//! Running `trippy edits sam` as a child process, and reading what it says.
//!
//! Module: `trips_viewer::sam_child` (binary only, native only)
//! Purpose: `docs/EDITOR.md` §3's "4. SAM 3 lift (E5)", viewer half. SAM 3 is
//!     never imported into trippy's process and never into the viewer's either:
//!     the lift is a Python command, so the viewer's SAM tool is a small
//!     process supervisor. This module is that supervisor and nothing else —
//!     build the command, spawn it, stream its stdout into the panel, kill it
//!     when Cancel is pressed, and parse the one JSON line it ends with. The
//!     panel, the prompt geometry and the region import live elsewhere
//!     (`edit_ui.rs`, `edit::sam`), so this half is testable against a FAKE
//!     child that is just `/bin/sh -c` (the tests at the bottom).
//! Invariants:
//!     - **Exactly one child at a time, spawned only when Jordan asks.** This
//!     	is his interactive use of his own GPU, which `AGENTS.md` §6 allows
//!       outside the queue; a tool that could fan out into several SAM runs
//!       would not be. [`SamJob::spawn`] is the only place a process is
//!       created and [`crate::edit_ui::EditSession`] holds at most one.
//!     - The child is killed on Cancel **and on drop**, so closing the window
//!       or opening another bundle cannot leave a SAM run holding the GPU.
//!     - `trippy edits sam`'s output contract (see its docstring): progress
//!       lines are prefixed `sam: `, and the LAST stdout line is the whole
//!       summary as one compact JSON object. Anything else on stdout is shown
//!       and ignored; stderr is shown prefixed `! ` and never parsed as JSON.
//!     - An exit code of 0 with no JSON line is a FAILURE, not a silent
//!       success: the region file might exist but the counts the panel reports
//!       would be invented.
//!     - `--scene` is deliberately NOT passed. `bundle.json`'s own `scene_root`
//!       (added 2026-09-07, `trippy.render.bundle.bundle_document`) is where
//!       the child gets it; a bundle without one cannot run the tool at all,
//!       and the panel says so rather than guessing where the photographs are.
//! Units: pixels for the prompt (the capture VIEW's grid — see
//!     [`crate::edit::sam`]); `mix` is 0 = splat, 1 = TRIPS; elapsed times are
//!     seconds.
//! Related docs: `docs/EDITOR.md` §3 "4. SAM 3 lift (E5)", §6's E5 row;
//!     `docs/USER_GUIDE.md` "SAM tool"; `trippy/cli.py` (`edits sam`).

use std::io::{BufRead, BufReader};
use std::path::{Path, PathBuf};
use std::process::{Child, Command, Stdio};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};

use trips_viewer::edit::model::Op;

/// An explicit interpreter, overriding everything else. Highest priority so a
/// bundle exported on another checkout can still be driven from this one.
pub const PYTHON_ENV: &str = "TRIPPY_PYTHON";

/// The trippy checkout to run `python -m trippy.cli` from, overriding
/// `bundle.json`'s `trippy_root`.
pub const ROOT_ENV: &str = "TRIPPY_ROOT";

/// Where a checkout's interpreter lives, relative to its root
/// (`scripts/bootstrap.sh` creates it there with `uv sync`).
pub const VENV_PYTHON: &str = ".venv/bin/python";

/// The environment variable that makes the child synthesise its mask.
///
/// `trippy.constants.SAM_FAKE_ENV`. Set alongside `--fake` so the switch works
/// even against a checkout whose CLI predates the flag.
pub const FAKE_ENV: &str = "TRIPPY_SAM_FAKE";

/// How many output lines the panel keeps. A SAM run prints a handful; a runaway
/// child must not grow the viewer's heap while Jordan watches it.
pub const MAX_LINES: usize = 200;

/// What Jordan pointed at, in the capture view's own pixel grid.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum SamPrompt {
    /// A drag rectangle, `[x0, y0, x1, y1]`, already normalised and clamped.
    Box([f64; 4]),
    /// An Alt-click.
    Point((f64, f64)),
}

impl SamPrompt {
    /// The `--box`/`--point` flag and its values.
    fn args(self) -> Vec<String> {
        match self {
            Self::Box(b) => {
                let mut out = vec!["--box".to_owned()];
                out.extend(b.iter().map(|v| format!("{v:.3}")));
                out
            }
            Self::Point((u, v)) => {
                vec!["--point".to_owned(), format!("{u:.3}"), format!("{v:.3}")]
            }
        }
    }

    /// A one-line description for the panel and the region's own name.
    #[must_use]
    pub fn label(self) -> String {
        match self {
            Self::Box([x0, y0, x1, y1]) => {
                format!("box ({x0:.0}, {y0:.0})-({x1:.0}, {y1:.0})")
            }
            Self::Point((u, v)) => format!("point ({u:.0}, {v:.0})"),
        }
    }
}

/// Everything one `trippy edits sam` invocation needs.
#[derive(Debug, Clone)]
pub struct SamRequest {
    /// The bundle directory (`bundle.json` + `points.npz`).
    pub bundle_dir: PathBuf,
    /// The capture view to prompt, by image file name.
    pub view_name: String,
    /// The prompt, in that view's pixels.
    pub prompt: SamPrompt,
    /// `--views-around`: neighbouring views that also segment and vote.
    pub views_around: usize,
    /// The op the imported region gets.
    pub op: Op,
    /// The mix the imported region gets (ignored by `delete`).
    pub mix: f64,
    /// `--out`: a throwaway `edits.json` the region is read back from. Never
    /// the session's own file — the import goes through the undo log.
    pub out: PathBuf,
    /// `--device`: `cpu` or `mps`.
    pub device: String,
    /// `--fake`: synthesise the mask instead of loading SAM 3.
    pub fake: bool,
}

/// The interpreter and the checkout to run it from.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Interpreter {
    /// The python binary.
    pub python: PathBuf,
    /// The trippy checkout, used as the child's working directory so
    /// `-m trippy.cli` resolves whether or not the package is installed.
    pub root: PathBuf,
}

/// Resolve the interpreter from explicit values, in priority order.
///
/// Pure, so the tests do not have to mutate the process environment.
///
/// # Arguments
/// - `python_env`: `$TRIPPY_PYTHON`, if set.
/// - `root_env`: `$TRIPPY_ROOT`, if set.
/// - `manifest_root`: `bundle.json`'s `trippy_root`, if the bundle has one.
///
/// # Errors
/// Returns `Err` naming both environment variables when nothing points at a
/// checkout, or when the resolved interpreter is not on disk — a wrong path
/// found now is a message in the panel, and found later is a child that exits
/// with an OS error nobody can act on.
pub fn resolve_interpreter(
    python_env: Option<&str>,
    root_env: Option<&str>,
    manifest_root: Option<&str>,
) -> Result<Interpreter, String> {
    let root = root_env
        .filter(|s| !s.is_empty())
        .or(manifest_root.filter(|s| !s.is_empty()))
        .map(PathBuf::from);
    let python = match python_env.filter(|s| !s.is_empty()) {
        Some(explicit) => PathBuf::from(explicit),
        None => match &root {
            Some(root) => root.join(VENV_PYTHON),
            None => {
                return Err(format!(
                    "no trippy checkout to run `trippy edits sam` from: this bundle.json \
                     records no `trippy_root` (re-export it with `trippy export-bundle`), \
                     and neither ${ROOT_ENV} nor ${PYTHON_ENV} is set"
                ))
            }
        },
    };
    if !python.is_file() {
        return Err(format!(
            "{} is not there; set ${PYTHON_ENV} to a python that can `import trippy`, \
             or ${ROOT_ENV} to a checkout with a {VENV_PYTHON}",
            python.display()
        ));
    }
    // With no root we still have an interpreter; run it from the bundle's own
    // directory, which is harmless: `-m trippy.cli` then resolves through that
    // interpreter's own installed packages.
    let root = root.unwrap_or_else(|| PathBuf::from("."));
    Ok(Interpreter { python, root })
}

/// The same thing, reading this process's environment.
///
/// # Errors
/// As [`resolve_interpreter`].
pub fn resolve_interpreter_from_env(manifest_root: Option<&str>) -> Result<Interpreter, String> {
    let python = std::env::var(PYTHON_ENV).ok();
    let root = std::env::var(ROOT_ENV).ok();
    resolve_interpreter(python.as_deref(), root.as_deref(), manifest_root)
}

/// The full argument list, `python` excluded.
///
/// `--scene` is absent on purpose: see the module invariants.
#[must_use]
pub fn command_args(request: &SamRequest) -> Vec<String> {
    let mut args = vec![
        "-m".to_owned(),
        "trippy.cli".to_owned(),
        "edits".to_owned(),
        "sam".to_owned(),
        "--bundle".to_owned(),
        request.bundle_dir.display().to_string(),
        "--view".to_owned(),
        request.view_name.clone(),
    ];
    args.extend(request.prompt.args());
    args.extend([
        "--prompt-space".to_owned(),
        // The viewer measures the capture VIEW's pixels; the child scales them
        // into the photograph's, which is the only side that knows its size.
        "view".to_owned(),
        "--views-around".to_owned(),
        request.views_around.to_string(),
        "--op".to_owned(),
        request.op.as_str().to_owned(),
        "--mix".to_owned(),
        format!("{:.4}", request.mix),
        "--out".to_owned(),
        request.out.display().to_string(),
        "--device".to_owned(),
        request.device.clone(),
    ]);
    if request.fake {
        args.push("--fake".to_owned());
    }
    args
}

/// The command line as one copy-pasteable string, for the panel and the log.
#[must_use]
pub fn command_line(interpreter: &Interpreter, request: &SamRequest) -> String {
    let mut parts = vec![interpreter.python.display().to_string()];
    parts.extend(command_args(request));
    parts.join(" ")
}

/// Build the `Command` [`SamJob::spawn`] runs.
#[must_use]
pub fn build_command(interpreter: &Interpreter, request: &SamRequest) -> Command {
    let mut command = Command::new(&interpreter.python);
    command.args(command_args(request));
    command.current_dir(&interpreter.root);
    // `-m trippy.cli` from a checkout that is not pip-installed needs the
    // checkout on the path; `current_dir` alone is not enough under every
    // python launcher, and an explicit PYTHONPATH costs nothing when it is.
    command.env("PYTHONPATH", &interpreter.root);
    if request.fake {
        command.env(FAKE_ENV, "1");
    }
    command
}

/// Where a run has got to. The whole state machine is this enum plus
/// [`SamJob::poll`]'s transitions out of `Running`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum SamState {
    /// The child is alive.
    Running,
    /// Exit 0, and the final JSON line parsed.
    Finished,
    /// Non-zero exit, a missing/unparseable summary, or a spawn failure.
    Failed(String),
    /// [`SamJob::cancel`] killed it.
    Cancelled,
}

impl SamState {
    /// Whether this state can still change.
    #[must_use]
    pub const fn is_running(&self) -> bool {
        matches!(self, Self::Running)
    }
}

/// One running (or just-finished) `trippy edits sam`.
pub struct SamJob {
    child: Option<Child>,
    state: SamState,
    /// Everything the child printed, newest last, capped at [`MAX_LINES`].
    lines: Arc<Mutex<Vec<String>>>,
    /// The last stdout line that looked like a JSON object.
    summary_line: Arc<Mutex<Option<String>>>,
    /// The parsed summary, once the run has finished.
    summary: Option<serde_json::Value>,
    /// How many reader threads are still draining a pipe. [`Self::drain`]
    /// waits for this to reach zero, which is the only exact "the child's last
    /// line has been read" signal there is.
    readers: Arc<AtomicUsize>,
    started: std::time::Instant,
    finished_after: Option<f64>,
}

impl SamJob {
    /// Spawn `command` with both pipes captured and start draining them.
    ///
    /// Two reader threads rather than one: a child that fills its stderr pipe
    /// while the parent is only reading stdout deadlocks, and the SAM child
    /// writes to both.
    ///
    /// # Errors
    /// Returns `Err` when the process cannot be started at all (a wrong
    /// interpreter path, no execute permission).
    pub fn spawn(mut command: Command) -> Result<Self, String> {
        command.stdout(Stdio::piped()).stderr(Stdio::piped());
        let mut child = command
            .spawn()
            .map_err(|e| format!("could not start {command:?}: {e}"))?;
        let lines: Arc<Mutex<Vec<String>>> = Arc::new(Mutex::new(Vec::new()));
        let summary_line: Arc<Mutex<Option<String>>> = Arc::new(Mutex::new(None));
        let readers = Arc::new(AtomicUsize::new(0));

        if let Some(stdout) = child.stdout.take() {
            let lines = Arc::clone(&lines);
            let summary_line = Arc::clone(&summary_line);
            let readers = Arc::clone(&readers);
            readers.fetch_add(1, Ordering::SeqCst);
            std::thread::spawn(move || {
                for line in BufReader::new(stdout).lines().map_while(Result::ok) {
                    // The contract: the summary is a compact JSON OBJECT on its
                    // own line. Keeping the last such line (rather than only the
                    // very last line) means a stray trailing newline or a
                    // warning printed after it cannot lose the result.
                    if line.trim_start().starts_with('{') {
                        *summary_line.lock().expect("poisoned") = Some(line.clone());
                    }
                    push_line(&lines, line);
                }
                readers.fetch_sub(1, Ordering::SeqCst);
            });
        }
        if let Some(stderr) = child.stderr.take() {
            let lines = Arc::clone(&lines);
            let readers = Arc::clone(&readers);
            readers.fetch_add(1, Ordering::SeqCst);
            std::thread::spawn(move || {
                for line in BufReader::new(stderr).lines().map_while(Result::ok) {
                    push_line(&lines, format!("! {line}"));
                }
                readers.fetch_sub(1, Ordering::SeqCst);
            });
        }

        Ok(Self {
            child: Some(child),
            state: SamState::Running,
            lines,
            summary_line,
            summary: None,
            readers,
            started: std::time::Instant::now(),
            finished_after: None,
        })
    }

    /// Advance the state machine. Cheap; call it once per frame.
    ///
    /// `Running` -> `Finished` on exit 0 with a parseable summary, and ->
    /// `Failed` on anything else. Already-terminal states are returned as they
    /// are, so a poll after the panel has read the result is a no-op.
    pub fn poll(&mut self) -> &SamState {
        if !self.state.is_running() {
            return &self.state;
        }
        let Some(child) = self.child.as_mut() else {
            return &self.state;
        };
        match child.try_wait() {
            Ok(None) => {}
            Ok(Some(status)) => {
                self.finished_after = Some(self.started.elapsed().as_secs_f64());
                // The reader threads may still be draining the last few bytes
                // out of the pipes; the summary is the very last thing written,
                // so waiting for EOF is what makes reading it reliable.
                self.drain();
                self.state = if status.success() {
                    match self.parse_summary() {
                        Ok(value) => {
                            self.summary = Some(value);
                            SamState::Finished
                        }
                        Err(e) => SamState::Failed(e),
                    }
                } else {
                    SamState::Failed(format!(
                        "trippy edits sam exited {}{}",
                        status.code().map_or_else(|| "on a signal".to_owned(), |c| c.to_string()),
                        self.last_error_line()
                            .map_or_else(String::new, |l| format!(": {l}"))
                    ))
                };
                self.child = None;
            }
            Err(e) => {
                self.state = SamState::Failed(format!("waiting on the SAM child: {e}"));
                self.child = None;
            }
        }
        &self.state
    }

    /// Wait until the child exits, then return the final state.
    ///
    /// The headless `--sam-box` path (`main.rs`) uses this; the window never
    /// does, because blocking the UI thread is exactly what the Cancel button
    /// exists to make unnecessary.
    pub fn wait(&mut self) -> &SamState {
        if let Some(child) = self.child.as_mut() {
            let _ = child.wait();
        }
        self.poll()
    }

    /// Kill the child. Idempotent; a job that already finished is untouched.
    pub fn cancel(&mut self) {
        if let Some(child) = self.child.as_mut() {
            let _ = child.kill();
            let _ = child.wait();
            self.finished_after = Some(self.started.elapsed().as_secs_f64());
            self.state = SamState::Cancelled;
            self.child = None;
            self.drain();
        }
    }

    /// The current state, without advancing it.
    #[must_use]
    pub const fn state(&self) -> &SamState {
        &self.state
    }

    /// The summary object, once the state is [`SamState::Finished`].
    #[must_use]
    pub const fn summary(&self) -> Option<&serde_json::Value> {
        self.summary.as_ref()
    }

    /// Everything the child has printed so far, newest last.
    #[must_use]
    pub fn lines(&self) -> Vec<String> {
        self.lines.lock().expect("poisoned").clone()
    }

    /// Seconds since the spawn, or the run's total once it ended.
    #[must_use]
    pub fn elapsed(&self) -> f64 {
        self.finished_after
            .unwrap_or_else(|| self.started.elapsed().as_secs_f64())
    }

    /// Wait for the reader threads to reach EOF on a dead child's pipes.
    ///
    /// `try_wait` can report the exit before the last write has been read out
    /// of the pipe, and the summary is the very LAST thing the child writes.
    /// Bounded, and it cannot hang: the pipes are closed by the exit (or by the
    /// kill), so both threads are already finishing. Waiting on the reader
    /// count rather than on the summary line means a failed or cancelled run
    /// pays the same tiny wait as a successful one instead of the whole cap.
    fn drain(&self) {
        for _ in 0..DRAIN_POLLS {
            if self.readers.load(Ordering::SeqCst) == 0 {
                return;
            }
            std::thread::sleep(std::time::Duration::from_millis(DRAIN_POLL_MS));
        }
    }

    /// Parse the recorded JSON line, or explain why there is nothing to parse.
    fn parse_summary(&self) -> Result<serde_json::Value, String> {
        let line = self
            .summary_line
            .lock()
            .expect("poisoned")
            .clone()
            .ok_or_else(|| {
                "trippy edits sam exited 0 but printed no summary line; \
                 the region file cannot be trusted"
                    .to_owned()
            })?;
        serde_json::from_str(&line).map_err(|e| format!("the summary line is not JSON: {e}"))
    }

    /// The last stderr line, for a failure message worth reading.
    fn last_error_line(&self) -> Option<String> {
        self.lines
            .lock()
            .expect("poisoned")
            .iter()
            .rev()
            .find(|l| l.starts_with("! "))
            .map(|l| l[2..].to_owned())
    }
}

impl Drop for SamJob {
    fn drop(&mut self) {
        // Closing the window, or opening another bundle, must not leave a SAM
        // run holding the GPU (module invariants).
        self.cancel();
    }
}

/// How many times [`SamJob::drain`] checks for the summary line.
const DRAIN_POLLS: u32 = 100;

/// Milliseconds between those checks. 100 x 5 ms = half a second, which is
/// orders of magnitude more than a closed pipe needs and still not a hang.
const DRAIN_POLL_MS: u64 = 5;

/// Append one line, dropping the oldest once [`MAX_LINES`] is reached.
fn push_line(lines: &Arc<Mutex<Vec<String>>>, line: String) {
    let mut guard = lines.lock().expect("poisoned");
    if guard.len() >= MAX_LINES {
        guard.remove(0);
    }
    guard.push(line);
}

/// Read the region the child wrote back out of its throwaway `edits.json`.
///
/// The child appends ONE region, so the last one is this run's. It is returned
/// rather than applied here: the caller puts it through
/// [`trips_viewer::edit::EditDocument::add_region`] so the import is one undo
/// step like every other edit.
///
/// # Errors
/// Returns `Err` when the file is missing, malformed, or holds no regions.
pub fn imported_region(path: &Path) -> Result<trips_viewer::edit::Region, String> {
    let doc = trips_viewer::edit::EditDocument::load(path)?;
    doc.regions()
        .last()
        .cloned()
        .ok_or_else(|| format!("{} holds no regions", path.display()))
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A fake child: a shell script standing in for `trippy edits sam`.
    ///
    /// The state machine under test is the parent's, so what matters is that
    /// the fake reproduces the OUTPUT CONTRACT (progress lines, then one JSON
    /// line, then an exit code) — which `/bin/sh` does exactly and a mocked-out
    /// `Child` would only approximate.
    fn fake_child(script: &str) -> Command {
        let mut command = Command::new("/bin/sh");
        command.arg("-c").arg(script);
        command
    }

    fn run_to_completion(script: &str) -> SamJob {
        let mut job = SamJob::spawn(fake_child(script)).expect("/bin/sh spawns");
        job.wait();
        job
    }

    #[test]
    fn a_child_that_prints_progress_then_a_summary_finishes() {
        let job = run_to_completion(
            "printf 'sam: segmenting IMG_0001.jpg\\nsam: 42 points\\n'; \
             printf '{\"n_points\": 42, \"region_id\": \"r-abcd\"}\\n'",
        );
        assert_eq!(job.state(), &SamState::Finished);
        let summary = job.summary().expect("a finished job has a summary");
        assert_eq!(summary["n_points"], 42);
        assert_eq!(summary["region_id"], "r-abcd");
        let lines = job.lines();
        assert_eq!(lines.len(), 3, "{lines:?}");
        assert!(lines[0].starts_with("sam: segmenting"));
        assert!(job.elapsed() >= 0.0);
    }

    #[test]
    fn stderr_is_shown_prefixed_and_never_parsed_as_the_summary() {
        let job = run_to_completion(
            "echo '{\"n_points\": 1}' >&2; printf '{\"n_points\": 7}\\n'",
        );
        assert_eq!(job.state(), &SamState::Finished);
        assert_eq!(job.summary().expect("summary")["n_points"], 7);
        assert!(job.lines().iter().any(|l| l == "! {\"n_points\": 1}"));
    }

    #[test]
    fn a_non_zero_exit_fails_and_quotes_the_last_stderr_line() {
        let job = run_to_completion("echo 'trippy edits sam: no such view' >&2; exit 2");
        match job.state() {
            SamState::Failed(message) => {
                assert!(message.contains("exited 2"), "{message}");
                assert!(message.contains("no such view"), "{message}");
            }
            other => panic!("expected Failed, got {other:?}"),
        }
        assert!(job.summary().is_none());
    }

    #[test]
    fn exit_zero_with_no_summary_is_a_failure_not_a_silent_success() {
        let job = run_to_completion("echo 'sam: did some work'; exit 0");
        match job.state() {
            SamState::Failed(message) => assert!(message.contains("no summary line"), "{message}"),
            other => panic!("expected Failed, got {other:?}"),
        }
    }

    #[test]
    fn a_summary_that_is_not_json_fails_rather_than_being_ignored() {
        let job = run_to_completion("printf '{not json at all\\n'");
        match job.state() {
            SamState::Failed(message) => assert!(message.contains("not JSON"), "{message}"),
            other => panic!("expected Failed, got {other:?}"),
        }
    }

    #[test]
    fn the_last_json_line_wins_over_an_earlier_one() {
        let job = run_to_completion(
            "printf '{\"n_points\": 1}\\n'; printf 'sam: more\\n'; printf '{\"n_points\": 9}\\n'",
        );
        assert_eq!(job.summary().expect("summary")["n_points"], 9);
    }

    #[test]
    fn cancel_kills_a_running_child_and_is_idempotent() {
        let mut job = SamJob::spawn(fake_child("sleep 30")).expect("/bin/sh spawns");
        assert_eq!(job.poll(), &SamState::Running);
        job.cancel();
        assert_eq!(job.state(), &SamState::Cancelled);
        // A poll after the kill must not resurrect it or overwrite the reason.
        assert_eq!(job.poll(), &SamState::Cancelled);
        job.cancel();
        assert_eq!(job.state(), &SamState::Cancelled);
        assert!(job.summary().is_none());
    }

    #[test]
    fn a_finished_job_is_not_re_cancelled_by_drop() {
        let mut job = run_to_completion("printf '{\"n_points\": 3}\\n'");
        assert_eq!(job.state(), &SamState::Finished);
        job.cancel();
        assert_eq!(job.state(), &SamState::Finished, "Cancel after the fact is a no-op");
    }

    #[test]
    fn spawning_something_that_does_not_exist_is_an_error_not_a_panic() {
        let command = Command::new("/definitely/not/a/binary/trippy-sam");
        assert!(SamJob::spawn(command).is_err());
    }

    #[test]
    fn the_output_buffer_is_capped() {
        let job = run_to_completion(&format!(
            "i=0; while [ $i -lt {} ]; do echo \"sam: line $i\"; i=$((i+1)); done; \
             printf '{{\"n_points\": 0}}\\n'",
            MAX_LINES + 50
        ));
        assert_eq!(job.state(), &SamState::Finished);
        assert_eq!(job.lines().len(), MAX_LINES);
        assert!(
            job.lines().last().expect("lines").starts_with('{'),
            "the newest lines are the ones kept"
        );
    }

    // --- the command line ---------------------------------------------------

    fn a_request() -> SamRequest {
        SamRequest {
            bundle_dir: PathBuf::from("/tmp/bundle"),
            view_name: "IMG_0003.jpg".to_owned(),
            prompt: SamPrompt::Box([10.0, 20.0, 110.0, 220.0]),
            views_around: 2,
            op: Op::Fade,
            mix: 0.25,
            out: PathBuf::from("/tmp/sam/edits.json"),
            device: "mps".to_owned(),
            fake: false,
        }
    }

    #[test]
    fn the_command_is_the_one_the_editor_doc_names() {
        let args = command_args(&a_request());
        let joined = args.join(" ");
        assert!(joined.starts_with("-m trippy.cli edits sam "), "{joined}");
        assert!(joined.contains("--bundle /tmp/bundle"));
        assert!(joined.contains("--view IMG_0003.jpg"));
        assert!(joined.contains("--box 10.000 20.000 110.000 220.000"));
        assert!(joined.contains("--views-around 2"));
        assert!(joined.contains("--op fade"));
        assert!(joined.contains("--mix 0.2500"));
        assert!(joined.contains("--out /tmp/sam/edits.json"));
        assert!(joined.contains("--device mps"));
        assert!(joined.contains("--prompt-space view"));
        assert!(
            !joined.contains("--scene"),
            "the child reads scene_root out of bundle.json"
        );
        assert!(!joined.contains("--fake"));
    }

    #[test]
    fn a_point_prompt_and_the_fake_switch() {
        let mut request = a_request();
        request.prompt = SamPrompt::Point((123.5, 45.25));
        request.fake = true;
        let joined = command_args(&request).join(" ");
        assert!(joined.contains("--point 123.500 45.250"), "{joined}");
        assert!(joined.ends_with("--fake"), "{joined}");
    }

    #[test]
    fn a_prompt_labels_itself_for_the_panel() {
        assert_eq!(
            SamPrompt::Box([10.4, 20.6, 110.0, 220.0]).label(),
            "box (10, 21)-(110, 220)"
        );
        assert_eq!(SamPrompt::Point((1.2, 3.8)).label(), "point (1, 4)");
    }

    #[test]
    fn the_interpreter_comes_from_the_env_before_the_bundle() {
        let dir = std::env::temp_dir().join(format!("trips-sam-py-{}", std::process::id()));
        std::fs::create_dir_all(dir.join(".venv/bin")).unwrap();
        let python = dir.join(VENV_PYTHON);
        std::fs::write(&python, "#!/bin/sh\n").unwrap();

        let root = dir.display().to_string();
        // From the manifest.
        let from_manifest = resolve_interpreter(None, None, Some(&root)).unwrap();
        assert_eq!(from_manifest.python, python);
        assert_eq!(from_manifest.root, dir);
        // $TRIPPY_ROOT beats the manifest.
        let other = resolve_interpreter(None, Some(&root), Some("/nowhere")).unwrap();
        assert_eq!(other.python, python);
        // $TRIPPY_PYTHON beats both.
        let explicit = resolve_interpreter(Some("/bin/sh"), Some(&root), Some("/nowhere")).unwrap();
        assert_eq!(explicit.python, PathBuf::from("/bin/sh"));
        assert_eq!(explicit.root, dir, "the root still comes from $TRIPPY_ROOT");

        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn no_checkout_anywhere_names_both_environment_variables() {
        let message = resolve_interpreter(None, None, None).unwrap_err();
        assert!(message.contains(ROOT_ENV), "{message}");
        assert!(message.contains(PYTHON_ENV), "{message}");
        assert!(message.contains("trippy_root"), "{message}");
    }

    #[test]
    fn a_checkout_with_no_venv_says_which_path_is_missing() {
        let message = resolve_interpreter(None, Some("/nowhere/at/all"), None).unwrap_err();
        assert!(message.contains("/nowhere/at/all/.venv/bin/python"), "{message}");
    }

    #[test]
    fn importing_a_region_reads_the_last_one_the_child_wrote() {
        use trips_viewer::edit::model::{Op, Params, Region};

        let dir = std::env::temp_dir().join(format!("trips-sam-import-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("edits.json");

        let mut doc = trips_viewer::edit::EditDocument::new("trippy-bundle-1".to_owned());
        doc.add_region(
            &Region::new(
                "r-old".to_owned(),
                "an earlier one".to_owned(),
                Params::Pointset { point_ids: vec![1] },
                0.0,
                Op::Fade,
            ),
            None,
        )
        .unwrap();
        doc.add_region(
            &Region::new(
                "r-sam".to_owned(),
                "sam: box".to_owned(),
                Params::Pointset {
                    point_ids: vec![4, 5, 6],
                },
                0.0,
                Op::Delete,
            ),
            None,
        )
        .unwrap();
        doc.save(&path).unwrap();

        let region = imported_region(&path).expect("the last region");
        assert_eq!(region.id, "r-sam");
        assert!(matches!(
            region.params,
            Params::Pointset { ref point_ids } if point_ids == &vec![4, 5, 6]
        ));
        assert!(imported_region(&dir.join("missing.json")).is_err());
        std::fs::remove_dir_all(&dir).ok();
    }
}
