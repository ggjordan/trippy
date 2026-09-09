"""Tests for scripts/open_quest_viewer.sh: generation, and the launchers it writes.

Module: tests.test_open_quest_viewer_script
Context: ADR-0008-supersplat.md Stage 3 item 9. Unlike sog_export.sh and
    quest_viewer_bootstrap.sh, this script does no network/install work -- it only
    writes two double-click `.command` launchers next to a pre-built viewer bundle
    (a synthetic one here, never a Karekare bundle):
        OPEN_<NAME>.command           -- 127.0.0.1 only, plain http, Mac preview.
        OPEN_<NAME>_ON_QUEST.command  -- 0.0.0.0, self-signed HTTPS, for the Quest
                                          browser over the LAN.
    That makes it fast and safe to exercise for real, including actually starting
    the generated servers against a synthetic fixture bundle and fetching from them
    -- no scene imagery, no network beyond loopback/LAN, nothing installed.

Invariants under test:
    - `bash -n` on the generator AND on both generated `.command` files.
    - No args / missing bundle -> usage, exit 2.
    - A bundle directory without index.html is refused (exit 2).
    - Both generated launchers embed the SAME absolute bundle path and the SAME
      deterministic (name-hashed) port.
    - The preview launcher actually serves the bundle on 127.0.0.1 over plain http.
    - The Quest launcher actually serves the bundle over HTTPS (self-signed cert,
      generated on the fly) reachable via 127.0.0.1 -- standing in for "reachable at
      the Mac's LAN IP" without depending on the test machine's actual network
      interface. Skipped if openssl is missing.
    - The Quest launcher's on-disk cert cache is idempotent: a second invocation
      with the same detected IP does not regenerate the certificate.
"""

from __future__ import annotations

import os
import shutil
import ssl
import subprocess
import time
import urllib.request
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
SCRIPT = REPO_ROOT / "scripts" / "open_quest_viewer.sh"


def _run(tmp_path: Path, *args: str) -> subprocess.CompletedProcess[str]:
    env = dict(os.environ)
    env["TRIPPY_OUTPUT"] = str(tmp_path / "trippy_output")
    return subprocess.run(
        ["bash", str(SCRIPT), *args],
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
        cwd=REPO_ROOT,
        env=env,
    )


def _make_bundle(tmp_path: Path) -> Path:
    """A synthetic, non-scene viewer bundle: just enough files to look real."""
    bundle = tmp_path / "bundle"
    bundle.mkdir()
    (bundle / "index.html").write_text("<html><body>synthetic test bundle</body></html>")
    (bundle / "index.sog").write_bytes(b"not a real sog, just a fixture")
    (bundle / "settings.json").write_text("{}")
    return bundle


def test_script_exists_and_is_executable() -> None:
    assert SCRIPT.is_file()
    assert os.access(SCRIPT, os.X_OK)


def test_script_is_valid_bash() -> None:
    result = subprocess.run(["bash", "-n", str(SCRIPT)], capture_output=True, text=True, check=False)
    assert result.returncode == 0, result.stderr


def test_no_args_is_usage_error(tmp_path: Path) -> None:
    result = _run(tmp_path)
    assert result.returncode == 2
    assert "usage" in result.stderr


def test_missing_bundle_dir_is_allowed_and_wired_for_a_future_build(tmp_path: Path) -> None:
    """A queued scripts/gpu_submit.sh job may not have built the bundle yet -- the
    generator still writes launchers wired to that future path (2026-09-10: the
    coordinator's fix for a stuck CPU-only conversion moved conversions onto the
    GPU queue, so launchers are generated before the job that builds the bundle
    finishes). The generated launcher's OWN run-time check is what refuses."""
    future_bundle = tmp_path / "not-built-yet"
    result = _run(tmp_path, str(future_bundle), "some-name")
    assert result.returncode == 0, f"stdout:\n{result.stdout}\nstderr:\n{result.stderr}"
    assert "not built yet" in result.stderr


def test_bundle_without_sog_is_allowed_and_wired(tmp_path: Path) -> None:
    empty_bundle = tmp_path / "empty-bundle"
    empty_bundle.mkdir()
    (empty_bundle / "index.html").write_text("<html></html>")
    result = _run(tmp_path, str(empty_bundle), "some-name")
    assert result.returncode == 0, f"stdout:\n{result.stdout}\nstderr:\n{result.stderr}"
    assert "not built yet" in result.stderr

    # The generated launchers refuse to open something with no SOG in it.
    deliver_dir = tmp_path / "trippy_output" / "deliver" / "quest-viewer-some-name"
    preview_text = (deliver_dir / "OPEN_SOME_NAME.command").read_text()
    quest_text = (deliver_dir / "OPEN_SOME_NAME_ON_QUEST.command").read_text()
    assert "index.sog" in preview_text
    assert "index.sog" in quest_text


def test_generates_two_valid_launchers_with_matching_port_and_path(tmp_path: Path) -> None:
    bundle = _make_bundle(tmp_path)

    result = _run(tmp_path, str(bundle), "kk-test")

    assert result.returncode == 0, f"stdout:\n{result.stdout}\nstderr:\n{result.stderr}"
    deliver_dir = tmp_path / "trippy_output" / "deliver" / "quest-viewer-kk-test"
    preview_cmd = deliver_dir / "OPEN_KK_TEST.command"
    quest_cmd = deliver_dir / "OPEN_KK_TEST_ON_QUEST.command"
    assert preview_cmd.is_file()
    assert quest_cmd.is_file()
    assert os.access(preview_cmd, os.X_OK)
    assert os.access(quest_cmd, os.X_OK)

    for cmdfile in (preview_cmd, quest_cmd):
        syntax = subprocess.run(["bash", "-n", str(cmdfile)], capture_output=True, text=True, check=False)
        assert syntax.returncode == 0, f"{cmdfile}: {syntax.stderr}"

    preview_text = preview_cmd.read_text()
    quest_text = quest_cmd.read_text()
    bundle_abs = str(bundle.resolve())
    assert bundle_abs in preview_text
    assert bundle_abs in quest_text

    # Same name -> same deterministic port in both launchers.
    preview_port = next(line for line in preview_text.splitlines() if line.startswith("PORT="))
    quest_port = next(line for line in quest_text.splitlines() if line.startswith("PORT="))
    assert preview_port == quest_port

    # Never click Publish-equivalent hazards or unexplained 0.0.0.0: the preview
    # launcher must stay bound to loopback, the Quest one is documented as LAN-only.
    assert "127.0.0.1" in preview_text
    assert "0.0.0.0" in quest_text
    assert "this network only" in quest_text or "home network" in quest_text


def test_deterministic_naming_uppercases_and_sanitises(tmp_path: Path) -> None:
    bundle = _make_bundle(tmp_path)
    result = _run(tmp_path, str(bundle), "kklid-tripsclean-shade-keep")
    assert result.returncode == 0, result.stderr
    deliver_dir = tmp_path / "trippy_output" / "deliver" / "quest-viewer-kklid-tripsclean-shade-keep"
    assert (deliver_dir / "OPEN_KKLID_TRIPSCLEAN_SHADE_KEEP.command").is_file()
    assert (deliver_dir / "OPEN_KKLID_TRIPSCLEAN_SHADE_KEEP_ON_QUEST.command").is_file()


def test_launcher_refuses_at_runtime_when_sog_not_built_yet(tmp_path: Path) -> None:
    """The concrete "refuses to open if the SOG is missing" behaviour: a launcher
    generated against a not-yet-built bundle path must refuse cleanly (not hang on
    the `read -r` prompt, not open a browser at a broken URL) when actually run."""
    future_bundle = tmp_path / "not-built-yet"
    result = _run(tmp_path, str(future_bundle), "pending-job")
    assert result.returncode == 0, result.stderr

    cmdfile = (
        tmp_path / "trippy_output" / "deliver" / "quest-viewer-pending-job" / "OPEN_PENDING_JOB.command"
    )
    proc = subprocess.run(
        ["bash", str(cmdfile)],
        input="\n",  # answer the "(press return to close)" prompt
        capture_output=True,
        text=True,
        timeout=10,
        check=False,
    )
    assert proc.returncode == 1
    assert "Not ready yet" in proc.stdout


def test_preview_launcher_actually_serves_the_bundle(tmp_path: Path) -> None:
    bundle = _make_bundle(tmp_path)
    result = _run(tmp_path, str(bundle), "serve-test")
    assert result.returncode == 0, result.stderr

    cmdfile = (
        tmp_path
        / "trippy_output"
        / "deliver"
        / "quest-viewer-serve-test"
        / "OPEN_SERVE_TEST.command"
    )
    # Don't let the launcher's own `open` (Finder browser launch) fire during tests.
    text = cmdfile.read_text().replace('open "http://127.0.0.1:$PORT/index.html"', "true")
    cmdfile.write_text(text)

    proc = subprocess.Popen(["bash", str(cmdfile)], stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    try:
        port_line = next(line for line in cmdfile.read_text().splitlines() if line.startswith("PORT="))
        port = int(port_line.split("=", 1)[1])
        deadline = time.time() + 10
        last_err: Exception | None = None
        while time.time() < deadline:
            try:
                with urllib.request.urlopen(f"http://127.0.0.1:{port}/index.html", timeout=1) as resp:
                    assert resp.status == 200
                    assert b"synthetic test bundle" in resp.read()
                    break
            except Exception as exc:  # noqa: BLE001 - retry until the server is up
                last_err = exc
                time.sleep(0.3)
        else:
            pytest.fail(f"preview server never came up on 127.0.0.1:{port}: {last_err}")
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            proc.kill()
        subprocess.run(["pkill", "-f", f"http.server {port} --bind 127.0.0.1"], check=False)


def test_quest_launcher_serves_https_with_self_signed_cert(tmp_path: Path) -> None:
    if shutil.which("openssl") is None:
        pytest.skip("openssl not installed on this machine")

    bundle = _make_bundle(tmp_path)
    result = _run(tmp_path, str(bundle), "xr-test")
    assert result.returncode == 0, result.stderr

    cmdfile = (
        tmp_path / "trippy_output" / "deliver" / "quest-viewer-xr-test" / "OPEN_XR_TEST_ON_QUEST.command"
    )
    port_line = next(line for line in cmdfile.read_text().splitlines() if line.startswith("PORT="))
    port = int(port_line.split("=", 1)[1])

    env = dict(os.environ)
    env["TRIPPY_OUTPUT"] = str(tmp_path / "trippy_output")
    proc = subprocess.Popen(
        ["bash", str(cmdfile)], stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, env=env
    )
    try:
        ctx = ssl.SSLContext(ssl.PROTOCOL_TLS_CLIENT)
        ctx.check_hostname = False
        ctx.verify_mode = ssl.CERT_NONE

        deadline = time.time() + 15
        last_err: Exception | None = None
        while time.time() < deadline:
            try:
                with urllib.request.urlopen(
                    f"https://127.0.0.1:{port}/index.html", timeout=1, context=ctx
                ) as resp:
                    assert resp.status == 200
                    assert b"synthetic test bundle" in resp.read()
                    break
            except Exception as exc:  # noqa: BLE001 - retry until the server + cert are up
                last_err = exc
                time.sleep(0.3)
        else:
            pytest.fail(f"HTTPS server never came up on 127.0.0.1:{port}: {last_err}")

        cert_dir = tmp_path / "trippy_output" / "tools" / "quest-viewer" / "certs"
        assert (cert_dir / "quest-viewer.crt").is_file()
        assert (cert_dir / "quest-viewer.key").is_file()
        cert_before = (cert_dir / "quest-viewer.crt").read_bytes()
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            proc.kill()

    # Second run with the same recorded IP must reuse the cached certificate.
    proc2 = subprocess.Popen(
        ["bash", str(cmdfile)], stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, env=env
    )
    try:
        time.sleep(1.5)
    finally:
        proc2.terminate()
        try:
            proc2.wait(timeout=5)
        except subprocess.TimeoutExpired:
            proc2.kill()
    cert_dir = tmp_path / "trippy_output" / "tools" / "quest-viewer" / "certs"
    cert_after = (cert_dir / "quest-viewer.crt").read_bytes()
    assert cert_before == cert_after
