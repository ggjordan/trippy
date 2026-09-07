//! The egui application: input, the ms readout, and the settings panel.
//!
//! Module: `trips_viewer::app`
//! Purpose: the interactive shell around [`crate::renderer`]. Deliberately one
//!     scene pane plus a small overlay — this viewer exists so Jordan can fly
//!     through a TRIPS scene and flip between the network's frame and the
//!     evidence behind it, not to retrain anything.
//! Invariants:
//!     - The camera starts **pinned** to the bundle's default view and stays
//!       bit-identical to it until the user moves; see
//!       [`crate::camera::Controller`].
//!     - Mouse input is read from the scene [`egui::Response`] and NEVER gated
//!       on `Context::egui_wants_pointer_input`. That predicate is true while
//!       *any* widget is being interacted with — including this canvas, which
//!       is allocated with `Sense::click_and_drag()` — so gating on it
//!       swallowed every drag the moment the button went down. `dragged_by` is
//!       already scoped to drags that started on this widget, which is the
//!       test actually wanted, and is what Brush's own camera controls use.
//!     - The render happens on the UI thread, blocking on the rasteriser's one
//!       readback. That costs the CPU/GPU overlap a viewer would normally get
//!       and is why the reported frame time is the honest one — nothing is
//!       hidden in a queue. `docs/LIMITATIONS.md` records it.
//!     - Every performance lever is a checkbox that starts OFF, so the app
//!       opens rendering the exact pipeline and any speed-up is something the
//!       user (or the launcher) asked for.
//! Units: milliseconds and frames per second in the readout; world units per
//!     second for the fly speed.
//! Related docs: `docs/USER_GUIDE.md`;
//!     `docs/decisions/ADR-0006-viewer-integration.md`.

use brush_pyramid::gpu::block_on;
use eframe::egui;

use crate::blit::{BlitCallback, BlitResources};
use crate::blend::{Blend, BlendMode, GATE_SCALE_MAX, GATE_SCALE_MIN};
use crate::bundle::Bundle;
use crate::edit_ui::{EditSession, SamGesture};
use trips_viewer::edit::cluster::ClickCamera;
use trips_viewer::edit::sam as sam_geom;
use crate::camera::{Controller, Mode};
use crate::renderer::{ExposureMode, Renderer, Settings, ViewMode, MANUAL_EXPOSURE_LIMIT};

/// How many recent frame intervals the fps readout averages over.
const FPS_WINDOW: usize = 30;

/// egui scroll units in one wheel notch. macOS trackpads report a continuous
/// delta, so this is a divisor rather than a count.
const SCROLL_NOTCH: f32 = 50.0;

/// The orbit pivot counts as "at the edge of the capture box" once it leaves
/// the box shrunk by this factor, i.e. within 0.1% of a wall.
const EDGE_OF_BOX: f32 = 0.999;

/// Render-scale presets the `-`/`=` keys step between.
const SCALE_STEPS: [f32; 4] = [0.5, 0.75, 0.9, 1.0];

/// The SAM tool's drag rectangle: `edit::apply::PREVIEW_TINT`'s magenta, so the
/// box Jordan is drawing and the points it selects are the one colour.
const SAM_MARQUEE_COLOUR: egui::Color32 = egui::Color32::from_rgb(255, 0, 255);

/// Stroke width of that rectangle, egui points.
const SAM_MARQUEE_WIDTH: f32 = 1.5;

/// The viewer.
pub struct ViewerApp {
    renderer: Renderer,
    controller: Controller,
    views: Vec<crate::bundle::BundleView>,
    scene_name: String,
    mode: ViewMode,
    settings: Settings,
    /// Which per-image exposure the tone mapper applies (panel + `X` key).
    exposure: ExposureMode,
    /// The EV the manual slider last held, kept across a trip through the
    /// other modes so switching back does not reset it.
    manual_exposure_ev: f32,
    show_panel: bool,
    intervals: std::collections::VecDeque<f64>,
    last_frame: Option<std::time::Instant>,
    last_stats: Option<crate::renderer::FrameStats>,
    error: Option<String>,
    /// The Blend panel's state (splat vs TRIPS).
    blend: Blend,
    /// Whether this bundle carries a `blend` block at all. False hides the
    /// panel entirely, so a non-hybrid scene's UI is exactly what it was.
    has_blend: bool,
    /// The editor: `edits.json`, the Regions/Inspector/Tools panels, the shade
    /// finder. Toggled with `M` and hidden by default, so a viewing session is
    /// exactly the session v0.6.0 shipped (`docs/EDITOR.md` §4).
    edit: EditSession,
    /// The SAM tool's live drag rectangle, `(start, current)` in egui points.
    ///
    /// Kept here rather than in [`EditSession`] because it is a WINDOW thing:
    /// it exists only to be painted over the render, and the session is handed
    /// the finished gesture in render pixels once the button comes up.
    sam_drag: Option<(egui::Pos2, egui::Pos2)>,
}

impl ViewerApp {
    /// Build the app, sharing eframe's wgpu device with Burn.
    ///
    /// # Arguments
    /// - `cc`: eframe's creation context; must be the wgpu backend.
    /// - `bundle`: the loaded scene.
    /// - `settings`: initial performance levers (from the command line).
    /// - `splat_args`: how to reach the live Gaussian splat, if at all.
    /// - `edits`: `--edits`, or `None` for `<bundle>/edits.json`.
    ///
    /// # Errors
    /// Returns `Err` if eframe is not on wgpu, or the weights are rejected.
    pub fn new(
        cc: &eframe::CreationContext<'_>,
        bundle: Bundle,
        settings: Settings,
        splat_args: &crate::SplatArgs,
        edits: Option<&std::path::Path>,
    ) -> Result<Self, String> {
        let state = cc
            .wgpu_render_state
            .as_ref()
            .ok_or("trips-viewer needs the wgpu backend")?;
        BlitResources::install(state);

        // Hand Burn the device egui is already drawing with, so the pyramid's
        // output buffer can be bound into egui's render pass directly.
        let burn_device = crate::init_burn_on(
            state.adapter.clone(),
            state.device.clone(),
            state.queue.clone(),
        );

        let views = bundle.manifest.views.clone();
        let scene_name = if bundle.manifest.name.is_empty() {
            bundle
                .dir
                .file_name()
                .map_or_else(|| "scene".to_owned(), |n| n.to_string_lossy().into_owned())
        } else {
            bundle.manifest.name.clone()
        };
        let up = bundle.manifest.up;
        // The home view is a real capture pose, so the viewer opens on
        // something a camera actually saw. Scale comes from the same place:
        // never from `renderer.bounds()`, which is the POINT bounding box and
        // on the horse bundle is 12 990 units across because of the far-field
        // environment sphere -- the 1948 u/s fly speed Jordan was given.
        let home = bundle.home_view_position();
        // The Blend panel starts at the run's own recorded gate_scale but in
        // TRIPS-only mode -- see `Blend::for_bundle` for why the first frame is
        // deliberately the one the run's metrics describe.
        let blend = Blend::for_bundle(
            bundle
                .manifest
                .blend
                .as_ref()
                .map_or(1.0, |b| b.gate_scale),
        );
        let has_blend = bundle.manifest.blend.is_some();

        let bundle_dir = bundle.dir.clone();
        let bundle_format = bundle.manifest.format.clone();
        let blend_manifest = bundle.manifest.blend.clone();
        // The SAM tool spawns `trippy edits sam` from the checkout the bundle
        // was exported by, and lets that child find the photographs from
        // `scene_root` -- so both come off the manifest before it moves into
        // the renderer (`crate::sam_child`).
        let trippy_root = bundle.manifest.trippy_root.clone();
        let has_scene_root = bundle.manifest.scene_root.is_some();
        let mut renderer = Renderer::new(bundle, burn_device)?;
        // On the SAME device eframe just handed Burn, so the splat image and the
        // TRIPS frame are two tensors on one allocator (`crate::splat`'s first
        // invariant). Blocking here is deliberate: the window has not been drawn
        // yet, and a half-loaded scene would be worse than a slow first frame.
        crate::attach_live_splat(
            &mut renderer,
            &bundle_dir,
            blend_manifest.as_ref(),
            splat_args,
        );
        let controller = Controller::new(&views, home, up);
        // The session is built AFTER the renderer, because a bundle reopened
        // with a saved `edits.json` must render edited on its first frame and
        // `apply` needs the renderer's own point cloud to compose against.
        let mut edit = EditSession::open_with_path(&bundle_dir, &bundle_format, edits);
        edit.set_bundle_paths(trippy_root.as_deref(), has_scene_root);
        // The click tool's `max_radius` is the bundle's own median camera
        // spacing, which needs the views and the cloud -- both of which exist
        // only now (`trippy.edit.cluster.default_max_radius_from_bundle`).
        edit.init_click_defaults(&views, &renderer, controller.scene().diameter());
        edit.estimate_shade_depths(&renderer);
        edit.refresh_shade(&renderer);
        if let Err(e) = edit.apply(&mut renderer) {
            log::warn!("edits not applied: {e}");
        }

        cc.egui_ctx
            .options_mut(|o| o.theme_preference = egui::ThemePreference::Dark);

        Ok(Self {
            renderer,
            controller,
            views,
            scene_name,
            mode: ViewMode::default(),
            settings,
            exposure: ExposureMode::default(),
            manual_exposure_ev: 0.0,
            show_panel: true,
            intervals: std::collections::VecDeque::with_capacity(FPS_WINDOW),
            last_frame: None,
            last_stats: None,
            error: None,
            blend,
            has_blend,
            edit,
            sam_drag: None,
        })
    }

    /// Set the initial blend state (from `--blend-mode` / `--gate-scale` / `--mix`).
    pub fn set_blend(&mut self, blend: Blend) {
        self.blend = blend;
    }

    /// Set the initial view mode (from `--mode`).
    pub fn set_mode(&mut self, mode: ViewMode) {
        self.mode = mode;
    }

    /// Show the editor's panels from the first frame (`--edit`), as if `M` had
    /// been pressed. Nothing else about the session changes.
    pub fn set_edit_mode(&mut self, active: bool) {
        self.edit.active = active;
    }

    /// Set the initial navigation mode (orbit by default, `--free` for fly).
    pub fn set_navigation(&mut self, mode: Mode) {
        self.controller.set_mode(mode);
    }

    /// Frames per second over the last [`FPS_WINDOW`] frames, or `None` before
    /// enough have been drawn.
    fn fps(&self) -> Option<f64> {
        if self.intervals.len() < 2 {
            return None;
        }
        let mean = self.intervals.iter().sum::<f64>() / self.intervals.len() as f64;
        (mean > 0.0).then(|| 1000.0 / mean)
    }

    /// Mean frame interval, milliseconds.
    fn frame_ms(&self) -> Option<f64> {
        (!self.intervals.is_empty())
            .then(|| self.intervals.iter().sum::<f64>() / self.intervals.len() as f64)
    }

    fn record_interval(&mut self) {
        let now = std::time::Instant::now();
        if let Some(previous) = self.last_frame {
            if self.intervals.len() == FPS_WINDOW {
                self.intervals.pop_front();
            }
            self.intervals
                .push_back((now - previous).as_secs_f64() * 1e3);
        }
        self.last_frame = Some(now);
    }

    /// Consume keyboard and mouse for this frame.
    ///
    /// Drags come from `response`, which egui has already scoped to this
    /// widget *and* to gestures that started on it: a drag begun on the HUD
    /// window never reaches the camera, and a drag begun on the canvas keeps
    /// working after the pointer wanders over the HUD. That is the whole
    /// contract, and it is why no `Context::egui_wants_pointer_input` check
    /// appears here — see the module invariants for what that cost.
    fn handle_input(&mut self, ctx: &egui::Context, response: &egui::Response, dt: f32) {
        let viewport_height = response.rect.height();
        let delta = response.drag_delta();
        // The SAM tool's marquee (`docs/EDITOR.md` §3, E5). It takes the
        // PRIMARY drag while that tool has focus, which is what "drag a box on
        // the render" means; orbit is still one modifier away (Shift-drag) and
        // right/middle-drag still pans, so navigation loses nothing permanent
        // and nothing at all outside this one tool.
        let sam_box_drag = self.edit.sam_tool_active()
            && response.dragged_by(egui::PointerButton::Primary)
            && !ctx.input(|i| i.modifiers.shift);
        if sam_box_drag {
            if let Some(pos) = response.interact_pointer_pos() {
                let start = self.sam_drag.map_or(pos, |(s, _)| s);
                self.sam_drag = Some((start, pos));
            }
        } else if response.dragged_by(egui::PointerButton::Primary) {
            self.controller.drag(delta.x, delta.y);
        } else if response.dragged_by(egui::PointerButton::Secondary)
            || response.dragged_by(egui::PointerButton::Middle)
        {
            self.controller.pan(delta.x, delta.y, viewport_height);
        }
        // The gesture is finished when the button comes up; only then is it
        // worth converting, and only then is a tiny drag known to be a click.
        if !response.dragged_by(egui::PointerButton::Primary) {
            if let Some((start, end)) = self.sam_drag.take() {
                let ppp = ctx.pixels_per_point();
                let scale = self.settings.render_scale;
                let min = (response.rect.min.x, response.rect.min.y);
                let a = sam_geom::render_pixel((start.x, start.y), min, ppp, scale);
                let b = sam_geom::render_pixel((end.x, end.y), min, ppp, scale);
                if sam_geom::is_box_drag(a, b) {
                    self.edit.request_sam(SamGesture::Box(a, b));
                }
            }
        }
        if response.dragged() {
            ctx.set_cursor_icon(egui::CursorIcon::Grabbing);
        } else if response.hovered() {
            ctx.set_cursor_icon(egui::CursorIcon::Grab);
        }

        // Shift-click selects an object (`docs/EDITOR.md` §4, E4). Read from
        // the same `response` every drag is, so a Shift-click that began on the
        // Edit window never reaches the canvas; and it is a CLICK, not a drag,
        // so Shift-dragging still orbits and navigation loses nothing. The
        // pixel handed on is in the RENDER's own coordinates -- egui points
        // scaled by the display's `pixels_per_point` and the render-scale
        // lever -- because that is the camera the click will be projected
        // against (`ui`'s own `width`/`height`).
        if self.edit.active && response.clicked() && ctx.input(|i| i.modifiers.shift) {
            if let Some(pos) = response.interact_pointer_pos() {
                let ppp = ctx.pixels_per_point();
                let scale = self.settings.render_scale;
                self.edit.request_click(sam_geom::render_pixel(
                    (pos.x, pos.y),
                    (response.rect.min.x, response.rect.min.y),
                    ppp,
                    scale,
                ));
            }
        }

        // Alt-click is the SAM tool's point prompt (`docs/EDITOR.md` §3, E5).
        // Alt is bound to nothing else in this viewer, so this costs no
        // existing gesture; and it is a CLICK, so an Alt-drag is still a drag.
        if self.edit.active
            && response.clicked()
            && ctx.input(|i| i.modifiers.alt && !i.modifiers.shift)
        {
            if let Some(pos) = response.interact_pointer_pos() {
                let ppp = ctx.pixels_per_point();
                let scale = self.settings.render_scale;
                self.edit.request_sam(SamGesture::Point(sam_geom::render_pixel(
                    (pos.x, pos.y),
                    (response.rect.min.x, response.rect.min.y),
                    ppp,
                    scale,
                )));
            }
        }

        // Only a text field should be allowed to eat the movement keys. The
        // broader `egui_wants_keyboard_input` is true whenever ANY widget holds
        // focus, so clicking a checkbox in the panel would have killed WASD.
        if ctx.text_edit_focused() {
            return;
        }

        // `M` is read here rather than in `EditSession::handle_keys` so it
        // works while the edit panels are hidden -- it is the key that shows them.
        let mut edit_mode_toggled = false;
        ctx.input(|i| {
            if i.key_pressed(egui::Key::V) {
                self.mode = self.mode.next();
            }
            if i.key_pressed(egui::Key::X) {
                self.exposure = next_exposure_mode(self.exposure, self.manual_exposure_ev);
            }
            if i.key_pressed(egui::Key::Tab) {
                self.show_panel = !self.show_panel;
            }
            if i.key_pressed(egui::Key::Minus) {
                self.step_scale(-1);
            }
            if i.key_pressed(egui::Key::Equals) || i.key_pressed(egui::Key::Plus) {
                self.step_scale(1);
            }
            if i.key_pressed(egui::Key::F) {
                self.controller.set_mode(self.controller.mode().toggled());
            }
            if i.key_pressed(egui::Key::R) {
                self.controller.reset(&self.views);
            }
            if i.key_pressed(egui::Key::N) {
                self.controller.step_view(&self.views, 1);
            }
            if i.key_pressed(egui::Key::P) {
                self.controller.step_view(&self.views, -1);
            }
            if i.key_pressed(egui::Key::B) {
                self.blend.mode = self.blend.mode.next();
            }
            if i.key_pressed(egui::Key::M) {
                edit_mode_toggled = true;
            }

            let axis = |positive: egui::Key, negative: egui::Key| -> f32 {
                f32::from(i.key_down(positive)) - f32::from(i.key_down(negative))
            };
            let forward = axis(egui::Key::W, egui::Key::S);
            let right = axis(egui::Key::D, egui::Key::A);
            // Q/E move along the scene's up axis; on a Y-down TRIPS scene "up"
            // is -Y, which is why this uses the bundle's vector rather than a
            // hardcoded axis.
            let up = axis(egui::Key::E, egui::Key::Q);
            self.controller.fly(forward, right, up, dt);

            if response.hovered() && i.smooth_scroll_delta.y != 0.0 {
                self.controller.scroll(i.smooth_scroll_delta.y / SCROLL_NOTCH);
            }
        });
        if edit_mode_toggled {
            self.edit.active = !self.edit.active;
        }
        self.edit.handle_keys(ctx, self.controller.scene().diameter());
    }

    fn step_scale(&mut self, direction: i32) {
        let current = SCALE_STEPS
            .iter()
            .position(|s| (s - self.settings.render_scale).abs() < 1e-3)
            .unwrap_or(SCALE_STEPS.len() - 1) as i32;
        let next = (current + direction).clamp(0, SCALE_STEPS.len() as i32 - 1) as usize;
        self.settings.render_scale = SCALE_STEPS[next];
    }

    /// The Blend panel: how much of the frame is splat and how much is TRIPS.
    ///
    /// Drawn only for a bundle that carries a `blend` block -- a hybrid (design
    /// A) run, or any run seeded from a Gaussian `.ply`; on every other scene
    /// this is a no-op and the overlay is unchanged.
    ///
    /// The controls deliberately state what they cannot do, and since v0.6.0
    /// there are two quite different things they might be doing:
    ///
    /// - **live** — `bundle.json`'s `blend.splat_ply` is loaded and Brush's
    ///   `brush-render` rasterises it at the viewer's own pose, so every mode
    ///   works everywhere and the readout says `live`;
    /// - **precomputed** — no ply (or `--no-live-splat`), so the splat half comes
    ///   from renders of the capture views carried in the bundle, and away from
    ///   those views there is no splat to show. The panel says so rather than
    ///   fading to black or reusing a neighbouring view's pixels.
    ///
    /// See `docs/USER_GUIDE.md` "Blend panel".
    fn blend_panel(&mut self, ui: &mut egui::Ui) {
        if !self.has_blend {
            return;
        }
        ui.separator();
        ui.label("Blend (B): splat vs TRIPS");

        let frame_index = self.controller.reference().index;
        let has_splat = self.renderer.has_splat(frame_index);
        let has_gate = self.renderer.has_gate();

        ui.horizontal(|ui| {
            for mode in [
                BlendMode::Trips,
                BlendMode::Splat,
                BlendMode::Gated,
                BlendMode::Mix,
                BlendMode::Split,
            ] {
                let usable = (!mode.needs_splat() || has_splat) && (!mode.needs_gate() || has_gate);
                ui.add_enabled_ui(usable, |ui| {
                    ui.selectable_value(&mut self.blend.mode, mode, mode.label());
                });
            }
        });

        if self.blend.mode.needs_gate() {
            ui.add(
                egui::Slider::new(&mut self.blend.gate_scale, GATE_SCALE_MIN..=GATE_SCALE_MAX)
                    .text("gate scale (0 = TRIPS, 1 = as trained, 2 = splat)"),
            );
        }
        if self.blend.mode == BlendMode::Mix {
            ui.add(
                egui::Slider::new(&mut self.blend.mix, 0.0..=1.0)
                    .text("mix (0 = splat, 1 = TRIPS)"),
            );
        }
        if self.blend.mode == BlendMode::Split {
            ui.add(
                egui::Slider::new(&mut self.blend.split, 0.0..=1.0)
                    .text("split (splat left, TRIPS right)"),
            );
        }

        // What the last frame ACTUALLY drew, which is not always what was asked.
        if let Some(status) = self.last_stats.as_ref().and_then(|s| s.blend) {
            if let Some(note) = status.note() {
                ui.colored_label(egui::Color32::from_rgb(255, 200, 120), note);
            }
        }
        if !has_gate {
            ui.label("no gate head in these weights: the gated blend is unavailable");
        }
        let splat_views = self.renderer.splat_views();
        let source = if self.renderer.has_live_splat() {
            let detail = self
                .renderer
                .live_splat()
                .map_or_else(String::new, |s| format!(" ({} Gaussians)", s.num_splats()));
            format!("yes — rendered LIVE at this pose from blend.splat_ply{detail}")
        } else if has_splat {
            "yes — a precomputed render of this capture view".to_owned()
        } else if !self.controller.is_pinned() {
            "no — you have flown off the capture views, and no ply is loaded".to_owned()
        } else {
            "no — this view has no stored render, and no ply is loaded".to_owned()
        };
        ui.label(format!(
            "splat at this pose: {source}  [{} of {} views carry a precomputed render]",
            splat_views.len(),
            self.views.len()
        ));
    }

    /// One line saying what the edit layer is doing to this frame.
    ///
    /// Always drawn, even with the Edit window closed: an edited scene that
    /// does not say it is edited is exactly the honesty failure `AGENTS.md` §7
    /// is about.
    fn edit_readout(&mut self, ui: &mut egui::Ui) {
        let Some((deleted, touched)) = self.renderer.edit_summary() else {
            if self.edit.doc.regions().is_empty() {
                return;
            }
            ui.separator();
            ui.label("Edit (M): regions loaded, none of them changes this frame");
            return;
        };
        ui.separator();
        ui.label(format!(
            "Edit (M): {} regions | {deleted} points deleted | {touched} points region-mixed{}",
            self.edit.doc.regions().len(),
            if self.edit.dirty { " | UNSAVED (Cmd-S)" } else { "" }
        ));
        if let Some(status) = self.last_stats.as_ref().and_then(|s| s.blend) {
            if status.edit_needs_splat {
                ui.colored_label(
                    egui::Color32::from_rgb(255, 200, 120),
                    "a region asks for the splat (mix < 1) and none is available at this pose:                      the frame is showing unedited TRIPS there",
                );
            }
        }
    }

    /// The overlay: what is being shown, how fast, and the levers.
    fn overlay(&mut self, ui: &mut egui::Ui) {
        let fps = self.fps();
        let heading = match (self.frame_ms(), fps) {
            (Some(ms), Some(fps)) => format!("{ms:.1} ms  ({fps:.1} fps)"),
            _ => "measuring...".to_owned(),
        };
        ui.label(egui::RichText::new(heading).size(18.0).strong());

        if let Some(stats) = self.last_stats {
            ui.label(format!(
                "{}  |  {}x{}  |  {} fragment slots  |  submit {:.1} ms",
                self.mode.label(),
                stats.width,
                stats.height,
                stats.fragment_slots,
                stats.frame_ms
            ));
            if let Some(s) = stats.stages {
                ui.label(format!(
                    "upload {:.1} | project {:.1} | prefix {:.1} | emit {:.1} | sort {:.1} \
                     ({} passes) | segment {:.1} | blend {:.1} ms",
                    s.upload_ms,
                    s.project_count_ms,
                    s.prefix_ms,
                    s.emit_ms,
                    s.sort_ms,
                    s.radix_passes,
                    s.segment_ms,
                    s.blend_ms
                ));
            }
        }
        if let Some(error) = &self.error {
            ui.colored_label(egui::Color32::from_rgb(255, 120, 120), error);
        }

        ui.separator();
        ui.horizontal(|ui| {
            ui.label("view (V):");
            for mode in [ViewMode::Network, ViewMode::RawLevel0, ViewMode::Coverage] {
                ui.selectable_value(&mut self.mode, mode, mode.label());
            }
        });

        ui.add(
            egui::Slider::new(&mut self.settings.render_scale, 0.4..=1.0)
                .text("render scale (-/=)"),
        );
        ui.checkbox(&mut self.settings.packed_sort, "packed 32-bit sort key");
        ui.checkbox(&mut self.settings.cap_fragments, "cap fragments per point");
        ui.checkbox(&mut self.settings.half_features, "f16 features");
        ui.checkbox(&mut self.settings.half_net, "f16 network");
        ui.checkbox(&mut self.settings.profile, "per-stage profile (adds syncs)");
        if self.settings.is_exact() {
            ui.label("exact pipeline");
        } else {
            ui.colored_label(
                egui::Color32::from_rgb(255, 200, 120),
                "approximate: speed levers are on",
            );
        }

        ui.separator();
        ui.horizontal(|ui| {
            ui.label("navigate (F):");
            let mut mode = self.controller.mode();
            let mut changed = false;
            for candidate in [Mode::Orbit, Mode::Free] {
                changed |= ui
                    .selectable_value(&mut mode, candidate, candidate.label())
                    .changed();
            }
            if changed {
                self.controller.set_mode(mode);
            }
        });
        // Speed is quoted twice on purpose: world units per second is what the
        // camera does, and "scenes per second" is what it MEANS. The second
        // number is the same in every scene, which is exactly what the old
        // "fly 1948.53 u/s" readout could not tell anyone.
        let scene = self.controller.scene();
        ui.label(format!(
            "{} points | capture area {:.1} u across | fly {:.3} u/s = {:.3} scene/s ({}) | \
             pivot {:.2} u away",
            self.renderer.num_points(),
            scene.diameter(),
            self.controller.move_speed(),
            self.controller.speed_in_scenes(),
            if self.controller.mode() == Mode::Free {
                "scroll = faster"
            } else {
                "scroll zooms"
            },
            self.controller.orbit_distance(),
        ));
        ui.label(
            "left-drag orbit/look | right- or middle-drag pan | WASD move, Q/E up/down\n\
             scroll = faster (up to 50x; in orbit mode it zooms) | F orbit-free\n\
             R home view | N / P next / previous capture view | V honesty view | X exposure",
        );

        // Exposure is the ONE per-image term of the tone mapper (white balance
        // and vignette are frozen in every config we train), so it is the only
        // thing that makes two views of the same scene differ in brightness,
        // and a free-flown pose has no image whose exposure is "the right
        // one". See docs/LIMITATIONS.md "Per-image exposure".
        ui.horizontal(|ui| {
            ui.label("exposure (X):");
            let manual = ExposureMode::Manual(self.manual_exposure_ev);
            for candidate in [ExposureMode::Auto, ExposureMode::View, ExposureMode::Median, manual] {
                let selected = std::mem::discriminant(&self.exposure) == std::mem::discriminant(&candidate);
                if ui.selectable_label(selected, candidate.label()).clicked() {
                    self.exposure = candidate;
                }
            }
        });
        if let ExposureMode::Manual(_) = self.exposure {
            if ui
                .add(
                    egui::Slider::new(
                        &mut self.manual_exposure_ev,
                        -MANUAL_EXPOSURE_LIMIT..=MANUAL_EXPOSURE_LIMIT,
                    )
                    .text("EV (gain = 2^-EV)"),
                )
                .changed()
            {
                self.exposure = ExposureMode::Manual(self.manual_exposure_ev);
            }
        }
        let applied = self
            .exposure
            .resolve(self.controller.is_pinned(), self.renderer.median_exposure())
            .or_else(|| self.renderer.view_exposure(self.controller.reference().index));
        match applied {
            Some(ev) => ui.label(format!(
                "  applying EV {ev:+.3} (gain {:.2}x){}",
                (-ev).exp2(),
                if self.exposure == ExposureMode::Auto && self.controller.is_pinned() {
                    " — this view's own; move off it for the scene median"
                } else {
                    ""
                }
            )),
            None => ui.label("  this scene has no per-image exposure"),
        };
        if self.controller.is_lost() {
            ui.colored_label(
                egui::Color32::from_rgb(255, 200, 120),
                "you have flown outside the captured area — press R to reset",
            );
        }
        // Orbit mode pins the pivot inside the camera box, so W eventually
        // stops moving. Say so, rather than letting it look like a dead key.
        if self.controller.mode() == Mode::Orbit
            && !scene
                .bounds
                .expanded(EDGE_OF_BOX)
                .contains(self.controller.target())
        {
            ui.colored_label(
                egui::Color32::from_rgb(255, 200, 120),
                "at the edge of the captured area — press F to fly past it",
            );
        }

        self.blend_panel(ui);
        self.edit_readout(ui);

        ui.horizontal(|ui| {
            if ui.button("R: home").clicked() {
                self.controller.reset(&self.views);
            }
            if ui.button("P: prev").clicked() {
                self.controller.step_view(&self.views, -1);
            }
            if ui.button("N: next").clicked() {
                self.controller.step_view(&self.views, 1);
            }
        });
        egui::ComboBox::from_label("jump to view")
            .selected_text(if self.controller.is_pinned() {
                format!("view {}", self.controller.reference().index)
            } else {
                format!("free (from view {})", self.controller.reference().index)
            })
            .show_ui(ui, |ui| {
                // A hundred-plus dataset views: show them all, the combo box
                // scrolls.
                let mut chosen = None;
                for (position, view) in self.views.iter().enumerate() {
                    let label = if view.name.is_empty() {
                        format!("view {}", view.index)
                    } else {
                        format!("{} ({})", view.index, view.name)
                    };
                    let selected = position == self.controller.view_position();
                    if ui.selectable_label(selected, label).clicked() {
                        chosen = Some(position);
                    }
                }
                if let Some(position) = chosen {
                    self.controller.snap_to_position(&self.views, position);
                }
            });
    }
}

impl eframe::App for ViewerApp {
    // eframe 0.36 hands the app a `Ui` rather than calling `update(ctx)`; the
    // `Ui` has no margin or background, which is exactly what a full-bleed
    // render target wants.
    fn ui(&mut self, ui: &mut egui::Ui, _frame: &mut eframe::Frame) {
        let ctx = ui.ctx().clone();
        let ctx = &ctx;
        self.record_interval();
        let dt = self
            .frame_ms()
            .map_or(1.0 / 60.0, |ms| (ms / 1000.0) as f32)
            .clamp(1.0 / 240.0, 0.1);

        let rect = ui.available_rect_before_wrap();
        let response = ui.allocate_rect(rect, egui::Sense::click_and_drag());
        self.handle_input(ctx, &response, dt);

        let ppp = ctx.pixels_per_point();
        let scale = self.settings.render_scale.clamp(0.1, 1.0);
        let width = ((rect.width() * ppp * scale).round() as usize).max(16);
        let height = ((rect.height() * ppp * scale).round() as usize).max(16);

        // The editor's work happens HERE, before the render, and only when a
        // widget or a key asked for it: `refresh_shade` re-thresholds the
        // audit's numbers, `apply` recomposes the per-point weights and
        // re-uploads. Both are no-ops on a frame where nothing changed, which
        // is why flying through an edited scene costs what flying through an
        // unedited one does.
        // The camera is built BEFORE the edit work because a Shift-click is
        // resolved against it: the pixel was recorded in this frame's render
        // coordinates and has to be projected with this frame's camera.
        let reference = self.controller.reference().clone();
        let camera = self.controller.render_camera(width, height, &reference);
        let frame_index = reference.index;

        self.edit
            .resolve_click(&ClickCamera::from_render_camera(&camera), &self.renderer);
        // The SAM tool's own two steps, in the same place and for the same
        // reason: the gesture was measured in this frame's render pixels, so it
        // is mapped with this frame's camera; and the child's state machine is
        // advanced once per frame so its progress reaches the panel while it
        // runs (`docs/EDITOR.md` §3, E5).
        self.edit.resolve_sam(
            &camera,
            &reference,
            self.controller.is_pinned(),
            &self.views,
            self.controller.position,
        );
        if let Some(position) = self.edit.take_sam_snap() {
            self.controller.snap_to_position(&self.views, position);
        }
        self.edit.poll_sam();
        self.edit.refresh_shade(&self.renderer);
        if let Err(e) = self.edit.apply(&mut self.renderer) {
            self.error = Some(e);
        }
        // The renderer cannot see whether the camera is still on its reference
        // view, and `ExposureMode::Auto` is defined in terms of exactly that.
        self.renderer
            .set_exposure(self.exposure, self.controller.is_pinned());

        match block_on(self.renderer.render(
            &camera,
            frame_index,
            self.mode,
            &self.settings,
            self.blend,
        )) {
            Ok(frame) => {
                self.last_stats = Some(frame.stats);
                self.error = None;
                BlitCallback::new(&frame).paint_into(ui, rect);
            }
            Err(message) => self.error = Some(message),
        }

        // The marquee, drawn over the finished frame so it is never part of the
        // render (and so `--screenshot` never contains it).
        if let Some((start, end)) = self.sam_drag {
            let painter = ui.painter_at(rect);
            painter.rect_stroke(
                egui::Rect::from_two_pos(start, end),
                0.0,
                egui::Stroke::new(SAM_MARQUEE_WIDTH, SAM_MARQUEE_COLOUR),
                egui::StrokeKind::Middle,
            );
        }

        if self.edit.active {
            let scene = self.controller.scene();
            let look_at = self.controller.target();
            let diameter = scene.diameter();
            let points = self.renderer.num_points();
            egui::Window::new("Edit (M)")
                .default_pos(rect.min + egui::vec2(rect.width() - 460.0, 12.0))
                .default_width(430.0)
                .resizable(true)
                .show(ctx, |ui| self.edit.ui(ui, look_at, diameter, points));
        }

        if self.show_panel {
            egui::Window::new(format!("TRIPS — {}", self.scene_name))
                .default_pos(rect.min + egui::vec2(12.0, 12.0))
                .resizable(false)
                .show(ctx, |ui| self.overlay(ui));
        } else {
            egui::Area::new(egui::Id::new("trips-readout"))
                .fixed_pos(rect.min + egui::vec2(12.0, 12.0))
                .show(ctx, |ui| {
                    let text = match (self.frame_ms(), self.fps()) {
                        (Some(ms), Some(fps)) => format!(
                            "{ms:.1} ms ({fps:.1} fps) — {} — {}",
                            self.mode.label(),
                            self.controller.mode().label()
                        ),
                        _ => "measuring...".to_owned(),
                    };
                    ui.label(egui::RichText::new(text).size(16.0).strong());
                    if self.controller.is_lost() {
                        ui.colored_label(
                            egui::Color32::from_rgb(255, 200, 120),
                            "outside the captured area — press R to reset",
                        );
                    }
                });
        }

        // The renderer only produces a frame when asked, and flying needs a
        // continuous stream, so never idle.
        ctx.request_repaint();
    }
}

/// Cycle order for the `X` key: auto -> view -> median -> manual -> auto.
///
/// A free function so the order is testable without an egui context.
#[must_use]
fn next_exposure_mode(current: ExposureMode, manual_ev: f32) -> ExposureMode {
    match current {
        ExposureMode::Auto => ExposureMode::View,
        ExposureMode::View => ExposureMode::Median,
        ExposureMode::Median => ExposureMode::Manual(manual_ev),
        ExposureMode::Manual(_) => ExposureMode::Auto,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_x_key_cycles_every_exposure_mode_and_returns() {
        let mut mode = ExposureMode::default();
        assert_eq!(mode, ExposureMode::Auto);
        mode = next_exposure_mode(mode, 1.5);
        assert_eq!(mode, ExposureMode::View);
        mode = next_exposure_mode(mode, 1.5);
        assert_eq!(mode, ExposureMode::Median);
        mode = next_exposure_mode(mode, 1.5);
        assert_eq!(mode, ExposureMode::Manual(1.5));
        mode = next_exposure_mode(mode, 1.5);
        assert_eq!(mode, ExposureMode::Auto);
    }
}
