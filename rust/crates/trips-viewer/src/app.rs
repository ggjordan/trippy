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
use crate::bundle::Bundle;
use crate::camera::{Controller, Mode};
use crate::renderer::{Renderer, Settings, ViewMode};

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

/// The viewer.
pub struct ViewerApp {
    renderer: Renderer,
    controller: Controller,
    views: Vec<crate::bundle::BundleView>,
    scene_name: String,
    mode: ViewMode,
    settings: Settings,
    show_panel: bool,
    intervals: std::collections::VecDeque<f64>,
    last_frame: Option<std::time::Instant>,
    last_stats: Option<crate::renderer::FrameStats>,
    error: Option<String>,
}

impl ViewerApp {
    /// Build the app, sharing eframe's wgpu device with Burn.
    ///
    /// # Arguments
    /// - `cc`: eframe's creation context; must be the wgpu backend.
    /// - `bundle`: the loaded scene.
    /// - `settings`: initial performance levers (from the command line).
    ///
    /// # Errors
    /// Returns `Err` if eframe is not on wgpu, or the weights are rejected.
    pub fn new(
        cc: &eframe::CreationContext<'_>,
        bundle: Bundle,
        settings: Settings,
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

        let renderer = Renderer::new(bundle, burn_device)?;
        let controller = Controller::new(&views, home, up);

        cc.egui_ctx
            .options_mut(|o| o.theme_preference = egui::ThemePreference::Dark);

        Ok(Self {
            renderer,
            controller,
            views,
            scene_name,
            mode: ViewMode::default(),
            settings,
            show_panel: true,
            intervals: std::collections::VecDeque::with_capacity(FPS_WINDOW),
            last_frame: None,
            last_stats: None,
            error: None,
        })
    }

    /// Set the initial view mode (from `--mode`).
    pub fn set_mode(&mut self, mode: ViewMode) {
        self.mode = mode;
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
        if response.dragged_by(egui::PointerButton::Primary) {
            self.controller.drag(delta.x, delta.y);
        } else if response.dragged_by(egui::PointerButton::Secondary)
            || response.dragged_by(egui::PointerButton::Middle)
        {
            self.controller.pan(delta.x, delta.y, viewport_height);
        }
        if response.dragged() {
            ctx.set_cursor_icon(egui::CursorIcon::Grabbing);
        } else if response.hovered() {
            ctx.set_cursor_icon(egui::CursorIcon::Grab);
        }

        // Only a text field should be allowed to eat the movement keys. The
        // broader `egui_wants_keyboard_input` is true whenever ANY widget holds
        // focus, so clicking a checkbox in the panel would have killed WASD.
        if ctx.text_edit_focused() {
            return;
        }

        ctx.input(|i| {
            if i.key_pressed(egui::Key::V) {
                self.mode = self.mode.next();
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
    }

    fn step_scale(&mut self, direction: i32) {
        let current = SCALE_STEPS
            .iter()
            .position(|s| (s - self.settings.render_scale).abs() < 1e-3)
            .unwrap_or(SCALE_STEPS.len() - 1) as i32;
        let next = (current + direction).clamp(0, SCALE_STEPS.len() as i32 - 1) as usize;
        self.settings.render_scale = SCALE_STEPS[next];
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
                "scroll changes it"
            } else {
                "scroll zooms"
            },
            self.controller.orbit_distance(),
        ));
        ui.label(
            "left-drag orbit/look | right- or middle-drag pan | WASD move, Q/E up/down\n\
             R home view | N / P next / previous capture view | F orbit-free | V honesty view",
        );
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

        let reference = self.controller.reference().clone();
        let camera = self.controller.render_camera(width, height, &reference);
        let frame_index = reference.index;

        match block_on(self.renderer.render(
            &camera,
            frame_index,
            self.mode,
            &self.settings,
        )) {
            Ok(frame) => {
                self.last_stats = Some(frame.stats);
                self.error = None;
                BlitCallback::new(&frame).paint_into(ui, rect);
            }
            Err(message) => self.error = Some(message),
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
