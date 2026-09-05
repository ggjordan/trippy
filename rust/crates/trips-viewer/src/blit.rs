//! Putting a Burn/CubeCL buffer on screen through egui, with no copy.
//!
//! Module: `trips_viewer::blit`
//! Purpose: the last millimetre of the pipeline. `brush-pyramid` composites
//!     into a `CubeTensor<WgpuRuntime>` and `brush-unet` produces a
//!     `Tensor<4>` on the same device; this binds whichever buffer the current
//!     view mode selected straight into egui's render pass as a read-only
//!     storage buffer, so the frame never leaves the GPU.
//! Invariants:
//!     - Burn and eframe **share one `wgpu::Device`** (see
//!       [`crate::main`]'s `burn_wgpu::init_device` call). Binding a buffer
//!       from one device into another device's pipeline is undefined and wgpu
//!       will refuse it; this whole module is only sound because of that.
//!     - The bind group is rebuilt every `prepare`, because a new frame is a
//!       new allocation. Cheap (a bind group is a descriptor, not memory) and
//!       the alternative — caching by buffer identity — would keep the
//!       previous frame's allocation alive.
//!     - `as_wgpu_bind_resource()` is used rather than `as_entire_binding()`:
//!       CubeCL's allocator may hand back a *view* into a larger page, and
//!       the resource knows its own offset and size.
//! Related docs: `apps/brush-app/src/ui/splat_backbuffer.rs` in the Brush fork
//!     (Apache-2.0, ArthurBrussee) — the same technique, for its RGBA8 splat
//!     backbuffer; `docs/decisions/ADR-0006-viewer-integration.md`.

use burn_wgpu::{CubeTensor, WgpuRuntime};
use eframe::egui_wgpu::{self, wgpu, CallbackTrait};

use crate::renderer::RenderedFrame;

/// Mirror of `blit.wgsl`'s `Uniforms`. `repr(C)` and 16-byte sized, which is
/// the minimum uniform buffer binding size WebGPU guarantees.
#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
struct Uniforms {
    img_width: u32,
    img_height: u32,
    channels: u32,
    mode: u32,
}

/// The pipeline and uniform buffer, created once and stashed in egui's
/// `callback_resources`.
pub struct BlitResources {
    pipeline: wgpu::RenderPipeline,
    uniform_buffer: wgpu::Buffer,
    bind_group_layout: wgpu::BindGroupLayout,
    /// Rebuilt every `prepare` against the current frame's buffer.
    bind_group: Option<wgpu::BindGroup>,
}

impl BlitResources {
    /// Build the pipeline for `target_format` and register it.
    ///
    /// # Arguments
    /// - `state`: eframe's render state, which owns the shared device.
    pub fn install(state: &eframe::egui_wgpu::RenderState) {
        let resources = Self::new(&state.device, state.target_format);
        state
            .renderer
            .write()
            .callback_resources
            .insert(resources);
    }

    fn new(device: &wgpu::Device, target_format: wgpu::TextureFormat) -> Self {
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("trips blit shader"),
            source: wgpu::ShaderSource::Wgsl(include_str!("shaders/blit.wgsl").into()),
        });
        let uniform_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("trips blit uniforms"),
            size: std::mem::size_of::<Uniforms>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("trips blit layout"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::VERTEX | wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
            ],
        });
        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("trips blit pipeline layout"),
            bind_group_layouts: &[Some(&bind_group_layout)],
            immediate_size: 0,
        });
        let pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("trips blit pipeline"),
            layout: Some(&pipeline_layout),
            vertex: wgpu::VertexState {
                module: &shader,
                entry_point: Some("vs_main"),
                // Fullscreen triangle: no vertex buffers.
                buffers: &[],
                compilation_options: wgpu::PipelineCompilationOptions::default(),
            },
            fragment: Some(wgpu::FragmentState {
                module: &shader,
                entry_point: Some("fs_main"),
                targets: &[Some(wgpu::ColorTargetState {
                    format: target_format,
                    blend: Some(wgpu::BlendState::ALPHA_BLENDING),
                    write_mask: wgpu::ColorWrites::ALL,
                })],
                compilation_options: wgpu::PipelineCompilationOptions::default(),
            }),
            primitive: wgpu::PrimitiveState {
                topology: wgpu::PrimitiveTopology::TriangleList,
                strip_index_format: None,
                front_face: wgpu::FrontFace::Ccw,
                cull_mode: None,
                unclipped_depth: false,
                polygon_mode: wgpu::PolygonMode::Fill,
                conservative: false,
            },
            depth_stencil: None,
            multisample: wgpu::MultisampleState::default(),
            cache: None,
            multiview_mask: None,
        });
        Self {
            pipeline,
            uniform_buffer,
            bind_group_layout,
            bind_group: None,
        }
    }
}

/// One frame's paint callback.
pub struct BlitCallback {
    buffer: CubeTensor<WgpuRuntime>,
    uniforms: Uniforms,
}

impl BlitCallback {
    /// Wrap a finished frame for painting into `rect`.
    #[must_use]
    pub fn new(frame: &RenderedFrame) -> Self {
        Self {
            buffer: frame.buffer.clone(),
            uniforms: Uniforms {
                img_width: frame.width,
                img_height: frame.height,
                channels: frame.channels,
                mode: frame.mode.shader_code(),
            },
        }
    }

    /// Add this frame to `ui`'s painter.
    pub fn paint_into(self, ui: &egui::Ui, rect: egui::Rect) {
        ui.painter()
            .add(eframe::egui_wgpu::Callback::new_paint_callback(rect, self));
    }
}

impl CallbackTrait for BlitCallback {
    fn prepare(
        &self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        _screen: &egui_wgpu::ScreenDescriptor,
        _encoder: &mut wgpu::CommandEncoder,
        resources: &mut egui_wgpu::CallbackResources,
    ) -> Vec<wgpu::CommandBuffer> {
        let Some(res) = resources.get_mut::<BlitResources>() else {
            return Vec::new();
        };
        queue.write_buffer(
            &res.uniform_buffer,
            0,
            bytemuck::cast_slice(&[self.uniforms]),
        );

        let Ok(handle) = self.buffer.client.get_resource(self.buffer.handle.clone()) else {
            // The allocation went away between render and paint. Skip this
            // frame rather than tear down the app; the next one repaints.
            res.bind_group = None;
            return Vec::new();
        };
        let resource = handle.resource();
        res.bind_group = Some(device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("trips blit bind group"),
            layout: &res.bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: res.uniform_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: resource.as_wgpu_bind_resource(),
                },
            ],
        }));
        Vec::new()
    }

    fn paint(
        &self,
        _info: egui::PaintCallbackInfo,
        render_pass: &mut wgpu::RenderPass<'static>,
        resources: &egui_wgpu::CallbackResources,
    ) {
        let Some(res) = resources.get::<BlitResources>() else {
            return;
        };
        let Some(bind_group) = res.bind_group.as_ref() else {
            return;
        };
        render_pass.set_pipeline(&res.pipeline);
        render_pass.set_bind_group(0, bind_group, &[]);
        render_pass.draw(0..3, 0..1);
    }
}

/// The blit shader's mode codes, kept next to the enum they mirror so a new
/// view mode cannot silently render as an old one.
#[cfg(test)]
mod tests {
    use super::*;
    use crate::renderer::ViewMode;

    #[test]
    fn shader_codes_match_the_wgsl_constants() {
        let wgsl = include_str!("shaders/blit.wgsl");
        for (mode, name) in [
            (ViewMode::Network, "MODE_NETWORK"),
            (ViewMode::RawLevel0, "MODE_RAW"),
            (ViewMode::Coverage, "MODE_COVERAGE"),
        ] {
            let needle = format!("const {name}: u32 = {}u;", mode.shader_code());
            assert!(wgsl.contains(&needle), "blit.wgsl is missing `{needle}`");
        }
    }

    #[test]
    fn the_uniform_block_is_sixteen_bytes() {
        // WebGPU's minimum guaranteed uniform buffer binding size.
        assert_eq!(std::mem::size_of::<Uniforms>(), 16);
    }
}
