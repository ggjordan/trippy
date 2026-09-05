//! Putting a Burn/CubeCL buffer on the canvas, with no copy.
//!
//! Module: `trips_web::blit`
//! Purpose: the web twin of `trips-viewer`'s `src/blit.rs`. Same shader
//!     (`trips_viewer::BLIT_WGSL`), same uniform block, same "bind the
//!     rasteriser's own allocation as a read-only storage buffer" trick —
//!     minus egui, because there is no egui here. It owns a plain
//!     `wgpu::RenderPipeline` and draws one fullscreen triangle into the
//!     surface texture.
//! Invariants:
//!     - Burn and this pipeline **share one `wgpu::Device`** (see
//!       [`crate::gpu`]). The whole module is only sound because of that.
//!     - The bind group is rebuilt every frame, because a new frame is a new
//!       allocation. A bind group is a descriptor, not memory.
//!     - `as_wgpu_bind_resource()` rather than `as_entire_binding()`: CubeCL's
//!       allocator may hand back a *view* into a larger page, and only the
//!       resource knows its own offset and size.
//!     - The uniform block is 16 bytes, WebGPU's minimum guaranteed uniform
//!       buffer binding size, and its lane order must match `blit.wgsl`'s
//!       `Uniforms` exactly. `tests::the_uniform_block_is_sixteen_bytes` and
//!       the native crate's `shader_codes_match_the_wgsl_constants` pin both
//!       halves.
//! Related docs: `docs/WEB_VIEWER.md`; `rust/crates/trips-viewer/src/blit.rs`.

use burn_wgpu::{CubeTensor, WgpuRuntime};
use trips_viewer::renderer::RenderedFrame;

/// Mirror of `blit.wgsl`'s `Uniforms`.
#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
struct Uniforms {
    img_width: u32,
    img_height: u32,
    channels: u32,
    mode: u32,
}

/// The blit pipeline and its uniform buffer, created once.
pub struct Blit {
    pipeline: wgpu::RenderPipeline,
    uniform_buffer: wgpu::Buffer,
    bind_group_layout: wgpu::BindGroupLayout,
}

impl Blit {
    /// Build the pipeline for `target_format`.
    ///
    /// # Arguments
    /// - `device`: the shared wgpu device.
    /// - `target_format`: the surface's texture format.
    #[must_use]
    pub fn new(device: &wgpu::Device, target_format: wgpu::TextureFormat) -> Self {
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("trips blit shader"),
            // The *same string* the native viewer compiles, so the two front
            // ends cannot disagree about how a frame is laid out.
            source: wgpu::ShaderSource::Wgsl(trips_viewer::BLIT_WGSL.into()),
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
                buffers: &[],
                compilation_options: wgpu::PipelineCompilationOptions::default(),
            },
            fragment: Some(wgpu::FragmentState {
                module: &shader,
                entry_point: Some("fs_main"),
                targets: &[Some(wgpu::ColorTargetState {
                    format: target_format,
                    // The canvas is opaque and this is the only draw, so there
                    // is nothing to blend against; REPLACE keeps the written
                    // bytes byte-identical to what the shader produced, which
                    // is what the screenshot PSNR check depends on.
                    blend: Some(wgpu::BlendState::REPLACE),
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
        }
    }

    /// Draw `frame` into `view`, filling it.
    ///
    /// # Arguments
    /// - `device`, `queue`: the shared device and its queue.
    /// - `view`: the surface texture's view for this frame.
    /// - `frame`: the finished, still-device-resident TRIPS frame.
    ///
    /// # Errors
    /// Returns `Err` if the frame's allocation could not be resolved to a
    /// bindable resource, which means it was freed between render and paint.
    pub fn draw(
        &self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        view: &wgpu::TextureView,
        frame: &RenderedFrame,
    ) -> Result<(), String> {
        let uniforms = Uniforms {
            img_width: frame.width,
            img_height: frame.height,
            channels: frame.channels,
            mode: frame.mode.shader_code(),
        };
        queue.write_buffer(&self.uniform_buffer, 0, bytemuck::cast_slice(&[uniforms]));

        let bind_group = self.bind_group(device, &frame.buffer)?;

        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("trips blit encoder"),
        });
        {
            let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("trips blit pass"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view,
                    depth_slice: None,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(wgpu::Color::BLACK),
                        store: wgpu::StoreOp::Store,
                    },
                })],
                depth_stencil_attachment: None,
                timestamp_writes: None,
                occlusion_query_set: None,
                multiview_mask: None,
            });
            pass.set_pipeline(&self.pipeline);
            pass.set_bind_group(0, &bind_group, &[]);
            pass.draw(0..3, 0..1);
        }
        queue.submit(Some(encoder.finish()));
        Ok(())
    }

    fn bind_group(
        &self,
        device: &wgpu::Device,
        buffer: &CubeTensor<WgpuRuntime>,
    ) -> Result<wgpu::BindGroup, String> {
        let handle = buffer
            .client
            .get_resource(buffer.handle.clone())
            .map_err(|e| format!("the frame's buffer went away before it could be drawn: {e:?}"))?;
        Ok(device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("trips blit bind group"),
            layout: &self.bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: self.uniform_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: handle.resource().as_wgpu_bind_resource(),
                },
            ],
        }))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_uniform_block_is_sixteen_bytes() {
        // WebGPU's minimum guaranteed uniform buffer binding size.
        assert_eq!(std::mem::size_of::<Uniforms>(), 16);
    }

    #[test]
    fn the_shader_is_the_native_viewers_shader() {
        // Not a copy: the same `&'static str`. If this ever becomes two files
        // the screenshot PSNR check stops proving what it claims to prove.
        assert!(trips_viewer::BLIT_WGSL.contains("const MODE_NETWORK: u32 = 0u;"));
        assert!(trips_viewer::BLIT_WGSL.contains("fn fs_main"));
    }
}
