// tiled-fork: begin example
//! MSAA line demo routed through typed subpass inputs.
//!
//! Two subpasses share one render pass:
//!   Subpass 0 (gbuffer):  draws the line list into an MSAA color target.
//!   Subpass 1 (present):  reads that target via `subpass_input` /
//!                         `subpass_input_multisampled` and writes the
//!                         present output.
//!
//! When `sample_count > 1` the present subpass writes into an
//! intermediate MSAA texture and a follow-up regular render pass resolves
//! that texture into the sRGB swapchain via `textureLoad` averaging in
//! `resolve.wgsl`. When `sample_count == 1` the present subpass writes
//! directly into the swapchain and the resolve pass is skipped.
//!
//! Left/Right arrow keys toggle between 1x and the adapter-maximum MSAA
//! sample count.
//!
//! Adapted from `infosia/wgpu-tiled`'s example of the same name. The
//! reference uses `RenderGraphBuilder` and extended `Limits`
//! (`max_subpasses`, `max_subpass_color_attachments`,
//! `max_input_attachments`); this fork constructs the
//! `SubpassRenderPassDescriptor` literally and uses the Phase-11h
//! `SubpassRenderPipelineDescriptor::new` to build subpass-aware
//! pipelines. The WGSL shaders are copied verbatim.

use bytemuck::{Pod, Zeroable};
use std::borrow::Cow;
use wgpu::util::DeviceExt;
use winit::{
    event::{ElementState, KeyEvent, WindowEvent},
    keyboard::{Key, NamedKey},
};

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct Vertex {
    pos: [f32; 2],
    color: [f32; 4],
}

struct ActivePipelines {
    gbuffer_pipeline: wgpu::RenderPipeline,
    present_pipeline: wgpu::RenderPipeline,
    present_bind_group: wgpu::BindGroup,
    resolve_pipeline: Option<wgpu::RenderPipeline>,
    resolve_bind_group: Option<wgpu::BindGroup>,
    _lines_texture: wgpu::Texture,
    lines_view: wgpu::TextureView,
    _output_msaa_texture: Option<wgpu::Texture>,
    output_msaa_view: Option<wgpu::TextureView>,
}

struct Example {
    config: wgpu::SurfaceConfiguration,
    gbuffer_shader: wgpu::ShaderModule,
    present_shader: wgpu::ShaderModule,
    resolve_shader: wgpu::ShaderModule,
    vertex_buffer: wgpu::Buffer,
    vertex_count: u32,
    sample_count: u32,
    max_sample_count: u32,
    rebuild: bool,
    active: ActivePipelines,
}

fn output_format(config: &wgpu::SurfaceConfiguration) -> wgpu::TextureFormat {
    config
        .view_formats
        .first()
        .copied()
        .unwrap_or(config.format)
}

fn create_line_vertices() -> Vec<Vertex> {
    let mut vertex_data = Vec::new();
    let max = 50;
    for i in 0..max {
        let percent = i as f32 / max as f32;
        let (sin, cos) = (percent * 2.0 * core::f32::consts::PI).sin_cos();
        vertex_data.push(Vertex {
            pos: [0.0, 0.0],
            color: [1.0, -sin, cos, 1.0],
        });
        vertex_data.push(Vertex {
            pos: [cos, sin],
            color: [sin, -cos, 1.0, 1.0],
        });
    }
    vertex_data
}

fn max_sample_count(adapter: &wgpu::Adapter, format: wgpu::TextureFormat) -> u32 {
    let flags = adapter.get_texture_format_features(format).flags;
    if flags.contains(wgpu::TextureFormatFeatureFlags::MULTISAMPLE_X16) {
        16
    } else if flags.contains(wgpu::TextureFormatFeatureFlags::MULTISAMPLE_X8) {
        8
    } else if flags.contains(wgpu::TextureFormatFeatureFlags::MULTISAMPLE_X4) {
        4
    } else if flags.contains(wgpu::TextureFormatFeatureFlags::MULTISAMPLE_X2) {
        2
    } else {
        1
    }
}

fn build_subpass_target_base(
    output_format: wgpu::TextureFormat,
) -> wgpu::SubpassTarget {
    wgpu::SubpassTarget {
        index: 0,
        color_attachment_formats: vec![Some(output_format), Some(output_format)],
        depth_stencil_format: None,
        subpass_descs: vec![
            wgpu::SubpassTargetDesc {
                color_attachment_indices: vec![0],
                uses_depth_stencil: false,
                input_attachment_indices: vec![],
            },
            wgpu::SubpassTargetDesc {
                color_attachment_indices: vec![1],
                uses_depth_stencil: false,
                input_attachment_indices: vec![0],
            },
        ],
        dependencies: vec![wgpu::SubpassDependency {
            src_subpass: wgpu::SubpassIndex(0),
            dst_subpass: wgpu::SubpassIndex(1),
            dependency_type: wgpu::SubpassDependencyType::ColorToInput,
            by_region: true,
        }],
    }
}

fn create_active_pipelines(
    device: &wgpu::Device,
    config: &wgpu::SurfaceConfiguration,
    sample_count: u32,
    gbuffer_shader: &wgpu::ShaderModule,
    present_shader: &wgpu::ShaderModule,
    resolve_shader: &wgpu::ShaderModule,
) -> ActivePipelines {
    let format = output_format(config);
    let subpass_target_base = build_subpass_target_base(format);

    let gbuffer_base = wgpu::RenderPipelineDescriptor {
        label: Some("subpass_msaa gbuffer_pipeline"),
        layout: None,
        vertex: wgpu::VertexState {
            module: gbuffer_shader,
            entry_point: Some("vs_main"),
            compilation_options: Default::default(),
            buffers: &[wgpu::VertexBufferLayout {
                array_stride: core::mem::size_of::<Vertex>() as wgpu::BufferAddress,
                step_mode: wgpu::VertexStepMode::Vertex,
                attributes: &wgpu::vertex_attr_array![0 => Float32x2, 1 => Float32x4],
            }],
        },
        fragment: Some(wgpu::FragmentState {
            module: gbuffer_shader,
            entry_point: Some("fs_main"),
            compilation_options: Default::default(),
            targets: &[Some(format.into())],
        }),
        primitive: wgpu::PrimitiveState {
            topology: wgpu::PrimitiveTopology::LineList,
            front_face: wgpu::FrontFace::Ccw,
            ..Default::default()
        },
        depth_stencil: None,
        multisample: wgpu::MultisampleState {
            count: sample_count,
            ..Default::default()
        },
        multiview_mask: None,
        cache: None,
    };
    let gbuffer_pipeline = device.create_subpass_render_pipeline(
        &wgpu::SubpassRenderPipelineDescriptor::new(
            gbuffer_base,
            wgpu::SubpassTarget {
                index: 0,
                ..subpass_target_base.clone()
            },
        ),
    );

    let present_entry_point = if sample_count == 1 {
        "fs_main_1x"
    } else {
        "fs_main_msaa"
    };
    let present_base = wgpu::RenderPipelineDescriptor {
        label: Some("subpass_msaa present_pipeline"),
        layout: None,
        vertex: wgpu::VertexState {
            module: present_shader,
            entry_point: Some("vs_main"),
            compilation_options: Default::default(),
            buffers: &[],
        },
        fragment: Some(wgpu::FragmentState {
            module: present_shader,
            entry_point: Some(present_entry_point),
            compilation_options: Default::default(),
            targets: &[Some(format.into())],
        }),
        primitive: Default::default(),
        depth_stencil: None,
        multisample: wgpu::MultisampleState {
            count: sample_count,
            ..Default::default()
        },
        multiview_mask: None,
        cache: None,
    };
    let present_pipeline = device.create_subpass_render_pipeline(
        &wgpu::SubpassRenderPipelineDescriptor::new(
            present_base,
            wgpu::SubpassTarget {
                index: 1,
                ..subpass_target_base
            },
        ),
    );

    let extent = wgpu::Extent3d {
        width: config.width,
        height: config.height,
        depth_or_array_layers: 1,
    };
    let lines_texture = device.create_texture(&wgpu::TextureDescriptor {
        label: Some("subpass_msaa lines_color"),
        size: extent,
        mip_level_count: 1,
        sample_count,
        dimension: wgpu::TextureDimension::D2,
        format,
        // Persistent (DRAM-backed) attachment; the public-API
        // `SubpassColorAttachment::Transient` path is still
        // `TransientNotWired` in this fork.
        usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
        view_formats: &[],
    });
    let lines_view = lines_texture.create_view(&wgpu::TextureViewDescriptor::default());

    let present_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("subpass_msaa present_bind_group"),
        layout: &present_pipeline.get_bind_group_layout(0),
        entries: &[wgpu::BindGroupEntry {
            binding: 0,
            resource: wgpu::BindingResource::TextureView(&lines_view),
        }],
    });

    let (output_msaa_texture, output_msaa_view, resolve_pipeline, resolve_bind_group) =
        if sample_count == 1 {
            (None, None, None, None)
        } else {
            let texture = device.create_texture(&wgpu::TextureDescriptor {
                label: Some("subpass_msaa output_msaa"),
                size: extent,
                mip_level_count: 1,
                sample_count,
                dimension: wgpu::TextureDimension::D2,
                format,
                usage: wgpu::TextureUsages::RENDER_ATTACHMENT
                    | wgpu::TextureUsages::TEXTURE_BINDING,
                view_formats: &[],
            });
            let view = texture.create_view(&wgpu::TextureViewDescriptor::default());

            let pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
                label: Some("subpass_msaa resolve_pipeline"),
                layout: None,
                vertex: wgpu::VertexState {
                    module: resolve_shader,
                    entry_point: Some("vs_main"),
                    compilation_options: Default::default(),
                    buffers: &[],
                },
                fragment: Some(wgpu::FragmentState {
                    module: resolve_shader,
                    entry_point: Some("fs_main"),
                    compilation_options: Default::default(),
                    targets: &[Some(format.into())],
                }),
                primitive: Default::default(),
                depth_stencil: None,
                multisample: Default::default(),
                multiview_mask: None,
                cache: None,
            });
            let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("subpass_msaa resolve_bind_group"),
                layout: &pipeline.get_bind_group_layout(0),
                entries: &[wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::TextureView(&view),
                }],
            });

            (Some(texture), Some(view), Some(pipeline), Some(bind_group))
        };

    ActivePipelines {
        gbuffer_pipeline,
        present_pipeline,
        present_bind_group,
        resolve_pipeline,
        resolve_bind_group,
        _lines_texture: lines_texture,
        lines_view,
        _output_msaa_texture: output_msaa_texture,
        output_msaa_view,
    }
}

impl crate::framework::Example for Example {
    fn optional_features() -> wgpu::Features {
        wgpu::Features::TEXTURE_ADAPTER_SPECIFIC_FORMAT_FEATURES
    }

    fn required_features() -> wgpu::Features {
        wgpu::Features::MULTI_SUBPASS
    }

    fn required_limits() -> wgpu::Limits {
        wgpu::Limits::downlevel_webgl2_defaults()
    }

    fn required_downlevel_capabilities() -> wgpu::DownlevelCapabilities {
        wgpu::DownlevelCapabilities {
            flags: wgpu::DownlevelFlags::MULTISAMPLED_SHADING,
            shader_model: wgpu::ShaderModel::Sm5,
            ..wgpu::DownlevelCapabilities::default()
        }
    }

    fn init(
        config: &wgpu::SurfaceConfiguration,
        adapter: &wgpu::Adapter,
        device: &wgpu::Device,
        _queue: &wgpu::Queue,
    ) -> Self {
        log::info!("Press left/right arrow keys to toggle sample_count.");

        let format = output_format(config);
        let max_sample_count = max_sample_count(adapter, format);
        let sample_count = max_sample_count;

        let gbuffer_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("subpass_msaa gbuffer_shader"),
            source: wgpu::ShaderSource::Wgsl(Cow::Borrowed(include_str!("gbuffer.wgsl"))),
        });
        let present_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("subpass_msaa present_shader"),
            source: wgpu::ShaderSource::Wgsl(Cow::Borrowed(include_str!("present.wgsl"))),
        });
        let resolve_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("subpass_msaa resolve_shader"),
            source: wgpu::ShaderSource::Wgsl(Cow::Borrowed(include_str!("resolve.wgsl"))),
        });

        let vertex_data = create_line_vertices();
        let vertex_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("subpass_msaa vertex_buffer"),
            contents: bytemuck::cast_slice(&vertex_data),
            usage: wgpu::BufferUsages::VERTEX,
        });
        let vertex_count = vertex_data.len() as u32;

        let active = create_active_pipelines(
            device,
            config,
            sample_count,
            &gbuffer_shader,
            &present_shader,
            &resolve_shader,
        );

        Self {
            config: config.clone(),
            gbuffer_shader,
            present_shader,
            resolve_shader,
            vertex_buffer,
            vertex_count,
            sample_count,
            max_sample_count,
            rebuild: false,
            active,
        }
    }

    fn update(&mut self, event: WindowEvent) {
        if let WindowEvent::KeyboardInput {
            event:
                KeyEvent {
                    logical_key,
                    state: ElementState::Pressed,
                    ..
                },
            ..
        } = event
        {
            match logical_key {
                Key::Named(NamedKey::ArrowLeft) => {
                    if self.sample_count == self.max_sample_count {
                        self.sample_count = 1;
                        self.rebuild = true;
                    }
                }
                Key::Named(NamedKey::ArrowRight) => {
                    if self.sample_count == 1 {
                        self.sample_count = self.max_sample_count;
                        self.rebuild = true;
                    }
                }
                _ => {}
            }
        }
    }

    fn resize(
        &mut self,
        config: &wgpu::SurfaceConfiguration,
        _device: &wgpu::Device,
        _queue: &wgpu::Queue,
    ) {
        self.config = config.clone();
        self.rebuild = true;
    }

    fn render(&mut self, view: &wgpu::TextureView, device: &wgpu::Device, queue: &wgpu::Queue) {
        if self.rebuild {
            self.active = create_active_pipelines(
                device,
                &self.config,
                self.sample_count,
                &self.gbuffer_shader,
                &self.present_shader,
                &self.resolve_shader,
            );
            self.rebuild = false;
        }

        let extent = wgpu::Extent3d {
            width: self.config.width,
            height: self.config.height,
            depth_or_array_layers: 1,
        };

        // Output target for the present subpass:
        //   1x:  the swapchain view directly.
        //   Nx:  an MSAA intermediate; a follow-up regular pass resolves
        //        that intermediate into the swapchain.
        // The `output_msaa_view` invariant (Some iff sample_count > 1) is
        // enforced in `create_active_pipelines`; if it ever desynchronizes
        // we skip the frame with a warning rather than panic.
        let output_view = if self.sample_count == 1 {
            view
        } else {
            match self.active.output_msaa_view.as_ref() {
                Some(v) => v,
                None => {
                    log::warn!("subpass_msaa: output_msaa_view missing at sample_count > 1; skipping frame");
                    return;
                }
            }
        };
        let output_attachment = wgpu::RenderPassColorAttachment {
            view: output_view,
            depth_slice: None,
            resolve_target: None,
            ops: wgpu::Operations {
                load: wgpu::LoadOp::Clear(wgpu::Color::BLACK),
                store: wgpu::StoreOp::Store,
            },
        };

        let lines_attachment = wgpu::RenderPassColorAttachment {
            view: &self.active.lines_view,
            depth_slice: None,
            resolve_target: None,
            ops: wgpu::Operations {
                load: wgpu::LoadOp::Clear(wgpu::Color::BLACK),
                store: wgpu::StoreOp::Discard,
            },
        };

        let subpasses = [
            wgpu::SubpassDescriptor {
                color_attachments: &[Some(wgpu::SubpassColorAttachment::Persistent(
                    lines_attachment.clone(),
                ))],
                color_attachment_indices: &[0],
                depth_stencil_attachment: None,
                input_attachments: &[],
            },
            wgpu::SubpassDescriptor {
                color_attachments: &[Some(wgpu::SubpassColorAttachment::Persistent(
                    output_attachment.clone(),
                ))],
                color_attachment_indices: &[1],
                depth_stencil_attachment: None,
                input_attachments: &[wgpu::SubpassInputAttachment {
                    binding: 0,
                    source: wgpu::SubpassInputSource::Color {
                        subpass: wgpu::SubpassIndex(0),
                        attachment_index: 0,
                    },
                }],
            },
        ];

        let subpass_dependencies = [wgpu::SubpassDependency {
            src_subpass: wgpu::SubpassIndex(0),
            dst_subpass: wgpu::SubpassIndex(1),
            dependency_type: wgpu::SubpassDependencyType::ColorToInput,
            by_region: true,
        }];

        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("subpass_msaa encoder"),
        });

        {
            let mut pass = encoder.begin_subpass_render_pass(&wgpu::SubpassRenderPassDescriptor {
                label: Some("subpass_msaa pass"),
                extent,
                sample_count: self.sample_count,
                color_attachments: &[Some(lines_attachment), Some(output_attachment)],
                depth_stencil_attachment: None,
                subpasses: &subpasses,
                subpass_dependencies: &subpass_dependencies,
                transient_memory_hint: wgpu::TransientMemoryHint::Auto,
                active_subpass_mask: None,
                multiview_mask: None,
                timestamp_writes: None,
                occlusion_query_set: None,
            });

            pass.set_pipeline(&self.active.gbuffer_pipeline);
            pass.set_vertex_buffer(0, self.vertex_buffer.slice(..));
            pass.draw(0..self.vertex_count, 0..1);

            pass.next_subpass();
            pass.set_pipeline(&self.active.present_pipeline);
            pass.set_bind_group(0, &self.active.present_bind_group, &[]);
            pass.draw(0..3, 0..1);
        }

        if self.sample_count > 1 {
            let (Some(resolve_pipeline), Some(resolve_bind_group)) = (
                self.active.resolve_pipeline.as_ref(),
                self.active.resolve_bind_group.as_ref(),
            ) else {
                log::warn!(
                    "subpass_msaa: resolve pipeline/bind-group missing at sample_count > 1; \
                     submitting without resolve pass"
                );
                queue.submit(Some(encoder.finish()));
                return;
            };
            let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("subpass_msaa resolve"),
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
                multiview_mask: None,
                timestamp_writes: None,
                occlusion_query_set: None,
            });
            pass.set_pipeline(resolve_pipeline);
            pass.set_bind_group(0, resolve_bind_group, &[]);
            pass.draw(0..3, 0..1);
        }

        queue.submit(Some(encoder.finish()));
    }
}

pub fn main() {
    crate::framework::run::<Example>("subpass-msaa");
}
// tiled-fork: end example
