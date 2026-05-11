// tiled-fork: begin example
//! Deferred Rendering with 3-Subpass Render Graph
//!
//! Demonstrates a TBDR-friendly deferred-shading pipeline:
//!   Subpass 0 (G-Buffer):  renders geometry to albedo + normal targets
//!   Subpass 1 (Lighting):  reads G-Buffer via input attachments, Blinn-Phong
//!                          with 4 point lights -> HDR target
//!   Subpass 2 (Composite): reads HDR via input attachment, Reinhard tonemap
//!                          -> sRGB swapchain
//!
//! Adapted from `infosia/wgpu-tiled`'s example of the same name. The
//! reference uses `RenderGraphBuilder` (not ported in this fork) and
//! extended `Limits` fields (also not ported per the "no breaking
//! changes" rule). This port constructs `wgpu::SubpassTarget`,
//! `SubpassRenderPassDescriptor`, and `SubpassDescriptor` literally,
//! and creates pipelines through `SubpassRenderPipelineDescriptor::new`
//! plus `Device::create_subpass_render_pipeline`. Visually equivalent;
//! the underlying tile-memory optimization (`TRANSIENT_ATTACHMENTS`)
//! is not exercised because the public-API path for transient subpass
//! attachments still returns `TransientNotWired` in this fork. The
//! attachments here are regular DRAM-backed textures.
//!
//! The three WGSL shaders are copied verbatim from the reference fork.
//!
//! Requires `Features::MULTI_SUBPASS`, which in this fork is advertised
//! only on Vulkan (always) and Metal/GLES (gated on tile-shading /
//! framebuffer-fetch). DX12 is permanently excluded.
//!
//! ## Known runtime gap (Phase 11i)
//!
//! As of Phase 7b this example **compiles but fails at runtime** when
//! creating the lighting pipeline. The implicit pipeline-layout
//! derivation in `wgpu-core/src/validation.rs` cannot translate the
//! shader's `subpass_input<f32>` binding (`naga::ImageClass::Subpass`)
//! into a `wgt::BindingType` because no `BindingType::SubpassInput`
//! variant exists yet -- the validator returns
//! `BindingError::TiledNotImplemented`. Wiring that variant (plus the
//! bind-group-layout entry validation + HAL descriptor-set translation)
//! is queued as **Phase 11i**. Once Phase 11i lands the example will
//! run end-to-end.
//!
//! The example is committed in its target shape so the API surface is
//! reviewable and the gap is concrete.

use bytemuck::{Pod, Zeroable};
use glam::{Mat4, Vec3};
use std::borrow::Cow;
use wgpu::util::DeviceExt;

#[repr(C)]
#[derive(Copy, Clone, Debug, Pod, Zeroable)]
struct Vertex {
    position: [f32; 3],
    normal: [f32; 3],
    color: [f32; 3],
}

#[repr(C)]
#[derive(Copy, Clone, Debug, Pod, Zeroable)]
struct Uniforms {
    view_proj: [[f32; 4]; 4],
}

#[repr(C)]
#[derive(Copy, Clone, Debug, Pod, Zeroable)]
struct LightParams {
    lights: [[f32; 4]; 4],
    camera_pos: [f32; 3],
    time: f32,
    inv_view_proj: [[f32; 4]; 4],
    screen_size: [f32; 2],
    _padding: [f32; 2],
}

const GRID_SIZE: u32 = 5;
const INSTANCE_COUNT: u32 = GRID_SIZE * GRID_SIZE;

const ALBEDO_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Rgba8Unorm;
const NORMAL_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Rgba16Float;
const LIT_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Rgba16Float;
const DEPTH_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Depth32Float;

fn create_cube_vertices() -> (Vec<Vertex>, Vec<u16>) {
    let positions: &[[f32; 3]] = &[
        [-1.0, -1.0, 1.0],
        [1.0, -1.0, 1.0],
        [1.0, 1.0, 1.0],
        [-1.0, 1.0, 1.0],
        [-1.0, -1.0, -1.0],
        [-1.0, 1.0, -1.0],
        [1.0, 1.0, -1.0],
        [1.0, -1.0, -1.0],
        [-1.0, 1.0, -1.0],
        [-1.0, 1.0, 1.0],
        [1.0, 1.0, 1.0],
        [1.0, 1.0, -1.0],
        [-1.0, -1.0, -1.0],
        [1.0, -1.0, -1.0],
        [1.0, -1.0, 1.0],
        [-1.0, -1.0, 1.0],
        [1.0, -1.0, -1.0],
        [1.0, 1.0, -1.0],
        [1.0, 1.0, 1.0],
        [1.0, -1.0, 1.0],
        [-1.0, -1.0, -1.0],
        [-1.0, -1.0, 1.0],
        [-1.0, 1.0, 1.0],
        [-1.0, 1.0, -1.0],
    ];
    let normals: &[[f32; 3]] = &[
        [0.0, 0.0, 1.0],
        [0.0, 0.0, 1.0],
        [0.0, 0.0, 1.0],
        [0.0, 0.0, 1.0],
        [0.0, 0.0, -1.0],
        [0.0, 0.0, -1.0],
        [0.0, 0.0, -1.0],
        [0.0, 0.0, -1.0],
        [0.0, 1.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, -1.0, 0.0],
        [0.0, -1.0, 0.0],
        [0.0, -1.0, 0.0],
        [0.0, -1.0, 0.0],
        [1.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [-1.0, 0.0, 0.0],
        [-1.0, 0.0, 0.0],
        [-1.0, 0.0, 0.0],
        [-1.0, 0.0, 0.0],
    ];
    let face_colors: &[[f32; 3]] = &[
        [1.0, 0.3, 0.3],
        [0.3, 1.0, 0.3],
        [0.3, 0.3, 1.0],
        [1.0, 1.0, 0.3],
        [1.0, 0.3, 1.0],
        [0.3, 1.0, 1.0],
    ];

    let vertices: Vec<Vertex> = (0..24)
        .map(|i| Vertex {
            position: positions[i],
            normal: normals[i],
            color: face_colors[i / 4],
        })
        .collect();

    let indices: Vec<u16> = (0..6u16)
        .flat_map(|face| {
            let base = face * 4;
            [base, base + 1, base + 2, base, base + 2, base + 3]
        })
        .collect();

    (vertices, indices)
}

fn create_attachment_textures(
    device: &wgpu::Device,
    width: u32,
    height: u32,
    output_format: wgpu::TextureFormat,
) -> (
    wgpu::Texture,
    wgpu::Texture,
    wgpu::Texture,
    wgpu::Texture,
) {
    let size = wgpu::Extent3d {
        width,
        height,
        depth_or_array_layers: 1,
    };
    let usage = wgpu::TextureUsages::RENDER_ATTACHMENT;
    let albedo = device.create_texture(&wgpu::TextureDescriptor {
        label: Some("deferred_rendering albedo"),
        size,
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format: ALBEDO_FORMAT,
        usage,
        view_formats: &[],
    });
    let normal = device.create_texture(&wgpu::TextureDescriptor {
        label: Some("deferred_rendering normal"),
        size,
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format: NORMAL_FORMAT,
        usage,
        view_formats: &[],
    });
    let depth = device.create_texture(&wgpu::TextureDescriptor {
        label: Some("deferred_rendering depth"),
        size,
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format: DEPTH_FORMAT,
        usage,
        view_formats: &[],
    });
    let lit = device.create_texture(&wgpu::TextureDescriptor {
        label: Some("deferred_rendering lit (HDR)"),
        size,
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format: LIT_FORMAT,
        usage,
        view_formats: &[],
    });
    let _ = output_format;
    (albedo, normal, depth, lit)
}

fn build_subpass_target_base(output_format: wgpu::TextureFormat) -> wgpu::SubpassTarget {
    wgpu::SubpassTarget {
        index: 0,
        color_attachment_formats: vec![
            Some(ALBEDO_FORMAT),
            Some(NORMAL_FORMAT),
            Some(LIT_FORMAT),
            Some(output_format),
        ],
        depth_stencil_format: Some(DEPTH_FORMAT),
        subpass_descs: vec![
            wgpu::SubpassTargetDesc {
                color_attachment_indices: vec![0, 1],
                uses_depth_stencil: true,
                input_attachment_indices: vec![],
            },
            wgpu::SubpassTargetDesc {
                color_attachment_indices: vec![2],
                uses_depth_stencil: false,
                input_attachment_indices: vec![0, 1],
            },
            wgpu::SubpassTargetDesc {
                color_attachment_indices: vec![3],
                uses_depth_stencil: false,
                input_attachment_indices: vec![2],
            },
        ],
        dependencies: vec![
            wgpu::SubpassDependency {
                src_subpass: wgpu::SubpassIndex(0),
                dst_subpass: wgpu::SubpassIndex(1),
                dependency_type: wgpu::SubpassDependencyType::ColorToInput,
                by_region: true,
            },
            wgpu::SubpassDependency {
                src_subpass: wgpu::SubpassIndex(1),
                dst_subpass: wgpu::SubpassIndex(2),
                dependency_type: wgpu::SubpassDependencyType::ColorToInput,
                by_region: true,
            },
        ],
    }
}

struct Example {
    gbuffer_pipeline: wgpu::RenderPipeline,
    lighting_pipeline: wgpu::RenderPipeline,
    composite_pipeline: wgpu::RenderPipeline,
    vertex_buf: wgpu::Buffer,
    index_buf: wgpu::Buffer,
    index_count: u32,
    uniform_buf: wgpu::Buffer,
    uniform_bind_group: wgpu::BindGroup,
    light_buf: wgpu::Buffer,
    lighting_bind_group: wgpu::BindGroup,
    composite_bind_group: wgpu::BindGroup,
    _albedo_texture: wgpu::Texture,
    _normal_texture: wgpu::Texture,
    _depth_texture: wgpu::Texture,
    _lit_texture: wgpu::Texture,
    albedo_view: wgpu::TextureView,
    normal_view: wgpu::TextureView,
    depth_view: wgpu::TextureView,
    lit_view: wgpu::TextureView,
    start_time: std::time::Instant,
    width: u32,
    height: u32,
    output_format: wgpu::TextureFormat,
}

impl crate::framework::Example for Example {
    fn required_features() -> wgpu::Features {
        wgpu::Features::MULTI_SUBPASS
    }

    fn required_limits() -> wgpu::Limits {
        wgpu::Limits::downlevel_webgl2_defaults()
    }

    fn init(
        config: &wgpu::SurfaceConfiguration,
        _adapter: &wgpu::Adapter,
        device: &wgpu::Device,
        _queue: &wgpu::Queue,
    ) -> Self {
        let width = config.width;
        let height = config.height;
        let output_format = config
            .view_formats
            .first()
            .copied()
            .unwrap_or(config.format);

        let (albedo_texture, normal_texture, depth_texture, lit_texture) =
            create_attachment_textures(device, width, height, output_format);
        let albedo_view = albedo_texture.create_view(&wgpu::TextureViewDescriptor::default());
        let normal_view = normal_texture.create_view(&wgpu::TextureViewDescriptor::default());
        let depth_view = depth_texture.create_view(&wgpu::TextureViewDescriptor::default());
        let lit_view = lit_texture.create_view(&wgpu::TextureViewDescriptor::default());

        let (vertices, indices) = create_cube_vertices();
        let vertex_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("deferred_rendering vertex"),
            contents: bytemuck::cast_slice(&vertices),
            usage: wgpu::BufferUsages::VERTEX,
        });
        let index_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("deferred_rendering index"),
            contents: bytemuck::cast_slice(&indices),
            usage: wgpu::BufferUsages::INDEX,
        });

        let subpass_target_base = build_subpass_target_base(output_format);

        // -- Subpass 0: G-Buffer pipeline ---------------------------------
        let gbuffer_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("deferred_rendering gbuffer.wgsl"),
            source: wgpu::ShaderSource::Wgsl(Cow::Borrowed(include_str!("gbuffer.wgsl"))),
        });
        let uniform_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("deferred_rendering uniforms"),
            size: core::mem::size_of::<Uniforms>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let gbuffer_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("deferred_rendering gbuffer bgl"),
            entries: &[wgpu::BindGroupLayoutEntry {
                binding: 0,
                visibility: wgpu::ShaderStages::VERTEX,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Uniform,
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            }],
        });
        let uniform_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("deferred_rendering uniform bg"),
            layout: &gbuffer_bgl,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: uniform_buf.as_entire_binding(),
            }],
        });
        let gbuffer_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("deferred_rendering gbuffer layout"),
            bind_group_layouts: &[Some(&gbuffer_bgl)],
            ..Default::default()
        });
        let gbuffer_pipeline_base = wgpu::RenderPipelineDescriptor {
            label: Some("deferred_rendering gbuffer pipeline"),
            layout: Some(&gbuffer_layout),
            vertex: wgpu::VertexState {
                module: &gbuffer_shader,
                entry_point: Some("vs_main"),
                compilation_options: Default::default(),
                buffers: &[wgpu::VertexBufferLayout {
                    array_stride: core::mem::size_of::<Vertex>() as u64,
                    step_mode: wgpu::VertexStepMode::Vertex,
                    attributes: &wgpu::vertex_attr_array![0 => Float32x3, 1 => Float32x3, 2 => Float32x3],
                }],
            },
            fragment: Some(wgpu::FragmentState {
                module: &gbuffer_shader,
                entry_point: Some("fs_main"),
                compilation_options: Default::default(),
                targets: &[Some(ALBEDO_FORMAT.into()), Some(NORMAL_FORMAT.into())],
            }),
            primitive: wgpu::PrimitiveState {
                topology: wgpu::PrimitiveTopology::TriangleList,
                cull_mode: Some(wgpu::Face::Back),
                ..Default::default()
            },
            depth_stencil: Some(wgpu::DepthStencilState {
                format: DEPTH_FORMAT,
                depth_write_enabled: Some(true),
                depth_compare: Some(wgpu::CompareFunction::Less),
                stencil: Default::default(),
                bias: Default::default(),
            }),
            multisample: Default::default(),
            multiview_mask: None,
            cache: None,
        };
        let gbuffer_pipeline = device.create_subpass_render_pipeline(
            &wgpu::SubpassRenderPipelineDescriptor::new(
                gbuffer_pipeline_base,
                wgpu::SubpassTarget {
                    index: 0,
                    ..subpass_target_base.clone()
                },
            ),
        );

        // -- Subpass 1: Lighting pipeline ---------------------------------
        let lighting_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("deferred_rendering lighting.wgsl"),
            source: wgpu::ShaderSource::Wgsl(Cow::Borrowed(include_str!("lighting.wgsl"))),
        });
        let light_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("deferred_rendering light params"),
            size: core::mem::size_of::<LightParams>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let lighting_pipeline_base = wgpu::RenderPipelineDescriptor {
            label: Some("deferred_rendering lighting pipeline"),
            layout: None,
            vertex: wgpu::VertexState {
                module: &lighting_shader,
                entry_point: Some("vs_main"),
                compilation_options: Default::default(),
                buffers: &[],
            },
            fragment: Some(wgpu::FragmentState {
                module: &lighting_shader,
                entry_point: Some("fs_main"),
                compilation_options: Default::default(),
                targets: &[Some(LIT_FORMAT.into())],
            }),
            primitive: Default::default(),
            depth_stencil: None,
            multisample: Default::default(),
            multiview_mask: None,
            cache: None,
        };
        let lighting_pipeline = device.create_subpass_render_pipeline(
            &wgpu::SubpassRenderPipelineDescriptor::new(
                lighting_pipeline_base,
                wgpu::SubpassTarget {
                    index: 1,
                    ..subpass_target_base.clone()
                },
            ),
        );
        let lighting_bgl = lighting_pipeline.get_bind_group_layout(0);
        let lighting_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("deferred_rendering lighting bg"),
            layout: &lighting_bgl,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::TextureView(&albedo_view),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::TextureView(&normal_view),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: light_buf.as_entire_binding(),
                },
            ],
        });

        // -- Subpass 2: Composite pipeline --------------------------------
        let composite_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("deferred_rendering composite.wgsl"),
            source: wgpu::ShaderSource::Wgsl(Cow::Borrowed(include_str!("composite.wgsl"))),
        });
        let composite_pipeline_base = wgpu::RenderPipelineDescriptor {
            label: Some("deferred_rendering composite pipeline"),
            layout: None,
            vertex: wgpu::VertexState {
                module: &composite_shader,
                entry_point: Some("vs_main"),
                compilation_options: Default::default(),
                buffers: &[],
            },
            fragment: Some(wgpu::FragmentState {
                module: &composite_shader,
                entry_point: Some("fs_main"),
                compilation_options: Default::default(),
                targets: &[Some(output_format.into())],
            }),
            primitive: Default::default(),
            depth_stencil: None,
            multisample: Default::default(),
            multiview_mask: None,
            cache: None,
        };
        let composite_pipeline = device.create_subpass_render_pipeline(
            &wgpu::SubpassRenderPipelineDescriptor::new(
                composite_pipeline_base,
                wgpu::SubpassTarget {
                    index: 2,
                    ..subpass_target_base
                },
            ),
        );
        let composite_bgl = composite_pipeline.get_bind_group_layout(0);
        let composite_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("deferred_rendering composite bg"),
            layout: &composite_bgl,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: wgpu::BindingResource::TextureView(&lit_view),
            }],
        });

        Self {
            gbuffer_pipeline,
            lighting_pipeline,
            composite_pipeline,
            vertex_buf,
            index_buf,
            index_count: indices.len() as u32,
            uniform_buf,
            uniform_bind_group,
            light_buf,
            lighting_bind_group,
            composite_bind_group,
            _albedo_texture: albedo_texture,
            _normal_texture: normal_texture,
            _depth_texture: depth_texture,
            _lit_texture: lit_texture,
            albedo_view,
            normal_view,
            depth_view,
            lit_view,
            start_time: std::time::Instant::now(),
            width,
            height,
            output_format,
        }
    }

    fn resize(
        &mut self,
        config: &wgpu::SurfaceConfiguration,
        device: &wgpu::Device,
        _queue: &wgpu::Queue,
    ) {
        self.width = config.width;
        self.height = config.height;
        let (albedo, normal, depth, lit) =
            create_attachment_textures(device, self.width, self.height, self.output_format);
        self.albedo_view = albedo.create_view(&wgpu::TextureViewDescriptor::default());
        self.normal_view = normal.create_view(&wgpu::TextureViewDescriptor::default());
        self.depth_view = depth.create_view(&wgpu::TextureViewDescriptor::default());
        self.lit_view = lit.create_view(&wgpu::TextureViewDescriptor::default());
        self._albedo_texture = albedo;
        self._normal_texture = normal;
        self._depth_texture = depth;
        self._lit_texture = lit;

        // Subpass-input bind groups reference the texture views above and
        // must be rebuilt whenever the views are recreated.
        let lighting_bgl = self.lighting_pipeline.get_bind_group_layout(0);
        self.lighting_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("deferred_rendering lighting bg"),
            layout: &lighting_bgl,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::TextureView(&self.albedo_view),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::TextureView(&self.normal_view),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: self.light_buf.as_entire_binding(),
                },
            ],
        });
        let composite_bgl = self.composite_pipeline.get_bind_group_layout(0);
        self.composite_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("deferred_rendering composite bg"),
            layout: &composite_bgl,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: wgpu::BindingResource::TextureView(&self.lit_view),
            }],
        });
    }

    fn update(&mut self, _event: winit::event::WindowEvent) {}

    fn render(&mut self, view: &wgpu::TextureView, device: &wgpu::Device, queue: &wgpu::Queue) {
        let time = self.start_time.elapsed().as_secs_f32();
        let aspect = self.width as f32 / self.height.max(1) as f32;

        let eye = Vec3::new(
            12.0 * (time * 0.3).cos(),
            8.0,
            12.0 * (time * 0.3).sin() + 15.0,
        );
        let target = Vec3::ZERO;
        let view_mat = Mat4::look_at_rh(eye, target, Vec3::Y);
        let proj = Mat4::perspective_rh(std::f32::consts::FRAC_PI_4, aspect, 0.1, 100.0);
        let view_proj = proj * view_mat;

        queue.write_buffer(
            &self.uniform_buf,
            0,
            bytemuck::cast_slice(&[Uniforms {
                view_proj: view_proj.to_cols_array_2d(),
            }]),
        );

        let t = time;
        let inv_view_proj = view_proj.inverse();
        queue.write_buffer(
            &self.light_buf,
            0,
            bytemuck::cast_slice(&[LightParams {
                lights: [
                    [10.0 * (t * 0.7).cos(), 8.0, 10.0 * (t * 0.7).sin(), 50.0],
                    [-8.0 * (t * 0.5).cos(), 6.0, -8.0 * (t * 0.5).sin(), 40.0],
                    [6.0 * (t * 1.1).sin(), 4.0, 6.0 * (t * 1.1).cos(), 35.0],
                    [-5.0, 10.0 + 3.0 * (t * 0.3).sin(), 5.0, 45.0],
                ],
                camera_pos: [eye.x, eye.y, eye.z],
                time: t,
                inv_view_proj: inv_view_proj.to_cols_array_2d(),
                screen_size: [self.width as f32, self.height as f32],
                _padding: [0.0; 2],
            }]),
        );

        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("deferred_rendering encoder"),
        });

        let extent = wgpu::Extent3d {
            width: self.width,
            height: self.height,
            depth_or_array_layers: 1,
        };

        let gbuffer_albedo_attachment = wgpu::RenderPassColorAttachment {
            view: &self.albedo_view,
            depth_slice: None,
            resolve_target: None,
            ops: wgpu::Operations {
                load: wgpu::LoadOp::Clear(wgpu::Color::TRANSPARENT),
                store: wgpu::StoreOp::Store,
            },
        };
        let gbuffer_normal_attachment = wgpu::RenderPassColorAttachment {
            view: &self.normal_view,
            depth_slice: None,
            resolve_target: None,
            ops: wgpu::Operations {
                load: wgpu::LoadOp::Clear(wgpu::Color::TRANSPARENT),
                store: wgpu::StoreOp::Store,
            },
        };
        let lit_attachment = wgpu::RenderPassColorAttachment {
            view: &self.lit_view,
            depth_slice: None,
            resolve_target: None,
            ops: wgpu::Operations {
                load: wgpu::LoadOp::Clear(wgpu::Color::BLACK),
                store: wgpu::StoreOp::Store,
            },
        };
        let output_attachment = wgpu::RenderPassColorAttachment {
            view,
            depth_slice: None,
            resolve_target: None,
            ops: wgpu::Operations {
                load: wgpu::LoadOp::Clear(wgpu::Color::BLACK),
                store: wgpu::StoreOp::Store,
            },
        };

        let subpasses = [
            wgpu::SubpassDescriptor {
                color_attachments: &[
                    Some(wgpu::SubpassColorAttachment::Persistent(
                        gbuffer_albedo_attachment.clone(),
                    )),
                    Some(wgpu::SubpassColorAttachment::Persistent(
                        gbuffer_normal_attachment.clone(),
                    )),
                ],
                color_attachment_indices: &[0, 1],
                depth_stencil_attachment: Some(wgpu::SubpassDepthStencilAttachment::Persistent(
                    wgpu::RenderPassDepthStencilAttachment {
                        view: &self.depth_view,
                        depth_ops: Some(wgpu::Operations {
                            load: wgpu::LoadOp::Clear(1.0),
                            store: wgpu::StoreOp::Discard,
                        }),
                        stencil_ops: None,
                    },
                )),
                input_attachments: &[],
            },
            wgpu::SubpassDescriptor {
                color_attachments: &[Some(wgpu::SubpassColorAttachment::Persistent(
                    lit_attachment.clone(),
                ))],
                color_attachment_indices: &[2],
                depth_stencil_attachment: None,
                input_attachments: &[
                    wgpu::SubpassInputAttachment {
                        binding: 0,
                        source: wgpu::SubpassInputSource::Color {
                            subpass: wgpu::SubpassIndex(0),
                            attachment_index: 0,
                        },
                    },
                    wgpu::SubpassInputAttachment {
                        binding: 1,
                        source: wgpu::SubpassInputSource::Color {
                            subpass: wgpu::SubpassIndex(0),
                            attachment_index: 1,
                        },
                    },
                ],
            },
            wgpu::SubpassDescriptor {
                color_attachments: &[Some(wgpu::SubpassColorAttachment::Persistent(
                    output_attachment.clone(),
                ))],
                color_attachment_indices: &[3],
                depth_stencil_attachment: None,
                input_attachments: &[wgpu::SubpassInputAttachment {
                    binding: 0,
                    source: wgpu::SubpassInputSource::Color {
                        subpass: wgpu::SubpassIndex(1),
                        attachment_index: 0,
                    },
                }],
            },
        ];

        let subpass_dependencies = [
            wgpu::SubpassDependency {
                src_subpass: wgpu::SubpassIndex(0),
                dst_subpass: wgpu::SubpassIndex(1),
                dependency_type: wgpu::SubpassDependencyType::ColorToInput,
                by_region: true,
            },
            wgpu::SubpassDependency {
                src_subpass: wgpu::SubpassIndex(1),
                dst_subpass: wgpu::SubpassIndex(2),
                dependency_type: wgpu::SubpassDependencyType::ColorToInput,
                by_region: true,
            },
        ];

        {
            let mut pass = encoder.begin_subpass_render_pass(&wgpu::SubpassRenderPassDescriptor {
                label: Some("deferred_rendering pass"),
                extent,
                sample_count: 1,
                color_attachments: &[
                    Some(gbuffer_albedo_attachment),
                    Some(gbuffer_normal_attachment),
                    Some(lit_attachment),
                    Some(output_attachment),
                ],
                depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                    view: &self.depth_view,
                    depth_ops: Some(wgpu::Operations {
                        load: wgpu::LoadOp::Clear(1.0),
                        store: wgpu::StoreOp::Discard,
                    }),
                    stencil_ops: None,
                }),
                subpasses: &subpasses,
                subpass_dependencies: &subpass_dependencies,
                transient_memory_hint: wgpu::TransientMemoryHint::Auto,
                active_subpass_mask: None,
                multiview_mask: None,
                timestamp_writes: None,
                occlusion_query_set: None,
            });

            // Subpass 0: G-Buffer geometry (instanced 5x5 cube grid).
            pass.set_pipeline(&self.gbuffer_pipeline);
            pass.set_bind_group(0, &self.uniform_bind_group, &[]);
            pass.set_vertex_buffer(0, self.vertex_buf.slice(..));
            pass.set_index_buffer(self.index_buf.slice(..), wgpu::IndexFormat::Uint16);
            pass.draw_indexed(0..self.index_count, 0, 0..INSTANCE_COUNT);

            // Subpass 1: Lighting (fullscreen triangle, reads G-Buffer via inputs).
            pass.next_subpass();
            pass.set_pipeline(&self.lighting_pipeline);
            pass.set_bind_group(0, &self.lighting_bind_group, &[]);
            pass.draw(0..3, 0..1);

            // Subpass 2: Composite (fullscreen triangle, Reinhard tonemap).
            pass.next_subpass();
            pass.set_pipeline(&self.composite_pipeline);
            pass.set_bind_group(0, &self.composite_bind_group, &[]);
            pass.draw(0..3, 0..1);
        }

        queue.submit(Some(encoder.finish()));
    }
}

pub fn main() {
    crate::framework::run::<Example>("Deferred Rendering (3-Subpass Demo)");
}
// tiled-fork: end example
