// tiled-fork: begin example
//! Headless smoke-test for the multi-subpass surface.
//!
//! Exercises the public `wgpu::CommandEncoder::begin_subpass_render_pass`
//! path end-to-end without any user-visible rendering: a 2-subpass
//! render pass with two persistent color attachments, one
//! `next_subpass` advance, and a queue submit.
//!
//! Adapted from `infosia/wgpu-tiled`'s example of the same name.
//! That fork uses an extended upstream `RenderPassDescriptor` with
//! subpass fields and a `RenderGraphBuilder` declarative API; this
//! fork uses a separate `SubpassRenderPassDescriptor` constructed
//! literally (the RenderGraphBuilder is not ported).

async fn run() {
    let instance = wgpu::Instance::default();
    let adapter = instance
        .request_adapter(&wgpu::RequestAdapterOptions::default())
        .await
        .expect("No suitable adapter found");
    let (device, queue) = adapter
        .request_device(&wgpu::DeviceDescriptor {
            label: Some("subpass_render_graph device"),
            required_features: wgpu::Features::empty(),
            required_limits: wgpu::Limits::downlevel_defaults(),
            experimental_features: wgpu::ExperimentalFeatures::disabled(),
            memory_hints: wgpu::MemoryHints::MemoryUsage,
            trace: wgpu::Trace::Off,
        })
        .await
        .expect("Failed to create device");

    if !device.features().contains(wgpu::Features::MULTI_SUBPASS) {
        log::warn!("Adapter does not support MULTI_SUBPASS; skipping headless subpass graph pass");
        return;
    }

    let format = wgpu::TextureFormat::Rgba8Unorm;
    let extent = wgpu::Extent3d {
        width: 4,
        height: 4,
        depth_or_array_layers: 1,
    };
    let usage = wgpu::TextureUsages::RENDER_ATTACHMENT;
    let gbuffer_texture = device.create_texture(&wgpu::TextureDescriptor {
        label: Some("subpass_render_graph gbuffer (subpass 0)"),
        size: extent,
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format,
        usage,
        view_formats: &[],
    });
    let output_texture = device.create_texture(&wgpu::TextureDescriptor {
        label: Some("subpass_render_graph output (subpass 1)"),
        size: extent,
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format,
        usage,
        view_formats: &[],
    });
    let gbuffer_view = gbuffer_texture.create_view(&wgpu::TextureViewDescriptor::default());
    let output_view = output_texture.create_view(&wgpu::TextureViewDescriptor::default());

    // 2-subpass graph (gbuffer -> composite). The reference fork
    // constructs this via RenderGraphBuilder; we construct it
    // literally. Persistent attachment indices: 0 = gbuffer, 1 = output.
    let subpasses = [
        wgpu::SubpassDescriptor {
            color_attachments: &[Some(wgpu::SubpassColorAttachment::Persistent(
                wgpu::RenderPassColorAttachment {
                    view: &gbuffer_view,
                    depth_slice: None,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(wgpu::Color::BLACK),
                        store: wgpu::StoreOp::Store,
                    },
                },
            ))],
            color_attachment_indices: &[0],
            depth_stencil_attachment: None,
            input_attachments: &[],
        },
        wgpu::SubpassDescriptor {
            color_attachments: &[Some(wgpu::SubpassColorAttachment::Persistent(
                wgpu::RenderPassColorAttachment {
                    view: &output_view,
                    depth_slice: None,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(wgpu::Color::BLACK),
                        store: wgpu::StoreOp::Store,
                    },
                },
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
        label: Some("subpass_render_graph encoder"),
    });
    {
        let mut pass = encoder.begin_subpass_render_pass(&wgpu::SubpassRenderPassDescriptor {
            label: Some("subpass_render_graph pass"),
            extent,
            sample_count: 1,
            color_attachments: &[
                Some(wgpu::RenderPassColorAttachment {
                    view: &gbuffer_view,
                    depth_slice: None,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(wgpu::Color::BLACK),
                        store: wgpu::StoreOp::Store,
                    },
                }),
                Some(wgpu::RenderPassColorAttachment {
                    view: &output_view,
                    depth_slice: None,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(wgpu::Color::BLACK),
                        store: wgpu::StoreOp::Store,
                    },
                }),
            ],
            depth_stencil_attachment: None,
            subpasses: &subpasses,
            subpass_dependencies: &subpass_dependencies,
            transient_memory_hint: wgpu::TransientMemoryHint::Auto,
            active_subpass_mask: None,
            multiview_mask: None,
            timestamp_writes: None,
            occlusion_query_set: None,
        });
        pass.next_subpass();
        // `pass` is dropped here -- the SubpassRenderPass Drop impl ends
        // the HAL render pass and unlocks the parent encoder.
    }
    queue.submit(Some(encoder.finish()));
    let _ = device.poll(wgpu::PollType::wait_indefinitely());
}

pub fn main() {
    #[cfg(not(target_arch = "wasm32"))]
    {
        env_logger::builder()
            .filter_level(log::LevelFilter::Info)
            .format_timestamp_nanos()
            .init();
        pollster::block_on(run());
    }
    #[cfg(target_arch = "wasm32")]
    {
        std::panic::set_hook(Box::new(console_error_panic_hook::hook));
        console_log::init_with_level(log::Level::Info).expect("could not initialize logger");
        crate::utils::add_web_nothing_to_see_msg();
        wasm_bindgen_futures::spawn_local(run());
    }
}
// tiled-fork: end example
