// tiled-fork: begin vulkan-backend
//! Tiled-rendering trait surface for the Vulkan backend.
//!
//! Provides real implementations for transient attachment resource
//! creation, multi-subpass `VkRenderPass` construction, the
//! `next_subpass` / `begin_subpass_render_pass` command-encoder hooks,
//! and (Phase 9a3) input-attachment descriptor sets so `subpassLoad(...)`
//! reads work end-to-end at draw time.
//!
//! `dispatch_transient` is still a stub and will be wired alongside the
//! Apple-GPU programmable tile-dispatch story in a later phase.

use alloc::{format, string::ToString, vec, vec::Vec};
use ash::vk;
use core::mem;

use super::{command, conv};
use super::device::CompiledStage;
// `begin_debug_marker` and `write_timestamp` are inherent methods on the
// upstream `CommandEncoder` trait; bring it into scope so the multi-
// subpass begin path can call them directly.
use crate::CommandEncoder as _;
use crate::{DeviceError, PipelineError};

/// A transient color or depth/stencil attachment whose backing memory is
/// allocated `TRANSIENT_ATTACHMENT | INPUT_ATTACHMENT` (plus the appropriate
/// color or depth/stencil usage). The image lifetime is bounded by an enclosing
/// subpass render pass and the contents are never store-able.
#[derive(Debug)]
pub struct TransientAttachment {
    pub(super) raw_image: vk::Image,
    pub(super) raw_view: vk::ImageView,
    #[allow(dead_code)]
    pub(super) format: wgt::TextureFormat,
    #[allow(dead_code)]
    pub(super) width: u32,
    #[allow(dead_code)]
    pub(super) height: u32,
    #[allow(dead_code)]
    pub(super) sample_count: u32,
    pub(super) allocation: gpu_allocator::vulkan::Allocation,
}

#[derive(Debug)]
pub struct TransientDispatch;

crate::impl_dyn_resource!(TransientAttachment, TransientDispatch);

impl crate::DynTransientAttachment for TransientAttachment {}
impl crate::DynTransientDispatch for TransientDispatch {}

impl crate::TiledApi for super::Api {
    type TransientAttachment = TransientAttachment;
    type TransientDispatch = TransientDispatch;
}

impl crate::TiledDevice for super::Device {
    unsafe fn create_transient_attachment(
        &self,
        desc: &wgt::TransientAttachmentDescriptor,
    ) -> Result<TransientAttachment, DeviceError> {
        if !self
            .shared
            .features
            .contains(wgt::Features::TRANSIENT_ATTACHMENTS)
        {
            log::error!(
                "vulkan: TRANSIENT_ATTACHMENTS feature not enabled on this device"
            );
            return Err(DeviceError::Unexpected);
        }

        let (width, height) = match desc.size {
            wgt::TransientSize::Explicit { width, height } => (width, height),
            wgt::TransientSize::MatchTarget => {
                // `MatchTarget` resolution happens in wgpu-core in a later
                // phase; the HAL only accepts explicit sizes.
                log::error!(
                    "vulkan: create_transient_attachment received unresolved TransientSize::MatchTarget"
                );
                return Err(DeviceError::Unexpected);
            }
            // `TransientSize` is `#[non_exhaustive]`. Any future variant must
            // be resolved by wgpu-core before reaching the HAL.
            _ => {
                log::error!(
                    "vulkan: create_transient_attachment received unsupported TransientSize variant"
                );
                return Err(DeviceError::Unexpected);
            }
        };
        if width == 0 || height == 0 || desc.sample_count == 0 {
            log::error!(
                "vulkan: create_transient_attachment received invalid descriptor: width={}, height={}, sample_count={}",
                width,
                height,
                desc.sample_count
            );
            return Err(DeviceError::Unexpected);
        }

        let raw_format = self.shared.private_caps.map_texture_format(desc.format);
        let depth_stencil = desc.format.is_depth_stencil_format();
        let usage = vk::ImageUsageFlags::TRANSIENT_ATTACHMENT
            | vk::ImageUsageFlags::INPUT_ATTACHMENT
            | if depth_stencil {
                vk::ImageUsageFlags::DEPTH_STENCIL_ATTACHMENT
            } else {
                vk::ImageUsageFlags::COLOR_ATTACHMENT
            };
        let vk_info = vk::ImageCreateInfo::default()
            .image_type(vk::ImageType::TYPE_2D)
            .format(raw_format)
            .extent(vk::Extent3D {
                width,
                height,
                depth: 1,
            })
            .mip_levels(1)
            .array_layers(1)
            .samples(vk::SampleCountFlags::from_raw(desc.sample_count))
            .tiling(vk::ImageTiling::OPTIMAL)
            .usage(usage)
            .sharing_mode(vk::SharingMode::EXCLUSIVE)
            .initial_layout(vk::ImageLayout::UNDEFINED);

        let raw_image = unsafe { self.shared.raw.create_image(&vk_info, None) }
            .map_err(super::map_host_device_oom_err)?;

        let mut requirements = unsafe { self.shared.raw.get_image_memory_requirements(raw_image) };

        // Prefer `LAZILY_ALLOCATED` memory -- this is the entire point of
        // the TRANSIENT_ATTACHMENTS feature on tile-based GPUs. The adapter
        // gate (`supports_lazily_allocated`) ensures at least one memory
        // type advertises it, but `find_memory_type_index` may still return
        // `None` if no memory type satisfies BOTH the image's
        // `memory_type_bits` and the LAZILY_ALLOCATED property; in that
        // case fall through to the regular allocator path so the call
        // doesn't fail outright. Mirrors `device.rs::create_image` for
        // ordinary `TextureUses::TRANSIENT` textures.
        if let Some(idx) = self.find_memory_type_index(
            requirements.memory_type_bits,
            vk::MemoryPropertyFlags::LAZILY_ALLOCATED,
        ) {
            requirements.memory_type_bits = 1 << idx;
        }

        self.error_if_would_oom_on_resource_allocation(false, requirements.size)
            .inspect_err(|_| unsafe {
                self.shared.raw.destroy_image(raw_image, None);
            })?;

        let allocation = self
            .mem_allocator
            .lock()
            .allocate(&gpu_allocator::vulkan::AllocationCreateDesc {
                name: "Transient attachment",
                requirements: vk::MemoryRequirements {
                    memory_type_bits: requirements.memory_type_bits & self.valid_ash_memory_types,
                    ..requirements
                },
                location: gpu_allocator::MemoryLocation::GpuOnly,
                linear: false,
                allocation_scheme: gpu_allocator::vulkan::AllocationScheme::GpuAllocatorManaged,
            })
            .inspect_err(|_| unsafe {
                self.shared.raw.destroy_image(raw_image, None);
            })?;
        self.counters.texture_memory.add(allocation.size() as isize);

        let bind_result = unsafe {
            self.shared
                .raw
                .bind_image_memory(raw_image, allocation.memory(), allocation.offset())
        };
        if let Err(err) = bind_result {
            self.counters.texture_memory.sub(allocation.size() as isize);
            if let Err(free_err) = self.mem_allocator.lock().free(allocation) {
                log::warn!("Failed to free transient attachment allocation: {free_err}");
            }
            unsafe {
                self.shared.raw.destroy_image(raw_image, None);
            }
            return Err(super::map_host_device_oom_err(err));
        }

        let view_aspect = conv::map_aspects(crate::FormatAspects::new(
            desc.format,
            wgt::TextureAspect::All,
        ));
        let view_info = vk::ImageViewCreateInfo::default()
            .image(raw_image)
            .view_type(vk::ImageViewType::TYPE_2D)
            .format(raw_format)
            .subresource_range(vk::ImageSubresourceRange {
                aspect_mask: view_aspect,
                base_mip_level: 0,
                level_count: 1,
                base_array_layer: 0,
                layer_count: 1,
            });
        let raw_view = match unsafe { self.shared.raw.create_image_view(&view_info, None) } {
            Ok(raw) => raw,
            Err(err) => {
                self.counters.texture_memory.sub(allocation.size() as isize);
                if let Err(free_err) = self.mem_allocator.lock().free(allocation) {
                    log::warn!("Failed to free transient attachment allocation: {free_err}");
                }
                unsafe {
                    self.shared.raw.destroy_image(raw_image, None);
                }
                return Err(super::map_host_device_oom_and_ioca_err(err));
            }
        };

        Ok(TransientAttachment {
            raw_image,
            raw_view,
            format: desc.format,
            width,
            height,
            sample_count: desc.sample_count,
            allocation,
        })
    }

    unsafe fn destroy_transient_attachment(&self, attachment: TransientAttachment) {
        unsafe {
            self.shared
                .raw
                .destroy_image_view(attachment.raw_view, None);
            self.shared.raw.destroy_image(attachment.raw_image, None);
        }
        self.counters
            .texture_memory
            .sub(attachment.allocation.size() as isize);
        if let Err(err) = self.mem_allocator.lock().free(attachment.allocation) {
            log::warn!("Failed to free transient attachment allocation: {err}");
        }
    }

    unsafe fn create_transient_dispatch(
        &self,
        _desc: &wgt::TransientDispatchDescriptor,
    ) -> Result<TransientDispatch, DeviceError> {
        Err(DeviceError::Unexpected)
    }

    unsafe fn destroy_transient_dispatch(&self, _dispatch: TransientDispatch) {}

    unsafe fn create_subpass_render_pipeline(
        &self,
        desc: &crate::RenderPipelineDescriptor<
            '_,
            super::PipelineLayout,
            super::ShaderModule,
            super::PipelineCache,
        >,
        subpass_target: &wgt::SubpassTarget,
    ) -> Result<super::RenderPipeline, PipelineError> {
        unsafe { create_subpass_render_pipeline_impl(self, desc, subpass_target) }
    }
}

impl crate::TiledCommandEncoder for super::CommandEncoder {
    unsafe fn begin_subpass_render_pass(
        &mut self,
        desc: &crate::SubpassRenderPassDescriptor<
            '_,
            <super::Api as crate::Api>::QuerySet,
            <super::Api as crate::Api>::TextureView,
        >,
    ) {
        // The trait signature does not return a `Result`, so any failure
        // is logged and the encoder is left without a started render
        // pass. This mirrors the upstream `unwrap()`s in
        // `begin_render_pass`; once Phase 9b lifts the upstream `Result`
        // shape we can propagate errors here too.
        if let Err(err) = unsafe { begin_subpass_render_pass_impl(self, desc) } {
            log::error!("vulkan: begin_subpass_render_pass failed: {err:?}");
        }
    }

    unsafe fn next_subpass(&mut self) {
        let Some(state) = self.subpass_state.as_mut() else {
            // Not in a subpass-mode render pass; ignore (matches the
            // behaviour the rest of the HAL takes when an unsupported
            // call is issued — see `begin_render_pass`'s `unwrap`s).
            log::error!("vulkan: next_subpass called outside a subpass render pass");
            return;
        };
        if state.subpass_count == 0 {
            return;
        }
        let Some(current_subpass) = state.current_index else {
            // Every subpass is culled; nothing to advance to.
            return;
        };
        let next_subpass = command::next_active_subpass_index(
            current_subpass,
            state.subpass_count,
            state.active_subpass_mask,
        );
        // Issue `vkCmdNextSubpass` once per subpass we step over —
        // including the culled middle subpasses — because Vulkan
        // demands sequential traversal.
        if current_subpass + 1 < state.subpass_count {
            let final_target = next_subpass.min(state.subpass_count - 1);
            for _ in (current_subpass + 1)..=final_target {
                unsafe {
                    self.device
                        .raw
                        .cmd_next_subpass(self.active, vk::SubpassContents::INLINE);
                }
            }
        }
        state.current_index = (next_subpass < state.subpass_count).then_some(next_subpass);
        // tiled-fork: begin reset-input-attachment-on-next-subpass
        // Force a re-bind of the input-attachment set on the new subpass; the
        // current pipeline (if any) will rebind on the next draw via
        // `set_render_pipeline`.
        self.active_input_attachment_descriptor_set = None;
        // tiled-fork: end reset-input-attachment-on-next-subpass
    }

    unsafe fn dispatch_transient(&mut self, _dispatch: &TransientDispatch) {
        // TODO: real programmable tile-dispatch implementation. The reference
        // fork's design carries this for forward-compat with Apple-GPU tile
        // shading; no wgpu-core path exercises it today.
        #[allow(clippy::todo)]
        {
            todo!("vulkan: dispatch_transient is reserved for future tile-dispatch support");
        }
    }
}

/// Body of `begin_subpass_render_pass`, factored out so we can use `?`
/// on every fallible step. Errors are logged at the trait boundary.
unsafe fn begin_subpass_render_pass_impl(
    encoder: &mut super::CommandEncoder,
    desc: &crate::SubpassRenderPassDescriptor<
        '_,
        <super::Api as crate::Api>::QuerySet,
        <super::Api as crate::Api>::TextureView,
    >,
) -> Result<(), DeviceError> {
    use arrayvec::ArrayVec;

    let mut vk_clear_values =
        ArrayVec::<vk::ClearValue, { super::MAX_TOTAL_ATTACHMENTS }>::new();
    let mut rp_key = super::RenderPassKey {
        colors: ArrayVec::default(),
        depth_stencil: None,
        sample_count: desc.sample_count,
        multiview_mask: desc.multiview_mask,
        subpasses: Vec::new(),
        subpass_dependencies: desc.subpass_dependencies.to_vec(),
    };
    let mut fb_key = super::FramebufferKey {
        raw_pass: vk::RenderPass::null(),
        attachment_views: ArrayVec::default(),
        attachment_identities: ArrayVec::default(),
        extent: desc.extent,
    };

    // Per-color-slot indices into `vk_attachments`. `None` slots in the
    // descriptor produce `None` here; `SubpassKey` lookups use these
    // tables to map color slot indices → real Vulkan attachment indices.
    let mut color_attachment_indices: Vec<Option<u32>> = vec![None; desc.color_attachments.len()];
    let mut resolve_attachment_indices: Vec<Option<u32>> =
        vec![None; desc.color_attachments.len()];
    let mut depth_stencil_attachment_index: Option<u32> = None;
    let mut next_attachment_index: u32 = 0;

    // tiled-fork: begin reset-active-views
    encoder.active_color_attachment_views.clear();
    encoder
        .active_color_attachment_views
        .reserve(desc.color_attachments.len());
    encoder.active_depth_stencil_view = None;
    encoder.active_input_attachment_descriptor_set = None;
    // tiled-fork: end reset-active-views

    for (color_slot, cat) in desc.color_attachments.iter().enumerate() {
        if let Some(cat) = cat.as_ref() {
            color_attachment_indices[color_slot] = Some(next_attachment_index);
            next_attachment_index += 1;
            let color_view = if cat.target.view.dimension == wgt::TextureViewDimension::D3 {
                let key = super::TempTextureViewKey {
                    texture: cat.target.view.raw_texture,
                    texture_identity: cat.target.view.texture_identity,
                    format: cat.target.view.raw_format,
                    mip_level: cat.target.view.base_mip_level,
                    depth_slice: cat
                        .depth_slice
                        .ok_or(DeviceError::Unexpected)?,
                };
                encoder.make_temp_texture_view(key)?
            } else {
                cat.target.view.identified_raw_view()
            };
            // tiled-fork: begin record-color-view
            encoder
                .active_color_attachment_views
                .push(Some(color_view.raw));
            // tiled-fork: end record-color-view

            vk_clear_values.push(vk::ClearValue {
                color: unsafe { cat.make_vk_clear_color() },
            });
            let color = super::ColorAttachmentKey {
                base: cat.target.make_attachment_key(cat.ops),
                resolve: cat.resolve_target.as_ref().map(|target| {
                    target.make_attachment_key(
                        crate::AttachmentOps::LOAD_CLEAR | crate::AttachmentOps::STORE,
                    )
                }),
            };

            rp_key.colors.push(Some(color));
            fb_key.push_view(color_view);
            if let Some(ref at) = cat.resolve_target {
                resolve_attachment_indices[color_slot] = Some(next_attachment_index);
                next_attachment_index += 1;
                vk_clear_values.push(unsafe { mem::zeroed() });
                fb_key.push_view(at.view.identified_raw_view());
            }
        } else {
            rp_key.colors.push(None);
            // tiled-fork: begin record-color-view-empty
            encoder.active_color_attachment_views.push(None);
            // tiled-fork: end record-color-view-empty
        }
    }
    if let Some(ref ds) = desc.depth_stencil_attachment {
        depth_stencil_attachment_index = Some(next_attachment_index);
        next_attachment_index += 1;
        vk_clear_values.push(vk::ClearValue {
            depth_stencil: vk::ClearDepthStencilValue {
                depth: ds.clear_value.0,
                stencil: ds.clear_value.1,
            },
        });
        rp_key.depth_stencil = Some(super::DepthStencilAttachmentKey {
            base: ds.target.make_attachment_key(ds.depth_ops),
            stencil_ops: ds.stencil_ops,
        });
        fb_key.push_view(ds.target.view.identified_raw_view());
        // tiled-fork: begin record-ds-view
        encoder.active_depth_stencil_view = Some(ds.target.view.raw);
        // tiled-fork: end record-ds-view
    }
    let _ = next_attachment_index;

    let subpass_count = desc.subpasses.len() as u32;

    // tiled-fork: begin populate-active-subpass-tables
    encoder.active_subpass_color_attachment_indices.clear();
    encoder
        .active_subpass_color_attachment_indices
        .reserve(desc.subpasses.len());
    encoder.active_subpass_input_attachments.clear();
    encoder
        .active_subpass_input_attachments
        .reserve(desc.subpasses.len());
    for subpass in desc.subpasses {
        encoder
            .active_subpass_color_attachment_indices
            .push(subpass.color_attachment_indices.to_vec());
        encoder
            .active_subpass_input_attachments
            .push(subpass.input_attachments.to_vec());
    }
    // tiled-fork: end populate-active-subpass-tables

    // Build per-subpass `SubpassKey` entries from the descriptor.
    rp_key.subpasses.reserve(desc.subpasses.len());
    for subpass in desc.subpasses {
        let persistent_color_count = subpass
            .color_attachments
            .iter()
            .filter(|attachment| {
                matches!(
                    attachment,
                    Some(crate::SubpassColorAttachment::Persistent(_))
                )
            })
            .count();
        if !subpass.color_attachments.is_empty()
            && persistent_color_count != subpass.color_attachment_indices.len()
        {
            log::error!(
                "vulkan: subpass color_attachments/color_attachment_indices mismatch: persistent={}, indices={}",
                persistent_color_count,
                subpass.color_attachment_indices.len()
            );
            return Err(DeviceError::Unexpected);
        }
        let mut subpass_color_attachment_indices: Vec<Option<u32>> = Vec::new();
        let mut subpass_resolve_attachment_indices: Vec<u32> = Vec::new();
        if !subpass.color_attachments.is_empty() {
            let mut attachment_slot_iter = subpass.color_attachment_indices.iter().copied();
            for color_attachment in subpass.color_attachments {
                match color_attachment {
                    Some(crate::SubpassColorAttachment::Persistent(_)) => {
                        let attachment_slot = attachment_slot_iter.next();
                        let color_attachment_index = attachment_slot.and_then(|slot| {
                            color_attachment_indices.get(slot as usize).copied().flatten()
                        });
                        let resolve_attachment_index = attachment_slot
                            .and_then(|slot| {
                                resolve_attachment_indices
                                    .get(slot as usize)
                                    .copied()
                                    .flatten()
                            })
                            .unwrap_or(vk::ATTACHMENT_UNUSED);
                        subpass_color_attachment_indices.push(color_attachment_index);
                        subpass_resolve_attachment_indices.push(resolve_attachment_index);
                    }
                    Some(crate::SubpassColorAttachment::Transient { .. }) => {
                        // TODO(wgpu-core bridge): wire transient color
                        // attachments through render pass / framebuffer
                        // construction once wgpu-core resolves
                        // `transient_index` -> `TransientAttachment` view.
                        log::error!(
                            "vulkan: transient subpass color attachments are not wired to render pass / framebuffer attachment lists yet"
                        );
                        return Err(DeviceError::Unexpected);
                    }
                    None => {
                        subpass_color_attachment_indices.push(None);
                        subpass_resolve_attachment_indices.push(vk::ATTACHMENT_UNUSED);
                    }
                }
            }
        } else {
            for &attachment_slot in subpass.color_attachment_indices {
                let color_attachment_index = color_attachment_indices
                    .get(attachment_slot as usize)
                    .copied()
                    .flatten();
                let resolve_attachment_index = resolve_attachment_indices
                    .get(attachment_slot as usize)
                    .copied()
                    .flatten()
                    .unwrap_or(vk::ATTACHMENT_UNUSED);
                subpass_color_attachment_indices.push(color_attachment_index);
                subpass_resolve_attachment_indices.push(resolve_attachment_index);
            }
        }

        // `SubpassInputSource::Color::attachment_index` is the source
        // subpass's *local* color-output slot (zero-based within that
        // subpass's `color_attachment_indices`). Translate to the
        // pass-level color slot first, then to the vk attachment index.
        // Must match the descriptor-set resolution path in
        // `subpass_input_descriptor_image_info`.
        let input_attachment_indices = subpass
            .input_attachments
            .iter()
            .map(|input_attachment| match input_attachment.source {
                wgt::SubpassInputSource::Color {
                    subpass: source_subpass,
                    attachment_index,
                } => encoder
                    .active_subpass_color_attachment_indices
                    .get(source_subpass.0 as usize)
                    .and_then(|locals| locals.get(attachment_index as usize))
                    .copied()
                    .and_then(|pass_level_slot| {
                        color_attachment_indices
                            .get(pass_level_slot as usize)
                            .copied()
                            .flatten()
                    })
                    .unwrap_or(vk::ATTACHMENT_UNUSED),
                wgt::SubpassInputSource::Depth { .. } => {
                    depth_stencil_attachment_index.unwrap_or(vk::ATTACHMENT_UNUSED)
                }
                // `SubpassInputSource` is `#[non_exhaustive]`; unknown
                // variants resolve to `ATTACHMENT_UNUSED` so the render
                // pass stays valid even if a future source kind is added.
                _ => vk::ATTACHMENT_UNUSED,
            })
            .collect::<Vec<_>>();

        let depth_stencil_index = match subpass.depth_stencil_attachment.as_ref() {
            Some(crate::SubpassDepthStencilAttachment::Persistent(_)) => {
                depth_stencil_attachment_index
            }
            Some(crate::SubpassDepthStencilAttachment::Transient { .. }) => {
                // TODO(wgpu-core bridge): wire transient depth/stencil
                // through attachment lists once wgpu-core resolves
                // `transient_index` -> `TransientAttachment` view.
                log::error!(
                    "vulkan: transient subpass depth/stencil attachments are not wired to render pass / framebuffer attachment lists yet"
                );
                return Err(DeviceError::Unexpected);
            }
            None => None,
        };

        rp_key.subpasses.push(super::SubpassKey {
            color_attachment_indices: subpass_color_attachment_indices,
            input_attachment_indices,
            depth_stencil_index,
            resolve_attachment_indices: subpass_resolve_attachment_indices,
        });
    }

    // tiled-fork: begin prepare-input-attachment-descriptors
    // Allocate the per-subpass input-attachment descriptor sets now so they
    // are ready when `set_render_pipeline` binds them on each subpass.
    //
    // If allocation fails the encoder is rolled back to a state where
    // `subpass_state` is `None` and the active-attachment side-tables are
    // cleared, so a subsequent `end_render_pass` won't observe stale
    // partial state from this aborted begin.
    unsafe {
        if let Err(err) = prepare_input_attachment_descriptor_sets(encoder) {
            encoder.active_color_attachment_views.clear();
            encoder.active_depth_stencil_view = None;
            encoder.active_subpass_color_attachment_indices.clear();
            encoder.active_subpass_input_attachments.clear();
            encoder.subpass_state = None;
            return Err(err);
        }
    }
    // tiled-fork: end prepare-input-attachment-descriptors

    let render_area = vk::Rect2D {
        offset: vk::Offset2D { x: 0, y: 0 },
        extent: vk::Extent2D {
            width: desc.extent.width,
            height: desc.extent.height,
        },
    };
    let vk_viewports = [vk::Viewport {
        x: 0.0,
        y: desc.extent.height as f32,
        width: desc.extent.width as f32,
        height: -(desc.extent.height as f32),
        min_depth: 0.0,
        max_depth: 1.0,
    }];

    let raw_pass = encoder.device.make_render_pass(rp_key)?;
    fb_key.raw_pass = raw_pass;
    let raw_framebuffer = encoder.make_framebuffer(fb_key)?;

    let vk_info = vk::RenderPassBeginInfo::default()
        .render_pass(raw_pass)
        .render_area(render_area)
        .clear_values(&vk_clear_values)
        .framebuffer(raw_framebuffer);

    if let Some(label) = desc.label {
        unsafe { encoder.begin_debug_marker(label) };
        encoder.rpass_debug_marker_active = true;
    }

    if let Some(timestamp_writes) = desc.timestamp_writes.as_ref() {
        if let Some(index) = timestamp_writes.beginning_of_pass_write_index {
            unsafe {
                encoder.write_timestamp(timestamp_writes.query_set, index);
            }
        }
        encoder.end_of_pass_timer_query = timestamp_writes
            .end_of_pass_write_index
            .map(|index| (timestamp_writes.query_set.raw, index));
    }

    unsafe {
        encoder
            .device
            .raw
            .cmd_set_viewport(encoder.active, 0, &vk_viewports);
        encoder
            .device
            .raw
            .cmd_set_scissor(encoder.active, 0, &[render_area]);
        encoder.device.raw.cmd_begin_render_pass(
            encoder.active,
            &vk_info,
            vk::SubpassContents::INLINE,
        );
    }

    // Initial-subpass cursor placement. If the very first subpass is
    // culled we step forward with empty `vkCmdNextSubpass` calls until we
    // hit one that is active. If every subpass is culled we drain to the
    // last subpass so the matching `end_render_pass` sees a properly-
    // walked render pass.
    let mut current_index: Option<u32> = None;
    if subpass_count > 0 {
        if let Some(active_subpass_index) =
            command::first_active_subpass_index(subpass_count, desc.active_subpass_mask)
        {
            for _ in 0..active_subpass_index {
                unsafe {
                    encoder
                        .device
                        .raw
                        .cmd_next_subpass(encoder.active, vk::SubpassContents::INLINE);
                }
            }
            current_index = Some(active_subpass_index);
        } else {
            for _ in 1..subpass_count {
                unsafe {
                    encoder
                        .device
                        .raw
                        .cmd_next_subpass(encoder.active, vk::SubpassContents::INLINE);
                }
            }
        }
    }

    encoder.subpass_state = Some(super::SubpassState {
        subpass_count,
        current_index,
        active_subpass_mask: if subpass_count > 0 {
            desc.active_subpass_mask
        } else {
            None
        },
    });

    encoder.bind_point = vk::PipelineBindPoint::GRAPHICS;

    Ok(())
}

// tiled-fork: begin input-attachment-descriptor-prep
/// Build the per-subpass input-attachment descriptor sets for the active
/// render pass. This must be called after the encoder's
/// `active_color_attachment_views`, `active_depth_stencil_view`, and
/// `active_subpass_input_attachments` have been populated.
///
/// Subpasses that declare no input attachments contribute no set; the
/// descriptor pool is sized to the count of subpasses that *do* declare
/// inputs. The resulting `Option<vk::DescriptorSet>` table is consumed at
/// draw time by [`bind_pipeline_input_attachments`].
unsafe fn prepare_input_attachment_descriptor_sets(
    encoder: &mut super::CommandEncoder,
) -> Result<(), DeviceError> {
    encoder.subpass_input_attachment_descriptor_sets.clear();
    encoder
        .subpass_input_attachment_descriptor_sets
        .resize(encoder.active_subpass_input_attachments.len(), None);

    let set_count: u32 = encoder
        .active_subpass_input_attachments
        .iter()
        .filter(|inputs| !inputs.is_empty())
        .count() as u32;
    if set_count == 0 {
        return Ok(());
    }
    if encoder.device.max_input_attachments == 0 {
        log::error!(
            "vulkan: backend reports zero input attachments — cannot bind subpass inputs"
        );
        return Err(DeviceError::Unexpected);
    }
    // The shared layout reserves `max_input_attachments` bindings per set;
    // size the pool accordingly so any binding index in `[0,
    // max_input_attachments)` is allocatable.
    let descriptor_count = set_count.saturating_mul(encoder.device.max_input_attachments);

    let pool_sizes = [vk::DescriptorPoolSize {
        ty: vk::DescriptorType::INPUT_ATTACHMENT,
        descriptor_count,
    }];
    let pool_info = vk::DescriptorPoolCreateInfo::default()
        .max_sets(set_count)
        .pool_sizes(&pool_sizes);
    let descriptor_pool = unsafe {
        encoder
            .device
            .raw
            .create_descriptor_pool(&pool_info, None)
            .map_err(super::map_host_device_oom_err)?
    };
    encoder
        .input_attachment_descriptor_pools
        .push(descriptor_pool);

    let set_layouts =
        vec![encoder.device.input_attachment_descriptor_set_layout; set_count as usize];
    let set_allocate_info = vk::DescriptorSetAllocateInfo::default()
        .descriptor_pool(descriptor_pool)
        .set_layouts(&set_layouts);
    let descriptor_sets = unsafe {
        encoder
            .device
            .raw
            .allocate_descriptor_sets(&set_allocate_info)
            .map_err(super::map_host_device_oom_err)?
    };

    let mut set_iter = descriptor_sets.into_iter();
    let inputs_len = encoder.active_subpass_input_attachments.len();
    for subpass_index in 0..inputs_len {
        if encoder.active_subpass_input_attachments[subpass_index].is_empty() {
            continue;
        }
        let descriptor_set = match set_iter.next() {
            Some(set) => set,
            None => {
                log::error!("vulkan: descriptor set allocation count mismatch");
                return Err(DeviceError::Unexpected);
            }
        };
        encoder.subpass_input_attachment_descriptor_sets[subpass_index] = Some(descriptor_set);

        // Snapshot the inputs to release the borrow on `encoder`'s vector
        // before reading the (also-borrowed) view tables in the helper.
        let inputs = encoder.active_subpass_input_attachments[subpass_index].clone();

        let mut image_infos = Vec::with_capacity(inputs.len());
        for input_attachment in inputs.iter().copied() {
            let Some(image_info) = subpass_input_descriptor_image_info(encoder, input_attachment)
            else {
                log::error!(
                    "vulkan: invalid subpass input attachment source at subpass {} binding {}",
                    subpass_index,
                    input_attachment.binding
                );
                return Err(DeviceError::Unexpected);
            };
            image_infos.push((input_attachment.binding, image_info));
        }
        let mut writes = Vec::with_capacity(image_infos.len());
        for (binding, image_info) in image_infos.iter() {
            writes.push(
                vk::WriteDescriptorSet::default()
                    .dst_set(descriptor_set)
                    .dst_binding(*binding)
                    .descriptor_type(vk::DescriptorType::INPUT_ATTACHMENT)
                    .image_info(core::slice::from_ref(image_info)),
            );
        }
        unsafe { encoder.device.raw.update_descriptor_sets(&writes, &[]) };
    }

    Ok(())
}

/// Resolve a single subpass-input source to a `VkDescriptorImageInfo`.
fn subpass_input_descriptor_image_info(
    encoder: &super::CommandEncoder,
    input: wgt::SubpassInputAttachment,
) -> Option<vk::DescriptorImageInfo> {
    match input.source {
        wgt::SubpassInputSource::Color {
            subpass,
            attachment_index,
        } => encoder
            .active_subpass_color_attachment_indices
            .get(subpass.0 as usize)
            .and_then(|source_subpass| source_subpass.get(attachment_index as usize))
            .copied()
            .and_then(|source_attachment_slot| {
                encoder
                    .active_color_attachment_views
                    .get(source_attachment_slot as usize)
                    .copied()
            })
            .flatten()
            .map(|view| {
                vk::DescriptorImageInfo::default()
                    .image_view(view)
                    .image_layout(vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL)
            }),
        wgt::SubpassInputSource::Depth { .. } => {
            encoder.active_depth_stencil_view.map(|view| {
                vk::DescriptorImageInfo::default()
                    .image_view(view)
                    .image_layout(vk::ImageLayout::DEPTH_STENCIL_READ_ONLY_OPTIMAL)
            })
        }
        // `SubpassInputSource` is `#[non_exhaustive]`; future variants are
        // treated as missing so the call site can return `Unexpected`.
        _ => None,
    }
}

/// Bind the input-attachment descriptor set for the active subpass on
/// behalf of `set_render_pipeline`. No-op when the pipeline does not
/// declare subpass-input bindings.
pub(super) unsafe fn bind_pipeline_input_attachments(
    encoder: &mut super::CommandEncoder,
    pipeline: &super::RenderPipeline,
) {
    if pipeline.input_attachments.is_empty() {
        encoder.active_input_attachment_descriptor_set = None;
        return;
    }
    if pipeline.input_attachment_descriptor_set_index == u32::MAX {
        log::error!(
            "vulkan: pipeline has subpass inputs but its layout has no spare descriptor set slot"
        );
        return;
    }
    let Some(state) = encoder.subpass_state.as_ref() else {
        log::error!(
            "vulkan: render pipeline with input attachments bound outside a subpass render pass"
        );
        return;
    };
    let Some(active_subpass_index) = state.current_index else {
        log::error!(
            "vulkan: render pipeline with input attachments bound while no subpass is active"
        );
        return;
    };
    let Some(descriptor_set) = encoder
        .subpass_input_attachment_descriptor_sets
        .get(active_subpass_index as usize)
        .copied()
        .flatten()
    else {
        log::error!(
            "vulkan: missing input attachment descriptor set for subpass {}",
            active_subpass_index
        );
        return;
    };
    let subpass_inputs = encoder
        .active_subpass_input_attachments
        .get(active_subpass_index as usize)
        .map_or(&[][..], |inputs| inputs.as_slice());
    if pipeline.input_attachments.iter().any(|binding| {
        !subpass_inputs
            .iter()
            .any(|input_attachment| input_attachment.binding == binding.binding)
    }) {
        log::error!(
            "vulkan: pipeline input attachment binding is missing in active subpass {}",
            active_subpass_index
        );
        return;
    }
    if encoder.active_input_attachment_descriptor_set == Some(descriptor_set) {
        return;
    }
    unsafe {
        encoder.device.raw.cmd_bind_descriptor_sets(
            encoder.active,
            vk::PipelineBindPoint::GRAPHICS,
            pipeline.layout,
            pipeline.input_attachment_descriptor_set_index,
            core::slice::from_ref(&descriptor_set),
            &[],
        );
    }
    encoder.active_input_attachment_descriptor_set = Some(descriptor_set);
}
// tiled-fork: end input-attachment-descriptor-prep

// tiled-fork: begin subpass-pipeline-creation
/// Implementation of [`crate::TiledDevice::create_subpass_render_pipeline`].
///
/// Builds a compatible `VkRenderPass` from `subpass_target` and rewrites the
/// fragment shader's `subpassInput` resource bindings to land on the shared
/// input-attachment descriptor set, then defers to the same naga → SPIR-V
/// → `vkCreateGraphicsPipelines` path that powers
/// [`crate::Device::create_render_pipeline`].
unsafe fn create_subpass_render_pipeline_impl(
    device: &super::Device,
    desc: &crate::RenderPipelineDescriptor<
        '_,
        super::PipelineLayout,
        super::ShaderModule,
        super::PipelineCache,
    >,
    subpass_target: &wgt::SubpassTarget,
) -> Result<super::RenderPipeline, PipelineError> {
    use arrayvec::ArrayVec;

    // 1. Build the compatible `RenderPassKey`.
    let mut compatible_rp_key = super::RenderPassKey {
        sample_count: desc.multisample.count,
        multiview_mask: desc.multiview_mask,
        ..Default::default()
    };
    compatible_rp_key.subpass_dependencies = subpass_target.dependencies.clone();

    // Color attachments come from `subpass_target.color_attachment_formats`.
    for format in &subpass_target.color_attachment_formats {
        let key = format.as_ref().map(|format| super::ColorAttachmentKey {
            base: super::AttachmentKey::compatible(
                device.shared.private_caps.map_texture_format(*format),
                vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL,
            ),
            resolve: None,
        });
        compatible_rp_key.colors.push(key);
    }
    // Depth/stencil format may come from the subpass target or the
    // pipeline's depth-stencil state.
    let compatible_depth_stencil_layout = desc.depth_stencil.as_ref().map_or(
        vk::ImageLayout::DEPTH_STENCIL_ATTACHMENT_OPTIMAL,
        |depth_stencil| {
            if depth_stencil.is_read_only(desc.primitive.cull_mode) {
                vk::ImageLayout::DEPTH_STENCIL_READ_ONLY_OPTIMAL
            } else {
                vk::ImageLayout::DEPTH_STENCIL_ATTACHMENT_OPTIMAL
            }
        },
    );
    let compatible_depth_stencil_format = subpass_target
        .depth_stencil_format
        .or_else(|| desc.depth_stencil.as_ref().map(|state| state.format));
    if let Some(format) = compatible_depth_stencil_format {
        compatible_rp_key.depth_stencil = Some(super::DepthStencilAttachmentKey {
            base: super::AttachmentKey::compatible(
                device.shared.private_caps.map_texture_format(format),
                compatible_depth_stencil_layout,
            ),
            stencil_ops: crate::AttachmentOps::all(),
        });
    }

    // 2. Compute color-attachment-slot → render-pass-attachment-index
    // (dense), to populate `SubpassKey::input_attachment_indices`.
    let mut next_attachment_index: u32 = 0;
    let mut color_attachment_indices: Vec<Option<u32>> =
        vec![None; compatible_rp_key.colors.len()];
    for (slot, color_key) in compatible_rp_key.colors.iter().enumerate() {
        if color_key.is_some() {
            color_attachment_indices[slot] = Some(next_attachment_index);
            next_attachment_index += 1;
        }
    }
    let depth_stencil_attachment_index = compatible_rp_key.depth_stencil.as_ref().map(|_| {
        let index = next_attachment_index;
        next_attachment_index += 1;
        index
    });
    let _ = next_attachment_index;

    // Assemble per-subpass `SubpassKey` entries.
    compatible_rp_key
        .subpasses
        .reserve(subpass_target.subpass_descs.len());
    for subpass in &subpass_target.subpass_descs {
        let color_attachment_indices_for_subpass = subpass
            .color_attachment_indices
            .iter()
            .map(|&attachment_slot| {
                color_attachment_indices
                    .get(attachment_slot as usize)
                    .copied()
                    .flatten()
            })
            .collect::<Vec<_>>();
        let resolve_attachment_indices =
            vec![vk::ATTACHMENT_UNUSED; color_attachment_indices_for_subpass.len()];
        let input_attachment_indices = subpass
            .input_attachment_indices
            .iter()
            .map(|&attachment_slot| {
                if attachment_slot == u32::MAX {
                    return depth_stencil_attachment_index.unwrap_or(vk::ATTACHMENT_UNUSED);
                }
                color_attachment_indices
                    .get(attachment_slot as usize)
                    .copied()
                    .flatten()
                    .unwrap_or(vk::ATTACHMENT_UNUSED)
            })
            .collect::<Vec<_>>();
        compatible_rp_key.subpasses.push(super::SubpassKey {
            color_attachment_indices: color_attachment_indices_for_subpass,
            input_attachment_indices,
            depth_stencil_index: subpass
                .uses_depth_stencil
                .then_some(depth_stencil_attachment_index)
                .flatten(),
            resolve_attachment_indices,
        });
    }

    let current_subpass_desc = subpass_target
        .subpass_descs
        .get(subpass_target.index as usize)
        .ok_or_else(|| {
            PipelineError::Linkage(
                wgt::ShaderStages::FRAGMENT,
                "subpass target index is out of range".to_string(),
            )
        })?;

    // 3. Walk the fragment shader for `ImageClass::Subpass` globals,
    // remap their bindings to the shared input-attachment descriptor set,
    // and remember them on the pipeline so `set_render_pipeline` can bind
    // the per-subpass descriptor set at draw time.
    let mut pipeline_input_attachments: Vec<super::PipelineInputAttachmentBinding> = Vec::new();
    let mut fragment_binding_map = desc.layout.binding_map.clone();
    if let Some(stage) = desc.fragment_stage.as_ref() {
        if let super::ShaderModule::Intermediate {
            ref naga_shader, ..
        } = *stage.module
        {
            let entry_point_index = naga_shader
                .module
                .entry_points
                .iter()
                .position(|ep| ep.stage == naga::ShaderStage::Fragment && ep.name == stage.entry_point)
                .ok_or_else(|| {
                    PipelineError::Linkage(
                        wgt::ShaderStages::FRAGMENT,
                        format!(
                            "could not find fragment entry point '{}' in naga shader module",
                            stage.entry_point
                        ),
                    )
                })?;
            let entry_point_info = naga_shader.info.get_entry_point(entry_point_index);
            for (var_handle, global) in naga_shader.module.global_variables.iter() {
                if entry_point_info[var_handle].is_empty() {
                    continue;
                }
                let Some(binding) = global.binding.as_ref() else {
                    continue;
                };
                let class = match naga_shader.module.types[global.ty].inner {
                    naga::TypeInner::Image { class, .. } => class,
                    naga::TypeInner::BindingArray { base, .. } => match naga_shader.module.types
                        [base]
                        .inner
                    {
                        naga::TypeInner::Image { class, .. } => class,
                        _ => continue,
                    },
                    _ => continue,
                };
                if !class.is_subpass_input() {
                    continue;
                }
                let input_attachment_binding = binding.binding;
                if input_attachment_binding >= device.shared.max_input_attachments {
                    return Err(PipelineError::Linkage(
                        wgt::ShaderStages::FRAGMENT,
                        format!(
                            "input attachment binding {} exceeds backend limit {}",
                            input_attachment_binding, device.shared.max_input_attachments
                        ),
                    ));
                }
                let Some(&attachment_index) = current_subpass_desc
                    .input_attachment_indices
                    .get(input_attachment_binding as usize)
                else {
                    return Err(PipelineError::Linkage(
                        wgt::ShaderStages::FRAGMENT,
                        format!(
                            "input attachment binding {} is out of range for subpass {}",
                            input_attachment_binding, subpass_target.index
                        ),
                    ));
                };
                if attachment_index == u32::MAX
                    && !matches!(
                        class,
                        naga::ImageClass::Subpass {
                            aspect: naga::SubpassAspect::Depth,
                            ..
                        } | naga::ImageClass::Subpass {
                            aspect: naga::SubpassAspect::Stencil,
                            ..
                        }
                    )
                {
                    return Err(PipelineError::Linkage(
                        wgt::ShaderStages::FRAGMENT,
                        format!(
                            "input attachment binding {} is unbound in subpass {}",
                            input_attachment_binding, subpass_target.index
                        ),
                    ));
                }
                fragment_binding_map.insert(
                    *binding,
                    naga::back::spv::BindingInfo {
                        binding_array_size: None,
                        binding: input_attachment_binding,
                        descriptor_set: desc.layout.input_attachment_descriptor_set_index,
                    },
                );
                if !pipeline_input_attachments
                    .iter()
                    .any(|mapped| mapped.binding == input_attachment_binding)
                {
                    pipeline_input_attachments.push(super::PipelineInputAttachmentBinding {
                        binding: input_attachment_binding,
                    });
                }
            }
        } else if !current_subpass_desc.input_attachment_indices.is_empty() {
            return Err(PipelineError::Linkage(
                wgt::ShaderStages::FRAGMENT,
                "vulkan subpass input attachments require naga shader modules".to_string(),
            ));
        }
    }
    if !pipeline_input_attachments.is_empty()
        && desc.layout.input_attachment_descriptor_set_index == u32::MAX
    {
        return Err(PipelineError::Linkage(
            wgt::ShaderStages::FRAGMENT,
            "pipeline layout has no spare descriptor set slot for input attachments".to_string(),
        ));
    }
    pipeline_input_attachments.sort_unstable_by_key(|mapped| mapped.binding);

    // 4. Build the rest of the pipeline create-info using the same logic
    // as `Device::create_render_pipeline`. We re-implement the relevant
    // pieces here rather than call the upstream method because we need to
    // override (a) the binding map for the fragment stage and (b) the
    // render pass + subpass index.

    let dynamic_states = [
        vk::DynamicState::VIEWPORT,
        vk::DynamicState::SCISSOR,
        vk::DynamicState::BLEND_CONSTANTS,
        vk::DynamicState::STENCIL_REFERENCE,
    ];
    let mut stages = ArrayVec::<_, { crate::MAX_CONCURRENT_SHADER_STAGES }>::new();
    let mut vertex_buffers = Vec::new();
    let mut vertex_attributes = Vec::new();

    if let crate::VertexProcessor::Standard {
        vertex_buffers: desc_vertex_buffers,
        vertex_stage: _,
    } = &desc.vertex_processor
    {
        vertex_buffers = Vec::with_capacity(desc_vertex_buffers.len());
        for (i, vb) in desc_vertex_buffers.iter().enumerate() {
            vertex_buffers.push(vk::VertexInputBindingDescription {
                binding: i as u32,
                stride: vb.array_stride as u32,
                input_rate: match vb.step_mode {
                    wgt::VertexStepMode::Vertex => vk::VertexInputRate::VERTEX,
                    wgt::VertexStepMode::Instance => vk::VertexInputRate::INSTANCE,
                },
            });
            for at in vb.attributes {
                vertex_attributes.push(vk::VertexInputAttributeDescription {
                    location: at.shader_location,
                    binding: i as u32,
                    format: conv::map_vertex_format(at.format),
                    offset: at.offset as u32,
                });
            }
        }
    }

    let vk_vertex_input = vk::PipelineVertexInputStateCreateInfo::default()
        .vertex_binding_descriptions(&vertex_buffers)
        .vertex_attribute_descriptions(&vertex_attributes);

    let vk_input_assembly = vk::PipelineInputAssemblyStateCreateInfo::default()
        .topology(conv::map_topology(desc.primitive.topology))
        .primitive_restart_enable(desc.primitive.strip_index_format.is_some());

    let mut compiled_vs = None;
    let mut compiled_ms = None;
    let mut compiled_ts = None;
    match &desc.vertex_processor {
        crate::VertexProcessor::Standard {
            vertex_buffers: _,
            vertex_stage,
        } => {
            let cs = device.compile_stage(
                vertex_stage,
                naga::ShaderStage::Vertex,
                &desc.layout.binding_map,
            )?;
            stages.push(cs.create_info);
            compiled_vs = Some(cs);
        }
        crate::VertexProcessor::Mesh {
            task_stage,
            mesh_stage,
        } => {
            if let Some(t) = task_stage.as_ref() {
                let cs = device.compile_stage(
                    t,
                    naga::ShaderStage::Task,
                    &desc.layout.binding_map,
                )?;
                stages.push(cs.create_info);
                compiled_ts = Some(cs);
            }
            let cs = device.compile_stage(
                mesh_stage,
                naga::ShaderStage::Mesh,
                &desc.layout.binding_map,
            )?;
            stages.push(cs.create_info);
            compiled_ms = Some(cs);
        }
    }
    let compiled_fs = match desc.fragment_stage {
        Some(ref stage) => {
            let compiled =
                device.compile_stage(stage, naga::ShaderStage::Fragment, &fragment_binding_map)?;
            stages.push(compiled.create_info);
            Some(compiled)
        }
        None => None,
    };

    let mut vk_rasterization = vk::PipelineRasterizationStateCreateInfo::default()
        .polygon_mode(conv::map_polygon_mode(desc.primitive.polygon_mode))
        .front_face(conv::map_front_face(desc.primitive.front_face))
        .line_width(1.0)
        .depth_clamp_enable(desc.primitive.unclipped_depth);
    if let Some(face) = desc.primitive.cull_mode {
        vk_rasterization = vk_rasterization.cull_mode(conv::map_cull_face(face))
    }
    let mut vk_rasterization_conservative_state =
        vk::PipelineRasterizationConservativeStateCreateInfoEXT::default()
            .conservative_rasterization_mode(vk::ConservativeRasterizationModeEXT::OVERESTIMATE);
    if desc.primitive.conservative {
        vk_rasterization = vk_rasterization.push_next(&mut vk_rasterization_conservative_state);
    }

    let mut vk_depth_stencil = vk::PipelineDepthStencilStateCreateInfo::default();
    if let Some(ref ds) = desc.depth_stencil {
        if ds.is_depth_enabled() {
            vk_depth_stencil = vk_depth_stencil
                .depth_test_enable(true)
                .depth_write_enable(ds.depth_write_enabled.unwrap_or_default())
                .depth_compare_op(conv::map_comparison(ds.depth_compare.unwrap_or_default()));
        }
        if ds.stencil.is_enabled() {
            let s = &ds.stencil;
            let front = conv::map_stencil_face(&s.front, s.read_mask, s.write_mask);
            let back = conv::map_stencil_face(&s.back, s.read_mask, s.write_mask);
            vk_depth_stencil = vk_depth_stencil
                .stencil_test_enable(true)
                .front(front)
                .back(back);
        }
        if ds.bias.is_enabled() {
            vk_rasterization = vk_rasterization
                .depth_bias_enable(true)
                .depth_bias_constant_factor(ds.bias.constant as f32)
                .depth_bias_clamp(ds.bias.clamp)
                .depth_bias_slope_factor(ds.bias.slope_scale);
        }
    }

    let vk_viewport = vk::PipelineViewportStateCreateInfo::default()
        .flags(vk::PipelineViewportStateCreateFlags::empty())
        .scissor_count(1)
        .viewport_count(1);

    let vk_sample_mask = [
        desc.multisample.mask as u32,
        (desc.multisample.mask >> 32) as u32,
    ];
    let vk_multisample = vk::PipelineMultisampleStateCreateInfo::default()
        .rasterization_samples(vk::SampleCountFlags::from_raw(desc.multisample.count))
        .alpha_to_coverage_enable(desc.multisample.alpha_to_coverage_enabled)
        .sample_mask(&vk_sample_mask);

    let blend_attachment_count = current_subpass_desc.color_attachment_indices.len();
    if desc.color_targets.len() > blend_attachment_count {
        return Err(PipelineError::Linkage(
            wgt::ShaderStages::FRAGMENT,
            "fragment targets exceed active subpass color attachment count".to_string(),
        ));
    }
    let mut vk_attachments = Vec::with_capacity(blend_attachment_count);
    for index in 0..blend_attachment_count {
        let attachment = if let Some(cat) = desc
            .color_targets
            .get(index)
            .and_then(|target| target.as_ref())
        {
            let mut vk_attachment = vk::PipelineColorBlendAttachmentState::default()
                .color_write_mask(vk::ColorComponentFlags::from_raw(cat.write_mask.bits()));
            if let Some(ref blend) = cat.blend {
                let (color_op, color_src, color_dst) = conv::map_blend_component(&blend.color);
                let (alpha_op, alpha_src, alpha_dst) = conv::map_blend_component(&blend.alpha);
                vk_attachment = vk_attachment
                    .blend_enable(true)
                    .color_blend_op(color_op)
                    .src_color_blend_factor(color_src)
                    .dst_color_blend_factor(color_dst)
                    .alpha_blend_op(alpha_op)
                    .src_alpha_blend_factor(alpha_src)
                    .dst_alpha_blend_factor(alpha_dst);
            }
            vk_attachment
        } else {
            vk::PipelineColorBlendAttachmentState::default()
        };
        vk_attachments.push(attachment);
    }
    let vk_color_blend =
        vk::PipelineColorBlendStateCreateInfo::default().attachments(&vk_attachments);

    let vk_dynamic_state =
        vk::PipelineDynamicStateCreateInfo::default().dynamic_states(&dynamic_states);

    let raw_pass = device.shared.make_render_pass(compatible_rp_key)?;

    let vk_infos = [{
        vk::GraphicsPipelineCreateInfo::default()
            .layout(desc.layout.raw)
            .stages(&stages)
            .vertex_input_state(&vk_vertex_input)
            .input_assembly_state(&vk_input_assembly)
            .rasterization_state(&vk_rasterization)
            .viewport_state(&vk_viewport)
            .multisample_state(&vk_multisample)
            .depth_stencil_state(&vk_depth_stencil)
            .color_blend_state(&vk_color_blend)
            .dynamic_state(&vk_dynamic_state)
            .render_pass(raw_pass)
            .subpass(subpass_target.index)
    }];

    let pipeline_cache = desc
        .cache
        .map(|it| it.raw)
        .unwrap_or(vk::PipelineCache::null());

    let mut raw_vec = {
        profiling::scope!("vkCreateGraphicsPipelines");
        unsafe {
            device
                .shared
                .raw
                .create_graphics_pipelines(pipeline_cache, &vk_infos, None)
                .map_err(|(_, e)| super::map_pipeline_err(e))
        }?
    };

    let raw = raw_vec
        .pop()
        .ok_or(PipelineError::Device(DeviceError::Unexpected))?;
    if let Some(label) = desc.label {
        unsafe { device.shared.set_object_name(raw, label) };
    }

    if let Some(CompiledStage {
        temp_raw_module: Some(raw_module),
        ..
    }) = compiled_vs
    {
        unsafe { device.shared.raw.destroy_shader_module(raw_module, None) };
    }
    if let Some(CompiledStage {
        temp_raw_module: Some(raw_module),
        ..
    }) = compiled_ts
    {
        unsafe { device.shared.raw.destroy_shader_module(raw_module, None) };
    }
    if let Some(CompiledStage {
        temp_raw_module: Some(raw_module),
        ..
    }) = compiled_ms
    {
        unsafe { device.shared.raw.destroy_shader_module(raw_module, None) };
    }
    if let Some(CompiledStage {
        temp_raw_module: Some(raw_module),
        ..
    }) = compiled_fs
    {
        unsafe { device.shared.raw.destroy_shader_module(raw_module, None) };
    }

    device.counters.render_pipelines.add(1);

    Ok(super::RenderPipeline {
        raw,
        is_multiview: desc.multiview_mask.is_some(),
        layout: desc.layout.raw,
        input_attachment_descriptor_set_index: desc.layout.input_attachment_descriptor_set_index,
        input_attachments: pipeline_input_attachments,
        subpass_index: subpass_target.index,
    })
}
// tiled-fork: end subpass-pipeline-creation

#[cfg(test)]
mod tests {
    use super::command::{first_active_subpass_index, next_active_subpass_index};
    use wgt::{ActiveSubpassMask, SubpassIndex};

    #[test]
    fn subpass_traversal_sequential() {
        let mask = None;
        assert_eq!(first_active_subpass_index(3, mask), Some(0));
        assert_eq!(next_active_subpass_index(0, 3, mask), 1);
        assert_eq!(next_active_subpass_index(1, 3, mask), 2);
        assert_eq!(next_active_subpass_index(2, 3, mask), 3);
    }

    #[test]
    fn subpass_traversal_skips_culled_middle() {
        let mask = Some(
            ActiveSubpassMask::NONE
                .with(SubpassIndex(0))
                .with(SubpassIndex(2)),
        );
        assert_eq!(first_active_subpass_index(3, mask), Some(0));
        assert_eq!(next_active_subpass_index(0, 3, mask), 2);
        assert_eq!(next_active_subpass_index(2, 3, mask), 3);
    }

    #[test]
    fn subpass_traversal_skips_initial_culled() {
        let mask = Some(
            ActiveSubpassMask::NONE
                .with(SubpassIndex(2))
                .with(SubpassIndex(3)),
        );
        assert_eq!(first_active_subpass_index(4, mask), Some(2));
    }

    #[test]
    fn subpass_traversal_all_culled() {
        let mask = Some(ActiveSubpassMask::NONE);
        assert_eq!(first_active_subpass_index(4, mask), None);
    }

    #[test]
    fn subpass_traversal_zero_subpasses() {
        let mask = Some(ActiveSubpassMask::NONE);
        assert_eq!(first_active_subpass_index(0, mask), None);
    }
}
// tiled-fork: end vulkan-backend
