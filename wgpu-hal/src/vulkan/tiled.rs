// tiled-fork: begin vulkan-backend
//! Tiled-rendering trait surface for the Vulkan backend.
//!
//! Provides real implementations for transient attachment resource
//! creation, multi-subpass `VkRenderPass` construction, and the
//! `next_subpass` / `begin_subpass_render_pass` command-encoder hooks.
//!
//! Phase 9a3 will wire input-attachment descriptor sets so
//! `subpassLoad(...)` reads work end-to-end at draw time, and will
//! provide a real `dispatch_transient` body. Until then a pipeline that
//! consumes a `subpassInput` will get a correct `VkRenderPass` but its
//! shader-side `subpassLoad` calls will not bind to anything.

use alloc::{vec, vec::Vec};
use ash::vk;
use core::mem;

use super::{command, conv};
// `begin_debug_marker` and `write_timestamp` are inherent methods on the
// upstream `CommandEncoder` trait; bring it into scope so the multi-
// subpass begin path can call them directly.
use crate::CommandEncoder as _;
use crate::DeviceError;

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
    }

    unsafe fn dispatch_transient(&mut self, _dispatch: &TransientDispatch) {
        // TODO(Phase 9a3): real transient dispatch implementation.
        #[allow(clippy::todo)]
        {
            todo!("vulkan: dispatch_transient is deferred to Phase 9a3");
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
    }
    let _ = next_attachment_index;

    let subpass_count = desc.subpasses.len() as u32;

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
                        // TODO(Phase 9a3): wire transient color attachments
                        // through render pass / framebuffer construction.
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

        let input_attachment_indices = subpass
            .input_attachments
            .iter()
            .map(|input_attachment| match input_attachment.source {
                wgt::SubpassInputSource::Color {
                    attachment_index, ..
                } => color_attachment_indices
                    .get(attachment_index as usize)
                    .copied()
                    .flatten()
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
                // TODO(Phase 9a3): wire transient depth/stencil through
                // attachment lists.
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
