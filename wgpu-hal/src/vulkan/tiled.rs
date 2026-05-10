// tiled-fork: begin vulkan-backend
//! Tiled-rendering trait surface for the Vulkan backend.
//!
//! This phase provides real implementations for transient attachment
//! resource creation and destruction. Multi-subpass `VkRenderPass`
//! construction, input-attachment descriptor wiring, and the
//! `next_subpass`/`dispatch_transient` command-encoder hooks remain
//! `todo!()` for a later phase. `TransientDispatch` likewise remains a
//! unit-struct stub until compute-style transient dispatch lands.

use ash::vk;

use super::conv;
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
        _desc: &crate::SubpassRenderPassDescriptor<
            '_,
            <super::Api as crate::Api>::QuerySet,
            <super::Api as crate::Api>::TextureView,
        >,
    ) {
        todo!("tiled rendering is not yet wired up for the Vulkan backend")
    }

    unsafe fn next_subpass(&mut self) {
        todo!("tiled rendering is not yet wired up for the Vulkan backend")
    }

    unsafe fn dispatch_transient(&mut self, _dispatch: &TransientDispatch) {
        todo!("tiled rendering is not yet wired up for the Vulkan backend")
    }
}
// tiled-fork: end vulkan-backend
