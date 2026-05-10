// tiled-fork: begin vulkan-backend
//! Phase 2 stub of the tiled-rendering trait surface for the Vulkan backend.
//!
//! All methods either return `Err(DeviceError::Unexpected)` or `todo!()`.
//! Real implementations (transient images via `VK_IMAGE_USAGE_TRANSIENT_ATTACHMENT_BIT`,
//! multi-subpass `VkRenderPass`, descriptor-set binding for input attachments)
//! land in a dedicated later phase. This stub exists so that the workspace
//! compiles end-to-end after Phase 1 introduced the extension traits.

use crate::DeviceError;

#[derive(Debug)]
pub struct TransientAttachment;

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
        _desc: &wgt::TransientAttachmentDescriptor,
    ) -> Result<TransientAttachment, DeviceError> {
        Err(DeviceError::Unexpected)
    }

    unsafe fn destroy_transient_attachment(&self, _attachment: TransientAttachment) {}

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
