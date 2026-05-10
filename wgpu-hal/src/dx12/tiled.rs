// tiled-fork: begin dx12-backend
//! Phase 2 stub of the tiled-rendering trait surface for the DX12 backend.
//!
//! DX12 has no direct equivalent of Vulkan input attachments or Metal
//! framebuffer fetch, so all methods stay as `Err(DeviceError::Unexpected)` /
//! `todo!()` permanently. The fork's `MULTI_SUBPASS` /
//! `TRANSIENT_ATTACHMENTS` features are not advertised on DX12, so callers
//! should never reach these methods.

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
        todo!("tiled rendering is intentionally not supported on the DX12 backend")
    }

    unsafe fn next_subpass(&mut self) {
        todo!("tiled rendering is intentionally not supported on the DX12 backend")
    }

    unsafe fn dispatch_transient(&mut self, _dispatch: &TransientDispatch) {
        todo!("tiled rendering is intentionally not supported on the DX12 backend")
    }
}
// tiled-fork: end dx12-backend
