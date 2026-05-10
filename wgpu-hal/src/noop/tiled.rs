// tiled-fork: begin noop-backend
//! Phase 2 stub of the tiled-rendering trait surface for the noop backend.
//!
//! All methods either return `Err(DeviceError::Unexpected)` or `todo!()`.
//! Real implementations land in later phases (when wgpu-core needs the surface
//! to be reachable end-to-end). This stub exists so that the workspace
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

impl crate::TiledDevice for super::Context {
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

impl crate::TiledCommandEncoder for super::CommandBuffer {
    unsafe fn begin_subpass_render_pass(
        &mut self,
        _desc: &crate::SubpassRenderPassDescriptor<
            '_,
            <super::Api as crate::Api>::QuerySet,
            <super::Api as crate::Api>::TextureView,
        >,
    ) {
        todo!("tiled rendering is not yet wired up for the noop backend")
    }

    unsafe fn next_subpass(&mut self) {
        todo!("tiled rendering is not yet wired up for the noop backend")
    }

    unsafe fn dispatch_transient(&mut self, _dispatch: &TransientDispatch) {
        todo!("tiled rendering is not yet wired up for the noop backend")
    }
}
// tiled-fork: end noop-backend
