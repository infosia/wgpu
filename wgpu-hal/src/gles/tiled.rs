// tiled-fork: begin gles-backend
//! Phase 2 stub of the tiled-rendering trait surface for the GLES backend.
//!
//! All methods either return `Err(DeviceError::Unexpected)` or `todo!()`.
//! Real implementations (renderbuffer transients with `glInvalidateFramebuffer`,
//! framebuffer-fetch / multi-pass-fallback subpass execution) land in a
//! dedicated later phase. This stub exists so that the workspace compiles
//! end-to-end after Phase 1 introduced the extension traits.

use crate::{DeviceError, PipelineError};

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

    unsafe fn create_subpass_render_pipeline(
        &self,
        desc: &crate::RenderPipelineDescriptor<
            '_,
            <super::Api as crate::Api>::PipelineLayout,
            <super::Api as crate::Api>::ShaderModule,
            <super::Api as crate::Api>::PipelineCache,
        >,
        _subpass_target: &wgt::SubpassTarget,
    ) -> Result<<super::Api as crate::Api>::RenderPipeline, PipelineError> {
        // TODO(Phase 9b): GLES will lower subpasses to multi-pass with
        // framebuffer-fetch where available. Until then we forward to the
        // single-subpass pipeline so the surface is reachable.
        unsafe { <Self as crate::Device>::create_render_pipeline(self, desc) }
    }
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
        todo!("tiled rendering is not yet wired up for the GLES backend")
    }

    unsafe fn next_subpass(&mut self) {
        todo!("tiled rendering is not yet wired up for the GLES backend")
    }

    unsafe fn dispatch_transient(&mut self, _dispatch: &TransientDispatch) {
        todo!("tiled rendering is not yet wired up for the GLES backend")
    }
}
// tiled-fork: end gles-backend
