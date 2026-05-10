// tiled-fork: begin dynamic
//! Dynamic-dispatch surface for the tiled rendering extension.
//!
//! Phase 1 of the upstream-friendly fork plan documented in `TILED.md`.
//!
//! Mirrors `wgpu-hal/src/tiled.rs` but in `dyn`-typed form so that wgpu-core
//! can drive tiled-aware backends without naming the concrete `Api` type.

use alloc::boxed::Box;
use core::fmt;

use super::{
    DynDevice, DynPipelineCache, DynPipelineLayout, DynRenderPipeline, DynResource,
    DynResourceExt as _, DynShaderModule,
};
use crate::tiled::{TiledApi, TiledCommandEncoder, TiledDevice};
use crate::{Api, CommandEncoder, Device, DeviceError, PipelineError, RenderPipelineDescriptor};

// ----- Resource marker traits ---------------------------------------------

/// Marker trait for backend-specific transient attachment resources.
///
/// Used by [`DynTiledDevice`] to box transient attachments behind a trait
/// object so wgpu-core doesn't need the concrete backend type.
pub trait DynTransientAttachment: DynResource + fmt::Debug {}

/// Marker trait for backend-specific transient-dispatch resources.
pub trait DynTransientDispatch: DynResource + fmt::Debug {}

// ----- Dyn extension traits -----------------------------------------------

/// Dynamic-dispatch counterpart to [`TiledDevice`].
pub trait DynTiledDevice: DynDevice {
    /// Create a transient attachment, returning the boxed dyn-typed resource.
    ///
    /// # Safety
    /// See [`TiledDevice::create_transient_attachment`].
    unsafe fn create_transient_attachment_dyn(
        &self,
        desc: &wgt::TransientAttachmentDescriptor,
    ) -> Result<Box<dyn DynTransientAttachment>, DeviceError>;

    /// Destroy a transient attachment.
    ///
    /// # Safety
    /// See [`TiledDevice::destroy_transient_attachment`].
    unsafe fn destroy_transient_attachment_dyn(
        &self,
        attachment: Box<dyn DynTransientAttachment>,
    );

    /// Create a programmable tile-dispatch resource.
    ///
    /// # Safety
    /// See [`TiledDevice::create_transient_dispatch`].
    unsafe fn create_transient_dispatch_dyn(
        &self,
        desc: &wgt::TransientDispatchDescriptor,
    ) -> Result<Box<dyn DynTransientDispatch>, DeviceError>;

    /// Destroy a programmable tile-dispatch resource.
    ///
    /// # Safety
    /// See [`TiledDevice::destroy_transient_dispatch`].
    unsafe fn destroy_transient_dispatch_dyn(&self, dispatch: Box<dyn DynTransientDispatch>);

    /// Create a render pipeline that targets a specific subpass.
    ///
    /// # Safety
    /// See [`TiledDevice::create_subpass_render_pipeline`].
    unsafe fn create_subpass_render_pipeline_dyn(
        &self,
        desc: &RenderPipelineDescriptor<
            dyn DynPipelineLayout,
            dyn DynShaderModule,
            dyn DynPipelineCache,
        >,
        subpass_target: &wgt::SubpassTarget,
    ) -> Result<Box<dyn DynRenderPipeline>, PipelineError>;
}

/// Dynamic-dispatch counterpart to [`TiledCommandEncoder`].
pub trait DynTiledCommandEncoder {
    /// Advance to the next subpass.
    ///
    /// # Safety
    /// See [`TiledCommandEncoder::next_subpass`].
    unsafe fn next_subpass_dyn(&mut self);

    /// Issue a programmable tile dispatch.
    ///
    /// # Safety
    /// See [`TiledCommandEncoder::dispatch_transient`].
    unsafe fn dispatch_transient_dyn(&mut self, dispatch: &dyn DynTransientDispatch);
}

// ----- Blanket impls ------------------------------------------------------
//
// A backend that implements `TiledDevice`/`TiledCommandEncoder` and wires its
// `Api::Device`/`Api::CommandEncoder` types into the upstream `DynResource`
// graph automatically gets the dyn variants for free, mirroring how upstream
// `DynDevice` is provided to anything that implements `Device + DynResource`.

impl<D> DynTiledDevice for D
where
    D: TiledDevice + DynResource,
    <D as Device>::A: TiledApi,
{
    unsafe fn create_transient_attachment_dyn(
        &self,
        desc: &wgt::TransientAttachmentDescriptor,
    ) -> Result<Box<dyn DynTransientAttachment>, DeviceError> {
        let attachment = unsafe { <D as TiledDevice>::create_transient_attachment(self, desc) }?;
        Ok(Box::new(attachment))
    }

    unsafe fn destroy_transient_attachment_dyn(
        &self,
        attachment: Box<dyn DynTransientAttachment>,
    ) {
        // SAFETY: caller guarantees `attachment` was produced by the matching
        // `create_transient_attachment_dyn`, so the unboxing is to the right
        // concrete type.
        let attachment = unsafe { super::DynResourceExt::unbox(attachment) };
        unsafe { <D as TiledDevice>::destroy_transient_attachment(self, attachment) }
    }

    unsafe fn create_transient_dispatch_dyn(
        &self,
        desc: &wgt::TransientDispatchDescriptor,
    ) -> Result<Box<dyn DynTransientDispatch>, DeviceError> {
        let dispatch = unsafe { <D as TiledDevice>::create_transient_dispatch(self, desc) }?;
        Ok(Box::new(dispatch))
    }

    unsafe fn destroy_transient_dispatch_dyn(&self, dispatch: Box<dyn DynTransientDispatch>) {
        // SAFETY: caller guarantees `dispatch` was produced by the matching
        // `create_transient_dispatch_dyn`.
        let dispatch = unsafe { super::DynResourceExt::unbox(dispatch) };
        unsafe { <D as TiledDevice>::destroy_transient_dispatch(self, dispatch) }
    }

    unsafe fn create_subpass_render_pipeline_dyn(
        &self,
        desc: &RenderPipelineDescriptor<
            dyn DynPipelineLayout,
            dyn DynShaderModule,
            dyn DynPipelineCache,
        >,
        subpass_target: &wgt::SubpassTarget,
    ) -> Result<Box<dyn DynRenderPipeline>, PipelineError> {
        let desc = RenderPipelineDescriptor::<
            <<D as Device>::A as Api>::PipelineLayout,
            <<D as Device>::A as Api>::ShaderModule,
            <<D as Device>::A as Api>::PipelineCache,
        > {
            label: desc.label,
            layout: desc.layout.expect_downcast_ref(),
            vertex_processor: match &desc.vertex_processor {
                crate::VertexProcessor::Standard {
                    vertex_buffers,
                    vertex_stage,
                } => crate::VertexProcessor::Standard {
                    vertex_buffers,
                    vertex_stage: vertex_stage.clone().expect_downcast(),
                },
                crate::VertexProcessor::Mesh {
                    task_stage: task,
                    mesh_stage: mesh,
                } => crate::VertexProcessor::Mesh {
                    task_stage: task.as_ref().map(|a| a.clone().expect_downcast()),
                    mesh_stage: mesh.clone().expect_downcast(),
                },
            },
            primitive: desc.primitive,
            depth_stencil: desc.depth_stencil.clone(),
            multisample: desc.multisample,
            fragment_stage: desc.fragment_stage.clone().map(|f| f.expect_downcast()),
            color_targets: desc.color_targets,
            multiview_mask: desc.multiview_mask,
            cache: desc.cache.map(|c| c.expect_downcast_ref()),
        };

        unsafe { <D as TiledDevice>::create_subpass_render_pipeline(self, &desc, subpass_target) }
            .map(|b| -> Box<dyn DynRenderPipeline> { Box::new(b) })
    }
}

impl<E> DynTiledCommandEncoder for E
where
    E: TiledCommandEncoder,
    <E as CommandEncoder>::A: TiledApi,
{
    unsafe fn next_subpass_dyn(&mut self) {
        unsafe { <E as TiledCommandEncoder>::next_subpass(self) }
    }

    unsafe fn dispatch_transient_dyn(&mut self, dispatch: &dyn DynTransientDispatch) {
        let dispatch = super::DynResourceExt::expect_downcast_ref(dispatch);
        unsafe { <E as TiledCommandEncoder>::dispatch_transient(self, dispatch) }
    }
}
// tiled-fork: end dynamic
