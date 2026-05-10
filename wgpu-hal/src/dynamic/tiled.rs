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
    DynCommandEncoder, DynDevice, DynPipelineCache, DynPipelineLayout, DynQuerySet,
    DynRenderPipeline, DynResource, DynResourceExt as _, DynShaderModule, DynTextureView,
};
use crate::tiled::{
    Subpass, SubpassDepthStencilAttachment, SubpassRenderPassDescriptor, TiledApi,
    TiledCommandEncoder, TiledDevice,
};
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
///
/// `DynTiledCommandEncoder: DynCommandEncoder` so wgpu-core can store a
/// single `Box<dyn DynTiledCommandEncoder>` and reach both the upstream
/// command-encoder API and the tiled extension API through trait upcasting.
pub trait DynTiledCommandEncoder: DynCommandEncoder {
    /// Begin a multi-subpass render pass.
    ///
    /// # Safety
    /// See [`TiledCommandEncoder::begin_subpass_render_pass`].
    unsafe fn begin_subpass_render_pass_dyn(
        &mut self,
        desc: &SubpassRenderPassDescriptor<'_, dyn DynQuerySet, dyn DynTextureView>,
    );

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
    // tiled-fork: begin trait-bound (DynDevice supertrait)
    // `DynTiledDevice: DynDevice` and the blanket `DynDevice` impl now
    // requires `<D::A as Api>::CommandEncoder: TiledCommandEncoder` so it
    // can box created encoders as `Box<dyn DynTiledCommandEncoder>`.
    // Re-state the bound here so this `DynTiledDevice` impl can itself
    // satisfy its `DynDevice` supertrait.
    <<D as Device>::A as Api>::CommandEncoder: TiledCommandEncoder,
    // tiled-fork: end trait-bound (DynDevice supertrait)
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
    E: TiledCommandEncoder + DynResource,
    <E as CommandEncoder>::A: TiledApi,
{
    unsafe fn begin_subpass_render_pass_dyn(
        &mut self,
        desc: &SubpassRenderPassDescriptor<'_, dyn DynQuerySet, dyn DynTextureView>,
    ) {
        use alloc::vec::Vec;

        type ApiOf<E> = <E as CommandEncoder>::A;

        // Downcast persistent (DRAM) color attachments.
        let color_attachments: Vec<_> = desc
            .color_attachments
            .iter()
            .map(|attachment| {
                attachment
                    .as_ref()
                    .map(|attachment| attachment.expect_downcast())
            })
            .collect();

        // Downcast persistent depth/stencil attachment.
        let depth_stencil_attachment = desc
            .depth_stencil_attachment
            .as_ref()
            .map(|ds| ds.expect_downcast());

        // Each subpass owns its own attachment vectors that must outlive the
        // concrete `Subpass` slice we hand off below.
        let subpass_color_attachments: Vec<Vec<_>> = desc
            .subpasses
            .iter()
            .map(|subpass| {
                subpass
                    .color_attachments
                    .iter()
                    .map(|att| att.as_ref().map(|a| a.expect_downcast()))
                    .collect()
            })
            .collect();

        let subpasses: Vec<Subpass<<ApiOf<E> as Api>::TextureView>> = desc
            .subpasses
            .iter()
            .zip(subpass_color_attachments.iter())
            .map(|(subpass, color_attachments)| Subpass::<<ApiOf<E> as Api>::TextureView> {
                color_attachments,
                color_attachment_indices: subpass.color_attachment_indices,
                depth_stencil_attachment: subpass
                    .depth_stencil_attachment
                    .as_ref()
                    .map(SubpassDepthStencilAttachment::expect_downcast),
                input_attachments: subpass.input_attachments,
            })
            .collect();

        let timestamp_writes = desc
            .timestamp_writes
            .as_ref()
            .map(|writes| writes.expect_downcast());

        let occlusion_query_set = desc
            .occlusion_query_set
            .map(|set| set.expect_downcast_ref());

        let concrete = SubpassRenderPassDescriptor::<
            '_,
            <ApiOf<E> as Api>::QuerySet,
            <ApiOf<E> as Api>::TextureView,
        > {
            label: desc.label,
            extent: desc.extent,
            sample_count: desc.sample_count,
            color_attachments: &color_attachments,
            depth_stencil_attachment,
            subpasses: &subpasses,
            subpass_dependencies: desc.subpass_dependencies,
            transient_memory_hint: desc.transient_memory_hint,
            active_subpass_mask: desc.active_subpass_mask,
            multiview_mask: desc.multiview_mask,
            timestamp_writes,
            occlusion_query_set,
        };
        unsafe { <E as TiledCommandEncoder>::begin_subpass_render_pass(self, &concrete) }
    }

    unsafe fn next_subpass_dyn(&mut self) {
        unsafe { <E as TiledCommandEncoder>::next_subpass(self) }
    }

    unsafe fn dispatch_transient_dyn(&mut self, dispatch: &dyn DynTransientDispatch) {
        let dispatch = super::DynResourceExt::expect_downcast_ref(dispatch);
        unsafe { <E as TiledCommandEncoder>::dispatch_transient(self, dispatch) }
    }
}
// tiled-fork: end dynamic
