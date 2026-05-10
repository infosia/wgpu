// tiled-fork: begin types
//! Public `wgpu` API surface for tile-based, multi-subpass render passes.
//!
//! Phase 11d3 of the upstream-friendly fork plan in `TILED.md`.
//!
//! This module mirrors the wgpu-core `command::subpass` types
//! ([`wgc::command::SubpassRenderPassDescriptor`] and friends) at the
//! public-API layer. Construct a [`SubpassRenderPassDescriptor`] and pass
//! it to [`crate::CommandEncoder::begin_subpass_render_pass`] to obtain a
//! [`SubpassRenderPass`] handle.
//!
//! Per-subpass draw machinery (`set_pipeline`, `set_bind_group`,
//! `set_vertex_buffer`, `set_index_buffer`, `draw`, `draw_indexed`,
//! `set_viewport`, `set_scissor_rect`) was wired in Phase 11d4. See
//! the `wgpu-core::command::subpass` module docs for the known
//! validation/tracker gaps in that path (callers are currently
//! responsible for keeping resource handles alive until submission).
//!
//! [`wgc::command::SubpassRenderPassDescriptor`]: wgc::command::SubpassRenderPassDescriptor

use core::num::NonZeroU32;
use core::ops::Range;

use crate::*;

// tiled-fork: begin subpass-pipeline-descriptor
/// Wrapper descriptor for creating a render pipeline that targets a
/// specific subpass within a multi-subpass render pass.
///
/// Wraps [`RenderPipelineDescriptor`] per the fork's no-breaking-changes
/// rule (TILED.md "Two key wraps" #2). The base descriptor stays
/// upstream-unchanged; this fork-only wrapper carries the additional
/// [`wgt::SubpassTarget`] that Vulkan needs at pipeline-creation time to
/// build a compatible `VkRenderPass`. Metal and GLES consult the target
/// for input-attachment format derivation; backends that do not need
/// per-subpass pipeline binding simply forward to
/// [`Device::create_render_pipeline`].
///
/// Pass to [`Device::create_subpass_render_pipeline`] to build a
/// pipeline that targets the named subpass. The struct is
/// `#[non_exhaustive]`, so external callers must construct it via a
/// helper or using struct-update syntax against a value the fork's
/// own code returns; today this is only viable inside the `wgpu`
/// crate (Phase 11f -- a public Default impl is a candidate
/// follow-up).
#[derive(Clone, Debug)]
#[non_exhaustive]
pub struct SubpassRenderPipelineDescriptor<'a> {
    /// The upstream descriptor for the pipeline.
    pub base: RenderPipelineDescriptor<'a>,
    /// Subpass target metadata: which subpass this pipeline runs in,
    /// the full pass's color/depth formats, subpass descriptors, and
    /// dependencies. Used by Vulkan to build a compatible `VkRenderPass`
    /// at pipeline-creation time; Metal/GLES consult this for
    /// input-attachment format derivation.
    pub subpass_target: wgt::SubpassTarget,
}
#[cfg(send_sync)]
static_assertions::assert_impl_all!(SubpassRenderPipelineDescriptor<'_>: Send, Sync);
// tiled-fork: end subpass-pipeline-descriptor

/// Per-subpass color attachment.
///
/// Mirrors [`wgc::command::SubpassColorAttachment`]. The `Persistent`
/// variant references one of the descriptor's persistent
/// [`RenderPassColorAttachment`]s by reusing the same shape; the
/// `Transient` variant is reserved for the future tile-memory bridge phase
/// and currently returns
/// [`wgc::command::SubpassRenderPassError::TransientNotWired`].
#[derive(Clone, Debug)]
#[non_exhaustive]
pub enum SubpassColorAttachment<'a> {
    /// A regular DRAM-backed color attachment, slotted into the parent
    /// render pass's `color_attachments` array via
    /// [`SubpassDescriptor::color_attachment_indices`].
    Persistent(RenderPassColorAttachment<'a>),
    /// A tile-memory-only color attachment.
    Transient {
        /// Index into the caller's transient-attachment table.
        transient_index: u32,
        /// Load operation; store is implicitly `Discard`.
        ops: wgt::TransientOps<wgt::Color>,
        /// Clear value used when `ops.load == TransientLoadOp::Clear`.
        clear_value: wgt::Color,
    },
}
#[cfg(send_sync)]
static_assertions::assert_impl_all!(SubpassColorAttachment<'_>: Send, Sync);

/// Per-subpass depth/stencil attachment.
///
/// Mirrors [`wgc::command::SubpassDepthStencilAttachment`].
#[derive(Clone, Debug)]
#[non_exhaustive]
pub enum SubpassDepthStencilAttachment<'a> {
    /// A regular DRAM-backed depth/stencil attachment.
    Persistent(RenderPassDepthStencilAttachment<'a>),
    /// A tile-memory-only depth/stencil attachment.
    Transient {
        /// Index into the caller's transient-attachment table.
        transient_index: u32,
        /// Depth load operation; store is implicitly `Discard`.
        depth_ops: wgt::TransientOps<f32>,
        /// Stencil load operation; store is implicitly `Discard`.
        stencil_ops: wgt::TransientOps<u32>,
        /// Clear values for `(depth, stencil)`.
        clear_value: (f32, u32),
    },
}
#[cfg(send_sync)]
static_assertions::assert_impl_all!(SubpassDepthStencilAttachment<'_>: Send, Sync);

/// One subpass in a [`SubpassRenderPassDescriptor`].
///
/// Mirrors [`wgc::command::SubpassDescriptor`].
#[derive(Clone, Debug, Default)]
#[non_exhaustive]
pub struct SubpassDescriptor<'a> {
    /// Per-slot color attachment descriptions for this subpass.
    pub color_attachments: &'a [Option<SubpassColorAttachment<'a>>],
    /// Indices into [`SubpassRenderPassDescriptor::color_attachments`].
    pub color_attachment_indices: &'a [u32],
    /// Optional per-subpass depth/stencil attachment.
    pub depth_stencil_attachment: Option<SubpassDepthStencilAttachment<'a>>,
    /// Input attachments read by this subpass.
    pub input_attachments: &'a [wgt::SubpassInputAttachment],
}
#[cfg(send_sync)]
static_assertions::assert_impl_all!(SubpassDescriptor<'_>: Send, Sync);

/// Descriptor for a multi-subpass render pass.
///
/// Mirrors [`wgc::command::SubpassRenderPassDescriptor`]. The pass-level
/// persistent color and depth/stencil attachments are described with the
/// same [`RenderPassColorAttachment`] / [`RenderPassDepthStencilAttachment`]
/// shape used by [`RenderPassDescriptor`].
#[derive(Clone, Debug, Default)]
#[non_exhaustive]
pub struct SubpassRenderPassDescriptor<'a> {
    /// Debug label of the subpass render pass. Shown in graphics
    /// debuggers for easy identification.
    pub label: Label<'a>,
    /// Render-target extent.
    pub extent: wgt::Extent3d,
    /// Sample count for all attachments.
    pub sample_count: u32,
    /// Persistent color attachments, indexed by
    /// [`SubpassDescriptor::color_attachment_indices`].
    pub color_attachments: &'a [Option<RenderPassColorAttachment<'a>>],
    /// Persistent depth/stencil attachment for the render pass.
    pub depth_stencil_attachment: Option<RenderPassDepthStencilAttachment<'a>>,
    /// Subpasses, executed in order.
    pub subpasses: &'a [SubpassDescriptor<'a>],
    /// Subpass synchronization dependencies.
    pub subpass_dependencies: &'a [wgt::SubpassDependency],
    /// Backend hint for transient-memory behavior.
    pub transient_memory_hint: wgt::TransientMemoryHint,
    /// Optional bitmask selecting which subpasses are active.
    pub active_subpass_mask: Option<wgt::ActiveSubpassMask>,
    /// Multiview layer count, identical semantics to [`RenderPassDescriptor`].
    pub multiview_mask: Option<NonZeroU32>,
    /// Optional pass-level timestamp writes.
    pub timestamp_writes: Option<RenderPassTimestampWrites<'a>>,
    /// Optional occlusion query set.
    pub occlusion_query_set: Option<&'a QuerySet>,
}
#[cfg(send_sync)]
static_assertions::assert_impl_all!(SubpassRenderPassDescriptor<'_>: Send, Sync);

/// In-progress recording of a multi-subpass render pass.
///
/// Created by [`crate::CommandEncoder::begin_subpass_render_pass`].
///
/// In Phase 11d3 the only valid operations on this handle are
/// [`Self::next_subpass`], [`Self::current_subpass_index`], and ending the
/// pass (either explicitly via [`Self::end`] or implicitly via `Drop`).
/// Per-subpass draw machinery (`set_pipeline`, `draw`, etc.) lands in
/// Phase 11d4.
#[derive(Debug)]
pub struct SubpassRenderPass<'encoder> {
    pub(crate) inner: dispatch::DispatchSubpassRenderPass,

    /// This lifetime is used to protect the [`CommandEncoder`] from being
    /// used while the pass is alive. `PhantomDrop` prevents the lifetime
    /// from being shortened.
    pub(crate) _encoder_guard: crate::api::PhantomDrop<&'encoder ()>,
}

#[cfg(send_sync)]
static_assertions::assert_impl_all!(SubpassRenderPass<'_>: Send, Sync);

crate::cmp::impl_eq_ord_hash_proxy!(SubpassRenderPass<'_> => .inner);

impl SubpassRenderPass<'_> {
    /// Drops the lifetime relationship to the parent command encoder,
    /// making usage of the encoder while this pass is recorded a run-time
    /// error instead.
    ///
    /// See [`RenderPass::forget_lifetime`] for the same pattern on
    /// upstream render passes.
    pub fn forget_lifetime(self) -> SubpassRenderPass<'static> {
        SubpassRenderPass {
            inner: self.inner,
            _encoder_guard: crate::api::PhantomDrop::default(),
        }
    }

    /// Advance to the next subpass.
    ///
    /// Errors are routed through the encoder's error sink rather than
    /// being returned, matching the upstream `RenderPass` ergonomics.
    pub fn next_subpass(&mut self) {
        self.inner.next_subpass();
    }

    /// Returns the index of the subpass currently being recorded into,
    /// or `None` if the pass has already ended.
    pub fn current_subpass_index(&self) -> Option<u32> {
        self.inner.current_subpass_index()
    }

    /// End the subpass render pass explicitly.
    ///
    /// Equivalent to dropping the [`SubpassRenderPass`]; this method
    /// exists for parity with WebGPU `GPURenderPassEncoder.end()`.
    /// Calling [`Self::next_subpass`] or [`Self::end`] after this method
    /// is a validation error reported through the encoder's error sink.
    pub fn end(&mut self) {
        self.inner.end();
    }

    // tiled-fork: begin draw-machinery (api)
    /// Sets the active render pipeline.
    ///
    /// Subsequent draw calls will exhibit the behavior defined by `pipeline`.
    /// Mirrors [`RenderPass::set_pipeline`].
    pub fn set_pipeline(&mut self, pipeline: &RenderPipeline) {
        self.inner.set_pipeline(&pipeline.inner);
    }

    /// Sets the active bind group for a given bind group index.
    ///
    /// Mirrors [`RenderPass::set_bind_group`]. Note: the active pipeline
    /// must already have been set with [`Self::set_pipeline`] when this is
    /// called, because the eager-dispatch HAL backend needs the pipeline
    /// layout from the most recently bound pipeline.
    pub fn set_bind_group<'a, BG>(&mut self, index: u32, bind_group: BG, offsets: &[DynamicOffset])
    where
        Option<&'a BindGroup>: From<BG>,
    {
        let bg: Option<&'a BindGroup> = bind_group.into();
        let bg = bg.map(|bg| &bg.inner);
        self.inner.set_bind_group(index, bg, offsets);
    }

    /// Assigns a vertex buffer to a slot.
    ///
    /// Mirrors [`RenderPass::set_vertex_buffer`].
    pub fn set_vertex_buffer(&mut self, slot: u32, buffer_slice: BufferSlice<'_>) {
        self.inner.set_vertex_buffer(
            slot,
            &buffer_slice.buffer.inner,
            buffer_slice.offset,
            Some(buffer_slice.size),
        );
    }

    /// Sets the active index buffer.
    ///
    /// Mirrors [`RenderPass::set_index_buffer`].
    pub fn set_index_buffer(&mut self, buffer_slice: BufferSlice<'_>, index_format: IndexFormat) {
        self.inner.set_index_buffer(
            &buffer_slice.buffer.inner,
            index_format,
            buffer_slice.offset,
            Some(buffer_slice.size),
        );
    }

    /// Draws primitives from the active vertex buffer(s).
    ///
    /// Mirrors [`RenderPass::draw`].
    pub fn draw(&mut self, vertices: Range<u32>, instances: Range<u32>) {
        self.inner.draw(vertices, instances);
    }

    /// Draws indexed primitives using the active index buffer and the
    /// active vertex buffers.
    ///
    /// Mirrors [`RenderPass::draw_indexed`].
    pub fn draw_indexed(&mut self, indices: Range<u32>, base_vertex: i32, instances: Range<u32>) {
        self.inner.draw_indexed(indices, base_vertex, instances);
    }

    /// Sets the viewport used during the rasterization stage.
    ///
    /// Mirrors [`RenderPass::set_viewport`].
    pub fn set_viewport(&mut self, x: f32, y: f32, w: f32, h: f32, min_depth: f32, max_depth: f32) {
        self.inner.set_viewport(x, y, w, h, min_depth, max_depth);
    }

    /// Sets the scissor rectangle used during the rasterization stage.
    ///
    /// Mirrors [`RenderPass::set_scissor_rect`].
    pub fn set_scissor_rect(&mut self, x: u32, y: u32, width: u32, height: u32) {
        self.inner.set_scissor_rect(x, y, width, height);
    }
    // tiled-fork: end draw-machinery (api)

    /// Returns the custom backend implementation of this subpass render
    /// pass, if any.
    #[cfg(custom)]
    pub fn as_custom<T: custom::SubpassRenderPassInterface>(&self) -> Option<&T> {
        self.inner.as_custom()
    }
}

// `Drop` for `SubpassRenderPass` is delegated to the inner
// `DispatchSubpassRenderPass`, whose `Drop` impl in each backend ends the
// subpass render pass if the user hasn't already done so. This mirrors
// the pattern used by [`crate::RenderPass`] and [`crate::ComputePass`],
// which also have no outer `Drop` impl: implementing `Drop` on the outer
// type would forbid moving fields out (e.g. in `forget_lifetime`).
// tiled-fork: end types
