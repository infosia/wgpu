// tiled-fork: begin types
//! wgpu-core entry points for multi-subpass render passes.
//!
//! Phase 11d2 introduced the begin/next/end machinery; Phase 11d4 added
//! the per-subpass draw machinery (`set_pipeline`, `set_bind_group`,
//! `set_vertex_buffer`, `set_index_buffer`, `draw`, `draw_indexed`,
//! `set_viewport`, `set_scissor_rect`). The module bridges the public
//! `wgpu` API surface for subpass-mode render passes down to the HAL
//! [`hal::DynTiledCommandEncoder`] installed in Phase 11d1.
//!
//! `Transient` color and depth/stencil attachments are deferred: the
//! wgpu-core transient-attachment table is not yet wired (a future
//! bridge phase). Today, both `Transient` arms return
//! [`SubpassRenderPassError::TransientNotWired`].
//!
//! ## Eager vs. deferred — known caller constraint
//!
//! Upstream wgpu-core's `RenderPass` records HAL commands into a
//! deferred `commands` queue and replays them against the HAL encoder
//! during `command_encoder_finish`. The Phase 11d2/11d4 subpass
//! machinery diverges from that model: it issues **eager** HAL calls
//! at the moment the corresponding `Global::*` method runs.
//!
//! **Caller constraint**: a subpass-mode render pass must be the first
//! command-encoder operation. If the caller queued any other commands
//! (e.g. `copy_buffer_to_buffer`) into `cmd_buf.commands` before calling
//! `command_encoder_begin_subpass_render_pass`, those commands will be
//! replayed at `finish`-time *after* the subpass pass has already been
//! encoded into the HAL stream, producing wrong order at the HAL level.
//!
//! ## Known limitations of Phase 11d4
//!
//! The per-subpass draw machinery is intentionally **thin**. It assumes
//! the caller upholds the same invariants the upstream `RenderPass`
//! validator would enforce. Specifically:
//!
//! 1. ~~No resource-tracker registration~~ — **fixed in Phase 11g.**
//!    `set_pipeline`, `set_bind_group`, `set_vertex_buffer`, and
//!    `set_index_buffer` now insert their resource Arcs into the
//!    command buffer's tracker so the resources stay alive until queue
//!    submission. The pattern mirrors upstream `RenderPass`'s tracker
//!    insertion. Buffer bindings additionally record `BufferUses::VERTEX`
//!    or `BufferUses::INDEX` against the buffer tracker via
//!    `BufferTracker::set_single` (the upstream per-draw
//!    `UsageScope::merge_single` barrier-plan path is still deferred to
//!    a future validation-hardening pass).
//! 2. **No pipeline/bind-group/buffer-usage validation**: `set_pipeline`
//!    skips `pass_context.check_compatible`; `set_bind_group` skips
//!    `BindGroupLayout::is_compatible`; `set_*_buffer` skips
//!    `BufferUsages::INDEX/VERTEX` checks; `draw`/`draw_indexed` skip
//!    vertex-buffer-limit checks. Most of these are pre-condition
//!    contracts the public `wgpu::SubpassRenderPass` API can't verify
//!    safely. Future hardening should layer them in.
//! 3. **`set_bind_group(None, ...)`** silently elides the HAL call
//!    rather than unbinding. State drifts from the upstream binder
//!    model.
//! 4. **`set_viewport`/`set_scissor_rect` range/zero-size checks** are
//!    not performed. Out-of-bounds values reach the HAL.
//!
//! Each per-subpass method also calls `cmd_buf_data.invalidate(...)`
//! on dispatch failure (Phase 11d4 review M5 fix) so a subsequent
//! `end_subpass_render_pass` short-circuits the HAL `end_render_pass`
//! call when the encoder is in an inconsistent state.

use alloc::{
    borrow::Cow,
    format,
    string::{String, ToString},
    sync::Arc,
    vec::Vec,
};
use core::num::NonZeroU32;
use core::ops::Range;

use arrayvec::ArrayVec;
use thiserror::Error;
use wgt::{
    error::{ErrorType, WebGpuError},
    BufferAddress, BufferSize, DynamicOffset, IndexFormat, TextureUsages,
};

use crate::binding_model::PipelineLayout;
use crate::command::render::{
    RenderPassColorAttachment, RenderPassDepthStencilAttachment, ResolvedPassChannel,
};
use crate::command::{
    ArcPassTimestampWrites, ArcRenderPassColorAttachment, CommandBufferMutable, CommandEncoder,
    CommandEncoderError, CommandEncoderStatus, EncoderStateError, LoadOp, PassTimestampWrites,
    ResolvedRenderPassDepthStencilAttachment, StoreOp,
};
use crate::device::{Device, DeviceError, MissingFeatures};
use crate::global::Global;
use crate::id;
use crate::resource::{
    Buffer, DestroyedResourceError, InvalidResourceError, Labeled, MissingTextureUsageError,
    ParentDevice, QuerySet, RawResourceAccess, TextureView,
};
use crate::Label;

// ----- Public descriptor types --------------------------------------------

/// One subpass in a [`SubpassRenderPassDescriptor`].
///
/// Mirrors the HAL [`hal::Subpass`] type but uses [`id::TextureViewId`] in
/// place of raw HAL views so wgpu-core can keep the registry as the source
/// of truth.
#[derive(Clone, Debug)]
pub struct SubpassDescriptor<'a> {
    /// Per-slot color attachment descriptions for this subpass.
    pub color_attachments: Cow<'a, [Option<SubpassColorAttachment>]>,
    /// Indices into [`SubpassRenderPassDescriptor::color_attachments`].
    pub color_attachment_indices: Cow<'a, [u32]>,
    /// Optional per-subpass depth/stencil attachment.
    pub depth_stencil_attachment: Option<SubpassDepthStencilAttachment>,
    /// Input attachments read by this subpass.
    pub input_attachments: Cow<'a, [wgt::SubpassInputAttachment]>,
}

/// Per-subpass color attachment, parallel to HAL's `SubpassColorAttachment`.
#[derive(Clone, Debug)]
pub enum SubpassColorAttachment {
    /// A regular DRAM-backed color attachment, slotted into the parent
    /// render pass's `color_attachments` array via
    /// [`SubpassDescriptor::color_attachment_indices`].
    Persistent(RenderPassColorAttachment),
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

/// Per-subpass depth/stencil attachment, parallel to HAL's
/// `SubpassDepthStencilAttachment`.
#[derive(Clone, Debug)]
pub enum SubpassDepthStencilAttachment {
    /// A regular DRAM-backed depth/stencil attachment.
    Persistent(RenderPassDepthStencilAttachment<id::TextureViewId>),
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

/// Descriptor for a multi-subpass render pass.
///
/// Parallel to upstream `RenderPassDescriptor`, but carries the subpass list
/// and the additional knobs required by the HAL surface
/// ([`hal::SubpassRenderPassDescriptor`]).
#[derive(Clone, Debug)]
pub struct SubpassRenderPassDescriptor<'a> {
    /// Optional human-readable label.
    pub label: Label<'a>,
    /// Render-target extent.
    pub extent: wgt::Extent3d,
    /// Sample count for all attachments.
    pub sample_count: u32,
    /// Persistent color attachments, indexed by
    /// [`SubpassDescriptor::color_attachment_indices`].
    pub color_attachments: Cow<'a, [Option<RenderPassColorAttachment>]>,
    /// Persistent depth/stencil attachment for the render pass.
    pub depth_stencil_attachment: Option<RenderPassDepthStencilAttachment<id::TextureViewId>>,
    /// Subpasses, executed in order.
    pub subpasses: Cow<'a, [SubpassDescriptor<'a>]>,
    /// Subpass synchronization dependencies.
    pub subpass_dependencies: Cow<'a, [wgt::SubpassDependency]>,
    /// Backend hint for transient-memory behavior.
    pub transient_memory_hint: wgt::TransientMemoryHint,
    /// Optional bitmask selecting which subpasses are active.
    pub active_subpass_mask: Option<wgt::ActiveSubpassMask>,
    /// Multiview layer count, identical semantics to upstream.
    pub multiview_mask: Option<NonZeroU32>,
    /// Optional pass-level timestamp writes.
    pub timestamp_writes: Option<PassTimestampWrites>,
    /// Optional occlusion query set.
    pub occlusion_query_set: Option<id::QuerySetId>,
}

// ----- Pass handle -------------------------------------------------------

/// Opaque handle for an in-progress multi-subpass render pass.
///
/// Holds an [`Arc<CommandEncoder>`] that keeps the parent encoder locked for
/// the lifetime of the pass. The encoder transitions back to `Recording`
/// when [`Global::render_pass_end_subpass_render_pass`] runs, mirroring the
/// upstream `RenderPass::end` lifecycle.
pub struct SubpassRenderPass {
    /// Parent command encoder. `Some` while the pass is open;
    /// `None` after [`Global::render_pass_end_subpass_render_pass`].
    parent: Option<Arc<CommandEncoder>>,
    /// Number of subpasses declared at begin-time.
    subpass_count: u32,
    /// Index of the subpass currently being recorded into.
    current_subpass: u32,
    /// Recorded label for diagnostics.
    label: Option<String>,
    /// First-error sticky slot. If `next_subpass`/`end` is called on a
    /// pass that already failed, the error is reported again rather than
    /// causing UB.
    error: Option<SubpassRenderPassError>,
    // tiled-fork: begin draw-state
    /// Pipeline layout of the most recently-set render pipeline. Required
    /// for [`Global::render_pass_set_bind_group`] because the HAL
    /// `set_bind_group` entry point takes a pipeline layout, but the public
    /// API only passes a bind group.
    current_pipeline_layout: Option<Arc<PipelineLayout>>,
    // tiled-fork: end draw-state
}

impl core::fmt::Debug for SubpassRenderPass {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_struct("SubpassRenderPass")
            .field("label", &self.label)
            .field("subpass_count", &self.subpass_count)
            .field("current_subpass", &self.current_subpass)
            .field("ended", &self.parent.is_none())
            .field("error", &self.error)
            .finish()
    }
}

impl Drop for SubpassRenderPass {
    /// If the user forgot to call
    /// [`Global::render_pass_end_subpass_render_pass`], end the HAL render
    /// pass and unlock the parent encoder so the encoder isn't left with an
    /// active render pass when its own `Drop` runs (Vulkan requires
    /// `vkCmdEndRenderPass` before `vkEndCommandBuffer`/discard).
    ///
    /// All errors during cleanup are logged. We can't surface them through
    /// `Drop`'s `()` return.
    fn drop(&mut self) {
        let Some(parent) = self.parent.take() else {
            return;
        };

        let mut cmd_buf_data = parent.data.lock();
        let dispatch_result: Result<(), SubpassRenderPassError> =
            with_locked_encoder_mut(&mut cmd_buf_data, |cmd_buf| {
                // SAFETY: pass was opened successfully (parent is Some),
                // matching the begin/end contract.
                unsafe { cmd_buf.encoder.raw.as_mut().end_render_pass() };
                cmd_buf.encoder.close_if_open()?;
                Ok(())
            });
        let unlock_result = cmd_buf_data.unlock_encoder();
        drop(cmd_buf_data);

        if let Err(err) = dispatch_result {
            log::error!(
                "tiled-fork: SubpassRenderPass dropped without explicit end; HAL end_render_pass failed: {err:?}"
            );
        }
        if let Err(err) = unlock_result {
            log::error!(
                "tiled-fork: SubpassRenderPass dropped without explicit end; encoder unlock failed: {err:?}"
            );
        }
    }
}

impl SubpassRenderPass {
    fn new_invalid(err: SubpassRenderPassError, label: Option<String>) -> Self {
        Self {
            parent: None,
            subpass_count: 0,
            current_subpass: 0,
            label,
            error: Some(err),
            current_pipeline_layout: None,
        }
    }

    fn new_open(parent: Arc<CommandEncoder>, subpass_count: u32, label: Option<String>) -> Self {
        Self {
            parent: Some(parent),
            subpass_count,
            current_subpass: 0,
            label,
            error: None,
            current_pipeline_layout: None,
        }
    }

    /// Returns the label originally passed to `begin_subpass_render_pass`.
    #[inline]
    pub fn label(&self) -> Option<&str> {
        self.label.as_deref()
    }
}

// ----- Errors -----------------------------------------------------------

/// Errors that can occur driving a [`SubpassRenderPass`].
#[derive(Clone, Debug, Error)]
#[non_exhaustive]
pub enum SubpassRenderPassError {
    /// Underlying device error.
    #[error(transparent)]
    Device(#[from] DeviceError),
    /// Encoder state machine rejected the begin/next/end call.
    #[error(transparent)]
    EncoderState(#[from] EncoderStateError),
    /// The device is missing a feature required by this entry point.
    #[error(transparent)]
    MissingFeatures(#[from] MissingFeatures),
    /// One of the resource handles in the descriptor was invalid.
    #[error(transparent)]
    InvalidResource(#[from] InvalidResourceError),
    /// A texture view referenced by the descriptor has been destroyed.
    #[error(transparent)]
    DestroyedResource(#[from] DestroyedResourceError),
    /// A texture view does not have the required usage flags.
    #[error(transparent)]
    MissingTextureUsage(#[from] MissingTextureUsageError),
    /// `next_subpass` was called past the last subpass.
    #[error("next_subpass called past the final subpass (current = {current}, count = {count})")]
    NextSubpassPastEnd {
        /// The current subpass index when `next_subpass` was called.
        current: u32,
        /// The number of subpasses declared in the descriptor.
        count: u32,
    },
    /// A method on the pass was called after it had already ended.
    #[error("subpass render pass has already ended")]
    AlreadyEnded,
    /// A pass-level operation was rejected because the descriptor
    /// declared zero subpasses.
    #[error("subpass render pass requires at least one subpass in the descriptor")]
    EmptySubpassList,
    /// `Transient` attachment arms are not wired in this fork bridge yet.
    #[error(
        "transient attachments are not yet wired through wgpu-core (follow-up bridge phase). \
         Use Persistent attachments for now."
    )]
    TransientNotWired,
    /// Generic descriptor-level validation failure (used for cases that
    /// don't match any of the above categories yet).
    #[error("subpass render pass descriptor failed validation: {0}")]
    DescriptorInvalid(String),
    // tiled-fork: begin draw-errors
    /// `set_bind_group` was called before `set_pipeline`, so we don't yet
    /// know which pipeline layout to bind against.
    #[error("set_bind_group called on a subpass render pass with no pipeline set")]
    MissingPipelineForBindGroup,
    // tiled-fork: end draw-errors
}

impl WebGpuError for SubpassRenderPassError {
    fn webgpu_error_type(&self) -> ErrorType {
        match self {
            Self::Device(e) => e.webgpu_error_type(),
            Self::EncoderState(e) => e.webgpu_error_type(),
            Self::MissingFeatures(e) => e.webgpu_error_type(),
            Self::InvalidResource(e) => e.webgpu_error_type(),
            Self::DestroyedResource(e) => e.webgpu_error_type(),
            Self::MissingTextureUsage(e) => e.webgpu_error_type(),
            Self::NextSubpassPastEnd { .. }
            | Self::AlreadyEnded
            | Self::EmptySubpassList
            | Self::TransientNotWired
            | Self::DescriptorInvalid(_)
            | Self::MissingPipelineForBindGroup => ErrorType::Validation,
        }
    }
}

impl From<SubpassRenderPassError> for CommandEncoderError {
    fn from(err: SubpassRenderPassError) -> Self {
        match err {
            SubpassRenderPassError::Device(e) => CommandEncoderError::Device(e),
            SubpassRenderPassError::EncoderState(e) => CommandEncoderError::State(e),
            SubpassRenderPassError::MissingFeatures(e) => CommandEncoderError::MissingFeatures(e),
            SubpassRenderPassError::InvalidResource(e) => CommandEncoderError::InvalidResource(e),
            SubpassRenderPassError::DestroyedResource(e) => {
                CommandEncoderError::DestroyedResource(e)
            }
            // The remaining variants are subpass-specific validation errors
            // that don't have a precise upstream `CommandEncoderError`
            // counterpart. Surface them via the closest existing variant
            // until a dedicated subpass arm is added.
            SubpassRenderPassError::MissingTextureUsage(_)
            | SubpassRenderPassError::NextSubpassPastEnd { .. }
            | SubpassRenderPassError::AlreadyEnded
            | SubpassRenderPassError::EmptySubpassList
            | SubpassRenderPassError::TransientNotWired
            | SubpassRenderPassError::DescriptorInvalid(_)
            | SubpassRenderPassError::MissingPipelineForBindGroup => {
                CommandEncoderError::State(EncoderStateError::Invalid)
            }
        }
    }
}

// ----- Global methods ---------------------------------------------------

impl Global {
    /// Begin a multi-subpass render pass on the given command encoder.
    ///
    /// Phase 11d2 implementation. The descriptor is resolved to HAL types
    /// using the registry, the encoder is transitioned to its locked
    /// state (matching the upstream render-pass lifecycle), and the
    /// HAL [`hal::DynTiledCommandEncoder::begin_subpass_render_pass_dyn`]
    /// is invoked eagerly.
    pub fn command_encoder_begin_subpass_render_pass(
        &self,
        encoder_id: id::CommandEncoderId,
        desc: &SubpassRenderPassDescriptor<'_>,
    ) -> (SubpassRenderPass, Option<SubpassRenderPassError>) {
        profiling::scope!("CommandEncoder::begin_subpass_render_pass");

        let hub = &self.hub;
        let label_owned: Option<String> = desc.label.as_deref().map(ToString::to_string);

        let cmd_enc = hub.command_encoders.get(encoder_id);
        let device = cmd_enc.device.clone();

        // Eager feature gate.
        if let Err(e) = device.require_features(wgt::Features::MULTI_SUBPASS) {
            let err: SubpassRenderPassError = e.into();
            return (
                SubpassRenderPass::new_invalid(err.clone(), label_owned),
                Some(err),
            );
        }
        if let Err(e) = device.check_is_valid() {
            let err: SubpassRenderPassError = SubpassRenderPassError::Device(e);
            return (
                SubpassRenderPass::new_invalid(err.clone(), label_owned),
                Some(err),
            );
        }

        // Resolve & validate the descriptor against the registry.
        let resolved = match resolve_descriptor(hub, &device, desc) {
            Ok(resolved) => resolved,
            Err(err) => {
                return (
                    SubpassRenderPass::new_invalid(err.clone(), label_owned),
                    Some(err),
                );
            }
        };

        // Transition the encoder Recording -> Locked.
        let mut cmd_buf_data = cmd_enc.data.lock();
        match cmd_buf_data.lock_encoder() {
            Ok(()) => {}
            Err(state_err) => {
                drop(cmd_buf_data);
                let err: SubpassRenderPassError = state_err.clone().into();
                let immediate = match state_err {
                    EncoderStateError::Ended | EncoderStateError::Submitted => Some(err.clone()),
                    EncoderStateError::Locked
                    | EncoderStateError::Invalid
                    | EncoderStateError::Unlocked => None,
                };
                return (SubpassRenderPass::new_invalid(err, label_owned), immediate);
            }
        }

        // Build & dispatch the HAL descriptor while holding the lock.
        let dispatch_result: Result<(), SubpassRenderPassError> =
            with_locked_encoder_mut(&mut cmd_buf_data, |cmd_buf| {
                cmd_buf.encoder.close_if_open()?;
                let _ = cmd_buf.encoder.open_pass(label_owned.as_deref())?;

                let snatch_guard = device.snatchable_lock.read();
                // tiled-fork: emit pre-pass texture-layout barriers so the
                // resource tracker matches the layouts the HAL render pass
                // expects on entry. Without this the Vulkan validation layer
                // reports `InvalidImageLayout` (expected
                // COLOR_ATTACHMENT_OPTIMAL, got UNDEFINED) and the
                // post-pass swapchain transition to PRESENT_SRC never fires.
                emit_pre_pass_barriers(cmd_buf, &resolved, &snatch_guard);
                dispatch_hal_begin(
                    cmd_buf.encoder.raw.as_mut(),
                    &resolved,
                    &snatch_guard,
                    &device,
                )?;
                drop(snatch_guard);
                Ok(())
            });

        match dispatch_result {
            Ok(()) => {
                let count = resolved.subpasses.len() as u32;
                drop(cmd_buf_data);
                (
                    SubpassRenderPass::new_open(cmd_enc, count, label_owned),
                    None,
                )
            }
            Err(err) => {
                cmd_buf_data.invalidate(EncoderStateError::Invalid);
                drop(cmd_buf_data);
                (
                    SubpassRenderPass::new_invalid(err.clone(), label_owned),
                    Some(err),
                )
            }
        }
    }

    /// Advance the active subpass-mode render pass to its next subpass.
    ///
    /// Phase 11d2 implementation: validates the current state, dispatches
    /// to [`hal::DynTiledCommandEncoder::next_subpass_dyn`], and updates the
    /// pass's `current_subpass` counter.
    pub fn render_pass_next_subpass(
        &self,
        pass: &mut SubpassRenderPass,
    ) -> Result<(), SubpassRenderPassError> {
        if let Some(err) = pass.error.clone() {
            return Err(err);
        }
        let parent = pass
            .parent
            .as_ref()
            .ok_or(SubpassRenderPassError::AlreadyEnded)?;

        if pass.subpass_count == 0 {
            return Err(SubpassRenderPassError::EmptySubpassList);
        }
        if pass.current_subpass + 1 >= pass.subpass_count {
            return Err(SubpassRenderPassError::NextSubpassPastEnd {
                current: pass.current_subpass,
                count: pass.subpass_count,
            });
        }

        let mut cmd_buf_data = parent.data.lock();
        let dispatch_result: Result<(), SubpassRenderPassError> =
            with_locked_encoder_mut(&mut cmd_buf_data, |cmd_buf| {
                // SAFETY: a successful `begin` opened the pass; we are
                // still inside the same locked encoder.
                unsafe { cmd_buf.encoder.raw.as_mut().next_subpass_dyn() };
                Ok(())
            });
        drop(cmd_buf_data);

        match dispatch_result {
            Ok(()) => {
                pass.current_subpass += 1;
                Ok(())
            }
            Err(err) => {
                pass.error = Some(err.clone());
                Err(err)
            }
        }
    }

    /// Returns the current subpass index of the active subpass-mode render
    /// pass, or `None` if the pass has already ended.
    pub fn render_pass_current_subpass_index(&self, pass: &SubpassRenderPass) -> Option<u32> {
        if pass.parent.is_some() {
            Some(pass.current_subpass)
        } else {
            None
        }
    }

    /// End an active multi-subpass render pass.
    ///
    /// Calls [`hal::CommandEncoder::end_render_pass`] on the underlying
    /// dyn-encoder, then transitions the encoder back to `Recording`.
    pub fn render_pass_end_subpass_render_pass(
        &self,
        pass: &mut SubpassRenderPass,
    ) -> Result<(), SubpassRenderPassError> {
        let parent = pass
            .parent
            .take()
            .ok_or(SubpassRenderPassError::AlreadyEnded)?;

        let mut cmd_buf_data = parent.data.lock();

        let dispatch_result: Result<(), SubpassRenderPassError> =
            with_locked_encoder_mut(&mut cmd_buf_data, |cmd_buf| {
                // SAFETY: the pass was successfully begun, so the encoder is
                // open with a subpass-mode render pass active.
                unsafe { cmd_buf.encoder.raw.as_mut().end_render_pass() };
                cmd_buf.encoder.close_if_open()?;
                Ok(())
            });

        // Always attempt to transition back to Recording, regardless of
        // dispatch outcome. If unlocking fails (encoder was invalidated
        // mid-pass), surface that error too.
        let unlock_result = cmd_buf_data.unlock_encoder();
        drop(cmd_buf_data);

        if let Some(err) = pass.error.clone() {
            return Err(err);
        }
        dispatch_result?;
        unlock_result.map_err(SubpassRenderPassError::from)?;
        Ok(())
    }

    // tiled-fork: begin draw-machinery
    /// Set the active render pipeline for the current subpass.
    pub fn subpass_render_pass_set_pipeline(
        &self,
        pass: &mut SubpassRenderPass,
        pipeline_id: id::RenderPipelineId,
    ) -> Result<(), SubpassRenderPassError> {
        if let Some(err) = pass.error.clone() {
            return Err(err);
        }
        let parent = pass
            .parent
            .as_ref()
            .ok_or(SubpassRenderPassError::AlreadyEnded)?
            .clone();

        let hub = &self.hub;
        let pipeline = match hub.render_pipelines.get(pipeline_id).get() {
            Ok(pipeline) => pipeline,
            Err(e) => {
                let err = SubpassRenderPassError::InvalidResource(e);
                pass.error = Some(err.clone());
                return Err(err);
            }
        };
        let layout = pipeline.layout.clone();

        let mut cmd_buf_data = parent.data.lock();
        let dispatch_result: Result<(), SubpassRenderPassError> =
            with_locked_encoder_mut(&mut cmd_buf_data, |cmd_buf| {
                // tiled-fork: begin tracker (set_pipeline)
                // Pin the pipeline Arc into the command buffer's tracker so
                // it stays alive until queue submission. Mirrors upstream
                // `RenderPass::set_pipeline` -> `state.pass.base.tracker.render_pipelines.insert_single(pipeline)`.
                cmd_buf
                    .trackers
                    .render_pipelines
                    .insert_single(pipeline.clone());
                // tiled-fork: end tracker (set_pipeline)
                // SAFETY: pass was opened, encoder is in a subpass-mode
                // render pass per the begin/end contract.
                unsafe {
                    cmd_buf.encoder.raw.as_mut().set_render_pipeline(pipeline.raw());
                }
                Ok(())
            });
        // On dispatch failure invalidate the encoder so the subsequent
        // end_subpass_render_pass short-circuits the HAL end_render_pass
        // call (the encoder state is now inconsistent).
        if dispatch_result.is_err() {
            cmd_buf_data.invalidate(EncoderStateError::Invalid);
        }
        drop(cmd_buf_data);

        match dispatch_result {
            Ok(()) => {
                pass.current_pipeline_layout = Some(layout);
                Ok(())
            }
            Err(err) => {
                pass.error = Some(err.clone());
                Err(err)
            }
        }
    }

    /// Set a bind group on the current subpass.
    pub fn subpass_render_pass_set_bind_group(
        &self,
        pass: &mut SubpassRenderPass,
        index: u32,
        bind_group_id: Option<id::BindGroupId>,
        offsets: &[DynamicOffset],
    ) -> Result<(), SubpassRenderPassError> {
        if let Some(err) = pass.error.clone() {
            return Err(err);
        }
        let parent = pass
            .parent
            .as_ref()
            .ok_or(SubpassRenderPassError::AlreadyEnded)?
            .clone();

        let hub = &self.hub;
        let layout = match pass.current_pipeline_layout.as_ref() {
            Some(layout) => layout.clone(),
            None => {
                // `set_bind_group(None, ...)` could in principle be a
                // pipeline-layout-free clear, but the HAL `set_bind_group`
                // entry point still wants a layout, so require a pipeline
                // here regardless.
                let err = SubpassRenderPassError::MissingPipelineForBindGroup;
                pass.error = Some(err.clone());
                return Err(err);
            }
        };

        let bind_group = if let Some(id) = bind_group_id {
            match hub.bind_groups.get(id).get() {
                Ok(bg) => Some(bg),
                Err(e) => {
                    let err = SubpassRenderPassError::InvalidResource(e);
                    pass.error = Some(err.clone());
                    return Err(err);
                }
            }
        } else {
            None
        };

        let device = parent.device.clone();
        let snatch_guard = device.snatchable_lock.read();

        let raw_bg = match &bind_group {
            Some(bg) => Some(match bg.try_raw(&snatch_guard) {
                Ok(raw) => raw,
                Err(e) => {
                    drop(snatch_guard);
                    let err = SubpassRenderPassError::DestroyedResource(e);
                    pass.error = Some(err.clone());
                    return Err(err);
                }
            }),
            None => None,
        };

        let mut cmd_buf_data = parent.data.lock();
        let dispatch_result: Result<(), SubpassRenderPassError> =
            with_locked_encoder_mut(&mut cmd_buf_data, |cmd_buf| {
                if let (Some(raw_bg), Some(bg)) = (raw_bg, bind_group.as_ref()) {
                    // tiled-fork: begin tracker (set_bind_group)
                    // Pin the bind-group Arc into the command buffer's
                    // tracker so it stays alive until queue submission.
                    // Mirrors upstream `RenderPass::set_bind_group` ->
                    // `state.pass.base.tracker.bind_groups.insert_single(bind_group)`.
                    // `raw_bg` and `bind_group` are constructed as paired
                    // Options above, so this destructure always matches when
                    // `raw_bg.is_some()`.
                    cmd_buf.trackers.bind_groups.insert_single(bg.clone());
                    // tiled-fork: end tracker (set_bind_group)
                    // SAFETY: pass is open in subpass mode; layout is from
                    // the most recently bound pipeline.
                    unsafe {
                        cmd_buf
                            .encoder
                            .raw
                            .as_mut()
                            .set_bind_group(layout.raw(), index, raw_bg, offsets);
                    }
                } else {
                    // No bind group: there's no per-backend HAL "clear bind
                    // group" call, so this is a no-op at the HAL level.
                    // Upstream tracks this in the binder; for the eager
                    // subpass mode, we simply elide the HAL call.
                    log::trace!(
                        "tiled-fork: SubpassRenderPass::set_bind_group(index={index}, None) elided at HAL level"
                    );
                }
                Ok(())
            });
        drop(cmd_buf_data);
        drop(snatch_guard);

        match dispatch_result {
            Ok(()) => Ok(()),
            Err(err) => {
                pass.error = Some(err.clone());
                Err(err)
            }
        }
    }

    /// Bind a vertex buffer to a slot.
    pub fn subpass_render_pass_set_vertex_buffer(
        &self,
        pass: &mut SubpassRenderPass,
        slot: u32,
        buffer_id: id::BufferId,
        offset: BufferAddress,
        size: Option<BufferSize>,
    ) -> Result<(), SubpassRenderPassError> {
        self.subpass_render_pass_set_buffer_inner(
            pass,
            BufferBindKind::Vertex { slot },
            buffer_id,
            offset,
            size,
        )
    }

    /// Bind an index buffer.
    pub fn subpass_render_pass_set_index_buffer(
        &self,
        pass: &mut SubpassRenderPass,
        buffer_id: id::BufferId,
        format: IndexFormat,
        offset: BufferAddress,
        size: Option<BufferSize>,
    ) -> Result<(), SubpassRenderPassError> {
        self.subpass_render_pass_set_buffer_inner(
            pass,
            BufferBindKind::Index { format },
            buffer_id,
            offset,
            size,
        )
    }

    /// Issue a non-indexed draw.
    pub fn subpass_render_pass_draw(
        &self,
        pass: &mut SubpassRenderPass,
        vertices: Range<u32>,
        instances: Range<u32>,
    ) -> Result<(), SubpassRenderPassError> {
        if let Some(err) = pass.error.clone() {
            return Err(err);
        }
        let parent = pass
            .parent
            .as_ref()
            .ok_or(SubpassRenderPassError::AlreadyEnded)?
            .clone();

        let mut cmd_buf_data = parent.data.lock();
        let dispatch_result: Result<(), SubpassRenderPassError> =
            with_locked_encoder_mut(&mut cmd_buf_data, |cmd_buf| {
                // SAFETY: pass open in subpass mode; pipeline must be bound
                // by the caller (HAL contract).
                unsafe {
                    cmd_buf.encoder.raw.as_mut().draw(
                        vertices.start,
                        vertices.end - vertices.start,
                        instances.start,
                        instances.end - instances.start,
                    );
                }
                Ok(())
            });
        drop(cmd_buf_data);

        match dispatch_result {
            Ok(()) => Ok(()),
            Err(err) => {
                pass.error = Some(err.clone());
                Err(err)
            }
        }
    }

    /// Issue an indexed draw.
    pub fn subpass_render_pass_draw_indexed(
        &self,
        pass: &mut SubpassRenderPass,
        indices: Range<u32>,
        base_vertex: i32,
        instances: Range<u32>,
    ) -> Result<(), SubpassRenderPassError> {
        if let Some(err) = pass.error.clone() {
            return Err(err);
        }
        let parent = pass
            .parent
            .as_ref()
            .ok_or(SubpassRenderPassError::AlreadyEnded)?
            .clone();

        let mut cmd_buf_data = parent.data.lock();
        let dispatch_result: Result<(), SubpassRenderPassError> =
            with_locked_encoder_mut(&mut cmd_buf_data, |cmd_buf| {
                // SAFETY: pass open in subpass mode; pipeline + index buffer
                // must be bound by the caller (HAL contract).
                unsafe {
                    cmd_buf.encoder.raw.as_mut().draw_indexed(
                        indices.start,
                        indices.end - indices.start,
                        base_vertex,
                        instances.start,
                        instances.end - instances.start,
                    );
                }
                Ok(())
            });
        drop(cmd_buf_data);

        match dispatch_result {
            Ok(()) => Ok(()),
            Err(err) => {
                pass.error = Some(err.clone());
                Err(err)
            }
        }
    }

    /// Set the viewport.
    pub fn subpass_render_pass_set_viewport(
        &self,
        pass: &mut SubpassRenderPass,
        x: f32,
        y: f32,
        width: f32,
        height: f32,
        min_depth: f32,
        max_depth: f32,
    ) -> Result<(), SubpassRenderPassError> {
        if let Some(err) = pass.error.clone() {
            return Err(err);
        }
        let parent = pass
            .parent
            .as_ref()
            .ok_or(SubpassRenderPassError::AlreadyEnded)?
            .clone();

        let mut cmd_buf_data = parent.data.lock();
        let dispatch_result: Result<(), SubpassRenderPassError> =
            with_locked_encoder_mut(&mut cmd_buf_data, |cmd_buf| {
                let rect = hal::Rect {
                    x,
                    y,
                    w: width,
                    h: height,
                };
                // SAFETY: pass open in subpass mode.
                unsafe {
                    cmd_buf
                        .encoder
                        .raw
                        .as_mut()
                        .set_viewport(&rect, min_depth..max_depth);
                }
                Ok(())
            });
        drop(cmd_buf_data);

        match dispatch_result {
            Ok(()) => Ok(()),
            Err(err) => {
                pass.error = Some(err.clone());
                Err(err)
            }
        }
    }

    /// Set the scissor rectangle.
    pub fn subpass_render_pass_set_scissor_rect(
        &self,
        pass: &mut SubpassRenderPass,
        x: u32,
        y: u32,
        width: u32,
        height: u32,
    ) -> Result<(), SubpassRenderPassError> {
        if let Some(err) = pass.error.clone() {
            return Err(err);
        }
        let parent = pass
            .parent
            .as_ref()
            .ok_or(SubpassRenderPassError::AlreadyEnded)?
            .clone();

        let mut cmd_buf_data = parent.data.lock();
        let dispatch_result: Result<(), SubpassRenderPassError> =
            with_locked_encoder_mut(&mut cmd_buf_data, |cmd_buf| {
                let rect = hal::Rect {
                    x,
                    y,
                    w: width,
                    h: height,
                };
                // SAFETY: pass open in subpass mode.
                unsafe {
                    cmd_buf.encoder.raw.as_mut().set_scissor_rect(&rect);
                }
                Ok(())
            });
        drop(cmd_buf_data);

        match dispatch_result {
            Ok(()) => Ok(()),
            Err(err) => {
                pass.error = Some(err.clone());
                Err(err)
            }
        }
    }

    /// Shared `set_vertex_buffer` / `set_index_buffer` body.
    fn subpass_render_pass_set_buffer_inner(
        &self,
        pass: &mut SubpassRenderPass,
        kind: BufferBindKind,
        buffer_id: id::BufferId,
        offset: BufferAddress,
        size: Option<BufferSize>,
    ) -> Result<(), SubpassRenderPassError> {
        if let Some(err) = pass.error.clone() {
            return Err(err);
        }
        let parent = pass
            .parent
            .as_ref()
            .ok_or(SubpassRenderPassError::AlreadyEnded)?
            .clone();

        let hub = &self.hub;
        let buffer: Arc<Buffer> = match hub.buffers.get(buffer_id).get() {
            Ok(b) => b,
            Err(e) => {
                let err = SubpassRenderPassError::InvalidResource(e);
                pass.error = Some(err.clone());
                return Err(err);
            }
        };

        let device = parent.device.clone();
        let snatch_guard = device.snatchable_lock.read();
        let (binding, _resolved_size) = match buffer.binding(offset, size, &snatch_guard) {
            Ok(pair) => pair,
            Err(e) => {
                drop(snatch_guard);
                let err = SubpassRenderPassError::DescriptorInvalid(format!(
                    "buffer binding error: {e}"
                ));
                pass.error = Some(err.clone());
                return Err(err);
            }
        };

        let mut cmd_buf_data = parent.data.lock();
        let dispatch_result: Result<(), SubpassRenderPassError> =
            with_locked_encoder_mut(&mut cmd_buf_data, |cmd_buf| {
                // tiled-fork: begin tracker (set_buffer_inner)
                // Pin the buffer Arc into the command buffer's BufferTracker
                // so it stays alive until queue submission. We use
                // `set_single` (not `merge_single`) because:
                //
                //   * Phase 11g's goal is lifetime: the BufferTracker holds
                //     `Arc<Buffer>` and frees it at submission, which is
                //     what we need.
                //   * Upstream's `merge_single` lives on a `UsageScope`
                //     (not the BufferTracker) and is used for per-draw
                //     usage-state merging into the pass-scoped barrier
                //     plan; that machinery isn't yet wired for subpass
                //     mode (a future hardening pass).
                //
                // The returned `Option<PendingTransition>` would describe a
                // state transition that the (eager) HAL dispatch already
                // performed externally; we ignore it.
                let usage = match kind {
                    BufferBindKind::Vertex { .. } => wgt::BufferUses::VERTEX,
                    BufferBindKind::Index { .. } => wgt::BufferUses::INDEX,
                };
                let _ = cmd_buf.trackers.buffers.set_single(&buffer, usage);
                // tiled-fork: end tracker (set_buffer_inner)
                match kind {
                    BufferBindKind::Vertex { slot } => {
                        // SAFETY: pass open in subpass mode; binding has the
                        // buffer's lifetime via snatch_guard.
                        unsafe {
                            cmd_buf
                                .encoder
                                .raw
                                .as_mut()
                                .set_vertex_buffer(slot, binding);
                        }
                    }
                    BufferBindKind::Index { format } => {
                        // SAFETY: same as above.
                        unsafe {
                            cmd_buf
                                .encoder
                                .raw
                                .as_mut()
                                .set_index_buffer(binding, format);
                        }
                    }
                }
                Ok(())
            });
        drop(cmd_buf_data);
        drop(snatch_guard);

        match dispatch_result {
            Ok(()) => Ok(()),
            Err(err) => {
                pass.error = Some(err.clone());
                Err(err)
            }
        }
    }
    // tiled-fork: end draw-machinery
}

// tiled-fork: begin draw-machinery-helpers
#[derive(Copy, Clone, Debug)]
enum BufferBindKind {
    Vertex { slot: u32 },
    Index { format: IndexFormat },
}
// tiled-fork: end draw-machinery-helpers

// ----- Resolution / dispatch helpers (private) --------------------------

struct ResolvedDescriptor {
    label: Option<String>,
    extent: wgt::Extent3d,
    sample_count: u32,
    color_attachments:
        ArrayVec<Option<ArcRenderPassColorAttachment>, { hal::MAX_COLOR_ATTACHMENTS }>,
    depth_stencil_attachment: Option<ResolvedRenderPassDepthStencilAttachment<Arc<TextureView>>>,
    subpasses: Vec<ResolvedSubpass>,
    subpass_dependencies: Vec<wgt::SubpassDependency>,
    transient_memory_hint: wgt::TransientMemoryHint,
    active_subpass_mask: Option<wgt::ActiveSubpassMask>,
    multiview_mask: Option<NonZeroU32>,
    timestamp_writes: Option<ArcPassTimestampWrites>,
    occlusion_query_set: Option<Arc<QuerySet>>,
}

struct ResolvedSubpass {
    color_attachments: Vec<Option<ResolvedSubpassColorAttachment>>,
    color_attachment_indices: Vec<u32>,
    depth_stencil_attachment: Option<ResolvedSubpassDepthStencilAttachment>,
    input_attachments: Vec<wgt::SubpassInputAttachment>,
}

enum ResolvedSubpassColorAttachment {
    Persistent(ArcRenderPassColorAttachment),
    /// Reserved for future bridge phase. The resolver currently rejects
    /// `Transient` arms before reaching this variant, but we keep it so
    /// the resolved structure mirrors the public descriptor shape.
    #[allow(dead_code)]
    Transient {
        transient_index: u32,
        ops: wgt::TransientOps<wgt::Color>,
        clear_value: wgt::Color,
    },
}

enum ResolvedSubpassDepthStencilAttachment {
    Persistent(ResolvedRenderPassDepthStencilAttachment<Arc<TextureView>>),
    #[allow(dead_code)]
    Transient {
        transient_index: u32,
        depth_ops: wgt::TransientOps<f32>,
        stencil_ops: wgt::TransientOps<u32>,
        clear_value: (f32, u32),
    },
}

fn resolve_descriptor(
    hub: &crate::hub::Hub,
    device: &Arc<Device>,
    desc: &SubpassRenderPassDescriptor<'_>,
) -> Result<ResolvedDescriptor, SubpassRenderPassError> {
    let texture_views = hub.texture_views.read();
    let query_sets = hub.query_sets.read();

    let max_color_attachments = device.limits.max_color_attachments as usize;
    if desc.color_attachments.len() > max_color_attachments {
        return Err(SubpassRenderPassError::DescriptorInvalid(format!(
            "too many color attachments: given {}, limit {}",
            desc.color_attachments.len(),
            max_color_attachments
        )));
    }

    // -- Persistent color attachments -------------------------------------
    let mut resolved_colors = ArrayVec::new();
    for color in desc.color_attachments.iter() {
        match color {
            Some(att) => {
                let view = texture_views.get(att.view).get()?;
                view.same_device(device)?;
                if !view.desc.usage.contains(TextureUsages::RENDER_ATTACHMENT) {
                    return Err(SubpassRenderPassError::MissingTextureUsage(
                        MissingTextureUsageError {
                            res: view.error_ident(),
                            actual: view.desc.usage,
                            expected: TextureUsages::RENDER_ATTACHMENT,
                        },
                    ));
                }
                if view.desc.usage.contains(TextureUsages::TRANSIENT)
                    && att.store_op != StoreOp::Discard
                {
                    return Err(SubpassRenderPassError::DescriptorInvalid(
                        "TRANSIENT color attachment requires StoreOp::Discard".into(),
                    ));
                }
                let resolve_target = if let Some(rt_id) = att.resolve_target {
                    let rt = texture_views.get(rt_id).get()?;
                    rt.same_device(device)?;
                    if !rt.desc.usage.contains(TextureUsages::RENDER_ATTACHMENT) {
                        return Err(SubpassRenderPassError::MissingTextureUsage(
                            MissingTextureUsageError {
                                res: rt.error_ident(),
                                actual: rt.desc.usage,
                                expected: TextureUsages::RENDER_ATTACHMENT,
                            },
                        ));
                    }
                    Some(rt)
                } else {
                    None
                };
                resolved_colors.push(Some(ArcRenderPassColorAttachment {
                    view,
                    depth_slice: att.depth_slice,
                    resolve_target,
                    load_op: att.load_op,
                    store_op: att.store_op,
                }));
            }
            None => resolved_colors.push(None),
        }
    }

    // -- Persistent depth/stencil ----------------------------------------
    let resolved_depth_stencil = if let Some(ds) = &desc.depth_stencil_attachment {
        Some(resolve_persistent_depth_stencil(&texture_views, device, ds)?)
    } else {
        None
    };

    // -- Subpasses --------------------------------------------------------
    let mut resolved_subpasses = Vec::with_capacity(desc.subpasses.len());
    for subpass in desc.subpasses.iter() {
        let mut sub_colors = Vec::with_capacity(subpass.color_attachments.len());
        for color in subpass.color_attachments.iter() {
            match color {
                Some(SubpassColorAttachment::Persistent(att)) => {
                    let view = texture_views.get(att.view).get()?;
                    view.same_device(device)?;
                    if !view.desc.usage.contains(TextureUsages::RENDER_ATTACHMENT) {
                        return Err(SubpassRenderPassError::MissingTextureUsage(
                            MissingTextureUsageError {
                                res: view.error_ident(),
                                actual: view.desc.usage,
                                expected: TextureUsages::RENDER_ATTACHMENT,
                            },
                        ));
                    }
                    let resolve_target = if let Some(rt_id) = att.resolve_target {
                        let rt = texture_views.get(rt_id).get()?;
                        rt.same_device(device)?;
                        if !rt.desc.usage.contains(TextureUsages::RENDER_ATTACHMENT) {
                            return Err(SubpassRenderPassError::MissingTextureUsage(
                                MissingTextureUsageError {
                                    res: rt.error_ident(),
                                    actual: rt.desc.usage,
                                    expected: TextureUsages::RENDER_ATTACHMENT,
                                },
                            ));
                        }
                        Some(rt)
                    } else {
                        None
                    };
                    sub_colors.push(Some(ResolvedSubpassColorAttachment::Persistent(
                        ArcRenderPassColorAttachment {
                            view,
                            depth_slice: att.depth_slice,
                            resolve_target,
                            load_op: att.load_op,
                            store_op: att.store_op,
                        },
                    )));
                }
                Some(SubpassColorAttachment::Transient { .. }) => {
                    log::error!(
                        "tiled-fork: SubpassColorAttachment::Transient is not yet wired in wgpu-core"
                    );
                    return Err(SubpassRenderPassError::TransientNotWired);
                }
                None => sub_colors.push(None),
            }
        }
        let sub_ds = match &subpass.depth_stencil_attachment {
            Some(SubpassDepthStencilAttachment::Persistent(ds)) => Some(
                ResolvedSubpassDepthStencilAttachment::Persistent(
                    resolve_persistent_depth_stencil(&texture_views, device, ds)?,
                ),
            ),
            Some(SubpassDepthStencilAttachment::Transient { .. }) => {
                log::error!(
                    "tiled-fork: SubpassDepthStencilAttachment::Transient is not yet wired in wgpu-core"
                );
                return Err(SubpassRenderPassError::TransientNotWired);
            }
            None => None,
        };
        resolved_subpasses.push(ResolvedSubpass {
            color_attachments: sub_colors,
            color_attachment_indices: subpass.color_attachment_indices.to_vec(),
            depth_stencil_attachment: sub_ds,
            input_attachments: subpass.input_attachments.to_vec(),
        });
    }

    // -- Timestamp writes & occlusion query set --------------------------
    let timestamp_writes = if let Some(tw) = &desc.timestamp_writes {
        let qs = query_sets.get(tw.query_set).get()?;
        qs.same_device(device)?;
        Some(ArcPassTimestampWrites {
            query_set: qs,
            beginning_of_pass_write_index: tw.beginning_of_pass_write_index,
            end_of_pass_write_index: tw.end_of_pass_write_index,
        })
    } else {
        None
    };

    let occlusion_query_set = if let Some(qs_id) = desc.occlusion_query_set {
        let qs = query_sets.get(qs_id).get()?;
        qs.same_device(device)?;
        Some(qs)
    } else {
        None
    };

    Ok(ResolvedDescriptor {
        label: desc.label.as_deref().map(ToString::to_string),
        extent: desc.extent,
        sample_count: desc.sample_count,
        color_attachments: resolved_colors,
        depth_stencil_attachment: resolved_depth_stencil,
        subpasses: resolved_subpasses,
        subpass_dependencies: desc.subpass_dependencies.to_vec(),
        transient_memory_hint: desc.transient_memory_hint,
        active_subpass_mask: desc.active_subpass_mask,
        multiview_mask: desc.multiview_mask,
        timestamp_writes,
        occlusion_query_set,
    })
}

fn resolve_persistent_depth_stencil(
    texture_views: &crate::lock::RwLockReadGuard<
        '_,
        crate::storage::Storage<crate::resource::Fallible<TextureView>>,
    >,
    device: &Arc<Device>,
    ds: &RenderPassDepthStencilAttachment<id::TextureViewId>,
) -> Result<ResolvedRenderPassDepthStencilAttachment<Arc<TextureView>>, SubpassRenderPassError> {
    let view = texture_views.get(ds.view).get()?;
    view.same_device(device)?;
    if !view.desc.usage.contains(TextureUsages::RENDER_ATTACHMENT) {
        return Err(SubpassRenderPassError::MissingTextureUsage(
            MissingTextureUsageError {
                res: view.error_ident(),
                actual: view.desc.usage,
                expected: TextureUsages::RENDER_ATTACHMENT,
            },
        ));
    }
    let format = view.desc.format;
    let depth = if format.has_depth_aspect() {
        resolve_pass_channel(&ds.depth, 0.0)?
    } else {
        ResolvedPassChannel::ReadOnly
    };
    let stencil = if format.has_stencil_aspect() {
        resolve_pass_channel(&ds.stencil, 0)?
    } else {
        ResolvedPassChannel::ReadOnly
    };
    Ok(ResolvedRenderPassDepthStencilAttachment {
        view,
        depth,
        stencil,
    })
}

/// Inline implementation of `PassChannel::resolve`. We don't call the
/// upstream private method because it lives in another module; instead,
/// reproduce its logic with a default clear value when the channel doesn't
/// supply one, since this entry point has no AttachmentError variant in
/// the subpass error enum.
fn resolve_pass_channel<V>(
    channel: &crate::command::PassChannel<Option<V>>,
    default_clear: V,
) -> Result<ResolvedPassChannel<V>, SubpassRenderPassError>
where
    V: Copy + Default,
{
    if channel.read_only {
        if channel.load_op.is_some() {
            return Err(SubpassRenderPassError::DescriptorInvalid(
                "depth/stencil channel marked read_only but supplies load_op".into(),
            ));
        }
        if channel.store_op.is_some() {
            return Err(SubpassRenderPassError::DescriptorInvalid(
                "depth/stencil channel marked read_only but supplies store_op".into(),
            ));
        }
        Ok(ResolvedPassChannel::ReadOnly)
    } else {
        let load = match channel
            .load_op
            .ok_or_else(|| {
                SubpassRenderPassError::DescriptorInvalid(
                    "depth/stencil channel missing load_op".into(),
                )
            })?
        {
            LoadOp::Clear(clear_value) => LoadOp::Clear(clear_value.unwrap_or(default_clear)),
            LoadOp::DontCare(t) => LoadOp::DontCare(t),
            LoadOp::Load => LoadOp::Load,
        };
        let store = channel.store_op.ok_or_else(|| {
            SubpassRenderPassError::DescriptorInvalid("depth/stencil channel missing store_op".into())
        })?;
        Ok(ResolvedPassChannel::Operational(wgt::Operations {
            load,
            store,
        }))
    }
}

/// Emit `transition_textures` barriers ahead of `vkCmdBeginRenderPass` so
/// the wgpu-core resource tracker reflects the layouts the HAL render pass
/// expects on entry. Walks the descriptor's persistent color, color-resolve,
/// and depth/stencil attachments, sets their tracker state to the matching
/// `TextureUses`, and emits any pending transitions as one batched HAL call.
///
/// Errors during raw-view lookup (e.g. a destroyed texture) are silently
/// skipped because the subsequent `dispatch_hal_begin` will re-report the
/// missing view through `SubpassRenderPassError::DescriptorInvalid`, which
/// surfaces a better diagnostic.
fn emit_pre_pass_barriers(
    cmd_buf: &mut CommandBufferMutable,
    resolved: &ResolvedDescriptor,
    snatch_guard: &crate::snatch::SnatchGuard<'_>,
) {
    let mut barriers: Vec<hal::TextureBarrier<'_, dyn hal::DynTexture>> = Vec::new();

    // Color attachments + their resolve targets. Each attachment view's
    // Arc is registered in `trackers.views` so it stays alive until queue
    // submission completes -- mirrors upstream `RenderPassInfo::start`
    // (see `wgpu-core/src/command/render.rs:1499-1505`). Without this the
    // view (and any cached `VkFramebuffer` referencing it) gets destroyed
    // while a pending command buffer still references it, surfacing as
    // `VUID-vkDestroyImageView-imageView-01026` /
    // `VUID-vkDestroyFramebuffer-framebuffer-00892` in the Vulkan
    // validation layer.
    for color in &resolved.color_attachments {
        let Some(at) = color else { continue };
        cmd_buf.trackers.views.insert_single(at.view.clone());
        push_transitions(
            &mut cmd_buf.trackers,
            &at.view.parent,
            at.view.selector.clone(),
            wgt::TextureUses::COLOR_TARGET,
            snatch_guard,
            &mut barriers,
        );
        if let Some(ref resolve_view) = at.resolve_target {
            cmd_buf.trackers.views.insert_single(resolve_view.clone());
            push_transitions(
                &mut cmd_buf.trackers,
                &resolve_view.parent,
                resolve_view.selector.clone(),
                wgt::TextureUses::COLOR_TARGET,
                snatch_guard,
                &mut barriers,
            );
        }
    }

    // Depth/stencil attachment. Phase 11j default: treat as
    // `DEPTH_STENCIL_WRITE` because the deferred-rendering example and the
    // typical multi-subpass shape write depth in the geometry pass. A
    // read-only-depth refinement (matching upstream
    // `RenderPassInfo::start`) is a Phase 11j follow-up.
    if let Some(ref ds) = resolved.depth_stencil_attachment {
        cmd_buf.trackers.views.insert_single(ds.view.clone());
        push_transitions(
            &mut cmd_buf.trackers,
            &ds.view.parent,
            ds.view.selector.clone(),
            wgt::TextureUses::DEPTH_STENCIL_WRITE,
            snatch_guard,
            &mut barriers,
        );
    }

    if !barriers.is_empty() {
        unsafe {
            cmd_buf.encoder.raw.as_mut().transition_textures(&barriers);
        }
    }
}

fn push_transitions<'a>(
    trackers: &mut crate::track::Tracker,
    texture: &'a Arc<crate::resource::Texture>,
    selector: wgt::TextureSelector,
    usage: wgt::TextureUses,
    snatch_guard: &'a crate::snatch::SnatchGuard<'_>,
    out: &mut Vec<hal::TextureBarrier<'a, dyn hal::DynTexture>>,
) {
    let pending: Vec<_> = trackers
        .textures
        .set_single(texture, selector, usage)
        .collect();
    if let Some(raw) = texture.raw(snatch_guard) {
        for p in pending {
            out.push(p.into_hal(raw));
        }
    }
}

/// Build the HAL descriptor inline and call `begin_subpass_render_pass_dyn`.
fn dispatch_hal_begin(
    raw: &mut dyn hal::DynTiledCommandEncoder,
    resolved: &ResolvedDescriptor,
    snatch_guard: &crate::snatch::SnatchGuard<'_>,
    device: &Device,
) -> Result<(), SubpassRenderPassError> {
    // -- Persistent color attachments -> hal::ColorAttachment ------------
    let mut hal_colors: Vec<Option<hal::ColorAttachment<'_, dyn hal::DynTextureView>>> =
        Vec::with_capacity(resolved.color_attachments.len());
    for att in resolved.color_attachments.iter() {
        match att {
            Some(att) => {
                let resolve = if let Some(rt) = &att.resolve_target {
                    Some(hal::Attachment {
                        view: rt.try_raw(snatch_guard)?,
                        usage: wgt::TextureUses::COLOR_TARGET,
                    })
                } else {
                    None
                };
                hal_colors.push(Some(hal::ColorAttachment {
                    target: hal::Attachment {
                        view: att.view.try_raw(snatch_guard)?,
                        usage: wgt::TextureUses::COLOR_TARGET,
                    },
                    depth_slice: att.depth_slice,
                    resolve_target: resolve,
                    ops: load_store_to_hal_ops(att.load_op, att.store_op),
                    clear_value: load_clear(att.load_op),
                }));
            }
            None => hal_colors.push(None),
        }
    }

    // -- Persistent depth/stencil -> hal::DepthStencilAttachment --------
    let hal_depth_stencil = if let Some(ds) = &resolved.depth_stencil_attachment {
        let usage = if ds.depth_is_readonly()
            && ds.stencil_is_readonly()
            && device
                .downlevel
                .flags
                .contains(wgt::DownlevelFlags::READ_ONLY_DEPTH_STENCIL)
        {
            wgt::TextureUses::DEPTH_STENCIL_READ | wgt::TextureUses::RESOURCE
        } else {
            wgt::TextureUses::DEPTH_STENCIL_WRITE
        };
        Some(hal::DepthStencilAttachment {
            target: hal::Attachment {
                view: ds.view.try_raw(snatch_guard)?,
                usage,
            },
            depth_ops: channel_to_hal_ops(&ds.depth),
            stencil_ops: channel_to_hal_ops(&ds.stencil),
            clear_value: (channel_clear(&ds.depth), channel_clear(&ds.stencil)),
        })
    } else {
        None
    };

    // -- Per-subpass color attachment vectors ---------------------------
    let mut sub_colors: Vec<Vec<Option<hal::SubpassColorAttachment<'_, dyn hal::DynTextureView>>>> =
        Vec::with_capacity(resolved.subpasses.len());
    let mut sub_ds: Vec<Option<hal::SubpassDepthStencilAttachment<'_, dyn hal::DynTextureView>>> =
        Vec::with_capacity(resolved.subpasses.len());
    for subpass in resolved.subpasses.iter() {
        let mut colors: Vec<Option<hal::SubpassColorAttachment<'_, dyn hal::DynTextureView>>> =
            Vec::with_capacity(subpass.color_attachments.len());
        for color in subpass.color_attachments.iter() {
            match color {
                Some(ResolvedSubpassColorAttachment::Persistent(att)) => {
                    let resolve = if let Some(rt) = &att.resolve_target {
                        Some(hal::Attachment {
                            view: rt.try_raw(snatch_guard)?,
                            usage: wgt::TextureUses::COLOR_TARGET,
                        })
                    } else {
                        None
                    };
                    colors.push(Some(hal::SubpassColorAttachment::Persistent(
                        hal::ColorAttachment {
                            target: hal::Attachment {
                                view: att.view.try_raw(snatch_guard)?,
                                usage: wgt::TextureUses::COLOR_TARGET,
                            },
                            depth_slice: att.depth_slice,
                            resolve_target: resolve,
                            ops: load_store_to_hal_ops(att.load_op, att.store_op),
                            clear_value: load_clear(att.load_op),
                        },
                    )));
                }
                Some(ResolvedSubpassColorAttachment::Transient { .. }) => {
                    // The resolver always rejects `Transient` arms in this
                    // phase (returns TransientNotWired). If we ever reach
                    // this branch, surface it as a validation error rather
                    // than panicking.
                    return Err(SubpassRenderPassError::TransientNotWired);
                }
                None => colors.push(None),
            }
        }
        sub_colors.push(colors);

        let ds = match &subpass.depth_stencil_attachment {
            Some(ResolvedSubpassDepthStencilAttachment::Persistent(ds)) => {
                let usage = if ds.depth_is_readonly()
                    && ds.stencil_is_readonly()
                    && device
                        .downlevel
                        .flags
                        .contains(wgt::DownlevelFlags::READ_ONLY_DEPTH_STENCIL)
                {
                    wgt::TextureUses::DEPTH_STENCIL_READ | wgt::TextureUses::RESOURCE
                } else {
                    wgt::TextureUses::DEPTH_STENCIL_WRITE
                };
                Some(hal::SubpassDepthStencilAttachment::Persistent(
                    hal::DepthStencilAttachment {
                        target: hal::Attachment {
                            view: ds.view.try_raw(snatch_guard)?,
                            usage,
                        },
                        depth_ops: channel_to_hal_ops(&ds.depth),
                        stencil_ops: channel_to_hal_ops(&ds.stencil),
                        clear_value: (channel_clear(&ds.depth), channel_clear(&ds.stencil)),
                    },
                ))
            }
            Some(ResolvedSubpassDepthStencilAttachment::Transient { .. }) => {
                return Err(SubpassRenderPassError::TransientNotWired);
            }
            None => None,
        };
        sub_ds.push(ds);
    }

    // Borrow-stable index/input arrays.
    let sub_indices: Vec<&[u32]> = resolved
        .subpasses
        .iter()
        .map(|s| s.color_attachment_indices.as_slice())
        .collect();
    let sub_inputs: Vec<&[wgt::SubpassInputAttachment]> = resolved
        .subpasses
        .iter()
        .map(|s| s.input_attachments.as_slice())
        .collect();

    // Drain `sub_ds` into per-subpass `Subpass` values. `Option::take`
    // gives us each owned attachment exactly once, sidestepping the
    // missing `Clone` bound on `SubpassDepthStencilAttachment<'_, dyn _>`.
    //
    // `hal::Subpass` is `#[non_exhaustive]`, so we cannot use struct-literal
    // syntax cross-crate. Instead, start from `Default::default()` and
    // mutate through the public field accessors.
    let subpass_count = resolved.subpasses.len();
    let mut subpasses: Vec<hal::Subpass<'_, dyn hal::DynTextureView>> =
        Vec::with_capacity(subpass_count);
    for i in 0..subpass_count {
        let mut sp: hal::Subpass<'_, dyn hal::DynTextureView> = Default::default();
        sp.color_attachments = sub_colors[i].as_slice();
        sp.color_attachment_indices = sub_indices[i];
        sp.depth_stencil_attachment = sub_ds[i].take();
        sp.input_attachments = sub_inputs[i];
        subpasses.push(sp);
    }

    let timestamp_writes_hal =
        resolved
            .timestamp_writes
            .as_ref()
            .map(|tw| hal::PassTimestampWrites::<'_, dyn hal::DynQuerySet> {
                query_set: tw.query_set.raw(),
                beginning_of_pass_write_index: tw.beginning_of_pass_write_index,
                end_of_pass_write_index: tw.end_of_pass_write_index,
            });

    let occlusion_qs: Option<&dyn hal::DynQuerySet> = resolved
        .occlusion_query_set
        .as_ref()
        .map(|qs| qs.raw());

    // `hal::SubpassRenderPassDescriptor` is `#[non_exhaustive]`; same
    // pattern as `hal::Subpass` above.
    let hal_label_str = resolved.label.as_deref();
    let mut hal_desc: hal::SubpassRenderPassDescriptor<
        '_,
        dyn hal::DynQuerySet,
        dyn hal::DynTextureView,
    > = Default::default();
    hal_desc.label = hal_label_str;
    hal_desc.extent = resolved.extent;
    hal_desc.sample_count = resolved.sample_count;
    hal_desc.color_attachments = hal_colors.as_slice();
    hal_desc.depth_stencil_attachment = hal_depth_stencil;
    hal_desc.subpasses = subpasses.as_slice();
    hal_desc.subpass_dependencies = resolved.subpass_dependencies.as_slice();
    hal_desc.transient_memory_hint = resolved.transient_memory_hint;
    hal_desc.active_subpass_mask = resolved.active_subpass_mask;
    hal_desc.multiview_mask = resolved.multiview_mask;
    hal_desc.timestamp_writes = timestamp_writes_hal;
    hal_desc.occlusion_query_set = occlusion_qs;

    // SAFETY: feature gate verified above; encoder is open via the
    // caller's `open_pass`; descriptor lifetime extends to end of this
    // function (inclusive of the synchronous HAL call).
    unsafe { raw.begin_subpass_render_pass_dyn(&hal_desc) };
    Ok(())
}

// Tiny extension trait so we can ask a `ResolvedRenderPassDepthStencilAttachment`
// whether each channel is read-only without re-importing the inner enum.
trait ResolvedDepthStencilExt {
    fn depth_is_readonly(&self) -> bool;
    fn stencil_is_readonly(&self) -> bool;
}

impl<TV> ResolvedDepthStencilExt for ResolvedRenderPassDepthStencilAttachment<TV> {
    fn depth_is_readonly(&self) -> bool {
        matches!(self.depth, ResolvedPassChannel::ReadOnly)
    }
    fn stencil_is_readonly(&self) -> bool {
        matches!(self.stencil, ResolvedPassChannel::ReadOnly)
    }
}

fn load_store_to_hal_ops<V>(load: LoadOp<V>, store: StoreOp) -> hal::AttachmentOps {
    let load_bits = match load {
        LoadOp::Load => hal::AttachmentOps::LOAD,
        LoadOp::Clear(_) => hal::AttachmentOps::LOAD_CLEAR,
        LoadOp::DontCare(_) => hal::AttachmentOps::LOAD_DONT_CARE,
    };
    let store_bits = match store {
        StoreOp::Store => hal::AttachmentOps::STORE,
        StoreOp::Discard => hal::AttachmentOps::STORE_DISCARD,
    };
    load_bits | store_bits
}

fn load_clear<V: Default + Copy>(load: LoadOp<V>) -> V {
    match load {
        LoadOp::Clear(v) => v,
        _ => V::default(),
    }
}

fn channel_to_hal_ops<V>(channel: &ResolvedPassChannel<V>) -> hal::AttachmentOps
where
    V: Copy + Default,
{
    match channel {
        ResolvedPassChannel::ReadOnly => hal::AttachmentOps::LOAD | hal::AttachmentOps::STORE,
        ResolvedPassChannel::Operational(ops) => {
            let load = match ops.load {
                LoadOp::Load => hal::AttachmentOps::LOAD,
                LoadOp::Clear(_) => hal::AttachmentOps::LOAD_CLEAR,
                LoadOp::DontCare(_) => hal::AttachmentOps::LOAD_DONT_CARE,
            };
            let store = match ops.store {
                StoreOp::Store => hal::AttachmentOps::STORE,
                StoreOp::Discard => hal::AttachmentOps::STORE_DISCARD,
            };
            load | store
        }
    }
}

fn channel_clear<V: Copy + Default>(channel: &ResolvedPassChannel<V>) -> V {
    match channel {
        ResolvedPassChannel::Operational(ops) => match ops.load {
            LoadOp::Clear(v) => v,
            _ => V::default(),
        },
        _ => V::default(),
    }
}

/// Run a closure with `&mut CommandBufferMutable` while the encoder is in
/// the [`CommandEncoderStatus::Locked`] state.
fn with_locked_encoder_mut<R>(
    status: &mut CommandEncoderStatus,
    f: impl FnOnce(&mut CommandBufferMutable) -> Result<R, SubpassRenderPassError>,
) -> Result<R, SubpassRenderPassError> {
    match status {
        CommandEncoderStatus::Locked(inner) => f(inner),
        CommandEncoderStatus::Recording(_) => Err(SubpassRenderPassError::EncoderState(
            EncoderStateError::Unlocked,
        )),
        CommandEncoderStatus::Finished(_) | CommandEncoderStatus::Consumed => Err(
            SubpassRenderPassError::EncoderState(EncoderStateError::Ended),
        ),
        CommandEncoderStatus::Error(_) => Err(SubpassRenderPassError::EncoderState(
            EncoderStateError::Invalid,
        )),
        // CommandEncoderStatus::Transitioning is only ever observed inside
        // `mem::replace` calls in lock_encoder/unlock_encoder/finish; an
        // outside caller can't see it. Per CLAUDE.md "no panics in library
        // code" we surface a benign Invalid error rather than `unreachable!()`.
        CommandEncoderStatus::Transitioning => Err(SubpassRenderPassError::EncoderState(
            EncoderStateError::Invalid,
        )),
    }
}
// tiled-fork: end types
