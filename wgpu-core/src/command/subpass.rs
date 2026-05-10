// tiled-fork: begin types
//! Phase 4 scaffolding: wgpu-core entry points for multi-subpass render passes.
//!
//! This module is intentionally minimal: it introduces a new
//! [`SubpassRenderPass`] handle type and three global methods
//! (`command_encoder_begin_subpass_render_pass`,
//! `render_pass_next_subpass`, `render_pass_current_subpass_index`).
//! The first two return `SubpassRenderPassError::NotImplemented`; the
//! third returns `None`. Per-backend HAL support and full validation
//! logic land in Phase 9 of the renumbered plan.
//!
//! The wgpu public API (Phase 5) targets these stubs so that the
//! end-to-end call path (`wgpu::CommandEncoder::begin_subpass_render_pass`
//! → `Global::command_encoder_begin_subpass_render_pass` → HAL) is wired
//! before any one layer is fully implemented.

use alloc::sync::Arc;

use thiserror::Error;
use wgt::error::{ErrorType, WebGpuError};

use crate::command::CommandEncoder;
use crate::device::DeviceError;
use crate::global::Global;
use crate::id;

/// Opaque handle for an in-progress multi-subpass render pass.
///
/// In Phase 4 this is a placeholder: it cannot be constructed by callers
/// because `command_encoder_begin_subpass_render_pass` always returns
/// `Err(...)`. The shape follows the upstream `RenderPass` so that Phase 9
/// can flesh it out without renaming the public surface.
pub struct SubpassRenderPass {
    /// Parent command encoder; populated when
    /// [`Global::command_encoder_begin_subpass_render_pass`] becomes real
    /// in Phase 9. The Phase 4 stub always leaves it `None`, hence
    /// `#[allow(dead_code)]`.
    #[allow(dead_code)]
    parent: Option<Arc<CommandEncoder>>,
}

impl core::fmt::Debug for SubpassRenderPass {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_struct("SubpassRenderPass").finish_non_exhaustive()
    }
}

/// Errors that can occur driving a [`SubpassRenderPass`].
#[derive(Clone, Debug, Error)]
#[non_exhaustive]
pub enum SubpassRenderPassError {
    /// Underlying device error.
    #[error(transparent)]
    Device(#[from] DeviceError),
    /// The fork's tiled-rendering surface is not yet wired end-to-end.
    ///
    /// Returned from every Phase 4 stub so that callers see a clear
    /// "not implemented" signal rather than a misleading parse/validation
    /// error.
    #[error("tiled-rendering surface is not yet wired end-to-end (fork-only stub)")]
    NotImplemented,
}

impl WebGpuError for SubpassRenderPassError {
    fn webgpu_error_type(&self) -> ErrorType {
        match self {
            Self::Device(e) => e.webgpu_error_type(),
            Self::NotImplemented => ErrorType::Internal,
        }
    }
}

impl Global {
    /// Begin a multi-subpass render pass on the given command encoder.
    ///
    /// In Phase 4 this always returns
    /// [`SubpassRenderPassError::NotImplemented`]. The signature is shaped
    /// so that the public `wgpu::CommandEncoder::begin_subpass_render_pass`
    /// API can call through unchanged once the body is real.
    pub fn command_encoder_begin_subpass_render_pass(
        &self,
        _encoder_id: id::CommandEncoderId,
    ) -> (SubpassRenderPass, Option<SubpassRenderPassError>) {
        (
            SubpassRenderPass { parent: None },
            Some(SubpassRenderPassError::NotImplemented),
        )
    }

    /// Advance the active subpass-mode render pass to its next subpass.
    ///
    /// Phase 4 stub: always returns
    /// [`SubpassRenderPassError::NotImplemented`].
    pub fn render_pass_next_subpass(
        &self,
        _pass: &mut SubpassRenderPass,
    ) -> Result<(), SubpassRenderPassError> {
        Err(SubpassRenderPassError::NotImplemented)
    }

    /// Returns the current subpass index of the active subpass-mode render
    /// pass, or `None` if the pass is not in subpass mode.
    ///
    /// Phase 4 stub: always returns `None`.
    pub fn render_pass_current_subpass_index(
        &self,
        _pass: &SubpassRenderPass,
    ) -> Option<u32> {
        None
    }
}
// tiled-fork: end types
