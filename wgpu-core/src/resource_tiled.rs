// tiled-fork: begin types
//! Resource wrappers for tile-based deferred rendering.
//!
//! Phase 3 of the upstream-friendly fork plan in `TILED.md`.
//!
//! These wrap the HAL transient-resource handles. In Phase 3 the wrappers
//! exist as types and are wired into the registry/tracker, but they hold
//! no HAL resource yet — `Device::create_transient_attachment` /
//! `Device::create_transient_dispatch` return `Err(DeviceError::Unexpected)`
//! until the per-backend HAL implementations land in a later phase.

use alloc::string::String;
use alloc::sync::Arc;

use thiserror::Error;
use wgt::error::{ErrorType, WebGpuError};

use crate::device::{Device, DeviceError, MissingFeatures};
use crate::resource::TrackingData;

/// A tile-memory-only render attachment.
///
/// Backed by `MTLStorageModeMemoryless` on Metal,
/// `VK_IMAGE_USAGE_TRANSIENT_ATTACHMENT_BIT` on Vulkan, and a regular GL
/// renderbuffer that gets `glInvalidateFramebuffer`-d on GLES.
#[derive(Debug)]
pub struct TransientAttachment {
    // The four fields below become live once
    // `Device::create_transient_attachment` does real HAL work and the
    // render graph queries them. The per-field `#[allow(dead_code)]`
    // markers can be removed in the same patch that lights up that path.
    #[allow(dead_code)]
    pub(crate) device: Arc<Device>,
    #[allow(dead_code)]
    pub(crate) label: String,
    #[allow(dead_code)]
    pub(crate) tracking_data: TrackingData,
    #[allow(dead_code)]
    pub(crate) desc: wgt::TransientAttachmentDescriptor,
}

crate::impl_resource_type!(TransientAttachment);
crate::impl_labeled!(TransientAttachment);
crate::impl_parent_device!(TransientAttachment);
crate::impl_storage_item!(TransientAttachment);
crate::impl_trackable!(TransientAttachment);

/// A programmable tile-dispatch resource.
///
/// Reserved for future Apple-GPU tile-dispatch support; currently all
/// backends return `Err(DeviceError::Unexpected)` from
/// [`Device::create_transient_dispatch`](crate::device::Device::create_transient_dispatch).
#[derive(Debug)]
pub struct TransientDispatch {
    // Same scaffolding pattern as `TransientAttachment` above; the
    // per-field `#[allow(dead_code)]` markers can be removed in the same
    // patch that lights up the real backend path.
    #[allow(dead_code)]
    pub(crate) device: Arc<Device>,
    #[allow(dead_code)]
    pub(crate) label: String,
    #[allow(dead_code)]
    pub(crate) tracking_data: TrackingData,
    #[allow(dead_code)]
    pub(crate) desc: wgt::TransientDispatchDescriptor,
}

crate::impl_resource_type!(TransientDispatch);
crate::impl_labeled!(TransientDispatch);
crate::impl_parent_device!(TransientDispatch);
crate::impl_storage_item!(TransientDispatch);
crate::impl_trackable!(TransientDispatch);

/// Errors that can occur creating a [`TransientAttachment`].
#[derive(Clone, Debug, Error)]
#[non_exhaustive]
pub enum CreateTransientAttachmentError {
    /// Underlying device error.
    #[error(transparent)]
    Device(#[from] DeviceError),
    /// Missing required device feature.
    #[error(transparent)]
    MissingFeatures(#[from] MissingFeatures),
}

impl WebGpuError for CreateTransientAttachmentError {
    fn webgpu_error_type(&self) -> ErrorType {
        match self {
            Self::Device(e) => e.webgpu_error_type(),
            Self::MissingFeatures(e) => e.webgpu_error_type(),
        }
    }
}

/// Errors that can occur creating a [`TransientDispatch`].
#[derive(Clone, Debug, Error)]
#[non_exhaustive]
pub enum CreateTransientDispatchError {
    /// Underlying device error.
    #[error(transparent)]
    Device(#[from] DeviceError),
    /// Missing required device feature.
    #[error(transparent)]
    MissingFeatures(#[from] MissingFeatures),
}

impl WebGpuError for CreateTransientDispatchError {
    fn webgpu_error_type(&self) -> ErrorType {
        match self {
            Self::Device(e) => e.webgpu_error_type(),
            Self::MissingFeatures(e) => e.webgpu_error_type(),
        }
    }
}
// tiled-fork: end types
