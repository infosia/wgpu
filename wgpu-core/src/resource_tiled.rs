// tiled-fork: begin types
//! Resource wrappers for tile-based deferred rendering.
//!
//! Phase 3 introduced these as scaffolding placeholders. Phase 11c lit up
//! `TransientAttachment` end-to-end: the wrapper now owns a real
//! `Box<dyn hal::DynTransientAttachment>` allocated by Phase 9 backends
//! (Vulkan: `VK_IMAGE_USAGE_TRANSIENT_ATTACHMENT_BIT` + `LAZILY_ALLOCATED`,
//! Metal: `MTLStorageMode::Memoryless`, GLES: `glRenderbuffer` +
//! `glInvalidateFramebuffer`). `TransientDispatch` is still scaffolding
//! (the programmable-tile-dispatch surface is reserved for a future
//! Apple-GPU phase).

use alloc::boxed::Box;
use alloc::string::String;
use alloc::sync::Arc;
use core::mem::ManuallyDrop;

use thiserror::Error;
use wgt::error::{ErrorType, WebGpuError};

use crate::device::{Device, DeviceError, MissingFeatures};
use crate::resource::{Labeled, TrackingData};
use crate::resource_log;

/// A tile-memory-only render attachment.
///
/// Backed by `MTLStorageModeMemoryless` on Metal,
/// `VK_IMAGE_USAGE_TRANSIENT_ATTACHMENT_BIT` on Vulkan, and a regular GL
/// renderbuffer that gets `glInvalidateFramebuffer`-d on GLES.
#[derive(Debug)]
pub struct TransientAttachment {
    pub(crate) raw: ManuallyDrop<Box<dyn hal::DynTransientAttachment>>,
    pub(crate) device: Arc<Device>,
    /// The `label` carried for diagnostics; `wgt::TransientAttachmentDescriptor`
    /// does not currently have a `label` field, so this is a synthetic
    /// identifier derived from the format + extent. When that descriptor
    /// gains a label, this field will mirror it.
    pub(crate) label: String,
    pub(crate) tracking_data: TrackingData,
    /// Stored for the render-graph and validation work in Phase 11d+; the
    /// HAL handle does not expose its descriptor and the resource may need
    /// to be matched against a `SubpassRenderPassDescriptor`'s transient
    /// table.
    #[allow(dead_code)]
    pub(crate) desc: wgt::TransientAttachmentDescriptor,
}

impl Drop for TransientAttachment {
    fn drop(&mut self) {
        resource_log!("Destroy raw {}", self.error_ident());
        // SAFETY: We are in the Drop impl and we don't use self.raw anymore after this point.
        let raw = unsafe { ManuallyDrop::take(&mut self.raw) };
        unsafe {
            self.device.raw_tiled().destroy_transient_attachment_dyn(raw);
        }
    }
}

impl TransientAttachment {
    /// Direct access to the underlying HAL resource. Used by the render-graph
    /// build path in subsequent bridge phases; marked `#[allow(dead_code)]`
    /// until those callers land.
    #[allow(dead_code)]
    pub(crate) fn raw(&self) -> &dyn hal::DynTransientAttachment {
        self.raw.as_ref()
    }
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
    // Same scaffolding pattern as the original `TransientAttachment` above;
    // the per-field `#[allow(dead_code)]` markers can be removed in the
    // same patch that lights up the programmable-tile-dispatch surface.
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
