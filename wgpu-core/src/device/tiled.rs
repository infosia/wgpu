// tiled-fork: begin types
//! Phase 3 scaffolding: `Device` methods for creating tile-memory resources.
//!
//! These methods currently call `check_is_valid()` and the
//! feature-gate, then immediately return `Err(DeviceError::Unexpected)`.
//! The per-backend HAL stubs do the same. Real plumbing lands in a later
//! phase along with the backend-real implementations.

use alloc::sync::Arc;

use crate::device::{Device, DeviceError};
use crate::resource_tiled::{
    CreateTransientAttachmentError, CreateTransientDispatchError, TransientAttachment,
    TransientDispatch,
};

impl Device {
    /// Create a transient render attachment.
    ///
    /// Requires [`Features::TRANSIENT_ATTACHMENTS`].
    ///
    /// [`Features::TRANSIENT_ATTACHMENTS`]: wgt::Features::TRANSIENT_ATTACHMENTS
    pub fn create_transient_attachment(
        self: &Arc<Self>,
        _desc: &wgt::TransientAttachmentDescriptor,
    ) -> Result<Arc<TransientAttachment>, CreateTransientAttachmentError> {
        self.check_is_valid()?;
        self.require_features(wgt::Features::TRANSIENT_ATTACHMENTS)?;
        // Backends do not yet implement the HAL surface end-to-end; route
        // every call through `DeviceError::Unexpected` for now. The per-
        // backend `tiled.rs` stubs return the same error.
        Err(CreateTransientAttachmentError::Device(
            DeviceError::from_hal(hal::DeviceError::Unexpected),
        ))
    }

    /// Create a programmable tile-dispatch resource.
    ///
    /// Requires [`Features::PROGRAMMABLE_TILE_DISPATCH`].
    ///
    /// [`Features::PROGRAMMABLE_TILE_DISPATCH`]: wgt::Features::PROGRAMMABLE_TILE_DISPATCH
    pub fn create_transient_dispatch(
        self: &Arc<Self>,
        _desc: &wgt::TransientDispatchDescriptor,
    ) -> Result<Arc<TransientDispatch>, CreateTransientDispatchError> {
        self.check_is_valid()?;
        self.require_features(wgt::Features::PROGRAMMABLE_TILE_DISPATCH)?;
        Err(CreateTransientDispatchError::Device(
            DeviceError::from_hal(hal::DeviceError::Unexpected),
        ))
    }
}
// tiled-fork: end types
