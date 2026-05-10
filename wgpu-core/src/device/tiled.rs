// tiled-fork: begin types
//! `Device` methods for creating tile-memory resources.
//!
//! `create_transient_attachment` is fully wired (Phase 11c) and routes
//! through to the per-backend HAL implementations from Phase 9.
//! `create_transient_dispatch` is still a stub returning
//! `Err(DeviceError::Unexpected)`; the programmable-tile-dispatch
//! surface is reserved for a future Apple-GPU phase.

use alloc::format;
use alloc::sync::Arc;
use core::mem::ManuallyDrop;

use crate::device::{Device, DeviceError};
use crate::resource::TrackingData;
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
        desc: &wgt::TransientAttachmentDescriptor,
    ) -> Result<Arc<TransientAttachment>, CreateTransientAttachmentError> {
        self.check_is_valid()?;
        self.require_features(wgt::Features::TRANSIENT_ATTACHMENTS)?;

        // SAFETY: HAL backends validate the descriptor (size, sample count,
        // format); see the per-backend `tiled.rs` files. No caller-side
        // preconditions for this dyn-dispatched call.
        let raw = unsafe { self.raw_tiled().create_transient_attachment_dyn(desc) }
            .map_err(|hal_err| {
                CreateTransientAttachmentError::Device(self.handle_hal_error(hal_err))
            })?;

        // Synthesize a diagnostics label until
        // `wgt::TransientAttachmentDescriptor` gains a `label` field.
        let extent = match desc.size {
            wgt::TransientSize::Explicit { width, height } => format!("{width}x{height}"),
            _ => "match-target".into(),
        };
        let label = format!(
            "TransientAttachment<{:?},{},samples={}>",
            desc.format, extent, desc.sample_count
        );

        let attachment = TransientAttachment {
            raw: ManuallyDrop::new(raw),
            device: self.clone(),
            label,
            tracking_data: TrackingData::new(self.tracker_indices.transient_attachments.clone()),
            desc: *desc,
        };

        Ok(Arc::new(attachment))
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
        // Programmable tile dispatch is reserved for a future Apple-GPU
        // phase; no backend implements it today.
        Err(CreateTransientDispatchError::Device(
            DeviceError::from_hal(hal::DeviceError::Unexpected),
        ))
    }
}
// tiled-fork: end types
