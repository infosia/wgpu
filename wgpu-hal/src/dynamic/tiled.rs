// tiled-fork: begin dynamic
//! Dynamic-dispatch surface for the tiled rendering extension.
//!
//! Phase 1 of the upstream-friendly fork plan documented in `TILED.md`.
//!
//! Mirrors `wgpu-hal/src/tiled.rs` but in `dyn`-typed form so that wgpu-core
//! can drive tiled-aware backends without naming the concrete `Api` type.

use alloc::boxed::Box;
use core::fmt;

use super::{DynDevice, DynResource};
use crate::tiled::{TiledApi, TiledCommandEncoder, TiledDevice};
use crate::{CommandEncoder, Device, DeviceError};

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
