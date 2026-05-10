use alloc::boxed::Box;

use crate::{
    Adapter, Api, DeviceError, OpenDevice, SurfaceCapabilities, TextureFormatCapabilities,
};

// tiled-fork: begin import (DynTiledDevice)
// `DynOpenDevice.device` is stored as `Box<dyn DynTiledDevice>` so the
// fork's tiled-rendering surface (transient attachments, multi-subpass
// passes) is reachable from wgpu-core. `DynTiledDevice: DynDevice`
// (Phase 1) so trait upcasting (stable in Rust 1.86+) makes existing
// `&dyn DynDevice` callers continue to work transparently.
use super::DynTiledDevice;
// tiled-fork: end import (DynTiledDevice)
use super::{DynQueue, DynResource, DynResourceExt, DynSurface};

pub struct DynOpenDevice {
    // tiled-fork: begin field (DynTiledDevice storage)
    pub device: Box<dyn DynTiledDevice>,
    // tiled-fork: end field (DynTiledDevice storage)
    pub queue: Box<dyn DynQueue>,
}

// tiled-fork: begin trait-bound (TiledApi)
// `Box::new(open_device.device)` coerces to `Box<dyn DynTiledDevice>`,
// which requires the concrete `<A>::Device` to impl `DynTiledDevice`.
// The blanket impl in `dynamic/tiled.rs` provides that for any
// `A::Device: TiledDevice + DynResource` whose `A` impls `TiledApi`.
impl<A: Api + crate::TiledApi> From<OpenDevice<A>> for DynOpenDevice
where
    <A as Api>::Device: crate::TiledDevice,
    // Phase 11d1: `DynTiledDevice: DynDevice`, and the `DynDevice` blanket
    // requires the concrete encoder to impl `TiledCommandEncoder`.
    <A as Api>::CommandEncoder: crate::TiledCommandEncoder,
{
// tiled-fork: end trait-bound (TiledApi)
    fn from(open_device: OpenDevice<A>) -> Self {
        Self {
            device: Box::new(open_device.device),
            queue: Box::new(open_device.queue),
        }
    }
}

pub trait DynAdapter: DynResource {
    unsafe fn open(
        &self,
        features: wgt::Features,
        limits: &wgt::Limits,
        memory_hints: &wgt::MemoryHints,
    ) -> Result<DynOpenDevice, DeviceError>;

    unsafe fn texture_format_capabilities(
        &self,
        format: wgt::TextureFormat,
    ) -> TextureFormatCapabilities;

    unsafe fn surface_capabilities(&self, surface: &dyn DynSurface) -> Option<SurfaceCapabilities>;

    unsafe fn get_presentation_timestamp(&self) -> wgt::PresentationTimestamp;

    fn get_ordered_buffer_usages(&self) -> wgt::BufferUses;

    fn get_ordered_texture_usages(&self) -> wgt::TextureUses;

    // tiled-fork: begin tiled-caps
    /// Forwards to [`Adapter::tiled_capabilities`].
    fn tiled_capabilities(&self) -> wgt::TiledCapabilities;
    // tiled-fork: end tiled-caps
}

impl<A: Adapter + DynResource> DynAdapter for A
where
    // tiled-fork: begin trait-bound (DynTiledDevice)
    // The blanket adapter impl needs to box the concrete `<A::A>::Device`
    // as `Box<dyn DynTiledDevice>`. Concrete backend devices satisfy
    // this via the blanket `impl<D: TiledDevice + DynResource>
    // DynTiledDevice for D` in `dynamic/tiled.rs`. Spelling the bounds
    // out here keeps them visible at the call site.
    A::A: crate::TiledApi,
    <A::A as Api>::Device: crate::TiledDevice,
    // Phase 11d1: `DynTiledDevice: DynDevice`, and the `DynDevice` blanket
    // requires the concrete encoder to impl `TiledCommandEncoder` so it
    // can box created encoders as `Box<dyn DynTiledCommandEncoder>`.
    <A::A as Api>::CommandEncoder: crate::TiledCommandEncoder,
    // tiled-fork: end trait-bound (DynTiledDevice)
{
    unsafe fn open(
        &self,
        features: wgt::Features,
        limits: &wgt::Limits,
        memory_hints: &wgt::MemoryHints,
    ) -> Result<DynOpenDevice, DeviceError> {
        unsafe { A::open(self, features, limits, memory_hints) }.map(|open_device| DynOpenDevice {
            device: Box::new(open_device.device),
            queue: Box::new(open_device.queue),
        })
    }

    unsafe fn texture_format_capabilities(
        &self,
        format: wgt::TextureFormat,
    ) -> TextureFormatCapabilities {
        unsafe { A::texture_format_capabilities(self, format) }
    }

    unsafe fn surface_capabilities(&self, surface: &dyn DynSurface) -> Option<SurfaceCapabilities> {
        let surface = surface.expect_downcast_ref();
        unsafe { A::surface_capabilities(self, surface) }
    }

    unsafe fn get_presentation_timestamp(&self) -> wgt::PresentationTimestamp {
        unsafe { A::get_presentation_timestamp(self) }
    }

    fn get_ordered_buffer_usages(&self) -> wgt::BufferUses {
        A::get_ordered_buffer_usages(self)
    }

    fn get_ordered_texture_usages(&self) -> wgt::TextureUses {
        A::get_ordered_texture_usages(self)
    }

    // tiled-fork: begin tiled-caps
    fn tiled_capabilities(&self) -> wgt::TiledCapabilities {
        A::tiled_capabilities(self)
    }
    // tiled-fork: end tiled-caps
}
