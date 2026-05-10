// tiled-fork: begin types
//! Tile-memory adapter capabilities, queried via [`Adapter::tiled_capabilities`].
//!
//! Phase 5 of the upstream-friendly fork plan in `TILED.md`.
//!
//! `TiledCapabilities` lives in `wgpu-types` so wgpu-hal and wgpu-core
//! can reference it without a forward dependency on the `wgpu` crate. It
//! is re-exported from `wgpu` to preserve the `wgpu::TiledCapabilities`
//! path.
//!
//! [`Adapter::tiled_capabilities`]: crate::Adapter::tiled_capabilities

pub use wgpu_types::TiledCapabilities;
// tiled-fork: end types
