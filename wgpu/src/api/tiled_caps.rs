// tiled-fork: begin types
//! Tile-memory adapter capabilities, queried via [`Adapter::tiled_capabilities`].
//!
//! Phase 5 of the upstream-friendly fork plan in `TILED.md`.
//!
//! `TiledCapabilities` is a sibling type to [`Limits`](crate::Limits) so that
//! upstream `Limits` can stay untouched. Backends populate the four fields
//! when they support tile-based deferred rendering; current Phase 5 stubs
//! return all-zero (no tiled support) until Phase 9 lights up the per-backend
//! HAL implementations.

/// Tile-memory adapter capabilities reported by an [`Adapter`](crate::Adapter).
///
/// Returned from [`Adapter::tiled_capabilities`](crate::Adapter::tiled_capabilities).
///
/// All fields are advisory: a value of `0` means "the adapter does not
/// support tile-based rendering or does not report this metric". Real
/// values are populated by Phase 9 backend implementations.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
#[non_exhaustive]
pub struct TiledCapabilities {
    /// Maximum number of subpasses in a single render pass.
    pub max_subpasses: u32,
    /// Maximum color attachments writable from a single subpass.
    pub max_subpass_color_attachments: u32,
    /// Maximum input attachments readable in a single subpass.
    pub max_input_attachments: u32,
    /// Estimated tile-memory size in bytes, advisory.
    pub estimated_tile_memory_bytes: u32,
}

impl TiledCapabilities {
    /// Returns capabilities that report no tile-based rendering support.
    ///
    /// This is the same as [`TiledCapabilities::default()`].
    #[must_use]
    pub const fn none() -> Self {
        Self {
            max_subpasses: 0,
            max_subpass_color_attachments: 0,
            max_input_attachments: 0,
            estimated_tile_memory_bytes: 0,
        }
    }
}
// tiled-fork: end types

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn none_equals_default() {
        assert_eq!(TiledCapabilities::none(), TiledCapabilities::default());
    }

    #[test]
    fn none_is_all_zero() {
        let caps = TiledCapabilities::none();
        assert_eq!(caps.max_subpasses, 0);
        assert_eq!(caps.max_subpass_color_attachments, 0);
        assert_eq!(caps.max_input_attachments, 0);
        assert_eq!(caps.estimated_tile_memory_bytes, 0);
    }
}
