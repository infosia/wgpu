// tiled-fork: begin types
//! Backend-agnostic types for the tile-based deferred rendering (TBDR) extension.
//!
//! This module is fork-only and lives outside the upstream `render.rs` so that
//! the diff against upstream stays clean. See `TILED.md` at the repo root for
//! the porting plan and conventions.
//!
//! ## Serde naming convention
//!
//! Enums use `serde(rename_all = "kebab-case")` (matches WebGPU IDL conventions
//! for tagged unions). Structs use `serde(rename_all = "camelCase")` (matches
//! WebGPU IDL conventions for property bags).

#[cfg(feature = "serde")]
use serde::{Deserialize, Serialize};

use alloc::vec::Vec;

use crate::TextureFormat;

/// Sentinel input-attachment slot index that denotes the render pass
/// depth/stencil attachment instead of a color attachment slot.
///
/// Used in [`SubpassTargetDesc::input_attachment_indices`] entries.
pub const DEPTH_STENCIL_INPUT_ATTACHMENT_INDEX: u32 = u32::MAX;

/// Size specification for a transient attachment.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Default)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
#[cfg_attr(feature = "serde", serde(rename_all = "kebab-case"))]
#[non_exhaustive]
pub enum TransientSize {
    /// Match the render pass target size.
    #[default]
    MatchTarget,
    /// Use an explicit transient attachment size.
    Explicit {
        /// Explicit width.
        width: u32,
        /// Explicit height.
        height: u32,
    },
}

/// Descriptor for creating a transient attachment.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
#[cfg_attr(feature = "serde", serde(rename_all = "camelCase"))]
#[non_exhaustive]
pub struct TransientAttachmentDescriptor {
    /// Texture format of the transient attachment.
    pub format: TextureFormat,
    /// Attachment dimensions.
    pub size: TransientSize,
    /// Sample count for the attachment.
    pub sample_count: u32,
}

/// Operation to perform for a transient attachment at pass start.
///
/// Transient attachments have no persistent contents, so there is no `Load`
/// variant - either clear with a value, or treat the contents as undefined.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
#[cfg_attr(feature = "serde", serde(rename_all = "kebab-case"))]
#[non_exhaustive]
pub enum TransientLoadOp<V> {
    /// Clear the transient attachment with a value.
    Clear(V),
    /// Existing contents are undefined.
    DontCare,
}

impl<V: Default> Default for TransientLoadOp<V> {
    fn default() -> Self {
        Self::Clear(V::default())
    }
}

/// Operations for transient attachment usage.
///
/// Store is always implicitly `Discard` for transient attachments.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
#[non_exhaustive]
pub struct TransientOps<V> {
    /// Load operation.
    pub load: TransientLoadOp<V>,
}

impl<V: Default> Default for TransientOps<V> {
    fn default() -> Self {
        Self {
            load: TransientLoadOp::default(),
        }
    }
}

/// A subpass index inside a render pass.
#[repr(transparent)]
#[derive(Copy, Clone, Ord, PartialOrd, Eq, PartialEq, Hash, Debug, Default)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct SubpassIndex(pub u32);

/// Source for a subpass input attachment.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
#[cfg_attr(feature = "serde", serde(rename_all = "kebab-case"))]
#[non_exhaustive]
pub enum SubpassInputSource {
    /// Color attachment source from a previous subpass.
    Color {
        /// Source subpass.
        subpass: SubpassIndex,
        /// Source color attachment slot index.
        attachment_index: u32,
    },
    /// Depth/stencil attachment source from a previous subpass.
    Depth {
        /// Source subpass.
        subpass: SubpassIndex,
    },
}

/// Input attachment declaration for one binding.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
#[cfg_attr(feature = "serde", serde(rename_all = "camelCase"))]
#[non_exhaustive]
pub struct SubpassInputAttachment {
    /// Binding slot used by the shader.
    pub binding: u32,
    /// Input source.
    pub source: SubpassInputSource,
}

/// High-level dependency type between two subpasses.
///
/// Vulkan maps these to specific stage/access flags internally; Metal and GLES
/// ignore dependencies (they fall out of tile-memory semantics or multi-pass
/// fallback respectively).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
#[cfg_attr(feature = "serde", serde(rename_all = "kebab-case"))]
#[non_exhaustive]
pub enum SubpassDependencyType {
    /// Color writes become input attachment reads.
    ColorToInput,
    /// Depth/stencil writes become input attachment reads.
    DepthToInput,
    /// Color and depth/stencil writes become input attachment reads.
    ColorDepthToInput,
}

/// Synchronization dependency between two subpasses.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
#[cfg_attr(feature = "serde", serde(rename_all = "camelCase"))]
#[non_exhaustive]
pub struct SubpassDependency {
    /// Source subpass.
    pub src_subpass: SubpassIndex,
    /// Destination subpass.
    pub dst_subpass: SubpassIndex,
    /// Dependency flavor.
    pub dependency_type: SubpassDependencyType,
    /// Whether this dependency is by-region (the critical TBDR optimization).
    pub by_region: bool,
}

/// Aspect kind of a subpass input binding.
///
/// This type is consumed by `wgpu-core`'s subpass-input validation in Phase 3.
/// It is exposed here so that the `BindingType` extension in Phase 1 can carry
/// a value of this type without a forward dependency.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
#[cfg_attr(feature = "serde", serde(rename_all = "camelCase"))]
#[non_exhaustive]
pub enum SubpassInputAspect {
    /// Read color aspect of a color attachment.
    Color,
    /// Read depth aspect of a depth/stencil attachment.
    Depth,
    /// Read stencil aspect of a depth/stencil attachment.
    Stencil,
}

/// Validation and compatibility metadata for one subpass.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
#[cfg_attr(feature = "serde", serde(rename_all = "camelCase"))]
#[non_exhaustive]
pub struct SubpassLayout {
    /// Color target formats.
    pub color_formats: Vec<Option<TextureFormat>>,
    /// Depth/stencil target format.
    pub depth_stencil_format: Option<TextureFormat>,
    /// Sample count.
    pub sample_count: u32,
    /// Input attachment formats.
    pub input_attachment_formats: Vec<TextureFormat>,
}

impl Default for SubpassLayout {
    fn default() -> Self {
        Self {
            color_formats: Vec::new(),
            depth_stencil_format: None,
            sample_count: 1,
            input_attachment_formats: Vec::new(),
        }
    }
}

/// Subpass attachment usage metadata for one subpass in a target.
#[derive(Clone, Debug, PartialEq, Eq, Hash, Default)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
#[cfg_attr(feature = "serde", serde(rename_all = "camelCase"))]
#[non_exhaustive]
pub struct SubpassTargetDesc {
    /// Color attachment indices written by this subpass.
    pub color_attachment_indices: Vec<u32>,
    /// Whether this subpass uses depth/stencil.
    pub uses_depth_stencil: bool,
    /// Input attachment slot indices, ordered by the consuming subpass's
    /// input-attachment binding order.
    ///
    /// Each entry is either:
    /// - a color attachment slot index into [`SubpassTarget::color_attachment_formats`], or
    /// - [`DEPTH_STENCIL_INPUT_ATTACHMENT_INDEX`], which denotes the render pass
    ///   depth/stencil attachment.
    pub input_attachment_indices: Vec<u32>,
}

/// Complete render pass compatibility target for subpass-aware pipelines.
///
/// Vulkan requires a compatible `VkRenderPass` at pipeline creation time;
/// `SubpassTarget` carries the full subpass structure so the Vulkan backend
/// can construct it. Metal uses it for input-attachment format derivation.
/// GLES ignores it.
#[derive(Clone, Debug, PartialEq, Eq, Hash, Default)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
#[cfg_attr(feature = "serde", serde(rename_all = "camelCase"))]
#[non_exhaustive]
pub struct SubpassTarget {
    /// Pipeline's active subpass index within `subpass_descs`.
    pub index: u32,
    /// Render pass color attachment formats.
    pub color_attachment_formats: Vec<Option<TextureFormat>>,
    /// Render pass depth/stencil format.
    pub depth_stencil_format: Option<TextureFormat>,
    /// Subpass usage descriptions.
    pub subpass_descs: Vec<SubpassTargetDesc>,
    /// Subpass dependencies.
    pub dependencies: Vec<SubpassDependency>,
}

/// Backend hint for transient-memory behavior.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Default)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
#[cfg_attr(feature = "serde", serde(rename_all = "kebab-case"))]
#[non_exhaustive]
pub enum TransientMemoryHint {
    /// Let the backend pick.
    #[default]
    Auto,
    /// Prefer tile-memory-backed transients.
    PreferTileMemory,
    /// Prefer larger tile sizing when available.
    PreferLargerTiles,
}

/// Descriptor for programmable tile dispatch.
///
/// Reserved for future backend implementations; all backends currently return
/// `Unexpected` when this is used.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Default)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
#[cfg_attr(feature = "serde", serde(rename_all = "camelCase"))]
#[non_exhaustive]
pub struct TransientDispatchDescriptor {
    /// Tile width in pixels.
    pub tile_width: u32,
    /// Tile height in pixels.
    pub tile_height: u32,
}

/// Bitmask selecting active subpasses.
///
/// This mask is currently limited to 32 subpasses (`u32::BITS`).
#[repr(transparent)]
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Default)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct ActiveSubpassMask(pub u32);

impl ActiveSubpassMask {
    /// Maximum number of subpasses addressable by this mask.
    pub const MAX_SUBPASSES: u32 = u32::BITS;
    /// All 32 possible subpasses active.
    pub const ALL: Self = Self(u32::MAX);
    /// No subpasses active.
    pub const NONE: Self = Self(0);

    /// Returns true when a given subpass index is active.
    #[must_use]
    pub fn is_active(self, index: SubpassIndex) -> bool {
        debug_assert!(index.0 < Self::MAX_SUBPASSES);
        let bit = 1u32.checked_shl(index.0).unwrap_or(0);
        (self.0 & bit) != 0
    }

    /// Returns a copy with the given subpass marked active.
    #[must_use]
    pub fn with(self, index: SubpassIndex) -> Self {
        debug_assert!(index.0 < Self::MAX_SUBPASSES);
        let bit = 1u32.checked_shl(index.0).unwrap_or(0);
        Self(self.0 | bit)
    }

    /// Returns a copy with the given subpass marked inactive.
    #[must_use]
    pub fn without(self, index: SubpassIndex) -> Self {
        debug_assert!(index.0 < Self::MAX_SUBPASSES);
        let bit = 1u32.checked_shl(index.0).unwrap_or(0);
        Self(self.0 & !bit)
    }

    /// Returns the number of active subpasses.
    #[must_use]
    pub fn count_active(self) -> u32 {
        self.0.count_ones()
    }

    /// Build a mask from an index slice.
    #[must_use]
    pub fn from_indices(indices: &[SubpassIndex]) -> Self {
        let mut mask = Self::NONE;
        for &index in indices {
            mask = mask.with(index);
        }
        mask
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use alloc::vec;
    use core::hash::{Hash, Hasher};

    fn hash_of<T: Hash>(value: T) -> u64 {
        let mut hasher = std::collections::hash_map::DefaultHasher::new();
        value.hash(&mut hasher);
        hasher.finish()
    }

    #[test]
    fn transient_size_default_is_match_target() {
        assert_eq!(TransientSize::default(), TransientSize::MatchTarget);
    }

    #[test]
    fn transient_size_explicit_roundtrip() {
        let size = TransientSize::Explicit {
            width: 64,
            height: 32,
        };
        assert_eq!(
            size,
            TransientSize::Explicit {
                width: 64,
                height: 32
            }
        );
    }

    #[test]
    fn transient_attachment_descriptor_equality() {
        let a = TransientAttachmentDescriptor {
            format: TextureFormat::Rgba8Unorm,
            size: TransientSize::MatchTarget,
            sample_count: 1,
        };
        let b = a;
        assert_eq!(a, b);
    }

    #[test]
    fn transient_load_op_default_is_clear() {
        assert_eq!(TransientLoadOp::<u32>::default(), TransientLoadOp::Clear(0));
    }

    #[test]
    fn transient_load_op_equality() {
        assert_eq!(TransientLoadOp::DontCare::<u32>, TransientLoadOp::DontCare);
    }

    #[test]
    fn transient_ops_default_uses_clear() {
        assert_eq!(
            TransientOps::<u32>::default(),
            TransientOps {
                load: TransientLoadOp::Clear(0),
            }
        );
    }

    #[test]
    fn transient_ops_hash_matches_equal_values() {
        let a = TransientOps {
            load: TransientLoadOp::Clear(7u32),
        };
        let b = TransientOps {
            load: TransientLoadOp::Clear(7u32),
        };
        assert_eq!(hash_of(a), hash_of(b));
    }

    #[test]
    fn subpass_index_ordering() {
        assert!(SubpassIndex(1) > SubpassIndex(0));
    }

    #[test]
    fn subpass_index_hash_matches_equal_values() {
        assert_eq!(hash_of(SubpassIndex(9)), hash_of(SubpassIndex(9)));
    }

    #[test]
    fn subpass_input_source_color_equality() {
        let source = SubpassInputSource::Color {
            subpass: SubpassIndex(2),
            attachment_index: 1,
        };
        assert_eq!(
            source,
            SubpassInputSource::Color {
                subpass: SubpassIndex(2),
                attachment_index: 1,
            }
        );
    }

    #[test]
    fn subpass_input_source_depth_equality() {
        assert_eq!(
            SubpassInputSource::Depth {
                subpass: SubpassIndex(1),
            },
            SubpassInputSource::Depth {
                subpass: SubpassIndex(1),
            }
        );
    }

    #[test]
    fn subpass_input_attachment_equality() {
        let input = SubpassInputAttachment {
            binding: 3,
            source: SubpassInputSource::Depth {
                subpass: SubpassIndex(0),
            },
        };
        assert_eq!(input, input);
    }

    #[test]
    fn subpass_dependency_type_equality() {
        assert_eq!(
            SubpassDependencyType::ColorDepthToInput,
            SubpassDependencyType::ColorDepthToInput
        );
    }

    #[test]
    fn subpass_dependency_equality() {
        let dep = SubpassDependency {
            src_subpass: SubpassIndex(0),
            dst_subpass: SubpassIndex(1),
            dependency_type: SubpassDependencyType::ColorToInput,
            by_region: true,
        };
        assert_eq!(dep, dep);
    }

    #[test]
    fn subpass_layout_default_is_empty_sample_count_one() {
        let layout = SubpassLayout::default();
        assert!(layout.color_formats.is_empty());
        assert!(layout.input_attachment_formats.is_empty());
        assert_eq!(layout.depth_stencil_format, None);
        assert_eq!(layout.sample_count, 1);
    }

    #[test]
    fn subpass_layout_hash_matches_equal_values() {
        let a = SubpassLayout {
            color_formats: vec![Some(TextureFormat::Rgba8Unorm)],
            depth_stencil_format: Some(TextureFormat::Depth32Float),
            sample_count: 4,
            input_attachment_formats: vec![TextureFormat::Rgba8Unorm],
        };
        let b = a.clone();
        assert_eq!(hash_of(a), hash_of(b));
    }

    #[test]
    fn subpass_target_desc_default() {
        let desc = SubpassTargetDesc::default();
        assert!(desc.color_attachment_indices.is_empty());
        assert!(!desc.uses_depth_stencil);
        assert!(desc.input_attachment_indices.is_empty());
    }

    #[test]
    fn subpass_target_default() {
        let target = SubpassTarget::default();
        assert_eq!(target.index, 0);
        assert!(target.color_attachment_formats.is_empty());
        assert_eq!(target.depth_stencil_format, None);
        assert!(target.subpass_descs.is_empty());
        assert!(target.dependencies.is_empty());
    }

    #[test]
    fn subpass_target_hash_matches_equal_values() {
        let target = SubpassTarget {
            index: 1,
            color_attachment_formats: vec![Some(TextureFormat::Rgba8Unorm)],
            depth_stencil_format: None,
            subpass_descs: vec![SubpassTargetDesc {
                color_attachment_indices: vec![0],
                uses_depth_stencil: false,
                input_attachment_indices: vec![],
            }],
            dependencies: vec![SubpassDependency {
                src_subpass: SubpassIndex(0),
                dst_subpass: SubpassIndex(1),
                dependency_type: SubpassDependencyType::ColorToInput,
                by_region: true,
            }],
        };
        assert_eq!(hash_of(target.clone()), hash_of(target));
    }

    #[test]
    fn transient_memory_hint_default_is_auto() {
        assert_eq!(TransientMemoryHint::default(), TransientMemoryHint::Auto);
    }

    #[test]
    fn transient_dispatch_descriptor_equality() {
        let dispatch = TransientDispatchDescriptor {
            tile_width: 8,
            tile_height: 16,
        };
        assert_eq!(dispatch, dispatch);
    }

    #[test]
    fn active_subpass_mask_constants() {
        assert_eq!(ActiveSubpassMask::NONE.0, 0);
        assert_eq!(ActiveSubpassMask::ALL.0, u32::MAX);
    }

    #[test]
    fn active_subpass_mask_with_sets_bit() {
        let mask = ActiveSubpassMask::NONE.with(SubpassIndex(3));
        assert!(mask.is_active(SubpassIndex(3)));
    }

    #[test]
    fn active_subpass_mask_without_clears_bit() {
        let mask = ActiveSubpassMask::ALL.without(SubpassIndex(5));
        assert!(!mask.is_active(SubpassIndex(5)));
    }

    #[test]
    fn active_subpass_mask_count_active() {
        let mask = ActiveSubpassMask::NONE
            .with(SubpassIndex(0))
            .with(SubpassIndex(7))
            .with(SubpassIndex(31));
        assert_eq!(mask.count_active(), 3);
    }

    #[test]
    fn active_subpass_mask_from_indices_empty() {
        assert_eq!(
            ActiveSubpassMask::from_indices(&[]),
            ActiveSubpassMask::NONE
        );
    }

    #[test]
    fn active_subpass_mask_from_indices_multiple() {
        let mask = ActiveSubpassMask::from_indices(&[
            SubpassIndex(0),
            SubpassIndex(2),
            SubpassIndex(2),
            SubpassIndex(30),
        ]);
        assert!(mask.is_active(SubpassIndex(0)));
        assert!(mask.is_active(SubpassIndex(2)));
        assert!(mask.is_active(SubpassIndex(30)));
        assert_eq!(mask.count_active(), 3);
    }

    #[test]
    fn active_subpass_mask_from_indices_max_index() {
        let mask = ActiveSubpassMask::from_indices(&[SubpassIndex(31)]);
        assert!(mask.is_active(SubpassIndex(31)));
        assert_eq!(mask.count_active(), 1);
    }

    #[test]
    fn active_subpass_mask_with_chain() {
        let mask = ActiveSubpassMask::NONE
            .with(SubpassIndex(1))
            .with(SubpassIndex(4))
            .without(SubpassIndex(1));
        assert!(!mask.is_active(SubpassIndex(1)));
        assert!(mask.is_active(SubpassIndex(4)));
    }

    #[test]
    fn active_subpass_mask_without_unset_is_noop() {
        let mask = ActiveSubpassMask::NONE.without(SubpassIndex(8));
        assert_eq!(mask, ActiveSubpassMask::NONE);
    }

    #[test]
    fn active_subpass_mask_hash_matches_equal_values() {
        assert_eq!(
            hash_of(ActiveSubpassMask::from_indices(&[
                SubpassIndex(0),
                SubpassIndex(4)
            ])),
            hash_of(ActiveSubpassMask::from_indices(&[
                SubpassIndex(4),
                SubpassIndex(0)
            ]))
        );
    }

    #[test]
    fn active_subpass_mask_is_active_false_for_unset_bits() {
        let mask = ActiveSubpassMask::from_indices(&[SubpassIndex(1)]);
        assert!(!mask.is_active(SubpassIndex(0)));
    }

    #[cfg(debug_assertions)]
    #[test]
    #[should_panic]
    fn active_subpass_mask_is_active_panics_on_out_of_range() {
        let _ = ActiveSubpassMask::NONE.is_active(SubpassIndex(32));
    }

    #[cfg(debug_assertions)]
    #[test]
    #[should_panic]
    fn active_subpass_mask_with_panics_on_out_of_range() {
        let _ = ActiveSubpassMask::NONE.with(SubpassIndex(32));
    }

    #[cfg(debug_assertions)]
    #[test]
    #[should_panic]
    fn active_subpass_mask_without_panics_on_out_of_range() {
        let _ = ActiveSubpassMask::NONE.without(SubpassIndex(32));
    }

    #[cfg(debug_assertions)]
    #[test]
    #[should_panic]
    fn active_subpass_mask_from_indices_panics_on_out_of_range() {
        let _ = ActiveSubpassMask::from_indices(&[SubpassIndex(32)]);
    }

    // Release-mode contract: out-of-range indices silently no-op (no panic).
    // The `should_panic` tests above only run under `debug_assertions`; these
    // lock in the release-mode behavior so it can't drift.
    #[cfg(not(debug_assertions))]
    #[test]
    fn active_subpass_mask_is_active_release_silent_noop() {
        assert!(!ActiveSubpassMask::ALL.is_active(SubpassIndex(32)));
    }

    #[cfg(not(debug_assertions))]
    #[test]
    fn active_subpass_mask_with_release_silent_noop() {
        assert_eq!(
            ActiveSubpassMask::NONE.with(SubpassIndex(32)),
            ActiveSubpassMask::NONE
        );
    }

    #[cfg(not(debug_assertions))]
    #[test]
    fn active_subpass_mask_without_release_silent_noop() {
        assert_eq!(
            ActiveSubpassMask::ALL.without(SubpassIndex(32)),
            ActiveSubpassMask::ALL
        );
    }

    #[cfg(not(debug_assertions))]
    #[test]
    fn active_subpass_mask_from_indices_release_silent_noop() {
        assert_eq!(
            ActiveSubpassMask::from_indices(&[SubpassIndex(32)]),
            ActiveSubpassMask::NONE
        );
    }
}
// tiled-fork: end types
