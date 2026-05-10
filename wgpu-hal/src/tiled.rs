// tiled-fork: begin types
//! Tile-based deferred rendering (TBDR) extension surface for `wgpu-hal`.
//!
//! Phase 1 of the upstream-friendly fork plan documented in `TILED.md` at the
//! repo root. This module is fork-only and adds **extension traits** rather
//! than methods on upstream traits, so upstream `Api`, `Device`, and
//! `CommandEncoder` are not touched.
//!
//! Conceptually, a backend opts into tiled support by implementing:
//!
//! - [`TiledApi`] for its `Api` type, declaring the two transient associated
//!   types (`TransientAttachment` and `TransientDispatch`).
//! - [`TiledDevice`] for its `Device` type, supplying create/destroy methods
//!   for those resources.
//! - [`TiledCommandEncoder`] for its `CommandEncoder` type, supplying
//!   `begin_subpass_render_pass`, `next_subpass`, and `dispatch_transient`.
//!
//! Backends that do not implement these traits are unaffected; the upstream
//! single-pass HAL surface remains the only entry point for them.

use core::num::NonZeroU32;

use crate::dynamic::{DynQuerySet, DynTextureView};
use crate::{
    Api, ColorAttachment, CommandEncoder, DepthStencilAttachment, Device, DeviceError,
    DynTransientAttachment, DynTransientDispatch, Label, PassTimestampWrites, PipelineError,
    RenderPipelineDescriptor,
};

// ----- Extension traits ---------------------------------------------------

/// Backend declaration of the tile-memory resources the API exposes.
///
/// Implemented for `Api` types that support tiled rendering. Backends that
/// don't support tiling (e.g., DX12 in this fork) simply do not implement
/// this trait.
pub trait TiledApi: Api {
    /// Backend-specific transient attachment type.
    ///
    /// On Vulkan: `VkImage` + `VkImageView` allocated with
    /// `VK_IMAGE_USAGE_TRANSIENT_ATTACHMENT_BIT`.
    /// On Metal: `MTLTexture` with `MTLStorageModeMemoryless`.
    /// On GLES: GL renderbuffer that gets `glInvalidateFramebuffer`-d.
    type TransientAttachment: DynTransientAttachment;

    /// Backend-specific programmable tile-dispatch resource.
    ///
    /// Reserved for future Apple-GPU tile-dispatch support; backends currently
    /// return `Err(DeviceError::Unexpected)` from
    /// [`TiledDevice::create_transient_dispatch`] when invoked.
    type TransientDispatch: DynTransientDispatch;
}

/// Device-side extension trait for tiled rendering.
///
/// This trait extends [`Device`] with create/destroy methods for tile-memory
/// resources. A backend implements this trait in addition to the upstream
/// `Device` trait.
pub trait TiledDevice: Device
where
    <Self as Device>::A: TiledApi,
{
    /// Create a transient attachment.
    ///
    /// # Safety
    /// See [`Device::create_buffer`] for general HAL safety rules.
    unsafe fn create_transient_attachment(
        &self,
        desc: &wgt::TransientAttachmentDescriptor,
    ) -> Result<<<Self as Device>::A as TiledApi>::TransientAttachment, DeviceError>;

    /// Destroy a transient attachment.
    ///
    /// # Safety
    /// `attachment` must have been created by this device, and must not be
    /// referenced by any pending command buffer.
    unsafe fn destroy_transient_attachment(
        &self,
        attachment: <<Self as Device>::A as TiledApi>::TransientAttachment,
    );

    /// Create a programmable tile-dispatch resource.
    ///
    /// # Safety
    /// See [`Device::create_buffer`] for general HAL safety rules. All current
    /// backends return `Err(DeviceError::Unexpected)`.
    unsafe fn create_transient_dispatch(
        &self,
        desc: &wgt::TransientDispatchDescriptor,
    ) -> Result<<<Self as Device>::A as TiledApi>::TransientDispatch, DeviceError>;

    /// Destroy a programmable tile-dispatch resource.
    ///
    /// # Safety
    /// `dispatch` must have been created by this device.
    unsafe fn destroy_transient_dispatch(
        &self,
        dispatch: <<Self as Device>::A as TiledApi>::TransientDispatch,
    );

    /// Create a render pipeline targeting a specific subpass within a
    /// multi-subpass render pass.
    ///
    /// The `subpass_target` argument carries the full subpass structure
    /// (color/depth formats, per-subpass attachment descriptors, dependencies)
    /// that backends like Vulkan need to construct a *compatible* render pass
    /// at pipeline-creation time. It is passed as a separate argument rather
    /// than added as a field on [`RenderPipelineDescriptor`] to keep the
    /// upstream-shared descriptor type unchanged for easier merging.
    ///
    /// Backends that do not need a compatible render pass at pipeline creation
    /// (Metal, GLES) may simply forward to `Device::create_render_pipeline`
    /// after consulting `subpass_target` for input-attachment format
    /// derivation as needed.
    ///
    /// # Safety
    /// See [`Device::create_render_pipeline`].
    #[allow(clippy::type_complexity)]
    unsafe fn create_subpass_render_pipeline(
        &self,
        desc: &RenderPipelineDescriptor<
            '_,
            <<Self as Device>::A as Api>::PipelineLayout,
            <<Self as Device>::A as Api>::ShaderModule,
            <<Self as Device>::A as Api>::PipelineCache,
        >,
        subpass_target: &wgt::SubpassTarget,
    ) -> Result<<<Self as Device>::A as Api>::RenderPipeline, PipelineError>;
}

/// Command-encoder-side extension trait for tiled rendering.
///
/// This trait adds the multi-subpass entry points to a backend's
/// `CommandEncoder`. Once a subpass-mode pass has been started via
/// [`begin_subpass_render_pass`], the upstream
/// [`CommandEncoder::end_render_pass`] terminates it.
///
/// [`begin_subpass_render_pass`]: TiledCommandEncoder::begin_subpass_render_pass
pub trait TiledCommandEncoder: CommandEncoder
where
    <Self as CommandEncoder>::A: TiledApi,
{
    /// Begin a multi-subpass render pass.
    ///
    /// # Safety
    /// See [`CommandEncoder::begin_render_pass`]. Additionally, every
    /// `transient_index` referenced inside `desc.subpasses` must be a valid
    /// index into a transient-attachment table maintained by the caller.
    unsafe fn begin_subpass_render_pass(
        &mut self,
        desc: &SubpassRenderPassDescriptor<
            '_,
            <<Self as CommandEncoder>::A as Api>::QuerySet,
            <<Self as CommandEncoder>::A as Api>::TextureView,
        >,
    );

    /// Advance to the next subpass within the active subpass-mode render pass.
    ///
    /// # Safety
    /// The encoder must be inside a subpass-mode render pass started via
    /// [`begin_subpass_render_pass`], and there must be at least one more
    /// subpass remaining.
    ///
    /// [`begin_subpass_render_pass`]: TiledCommandEncoder::begin_subpass_render_pass
    unsafe fn next_subpass(&mut self);

    /// Issue a programmable tile dispatch.
    ///
    /// # Safety
    /// The encoder must be inside a subpass-mode render pass and the
    /// `dispatch` resource must come from the same `TiledApi`.
    unsafe fn dispatch_transient(
        &mut self,
        dispatch: &<<Self as CommandEncoder>::A as TiledApi>::TransientDispatch,
    );
}

// ----- HAL descriptor types -----------------------------------------------

/// Per-subpass color attachment.
///
/// `Persistent` mirrors the upstream [`ColorAttachment`] for color targets
/// that are written through to DRAM. `Transient` references a slot in a
/// transient-attachment table whose contents live only in tile memory.
#[derive(Clone, Debug)]
#[non_exhaustive]
pub enum SubpassColorAttachment<'a, T: DynTextureView + ?Sized> {
    /// A regular DRAM-backed color attachment, slotted into the parent
    /// render pass's `color_attachments` array via
    /// [`Subpass::color_attachment_indices`].
    Persistent(ColorAttachment<'a, T>),
    /// A tile-memory-only color attachment.
    #[non_exhaustive]
    Transient {
        /// Index into the caller's transient-attachment table.
        transient_index: u32,
        /// Load operation; store is implicitly `Discard`.
        ops: wgt::TransientOps<wgt::Color>,
        /// Clear value used when `ops.load == TransientLoadOp::Clear`.
        clear_value: wgt::Color,
    },
}

/// Per-subpass depth/stencil attachment.
#[derive(Clone, Debug)]
#[non_exhaustive]
pub enum SubpassDepthStencilAttachment<'a, T: DynTextureView + ?Sized> {
    /// A regular DRAM-backed depth/stencil attachment, identifying the
    /// render pass's `depth_stencil_attachment`.
    Persistent(DepthStencilAttachment<'a, T>),
    /// A tile-memory-only depth/stencil attachment.
    #[non_exhaustive]
    Transient {
        /// Index into the caller's transient-attachment table.
        transient_index: u32,
        /// Depth load operation; store is implicitly `Discard`.
        depth_ops: wgt::TransientOps<f32>,
        /// Stencil load operation; store is implicitly `Discard`.
        stencil_ops: wgt::TransientOps<u32>,
        /// Clear values for `(depth, stencil)`.
        clear_value: (f32, u32),
    },
}

/// One subpass in a multi-subpass render pass.
#[derive(Clone, Debug)]
#[non_exhaustive]
pub struct Subpass<'a, T: DynTextureView + ?Sized> {
    /// Optional per-slot color attachment descriptions.
    ///
    /// When non-empty, every `Persistent(..)` entry must consume one slot
    /// from `color_attachment_indices` in order.
    pub color_attachments: &'a [Option<SubpassColorAttachment<'a, T>>],
    /// Indices into [`SubpassRenderPassDescriptor::color_attachments`].
    ///
    /// In the non-empty `color_attachments` mode above, this list must
    /// contain exactly one index for each `Persistent(..)` entry.
    pub color_attachment_indices: &'a [u32],
    /// Optional per-subpass depth/stencil attachment.
    pub depth_stencil_attachment: Option<SubpassDepthStencilAttachment<'a, T>>,
    /// Input attachments read by this subpass.
    pub input_attachments: &'a [wgt::SubpassInputAttachment],
}

impl<T: DynTextureView + ?Sized> Default for Subpass<'_, T> {
    fn default() -> Self {
        Self {
            color_attachments: &[],
            color_attachment_indices: &[],
            depth_stencil_attachment: None,
            input_attachments: &[],
        }
    }
}

/// HAL descriptor for a multi-subpass render pass.
///
/// This is the **separate** entry-point descriptor that
/// [`TiledCommandEncoder::begin_subpass_render_pass`] consumes, intentionally
/// distinct from the upstream `RenderPassDescriptor` so that no fields are
/// added to the upstream type.
#[derive(Clone, Debug)]
#[non_exhaustive]
pub struct SubpassRenderPassDescriptor<
    'a,
    Q: DynQuerySet + ?Sized,
    T: DynTextureView + ?Sized,
> {
    /// Optional human-readable label.
    pub label: Label<'a>,
    /// Render-target extent.
    pub extent: wgt::Extent3d,
    /// Sample count for all attachments.
    pub sample_count: u32,
    /// Persistent color attachments, indexed by
    /// [`Subpass::color_attachment_indices`].
    pub color_attachments: &'a [Option<ColorAttachment<'a, T>>],
    /// Persistent depth/stencil attachment for the render pass.
    pub depth_stencil_attachment: Option<DepthStencilAttachment<'a, T>>,
    /// Subpasses, executed in order.
    pub subpasses: &'a [Subpass<'a, T>],
    /// Subpass synchronization dependencies.
    pub subpass_dependencies: &'a [wgt::SubpassDependency],
    /// Backend hint for transient-memory behavior.
    pub transient_memory_hint: wgt::TransientMemoryHint,
    /// Optional bitmask selecting which subpasses are active. When `None`,
    /// every subpass is active.
    pub active_subpass_mask: Option<wgt::ActiveSubpassMask>,
    /// Multiview layer count, identical semantics to upstream.
    pub multiview_mask: Option<NonZeroU32>,
    /// Optional pass-level timestamp writes.
    pub timestamp_writes: Option<PassTimestampWrites<'a, Q>>,
    /// Optional occlusion query set.
    pub occlusion_query_set: Option<&'a Q>,
}

impl<Q: DynQuerySet + ?Sized, T: DynTextureView + ?Sized> Default
    for SubpassRenderPassDescriptor<'_, Q, T>
{
    fn default() -> Self {
        Self {
            label: None,
            extent: wgt::Extent3d::default(),
            sample_count: 1,
            color_attachments: &[],
            depth_stencil_attachment: None,
            subpasses: &[],
            subpass_dependencies: &[],
            transient_memory_hint: wgt::TransientMemoryHint::Auto,
            active_subpass_mask: None,
            multiview_mask: None,
            timestamp_writes: None,
            occlusion_query_set: None,
        }
    }
}

// ----- expect_downcast helpers --------------------------------------------

impl<'a> SubpassColorAttachment<'a, dyn DynTextureView> {
    /// Downcast a dyn-typed subpass color attachment to a concrete texture-view backend.
    ///
    /// Takes `&self` so callers can rebuild a concrete descriptor from the
    /// dyn-typed `Subpass.color_attachments: &'a [Option<...>]` slice without
    /// needing the elements themselves to be `Clone` (the `derive(Clone)`
    /// blanket on the enum requires `T: Clone`, which `dyn DynTextureView`
    /// does not satisfy).
    ///
    /// # Panics
    /// `Persistent` arms panic if the underlying view is not of type `B`.
    /// `Transient` arms are infallible; they carry no view and are simply
    /// re-wrapped with the new generic parameter.
    pub fn expect_downcast<B: DynTextureView>(&self) -> SubpassColorAttachment<'a, B> {
        match self {
            Self::Persistent(p) => SubpassColorAttachment::Persistent(p.expect_downcast()),
            Self::Transient {
                transient_index,
                ops,
                clear_value,
            } => SubpassColorAttachment::Transient {
                transient_index: *transient_index,
                ops: *ops,
                clear_value: *clear_value,
            },
        }
    }
}

impl<'a> SubpassDepthStencilAttachment<'a, dyn DynTextureView> {
    /// Downcast a dyn-typed subpass depth/stencil attachment to a concrete texture-view backend.
    ///
    /// Takes `&self`; see [`SubpassColorAttachment::expect_downcast`] for
    /// rationale.
    ///
    /// # Panics
    /// `Persistent` arms panic if the underlying view is not of type `B`.
    /// `Transient` arms are infallible; they carry no view and are simply
    /// re-wrapped with the new generic parameter.
    pub fn expect_downcast<B: DynTextureView>(&self) -> SubpassDepthStencilAttachment<'a, B> {
        match self {
            Self::Persistent(p) => SubpassDepthStencilAttachment::Persistent(p.expect_downcast()),
            Self::Transient {
                transient_index,
                depth_ops,
                stencil_ops,
                clear_value,
            } => SubpassDepthStencilAttachment::Transient {
                transient_index: *transient_index,
                depth_ops: *depth_ops,
                stencil_ops: *stencil_ops,
                clear_value: *clear_value,
            },
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::noop;
    use crate::{Attachment, AttachmentOps};

    #[test]
    fn subpass_default_is_empty() {
        let subpass = Subpass::<'_, dyn DynTextureView>::default();
        assert!(subpass.color_attachments.is_empty());
        assert!(subpass.color_attachment_indices.is_empty());
        assert!(subpass.depth_stencil_attachment.is_none());
        assert!(subpass.input_attachments.is_empty());
    }

    #[test]
    fn subpass_render_pass_descriptor_default() {
        let desc =
            SubpassRenderPassDescriptor::<'_, dyn DynQuerySet, dyn DynTextureView>::default();
        assert!(desc.subpasses.is_empty());
        assert!(desc.subpass_dependencies.is_empty());
        assert!(desc.active_subpass_mask.is_none());
        assert_eq!(desc.transient_memory_hint, wgt::TransientMemoryHint::Auto);
        assert_eq!(desc.sample_count, 1);
        assert!(desc.color_attachments.is_empty());
        assert!(desc.depth_stencil_attachment.is_none());
        assert!(desc.timestamp_writes.is_none());
        assert!(desc.occlusion_query_set.is_none());
    }

    #[test]
    fn subpass_color_attachment_transient_fields() {
        let attachment: SubpassColorAttachment<'_, dyn DynTextureView> =
            SubpassColorAttachment::Transient {
                transient_index: 3,
                ops: wgt::TransientOps::new(wgt::TransientLoadOp::DontCare),
                clear_value: wgt::Color::RED,
            };
        match attachment {
            SubpassColorAttachment::Persistent(_) => panic!("expected Transient"),
            SubpassColorAttachment::Transient {
                transient_index,
                ops,
                clear_value,
            } => {
                assert_eq!(transient_index, 3);
                assert_eq!(ops.load, wgt::TransientLoadOp::DontCare);
                assert_eq!(clear_value, wgt::Color::RED);
            }
        }
    }

    #[test]
    fn subpass_depth_stencil_transient_fields() {
        let attachment: SubpassDepthStencilAttachment<'_, dyn DynTextureView> =
            SubpassDepthStencilAttachment::Transient {
                transient_index: 7,
                depth_ops: wgt::TransientOps::new(wgt::TransientLoadOp::DontCare),
                stencil_ops: wgt::TransientOps::new(wgt::TransientLoadOp::Clear(42)),
                clear_value: (0.5, 9),
            };
        match attachment {
            SubpassDepthStencilAttachment::Persistent(_) => panic!("expected Transient"),
            SubpassDepthStencilAttachment::Transient {
                transient_index,
                depth_ops,
                stencil_ops,
                clear_value,
            } => {
                assert_eq!(transient_index, 7);
                assert_eq!(depth_ops.load, wgt::TransientLoadOp::DontCare);
                assert_eq!(stencil_ops.load, wgt::TransientLoadOp::Clear(42));
                assert_eq!(clear_value, (0.5, 9));
            }
        }
    }

    #[test]
    fn subpass_color_attachment_persistent_round_trip() {
        let view = noop::Resource;
        let view_dyn: &dyn DynTextureView = &view;
        let attachment = SubpassColorAttachment::Persistent(ColorAttachment {
            target: Attachment {
                view: view_dyn,
                usage: wgt::TextureUses::COLOR_TARGET,
            },
            depth_slice: Some(2),
            resolve_target: None,
            ops: AttachmentOps::LOAD | AttachmentOps::STORE,
            clear_value: wgt::Color::GREEN,
        });
        let downcast = attachment.expect_downcast::<noop::Resource>();
        match downcast {
            SubpassColorAttachment::Persistent(color) => {
                assert!(core::ptr::eq(color.target.view, &view));
                assert_eq!(color.target.usage, wgt::TextureUses::COLOR_TARGET);
                assert_eq!(color.depth_slice, Some(2));
                assert_eq!(color.clear_value, wgt::Color::GREEN);
            }
            SubpassColorAttachment::Transient { .. } => panic!("expected Persistent"),
        }
    }

    #[test]
    fn subpass_depth_stencil_persistent_round_trip() {
        let view = noop::Resource;
        let view_dyn: &dyn DynTextureView = &view;
        let attachment = SubpassDepthStencilAttachment::Persistent(DepthStencilAttachment {
            target: Attachment {
                view: view_dyn,
                usage: wgt::TextureUses::DEPTH_STENCIL_WRITE,
            },
            depth_ops: AttachmentOps::LOAD | AttachmentOps::STORE,
            stencil_ops: AttachmentOps::LOAD | AttachmentOps::STORE,
            clear_value: (0.25, 5),
        });
        let downcast = attachment.expect_downcast::<noop::Resource>();
        match downcast {
            SubpassDepthStencilAttachment::Persistent(depth_stencil) => {
                assert!(core::ptr::eq(depth_stencil.target.view, &view));
                assert_eq!(
                    depth_stencil.target.usage,
                    wgt::TextureUses::DEPTH_STENCIL_WRITE
                );
                assert_eq!(depth_stencil.clear_value, (0.25, 5));
            }
            SubpassDepthStencilAttachment::Transient { .. } => panic!("expected Persistent"),
        }
    }

    #[test]
    fn subpass_render_pass_descriptor_carries_active_mask() {
        let mask = wgt::ActiveSubpassMask::from_indices(&[
            wgt::SubpassIndex(0),
            wgt::SubpassIndex(2),
        ]);
        let desc = SubpassRenderPassDescriptor::<'_, dyn DynQuerySet, dyn DynTextureView> {
            active_subpass_mask: Some(mask),
            ..Default::default()
        };
        // The default would be `None`; verify the explicit override sticks
        // and that mask round-trips via `is_active`.
        let stored = desc.active_subpass_mask.expect("set above");
        assert!(stored.is_active(wgt::SubpassIndex(0)));
        assert!(!stored.is_active(wgt::SubpassIndex(1)));
        assert!(stored.is_active(wgt::SubpassIndex(2)));
    }

    #[test]
    fn subpass_render_pass_descriptor_carries_subpasses() {
        let s0 = Subpass::<'_, dyn DynTextureView>::default();
        let s1 = Subpass::<'_, dyn DynTextureView>::default();
        let subpasses = [s0, s1];
        let desc = SubpassRenderPassDescriptor::<'_, dyn DynQuerySet, dyn DynTextureView> {
            subpasses: &subpasses,
            ..Default::default()
        };
        assert_eq!(desc.subpasses.len(), 2);
    }
}
// tiled-fork: end types
