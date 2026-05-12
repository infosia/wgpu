// tiled-fork: begin metal-backend
//! Metal backend implementation of the tiled-rendering trait surface.
//!
//! Phase 9b lights up the path through wgpu-core for the Metal backend:
//!
//! - **TransientAttachment**: a real `MTLTexture` allocated with
//!   `MTLStorageMode::Memoryless` when the device supports it, so its
//!   contents live in tile memory and never spill to system RAM. Falls back
//!   to `MTLStorageMode::Private` (with a warning) on hardware that exposes
//!   tile shading without memoryless storage.
//! - **Subpass state machine**: `begin_subpass_render_pass` translates the
//!   `SubpassRenderPassDescriptor` into the upstream single-pass entry
//!   point and records traversal state on the encoder; `next_subpass`
//!   advances the active subpass while honoring `ActiveSubpassMask` culls.
//!   Subpasses that reference `Transient` attachments are short-circuited
//!   with a `log::error!` until wgpu-core's transient-table bridge lands
//!   (mirrors Vulkan 9a3 and GLES 9c).
//!
//! The pipeline-side wiring (`create_subpass_render_pipeline` with
//! input-attachment formats and color-target remap) lives in `device.rs` to
//! reuse the existing pipeline-creation helpers.

use objc2::rc::{autoreleasepool, Retained};
use objc2::runtime::ProtocolObject;
use objc2_metal::{
    MTLDevice, MTLStorageMode, MTLTexture, MTLTextureDescriptor, MTLTextureType, MTLTextureUsage,
};

use crate::{DeviceError, PipelineError};

/// A transient color or depth/stencil attachment backed by an `MTLTexture`
/// allocated with `MTLStorageMode::Memoryless` whenever the device exposes
/// memoryless storage. The texture's contents only exist in tile memory
/// during a render pass; reading from them outside the pass is undefined.
#[derive(Debug)]
pub struct TransientAttachment {
    // The raw texture is consumed by the wgpu-core transient-table bridge
    // when it wires `Subpass::*::Transient` into the underlying
    // `MTLRenderPassDescriptor` color/depth attachments. That bridge is the
    // companion follow-up to the carve-out in `begin_subpass_render_pass`
    // below; until it lands the field is held only so the `MTLTexture`'s
    // refcount keeps the memoryless backing alive for the lifetime of the
    // attachment.
    #[allow(dead_code)]
    pub(super) raw: Retained<ProtocolObject<dyn MTLTexture>>,
    // Format / size metadata is read by the subpass attachment binding path
    // in later sub-phases; held here so the resource is self-describing.
    #[allow(dead_code)]
    pub(super) format: wgt::TextureFormat,
    #[allow(dead_code)]
    pub(super) width: u32,
    #[allow(dead_code)]
    pub(super) height: u32,
    #[allow(dead_code)]
    pub(super) sample_count: u32,
}

unsafe impl Send for TransientAttachment {}
unsafe impl Sync for TransientAttachment {}

#[derive(Debug)]
pub struct TransientDispatch;

crate::impl_dyn_resource!(TransientAttachment, TransientDispatch);

impl crate::DynTransientAttachment for TransientAttachment {}
impl crate::DynTransientDispatch for TransientDispatch {}

impl crate::TiledApi for super::Api {
    type TransientAttachment = TransientAttachment;
    type TransientDispatch = TransientDispatch;
}

/// Rebuild an `Attachment` field-by-field. The macro-derived `Clone` for
/// `Attachment<'_, T>` requires `T: Clone`, but Metal's `TextureView`
/// deliberately does not derive `Clone` (the underlying `Retained<MTLTexture>`
/// would let callers extend the texture's lifetime invisibly). Every field
/// in `Attachment` is either a `&T` reference (always `Copy`) or a `Copy`
/// scalar, so the rebuild is a pure structural copy.
fn rebuild_attachment<'a, T: crate::DynTextureView + ?Sized>(
    a: &crate::Attachment<'a, T>,
) -> crate::Attachment<'a, T> {
    crate::Attachment {
        view: a.view,
        usage: a.usage,
    }
}

fn rebuild_color<'a, T: crate::DynTextureView + ?Sized>(
    c: &crate::ColorAttachment<'a, T>,
) -> crate::ColorAttachment<'a, T> {
    crate::ColorAttachment {
        target: rebuild_attachment(&c.target),
        depth_slice: c.depth_slice,
        resolve_target: c.resolve_target.as_ref().map(rebuild_attachment),
        ops: c.ops,
        clear_value: c.clear_value,
    }
}

fn rebuild_ds<'a, T: crate::DynTextureView + ?Sized>(
    ds: &crate::DepthStencilAttachment<'a, T>,
) -> crate::DepthStencilAttachment<'a, T> {
    crate::DepthStencilAttachment {
        target: rebuild_attachment(&ds.target),
        depth_ops: ds.depth_ops,
        stencil_ops: ds.stencil_ops,
        clear_value: ds.clear_value,
    }
}

/// Find the first active subpass index according to `mask`. Returns `None`
/// when every subpass is culled or `subpass_count == 0`. Bits set beyond
/// `subpass_count` (or beyond `ActiveSubpassMask::MAX_SUBPASSES`) are
/// ignored.
pub(super) fn first_active_subpass_index(
    subpass_count: u32,
    mask: Option<wgt::ActiveSubpassMask>,
) -> Option<u32> {
    if subpass_count == 0 {
        return None;
    }
    let mask_bits = mask.unwrap_or(wgt::ActiveSubpassMask::ALL).0;
    let search_end = subpass_count.min(wgt::ActiveSubpassMask::MAX_SUBPASSES);
    (0..search_end).find(|&index| (mask_bits & (1u32 << index)) != 0)
}

/// Find the next active subpass index strictly after `current`. Returns
/// `subpass_count` (i.e. one past the last subpass) when no more active
/// subpasses remain — callers use this as a sentinel meaning
/// "drain to end".
pub(super) fn next_active_subpass_index(
    current: u32,
    subpass_count: u32,
    mask: Option<wgt::ActiveSubpassMask>,
) -> u32 {
    let mask_bits = mask.unwrap_or(wgt::ActiveSubpassMask::ALL).0;
    let search_end = subpass_count.min(wgt::ActiveSubpassMask::MAX_SUBPASSES);
    ((current + 1)..search_end)
        .find(|&index| (mask_bits & (1u32 << index)) != 0)
        .unwrap_or(subpass_count)
}

impl crate::TiledDevice for super::Device {
    unsafe fn create_transient_attachment(
        &self,
        desc: &wgt::TransientAttachmentDescriptor,
    ) -> Result<TransientAttachment, DeviceError> {
        if !self.features.contains(wgt::Features::TRANSIENT_ATTACHMENTS) {
            log::error!(
                "metal: create_transient_attachment called without TRANSIENT_ATTACHMENTS support"
            );
            return Err(DeviceError::Unexpected);
        }

        let (width, height) = match desc.size {
            wgt::TransientSize::Explicit { width, height } => (width, height),
            wgt::TransientSize::MatchTarget => {
                // `MatchTarget` resolution happens in wgpu-core in a later
                // phase; the HAL only accepts explicit sizes.
                log::error!(
                    "metal: create_transient_attachment received unresolved TransientSize::MatchTarget"
                );
                return Err(DeviceError::Unexpected);
            }
            // `TransientSize` is `#[non_exhaustive]`. Any future variant must
            // be resolved by wgpu-core before reaching the HAL.
            other => {
                log::error!(
                    "metal: create_transient_attachment received unsupported TransientSize variant: {other:?}"
                );
                return Err(DeviceError::Unexpected);
            }
        };
        if width == 0 || height == 0 || desc.sample_count == 0 {
            log::error!(
                "metal: create_transient_attachment received invalid descriptor: width={}, height={}, sample_count={}",
                width,
                height,
                desc.sample_count
            );
            return Err(DeviceError::Unexpected);
        }

        let mtl_format = self
            .shared
            .private_texture_format_caps
            .map_format(desc.format);

        autoreleasepool(|_| {
            let descriptor = MTLTextureDescriptor::new();
            let mtl_type = if desc.sample_count > 1 {
                unsafe { descriptor.setSampleCount(desc.sample_count as usize) };
                MTLTextureType::Type2DMultisample
            } else {
                MTLTextureType::Type2D
            };
            let storage_mode = if self.shared.private_caps.supports_memoryless_storage {
                MTLStorageMode::Memoryless
            } else {
                // The adapter gate (`supports_tile_shading &&
                // supports_memoryless_storage`) normally blocks this path;
                // keep a runtime fallback in case some future device exposes
                // tile shading without memoryless storage.
                log::warn!(
                    "metal: memoryless transient attachments are unsupported on this device, falling back to private storage"
                );
                MTLStorageMode::Private
            };

            descriptor.setTextureType(mtl_type);
            unsafe { descriptor.setWidth(width as usize) };
            unsafe { descriptor.setHeight(height as usize) };
            unsafe { descriptor.setMipmapLevelCount(1) };
            descriptor.setPixelFormat(mtl_format);
            descriptor.setUsage(MTLTextureUsage::RenderTarget);
            descriptor.setStorageMode(storage_mode);

            let raw = self
                .shared
                .device
                .newTextureWithDescriptor(&descriptor)
                .ok_or(DeviceError::OutOfMemory)?;

            Ok(TransientAttachment {
                raw,
                format: desc.format,
                width,
                height,
                sample_count: desc.sample_count,
            })
        })
    }

    unsafe fn destroy_transient_attachment(&self, _attachment: TransientAttachment) {
        // The `Retained<MTLTexture>` releases the underlying texture when
        // dropped; nothing else to do.
    }

    unsafe fn create_transient_dispatch(
        &self,
        _desc: &wgt::TransientDispatchDescriptor,
    ) -> Result<TransientDispatch, DeviceError> {
        // Programmable tile dispatch on Apple GPUs is the natural Metal
        // analogue but is not yet wired through the HAL trait surface.
        // Surface a recoverable error so callers can fall back to a
        // single-pass path.
        Err(DeviceError::Unexpected)
    }

    unsafe fn destroy_transient_dispatch(&self, _dispatch: TransientDispatch) {}

    unsafe fn create_subpass_render_pipeline(
        &self,
        desc: &crate::RenderPipelineDescriptor<
            '_,
            <super::Api as crate::Api>::PipelineLayout,
            <super::Api as crate::Api>::ShaderModule,
            <super::Api as crate::Api>::PipelineCache,
        >,
        subpass_target: &wgt::SubpassTarget,
    ) -> Result<<super::Api as crate::Api>::RenderPipeline, PipelineError> {
        // Gate on tile-shading hardware support up front so the caller gets a
        // crisp linkage error rather than a Metal validation failure later.
        if !self.supports_tile_shading() {
            return Err(PipelineError::Linkage(
                wgt::ShaderStages::FRAGMENT,
                "metal: create_subpass_render_pipeline requires tile-shading hardware support; \
                 advertise the MULTI_SUBPASS feature only on Apple4+/Mac2 GPUs"
                    .into(),
            ));
        }

        // Validate the subpass target shape early. Even though we currently
        // forward to the single-subpass pipeline path, the helpers ensure
        // wgpu-core's smoke tests get a deterministic error when a
        // descriptor is malformed (out-of-bounds index, conflicting fragment
        // output remap, etc.) and the same error surface as backends that
        // do consume the remap (Vulkan).
        let (_input_attachment_indices, color_attachment_indices) =
            super::device::get_subpass_attachment_remaps(subpass_target).map_err(|msg| {
                PipelineError::Linkage(wgt::ShaderStages::FRAGMENT, msg)
            })?;
        super::device::validate_subpass_output_remap(
            &color_attachment_indices,
            desc.color_targets.len(),
            subpass_target.color_attachment_formats.len(),
        )
        .map_err(|msg| PipelineError::Linkage(wgt::ShaderStages::FRAGMENT, msg))?;

        // Build the (group, binding) -> color attachment slot map for any
        // `subpass_input` globals in the fragment shader. The MSL writer
        // surfaces these as `[[color(N)]]` fragment-entry arguments through
        // `naga::back::msl::Options::subpass_color_slots`.
        let fragment_subpass_color_slots = if let Some(stage) = desc.fragment_stage.as_ref() {
            super::device::build_subpass_color_slot_map(stage, Some(subpass_target))?
        } else {
            naga::FastHashMap::default()
        };

        // Forward to the shared pipeline-creation helper, threading the
        // slot map and the parent `SubpassTarget` so the helper can set
        // the color-attachment pixel formats for any framebuffer-fetch
        // slot the fragment shader reads back via `[[color(N)]]`.
        unsafe {
            self.create_render_pipeline_inner(
                desc,
                &fragment_subpass_color_slots,
                Some(subpass_target),
            )
        }
    }
}

impl crate::TiledCommandEncoder for super::CommandEncoder {
    unsafe fn begin_subpass_render_pass(
        &mut self,
        desc: &crate::SubpassRenderPassDescriptor<
            '_,
            <super::Api as crate::Api>::QuerySet,
            <super::Api as crate::Api>::TextureView,
        >,
    ) {
        // Reject any subpass that references a `Transient` attachment until
        // wgpu-core's transient-table bridge lands. Mirrors Vulkan Phase 9a3
        // and GLES Phase 9c so smoke tests of the persistent path stay
        // reachable while transient wiring is being plumbed through core.
        for (subpass_index, subpass) in desc.subpasses.iter().enumerate() {
            for color in subpass.color_attachments.iter().flatten() {
                if matches!(color, crate::SubpassColorAttachment::Transient { .. }) {
                    log::error!(
                        "metal: begin_subpass_render_pass: subpass {subpass_index} references a \
                         transient color attachment; wgpu-core transient-table bridge not yet wired \
                         (returning early without recording any commands)"
                    );
                    return;
                }
            }
            if let Some(ds) = &subpass.depth_stencil_attachment {
                if matches!(ds, crate::SubpassDepthStencilAttachment::Transient { .. }) {
                    log::error!(
                        "metal: begin_subpass_render_pass: subpass {subpass_index} references a \
                         transient depth/stencil attachment; wgpu-core transient-table bridge not \
                         yet wired (returning early without recording any commands)"
                    );
                    return;
                }
            }
        }

        let subpass_count = desc.subpasses.len() as u32;
        let active_subpass_mask = if subpass_count > 0 {
            desc.active_subpass_mask
        } else {
            None
        };
        let current_index = first_active_subpass_index(subpass_count, active_subpass_mask);

        // Translate the subpass descriptor into an upstream single-pass
        // descriptor and reuse `begin_render_pass`.
        //
        // tiled-fork: Metal has *one* `MTLRenderCommandEncoder` per render
        // pass with a single fixed attachment table — every slot the
        // pipeline's `[[color(N)]]` arguments touch (across *all* subpasses
        // in the pass) needs a corresponding entry here. So we forward the
        // pass-level `desc.color_attachments` directly rather than picking
        // out the active subpass's local list. The implicit serialization
        // between subpasses comes from the hardware tile pipeline; advance
        // is a no-op on Metal.
        let _ = current_index; // active-subpass selection is encoded in pipelines, not here
        let mut color_attachments_storage: arrayvec::ArrayVec<
            Option<crate::ColorAttachment<'_, <super::Api as crate::Api>::TextureView>>,
            { crate::MAX_COLOR_ATTACHMENTS },
        > = arrayvec::ArrayVec::new();
        let depth_stencil_attachment = desc.depth_stencil_attachment.as_ref().map(rebuild_ds);

        for slot in desc.color_attachments.iter() {
            let attachment = slot.as_ref().map(rebuild_color);
            if color_attachments_storage.try_push(attachment).is_err() {
                log::error!(
                    "metal: begin_subpass_render_pass: descriptor declared more color \
                     attachments than MAX_COLOR_ATTACHMENTS; truncating"
                );
                break;
            }
        }

        // `PassTimestampWrites` borrows `&Q`, so it can't be cloned through
        // a `?Sized` parameter; rebuild it field-by-field instead.
        let timestamp_writes =
            desc.timestamp_writes
                .as_ref()
                .map(|tw| crate::PassTimestampWrites {
                    query_set: tw.query_set,
                    beginning_of_pass_write_index: tw.beginning_of_pass_write_index,
                    end_of_pass_write_index: tw.end_of_pass_write_index,
                });
        let single_pass_desc = crate::RenderPassDescriptor::<
            '_,
            <super::Api as crate::Api>::QuerySet,
            <super::Api as crate::Api>::TextureView,
        > {
            label: desc.label,
            extent: desc.extent,
            sample_count: desc.sample_count,
            color_attachments: &color_attachments_storage,
            depth_stencil_attachment,
            multiview_mask: desc.multiview_mask,
            timestamp_writes,
            occlusion_query_set: desc.occlusion_query_set,
        };

        // Record the pass via the upstream entry point. The error path is
        // swallowed because the `TiledCommandEncoder::begin_subpass_render_pass`
        // trait signature is `unsafe fn ... -> ()` (matches upstream
        // `CommandEncoder::begin_render_pass` and the wgpu-tiled reference).
        // A subsequent `set_render_pipeline`/draw/`end_render_pass` on a
        // failed begin will see no `subpass_state` (we return early) and hit
        // upstream's own validation. Refactoring to `Result` would change
        // the trait shape across all backends; defer that to a follow-up
        // trait-surface change.
        if let Err(err) =
            unsafe { <Self as crate::CommandEncoder>::begin_render_pass(self, &single_pass_desc) }
        {
            log::error!("metal: begin_subpass_render_pass: begin_render_pass failed: {err:?}");
            return;
        }

        self.subpass_state = Some(super::SubpassState {
            subpass_count,
            current_index,
            active_subpass_mask,
        });
    }

    unsafe fn next_subpass(&mut self) {
        let Some(state) = self.subpass_state.as_mut() else {
            log::error!("metal: next_subpass called outside a subpass render pass");
            return;
        };
        if state.subpass_count == 0 {
            return;
        }
        let Some(current) = state.current_index else {
            return;
        };
        let next =
            next_active_subpass_index(current, state.subpass_count, state.active_subpass_mask);
        debug_assert!(next <= state.subpass_count);

        // Metal serializes fragment work for sequential draws inside the
        // same `MTLRenderCommandEncoder` and exposes `subpassLoad(...)` via
        // `[[color(N)]]` tile memory reads, so subpass advance does not
        // re-encode anything — we just bookkeep the current index and let
        // the next pipeline binding observe the new subpass slot.
        state.current_index = (next < state.subpass_count).then_some(next);
    }

    unsafe fn dispatch_transient(&mut self, _dispatch: &TransientDispatch) {
        // Programmable tile dispatch is not currently routed through the
        // HAL; surface an error so wgpu-core can flag the unsupported call.
        log::error!("metal: dispatch_transient is not supported in this build");
    }
}

#[cfg(test)]
mod tests {
    use super::{first_active_subpass_index, next_active_subpass_index};

    #[test]
    fn subpass_traversal_sequential() {
        let mask = None;
        assert_eq!(first_active_subpass_index(3, mask), Some(0));
        assert_eq!(next_active_subpass_index(0, 3, mask), 1);
        assert_eq!(next_active_subpass_index(1, 3, mask), 2);
        assert_eq!(next_active_subpass_index(2, 3, mask), 3);
    }

    #[test]
    fn subpass_traversal_skips_culled_middle() {
        let mask = Some(
            wgt::ActiveSubpassMask::NONE
                .with(wgt::SubpassIndex(0))
                .with(wgt::SubpassIndex(2)),
        );
        assert_eq!(first_active_subpass_index(3, mask), Some(0));
        assert_eq!(next_active_subpass_index(0, 3, mask), 2);
        assert_eq!(next_active_subpass_index(2, 3, mask), 3);
    }

    #[test]
    fn subpass_traversal_skips_initial_culled() {
        let mask = Some(
            wgt::ActiveSubpassMask::NONE
                .with(wgt::SubpassIndex(2))
                .with(wgt::SubpassIndex(3)),
        );
        assert_eq!(first_active_subpass_index(4, mask), Some(2));
    }

    #[test]
    fn subpass_traversal_all_culled() {
        let mask = Some(wgt::ActiveSubpassMask::NONE);
        assert_eq!(first_active_subpass_index(4, mask), None);
    }

    #[test]
    fn subpass_traversal_zero_subpasses() {
        let mask = Some(wgt::ActiveSubpassMask::NONE);
        assert_eq!(first_active_subpass_index(0, mask), None);
    }
}
// tiled-fork: end metal-backend
