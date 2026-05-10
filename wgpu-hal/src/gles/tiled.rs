// tiled-fork: begin gles-backend
//! GLES backend implementation of the tiled-rendering trait surface.
//!
//! Two-tier strategy:
//!
//! - **Tier A** (`EXT_shader_framebuffer_fetch`, coherent variant): keep a
//!   single FBO across subpasses; tile-memory reads happen via `inout` color
//!   variables in the fragment shader. `next_subpass` only advances internal
//!   state.
//! - **Tier B** (always available): each subpass becomes a separate render
//!   pass / FBO. Subpass advance issues `glInvalidateFramebuffer` for the
//!   transient color/depth/stencil attachments of the outgoing subpass and
//!   binds a fresh FBO for the incoming one.
//!
//! Phase 9c implements the state machine, transient renderbuffer allocation,
//! and the feature/capability surface. Wiring `Subpass::*Transient` arms
//! through to actual FBO attachments is gated on the wgpu-core bridge that
//! populates the transient table from `TextureUsages::TRANSIENT`; until that
//! lands the encoder logs and returns early on `Transient` arms (matches the
//! Vulkan backend's Phase 9a3 behavior).

use glow::HasContext as _;

use crate::{DeviceError, PipelineError};

#[derive(Debug)]
pub struct TransientAttachment {
    pub(super) raw: glow::Renderbuffer,
    // The fields below are read by the FBO-rewire path that consumes
    // transient attachments. They are kept around for diagnostic logging
    // and to support the `MatchTarget` resolution path that wgpu-core will
    // perform before reaching the HAL.
    #[allow(dead_code)]
    pub(super) format: wgt::TextureFormat,
    #[allow(dead_code)]
    pub(super) width: u32,
    #[allow(dead_code)]
    pub(super) height: u32,
    #[allow(dead_code)]
    pub(super) sample_count: u32,
}

#[cfg(send_sync)]
unsafe impl Sync for TransientAttachment {}
#[cfg(send_sync)]
unsafe impl Send for TransientAttachment {}

#[derive(Debug)]
pub struct TransientDispatch;

crate::impl_dyn_resource!(TransientAttachment, TransientDispatch);

impl crate::DynTransientAttachment for TransientAttachment {}
impl crate::DynTransientDispatch for TransientDispatch {}

impl crate::TiledApi for super::Api {
    type TransientAttachment = TransientAttachment;
    type TransientDispatch = TransientDispatch;
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
        if !self
            .shared
            .features
            .contains(wgt::Features::TRANSIENT_ATTACHMENTS)
        {
            log::error!(
                "gles: create_transient_attachment called without TRANSIENT_ATTACHMENTS support"
            );
            return Err(DeviceError::Unexpected);
        }

        let (width, height) = match desc.size {
            wgt::TransientSize::Explicit { width, height } => (width, height),
            wgt::TransientSize::MatchTarget => {
                log::error!(
                    "gles: create_transient_attachment received unresolved TransientSize::MatchTarget; \
                     wgpu-core must resolve sizes before invoking the HAL"
                );
                return Err(DeviceError::Unexpected);
            }
            // `TransientSize` is `#[non_exhaustive]`. New variants land in
            // wgpu-types; until they're wired here treat them as unsupported.
            other => {
                log::error!(
                    "gles: create_transient_attachment received unsupported TransientSize variant: {other:?}"
                );
                return Err(DeviceError::Unexpected);
            }
        };
        if width == 0 || height == 0 || desc.sample_count == 0 {
            log::error!(
                "gles: create_transient_attachment received invalid descriptor: width={}, height={}, sample_count={}",
                width,
                height,
                desc.sample_count
            );
            return Err(DeviceError::Unexpected);
        }

        let gl = &self.shared.context.lock();
        let format_desc = self.shared.describe_texture_format(desc.format);
        let raw = unsafe { gl.create_renderbuffer() }.map_err(|_| DeviceError::OutOfMemory)?;
        unsafe { gl.bind_renderbuffer(glow::RENDERBUFFER, Some(raw)) };
        if desc.sample_count > 1 {
            unsafe {
                gl.renderbuffer_storage_multisample(
                    glow::RENDERBUFFER,
                    desc.sample_count as i32,
                    format_desc.internal,
                    width as i32,
                    height as i32,
                )
            };
        } else {
            unsafe {
                gl.renderbuffer_storage(
                    glow::RENDERBUFFER,
                    format_desc.internal,
                    width as i32,
                    height as i32,
                )
            };
        }
        let storage_error = unsafe { gl.get_error() };
        unsafe { gl.bind_renderbuffer(glow::RENDERBUFFER, None) };
        if storage_error != glow::NO_ERROR {
            unsafe { gl.delete_renderbuffer(raw) };
            if storage_error == glow::OUT_OF_MEMORY {
                return Err(DeviceError::OutOfMemory);
            }
            log::error!(
                "gles: create_transient_attachment failed during renderbuffer storage (gl error {storage_error:#x})"
            );
            return Err(DeviceError::Unexpected);
        }

        Ok(TransientAttachment {
            raw,
            format: desc.format,
            width,
            height,
            sample_count: desc.sample_count,
        })
    }

    unsafe fn destroy_transient_attachment(&self, attachment: TransientAttachment) {
        let gl = &self.shared.context.lock();
        unsafe { gl.delete_renderbuffer(attachment.raw) };
    }

    unsafe fn create_transient_dispatch(
        &self,
        _desc: &wgt::TransientDispatchDescriptor,
    ) -> Result<TransientDispatch, DeviceError> {
        // Programmable tile dispatch is Apple-GPU-only; GLES has no analogue.
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
        _subpass_target: &wgt::SubpassTarget,
    ) -> Result<<super::Api as crate::Api>::RenderPipeline, PipelineError> {
        // GLES does not require a "compatible render pass" at pipeline-creation
        // time. Subpass-input formats are derived at draw time from the bound
        // FBO (Tier B) or from `inout` declarations in GLSL (Tier A).
        // Phase 6e wired the GLSL writer's `use_framebuffer_fetch` option;
        // the pipeline-layout consumer that flips that option per pipeline is
        // a Phase 10 concern. For now forward to the upstream single-subpass
        // pipeline path.
        unsafe { <Self as crate::Device>::create_render_pipeline(self, desc) }
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
        // wgpu-core's transient-table bridge lands. This mirrors the Vulkan
        // Phase 9a3 carve-out so smoke tests of the persistent path stay
        // reachable while transient wiring is being plumbed through core.
        for (subpass_index, subpass) in desc.subpasses.iter().enumerate() {
            for color in subpass.color_attachments.iter().flatten() {
                if matches!(color, crate::SubpassColorAttachment::Transient { .. }) {
                    log::error!(
                        "gles: begin_subpass_render_pass: subpass {subpass_index} references a \
                         transient color attachment; wgpu-core transient-table bridge not yet wired \
                         (returning early without recording any commands)"
                    );
                    return;
                }
            }
            if let Some(ds) = &subpass.depth_stencil_attachment {
                if matches!(ds, crate::SubpassDepthStencilAttachment::Transient { .. }) {
                    log::error!(
                        "gles: begin_subpass_render_pass: subpass {subpass_index} references a \
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
        let use_framebuffer_fetch = self
            .private_caps
            .contains(super::PrivateCapabilities::SHADER_FRAMEBUFFER_FETCH);

        // Translate the subpass descriptor into the upstream single-pass
        // descriptor and reuse `begin_render_pass`. For the persistent-only
        // path, the first active subpass alone determines what attachments
        // the FBO is bound to; subpass advance later issues an invalidate +
        // rebind for Tier B (or is a no-op for Tier A).
        let active_subpass = current_index.and_then(|idx| desc.subpasses.get(idx as usize));
        let mut color_attachments_storage: arrayvec::ArrayVec<
            Option<crate::ColorAttachment<'_, super::TextureView>>,
            { crate::MAX_COLOR_ATTACHMENTS },
        > = arrayvec::ArrayVec::new();
        let depth_stencil_attachment = desc.depth_stencil_attachment.clone();

        if let Some(subpass) = active_subpass {
            // Build a color-attachment slice from the subpass's own
            // persistent color list (in slot order). Slots without a
            // `Persistent` entry become `None`.
            //
            // TODO(wgpu-core bridge): per the `Subpass::color_attachments`
            // doc contract, each `Persistent(..)` entry is supposed to
            // *consume* one slot from `color_attachment_indices` in order,
            // i.e. the parent-pass slot is `color_attachment_indices[N]`,
            // not the position within `subpass.color_attachments`. With
            // today's wgpu-core gap (`begin_subpass_render_pass` is
            // unreachable from the public API), this only matters once the
            // bridge lands -- but the implicit identity-mapping assumption
            // here will need to be replaced with proper `*_indices`
            // consumption at that point.
            for slot in subpass.color_attachments.iter() {
                let attachment = match slot {
                    Some(crate::SubpassColorAttachment::Persistent(p)) => Some(p.clone()),
                    Some(crate::SubpassColorAttachment::Transient { .. }) => {
                        // Already short-circuited above; defense in depth.
                        None
                    }
                    None => None,
                };
                if color_attachments_storage.try_push(attachment).is_err() {
                    log::error!(
                        "gles: begin_subpass_render_pass: subpass declared more color attachments \
                         than MAX_COLOR_ATTACHMENTS; truncating"
                    );
                    break;
                }
            }
        } else {
            // No active subpass: still record an empty pass so end_render_pass
            // sees a coherent state.
            for _ in 0..desc.color_attachments.len() {
                let _ = color_attachments_storage.try_push(None);
            }
        }

        // `PassTimestampWrites` borrows `&Q`, so it can't `clone()` through
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
        // failed begin will see no `subpass_state` (we return early) and
        // hit upstream's own validation. Refactoring to `Result` would
        // change the trait shape across all backends; defer that to a
        // follow-up trait-surface change.
        if let Err(err) =
            unsafe { <Self as crate::CommandEncoder>::begin_render_pass(self, &single_pass_desc) }
        {
            log::error!("gles: begin_subpass_render_pass: begin_render_pass failed: {err:?}");
            return;
        }

        self.subpass_state = Some(super::SubpassState {
            subpass_count,
            current_index,
            active_subpass_mask,
            use_framebuffer_fetch,
        });
    }

    unsafe fn next_subpass(&mut self) {
        let Some(state) = self.subpass_state.as_mut() else {
            log::error!("gles: next_subpass called outside a subpass render pass");
            return;
        };
        if state.subpass_count == 0 {
            return;
        }
        let Some(current) = state.current_index else {
            return;
        };
        let next = next_active_subpass_index(current, state.subpass_count, state.active_subpass_mask);
        debug_assert!(next <= state.subpass_count);

        if !state.use_framebuffer_fetch {
            // Tier B: invalidate any STORE_DISCARD attachments accumulated by
            // the previous subpass (the inner `begin_render_pass` already
            // pushed them onto `state.invalidate_attachments`) and let the
            // upstream `end_render_pass`/`begin_render_pass` machinery handle
            // FBO rebind on the next call. For Phase 9c the bound FBO is
            // unchanged across the (single) persistent subpass we currently
            // emit, so the per-advance discard hint is best-effort: it nudges
            // the driver to drop tile-memory contents for transient slots.
            if !self.state.invalidate_attachments.is_empty() {
                self.cmd_buffer
                    .commands
                    .push(super::Command::InvalidateAttachments(
                        self.state.invalidate_attachments.clone(),
                    ));
                self.state.invalidate_attachments.clear();
            }
        }
        // Tier A: nothing else to do — the same FBO carries every subpass.

        state.current_index = (next < state.subpass_count).then_some(next);
    }

    unsafe fn dispatch_transient(&mut self, _dispatch: &TransientDispatch) {
        // Programmable tile dispatch is not supported on GLES.
        log::error!("gles: dispatch_transient is not supported on the GLES backend");
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
// tiled-fork: end gles-backend
