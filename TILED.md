# wgpu Tiled Rendering Fork Plan

This branch (`feature/tiled`) ports the tile-based deferred rendering (TBDR) extension
from `../wgpu-tiled` (sibling clone of `infosia/wgpu-tiled`) into upstream `wgpu`,
**deliberately reshaped to minimize divergence from upstream**.

This document is the design plan. The reference implementation we are porting from
lives in `../wgpu-tiled` and is described in detail in that repo's `TILED.md`.

---

## Goals

1. **Port the tiled rendering feature set** from `wgpu-tiled`:
   transient attachments, multi-subpass render passes, render-graph builder,
   subpass inputs, framebuffer fetch, naga IR + backend support.

2. **Make the diff easy to follow upstream over time.**
   We do **not** plan to PR this upstream, but every upstream change to `wgpu`
   should rebase cleanly with minimal merge work.

## Guiding principle

> Do not change existing API signatures that introduce a breaking change whenever
> possible. Adding properties/methods to existing classes is acceptable only when
> it is expected to be easy to merge to upstream. It is fine to add new classes,
> methods, and public APIs.

In practice this means:

- **Wrap, don't extend** — when an existing type would require new fields,
  introduce a wrapping type instead (e.g., `SubpassRenderPipelineDescriptor`
  wraps `RenderPipelineDescriptor` rather than adding a field to it).
- **Sidecar entry points** — when `RenderPassDescriptor` would need new fields,
  add a sibling `SubpassRenderPassDescriptor` and a new
  `begin_subpass_render_pass()` method instead of expanding the existing one.
- **Extension traits** — instead of adding methods to upstream HAL traits
  (`Api`, `Device`, `CommandEncoder`), define new sub-traits
  (`TiledApi`, `TiledDevice`, `TiledCommandEncoder`) that backends opt into.
- **Pure additions only** to upstream-shared files — new modules, new flag
  constants, new inherent methods. No struct field additions to public types.
- **Crate-internal extensions are OK** when they are not part of the public
  API surface (e.g., `pub(crate)` enum variants, `pub(super)` struct fields).

## File layout

### Fork-only files (where the bulk of the diff lives)

```
wgpu-types/src/tiled.rs                       # 16 backend-agnostic tiled types
wgpu-hal/src/tiled.rs                         # TiledApi/TiledDevice/TiledCommandEncoder
wgpu-hal/src/dynamic/tiled.rs                 # DynTiledDevice/DynTiledCommandEncoder
wgpu-hal/src/{vulkan,metal,gles,dx12,noop}/tiled.rs
                                              # per-backend impls + TransientAttachment
wgpu-core/src/command/subpass.rs              # validation, SubpassRenderPassInfo, errors
wgpu-core/src/resource_tiled.rs               # TransientAttachment resource wrapper
wgpu-core/src/device/tiled.rs                 # Device::create_transient_attachment
wgpu/src/api/subpass.rs                       # SubpassRenderPassDescriptor
                                              # + SubpassRenderPipelineDescriptor
wgpu/src/api/render_graph.rs                  # RenderGraphBuilder (verbatim from wgpu-tiled)
wgpu/src/api/tiled_caps.rs                    # TiledCapabilities + Adapter::tiled_capabilities()
wgpu/src/dispatch_tiled.rs                    # SubpassRenderPassInterface trait
examples/features/src/{deferred_rendering,subpass_render_graph,subpass_msaa}/
naga/tests/in/wgsl/{subpass-*,framebuffer-fetch-*}.wgsl
```

### Upstream-shared files we touch (kept minimal, marker-tagged)

| File | Edit | Lines | Type |
|---|---|---:|---|
| `wgpu-types/src/features.rs` | 4 new `Features` flag constants | ~10 | additive constants |
| `wgpu-types/src/lib.rs` | `mod tiled; pub use tiled::*;` | ~3 | module + re-exports |
| `wgpu-hal/src/lib.rs` | `mod tiled; pub use tiled::*;` | ~3 | module + re-exports |
| `wgpu-hal/src/dynamic/mod.rs` | `mod tiled;` | ~2 | module |
| `wgpu-hal/src/{5 backends}/mod.rs` | `mod tiled;` | ~2 each | module |
| `wgpu-hal/src/{5 backends}/command.rs` | `pub(super) subpass_state: Option<SubpassState>` field + 1 dispatch line | ~5 each | crate-internal field |
| `wgpu-core/src/command/render.rs` | `mod subpass; pub use subpass::*;` + 1 match arm | ~10 | additive |
| `wgpu-core/src/command/render_command.rs` | `RenderCommand::NextSubpass` variant | ~2 | crate-internal enum |
| `wgpu-core/src/{id,hub,track/mod,resource}.rs` | small registry additions | ~15 total | crate-internal |
| `wgpu-core/src/device/{mod,resource}.rs` | `mod tiled;` | ~3 | module |
| `wgpu/src/lib.rs` | re-export block | ~15 | re-exports |
| `wgpu/src/api/{render_pass,command_encoder,device,adapter}.rs` | inherent methods only | ~50 total | additive methods |
| `wgpu/src/backend/wgpu_core.rs` | new methods inside existing impl blocks | ~30 | additive |
| `wgpu/src/backend/webgpu.rs` | no-op stubs in existing impl blocks | ~15 | additive |
| naga IR + ~10 backend files (already-refactored variant) | enum variant additions, match arms | ~400 | accepted breakage |

**Total: ~600 LOC of upstream-shared edits** vs ~14,664 LOC in `wgpu-tiled` —
a ~24× reduction in upstream divergence surface.

## Two key wraps (the heart of the no-breaking-changes rule)

### 1. `Limits` extension → separate `TiledCapabilities` query

`wgpu-tiled` adds 4 fields to `wgpu_types::Limits`. That breaks every
`Limits { ... }` struct literal. Instead:

```rust
// Untouched: wgpu_types::Limits

// New, fork-only: wgpu/src/api/tiled_caps.rs
pub struct TiledCapabilities {
    pub max_subpasses: u32,
    pub max_subpass_color_attachments: u32,
    pub max_input_attachments: u32,
    pub estimated_tile_memory_bytes: u32,
}

// New inherent method on Adapter:
let caps = adapter.tiled_capabilities();
```

### 2. `RenderPipelineDescriptor` extension → wrapping `SubpassRenderPipelineDescriptor`

`wgpu-tiled` adds an `Option<SubpassTarget>` field to `RenderPipelineDescriptor`.
Vulkan needs the subpass info at pipeline creation time, so we cannot defer it.
Instead of extending the descriptor, we wrap it:

```rust
// Untouched: RenderPipelineDescriptor

// New, fork-only: wgpu/src/api/subpass.rs
pub struct SubpassRenderPipelineDescriptor<'a> {
    pub base: RenderPipelineDescriptor<'a>,    // upstream descriptor unchanged
    pub subpass_target: SubpassTarget,
}

// New inherent method on Device:
let pipeline = device.create_subpass_render_pipeline(&desc);
```

The same pattern applies to `RenderPassDescriptor` → `SubpassRenderPassDescriptor`
and `CommandEncoder::begin_render_pass` → `begin_subpass_render_pass`.

## Phase ordering (each phase = one focused commit)

The original plan had 9 phases (0-8). During execution, Phase 3
(wgpu-core integration) and Phase 4 (public API) were each scoped wide
enough that they were split into scaffolding-only commits + a deferred
"real implementation" phase, to keep each commit reviewable under the
single-commit-per-phase rule. The current sequence:

1. **Phase 0** — `wgpu-types`: foundation types in `tiled.rs`, 5 feature flag bits, `mod` registration. Limits and `RenderPipelineDescriptor` are **not** touched.
2. **Phase 1** — `wgpu-hal` extension traits in `tiled.rs`. Upstream `Api`/`Device`/`CommandEncoder` traits unchanged.
3. **Phase 2** — Per-backend HAL stubs in `{vulkan,metal,gles,dx12,noop}/tiled.rs`. Each backend's `mod.rs` gets a single `mod tiled;` line.
4. **Phase 3** — `wgpu-core` scaffolding: `resource_tiled.rs`, `device/tiled.rs`, registry/tracker slots, IDs. Methods return `Err(DeviceError::from_hal(Unexpected))`.
5. **Phase 4** — `wgpu-core` subpass scaffolding: `command/subpass.rs` with stub `Global` methods returning `SubpassRenderPassError::NotImplemented`.
6. **Phase 5** — `wgpu` public API: `api/subpass.rs`, `api/render_graph.rs`, `api/tiled_caps.rs`, `dispatch_tiled.rs`. Existing types receive only purely-additive inherent methods.
7. **Phase 6** — naga: cherry-pick the **already-refactored** subpass variant from `wgpu-tiled` (post `4ba4a1c11`, post `b1c2d5d50`). Do not redo the IR work.
8. **Phase 7** — Examples: `deferred_rendering`, `subpass_render_graph`, `subpass_msaa`. **No** churn to existing examples.
9. **Phase 8** — Tests + benches: copy from `wgpu-tiled` for new tiled-related tests. **No** churn to existing tests.
10. **Phase 9** — Backend real implementations (Vulkan / Metal / GLES). Replaces the per-backend `todo!()` stubs and lights up the path through wgpu-core.
11. **Phase 10** — Docs: refresh `TILED.md`, add `docs/tiled-fork-conventions.md` describing the marker convention and rebase workflow.

## Conventions

### Marker comments

Every edit to an upstream-shared file is wrapped:

```rust
// tiled-fork: begin <short-tag>
... fork code ...
// tiled-fork: end <short-tag>
```

`git grep "tiled-fork:"` then enumerates every divergence point in seconds.

### Module naming

All fork-only modules are named `tiled` or `subpass` so that
`git grep -l "mod tiled"` and `git grep -l "mod subpass"` enumerate the
fork-only entry points.

### Type design

- `#[non_exhaustive]` on every new public type **with named fields or variants**,
  so future fork-internal field additions don't compound the divergence problem.
  Exempt: transparent `#[repr(transparent)]` newtype wrappers
  (e.g., `SubpassIndex(pub u32)`, `ActiveSubpassMask(pub u32)`) where the entire
  point is external literal construction. We will not add fields to a
  `repr(transparent)` newtype, so the rule has no work to do there.
- New types live in fork-only files; never add fields to upstream public types.

### Commits

- One phase per commit.
- Each commit edits ≤ 3 upstream-shared files when possible.
- Commit messages prefixed `tiled-fork(<phase>):` for grep-friendliness.

## Escape hatches

If a planned edit is harder to merge than expected, the fallback chain is:

1. **First fallback** — extract into a new file via `mod` registration; leave
   only 1 line of `mod foo;` in the upstream-shared file.
2. **Second fallback** — introduce a wrapping/sibling type (the pattern used
   for `Limits` → `TiledCapabilities` and `RenderPipelineDescriptor`
   → `SubpassRenderPipelineDescriptor`).
3. **Third fallback** — gate behind a fork-only Cargo feature flag so the
   upstream-shared file stays untouched when the flag is off. Last resort
   because it doubles maintenance.

## Known unavoidable sources of upstream divergence

1. **`SubpassState` field on backend `CommandEncoder`/`CommandState`.**
   Backends need persistent subpass tracking. Mitigated by a single
   `pub(super) subpass_state: Option<SubpassState>` field, where `SubpassState`
   itself lives in the new `tiled.rs` submodule.

2. **naga `ImageClass::SubpassInput*` / `Expression::SubpassLoad` variants.**
   Adding variants to a public enum is breaking for external naga consumers
   that exhaustively match. Unavoidable — there is no way to add subpass
   support to naga IR without expanding the IR. Upstream naga adds variants
   regularly.

4. **`naga::back::glsl::Options` struct field addition.** Phase 6e adds a
   `use_framebuffer_fetch: bool` field to `naga::back::glsl::Options`. This is
   a known-breaking change for any out-of-tree downstream `naga` consumer
   that constructs `Options { ... }` via struct literal. We do NOT mark the
   struct `#[non_exhaustive]` because doing so would block the standard
   `Options { ..Default::default() }` workaround pattern in external crates
   (including in-tree `wgpu-hal`). Mitigation: marker comment + this entry.

3. **Feature-flag bit allocation.** This fork allocates bits 14, 16, 24, 25,
   and 35 inside `FeaturesWGPU` (matching `wgpu-tiled`). All five bits are
   currently free in upstream `wgpu` v29.0.3:
   - bit 14 = `MULTI_SUBPASS`
   - bit 16 = `PROGRAMMABLE_TILE_DISPATCH`
   - bit 24 = `FRAMEBUFFER_FETCH`
   - bit 25 = `SHADER_FRAMEBUFFER_FETCH`
   - bit 35 = `TRANSIENT_ATTACHMENTS` (formerly `SHADER_PRIMITIVE_INDEX`)

   We deliberately match `wgpu-tiled`'s allocation rather than reserving a
   high-bit range, because (a) cherry-picking later `wgpu-tiled` work into
   this fork is easier when bit assignments line up, and (b) the natural
   "fill-the-holes" pattern is what upstream itself uses. Risk: if upstream
   later allocates one of these bits, we collide and must renumber. Mitigation:
   `tiled-fork:` markers on every flag site make a renumber a 5-line edit.

5. **Construction-side subpass descriptors are not `#[non_exhaustive]`.**
   The Type-design convention in this document mandates `#[non_exhaustive]`
   on every new public type with named fields or variants, exempting only
   `repr(transparent)` newtypes. Phase 7a takes a documented exception for
   six construction-side public types because external callers must
   literal-construct them via struct expressions (just like upstream's
   `RenderPassDescriptor`, which is also not `#[non_exhaustive]`):

   - `wgpu::SubpassRenderPassDescriptor`
   - `wgpu::SubpassDescriptor`
   - `wgpu::SubpassColorAttachment`
   - `wgpu::SubpassDepthStencilAttachment`
   - `wgt::SubpassInputAttachment`
   - `wgt::SubpassDependency`
   - `wgt::SubpassTarget`        *(Phase 7b)*
   - `wgt::SubpassTargetDesc`    *(Phase 7b)*

   `SubpassRenderPipelineDescriptor` keeps `#[non_exhaustive]` because its
   only required public field is the upstream `RenderPipelineDescriptor` it
   wraps; future fork-internal subpass-target fields can be added without
   pinning down a full struct-literal contract. All six exempted sites carry
   a `// tiled-fork: non-exhaustive-relax` marker so a future merge that
   re-applies upstream's attribute is grep-detectable. Mitigation if upstream
   ever marks one of these types `#[non_exhaustive]`: drop the marker comment
   and re-add the attribute — the change is one line per site.

## Reference

The reference implementation is `../wgpu-tiled` at branch `main`,
fork-base `wgpu v29.0.1` (`923b89695`).

That repo's `TILED.md` documents the implementation in detail. This document
is the **port plan**; it covers what we keep, what we reshape, and why.

## Status

- [x] Plan agreed and saved to TILED.md
- [x] Phase 0 — wgpu-types foundation
- [x] Phase 1 — wgpu-hal extension traits
- [x] Phase 2 — Per-backend HAL impls (stubs; real impls in later phases)
- [x] Phase 3 — wgpu-core scaffolding (resource wrappers, Device methods, IDs)
- [x] Phase 4 — wgpu-core subpass scaffolding (NotImplemented stubs; full validation in Phase 9)
- [x] Phase 5 — wgpu public API (TiledCapabilities query; descriptors & methods deferred)
- [x] Phase 6a — naga IR additions + match-arm stubs (collapsed Subpass variant; backends Phase 6b)
- [x] Phase 6b1 — naga WGSL frontend: subpass_input types + subpassLoad builtin (excludes @color)
- [x] Phase 6b2 — naga WGSL frontend: @color(N) framebuffer fetch attribute
- [x] Phase 6c — naga SPIR-V backend: SubpassData type + OpImageRead emission, SPV_EXT_shader_tile_image
- [x] Phase 6d — naga MSL backend: [[color(N)]] fragment-arg emission for subpass + framebuffer fetch
- [x] Phase 6e — naga GLSL backend: subpassInput/inout dual-mode emission
- [x] Phase 7 — Examples
  - [x] Phase 7a — `subpass_render_graph` headless smoke test (2-subpass persistent attachments; no shaders/draws yet)
  - [x] Phase 7b — `deferred_rendering` (3-subpass G-buffer/lighting/composite). Runs end-to-end on Vulkan after Phase 11i (60 FPS verified locally).
  - [x] Phase 7c — `subpass_msaa` (2-subpass MSAA line demo). Runs end-to-end on Vulkan at 60 FPS with zero validation messages in debug mode. Left/Right arrow keys toggle between 1x and adapter-max MSAA; MSAA mode adds a follow-up regular pass to resolve the MSAA intermediate into the swapchain. Phase 7 set complete.
- [x] Phase 8 — Tests + benches (naga snapshot fixtures: subpass-* + framebuffer-fetch-*)
- [x] Phase 9 — Backend real impls (Vulkan, Metal, GLES)
  - [x] Phase 9a1 — Vulkan TransientAttachment (real VkImage + LAZILY_ALLOCATED)
  - [x] Phase 9a2 — Vulkan multi-subpass VkRenderPass + state machine
  - [x] Phase 9a3 — Vulkan input-attachment descriptor sets + create_subpass_render_pipeline
  - [x] Phase 9c — GLES TransientAttachment + multi-subpass state machine (Tier A/B detection)
  - [x] Phase 9b — Metal TransientAttachment + multi-subpass state machine + create_subpass_render_pipeline validation
- [x] Phase 10 — Docs (`docs/tiled-fork-conventions.md` + this status block)
- [x] Phase 11 — wgpu-core bridge to expose Phase 9 HAL through the public API
  - [x] Phase 11a — Box<dyn DynTiledDevice> storage in wgpu-core's `Device::raw`
  - [x] Phase 11b — `Adapter::tiled_capabilities()` reports real values from HAL
  - [x] Phase 11c — `Device::create_transient_attachment` actually invokes HAL
  - [x] Phase 11d — Public `wgpu::SubpassRenderPassDescriptor` API + `begin_subpass_render_pass`
    - [x] Phase 11d1 — `DynTiledCommandEncoder: DynCommandEncoder` + Box storage
    - [x] Phase 11d2 — wgpu-core SubpassRenderPass machinery (begin/next/end; persistent attachments)
    - [x] Phase 11d3 — public wgpu::SubpassRenderPassDescriptor + begin_subpass_render_pass
    - [x] Phase 11d4 — per-subpass draw machinery (set_pipeline, set_bind_group, set_vertex_buffer, set_index_buffer, draw, draw_indexed, set_viewport, set_scissor_rect)
  - [ ] Phase 11e — `RenderPass::next_subpass` + `current_subpass_index` on upstream `RenderPass`
        (absorbed: the new `SubpassRenderPass` handle already exposes both methods; this sub-phase is
        only needed if we ever want subpass advance from inside an *upstream* single-pass `RenderPass`,
        which has no obvious caller. Left open as a marker, low priority.)
  - [x] Phase 11f — Public `SubpassRenderPipelineDescriptor` + `Device::create_subpass_render_pipeline`
  - [x] Phase 11g — Resource-tracker registration in `SubpassRenderPass::set_*` (pipeline / bind-group / vertex / index)
  - [x] Phase 11h — `SubpassRenderPipelineDescriptor::new` constructor so external callers can build the descriptor without dropping `#[non_exhaustive]`
  - [x] Phase 11i — `BindingType::SubpassInput` variant in `wgt::BindingType` plus the bind-group-layout / pipeline-layout / HAL plumbing needed to consume it. Unblocks Phase 7b/7c: deferred_rendering now runs end-to-end on Vulkan (60 FPS verified locally on NVIDIA RTX 5060 Ti).
  - [x] Phase 11i2 — Scalar-kind validation in `wgt::SubpassInputAspect::Color`. `Color` carries a `sample_type: SubpassInputSampleType` (Float/Sint/Uint) mirroring upstream `BindingType::Texture::sample_type`; `derive_binding_type` translates `naga::ScalarKind` and `check_binding_use` verifies it matches the shader's `subpass_input<T>` scalar kind. A `subpass_input<i32>` bound against an `f32` attachment now fails at pipeline-layout-derivation time instead of producing undefined GPU output.
  - [x] Phase 11j — Vulkan validation cleanup for `SubpassRenderPass`. All four debug-mode validation errors resolved:
    - [x] Pre-pass texture barriers. `command_encoder_begin_subpass_render_pass` now calls `emit_pre_pass_barriers` which sets the wgpu-core tracker state for each color / resolve / depth-stencil attachment and emits any pending layout transitions via `transition_textures` before `vkCmdBeginRenderPass`. Fixes the `InvalidImageLayout` validation error.
    - [x] Swapchain post-pass layout. With the tracker correctly knowing the surface texture is in `COLOR_TARGET` state after the pass, `Queue::submit`'s present transition fires correctly, clearing `VkPresentInfoKHR-pImageIndices-01430`.
    - [x] Acquire-semaphore cascade. Cleared as a cascade of the swapchain layout fix.
    - [x] Framebuffer-cache lifecycle. Two-part fix: (1) attachment views are now registered in `trackers.views` (matching upstream `RenderPassInfo::start`), keeping the view's `Arc` alive until queue submission completes; (2) the `VkFramebuffer` cache moved from per-`CommandEncoder` to `DeviceShared` with a reverse-lookup from view identity to dependent framebuffer keys, and `Device::destroy_texture_view` now sweeps the dependent framebuffers before destroying the underlying `VkImageView`. Verified: deferred_rendering runs at 60 FPS in debug mode with **zero validation messages**.
    - [x] Read-only-depth refinement (Phase 11j3). `emit_pre_pass_barriers` now mirrors upstream's `RenderPassInfo::start` pattern: when both depth and stencil aspects are read-only AND the device supports `DownlevelFlags::READ_ONLY_DEPTH_STENCIL`, the transition target is `DEPTH_STENCIL_READ | RESOURCE` so the same attachment can be sampled in the pass; otherwise the typical `DEPTH_STENCIL_WRITE`.

## Snapshot at session end (2026-05-11)

The fork has landed Phases 0-6e, 8, 9 (a1/a2/a3, b, c), 10, 11 (a, b, c,
d1/d2/d3/d4, f) across 28 commits on `feature/tiled`. Cumulative diff
against the branch base is ~12,000 LOC across ~130 files (~95% of it
in fork-only new files). `git grep "tiled-fork:"` enumerates every
divergence point against upstream-shared files.

### What works end-to-end on Vulkan / Metal / GLES

A user can drive a full multi-subpass deferred-shading pipeline from
the public `wgpu` API:

```rust
let caps = adapter.tiled_capabilities();
//   -> real max_subpasses / max_input_attachments / max_color_attachments
//      / estimated_tile_memory_bytes from the physical device.

let transient = device.create_transient_attachment(&desc)?;
//   -> Vulkan: real `VkImage` with `TRANSIENT_ATTACHMENT | INPUT_ATTACHMENT`
//      and `LAZILY_ALLOCATED` memory.
//      Metal: `MTLTexture` with `MTLStorageMode::Memoryless`.
//      GLES:  `glRenderbuffer` (MSAA-capable).
//   Drop cleans up via `device.raw_tiled().destroy_transient_attachment_dyn`.

let pipeline = device.create_subpass_render_pipeline(&SubpassRenderPipelineDescriptor {
    base: rp_desc,
    subpass_target: target,
})?;
//   -> Vulkan: builds a compatible VkRenderPass from `SubpassTarget` and
//      records input-attachment descriptor-set bindings.
//      Metal / GLES: forwards to `create_render_pipeline` (no compat pass
//      needed; format derivation will land later).

let mut pass = encoder.begin_subpass_render_pass(&SubpassRenderPassDescriptor { ... });
//   -> Vulkan: vkCmdBeginRenderPass on the multi-subpass VkRenderPass +
//      allocates per-subpass `VK_DESCRIPTOR_TYPE_INPUT_ATTACHMENT` sets.
//      Metal: prepares a single-encoder multi-subpass state machine.
//      GLES:  Tier A (framebuffer fetch) no-op advance OR Tier B
//             FBO-rebind path with `glInvalidateFramebuffer`.
pass.set_pipeline(&pipeline);
pass.set_bind_group(0, Some(&bg), &[]);
pass.set_vertex_buffer(0, vbuf.slice(..));
pass.set_index_buffer(ibuf.slice(..), IndexFormat::Uint32);
pass.set_viewport(0.0, 0.0, w, h, 0.0, 1.0);
pass.set_scissor_rect(0, 0, w as u32, h as u32);
pass.draw_indexed(0..n, 0, 0..1);
pass.next_subpass();
pass.set_pipeline(&pipeline_lighting);
pass.draw(0..6, 0..1);
pass.end();
//   -> vkCmdEndRenderPass + unlock encoder. Drop is idempotent so
//      forgetting `.end()` still releases the HAL render pass.

// WGSL using the new types compiles correctly to every backend:
//   var gbuffer: subpass_input<f32>;
//   ... subpassLoad(gbuffer) ...
//   @fragment fn main(@color(0) prev: vec4<f32>) -> ... { ... }
```

Cross-cutting verifications:
- `cargo check --workspace`               : clean
- `cargo clippy --workspace -D warnings`  : clean (every backend combo)
- `cargo test -p wgpu-types`              : 85 unit + 14 doc tests
- `cargo test -p wgpu-hal --features vulkan,gles tiled` : 18 tests
- `cargo test -p naga --lib`              : 135 tests
- `cargo test -p naga --test naga`        : 200 snapshot tests (incl.
                                            subpass-* + framebuffer-fetch-*)
- `cargo test -p wgpu-core --lib`         : 42 tests
- `cargo test -p wgpu --lib`              : 17 tests

### What's deferred / still rough

These are documented limitations that wouldn't block visual examples
but should be addressed in subsequent passes:

- **Phase 7 — visual examples.** Phase 7a landed a headless
  `subpass_render_graph` smoke test (2-subpass persistent attachments,
  no shaders/draws — exercises `begin_subpass_render_pass` +
  `next_subpass` + drop end-to-end). Phases 7b (`deferred_rendering`)
  and 7c (`subpass_msaa`) are deferred: both reference-fork examples
  rely on `RenderGraphBuilder` (declarative subpass graph) and the
  reference's extended `Limits` (`max_subpasses`,
  `max_input_attachments`, `max_subpass_color_attachments`), neither
  of which is ported in this fork per the "no breaking changes" rule.
  Porting them requires rewriting the example bodies to build
  `SubpassRenderPassDescriptor` / `SubpassDescriptor` arrays
  literally and using `TiledCapabilities` for capacity queries — a
  ~1,500-LOC follow-up that mirrors the reference scene visually but
  uses a different declaration surface.

- ~~Resource-tracker registration in `SubpassRenderPass`~~ — **fixed
  in Phase 11g.** `set_pipeline`, `set_bind_group`, `set_vertex_buffer`,
  and `set_index_buffer` now insert their resource Arcs into
  `cmd_buf.trackers` so they survive until queue submission. Buffer
  bindings additionally record `BufferUses::VERTEX|INDEX` via
  `BufferTracker::set_single`. (The per-draw barrier-plan side of
  upstream's tracking — `UsageScope::merge_single` — is still not
  wired; that's a separate sub-pass on the per-draw validation
  hardening listed below.)

- **wgpu-side draw validation gaps in `SubpassRenderPass`.**
  `set_pipeline` skips `pass_context.check_compatible` and pipeline
  format checks. `set_bind_group` skips `BindGroupLayout::is_compatible`
  against the pipeline layout and `same_device`. `set_*_buffer` skips
  `BufferUsages::INDEX | VERTEX` and `same_device` checks.
  `draw` / `draw_indexed` skip vertex-buffer-limit checks.
  `set_viewport` / `set_scissor_rect` skip range / zero-size checks.
  All of these mirror upstream's checks; replicating them is layered
  validation work, not architectural.

- **`set_bind_group(index, None, &[])`** silently elides the HAL call
  rather than unbinding. State drifts from the upstream binder model;
  should be either an explicit unbind or an error.

- **`Transient` subpass attachments.** `SubpassColorAttachment::Transient`
  and `SubpassDepthStencilAttachment::Transient` arms currently return
  `SubpassRenderPassError::TransientNotWired`. The wgpu-core
  transient-attachment table that resolves `transient_index` to an
  `Arc<TransientAttachment>` for the render-pass binder isn't built
  yet. Only the *Persistent* arms work today. Without this, the
  bandwidth savings the feature exists for aren't reachable from the
  public API even though the HAL allocator path is real.

- **`dispatch_transient`.** Forward-compat scaffolding for Apple-GPU
  programmable tile dispatch; still `todo!()` on every backend. No
  wgpu-core path exercises it.

- **Phase 11e** — `RenderPass::next_subpass` / `current_subpass_index`
  on the upstream `RenderPass` type. Effectively absorbed by the new
  `SubpassRenderPass` handle (which already exposes both); left in
  the status list as a marker. Low priority unless a caller surfaces
  that wants subpass advance from inside an upstream single-pass
  `RenderPass`.

- **Subpass-input *globals* on MSL.** Phase 6d's MSL backend returns
  `FeatureNotImplemented` for the global-variable form
  (`var gbuf: subpass_input<f32>;`); the entry-point `@color(N)` form
  works. Lifting globals to `[[color(N)]]` arguments needs an
  `Options::subpass_color_slots` mapping that's a follow-up. Vulkan
  and GLES handle the global form.

- **GLES Tier B FBO-rebind-per-subpass.** Phase 9c lit up the
  state-machine + per-advance `glInvalidateFramebuffer` discard hint,
  but does not yet rebind a fresh FBO with each subpass's attachment
  set. Multi-subpass execution silently renders only into the first
  subpass's attachments when Tier A (`EXT_shader_framebuffer_fetch`)
  is unavailable. The adapter gates `MULTI_SUBPASS` /
  `TRANSIENT_ATTACHMENTS` on `has_framebuffer_fetch` to avoid the
  miscompile.

- **DX12 backend.** Permanent stub: returns
  `Err(DeviceError::Unexpected)` from every tiled-trait method.
  `MULTI_SUBPASS` / `TRANSIENT_ATTACHMENTS` are never advertised on
  DX12, so callers should never reach those methods.

- **Eager-dispatch caller constraint on `begin_subpass_render_pass`.**
  Documented in `wgpu-core/src/command/subpass.rs` module docs: a
  subpass-mode render pass must be the first command-encoder
  operation. If the caller queued any other commands (e.g.
  `copy_buffer_to_buffer`) before calling `begin_subpass_render_pass`,
  those commands will be replayed at `finish`-time *after* the
  subpass pass is already encoded into the HAL stream — wrong order
  at the HAL level. Reconciling this with upstream's deferred-replay
  model is a future architectural pass.

- **Trace surface.** wgpu-core's `trace::Action` is not extended for
  subpass-aware pipelines. Replays of fork-recorded traces re-create
  subpass pipelines as ordinary pipelines (loss of fidelity, not a
  crash).

- **`naga::back::glsl::Options` field addition.** The
  `use_framebuffer_fetch` field is a known-breaking change for
  external naga consumers; `#[non_exhaustive]` would block the
  in-tree `wgpu-hal::gles` initializer's `Options { ... }` pattern,
  so the breakage is accepted. See `docs/tiled-fork-conventions.md`
  "Known divergences" #2.
