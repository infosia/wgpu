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
- [ ] Phase 7 — Examples
- [ ] Phase 8 — Tests + benches
- [ ] Phase 9 — Backend real impls (Vulkan, Metal, GLES)
- [ ] Phase 10 — Docs
