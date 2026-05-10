# Tiled-Rendering Fork Conventions

This document describes the conventions used by the tiled-rendering fork
(`feature/tiled` branch) and the workflow for rebasing the fork on a new
upstream `wgpu` release. It complements the design plan in `TILED.md`.

## Marker convention

Every edit to an upstream-shared file is wrapped in `tiled-fork:` markers
so the divergence is grep-discoverable:

```rust
// tiled-fork: begin <short-tag>
... fork code ...
// tiled-fork: end <short-tag>
```

The tag describes the purpose, e.g. `module`, `re-export`, `variant
(Subpass)`, `arm (SubpassLoad)`, `tiled-caps`, `field-init
(use_framebuffer_fetch)`. Tags are not unique — the same tag may appear
in different files, but each begin/end pair must match.

`git grep "tiled-fork:"` enumerates every divergence point. Counting
begin/end markers (must always match) is a quick sanity check during
rebase:

```bash
git grep -c "tiled-fork: begin" | awk -F: '{s+=$2} END {print s}'
git grep -c "tiled-fork: end"   | awk -F: '{s+=$2} END {print s}'
```

## Module naming

Fork-only modules are named `tiled` or `subpass` so they're easy to find:

```bash
git grep -l "^mod tiled\b"
git grep -l "^mod subpass\b"
git grep -l "^pub mod resource_tiled\b"
```

Lists the primary fork-only entry-point files.

## What's the rebase workflow?

After upstream `wgpu` advances:

1. **Fetch upstream and merge into a working branch:**
   ```bash
   git fetch upstream
   git switch -c rebase-attempt
   git merge upstream/trunk
   ```

2. **Resolve conflicts.** Conflicts will mostly be in upstream-shared
   files. The `tiled-fork:` markers tell you which side to keep:
   - The fork's begin/end blocks should be preserved.
   - Lines outside markers should match upstream.

3. **Locate every divergence point** with the grep commands above. For
   each fork-only file (those listed by the `mod` greps), verify it
   compiles against the new upstream IR / trait surface:

   ```bash
   cargo check --workspace
   cargo clippy --workspace --all-targets -- -D warnings
   cargo test -p naga --lib
   cargo test -p naga --test naga
   cargo test -p wgpu-types --features std,serde
   ```

4. **Re-run the snapshot tests.** If naga's IR or backend output format
   changed upstream, the snapshot outputs in `naga/tests/out/{spv,msl,
   glsl,wgsl}/wgsl-{subpass,framebuffer-fetch}-*.{spvasm,metal,glsl,wgsl}`
   may need regeneration. Just running `cargo test -p naga --test naga
   convert_snapshots_wgsl` regenerates them; commit any changes after
   visual inspection.

5. **Verify the M1-M5 + M7 mitigations are still satisfied.** See the
   "What makes naga rebase-risky" section below.

## What makes naga rebase-risky

naga is the highest-divergence component. The fork extends three IR
enums (`ImageClass`, `Expression`, `Binding`) and adds a new public
type (`SubpassAspect`). To minimize rebase friction:

- **M1 — Variant collapsing.** `ImageClass::Subpass { aspect, multi }`
  is one variant carrying an `aspect` enum, not three separate
  variants. Cuts ~2/3 of match-arm pressure across naga.
- **M2 — End-of-enum placement.** New variants live AT THE END of
  `ImageClass`, `Expression`, `Binding`. When upstream adds a variant
  it goes above ours, not interleaved.
- **M3 — End-of-match arm placement.** Every new arm sits at the END
  of its match block, between the last upstream arm and the closing
  `}`. Marker-tagged. Same logic as M2.
- **M4 — Per-backend submodule sequestration.** Where possible, helper
  functions live in fork-only files (e.g. `naga/src/back/spv/image.rs`'s
  `write_subpass_load`); the match arms in upstream files are short
  forwarders.
- **M5 — Sub-phase splitting.** Phase 6 (naga) was split into 6a
  (IR + stubs), 6b1 (WGSL types), 6b2 (WGSL @color), 6c (SPIR-V), 6d
  (MSL), 6e (GLSL) so each commit is reviewable and bisectable.
- **M7 — Test fixture isolation.** New WGSL snapshot inputs use the
  `subpass-*` and `framebuffer-fetch-*` prefixes; outputs go in the
  matching `wgsl-subpass-*` / `wgsl-framebuffer-fetch-*` files. Easy
  to spot in git.

## Phase commit history

The fork was built incrementally, one focused commit per phase, each
with a fresh-Opus Phase Review against `TILED.md` and `CLAUDE.md`:

| Phase | Title | Lines | Files | Notes |
|---|---|---:|---:|---|
| 0 | wgpu-types foundation types | +1081 | 4 | 16 backend-agnostic types, 5 feature flags |
| 1 | wgpu-hal extension traits | +669 | 6 | TiledApi/TiledDevice/TiledCommandEncoder |
| 2 | per-backend HAL stubs | ~600 | 10 | 5 backends × tiled.rs + mod.rs hook |
| 3 | wgpu-core scaffolding | +219 | 8 | resource wrappers, registry slots, IDs |
| 4 | wgpu-core subpass scaffolding | +135 | 3 | Global stubs + SubpassRenderPass |
| 5 | wgpu public API tiled capabilities | +114 | 7 | Adapter::tiled_capabilities |
| 6a | naga IR additions + match-arm stubs | +533 | 32 | Collapsed Subpass variant + 110 markers |
| 6b1 | naga WGSL frontend (subpass types) | +253 | 8 | subpass_input<T> + subpassLoad |
| 6b2 | naga WGSL @color(N) | +252 | 7 | framebuffer-fetch attribute |
| 6c | naga SPIR-V backend emission | +355 | 6 | OpImageRead + InputAttachmentIndex |
| 6d | naga MSL backend emission | +187 | 3 | [[color(N)]] fragment-arg |
| 6e | naga GLSL backend emission | +381 | 6 | dual-mode subpassInput / inout |
| 8 | naga snapshot fixtures | +500 | ~30 | subpass-* + framebuffer-fetch-* |

Total: ~5,000 LOC of fork-only new code, ~250 LOC of marker-tagged
upstream-shared edits.

## Known divergences from upstream `wgpu`

These are the unavoidable surfaces where the fork diverges from
upstream. Documented here so a rebase doesn't accidentally undo them:

1. **`FeaturesWGPU` bit allocation.** Bits 14, 16, 24, 25, 35 used.
   See TILED.md "Known unavoidable sources of upstream divergence" #3.

2. **`naga::back::glsl::Options::use_framebuffer_fetch` field.**
   Adding a field to a public non-`#[non_exhaustive]` struct is a
   breaking change for downstream `naga` consumers. We did not add
   `#[non_exhaustive]` because it would break the in-tree
   `wgpu-hal::gles::device` initializer. See TILED.md #4.

3. **`SubpassState` field on backend `CommandEncoder`/`CommandState`.**
   Per-backend internal state; not part of public API.

4. **naga `ImageClass::Subpass`, `Expression::SubpassLoad`,
   `Binding::ColorAttachmentRead` variants.** Public enum extensions;
   breaking for naga library consumers who exhaustively match. Same
   risk profile as upstream naga's own variant additions.

## What's still deferred

Phase 9 (real wgpu-hal backends — Vulkan transient images, Metal
memoryless tiles, GLES `EXT_shader_framebuffer_fetch` runtime path)
is not yet ported. Phase 7 (visual examples) depends on Phase 9.
Until those land, the public API surface is reachable but every
end-to-end attempt eventually returns
`SubpassRenderPassError::NotImplemented` /
`DeviceError::Unexpected`.
