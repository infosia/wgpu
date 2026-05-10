# CLAUDE.md — Permanent Rules for Claude Code

This file captures **phase-agnostic rules** that apply to all Claude Code
sessions working on wgpu, regardless of which phase is active.


## Critical Rules (Apply Always)

### Private Information Check

At the end of each phase, scan all changed and new files for **private or sensitive information** before committing:

- **Filesystem paths** — No absolute paths like `/Users/...`, `/home/...`, or `C:\Users\...`
- **Person names** — No real names in source code, comments, or strings (git author config is fine)
- **Passwords / secrets** — No hardcoded passwords, API keys, tokens, or credentials
- **Email addresses** — No personal email addresses in source files
- **Internal URLs / IPs** — No hardcoded internal hostnames or IP addresses
- **Mobile signing values** — No Apple Team IDs, signing identity names,
  device UDIDs, bundle-ID prefixes, provisioning-profile UUIDs, Android
  keystore aliases, or APK signing keys. Mobile runners (Phase 4 onward)
  read these from environment variables; the values must never appear in
  scripts, docs, examples, dotfiles checked into the repo, or commit
  messages, and the physical artifacts (`*.mobileprovision`, `*.p12`,
  `*.keystore`, `*.jks`) must stay outside the working tree.

If any are found, stop committing and remove or replace them.

### MUST follow

1. **No panics in library code.** Never use `panic!`, `unwrap()`, `expect()`,
   `unreachable!()`, or indexing that can panic. Use `?` with `Result`.
   During phases where bodies are stubs (like Phase D), use `todo!()`.
   `panic!("…")` is only acceptable as a `todo!()` substitute inside `const fn`
   bodies (where `todo!()` does not expand).

## Phase Review — Clean Review Then Fix

This workflow is **mandatory at the end of every phase** (Phase D
onward, including any new phase opened on demand). Its purpose is to
catch issues the implementing agent missed due to context saturation or
tunnel vision.

1. **Spawn a fresh agent with no session context** to review all files
   changed in the phase. The reviewer must:
   - Start without any knowledge of the design discussion or implementation
     history
   - Read `TILD.md`
     phase-specific review-context document
   - Output findings tagged by severity: `CRITICAL`, `MAJOR`, `MINOR`
   - For each finding, cite file and line, explain the issue, and recommend
     a fix

2. **Fix findings in order**: `CRITICAL` → `MAJOR` → `MINOR`. Do not skip
   severity tiers. Document each fix in the phase's completion notes.
   Findings matching known-intentional decisions (captured in a phase's
   review-context document) are not fixes — mark them as "intentional,
   not a finding".

3. **Re-run the build gates after all fixes**:

   Both must succeed with zero errors. Clippy warnings should be fixed or
   explicitly allowed with a justification comment.

The fresh-agent review is the final gate before declaring a phase complete.
Do not advance to the next phase without a clean review + fix cycle.

### Single-commit-per-phase rule

**Default for autonomous phases: one commit at phase end.** All
implementation work, Phase Review fixes, and spec amendments belong in
a single commit covering the whole phase. Do not create intermediate
"checkpoint" commits.

Why: the single commit is what reviewers, downstream agents, and
`git log` consumers actually navigate by. Intermediate commits create
noise without adding bisectability (a half-implemented phase is rarely
useful as a bisect target), and the Phase Review's whole point is that
the *committed* state is reviewed — not a sequence of mid-flight
states.

Concrete shape:

- Implement on the working tree (no `git commit` calls until the end).
- Run the gate once on the working tree.
- Spawn the fresh-agent Phase Review on `git diff HEAD` plus untracked
  files.
- Fix CRITICAL → MAJOR → MINOR on the working tree.
- Re-run the gate (mandatory after any post-review code change).
- `git add` the specific files (never `git add -A` per the git-safety
  rules below) and `git commit` once. Use a HEREDOC body summarising
  the phase's deliverables.

Exceptions (commit separately, before the phase commit):

- Trivial out-of-scope fixes the agent stumbled onto (typos, broken
  unrelated imports). One small commit, clearly labelled, before the
  main phase commit.
- A docs-only phase that *is* a single commit by nature — no change.

Sessions not labelled autonomous (e.g. interactive tutorials,
exploratory work, debugging sessions) are not bound by this rule;
operator-driven flows commit per the operator's preference.

