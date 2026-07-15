# M121 macOS release evidence hardening plan

## Objective

Close two public-release P1 gaps without contacting WING or widening live
authority: remove the false-green npm release surface, provide a fixed-shell
clean-environment launcher, and make recorded app/DMG notarization evidence
immutable and structurally replayable without overstating external authenticity.

## Checkpoints

### CP01 — clean-environment release launcher

- Remove `mac:release` from package scripts because npm `script-shell` can
  return success without running project code.
- Add `/bin/sh -p scripts/run-macos-release`, unset Node startup inputs and use
  `/usr/bin/env -i` before the inner Node verifier starts.
- Reject direct `.mjs` entry without the clean-environment marker.
- Keep `build-macos.mjs` rejecting all npm configuration inputs.
- Replace inherited PATH before the unchanged core builder starts.
- Pass a real hostile-script-shell integration test and direct-launch mutation
  tests.

Canonical invocation from `frontend/`:

```bash
AUTOMIXER_NODE_BIN=/absolute/path/to/node-v22.22.3/bin/node \
  /bin/sh -p scripts/run-macos-release
```
- Commit CP01 independently.

### CP02 — notarization evidence v2

- Disable electron-builder internal notarization.
- Await a repository-owned app notarization hook before ZIP/DMG targets.
- Fetch and retain raw Apple log JSON for every accepted app/DMG submission.
- Bind log UUID/status/statusCode/archive/SHA/time and the app ticket cdhash.
- Require stable contained app/DMG evidence and log files in both verifier
  passes and manifest v2.
- Pass signing, race, symlink, verifier and build-profile contract tests.
- Commit CP02 independently.

### CP03 — immutable cross-domain release evidence

- Add a separate `automixer.public_release_evidence_bundle.v1`; do not overload
  or upgrade the package release manifest.
- Bind one exact clean Git HEAD/tree to retained package evidence, a recomputed
  CP05 non-marker source binding, committed attended runner/backend source, one
  full accepted M110 `execution-summary.json`, its exact route proof and the
  immutable M006 restore contract. Do not claim an arbitrary log as
  proof-owned until the runner emits a durable byte identity for it.
- Require the executing public-release authority closure to match exact HEAD
  blobs and executable modes at construction, pre-publication and post-replay
  cuts. Execute spawned evidence helpers from pinned selected Git-blob bytes,
  never mutable helper pathnames. Reject hidden Git
  index flags and retain a canonical commit/tree/blob membership proof for the
  authority plus the five detached source inputs.
- Run every Git inspection with a fixed Git-only environment, disabled
  system/global config, disabled fsmonitor/hooks/lazy-fetch/replace objects,
  an explicit selected worktree, and a fail-closed local-config allowlist that
  rejects include/attribute/filter/diff/pager/hook command surfaces; the public launcher
  must replace inherited `HOME`, temp and locale values rather than merely
  allowlisting their names.
- Retain proposal-review digests, exact operation counts/scopes/IDs and durable
  proof/commit terminals in the M110 receipt so detached validation can reject
  skeletal reset/M006 claims. Recompute the canonical reset and all three M006
  operation digests from the fixed contract and exact route/runtime safety
  binding; proof/commit SHA shape alone is not semantic evidence.
- Copy mutable/private M110 inputs into a fresh `0700` payload, normalize every
  retained file to `0600`, and validate fixed paths/modes before parsing.
  Bind raw input bytes through `O_NOFOLLOW` descriptors, then require the stable
  JSON read and private copy to match that exact size/SHA. Hold the selected
  payload root and hash its exact flat file set from a helper cwd pinned to the
  selected root inode, so replacing an ancestor pathname cannot redirect the
  canonical walk.
  Require both raw input files and every copied private payload inode to remain
  single-link through final replay, so a hardlink cannot publish private bytes.
  Require both raw input files to live outside the distributable release root.
  Publish only a sanitized manifest; classify retained raw bytes as private
  audit evidence because they can contain LAN addresses, channel state,
  absolute historical paths and request nonces.
- Require the central M110 summary to be the runner's exact sorted compact
  Python JSON form with one newline, unique keys, canonical numbers and no
  capability material. Validate the fixed `+3 dB` true-peak allowance and
  whole-second summary/fractional acknowledgement boundary. Require the route
  artifact's exact Python canonical form without a final newline.
- Detached verification requires exact single serialized byte forms for the
  retained release manifest and public projection, then replays retained
  byte/semantic bindings without launching the app/backend or claiming a new
  Gatekeeper/native-trust result.
- Reconstruct the complete sanitized projection from replayed evidence, verify
  the candidate before publication, require the public manifest's exact single
  serialized byte form (no duplicate keys), and remove unchanged published
  bytes if a final post-publication check fails.
- Re-run retained-package verification at standalone and builder return cuts,
  then recheck raw-input closure, private payload and public-manifest identities
  immediately before returning a structural verdict. Bind build-profile
  semantics to the exact size/SHA identity returned by its stable reader.
- Reject route-only/HW01 summaries, source mismatches, stale CP05, incomplete
  M006/lanes/readiness, missing SHA-bound listening, or incomplete teardown.
- Exercise the real retained-v2 verifier success path, not only a package stub;
  bind `buildProfile.size` in the smoke artifact manifest and prove that a later
  ZIP mutation invalidates production public verification.
- Treat a fully synthetic passing fixture as proof of structural validation,
  not authenticity. Emit `structural_consistency_verified`, require an external
  authenticated attestation, and keep `readinessClosureAllowed=false` until a
  pinned release signature/transparency identity is designed and verified.
- Require runner M006 scope validation to use the server's exact prepared-write
  order, not sorted metadata keys, with golden hashes for all three stages.
- Do not issue readiness-closing evidence during offline tooling work. A future
  authenticated attestation must bind a notarized retained package and complete
  M110 HW02-HW05 evidence from the same final HEAD/tree after CP05 rebind.

### CP04 — validation and live handoff

- Run focused frontend tests, including direct-entry `NODE_OPTIONS`, CLI preload
  arguments, mutable helper-runtime identity, and `GIT_*` environment poisoning
  rejection, hostile inherited-HOME Git config isolation, reordered-manifest
  rejection, repository-local fsmonitor/attribute/filter command rejection,
  mode-only/case-only Git drift, raw-input/private-payload hardlink containment,
  descriptor-bound raw bytes, pinned-root payload/tree walks and build-profile
  semantic-read identity, then full frontend
  tests/build and `git diff --check`.
- Run broader no-WING/full-suite validation as resources allow.
- Release the writer lease.
- Create exactly one marker-only CP05 rebind for the final non-marker tree.
- Only then resume M005/M006 promotion and attended M110 HW02-HW05.

## Safety boundary

M121 tooling does not contact backend, CoreAudio, OSC or WING. It grants no live write,
normal allowlist, readiness closure or strict-score increase. Existing real
HW01 evidence remains historical and source-bound; the final hardware run must
bind to the post-M121 CP05 marker. The public manifest is structural-only,
requires external authenticated attestation, and cannot create readiness or
authority.

## Rollback

Revert CP02 without reusing any partial notary run directory, then revert CP01.
Never relabel a v1 or partial artifact as v2 and never rebind CP05 until the
selected source tree is committed and validated.
