# M121 macOS release evidence hardening plan

## Objective

Close two public-release P1 gaps without contacting WING or widening live
authority: remove the false-green npm release surface, provide a fixed-shell
clean-environment launcher, and make successful app/DMG notarization immutable and
independently verifiable.

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

### CP03 — validation and live handoff

- Run focused frontend tests, full frontend tests/build and `git diff --check`.
- Run broader no-WING/full-suite validation as resources allow.
- Release the writer lease.
- Create exactly one marker-only CP05 rebind for the final non-marker tree.
- Only then resume M005/M006 promotion and attended M110 HW02-HW05.

## Safety boundary

M121 does not contact backend, CoreAudio, OSC or WING. It grants no live write,
normal allowlist, readiness closure or strict-score increase. Existing real
HW01 evidence remains historical and source-bound; the final hardware run must
bind to the post-M121 CP05 marker.

## Rollback

Revert CP02 without reusing any partial notary run directory, then revert CP01.
Never relabel a v1 or partial artifact as v2 and never rebind CP05 until the
selected source tree is committed and validated.
