# Audio Workbench: current-render identity verification

## Context
The Premiera preflight reproduced an approval bug: a file replaced at the same path could retain the previous render's successful checks. A prior CI run also stopped at the existing optimizer test because the CI environment did not install Optuna.

## Options considered
1. Trust stored paths, sizes and timestamps.
2. Recompute the content identity before recording evidence or checking readiness.
3. Require a new content-addressed immutable store immediately.

## Decision
Use option 2 for the current MVP. A missing, unreadable or changed file invalidates its checks and cannot finalize. Recording evidence for that old identity raises an error. Direct accepted-decision logging must pass coverage as well. Restoring bytes after invalidation does not restore approvals automatically.

## Why this won
This closes the reproduced stale-file defect without a storage migration or new production dependencies. Size/mtime-only verification is insufficient when files are replaced in place.

## Rejected alternatives
A full immutable-store migration is deferred. Checksums do not certify artistic quality, so existing evidence must not be advertised as independent musical approval.

## Implementation plan
Update core.py, add current-file regressions, use scipy.integrate.trapezoid for compatibility with the already supported NumPy range. Install the already declared Optuna integration in CI for its existing tests, not in the live backend requirements. Include FFmpeg and Workbench compilation in that workflow.

## Test plan
Local verification used the three modules from base c2edcd948bca38b20df1ec07f5bf1a20946358a2, with only core.py modified. Nine behavioural audit tests and eight focused identity/compatibility cases passed. Scenarios include deleted/corrupt audio, same-size replacement with preserved mtime, restored bytes, refused evidence writes and refused accepted-decision writes. Full repository CI must be checked separately.

## Risks and rollback
Hashing reads the file on each readiness check. This is an offline MVP, not an audio-thread operation. This patch does not make render files immutable against concurrent modification after validation, does not calibrate observers and does not certify arbitrary supplied evidence. Revert this commit to roll back; source audio has not been modified.
