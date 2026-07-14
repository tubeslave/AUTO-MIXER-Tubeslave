# Independent Proposal — M116 M110 HW01 No-Signal Truth Sync

## 1. Short thesis

Perform a docs/state/test-only truth sync: replace the missing-device blocker
with an attempted-HW01 signal blocker, without raising readiness.

## 2. What I understood

The physical WING and 48x48/48 kHz CoreAudio stream were available, route-only
contact occurred and capture was silent. No route authority or artifact was
issued, and reset/correction/write counts remained zero. Serial-bound USB
identity verification was not reached because the signal gate runs first.

## 3. Proposed solution

Make `STATE.json` canonical, append one `supervised_` diagnostic ledger row,
synchronize the active mirrors, regenerate the progress report, and update the
truth rails. Bind the local diagnostic paths and hashes, but keep them distinct
from accepted hardware evidence.

## 4. Likely files to touch

The current state, ledger, operator reports, active milestone/roadmap/plans,
project-layer task/blocker mirrors, progress report, M116 Council artifacts and
`tests/test_wing_mvp_rails.py`. Rebind only the two CP05 markers afterward.

## 5. Alternatives considered

Leaving truth stale, declaring HW01 complete, rewriting historical ledger rows,
or adding a new runner schema here were rejected. The first three are false or
destructive; the last is useful but is a separate product-code milestone.

## 6. Risks

Do not mistake hardware detection for route proof, silence for a routing fault,
CoreAudio identity for serial-bound USB identity, or current attended contact
for a change to historical no-WING flags.

## 7. Test plan

Parse JSON/JSONL; run capture/route/runner tests, WING truth rails, progress
generation, full tests, frontend, no-WING and diff checks. Commit non-marker
truth first, then compute and commit a marker-only CP05 rebind and verify it on
the clean final HEAD.

## 8. Where the other agent may disagree

A richer structured 48-channel failure artifact would improve diagnostics, but
should not expand this truth-only checkpoint or be confused with accepted HW01.
