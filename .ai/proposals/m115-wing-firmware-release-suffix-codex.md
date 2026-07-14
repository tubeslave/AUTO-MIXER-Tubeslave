# Codex Proposal — M115 WING Firmware Release Suffix

## 1. Short thesis

Extend only the existing fw3.0.5 suffix grammar with an optional exact
`:release` terminal tag and exercise the real readback string through the
fixed-route proof tests.

## 2. What I understood

The console is reachable and the M114 capture race is closed. HW01 now fails
before signal/headroom collection because the real WING Rack build descriptor
contains a colon that the current regex does not admit. This is a parser defect,
not evidence for accepting another firmware family.

## 3. Proposed solution

Keep the full-match base at `3.0.5`. Preserve the existing optional build
suffix character set, then permit only one optional terminal `:release` tag.
Use the observed `3.0.5-4-ga4ea833d:release` in the route regression test and
add a rejection case for a different base or an arbitrary colon tag.

## 4. Likely files to touch

- `backend/server.py`
- `tests/test_master_capture_route_proof.py`
- M115 Council/ADR/plan documents

## 5. Alternatives considered

Accept every colon in the suffix, parse only the first three numeric fields, or
pin the entire observed string. The first two are unnecessarily broad; the
last would reject another legitimate fw3.0.5 release build.

## 6. Risks

Vendor formatting could later use a different terminal tag. That should remain
a new fail-closed evidence event rather than being guessed now.

## 7. Test plan

Run the master route suite, attended runner suite, combined M110 slice, WING
rails and the full standard suite. Rebind CP05 marker-only, then repeat HW01
from a clean detached checkout.

## 8. Where another agent may disagree

An alternate review may prefer a dedicated parser or pinning the exact build
hash. The decision should favor the smallest grammar that explains current
physical evidence while preserving established accepted forms.

