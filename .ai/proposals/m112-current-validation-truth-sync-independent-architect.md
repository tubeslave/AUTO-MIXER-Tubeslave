# Independent Architect Proposal — M112

## 1. Short thesis

Use two commits: a non-marker truth sync tied to validated baseline
`451d0fdf3`, followed by a marker-only CP05 rebind.

## 2. What I understood

M111 and its inventory repair are committed; `7160 passed, 3 skipped`, 150
rails and the 66/66 no-WING capsule are verified. Strict live readiness remains
88, allowlist and writes remain zero, and physical WING USB is absent.

## 3. Proposed solution

Synchronize every canonical state/report surface, append a fail-closed ledger
row, fix the skipped-count regexes, canonicalize only three capsule files and
then rebind CP05 after the non-marker commit. Keep the launch-only proof bound
to `5c054eb9e`.

## 4. Likely files to touch

STATE, ledger, heartbeat, reports, milestone/roadmap/plans, project-layer
truth, rail tests, the three capsule files and finally the CP05 marker pair.

## 5. Alternatives considered

Self-referential commit hashes, post-marker documentation edits, an untracked
canonical capsule and committing all generated capsule files were rejected.

## 6. Risks

Marker invalidation after any later tracked edit, exact-one-skipped parsers,
and accidental conflation of `hardware_enabled` with live authority.

## 7. Test plan

CP05 verifier, 150 rails, focused M111/M006 tests, detached capsule verifier,
full suite, exact Git file inventory, diff check and independent review.

## 8. Where the other agent may disagree

The three capsule files add repository weight, but tracking them is preferable
to a canonical reference to ephemeral evidence.
