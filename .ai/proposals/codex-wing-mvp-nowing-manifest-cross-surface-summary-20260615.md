# Proposal: no-WING Manifest Cross-Surface Summary

## Short thesis

Expose the already-computed manifest cross-surface digest verdicts directly in
no-WING `summary.json` and `summary.md`.

## What I understood

The offline no-WING bundle writes `artifact-manifest.json` and proves that
`summary.json` and `summary.md` carry the expected manifest contract digest.
The Project Director Current First Command also expects operators to see those
cross-surface verdicts in the no-WING summary surface before supervised proof
planning.

## Proposed solution

- Add the missing cross-surface digest verdict fields to the required manifest
  summary surface lists.
- Render the fields in `summary.json` and operator-readable `summary.md`.
- Keep `artifact-manifest.json` as the byte-level verifier.
- Extend focused no-WING tests for summary embedding and missing-line failure.

## Likely files to touch

- `tools/no_wing_release_check.py`
- `tests/test_no_wing_release_check.py`
- `Docs/MVP_WING_SUPERVISED.md`
- `Docs/INDEX.md`
- `.ai/project-layer/wing_mvp_blockers.md`

## Alternatives considered

- Only rely on `artifact-manifest.json`: rejected because the operator summary
  would still hide fields named in the Current First Command.
- Recompute the manifest after writing it and rewrite `summary.json`: rejected
  as unnecessary churn; the manifest remains the byte-level verifier.

## Risks

- The summary surface list can drift if future manifest fields are added.
- A false sense of readiness is possible if the wording is not explicit that
  this remains no-WING/offline.

## Test plan

- Focused no-WING summary tests.
- no-WING quick check.
- Project Director check.
- `git diff --check`.
- Standard pytest if the focused path is clean.

## Where the other agent may disagree

Kimi may ask to make the manifest cross-surface check update `summary.json`
after the manifest is written. That is stricter, but heavier; the current
change closes the operator-visible surface gap while preserving the existing
manifest verifier.
