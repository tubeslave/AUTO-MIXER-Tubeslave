# Proposal: no-WING Manifest Cross-Surface Subcheck Render

## Short thesis

Expose individual manifest cross-surface verdicts in no-WING `summary.json` and
operator-readable `summary.md`, not only the aggregate mismatch list.

## What I understood

The first WING MVP blocker is still real read-only connection proof. Offline
gate artifacts should make drift diagnosable for the operator without implying
readiness or live authority.

## Proposed solution

- Add the individual cross-surface subcheck fields to the manifest summary
  surface required fields and Markdown required lines.
- Render those subchecks in `summary.md`.
- Replace their rendered values after the cross-surface check computes the
  actual verdicts.
- Extend focused tests around summary embedding, missing-line detection, and
  stale-digest failure.

## Likely files to touch

- `tools/no_wing_release_check.py`
- `tests/test_no_wing_release_check.py`
- `Docs/INDEX.md`
- `.ai/project-layer/wing_mvp_blockers.md`

## Alternatives considered

- Keep only `artifact_manifest_contract_cross_surface_mismatches`: rejected
  because the operator can see which check names failed but not the full set of
  expected subcheck verdict rows.
- Defer until a schema renderer exists: rejected because this focused surface
  fix is small and keeps the current gate honest.

## Risks

The surface grows by four fields. That is acceptable because the fields are
already computed and directly explain manifest contract drift.

## Test plan

- Targeted manifest summary and cross-surface tests.
- Focused no-WING, Project Director, and WING MVP rails tests.
- no-WING quick check.
- Standard pytest.

## Where the other agent may disagree

Kimi may prefer replacing the hand-written Markdown summary with a declarative
renderer. That would be useful later, but it is more invasive than this
operator-facing diagnostic fix.
