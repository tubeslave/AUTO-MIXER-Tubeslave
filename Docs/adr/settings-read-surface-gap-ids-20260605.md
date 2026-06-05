# ADR: settings-read-surface-gap-ids-20260605

## Context

Product Layer proof quality includes `physical_fixture_missing_surface_ids`.
The adapter and Settings currently show only the aggregate physical gap count,
which makes it harder to choose the next hardware proof target.

## Options considered

1. Add compact gap IDs to the existing supervised proof evidence row and
   Settings inspector.
2. Add a new read-surface remediation page.
3. Leave the IDs in backend artifacts only.

## Decision

Use option 1. The adapter will preserve the gap IDs in read-only profile and
next-step text, while Settings will show the first missing surface plus the
remaining count.

## Why this won

It improves operator explainability and next-proof selection with minimal UI
surface area and no live mixer access.

## Rejected alternatives

A new remediation page is larger than needed for this evidence slice. Backend
artifact-only visibility is not enough because the operator cockpit should show
why autonomous read readiness remains blocked.

## Implementation plan

1. Read `physical_fixture_missing_surface_ids` in the frontend adapter.
2. Add a compact `read_surface_gap_ids` profile and next-step text.
3. Render a compact Settings inspector row.
4. Extend adapter contract and browser e2e fixtures/assertions.

## Test plan

- `node frontend/src/services/__tests__/operatorAdaptersContract.test.mjs`
- `npm --prefix frontend test`
- `npm --prefix frontend run build`
- `npm --prefix frontend run test:e2e -- automixer-prototype.spec.ts`
- `python3 tools/memory/memory_lint.py`
- `PYTHONPATH=backend python3 -m pytest tests/ -x --tb=short -q`

## Risks and rollback

Risk: gap labels could be mistaken for current live read authority. The UI keeps
them under proof evidence, marks physical gaps separately, and does not add
controls.

Rollback is removing the adapter profile, Settings row, tests, and this ADR.
