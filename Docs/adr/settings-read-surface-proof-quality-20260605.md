# ADR: settings-read-surface-proof-quality-20260605

## Context

No-WING summaries now expose read-surface proof-quality fields, and the Product
Layer normalizes them into `supervised_proof_evidence.summary`. The frontend
adapter carries these fields in module status profiles, but Settings does not
yet summarize them for operators.

## Options considered

1. Add read-surface proof quality rows to the existing Settings Module Status
   inspector.
2. Add a dedicated read-surface proof page.
3. Keep proof quality visible only in module table next-step text.

## Decision

Use option 1. Settings will show metadata availability, physical readback gaps,
fixture test count, and contract/gap gate state as read-only inspector rows.

## Why this won

It makes proof quality browser-visible with a narrow UI diff and preserves the
existing separation between proof source availability and live write authority.

## Rejected alternatives

A dedicated page is premature before physical hardware evidence exists.
Next-step text alone is too easy to miss and does not satisfy operator summary
visibility.

## Implementation plan

1. Parse the proof-quality profile tags already emitted by the adapter.
2. Add Settings inspector rows under Module Status.
3. Extend the backend DTO e2e fixture with read-surface proof-quality metadata.
4. Assert browser-visible Settings rows and retain no-live-write checks.

## Test plan

- `npm --prefix frontend test`
- `npm --prefix frontend run build`
- `npm --prefix frontend run test:e2e -- automixer-prototype.spec.ts`
- `python3 tools/memory/memory_lint.py`
- `PYTHONPATH=backend python3 -m pytest tests/ -x --tb=short -q`

## Risks and rollback

Risk: proof-quality metadata could be mistaken for hardware readiness. The UI
keeps physical gap counts visible and does not add live apply controls.

Rollback is removing the Settings rows, e2e fixture/assertions, and this ADR.
