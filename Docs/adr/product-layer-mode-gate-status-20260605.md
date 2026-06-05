# ADR: product-layer-mode-gate-status-20260605

## Context

The Product Layer processing snapshot already exposes promotion readiness, write
surface inventory, read surface inventory, and safety coverage. The
`autonomy_mode_transition_gate` is registered as shadow-only, but it was not
included in the snapshot or frontend module status table.

## Options considered

1. Add the mode gate to the Product Layer snapshot and frontend Settings table.
2. Keep mode gate visibility only in proof reports.
3. Reconstruct mode status from promotion gap rows in the frontend.

## Decision

Use option 1. Include `autonomy_mode_transition_gate` in
`build_processing_registry_snapshot()` and map its rows to read-only
`mode_gate:*` rows in the frontend adapter. The Settings Mode Policy summary
should use those rows for Supervised and Full Auto status.

## Why this won

This keeps the canonical backend gate as the source of truth and makes the
four-mode architecture visible in the operator UI without adding any live-write
path.

## Rejected alternatives

Proof-report-only visibility is too far from the daily Settings policy surface.
Frontend reconstruction would duplicate backend mode policy and risk drift.

## Implementation plan

1. Build one `autonomy_promotion_readiness` snapshot in processing registry.
2. Build `autonomy_mode_transition_gate` from that readiness snapshot.
3. Add the gate to the processing snapshot payload.
4. Map gate rows to status-only frontend module rows.
5. Summarize the gate rows in Settings Mode Policy.
6. Extend backend and frontend contract tests.

## Test plan

- `PYTHONPATH=backend python3 -m pytest tests/test_processing_registry.py tests/test_autonomy_mode_transition_gate.py -q`
- `npm --prefix frontend test`
- `cd frontend && npx tsc --noEmit`
- `python3 -m py_compile backend/processing_registry.py backend/autonomy_mode_transition_gate.py`

## Risks and rollback

Risk: displaying Supervised current write kinds could be mistaken for a new UI
apply authority. The rows are status-only, with `canWriteLive=false` and empty
`liveWriteKinds`.

Rollback is removing the snapshot field, frontend mapping, Settings summary,
test assertions, and this ADR.
