# ADR: FX Send Monitor/Return-safe Proposal Layer

Date: 2026-06-15
Status: accepted

## Context

The Automixer goal requires automatic FX management and send tuning. FX/send
writes are not live-approved because they can affect feedback, intelligibility,
monitor mixes, and broad FX rack state. Before WING proof, FX work must remain
offline, dry-run, proposal-only, and blocked from live apply.

## Options Considered

1. Keep FX / Sends as `proposal_only_needed`.
2. Promote the existing `fx_send_proposal` analyzer to a stricter proposal-only
   contract with monitor and return safety.
3. Add live FX/send execution behind the current Product Layer.

## Decision

Use option 2. `fx_send_proposal` now emits Product Layer style proposals only
when feedback, intelligibility, return-level, monitor-safety, freshness, schema,
and source-identity gates pass. Monitor risk or stale monitor context keeps the
result `analysis_only`.

## Why This Won

It advances autonomous FX decisions while preserving the supervised WING MVP
rail. The method can explain safe FX/send intent to the operator without
creating live writes, queue imports, or mixer-client side effects.

## Rejected Alternatives

- A new FX planner would duplicate the existing transport-free proposal layer.
- Live FX/send execution is premature because send/return readback, rollback,
  feedback proof, monitor proof, emergency stop, and hardware proof are missing.

## Implementation Plan

- Add monitor-safety fields and evidence to `backend/fx_send_proposal.py`.
- Add `old_value` and `rollback_value` metadata to operator proposals.
- Update proposal contracts, registry, capability matrix, and backlog.
- Add tests for proposal payload, monitor blocking, and contract visibility.

## Test Plan

- `PYTHONPATH=backend python -m py_compile ...`
- `PYTHONPATH=backend python -m pytest -q tests/test_fx_send_proposal.py tests/test_auto_fx.py tests/test_auto_reverb.py tests/test_project_goal_capability_matrix.py tests/test_processing_registry.py`
- `PYTHONPATH=backend python -m pytest tests/ -x --tb=short -q`
- `PYTHONPATH=backend python tools/no_wing_release_check.py --quick --skip-frontend`

## Risks and Rollback

The stricter monitor guard may suppress useful proposals until monitor context
is present. Roll back only the monitor-specific blockers if needed, while
keeping the non-live proposal boundary intact.
