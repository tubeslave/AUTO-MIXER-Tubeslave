# ADR: Pan Stereo Mono/Phase-safe Proposal Layer

Date: 2026-06-15
Status: accepted

## Context

The Automixer goal requires automatic panning and stereo placement. The current
live WING boundary does not allow pan writes. Pan work therefore has to remain
offline, dry-run, proposal-only, and blocked from live apply until readback,
rollback, emergency-stop, and hardware proof exist.

## Options Considered

1. Keep pan as `proposal_only_needed`.
2. Promote existing `pan_spatial_proposal` to a stricter proposal-only contract.
3. Add live pan execution behind the current Product Layer.

## Decision

Use option 2. `pan_spatial_proposal` now emits Product Layer style operator
proposals only when the move is safe enough for review. Non-center pan moves
require fresh mono/phase evidence. Missing, stale, or unsafe phase evidence
keeps the result `analysis_only`.

## Why This Won

It moves the autonomous-mixing goal forward without bypassing the supervised
WING MVP rail. The pan method now has source-role, stage/stereo context,
mono/phase safety, source-identity trust, current-pan freshness, readback, and
rollback evidence in one proposal contract.

## Rejected Alternatives

- A separate pan adapter would duplicate the existing transport-free analyzer.
- Live pan execution is premature because there is no physical pan readback,
  rollback proof, or operator-approved pan write scope.

## Implementation Plan

- Add mono/phase fields and blockers in `backend/pan_spatial_proposal.py`.
- Add `old_value` and `rollback_value` metadata to operator proposals.
- Update processing contracts, registry, capability matrix, and backlog.
- Cover safe, missing, and unsafe phase cases with tests.

## Test Plan

- `PYTHONPATH=backend python -m py_compile ...`
- `PYTHONPATH=backend python -m pytest -q tests/test_pan_spatial_proposal.py tests/test_project_goal_capability_matrix.py tests/test_processing_registry.py`
- `PYTHONPATH=backend python -m pytest tests/ -x --tb=short -q`

## Risks and Rollback

The stricter gate may suppress useful pan suggestions until phase telemetry is
available. Roll back by removing only the non-center mono/phase requirement,
while keeping the non-live proposal boundary intact.
