# ADR: Corrective EQ Protected-band Proposal Layer

Date: 2026-06-15
Status: accepted

## Context

The Automixer goal requires automatic equalization. Direct EQ writes remain
blocked because a wrong band, stale current state, feedback-notch overwrite, or
unsafe boost can harm live sound. Before WING proof, EQ work must remain
offline, dry-run, proposal-only, and blocked from live apply.

## Options Considered

1. Keep EQ as `proposal_only_needed`.
2. Add a dedicated corrective EQ proposal-only contract with band identity and
   protected-band safety.
3. Promote EQ writes behind the current Product Layer.

## Decision

Use option 2. `corrective_eq_proposal` emits operator proposals only when band
identity, frequency match, protected-band policy, bounded gain/headroom, current
EQ freshness, phase-context freshness, and source-identity trust evidence pass.

## Why This Won

It advances autonomous EQ decisions while preserving the supervised WING MVP
rail. The method can explain corrective EQ intent to the operator without
creating live writes, importing into a live queue, or touching mixer clients.

## Rejected Alternatives

- Extending direct AutoEQ apply/reset would blur a quarantined write surface.
- Live EQ execution is premature because physical EQ-band readback, rollback,
  feedback proof, and hardware proof are missing.

## Implementation Plan

- Add `backend/corrective_eq_proposal.py`.
- Add a Product Layer proposal contract for `corrective_eq_proposal`.
- Register the method in processing registry and capability matrix.
- Update the correction backlog from `proposal_only_needed` to `proposal_only`.
- Add tests for transport freedom, protected-band blockers, unsafe boost
  blockers, stale context blockers, proposal payload, registry visibility, and
  capability visibility.

## Test Plan

- `PYTHONPATH=backend python -m py_compile ...`
- `PYTHONPATH=backend python -m pytest -q tests/test_corrective_eq_proposal.py tests/test_auto_eq_fail_closed.py tests/test_project_goal_capability_matrix.py tests/test_processing_registry.py`
- `PYTHONPATH=backend python -m pytest tests/ -x --tb=short -q`
- `PYTHONPATH=backend python tools/no_wing_release_check.py --quick --skip-frontend`

## Risks and Rollback

The stricter band identity and freshness gates may suppress useful proposals
until real readback evidence exists. Roll back the new method registration if
needed, while keeping the AutoEQ direct apply/reset quarantine intact.
