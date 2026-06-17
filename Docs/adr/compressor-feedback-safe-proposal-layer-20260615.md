# ADR: Compressor Feedback-safe Proposal Layer

Date: 2026-06-15
Status: accepted

## Context

The Automixer goal requires automatic compression and console compressor
control. The current live executor remains supervised fader/gain only.
Compressor proposals already use dynamics, gain-reduction, headroom, freshness,
and source-identity evidence, but the correction backlog still required
feedback safety before compression could be considered a real proposal-only
method family.

## Options Considered

1. Keep compressor marked `proposal_only_needed`.
2. Add feedback-safety gating to the existing compressor proposal layer.
3. Promote compressor writes behind the current Product Layer.

## Decision

Use option 2. `compressor_dynamics_proposal` now treats high feedback risk as a
blocker when the recommendation would add makeup gain, and it exposes explicit
feedback-safety, old-value, and rollback-value evidence.

## Why This Won

It advances automatic compression while preserving the supervised WING MVP rail.
The method can explain compressor intent to the operator without creating live
writes, importing into a live queue, or touching mixer clients.

## Rejected Alternatives

- Updating documentation only would overstate the compressor safety contract.
- Live compressor execution is premature because dynamics readback, rollback,
  feedback proof, emergency stop behavior, and hardware proof are missing.

## Implementation Plan

- Add feedback-risk input and feedback-safety evidence.
- Block high-feedback-risk makeup proposals.
- Add old-value and rollback-value readback sources to operator proposals.
- Update Product Layer contracts, registry, capability matrix, and backlog.
- Add tests for feedback blocking and proposal payload evidence.

## Test Plan

- `PYTHONPATH=backend python -m py_compile ...`
- `PYTHONPATH=backend python -m pytest -q tests/test_compressor_dynamics_proposal.py tests/test_auto_compressor_fail_closed.py tests/test_compressor.py tests/test_project_goal_capability_matrix.py tests/test_processing_registry.py`
- `PYTHONPATH=backend python -m pytest tests/ -x --tb=short -q`
- `PYTHONPATH=backend python tools/no_wing_release_check.py --quick --skip-frontend`

## Risks and Rollback

The feedback guard may suppress useful compressor proposals for open-mic or
feedback-prone sources until better feedback evidence exists. Roll back the
feedback-specific blocker if needed, while keeping compressor writes outside
the live executor.
