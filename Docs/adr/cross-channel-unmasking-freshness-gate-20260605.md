# ADR: cross-channel-unmasking-freshness-gate-20260605

## Context

The project goal requires automatic cross-channel unmasking and frequency
separation. `cross_adaptive_eq` can compute mirror-EQ cuts and boosts from
band-energy overlap and source priority, but direct EQ writes are not inside the
current proven live executor. Unmasking must therefore advance through
proposal-only contracts with fresh evidence, readback, and rollback.

## Options considered

1. Leave `cross_adaptive_eq` as raw adjustment output only.
2. Add a proposal-only wrapper that gates on provided current-state freshness.
3. Promote mirror EQ directly into live EQ writes.

## Decision

Use option 2. Preserve `calculate_corrections` for compatibility and add
`analyze_unmasking_proposals` as the contract-bearing API. Fresh or legacy
missing freshness can emit linked proposal records. Stale, unknown, or invalid
provided freshness returns no proposals and records a blocker.

## Why this won

This moves cross-channel unmasking toward Assisted Mixing while preserving the
current safety boundary. It creates the proposal data needed for operator review
and future replay/hardware proof without adding any OSC, WING, dLive, or Mixing
Station write path.

## Rejected alternatives

Raw-only output was rejected because it does not satisfy the capability-matrix
next step for linked proposals with rollback evidence. Direct live EQ writes were
rejected because EQ-band readback, sum-bus verification, rollback, and hardware
proof are not complete.

## Implementation plan

- Add optional current-state freshness normalization to `cross_adaptive_eq`.
- Attach masking-pair metadata to existing `MirrorEQAdjustment` objects.
- Add proposal-only action records with old-value source, proposed value,
  rollback source, readback requirement, and safety evidence.
- Gate stale provided freshness to no proposal output.
- Update registry, proposal-only contract, and capability matrix.
- Add focused tests for fresh, stale, compatibility, and contract evidence.

## Test plan

- `PYTHONPATH=backend python -m py_compile backend/cross_adaptive_eq.py backend/processing_registry.py backend/processing_proposal_adapters.py backend/project_goal_capability_matrix.py tests/test_cross_adaptive_eq.py`
- `PYTHONPATH=backend python -m pytest tests/test_cross_adaptive_eq.py tests/test_processing_registry.py tests/test_project_goal_capability_matrix.py -q`
- `PYTHONPATH=backend python -m pytest tests/ -x --tb=short -q`
- `/opt/homebrew/bin/git diff --check`

## Risks and rollback

Risk: callers that pass stale freshness produce no unmasking proposals until they
refresh masking context. Rollback is removing the proposal wrapper, contract
evidence additions, matrix text, and focused tests; the legacy adjustment API is
kept compatible throughout.
