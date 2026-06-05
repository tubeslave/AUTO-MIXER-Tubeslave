# ADR: frequency-separation-proposal-20260605

## Context

The project goal requires automatic frequency separation of instruments. Existing
EQ and cross-adaptive layers can identify tonal and masking candidates, but the
system also needs an explicit source/stem frequency-lane decision before safely
deriving concrete EQ, fader, or pan moves.

## Options considered

1. Keep frequency separation represented only by `cross_adaptive_eq`.
2. Add a dedicated proposal-only source/stem frequency-lane analyzer.
3. Promote frequency separation directly into live EQ or fader writes.

## Decision

Use option 2. Add `frequency_separation_proposal` as a transport-free analyzer
that emits freshness-gated, operator-facing frequency-lane recommendations.
Stale provided source/stem spectrum context returns no proposals and records a
blocker. Fresh or legacy missing freshness may produce proposal-only records.

## Why this won

This creates an explainable bridge between arrangement role, inter-channel
analysis, and future EQ/fader moves without expanding live authority. It gives
operator review a clear reason for why one source should yield a frequency lane
to another source or stem.

## Rejected alternatives

Using only `cross_adaptive_eq` was rejected because pairwise mirror EQ does not
fully describe source/stem frequency ownership. Direct live writes were rejected
because fresh spectrum readback, sum-bus verification, rollback, and hardware
proof are not complete.

## Implementation plan

- Add source/stem band-energy input dataclasses.
- Normalize current-state freshness and block stale provided freshness.
- Score lane ownership from arrangement role, source type, priority, confidence,
  and stem contribution.
- Emit bounded subtractive frequency-lane proposals with readback and rollback
  evidence.
- Register the method, proposal-only contract, and capability matrix row.
- Add focused tests for fresh, stale, operator proposal, registry/contract, and
  transport-free behavior.

## Test plan

```bash
PYTHONPATH=backend python -m py_compile \
  backend/frequency_separation_proposal.py \
  backend/processing_registry.py \
  backend/processing_proposal_adapters.py \
  backend/project_goal_capability_matrix.py \
  tests/test_frequency_separation_proposal.py
PYTHONPATH=backend python -m pytest \
  tests/test_frequency_separation_proposal.py \
  tests/test_processing_registry.py \
  tests/test_project_goal_capability_matrix.py -q
PYTHONPATH=backend python -m pytest tests/ -x --tb=short -q
/opt/homebrew/bin/git diff --check
```

## Risks and rollback

Risk: early heuristics may overstate role ownership in dense arrangements until
replay tuning is available. Rollback is removing the analyzer, registry entry,
proposal-only contract, capability-row method id, and focused tests. No live
transport path is added.
