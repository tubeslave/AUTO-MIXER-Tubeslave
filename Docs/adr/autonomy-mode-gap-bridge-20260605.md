# ADR: autonomy-mode-gap-bridge-20260605

## Context

The mode transition gate decides which of the four target operating modes is
available, limited, or blocked. It already consumes autonomy readiness, but it
does not expose the new promotion gap summary that explains the remaining
autonomy blockers by category.

## Options considered

1. Add readiness gap fields to each mode gate row.
2. Keep gap visibility only in the readiness snapshot and frontend Settings.
3. Create a separate mode remediation snapshot.

## Decision

Use option 1. Mode gate rows should include the readiness gap counts and target
IDs, and Full Autonomous should list nonzero gap buckets as blockers.

## Why this won

It improves explainability of the four-mode architecture without changing
runtime mode policy or live-write authority.

## Rejected alternatives

Frontend-only visibility leaves backend reports and agents without canonical
mode-level evidence. A separate remediation snapshot would duplicate readiness
parsing before there is a need for a new API surface.

## Implementation plan

1. Read `summary.promotion_gap_summary` from autonomy readiness.
2. Normalize bucket counts and target IDs.
3. Add the normalized data to `AutonomyModeGateRow`.
4. Add `readiness_gap_<bucket>` blockers to Auto mode only.
5. Extend mode gate tests.

## Test plan

- `PYTHONPATH=backend python3 -m pytest tests/test_autonomy_mode_transition_gate.py -q`
- `PYTHONPATH=backend python3 -m pytest tests/test_autonomy_mode_transition_gate.py tests/test_autonomy_promotion_readiness.py tests/test_processing_registry.py -q`

## Risks and rollback

Risk: broad autonomous gaps could make Supervised look unavailable. The row
fields are informational for Supervised; only Auto uses the buckets as blockers.

Rollback is removing the row fields, helper functions, test assertions, and this
ADR.
