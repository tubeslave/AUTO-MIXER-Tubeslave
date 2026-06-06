# ADR: source-identity-memory-trust-promotion-priority-gate-20260606

## Context

Source identity memory-trust promotion now has an operator-gated executor,
apply audit, and rollback verification. Verified rollback records are strong
evidence, but they are not yet a proposal-priority policy decision.

## Options Considered

1. Let rollback verification directly affect downstream proposal priority.
2. Treat all verified rollback records as priority-policy inputs without another
   source identity gate check.
3. Add a separate read-only downstream-priority eligibility gate.

## Decision

Use option 3. Add a non-live priority gate that consumes verified rollback
records, reuses the shared source identity downstream gate, and emits read-only
eligibility records for a future proposal-priority consumer.

## Why This Won

It keeps apply, audit, rollback verification, eligibility, and policy consumption
as separate contracts. The system moves closer to autonomous learning while
preserving explicit proof boundaries before behavior changes.

## Rejected Alternatives

- Direct priority mutation was rejected because proposal ordering is runtime
  behavior and needs its own policy consumer, audit, and rollback.
- Skipping the shared downstream gate was rejected because verified rollback
  evidence can still contain held, blocked, unknown, or stale trust states.
- Queue import was rejected because eligibility evidence is not an operator task.

## Implementation Plan

1. Add the backend priority-gate module.
2. Validate rollback-verification envelope, summary, records, duplicate IDs, and
   safety flags.
3. Evaluate verified records with the shared source identity downstream gate.
4. Emit eligible records only for trusted downstream-gate decisions.
5. Keep Product Layer preview non-mutating and expose zero records by default.
6. Register the method as shadow-only memory-learning infrastructure.
7. Add focused gate, Product Layer, registry, and capability tests.

## Test Plan

- `PYTHONPATH=backend python3 -m pytest tests/test_source_identity_calibration_memory_trust_promotion_priority_gate.py tests/test_source_identity_calibration_memory_trust_promotion_rollback_verification.py tests/test_operator_product_state.py tests/test_processing_registry.py tests/test_project_goal_capability_matrix.py -q`
- `PYTHONPATH=backend python3 -m py_compile backend/source_identity_calibration_memory_trust_promotion_priority_gate.py backend/processing_pipeline_preview.py backend/processing_registry.py backend/project_goal_capability_matrix.py`
- `python3 tools/memory/memory_lint.py`
- `/opt/homebrew/bin/git diff --check`
- `PYTHONPATH=backend python3 -m pytest tests/ -x --tb=short -q`

## Risks And Rollback

Risk: eligible records could be mistaken for actual proposal-priority mutation.
Mitigation: records keep `direct_downstream_priority_mutation=false`,
`downstream_priority_change_allowed_by_gate=false`, and require a separate
proposal-priority consumer.

Risk: forged rollback verification could bypass safety. Mitigation: the gate
validates envelope, summary counts, IDs, timestamps, source identity fields, and
record-level authority flags before emitting records.

Rollback: remove the priority-gate call from `processing_pipeline_preview.py`,
remove registry/capability rows, and keep rollback verification as the terminal
source identity promotion evidence artifact.
