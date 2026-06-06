# ADR: source-identity-memory-trust-promotion-rollback-verification-20260606

## Context

Applied source identity memory-trust promotion now has rollback audit records,
but those records are not verified. Before applied trust can inform future
downstream proposal priority, rollback evidence must be reviewed as its own
contract.

## Options Considered

1. Treat apply-audit records as sufficient evidence.
2. Mark rollback verified inside the audit module.
3. Add a separate read-only rollback verification gate.

## Decision

Use option 3. Add a non-live verifier that consumes apply-audit records and
explicit operator verification records, then emits verified rollback evidence
for a future downstream-priority gate.

## Why This Won

This keeps apply, audit, rollback verification, and downstream priority as
separate steps. It gives the future priority gate stronger evidence without
mutating trust stores, queues, models, runtime memory, or live mixer state.

## Rejected Alternatives

- Audit-only evidence was rejected because `rollback_verified=false` is not a
  sufficient proof for downstream policy.
- In-audit verification was rejected because it hides a distinct operator review
  decision inside the audit artifact.
- Immediate priority authority was rejected because proposal-priority policy
  still needs a separate reviewed gate.

## Implementation Plan

1. Add the backend rollback verification module.
2. Validate apply-audit envelope, summary, safety, and record fields.
3. Validate operator verification records against apply-audit records.
4. Emit verified records only when all apply-audit records are verified.
5. Keep Product Layer preview non-mutating and expose zero verified records by
   default.
6. Register the method as shadow-only memory-learning infrastructure.
7. Add focused verifier, Product Layer, registry, and capability tests.

## Test Plan

- `PYTHONPATH=backend python3 -m pytest tests/test_source_identity_calibration_memory_trust_promotion_rollback_verification.py tests/test_source_identity_calibration_memory_trust_promotion_apply_audit.py tests/test_operator_product_state.py tests/test_processing_registry.py tests/test_project_goal_capability_matrix.py -q`
- `PYTHONPATH=backend python3 -m py_compile backend/source_identity_calibration_memory_trust_promotion_rollback_verification.py backend/processing_pipeline_preview.py backend/processing_registry.py backend/project_goal_capability_matrix.py`
- `python3 tools/memory/memory_lint.py`
- `/opt/homebrew/bin/git diff --check`
- `PYTHONPATH=backend python3 -m pytest tests/ -x --tb=short -q`

## Risks And Rollback

Risk: verified rollback records could be mistaken for downstream priority
authority. Mitigation: verification records keep direct priority mutation false
and require a separate downstream-priority gate.

Risk: forged verification records could bypass rollback review. Mitigation: the
verifier validates matching apply-audit IDs, source IDs, rollback targets,
timestamps, decisions, and safety flags.

Rollback: remove the verifier call from `processing_pipeline_preview.py`, remove
registry/capability rows, and keep apply-audit records as the terminal rollback
evidence artifact.
