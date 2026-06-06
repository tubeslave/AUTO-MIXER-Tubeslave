# ADR: source-identity-memory-trust-promotion-apply-audit-20260606

## Context

The source identity memory-trust promotion executor can apply accepted dry-run
specs to an explicit trust store under operator acknowledgement. Applied trust
still must not become downstream proposal-priority authority without rollback
evidence and a separate reviewed gate.

## Options Considered

1. Let the executor output be the final evidence artifact.
2. Persist rollback records directly into runtime memory.
3. Add a read-only apply audit that validates executor output and records
   rollback targets.

## Decision

Use option 3. Add a non-live audit layer that consumes executor output, emits
rollback audit records only for valid applied results, and rejects forged
authority or summary drift.

## Why This Won

This makes applied trust-store changes auditable while keeping Product Layer
preview, queues, models, live mixer transport, and downstream proposal priority
unchanged. It also gives future priority policy a concrete evidence contract to
depend on instead of trusting executor output directly.

## Rejected Alternatives

- Executor-only evidence was rejected because it couples apply and audit
  responsibilities too tightly.
- Direct runtime persistence was rejected because persistent trust-store
  ownership is not reviewed yet.
- Immediate priority authority was rejected because rollback evidence must be
  verified before applied trust influences proposal ordering.

## Implementation Plan

1. Add the backend apply audit module.
2. Validate executor envelope, status, summary, safety, apply records, target
   paths, and duplicate IDs.
3. Emit rollback audit records with `rollback_target_trust_state`.
4. Keep Product Layer preview non-mutating and expose zero audit records.
5. Register the method as shadow-only memory-learning infrastructure.
6. Add focused audit, Product Layer, registry, and capability tests.

## Test Plan

- `PYTHONPATH=backend python3 -m pytest tests/test_source_identity_calibration_memory_trust_promotion_apply_audit.py tests/test_source_identity_calibration_memory_trust_promotion_executor.py tests/test_operator_product_state.py tests/test_processing_registry.py tests/test_project_goal_capability_matrix.py -q`
- `PYTHONPATH=backend python3 -m py_compile backend/source_identity_calibration_memory_trust_promotion_apply_audit.py backend/processing_pipeline_preview.py backend/processing_registry.py backend/project_goal_capability_matrix.py`
- `python3 tools/memory/memory_lint.py`
- `/opt/homebrew/bin/git diff --check`
- `PYTHONPATH=backend python3 -m pytest tests/ -x --tb=short -q`

## Risks And Rollback

Risk: audit records could be confused with proposal-priority authority.
Mitigation: audit records and summaries preserve direct downstream-priority
mutation false and live authority false.

Risk: malformed executor output could make invalid applied trust look safe.
Mitigation: the audit validates executor safety, record fields, summary counts,
target paths, and duplicate IDs before emitting records.

Rollback: remove the audit builder call from `processing_pipeline_preview.py`,
remove registry/capability rows, and leave the operator-gated executor as the
terminal trust-promotion artifact.
