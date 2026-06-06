# ADR: source-identity-memory-trust-promotion-executor-20260606

## Context

The source identity memory-trust promotion chain now reaches deterministic
dry-run apply specs. To move toward real long-term learning, the system needs a
controlled executor that proves apply semantics without widening live mixer
authority.

## Options Considered

1. Keep dry-run specs only.
2. Persist trust promotion directly into runtime configuration.
3. Add an operator-gated executor for an explicit trust store.

## Decision

Use option 3. Add a non-live executor that validates dry-run specs and applies
them only to a caller-provided trust store when `apply=true` and
`operator_ack=true`.

## Why This Won

This proves the apply path while keeping Product Layer preview, live mixer
transport, queues, and classifier models untouched. It also keeps persistent
runtime trust storage and downstream priority policy as separate, auditable
steps.

## Rejected Alternatives

- Dry-run-only was rejected because it leaves apply behavior untested.
- Direct runtime persistence was rejected because rollback audit and persistence
  ownership are not reviewed yet.
- Product Layer queue execution was rejected because memory trust promotion is
  not a mixer operation.

## Implementation Plan

1. Add a backend source identity promotion executor.
2. Validate dry-run envelope, status, summary counts, records, target paths, and
   idempotency keys.
3. Require operator acknowledgement and an explicit target store before apply.
4. Keep Product Layer preview non-mutating.
5. Register the executor as non-live memory-learning infrastructure.
6. Add focused executor, Product Layer, registry, and capability tests.

## Test Plan

- `PYTHONPATH=backend python3 -m pytest tests/test_source_identity_calibration_memory_trust_promotion_executor.py tests/test_source_identity_calibration_memory_trust_promotion_executor_dry_run.py tests/test_operator_product_state.py tests/test_processing_registry.py tests/test_project_goal_capability_matrix.py -q`
- `PYTHONPATH=backend python3 -m py_compile backend/source_identity_calibration_memory_trust_promotion_executor.py backend/source_identity_calibration_memory_trust_promotion_executor_dry_run.py backend/processing_pipeline_preview.py backend/processing_registry.py backend/project_goal_capability_matrix.py`
- `python3 tools/memory/memory_lint.py`
- `/opt/homebrew/bin/git diff --check`
- `PYTHONPATH=backend python3 -m pytest tests/ -x --tb=short -q`

## Risks And Rollback

Risk: executor apply could be confused with live console control. Mitigation:
the executor never imports mixer clients, never mutates queues, never writes
models, and has `can_write_live=false`.

Risk: applied trust could influence downstream proposals later. Mitigation:
direct downstream-priority mutation remains false; using applied trust for
priority requires a separate persistent trust-store and rollback audit.

Rollback: remove the executor call from `processing_pipeline_preview.py`, remove
registry/capability rows, and keep executor dry-run specs as the terminal
evidence artifact.
