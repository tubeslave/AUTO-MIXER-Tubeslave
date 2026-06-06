# ADR: execution-gate result consumer for promotion queue import

## Context

The promotion queue-import path can verify dry-run results and expose a generic execution-gate request template. The generic gate can produce executor-ready rows, but it is not an executor, idempotency reservation, queue mutation, or live-write authority.

## Options considered

1. Feed generic execution-gate results directly into the executor.
2. Add a promotion execution-gate result consumer that verifies provenance before exposing an executor request template.
3. Invoke the generic sequence executor from inside the consumer.

## Decision

Use option 2. Add a read-only promotion execution-gate result consumer that verifies the prior promotion dry-run result consumer, delegates gate validation to the existing generic queue-import execution-gate result consumer, and emits only an executor request template.

## Why this won

It keeps the promotion chain explicit while reusing the authoritative generic execution-gate validation contract. It also keeps executor activation outside the verifier: the emitted request remains `execute=false`, `operator_ack=false`, and `approved=false`.

## Rejected alternatives

- Direct gate-to-executor was rejected because it bypasses promotion provenance checks.
- Direct executor invocation was rejected because it would combine verification with execution.
- Duplicating generic validation was rejected because it increases drift risk for gate readiness rules.

## Implementation plan

- Add the promotion execution-gate result consumer module.
- Register it in Product Layer dispatch, processing registry, and capability matrix.
- Add focused tests for missing inputs, ready evidence, blocked rows, forged evidence, registry, and matrix coverage.
- Extend Product Layer dispatch coverage through executor request exposure while preserving no live write and no idempotency reservation.

## Test plan

- `PYTHONPATH=backend python3 -m pytest tests/test_operator_proposal_apply_sequence_policy_memory_trust_promotion_queue_import_promotion_execution_gate_result_consumer.py tests/test_operator_product_state.py::test_sequence_policy_queue_import_promotion_queue_importer_result_consumer_feeds_preflight_plan_without_live_write tests/test_project_goal_capability_matrix.py tests/test_processing_registry.py -q`
- `python3 -m py_compile` for changed backend modules.
- `python3 tools/memory/memory_lint.py`
- `/opt/homebrew/bin/git diff --check`
- `PYTHONPATH=backend python3 -m pytest tests/ -x --tb=short -q`

## Risks and rollback

Risk: the wrapper rejects valid future generic execution-gate schema changes. Rollback is to remove the new consumer and dispatch registration, leaving the promotion dry-run result consumer as the final promotion boundary before generic execution gating.
