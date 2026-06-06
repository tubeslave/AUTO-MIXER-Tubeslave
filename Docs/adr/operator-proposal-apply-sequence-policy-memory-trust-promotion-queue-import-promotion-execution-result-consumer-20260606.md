# ADR: execution-result consumer for promotion queue import

## Context

The promotion queue-import path can verify execution-gate results and expose a generic sequence-executor request template. If an operator later runs that executor with explicit execute, acknowledgement, and approval, the resulting delegated `apply_operator_proposal` evidence still needs provenance verification before execution audit.

## Options considered

1. Feed generic executor results directly into the audit handler.
2. Add a promotion execution-result consumer that verifies provenance before exposing an audit request template.
3. Invoke the generic execution-audit handler from inside the consumer.

## Decision

Use option 2. Add a read-only promotion execution-result consumer that verifies the prior promotion execution-gate result consumer, delegates executor-result validation to the existing generic queue-import execution-result consumer, and emits only an audit request template.

## Why this won

It keeps the promotion chain explicit while reusing the authoritative generic execution-result validation contract. It also keeps audit activation outside the verifier.

## Rejected alternatives

- Direct executor-to-audit was rejected because it bypasses promotion provenance checks.
- Direct audit invocation was rejected because it would combine verification with audit execution.
- Duplicating generic validation was rejected because it increases drift risk for delegated-apply safety rules.

## Implementation plan

- Add the promotion execution-result consumer module.
- Register it in Product Layer dispatch, processing registry, and capability matrix.
- Add focused tests for missing inputs, applied evidence, blocked rows, forged evidence, registry, and matrix coverage.
- Extend Product Layer dispatch coverage through execution-audit request exposure while preserving no consumer-side live write and no idempotency reservation.

## Test plan

- `PYTHONPATH=backend python3 -m pytest tests/test_operator_proposal_apply_sequence_policy_memory_trust_promotion_queue_import_promotion_execution_result_consumer.py tests/test_operator_product_state.py::test_sequence_policy_queue_import_promotion_queue_importer_result_consumer_feeds_preflight_plan_without_live_write tests/test_project_goal_capability_matrix.py tests/test_processing_registry.py -q`
- `python3 -m py_compile` for changed backend modules.
- `python3 tools/memory/memory_lint.py`
- `/opt/homebrew/bin/git diff --check`

## Risks and rollback

Risk: the wrapper rejects valid future generic execution-result schema changes. Rollback is to remove the new consumer and dispatch registration, leaving the promotion execution-gate result consumer as the final promotion boundary before generic execution.
