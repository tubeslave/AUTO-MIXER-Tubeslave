# ADR: execution-audit result consumer for promotion queue import

## Context

The promotion queue-import path can verify execution results and expose a generic execution-audit request template. A passed generic audit is policy-gate input, but the promotion path still needs provenance checks before exposing that policy gate.

## Options considered

1. Feed generic execution-audit results directly into the policy gate.
2. Add a promotion execution-audit result consumer that verifies provenance before exposing an audit-policy-gate request template.
3. Invoke the generic audit-policy gate from inside the consumer.

## Decision

Use option 2. Add a read-only promotion execution-audit result consumer that verifies the prior promotion execution-result consumer, delegates audit validation to the existing generic queue-import execution-audit result consumer, and emits only an audit-policy-gate request template.

## Why this won

It keeps the promotion chain explicit while reusing the authoritative generic execution-audit validation contract. It also keeps policy-gate activation outside the verifier.

## Rejected alternatives

- Direct audit-to-policy-gate was rejected because it bypasses promotion provenance checks.
- Direct policy-gate invocation was rejected because it would combine verification with policy evidence generation.
- Duplicating generic validation was rejected because it increases drift risk for audit safety rules.

## Implementation plan

- Add the promotion execution-audit result consumer module.
- Register it in Product Layer dispatch, processing registry, and capability matrix.
- Add focused tests for missing inputs, passed evidence, blocked rows, forged evidence, registry, and matrix coverage.
- Extend Product Layer dispatch coverage through audit-policy-gate request exposure while preserving no consumer-side live write and no policy mutation.

## Test plan

- `PYTHONPATH=backend python3 -m pytest tests/test_operator_proposal_apply_sequence_policy_memory_trust_promotion_queue_import_promotion_execution_audit_result_consumer.py tests/test_operator_product_state.py::test_sequence_policy_queue_import_promotion_queue_importer_result_consumer_feeds_preflight_plan_without_live_write tests/test_project_goal_capability_matrix.py tests/test_processing_registry.py -q`
- `python3 -m py_compile` for changed backend modules.
- `python3 tools/memory/memory_lint.py`
- `/opt/homebrew/bin/git diff --check`

## Risks and rollback

Risk: the wrapper rejects valid future generic execution-audit schema changes. Rollback is to remove the new consumer and dispatch registration, leaving the promotion execution-result consumer as the final promotion boundary before generic execution audit.
