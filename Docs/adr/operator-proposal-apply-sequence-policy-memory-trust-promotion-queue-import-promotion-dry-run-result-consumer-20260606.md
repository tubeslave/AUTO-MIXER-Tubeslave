# ADR: dry-run result consumer for promotion queue import

## Context

The promotion queue-import path can verify sequence-plan results and expose a generic dry-run request template. The generic dry-run handler rehearses planned apply steps but is not an execution gate, executor, idempotency reservation, or live-write authority.

## Options considered

1. Feed generic dry-run results directly into the execution gate.
2. Add a promotion dry-run result consumer that verifies provenance before exposing an execution-gate request template.
3. Invoke the generic execution-gate handler from inside the consumer.

## Decision

Use option 2. Add a read-only promotion dry-run result consumer that verifies the prior promotion sequence-plan result consumer, delegates dry-run validation to the existing generic queue-import dry-run result consumer, and emits only an execution-gate request template.

## Why this won

It keeps the promotion chain explicit while reusing the authoritative generic dry-run validation contract. It also keeps operator acknowledgement outside the verifier: the emitted request remains `operator_ack=false`.

## Rejected alternatives

- Direct dry-run-to-gate was rejected because it bypasses promotion provenance checks.
- Direct gate invocation was rejected because it would combine verification with execution gating.
- Duplicating generic validation was rejected because it increases drift risk for dry-run readiness rules.

## Implementation plan

- Add the promotion dry-run result consumer module.
- Register it in Product Layer dispatch, processing registry, and capability matrix.
- Add focused tests for missing inputs, ready evidence, blocked rows, forged evidence, registry, and matrix coverage.
- Extend Product Layer dispatch coverage through execution-gate request exposure while preserving no live write and no idempotency reservation.

## Test plan

- `PYTHONPATH=backend python3 -m pytest tests/test_operator_proposal_apply_sequence_policy_memory_trust_promotion_queue_import_promotion_dry_run_result_consumer.py tests/test_operator_product_state.py::test_sequence_policy_queue_import_promotion_queue_importer_result_consumer_feeds_preflight_plan_without_live_write tests/test_project_goal_capability_matrix.py tests/test_processing_registry.py -q`
- `python3 -m py_compile` for changed backend modules.
- `python3 tools/memory/memory_lint.py`
- `/opt/homebrew/bin/git diff --check`

## Risks and rollback

Risk: the wrapper rejects valid future generic dry-run schema changes. Rollback is to remove the new consumer and dispatch registration, leaving the promotion sequence-plan result consumer as the final promotion boundary before generic dry-run.
