# ADR: source-identity-queue-import-handler-20260606

## Context

The source identity queue importer can produce dry-run import specs and mutate
only a caller-provided operator proposal queue after acknowledgement. Operators
still need a Product Layer dispatch action to request that behavior.

## Options Considered

1. Import source identity queue plans automatically during preview.
2. Add a new server route.
3. Add a Product Layer handler for dry-run and acknowledged import.

## Decision

Use option 3. Add `import_source_identity_queue_ordering_plan` to the Product
Layer handler registry.

## Why This Won

It exposes the new queue-import boundary in the same handler layer as other
operator actions while keeping `server.py` as a coordinator and preserving
preview non-mutation.

## Rejected Alternatives

- Automatic preview import was rejected because preview builders must remain
  read-only.
- A new server route was rejected because Product Layer action handlers already
  own queue operations.
- Chaining directly into console apply was rejected because queue import and
  live apply need separate acknowledgements and safety gates.

## Implementation Plan

1. Add the Product Layer handler.
2. Use explicit plan/payloads when supplied, otherwise derive them from the main
   pipeline preview.
3. Call the queue importer with `apply=false` by default.
4. Pass `server.operator_proposal_queue` only with `apply=true` and
   `operator_ack=true`.
5. Test dry-run, missing ack, and acknowledged import behavior.

## Test Plan

- `PYTHONPATH=backend python3 -m pytest tests/test_operator_product_state.py tests/test_source_identity_proposal_queue_importer.py -q`
- `PYTHONPATH=backend python3 -m py_compile backend/handlers/product_state_handlers.py backend/source_identity_proposal_queue_importer.py`
- `python3 tools/memory/memory_lint.py`
- `/opt/homebrew/bin/git diff --check`
- `PYTHONPATH=backend python3 -m pytest tests/ -x --tb=short -q`

## Risks And Rollback

Risk: queue import may be mistaken for console apply. Mitigation: handler only
imports queue entries; live apply still requires the existing proposal apply
handler, approval, idempotency key, and supervised manual write gate.

Rollback: remove the handler registration and keep the queue importer callable
only from direct backend tests or future reviewed UI work.
