# Brief: source-identity-queue-import-handler-20260606

## Context

The source identity queue importer can validate queue-ordering plans and mutate
only an explicit target queue, but there is no Product Layer dispatch action for
operators to request dry-run or acknowledged import.

## Objective

Add a Product Layer handler for source identity queue import that defaults to
dry-run, requires explicit operator acknowledgement for apply, and never writes
to the live console.

## Safety Constraints

- Do not add routes to `server.py`; use existing Product Layer handler
  registration.
- Default handler behavior must be dry-run and non-mutating.
- `apply=true` must still require `operator_ack=true`.
- Queue mutation may target only `server.operator_proposal_queue`.
- Do not call mixer clients or supervised console apply paths.
- Imported queue entries remain proposals; console writes still require the
  existing proposal approval and supervised manual write gate.

## Expected Evidence

- Handler returns importer dry-run payload without queue mutation by default.
- Handler blocks `apply=true` without `operator_ack=true`.
- Handler imports into `operator_proposal_queue` with ack and preserves planned
  order.
- Handler tests prove mixer write functions are not called.
