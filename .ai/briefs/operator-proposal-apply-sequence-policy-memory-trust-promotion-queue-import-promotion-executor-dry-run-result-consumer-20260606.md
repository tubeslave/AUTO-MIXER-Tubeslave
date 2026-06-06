# Operator Proposal Apply Sequence Policy Memory Trust Promotion Queue Import Promotion Executor Dry-Run Result Consumer

## Context

The queue-import promotion path can verify accepted promotion records, build a
read-only execution plan, and verify that plan before requesting an executor
dry-run. The next boundary is consuming the dry-run result without turning it
into policy mutation.

## Goal

Add a transport-free result consumer that verifies promotion executor dry-run
records against queue-import execution-plan provenance and emits only a promotion
executor request template.

## Constraints

- Do not invoke the executor.
- Do not apply sequence-policy promotion.
- Keep `apply=False` and `operator_ack=False` in emitted templates.
- Do not mutate queues, proposal state, policy stores, runtime memory, models,
  append-only memory, idempotency, or mixer transport.
- Keep live authority and full autonomous readiness false.

## Test Plan

- Add focused tests for not-ready, valid executor request emission, forged dry-run
  rejection, forged prior consumer rejection, registry, and capability matrix.
- Extend Product Layer flow through the new verifier and executor preview.
- Run py_compile, focused pytest, broader promotion-chain pytest, memory lint,
  diff whitespace checks, and the standard backend suite when practical.
