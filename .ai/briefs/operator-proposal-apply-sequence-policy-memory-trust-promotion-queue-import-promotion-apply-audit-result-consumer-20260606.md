# Operator Proposal Apply Sequence Policy Memory Trust Promotion Queue Import Promotion Apply Audit Result Consumer

## Context

The queue-import promotion path can now verify applied executor results before
emitting a read-only apply-audit request. The next boundary is consuming the
apply-audit result before rollback verification can be requested.

## Goal

Add a transport-free result consumer that verifies promotion apply-audit records
against queue-import executor-result provenance and emits only a rollback
verification request template.

## Constraints

- Do not invoke rollback verification from the consumer.
- Do not verify rollback inside the consumer.
- Require apply-audit records before exposing the rollback-verification request.
- Do not mutate queues, proposal state, policy stores, runtime memory, models,
  append-only memory, idempotency, or mixer transport.
- Keep live authority and full autonomous readiness false.

## Test Plan

- Add focused tests for not-ready, audit-blocked, valid rollback-verification
  request emission, forged audit rejection, forged prior consumer rejection,
  registry, and capability matrix.
- Extend Product Layer flow through applied executor result, read-only apply
  audit, verifier, and rollback-verification not-ready state without operator
  verification records.
- Run py_compile, focused pytest, memory lint, diff whitespace checks, and
  promotion-chain pytest.
