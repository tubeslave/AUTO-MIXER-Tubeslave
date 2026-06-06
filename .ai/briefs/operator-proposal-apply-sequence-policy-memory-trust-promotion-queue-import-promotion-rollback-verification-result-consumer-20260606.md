# Operator Proposal Apply Sequence Policy Memory Trust Promotion Queue Import Promotion Rollback Verification Result Consumer

## Context

The queue-import promotion path can now verify apply-audit records before a
rollback-verification request is prepared. The next boundary is consuming the
rollback-verification result before sequence-policy gate evidence can be built.

## Goal

Add a transport-free result consumer that verifies rollback-verification records
against queue-import apply-audit provenance and emits only a sequence-policy
gate request template.

## Constraints

- Do not invoke the sequence-policy gate from the consumer.
- Do not create policy records inside the consumer.
- Require verified rollback records before exposing the gate request.
- Do not mutate queues, proposal state, policy stores, runtime memory, models,
  append-only memory, idempotency, or mixer transport.
- Keep live authority and full autonomous readiness false.

## Test Plan

- Add focused tests for not-ready, rollback-blocked, valid sequence-policy gate
  request emission, forged rollback rejection, forged prior consumer rejection,
  registry, and capability matrix.
- Extend Product Layer flow through rollback verification, verifier, and
  read-only sequence-policy gate.
- Run py_compile, focused pytest, memory lint, diff whitespace checks, and
  promotion-chain pytest.
