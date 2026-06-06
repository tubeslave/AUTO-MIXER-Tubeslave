# Operator Proposal Apply Sequence Policy Memory Trust Promotion Queue Import Promotion Policy Consumer Result Consumer

## Context

The queue-import promotion path can verify sequence-policy gate results before
a policy consumer is requested. The next boundary is consuming the policy
consumer result before an ordering preview can be requested.

## Goal

Add a transport-free result consumer that verifies policy-consumer records
against queue-import sequence-policy gate provenance and emits only an
ordering-preview request template.

## Constraints

- Do not invoke ordering preview from the result consumer.
- Do not reorder proposals inside this result consumer.
- Require policy-consumer records before exposing the ordering-preview request.
- Do not mutate queues, proposal state, proposal order, policy stores, runtime
  memory, models, append-only memory, idempotency, or mixer transport.
- Keep live authority and full autonomous readiness false.

## Test Plan

- Add focused tests for not-ready, policy-consumer-blocked, valid
  ordering-preview request emission, forged policy-consumer rejection, forged
  prior consumer rejection, registry, and capability matrix.
- Extend Product Layer flow through policy-consumer result verification and
  read-only ordering preview.
- Run py_compile, focused pytest, memory lint, diff whitespace checks, and
  relevant promotion-chain pytest.
