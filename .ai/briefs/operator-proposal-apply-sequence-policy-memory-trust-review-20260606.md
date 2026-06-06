# Brief: operator-proposal-apply-sequence-policy-memory-trust-review-20260606

## Context

The supervised sequence policy memory writer and index can produce structured
evidence from append-only memory blocks. The next safe learning step is to
surface indexed records for operator policy/trust review without granting
execution authority.

## Objective

Add a read-only trust-review layer that consumes sequence policy memory index
records and emits operator-review candidates only when a fresh read-only review
context is provided.

## Constraints

- Do not mutate queues, policies, models, runtime memory, memory files, mixer
  transport, or console state.
- Reject forged index envelopes, summary count drift, unsafe record-level flags,
  fader values above `0 dBFS`, and unsafe freshness context.
- Register as `shadow_only`; do not add proposal or executor authority.

## Expected Proof

- Focused tests for no records, fresh candidate creation, stale context,
  forged index/record authority, and unsafe freshness.
- Product Layer preview visibility with one candidate from valid memory.
- Registry/capability tests proving no live or executor widening.
- Compile, focused pytest, memory lint, and diff hygiene.
