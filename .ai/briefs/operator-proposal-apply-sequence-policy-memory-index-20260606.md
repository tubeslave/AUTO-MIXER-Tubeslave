# Brief: operator-proposal-apply-sequence-policy-memory-index-20260606

## Context

The operator proposal apply-sequence chain can now execute gated supervised
fader/gain rows, audit the execution, draft policy memory, and append that
memory only with explicit operator acknowledgement.

## Objective

Add the next learning step: a read-only index of appended supervised sequence
policy memory blocks. The index should expose durable evidence for review while
preserving the current live safety boundary.

## Constraints

- Read only from explicit text or `.ai/memory/project.md`.
- Do not write memory, queues, policy stores, models, runtime memory, or console
  state.
- Reject malformed records, unsafe runtime-authority claims, unverified
  readback/proof, and fader values above `0 dBFS`.
- Register as `shadow_only` memory-learning evidence, not a proposal or
  executor.

## Expected Proof

- Focused parser tests for valid, missing, malformed, and bad-target inputs.
- Product Layer preview visibility test.
- Registry and capability matrix tests proving no live/executor widening.
- `py_compile`, focused pytest, memory lint, and diff hygiene.
