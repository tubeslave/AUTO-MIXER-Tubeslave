# Operator Proposal Apply-Sequence Policy Memory Trust Promotion Queue Import Promotion Ordering-Preview Result Consumer

## Context

The queue-import promotion chain can verify promotion policy-consumer records
and expose a read-only ordering-preview request. The next safe step is to verify
the produced ordering-preview rows before exposing the already existing
queue-ordering-plan builder.

## Goal

Add a transport-free, fail-closed result consumer that validates ordering-preview
rows against queue-import policy-consumer provenance and emits only a
queue-ordering-plan request template.

## Constraints

- Do not mutate proposals, queues, policy stores, memory, models, or live mixer
  state.
- Do not reserve idempotency keys.
- Do not grant queue import, direct sequence execution, live authority, or Full
  Autonomous readiness.
- Register as `shadow_only` decision evidence.
- Add Product Layer wiring, focused tests, ADR, and durable memory.

## Validation

- Focused pytest for the new consumer and Product Layer chain.
- Registry and goal-capability matrix checks.
- `python3 tools/memory/memory_lint.py`.
- `PYTHONPATH=backend python3 -m pytest tests/ -x --tb=short -q` before merge
  if feasible.
