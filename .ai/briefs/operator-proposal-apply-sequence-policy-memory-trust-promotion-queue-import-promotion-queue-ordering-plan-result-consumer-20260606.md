# Operator Proposal Apply-Sequence Policy Memory Trust Promotion Queue Import Promotion Queue-Ordering-Plan Result Consumer

## Context

The queue-import promotion chain can verify ordering-preview rows and request a
read-only queue-ordering plan. The next safe step is to verify the produced
queue-ordering-plan records before exposing the existing queue importer.

## Goal

Add a transport-free, fail-closed result consumer that validates
queue-ordering-plan records against queue-import ordering-preview provenance and
emits only a queue-importer request template.

## Constraints

- Do not import queues.
- Do not reserve idempotency keys.
- Do not mutate proposals, policy stores, memory, models, or live mixer state.
- Keep `apply=false` and `operator_ack=false` in the emitted template.
- Register as `shadow_only` decision evidence.
- Add Product Layer wiring, focused tests, ADR, and durable memory.

## Validation

- Focused pytest for the new consumer and Product Layer chain.
- Registry and goal-capability matrix checks.
- `python3 tools/memory/memory_lint.py`.
- `PYTHONPATH=backend python3 -m pytest tests/ -x --tb=short -q` before merge
  if feasible.
