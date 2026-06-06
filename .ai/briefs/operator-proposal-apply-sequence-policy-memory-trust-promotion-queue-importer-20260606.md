# Brief: Operator Proposal Apply Sequence Policy Memory Trust Promotion Queue Importer

## Context

Sequence-policy queue-ordering plan records can now describe a future operator
queue order from trusted sequence-memory evidence. The next boundary is an
operator-gated importer that can preview imports safely and, only with explicit
acknowledgement, place those proposals into the operator proposal queue.

## Goal

Add a sequence-policy queue importer that consumes queue-ordering plan records
and proposal payloads. Product Layer preview must remain dry-run only. A caller
may import into an explicit target queue only with `apply=true` and
`operator_ack=true`.

## Constraints

- No live OSC/WING writes.
- No mixer client calls, supervised console apply, direct sequence execution,
  policy-store mutation, runtime-memory mutation, model mutation, or proposal
  state mutation.
- Product Layer preview must not pass a target queue and must use dry-run.
- Queue mutation is allowed only for an explicit caller-provided operator queue
  with operator acknowledgement.
- Imported records must still require the normal proposal apply preflight before
  any live write.
- Direct sequence execution still requires a separate sequence consumer.

## Validation

- Focused pytest for queue importer, Product Layer preview, handler, decision
  queue visibility, registry, and capability matrix.
- `python3 tools/memory/memory_lint.py`.
