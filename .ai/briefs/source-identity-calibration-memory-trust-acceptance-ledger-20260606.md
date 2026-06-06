# Brief: source-identity-calibration-memory-trust-acceptance-ledger-20260606

## Context

Source identity memory trust-review candidates can now be surfaced when indexed
memory and fresh replay context agree. The next safe step is a read-only ledger
for operator accepted/rejected decisions on those candidates.

## Objective

Add a transport-free source identity memory trust acceptance ledger that records
operator decisions against memory-trust review candidates without promoting
runtime trust or downstream proposal priority.

## Safety Constraints

- Do not mutate source identity models or runtime trust state.
- Do not change downstream proposal priority.
- Do not import accepted records into the decision queue.
- Do not create mixer write requests, `requested_change`, or live apply paths.
- Treat unsafe review envelopes, summary drift, candidate mismatches, invalid
  timestamps, or unsafe acceptance claims as replacement-required.
- Preserve `queue_mutation=false`, `model_mutation=false`,
  `runtime_memory_mutation=false`, `downstream_priority_mutation=false`, and
  `live_write=false`.

## Expected Evidence

- A backend ledger validates memory trust-review candidates and operator
  acceptance records.
- Product Layer preview exposes acceptance counts without queue mutation.
- The processing registry and project goal matrix register the ledger as
  `shadow_only` memory-learning evidence, not a proposal or live executor.
- Focused ledger, Product Layer, registry, and capability tests pass.
