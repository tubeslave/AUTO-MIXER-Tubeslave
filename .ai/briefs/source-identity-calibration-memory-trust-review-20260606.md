# Brief: source-identity-calibration-memory-trust-review-20260606

## Context

The source identity calibration memory index can expose appended memory records
as read-only evidence. The next safe step is an operator-review boundary that
checks indexed memory against fresh replay/source context before any future
downstream trust promotion is considered.

## Objective

Add a transport-free source identity calibration memory trust-review contract
that turns valid indexed memory records into operator-review candidates only
when the current source identity replay context is fresh.

## Safety Constraints

- Do not mutate source identity models or runtime trust state.
- Do not change downstream proposal priority.
- Do not import review candidates into the decision queue.
- Do not create mixer write requests, `requested_change`, or live apply paths.
- Treat unsafe index envelope, summary, record flags, or stale replay context as
  not ready or invalid.
- Preserve `queue_mutation=false`, `model_mutation=false`,
  `runtime_memory_mutation=false`, `downstream_priority_mutation=false`, and
  `live_write=false`.

## Expected Evidence

- A backend trust-review module validates memory index envelopes, summaries,
  records, and fresh replay context.
- Product Layer preview exposes review status and candidate counts without
  queue mutation.
- The processing registry and project goal matrix register the review layer as
  `shadow_only` memory-learning evidence, not a proposal or live executor.
- Focused trust-review, Product Layer, registry, and capability tests pass.
