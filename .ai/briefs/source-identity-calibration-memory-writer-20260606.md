# Brief: source-identity-calibration-memory-writer-20260606

## Context

Source identity calibration memory drafts now provide append-only record text,
but there is still no audited contract that can apply those drafts. The next
safe step is an append-only writer that stays dry-run in preview and only writes
when explicitly invoked with operator acknowledgement.

## Objective

Add a transport-free source identity calibration memory writer contract that
validates memory draft records and can append them to project memory only when
`apply=true` and `operator_ack=true`.

## Safety Constraints

- Product Layer preview must call the writer as dry-run only.
- Do not mutate source identity models or downstream proposal priority.
- Do not import writer results into the decision queue.
- Do not create mixer write requests, `requested_change`, or live apply paths.
- Treat unsafe draft envelope, summary, or record claims as invalid.
- Preserve `queue_mutation=false`, `model_mutation=false`,
  `runtime_memory_mutation=false`, `downstream_priority_mutation=false`, and
  `live_write=false`.

## Expected Evidence

- A backend writer validates draft identity, status, timestamps, safety flags,
  summary counts, summary identity lists, target memory path, and draft-record
  false-authority flags.
- Dry-run writer status appears in `main_pipeline_preview` without writing
  memory.
- Apply mode writes only to a supplied root path when `operator_ack=true`.
- The processing registry and project goal matrix register the writer as
  `shadow_only` memory-learning evidence, not a live executor.
- Focused writer, Product Layer, registry, and capability tests pass.
