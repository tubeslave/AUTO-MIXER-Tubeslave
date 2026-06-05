# Brief: source-identity-replay-preview-20260606

## Context

Source identity replay calibration can recommend whether a source type should be
trusted for downstream proposal priority, held for review, or blocked. That
evidence is central to automatic source typing, arrangement-aware mixing, EQ,
dynamics, FX, and adaptation decisions, but it must stay operator-reviewed
before it influences downstream priority.

## Objective

Expose replay-calibrated source identity trust candidates in the main pipeline
preview as review-only proposals when replay events are available.

## Safety Constraints

- Do not mutate source-identity models or long-term memory.
- Do not import calibration candidates into the operator proposal queue.
- Do not change downstream proposal priority automatically.
- Do not create mixer write requests, `requested_change`, or live apply paths.
- Preserve `queue_mutation=false`, `model_mutation=false`,
  `memory_mutation=false`, `downstream_priority_mutation=false`, and
  `live_write=false`.

## Expected Evidence

- `main_pipeline_preview` includes source-identity replay calibration proposals
  only when replay events exist.
- The proposals are preview-only, not queueable, and not live-applicable.
- Missing replay events keep the adapter in standby and preserve existing
  preview behavior.
- Focused source-identity and Product Layer tests pass.
