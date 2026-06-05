# ADR: source-identity-replay-preview-20260606

## Context

Automatic source typing is a canonical capability, but downstream proposal
families still need replay-calibrated trust before source-specific priority can
be used safely. `source_identity_replay_calibration` already summarizes accepted
and rejected outcomes, but the main pipeline preview did not surface those
calibration candidates for operator review.

## Options Considered

1. Keep replay calibration as a standalone helper only.
2. Import replay calibration candidates into the operator proposal queue.
3. Expose replay calibration candidates in `main_pipeline_preview` as
   preview-only operator-review proposals.

## Decision

Use option 3. The main pipeline preview now reads optional source identity
replay events, analyzes them through the existing calibration method, and
surfaces preview-only calibration proposals.

## Why This Won

The preview is the right boundary for evidence that informs future proposals but
must not yet mutate queues, models, memory, downstream priority, or live console
state. Operators can see whether a source type is ready for trust review while
all authority remains blocked.

## Rejected Alternatives

- Standalone-only calibration was rejected because it leaves an important source
  typing decision invisible in the main operator workflow.
- Queue import was rejected because source identity calibration is not a
  supervised fader/gain apply candidate.
- Direct model or memory mutation was rejected because persistence needs a
  separate operator acceptance and audit contract.

## Implementation Plan

1. Add a replay-calibration proposal adapter that forces preview-only behavior.
2. Let the preview builder read optional replay events and cache freshness from
   server attributes or config.
3. Convert valid mapping records into `SourceIdentityReplayEvent` instances.
4. Add preview summary counts, adapter status, and metadata flags for no queue,
   model, memory, downstream-priority, or live mutations.
5. Cover the adapter and Product Layer preview behavior in focused tests.

## Test Plan

- `PYTHONPATH=backend python3 -m pytest tests/test_source_identity_replay_calibration.py tests/test_operator_product_state.py -q`
- `PYTHONPATH=backend python3 -m py_compile backend/processing_pipeline_preview.py backend/processing_proposal_adapters.py backend/source_identity_replay_calibration.py`
- `python3 tools/memory/memory_lint.py`
- `/opt/homebrew/bin/git diff --check`

## Risks And Rollback

Risk: operators may overread trust candidates as already-applied source typing
state. Mitigation: preview evidence forces `queue_mutation=false`,
`model_mutation=false`, `memory_mutation=false`,
`downstream_priority_mutation=false`, and `live_write=false`.

Rollback: remove the source-identity replay adapter from
`processing_pipeline_preview.py` and keep the standalone replay-calibration
module and proposal-only contract unchanged.
