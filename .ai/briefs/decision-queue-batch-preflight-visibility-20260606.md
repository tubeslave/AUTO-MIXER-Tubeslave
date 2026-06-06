# Brief: decision-queue-batch-preflight-visibility-20260606

## Context

The Product Layer now has `preflight_operator_proposal_apply_batch`, an
explicit dry-run handler for pending apply candidates. The decision queue
snapshot should advertise this batch preflight path without running it.

## Objective

Expose snapshot-only batch-preflight visibility in `build_decision_queue` so
operator UI can see which pending proposals the batch dry-run would select by
default and which analysis-only proposals it would skip.

## Safety Constraints

- Do not call `preflight_operator_proposal_apply_batch` from the queue snapshot.
- Do not reserve idempotency keys.
- Do not mutate proposal state, queue order, cumulative delta history, or
  source-identity state.
- Do not call `_apply_manual_console_write` or mixer clients.
- Keep live apply behind the existing `apply_operator_proposal` supervised gate.

## Expected Evidence

- `decision_queue.summary.batch_preflight` names the batch handler.
- The summary lists default candidate IDs and skipped analysis-only IDs.
- The summary states `runs_preflight=false`, `live_write=false`, and
  `queue_mutation=false`.
- Focused tests prove the snapshot remains read-only.
