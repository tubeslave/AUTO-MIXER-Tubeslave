# ADR: WING Read-Only Preflight Event-Log Crosscheck

## Context

`wing_readonly_preflight_runner.py` records both `summary.json` and
`events.jsonl`. The summary describes the read-only WING preflight result, while
the event log is the operator-visible request/response trace. Evidence intake
previously validated only the summary.

## Options considered

1. Continue trusting summary fields only.
2. Require signed artifact bundles before any readiness review.
3. Validate filesystem summaries against their event logs and fail closed on
   mismatch.

## Decision

Use option 3. Discovered filesystem artifacts require a present, readable event
log whose send request types and readback rows match the summary. Inline
payload validation remains available for focused tests and synthetic snapshots.

## Why this won

The next supervised WING proof depends on trustworthy read-only evidence. A
summary that omits a write-like event, loses the event log, or disagrees with
the recorded readbacks is not clean enough for manual readiness review.

## Rejected alternatives

Summary-only validation is too easy to over-trust. Signed bundles are stronger
but introduce key management and workflow overhead before there is a captured
real preflight artifact to protect.

## Implementation plan

- Resolve `events_path` for filesystem `summary.json` artifacts.
- Load JSONL events defensively.
- Require strict `send`/`recv` pairs for each runner step.
- Require contiguous event `seq` values from `1` through the event count.
- Compare event-log send request types to summary `sent_request_types`.
- Require the read-only runner sequence: `get_operator_mode`, `get_state`, then
  one or more `query_mixer_parameter` requests.
- Reject unexpected or forbidden send request types.
- Compare event-log readback payloads to summary `readbacks`.
- Require each query request to match its readback channel, value type, and
  address.
- Preserve read-only, non-authoritative readiness semantics.

## Test plan

- Valid filesystem artifact with matching event log.
- Missing event log.
- Tampered event log containing `set_gain`.
- Tampered event log with a reordered request sequence.
- Tampered event log with grouped sends/receives instead of stepwise responses.
- Tampered event log with missing or non-contiguous sequence numbers.
- Tampered event log with query/readback identity mismatch.
- Existing runner, readiness, supervised plan, and no-WING release-check tests.

## Risks and rollback

Risk: manually copied summaries without event logs will be rejected. Rollback is
a small revert of event-log validation and tests; no live mixer state is touched.
