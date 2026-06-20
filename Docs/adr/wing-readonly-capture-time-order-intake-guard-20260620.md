# WING Read-Only Capture Time Order Intake Guard

## Context

M001 is waiting for a real WING Rack read-only artifact. M102 and M103 require
runner-shaped `run_id` values with valid UTC timestamp prefixes, but accepted
candidate intake also needs to prove a coherent capture timeline. The runner
creates the `run_id` at capture start and writes `captured_at` at summary time,
so `captured_at` can be later but must not be earlier than the `run_id`
timestamp.

## Options Considered

1. Leave capture-time ordering to manual review.
2. Reject candidates whose `captured_at` is earlier than the UTC timestamp in
   `run_id`.
3. Require exact equality between `run_id` timestamp and `captured_at`.

## Decision

Use option 2. Candidate intake now reports
`captured_at_before_run_id_timestamp` when the summary capture timestamp is
earlier than the UTC capture timestamp encoded in `run_id`. File-backed event
logs also report `event_log_timestamp_before_run_id:<index>` when an event is
timestamped before the run id capture time.

## Why This Won

This guards against stale, copied or cross-run evidence that preserves valid
individual timestamps but has an impossible capture order. Exact equality was
rejected because the runner writes `captured_at` at summary completion, after
the run directory and run id already exist.

## Rejected Alternatives

Manual-only review was rejected because timestamp ordering is deterministic and
cheap to validate before a candidate reaches operator decision records. Exact
equality was rejected because real runs can take time and event timestamps can
have millisecond precision while run ids use second precision.

## Implementation Plan

- Parse the timestamp prefix of canonical WING read-only `run_id` values as UTC.
- Reject summaries whose `captured_at` is earlier than that parsed timestamp.
- Reject file-backed event logs whose event timestamps are earlier than that
  parsed timestamp.
- Add focused regression coverage and rail/state/report coverage.

## Test Plan

- Focused WING read-only evidence tests.
- M104 rail.
- Full WING MVP rail.
- JSON/JSONL validation and memory lint before commit.

## Risks and Rollback

The main risk is rejecting hand-edited historical fixtures whose timestamps were
not runner-coherent. That is intended for M001: real hardware proof should
preserve a coherent runner-generated capture timeline. Roll back by reverting
this ADR, the M104 plan/report/state entries, the test and the ordering guard.
No live transport or console state is affected.
