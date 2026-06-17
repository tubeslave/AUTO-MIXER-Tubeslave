# ADR: WING MVP Read-Only Event Log Run-Id Binding

## 1. Context

Real WING read-only preflight artifacts consist of `summary.json` plus
`events.jsonl`. The validator already checks request order, allowed request
types, physical readbacks, expected-plan hashes, prestart-summary hashes, and
artifact paths. The event rows did not carry their own run id, which left a
small identity gap if event logs were copied between runs.

## 2. Options considered

- Keep path and SHA256 checks only.
- Add a separate real-run artifact manifest.
- Add row-level `run_id` binding to the event log.

## 3. Decision

Add row-level `run_id` binding to `events.jsonl` and validate it against
`summary.json`.

## 4. Why this won

It is the smallest safe integrity improvement for the first WING blocker. It
does not contact hardware, does not grant authority, and improves future manual
review of real artifacts.

## 5. Rejected alternatives

Path/hash checks alone were rejected because they do not make the event log's
internal identity explicit. A separate manifest was deferred because the current
need is row-level run identity, not a new package format.

## 6. Implementation plan

- Record `run_id` in every event row written by the read-only runner.
- Validate event row `run_id` values against the summary `run_id`.
- Surface `event_log_run_id_matches_summary` in artifact evidence and review
  markdown.
- Add focused positive and negative tests.

## 7. Test plan

Run focused runner/evidence/review tests, WING rail tests, full pytest,
no-WING runtime, `git diff --check`, memory lint, and Project Director.

## 8. Risks and rollback

Older event logs without `run_id` will fail validation. Roll back by removing
the event-row requirement and review verdict. This ADR does not contact WING,
grant live authority, or authorize mixer writes.
