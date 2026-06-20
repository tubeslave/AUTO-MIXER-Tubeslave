# WING Read-Only Run ID Timestamp Intake Guard

## Context

M001 is waiting for a real WING Rack read-only artifact. M102 required the
runner-generated `YYYYMMDDTHHMMSSZ_wing_readonly_preflight` run-id format, but
that regex still accepted impossible timestamps such as
`20261340T250000Z_wing_readonly_preflight` or
`99999999T999999Z_wing_readonly_preflight`. The real runner emits timestamps
from UTC datetime formatting, so impossible dates or times should not pass the
first accepted-candidate boundary.

## Options Considered

1. Leave timestamp validity to manual review.
2. Reject candidates whose run-id timestamp cannot be parsed as UTC.
3. Accept any matching digit pattern and rely on summary/events consistency.

## Decision

Use option 2. Candidate intake now reports
`run_id_not_valid_utc_timestamp` when the timestamp prefix in `run_id` cannot be
parsed as `%Y%m%dT%H%M%SZ`.

## Why This Won

The accepted-candidate boundary should distinguish a runner-produced capture id
from a forged canonical-looking string. Semantic timestamp validation preserves
the audit trail without contacting hardware or widening live authority.

## Rejected Alternatives

Manual-only review was rejected because impossible timestamps can be caught
deterministically before a bad candidate looks ready. Accepting any matching
digit pattern was rejected because M102 is a format guard; M103 adds the
semantic timestamp check needed to make that format meaningful.

## Implementation Plan

- Parse the timestamp prefix of canonical-looking WING read-only `run_id`
  values.
- Add focused regression coverage for a self-consistent impossible-timestamp
  candidate that previously validated as accepted.
- Add rail coverage and update state/report surfaces.

## Test Plan

- Focused WING read-only evidence tests.
- M103 rail.
- Full WING MVP rail.
- JSON/JSONL validation and memory lint before commit.

## Risks and Rollback

The main risk is rejecting hand-edited captures that use an impossible
timestamp-like directory name. That is intended for M001: real hardware proof
should preserve the runner-generated UTC capture id. Roll back by reverting
this ADR, the M103 plan/report/state entries, the test and the timestamp guard.
No live transport or console state is affected.
