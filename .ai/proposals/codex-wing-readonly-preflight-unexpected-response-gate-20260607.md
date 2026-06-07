# Proposal: Fail Closed On Unexpected WING Preflight Responses

## Short Thesis

The read-only preflight runner should stop immediately if the backend response
type does not match the request in flight.

## What I Understood

The current WING MVP needs a clean hardware artifact before supervised proof.
The event log contract is strict `send`/`recv` pairs, so the runner should not
continue after noisy, stale, or unrelated backend responses.

## Proposed Solution

Replace the retry-until-expected behavior in `send_and_wait` with a one-response
gate. After a request is sent, the next received payload must have one of the
expected response types. If not, raise a typed error, write a blocked summary
with `reason: unexpected_response_type`, preserve the actual payload, and exit
without sending further readback queries.

## Likely Files To Touch

- `tools/wing_readonly_preflight_runner.py`
- `tools/no_wing_release_check.py`
- `tests/test_wing_readonly_preflight_runner.py`
- `Docs/runbooks/WING_READONLY_PREFLIGHT.md`
- `Docs/adr/wing-readonly-preflight-unexpected-response-gate-20260607.md`
- `Docs/reports/AUTOMIXER_WING_READONLY_PREFLIGHT_UNEXPECTED_RESPONSE_GATE_20260607.md`
- `Docs/INDEX.md`

## Alternatives Considered

- Keep tolerating extra events and rely on evidence review. Rejected because it
  lets the hardware run continue after the event log is already invalid.
- Filter known heartbeat/status messages. Rejected for this MVP because a
  clean proof should be stricter than a production streaming client.

## Risks

- Real backend sessions with unsolicited events may fail preflight until the
  operator runs from a quiet backend state. This is acceptable for proof mode.

## Test Plan

- Focused runner tests for unexpected response blocking.
- no-WING quick gate.
- Direction guard, memory lint, diff check, and full pytest before commit.

## Where The Other Agent May Disagree

Kimi may prefer buffering unrelated backend events. For this first proof, the
strict path is safer and produces a stronger audit artifact.
