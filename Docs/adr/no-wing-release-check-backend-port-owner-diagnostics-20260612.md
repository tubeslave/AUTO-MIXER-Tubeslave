# no-wing-release-check-backend-port-owner-diagnostics-20260612

## Context

The WING read-only preflight can fail before it proves anything if the expected
Automixer backend port is already occupied by another local service. The
no-WING release check already failed closed on an unexpected listener, but its
summary only reported `listener_detected: true`, leaving the operator to run
separate diagnostics before knowing what owns the port.

## Options considered

1. Keep the port check as a boolean pass/fail.
2. Automatically stop the listener when the port is busy.
3. Keep the fail-closed result and add best-effort listener owner diagnostics.

## Decision

Use option 3. When `backend_port_free` fails, the release check now adds
`listener_processes` from a best-effort local `lsof` query. If that lookup is not
available or fails, the check remains failed and records `listener_process_error`
instead. The generated markdown summary also surfaces the listener owner and the
operator next step so the diagnosis is visible without opening raw JSON.

## Why this won

The diagnostic makes the first WING proof blocker actionable without weakening
the no-WING safety boundary. The runner still does not contact WING, does not
send mixer-control traffic, and does not mutate the host by killing or stopping
the process.

## Rejected alternatives

Boolean-only output was rejected because it hides a common local setup problem.
Automatic process cleanup was rejected because a release gate must not destroy
unrelated local services or assume operator intent.

## Implementation plan

1. Add a small `lsof -F pcn` parser for listener owner details.
2. Attach listener owner data only when the socket probe confirms the port is
   busy.
3. Surface the same detail in `summary.md` for operator review.
4. Preserve previous pass/fail behavior and keep the lookup best-effort.
5. Cover the busy-port detail, markdown output, and parser behavior with
   focused tests.

## Test plan

- `PYTHONPATH=backend python3 -m pytest -q tests/test_no_wing_release_check.py`

## Risks and rollback

`lsof` output is platform-specific. The code treats the lookup as diagnostic
only, so lookup failures do not change the safety result. Roll back by removing
the helper and the extra detail fields if downstream consumers require the
older compact summary.
