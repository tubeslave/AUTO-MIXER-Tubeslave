# Proposal: WING Read-Only Preflight Listener Command-Line Diagnostics

## Short thesis

When the no-WING prestart gate finds a listener on the backend port, the artifact should include the listener's command line when available so the operator can identify it without probing or stopping the process.

## What I understood

The first WING MVP blocker remains open until real read-only preflight proof exists. The current local blocker is often a busy backend port. We can make that diagnostic safer and more useful while staying read-only and offline.

## Proposed solution

- Keep the existing `lsof` listener discovery.
- For each discovered PID, run `ps -p <pid> -o command=` and attach `command_line`.
- If `ps` fails, attach `command_line_error` and keep the original listener record.
- Extend tests for success and failure of the command-line probe.

## Likely files to touch

- `tools/no_wing_release_check.py`
- `tests/test_no_wing_release_check.py`
- `Docs/adr/wing-readonly-preflight-listener-command-line-diagnostics-20260611.md`
- `Docs/reports/AUTOMIXER_WING_READONLY_PREFLIGHT_LISTENER_COMMAND_LINE_DIAGNOSTICS_20260611.md`

## Alternatives considered

- Probe the listener over WebSocket: rejected because the no-WING prestart gate should not contact local control services.
- Leave only PID and command name: rejected because `Python` alone is too ambiguous for operator confirmation.

## Risks

`ps` may be unavailable or denied. The implementation records that as diagnostic metadata and does not convert a valid `lsof` result into a different failure mode.

## Test plan

- Focused listener diagnostics tests.
- Adjacent no-WING/WING focused tests.
- Full backend suite before commit.

## Where the other agent may disagree

Kimi may prefer a stronger identity check against the running backend. That may be valid later, but this slice avoids local network probing and only improves the existing diagnostic artifact.
