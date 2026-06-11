# Proposal: WING Read-Only Preflight Backend Port Listener Confirmation

## Short thesis

The no-WING prestart gate should produce explicit machine-readable evidence that a busy backend port blocks real preflight until the operator confirms whether the listener is the intended Automixer backend.

## What I understood

The first WING MVP blocker remains `WING read-only connection proof`. We cannot close it without real hardware artifacts, but we can improve the required prestart diagnostic so a busy `8765` listener is handled safely and predictably.

## Proposed solution

- Add explicit busy-port detail fields:
  - `preflight_blocked_until_listener_confirmed`
  - `intended_backend_confirmation_required`
  - `safe_to_stop_listener_by_gate`
  - `fail_closed_reason`
- Keep `listener_processes` and `operator_next_step`.
- Extend tests for direct port checks and release-check summaries.

## Likely files to touch

- `tools/no_wing_release_check.py`
- `tests/test_no_wing_release_check.py`
- `Docs/adr/wing-readonly-preflight-backend-port-listener-confirmation-20260611.md`
- `Docs/reports/AUTOMIXER_WING_READONLY_PREFLIGHT_BACKEND_PORT_LISTENER_CONFIRMATION_20260611.md`

## Alternatives considered

- Automatically stop the listener: rejected because the gate cannot know whether it is the intended backend.
- Treat any listener as pass when it looks like Python: rejected because the operator must confirm intended backend identity.

## Risks

The change is stricter in wording but does not alter pass/fail behavior. It may cause tests that hard-code detail shapes to need updates.

## Test plan

- Focused backend-port tests.
- Full no-WING focused test file.
- Full backend suite before commit.

## Where the other agent may disagree

Kimi may prefer probing the listener over WebSocket to identify Automixer. That would contact a local service during the no-WING gate and could blur the prestart boundary; this slice keeps the gate diagnostic and non-mutating.
