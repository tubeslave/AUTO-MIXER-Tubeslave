# Brief: WING Read-Only Preflight Offline Plan Preview

## Context

The current WING proof gate needs a real read-only preflight before any
supervised fader/gain proof. Operators also need to see the exact request
sequence before opening a backend/WING session, so channel confirmation mistakes
can be caught without contacting hardware.

## Goal

Add a `--plan-only` preview path for `tools/wing_readonly_preflight_runner.py`
and make no-WING checks verify that the preview command matches the real
read-only capture command.

## Constraints

- Plan-only mode must not open WebSocket.
- Plan-only mode must not query WING.
- Plan-only mode must not create physical proof artifacts.
- Plan-only mode must keep live authority and write capability false.
- Plan-only mode must name the current MVP blocker and manual review boundary.
- Plan-only mode must expose fail-closed conditions and post-run review commands.
- The handoff packet and no-WING summary must expose the preview command.

## Validation

- Unit tests for successful plan preview and blocked channel confirmation.
- Handoff packet tests for `preview_readonly_request_plan`.
- no-WING release-check tests for plan/capture consistency.
- no-WING quick smoke and full pytest.
