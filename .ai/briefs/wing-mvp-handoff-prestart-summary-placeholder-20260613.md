# Brief: WING MVP Handoff Prestart Summary Placeholder

Date: 2026-06-13
Status: implemented

## Task

Keep the first WING MVP blocker focused on the read-only connection proof by
binding the offline handoff packet to the passed no-WING prestart `summary.json`
that the read-only runner now requires.

## Context

`tools/wing_readonly_preflight_runner.py` fails closed unless both plan-only and
real read-only runs receive `--prestart-summary`. The handoff packet and no-WING
release checks still needed to expose and validate that operator-facing path.

## Safety Boundary

This task is offline and read-only. It does not contact WING, does not open a
backend WebSocket, does not mutate console state, and does not grant live
authority or readiness closure.

## Done

- Handoff packet commands include `--prestart-summary`.
- Handoff packet declares the prestart summary path placeholder.
- no-WING release checks validate matching plan/capture/handoff prestart paths.
- no-WING release checks create a prestart summary artifact for their offline
  runner plan check only after the backend-port gate passes.
- MVP docs continue to require real WING artifacts and manual review.
