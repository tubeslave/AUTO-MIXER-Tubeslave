# Brief: frontend live cockpit

## Objective

Raise the Automixer frontend from a visually convincing prototype to a truthful
production-grade operator cockpit scoring at least 95/100.

## Required outcome

- Connect to the Product Layer automatically in normal operation.
- Show real WING, audio-capture, meter, channel, master and safety state.
- Never substitute mock measurements when a live field is unavailable.
- Wire every active control to a canonical backend action; otherwise render it
  read-only or blocked with an operator-readable reason.
- Preserve observation-only WING authority and fail-closed mode gates.
- Support desktop sound-engineer work and a compact mobile musician/operator
  view.
- Prove the result with tests, real-browser QA and screenshots.

## Safety boundary

This frontend goal does not widen console-write authority. It may expose only
the backend capabilities already allowed by the supervised safety contract.

## Acceptance evidence

1. Real WING Rack readback at `192.168.1.102`.
2. Real 48-channel, 48 kHz audio-capture status.
3. Working Refresh, Analyze, WING read-only connect and meter Start/Stop paths.
4. Disabled autonomous mode while backend apply authority is denied.
5. No browser console errors or warnings.
6. Frontend tests, production build, E2E and targeted meter-handler tests pass.
7. Desktop, system, mobile and before/after screenshots are stored under
   `Docs/reports/frontend_cockpit_20260710/screenshots/`.
