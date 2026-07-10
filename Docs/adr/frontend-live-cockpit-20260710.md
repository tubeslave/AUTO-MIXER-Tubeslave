# ADR: Frontend live cockpit truth contract

## 1. Context

The frontend looked like a live mixing console but defaulted to mock data,
filled absent live fields with plausible measurements, exposed local-only
transport actions and displayed placeholder system health. This was unsafe for
soundcheck and concert use because the operator could not distinguish console
readback from presentation data.

## 2. Options considered

1. Keep the prototype and add stronger mock labels.
2. Make all visual controls directly writable to WING.
3. Make the cockpit readback-first, wire only canonical backend actions and
   block every unsupported operation.

## 3. Decision

Choose option 3. The frontend autoconnects by default, consumes Product Layer,
WING connection, mixer update, audio-capture and meter events, and renders
availability/provenance explicitly. WING connect remains observation-only.

## 4. Why this won

It gives the operator accurate console and audio state without expanding live
authority. It also preserves the dense existing console UI and makes demo mode
available only through explicit local configuration.

## 5. Rejected alternatives

Labels alone do not remove fake measurements or inert controls. Direct broad
WING controls are rejected because EQ, dynamics, routing, snapshot and full-auto
writes lack the required backend/hardware proof.

## 6. Implementation plan

1. Extend frontend DTOs with WING/audio/availability state.
2. Autoconnect and subscribe to realtime readback events.
3. Remove live fallback measurements and fake transport/system state.
4. Add a truthful runtime control bar and system topology summary.
5. Align desktop and mobile mode gates and meter controls.
6. Protect event state from stale dashboard snapshots.
7. Add regression/E2E coverage and real-browser screenshots.

## 7. Test plan

- `npm test`
- `npm run build`
- `npx playwright test --reporter=dot`
- `PYTHONPATH=backend python3 -m pytest tests/test_audio_handlers.py tests/test_meter_stream.py -q`
- Real in-app browser QA against WING `192.168.1.102`, including Start/Stop,
  Refresh, channel selection, desktop/mobile mode gates and console logs.

## 8. Risks and rollback

The production JavaScript chunk remains slightly above 500 kB and should be
split later. Rollback is a frontend code revert; it must not alter the current
observation-only WING policy or reintroduce synthetic live measurements.
