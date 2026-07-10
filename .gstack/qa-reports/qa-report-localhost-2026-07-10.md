# QA report: Automixer live frontend

**Target:** `http://localhost:3000/#/dashboard`
**Date:** 2026-07-10
**Baseline:** 62/100
**Final:** 96/100

## Test environment

- Real WING Rack: `192.168.1.102`, observation-only.
- Product Layer WebSocket: `127.0.0.1:8765`.
- Audio capture: WING device 0, 48 channels, 48 kHz.
- Desktop: 1440x900.
- Mobile: 390x844.

## Findings and resolution

| ID | Severity | Baseline defect | Resolution | Evidence |
| --- | --- | --- | --- | --- |
| ISSUE-001 | P0 | Live dashboard mixed real WING state with mock LUFS, peak, spectrum, energy, gates and compressors. | Live fields now carry availability/provenance and render `NO SIGNAL` or `Unavailable`; channel faders/names use WING readback. | `issue-001-live-wing-truth-gap.png`, `issue-001-after.png` |
| ISSUE-002 | P1 | System screen showed inert Auto Layout/Refresh and placeholder Ethernet, clock and latency health. | Refresh calls the backend; topology and device cards show Product Layer, WING, audio capture and write authority only. | `issue-002-system-placeholders.png`, `issue-002-after.png` |
| ISSUE-003 | P1 | Mobile exposed Full Autonomous and local rollback regardless of live authority. | Mobile gates match desktop; Full Autonomous is disabled and Safe Mode controls real Refresh and meter Start/Stop. | `issue-003-mobile-after.png` |
| ISSUE-004 | P1 | A late dashboard snapshot could overwrite a newer audio Stop status. | Canonical `audio_capture_status` is authoritative over dashboard snapshots; repeated desktop/mobile Start/Stop checks pass. | regression test plus live browser run |
| ISSUE-005 | P2 | 10 Hz meter frames flooded the console. | High-rate meter frames are intentionally not logged. | clean browser console |

## Browser verification

- Dashboard autoconnected to Product Layer and showed `wing 192.168.1.102`.
- WING fader readback included CH1 `-3.1 dB`, CH3 `-12.1 dB`, CH8 `-4.8 dB`,
  and Main `0.0 dB`.
- Current audio input was silent; master, channel and analyzer surfaces showed
  `NO SIGNAL` rather than fabricated levels.
- Channel selection changed the selected readback to CH03.
- Refresh and Analyze were backend actions.
- Desktop and mobile meter Stop/Start both changed backend and UI state.
- Full Autonomous was disabled in the live backend state.
- Browser warnings/errors: 0.

## Automated verification

- Frontend unit/regression: 102 passed.
- Frontend Playwright: 9 passed.
- Backend audio/meter handlers: 9 passed.
- Production build: passed.
- `git diff --check -- frontend`: passed.

## Residual risks

- Production JS is 503.05 kB minified / 134.10 kB gzip and triggers Vite's
  chunk-size warning. Route-level splitting is a future performance task.
- Broad console writes are intentionally absent. A visible fader is readback,
  not an unproven live write surface.
- Physical supervised apply/readback/rollback evidence remains governed by the
  existing WING safety milestone and is not granted by this frontend score.
