# Frontend cockpit audit — 2026-07-10

## Result

The frontend moved from **62/100** to **96/100** for the scoped production
operator-cockpit goal.

| Area | Weight | Baseline | Final | Evidence |
| --- | ---: | ---: | ---: | --- |
| Truthful live data and provenance | 25 | 8 | 25 | WING/audio/Product Layer readback; no fake live metrics |
| Functional controls | 20 | 9 | 19 | backend Refresh/Analyze, read-only Connect, desktop/mobile meter Start/Stop |
| Operator ergonomics | 20 | 15 | 19 | dense desktop cockpit, selected-channel focus, compact mobile view |
| Live safety transparency | 15 | 12 | 15 | observation-only authority, apply blockers and mode gates visible |
| Responsive/accessibility | 10 | 8 | 9 | semantic labels, disabled modes, 390x844 compact view |
| Resilience/performance | 10 | 10 | 9 | stale-status race fixed; 503.05 kB chunk warning remains |
| **Total** | **100** | **62** | **96** | |

## What changed

- Normal frontend startup connects to the Product Layer unless explicit demo
  mode is selected.
- WING connection, mixer updates, capture state and meter frames update the UI.
- Master/channel/analyzer values no longer fall back to plausible mock numbers
  in live mode.
- Fake Play, Record, Snapshot, rollback and elapsed-time transport was replaced
  by a backend/console/audio/authority/safety runtime bar.
- System topology removed placeholder network and clock health.
- Director Review/Dismiss routes use the real task/proposal workflow.
- Desktop and mobile modes share the same backend capability gates.
- Audio-capture events cannot be overwritten by late dashboard snapshots.

## Real-session evidence

- WING Rack `192.168.1.102` connected in observation-only mode.
- 48-channel / 48 kHz WING audio capture running.
- Confirmed fader readback: CH1 `-3.1 dB`, CH3 `-12.1 dB`, CH8 `-4.8 dB`,
  Main `0.0 dB`.
- Current signal was silence, correctly displayed as `NO SIGNAL`.
- Desktop/mobile meter Start/Stop and backend Refresh were exercised in the
  real browser.
- Browser console had zero warnings or errors.

## Screenshots

- Baseline: `screenshots/01-baseline-dashboard.png`
- Final desktop: `screenshots/06-final-live-cockpit.png`
- Final mobile: `screenshots/07-final-mobile-operator.png`
- Final system: `screenshots/08-final-system-state.png`
- Same-viewport comparison: `screenshots/09-before-after-comparison.jpg`

## Honest boundary

The 96/100 score covers frontend truth, ergonomics and safe control wiring. It
does not claim Full Autonomous readiness or broad live WING writes. Controls
without canonical backend authority are deliberately read-only or disabled.
