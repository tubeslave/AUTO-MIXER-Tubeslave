# MuQ EWMA Drift

## Task

Add optional per-stem EWMA metrics and drift detection on MuQ quality observations.

## Safety Constraints

- The feature must be disabled unless configured.
- CRIT drift must prefer restoring last known good parameters and freezing ML
  correction for the affected stem rather than applying more correction.
- The implementation must not bypass existing safety gates, fader ceilings,
  true-peak checks, or feedback handling.

## Expected Behavior

- Track EWMA, baseline, drift, debounce timers, and state per stem.
- Support optional frequency-mask overrides such as a vocal 300-800 Hz band.
- Log NORMAL/WARN/CRIT transitions and expose an OSC-style visualization payload.
- Freeze stem ML corrections after CRIT until the stem has stayed NORMAL for the
  configured recovery window.

## Tests

- Synthetic noisy/stair-step MuQ sequences.
- False-trigger checks below thresholds.
- Debounce and hysteresis unit tests.
