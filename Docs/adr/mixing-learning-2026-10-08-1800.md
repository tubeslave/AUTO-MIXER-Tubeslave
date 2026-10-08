# ADR: separate transport latency from acoustic arrival

- status: accepted for research guidance
- date: `2026-10-08`
- update: `ML-2026-10-08-1800`
- auto_apply: `false`

## Context

The 15:00 search created a candidate rule for correcting residual parallel-path latency. Deep review found that the strongest peer-reviewed source concerns delay estimation between microphone recordings of one acoustic source, while the vendor example concerns a nonlinear external compressor path. Official Ableton and Cubase documentation also implement external-latency compensation differently.

## Decision

1. Classify delay as `transport_roundtrip` or `acoustic_arrival` before proposing correction.
2. Calibrate external hardware with a DAW ping or bypass/unity loopback; do not estimate its transport delay from a compressed signal if a cleaner calibration path exists.
3. Treat GCC-PHAT as an analysis candidate for same-source multi-mic signals, with explicit window, frame, search, confidence and sign contracts.
4. Do not transfer the paper's or vendor article's sample counts, frame sizes or the repository's thresholds into musical rules.
5. Preserve intentional room/overhead timing and require loudness-matched listening before accepting a correction.
6. Record the current GCC-PHAT OSC actuation path as a research risk; do not modify or enable it in this update.

## Consequences

`RULE-ML-20261008-1500-01` is narrowed and `EXP-ML-20261008-1500-01` becomes `not_run_blocked` until an actual hardware route and test material are defined. The current implementation's Hann window is directionally supported, but its thresholds and automatic hardware write remain unvalidated. The next executable project experiment remains `EXP-ML-20261001-1800-01`.
