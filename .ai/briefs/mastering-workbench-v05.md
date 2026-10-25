# Mastering Workbench v0.5: Musical Regression Critic

The quality baseline remains v0.3. v0.4 is a Delivery Optimizer candidate, not automatically a quality upgrade.

New critics compare a candidate against the quality baseline after level matching:
- Transient Critic: kick-band and snare-presence onset retention.
- Low-End Punch Critic: 35–180 Hz attack/body ratio on matched base-defined events.
- Spectral Shift Critic: level-insensitive six-band tonal displacement.

A louder candidate must pass these musical guards in addition to peak/crest/stereo checks.
Metrics are heuristics, not substitutes for listening. Human preference remains the final acceptance gate.
