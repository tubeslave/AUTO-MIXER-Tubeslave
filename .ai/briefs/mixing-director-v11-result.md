# Mixing Director v1.1 Dynamics experiment

Baseline: v1.0 faders + pan + section automation. Human feedback: balance was strong, but individual source dynamics caused instruments to rise/fall too much.

v1.1 adds only slow pre-compressor rides and role-aware compression. EQ, saturation, reverb and limiting remain disabled.

Ptitsa diagnostics:
- all 18 tracks passed Dynamics Critic;
- Valera vocal active spread: 6.21 -> 4.65 dB, p95 GR 1.97 dB;
- Nikita vocal: 14.91 -> 11.83 dB, p95 GR 2.78 dB;
- bass: 6.40 -> 4.73 dB, p95 GR 2.11 dB;
- guitar: 2.51 -> 1.96 dB, p95 GR 0.48 dB;
- kick close mics: spread reduced about 1.7–1.8 dB, p95 GR about 1.7–1.8 dB;
- snare close mics: spread reduced about 2.5–2.8 dB, p95 GR about 2.8 dB;
- cymbals remain lightly controlled.

Post-dynamics approximate balance remains inside v1.0 guardrails:
- vocal/music: -4.22 dB;
- kick/bass: +0.18 dB.

No automatic rebalance was required. Human level-matched A/B remains the final acceptance gate.
