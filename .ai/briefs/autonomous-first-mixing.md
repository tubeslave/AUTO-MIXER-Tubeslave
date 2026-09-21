# Autonomous-first mixing policy

Evidence from Premiera showed the strongest baseline came from the no-reference/no-guided-intent pass. The system therefore changes its default order.

1. Build a technically valid autonomous mix from audio + track roles.
2. Self-critique balance, masking, dynamics, stereo, section contrast and artifacts.
3. Freeze a baseline.
4. Only then inspect optional references or short user intent.
5. Treat them as priors/constraints, never as compulsory spectral or level targets.
6. Generate bounded causal deltas and keep a change only when it improves the protected objectives.
7. Human corrections become context-tagged calibration evidence. One song must not become a universal rule.
8. Dense distorted-guitar sections receive an explicit over-weighting guard.
9. Prefer bracket/binary search after a human says "too much" then "too little".
10. Stop polishing when remaining moves are perceptually small or evidence is unstable.

Premiera calibration evidence:
- autonomous v23 was preferred as the overall starting point;
- v25 guitar level was still high, v26 was too low, midpoint v27 was closer;
- vocal was subsequently lowered by about 1.4 dB by v29;
- these values are song-specific evidence, not fixed defaults for future songs.
