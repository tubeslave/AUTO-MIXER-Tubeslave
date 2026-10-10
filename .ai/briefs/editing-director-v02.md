# Editing Director v0.2: groove-aware local editing

Do not quantize the performance.

Infer a pulse/subdivision map only as evidence. A drum event is eligible for an automatic timing edit when:
- its deviation is clear but bounded;
- neighboring events are close to the inferred groove;
- the event is isolated rather than part of a fill;
- the correction is capped (12 ms in the Ptitsa validation);
- all drum microphones receive the identical smooth local warp.

Ptitsa validation inferred 86 BPM and accepted six isolated local corrections.
Bass was not shifted automatically because its kick-relative offset had too much dispersion.
Guitar required no meaningful global correction.
