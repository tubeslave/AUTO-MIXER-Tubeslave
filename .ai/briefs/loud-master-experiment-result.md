# Loud Master experiment result: Ptitsa

Human level-matched evaluation:
- Quality baseline: Mastering Workbench v0.8
- Loud candidate: staged LoudMaster v11
- Loud-section target: about -8 LUFS
- Human result: no audible negative consequence; all elements remain clearly audible.

Measured staged loud master:
- loudest 3 s: -8.02 LUFS
- integrated: -10.78 LUFS-I
- final limiter max GR: 2.94 dB
- final limiter p95 GR in worst chunk: 1.14 dB
- bus density max GR: ~1.2 dB
- multiband peak conditioning: <=1 dB per band
- oversampled clipper: <=2.5 dB rare-peak reduction
- experimental true peak: about -0.52 dBTP

Decision:
- Staged loud mastering is validated as a useful mode for this material.
- Peak reduction is not automatically classified as damage; denser drums can be desirable.
- Default loud-master architecture should distribute crest reduction before the final limiter.
- Final limiter remains hard-budgeted at <=3 dB GR.
- Keep Quality Master and Loud Delivery Master as separate artifacts.
- Do not universalize -8 LUFS to every song. Search the loud-section target and processing budget per material, with -8 LUFS available as an explicit requested target.
- Production delivery should provide an optional safer true-peak ceiling (e.g. around -1 dBTP) separately from the approved loudness experiment.
