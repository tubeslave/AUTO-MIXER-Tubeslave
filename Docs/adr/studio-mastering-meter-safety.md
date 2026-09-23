# ADR: STUDIO mastering safety measurements

## Context

The project canon requires ITU-R BS.1770-4 loudness gating and 4x true-peak measurement. The existing `backend/auto_mastering.py` still contains an RMS loudness approximation and sample-peak limiting, while the repository already has reusable K-weighting and true-peak DSP in `backend/lufs_gain_staging.py`. Autonomous mastering needs trustworthy objective evidence before it may decide that a candidate is even technically safe for human review.

## Options considered

1. Add `pyloudnorm` or another mastering dependency.
2. Duplicate K-weighting and oversampling DSP inside the STUDIO path.
3. Reuse `KWeightingFilter` and `TruePeakMeter`, adding only integrated gating and a measurement facade.
4. Keep RMS/sample-peak approximations until a later mastering rewrite.

## Decision

Use option 3. Add `backend/studio_mastering_metrics.py` as a side-effect-free offline measurement facade. It computes 400 ms loudness blocks at 100 ms hop, applies the -70 LUFS absolute gate and -10 LU relative gate, reports sample peak / true peak / RMS / crest factor, and exposes a true-peak safety attenuation helper that can only reduce gain.

The current bounded change does not alter `AutoMaster` output. Wiring these measurements into the mastering controller is a separate task because that will change rendered audio and therefore requires human listening before subjective acceptance.

## Why this won

It follows existing project DSP, adds no production dependency, catches intersample overs that sample peak misses, and cleanly separates objective measurement from creative mastering decisions.

## Rejected alternatives

- New third-party loudness dependency: unnecessary for this task and conflicts with the small-diff/no-new-dependency rule.
- A second custom K-weighting/oversampling implementation: creates duplicate DSP that can drift from the project's tested meters.
- Leaving RMS/sample peak in place: blocks reliable mastering validation and contradicts the project canon.

## Implementation plan

1. Add `StudioMasteringMeter` and immutable measurement evidence.
2. Add synthetic tests for LUFS calibration, stereo energy, silence, intersample peak detection, true-peak attenuation, and no-boost behavior.
3. Open a PR so the normal Python 3.10/3.11/3.12 test matrix runs.
4. In the next STUDIO task, replace `AutoMaster` RMS/sample-peak acceptance evidence with this facade and add human-review gating for any resulting render change.

## Test plan

Run the targeted synthetic tests locally, then require normal repository CI before merge. No external services or paid credits are needed.

## Risks and rollback

The initial implementation intentionally supports the current STUDIO mono/stereo use case with unit channel weights and does not guess surround speaker roles. If tests or review find a problem, rollback is deletion of the new module/tests; existing mastering behavior is untouched by this ADR's first implementation step.
