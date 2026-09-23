# ADR: Wire standards-based metering into offline AutoMaster

## Context

The accepted STUDIO mastering meter provides gated integrated loudness and 4x true-peak measurement/safety attenuation. `AutoMaster` still uses RMS as its operational loudness target and sample peak as its operational ceiling in the built-in and fallback paths. Leaving those approximations in the actual render path makes the new measurement primitive observational only.

## Options considered

1. Rewrite `AutoMaster` and its public API around a new mastering result model.
2. Modify the giant offline orchestrator first.
3. Keep the public API stable and replace only the operational loudness/ceiling measurements inside `AutoMaster`, while retaining explicit compatibility helpers.
4. Leave the existing render path unchanged and use the new meter only for reports.

## Decision

Use option 3. `AutoMaster._estimate_lufs()` and `_limit()` remain compatibility helpers, but actual built-in, fallback and reference-master output paths use `StudioMasteringMeter` for integrated LUFS and true-peak safety. Existing EQ matching and compressor policy are out of scope.

## Why this won

It is the smallest change that makes the accepted standards meter affect real STUDIO mastering behavior while preserving return types used by the offline orchestrator and tests. It also avoids an unrelated rewrite of `tools/offline_agent_mix.py`.

## Rejected alternatives

- Public API rewrite: too broad for one Director cycle and creates unnecessary migration risk.
- Editing the offline monolith first: larger blast radius and duplicates mastering safety logic.
- Reporting only: would leave real mastering constrained by sample peak/RMS rather than the accepted safety primitive.

## Implementation plan

1. Import `StudioMasteringMeter` in `backend/auto_mastering.py`.
2. Use integrated LUFS for operational target gain in built-in/fallback paths.
3. Use 4x true-peak attenuation for actual built-in/fallback/reference outputs.
4. Preserve legacy `_estimate_lufs()` and `_limit()` behavior for compatibility.
5. Add focused regression tests, then run full CI.

## Test plan

- Existing `test_auto_mastering.py` and `test_mastering.py` remain green.
- Verify built-in result LUFS equals the standards meter on the returned audio.
- Verify `_match_target_loudness()` corrects an intersample overshoot even when sample peak is already below the ceiling.
- Require the normal full repository CI before merge.

## Risks and rollback

Integrated LUFS can request different gain than the legacy RMS proxy and true-peak attenuation can reduce level where sample peak looked safe. Those are intended technical corrections but are audible. Human A/B remains mandatory before any rendered audio becomes a baseline. Rollback is the single integration commit; the standalone meter remains valid.