# STUDIO Offline Master Delivery v1 — 2026-09-24

## Goal

Connect the accepted `MasteringTargetController` to a real file-oriented offline delivery path without touching live control or granting machine authority over subjective acceptance.

## Repository evidence

The controller can already search bounded pregain/maximizer candidates through the mastering safety gate, but it is not yet responsible for actual WAV/MP3 delivery. The Ptitsa fresh-mix experiment also exposed a real mastering defect: the sample-domain maximizer can leave an intersample overshoot, so safety must be measured and enforced on the actual post-maximizer signal before export.

## Bounded task

- add a STUDIO-only `render_offline_master()` / `deliver_master()` facade;
- preserve and hash the input before/after the run;
- export only a machine-safe candidate, never a rejected one;
- remeasure the written PCM24 WAV and decoded 320 kbps MP3;
- keep every audible result `pending_human_review` and `baseline_eligible=false`;
- reject invalid, silent, non-stereo or too-short inputs;
- fail on output-directory reuse rather than overwriting evidence;
- add linked static true-peak attenuation after the existing maximizer so intersample overshoots cannot pass the safety gate;
- do not change Mixing Directors, cleanup, live-console code, OSC, or any accepted audio baseline.

## Acceptance

Synthetic reachable target delivers WAV+MP3 with current evidence; an unreachable target returns the exact source and no fake mastered audio; source bytes stay immutable; true-peak overshoot regression is covered; full Python 3.10/3.11/3.12 repository CI must pass before merge.

## Human listening

Machine validation only establishes engineering/safety correctness. Musical mastering output remains pending explicit human listening acceptance.
