# production_mix_v1

## 1. Context

AUTO-MIXER-Tubeslave already has legacy live/soundcheck flows, a large offline
renderer, Decision Engine v2, Safety Gate v2, MuQ-Eval, perceptual/MERT shadow
scoring, and AutoFOH analysis helpers. The new request is for a separate
production offline/live-ready pipeline that must not change default behavior.

## 2. Options Considered

- Extend the existing offline renderer.
- Extend Decision Engine v2.
- Add a new independent `production_mix_v1` package.

## 3. Decision

Add `production_mix_v1` as a separate package under `automixer/` with its own
config, CLI, reports, tests, and rollback path.

## 4. Why This Won

It keeps legacy behavior stable, allows candidate-based audio scoring, keeps OSC
behind a queue, and makes rollback a file-level deletion rather than a cross-cut
through backend live code.

## 5. Rejected Alternatives

- Rewriting `tools/offline_agent_mix.py`: too risky for an existing working
  offline path.
- Depending on `ai_mixing_pipeline`: source files are absent in this worktree.
- Making MuQ/MERT mandatory: model dependencies may be unavailable.

## 6. Implementation Plan

1. Add pipeline config.
2. Add isolated modules for audio loading/rendering, analyzers, candidates,
   critics, decision, safety, OSC queue, reports, and CLI.
3. Keep dry-run as default for offline and live.
4. Add unit tests and smoke run.

## 7. Test Plan

- `PYTHONPATH=backend python -m pytest tests/test_production_mix_v1.py -q`
- `PYTHONPATH=backend python -m automixer.production_mix_v1 --smoke --dry-run`
- Run broader tests if local environment permits.

## 8. Risks And Rollback

Risks are heavy optional dependency imports, too-weak fallback scoring, and
unsafe live apply. Rollback is to remove the new package, config, docs, ADR, and
tests. No default pipeline, server route, or mixer client is modified.

## 9. Musical Recipe Refinement

The first SONG REPA verification showed that standalone `final_linear_safety_trim`
could win as a technical candidate. The pipeline now separates candidate scoring
into `musical_score`, `technical_safety_score`, and `final_score`. The Decision
Engine selects a musical candidate first, then the pipeline applies final linear
safety trim as a utility post-process.

`production_mix_v1` also inherits the explicitly requested
`FX15_SECTION_MOD_SPACE_AIR_MINUS_080` reference recipe through adapters/config:
FX15 section/mod space sends, final air trim -0.80 dB, vocal/bass/drum
compression recipes, drum room glue, and parallel compression candidates.
Fallback/proxy critic scores remain non-authoritative, and technical-only wins
require review instead of automatic acceptance.

## 10. Render Verification Boundary

The second SONG REPA verification showed that planned FX/compression actions can
look musical in the report while not proving that the rendered audio changed.
`production_mix_v1` now tracks every action as `planned`, `rendered`,
`verified`, or `failed`. Offline acceptance requires an actual render and
required musical actions to be rendered/verified; live dry-run can only produce
`accepted_plan_only`.

Musical score is no longer allowed to increase from action labels alone. It is
derived from rendered before/after audio metrics and real critic scores. Combined
musical candidates are hard-rejected when required fader balance, compression
parameters, FX route/return fields, FX wetness, compression gain-reduction proxy,
or balance safeguards are missing. Reports include action status counts,
render-verification metrics, balance before/after, FX wetness, measured dynamics
proxy, and decision state.

## 11. Role Confidence And Recipe Transfer Guard

New songs must not inherit fixed SONG REPA role/group balance values. The
accepted reference recipe may transfer only taste-level decisions: FX space,
final air trim, limiter target, and gentle bus glue. Fixed group LUFS targets,
large gain moves, channel-specific compression GR, and unverified pan maps are
blocked unless explicitly approved.

`production_mix_v1` now classifies every stem through `RoleDetectionResult`
with confidence, evidence, ambiguity, and an action level. Role-specific
processing is allowed only at confidence >= 0.85. Ambiguous stems are limited to
universal safe actions or no-op/review. A local `config/role_overrides.yaml`
lets the operator pin names such as Playback, ROOM DR, Ride, and Hi-Hat with
confidence 1.0.

The decision layer now rejects/reviews aggressive candidates when MuQ-Eval is
unavailable or fallback-only, when final score delta is < 0.01 with more than
10 actions, when large gain or ambience-destroying cuts are present, and when
measured compression GR exceeds role limits. Dry-run reports separate offline
render acceptance from console action acceptance and live readiness.

## 12. Material Improvement And Low-Signal Guard

The MIX report showed that a single bounded crest-control action could be
accepted with only a 0.0011 final delta while MuQ-Eval was unavailable. The
Decision Engine now uses `material_improvement` rather than raw
`metrics_improved`. When MuQ-Eval is unavailable/fallback/excluded, offline
acceptance requires at least 0.010 final-score delta and 0.020 musical-score
delta; otherwise the pipeline falls back to `candidate_000_no_change` or
review. Single-action micro-fixes are accepted only when they explicitly fix a
critical safety issue.

Low-signal sources below -45 dBFS RMS/LUFS cannot receive compression,
parallel compression, EQ boosts, or gain boosts. Crest factor is taken from the
shared analysis metrics, and any crest-factor disagreement above 3 dB between
role evidence, action parameters, and analysis is reported as
`metrics_conflict` and rejected/reviewed.

Reference-taste candidates are typed separately from large remixes, so FX15
reference candidates are not rejected as `review_required_muq_excluded_large_remix`.
Playback files default to `music_stem` with `universal_safe` permissions unless
an operator explicitly opts into role-specific playback handling.
