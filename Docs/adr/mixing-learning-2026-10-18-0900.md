# ADR: Add a perturbation-robustness gate for objective audio metrics

- Status: `accepted_for_research_workflow`
- Date: 2026-10-18
- Update: [ML-2026-10-18-0900](../mixing_learning_updates/ML-2026-10-18-0900.md)
- Scope: research and experiment planning only
- Runtime effect: none

## Context

The complete PADP preprint reports four waveform perturbation families intended to preserve perceived content while probing objective audio-quality metrics and foundation-model embeddings. In two small mono MUSHRA tests, several perturbation settings were judged near the hidden reference, yet metric and embedding scores did not move consistently with listener ratings. DeePAQ and ViSQOL differed substantially in rank correlation on the constructed PADP set.

The evidence is narrow: only 12 source items, 8–9 retained listeners, mono left-channel evaluation, post-screening, a chosen mean difference-grade threshold, 12 sampled APX configurations out of 4096, and no released implementation/stimulus/results manifest found. The paper does not validate rock-mix quality, stereo invariance or any production benefit.

## Decision

Add a research-only gate: an objective metric proposed for mix/DSP acceptance must be stress-tested on a block of human-verified transparent perturbations before it can influence project decisions. Use one perturbation family and one fixed setting at a time, preserve raw outputs, make separate attenuation-only BS.1770-matched listening copies, pin every implementation/version and predeclare both the human-transparency criterion and metric tolerance.

Metric instability flags a limitation of the metric/condition; it does not validate the perturbation as a mixing processor. PADP parameters remain source provenance, not defaults. No metric becomes a sole acceptance gate, and no rule is auto-applied.

## Consequences

- One complete Source Card, nine Knowledge Cards, two metadata/companion-only Video Source Cards, one bounded candidate rule and one blocked experiment are added.
- The two videos remain `queued_source`; no transcript/captions were obtained and `videos_studied:0`.
- No legal raw multitrack is added.
- `RULE-ML-20261018-0900-01` has `auto_apply:false`.
- `EXP-ML-20261018-0900-01` remains `not_run_blocked` pending implementation, licence, metric-version and approved-audio gates.
- The next executable experiment remains `EXP-ML-PENA-AB15-LISTEN-20261010-1800`.
- No audio, code, model, DSP or runtime changes were made; citation checking remains partial.
