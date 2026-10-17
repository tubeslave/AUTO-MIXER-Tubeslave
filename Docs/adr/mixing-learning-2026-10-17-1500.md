# ADR: Keep two DAFx 2026 findings research-only until task transfer is tested

- Status: `accepted_for_research_workflow`
- Date: 2026-10-17
- Update: [ML-2026-10-17-1500](../mixing_learning_updates/ML-2026-10-17-1500.md)
- Scope: research and experiment planning only
- Runtime effect: none

## Context

Two open DAFx 2026 papers were absent from the shared index. One estimates physical plate parameters from controlled synthetic impulse responses; the other repurposes a diffusion model for several audio-to-audio transformations. Both are relevant to research tooling, but neither establishes a mixing-quality rule. The plate paper lacks real-unit and listening validation. The diffusion paper reports task- and model-specific controls, incomplete percussive/source suppression, and metrics that do not protect mix-critical transients, stereo image, groove or source identity.

## Decision

Add bounded Source and Knowledge Cards, but keep both methods research-only. Add one-factor, blocked experiment plans: phase-channel ablation for the plate estimator, and noise-injection A/B for warm initialization. Preserve raw outputs and create separate attenuation-only loudness-matched listening copies for any future audio comparison.

Do not copy `tau_init≈0.8` as a default, do not interpret plate-parameter NMSE as perceptual reverb quality, and do not use warm initialization as automatic source cleanup. All candidate rules remain `auto_apply:false`; both experiments remain `not_run_blocked`.

## Consequences

- Two Source Cards, seven Knowledge Cards, two candidate rules and two blocked experiment plans are added.
- Three official YouTube links remain metadata-only queued sources; no video-derived rule is created.
- No audio, code, model, plug-in, DSP engine or runtime rule is changed.
- The next executable experiment remains `EXP-ML-PENA-AB15-LISTEN-20261010-1800`.
- Citation checking remains partial; lack of a discovered correction or contradiction is not confirmation.
