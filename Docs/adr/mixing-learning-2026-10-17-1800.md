# ADR: Downgrade warm-initialization thresholds and pin research artifacts

- Status: `accepted_for_research_workflow`
- Date: 2026-10-17
- Update: [ML-2026-10-17-1800](../mixing_learning_updates/ML-2026-10-17-1800.md)
- Scope: research and experiment planning only
- Runtime effect: none

## Context

The complete warm-initialization paper and appendix show that the frequently highlighted region near `tau_init≈0.8` is based on model-specific plots plus informal listening. No controlled listener protocol, loudness matching or inferential analysis is reported. FAD references also use different classifier-free guidance from transfer renders, and the pitch-set Jaccard metric discards temporal, dynamic and stereo information. The official companion repository contains qualitative WAV examples but not implementation code, a frozen environment, model revision, seed manifest, evaluation scripts or an explicit licence.

The official DAFx plate challenge repository improves synthetic reproducibility by exposing the simulator, data, baselines and evaluators. It also confirms that the challenge is entirely synthetic and that normalized WAV previews discard absolute-amplitude cues. No explicit repository licence or measured real-plate validation set was identified.

## Decision

Downgrade all warm-initialization numeric operating-point language to exploratory, model-specific informal-listening evidence. Retain the method only as a blocked research hypothesis. Revise the planned `lambda=0` versus `lambda=1` comparison to use a predeclared paired seed block, immutable provenance, raw preserved renders and separate attenuation-only BS.1770-matched blind copies. Do not use drums or claim cleanup/separation.

Pin both official artifact repositories, but do not ingest their audio/data or run their code until licence and reproducibility gates are satisfied. For any future plate experiment, use unnormalized NPZ displacement and require a separate rights-cleared measured-plate external test before promotion.

All candidate rules remain `auto_apply:false`; both experiments remain `not_run_blocked`.

## Consequences

- One Source Card is promoted to complete reading; two Source Cards are revised with artifact context.
- Two Artifact Audits and nine Knowledge Cards are added.
- Two candidate rules and two blocked experiment plans are revised.
- The companion examples remain qualitative evidence only; no audio was downloaded or played.
- No runtime, DSP, repository implementation, model or audio asset changes.
- The next executable experiment remains `EXP-ML-PENA-AB15-LISTEN-20261010-1800`.
- Citation checking remains partial; absence of a discovered correction or contradiction is not confirmation.
