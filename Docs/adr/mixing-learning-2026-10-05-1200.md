# ADR — Mixing Learning deep review 2026-10-05 12:00 MSK

## Status

Accepted as source-grounded knowledge update. No automatic DSP/runtime action is authorised.

## Context

The search found a previously missing DAFx-26 paper on stability-regularised recurrent virtual-analogue modelling and an associated public artifact. It also found two useful rock-vocal educational videos whose captions were unavailable, plus first-party companion articles.

The scientific result is narrower than a production rule: its strongest automation evidence is a zero-input control-noise test; validation is reused as test, best-of-three runs are reported without dispersion, and programme-audio/listener/target-host evidence is absent. The DAFx and arXiv records are the same version family. The repository licence contents could not be verified.

## Decision

1. Add one full-text scientific Source Card under stable ID `arxiv:2509.15622`, with DAFx paper 26 as an alias rather than independent confirmation.
2. Add an automation-specific recurrent-effect validation rule with `auto_apply:false`; never infer safety from static-fit metrics alone.
3. Plan a matched `L2`-regularisation A/B, but keep it blocked and `not_run` until the artifact, licence, manifests and rights-clear audio are reproducible.
4. Treat the Billy Decker template method as a routing scaffold; retain all numeric settings as demonstration provenance only.
5. Add two videos as metadata-only `queued_source`. The full companion articles may support Source/Knowledge Cards, but the videos are not marked studied without transcript/content.
6. Plan a separate vocal-reverb HF-damping A/B with one changing factor and BS.1770 matching; keep it `not_run`.
7. Keep `EXP-ML-20261001-1800-01` as the next immediately executable project experiment.

## Consequences

- New rules remain `auto_apply:false`.
- Experiments remain `not_run`; no audio, model, DSP, runtime or repository code changes occur.
- The video queue becomes `91 / 8 / 0 / 83` (`total / transcript-complete / partial / metadata-only`).
- `citation_check:partial`: no complete downstream-citation audit was available, and Scite/Consensus were not used.
