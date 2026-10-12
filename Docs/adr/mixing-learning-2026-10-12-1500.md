# ADR: retain DiffVox as topology-conditioned vocal-FX evidence

- **ADR ID:** `mixing-learning-2026-10-12-1500`
- **Status:** accepted for the research queue; no runtime promotion
- **Date:** 2026-10-12
- **Update:** `ML-2026-10-12-1500`

## Context

DiffVox provides full-text DAFx evidence and MIT-licensed code for inverse fitting and analysing a fixed vocal-effects chain. Its parameter distribution is structured and non-Gaussian, but its evaluation is objective, single-track, topology-specific and partly based on unavailable private audio.

## Decision

Store the work as evidence for **coupled, shadow-only vocal-FX candidate search**, not as a source of vocal presets or automatic control values. Any project trial must use a frozen topology/vector, separate raw measurement renders from BS.1770-matched blind listening copies, protect intelligibility/identity/attack/mono/headroom and require Dmitry's recorded verdict. `auto_apply:false`.

Do not attribute inverse-fitted controls to the original engineer, do not copy dataset means as rock presets, and do not generalize the objective spatial-effect ablation to listener preference.

## Consequences

- One bounded rule and one blocked “Пена” experiment enter the queue.
- The experiment can evaluate only the whole frozen spatial-return bundle, not its individual controls.
- The later `arXiv:2505.11315` style-transfer work and artifact/data parity receive deep-review priority.
- No audio, DSP, runtime or model changes are authorized.

