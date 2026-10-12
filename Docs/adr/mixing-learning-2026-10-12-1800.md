# ADR: Gaussian prior for vocal-FX search stays shadow-only

- **Date:** 2026-10-12
- **Status:** accepted for research queue; no runtime promotion
- **Update:** `ML-2026-10-12-1800`

## Context

The WASPAA 2025 DiffVox follow-up adds a full-covariance Gaussian parameter prior to ST-ITO. It reports better architecture-aligned objectives, but the preset distribution is explicitly non-Gaussian and the retained 16-listener comparison is not significant (`p=.117`). The renderer, data and oracle are topology-conditioned.

## Decision

Treat the prior as an optional shadow-search regularizer only. Do not use the published mean, alpha or recovered controls as a production vocal preset. Any test must change only the prior term, keep raw measurement renders separate from BS.1770-matched listening copies, and include both centre and deliberate-outlier targets.

## Consequences

- `RULE-ML-20261012-1500-01` remains `auto_apply:false` and gains an outlier-cancellation gate.
- `EXP-ML-20261012-1800-01` is `not_run_blocked` until protocol/checkpoint/data parity and a frozen manifest exist.
- Listener choice and protected qualities override PMSE or embedding gains.
- No DSP/runtime/model/audio change is authorized.

