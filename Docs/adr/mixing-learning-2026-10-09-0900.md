# ADR — PEACE remains a shadow retrieval candidate

- date: 2026-10-09
- status: accepted for research queue
- update: [ML-2026-10-09-0900](../mixing_learning_updates/ML-2026-10-09-0900.md)

## Context

PEACE connects effected audio to Faust DSP code in a shared embedding. Its results are strong enough to justify a version-pinned offline retrieval study, but performance varies widely by effect family and the paper contains no listener study. The author repository also documents scope restrictions and a proceedings/arXiv preprocessing difference.

## Decision

1. Treat PEACE as a shadow/offline retrieval candidate only.
2. Do not use embedding similarity as mix quality, listener preference or an automatic application score.
3. Require declared checkpoint, encoder, preprocessing, padding/version, parameter visibility and gallery.
4. Report per-effect-family results and explicit inactive-effect/unsupported-graph failures.
5. Keep every related rule `auto_apply:false`; do not download weights or run paid compute in the learning automation.
6. Permit a later free local reproduction only after an approved test corpus, licence review and resource budget exist.

## Consequences

- Reverb retrieval evidence cannot be transferred to compressor, gate, limiter or distortion.
- Parameter-masked retrieval cannot be treated as parameter estimation.
- Production DSP and runtime remain unchanged.
- `EXP-ML-20261009-0900-01` stays `not_run_blocked`.
