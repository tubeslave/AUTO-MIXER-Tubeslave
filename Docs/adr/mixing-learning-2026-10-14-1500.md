# ADR: MPASCNet remains a checkpoint-gated restoration hypothesis

- **ID:** `ADR-ML-2026-10-14-1500`
- **Status:** accepted for research queue; not runtime policy
- **Date:** 2026-10-14
- **Related update:** [ML-2026-10-14-1500](../mixing_learning_updates/ML-2026-10-14-1500.md)
- **Previous search:** [ML-2026-10-14-0900](../mixing_learning_updates/ML-2026-10-14-0900.md)

## Context

MPASCNet reports strong objective gains for synthetic low-bitrate MP3 restoration and publishes code plus a checkpoint card. The paper has no listening test, uses short excerpts, treats stereo channels independently and does not establish transfer to rock mastering. The inspected validation generator samples from the merged training-data pool without recording source identity. The checkpoint card is visible only at metadata level: file, immutable revision and checksum were not exposed. Code and checkpoint cards also declare different licences.

## Decision

Keep MPASCNet as a **shadow-only, checkpoint-gated restoration candidate**. It is not a mixing/mastering rule and must not be auto-applied. Before any inference, record the exact checkpoint manifest and rights, lock the local environment and establish validation/source provenance. The first allowed experiment is restoration off/on on a locally generated MP3 degradation of project-owned lossless audio, with the lossless source retained as reference. Stereo, transients, timbre, loudness and true peak are protected outcomes; objective metrics are diagnostics, not acceptance criteria by themselves.

## Consequences

- New rule `RULE-ML-20261014-1500-01` remains `auto_apply:false`.
- New experiment `EXP-ML-20261014-1500-01` remains `not_run_blocked`.
- A metric-only improvement cannot promote the method.
- Code licence, checkpoint-card licence and dataset/audio rights remain separate records.
- No audio, DSP engine, runtime or model repository was changed.

