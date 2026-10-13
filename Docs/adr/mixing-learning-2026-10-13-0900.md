# ADR: keep reference-free pitch correction sparse, shadow-only and harm-aware

- **Date:** 2026-10-13
- **Status:** accepted for research queue; no runtime promotion
- **Update:** `ML-2026-10-13-0900`

## Context

A fresh ICASSP 2027 submission reports that a factorised note-edit model plus a tonal prior improves exact-note accuracy over hard key snapping and BERT-APC on a private paired singing dataset. The same results show an important trade-off: the full method repairs more erroneous notes but changes more initially correct notes than BERT-APC. Its audio renderer, data and project artifacts do not provide a reproducible full-mix rock validation, and listener means are not accompanied by reported inferential statistics.

## Decision

Admit only a bounded, shadow-only rule candidate:

1. decide whether one predeclared note should be edited before choosing an offset;
2. retain bypass as the explicit baseline;
3. evaluate repair and clean-note harm separately;
4. test one sealed note-region/offset manifest as the sole A/B factor;
5. preserve raw measurement renders and create separate BS.1770-matched blind copies;
6. reject edits that damage timing, consonants, formants, vibrato, transitions, harmony role or emotional intent;
7. keep `auto_apply:false`.

Do not transfer the paper's tonal-prior weight, hard-snap a performance, infer artistic correctness from a professional reference alone, or interpret project-page demos as independent evidence.

## Consequences

- `RULE-ML-20261013-0900-01` is candidate-only.
- `EXP-ML-20261013-0900-01` remains `not_run_blocked` until a sealed note/target/offset manifest and listening key exist.
- The existing executable listening test remains first in the queue.
- No audio, DSP, runtime, repository engine or model was changed.

