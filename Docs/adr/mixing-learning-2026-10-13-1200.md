# ADR — Mixing Learning deep review 2026-10-13 12:00 MSK

- **Status:** accepted for knowledge-base documentation; no automatic processing authority
- **Decision ID:** `ADR-ML-2026-10-13-1200`
- **Context:** follow-up to `ML-2026-10-13-0900`

## Decision

Treat context-aware target-note inference as a bounded research signal only. Separate three claims that the literature often joins: target-note selection, acoustic rendering, and listener preference. Before any non-zero sparse pitch edit, require a renderer zero-shift null on project-owned audio. Retain objective pitch error as a diagnostic, never as the sole acceptance criterion.

## Evidence

Full reading of BERT-APC shows strong RPA gains under its data and metric, especially for highly detuned material, but leaves singer/song-disjointness unstated, uses a rating-level one-way ANOVA despite repeated listeners/excerpts, reports no expression-equivalence test and does not release a reproducible model package. Its public demo also has statistic drift from the current paper and acknowledges occasional TD-PSOLA clicks.

An independent full ICASSP paper shows that standard pitch metrics can rank systems opposite to perceived naturalness/fluency in its correction pipeline. That study is small and reference-based, so it supports an evaluation requirement rather than a processor choice.

## Consequences

- Revise `RULE-ML-20261013-0900-01` with a renderer-null prerequisite and perception/metric separation.
- Add `EXP-ML-20261013-1200-01` as `not_run_blocked`.
- Do not promote either BERT-APC or RF-SPC to automatic application.
- Do not claim preservation of expression from similar arithmetic MOS means.
- Require listener/excerpt-aware analysis for any future listening dataset.
- Keep `auto_apply:false`; no DSP, audio, runtime or model change.
