# ADR: role-aware bass layers remain shadow candidates

- **ID:** `ADR-ML-2026-10-12-0900`
- **Status:** accepted
- **Update:** `ML-2026-10-12-0900`

## Context

The fully read Produce Like A Pro article presents five professional bass-mixing examples. Across them, the bass role changes by artist and section; clean low-frequency weight, articulation and distorted character may be carried by different paths; multi-source combinations require alignment checks; and compression judgments require level matching and awareness of peak control already produced by distortion. The article is professional editorial evidence, not a controlled study, and the five examples share one publisher and selected-course context.

The linked and newly discovered YouTube pages did not yield permitted timestamped transcripts. A watch-page attempt stopped at a human-verification challenge. No bypass, alternative transcript scrape or inferred settings were used.

## Decision

Add the article as one fully read Source Card and merge its 2025 predecessor and linked video into one version family. Add only shadow candidate rules: role-conditioned optional bass layers and an alignment gate for combined paths. Treat all quoted frequency regions and named processors as source-specific hypotheses, not presets.

Plan one lower-priority project A/B in which a completely frozen distorted return is the only enabled/disabled factor. Keep it blocked until an eligible “Пена” bass source, baseline and return are frozen, and until the already executable AB15 listener test is completed. Keep `auto_apply:false`.

Retain three new videos as `queued_source` at metadata-only depth. Do not copy article findings into the video cards as if their transcripts had been read.

## Consequences

The shared base gains a concrete, source-grounded bass workflow without promoting a universal chain or automatic preset. The scientific search records a genuine negative result after deduplication. The next executable experiment remains `EXP-ML-PENA-AB15-LISTEN-20261010-1800`; the new bass experiment is `not_run_blocked`. No audio, DSP, runtime, repository code, model or training state changes.
