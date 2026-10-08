# ADR: separate crest factor, peak-to-loudness and LRA

- date: `2026-10-08`
- status: `accepted_documentation_only`
- update: `ML-2026-10-08-1200`
- preceding search: `ML-2026-10-08-0900`

## Context

The 09:00 search added a vendor tutorial and a candidate rule requiring a measurement contract for crest factor. The next deep review had to determine which parts are supported by open primary standards and prevent ambiguous use of `crest factor` for quantities mixing true peak, RMS and loudness.

## Decision

1. Add ITU-R BS.1770-5 as a full primary standard Source Card and upgrade the existing EBU Tech 3341/3342 cards instead of duplicating them.
2. Treat mathematical crest factor, peak-to-loudness and LRA as separate metric families.
3. Reserve `crest factor` for a declared peak/RMS ratio under one fixed window and channel aggregation.
4. Label true-peak-minus-short-term-loudness as the project-local descriptor `PLD-S`; it is not a standard crest factor.
5. Do not use LRA to decide an event/short-section attack A/B; values observed before 60 s are provisional.
6. Revise `RULE-ML-20261008-0900-01` and `EXP-ML-20261008-0900-01`; keep `auto_apply:false` and `not_run`.
7. Keep `EXP-ML-20261001-1800-01` as the next executable experiment.

## Consequences

The database gains a standards-grounded vocabulary and a reproducible measurement contract without importing broadcast targets as music-production goals. Vendor genre ranges remain illustrative only. No production parameter, audio asset, DSP implementation, model or runtime is changed.
