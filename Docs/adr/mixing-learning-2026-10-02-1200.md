# ADR: Require same-path nulls and objective-specific PTFR claims

- **Decision ID:** `ADR-ML-2026-10-02-1200`
- **Status:** accepted for knowledge-base policy; no runtime implementation
- **Date:** 2026-10-02
- **Related update:** `ML-2026-10-02-1200`

## Decision

Any future Giant-FFT group-delay A/B must compare variants through the same frozen analysis/synthesis path, pass a `Δt = 0` null preflight and change only one displacement expressed in time. PTFR/TIV optimizer claims must name the exact representation, objective and data regime; neither a universal optimizer nor monotonic benefit from `2-opt` is accepted.

## Rationale

The Giant-FFT paper's exact reconstruction statement concerns the unchanged analysis/synthesis pass, while its transformations are supported by qualitative examples without a matched windowed baseline or listener test. Bin-count parameters depend on sample rate, padded FFT length and duration. In the PTFR table, different methods win vector and matrix objectives, and `2-opt` worsens two reported musical comparisons even while improving others. Both official demos are same-author qualitative evidence without verified public code, dataset or separate audio-asset licence.

## Consequences

- Freeze and log the complete Giant-FFT segmentation manifest before rendering A/B.
- Use untouched source audio only as diagnostic C; A and B must traverse the same path.
- Reject the experiment before listening if the no-op null or grouping-stability gate fails.
- Keep PTFR/TIV out of timing, phase, EQ, gain, dynamics and mix-quality controls.
- Do not ingest or rehost demo audio based only on the article's CC BY 4.0 statement.
- `auto_apply:false`; all affected experiments remain `not_run`.
