# STUDIO ADR: Bass Compression Director v1

## Decision

Introduce a bass-specific compression proposal layer instead of copying the vocal result or reusing the generic role formula unchanged.

The director derives a small three-candidate family from the source's measured macro attack and inter-event spacing, but constrains the result to bass-appropriate timing:

- attack: 5–25 ms
- release: 75–180 ms
- ratio: 3.5:1 to 4:1
- max GR: 5 dB
- threshold calibrated against **actual active-P95 gain reduction** of the shared causal compressor

It deliberately returns no winner or ranking.

## Bass-specific technical objective

The technical screen uses fixed event windows derived once from the pre-compression bass. It compares each rendered candidate with an explicit reference bass and checks two level-invariant quantities:

1. note/body RMS spread (P90-P10), where an increase above +0.05 dB is rejected;
2. median attack-peak to body-RMS contrast, where more than 0.35 dB loss is rejected.

A pass only makes the candidate eligible for full-session rerender and human A/B. It never means musical acceptance or baseline promotion.

## Why this was needed

The existing generic Compression Director produced release times of roughly 277–553 ms and targeted 1.8–3.4 dB active-P95 GR on the Belye Stai bass. Although those candidates converged technically, all three increased bass body-level spread relative to the current bass (2.781 dB -> 4.626 / 3.916 / 3.238 dB).

The bass-specific v1 family instead shortened recovery according to inter-event spacing and kept materially more control. On the same full 207 s source, after matching the candidate source level to the current bass:

- `preserve_transient`: body spread delta -0.009 dB; attack/body contrast delta -0.106 dB
- `balanced`: body spread delta -0.134 dB; attack/body contrast delta -0.096 dB
- `control`: body spread delta -0.170 dB; attack/body contrast delta -0.087 dB

All three passed the local bass-specific technical screen.

Each was then rerendered through the complete Belye Stai session while keeping the user-approved vocal v2 frozen. None triggered the protected full-mix guards used for vocal intelligibility, harshness, punch or stereo width. The observed full-mix deltas are tiny and are evidence of contextual safety, not evidence of a musical winner.

## Boundaries

- Studio/offline only.
- No learned audio model or paid service.
- No mastering change.
- No automatic winner.
- No audio baseline promotion.
- Human listening remains required before choosing among surviving candidates.

## Next step

Produce a level-matched contextual A/B for the surviving bass family and collect human listening preference. After that, repeat the instrument-specific pattern for drums, starting with kick/snare because their attack/body/recovery objectives differ again from bass and vocal.
