# ADR: Bound group-delay manipulation to offline creative research

- **Decision ID:** `ADR-ML-2026-10-02-0900`
- **Status:** accepted for knowledge-base policy; no runtime implementation
- **Date:** 2026-10-02
- **Related update:** `ML-2026-10-02-0900`

## Decision

Keep Giant-FFT group-delay manipulation as an offline creative-research candidate only. Do not treat it as transparent time editing, multichannel microphone phase alignment, bleed repair, or an automatic mixing action. Keep pitch-aligned time-frequency representation methods in a separate harmonic/arrangement domain.

## Rationale

The reviewed DAFx26 group-delay paper demonstrates coherent temporal displacement mainly on isolated or non-overlapping pitched events and explicitly states that shared frequency bins cannot be moved independently. It includes qualitative examples but no controlled listener test or independent reproduction. The PTFR paper optimizes pitch-representation distances on synthetic/internal data and supplies no waveform timing or mix-quality evidence.

## Consequences

- Any group-delay audio trial requires separate approval, a project-owned isolated source, raw safety renders and a randomized loudness-matched A/B.
- Paper-specific bin counts are not portable defaults.
- PTFR/TIV distances must not control mic alignment, EQ, gain, dynamics or mix-bus processing.
- `auto_apply:false`; `EXP-ML-20261002-0900-01` remains `not_run`.
