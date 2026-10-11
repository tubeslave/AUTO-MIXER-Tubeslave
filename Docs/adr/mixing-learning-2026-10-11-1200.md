# ADR: restoration metrics cannot bypass the multitrack listener gate

- **ID:** ADR-ML-2026-10-11-1200
- **Date:** 2026-10-11
- **Status:** accepted
- **Related search:** [ML-2026-10-11-0900](../mixing_learning_updates/ML-2026-10-11-0900.md)
- **Related deep review:** [ML-2026-10-11-1200](../mixing_learning_updates/ML-2026-10-11-1200.md)

## Context

The MSR Challenge reports Spearman `rho=0.9` between objective and subjective system rankings. Full review shows that this is a comparison of only five systems, with objective scores from a 1,000-clip reference set and subjective MOS from a different 500-clip blind real-world set. It is not per-clip or per-stem metric validation. MSRBench also shows strong phase/perceptual disagreement and describes restoration as increasingly generative when information is destroyed.

The exact October 2026 implementation remains unavailable. A prior same-team MIT repository exists, but implements the March HiFi++/LoRA challenge system, not the October Apollo-based method.

## Decision

1. MMSNR, Zimtohrli, FAD, SI-SDR and similar metrics remain diagnostic panels, not acceptance authorities.
2. Restoration targets must be named explicitly: dry tracks, processed group stems, or final mix.
3. Related repositories cannot substitute for exact parity code, weights, config and preprocessing.
4. Every candidate must pass a project-owned multitrack A/B with one changed factor, raw metric renders, separate BS.1770-matched listening copies and a recorded Dmitry verdict.
5. Transient sources require explicit attack, phase and bleed protection.
6. Rules remain `auto_apply:false`; blocked experiments remain `not_run_blocked`.

## Consequences

`EXP-ML-20261011-0900-01` is revised to use the 13 verified `Пена` processed stems, a frozen 13-to-8 mapping that nulls back to the premaster, and Stage-1 on/off as the only factor. FAD is not decisive per clip. The experiment remains blocked until the exact October artifact is version-pinned.

The next executable work remains `EXP-ML-PENA-AB15-LISTEN-20261010-1800`.
