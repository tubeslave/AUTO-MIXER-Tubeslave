# ADR — Gate time-stretch evidence by executable parity and content class

- **ID:** `ADR-ML-2026-10-13-1800`
- **Status:** accepted for research queue; not runtime policy
- **Date:** 2026-10-13
- **Related update:** [ML-2026-10-13-1800](../mixing_learning_updates/ML-2026-10-13-1800.md)
- **Source search:** [ML-2026-10-13-1500](../mixing_learning_updates/ML-2026-10-13-1500.md)

## Context

The search update retained a DAFx26 extrema/keyframe time-stretch paper, the Capicola repository and a planned bypass-versus-1.0x null. A deeper code audit found that the current Capicola 1.1.0 live path adds transient-driven multi-grain slicing that the repository itself excludes from the paper’s public baseline examples. Public host tests use synthetic tolerance checks rather than a sample-null or music transparency test.

An independent full subjective dataset paper shows that TSM quality depends on content and ratio direction. Its material is short, mono and normalised; it does not establish stereo/multitrack or project-rock transfer. A later objective metric currently has only metadata/abstract-level review.

## Decision

1. Treat paper algorithm, published examples and current firmware as separate evidence objects.
2. Do not infer current-firmware quality from the paper listening result without executable/mode parity.
3. Require a frozen executable/commit/mode and target-host test pass before audio evaluation.
4. Run an exact 1.0x round-trip control separately on exposed vocal and drum/percussion material.
5. Permit only one constant delay correction for raw residual diagnostics; do not time-warp or gain-fit the raw null.
6. Create loudness-matched, true-peak-safe copies separately for blind listening.
7. Test one fixed non-unity ratio only after both unity controls pass.
8. Do not use paper defaults, Capicola defaults or abstract-level objective scores as project presets or acceptance thresholds.
9. Keep the rule `auto_apply:false` and experiment `not_run_blocked`.

## Consequences

- The earlier vocal-only null is expanded to two content classes.
- Current Capicola cannot be described as validated by the DAFx26 baseline evaluation.
- Host tests remain useful preflight evidence but do not replace project audio measurement/listening.
- AGPL/legal review is required before any Capicola integration.
- No DSP, audio, runtime or model change is authorised.
- The next executable task remains `EXP-ML-PENA-AB15-LISTEN-20261010-1800`.
