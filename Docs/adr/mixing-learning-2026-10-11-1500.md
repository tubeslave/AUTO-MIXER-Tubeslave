# ADR: measured-room interpolation and licence gate

- **ID:** ADR-ML-2026-10-11-1500
- **Status:** accepted
- **Update:** ML-2026-10-11-1500

## Context

The DAFx-26 Bunkervik demo provides a useful engineering example of aligning independently estimated room modes before interpolating position-dependent residues. Its objective fit metrics do not include perceptual or music-production validation. The official NEMUS binary is free, but its EULA expressly prohibits AI/ML training or validation with the software or generated audio and prohibits automated learning or cloning of its character.

## Decision

Add the paper as a fully read source and keep its conclusion bounded to the correspondence problem in dynamic measured-room interpolation. Do not treat NRMSE or CPU cost as evidence of better drum ambience. Exclude the NEMUS binary and its output from automixer training or validation; do not download or execute it in this update.

Create only a shadow candidate rule and a blocked manual multitrack A/B plan. It may be tested later with an already licensed local processor, one factor at a time, after the current “Пена” blind listening task. Keep auto_apply:false.

Keep both new YouTube items metadata-only because the visible transcript panels produced no timestamped rows. Do not extract settings or techniques from titles, descriptions or chapters.

## Consequences

The knowledge base gains a source-grounded caution about coherent room interpolation and an explicit licence boundary without changing the DSP engine or using restricted software. The immediate executable queue remains unchanged: EXP-ML-PENA-AB15-LISTEN-20261010-1800 is still first.
