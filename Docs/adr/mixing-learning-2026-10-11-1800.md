# ADR: measured-room interpolation requires correspondence-specific evidence

- **ID:** ADR-ML-2026-10-11-1800
- **Status:** accepted
- **Update:** ML-2026-10-11-1800
- **Supersedes in part:** ADR-ML-2026-10-11-1500

## Context

The 15:00 search converted the Bunkervik demo into a generic creative candidate: compare a static rock-drum room with moving receiver-position modulation in an arbitrary licensed processor. Deep review found that this is a construct mismatch.

The cited I3DA-2025 paper establishes modal extraction and efficient resynthesis of individual RIRs. It does not validate interpolation between positions. A separate, independent Acta Acustica study supplies listener evidence for a different Ambisonic method in which direct sound and early reflections are put into correspondence before interpolation. That study supports the general correspondence principle, but not the Bunkervik common-pole implementation or creative modulation in an arbitrary reverb.

## Decision

Narrow RULE-ML-20261011-1500-01 to correspondence-aware interpolation of a licensed multi-position RIR/ARIR set. Do not transfer it to generic static-versus-moving drum ambience or automatic processing.

Replace EXP-ML-20261011-1500-01 with a parity-gated, one-factor test:

- A: naive distance-weighted interpolation of the same RIR triplet;
- B: identical interpolation with direct-arrival TOA alignment only;
- retain raw renders and create separate BS.1770-matched listening copies;
- measure spectral-notch motion, early energy, attack, loudness, true peak, LR correlation and mono delta;
- require a recorded Dmitry verdict before any project rule promotion.

Keep the experiment not_run_blocked until an approved local implementation and legally reusable dataset are audited. The NEMUS Bunkervik binary remains excluded under its EULA. Do not import the paper-specific grid spacings, event count, update size, thresholds or CPU result as presets.

## Consequences

The knowledge base retains the useful engineering principle but removes an unsupported rock-mixing transfer. The independent listening study is stored as adjacent evidence, not direct replication. No audio, binary, dataset, DSP, runtime or model is changed.

The immediate executable queue remains EXP-ML-PENA-AB15-LISTEN-20261010-1800. The RIR-alignment experiment stays below it and blocked. All rules remain auto_apply:false.
