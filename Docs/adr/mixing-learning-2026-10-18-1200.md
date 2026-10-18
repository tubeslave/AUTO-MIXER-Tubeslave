# ADR: Bound PADP to an uncertainty-aware metric stress test

- Status: `accepted_for_research_workflow`
- Date: 2026-10-18
- Update: [ML-2026-10-18-1200](../mixing_learning_updates/ML-2026-10-18-1200.md)
- Scope: research and experiment planning only
- Runtime effect: none

## Context

PADP is a useful warning that waveform-different signals judged near-transparent can move objective metrics. The complete paper and targeted primary-source audit also expose limits that prevent operational promotion: DIF2 was not directly tested; APX assignment and random seed are missing; MP3 implementation, loudness compensation and alignment are unspecified; the SCD recommendation contains opposite inequality language; metric SRCCs have no uncertainty or held-out validation; DeePAQ is not an independent-lab confirmation; and the proposed data-augmentation use was not tested.

ITU-R BS.1534-3 treats MUSHRA as an intermediate-impairment method and asks for controlled/randomized design, listener reliability detail, inferential analysis, effect sizes and uncertainty reporting. ODAQ's full stereo/26-listener corpus is broader than PADP's undisclosed 12-item mono subset and cannot be inherited as evidence.

## Decision

Retain PADP only as a research diagnostic for candidate metric robustness. Revise the rule to require immutable renderer/codec/metric provenance, identity/alignment preflights, disjoint held-out material, raw evidence, separate attenuation-only BS.1770-matched blind copies, per-item/per-listener reporting and uncertainty-aware acceptance criteria. Do not use PADP parameters or the paper's `<10` mean-difference threshold as automixer defaults.

Revise the planned experiment to A/B one pinned SCD residual at `lambda=0.6`, explicitly as paper provenance rather than a default. The experiment remains blocked until rights, builds, alignment, preregistration and project-owned material are resolved. It tests metric stability, not audio improvement.

All rules remain `auto_apply:false`; the experiment remains `not_run_blocked`.

## Consequences

- One PADP Source Card is revised.
- One official-standard Source Card, one ODAQ Artifact Audit and ten Knowledge Cards are added.
- One candidate rule and one blocked experiment are revised.
- No video, reference or multitrack is promoted.
- No audio, code, model, metric, DSP or runtime change occurs.
- The next executable experiment remains `EXP-ML-PENA-AB15-LISTEN-20261010-1800`.
- Citation checking remains partial; absence of a discovered correction or contradiction is not confirmation.
