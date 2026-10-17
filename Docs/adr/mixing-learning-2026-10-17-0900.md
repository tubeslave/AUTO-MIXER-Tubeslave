# ADR: Record FNO sample-rate invariance as bounded simulated-circuit evidence

- Status: accepted_for_research_workflow
- Date: 2026-10-17
- Update: [ML-2026-10-17-0900](../mixing_learning_updates/ML-2026-10-17-0900.md)
- Scope: research and experiment planning only
- Runtime effect: none

## Context

The search found an unindexed DAFx 2026 paper reporting one Fourier Neural Operator across several sample-rate grids. The result is relevant to virtual-analog portability, but it is based on one simulated Big Muff Pi input stage, a small guitar/sweep dataset and objective metrics without listeners, hardware validation or a public reproducibility artifact.

## Decision

Add the paper as a full-text Source Card and retain only bounded claims:

- sample-rate-grid operation is not device, circuit, material or perceptual transfer;
- FNO-versus-LIDL conclusions are metric-dependent;
- the 24 kHz result has no LIDL-RNN comparator;
- internal upsampling is an experiment candidate, not a production setting;
- startup, overlap-add, latency, CPU and listener gates are required.

Add one blocked one-factor A/B plan for `rho=1` versus `rho=2` at 24 kHz. Keep `auto_apply:false` and `not_run_blocked` until a parity artifact, manifest and simulator reference exist.

The FDverb paper and two videos remain metadata-only queue items. Do not promote any of their techniques until full text or authorized transcripts are actually read.

## Consequences

- Seven Knowledge Cards and one candidate rule enter the source-grounded research layer.
- No DSP setting, runtime rule, model, dependency or audio asset changes.
- The next executable experiment remains `EXP-ML-PENA-AB15-LISTEN-20261010-1800`.
- The FNO experiment remains below executable work because the public artifact and parity fixture are missing.
- Citation checking remains partial; absence of a found replication or contradiction is not confirmation.
