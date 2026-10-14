# ADR — Gate compressor-objective comparisons on manifest, split and implementation parity

- **ID:** `ADR-ML-2026-10-14-1200`
- **Status:** accepted for research queue; not runtime policy
- **Date:** 2026-10-14
- **Related update:** [ML-2026-10-14-1200](../mixing_learning_updates/ML-2026-10-14-1200.md)
- **Source search:** [ML-2026-10-14-0900](../mixing_learning_updates/ML-2026-10-14-0900.md)

## Context

The DAFx26 study shows that direct gain-control error can distinguish fitted compressor models that waveform losses rank similarly. Its design deliberately fits and evaluates the same 30-second examples and therefore does not demonstrate held-out generalisation or listening preference.

The associated Hugging Face viewer flattens the repository's audio files into 40,577 rows, while the scientific dataset contains 19,710 input/output/control examples linked through metadata. Random row splits would leak source clips repeated across hardware permutations. The dataset card also names a calibration-summary file not visible in the checked metadata tree.

The public `torchcomp` dependency uses an RMS detector. The paper's peak detector, stereo maximum and shared gain modification is not present in the pinned public branch, so code parity is unresolved.

## Decision

1. Treat the automatic Hugging Face split as file browsing only, not as an experiment split.
2. Build paired examples through pinned metadata tables and hashes.
3. Split by verified original-song identity before expanding hardware permutations; evaluate unseen compressor settings separately.
4. Freeze control-trajectory alignment independently from audio I/O alignment and never tune lag on test material.
5. Require a public or reconstructed paper-parity patch plus unit tests before running the loss A/B.
6. Report direct gain error, waveform metrics and loudness-matched listening as distinct endpoints.
7. Keep hardware settings and dataset augmentation values as reproduction metadata, not mixing presets.
8. Keep `RULE-ML-20261014-0900-01` `auto_apply:false` and `EXP-ML-20261014-0900-01` `not_run_blocked`.

## Consequences

- The prior experiment plan is narrowed and delayed until manifest, provenance, alignment and implementation gates are satisfied.
- `Solid State Bus-Comp` is reused as an existing split-design comparator rather than duplicated or counted as independent evidence for the direct-control metric.
- No dataset audio is downloaded and no audio, DSP, runtime, model, build or training changes are authorised.
- The next executable project task remains `EXP-ML-PENA-AB15-LISTEN-20261010-1800`.

