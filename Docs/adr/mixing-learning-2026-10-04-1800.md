# ADR — Mixing Learning deep review 2026-10-04 18:00 MSK

- Status: accepted as source-grounded update
- Update: `ML-2026-10-04-1800`
- Preceding search: `ML-2026-10-04-1500`

## Decision

Treat Count-Density B1/B2 agreement between the participant-controlled 100-IR set and organiser-controlled 16-IR hidden set as matched-generator repeatability, not measured-plate or external generalisation. Prefer B1 only as the current benchmark comparator: it is smaller and better on reported total, gain and reconstructed-response metrics, while B2 improves decay only.

Strengthen modal-estimator evaluation with bulk-gain, response, overlap-band and artifact-reproducibility gates. Add one low-to-moderate-confidence hypothesis: explicit calibrated input-scale conditioning may improve gain recovery. Test it as a single-factor ablation only; do not promote it to DSP or mixing behaviour.

## Evidence

- Local/hidden totals: B1 `0.318/0.328`, B2 `0.330/0.337`, all from the same generator/ranges.
- Official response: B1 `0.882` rank 3; B2 `2.983` rank 10, with a representative approximately `30 dB` low response.
- B1 is about `3.1 M` parameters; B2 about `99.6 M` real-parameter equivalents.
- GitHub revision `4cf5307` contains only a 94-byte README and no licence or method files.
- Hugging Face returned PyTorch/MIT metadata but no README/file manifest through the connector; no checkpoint was downloaded or run.

## Consequences

- `RULE-ML-20261004-1500-01` gains checkpoint/preprocessing/environment/evaluation-manifest gates.
- New `RULE-ML-20261004-1800-01` remains `auto_apply:false`.
- `EXP-ML-20261004-1800-01` remains `not_run` and blocked by missing reproducible method artifacts and rights-clear test partitions.
- Video and reference queues are unchanged.
- The next executable project test remains `EXP-ML-20261001-1800-01` (parallel-drum density).

No audio, code, DSP, runtime or model state changed.
