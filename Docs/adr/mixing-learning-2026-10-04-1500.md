# ADR — Mixing Learning search 2026-10-04 15:00 MSK

- Status: accepted as source-grounded update
- Update: `ML-2026-10-04-1500`
- Preceding update: `ML-2026-10-04-1200`
- Preceding successful search: `ML-2026-10-04-0900`

## Decision

Add `DAFx26_challenge_87` as a new full-text source, merge the rediscovered organiser report into existing card `SRC-ML-20260924-0900-03`, and correct the Matrix-Pencil evidence state from “official hidden result absent from the participant paper” to “official result published in the existing organiser report”. Do not treat count-density, Matrix Pencil, or their benchmark rankings as mixing/reverb defaults.

Require paired per-mode and reconstructed-response scoring, explicit absolute-gain bias, count error and resolvable/overlap frequency bands for any future modal-estimator comparison. Keep all candidate rules `auto_apply:false` and all planned experiments `not_run`.

## Evidence

- Count-density B1/B2 were trained on `327,680` same-generator synthetic IRs; local results are strong but remain in-distribution.
- Official hidden-set B1/B2 totals are `0.328/0.337`; gain errors remain `0.847/0.925`.
- Matrix Pencil `3-B` official total is `0.743`, gain error `0.970`, response rank `7/10`.
- The official response metric reorders methods and exposes a roughly `30 dB` bias in B2 that the per-mode metric largely misses.
- The named public count-density repository is currently a no-code, no-licence placeholder.

## Consequences

- `RULE-ML-20261004-0900-01` is narrowed to research-comparator status.
- `RULE-ML-20261004-1500-01` formalizes paired modal evaluation; it is not an audio-processing rule.
- `EXP-ML-20261004-1500-01` remains blocked until rights-clear implementations and measured IRs exist.
- Two clipping/master-bus videos enter the queue as metadata-only; no claims are extracted.
- The next immediately executable audio test remains `EXP-ML-20261001-1800-01` (parallel-drum density).

No audio, code, DSP, runtime or model state changed.
