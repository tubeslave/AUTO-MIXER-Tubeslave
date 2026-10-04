# ADR — Mixing Learning search 2026-10-04 09:00 MSK

- Status: accepted as source-grounded update
- Update: `ML-2026-10-04-0900`
- Preceding search: `ML-2026-10-03-1500`

## Decision

Retain the diagonal complex-SSM/Matrix-Pencil paper as engineering evidence for an experiment design, not as proof of perceptual plate-reverb quality. Retain the Klein-bottle plate work and two videos in the deep-review queue at their actual reading depth. Add one transcript-grounded, non-automatic workflow rule: test a fixed stereo-bus chain late, against an already viable bypassed mix, with randomized BS.1770-matched copies.

## Guardrails

- Reconstruction metrics do not substitute for held-out measured IRs or listening tests.
- Parameter-recovery degradation, synthetic/in-sample design and missing paper-specific implementation remain explicit limitations.
- YouTube ASR numbers/plugin spellings are not defaults and require verification.
- No audio was downloaded or analysed; no code was run; no DSP/runtime/model change was made.
- All rules remain `auto_apply:false`; all experiments remain `not_run` until Dmitry evaluates controlled renders.

## Consequence

The next executable project experiment remains `EXP-ML-20261001-1800-01` (parallel-drum density), followed by `EXP-ML-20261002-1500-01` (pre-reverb compression). The new bus-chain test is third, after preregistration.
