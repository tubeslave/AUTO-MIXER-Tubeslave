# STUDIO decision — AutoMaster standards meter integration

Date: 2026-09-23
Status: machine-accepted, audio acceptance pending human listening

## Accepted engineering result

PR #113 was merged after both repository workflows passed. `AutoMaster` now uses the accepted `StudioMasteringMeter` in its actual built-in, fallback, and reference mastering output paths:

- gated integrated LUFS drives operational loudness targeting;
- 4x reconstructed true peak drives actual safety attenuation;
- `MasteringResult.lufs` reports integrated LUFS in real mastering results;
- legacy RMS `_estimate_lufs()` and sample-peak `_limit()` remain compatibility-only helpers.

Validation evidence:

- Stem Offline Test run `35913849318`: success.
- Tests run `35913849473`: success on Python 3.10, 3.11, and 3.12.
- Python 3.11 job: `951 passed, 2 skipped, 1 warning`.
- Integration tests explicitly cover an intersample overshoot where sample peak is already below -1 dBFS but reconstructed true peak is above -1 dBTP.

Merged code commit: `a5b472fcbcf5290510667a304e9da3774fee54b2`.

## Audio acceptance

No user mix/master was rendered or promoted in this task. The changed mastering path can make audible level/ceiling changes, therefore any future candidate produced by it remains pending human A/B before baseline promotion.

The balance-transparent CLEAN candidate for “Птицы” remains outside baseline until human listening acceptance.

## Next STUDIO goal

Wire the same standards evidence into `tools/offline_agent_mix.py::master_process` reporting and acceptance logic so every offline master candidate records pre/post integrated LUFS, sample peak, dBTP, crest change, safety attenuation, and a machine verdict that cannot promote a subjective master without human review.