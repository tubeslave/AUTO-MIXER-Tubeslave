# Stem Offline CI Test Run

This branch exists to trigger the `Stem Offline Test` workflow after the workflow was merged into `master`.

Expected artifact:

- `stem-training-loop-result`
  - `stem_training_fixture.wav`
  - `stem_training_fixed_mix.wav`
  - `stem_training_report.json`

Safety expectations:

- `OSC_DISABLED=true`
- mock splitter only
- no live audio thread
- no OSC commands
