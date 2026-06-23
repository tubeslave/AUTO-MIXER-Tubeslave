# Brief: Test runtime HuggingFace offline default

## Task

Make the local pytest runtime deterministic when optional perceptual MERT /
HuggingFace-backed components are unavailable or network access is restricted.

## Scope

- Do not change live mixer behavior.
- Keep optional MERT/MuQ features as fallbacks, not test-time network blockers.
- Set HuggingFace/Transformers offline environment defaults for pytest.
- Add a regression that MERT fallback works without explicitly passing
  `local_files_only: true`.

## Test Plan

- Run `tests/test_perceptual_evaluation.py`.
- Run focused WING rail suites to ensure the active MVP work remains intact.
