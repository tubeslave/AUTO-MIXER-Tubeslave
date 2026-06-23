# ADR: Test runtime HuggingFace offline default

## 1. Context

The standard pytest command previously stalled inside `huggingface_hub` during
optional perceptual MERT initialization. This repository should validate without
hardware or network, especially while preparing supervised WING proof gates.

## 2. Options considered

- Change production perceptual config to local-only.
- Force offline HuggingFace/Transformers environment defaults only in pytest.

## 3. Decision

Set offline HuggingFace/Transformers environment defaults in `tests/conftest.py`
and add a regression that MERT fallback works without an explicit
`local_files_only` override.

## 4. Why this won

It fixes local validation determinism without changing production configuration
or live mixer behavior.

## 5. Rejected alternatives

Changing production defaults was rejected because operators may intentionally
configure an online model cache/update path outside local pytest.

## 6. Implementation plan

- Set `HF_HUB_OFFLINE`, `TRANSFORMERS_OFFLINE`, `HF_DATASETS_OFFLINE`, and
  `HF_HUB_DISABLE_TELEMETRY` in `tests/conftest.py`.
- Preserve those values across autouse fixture environment isolation.
- Add a perceptual fallback regression.

## 7. Test plan

- `tests/test_perceptual_evaluation.py`
- Focused no-WING/handoff suite
- Project Director rails
- Hygiene checks

## 8. Risks and rollback

Rollback is limited to test fixtures and one regression test. No live WING,
OSC, mixer, or DSP runtime behavior is changed.
