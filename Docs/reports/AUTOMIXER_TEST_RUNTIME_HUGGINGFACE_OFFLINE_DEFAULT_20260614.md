# Test runtime HuggingFace offline default

Date: 2026-06-14

## Summary

Pytest now defaults HuggingFace/Transformers integrations to offline mode so
optional perceptual MERT fallback does not block local validation on network
access.

## Safety impact

- No live mixer behavior changes.
- No live authority changes.
- Keeps standard validation aligned with no-hardware/no-network test execution.

## Validation target

`tests/test_perceptual_evaluation.py` now includes a regression proving MERT
fallback works when the test does not explicitly pass `local_files_only: true`.
