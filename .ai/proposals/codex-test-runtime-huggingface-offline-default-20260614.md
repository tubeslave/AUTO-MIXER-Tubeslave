# Proposal: Test runtime HuggingFace offline default

## 1. Short thesis

Local pytest should force optional HuggingFace-backed model lookups offline so
the standard validation command does not hang in restricted environments.

## 2. What I understood

The active work still targets the supervised WING Rack MVP, but validation is a
prerequisite for landing that work. The previous full pytest run stalled inside
`huggingface_hub` while optional perceptual MERT support tried to resolve a
model over the network.

## 3. Proposed solution

Set `HF_HUB_OFFLINE`, `TRANSFORMERS_OFFLINE`, `HF_DATASETS_OFFLINE`, and
`HF_HUB_DISABLE_TELEMETRY` to `1` from `tests/conftest.py`, and add a regression
that MERT fallback works without per-test `local_files_only`.

## 4. Likely files to touch

- `tests/conftest.py`
- `tests/test_perceptual_evaluation.py`
- Council ADR/report files

## 5. Alternatives considered

Changing production config to `local_files_only: true` would avoid downloads but
would also alter runtime operator configuration. The test fixture is narrower
and preserves production configurability.

## 6. Risks

Tests that intentionally need online model access would need to opt out, but this
repo's test contract already favors no hardware and no network.

## 7. Test plan

Run the perceptual test module and the focused WING validation suites.

## 8. Where the other agent may disagree

Kimi may prefer changing the default product config. The narrower fixture avoids
surprising production behavior while still making validation deterministic.
