# Enable Full MERT - Codex Proposal

## Short Thesis

Add the missing HuggingFace runtime dependencies and require the configured
MERT backend in live config so tests expose loading failures immediately.

## What I Understood

The operator needs the real MERT evaluator running during current tests, not the
lightweight fallback. The backend should continue applying mixer corrections
through existing safety rules, while MERT contributes shadow scoring and
decision visibility.

## Proposed Solution

- Add `transformers`, `safetensors`, and `nnAudio` to Python requirements.
- Keep the generic fallback behavior available for tests and offline lightweight
  use, but add a `fallback_to_lightweight` config switch.
- Set live `config/automixer.yaml` to `backend: mert`,
  `local_files_only: false`, and `fallback_to_lightweight: false`.
- Prefer Apple Silicon MPS when available, then CUDA, then CPU.
- Verify by instantiating `PerceptualEvaluator` and extracting an embedding from
  a short synthetic signal.

## Likely Files To Touch

- `backend/perceptual/embedding_backend.py`
- `backend/perceptual/perceptual_evaluator.py`
- `backend/requirements.txt`
- `requirements.txt`
- `config/automixer.yaml`
- `config/perceptual.yaml`
- `backend/config_manager.py`
- `tests/test_perceptual_evaluation.py`

## Alternatives Considered

- Leave fallback enabled and only install packages. Rejected because it can hide
  broken model loading during live tests.
- Vendor model weights into the repository. Rejected because model files are
  large and project rules say not to commit ML models over 100 MB.

## Risks

- First model load may download hundreds of MB and take time.
- MERT inference is heavier than the lightweight backend.
- HuggingFace remote model loading uses trusted remote code for this model.

## Test Plan

- Run perceptual unit tests.
- Instantiate MERT from the live config and extract one embedding.
- Run the standard pytest command if time allows.

## Where The Other Agent May Disagree

Kimi might prefer keeping fallback enabled in live config to avoid disabling
perceptual evaluation if HuggingFace is unavailable. I think live testing needs
the failure to be visible because the operator specifically asked for full MERT.
