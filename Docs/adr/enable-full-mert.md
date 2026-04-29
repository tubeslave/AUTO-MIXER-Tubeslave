# Enable Full MERT

## Context

The live auto-mixer config already requested `perceptual.backend: mert`, but the
Python environment did not include the HuggingFace runtime needed by the
existing `MERTEmbeddingBackend`. That caused a fallback to the lightweight
embedding backend, hiding the fact that full MERT was not active.

## Options Considered

- Install and declare the missing dependencies, with live config requiring MERT.
- Install dependencies but keep silent fallback.
- Vendor model weights locally.

## Decision

Install and declare `transformers`, `safetensors`, and `nnAudio`, keep MERT in
shadow mode, and set live config to require the MERT backend without silent
fallback.

## Why This Won

It makes the operator's requested evaluator real and observable while preserving
the existing mixer safety pipeline. If model loading fails, tests and backend
startup logs show the problem instead of quietly using the wrong evaluator.

## Rejected Alternatives

Silent fallback was rejected for live tests because it obscures whether MERT is
actually involved. Vendoring weights was rejected because the model is large and
the repository should not store large ML artifacts.

## Implementation Plan

- Add dependency declarations.
- Add a `fallback_to_lightweight` switch to the embedding backend.
- Use MPS/CUDA/CPU automatic device selection.
- Clean the live perceptual config.
- Verify model loading with a short synthetic extraction.

## Test Plan

- `PYTHONPATH=backend ./backend/venv/bin/python -m pytest tests/test_perceptual_evaluation.py -q`
- Standard backend pytest command after dependency install.
- Direct MERT smoke test using `PerceptualEvaluator`.

## Risks And Rollback

MERT may increase CPU/GPU load and the first load may require a large download.
Rollback is to set `perceptual.backend: lightweight` or
`fallback_to_lightweight: true`, then restart the backend.
