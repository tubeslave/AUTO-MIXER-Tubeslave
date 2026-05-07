# Perceptual Evaluation Brief

## Task

Add an experimental perceptual quality layer to AUTO-MIXER-Tubeslave using
MERT-like audio embeddings, without changing the live channel-analysis,
decision, safety, or OSC command behavior by default.

## Constraints

- Default `perceptual.enabled` remains `false`.
- Shadow/offline mode only; no OSC blocking or mutation in this stage.
- Heavy embedding models must never run in the real-time audio callback.
- Missing `torch`, `transformers`, or a MERT model must fall back safely.
- Logs must support future agent/RL training with reward-style fields.

## Expected Output

- Independent `backend/perceptual/` module.
- Lightweight and optional MERT backends.
- Embedding metrics and `RewardSignal`.
- JSONL decision logging.
- Shadow integration around AutoFOH action evaluation.
- Tests and docs.
