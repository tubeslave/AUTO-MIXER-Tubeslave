# AI Mixing Offline Chain Brief

## Task

Add a safe offline AI mixing chain that loads multitrack stems, renders
candidates, evaluates them with optional critic adapters, gates the result
through a Safety Governor, and writes explainable reports.

## Constraints

- Do not modify live OSC/MIDI paths for the offline test mode.
- Heavy ML models are optional and must fall back gracefully.
- MuQ-Eval, Audiobox, MERT, CLAP, Essentia, PANNs/BEATs, and Demucs/Open-Unmix
  are role-specific sidecars, not direct mixer controllers.
- Reuse existing `mix_agent`, `backend/evaluation`, and `backend/perceptual`
  logic where practical.
- Keep all actions bounded, reversible, and explainable.
- Do not delete old offline experiments or monolithic mixers.

## Expected Output

- Root package `ai_mixing_pipeline/` with role-specific subpackages.
- `configs/ai_mixing_roles.yaml`.
- CLI: `python -m ai_mixing_pipeline.offline_test_runner`.
- Offline outputs: renders, decision JSONL, critic CSV, accepted/rejected JSON,
  before/after snapshots, and summary Markdown.
- Tests for config, critic interface, renderer, decision engine, safety governor,
  missing model fallbacks, and the full offline chain.
