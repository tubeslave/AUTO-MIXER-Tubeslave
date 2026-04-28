# Existing Modules Report

## Summary

The repository already contains several mature pieces that the new offline chain
should reuse instead of replacing. The safest extension point is a new
offline-only package that reads files, renders local WAV candidates, and writes
reports without importing or calling live mixer clients.

## Audio loading

- `mix_agent/analysis/loader.py` loads WAV/AIFF/FLAC/OGG stems and optional mix
  or reference files with `soundfile`.
- `tools/offline_agent_mix.py` has a much larger custom loader/renderer for
  offline folder mixing.
- `src/experiments/muq_eval_director.py` loads multitrack WAVs for a MuQ-based
  offline experiment.
- `backend/audio_capture.py` is the canonical real-time capture service and
  must not be duplicated for live paths.

## Multitrack, stem, and master processing

- `mix_agent` provides typed models, analysis pipeline, rule planner, reporting,
  and conservative offline render helpers.
- `tools/offline_agent_mix.py` contains the most detailed monolithic offline
  automix implementation, including channel plans, EQ, compression, expander,
  FX, groups, and source-grounded logging.
- `tools/chat_only_shared_mix.py` contains a smaller source/stem-oriented
  offline mixer.
- `src/experiments/muq_eval_director.py` renders MuQ-scored candidates with beam
  search.

## Channel analysis and identity

- `backend/ml/channel_classifier.py` is the project ML classifier.
- `backend/auto_fader_v2/core/channel_classifier.py` wraps the ML classifier
  with a fallback.
- `backend/channel_recognizer.py` and `mix_agent/analysis/loader.py` infer roles
  from names.
- `backend/auto_fader_v2/core/bleed_detector.py` and `backend/bleed_service.py`
  cover bleed-related logic.

## Corrections and action generation

- `mix_agent/rules/*` generates explainable offline recommendations.
- `backend/auto_soundcheck_engine.py` coordinates live AutoFOH actions.
- `backend/autofoh_safety.py` provides typed live correction actions and safety
  execution.
- `tools/offline_agent_mix.py` has a rich offline planning layer with source
  rules and measurement passes.

## DSP, EQ, gain, dynamics, pan, and FX

- `mix_agent/actions/base.py` has conservative offline gain, HPF, parametric EQ,
  pan, and render helpers.
- `backend/auto_eq.py`, `backend/cross_adaptive_eq.py`, `backend/lufs_gain_staging.py`,
  `backend/auto_compressor.py`, `backend/auto_gate_caig.py`, `backend/auto_panner.py`,
  `backend/auto_reverb.py`, and `backend/auto_fx.py` cover live/control DSP.
- `backend/ml/differentiable_console.py`, `backend/ml/processing_graph.py`, and
  external `Diff-MST`, `FxNorm-automix`, `automix-toolkit`, and `dasp-pytorch`
  are research/action-planner references.

## Logging and reports

- `backend/autofoh_logging.py` provides non-blocking JSONL logging for live
  AutoFOH.
- `backend/output_paths.py` defines shared output locations.
- `mix_agent/reporting/*` writes JSON and Markdown reports.
- `backend/evaluation/muq_eval_service.py` and `backend/perceptual/perceptual_evaluator.py`
  write advisory decision logs.

## Offline processing

- `mix_agent` is the best small reusable offline facade.
- `tools/offline_agent_mix.py` is feature-rich but large and should be treated
  as a reference, not rewritten during this task.
- `src/experiments/muq_eval_director.py` is the closest prior MuQ-directed
  offline candidate search.

## Real-time processing

- `backend/audio_capture.py` is the real-time audio source.
- `backend/auto_soundcheck_engine.py`, `backend/auto_fader_v2/controller.py`,
  and `backend/agents/*` are live workflows.
- Heavy AI models must not enter these callbacks or control loops.

## OSC, MIDI, and control layer

- `backend/wing_client.py` sends Behringer WING OSC.
- `backend/dlive_client.py` sends Allen & Heath dLive MIDI/TCP.
- `backend/mixer_client_base.py`, `backend/osc_manager.py`, `backend/routing.py`,
  and `backend/handlers/*` form the live control surface.
- Offline test mode must not call these modules to apply changes.

## Existing AI/ML/analysis roles

- MuQ-Eval: `backend/evaluation/muq_eval_service.py`, `config/muq_eval.yaml`,
  `external/MuQ-Eval/`, and `src/experiments/muq_eval_director.py`.
- MERT-like embeddings: `backend/perceptual/*`, `config/perceptual.yaml`, and
  `config/automixer.yaml`.
- Essentia-like technical analysis: current project uses NumPy/scipy/librosa
  style metrics in `mix_agent/analysis/*` and `backend/signal_metrics.py`; no
  direct Essentia adapter was found.
- CLAP/LAION-CLAP, Audiobox Aesthetics, BEATs, Demucs, Open-Unmix: no direct
  production adapters were found.
- PANNs: external `Diff-MST/mst/panns.py` exists as research code, not a
  project-level channel identity service.

## Partial implementations

- MERT is already represented by a robust optional/fallback embedding backend.
- MuQ-Eval already has an advisory service and missing-model fallback.
- Source separation and CLAP-style semantic scoring are present in research
  docs/knowledge, but not as callable project modules.
- The action-planner ideas exist across `mix_agent`, external repos, and
  `tools/offline_agent_mix.py`, but not as one unified candidate API.

## Missing pieces

- A stable common critic interface.
- A unified role config for all critics/analyzers.
- A one-command offline test runner that writes the requested artifact tree.
- A decision engine that combines critic scores with weight renormalization.
- A final offline Safety Governor for candidate actions and rendered audio.

## Files to avoid touching without need

- `backend/server.py`
- `backend/wing_client.py`
- `backend/dlive_client.py`
- `backend/auto_soundcheck_engine.py`
- `backend/autofoh_safety.py`
- `backend/audio_capture.py`
- `tools/offline_agent_mix.py`
- `src/experiments/muq_eval_director.py`
- `external/**`
- `.env`, secrets, local model files, and existing `.ai/memory/*`

## Recommended extension points

- New package: `ai_mixing_pipeline/`.
- Reuse `mix_agent.analysis` for safe file analysis and technical metrics.
- Reuse `mix_agent.actions.base` for conservative offline rendering primitives.
- Reuse `backend/evaluation/muq_eval_service.py` through an adapter.
- Reuse `backend/perceptual/embedding_backend.py` through an adapter.
- Keep output under user-specified `offline_test_output/`, not live session
  folders.
