# Decision/Correction Layer Audit

## Existing Modules

- Audio loading: `ai_mixing_pipeline.audio_utils.read_audio`, `audio_files`,
  `resample_audio`; older large tooling also exists in `tools/offline_agent_mix.py`.
- Multitrack/stem/master handling: `ai_mixing_pipeline.offline_test_runner.runner`
  loads `offline_test_input/multitrack`, optional `reference`, and
  `channel_map.json`.
- Initial mix and offline render: `ai_mixing_pipeline.sandbox_renderer.SandboxRenderer`
  already renders initial and candidate mixes with offline-only DSP and safe
  true-peak trimming.
- Critics/analyzers: `ai_mixing_pipeline.critics` contains MuQ-Eval,
  Audiobox Aesthetics, and CLAP adapters; `stem_critics.mert` contains MERT;
  `technical_analyzers` contains Essentia and PANNs/BEATs-style adapters.
- Decision logic: `ai_mixing_pipeline.decision_engine.DecisionEngine` selects
  the best previous offline-test candidate from critic deltas and safety score.
- Safety logic: `ai_mixing_pipeline.safety_governor.SafetyGovernor` checks
  action bounds, clipping, headroom, phase, vocal clarity, compression, and
  identity/bleed signals.
- DSP: existing offline renderer supports gain, high-pass, parametric EQ when
  available, compression, pan, and report-only FX send. Live DSP/OSC paths live
  under `backend/` and must stay isolated from offline correction.
- Logs/reports: `ai_mixing_pipeline.reports.writers` writes decision JSONL,
  critic CSV, accepted/rejected actions, snapshots, and summary markdown.
- CLI: `python -m ai_mixing_pipeline.offline_test_runner` already exists for
  the first offline chain.
- Tests: `tests/test_*ai_mixing*`, `test_offline_test_chain.py`,
  `test_sandbox_renderer.py`, `test_decision_engine.py`, and
  `test_safety_governor.py` cover the first pipeline and fallbacks.

## Reuse Points

- Use `audio_utils` for file IO, loudness matching, true peak/headroom metrics,
  and fallback measurement.
- Use existing critic adapters through a new `CriticBridge`.
- Keep existing `SandboxRenderer`, `DecisionEngine`, and `SafetyGovernor`
  intact; the new layer wraps similar responsibilities for action optimization
  and virtual mixing rather than changing the old offline-test API.
- Reuse project channel-map conventions and filename role inference.

## Missing Pieces Before This Change

- A typed action schema that can be optimized, rendered, serialized, and logged.
- A dedicated optimizer layer for Nevergrad/Optuna-style ask/tell candidates.
- A virtual mixer abstraction that uses real upstream `pymixconsole` when
  installed and falls back to a dependency-light local DSP mixer.
- A correction runner that filters unsafe actions before render, renders only
  offline, evaluates critics, and produces best mix plus mixer-state reports.

## Best Integration Point

The new layer belongs under `ai_mixing_pipeline/decision_layer/`. It is
offline-only, imports existing critics/analyzers through adapters, and writes
reports under the requested output run directory. This avoids touching
`backend/server.py`, live mixer clients, OSC/MIDI handlers, or real-time audio
threads.

## Files To Avoid Touching Without Need

- `backend/wing_client.py`, `backend/dlive_client.py`, and live control
  handlers: they send real mixer commands.
- `backend/audio_capture.py` and real-time processing code: the decision layer
  must not block audio callbacks.
- Existing critic adapters: they already provide graceful fallbacks and should
  remain stable.
- `tools/offline_agent_mix.py`: large legacy experiment; useful reference, but
  not a safe place for broad rewrites.

## Candidates For Cleanup Only

See `docs/CLEANUP_CANDIDATES_DECISION_LAYER.md`. No old files were deleted.
