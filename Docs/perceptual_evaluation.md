# Perceptual Evaluation

## Purpose

The perceptual layer is an experimental shadow/offline evaluator for AUTO-MIXER-Tubeslave. The existing live path still captures multichannel audio, computes channel metrics, selects safe corrections, and sends OSC commands exactly as before. The new evaluator adds a separate question: did the audio snapshot become perceptually closer to a useful target, or at least avoid obvious quality regressions?

The first implementation uses MERT/MERT-like audio-embedding ideas:

- `embedding_mse`
- `cosine_distance`
- `delta_score`
- confidence and verdict: `improved`, `worse`, or `neutral`
- optional batch `fad_like_distance`

## Current Integration Points

- Audio input: `backend/audio_capture.py` keeps per-channel ring buffers and exposes `get_buffer()`.
- Channel features: `backend/signal_metrics.py` and `backend/autofoh_analysis.py` compute loudness, dynamics, spectral bands, mix indexes, and stem aggregates.
- Decision path: `backend/auto_soundcheck_engine.py` creates typed actions, then routes them through `AutoFOHSafetyController`.
- OSC path: mixer clients such as `backend/wing_client.py` and `backend/dlive_client.py` execute `set_fader`, `set_eq_band`, `set_hpf`, `set_compressor`, and related writes.
- Shadow hook: `AutoSoundcheckEngine._execute_action()` captures a pre-action audio window, and `_evaluate_pending_actions()` captures the delayed post-action window and submits both to `PerceptualEvaluator`.

## Why Shadow/Offline First

MERT-class models are too heavy for a real-time audio callback. This module never runs in the audio callback and does not block or rewrite OSC commands. By default:

```yaml
perceptual:
  enabled: false
  mode: shadow
  block_osc_when_score_worse: false
```

Even if enabled, the module only logs scores. If it fails, the engine catches the error and live mixing continues.

## Configuration

Defaults live in `ConfigManager`, `config/automixer.yaml`, and the standalone example `config/perceptual.yaml`.

```yaml
perceptual:
  enabled: false
  mode: shadow
  backend: lightweight
  model_name: "m-a-p/MERT-v1-95M"
  sample_rate: 24000
  window_seconds: 5
  hop_seconds: 2
  evaluate_channels: true
  evaluate_mix_bus: true
  max_cpu_percent: 25
  log_scores: true
  block_osc_when_score_worse: false
  log_path: logs/perceptual_decisions.jsonl
  async_evaluation: true
```

To enable lightweight shadow scoring, set `perceptual.enabled: true`. OSC behavior remains unchanged.

## Backends

`lightweight` is the default safe backend. It uses deterministic numpy features: temporal level, crest factor, clipping/activity, spectral centroid/rolloff/flatness, and log-spaced band summaries. It is intended for development and data collection when MERT is not installed.

`mert` attempts to load `torch`, `transformers`, and `model_name`. If any import or model load fails, the evaluator logs a warning and falls back to `lightweight`.

For offline MERT testing:

```yaml
perceptual:
  enabled: true
  backend: mert
  model_name: "m-a-p/MERT-v1-95M"
  local_files_only: false
```

Use `local_files_only: true` when the model is already cached and network/model downloads must be avoided.

## Logs

Perceptual decisions are appended to:

```text
logs/perceptual_decisions.jsonl
```

Each JSONL row includes:

- `timestamp`
- `channel`
- `instrument`
- `action`
- `score_before`
- `score_after`
- `perceptual_score` / `delta_score`
- `mse`
- `cosine_distance`
- `verdict`
- `confidence`
- `features_before`
- `features_after`
- `reward_signal`
- `osc_sent`

The `features_before` and `features_after` fields intentionally store summaries, not raw audio or full embeddings.

## Reward Signal

`backend/perceptual/reward.py` defines `RewardSignal`:

- `engineering_score`
- `perceptual_score`
- `safety_score`
- `combined_score`

This keeps the perceptual layer usable later as a reward function for agent/RL training. Current shadow rows can be treated as offline training data once enough action/audio pairs are collected.

## Limitations

Without a true post-console mix bus or a reference embedding, channel-level live evaluation often sees raw input rather than processed output. In that case the result is best treated as a neutral data point or a lightweight proxy, not proof that the audience mix improved.

The next useful stage is to feed the evaluator from a real post-fader/post-main capture path and build reference embeddings per venue, song, or instrument role.
