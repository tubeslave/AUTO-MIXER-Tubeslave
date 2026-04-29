# AutoFOH Gap Analysis

## Existing Architecture

- Language/runtime: Python 3 backend, React/Electron frontend.
- Audio input: `backend/audio_capture.py` uses `sounddevice` with per-channel ring buffers and non-blocking test generators.
- Mixer control:
  - `backend/wing_client.py` for Behringer WING over OSC/UDP.
  - `backend/dlive_client.py` for Allen & Heath dLive over MIDI/TCP.
  - `backend/osc_manager.py` centralizes throttled OSC send/receive for the WING path.
- Channel-name parsing: `backend/channel_recognizer.py`.
- Per-channel analysis:
  - `backend/signal_metrics.py` for LUFS, true peak, dynamics, spectral metrics.
  - `backend/feedback_detector.py` for narrowband feedback/ringing detection.
- Automatic soundcheck/orchestration: `backend/auto_soundcheck_engine.py`.
- Config/tests:
  - YAML config via `backend/config_manager.py` and `config/automixer.yaml`.
  - Pytest test suite in `tests/`.

## Already Implemented

- Multichannel real-device capture is preserved.
- Raw channel signals can be analyzed before corrections are applied.
- Channel names are read from the console and used for preset selection.
- Automatic console correction is already supported and must remain supported.
- Per-channel LUFS/peak/dynamics/spectral analysis exists.
- Feedback detection exists and can already notch/reduce faders.
- Observation mode exists for safe test-only interception of writes.

## Main Gaps Against AutoFOH Requirements

- Channel classification is preset-oriented and not stem-aware; it has no explicit confidence, stem membership, allowed-control map, or override schema.
- The current engine applies a full processing chain directly from per-channel heuristics; it lacks typed action objects, risk/confidence gates, rollback/evaluation, and concert/runtime state permissions.
- Spectrum work is still mostly broad per-channel analysis; there is no named AutoFOH band model, 1/3-octave contribution view, or stem contribution matrix for culprit attribution.
- Unknown or weakly identified channels can still fall through to generic processing, which is risky even in test workflows.
- No persistent structured AutoFOH log/session report path exists yet.
- No soundcheck learning profile/state machine for show phases exists yet.

## Risky Current Behavior

- `AutoSoundcheckEngine` resets channels before analysis and then writes gain/HPF/EQ/compressor/fader/FX directly, with only local parameter clamps.
- Unknown/ambiguous channel names currently collapse to a generic/custom path instead of a conservative “observe only” default.
- The decision path is channel-local; there is no stem contribution proof before broad tonal decisions.
- Current engine state is operational (`discovering`, `analyzing`, `running`) rather than musical (`soundcheck`, `chorus`, `speech`, `emergency_feedback`).

## First Milestone Plan

1. Add foundational AutoFOH data models without replacing the existing engine.
2. Upgrade channel-name parsing to structured source/stem classification with confidence, overrides, priority, and allowed controls.
3. Add named band/index extraction and fractional-octave utilities with tests.
4. Add a stem contribution matrix utility with tests.
5. Integrate classification safety into `AutoSoundcheckEngine` so unknown/low-confidence channels are skipped for reset/auto-processing by default.

## Deferred To Later Milestones

- Full outbound correction safety layer with typed actions, rate limiting, support checks, and rollback evaluation.
- Soundcheck profile persistence/learning.
- Concert/runtime state machine and section-aware lead priority.
- Lead masking, mud/harshness/sibilance detectors, low-end controller, SPL/calibration support, and persistent JSONL session reporting.

## Planned Files

- New: `backend/autofoh_models.py`
- New: `backend/autofoh_analysis.py`
- Update: `backend/channel_recognizer.py`
- Update: `backend/auto_soundcheck_engine.py`
- Update: `backend/config_manager.py`
- Update: `config/automixer.yaml`
- New tests for classification, analysis bands, contribution matrix, and safer engine behavior
