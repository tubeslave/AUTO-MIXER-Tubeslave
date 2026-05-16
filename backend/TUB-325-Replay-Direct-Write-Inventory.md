# TUB-325 Replay and Direct-Write Inventory

Date: 2026-05-16

Scope:
- replay artifact coverage for arrangement-style metadata and auto-fader plans
- direct WING write surfaces still depending on the transport-level write gate
- safety implications for dry-run-first promotion

## Replay inventory status

Confirmed replay-safe building blocks already present:
- `ai_mixing_pipeline/decision_layer/replay_proposal_ranking.py`
  - proposals carry full `source_payload`
  - timeline artifacts persist `source_payload`, `replay_correlation_id`, `transport_policy`, and `live_mixer_writes`
  - all proposal builders default to `transport_policy="dry_run_only"` and `live_mixer_writes=False`
- `ai_mixing_pipeline/decision_layer/replay_executor.py`
  - proposals marked with operator confirmation requirements are blocked in replay instead of executed
- `tests/test_replay_proposal_ranking.py`
  - gain proposal replay inventory already preserved `arrangement_id` and `live_apply`
  - this pass adds the same regression lock for auto-fader/fader proposals and `readback_source`

Important repo fact:
- no runtime arrangement-automation module was found in this repo snapshot
- the only concrete arrangement marker currently visible is replay metadata such as `source_payload.arrangement_id`

Safety implication:
- replay evidence can preserve arrangement context once a producer emits it
- but the repo does not yet contain a single shared producer that turns legacy automation writes into replay-intent artifacts first

## Direct-write inventory

### Auto-soundcheck engine

Current direct write calls:
- `backend/auto_soundcheck_engine.py`
  - `set_fader(...)`
  - `set_gain(...)`
  - `set_eq_band(...)`
  - `set_mute(...)`

Current safety posture:
- these calls still target the mixer client directly
- on WING, unsupervised writes rely on `backend/wing_client.py` transport blocking
- they do not emit a shared replay/write-intent inventory artifact before the write attempt

### Auto-fader legacy controller

Current direct write calls:
- `backend/auto_fader.py`
  - read/query path: `send("/ch/.../fdr")`
  - write path: `set_channel_fader(...)` via replay-intent guard on WING

Current safety posture:
- readback/query is safe
- WING auto-balance and realtime fader writes emit replay/write-intent artifacts before any live write attempt
- blocked WING writes remain visible through transport-level gate telemetry

### Auto-fader v2 controller

Current direct write calls:
- `backend/auto_fader_v2/controller.py`
  - `set_gate_on(...)`
  - `set_channel_pan(...)`
  - AutoPanner and AutoReverb mixer application helpers

Current safety posture:
- WING fader writes now fail closed through the same replay-intent adapter used by legacy auto-fader:
  - startup reset to `0 dB`
  - peak-calibration fader targets
  - adaptive-noise-gate fader fallback when no dedicated gate setter exists
  - steady-state `_send_fader_command(...)` automation writes
- those WING fader paths now emit replay artifacts with:
  - `replay_correlation_id`
  - arrangement / scene metadata from `config.replay_write_intent`
  - explicit `readback_source`
  - post-limit requested target dB, so ceiling semantics are preserved in the artifact
- remaining non-fader surfaces still depend on mixer-client / transport blocking:
  - `set_gate_on(...)`
  - `set_channel_pan(...)`
  - spectral-masking and cross-adaptive `set_eq_band(...)`
  - AutoPanner and AutoReverb mixer application helpers

## What is stable after this pass

- replay artifacts now have explicit regression coverage for:
  - gain inventory metadata
  - auto-fader inventory metadata
  - arrangement-style ids
  - `live_apply`
  - `readback_source`
  - confirmation flags preserved in `source_payload`
- WING `auto_fader_v2` fader writes now share the same dry-run-first inventory contract as legacy `auto_fader`
- direct WING setter blocking already had tests for fader/gain classification
- this pass keeps the blocked-write inventory contract explicit: channel, write kind, requested value, source, and supervision reason

## Remaining gap

The main unresolved gap is architectural, not transport-level:
- legacy automation modules still attempt direct writes first
- replay artifacts are produced separately
- there is no shared "write intent -> replay artifact -> supervised/manual promotion" pipeline for auto-soundcheck or auto-fader

That means:
- WING safety is currently enforced by blocking, throttling, cooldown, approval, and readback
- deterministic replay exists for proposal artifacts
- but the two systems are not yet fully joined for non-fader legacy automation surfaces

## Recommended next safe tasks

1. Add a write-intent adapter for auto-fader and auto-soundcheck that emits replay proposals before any mixer write attempt.
2. Stamp those intents with `replay_correlation_id`, arrangement/scene metadata, and `readback_source`.
3. Keep WING transport writes dry-run-only for those paths until the intent artifact is mandatory.
4. Add tests that legacy automation surfaces produce replay intent rows even when WING writes are blocked.
5. Only after that, decide whether any of those surfaces should graduate to supervised/manual promotion.
