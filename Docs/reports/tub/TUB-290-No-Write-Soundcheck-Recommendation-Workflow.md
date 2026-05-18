# TUB-290: No-Write Soundcheck Recommendation Workflow

Date: 2026-05-16

## Goal

Extract the recommendation-producing parts of `backend/auto_soundcheck_engine.py`
into a replay-safe workflow boundary that never mutates a live console.

This task does not change runtime orchestration. It defines the dry-run output
surface that a later task can wire into handlers, reports, or supervised review.

## Extraction Map

Current `AutoSoundcheckEngine` methods that mix recommendation logic with mixer
mutation:

- `_apply_input_gain(...)`
  - output kind: `input_gain`
  - current state: `trim_db`
  - requested state: `trim_db`
- `_apply_hpf(...)`
  - output kind: `hpf`
  - current state: `enabled`, `freq_hz`
  - requested state: `enabled`, `freq_hz`
- `_apply_eq(...)`
  - output kind: `eq`
  - current state: `bands`
  - requested state: `bands`
- `_apply_compressor(...)`
  - output kind: `compressor`
  - current state: compressor settings
  - requested state: compressor settings
- `_apply_pan(...)`
  - output kind: `pan`
  - current state: `pan`
  - requested state: `pan`
- `_apply_gain_correction(...)` plus `_apply_fader(...)`
  - output kind: `fader`
  - current state: `fader_db`
  - requested state: `fader_db`
- `_apply_fx_sends(...)`
  - output kind: `fx_send`
  - current state: `send_level_db`
  - requested state: `send_level_db`
- `_handle_feedback_event(...)`
  - output kinds: `feedback_notch`, `feedback_fader_reduce`
  - current state: varies by action
  - requested state: notch or fader-reduction candidate

## Artifact Contract

The extracted workflow lives in
`backend/soundcheck_recommendation_workflow.py` and emits:

- recommendation rows with:
  - `recommendation_id`
  - `kind`
  - `family`
  - `action_type`
  - `mode`
  - `channel`
  - `target`
  - `current_state`
  - `requested_state`
  - `source_payload`
  - `safety`
  - `confidence`
  - `replay_correlation_id`
- replay proposals with:
  - `transport_policy="dry_run_only"`
  - `live_mixer_writes=false`
  - `auto_apply_blocked=true`
- replay bundle outputs:
  - `recommendations`
  - `proposals`
  - `ranking`
  - `timeline_artifact`
  - `graph_checkpoint`
  - `mode_summary`

`source_payload` is the preferred place for soundcheck metrics, channel labels,
recognized presets, and derived rationale such as LUFS deltas, crest-factor
adjustments, or feedback frequency evidence.

## Safety Mode Map

- `approval_required`
  - `input_gain`
  - `fader`
  - reason: these directly change gain structure or audience level
- `assisted`
  - `hpf`
  - `eq`
  - `compressor`
  - `pan`
  - `fx_send`
  - reason: tonal/spatial moves are still write candidates, but they are better
    reviewed as operator-assisted suggestions instead of blind automation
- `observe_only`
  - `feedback_notch`
  - `feedback_fader_reduce`
  - reason: feedback reactions are time-sensitive and high-risk; preserve them
    as evidence/candidates only until a supervised live pathway exists

## No-Write Boundary

The workflow intentionally has no mixer client dependency.

Required invariants:

- every emitted proposal is replay-safe only
- every emitted proposal has `transport_policy="dry_run_only"`
- every emitted proposal has `live_mixer_writes=false`
- recommendation generation is separated from `set_*` console calls
- runtime wiring stays unchanged in this task
