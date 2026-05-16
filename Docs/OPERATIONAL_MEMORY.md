# Operational Memory

## Purpose

`backend/operational_memory.py` is the centralized append-only memory layer for
execution continuity. It does not control the mixer and does not alter live OSC
behavior. It only persists evidence and GPT-readable summaries.

## Stored event families

- `session_event`
  - session lifecycle and config changes from `backend/session_manager.py`
- `write_intent`
  - replay-safe dry-run WING write intents from `backend/replay_write_intent_adapter.py`
- `telemetry_session`
  - WING telemetry capture start/stop records from `backend/wing_telemetry_recorder.py`
- `task_outcome`
  - issue-level outcomes, evidence paths, and execution summaries
- `context_sync`
  - agent handoff summaries and related event references
- `knowledge_snapshot`
  - durable JSON snapshots for shared operational facts

## Artifact layout

Default base path:

`artifacts/operational_memory/`

Files:

- `events.jsonl`
  - append-only raw event log
- `latest_snapshot.json`
  - machine-readable rollup of counts and recent evidence
- `latest_summary.md`
  - GPT/human-readable execution summary
- `knowledge_snapshots/*.json`
  - durable shared knowledge payloads

## Safety posture

- no live mixer writes
- no auto-promotion from dry-run to live control
- no mutation of replay evidence after append
- integrations are best-effort and fail open toward the primary runtime path

## Intended next integrations

- record shadow-mode report summaries after replay comparisons complete
- attach supervised-write run outcomes after physical WING tests
- add a Paperclip helper that writes `task_outcome` and `context_sync` at issue closeout
