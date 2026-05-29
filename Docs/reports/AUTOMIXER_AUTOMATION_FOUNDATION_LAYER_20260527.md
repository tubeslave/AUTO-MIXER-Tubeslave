# Automixer Automation Foundation Layer Report

Date: 2026-05-28
Status: implemented, dry-run/supervised-safe
Scope: documentation handoff for the 2026-05-27 automation foundation slice

## Summary

The project now has a foundation layer for safe agentic live-mixing work. It
does not make Automixer autonomous. It gives the system a consistent way to
record what it wanted to do, why it was safe or blocked, what was read before
apply, what was read back after apply, and what rollback evidence exists.

The live boundary remains unchanged: real console writes are limited to
operator-approved supervised `fader` and `gain` proposals through
`apply_operator_proposal` and the existing manual gate.

## Implemented Surfaces

- `backend/automation_trace.py` defines `AutomationDecisionTrace` and an
  append-only trace store.
- `backend/server.py` owns trace-store lifecycle and idempotency reservation
  for operator apply requests.
- `backend/handlers/product_state_handlers.py` exposes read-only DTO handlers
  and hardens `apply_operator_proposal`.
- `backend/operator_product_state.py` builds cockpit-safe DTOs:
  `decision_traces`, `safety_monitor`, `replay_runs`, and
  `channel_intelligence`.
- `backend/offline_validation_harness.py` writes replay artifacts:
  `decision_traces.jsonl`, `metrics_before_after.json`, and `report.md`.
- Frontend websocket, backend hook, adapters, and Task Queue panels consume the
  new DTOs as read-only evidence.
- The Task Queue cockpit now includes a compact `Replay Engine / Evidence`
  surface backed by `replay_runs`, `decision_traces`, and
  `channel_intelligence`. Its `Start Offline` action sends only
  `start_replay_run`, which is an offline replay request and not a live
  WING/OSC write.
- The backend `start_replay_run` handler now records a summary
  `offline_replay_v1` `replay_run` trace after writing replay artifacts. The
  trace carries `offline_only=true`, `live_write=false`, and
  `mixer_transport_allowed=false`, and can feed decision memory as offline
  evidence.
- `replay_runs` DTO rows expose run-level `trace_count`, `offline_only`, and
  `live_write` fields, and the cockpit Replay/Evidence panel renders those
  policy facts directly.
- The cockpit now treats `replay_run_started` as a local read-only replay-row
  echo and then refreshes `get_replay_runs`. E2E verifies the newly started run
  appears, `get_replay_runs` is requested again, and unsafe write messages are
  still absent.
- `backend/shadow_gain_rides.py` adds shadow-only vocal gain ride candidates.
  Explicit operator analysis can now write `gain_ride_shadow` traces with a
  `0.5 dB` max step, oscillation/pumping blockers, rollback evidence, and
  `live_apply_allowed=false`. These are evidence traces only, not queueable live
  proposals.
- `backend/decision_memory.py` adds decision memory v1. It derives append-only
  memory entries from verified/offline traces only, excludes `shadow_only`
  traces, and surfaces those rows through `memory_knowledge_snapshot` as context
  for operators and future proposal builders.

## Safety Rules Now Captured In Code

- `apply_operator_proposal` requires an `idempotency_key`.
- Duplicate idempotency keys are rejected before reaching the manual gate.
- Fader steps are bounded to `1.0 dB`.
- Gain ride steps are bounded to `0.5 dB`.
- Auto-generated upward fader/gain proposals are blocked unless explicitly
  operator-authored.
- Non-`fader`/`gain` actions remain proposal-only, shadow-only, offline-only, or
  quarantined.
- Offline replay and MuQ/perceptual scoring have no live-apply authority.

## Validation Evidence

The latest unrestricted local checks for this slice were green:

- `PYTHONPATH=backend python -m pytest tests/test_operator_product_state.py -q`
  -> `48 passed`
- `PYTHONPATH=backend python -m pytest tests/ -x --tb=short -q`
  -> `1208 passed, 1 skipped`
- `PYTHONPATH=backend python -m pytest tests/test_offline_validation_harness.py tests/test_wing_operator_proposal_validation_runner.py tests/test_gain_proof_dry_run_runner.py -q`
  -> `22 passed`
- `PYTHONPATH=backend python -m pytest tests/test_server.py -q`
  -> `110 passed`
- `npm --prefix frontend test -- --watch=false`
  -> `13 passed`
- `npm --prefix frontend run test:e2e`
  -> `8 passed`
- `npm --prefix frontend run build`
  -> passed
- `cd frontend && npx tsc --noEmit`
  -> passed
- `git diff --check`
  -> passed
- Browser-level visual smoke used a local Vite server at
  `http://127.0.0.1:5179/#/task-queue` with a 1280x720 screenshot. The
  Replay/Evidence panel was visible in the 16:9 cockpit layout and kept the
  `no live write` policy visible.

## Readiness Wording

Allowed claim:

- Automixer has a traceable, read-only/supervised foundation layer for proposal
  evidence, offline replay evidence, safety monitor DTOs, and cockpit display.

Not allowed claim:

- Automixer is autonomous-live-ready.
- EQ, compressor, gate, pan, phase/delay, FX, routing, snapshots, scenes,
  training, or rules can be live-applied.
- MuQ or replay scores can authorize live writes.

## Next Safe Work

1. Keep the next physical WING step narrow: one gain proof after explicit safe
   reset, physical pre-read, apply readback, rollback readback, and blocked
   proof.
