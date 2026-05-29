# Task automation-foundation-layer-20260527: Safe foundation layer for agentic live mixing

## Short thesis

Add traceable, replayable, supervised-only automation infrastructure without
promoting autonomous live mixing or widening WING write surfaces.

## What I understood

The current `main_live_assist` path remains canonical. Live writes stay limited
to operator-approved `fader` and `gain` proposals through
`apply_operator_proposal`. EQ, compressor, gate, pan, phase, FX, routing,
snapshots, training, rules, and autonomous apply remain proposal-only or
quarantined.

## Proposed solution

- Add append-only `AutomationDecisionTrace` JSONL evidence.
- Require `idempotency_key` on supervised apply requests.
- Block duplicate apply keys.
- Enforce small per-apply fader/gain delta limits before the manual gate.
- Expose read-only Product Layer DTOs for decision traces, safety monitor,
  replay runs, and channel intelligence.
- Extend offline validation artifacts with decision traces, metrics JSON, and a
  markdown replay report.
- Connect the frontend cockpit to backend DTOs without adding direct OSC/live
  controls.

## Likely files to touch

- `backend/automation_trace.py`
- `backend/server.py`
- `backend/handlers/product_state_handlers.py`
- `backend/operator_product_state.py`
- `backend/offline_validation_harness.py`
- `frontend/src/services/*`
- `frontend/src/components/prototype/TaskQueuePanels.tsx`
- focused tests and docs

## Risks

- Enforcing `idempotency_key` can break older callers unless runners/frontend are
  updated in the same slice.
- Delta limits may block older broad fader proposals; this is intentional for
  safety and should be adjusted only through a separate proof plan.
- Trace data must be treated as evidence, not as autonomous authority.

## Test plan

- Backend focused pytest for idempotency, delta limits, traces, safety monitor,
  and replay artifacts.
- Frontend TypeScript/build/e2e.
- `git diff --check`.

