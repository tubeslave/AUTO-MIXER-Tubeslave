# ADR: Automation Foundation Layer

## Context

AUTO-MIXER has a narrow supervised WING proof for fader/gain proposals, but this
does not make the system autonomous-live-ready. The next step is foundation
infrastructure: decision traces, safety DTOs, offline replay evidence, and
frontend panels.

## Options considered

1. Promote more live actions now.
2. Keep all work frontend/mock-only.
3. Add backend-backed evidence and replay infrastructure while preserving the
   supervised fader/gain boundary.

## Decision

Use option 3. The foundation layer records and displays decisions, blocks unsafe
or duplicate applies, and keeps non-fader/gain surfaces proposal-only.

## Why this won

It improves observability and operator workflow without adding new live write
capabilities. It also creates a concrete data contract for future replay,
memory, and promotion decisions.

## Rejected alternatives

- Autonomous apply: rejected until a separate hardware proof and rollback plan.
- Mock-only cockpit panels: rejected because the cockpit must reflect backend
  Product Layer state.
- Direct OSC controls for EQ/routing/FX/snapshots: rejected under the WING
  deployment boundary.

## Implementation plan

- Add `AutomationDecisionTrace` and append-only `decision_traces.jsonl`.
- Require `idempotency_key` for `apply_operator_proposal`.
- Enforce bounded fader/gain deltas before the manual write gate.
- Add read-only DTOs: `decision_traces`, `safety_monitor`, `replay_runs`, and
  `channel_intelligence`.
- Extend offline validation artifacts with trace, metrics, and report outputs.
- Connect cockpit panels to these DTOs.

## Test plan

- Focused backend tests for missing/duplicate idempotency, delta blocking,
  trace output, and safety monitor DTO.
- Replay artifact test updates.
- Frontend TypeScript, e2e, unit tests, build.

## Risks and rollback

Risk: older callers without `idempotency_key` will now be blocked.
Rollback: temporarily allow generated idempotency keys only for explicitly
marked legacy dry-run callers, not for live apply.

Risk: bounded delta limits block existing broad proposals.
Rollback: keep proposals visible as review-only and require a separate ADR to
raise limits.

