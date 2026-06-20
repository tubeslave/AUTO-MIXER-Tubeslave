# ADR: FX send monitor and return safety gates

## Context

Effects and sends are part of the AI Mixing Engineer goal, but they are
high-risk in live sound because reverb, delay and monitor-exposed sends can
create feedback, smear intelligibility or overload returns. M084 hardened
downstream EQ/frequency/balance proposals; FX sends need their own safety
boundary before they can be trusted as assisted recommendations.

The current WING MVP still has no hardware proof for FX writes. Any FX work in
this checkpoint must therefore stay offline, proposal-only and unable to grant
live authority.

## Options Considered

1. Keep FX send proposals permissive and rely only on feedback/headroom checks.
2. Require fresh read-only/no-authority send/return state, mapped
   no-authority FX schema hints, monitor-safety evidence and rollback/readback
   evidence before proposal candidates are emitted.
3. Promote FX send changes into queueable or live supervised actions after unit
   tests.

## Decision

Require FX send proposals to consume fresh read-only/no-authority send/return
context, a mapped no-authority FX parameter schema, monitor-safety context when
monitor exposure exists, and explicit old-value/rollback readback sources.
Missing, stale, generic, mismatched or live-authority context keeps the result
analysis-only.

## Why This Won

This moves the project from dry effect suggestions toward safer assisted FX send
decisions while preserving the live-sound boundary. The system can reason about
lead vocal space, delay/reverb send levels, return headroom and monitor risk,
but it cannot apply FX changes or contact a console.

## Rejected Alternatives

- Feedback/headroom checks alone do not cover monitor exposure, return-level
  safety, stale schema mappings or live-authority context.
- Promoting FX sends into live apply would skip M007/M008 replay gates, WING
  hardware proof, readback, rollback, safety-kernel integration and operator
  approval.

## Implementation Plan

- Add no-authority freshness checks to FX send/return context.
- Require mapped no-authority FX parameter schema evidence.
- Surface old-value and rollback-value readback sources in operator proposals.
- Add monitor-safety and return-level safety evidence and blockers.
- Update proposal-only contract evidence requirements for FX sends.
- Add tests for fresh context, missing context, live-authority context,
  monitor-safety blockers and schema blockers.
- Record M085 evidence while keeping M001/M002 blocked on real WING/operator
  validation.

## Test Plan

- `PYTHONPATH=backend /opt/homebrew/bin/python3 -m pytest tests/test_fx_send_proposal.py -q`
- `PYTHONPATH=backend /opt/homebrew/bin/python3 -m pytest tests/test_wing_mvp_rails.py::test_m085_fx_send_monitor_return_safety_gates_stay_readonly -q`
- Standard backend suite.
- Memory lint and JSON/JSONL validation.

## Risks and Rollback

Risk: FX proposals become more conservative until fresh read-only send/return,
schema and monitor-safety context is available. This is acceptable because FX
sends can directly affect feedback and intelligibility.

Rollback: remove the FX context/schema/monitor safety checks, the new tests and
M085 state/report updates. This does not affect live transport because the
change is offline/proposal-only.

No live transport, no OSC write, no proposal apply, no readiness closure and no
live authority are introduced by this decision.
