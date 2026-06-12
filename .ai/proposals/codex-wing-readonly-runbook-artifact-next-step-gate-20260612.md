# Proposal: Add artifact next-step guidance to WING read-only runbook

## Short Thesis

Operators should check `artifact_operator_next_steps` in the no-WING summary
before opening the backend/WING WebSocket.

## What I Understood

The first MVP blocker is still real WING read-only proof. Recent tooling makes
blocked expected-plan repair steps visible, but the runbook must tell the
operator to act on them before the hardware session starts.

## Proposed Solution

- Add a precondition covering
  `wing_readonly_preflight_evidence.artifact_operator_next_steps`.
- State the guidance is diagnostic only and does not approve writes or close
  readiness rows.
- Add a rails regression in `tests/test_wing_mvp_rails.py`.

## Likely Files To Touch

- `Docs/runbooks/WING_READONLY_PREFLIGHT.md`
- `tests/test_wing_mvp_rails.py`

## Alternatives Considered

Leaving the guidance only in no-WING summary tests was rejected because the
operator-facing runbook is the actual execution contract.

## Risks

The guidance could be mistaken for permission unless the boundary is explicit,
so the runbook states it does not approve live writes or close readiness rows.

## Test Plan

- Focused rails test.
- Adjacent WING/no-WING tests.
- Full pytest before commit.

## Where The Other Agent May Disagree

Kimi may prefer keeping runbook preconditions shorter. I prefer the explicit
field reference because it reduces operator ambiguity at the prestart gate.
