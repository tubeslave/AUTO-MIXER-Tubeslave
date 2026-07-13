# Soundcheck corrections require supervised transactions

## Context

The soundcheck bridge can emit legacy generic gain/fader proposals when a
recommendation is not represented by a supervised transaction child. The UI
normally hides apply controls in assist mode, but the queue's generic manual
write path only validated the value shape and current operator mode. A pending
legacy row could therefore become applyable after a later switch to a mode
with generic fader authority, bypassing the soundcheck family registry,
transaction preflight, rollback journal, and readiness accounting.

## Options considered

1. Delete every unmatched soundcheck recommendation.
2. Allow generic gain/fader apply when the current operator mode permits it.
3. Keep unmatched rows for operator review, but require every soundcheck
   correction to have a supervised transaction before it can reach an apply
   path.

## Decision

Use option 3. A proposal from `soundcheck_recommendation_bundle` without a
dictionary `supervised_transaction_plan` receives the immutable blocker
`soundcheck_correction_requires_supervised_transaction`. The queue exposes the
blocker, while preflight and apply repeat the source gate before idempotency
reservation, proposal claim, or manual console write.

## Why this won

The operator retains the musical recommendation and its evidence, but changing
operator mode cannot silently grant it a less strict execution path. Planned
transaction children continue through family-specific safety, readback,
rollback, and soundcheck readiness accounting.

## Rejected alternatives

- Deleting unmatched rows hides incomplete planner coverage and makes the
  soundcheck look more complete than it is.
- Trusting the UI or creation-time mode is not an authority boundary; backend
  apply authority can change while a proposal remains pending.

## Implementation plan

- Mark unmatched bridge rows as operator-review-only.
- Add the immutable source blocker in `operator_proposal_queue.py`.
- Repeat the blocker in apply preflight and apply handlers before any mutable
  authority step.
- Let default batch preflight exclude the rows through `can_apply=false`.

## Test plan

- Verify an unmatched fader remains pending but `can_apply=false` with the exact
  blocker.
- Create it in assist, switch to an apply-capable test mode, and prove both
  preflight and apply remain blocked without idempotency reservation or console
  send.
- Verify batch preflight excludes it and ordinary operator-authored fader
  proposals still use the existing manual gate.
- Preserve transaction-plan registry tests.

## Risks and rollback

Some legacy soundcheck rows will no longer offer an apply button. That is the
intended fail-closed state until planner coverage produces a transaction child.
Rollback is limited to the source blocker and handler defense-in-depth checks;
it does not alter console state or stored proposal evidence.
