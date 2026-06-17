# WING Read-Only Runbook Validate Inputs Step

## Task

Sync the WING read-only operator runbook with the new `validate_readonly_preflight_inputs` handoff step.

## Context

The handoff packet and no-WING checks now require an 11-step operator sequence with `validate_readonly_preflight_inputs` between offline preview and real capture. The runbook still described the old 10-step sequence and stale capture/review/final-gate indices.

## Constraints

- Documentation-only update.
- Preserve the read-only, operator-gated WING path.
- Do not claim readiness or grant live authority.
- Do not touch unrelated dirty worktree changes.

## Test Plan

- Search the runbook for stale 10-step and old index fields.
- Run focused WING handoff/no-WING tests.
- Run no-WING quick.
- Run full pytest before commit.
