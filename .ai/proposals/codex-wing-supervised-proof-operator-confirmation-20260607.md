# Codex Proposal: WING Supervised Proof Operator Confirmation - 2026-06-07

## Short thesis

Physical supervised WING proof should fail before WebSocket if the operator has
not explicitly reconfirmed the exact channel, value type, and delta that will be
used with `--execute`.

## What I understood

The repository is focused on the supervised WING Rack MVP. Current allowed live
scope is still only bounded fader and gain-down proof through Product Layer
proposal apply and rollback. The next real hardware action remains read-only
preflight and manual review first.

## Proposed solution

Add `--operator-confirm-channel`, `--operator-confirm-value-type`, and
`--operator-confirm-delta-db` to `tools/wing_operator_proposal_validation_runner.py`.
Require them only when `--execute` is present, and require exact match with the
requested action before any WebSocket connection opens. Emit the confirmation
facts into `summary.json`.

Update the supervised proof plan to include the flags and update
`tools/no_wing_release_check.py` so a stale proof-plan command fails release
validation.

## Likely files to touch

- `tools/wing_operator_proposal_validation_runner.py`
- `backend/supervised_wing_proof_plan.py`
- `tools/no_wing_release_check.py`
- focused tests for the runner, proof plan, and no-WING gate
- WING supervised proof runbook and report/index docs

## Alternatives considered

- Rely on prose runbook approval only. Rejected because a machine-checkable
  confirmation is safer and easier to audit.
- Require confirmation for dry-run planning too. Rejected because dry-run
  planning is non-mutating and useful before choosing the exact hardware action.

## Risks

Existing physical proof commands need three additional flags. That is intentional
because stale commands should fail closed.

## Test plan

- Compile changed Python modules.
- Run focused runner, proof-plan, and no-WING release-check tests.
- Run quick no-WING release check.
- Run direction guard and full pytest before commit.

## Where the other agent may disagree

Kimi may argue for an interactive prompt instead of CLI flags. CLI flags are
better here because the proof runner is artifact-driven and must be repeatable
in runbooks and release gates.
