# Codex Proposal: WING MVP Proof Plan Preflight Command Bindings

## Short thesis

Make the supervised proof plan's fallback read-only preflight command use the
same expected-plan and no-WING prestart-summary bindings as the WING runbook.

## What I understood

The current blocker is still the WING read-only connection proof. Handoff and
runbook commands already require a passed no-WING summary plus an expected
offline plan before the real read-only preflight. The supervised proof plan's
blocked-by-preflight step should not suggest a weaker command.

## Proposed solution

Add default `readonly_preflight_expected_plan` and
`readonly_preflight_prestart_summary` bindings to
`backend/supervised_wing_proof_plan.py`. Include those values in the read-only
preflight step and command. Extend `tools/no_wing_release_check.py` so the
`next_wing_proof_plan` validator and summary require and expose the same
fields.

## Likely files to touch

- `backend/supervised_wing_proof_plan.py`
- `tools/no_wing_release_check.py`
- `tests/test_supervised_wing_proof_plan.py`
- `tests/test_no_wing_release_check.py`
- `Docs/reports/AUTOMIXER_WING_MVP_PROOF_PLAN_PREFLIGHT_COMMAND_BINDINGS_20260614.md`
- `Docs/INDEX.md`

## Alternatives considered

Leave the proof plan command unchanged. Rejected because it would remain weaker
than the operator runbook and handoff packet.

Only document the command in the report. Rejected because the release gate
would not enforce the contract.

## Risks

The main risk is mistaking a stronger offline command contract for hardware
proof. This change remains non-executing and does not grant live authority or
close readiness.

## Test plan

Run focused proof-plan and no-WING tests, focused WING/readiness rail tests,
Project Director, direction guard, `py_compile`, memory lint, `git diff
--check`, and the standard full pytest command.

## Where the other agent may disagree

Kimi may prefer moving command construction into a shared WING preflight command
builder. That is reasonable later; this patch keeps the current surface small
and test-backed.
