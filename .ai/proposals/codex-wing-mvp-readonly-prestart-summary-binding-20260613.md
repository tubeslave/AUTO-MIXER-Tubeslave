# Codex Proposal: WING MVP Read-Only Prestart Summary Binding

## Short thesis

Bind read-only WING preflight artifacts to the actual no-WING prestart
`summary.json` by path and SHA256, then make review and supervised proof
planning fail closed without that binding.

## What I understood

The first MVP blocker is still `WING read-only connection proof`. Existing
artifacts require prestart contract fields, but those fields can be present
without proving which no-WING prestart run passed before the backend/WING
session.

## Proposed solution

- Add `--prestart-summary` to `tools/wing_readonly_preflight_runner.py`.
- Validate the referenced no-WING summary before plan-only and real runs.
- Store prestart summary identity and `backend_port_free` status in the
  read-only plan and captured summary.
- Update evidence review, decision ledger, and supervised proof plan checks.
- Document the operator command changes.

## Likely files to touch

- `tools/wing_readonly_preflight_runner.py`
- `backend/wing_readonly_preflight_evidence.py`
- `backend/wing_readonly_preflight_review_decision.py`
- `backend/supervised_wing_proof_plan.py`
- `tools/review_wing_readonly_preflight_artifacts.py`
- WING MVP docs, tests, ADR, and report.

## Alternatives considered

- Keep declarative prestart fields only.
- Add a report-only reminder to the runbook.
- Let the decision JSON carry the prestart hash instead of the preflight
  artifact.

## Risks

- Operators must pass one more path in the command.
- Existing offline fixtures must include a synthetic no-WING summary.

## Test plan

- Focused preflight runner, evidence, review, decision, and proof-plan tests.
- MVP rails and Project Director tests.
- No-WING quick gate observation remains fail-closed on the current busy port.

## Where the other agent may disagree

Kimi might prefer keeping the runner command shorter and validating the
prestart summary only in the offline review step. I prefer failing before
WebSocket because this is a prestart gate.
