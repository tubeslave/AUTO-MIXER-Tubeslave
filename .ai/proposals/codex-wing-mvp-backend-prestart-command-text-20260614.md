# Codex Proposal: WING MVP Backend Prestart Command Text

## Short thesis

Add a stable exact-string mirror for the backend prestart command while keeping
the existing argv-list contract unchanged.

## What I understood

The current WING blocker is still read-only connection proof. The no-WING
prestart gate already records `backend_port_prestart_check_command` as a list,
but the operator-facing rail says the summaries must show the exact command
string. Relying on Markdown rendering of the list is brittle.

## Proposed solution

Add `WING_READONLY_PREFLIGHT_BACKEND_PORT_PRESTART_CHECK_COMMAND_TEXT` to the
shared contract and propagate `backend_port_prestart_check_command_text` through
the preflight runner, handoff packet, no-WING summary, evidence intake, review
decision ledger, readiness overlay, and supervised proof plan summary.

## Likely files to touch

- `backend/wing_readonly_preflight_contract.py`
- `backend/wing_readonly_preflight_evidence.py`
- `backend/wing_readonly_preflight_review_decision.py`
- `backend/autonomy_promotion_readiness.py`
- `backend/supervised_wing_proof_plan.py`
- `tools/wing_readonly_preflight_runner.py`
- `tools/build_wing_readonly_preflight_handoff_packet.py`
- `tools/no_wing_release_check.py`
- WING preflight/review/proof-plan tests
- WING MVP rail docs and reports

## Alternatives considered

Keep using only the argv list. Rejected because the current operator rail asks
for an exact text match and downstream summaries should not depend on display
format.

Replace the list field with a string. Rejected because existing validation
depends on the structured argv list.

## Risks

The main risk is partial propagation, where one artifact has the list but not
the text mirror. Focused tests cover each handoff point.

## Test plan

Run focused WING preflight/review/proof-plan/no-WING tests, regenerate the MVP
progress report, run Project Director and direction guard, run memory lint,
`git diff --check`, and the full standard test suite.

## Where Kimi may disagree

Kimi may prefer deriving the string only at render time. The JSON field is more
auditable for downstream automation and does not remove the structured list.
