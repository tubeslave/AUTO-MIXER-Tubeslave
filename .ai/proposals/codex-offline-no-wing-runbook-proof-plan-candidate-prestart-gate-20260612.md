# Codex Proposal: Offline no-WING Runbook Proof Plan Candidate Prestart Gate

## Short thesis

The offline no-WING runbook should explicitly state that Next WING Proof Plan success requires the accepted candidate prestart trace fields and the specific blocker when that trace is invalid.

## What I understood

The code and release validator now enforce the candidate-prestart gate. The operator runbook still described the Next WING Proof Plan only as read-only, operator-required, and non-executing, which is now incomplete.

## Proposed solution

- Add runbook success criteria for `readonly_preflight_review_decision_candidate_prestart_contract_valid`.
- Add runbook success criteria for `readonly_preflight_review_decision_candidate_prestart_source_ids`.
- Require the specific `accepted_readonly_preflight_candidate_prestart_contract_required_before_supervised_proof` blocker when accepted ids exist but the candidate prestart trace is invalid.
- Add a regression test that reads the runbook and asserts those strings are present.

## Likely files to touch

- `Docs/OFFLINE_NO_WING_VALIDATION_RUNBOOK.md`
- `tests/test_no_wing_release_check.py`
- `.ai/briefs/`
- `.ai/proposals/`
- `Docs/adr/`
- `Docs/reports/`

## Alternatives considered

- Leave runbook wording generic. That risks an operator missing the new precise blocker remediation.
- Add a separate runbook section. A concise success-criteria extension is enough and keeps the checklist close to the existing `next_wing_proof_plan_schema` gate.

## Risks

- Minimal. This is documentation plus a text regression test.

## Test plan

- Focused no-WING runbook pytest.
- Affected no-WING pytest.
- Full standard pytest suite.
- Diff and memory lint.

## Where the other agent may disagree

Kimi may prefer moving these strings into generated docs. I would first lock the human runbook, then consider generation if this section changes frequently.
