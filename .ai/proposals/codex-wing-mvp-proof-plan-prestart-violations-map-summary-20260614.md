# Codex Proposal: WING MVP Proof Plan Prestart Violations Map Summary

## Short thesis

Make the no-WING release summary carry the proof-plan accepted-candidate
prestart violations map, and reject stale proof-plan summaries that disagree
with the decision-review ledger.

## What I understood

The blocker rail now requires two empty maps before supervised proof planning:
`accepted_candidate_prestart_contract_violations_by_source` in the
decision-review ledger/report, and
`readonly_preflight_review_decision_candidate_prestart_contract_violations_by_source`
in the supervised proof-plan summary. The backend proof plan already computes
the second field, but the no-WING summary dropped it.

## Proposed solution

Extend `tools/no_wing_release_check.py` so
`validate_supervised_proof_report_next_wing_plan` extracts the proof-plan map,
requires it to be a `dict[str, list[str]]`, compares it to the decision-review
ledger map, and treats any non-empty decision-review map as invalid for
supervised proof. Also include the map in the generated no-WING JSON and
Markdown summary.

## Likely files to touch

- `tools/no_wing_release_check.py`
- `tests/test_no_wing_release_check.py`
- `.ai/project-layer/wing_mvp_blockers.md`
- `Docs/INDEX.md`
- `Docs/reports/MVP_PROGRESS_LATEST.md`

## Alternatives considered

Leave the map only in the backend proof-plan object. Rejected because the
operator-facing no-WING summary is the gate artifact used before the real WING
session.

Only render the map without validating it. Rejected because stale proof-plan
summaries could still appear to satisfy the gate while hiding a non-empty
decision-review violation map.

## Risks

The change can fail older proof-plan fixtures or generated reports that omit
the map. That is intentional fail-closed behavior for supervised proof gating.

## Test plan

Run focused no-WING tests, supervised proof-plan tests, Project Director,
direction guard, memory lint, `git diff --check`, and the standard full pytest
command.

## Where Kimi may disagree

Kimi may argue this is duplicate state because the decision-review ledger
already has the violations map. The no-WING proof-plan summary is the operator
handoff surface, so repeating and comparing the map is useful fail-closed
redundancy.
