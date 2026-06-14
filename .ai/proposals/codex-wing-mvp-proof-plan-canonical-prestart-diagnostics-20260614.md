# Codex Proposal: WING MVP Proof Plan Canonical Prestart Diagnostics

## Short thesis

Make the supervised WING proof plan and no-WING summary show the helper-computed
canonical prestart trace verdict, not only the reported candidate fields.

## What I understood

The current blocker is still the real WING read-only connection proof. The
offline gate already recomputes accepted-candidate prestart validity from
`backend.wing_readonly_preflight_contract`, but an operator reading the no-WING
summary cannot see the expected source IDs and expected violations map that led
to the gate result.

## Proposed solution

Add `readonly_preflight_review_decision_expected_candidate_prestart_source_ids`
and
`readonly_preflight_review_decision_expected_candidate_prestart_contract_violations_by_source`
to the supervised proof plan summary. In the no-WING validator, parse the
reported fields, recompute the expected fields through the shared contract
helper, fail on mismatch, and expose the recomputed expected fields in JSON and
Markdown summaries.

## Likely files to touch

- `backend/supervised_wing_proof_plan.py`
- `tools/no_wing_release_check.py`
- `tests/test_no_wing_release_check.py`
- `Docs/reports/AUTOMIXER_WING_MVP_PROOF_PLAN_CANONICAL_PRESTART_DIAGNOSTICS_20260614.md`
- `Docs/INDEX.md`
- `.ai/project-layer/wing_mvp_blockers.md`

## Alternatives considered

Leave diagnostics inside validator-only comparisons. Rejected because a
failing gate should show the computed reason without requiring code inspection.

Expose only a boolean. Rejected because source IDs and per-source violations
are the useful operator review handles.

## Risks

The main risk is making report output noisier. The added fields are scoped to
the WING proof-plan summary and do not change execution authority. The gate
remains fail-closed.

## Test plan

Run focused no-WING tests, focused WING/readiness rail tests, Project Director,
direction guard, no-WING quick fail-closed check, `py_compile`, memory lint,
`git diff --check`, and the standard full pytest command.

## Where the other agent may disagree

Kimi may prefer surfacing these diagnostics only in the read-only preflight
review decision ledger. That is useful too, but the supervised proof plan is
where the operator decides whether a WING proof can even be attempted.
