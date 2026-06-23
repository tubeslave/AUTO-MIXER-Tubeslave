# Codex Proposal: WING MVP Review Decision Run-Id Verdict Field

## 1. Short thesis

Store the event-log run-id match verdict in WING read-only preflight decision
records so accepted operator decisions preserve the same machine evidence the
artifact review used.

## 2. What I understood

The review report already exposes `event_log_run_id_matches_summary: true`, and
the checklist requires the operator to confirm run-id binding. The durable
decision JSON should also carry the verdict, otherwise old records can look
complete while omitting a now-critical integrity fact.

## 3. Proposed solution

Add `event_log_run_id_matches_summary` to the shared decision-template required
fields, emit it from the pending decision template, and reject decision records
where the field is absent or not true.

## 4. Likely files to touch

- `backend/wing_readonly_preflight_contract.py`
- `backend/wing_readonly_preflight_review_decision.py`
- `tools/review_wing_readonly_preflight_artifacts.py`
- `tests/test_wing_readonly_preflight_review_decision.py`
- `tests/test_wing_readonly_preflight_review_tool.py`
- `tests/test_wing_readonly_preflight_review_decision_tool.py`
- `tests/test_no_wing_release_check.py`
- `Docs/runbooks/WING_READONLY_PREFLIGHT.md`

## 5. Alternatives considered

- Rely only on `confirm_event_log_run_id_binding`: rejected because checklist
  ids capture operator confirmation, not the machine verdict from the artifact.
- Rely only on event-log hashes: rejected because hashes bind files, not the
  semantic row-level `run_id` check.

## 6. Risks

Existing decision JSON records created before this field will fail closed and
must be replaced. That is expected for readiness evidence.

## 7. Test plan

Run focused review-decision, decision-tool, review-tool, handoff, no-WING, WING
rail, and Project Director tests, then standard pytest, no-WING quick runtime,
`git diff --check`, memory lint, and Project Director.

## 8. Where the other agent may disagree

Kimi may argue the field duplicates the checklist id. The counterpoint is that
durable acceptance should snapshot both the operator confirmation and the exact
machine verdict used to declare the artifact clean.
