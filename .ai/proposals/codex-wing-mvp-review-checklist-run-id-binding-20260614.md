# Codex Proposal: WING MVP Review Checklist Run-Id Binding

## 1. Short thesis

Add a required operator checklist item for event-log `run_id` binding so an
accepted WING read-only preflight decision explicitly confirms that the event
log belongs to the reviewed summary.

## 2. What I understood

The validator already blocks mismatched event-log run ids. The manual decision
record should make that same integrity check explicit, otherwise a complete
accepted decision can look current while relying on an older checklist.

## 3. Proposed solution

Extend the shared review checklist ids with
`confirm_event_log_run_id_binding`, add the matching review-tool label, and
add tests proving old checklist records are rejected.

## 4. Likely files to touch

- `backend/wing_readonly_preflight_contract.py`
- `tools/review_wing_readonly_preflight_artifacts.py`
- `tests/test_wing_readonly_preflight_review_decision.py`
- `tests/test_wing_readonly_preflight_review_tool.py`
- `tests/test_wing_readonly_preflight_handoff_packet.py`
- `tests/test_no_wing_release_check.py`
- `Docs/runbooks/WING_READONLY_PREFLIGHT.md`

## 5. Alternatives considered

- Rely on `confirm_event_log_readonly_sequence`: rejected because sequence and
  run identity are different review facts.
- Add free-text operator notes only: rejected because checklist ids are already
  validated by the ledger.

## 6. Risks

Existing accepted decision JSON created before this change will require
replacement with the expanded checklist. That is intentional for real readiness
evidence.

## 7. Test plan

Run focused review-decision, review-tool, handoff, and no-WING tests, then
standard pytest, no-WING quick runtime, `git diff --check`, memory lint, and
Project Director.

## 8. Where the other agent may disagree

Kimi may argue this is redundant because the validator already checks run ids.
The counterpoint is that accepted operator decisions are durable evidence and
must name each required manual confirmation explicitly.
