# Codex Proposal: WING Read-Only Preflight Session-Ready Review Checklist - 2026-06-07

## Short thesis

The manual review decision for a physical WING read-only preflight must include
an explicit `confirm_operator_session_ready` checklist id before it can be
accepted by the decision ledger.

## What I understood

The current MVP is not blocked by missing offline tooling as much as by needing
a real WING read-only artifact and a trustworthy manual review path. Since the
runner now records session-ready confirmation, the review layer must make the
operator explicitly inspect that field before allowing a decision record to
become the source for supervised proof planning.

## Proposed solution

Add `confirm_operator_session_ready` to the canonical
`REQUIRED_REVIEW_CHECKLIST_IDS` in
`backend/wing_readonly_preflight_review_decision.py`. Mirror the same id in the
offline review checklist, handoff packet, no-WING release gate, and runbook.

Expose `operator_session_confirmation_required` and
`operator_confirmed_session_ready` in read-only preflight evidence artifact rows
and review markdown so the checklist item has visible evidence to inspect.

## Likely files to touch

- `backend/wing_readonly_preflight_evidence.py`
- `backend/wing_readonly_preflight_review_decision.py`
- `tools/review_wing_readonly_preflight_artifacts.py`
- `tools/build_wing_readonly_preflight_handoff_packet.py`
- `tools/no_wing_release_check.py`
- focused WING preflight review/evidence/no-WING tests
- `Docs/runbooks/WING_READONLY_PREFLIGHT.md`
- `Docs/adr/wing-readonly-preflight-session-ready-review-checklist-20260607.md`
- `Docs/reports/AUTOMIXER_WING_READONLY_PREFLIGHT_SESSION_READY_REVIEW_CHECKLIST_20260607.md`

## Alternatives considered

- Rely on evidence validation only. Rejected because accepted operator decisions
  should document that the session-ready field was reviewed.
- Add free-form review notes instead of a checklist id. Rejected because the
  decision ledger needs a deterministic fail-closed contract.

## Risks

Older pending review decision templates without the new checklist id become
invalid. That is acceptable because the real read-only WING blocker is still
open and stale templates should be regenerated.

## Test plan

- Compile changed Python modules.
- Run focused WING preflight evidence, review, decision, handoff, and no-WING
  release-check tests.
- Run quick no-WING release check, direction guard, memory lint, diff check,
  and full pytest before commit.

## Where the other agent may disagree

Kimi may argue that session-ready confirmation is already covered by
`confirm_artifact_paths`. Keeping it separate is clearer: artifact paths prove
which files are reviewed, while session-ready confirmation proves the operator
checked the intended backend/WING session boundary.
