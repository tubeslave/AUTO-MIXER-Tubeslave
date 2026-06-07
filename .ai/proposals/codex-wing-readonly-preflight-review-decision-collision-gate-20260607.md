# Codex Proposal: WING Read-Only Preflight Review Decision Collision Gate

## Short Thesis

A real WING read-only preflight candidate should advance only when it has one
unambiguous operator review decision.

## What I Understood

The next supervised fader/gain proof is blocked until a clean preflight artifact
has an accepted decision record. If the decision root contains multiple valid
records for the same artifact, especially one accepted and one rejected, the
ledger must not pick the optimistic interpretation.

## Proposed Solution

After per-record validation, group valid decision records by
`artifact_source_id`. If more than one record targets the same source, mark all
records for that source invalid with a duplicate-source violation. If their
decisions disagree, also mark a conflicting-decision violation. The decision
review report should surface those source ids and return the existing blocked
exit code.

## Likely Files To Touch

- `backend/wing_readonly_preflight_review_decision.py`
- `tools/review_wing_readonly_preflight_decisions.py`
- `tools/no_wing_release_check.py`
- focused WING decision tests
- `Docs/runbooks/WING_READONLY_PREFLIGHT.md`
- `Docs/adr/wing-readonly-preflight-review-decision-collision-gate-20260607.md`
- `Docs/reports/AUTOMIXER_WING_READONLY_PREFLIGHT_REVIEW_DECISION_COLLISION_GATE_20260607.md`

## Alternatives Considered

- Accept the latest timestamped decision. This would silently overwrite operator
  intent and depends on filesystem ordering or clock trust.
- Accept any accepted record if no record is otherwise malformed. This is unsafe
  because a rejected record for the same artifact would be ignored.

## Risks

Operators may need to remove stale decision files before the ledger can proceed.
That is acceptable because readiness should depend on a single durable operator
decision.

## Test Plan

- `python3 -m py_compile backend/wing_readonly_preflight_review_decision.py tools/review_wing_readonly_preflight_decisions.py tools/no_wing_release_check.py`
- `PYTHONPATH=backend python3 -m pytest -q tests/test_wing_readonly_preflight_review_decision.py tests/test_wing_readonly_preflight_review_decision_tool.py`
- `PYTHONPATH=backend python3 -m pytest -q tests/test_wing_readonly_preflight_evidence.py tests/test_wing_readonly_preflight_review_tool.py tests/test_autonomy_promotion_readiness.py`
- `PYTHONPATH=backend python3 tools/no_wing_release_check.py --quick --skip-frontend --artifact-root /tmp/automixer_no_wing_review_decision_collision_gate_20260607`

## Where The Other Agent May Disagree

Kimi may argue that multiple JSON files can be resolved by timestamp. The safer
operator-proof contract is that ambiguity is a replacement requirement, not a
sorting problem.
