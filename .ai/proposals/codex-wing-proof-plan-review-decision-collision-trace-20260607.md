# Codex Proposal: WING Proof Plan Review Decision Collision Trace

## Short Thesis

The next WING supervised proof plan should explain exactly why a reviewed
read-only preflight candidate is still blocked when operator decision records
collide.

## What I Understood

The decision ledger already marks duplicate or conflicting decisions invalid.
The supervised proof plan consumes only counts and blockers, which is enough to
block execution but weak for operator repair. The source ids and invalid record
ids should be visible in the proof plan and no-WING summary.

## Proposed Solution

Copy the decision ledger's `invalid_record_ids`, `duplicate_source_ids`, and
`conflicting_source_ids` into the `supervised_wing_proof_plan` summary. Extend
the no-WING plan validator to require those fields to match the decision
overlay, and render them in generated summaries.

## Likely Files To Touch

- `backend/supervised_wing_proof_plan.py`
- `tools/no_wing_release_check.py`
- `tests/test_supervised_wing_proof_plan.py`
- `tests/test_no_wing_release_check.py`
- `Docs/runbooks/WING_SUPERVISED_FADER_GAIN_PROOF.md`
- `Docs/reports/AUTOMIXER_WING_PROOF_PLAN_REVIEW_DECISION_COLLISION_TRACE_20260607.md`

## Alternatives Considered

- Keep only counts in the proof plan. This blocks execution but makes operator
  repair slower and less predictable.
- Recompute duplicate decisions in the proof plan. This duplicates ledger logic
  and risks drift.

## Risks

Reports become a little more verbose. The tradeoff is worthwhile because the
extra fields identify the exact decision records that must be replaced before a
supervised proof.

## Test Plan

- `python3 -m py_compile backend/supervised_wing_proof_plan.py tools/no_wing_release_check.py`
- `PYTHONPATH=backend python3 -m pytest -q tests/test_supervised_wing_proof_plan.py tests/test_no_wing_release_check.py`
- `PYTHONPATH=backend python3 -m pytest -q tests/test_wing_readonly_preflight_review_decision.py tests/test_wing_readonly_preflight_review_decision_tool.py`
- `PYTHONPATH=backend python3 tools/no_wing_release_check.py --quick --skip-frontend --artifact-root /tmp/automixer_no_wing_proof_plan_decision_collision_trace_20260607`

## Where The Other Agent May Disagree

Kimi may prefer leaving diagnostics in the decision ledger only. The stronger
operator workflow is to carry the repair target into the exact proof-plan gate
that is blocking the next physical action.
