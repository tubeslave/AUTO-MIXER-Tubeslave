# ADR: WING Read-Only Preflight Decision-Gated Proof Plan

## Context

The supervised WING proof plan was already blocked until a clean read-only
preflight candidate existed. The next safety boundary is stricter: a clean
candidate is not enough by itself because the operator still has to review and
accept that candidate.

## Options considered

1. Let a clean preflight candidate unlock supervised fader/gain proof steps.
2. Require an accepted review decision ledger record before proof steps appear.
3. Remove proof-step planning until all readiness rows are manually closed.

## Decision

Require an accepted `wing_readonly_preflight_review_decision_ledger` record
before `backend/supervised_wing_proof_plan.py` emits supervised fader/gain proof
steps. If a clean preflight candidate exists but no accepted decision exists,
the plan reports `manual_review_ready` with zero proof steps and an explicit
review-decision blocker.

## Why this won

It preserves the correct WING MVP order without hiding the next action. The
operator can see that the preflight candidate is ready for decision review, but
the toolchain cannot present supervised write commands as the next step until
the accepted decision is recorded.

## Rejected alternatives

- Candidate-only unlock: too loose because it skips the durable operator
  decision gate.
- Wait for readiness-row closure: too strict because the supervised proof is
  part of the evidence needed before later readiness updates.

## Implementation plan

1. Add decision-ledger awareness to `backend/supervised_wing_proof_plan.py`.
2. Pass the ledger from `tools/evaluate_supervised_proof_artifacts.py`.
3. Teach `tools/no_wing_release_check.py` to reject proof steps without an
   accepted decision.
4. Update focused tests and WING MVP docs.

## Test plan

- `python3 -m py_compile backend/supervised_wing_proof_plan.py backend/wing_readonly_preflight_review_decision.py tools/evaluate_supervised_proof_artifacts.py tools/no_wing_release_check.py`
- `PYTHONPATH=backend python3 -m pytest -q tests/test_supervised_wing_proof_plan.py tests/test_supervised_proof_artifact_report.py tests/test_wing_readonly_preflight_review_decision.py tests/test_no_wing_release_check.py`
- `python3 tools/memory/memory_lint.py`
- `/opt/homebrew/bin/git diff --check`

## Risks and rollback

Risk is limited to offline planning/report validation. Rollback is to remove
the decision-ledger check from the proof plan and no-WING validator. No live
WING transport, proposal apply, rollback, emergency-stop, connect, or
disconnect path is changed.
