# Codex Proposal: Supervised WING Proof Plan

## 1. Short thesis

Generate the next physical proof checklist from current evidence, but keep the
tool non-executing and operator-controlled.

## 2. What I understood

The proof report and promotion evidence bridge can now say that current
artifacts are not enough to review fader/gain safety rows. The next gap is not
more static policy; it is a precise plan for the bounded WING proof runs needed
to create physical candidates.

## 3. Proposed solution

Add `backend/supervised_wing_proof_plan.py`. It reads promotion evidence rows
and produces:

- planned fader/gain proof steps;
- negative delta values only;
- existing runner command tuples;
- prerequisites and abort conditions;
- expected artifacts for proof review;
- safety metadata that denies execution, live widening, automatic safety
  update, and autonomous readiness.

Embed the plan into `tools/evaluate_supervised_proof_artifacts.py` output and
register it as a `shadow_only` method.

## 4. Likely files to touch

- `backend/supervised_wing_proof_plan.py`
- `tests/test_supervised_wing_proof_plan.py`
- `tools/evaluate_supervised_proof_artifacts.py`
- `tests/test_supervised_proof_artifact_report.py`
- `backend/processing_registry.py`
- `tests/test_processing_registry.py`
- `Docs/adr/supervised-wing-proof-plan-20260604.md`

## 5. Alternatives considered

- Execute the WING proof automatically: rejected because explicit operator
  approval is required.
- Leave the next proof as prose-only: rejected because the command shape and
  abort conditions should be testable.
- Plan positive gain tests: rejected because gain-up remains outside the safe
  proof scope.

## 6. Risks

- A generated command could be mistaken as permission. The plan explicitly sets
  execution authority to false and requires the operator.
- Existing blocked artifacts could be ignored. The plan carries blockers into
  the step prerequisites.
- The plan could drift from the runner CLI. Tests assert the command tuple uses
  the existing runner arguments.

## 7. Test plan

- Verify the plan module is transport-free.
- Verify it builds gain/fader proof steps without execution authority.
- Verify it skips value types already ready for manual safety review.
- Verify invalid artifacts block plan generation.
- Verify proof report embeds the plan.

## 8. Where the other agent may disagree

Kimi may prefer the next cycle to run WING proof directly. That requires
operator timing and hardware context. This plan makes the hardware run precise
and auditable without crossing the no-live-write boundary in this turn.
