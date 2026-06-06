# Codex Proposal: Supervised WING Proof Read-Only Preflight Gate

## Short Thesis

Make `supervised_wing_proof_plan` enforce the intended hardware-session order:
read-only preflight first, then explicit-approval supervised fader/gain proof.

## What I Understood

The current project focus is the supervised WING Rack MVP. Earlier work created
read-only preflight artifacts and surfaced them in readiness, but the next WING
proof plan could still present physical proof steps before that preflight
candidate existed.

## Proposed Solution

- Let `build_supervised_wing_proof_plan` inspect the
  `autonomy_promotion_readiness.summary.wing_readonly_preflight_evidence`
  overlay.
- Add `blocked_by_readonly_preflight`.
- Return zero proof steps until `readiness_update_candidate_available=true`,
  `valid_candidate_count>0`, and `invalid_candidate_count=0`.
- When blocked by preflight, include one non-executing read-only preflight
  checklist step using `tools/wing_readonly_preflight_runner.py`.
- Keep invalid supervised proof artifacts as a separate fail-closed blocker.
- Add no-WING release-check validation for the preflight prerequisite.
- Embed `autonomy_promotion_readiness` in the supervised proof artifact report
  and pass that exact snapshot into the next WING proof plan.

## Likely Files To Touch

- `backend/supervised_wing_proof_plan.py`
- `tools/evaluate_supervised_proof_artifacts.py`
- `tests/test_supervised_wing_proof_plan.py`
- `tests/test_supervised_proof_artifact_report.py`
- `tools/no_wing_release_check.py`
- `tests/test_no_wing_release_check.py`
- `.ai/memory/project.md`
- `Docs/INDEX.md`

## Alternatives Considered

- Keep emitting physical proof commands and rely on operator instructions.
  Rejected because the plan itself should encode the safe hardware order.
- Gate only inside the physical proof runner. Rejected because the operator
  evidence report would still advertise the wrong next step.

## Risks

The next proof report may show fewer runnable steps until a preflight artifact
exists. That is the intended fail-closed behavior and reduces the chance of a
supervised apply proof being attempted before read-only connection/readback is
captured.

## Test Plan

- `python3 -m py_compile backend/supervised_wing_proof_plan.py backend/autonomy_promotion_readiness.py backend/wing_readonly_preflight_evidence.py backend/processing_registry.py tools/evaluate_supervised_proof_artifacts.py tools/no_wing_release_check.py`
- `PYTHONPATH=backend python3 -m pytest -q tests/test_supervised_wing_proof_plan.py tests/test_supervised_proof_artifact_report.py`
- `PYTHONPATH=backend python3 -m pytest -q tests/test_no_wing_release_check.py`
- `python3 tools/memory/memory_lint.py`
- `/opt/homebrew/bin/git diff --check`
- `PYTHONPATH=backend python3 -m pytest tests/ -x --tb=short -q`

## Where The Other Agent May Disagree

Kimi may prefer leaving sequencing to an operator runbook. I would keep the gate
in the plan because this is the machine-readable object that downstream reports
and no-WING release validation inspect.
