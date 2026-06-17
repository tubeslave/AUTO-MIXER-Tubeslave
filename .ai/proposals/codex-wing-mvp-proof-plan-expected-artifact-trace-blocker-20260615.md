# Codex Proposal: WING MVP Proof Plan Expected Artifact Trace Blocker

## Short thesis

The supervised WING proof plan should treat an incomplete accepted-candidate
expected artifact trace as a hard blocker before physical proof steps are
listed.

## What I understood

The current Goal-mode direction is to work the first WING MVP blocker rather
than add broader proposal-only DSP methods. The handoff contract already says
the final supervised prestart gate requires an expected-artifact-bound accepted
source id, and no-WING summaries already surface the trace. The proof-plan
builder should enforce the same boundary directly.

## Proposed solution

- Add a proof-plan blocker:
  `accepted_readonly_preflight_candidate_expected_artifact_trace_required_before_supervised_proof`.
- Require
  `readonly_preflight_review_decision_accepted_candidate_expected_artifact_trace_complete`
  before `review_decision_accepted` can become true.
- Preserve existing prestart and artifact-root blockers.
- Add regression assertions for the success trace and the broken
  expected-artifact binding case.

## Likely files to touch

- `backend/supervised_wing_proof_plan.py`
- `tests/test_supervised_wing_proof_plan.py`
- `.ai/briefs/wing-mvp-proof-plan-expected-artifact-trace-blocker-20260615.md`
- `Docs/adr/wing-mvp-proof-plan-expected-artifact-trace-blocker-20260615.md`
- `Docs/reports/AUTOMIXER_WING_MVP_PROOF_PLAN_EXPECTED_ARTIFACT_TRACE_BLOCKER_20260615.md`
- `Docs/INDEX.md`

## Alternatives considered

- Rely only on no-WING value-gate validation. Rejected because the proof-plan
  builder should not list physical proof steps from an incomplete trace even if
  a later release check would fail.
- Fold this into the prestart blocker only. Rejected because the handoff
  contract names expected-artifact binding as its own supervised prestart
  condition.

## Risks

The main risk is duplicate blockers when a candidate fails both prestart and
expected-artifact trace checks. That is acceptable because both identify the
same fail-closed boundary from different operator review surfaces.

## Test plan

- `PYTHONPATH=backend python -m pytest -q tests/test_supervised_wing_proof_plan.py tests/test_no_wing_release_check.py tests/test_project_director_agent.py tests/test_codex_direction_guard.py`
- `PYTHONPATH=backend python -m pytest tests/ -x --tb=short -q`
- `PYTHONPATH=backend python tools/no_wing_release_check.py --quick --skip-frontend`
- `PYTHONPATH=backend python tools/memory/memory_lint.py`
- `/opt/homebrew/bin/git diff --check`

## Where the other agent may disagree

Kimi may prefer a new helper in `wing_readonly_preflight_contract.py` for the
trace completion verdict. This patch keeps the change inside the proof-plan
builder because the trace summary already exists there and the slice is narrow.
