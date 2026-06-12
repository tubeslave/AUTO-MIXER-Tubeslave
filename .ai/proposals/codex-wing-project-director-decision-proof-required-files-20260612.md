# Codex Proposal: wing-project-director-decision-proof-required-files-20260612

## Short thesis

Project Director should require the decision-review and supervised proof-plan
files because the current selected action depends on accepted-candidate
prestart validation.

## What I understood

The first WING MVP blocker is still waiting on real read-only proof. The
automation path must remain pointed at the first blocker and must not treat
offline contracts as live readiness.

## Proposed solution

- Add `backend/wing_readonly_preflight_review_decision.py`.
- Add `tools/review_wing_readonly_preflight_decisions.py`.
- Add `backend/supervised_wing_proof_plan.py`.
- Extend Project Director tests so these remain required.

## Likely files to touch

- `tools/codex_autoloop/project_director_agent.py`
- `tests/test_project_director_agent.py`
- `Docs/adr/wing-project-director-decision-proof-required-files-20260612.md`
- `Docs/reports/AUTOMIXER_WING_PROJECT_DIRECTOR_DECISION_PROOF_REQUIRED_FILES_20260612.md`

## Alternatives considered

- Rely only on no-WING release check. Rejected: the selected action names
  accepted review ids and proof-plan blockers, so future cycles should inspect
  their direct implementation.
- Add all WING modules. Rejected: this would make the director noisy and less
  tied to the first blocker.

## Risks

The required list can grow too broad. This patch adds only files directly tied
to accepted-decision and proof-plan gating.

## Test plan

- `PYTHONPATH=backend python3 -m pytest -q tests/test_project_director_agent.py`
- `PYTHONPATH=backend python3 tools/codex_autoloop/project_director_agent.py --format json --print`
- `PYTHONPATH=backend python3 tools/codex_autoloop/codex_direction_guard.py --format json`
- `/opt/homebrew/bin/git diff --check`

## Where the other agent may disagree

Kimi may prefer a smaller required-file list. The selected action already
depends on these files, so listing them reduces planning ambiguity without
adding execution authority.
