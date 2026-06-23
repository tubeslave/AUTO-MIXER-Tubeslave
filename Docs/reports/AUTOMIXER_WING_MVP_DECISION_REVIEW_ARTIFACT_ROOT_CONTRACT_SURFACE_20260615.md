# Automixer WING MVP Decision Review Artifact Root Contract Surface

Date: 2026-06-15

## Summary

The standalone WING read-only decision-review report now requires and exposes
accepted-candidate artifact-root contract validity before it can report an
accepted decision as ready.

## Changed

- Required accepted-candidate artifact-root contract validity for
  `accepted_decision_ready`.
- Added accepted-candidate artifact-root contract fields to decision-review JSON
  and markdown output.
- Added fallback artifact-root blocker rendering when an accepted source exists
  but the ledger summary lacks blocker coverage.
- Added regressions for valid and invalid artifact-root contract decision
  reports.

## Safety

This is read-only/offline validation. It does not contact WING, mutate console
state, update readiness, or grant live authority.

## Validation

Targeted validation:

```bash
PYTHONPATH=backend python -m pytest -q tests/test_wing_readonly_preflight_review_decision_tool.py
```

Result:

```text
11 passed
```

Focused validation:

```bash
PYTHONPATH=backend python -m pytest -q tests/test_wing_readonly_preflight_review_decision.py tests/test_supervised_wing_proof_plan.py tests/test_no_wing_release_check.py tests/test_wing_readonly_preflight_review_decision_tool.py
```

Result:

```text
361 passed, 8 skipped in 3.39s
```

WING-focused validation:

```bash
PYTHONPATH=backend python -m pytest -q tests/test_wing_readonly_preflight_evidence.py tests/test_wing_readonly_preflight_review_tool.py tests/test_wing_readonly_preflight_review_decision.py tests/test_wing_readonly_preflight_review_decision_tool.py tests/test_supervised_wing_proof_plan.py tests/test_no_wing_release_check.py tests/test_project_director_agent.py
```

Result:

```text
441 passed, 8 skipped in 3.66s
```

Standard validation:

```bash
PYTHONPATH=backend python -m pytest tests/ -x --tb=short -q
```

Result:

```text
3381 passed, 9 skipped in 49.36s
```

Release/service checks:

```bash
PYTHONPATH=backend python tools/no_wing_release_check.py --quick --skip-frontend
/opt/homebrew/bin/git diff --check
PYTHONPATH=backend python tools/memory/memory_lint.py
PYTHONPATH=backend python tools/codex_autoloop/project_director_agent.py --format json --print
```

Result:

```text
no_wing_release_check: passed; failed_checks: []
live_write: false
artifact_manifest_live_authority_granted: false
artifact_manifest_readiness_closure_allowed: false
next_wing_proof_plan_status: passed
review_decision_ledger_status: passed
git diff --check: passed
memory_lint.py: PASS
project_director_agent: on_target; guard_score: 90; selected_blocker: WING read-only connection proof; selected_blocker_status: tooling_ready_waits_real_wing; selected_evidence_file_count: 160
```

## Readiness Impact

No readiness row was closed. The first MVP blocker remains
`WING read-only connection proof` and still requires a real WING read-only
preflight capture plus manual operator review.
