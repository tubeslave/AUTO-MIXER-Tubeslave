# Automixer WING MVP Current-Command Invalid-Record Marker

Date: 2026-06-15

## Summary

Project Director Current First Command now explicitly requires
`readonly_preflight_review_decision_invalid_records` in the supervised
proof-plan summary field list. This keeps the director from treating stale
proof-plan guidance as baseline-complete after the invalid-record diagnostics
contract was added.

## Changed

- Added `readonly_preflight_review_decision_invalid_records` to Current First
  Command.
- Added the same marker to Project Director baseline coverage.
- Updated Codex autoloop prompt and README guidance.
- Added Project Director assertions for selected action, prompt, and README
  coverage.

## Safety

This is a direction-guard and documentation change. It does not contact WING,
mutate console state, update readiness rows, approve supervised proof, or grant
live authority.

## Validation

Targeted validation:

```bash
PYTHONPATH=backend python -m pytest -q tests/test_project_director_agent.py::test_project_director_selects_first_non_closed_mvp_blocker tests/test_project_director_agent.py::test_prompt_and_readme_document_project_director_agent
PYTHONPATH=backend python -m pytest -q tests/test_project_director_agent.py
```

Result:

```text
2 passed in 0.12s
4 passed in 0.13s
```

Standard validation:

```bash
PYTHONPATH=backend python -m pytest tests/ -x --tb=short -q
```

Result:

```text
3388 passed, 9 skipped in 49.98s
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
no-WING quick: passed; failed_checks: []; live_write: false; artifact_manifest_live_authority_granted: false; artifact_manifest_readiness_closure_allowed: false; next_wing_proof_plan: passed; review_decision_ledger: passed
git diff --check: passed
memory_lint: PASS
Project Director: guard_status=on_target; guard_score=90; selected_blocker=WING read-only connection proof; selected_blocker_status=tooling_ready_waits_real_wing; selected_action_source=Current First Command; selected_evidence_file_count=169; current_first_command_baseline_coverage_passed=true; current_first_command_baseline_coverage_marker_count=172; current_first_command_baseline_missing_markers=[]
```

## Readiness Impact

No readiness row was closed. The first MVP blocker remains
`WING read-only connection proof` and still requires a real WING read-only
preflight capture plus manual operator review.
