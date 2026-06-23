# Automixer WING MVP Review Decision Review-ID Required Gate

Date: 2026-06-15

## Summary

The WING read-only review-decision ledger now fails closed when an operator
decision record omits the required `review_id` field.

## Changed

- Added `review_decision_review_id_missing` for explicit missing `review_id`.
- Kept the fallback ID only as an internal label for invalid rows.
- Added a ledger regression proving a missing `review_id` cannot count as
  accepted.
- Updated the read-only runbook and rails test to keep `review_id` present.

## Safety

This is an offline validation change. It does not contact WING, mutate console
state, update readiness rows, or grant live authority.

## Validation

Targeted validation:

```bash
PYTHONPATH=backend python -m pytest -q tests/test_wing_readonly_preflight_review_decision.py::test_review_decision_ledger_rejects_missing_review_id tests/test_wing_mvp_rails.py::test_runbooks_keep_readonly_before_supervised_execute
```

Result:

```text
2 passed in 0.27s
```

Focused validation:

```bash
PYTHONPATH=backend python -m pytest -q tests/test_wing_readonly_preflight_evidence.py tests/test_wing_readonly_preflight_review_tool.py tests/test_wing_readonly_preflight_review_decision.py tests/test_wing_readonly_preflight_review_decision_tool.py tests/test_supervised_wing_proof_plan.py tests/test_no_wing_release_check.py tests/test_project_director_agent.py tests/test_wing_readonly_preflight_handoff_packet.py tests/test_wing_mvp_rails.py
```

Result:

```text
470 passed, 8 skipped in 4.09s
```

Standard validation:

```bash
PYTHONPATH=backend python -m pytest tests/ -x --tb=short -q
```

Result:

```text
3382 passed, 9 skipped in 49.66s
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
Project Director: guard_status=on_target; guard_score=90; selected_blocker=WING read-only connection proof; selected_blocker_status=tooling_ready_waits_real_wing; selected_action_source=Current First Command; selected_evidence_file_count=163; current_first_command_baseline_coverage_passed=true; current_first_command_baseline_missing_markers=[]
```

## Readiness Impact

No readiness row was closed. The first MVP blocker remains
`WING read-only connection proof` and still requires a real WING read-only
preflight capture plus manual operator review.
