# Automixer WING MVP no-WING Review-Decision Invalid-Record Diagnostics

Date: 2026-06-15

## Summary

The no-WING release summary now mirrors invalid WING read-only review decision
records from the ledger. The operator-readable `summary.md` can show stale
pending `template_status` markers and exact `violations` when the review
decision ledger fails closed.

## Changed

- Added normalized `invalid_records` to the no-WING review-decision ledger
  schema detail.
- Propagated `invalid_records` into `summary.json`.
- Rendered `invalid_records` in operator-readable `summary.md`.
- Added tests for validator detail and summary markdown visibility.

## Safety

This is an offline reporting change. It does not contact WING, mutate console
state, update readiness rows, approve supervised proof, or grant live authority.

## Validation

Initial targeted validation:

```bash
PYTHONPATH=backend python -m pytest -q tests/test_no_wing_release_check.py::test_no_wing_release_check_surfaces_invalid_wing_preflight_review_decision_records tests/test_no_wing_release_check.py::test_no_wing_release_check_summary_renders_invalid_wing_review_decision_records tests/test_no_wing_release_check.py::test_no_wing_release_check_summary_embeds_wing_preflight_review_decision tests/test_wing_readonly_preflight_review_decision_tool.py::test_decision_review_report_surfaces_unedited_template_status
```

Result:

```text
4 passed in 1.21s
```

Focused validation:

```bash
PYTHONPATH=backend python -m pytest -q tests/test_wing_readonly_preflight_evidence.py tests/test_wing_readonly_preflight_review_tool.py tests/test_wing_readonly_preflight_review_decision.py tests/test_wing_readonly_preflight_review_decision_tool.py tests/test_supervised_wing_proof_plan.py tests/test_no_wing_release_check.py tests/test_project_director_agent.py tests/test_wing_readonly_preflight_handoff_packet.py tests/test_wing_mvp_rails.py
```

Result:

```text
474 passed, 8 skipped in 3.83s
```

Standard validation:

```bash
PYTHONPATH=backend python -m pytest tests/ -x --tb=short -q
```

Result:

```text
3386 passed, 9 skipped in 49.99s
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
no-WING quick: passed; failed_checks: []; live_write: false; artifact_manifest_live_authority_granted: false; artifact_manifest_readiness_closure_allowed: false; next_wing_proof_plan: passed; review_decision_ledger: passed; review_decision_invalid_records: []
git diff --check: passed
memory_lint: PASS
Project Director: guard_status=on_target; guard_score=90; selected_blocker=WING read-only connection proof; selected_blocker_status=tooling_ready_waits_real_wing; selected_action_source=Current First Command; selected_evidence_file_count=167; current_first_command_baseline_coverage_passed=true; current_first_command_baseline_missing_markers=[]
```

## Readiness Impact

No readiness row was closed. The first MVP blocker remains
`WING read-only connection proof` and still requires a real WING read-only
preflight capture plus manual operator review.
