# Automixer WING MVP Review Decision Template Required-Edits Surface

Date: 2026-06-15

## Summary

The generated WING read-only review report now exposes the required operator
edits for pending review decision templates, including removing or replacing the
generated pending `template_status` marker before ledger acceptance.

## Changed

- Added `review_decision_template_required_edits` to review report JSON.
- Rendered `required_edits` in the Review Decision Template markdown section.
- Updated `--decision-template-output` CLI help to mention the pending
  `template_status` marker.
- Added targeted tests for report JSON, markdown, and stale-template ledger
  violation visibility.

## Safety

This is an offline report/help-text change. It does not contact WING, mutate
console state, update readiness rows, or grant live authority.

## Validation

Targeted validation:

```bash
PYTHONPATH=backend python -m pytest -q tests/test_wing_readonly_preflight_review_tool.py::test_review_decision_template_required_fields_match_shared_contract tests/test_wing_readonly_preflight_review_tool.py::test_review_report_accepts_clean_artifact_without_live_authority tests/test_wing_readonly_preflight_review_tool.py::test_review_decision_template_links_clean_candidate_but_requires_operator_edit
```

Result:

```text
3 passed in 0.04s
```

Focused validation:

```bash
PYTHONPATH=backend python -m pytest -q tests/test_wing_readonly_preflight_evidence.py tests/test_wing_readonly_preflight_review_tool.py tests/test_wing_readonly_preflight_review_decision.py tests/test_wing_readonly_preflight_review_decision_tool.py tests/test_supervised_wing_proof_plan.py tests/test_no_wing_release_check.py tests/test_project_director_agent.py tests/test_wing_readonly_preflight_handoff_packet.py tests/test_wing_mvp_rails.py
```

Result:

```text
471 passed, 8 skipped in 3.84s
```

Standard validation:

```bash
PYTHONPATH=backend python -m pytest tests/ -x --tb=short -q
```

Result:

```text
3383 passed, 9 skipped in 49.88s
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
Project Director: guard_status=on_target; guard_score=90; selected_blocker=WING read-only connection proof; selected_blocker_status=tooling_ready_waits_real_wing; selected_action_source=Current First Command; selected_evidence_file_count=165; current_first_command_baseline_coverage_passed=true; current_first_command_baseline_missing_markers=[]
```

## Readiness Impact

No readiness row was closed. The first MVP blocker remains
`WING read-only connection proof` and still requires a real WING read-only
preflight capture plus manual operator review.
