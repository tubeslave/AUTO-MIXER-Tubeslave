# Automixer WING MVP Decision Review Invalid-Record Diagnostics

Date: 2026-06-15

## Summary

The standalone WING read-only decision-review report now surfaces invalid
decision records directly in JSON and markdown. Operators can see stale pending
`template_status` markers and exact ledger `violations` without digging through
the nested ledger payload.

## Changed

- Added top-level `invalid_records` to decision-review reports.
- Rendered an `Invalid Decision Records` markdown section.
- Included `review_id`, `artifact_source_id`, `decision`, `template_status`,
  `violations`, and `unsafe_authority_flags`.
- Added a regression for accepted records that still carry
  `operator_must_replace_placeholders_before_ledger_acceptance`.

## Safety

This is an offline reporting change. It does not contact WING, mutate console
state, update readiness rows, approve supervised proof, or grant live authority.

## Validation

Initial targeted validation:

```bash
PYTHONPATH=backend python -m pytest -q tests/test_wing_readonly_preflight_review_decision_tool.py::test_decision_review_report_surfaces_unedited_template_status tests/test_wing_readonly_preflight_review_decision.py::test_review_decision_ledger_rejects_unedited_template_status
```

Result:

```text
2 passed in 0.09s
```

Focused validation:

```bash
PYTHONPATH=backend python -m pytest -q tests/test_wing_readonly_preflight_evidence.py tests/test_wing_readonly_preflight_review_tool.py tests/test_wing_readonly_preflight_review_decision.py tests/test_wing_readonly_preflight_review_decision_tool.py tests/test_supervised_wing_proof_plan.py tests/test_no_wing_release_check.py tests/test_project_director_agent.py tests/test_wing_readonly_preflight_handoff_packet.py tests/test_wing_mvp_rails.py
```

Result:

```text
472 passed, 8 skipped in 3.90s
```

Standard validation:

```bash
PYTHONPATH=backend python -m pytest tests/ -x --tb=short -q
```

Result:

```text
3384 passed, 9 skipped in 49.91s
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
Project Director: guard_status=on_target; guard_score=90; selected_blocker=WING read-only connection proof; selected_blocker_status=tooling_ready_waits_real_wing; selected_action_source=Current First Command; selected_evidence_file_count=166; current_first_command_baseline_coverage_passed=true; current_first_command_baseline_missing_markers=[]
```

## Readiness Impact

No readiness row was closed. The first MVP blocker remains
`WING read-only connection proof` and still requires a real WING read-only
preflight capture plus manual operator review.
