# Automixer WING MVP Handoff Invalid-Record Marker

Date: 2026-06-15

## Summary

The WING read-only handoff packet and no-WING handoff summary now expose
`required_supervised_proof_plan_summary_fields_include_invalid_records`. This
lets operators verify directly that the supervised proof-plan required field
contract still includes invalid review-decision record diagnostics.

## Changed

- Added the explicit invalid-record inclusion flag to handoff packet JSON.
- Rendered the flag in handoff markdown.
- Made no-WING handoff artifact validation fail closed when the declared flag is
  stale or missing.
- Propagated the flag into no-WING summary JSON and markdown.
- Added Project Director and WING rails coverage for the new marker.

## Safety

This is an offline handoff/reporting change. It does not contact WING, mutate
console state, update readiness rows, approve supervised proof, or grant live
authority.

## Validation

Targeted validation:

```bash
PYTHONPATH=backend python -m pytest -q tests/test_wing_readonly_preflight_handoff_packet.py tests/test_no_wing_release_check.py::test_no_wing_release_check_validates_wing_preflight_handoff_packet_artifact tests/test_no_wing_release_check.py::test_no_wing_release_check_rejects_handoff_stale_invalid_records_field_flag tests/test_no_wing_release_check.py::test_no_wing_release_check_summary_embeds_wing_preflight_handoff_packet_artifact tests/test_project_director_agent.py::test_project_director_selects_first_non_closed_mvp_blocker tests/test_project_director_agent.py::test_prompt_and_readme_document_project_director_agent tests/test_wing_mvp_rails.py::test_progress_tool_generates_latest_report_from_blocker_table
```

Result:

```text
27 passed in 1.89s
```

Focused validation:

```bash
PYTHONPATH=backend python -m pytest -q tests/test_wing_readonly_preflight_evidence.py tests/test_wing_readonly_preflight_review_tool.py tests/test_wing_readonly_preflight_review_decision.py tests/test_wing_readonly_preflight_review_decision_tool.py tests/test_supervised_wing_proof_plan.py tests/test_no_wing_release_check.py tests/test_project_director_agent.py tests/test_wing_readonly_preflight_handoff_packet.py tests/test_wing_mvp_rails.py
```

Result:

```text
477 passed, 8 skipped in 3.85s
```

Standard validation:

```bash
PYTHONPATH=backend python -m pytest tests/ -x --tb=short -q
```

Result:

```text
3389 passed, 9 skipped in 50.05s
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
no-WING quick: passed; failed_checks: []; live_write: false; artifact_manifest_live_authority_granted: false; artifact_manifest_readiness_closure_allowed: false; next_wing_proof_plan: passed; handoff_invalid_records_flag: true
git diff --check: passed
memory_lint: PASS
Project Director: guard_status=on_target; guard_score=90; selected_blocker=WING read-only connection proof; selected_blocker_status=tooling_ready_waits_real_wing; selected_action_source=Current First Command; selected_evidence_file_count=170; current_first_command_baseline_coverage_passed=true; current_first_command_baseline_coverage_marker_count=173; current_first_command_baseline_missing_markers=[]
```

## Readiness Impact

No readiness row was closed. The first MVP blocker remains
`WING read-only connection proof` and still requires a real WING read-only
preflight capture plus manual operator review.
