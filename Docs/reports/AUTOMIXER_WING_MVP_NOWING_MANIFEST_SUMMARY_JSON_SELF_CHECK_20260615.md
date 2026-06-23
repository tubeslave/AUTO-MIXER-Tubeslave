# Automixer WING MVP no-WING Manifest Summary JSON Self-Check

Date: 2026-06-15

## Summary

The no-WING manifest summary surface now computes
`artifact_manifest_summary_surface_missing_fields` from the required JSON field
list. A stale or incomplete `summary.json` can no longer claim
`missing_fields: []` without carrying the required manifest fields.

## Safety

This is an offline no-WING reporting change. It does not contact WING, query a
console, mutate mixer state, close readiness, or grant live authority.

## Validation

Targeted validation:

```bash
PYTHONPATH=backend python -m pytest -q tests/test_no_wing_release_check.py::test_no_wing_release_check_summary_embeds_next_wing_proof_plan tests/test_no_wing_release_check.py::test_no_wing_release_check_fails_when_manifest_summary_surface_missing_line tests/test_no_wing_release_check.py::test_no_wing_release_check_fails_when_manifest_summary_json_surface_field_missing tests/test_no_wing_release_check.py::test_no_wing_release_check_fails_when_manifest_contract_digest_line_missing tests/test_project_director_agent.py::test_project_director_selects_first_non_closed_mvp_blocker tests/test_wing_mvp_rails.py::test_progress_tool_generates_latest_report_from_blocker_table
```

Result:

```text
6 passed in 0.26s
```

Focused validation:

```bash
PYTHONPATH=backend python -m pytest -q tests/test_no_wing_release_check.py tests/test_project_director_agent.py tests/test_wing_mvp_rails.py
```

Result:

```text
315 passed, 8 skipped in 3.53s
```

Standard validation:

```bash
PYTHONPATH=backend python -m pytest tests/ -x --tb=short -q
```

Result:

```text
3390 passed, 9 skipped in 50.87s
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
no-WING quick: passed; artifact_manifest_summary_surface_missing_fields=[]; artifact_manifest_summary_surface_missing_field_lines=[]; artifact_manifest_contract_cross_surface_match=True; live_write=False; artifact_manifest_live_authority_granted=False; artifact_manifest_readiness_closure_allowed=False
git diff --check: passed
memory_lint: PASS
Project Director: guard_status=on_target; guard_score=90; selected_blocker=WING read-only connection proof; selected_blocker_status=tooling_ready_waits_real_wing; selected_action_source=Current First Command; selected_evidence_file_count=172; current_first_command_baseline_coverage_passed=true; current_first_command_baseline_missing_markers=[]
```

## Readiness Impact

No blocker was closed. The first MVP blocker remains `WING read-only connection
proof`, which still requires a real WING read-only preflight capture and manual
operator review.
