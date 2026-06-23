# Automixer WING MVP no-WING Manifest Summary Markdown Self-Check Render

Date: 2026-06-15

## Summary

The no-WING `summary.md` writer now renders computed manifest summary surface
self-check values after the Markdown surface validation runs. If a required
operator-readable manifest line is missing, `summary.md` no longer keeps the
static success rows `artifact_manifest_summary_surface_missing_field_lines: []`
and `artifact_manifest_summary_surface_field_lines_rendered: true`.

## Safety

This is an offline no-WING reporting change. It does not contact WING, query a
console, mutate mixer state, close readiness, or grant live authority.

## Validation

Targeted validation:

```bash
PYTHONPATH=backend python -m pytest -q tests/test_no_wing_release_check.py::test_no_wing_release_check_fails_when_manifest_summary_surface_missing_line tests/test_no_wing_release_check.py::test_no_wing_release_check_fails_when_manifest_summary_json_surface_field_missing tests/test_no_wing_release_check.py::test_no_wing_release_check_summary_markdown_renders_manifest_missing_line_self_check tests/test_no_wing_release_check.py::test_no_wing_release_check_fails_when_manifest_contract_digest_line_missing
```

Result:

```text
4 passed in 1.18s
```

Focused validation:

```bash
PYTHONPATH=backend python -m pytest -q tests/test_no_wing_release_check.py tests/test_project_director_agent.py tests/test_wing_mvp_rails.py
```

Result:

```text
316 passed, 8 skipped in 3.50s
```

Standard validation:

```bash
PYTHONPATH=backend python -m pytest tests/ -x --tb=short -q
```

Result:

```text
3391 passed, 9 skipped in 50.58s
```

Release/service checks:

```bash
PYTHONPATH=backend python tools/no_wing_release_check.py --quick --skip-frontend
PYTHONPATH=backend python tools/codex_autoloop/project_director_agent.py --format json --print
PYTHONPATH=backend python tools/memory/memory_lint.py
/opt/homebrew/bin/git diff --check
```

Result:

```text
no-WING quick: passed; artifact_manifest_summary_surface_missing_fields=[]; artifact_manifest_summary_surface_missing_field_lines=[]; artifact_manifest_summary_surface_field_lines_rendered=True; artifact_manifest_contract_cross_surface_match=True; live_write=False; artifact_manifest_live_authority_granted=False; artifact_manifest_readiness_closure_allowed=False
Project Director: guard_status=on_target; guard_score=90; selected_blocker=WING read-only connection proof; selected_blocker_status=tooling_ready_waits_real_wing; selected_action_source=Current First Command; selected_evidence_file_count=173; current_first_command_baseline_coverage_passed=true; current_first_command_baseline_missing_markers=[]
memory_lint: PASS
git diff --check: passed
```

## Readiness Impact

No blocker was closed. The first MVP blocker remains `WING read-only connection
proof`, which still requires a real WING read-only preflight capture and manual
operator review.
