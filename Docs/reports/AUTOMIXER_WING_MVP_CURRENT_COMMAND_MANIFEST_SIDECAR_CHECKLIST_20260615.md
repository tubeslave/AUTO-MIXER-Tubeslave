# Automixer WING MVP Current Command Manifest Sidecar Checklist

Date: 2026-06-15

## Summary

The Project Director current first command now requires the same no-WING
`artifact-manifest.json` summary-surface sidecar checklist as the read-only
runbook before a backend/WING WebSocket may be opened. The command names the
match flag, mismatch lists, detached manifest self-consistency fields, and
readiness/live-authority denial fields.

## Safety

This is a directive and rails-test change only. It does not contact WING, query
a console, mutate mixer state, close readiness, approve live writes, or grant
live authority.

## Validation

Targeted rails validation:

```bash
PYTHONPATH=backend python -m pytest -q tests/test_wing_mvp_rails.py::test_progress_tool_generates_latest_report_from_blocker_table tests/test_wing_mvp_rails.py::test_readonly_runbook_requires_no_wing_manifest_summary_surface_sidecar_gate
```

Result:

```text
2 passed in 0.06s
```

Project Director guard:

```bash
PYTHONPATH=backend python tools/codex_autoloop/project_director_agent.py --format json --print
```

Result:

```text
guard_status: drifting
guard_score: 30
guard_stop_the_line: true
selected_blocker: WING read-only connection proof
selected_blocker_status: tooling_ready_waits_real_wing
selected_action_source: Current First Command
selected_evidence_file_count: 205
current_first_command_baseline_coverage_passed: true
current_first_command_baseline_coverage_marker_count: 288
current_first_command_baseline_missing_markers: []
```

Focused WING/no-WING/Director validation:

```bash
PYTHONPATH=backend python -m pytest -q tests/test_no_wing_release_check.py tests/test_project_director_agent.py tests/test_wing_mvp_rails.py
```

Result:

```text
393 passed in 4.27s
```

no-WING quick validation:

```bash
PYTHONPATH=backend python tools/no_wing_release_check.py --quick --skip-frontend
```

Latest generated artifact root:
`/tmp/automixer_nowing_current_command_check/20260617T080322Z_no_wing_release_check`

Observed results:

```text
summary status: passed
summary.md contains artifact_manifest_self_consistency_summary_surface_match: true
summary.md contains same no-WING artifact directory
summary.md contains do not open the WING session
artifact-manifest.json artifact_manifest_self_consistency_summary_surface_match: true
artifact-manifest.json artifact_manifest_self_consistency_summary_surface_mismatches: []
artifact-manifest.json artifact_manifest_self_consistency_summary_surface_readiness_closure_allowed: false
artifact-manifest.json artifact_manifest_self_consistency_summary_surface_live_authority_granted: false
```

Full standard validation:

```bash
PYTHONPATH=backend python -m pytest tests/ -x --tb=short -q
```

Result:

```text
3758 passed, 1 skipped, 1 warning in 47.71s
```

Manifest verifier validation:

```bash
PYTHONPATH=backend python tools/verify_no_wing_artifact_manifest.py /tmp/automixer_nowing_current_command_check/20260617T080322Z_no_wing_release_check
```

Result:

```text
status: passed
failed_checks: []
artifact_manifest_self_consistency_match: true
artifact_manifest_contract_cross_surface_match: true
artifact_manifest_self_consistency_summary_surface_match: true
read_only: true
live_write: false
contacts_wing: false
contacts_backend: false
mutates_console_state: false
queries_console: false
executes_tests: false
readiness_closure_allowed: false
live_authority_granted: false
```
