# Automixer WING MVP Read-Only Runbook no-WING Manifest Sidecar Checklist

Date: 2026-06-15

## Summary

The WING read-only preflight runbook now requires operators to inspect the
detached no-WING `artifact-manifest.json` in the same artifact directory as the
passed prestart `summary.json` before opening a backend/WING WebSocket. The
checklist names the manifest self-consistency summary-surface match flag,
mismatch lists, and readiness/live-authority denial fields that must remain
clean.

## Safety

This is a runbook and rails-test change only. It does not contact WING, query a
console, mutate mixer state, close readiness, approve live writes, or grant live
authority.

## Validation

Targeted rails validation:

```bash
PYTHONPATH=backend python -m pytest -q tests/test_wing_mvp_rails.py::test_readonly_runbook_requires_no_wing_manifest_summary_surface_sidecar_gate
```

Result:

```text
1 passed in 0.02s
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
`/tmp/automixer_nowing_manifest_sidecar_check/20260617T075732Z_no_wing_release_check`

Observed detached sidecar verdict:

```text
artifact_manifest_self_consistency_summary_surface_match: true
artifact_manifest_self_consistency_summary_surface_json_mismatches: []
artifact_manifest_self_consistency_summary_surface_markdown_mismatches: []
artifact_manifest_self_consistency_summary_surface_mismatches: []
artifact_manifest_self_consistency_summary_surface_readiness_closure_allowed: false
artifact_manifest_self_consistency_summary_surface_live_authority_granted: false
artifact_manifest_self_consistency_match: true
artifact_manifest_self_consistency_mismatches: []
artifact_manifest_contract_cross_surface_match: true
artifact_manifest_contract_cross_surface_mismatches: []
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

Full standard validation:

```bash
PYTHONPATH=backend python -m pytest tests/ -x --tb=short -q
```

Result:

```text
3758 passed, 1 skipped, 1 warning in 47.20s
```

Manifest verifier validation:

```bash
PYTHONPATH=backend python tools/verify_no_wing_artifact_manifest.py /tmp/automixer_nowing_manifest_sidecar_check/20260617T075732Z_no_wing_release_check
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
