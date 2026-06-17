# WING MVP Supervised Prestart Trace Summary Fields

## Summary

The no-WING handoff summary now exposes top-level supervised prestart trace
fields for accepted-candidate prestart, artifact-root, and expected-artifact
requirements.

## Fields

- `supervised_prestart_gate_requires_accepted_candidate_prestart_trace`
- `supervised_prestart_gate_requires_accepted_candidate_artifact_root_trace`
- `supervised_prestart_gate_requires_accepted_candidate_expected_artifact_trace`

## Safety Boundary

- No WING hardware is contacted.
- No live write authority is granted.
- No readiness closure is allowed by these fields.
- The current WING read-only connection proof blocker remains open until a real
  `wing_readonly_preflight.v1` artifact is captured and manually reviewed.

## Validation

Current validation on 2026-06-17:

```bash
PYTHONPATH=backend /opt/homebrew/bin/python3 -m py_compile tools/no_wing_release_check.py tests/test_no_wing_release_check.py tests/test_project_director_agent.py
# PASS

PYTHONPATH=backend /opt/homebrew/bin/python3 -m pytest -q tests/test_no_wing_release_check.py tests/test_project_director_agent.py tests/test_wing_mvp_rails.py tests/test_wing_readonly_preflight_handoff_packet.py tests/test_supervised_wing_proof_plan.py tests/test_codex_direction_guard.py
# 449 passed in 4.55s

rm -rf /tmp/automixer_supervised_prestart_trace_summary_fields
PYTHONPATH=backend /opt/homebrew/bin/python3 tools/no_wing_release_check.py --quick --skip-frontend --artifact-root /tmp/automixer_supervised_prestart_trace_summary_fields
# artifact: /tmp/automixer_supervised_prestart_trace_summary_fields/20260617T102124Z_no_wing_release_check
# status='passed'
# failed_checks=[]
# live_authority_granted=False
# readiness_closure_allowed_by_packet=False
# full_autonomous_ready=False
# supervised_prestart_gate_requires_accepted_candidate_prestart_trace=True
# supervised_prestart_gate_requires_accepted_candidate_artifact_root_trace=True
# supervised_prestart_gate_requires_accepted_candidate_expected_artifact_trace=True
# project_director_current_first_command_baseline_coverage_marker_count=288
# project_director_current_first_command_baseline_missing_markers=[]
# next_manual_artifact_required='real wing_readonly_preflight.v1 summary.json'
# artifact_manifest_contract_cross_surface_match=True
# artifact_manifest_self_consistency_match=True
# artifact_manifest_self_consistency_summary_surface_match=True

PYTHONPATH=backend /opt/homebrew/bin/python3 tools/verify_no_wing_artifact_manifest.py /tmp/automixer_supervised_prestart_trace_summary_fields/20260617T102124Z_no_wing_release_check
# status='passed'
# failed_checks=[]
# artifact_manifest_contract_cross_surface_match=True
# artifact_manifest_self_consistency_match=True
# artifact_manifest_self_consistency_summary_surface_match=True
# read_only=True
# live_write=False
# contacts_wing=False
# contacts_backend=False
# mutates_console_state=False
# queries_console=False
# executes_tests=False
# readiness_closure_allowed=False
# live_authority_granted=False

PYTHONPATH=backend /opt/homebrew/bin/python3 tools/codex_autoloop/project_director_agent.py --format json --print
# guard_status='drifting'
# guard_score=30
# selected_blocker='WING read-only connection proof'
# selected_action_source='Current First Command'
# selected_evidence_file_count=205
# current_first_command_baseline_coverage_marker_count=288
# current_first_command_baseline_missing_markers=[]

PYTHONPATH=backend /opt/homebrew/bin/python3 -m pytest tests/ -x --tb=short -q
# 3758 passed, 1 skipped, 1 warning in 47.51s

PYTHONPATH=backend /opt/homebrew/bin/python3 tools/memory/memory_lint.py
# Status: PASS

/opt/homebrew/bin/git diff --check -- .ai/briefs/wing-mvp-supervised-prestart-trace-summary-fields-20260615.md .ai/proposals/codex-wing-mvp-supervised-prestart-trace-summary-fields-20260615.md Docs/adr/wing-mvp-supervised-prestart-trace-summary-fields-20260615.md Docs/reports/AUTOMIXER_WING_MVP_SUPERVISED_PRESTART_TRACE_SUMMARY_FIELDS_20260615.md
# PASS
```

These flat trace fields are offline review evidence only. They do not close
readiness or grant live authority.
