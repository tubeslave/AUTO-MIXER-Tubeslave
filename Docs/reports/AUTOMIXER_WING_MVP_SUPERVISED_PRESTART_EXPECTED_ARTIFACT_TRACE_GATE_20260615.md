# Automixer WING MVP Supervised Prestart Expected Artifact Trace Gate

## Summary

The structured `supervised_prestart_gate` now has to require
`requires_accepted_candidate_expected_artifact_trace: true` before supervised
fader/gain proof planning can be treated as aligned.

## Why It Matters

The operator step already names an expected-artifact-bound accepted source id.
This change makes the same requirement machine-checkable in the shared gate
object, alongside the existing prestart and artifact-root trace requirements.

## Safety Boundary

This is offline contract hardening only. It does not contact WING, start a
backend, grant live authority, allow readiness closure, or mutate console
state.

## Validation

Current validation on 2026-06-17:

```bash
PYTHONPATH=backend /opt/homebrew/bin/python3 -m pytest -q tests/test_no_wing_release_check.py tests/test_wing_readonly_preflight_handoff_packet.py tests/test_project_director_agent.py tests/test_supervised_wing_proof_plan.py tests/test_codex_direction_guard.py
# 441 passed in 4.46s

rm -rf /tmp/automixer_supervised_prestart_expected_artifact_trace_gate
PYTHONPATH=backend /opt/homebrew/bin/python3 tools/no_wing_release_check.py --quick --skip-frontend --artifact-root /tmp/automixer_supervised_prestart_expected_artifact_trace_gate
# artifact: /tmp/automixer_supervised_prestart_expected_artifact_trace_gate/20260617T101110Z_no_wing_release_check
# status='passed'
# failed_checks=[]
# live_authority_granted=False
# readiness_closure_allowed_by_packet=False
# full_autonomous_ready=False
# supervised_prestart_gate_requires_accepted_candidate_prestart_trace=True
# supervised_prestart_gate_requires_accepted_candidate_artifact_root_trace=True
# supervised_prestart_gate_requires_accepted_candidate_expected_artifact_trace=True
# backend_port_prestart_gate_matches_expected=True
# supervised_prestart_gate_matches_expected=True
# project_director_current_first_command_baseline_coverage_marker_count=288
# project_director_current_first_command_baseline_missing_markers=[]
# next_manual_artifact_required='real wing_readonly_preflight.v1 summary.json'
# artifact_manifest_contract_cross_surface_match=True
# artifact_manifest_self_consistency_match=True
# artifact_manifest_self_consistency_summary_surface_match=True

PYTHONPATH=backend /opt/homebrew/bin/python3 tools/verify_no_wing_artifact_manifest.py /tmp/automixer_supervised_prestart_expected_artifact_trace_gate/20260617T101110Z_no_wing_release_check
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
# selected_blocker_status='tooling_ready_waits_real_wing'
# selected_action_source='Current First Command'
# selected_evidence_file_count=205
# current_first_command_baseline_coverage_marker_count=288
# current_first_command_baseline_missing_markers=[]

PYTHONPATH=backend /opt/homebrew/bin/python3 -m pytest tests/ -x --tb=short -q
# 3758 passed, 1 skipped, 1 warning in 48.70s

PYTHONPATH=backend /opt/homebrew/bin/python3 tools/memory/memory_lint.py
# Status: PASS

/opt/homebrew/bin/git diff --check -- .ai/briefs/wing-mvp-supervised-prestart-expected-artifact-trace-gate-20260615.md .ai/proposals/codex-wing-mvp-supervised-prestart-expected-artifact-trace-gate-20260615.md Docs/adr/wing-mvp-supervised-prestart-expected-artifact-trace-gate-20260615.md Docs/reports/AUTOMIXER_WING_MVP_SUPERVISED_PRESTART_EXPECTED_ARTIFACT_TRACE_GATE_20260615.md
# PASS
```

This remains an offline contract gate. It does not close readiness or grant live
authority; the next manual artifact is still a real
`wing_readonly_preflight.v1` `summary.json`.
