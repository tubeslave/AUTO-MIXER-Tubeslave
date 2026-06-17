# Automixer WING MVP Proof Plan Expected Artifact Trace Blocker

Date: 2026-06-15

## Summary

The supervised WING proof-plan builder now treats an incomplete accepted
read-only preflight candidate expected artifact trace as a hard blocker before
listing physical fader/gain proof steps.

The new blocker is:

`accepted_readonly_preflight_candidate_expected_artifact_trace_required_before_supervised_proof`

This keeps the proof plan aligned with the no-WING handoff requirement for an
expected-artifact-bound accepted source id before
`WING_SUPERVISED_FADER_GAIN_PROOF`.

The Project Director Current First Command baseline also requires this blocker
marker, so the selected action cannot silently omit the expected-artifact trace
fail-closed boundary.

## Safety

This is offline plan logic only. It does not contact WING, query a console,
start a backend, mutate mixer state, close readiness, approve live writes, or
grant live authority.

## Validation

Current validation on 2026-06-17:

```bash
PYTHONPATH=backend /opt/homebrew/bin/python3 -m pytest -q tests/test_supervised_wing_proof_plan.py tests/test_no_wing_release_check.py tests/test_project_director_agent.py tests/test_codex_direction_guard.py
# 420 passed in 4.27s

rm -rf /tmp/automixer_expected_artifact_trace_blocker
PYTHONPATH=backend /opt/homebrew/bin/python3 tools/no_wing_release_check.py --quick --skip-frontend --artifact-root /tmp/automixer_expected_artifact_trace_blocker
# artifact: /tmp/automixer_expected_artifact_trace_blocker/20260617T100735Z_no_wing_release_check
# status='passed'
# failed_checks=[]
# live_authority_granted=False
# readiness_closure_allowed_by_packet=False
# full_autonomous_ready=False
# operator_sequence_count=10
# operator_sequence_count_matches_contract=True
# backend_port_prestart_gate_matches_expected=True
# supervised_prestart_gate_matches_expected=True
# supervised_prestart_gate_requires_accepted_candidate_expected_artifact_trace=True
# readonly_preflight_review_decision_accepted_candidate_expected_artifact_trace_source_ids=[]
# readonly_preflight_review_decision_accepted_candidate_expected_artifact_trace_missing_source_ids=[]
# readonly_preflight_review_decision_accepted_candidate_expected_artifact_trace_complete=False
# next_wing_proof_plan_accepted_candidate_expected_artifact_trace_value_gate_required=True
# next_wing_proof_plan_accepted_candidate_expected_artifact_trace_value_gate_violations=[]
# next_wing_proof_plan_accepted_candidate_expected_artifact_trace_value_gate_passed=True
# next_wing_proof_plan_accepted_candidate_expected_artifact_trace_value_gate_readiness_closure_allowed=False
# next_wing_proof_plan_accepted_candidate_expected_artifact_trace_value_gate_live_authority_granted=False
# next_manual_artifact_required='real wing_readonly_preflight.v1 summary.json'
# artifact_manifest_contract_cross_surface_match=True
# artifact_manifest_self_consistency_match=True
# artifact_manifest_self_consistency_summary_surface_match=True

PYTHONPATH=backend /opt/homebrew/bin/python3 tools/verify_no_wing_artifact_manifest.py /tmp/automixer_expected_artifact_trace_blocker/20260617T100735Z_no_wing_release_check
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
# 3758 passed, 1 skipped, 1 warning in 47.53s

PYTHONPATH=backend /opt/homebrew/bin/python3 tools/memory/memory_lint.py
# Status: PASS

/opt/homebrew/bin/git diff --check -- .ai/briefs/wing-mvp-proof-plan-expected-artifact-trace-blocker-20260615.md .ai/proposals/codex-wing-mvp-proof-plan-expected-artifact-trace-blocker-20260615.md Docs/adr/wing-mvp-proof-plan-expected-artifact-trace-blocker-20260615.md Docs/reports/AUTOMIXER_WING_MVP_PROOF_PLAN_EXPECTED_ARTIFACT_TRACE_BLOCKER_20260615.md
# PASS
```

This package does not close readiness. The accepted-candidate expected artifact
trace is still incomplete in no-WING/offline evidence, and the next manual
artifact remains a real `wing_readonly_preflight.v1` `summary.json`.
