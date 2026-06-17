# Automixer WING MVP Backend Prestart Summary JSON Surface

Date: 2026-06-15

## Summary

The no-WING `summary.json` now exposes the backend prestart summary binding at
the top level. Machine consumers can verify the bound prestart summary path,
SHA256, type, status, connection policy, offline safety flags, and match
verdicts without parsing the nested handoff packet artifact.

## Safety

This is an offline no-WING reporting change. It does not contact WING, query a
console, mutate mixer state, close readiness, approve live writes, or grant live
authority.

## Validation

Current validation on 2026-06-17:

```bash
PYTHONPATH=backend /opt/homebrew/bin/python3 -m py_compile tools/no_wing_release_check.py tests/test_no_wing_release_check.py
# PASS

PYTHONPATH=backend /opt/homebrew/bin/python3 -m pytest -q tests/test_no_wing_release_check.py::test_no_wing_release_check_summary_embeds_wing_preflight_handoff_packet_artifact
# 1 passed in 0.11s

rm -rf /tmp/automixer_backend_prestart_summary_surface
PYTHONPATH=backend /opt/homebrew/bin/python3 tools/no_wing_release_check.py --quick --skip-frontend --artifact-root /tmp/automixer_backend_prestart_summary_surface
# artifact: /tmp/automixer_backend_prestart_summary_surface/20260617T100343Z_no_wing_release_check
# status='passed'
# failed_checks=[]
# readiness_closure_allowed_by_packet=False
# live_authority_granted=False
# full_autonomous_ready=False
# backend_port_prestart_summary_surface_required=True
# backend_port_prestart_summary_required=True
# backend_port_prestart_summary_path_placeholder='/tmp/automixer_backend_prestart_summary_surface/20260617T100343Z_no_wing_release_check/backend_port_prestart/summary.json'
# plan_backend_port_prestart_summary_path='/tmp/automixer_backend_prestart_summary_surface/20260617T100343Z_no_wing_release_check/backend_port_prestart/summary.json'
# backend_port_prestart_summary_sha256_available=True
# backend_port_prestart_summary_sha256_matches_plan=True
# backend_port_prestart_summary_type='no_wing_release_check'
# backend_port_prestart_summary_status='passed'
# backend_port_prestart_summary_connection_policy='no_hardware_no_network_no_live_writes'
# backend_port_prestart_summary_read_only=True
# backend_port_prestart_summary_live_write=False
# backend_port_prestart_summary_contacts_backend=False
# backend_port_prestart_summary_contacts_wing=False
# backend_port_prestart_summary_mutates_console_state=False
# backend_port_prestart_summary_queries_console=False
# backend_port_prestart_check_status='passed'
# artifact_manifest_contract_cross_surface_match=True
# artifact_manifest_self_consistency_match=True
# artifact_manifest_self_consistency_summary_surface_match=True

PYTHONPATH=backend /opt/homebrew/bin/python3 tools/verify_no_wing_artifact_manifest.py /tmp/automixer_backend_prestart_summary_surface/20260617T100343Z_no_wing_release_check
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

PYTHONPATH=backend /opt/homebrew/bin/python3 -m pytest -q tests/test_no_wing_release_check.py tests/test_project_director_agent.py tests/test_wing_mvp_rails.py
# 393 passed in 4.28s

PYTHONPATH=backend /opt/homebrew/bin/python3 -m pytest tests/ -x --tb=short -q
# 3758 passed, 1 skipped, 1 warning in 47.44s

PYTHONPATH=backend /opt/homebrew/bin/python3 tools/memory/memory_lint.py
# Status: PASS

/opt/homebrew/bin/git diff --check -- .ai/briefs/wing-mvp-backend-prestart-summary-json-surface-20260615.md .ai/proposals/codex-wing-mvp-backend-prestart-summary-json-surface-20260615.md Docs/adr/wing-mvp-backend-prestart-summary-json-surface-20260615.md Docs/reports/AUTOMIXER_WING_MVP_BACKEND_PRESTART_SUMMARY_JSON_SURFACE_20260615.md
# PASS
```

The guard remains closed for live readiness: the first open blocker is still
`WING read-only connection proof`.
