# Automixer WING MVP Next Proof Plan Required Fields Summary JSON Surface

Date: 2026-06-15

## Summary

The no-WING `summary.json` now exposes next-proof-plan required and missing
supervised summary fields at the top level. Machine consumers can verify
`required_supervised_proof_plan_summary_fields`,
`missing_required_supervised_proof_plan_summary_fields`, the expected contract,
and the empty-missing-field verdict without parsing the nested
`next_wing_proof_plan`.

## Safety

This is an offline no-WING reporting change. It does not contact WING, query a
console, mutate mixer state, close readiness, approve live writes, or grant
live authority.

## Validation

Current validation on 2026-06-17:

```bash
PYTHONPATH=backend /opt/homebrew/bin/python3 -m pytest -q tests/test_no_wing_release_check.py::test_no_wing_release_check_summary_embeds_next_wing_proof_plan
# 1 passed in 0.10s

PYTHONPATH=backend /opt/homebrew/bin/python3 -m pytest -q tests/test_no_wing_release_check.py tests/test_project_director_agent.py tests/test_wing_mvp_rails.py
# 393 passed in 4.42s

rm -rf /tmp/automixer_next_proof_plan_required_fields_surface
PYTHONPATH=backend /opt/homebrew/bin/python3 tools/no_wing_release_check.py --quick --skip-frontend --artifact-root /tmp/automixer_next_proof_plan_required_fields_surface
# artifact: /tmp/automixer_next_proof_plan_required_fields_surface/20260617T101423Z_no_wing_release_check
# status='passed'
# failed_checks=[]
# live_authority_granted=False
# readiness_closure_allowed_by_packet=False
# full_autonomous_ready=False
# next_wing_proof_plan_required_fields_summary_surface_required=True
# missing_required_supervised_proof_plan_summary_fields=[]
# next_wing_proof_plan_required_supervised_proof_plan_summary_fields_match_contract=True
# next_wing_proof_plan_missing_required_supervised_proof_plan_summary_fields_empty=True
# next_wing_proof_plan_required_fields_summary_surface_readiness_closure_allowed=False
# next_wing_proof_plan_required_fields_summary_surface_live_authority_granted=False
# artifact_manifest_contract_cross_surface_match=True
# artifact_manifest_self_consistency_match=True
# artifact_manifest_self_consistency_summary_surface_match=True

PYTHONPATH=backend /opt/homebrew/bin/python3 tools/verify_no_wing_artifact_manifest.py /tmp/automixer_next_proof_plan_required_fields_surface/20260617T101423Z_no_wing_release_check
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
# 3758 passed, 1 skipped, 1 warning in 47.68s

PYTHONPATH=backend /opt/homebrew/bin/python3 tools/memory/memory_lint.py
# Status: PASS

/opt/homebrew/bin/git diff --check -- .ai/briefs/wing-mvp-next-proof-plan-required-fields-summary-json-surface-20260615.md .ai/proposals/codex-wing-mvp-next-proof-plan-required-fields-summary-json-surface-20260615.md Docs/adr/wing-mvp-next-proof-plan-required-fields-summary-json-surface-20260615.md Docs/reports/AUTOMIXER_WING_MVP_NEXT_PROOF_PLAN_REQUIRED_FIELDS_SUMMARY_JSON_SURFACE_20260615.md
# PASS
```

This remains a reporting surface only. Required-field completeness does not
close readiness or grant live authority; the first open blocker remains
`WING read-only connection proof`.
