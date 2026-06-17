# WING MVP Proof Plan Prestart Violations Step Gate

## Summary

The no-WING release checker now has an explicit fail-closed gate for supervised proof plans that still carry accepted-candidate prestart contract violations.

If `next_wing_proof_plan.summary.readonly_preflight_review_decision_candidate_prestart_contract_violations_by_source` is non-empty while physical proof steps are planned, `next_wing_proof_plan_schema` fails with:

- `next_wing_proof_plan_steps_with_candidate_prestart_violations`

The same condition also requires the existing blocker:

- `accepted_readonly_preflight_candidate_prestart_contract_required_before_supervised_proof`

## Why It Matters

The current WING MVP rail requires a valid accepted-candidate no-WING prestart trace before any supervised fader/gain proof. A non-empty prestart violations map means that proof is not ready, even if other plan fields look valid.

## Safety Boundary

- No live WING access was added.
- No mixer state mutation was added.
- No fader or gain increase was added.
- The first open blocker remains `WING read-only connection proof`.

## Validation

Current validation on 2026-06-17:

```bash
PYTHONPATH=backend /opt/homebrew/bin/python3 -m pytest -q tests/test_no_wing_release_check.py::test_no_wing_release_check_rejects_next_wing_plan_steps_with_candidate_prestart_violations
# 1 passed in 0.09s

PYTHONPATH=backend /opt/homebrew/bin/python3 -m pytest -q tests/test_wing_readonly_preflight_review_decision.py tests/test_supervised_wing_proof_plan.py tests/test_no_wing_release_check.py
# 439 passed in 4.20s

PYTHONPATH=backend /opt/homebrew/bin/python3 -m pytest -q tests/test_wing_readonly_preflight_evidence.py tests/test_wing_readonly_preflight_review_tool.py tests/test_wing_readonly_preflight_review_decision.py tests/test_supervised_wing_proof_plan.py tests/test_no_wing_release_check.py tests/test_project_director_agent.py
# 520 passed in 4.51s

rm -rf /tmp/automixer_proof_plan_prestart_violations_step_gate
PYTHONPATH=backend /opt/homebrew/bin/python3 tools/no_wing_release_check.py --quick --skip-frontend --artifact-root /tmp/automixer_proof_plan_prestart_violations_step_gate
# artifact: /tmp/automixer_proof_plan_prestart_violations_step_gate/20260617T095331Z_no_wing_release_check
# status='passed'
# failed_checks=[]
# readiness_closure_allowed_by_packet=False
# live_authority_granted=False
# full_autonomous_ready=False
# readonly_preflight_review_decision_candidate_prestart_contract_violations_by_source={}
# artifact_manifest_contract_cross_surface_match=True
# artifact_manifest_self_consistency_match=True
# artifact_manifest_self_consistency_summary_surface_match=True

PYTHONPATH=backend /opt/homebrew/bin/python3 tools/verify_no_wing_artifact_manifest.py /tmp/automixer_proof_plan_prestart_violations_step_gate/20260617T095331Z_no_wing_release_check
# status='passed'
# failed_checks=[]
# artifact_manifest_contract_cross_surface_match=True
# artifact_manifest_self_consistency_match=True
# artifact_manifest_self_consistency_summary_surface_match=True
# read_only=True
# live_write=False
# contacts_wing=False
# mutates_console_state=False
# queries_console=False
# readiness_closure_allowed=False
# live_authority_granted=False

PYTHONPATH=backend /opt/homebrew/bin/python3 tools/codex_autoloop/project_director_agent.py --format json --print
# guard_status='drifting'
# guard_score=30
# selected_blocker='WING read-only connection proof'
# selected_evidence_file_count=205
# current_first_command_baseline_coverage_marker_count=288
# current_first_command_baseline_missing_markers=[]

PYTHONPATH=backend /opt/homebrew/bin/python3 -m pytest tests/ -x --tb=short -q
# 3758 passed, 1 skipped, 1 warning in 46.50s

PYTHONPATH=backend /opt/homebrew/bin/python3 tools/memory/memory_lint.py
# Status: PASS

/opt/homebrew/bin/git diff --check -- .ai/briefs/wing-mvp-proof-plan-prestart-violations-step-gate-20260614.md .ai/proposals/codex-wing-mvp-proof-plan-prestart-violations-step-gate-20260614.md Docs/adr/wing-mvp-proof-plan-prestart-violations-step-gate-20260614.md Docs/reports/AUTOMIXER_WING_MVP_PROOF_PLAN_PRESTART_VIOLATIONS_STEP_GATE_20260614.md
# PASS
```

The guard remains intentionally closed for live readiness: the first open blocker is still `WING read-only connection proof`.
