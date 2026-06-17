# Automixer WING MVP Accepted Candidate Expected Artifact Trace Required Fields

Date: 2026-06-15

## Summary

The accepted-candidate expected artifact trace is now part of the canonical
required supervised proof-plan summary contract. Generated
`next_wing_proof_plan.summary` objects must carry source-id keyed trace fields
for expected artifacts, expected summary/events artifacts, summary/events path
match flags, missing trace sources, and the trace completion verdict.

This does not close readiness. In the current no-WING state the plan remains
`blocked_by_readonly_preflight`, the trace source list is empty, and
`readonly_preflight_review_decision_accepted_candidate_expected_artifact_trace_complete`
is `false`.

## Safety

This is an offline schema and reporting change. It does not contact WING, query
a console, mutate mixer state, close readiness, approve live writes, or grant
live authority.

## Validation

- PASS: `PYTHONPATH=backend /opt/homebrew/bin/python3 -m py_compile backend/supervised_wing_proof_plan.py backend/wing_readonly_preflight_contract.py tools/no_wing_release_check.py tests/test_no_wing_release_check.py tests/test_supervised_wing_proof_plan.py`.
- PASS: `PYTHONPATH=backend /opt/homebrew/bin/python3 -m pytest -q tests/test_supervised_wing_proof_plan.py`
  returned `22 passed in 0.05s`.
- PASS: `PYTHONPATH=backend /opt/homebrew/bin/python3 -m pytest -q tests/test_no_wing_release_check.py::test_no_wing_release_check_rejects_next_wing_plan_missing_required_summary_field tests/test_no_wing_release_check.py::test_no_wing_release_check_summary_embeds_next_wing_proof_plan`
  returned `32 passed in 0.21s`.
- PASS: `PYTHONPATH=backend /opt/homebrew/bin/python3 tools/no_wing_release_check.py --quick --skip-frontend --artifact-root /tmp/automixer_expected_artifact_required_fields`
  generated
  `/tmp/automixer_expected_artifact_required_fields/20260617T091202Z_no_wing_release_check/summary.json`
  with `status: passed`, `required_supervised_proof_plan_summary_fields`
  expanded to 31 fields, trace fields included in the required set,
  `missing_required_supervised_proof_plan_summary_fields: []`,
  `next_wing_proof_plan_required_supervised_proof_plan_summary_fields_match_contract:
  true`, `readonly_preflight_review_decision_accepted_candidate_expected_artifact_trace_complete:
  false`, and readiness/live-authority denial flags set to `false`.
- PASS: `PYTHONPATH=backend /opt/homebrew/bin/python3 tools/verify_no_wing_artifact_manifest.py /tmp/automixer_expected_artifact_required_fields/20260617T091202Z_no_wing_release_check`
  returned `failed_checks: []`, `artifact_manifest_self_consistency_match: true`,
  `artifact_manifest_contract_cross_surface_match: true`,
  `artifact_manifest_self_consistency_summary_surface_match: true`,
  `contacts_wing: false`, `mutates_console_state: false`,
  `queries_console: false`, and `live_authority_granted: false`.
- PASS: `PYTHONPATH=backend /opt/homebrew/bin/python3 tools/codex_autoloop/project_director_agent.py --format json --print`
  returned `guard_status: drifting`, `guard_score: 30`,
  selected `WING read-only connection proof`,
  reported `selected_evidence_file_count: 205`,
  `current_first_command_baseline_coverage_marker_count: 288`, and
  `current_first_command_baseline_missing_markers: []`.
- PASS: `PYTHONPATH=backend /opt/homebrew/bin/python3 -m pytest -q tests/test_no_wing_release_check.py tests/test_wing_readonly_preflight_handoff_packet.py tests/test_project_director_agent.py`
  returned `406 passed in 4.52s`.
- PASS: `PYTHONPATH=backend /opt/homebrew/bin/python3 -m pytest tests/ -x --tb=short -q`
  returned `3758 passed, 1 skipped, 1 warning in 49.40s`.
- PASS: `PYTHONPATH=backend /opt/homebrew/bin/python3 tools/memory/memory_lint.py`.
- PASS: `/opt/homebrew/bin/git diff --check`.
