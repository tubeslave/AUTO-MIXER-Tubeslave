# Automixer WING MVP Accepted Candidate Expected Artifact Trace Value Gate

Date: 2026-06-15

## Summary

The no-WING next proof-plan validator now compares accepted-candidate expected
artifact trace values in `next_wing_proof_plan.summary` against the canonical
trace computed from `wing_readonly_preflight_review_decision_ledger.summary.candidate_reviews`.

This catches proof plans that include every required trace field but carry stale
or incorrect accepted source ids, expected artifact lists, expected
summary/events artifact paths, summary/events match flags, missing source ids,
or trace completion verdict.

This does not close readiness. The current no-WING path remains offline and
does not grant live authority.

## Safety

This is an offline validation change. It does not contact WING, query a console,
mutate mixer state, close readiness, approve live writes, or grant live
authority.

## Validation

- PASS: `PYTHONPATH=backend /opt/homebrew/bin/python3 -m py_compile tools/no_wing_release_check.py tests/test_no_wing_release_check.py`.
- PASS: `PYTHONPATH=backend /opt/homebrew/bin/python3 -m pytest -q tests/test_no_wing_release_check.py::test_no_wing_release_check_rejects_next_wing_plan_expected_artifact_trace_mismatch tests/test_no_wing_release_check.py::test_no_wing_release_check_summary_embeds_next_wing_proof_plan`
  returned `9 passed in 0.15s`.
- PASS: `PYTHONPATH=backend /opt/homebrew/bin/python3 -m pytest -q tests/test_no_wing_release_check.py::test_no_wing_release_check_allows_next_wing_plan_blocked_by_preflight tests/test_no_wing_release_check.py::test_no_wing_release_check_allows_next_wing_plan_waiting_on_review_decision tests/test_no_wing_release_check.py::test_no_wing_release_check_rejects_next_wing_plan_expected_artifact_trace_mismatch`
  returned `10 passed in 0.14s`.
- PASS: `PYTHONPATH=backend /opt/homebrew/bin/python3 tools/no_wing_release_check.py --quick --skip-frontend --artifact-root /tmp/automixer_expected_artifact_value_gate`
  generated
  `/tmp/automixer_expected_artifact_value_gate/20260617T090735Z_no_wing_release_check/summary.json`
  with `status: passed`,
  `next_wing_proof_plan_required_supervised_proof_plan_summary_fields_match_contract:
  true`, empty accepted-candidate expected artifact trace source ids,
  `readonly_preflight_review_decision_accepted_candidate_expected_artifact_trace_complete:
  false`, `next_wing_proof_plan_accepted_candidate_expected_artifact_trace_value_gate_required:
  true`, `next_wing_proof_plan_accepted_candidate_expected_artifact_trace_value_gate_violations:
  []`, `next_wing_proof_plan_accepted_candidate_expected_artifact_trace_value_gate_passed:
  true`, `live_authority_granted: false`, and `full_autonomous_ready: false`.
- PASS: `PYTHONPATH=backend /opt/homebrew/bin/python3 tools/verify_no_wing_artifact_manifest.py /tmp/automixer_expected_artifact_value_gate/20260617T090735Z_no_wing_release_check`
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
  returned `406 passed in 4.50s`.
- PASS: `PYTHONPATH=backend /opt/homebrew/bin/python3 -m pytest tests/ -x --tb=short -q`
  returned `3758 passed, 1 skipped, 1 warning in 49.46s`.
- PASS: `PYTHONPATH=backend /opt/homebrew/bin/python3 tools/memory/memory_lint.py`.
- PASS: `/opt/homebrew/bin/git diff --check`.
