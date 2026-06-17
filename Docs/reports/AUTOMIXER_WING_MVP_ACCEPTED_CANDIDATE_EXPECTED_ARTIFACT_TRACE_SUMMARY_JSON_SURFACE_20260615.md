# Automixer WING MVP Accepted Candidate Expected Artifact Trace Summary JSON Surface

Date: 2026-06-15

## Summary

The no-WING `summary.json` now exposes the accepted-candidate expected artifact
trace for the next WING proof plan. The new surface includes source-id keyed
maps for `expected_artifacts`, `expected_summary_artifact`,
`expected_events_artifact`, and the summary/events path match flags.

Current no-WING runs still report empty maps and
`readonly_preflight_review_decision_accepted_candidate_expected_artifact_trace_complete:
false` until a real accepted WING read-only preflight candidate exists.

## Safety

This is an offline reporting change. It does not contact WING, query a console,
mutate mixer state, close readiness, approve live writes, or grant live
authority.

## Validation

Refreshed on 2026-06-17 after the expected-artifact trace required-field
package had landed:

- PASS: `PYTHONPATH=backend /opt/homebrew/bin/python3 -m py_compile backend/supervised_wing_proof_plan.py backend/wing_readonly_preflight_contract.py tools/no_wing_release_check.py tests/test_no_wing_release_check.py tests/test_supervised_wing_proof_plan.py`.
- PASS: `PYTHONPATH=backend /opt/homebrew/bin/python3 -m pytest -q tests/test_supervised_wing_proof_plan.py`
  returned `22 passed in 0.05s`.
- PASS: `PYTHONPATH=backend /opt/homebrew/bin/python3 -m pytest -q tests/test_no_wing_release_check.py::test_no_wing_release_check_rejects_next_wing_plan_missing_required_summary_field tests/test_no_wing_release_check.py::test_no_wing_release_check_summary_embeds_next_wing_proof_plan`
  returned `32 passed in 0.21s`.
- PASS: `PYTHONPATH=backend /opt/homebrew/bin/python3 tools/no_wing_release_check.py --quick --skip-frontend --artifact-root /tmp/automixer_expected_artifact_trace_summary_json_surface`
  generated
  `/tmp/automixer_expected_artifact_trace_summary_json_surface/20260617T091758Z_no_wing_release_check/summary.json`
  with `status: passed`,
  `readonly_preflight_review_decision_accepted_candidate_expected_artifact_trace_source_ids:
  []`,
  `readonly_preflight_review_decision_accepted_candidate_expected_artifact_trace_complete:
  false`, `readiness_closure_allowed_by_packet: false`,
  `live_authority_granted: false`, and `full_autonomous_ready: false`.
- PASS: `PYTHONPATH=backend /opt/homebrew/bin/python3 tools/verify_no_wing_artifact_manifest.py /tmp/automixer_expected_artifact_trace_summary_json_surface/20260617T091758Z_no_wing_release_check`
  reported `artifact_manifest_contract_cross_surface_match: true`,
  `artifact_manifest_self_consistency_match: true`, and `failed_checks: []`.
- PASS: `PYTHONPATH=backend /opt/homebrew/bin/python3 tools/codex_autoloop/project_director_agent.py --format json --print`
  selected `WING read-only connection proof`, reported
  `guard_status: drifting`, `guard_score: 30`,
  `selected_evidence_file_count: 205`, and kept current-command baseline
  coverage passing with no missing markers.
- PASS: `PYTHONPATH=backend /opt/homebrew/bin/python3 -m pytest -q tests/test_no_wing_release_check.py tests/test_wing_readonly_preflight_handoff_packet.py tests/test_project_director_agent.py`
  returned `406 passed in 4.45s`.
- PASS: `PYTHONPATH=backend /opt/homebrew/bin/python3 -m pytest tests/ -x --tb=short -q`
  returned `3758 passed, 1 skipped, 1 warning in 49.00s`.
- PASS: `PYTHONPATH=backend /opt/homebrew/bin/python3 tools/memory/memory_lint.py`.
- PASS: `/opt/homebrew/bin/git diff --check`.
