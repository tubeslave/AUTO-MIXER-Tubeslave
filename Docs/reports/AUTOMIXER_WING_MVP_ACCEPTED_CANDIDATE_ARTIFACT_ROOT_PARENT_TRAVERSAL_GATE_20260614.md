# Automixer WING MVP Accepted Candidate Artifact Root Parent Traversal Gate

Date: 2026-06-14

## Summary

Accepted WING read-only candidates now fail closed when reviewed summary,
event, or expected artifact paths contain parent traversal that would leave the
declared artifact root after normalization.

## Changed

- Strengthened the shared accepted-candidate artifact-root contract helper with
  POSIX path normalization.
- Added `*_contains_parent_traversal` violations for artifact path fields that
  include `..`.
- Kept `*_not_under_artifact_root` violations when normalized paths escape the
  declared root.
- Added regressions for direct helper use, supervised proof planning, and
  no-WING next-proof-plan validation.

## Safety

This is read-only/offline validation. It does not contact WING, mutate console
state, update readiness, or grant live authority.

## Validation

Refreshed on 2026-06-17:

Focused validation:

```bash
PYTHONPATH=backend /opt/homebrew/bin/python3 -m pytest -q tests/test_wing_readonly_preflight_review_decision.py tests/test_supervised_wing_proof_plan.py tests/test_no_wing_release_check.py
```

Result:

```text
439 passed in 4.21s
```

WING-focused validation:

```bash
PYTHONPATH=backend /opt/homebrew/bin/python3 -m pytest -q tests/test_wing_readonly_preflight_evidence.py tests/test_wing_readonly_preflight_review_tool.py tests/test_wing_readonly_preflight_review_decision.py tests/test_supervised_wing_proof_plan.py tests/test_no_wing_release_check.py tests/test_project_director_agent.py
```

Result:

```text
520 passed in 4.43s
```

Standard validation:

```bash
PYTHONPATH=backend /opt/homebrew/bin/python3 -m pytest tests/ -x --tb=short -q
```

Result:

```text
3758 passed, 1 skipped, 1 warning in 47.44s
```

Release/service checks:

```bash
PYTHONPATH=backend /opt/homebrew/bin/python3 tools/no_wing_release_check.py --quick --skip-frontend --artifact-root /tmp/automixer_artifact_root_parent_traversal_gate
PYTHONPATH=backend /opt/homebrew/bin/python3 tools/verify_no_wing_artifact_manifest.py /tmp/automixer_artifact_root_parent_traversal_gate/20260617T093418Z_no_wing_release_check
/opt/homebrew/bin/git diff --check -- .ai/briefs/wing-mvp-accepted-candidate-artifact-root-parent-traversal-gate-20260614.md .ai/proposals/codex-wing-mvp-accepted-candidate-artifact-root-parent-traversal-gate-20260614.md Docs/adr/wing-mvp-accepted-candidate-artifact-root-parent-traversal-gate-20260614.md Docs/reports/AUTOMIXER_WING_MVP_ACCEPTED_CANDIDATE_ARTIFACT_ROOT_PARENT_TRAVERSAL_GATE_20260614.md
PYTHONPATH=backend /opt/homebrew/bin/python3 tools/memory/memory_lint.py
PYTHONPATH=backend /opt/homebrew/bin/python3 tools/codex_autoloop/project_director_agent.py --format json --print
```

Result:

```text
no_wing_release_check: status=passed, failed_checks=[], readiness_closure_allowed_by_packet=false, live_authority_granted=false, full_autonomous_ready=false
manifest_verifier: status=passed, failed_checks=[], artifact_manifest_contract_cross_surface_match=true, artifact_manifest_self_consistency_match=true, artifact_manifest_self_consistency_summary_surface_match=true
git diff --check: passed
memory_lint: PASS
project_director: guard_status=drifting, guard_score=30, selected_blocker="WING read-only connection proof", selected_evidence_file_count=205, current_first_command_baseline_coverage_marker_count=288, current_first_command_baseline_missing_markers=[]
```

## Readiness Impact

No readiness row was closed. The first MVP blocker remains
`WING read-only connection proof` and still requires a real WING read-only
preflight capture plus manual operator review.
