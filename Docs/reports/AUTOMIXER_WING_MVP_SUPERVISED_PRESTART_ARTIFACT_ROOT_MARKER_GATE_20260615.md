# Automixer WING MVP Supervised Prestart Artifact-Root Marker Gate

Date: 2026-06-15

## Summary

The final WING supervised prestart gate now explicitly requires a valid
accepted-candidate artifact-root trace before supervised fader/gain proof
planning can proceed.

## Changed

- Added `valid accepted-candidate artifact-root trace` to the supervised
  prestart gate operator step and required markers.
- Added `requires_accepted_candidate_artifact_root_trace: true` to the
  structured supervised prestart gate.
- Updated the current first WING blocker command, Project Director baseline,
  runbook, MVP contract, and autoloop docs to expose the same marker.
- Added/updated exact-string regressions in handoff, no-WING, rails, and Project
  Director tests.

## Safety

This is an offline/read-only contract surface change. It does not contact WING,
mutate console state, close readiness, or grant live authority.

## Validation

Targeted validation:

```bash
PYTHONPATH=backend python -m pytest -q tests/test_wing_readonly_preflight_handoff_packet.py tests/test_no_wing_release_check.py tests/test_wing_mvp_rails.py tests/test_project_director_agent.py
```

Result:

```text
330 passed, 8 skipped in 3.64s
```

Focused validation:

```bash
PYTHONPATH=backend python -m pytest -q tests/test_wing_readonly_preflight_evidence.py tests/test_wing_readonly_preflight_review_tool.py tests/test_wing_readonly_preflight_review_decision.py tests/test_wing_readonly_preflight_review_decision_tool.py tests/test_supervised_wing_proof_plan.py tests/test_no_wing_release_check.py tests/test_project_director_agent.py tests/test_wing_readonly_preflight_handoff_packet.py tests/test_wing_mvp_rails.py
```

Result:

```text
469 passed, 8 skipped in 3.83s
```

Standard validation:

```bash
PYTHONPATH=backend python -m pytest tests/ -x --tb=short -q
```

Result:

```text
3381 passed, 9 skipped in 49.98s
```

Release/service checks:

```bash
PYTHONPATH=backend python tools/no_wing_release_check.py --quick --skip-frontend
/opt/homebrew/bin/git diff --check
PYTHONPATH=backend python tools/memory/memory_lint.py
PYTHONPATH=backend python tools/codex_autoloop/project_director_agent.py --format json --print
```

Result:

```text
no_wing_release_check: passed; failed_checks: []
live_write: false
artifact_manifest_live_authority_granted: false
artifact_manifest_readiness_closure_allowed: false
next_wing_proof_plan_status: passed
review_decision_ledger_status: passed
supervised_prestart_requires_artifact_root: true
git diff --check: passed
memory_lint.py: PASS
project_director_agent: on_target; guard_score: 90; selected_blocker: WING read-only connection proof; selected_blocker_status: tooling_ready_waits_real_wing; selected_action_source: Current First Command; selected_evidence_file_count: 161
```

## Readiness Impact

No readiness row was closed. The first MVP blocker remains
`WING read-only connection proof` and still requires a real WING read-only
preflight capture plus manual operator review.
