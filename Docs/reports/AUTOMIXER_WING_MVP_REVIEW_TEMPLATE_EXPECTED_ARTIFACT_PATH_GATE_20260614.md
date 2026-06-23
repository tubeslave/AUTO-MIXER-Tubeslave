# Automixer WING MVP Review Template Expected Artifact Path Gate

Date: 2026-06-14

## Summary

The WING read-only review decision template now rejects unbound expected
artifact paths. Summary and event artifact paths must match the expected
artifact paths before a manual review decision template can be generated.

## Changed

- Required summary/event paths to match their expected artifact paths.
- Required evidence path-match flags before decision template creation.
- Added a regression test that mutates a clean report with mismatched expected
  artifact path bindings and expects template generation to fail closed.

## Safety

This is read-only/offline review tooling. It does not contact WING, mutate
console state, update readiness, or grant live authority.

## Validation

Focused validation:

```bash
PYTHONPATH=backend python -m pytest -q tests/test_wing_readonly_preflight_evidence.py tests/test_wing_readonly_preflight_review_tool.py tests/test_wing_readonly_preflight_review_decision.py tests/test_no_wing_release_check.py tests/test_project_director_agent.py
```

Result:

```text
402 passed, 8 skipped
```

Standard validation:

```bash
PYTHONPATH=backend python -m pytest tests/ -x --tb=short -q
```

Result:

```text
3368 passed, 9 skipped
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
no_wing_release_check: status=passed, failed_checks=[], live_write=false, live_authority_granted=false, readiness_closure_allowed=false
git diff --check: passed
memory_lint: PASS
project_director: guard_status=on_target, guard_score=90, selected_blocker="WING read-only connection proof", selected_blocker_status=tooling_ready_waits_real_wing, selected_evidence_file_count=153
```
