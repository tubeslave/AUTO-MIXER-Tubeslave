# Automixer WING MVP Review Decision Ledger Artifact Root Contract Gate

Date: 2026-06-14

## Summary

The WING read-only review decision ledger now exposes accepted-candidate
artifact-root contract fields and refuses to mark a candidate as manually
available for readiness update when artifact-root metadata is invalid.

## Changed

- Added ledger-level accepted-candidate artifact-root contract fields.
- Required artifact-root contract validity for
  `manual_readiness_update_candidate_available`.
- Added the accepted-candidate artifact-root blocker to ledger `blocked_by`
  when needed.
- Preserved synthetic inline source-id support in the ledger without weakening
  strict proof-plan/no-WING path checks.

## Safety

This is read-only/offline validation. It does not contact WING, mutate console
state, update readiness, or grant live authority.

## Validation

Focused validation:

```bash
PYTHONPATH=backend python -m pytest -q tests/test_wing_readonly_preflight_review_decision.py tests/test_supervised_wing_proof_plan.py tests/test_no_wing_release_check.py
```

Result:

```text
349 passed, 8 skipped
```

WING-focused validation:

```bash
PYTHONPATH=backend python -m pytest -q tests/test_wing_readonly_preflight_evidence.py tests/test_wing_readonly_preflight_review_tool.py tests/test_wing_readonly_preflight_review_decision.py tests/test_supervised_wing_proof_plan.py tests/test_no_wing_release_check.py tests/test_project_director_agent.py
```

Result:

```text
429 passed, 8 skipped
```

Standard validation:

```bash
PYTHONPATH=backend python -m pytest tests/ -x --tb=short -q
```

Result:

```text
3379 passed, 9 skipped
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
PASS

no-WING quick summary:
- status: passed
- failed_checks: []
- live_write: false
- artifact_manifest_live_authority_granted: false
- artifact_manifest_readiness_closure_allowed: false

Project Director:
- guard_status: on_target
- guard_score: 90
- selected_blocker: WING read-only connection proof
- selected_blocker_status: tooling_ready_waits_real_wing
- selected_evidence_file_count: 158
```

## Readiness Impact

No readiness row was closed. The first MVP blocker remains
`WING read-only connection proof` and still requires a real WING read-only
preflight capture plus manual operator review.
