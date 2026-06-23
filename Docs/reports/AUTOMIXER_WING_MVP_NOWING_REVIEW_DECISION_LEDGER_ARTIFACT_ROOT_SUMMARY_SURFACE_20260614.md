# Automixer WING MVP no-WING Review Decision Ledger Artifact Root Summary Surface

Date: 2026-06-14

## Summary

The no-WING release summary now exposes review-decision ledger
accepted-candidate artifact-root contract fields and fails closed when an
accepted ledger candidate has invalid artifact-root metadata without the
artifact-root blocker or while still reporting a manual readiness candidate.

## Changed

- Added accepted-candidate artifact-root contract fields to no-WING
  review-decision snapshot detail.
- Added fail-closed validation for missing artifact-root blocker coverage.
- Rendered artifact-root contract fields in no-WING `summary.json` and
  operator-readable `summary.md`.
- Added no-WING regressions for baseline and invalid accepted-candidate ledger
  behavior.

## Safety

This is read-only/offline validation. It does not contact WING, mutate console
state, update readiness, or grant live authority.

## Validation

Targeted validation:

```bash
PYTHONPATH=backend python -m pytest -q tests/test_no_wing_release_check.py -k 'wing_preflight_review_decision'
```

Result:

```text
3 passed, 303 deselected
```

Focused validation:

```bash
PYTHONPATH=backend python -m pytest -q tests/test_wing_readonly_preflight_review_decision.py tests/test_supervised_wing_proof_plan.py tests/test_no_wing_release_check.py
```

Result:

```text
350 passed, 8 skipped
```

WING-focused validation:

```bash
PYTHONPATH=backend python -m pytest -q tests/test_wing_readonly_preflight_evidence.py tests/test_wing_readonly_preflight_review_tool.py tests/test_wing_readonly_preflight_review_decision.py tests/test_supervised_wing_proof_plan.py tests/test_no_wing_release_check.py tests/test_project_director_agent.py
```

Result:

```text
430 passed, 8 skipped
```

Standard validation:

```bash
PYTHONPATH=backend python -m pytest tests/ -x --tb=short -q
```

Result:

```text
3380 passed, 9 skipped
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
- review_decision_ledger_status: passed
- accepted_candidate_artifact_root_contract_valid: false
- accepted_candidate_artifact_roots: []
- accepted_candidate_artifact_root_contract_violations_by_source: {}

Project Director:
- guard_status: on_target
- guard_score: 90
- selected_blocker: WING read-only connection proof
- selected_blocker_status: tooling_ready_waits_real_wing
- selected_evidence_file_count: 159
```

## Readiness Impact

No readiness row was closed. The first MVP blocker remains
`WING read-only connection proof` and still requires a real WING read-only
preflight capture plus manual operator review.
