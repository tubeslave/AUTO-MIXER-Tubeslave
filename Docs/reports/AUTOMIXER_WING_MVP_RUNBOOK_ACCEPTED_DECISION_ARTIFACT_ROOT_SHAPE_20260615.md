# Automixer WING MVP Runbook Accepted-Decision Artifact-Root Shape

Date: 2026-06-15

## Summary

The WING read-only runbook minimum accepted-decision JSON now includes
`artifact_root`, matching the shared review-decision template contract.

## Changed

- Added `artifact_root` to the minimum accepted-decision JSON example.
- Added `artifact_root` to the fields that must match captured files.
- Added `artifact_root` to the fields operators must keep aligned with the
  generated decision template.
- Added a rails regression for the runbook wording.

## Safety

This is a documentation/test alignment change. It does not contact WING, mutate
console state, update readiness, or grant live authority.

## Validation

Targeted validation:

```bash
PYTHONPATH=backend python -m pytest -q tests/test_wing_mvp_rails.py::test_runbooks_keep_readonly_before_supervised_execute tests/test_wing_readonly_preflight_review_tool.py::test_review_decision_template_required_fields_match_shared_contract tests/test_wing_readonly_preflight_review_tool.py::test_review_decision_template_links_clean_candidate_but_requires_operator_edit tests/test_wing_readonly_preflight_review_decision.py::test_review_decision_ledger_rejects_missing_or_stale_artifact_root
```

Result:

```text
4 passed in 0.06s
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
3381 passed, 9 skipped in 49.80s
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
no-WING quick: passed; failed_checks: []; live_write: false; artifact_manifest_live_authority_granted: false; artifact_manifest_readiness_closure_allowed: false; next_wing_proof_plan: passed; review_decision_ledger: passed
git diff --check: passed
memory_lint: PASS
Project Director: guard_status=on_target; guard_score=90; selected_blocker=WING read-only connection proof; selected_blocker_status=tooling_ready_waits_real_wing; selected_action_source=Current First Command; selected_evidence_file_count=162
```

## Readiness Impact

No readiness row was closed. The first MVP blocker remains
`WING read-only connection proof` and still requires a real WING read-only
preflight capture plus manual operator review.
