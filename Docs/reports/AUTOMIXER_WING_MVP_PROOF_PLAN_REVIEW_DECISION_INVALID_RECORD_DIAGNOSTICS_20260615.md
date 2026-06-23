# Automixer WING MVP Proof-Plan Review-Decision Invalid-Record Diagnostics

Date: 2026-06-15

## Summary

The supervised WING proof-plan summary now carries normalized invalid
review-decision records. The no-WING gate validates that these proof-plan
records match the review-decision ledger and renders them in the operator
markdown summary.

## Changed

- Added `readonly_preflight_review_decision_invalid_records` to
  `next_wing_proof_plan.summary`.
- Added the field to the required supervised proof-plan summary contract.
- Made no-WING validation compare proof-plan invalid records with ledger
  invalid records.
- Propagated the field into no-WING `summary.json`.
- Rendered the field in no-WING `summary.md`.
- Added regression tests for proof-plan output, no-WING summary output, and
  stale invalid-record mismatch detection.

## Safety

This is an offline reporting and fail-closed consistency change. It does not
contact WING, mutate console state, update readiness rows, approve supervised
proof, or grant live authority.

## Validation

Targeted validation:

```bash
PYTHONPATH=backend python -m pytest -q tests/test_supervised_wing_proof_plan.py::test_plan_blocks_physical_proof_steps_for_conflicting_review_decisions tests/test_no_wing_release_check.py::test_no_wing_release_check_summary_embeds_next_wing_proof_plan tests/test_no_wing_release_check.py::test_no_wing_release_check_validates_next_wing_proof_plan_schema tests/test_no_wing_release_check.py::test_no_wing_release_check_rejects_next_wing_plan_stale_invalid_review_decision_records tests/test_no_wing_release_check.py::test_no_wing_release_check_rejects_next_wing_plan_missing_required_summary_field
```

Result:

```text
15 passed in 0.16s
```

Focused validation:

```bash
PYTHONPATH=backend python -m pytest -q tests/test_wing_readonly_preflight_evidence.py tests/test_wing_readonly_preflight_review_tool.py tests/test_wing_readonly_preflight_review_decision.py tests/test_wing_readonly_preflight_review_decision_tool.py tests/test_supervised_wing_proof_plan.py tests/test_no_wing_release_check.py tests/test_project_director_agent.py tests/test_wing_readonly_preflight_handoff_packet.py tests/test_wing_mvp_rails.py
```

Result:

```text
476 passed, 8 skipped in 3.85s
```

Standard validation:

```bash
PYTHONPATH=backend python -m pytest tests/ -x --tb=short -q
```

Result:

```text
3388 passed, 9 skipped in 50.01s
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
no-WING quick: passed; failed_checks: []; live_write: false; artifact_manifest_live_authority_granted: false; artifact_manifest_readiness_closure_allowed: false; next_wing_proof_plan: passed; review_decision_ledger: passed; proof_plan_invalid_records: []
git diff --check: passed
memory_lint: PASS
Project Director: guard_status=on_target; guard_score=90; selected_blocker=WING read-only connection proof; selected_blocker_status=tooling_ready_waits_real_wing; selected_action_source=Current First Command; selected_evidence_file_count=168; current_first_command_baseline_coverage_passed=true; current_first_command_baseline_missing_markers=[]
```

## Readiness Impact

No readiness row was closed. The first MVP blocker remains
`WING read-only connection proof` and still requires a real WING read-only
preflight capture plus manual operator review.
