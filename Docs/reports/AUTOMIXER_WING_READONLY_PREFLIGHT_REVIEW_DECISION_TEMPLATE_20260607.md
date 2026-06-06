# Automixer WING Read-Only Preflight Review Decision Template - 2026-06-07

## Summary

This slice adds an offline helper for the operator decision step after a clean
WING read-only preflight candidate is reviewed.

`tools/review_wing_readonly_preflight_artifacts.py` can now write a pending
decision JSON template with the correct `artifact_source_id`, `run_id`,
`summary_path`, `events_path`, and checklist ids.

`tools/review_wing_readonly_preflight_decisions.py` then validates the edited
decision JSON against the same captured preflight evidence. It returns success
only when an accepted decision record is present and still grants no live
authority.

`tools/no_wing_release_check.py` now exposes the same
`--decision-template-output` path in its operator-visible review command and
fails closed if that command stops writing a pending decision template.

The template is intentionally not accepted by the ledger until the operator
edits:

- `decision`
- `reviewed_by`
- `reviewed_at`

## Safety Boundary

- No live WING/OSC/WebSocket action was run for this change.
- No write, proposal, rollback, emergency-stop, connect, or disconnect path was
  added.
- The template does not grant live authority.
- The template does not close readiness rows.
- The decision ledger rejects the template until the operator records a real
  accepted or rejected decision.
- A rejected decision is valid review evidence but still exits nonzero because
  supervised proof must remain blocked.

## Files Changed

- `tools/review_wing_readonly_preflight_artifacts.py`
- `tools/review_wing_readonly_preflight_decisions.py`
- `tools/no_wing_release_check.py`
- `tests/test_wing_readonly_preflight_review_tool.py`
- `tests/test_wing_readonly_preflight_review_decision_tool.py`
- `tests/test_no_wing_release_check.py`
- `Docs/runbooks/WING_READONLY_PREFLIGHT.md`
- `Docs/reports/MVP_PROGRESS_LATEST.md`
- `Docs/adr/wing-readonly-preflight-review-decision-template-20260607.md`
- `Docs/reports/AUTOMIXER_WING_READONLY_PREFLIGHT_REVIEW_DECISION_TEMPLATE_20260607.md`
- `Docs/INDEX.md`
- `.ai/project-layer/wing_mvp_blockers.md`

## Validation

- `python3 -m py_compile tools/review_wing_readonly_preflight_artifacts.py tools/review_wing_readonly_preflight_decisions.py tools/no_wing_release_check.py`
  passed.
- `PYTHONPATH=backend python3 -m pytest -q tests/test_wing_readonly_preflight_review_tool.py tests/test_wing_readonly_preflight_review_decision.py tests/test_wing_readonly_preflight_review_decision_tool.py tests/test_no_wing_release_check.py`
  passed: `162 passed`.
- `PYTHONPATH=backend python3 tools/no_wing_release_check.py --quick --skip-frontend --artifact-root /tmp/automixer-no-wing-release-check-review-decision-template --command-timeout-sec 300`
  passed.
- Quick no-WING summary exposed `review_command_writes_decision_template:
  true`, `review_decision_template_contract_complete: true`,
  `tools/review_wing_readonly_preflight_decisions.py`, and the decision-review
  exit codes `0/2/3/4`.
- `python3 tools/memory/memory_lint.py` passed.
- `/opt/homebrew/bin/git diff --check` passed.

## Remaining WING MVP Blocker

The first blocker remains open. The operator still must run the WING read-only
preflight against the intended backend/WING session and then turn the pending
template into a real accepted or rejected decision JSON.
