# Automixer WING Read-Only Preflight Manual Review Checklist - 2026-06-07

## Summary

This slice makes the offline WING preflight artifact review report explicit
about the manual operator decision that must happen before readiness updates.

`tools/review_wing_readonly_preflight_artifacts.py` now emits:

- `manual_review_required_before_readiness_update`
- `manual_review_decision_required`
- `manual_review_checklist`

The Markdown report also renders a `Manual Review Gate` section. The checklist
asks the operator to confirm the captured artifact paths, the channel scope, the
physical readbacks, the read-only event-log sequence, and the fact that the
review grants no live authority.

The no-WING release gate now imports the review tool and reads the exported
`manual_review_checklist()` IDs. This avoids a false pass where checklist ID
strings remain in source comments but the actual review output stops carrying
the full manual gate.

## Safety Boundary

- No live WING/OSC/WebSocket action was run for this change.
- No write, proposal, operator-mode mutation, rollback, emergency-stop,
  connect, or disconnect request was added.
- The change affects offline review and no-WING release-check reporting only.
- A clean candidate still does not close readiness rows without manual review.

## Files Changed

- `tools/review_wing_readonly_preflight_artifacts.py`
- `tools/no_wing_release_check.py`
- `tests/test_wing_readonly_preflight_review_tool.py`
- `tests/test_no_wing_release_check.py`
- `Docs/runbooks/WING_READONLY_PREFLIGHT.md`
- `Docs/adr/wing-readonly-preflight-manual-review-checklist-20260607.md`
- `Docs/reports/AUTOMIXER_WING_READONLY_PREFLIGHT_MANUAL_REVIEW_CHECKLIST_20260607.md`
- `Docs/INDEX.md`

## Validation

- `python3 -m py_compile tools/review_wing_readonly_preflight_artifacts.py tools/no_wing_release_check.py` - PASS
- `PYTHONPATH=backend python3 -m pytest -q tests/test_wing_readonly_preflight_review_tool.py tests/test_no_wing_release_check.py` - `139 passed`
- `PYTHONPATH=backend python3 tools/no_wing_release_check.py --quick --skip-frontend --artifact-root /tmp/automixer-no-wing-release-check-manual-review-checklist-behavior --command-timeout-sec 300` - PASS; summary includes `review_manual_checklist_complete: true`

## Remaining WING MVP Blocker

The first blocker remains open. The operator still must run the read-only
preflight against the intended backend/WING session and manually review the
captured `summary.json` plus `events.jsonl`.
