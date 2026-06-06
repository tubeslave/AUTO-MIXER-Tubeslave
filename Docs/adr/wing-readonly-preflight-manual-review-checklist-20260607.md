# ADR: WING Read-Only Preflight Manual Review Checklist

## Context

The supervised WING MVP still waits for a real read-only WING preflight artifact
and manual operator review. The artifact review tool already validates captured
`summary.json` and `events.jsonl` files offline, but the review output also
needs to tell the operator exactly what must be confirmed before any readiness
matrix update.

## Options considered

1. Keep the checklist only in the runbook.
2. Add the checklist to the review tool output and protect it in the no-WING
   release check.
3. Treat a clean artifact as automatically closing the readiness row.

## Decision

Add a machine-readable manual review checklist to
`tools/review_wing_readonly_preflight_artifacts.py`, render it in Markdown, and
make `tools/no_wing_release_check.py` fail closed if the checklist is removed or
incomplete. The release check reads the exported `manual_review_checklist()`
behavior from the review tool, not just checklist ID strings in the source file.

## Why this won

The operator-visible review report is the artifact that will be used after the
real preflight run. Keeping the checklist there makes the next manual decision
auditable without granting live authority.

## Rejected alternatives

- Runbook-only checklist: too easy for generated review output to omit.
- Automatic readiness closure: unsafe because offline validation cannot replace
  operator review of the real WING run.

## Implementation plan

1. Add `manual_review_required_before_readiness_update`,
   `manual_review_decision_required`, and `manual_review_checklist` to the
   review report.
2. Render a `Manual Review Gate` Markdown section.
3. Add no-WING release-check fields and violations for incomplete checklist
   coverage, using the review tool's exported checklist behavior.
4. Update focused tests and the runbook.

## Test plan

- `python3 -m py_compile tools/review_wing_readonly_preflight_artifacts.py tools/no_wing_release_check.py`
- `PYTHONPATH=backend python3 -m pytest -q tests/test_wing_readonly_preflight_review_tool.py tests/test_no_wing_release_check.py`
- `PYTHONPATH=backend python3 tools/no_wing_release_check.py --quick --skip-frontend --artifact-root /tmp/automixer-no-wing-release-check-manual-review-checklist-behavior --command-timeout-sec 300`
- `python3 tools/memory/memory_lint.py`
- `/opt/homebrew/bin/git diff --check`

## Risks and rollback

Risk is limited to offline reporting and release-check validation. Rollback is
to remove the checklist fields, tests, and documentation; this does not affect
live WING transport or any write path.
