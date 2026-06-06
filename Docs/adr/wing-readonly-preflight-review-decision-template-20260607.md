# ADR: WING Read-Only Preflight Review Decision Template

## Context

After a clean read-only WING preflight artifact is captured, the operator must
record a decision JSON that matches the exact artifact source, run id, summary
path, events path, UTC review timestamp, and checklist ids. Hand-writing those
fields is easy to get wrong.

## Options considered

1. Keep the decision JSON fully manual.
2. Add an offline template writer that copies artifact identity fields from the
   reviewed candidate.
3. Let the review command write an accepted decision automatically.

## Decision

Add a pending review-decision template to
`tools/review_wing_readonly_preflight_artifacts.py`. The template fills the
artifact identity and checklist ids, but leaves `decision`, `reviewed_by`, and
`reviewed_at` invalid until the operator edits them.

Also add `tools/review_wing_readonly_preflight_decisions.py` as the focused
offline follow-up check for the edited decision JSON. It can report an accepted
decision as ready for the next manual WING MVP step, but it does not contact
WING, grant live authority, or close readiness rows.

## Why this won

The operator gets a lower-error path without losing the manual decision gate.
The decision ledger will reject the template until the operator explicitly
records an accepted or rejected review decision.

## Rejected alternatives

- Fully manual JSON remains too error-prone.
- Automatic acceptance is unsafe because it would collapse the artifact review
  and operator decision into one tool action.

## Implementation plan

1. Export review-decision template required fields from the review tool.
2. Add a template builder and optional `--decision-template-output`.
3. Fail closed when multiple clean candidates exist without an explicit source
   id.
4. Add a focused offline decision-review CLI with explicit exit codes for
   accepted, missing-candidate, invalid/missing decision, and rejected decision.
5. Teach no-WING release checks to verify the template contract and expose the
   same template-output path in the operator-visible review command.
6. Expose the focused decision-review command in no-WING summaries.
7. Update the runbook and focused tests.

## Test plan

- `python3 -m py_compile tools/review_wing_readonly_preflight_artifacts.py tools/review_wing_readonly_preflight_decisions.py tools/no_wing_release_check.py`
- `PYTHONPATH=backend python3 -m pytest -q tests/test_wing_readonly_preflight_review_tool.py tests/test_wing_readonly_preflight_review_decision.py tests/test_wing_readonly_preflight_review_decision_tool.py tests/test_no_wing_release_check.py`
- `PYTHONPATH=backend python3 tools/no_wing_release_check.py --quick --skip-frontend --artifact-root /tmp/automixer-no-wing-release-check-review-decision-template --command-timeout-sec 300`
- `python3 tools/memory/memory_lint.py`
- `/opt/homebrew/bin/git diff --check`

## Risks and rollback

Risk is limited to offline artifact review. Rollback is to remove the template
builder, decision-review CLI, no-WING contract checks, tests, and runbook
wording. No live WING transport or write path is affected.
