# ADR: WING MVP Operator Markdown Value-Gate Authority Denial

Date: 2026-06-15
Status: accepted

## Context

The WING MVP remains blocked on real read-only connection proof. no-WING
artifacts must make that boundary obvious to both tools and operators. The
operator Markdown summary already renders value-gate authority-denial fields,
but the required Markdown field-line contract did not include those lines.

## Options considered

1. Keep the rendered denial fields outside the required Markdown field-line
   contract.
2. Require the denial fields inside
   `operator_markdown_summary_required_field_lines`.

## Decision

Use option 2. The operator Markdown field-line contract now requires the
accepted-candidate value-gate summary-surface readiness-closure denial and
live-authority denial fields.

## Why this won

The WING proof path is safety-critical. A rendered-only denial line is weaker
than a field that no-WING validation explicitly requires before supervised
proof planning.

## Rejected alternatives

Keeping the previous contract was rejected because it allowed
`operator_markdown_summary_field_lines_rendered: true` without making the
authority-denial lines part of the contract.

## Implementation plan

- Extend `NEXT_WING_PROOF_PLAN_OPERATOR_MARKDOWN_REQUIRED_FIELD_LINES`.
- Assert the new required lines in no-WING tests.
- Add a Project Director baseline marker for the required field-line list.
- Sync MVP and runbook wording.

## Test plan

- `PYTHONPATH=backend python -m pytest -q tests/test_no_wing_release_check.py tests/test_project_director_agent.py`
- `PYTHONPATH=backend python tools/no_wing_release_check.py --quick --skip-frontend`
- `PYTHONPATH=backend python -m pytest tests/ -x --tb=short -q`

## Risks and rollback

Risk is limited to stricter no-WING validation. Roll back by removing the two
required field-line entries and the matching documentation/tests. No WING,
backend, or console mutation behavior is introduced.
