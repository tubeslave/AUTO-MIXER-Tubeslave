# ADR: WING MVP Operator Markdown Required Field Lines Self-Check

Date: 2026-06-15
Status: accepted

## Context

The no-WING operator `summary.md` is part of the first WING MVP blocker gate.
It renders both individual proof-plan fields and a
`operator_markdown_summary_required_field_lines` list. The validator already
checked individual field-line presence, but did not prove that the rendered
required-lines list itself matched the current contract.

## Options considered

1. Keep checking only individual field-line presence.
2. Also parse and compare the rendered required-lines list.

## Decision

Use option 2. The operator Markdown surface now exposes
`operator_markdown_summary_required_field_lines_missing_members` and
`operator_markdown_summary_required_field_lines_match_contract`, and fails
closed when the rendered list is stale or malformed.

## Why this won

The operator-visible contract must not drift from the runtime contract. A stale
required-lines list is a safety-review ambiguity before any WING session opens.

## Rejected alternatives

Relying on generated code paths alone was rejected because this validator is
specifically meant to catch stale or manually edited handoff surfaces.

## Implementation plan

- Extend the operator Markdown required field list to include the self-check
  fields.
- Parse the rendered required-lines list from Markdown.
- Mark the existing operator Markdown surface check failed on missing members.
- Update tests, Project Director markers, and operator docs.

## Test plan

- `PYTHONPATH=backend python -m pytest -q tests/test_no_wing_release_check.py tests/test_project_director_agent.py`
- `PYTHONPATH=backend python tools/no_wing_release_check.py --quick --skip-frontend`
- `PYTHONPATH=backend python -m pytest tests/ -x --tb=short -q`

## Risks and rollback

Risk is limited to stricter no-WING validation. Roll back by removing the list
self-check fields and parser. No WING or console mutation behavior is added.
