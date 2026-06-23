# ADR: WING MVP Decision Review Artifact Root Report

Date: 2026-06-14
Status: Accepted

## Context

The WING read-only review decision ledger now validates `artifact_root` on
operator-edited decisions. The decision-review report is the next operator
surface before supervised proof planning, so it should display the same root
binding instead of only paths and hashes.

## Options considered

1. Keep `artifact_root` internal to the ledger record.
2. Render `artifact_root` in decision-review candidate rows and assert it in
   no-WING summary coverage.
3. Infer the root from `summary_path` only in downstream review.

## Decision

Use option 2. The decision-review Markdown report and no-WING summary fixture
now expose the candidate `artifact_root`.

## Why this won

The validated root is part of the manual proof identity. Showing it beside the
accepted candidate's paths and prestart trace makes operator review and release
gate diagnostics more direct.

## Rejected alternatives

- Internal-only ledger field: too easy for operator-facing reports to hide a
  meaningful proof identity value.
- Derived root only: weaker than preserving the explicit runtime field.

## Implementation plan

- Render `artifact_root` in the decision-review candidate prestart section.
- Update focused fixtures to write `artifact_root` in both summary and decision
  record.
- Update no-WING summary coverage to assert the candidate root is preserved.
- Keep readiness/live authority flags denied.

## Test plan

- `PYTHONPATH=backend python -m pytest -q
  tests/test_wing_readonly_preflight_review_decision_tool.py`
- Full no-WING release-check module.
- Project Director rails, py_compile, memory lint, and `git diff --check`.

## Risks and rollback

Rollback removes the Markdown line, fixture assertions, and report. The change
is offline reporting only; it does not contact hardware, mutate console state,
grant live authority, or close readiness.
