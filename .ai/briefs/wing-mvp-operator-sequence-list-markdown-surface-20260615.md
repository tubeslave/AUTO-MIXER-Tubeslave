# Brief: WING MVP Operator Sequence List Markdown Surface

## Task

Fail closed when the operator-readable no-WING `summary.md` does not render the
promoted WING read-only handoff operator sequence list fields and self-check
verdicts.

## Context

The no-WING JSON summary now exposes and self-checks top-level
`operator_sequence` and `expected_operator_sequence` lists. Operators also rely
on `summary.md` before opening a backend/WING session, so the Markdown surface
must prove those same review lines are present.

## Scope

- Add a conditional Markdown-surface check for WING handoff operator sequence
  list fields.
- Run the check only when a WING read-only handoff artifact is present.
- Mark the no-WING summary failed when the required Markdown field lines are
  missing.
- Update tests, blocker docs, runbook docs, generated-progress source, and
  traceability docs.

## Safety Boundary

This is offline and read-only. It does not contact WING, start a backend,
execute handoff commands, grant live authority, or close the WING read-only
connection proof blocker.

## Validation

Run focused py_compile, no-WING handoff tests, no-WING quick, Project Director,
the standard full test command, memory lint, and diff whitespace checks.
