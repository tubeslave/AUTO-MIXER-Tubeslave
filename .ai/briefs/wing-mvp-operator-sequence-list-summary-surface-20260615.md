# Brief: WING MVP Operator Sequence List Summary Surface

## Task

Expose the actual and expected WING read-only handoff operator sequence lists in
the flat no-WING summary surface.

## Context

The WING read-only connection proof blocker already requires count, hash, and
contract match fields for the handoff operator sequence. Reviewers can prove the
sequence cardinality and digest, but the flat summary does not show the ordered
list itself without drilling into nested details.

## Scope

- Promote `operator_sequence` and `expected_operator_sequence` into the flat
  no-WING handoff artifact summary.
- Include both fields in the JSON and Markdown no-WING summary surfaces.
- Add focused tests for the new summary fields.
- Update blocker, runbook, generated progress source, and traceability docs.

## Safety Boundary

This is offline and read-only. It must not contact WING, open a backend/WING
session, execute handoff commands, grant live authority, or close the WING
read-only connection proof blocker.

## Validation

Run focused no-WING release and handoff packet tests, Project Director tests,
the standard full test command, memory lint, and diff whitespace checks.
