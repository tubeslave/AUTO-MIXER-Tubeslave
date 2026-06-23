# Brief: WING MVP Operator Sequence List Summary Surface Self-Check

## Task

Fail closed when the promoted top-level WING read-only handoff operator sequence
lists drift from the nested handoff artifact or shared checklist contract.

## Context

The no-WING summary now exposes `operator_sequence` and
`expected_operator_sequence` as flat review fields. That improves operator
inspection, but the promoted surface should also prove that those lists still
match the handoff artifact and expected contract.

## Scope

- Add a no-WING summary self-check for the promoted operator sequence lists.
- Mark the no-WING summary failed if the promoted lists are missing, not lists,
  differ from the handoff artifact, differ from the contract, or no longer match
  their count fields.
- Render the self-check in `summary.md`.
- Update focused tests, blocker docs, runbook docs, generated-progress source,
  and traceability records.

## Safety Boundary

This is offline and read-only. It does not contact WING, open a backend/WING
session, execute handoff commands, grant live authority, or close the WING
read-only connection proof blocker.

## Validation

Run focused py_compile, no-WING handoff tests, no-WING quick, Project Director,
the standard full test command, memory lint, and diff whitespace checks.
