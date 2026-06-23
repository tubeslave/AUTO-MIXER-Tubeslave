# ADR: WING MVP Supervised Prestart Marker Count Summary Fields

## Context

The supervised proof path depends on exactly five required final-gate markers:
`accepted_decision_ready`, accepted-candidate prestart trace,
artifact-root trace, expected-artifact source binding, and the supervised
fader/gain proof. The marker list and list-match verdicts were visible, but the
count was not exposed as a top-level no-WING summary field.

## Options Considered

1. Keep only the marker list and list-match verdicts.
2. Add top-level derived marker count and count-match verdict fields.
3. Replace list-match checks with count-only checks.

## Decision

Add top-level derived marker count fields while preserving the existing
list-match checks.

## Why This Won

It gives operators and Project Director a quick cardinality check without
weakening the stricter list equality checks. It does not contact WING hardware,
grant live authority, or close readiness.

## Rejected Alternatives

- List-only validation leaves cardinality implicit.
- Count-only validation would be too weak for a live-sound safety gate.

## Implementation Plan

- Derive the structured gate marker count.
- Derive the expected contract count.
- Compare the structured count to the contract count and top-level summary
  marker count.
- Render JSON/Markdown fields.
- Add tests and Project Director baseline markers.

## Test Plan

- `PYTHONPATH=backend python -m py_compile ...`
- Focused pytest for no-WING, Project Director, handoff packet, supervised proof
  plan, and direction guard tests.
- no-WING quick release check.
- Full pytest suite.
- Memory lint and `git diff --check`.

## Risks And Rollback

Rollback is limited to the derived count fields and docs. Existing marker-list
checks remain intact.
