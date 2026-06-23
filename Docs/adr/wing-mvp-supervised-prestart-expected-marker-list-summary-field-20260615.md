# ADR: WING MVP Supervised Prestart Expected Marker List Summary Field

## Context

The no-WING handoff summary exposes the actual structured supervised prestart
marker list and match verdicts. The expected contract list was visible only
indirectly through nested structured fields and constants.

## Options Considered

1. Keep the expected marker list implicit.
2. Add a flat expected marker list field.
3. Replace the flat marker fields with a structured marker contract object.

## Decision

Add `expected_supervised_prestart_gate_required_markers` as a flat no-WING
summary field.

## Why This Won

It makes the contract directly reviewable beside the actual structured marker
list, without changing live authority, readiness closure, or hardware behavior.

## Rejected Alternatives

- Implicit expected markers require reviewers to infer the contract.
- A broader structured contract refactor is unnecessary for this safety gap.

## Implementation Plan

- Derive the expected marker list from the shared supervised proof marker
  contract.
- Render JSON/Markdown fields.
- Add tests and Project Director baseline marker.

## Test Plan

- `PYTHONPATH=backend python -m py_compile ...`
- Focused pytest for no-WING, Project Director, handoff packet, supervised proof
  plan, and direction guard tests.
- no-WING quick release check.
- Full pytest suite.
- Memory lint and `git diff --check`.

## Risks And Rollback

Rollback is limited to the flat expected marker list and docs. Existing marker
checks remain intact.
