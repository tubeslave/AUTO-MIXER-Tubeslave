# Brief: WING MVP Read-Only Artifact Contract Constants

## Context

The active MVP blocker remains `WING read-only connection proof`. Several
offline tools refer to the same real WING proof artifact identity:
`wing_readonly_preflight.v1`, `summary.json`, `events.jsonl`, and the next
manual artifact requirement.

## Task

Move those artifact identity strings into the shared WING read-only preflight
contract and use them in the runner, handoff packet, no-WING release check,
evidence intake, supervised proof plan, and focused tests.

## Constraints

- Do not contact WING hardware.
- Do not start, stop, or reconfigure backend/WING listeners.
- Do not add live OSC/WING writes.
- Preserve existing artifact names and schema values.

## Validation

- Focused WING read-only preflight, handoff packet, no-WING release, and
  supervised proof plan tests.
- Project Director selected action still targets `WING read-only connection
  proof`.
- Direction guard remains `on_target`.
- Standard backend pytest before commit.
