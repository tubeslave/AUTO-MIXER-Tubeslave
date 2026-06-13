# Brief: WING MVP Current Evidence Latest Reports

## Context

The active MVP blocker remains `WING read-only connection proof`. Recent
validated reports now define the shared read-only artifact contract and the
stricter accepted-decision prestart ready gate, but the first blocker evidence
list still stopped at older preflight reports.

## Task

Make the current first blocker evidence cite the latest read-only artifact
contract and decision prestart ready gate reports so Project Director surfaces
them in `selected_evidence_files`.

## Constraints

- Do not contact WING hardware.
- Do not start, stop, or reconfigure backend/WING listeners.
- Do not add live OSC/WING writes.
- Do not close the current blocker from documentation or offline evidence.

## Validation

- Focused WING MVP rails and Project Director tests.
- Project Director still selects `WING read-only connection proof`.
- Direction guard remains `on_target`.
- Standard backend pytest before commit.
