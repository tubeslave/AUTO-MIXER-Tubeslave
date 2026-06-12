# Brief: WING Read-Only Preflight Decision Review Prestart Fields

## Task

Surface the validated prestart contract in the decision review report for WING
read-only preflight candidates.

## Context

Artifact review now displays prestart fields. The decision review report still
showed accepted/rejected source ids without the candidate prestart boundary,
which made the final manual decision review less self-contained.

## Constraints

- Do not contact WING hardware.
- Do not open a backend WebSocket WING session.
- Do not grant live authority or close readiness rows.
- Keep the change to offline ledger/report metadata and tests.

## Acceptance

- Decision ledger summary carries candidate prestart rows.
- Decision review report and markdown render those rows.
- Tests cover accepted decision markdown and ledger candidate metadata.
