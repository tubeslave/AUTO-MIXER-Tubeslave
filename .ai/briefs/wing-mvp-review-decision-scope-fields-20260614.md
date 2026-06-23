# Brief: WING MVP review decision scope fields

## Task

Require WING read-only preflight operator decision records to preserve the
scope of the reviewed read-only proof.

## Context

The artifact review already exposes queried channels, value types, operator
channel confirmation, operator session readiness, and physical readback counts.
Accepted decision JSON did not store those scope facts, so a durable decision
could be separated from the exact channel/value-type coverage it accepted.

## Constraints

- Do not contact WING or backend services.
- Do not add live authority or readiness closure.
- Keep the change scoped to the first MVP blocker.
- Preserve fail-closed behavior for stale or incomplete decision records.

## Acceptance

- Review decision templates include `channels`, `value_types`,
  `operator_confirmed_channels`, `operator_confirmed_session_ready`,
  `readback_count`, and `physical_readback_count`.
- The shared required-fields contract includes those fields.
- Decision records with stale scope fields are rejected by the ledger.
- Runbook, no-WING surfaces, and focused tests reflect the new scope fields.
