# Brief: WING MVP review decision request-id correlation field

## Task

Require WING read-only preflight operator decision records to preserve the
request-id policy and request/response correlation verdict used by artifact
review.

## Context

Read-only preflight artifacts are valid only when every physical readback can
be tied back to the expected `query_mixer_parameter` request by request id. The
review report exposes the canonical request-id policy, but accepted decision
JSON did not persist that policy or the correlation-required verdict.

## Constraints

- Do not contact WING or backend services.
- Do not add live authority or readiness closure.
- Keep the change scoped to the first MVP blocker.
- Preserve fail-closed behavior for older or incomplete decision records.

## Acceptance

- Review decision templates include `request_id_policy` and
  `request_id_correlation_required: true`.
- The shared required-fields contract includes both fields.
- Accepted or rejected decision records without the canonical policy and true
  correlation flag are rejected.
- Runbook, no-WING surfaces, and focused tests reflect the new fields.
