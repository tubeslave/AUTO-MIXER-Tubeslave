# WING MVP Plan Response Map Contract

## Task

Expose the read-only WING preflight request-to-response mapping in the offline
expected plan and bind runner, evidence intake, and no-WING validation to the
shared contract.

## Context

The supervised WING MVP still waits on real read-only WING evidence and manual
review. Recent hardening moved the accepted response types into a shared
contract, but the operator plan artifact still does not show that mapping.

## Safety Boundary

This is a no-WING contract hardening step. It does not contact hardware, grant
live authority, close readiness, or allow mixer writes.

## Expected Outcome

- Offline WING preflight plans include `expected_response_types_by_request_type`.
- Successful run summaries include the same mapping for expected-plan matching.
- Evidence intake and no-WING release checks fail closed on stale or missing
  response-type mappings.
- Focused tests and the standard suite pass before commit.
