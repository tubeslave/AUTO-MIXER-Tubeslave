# ADR: WING MVP Prestart Summary Contact Boundary

## Context

The supervised WING Rack MVP still waits on real `WING read-only connection
proof`. Accepted readonly-preflight candidates are now gated by a no-WING
prestart `summary.json` path and SHA256. The existing binding checks
`status: passed`, `live_write: false`, and `backend_port_prestart_check_status:
passed`, but did not require the summary itself to state that it made no backend
contact, no console query, and no test execution.

## Options Considered

1. Leave the boundary at packet level only.
2. Require no-contact fields only in documentation.
3. Require no-contact fields in the prestart summary payload and all accepted
   candidate gates.

## Decision

Use option 3. The no-WING prestart summary must expose `contacts_backend:
false`, `queries_console: false`, and `executes_tests: false`; accepted
candidate review/proof gates must reject traces that do not preserve those
values as bound summary fields.

## Why This Won

The accepted trace cites a concrete summary path and SHA256. Making that
artifact self-describing reduces ambiguity before any real backend/WING session
without granting live authority.

## Rejected Alternatives

Packet-only documentation was rejected because it does not protect stale or
partial accepted-candidate rows. Documentation-only fields were rejected because
they would not fail closed.

## Implementation Plan

- Emit the three fields from the no-WING prestart summary writer.
- Load, validate, and materialize them through WING readonly preflight plans and
  evidence rows.
- Require them in accepted review-decision and supervised proof-plan source
  gates.
- Update rails, docs, and tests.

## Test Plan

- Compile touched Python modules.
- Run focused WING readonly/prestart/review/proof/no-WING tests.
- Run no-WING quick validation and guard scripts.
- Run the full test suite.

## Risks And Rollback

The main risk is rejecting older artifacts. Rollback is to remove the three
required fields from validators and regenerate docs, but that would weaken the
prestart summary audit boundary and should not be used for live proof.
