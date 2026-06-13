# ADR: WING MVP Backend Prestart Template Review Fields

Date: 2026-06-13
Status: Accepted

## Context

The supervised WING Rack MVP is still blocked on real WING read-only connection
proof. The no-WING backend-port prestart gate already emits generated
backend-listener template fields showing that a busy-port template is pending
operator review and grants no authority. The shared failure-detail field list
did not include all of those template review markers.

## Options considered

1. Keep the fields only in separate busy-port summary prose.
2. Add the remaining template review fields to the shared prestart
   failure-detail contract.

## Decision

Use option 2. The shared
`WING_READONLY_PREFLIGHT_BACKEND_PORT_PRESTART_CHECK_FAILURE_DETAIL_FIELDS`
constant now includes safe-to-stop denial, live-authority denial, operator
placeholder fields, and required template edits.

## Why this won

A single exact field list is easier for no-WING summaries, handoff packets,
Project Director, runbooks, and tests to compare. It keeps generated templates
clearly pending until operator placeholders are replaced and the operator marks
the template as reviewed and accepted.

## Rejected alternatives

Leaving the fields as prose keeps useful evidence visible, but it allows future
contract consumers to treat the shared failure-detail list as complete while
missing operator-critical review fields.

## Implementation plan

- Extend the backend prestart failure-detail constant.
- Update tests that assert the shared no-WING contract.
- Update WING MVP rails, runbook, autoloop docs, and latest progress.
- Record the validated report in the current blocker evidence and index.

## Test plan

- Python compile check for touched modules.
- Focused no-WING, WING MVP rail, Project Director, and direction guard tests.
- no-WING quick gate with no hardware contact.
- Memory lint and full pytest before commit.

## Risks and rollback

Risk is limited to stricter documentation and contract checks. Rollback is to
remove the four added fields from the shared list and docs, but that would
reintroduce the split contract this ADR closes.
