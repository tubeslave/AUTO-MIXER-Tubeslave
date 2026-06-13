# ADR: WING MVP Backend Prestart Template Failure Detail Fields

Date: 2026-06-13
Status: accepted

## Context

The no-WING backend-port prestart gate now surfaces generated-template pending
and authority-denial fields when an existing listener blocks preflight. The
shared `backend_port_prestart_check_failure_detail_fields` contract still
listed only base listener-review fields, which let handoff/proof surfaces drift
from the summary detail.

## Options considered

1. Keep template fields only as separate summary prose.
2. Add a second template-specific required field list.
3. Extend the existing shared failure-detail field list.

## Decision

Use option 3. The shared failure-detail contract now includes:

- `listener_confirmation_template_status`
- `listener_confirmation_template_accepted_by_template`
- `listener_confirmation_template_execution_allowed_by_template`
- `listener_confirmation_template_readiness_closure_allowed_by_template`

## Why this won

The backend-port prestart failure is one operator gate. A single shared field
list keeps no-WING summaries, handoff packets, review decisions, proof plans,
Project Director, and runbooks aligned.

## Rejected alternatives

Separate prose is easy to miss during review. A second list would create another
contract that must be kept synchronized.

## Implementation plan

Update `backend/wing_readonly_preflight_contract.py`, update docs and autoloop
instructions, regenerate `MVP_PROGRESS_LATEST.md`, and add focused test
assertions.

## Test plan

Run focused no-WING and WING rail tests, Project Director/direction guard,
memory lint, no-WING quick fail-closed validation, and the standard full pytest
command.

## Risks and rollback

Rollback is to remove the four fields from the shared tuple and matching docs.
That would allow older handoff artifacts but would weaken operator-visible
template evidence before WING read-only preflight.
