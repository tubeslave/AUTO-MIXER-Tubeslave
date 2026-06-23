# ADR: WING MVP Handoff Required-False Detail Contract

## Context

The WING read-only connection proof remains open until a real hardware artifact
is captured and reviewed. Offline tooling now requires explicit backend-port
no-authority detail before the read-only runner opens a WebSocket, but the
operator handoff path also needs to preserve and validate those fields.

## Options considered

1. Validate the required-false detail only in the read-only runner.
2. Bind the required-false detail through the offline plan, handoff packet, and
   no-WING summary.

## Decision

Use option 2.

## Why this won

The handoff packet is the bridge from no-WING validation to the operator's real
WING preflight. If it drops or fails to compare no-authority detail fields, stale
evidence can look stronger than it is.

## Rejected alternatives

- Runner-only validation was rejected because it does not make the operator
  packet self-auditing.
- A separate field list was rejected because the shared read-only contract
  already defines the required backend-port false fields.

## Implementation plan

- Reuse the shared required-false field tuple in no-WING validation.
- Validate the plan artifact fields.
- Emit the fields in the handoff packet.
- Validate packet-vs-plan values and expose match verdicts.
- Render fields in Markdown summaries.
- Add focused regression tests.

## Test plan

- Focused no-WING pytest.
- Broader WING rails and no-WING tests.
- no-WING quick validation.
- Memory lint, whitespace diff check, and full test suite.

## Risks and rollback

This can reject older offline artifacts. Rollback removes the plan/packet
required-false field validation, but that should only happen if operators cannot
regenerate current no-WING evidence before a real WING run.
