# ADR: WING MVP Review Decision Scope Fields

## 1. Context

WING read-only preflight artifacts are scoped by channel, value type, operator
channel confirmation, operator session readiness, and physical readback counts.
The review report exposes those facts, but accepted operator decision records
did not persist them.

## 2. Options considered

- Keep scope only in the reviewed artifact.
- Rely on checklist ids and artifact hashes.
- Add scope fields to the decision record.

## 3. Decision

Add `channels`, `value_types`, `operator_confirmed_channels`,
`operator_confirmed_session_ready`, `readback_count`, and
`physical_readback_count` to the required WING read-only preflight decision
record fields.

## 4. Why this won

The decision record is the durable operator acceptance surface. Persisting the
scope keeps downstream supervised proof planning tied to exactly the channels
and value types that were physically read back and reviewed.

## 5. Rejected alternatives

Checklist-only confirmation was rejected because it hides the actual scope.
Hash-only binding was rejected because it is not an operator-readable scope
surface and can be audited only by reopening source artifacts.

## 6. Implementation plan

- Extend `WING_READONLY_PREFLIGHT_REVIEW_DECISION_TEMPLATE_REQUIRED_FIELDS`.
- Add scope fields to generated decision templates.
- Reject records whose scope fields differ from the clean candidate artifact.
- Update runbook examples and focused tests.

## 7. Test plan

Run focused review-decision, decision-tool, review-tool, handoff, no-WING, WING
rail, and Project Director tests, then standard pytest and no-WING quick
runtime.

## 8. Risks and rollback

Older decision JSON without scope fields will be rejected and must be replaced.
Rollback is removing the required fields and their validation/tests. This ADR
does not contact WING, mutate console state, grant live authority, or close
readiness.
