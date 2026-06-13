# ADR: WING MVP evidence value-types contract

## 1. Context

The supervised WING MVP is blocked on a real read-only connection proof. The
offline evidence intake decides whether captured `wing_readonly_preflight.v1`
artifacts can become manual review candidates. That decision must stay in lock
step with the runner/handoff value-type contract.

Evidence intake previously used `CURRENT_PROVEN_LIVE_WRITE_KINDS`. That product
policy is related, but it can expand independently of the current read-only
preflight scope.

## 2. Options considered

- Keep evidence intake tied to broader product live-write policy.
- Bind evidence intake to the shared WING read-only preflight value-type
  contract.
- Add a second local evidence-specific tuple.

## 3. Decision

Bind evidence intake to
`WING_READONLY_PREFLIGHT_RUNNER_SUPPORTED_VALUE_TYPES` and make no-WING
validation fail closed if evidence allowed value types drift.

## 4. Why this won

The current hardware runbook, handoff packet, runner, and expected artifacts all
describe the same read-only target. Evidence intake should not accept a wider
artifact unless the shared contract is intentionally changed.

## 5. Rejected alternatives

The broader live-write policy is not the source of truth for this read-only
proof. A second local tuple would be another drift point.

## 6. Implementation plan

1. Derive `ALLOWED_VALUE_TYPES` from the shared WING preflight contract.
2. Add no-WING detail fields for evidence allowed value types.
3. Fail no-WING validation on mismatch.
4. Add regression tests for both match and mismatch.

## 7. Test plan

- Focused evidence/no-WING tests.
- no-WING quick on an unused backend port.
- WING rails tests and full pytest.

## 8. Risks and rollback

Risk is limited to rejecting artifacts that use value types outside the current
runner/handoff contract. Rollback would restore the live-write policy source,
but that reopens the drift risk.
