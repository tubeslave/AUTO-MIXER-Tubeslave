# ADR: WING MVP no-WING prestart summary contract

## 1. Context

The WING read-only preflight flow requires a passed no-WING prestart
`summary.json` before the intended backend/WING session. The no-WING summary is
not live readiness, but it is a required artifact for the real proof path.

## 2. Options considered

- Keep no-WING summary type/policy/status literals local to
  `tools/no_wing_release_check.py`.
- Cross-test no-WING literals against runner/evidence constants.
- Use shared preflight contract constants in no-WING generation and validation.

## 3. Decision

Use shared `backend/wing_readonly_preflight_contract.py` constants for no-WING
prestart summary type, expected passed status, connection policy, backend check
name, and backend check status.

## 4. Why this won

The no-WING gate produces the artifact later consumed by the WING read-only
preflight runner, evidence intake, review ledger, and proof plan. A single
contract source prevents silent divergence.

## 5. Rejected alternatives

Independent literals plus tests were rejected because they still leave the
release gate as a separate contract owner.

## 6. Implementation plan

1. Add the shared expected prestart summary status.
2. Bind runner/evidence/no-WING status checks to shared constants.
3. Bind no-WING generated summaries and plan validation to shared constants.
4. Add regression tests for summary writing and drift rejection.

## 7. Test plan

- Focused WING/no-WING pytest set.
- no-WING quick release check.
- Direction guard and Project Director.
- Full pytest.

## 8. Risks and rollback

Risk: valid no-WING plans could be blocked if the shared constants are wrong.
Rollback is to revert this commit. No WING contact, live write, readiness
closure, or supervised execution is added.
