# ADR: WING MVP proof-plan required summary field parametric gate

## 1. Context

The no-WING proof-plan schema must fail closed when any canonical supervised
prestart proof-plan summary field is absent. The existing regression covered a
single field, while the Project Director directive names the broader invariant.

## 2. Options considered

- Keep one representative missing-field test.
- Parameterize the test over the canonical required field list.

## 3. Decision

Parameterize the missing-required-field regression over
`WING_READONLY_PREFLIGHT_SUPERVISED_PRESTART_REQUIRED_PROOF_PLAN_SUMMARY_FIELDS`.

## 4. Why this won

It keeps the test evidence aligned with the operator-facing no-WING contract and
prevents future required fields from slipping out of validation coverage.

## 5. Rejected alternatives

Representative-only coverage was rejected because it is weaker than the stated
fail-closed contract.

## 6. Implementation plan

- Convert the existing regression to `pytest.mark.parametrize`.
- Keep the validator implementation unchanged.

## 7. Test plan

- Parametrized missing-field regression.
- Focused no-WING/handoff suite.
- Project Director rails.

## 8. Risks and rollback

Rollback is a test-only revert. No live mixer authority or runtime behavior is
changed.
