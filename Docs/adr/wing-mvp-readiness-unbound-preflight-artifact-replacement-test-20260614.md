# ADR: WING MVP readiness unbound preflight artifact replacement test

## 1. Context

The standard pytest run exposed an outdated readiness regression. The fixture
artifact lacked expected-artifact binding but the test still expected
`readonly_preflight_candidate_available`.

## 2. Options considered

- Loosen runtime evidence handling and accept the unbound artifact.
- Update the test to expect replacement-required.

## 3. Decision

Update the test to expect
`readonly_preflight_artifact_replacement_required`.

## 4. Why this won

It preserves the current WING MVP fail-closed rule: incomplete or unbound
read-only preflight artifacts must not become readiness update candidates.

## 5. Rejected alternatives

Accepting the unbound artifact was rejected because it would make readiness sound
closable from incomplete proof evidence.

## 6. Implementation plan

- Rename the test.
- Update status, valid/invalid counts, readiness candidate flag, blocker ids, and
  violation expectations.

## 7. Test plan

- Targeted readiness regression.
- Standard pytest command until the next result.

## 8. Risks and rollback

Rollback is limited to test expectations. No live mixer behavior is changed.
