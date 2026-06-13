# ADR: WING MVP accepted-candidate prestart trace contract

## 1. Context

The supervised WING Rack MVP requires a real read-only preflight candidate and
manual review before any supervised fader/gain proof. The accepted candidate
must include a valid no-WING prestart trace with a bound summary path and
SHA256.

## 2. Options considered

- Keep separate review-decision and proof-plan implementations.
- Add tests that compare the two implementations.
- Move the prestart trace predicate into the shared WING preflight contract.

## 3. Decision

Use a shared helper in `backend/wing_readonly_preflight_contract.py` for
accepted-candidate prestart trace source ids.

## 4. Why this won

The helper keeps the fail-closed gate identical for the review decision ledger
and supervised proof plan. It also keeps no-WING prestart summary type,
connection policy, and backend check name in the same contract namespace as the
existing preflight constants.

## 5. Rejected alternatives

Separate implementations with cross-tests were rejected because they still
leave two independent predicates to drift.

## 6. Implementation plan

1. Add shared prestart summary constants and helper.
2. Bind runner/evidence constants to the shared contract.
3. Replace local accepted-candidate trace checks in review decision and proof
   plan code.
4. Add focused tests proving the shared helper accepts a bound trace and
   rejects a forged missing SHA.

## 7. Test plan

- `py_compile` touched modules.
- Focused WING read-only pytest set.
- no-WING quick release check.
- Direction guard and project director checks.

## 8. Risks and rollback

Risk: a regression could block valid reviewed candidates. Rollback is to revert
the helper call sites to the previous local predicates. No live-write authority
or readiness closure is introduced by this decision.
