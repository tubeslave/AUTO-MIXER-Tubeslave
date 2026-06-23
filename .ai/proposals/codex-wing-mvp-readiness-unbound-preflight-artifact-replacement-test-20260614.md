# Proposal: WING MVP readiness unbound preflight artifact replacement test

## 1. Short thesis

The readiness snapshot test should expect replacement-required for an unbound
read-only WING preflight artifact.

## 2. What I understood

The evidence builder now rejects artifacts that are not bound to expected summary
and event artifacts. The old test still expected
`readonly_preflight_candidate_available` while also asserting prestart contract
violations.

## 3. Proposed solution

Rename and update the regression so the unbound artifact produces
`readonly_preflight_artifact_replacement_required`, zero valid candidates, one
invalid candidate, and no readiness update candidate.

## 4. Likely files to touch

- `tests/test_autonomy_promotion_readiness.py`
- Council ADR/report files

## 5. Alternatives considered

Changing runtime readiness code to treat the artifact as valid was rejected
because it would weaken the WING fail-closed contract.

## 6. Risks

The change updates a test expectation only. It should not hide a real readiness
issue because the following artifact-bound positive test still covers the valid
candidate path.

## 7. Test plan

Run the targeted readiness test and then continue the standard suite.

## 8. Where the other agent may disagree

Kimi may ask for additional operator-facing diagnostics. That can be a follow-up;
this correction keeps the current fail-closed invariant truthful.
