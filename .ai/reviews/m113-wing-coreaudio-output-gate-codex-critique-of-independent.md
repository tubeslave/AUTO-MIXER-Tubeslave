# Codex Critique of the Independent M113 Proposal

## 1. Strongest part

The proposal catches the second-order boundary: fixing collection alone is not
enough if the attended runner's independent artifact verifier still accepts
the old input-only binding shape.

## 2. Possible mistakes or blind spots

It suggests broad canonical state changes before the implementation result is
known. The active M110 blocker, score and operator action do not change; the
minimal truthful state update should record the landed hardening and validation
without replacing M110 as active hardware work.

## 3. Where it is too complex, fragile, or costly

A new schema v2 or a new collector is unnecessary because no accepted M110-HW01
physical artifact exists. A broad new integration harness for output drift
would duplicate existing freshness machinery; one explicit output-mode refresh
failure test is enough beside the verifier and digest-binding tests.

## 4. What should be kept

Keep strict integer validation, blocked diagnostics, absence-of-authority
assertions, the complete runner fixture/semantic-verifier update, explicit v1
compatibility rationale and CP05 two-commit discipline.

## 5. Revised recommendation

Implement the narrow verifier and runner changes, add direct self-consistent
forged-artifact rejection for output mismatch, and add one explicit post-issue
output-mode refresh failure test in addition to the existing digest-drift
coverage. Record strict readiness as unchanged at 88.
