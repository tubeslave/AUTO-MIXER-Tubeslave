# Codex Proposal: WING MVP Proof Runner Contract

## Short Thesis

Move the read-only preflight proof runner id into the shared WING preflight
contract and make producer, consumer, and no-WING checks depend on it.

## What I Understood

The current MVP remains blocked on real WING read-only proof and manual review.
Offline work should harden artifact contracts without implying readiness or
granting live authority.

## Proposed Solution

Add `WING_READONLY_PREFLIGHT_PROOF_RUNNER` to the shared contract, import it in
the runner and evidence intake, and extend no-WING snapshot detail/tests to
compare runner/evidence proof-runner identity against the contract.

## Likely Files to Touch

- `backend/wing_readonly_preflight_contract.py`
- `backend/wing_readonly_preflight_evidence.py`
- `tools/wing_readonly_preflight_runner.py`
- `tools/no_wing_release_check.py`
- WING preflight and no-WING tests

## Alternatives Considered

Leaving the id as a literal is smaller, but it keeps a producer/consumer drift
point. A full schema object is broader than needed for this step.

## Risks

Risk is limited to fixture updates and no-WING snapshot expectations. The
artifact shape stays unchanged.

## Test Plan

Run focused WING preflight/no-WING tests, no-WING quick, direction rails,
memory lint, diff check, and the standard pytest suite.

## Where Kimi May Disagree

Kimi may prefer consolidating every artifact field into a schema class now.
For this MVP rail, one contract constant plus drift detection is lower risk.
