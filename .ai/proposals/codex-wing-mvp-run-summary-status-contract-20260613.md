# Codex Proposal: WING Run Summary Status Contract

## Short Thesis

Move the successful read-only preflight summary status and reason into the
shared WING preflight contract and make producer, consumer, and no-WING checks
depend on that source of truth.

## What I Understood

The current MVP needs stronger guarantees around real WING read-only proof
artifacts. Offline evidence must not close readiness, but it can prevent drift
between the runner and evidence intake before an operator captures real
artifacts.

## Proposed Solution

Add shared constants for the accepted run summary `status` and `reason`, import
them in the runner and evidence intake, and extend no-WING release detail to
compare runner/evidence values with the contract.

## Likely Files to Touch

- `backend/wing_readonly_preflight_contract.py`
- `backend/wing_readonly_preflight_evidence.py`
- `tools/wing_readonly_preflight_runner.py`
- `tools/no_wing_release_check.py`
- WING preflight and no-WING tests

## Alternatives Considered

Leaving the literals local is smaller, but it allows the producer and consumer
to drift silently. A schema library would be heavier than this narrow guard.

## Risks

The main risk is breaking existing tests that intentionally use literal fixture
payloads. The patch should update only the contract-sensitive expectations.

## Test Plan

Run focused WING preflight tests, no-WING quick checks, direction guards,
memory lint, diff check, and the standard pytest command.

## Where Kimi May Disagree

Kimi may prefer a typed artifact dataclass or schema object. For this narrow
MVP rail, constants plus release-check drift detection are lower risk.
