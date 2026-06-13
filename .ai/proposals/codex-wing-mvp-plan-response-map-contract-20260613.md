# Codex Proposal: WING Plan Response Map Contract

## Short Thesis

Publish the shared request-to-response contract in the offline WING preflight
plan so the operator can review expected response shapes before any real WING
connection is attempted.

## What I Understood

The current MVP is blocked on real read-only WING proof. The runner already
uses a shared response-type contract internally, but the no-WING plan artifact
does not expose that contract, so stale or incomplete operator plans are harder
to catch before the hardware step.

## Proposed Solution

Add `expected_response_types_by_request_type` to the runner's offline plan and
run summary, validate it in evidence intake, and make no-WING plan validation
reject drift from the shared contract.

## Likely Files to Touch

- `tools/wing_readonly_preflight_runner.py`
- `backend/wing_readonly_preflight_evidence.py`
- `tools/no_wing_release_check.py`
- WING preflight and no-WING tests
- Council brief, ADR, and report

## Alternatives Considered

Keeping the mapping runtime-only is smaller, but it leaves the operator-facing
plan without response-shape context. A full artifact schema migration would be
broader than needed for this guardrail.

## Risks

The main risk is fixture churn across existing offline tests. The change should
stay additive and fail closed when the new field is absent or stale.

## Test Plan

Run compile checks, focused WING preflight/no-WING tests, no-WING quick,
direction rails, memory lint, diff check, and the standard pytest command.

## Where Kimi May Disagree

Kimi may prefer a single schema object for all preflight plan fields. For this
MVP rail, exposing the existing contract with focused validation is lower risk.
