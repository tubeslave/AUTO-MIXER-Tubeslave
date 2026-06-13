# ADR: WING MVP Proof Runner Contract

## Context

The WING read-only preflight runner writes `proof_runner` into offline plans
and real proof summaries. Evidence intake and no-WING validation use that id to
reject unexpected artifact producers. The id was duplicated as a string.

## Options Considered

1. Keep the duplicated string.
2. Move the proof runner id into the shared contract.
3. Replace the summary contract with a full schema object.

## Decision

Move the proof runner id into `backend/wing_readonly_preflight_contract.py`.

## Why This Won

It removes a drift point while preserving the artifact shape and read-only
safety boundary.

## Rejected Alternatives

Duplicated literals can drift. A full schema object is broader than required
for this hardening step.

## Implementation Plan

Add the contract constant, consume it from runner/evidence/no-WING validation,
and add tests for contract matching and mismatch detection.

## Test Plan

Run focused WING preflight/no-WING tests, no-WING quick, direction guards,
memory lint, diff check, and the standard pytest suite.

## Risks and Rollback

Risk is limited to test fixture mismatch. Roll back this patch if the constant
binding is wrong; no live-write capability is introduced.
