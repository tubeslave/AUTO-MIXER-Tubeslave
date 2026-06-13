# ADR: WING MVP Run Summary Status Contract

## Context

The read-only WING preflight runner produces `summary.json` artifacts that
evidence intake validates before surfacing a candidate for manual review. The
successful run summary status and reason were duplicated as string literals.

## Options Considered

1. Keep the literals in runner and evidence intake.
2. Move the successful status and reason into shared contract constants.
3. Introduce a full artifact schema object for all summary fields.

## Decision

Use shared contract constants for the successful run summary status and reason.

## Why This Won

It removes producer/consumer drift without changing the artifact shape or
granting any new runtime authority.

## Rejected Alternatives

Local literals were rejected because they can drift. A full schema object was
rejected as too broad for this small hardening step.

## Implementation Plan

Add constants in `backend/wing_readonly_preflight_contract.py`, consume them in
the runner and evidence intake, and extend no-WING release checks and tests.

## Test Plan

Run focused WING preflight tests, no-WING quick checks, project direction
guards, memory lint, diff check, and the standard pytest suite.

## Risks and Rollback

Risk is limited to contract/test mismatch. Roll back by reverting this patch;
no runtime live-write path is introduced.
