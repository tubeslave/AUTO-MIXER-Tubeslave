# ADR: WING MVP Plan Response Map Contract

## Context

The read-only WING preflight runner builds an offline expected plan before an
operator can attempt a real WING proof. The runner and evidence intake already
share accepted response-type sets, but the expected plan did not expose that
mapping for operator review or stale-plan detection.

## Options Considered

1. Keep the response mapping runtime-only.
2. Add the mapping to the offline expected plan and successful run summary.
3. Replace the preflight artifacts with a full schema object.

## Decision

Add `expected_response_types_by_request_type` to the offline expected plan and
successful run summary, and validate it against the shared contract.

## Why This Won

It improves pre-hardware review and no-WING drift detection without changing
request execution, opening network connections, or granting live authority.

## Rejected Alternatives

Runtime-only mapping was rejected because operator plans remain less
inspectable. A full schema object was rejected as too broad for this targeted
readiness hardening step.

## Implementation Plan

Expose the mapping from `tools/wing_readonly_preflight_runner.py`, validate it
in `backend/wing_readonly_preflight_evidence.py`, add no-WING plan validation,
and update focused tests.

## Test Plan

Run compile checks, focused WING preflight/no-WING tests, no-WING quick,
project direction guards, memory lint, diff check, and the standard pytest
suite.

## Risks and Rollback

Risk is limited to stale fixtures or missing plan fields failing closed. Roll
back by reverting this patch; no live-write path is introduced.
