# ADR: read-surface-fixture-closure-plan-hardening-20260606

## Context

The fixture closure plan converts accepted physical readback review records into
manual fixture-baseline update candidates. Although it does not update fixture
files or grant live authority, it is near a future proof boundary where accepted
evidence could be mistaken for closed physical-readback gaps.

## Options Considered

1. Trust acceptance-ledger output without revalidation.
2. Reject every injected ledger and only build from the default ledger builder.
3. Revalidate the ledger envelope, summary, and accepted records inside the
   closure plan.

## Decision

Use option 3. The closure plan now independently validates ledger identity,
timestamps, status, safety flags, summary counts, surface-id summaries,
summary safety/authority claims, accepted-record structure, accepted decision
flags, candidate linkage, and duplicate/conflicting surface reviews before
surfacing manual closure candidates.

## Why This Won

The plan can receive an injected ledger mapping during tests and future
integrations. A second fail-closed barrier keeps forged or stale acceptance data
from becoming closure candidates while preserving the existing ledger builder as
the normal source of truth.

## Rejected Alternatives

- Trusting the ledger builder was rejected because it does not protect against
  injected or serialized ledger drift at the closure-plan boundary.
- Forcing all callers through the default builder was rejected because tests and
  future release artifacts need explicit ledger injection.
- Directly updating fixtures after validation was rejected because this slice is
  still no-WING and read-only.

## Implementation Plan

1. Add helper validation for ledger envelope identity and false safety fields.
2. Add accepted-record validation for required fields, UTC timestamps,
   `decision=accepted`, no conflicting rejected flag, candidate linkage, and
   false authority fields.
3. Add ledger summary count and surface-id consistency checks.
4. Add ledger summary safety checks so summary-level fixture closure, live
   authority, full autonomous, missing review, or missing physical-gap claims
   fail closed.
5. Suppress all closure candidates when any ledger or accepted-record violation
   is present.
6. Add regression tests for malformed records, ambiguous surfaces, summary
   drift, unsafe envelopes, and decision/candidate-link mismatch.

## Test Plan

- `PYTHONPATH=backend python3 -m pytest tests/test_read_surface_physical_readback_fixture_closure_plan.py -q`
- `PYTHONPATH=backend python3 -m pytest tests/test_no_wing_release_check.py tests/test_processing_registry.py tests/test_project_goal_capability_matrix.py -q`
- `PYTHONPATH=backend python3 -m py_compile backend/read_surface_physical_readback_fixture_closure_plan.py tools/no_wing_release_check.py`
- `python3 tools/memory/memory_lint.py`
- `/opt/homebrew/bin/git diff --check`

## Risks And Rollback

Risk: a future ledger schema change could make the closure plan reject otherwise
valid candidates until the hardening rules are updated. Mitigation: violation
ids are explicit and focused tests name the contract.

Rollback: remove the closure-plan revalidation helpers and keep the previous
valid-accepted-record filtering. This would restore prior behavior without
touching no-WING ordering, registry rows, or live-write surfaces.
