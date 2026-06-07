# Proposal: wing-readonly-preflight-transport-timeout-artifact-20260607

## Short thesis

Make read-only WING preflight transport timeouts first-class fail-closed
artifacts, then make the offline plan and release gate name that boundary.

## What I understood

The current MVP is still blocked on a real WING read-only preflight. Offline
tooling must not pretend to close it, but it can make the future hardware proof
safer and easier to review.

## Proposed solution

1. Add `transport_timeout_or_connection_loss` to the plan-only fail-closed
   conditions in `tools/wing_readonly_preflight_runner.py`.
2. Add the same condition to
   `WING_READONLY_PREFLIGHT_PLAN_REQUIRED_FAIL_CLOSED_CONDITIONS` in
   `tools/no_wing_release_check.py`.
3. Catch `asyncio.TimeoutError` separately in the runner and write a blocked
   summary with `fail_closed: true`, `timeout_sec`, and
   `blockers: ["transport_timeout_or_connection_loss"]`.
4. Keep generic transport errors fail-closed as well.
5. Update the runbook fail-closed list and focused tests.

## Likely files to touch

- `tools/wing_readonly_preflight_runner.py`
- `tools/no_wing_release_check.py`
- `tests/test_wing_readonly_preflight_runner.py`
- `tests/test_no_wing_release_check.py`
- `Docs/runbooks/WING_READONLY_PREFLIGHT.md`
- `Docs/adr/wing-readonly-preflight-transport-timeout-artifact-20260607.md`

## Alternatives considered

- Keep the generic transport error: rejected because timeout/connection-loss is
  an explicit WING MVP proof dimension and should be queryable in artifacts.
- Treat timeout as a successful proof of fail-closed behavior: rejected because
  it is only blocked evidence and still requires operator review.

## Risks

A timeout after a request can leave an odd send-only event log. That is
acceptable and should stay reviewable as incomplete/blocked evidence, not as a
valid read-only candidate.

## Test plan

- Focused runner timeout artifact test.
- Focused no-WING plan pass/fail tests.
- no-WING quick release check.
- Full repository pytest before commit.

## Where the other agent may disagree

An independent reviewer may want finer-grained timeout stages. The current
change keeps scope narrow while preserving enough detail for operator review:
sent request types, timeout seconds, event log, and blocked summary.
