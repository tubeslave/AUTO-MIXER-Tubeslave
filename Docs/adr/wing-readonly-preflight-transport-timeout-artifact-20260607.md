# ADR: wing-readonly-preflight-transport-timeout-artifact-20260607

## Context

The first supervised WING MVP blocker is still `WING read-only connection
proof`. A real preflight run can fail due to backend timeout or connection loss.
That failure must leave reviewable fail-closed evidence and must not look like a
missing or successful hardware proof.

## Options considered

1. Leave transport failures as generic `read_only_preflight_transport_error`.
2. Add timeout/connection-loss as a named fail-closed condition and artifact
   field.
3. Treat timeout as proof that the connection-loss blocker is closed.

## Decision

Use option 2.

## Why this won

The runner remains read-only and transport failures remain blocked, but the
operator and release gate can now identify the exact fail-closed boundary:
`transport_timeout_or_connection_loss`.

## Rejected alternatives

Generic transport errors were too weak for operator review. Treating timeout as
closed proof would overstate readiness because a controlled real hardware
artifact still needs manual review.

## Implementation plan

1. Add `transport_timeout_or_connection_loss` to the offline plan fail-closed
   list.
2. Make no-WING release validation require the condition.
3. Catch `asyncio.TimeoutError` separately and write a blocked summary with
   `fail_closed: true`.
4. Mark generic transport exceptions fail-closed with the same blocker.
5. Update runbook and tests.

## Test plan

- `python3 -m py_compile tools/wing_readonly_preflight_runner.py tools/no_wing_release_check.py`
- `PYTHONPATH=backend python3 -m pytest -q tests/test_wing_readonly_preflight_runner.py tests/test_no_wing_release_check.py::test_no_wing_release_check_validates_wing_preflight_plan_artifact tests/test_no_wing_release_check.py::test_no_wing_release_check_rejects_unsafe_wing_preflight_plan_artifact`
- `PYTHONPATH=backend python3 -m pytest -q tests/test_wing_readonly_preflight_evidence.py tests/test_wing_readonly_preflight_runner.py tests/test_wing_readonly_preflight_handoff_packet.py tests/test_wing_readonly_preflight_review_tool.py tests/test_wing_readonly_preflight_review_decision.py tests/test_wing_readonly_preflight_review_decision_tool.py tests/test_no_wing_release_check.py`
- `PYTHONPATH=backend python3 tools/no_wing_release_check.py --quick --skip-frontend --artifact-root /tmp/automixer_wing_readonly_transport_timeout_gate_20260607 --command-timeout-sec 300`
- `PYTHONPATH=backend python3 -m pytest tests/ -x --tb=short -q`

## Risks and rollback

Risk: a send-only timeout event log is incomplete by design. The artifact must
remain blocked and manually reviewed. Rollback is to remove the named condition
and timeout-specific summary fields, but that would weaken the future hardware
proof trail.
