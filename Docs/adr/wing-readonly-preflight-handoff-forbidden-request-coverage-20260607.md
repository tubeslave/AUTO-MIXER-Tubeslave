# ADR: wing-readonly-preflight-handoff-forbidden-request-coverage-20260607

## Context

The current first non-closed MVP blocker is `WING read-only connection proof`.
The offline handoff packet prepares the operator for the real read-only
preflight run, but it previously named only a narrow subset of forbidden live
request types. The event-log validator also now requires WING mode to be proven
from event payloads, so the offline plan should name that fail-closed condition.

## Options considered

1. Keep the shorter handoff list.
2. Import the runner's forbidden request constants into the handoff builder.
3. Expand the handoff list, make the no-WING handoff artifact gate enforce it,
   and align the plan fail-closed list with event-log WING-mode proof.

## Decision

Use option 3.

## Why this won

The packet stays transport-free and operator-readable, while release validation
now fails if the packet omits a forbidden live write, proposal, rollback,
operator-mode, or connect/disconnect request type. The plan-only artifact also
advertises `event_log_wing_mode_not_proven` as a fail-closed condition.

## Rejected alternatives

Keeping the shorter list left avoidable ambiguity in the operator handoff.
Importing runner internals into the packet builder was unnecessary for this
narrow change.

## Implementation plan

1. Expand `FORBIDDEN_LIVE_REQUEST_TYPES` in the handoff packet builder.
2. Add `WING_READONLY_PREFLIGHT_HANDOFF_REQUIRED_FORBIDDEN_REQUEST_TYPES` to the
   no-WING validator.
3. Inventory the handoff packet builder as lab-only proof tooling because it now
   contains write-token strings as forbidden request names.
4. Add `event_log_wing_mode_not_proven` to the plan-only artifact and release
   gate required fail-closed conditions.
5. Test that valid packets include the broader set.
6. Test that an unsafe packet missing `set_gain` fails closed.

## Test plan

- `PYTHONPATH=backend python3 -m pytest -q tests/test_wing_readonly_preflight_handoff_packet.py tests/test_no_wing_release_check.py::test_no_wing_release_check_validates_wing_preflight_handoff_packet_artifact tests/test_no_wing_release_check.py::test_no_wing_release_check_rejects_unsafe_wing_preflight_handoff_packet_artifact tests/test_no_wing_release_check.py::test_no_wing_release_check_rejects_unsafe_wing_preflight_plan_artifact tests/test_wing_readonly_preflight_runner.py::test_plan_only_outputs_readonly_request_plan_without_websocket_or_artifacts`
- `PYTHONPATH=backend python3 -m pytest tests/test_write_surface_inventory_static_scan.py -q --tb=short`
- `PYTHONPATH=backend python3 tools/no_wing_release_check.py --quick --skip-frontend --artifact-root artifacts/no_wing_release_check_current_cycle --command-timeout-sec 300`
- `PYTHONPATH=backend python3 -m pytest tests/ -x --tb=short -q`

## Risks and rollback

Risk: future request types can still drift. Rollback is to remove the added
required list and tests, but that would weaken the operator-visible preflight
boundary. This ADR does not contact WING, grant live authority, or close any MVP
blocker.
