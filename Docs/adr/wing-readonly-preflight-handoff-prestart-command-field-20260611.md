# ADR: WING Read-Only Preflight Handoff Prestart Command Field

## Context

The current MVP is blocked on real WING read-only connection proof. The operator handoff packet already tells the operator to run the prestart no-WING release check before opening the intended backend/WING session, but the exact command lived only inside the nested `commands` block.

## Options considered

1. Keep the command only in `commands.prestart_backend_port_check`.
2. Add a top-level safety field and validate it in no-WING artifact review.
3. Require a completed no-WING summary hash in the handoff packet.

## Decision

Use option 2. The handoff packet now includes `backend_port_prestart_check_command` at the top level, renders it in markdown, and no-WING artifact validation rejects mismatches.

## Why this won

The handoff packet is an operator-facing safety artifact. Its top-level boundary should state the exact prestart command alongside the existing required flags and diagnostic fields.

## Rejected alternatives

Option 1 leaves the safety boundary less self-contained. Option 3 may be useful later but adds a new artifact dependency outside this small hardening slice.

## Implementation plan

- Add the top-level command to `tools/build_wing_readonly_preflight_handoff_packet.py`.
- Render the command in the markdown packet.
- Add no-WING validation for the top-level command.
- Extend handoff and no-WING tests.

## Test plan

- `python3 -m py_compile tools/build_wing_readonly_preflight_handoff_packet.py tools/no_wing_release_check.py`
- `PYTHONPATH=backend python3 -m pytest -q tests/test_wing_readonly_preflight_handoff_packet.py tests/test_no_wing_release_check.py`
- Adjacent WING preflight/no-WING tests.

## Risks and rollback

Risk: older generated handoff packets without this field fail artifact validation. Rollback: remove the top-level field and corresponding no-WING validator assertion. No live write path is changed.
