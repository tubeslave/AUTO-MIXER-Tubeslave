# wing-readonly-preflight-backend-port-prestart-check-20260612

## Context

The WING read-only preflight requires an existing backend/WING session for the
real capture, but a local port conflict can make `ws://localhost:8765` point at a
different service before that session is started. Recent no-WING diagnostics can
identify the listener owner, but the offline WING plan and handoff packet did
not yet require that pre-start check.

## Options considered

1. Leave the backend-port check only in the runbook.
2. Run the no-WING gate automatically from the preflight runner.
3. Add a required pre-start backend-port check to the offline plan and handoff
   packet metadata.

## Decision

Use option 3. The offline plan and handoff packet now require
`backend_port_prestart_check_required=true`, record the canonical
`PYTHONPATH=backend python3 tools/no_wing_release_check.py --quick` command, and
state that it must pass before starting the intended backend/WING session. The
no-WING WING-readonly evidence overlay also surfaces the same command so the
operator-visible readiness summary starts with the pre-start port check. The
next WING proof plan validator also requires that the read-only preflight
overlay carried this pre-start check before it can be treated as aligned with
the supervised proof plan. The backend autonomy readiness snapshot now emits the
same pre-start check metadata, and the supervised proof plan summary preserves
it.

## Why this won

This keeps the real preflight runner focused on read-only evidence capture while
making the operator handoff self-contained. It also avoids pretending that the
no-WING gate can pass while the intended backend is already listening on the
same port.

## Rejected alternatives

Runbook-only guidance was rejected because artifact review could miss the port
precondition. Automatic execution from the preflight runner was rejected because
the runner should not mutate or police local services and the real run needs the
intended backend to be reachable.

## Implementation plan

1. Add pre-start backend-port check fields to the plan-only artifact.
2. Add the same fields and command to the handoff packet and markdown.
3. Extend no-WING release checks to validate the new plan/handoff contract.
4. Surface the pre-start command in the WING-readonly evidence overlay.
5. Require the same pre-start command in the `next_wing_proof_plan` preflight
   overlay validation.
6. Emit the pre-start check from backend autonomy readiness and carry it through
   supervised proof plan summaries.
7. Clarify the runbook sequence.

## Test plan

- `PYTHONPATH=backend python3 -m pytest -q tests/test_wing_readonly_preflight_runner.py tests/test_wing_readonly_preflight_handoff_packet.py tests/test_no_wing_release_check.py`
- `PYTHONPATH=backend python3 -m pytest tests/ -x --tb=short -q`

## Risks and rollback

The pre-start check can fail when another intended service is already using the
port. That is still safer than capturing a preflight against the wrong backend.
Rollback is to remove the pre-start fields and command validation from the
plan/handoff artifacts and revert this ADR.
