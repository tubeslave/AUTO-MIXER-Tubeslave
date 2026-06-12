# Codex Proposal: WING Read-Only Handoff First Prestart Gate

## Short thesis

The handoff packet should prove both ends of the operator sequence: the
backend-port prestart gate is first, and the supervised proof gate is last.

## What I understood

The current first WING MVP blocker remains open until real WING read-only
evidence exists. The offline handoff packet already includes the prestart
command and a final supervised gate, but the prestart command is not represented
as a structured first-step invariant.

## Proposed solution

- Add `backend_port_prestart_check_operator_sequence_step_index: 0`.
- Add `backend_port_prestart_check_must_be_first_operator_step: true`.
- Have the no-WING validator derive
  `backend_port_prestart_check_operator_sequence_step_is_first`.
- Fail closed if the packet moves the prestart gate after backend/WING startup.
- Update the runbook and rails tests.

## Likely files to touch

- `tools/build_wing_readonly_preflight_handoff_packet.py`
- `tools/no_wing_release_check.py`
- `tests/test_wing_readonly_preflight_handoff_packet.py`
- `tests/test_no_wing_release_check.py`
- `tests/test_wing_mvp_rails.py`
- `Docs/runbooks/WING_READONLY_PREFLIGHT.md`

## Alternatives considered

- Keep relying on prose in the first operator sequence item. Rejected because
  downstream validators should have machine-readable fields for the first gate.
- Add another report-only note. Rejected because the useful progress is a
  contract and fail-closed validator, not another narrative artifact.

## Risks

The packet schema becomes stricter, so stale packets may fail validation. That
is acceptable because stale handoff packets should be regenerated before real
hardware work.

## Test plan

- Focused handoff packet tests.
- Focused no-WING validator tests.
- WING MVP rails tests.
- Full repository pytest before commit.

## Where the other agent may disagree

Kimi may argue that the first sequence item was already clear enough. The
counterpoint is that first-step ordering is a safety boundary and should be
machine-checkable like the final supervised prestart gate.
