# WING MVP Handoff Gate Contract Source

Task id: `wing-mvp-handoff-gate-contract-source-20260613`

## Context

The first open MVP blocker remains `WING read-only connection proof`.
The WING read-only handoff packet and no-WING release validator both depend on
two structured gate contracts:

- `backend_port_prestart_gate`, which must be the first operator step before an
  intended backend/WING session starts.
- `supervised_prestart_gate`, which must be the final operator step before any
  supervised fader/gain proof planning.

Those gate shapes and marker lists were still duplicated in the packet builder
and no-WING validator.

## Constraints

- Do not contact WING hardware.
- Do not start, stop, or reconfigure the backend listener.
- Do not grant live authority from handoff or no-WING artifacts.
- Preserve existing packet fields and validator output.
- Keep the current blocker status unchanged until real WING artifacts are
  manually reviewed.

## Required Outcome

Move the handoff gate marker lists, step text, step indexes, runbook target, and
expected gate builders into `backend/wing_readonly_preflight_contract.py`, then
have the packet builder and no-WING validator consume that shared contract.
