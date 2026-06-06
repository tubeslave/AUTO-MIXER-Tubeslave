# Brief: WING Read-Only Preflight Runner

Task id: `wing-readonly-preflight-runner-20260606`
Date: 2026-06-06

## Goal

Reduce the supervised WING Rack MVP blockers for read-only connection proof,
physical console readback proof, and operator-visible evidence.

## Scope

- Add a runner that talks only to the already-running backend WebSocket.
- Use only read-only messages: `get_operator_mode`, `get_state`, and
  `query_mixer_parameter`.
- Write local artifacts with events and summary JSON.
- Block when backend is not in WING mode or physical readback is required but
  missing.

## Safety Boundary

No live WING/OSC write is performed by this task. The runner must not send
`set_fader`, `set_gain`, trim/preamp-gain writes, EQ, compressor, routing,
snapshots, reset, mute/unmute, insert, FX, `set_operator_mode`, proposal
apply/create, rollback, emergency stop, connect, or disconnect messages.

## Definition of Done

- Offline fake-WebSocket tests prove pass, non-WING block, and missing physical
  readback block cases.
- The operator report explains that this is tooling/readiness evidence, not a
  completed hardware proof.
- Memory lint and focused pytest pass.
