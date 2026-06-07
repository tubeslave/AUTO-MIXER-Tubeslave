# Brief: WING Read-Only Preflight Readback Identity Gate

## Task

Make the real WING read-only preflight runner block when a physical readback
response does not match the requested channel, value type, and OSC address.

## Context

The artifact review layer already rejects mismatched query/readback identity in
`events.jsonl`. The runner itself could still mark the run `passed` when the
backend returned a physical readback for the wrong channel or address. That is
too late for the first MVP blocker: the capture artifact should fail closed at
the source.

## Safety Boundary

- Do not query WING in this task.
- Do not add any write, proposal, rollback, emergency-stop, or operator-mode
  mutation request.
- Do not close the WING read-only blocker from offline tests.
- Keep supervised fader/gain proof blocked until real artifact review and an
  accepted decision JSON exist.

## Acceptance Criteria

- Runner compares every `mixer_parameter_value` readback identity to the
  `query_mixer_parameter` request.
- Mismatched identity adds a `readback_identity_mismatch` blocker and returns a
  blocked run.
- Offline plan fail-closed conditions include `readback_identity_mismatch`.
- no-WING release checks require that condition in plan artifacts.
- Focused tests cover a physical-but-wrong readback.
