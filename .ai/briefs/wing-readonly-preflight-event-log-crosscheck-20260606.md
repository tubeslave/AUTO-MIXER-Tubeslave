# Brief: WING Read-Only Preflight Event-Log Crosscheck

## Task

Harden `wing_readonly_preflight.v1` evidence intake so a filesystem
`summary.json` cannot be accepted as a readiness-review candidate unless its
`events.jsonl` exists and matches the recorded read-only runner flow.

## Constraints

- No live WING/OSC/WebSocket command is allowed for this implementation task.
- The evaluator must stay transport-free and read local artifacts only.
- Event-log validation must fail closed on missing, corrupt, mismatched, or
  mutation-looking send events.
- Event-log validation must reject missing, duplicate, skipped, or otherwise
  non-contiguous `seq` values.
- Inline unit-test payloads may remain supported without an event file.

## Safety Boundary

This can make operator-visible evidence harder to forge or accidentally copy
incompletely. It does not close any WING MVP blocker, grant live authority, or
allow supervised fader/gain apply.

## Validation

- Focused preflight evidence tests.
- Readiness and no-WING release-check tests that consume the same evidence.
