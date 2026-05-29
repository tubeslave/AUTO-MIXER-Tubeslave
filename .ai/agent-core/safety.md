# Agent Core Safety

## Purpose

This file is the global safety contract for agents working in Automixer. The
project controls live sound, so safety policy wins over target loudness,
automation speed, and model confidence.

## Non Negotiable Rules

- Live OSC writes are forbidden without explicit approval for the exact action.
- Live WING writes are forbidden without explicit approval for the exact action.
- All WING, OSC, Dante, gain, trim, fader, routing, snapshot, mute, EQ, dynamics,
  and FX actions default to audit-only or dry-run.
- Never raise a fader above `0 dBFS` without an explicit operator request.
- Preserve true-peak headroom and feedback safety before increasing gain.
- Pre-read is mandatory before any supervised parameter change.
- Rollback is mandatory before any supervised parameter change.
- Recommendations must not be treated as control commands.

## Mode Definitions

`audit-only` means the agent reads files, logs, traces, telemetry, or cached
state and produces a report. It does not prepare or send a runtime mutation.

`dry-run` means the agent computes or previews an action but does not mutate
Paperclip state, mixer state, OSC state, WING state, Dante routing, or audio
runtime state.

`supervised-write` means a bounded mutation may be performed only after explicit
operator approval and only with pre-read, safety check, readback, and rollback.

`live-write` means an actual command is sent to the console or live audio
runtime. Live-write mode is prohibited unless the operator explicitly approves
the exact target, parameter, range, timing, and rollback plan.

## Safety Check

Any gain, trim, fader, EQ, dynamics, routing, mute, snapshot, or FX mutation
must prove:

- target channel or bus identity is known;
- current parameter value was read first;
- proposed delta is bounded;
- true-peak and feedback constraints allow the action;
- fader ceiling is respected;
- action is idempotent or has duplicate protection;
- readback will verify the result;
- rollback can restore the pre-read value or a safer value.

## Failure Policy

If evidence is missing, stale, ambiguous, or unsafe, choose audit-only or
dry-run and report what is missing. Do not promote a recommendation into a
command.
