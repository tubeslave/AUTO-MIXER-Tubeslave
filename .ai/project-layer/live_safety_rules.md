# Live Safety Rules

## Purpose

These are the Automixer-specific safety rules for live-sound behavior.

## Hard Rules

- Live OSC writes are forbidden without explicit approval.
- Live WING writes are forbidden without explicit approval.
- All WING actions start in audit-only or dry-run mode.
- Gain, trim, and fader changes require safety check before execution.
- Rollback is mandatory for supervised write.
- Pre-read is mandatory before changing parameters.
- Recommendations and real control commands must remain separate.
- Never raise faders above `0 dBFS` without an explicit operator request.
- Preserve true-peak headroom and feedback safety before increasing gain.
- Feedback protection wins over target loudness and model confidence.

## Mode Definitions

`audit-only`: inspect and report. No runtime mutation.

`dry-run`: compute and preview. No runtime mutation.

`supervised-write`: explicit approval, bounded write, readback, and rollback.

`live-write`: real console mutation. Forbidden unless explicitly approved for
the exact action.

## Required Safety Check

For gain, trim, fader, EQ, compressor, gate, FX, routing, mute, or snapshot
changes, confirm:

- target identity;
- current value from pre-read;
- allowed range;
- bounded delta;
- headroom;
- feedback risk;
- operator approval;
- readback plan;
- rollback plan.

## Default Response To Ambiguity

If the task is ambiguous, stale, or missing approval, stay in audit-only or
dry-run mode and explain the missing evidence.
