# WING OSC Policy

## Purpose

This policy keeps Behringer WING Rack control separate from analysis,
recommendations, and reports.

## Default Mode

The default mode for all WING, OSC, Dante, gain, trim, fader, routing, mute,
snapshot, EQ, dynamics, and FX work is `audit-only` or `dry-run`.

## Mode Separation

`audit-only` reads code, docs, traces, telemetry, cached mixer state, or reports.
It does not build a command that can mutate the console.

`dry-run` builds or validates a proposed action without sending it. Dry-run may
write local artifacts, reports, or tests, but it must not mutate WING, OSC,
Dante, or live audio state.

`supervised-write` is a gated path for a bounded operator-approved action. It
requires pre-read, safety check, explicit approval, write, readback, and
rollback.

`live-write` sends a real command to the console or live runtime. Live-write is
forbidden without explicit approval for the exact target and value.

## Hard Restrictions

- Live OSC writes are forbidden without explicit approval.
- Live WING writes are forbidden without explicit approval.
- Routing, snapshots, mute, fader, gain, trim, EQ, compressor, gate, FX, and
  Dante changes must remain audit-only or dry-run unless explicitly approved.
- Recommendation rows, Paperclip tasks, LLM tool calls, and UI proposals are not
  live-write authority.

## Supervised Write Requirements

Before any supervised write:

- identify target console and channel or bus;
- pre-read the current value;
- compute bounded delta and rate limits;
- pass safety check;
- record rollback value or safer fallback;
- require explicit operator approval;
- write only the approved parameter;
- read back and verify;
- log evidence and rollback instructions.

## Rollback

Rollback must be specific enough for another operator or agent to execute. It
must name the pre-read value, target path, command boundary, and safer fallback
if the original value cannot be restored.
