# show-assist-auto-v1 gated live assistant

## Goal

Prepare the AutoFOH soundcheck/live path so it can behave as a sound engineer
assistant with automatic application only for methods that have a live-write
proof contract.

## Constraints

- Do not run real WING writes during implementation.
- Keep routing, snapshots, reset, channel naming, EQ, compression, gate, pan,
  phase, bus/DCA/master moves in proposal/shadow mode until separately proven.
- Require pre-read, readback, rollback metadata, and failure demotion for any
  promoted live action.
- Preserve the existing AutoFOH safety controller and current tests.

## Target

`show_assist_auto_v1` promotes only channel input trim, channel fader moves, and
emergency feedback notches by default. Other methods still participate in
analysis and recommendations, but they are imported to the operator queue
instead of writing to the console.

## Test Plan

- Focused gated executor tests.
- AutoSoundcheck routing test for dry-run recommendation bundle export.
- Manual WING observation-only gate test.
- Standard backend and frontend validation before use.
