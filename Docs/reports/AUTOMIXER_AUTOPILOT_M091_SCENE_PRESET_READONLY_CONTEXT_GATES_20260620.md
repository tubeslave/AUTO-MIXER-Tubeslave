# M091 Scene/Preset Read-Only Context Gates

## Summary

M091 scene preset read-only context gates is now covered as offline
scene/snapshot/preset management hardening. Scene/preset proposal candidates
require fresh read-only/no-authority scene/preset context, target preview,
scoped diffs, rollback evidence and destructive-scope limits before destructive
scene decisions can trust them.

This advances scene/snapshot/preset management, a separate console-control
capability required by the project goal. It is not another pass over pan, mute
or routing.

## Boundary

- No backend WebSocket was opened.
- No WING/OSC transport was opened.
- No live write was requested.
- No proposal apply path was enabled.
- No readiness closure was produced.
- No hardware evidence was produced.
- Full Autonomous Mixing Mode remains blocked.

## Validation

- Focused scene/preset validation: `22 passed`.
- Capability matrix focused validation: `1 passed`.
- M091 rail validation: recorded in `ops/autopilot/STATE.json`.
- Full WING rail: recorded in `ops/autopilot/STATE.json`.
- Clean staged full suite: recorded in `ops/autopilot/STATE.json`.
- Memory lint and JSON/JSONL validation: recorded in `ops/autopilot/STATE.json`.

## Operator Impact

Scene and preset changes remain proposal-only. Missing, stale or
live-authority-bearing current scene/preset context stays analysis-only, and
operator proposals continue to report `can_apply_live: false`.

## Remaining Blocker

M001 still waits for external WING/operator validation. Real scene/preset live
authority remains unavailable until a separate ADR, physical readback,
rollback proof, destructive-action proof and operator-approved hardware
evidence exist.
