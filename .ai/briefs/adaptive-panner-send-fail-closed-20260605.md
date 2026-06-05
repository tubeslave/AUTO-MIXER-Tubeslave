# Adaptive Panner Send Fail-Closed Boundary

## Task

Close the legacy `AutoPannerController._send_osc()` direct adaptive pan write path while preserving adaptive pan decisions as reviewable evidence.

## Context

- `backend/auto_panner_adaptive.py` computes multi-band adaptive panning decisions with genre templates and smoothed pan gains.
- `_send_osc()` historically called `mixer_client.set_channel_pan`.
- `backend/auto_panner.py` and `backend/auto_fader_v2/controller.py` direct pan apply paths are already fail-closed in this worktree.
- Pan writes remain quarantined until current-pan readback, mono compatibility, and rollback proof exist.

## Constraints

- Do not add pan writes to the live executor.
- Do not mutate a mixer client even if one is supplied.
- Preserve adaptive pan decision output and would-be command evidence for operator review.
- Keep panning proposal-only until physical pan readback proof exists.

## Acceptance

- `_send_osc()` returns `applied=False`.
- Fake mixer methods are not called.
- Would-be `set_channel_pan` commands are returned as `blocked_calls`.
- Runtime status exposes blocked write telemetry.
- Registry, write-surface inventory, capability matrix, processing inventory, tests, and ADR record the fail-closed boundary.
