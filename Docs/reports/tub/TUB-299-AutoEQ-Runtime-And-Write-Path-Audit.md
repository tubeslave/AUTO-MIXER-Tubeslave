# TUB-299 AutoEQ Runtime Compatibility and Write-Path Audit

## Scope

TUB-299 covers the websocket/server AutoEQ runtime path and the EQ write
surfaces that could modify a mixer. This audit is code-only and test-only: no
live OSC transport, WING runtime, or real mixer was touched.

## Runtime Compatibility

- `AutoMixerServer.start_multi_channel_auto_eq(...)` normalizes `mode` to
  `soundcheck` or `live`.
- `MultiChannelAutoEQController.start_multi_channel(...)` accepts the same
  `mode` argument, stores it on `controller.mode`, and stores it per active
  channel.
- Unknown modes fall back to `soundcheck`.
- Regression coverage now exercises the real controller with a stubbed
  `SpectrumAnalyzer`, so no audio device is opened during the test.

## AutoEQ Write-Path Registry

| Surface | Write risk | Current guard |
| --- | --- | --- |
| `start_auto_eq` | Can lead to auto-apply if enabled | WING deployment boundary forces `auto_apply=False` |
| `apply_eq_correction` | Calls `AutoEQController.apply_to_mixer()` | Blocked by `_block_quarantined_wing_write_surface(...)` for WING boundary |
| `reset_eq` | Can call controller reset or direct EQ OSC reset | Blocked by `_block_quarantined_wing_write_surface(...)` for WING boundary |
| `reset_all_eq` | Direct EQ reset loop over channels | Blocked by `_block_quarantined_wing_write_surface(...)` for WING boundary |
| `apply_channel_correction` | Calls multi-channel controller mixer writes | Blocked by `_block_quarantined_wing_write_surface(...)` for WING boundary |
| `apply_all_corrections` | Applies all multi-channel corrections | Blocked by `_block_quarantined_wing_write_surface(...)` for WING boundary |
| `AutoEQController.apply_to_mixer` | Direct mixer EQ writes | Internal write primitive; should remain behind server/manual gate |
| `AutoEQController.reset_eq` | Direct mixer EQ reset writes | Internal write primitive; should remain behind server/manual gate |
| `MultiChannelAutoEQController.apply_channel_correction` | Direct mixer EQ writes | Internal write primitive; should remain behind server/manual gate |
| `MultiChannelAutoEQController.apply_all_corrections` | Direct mixer EQ writes | Internal write primitive; should remain behind server/manual gate |

## Result

The original `mode=` runtime mismatch is covered at both server and controller
levels. The remaining direct EQ write primitives are still present by design,
but the websocket-facing WING deployment boundary blocks the risky apply/reset
surfaces before they reach the mixer client.

## Follow-Up

- Route remaining non-WING direct EQ primitives through the normalized
  `MixerClientBase.apply_live_change(...)` contract when those surfaces are
  promoted beyond guarded/manual operation.
- Keep live validation separate from this task: it requires explicit operator
  approval, readback, rollback, cooldown, and emergency-stop proof.
