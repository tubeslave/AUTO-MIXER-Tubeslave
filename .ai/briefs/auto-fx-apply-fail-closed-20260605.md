# AutoFX Apply Fail-Closed Boundary

## Task

Close the legacy `AutoFXPlanner.apply_to_mixer()` direct FX/send write path while preserving the FX plan as reviewable evidence for assisted mixing.

## Context

- `backend/auto_fx.py` builds useful FX buses and post-fader send plans.
- `apply_to_mixer()` historically called mixer-client methods such as `set_fx_model`, `set_fx_on`, `set_fx_mix`, `set_fx_parameter`, `set_fx_return_level`, and `set_channel_send`.
- FX, insert, and send writes are quarantined in `backend/write_surface_inventory.py`.
- The current live-approved executor remains supervised Product Layer fader/gain only.

## Constraints

- Do not add FX/send writes to the live executor.
- Do not mutate a mixer client even if one is supplied.
- Preserve the ability to inspect the intended FX plan for operator review.
- Keep useful FX behavior as proposal-only or lab/offline evidence until readback, feedback checks, and rollback proof exist.

## Acceptance

- `AutoFXPlanner.apply_to_mixer()` returns `applied=False`.
- Fake mixer methods are not called.
- Would-be FX/send commands are returned as `blocked_calls`.
- Registry, write-surface inventory, capability matrix, and processing inventory record the fail-closed boundary.
- Focused tests prove the no-mutation behavior and metadata boundary.
