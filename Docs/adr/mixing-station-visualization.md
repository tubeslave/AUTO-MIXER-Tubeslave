# Mixing Station Visualization Backend

## Context

Automixer already controls WING and dLive through dedicated clients and an
AutoFOH safety controller. Mixing Station Desktop should visualize the same
automation decisions without accidentally sending dangerous live-console changes.

## Options considered

1. Add Mixing Station as a new primary mixer client.
2. Add Mixing Station as a mirror backend after existing safety decisions.
3. Patch each legacy direct OSC/MIDI call site.

## Decision

Use a separate Mixing Station adapter and mirror `SafetyDecision` output from
`AutoSoundcheckEngine._execute_action()`.

## Why this won

This preserves existing WING/dLive logic, keeps dry-run/offline visualization as
the default, and makes all Mixing Station writes pass their own capability,
mapping, and safety checks.

## Rejected alternatives

Replacing the active mixer client was rejected because it would risk breaking
live-console behavior. Patching every direct legacy call was rejected for this
phase because it increases blast radius.

## Implementation plan

Add `backend/integrations/mixing_station/`, configuration under
`config/mixing_station*`, CLI scripts for connection/discovery/test/replay, JSONL
logging, and unit tests. Unknown dataPaths remain blocked with `needs_discovery`.

## Test plan

Run targeted Mixing Station tests and then the standard project pytest command:

`PYTHONPATH=backend python -m pytest tests/ -x --tb=short -q`

## Risks and rollback

If the adapter misbehaves, set `enabled: false` in `config/mixing_station.yaml`
or create `runtime/EMERGENCY_STOP`. Because the adapter is additive and disabled
by default, rollback is removing the new module/config/scripts/docs plus the
small `_execute_action()` mirror hook.
