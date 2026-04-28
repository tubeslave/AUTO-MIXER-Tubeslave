# Mixing Station Visualization

## Task

Add a safe Mixing Station Desktop integration so Automixer decisions can be visualized
for Behringer Wing Rack and Allen & Heath dLive profiles.

## Constraints

- Dry-run/offline visualization is the default.
- Live-control must remain disabled unless explicitly configured.
- All outgoing visualization/control events must pass a safety layer and be logged.
- Existing WING OSC and dLive MIDI/TCP clients must not be broken.
- Unknown Mixing Station dataPaths must be marked as requiring discovery.

## Integration Point

The safest existing point is `AutoSoundcheckEngine._execute_action()`, after
`AutoFOHSafetyController.execute()` has produced a `SafetyDecision`. The Mixing
Station backend should mirror this decision into its own `AutomixCorrection`
model and log/send it independently.

## Verification

- Unit tests for capabilities, mapping, safety, REST/OSC clients, dry-run adapter,
  and log replay.
- Standard test command before merge:
  `PYTHONPATH=backend python -m pytest tests/ -x --tb=short -q`
