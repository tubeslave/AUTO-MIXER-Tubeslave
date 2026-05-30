# Automixer Learning Telemetry

This package records local concert, soundcheck, rehearsal and offline-test
telemetry for future AI-agent training.

Safety boundary:

- No OSC/MIDI/WING writes.
- No mixer state changes.
- Config defaults keep telemetry disabled.
- Collectors read cached state or mock data only.

Default session root:

`data/telemetry/sessions/<session_id>/`

Files:

- `session.json`
- `scene_metadata.json`
- `channel_telemetry.jsonl`
- `bus_telemetry.jsonl`
- `spl_telemetry.jsonl`
- `engineer_actions.jsonl`
- `labels.jsonl`
- `manifest.json`

CLI:

```bash
python3 -m backend.telemetry.session_manager start --mode soundcheck --venue "Test Venue"
python3 -m backend.telemetry.session_manager stop --session-id <session_id>
python3 -m backend.telemetry.learning_export summarize --session-id <session_id>
```

The CLI is explicit operator action. Runtime auto-collection should still obey
`telemetry.enabled: false` unless deliberately enabled later.
