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
python3 -m backend.telemetry.streaming_alerts --session-id <session_id>
```

Streaming alerts:

- Reads existing session JSONL telemetry.
- Emits `alerts/alerts.jsonl`, `alerts/dashboard.csv`, and `alerts/summary.json`.
- Detects short-loudness drift, fader jumps, manual override bursts, clipping
  risk, and low headroom.
- Remains read-only and does not import WING/OSC clients.

Product Layer read model:

- `get_telemetry_alerts_snapshot` reads already-generated alert artifacts for
  cockpit/frontend display.
- The snapshot does not run the alert generator and does not create files; it
  only reports existing summaries and alert rows with `live_write=false`.
- Stored summary safety/path fields are treated as untrusted evidence; the
  snapshot normalizes session safety to read-only/no-live-write and reports the
  local `alerts/` artifact paths it actually inspected.

The CLI is explicit operator action. Runtime auto-collection should still obey
`telemetry.enabled: false` unless deliberately enabled later.
