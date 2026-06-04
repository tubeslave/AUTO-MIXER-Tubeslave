# Automixer Streaming Alerts Implementation - 2026-06-01

## Summary

Implemented a read-only `StreamingAlertEngine` for concert/soundcheck telemetry.
The engine consumes existing telemetry JSONL timelines and writes local alert
artifacts for frontend dashboards, QA, and future learning data.

## Implemented

- `backend.telemetry.streaming_alerts`
  - reads exported telemetry timeline rows;
  - detects `lufs_drift`, `fader_jump`, `manual_override_burst`,
    `clipping_risk`, and `headroom_low`;
  - writes `alerts.jsonl`, `dashboard.csv`, and `summary.json`;
  - exposes a CLI:

```bash
PYTHONPATH=. python3 -m backend.telemetry.streaming_alerts \
  --storage-dir data/telemetry/sessions \
  --session-id <session_id>
```

- Focused tests in `tests/test_telemetry_learning.py`.
- Documentation updates in `backend/telemetry/README.md` and
  `Docs/telemetry_learning_data.md`.

## Safety

No live OSC/WING writes were added or run. The new module does not import WING,
OSC, MIDI, or mixer client transports. It only reads local telemetry artifacts
and writes local alert/dashboard files.

## Validation

- `PYTHONPATH=backend python -m pytest tests/test_telemetry_learning.py -q`
  - `13 passed`
- `python3 -m py_compile backend/telemetry/streaming_alerts.py backend/telemetry/schemas.py backend/telemetry/learning_export.py`
  - passed
- `rg -n "set_fader\\(|set_gain\\(|route_output\\(|load_snap\\(|save_snap\\(|send\\(" backend/telemetry`
  - no matches

## Next Steps

1. Expose a read-only Product Layer DTO for latest telemetry alerts.
2. Add frontend panels/widgets for active channel/master alerts.
3. Add fixture sessions from real/replayed WING telemetry to tune thresholds.
4. Add optional robust window stats once real telemetry volume is available.
