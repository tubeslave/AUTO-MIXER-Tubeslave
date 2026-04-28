import json

from scripts.replay_automix_log_to_mixing_station import load_replay_corrections


def test_log_replay_loads_corrections_from_jsonl(tmp_path):
    log_path = tmp_path / "automix_to_mixing_station.jsonl"
    row = {
        "timestamp": "2026-04-27T12:00:00+00:00",
        "console_profile": "wing_rack",
        "mode": "offline_visualization",
        "channel_index": 0,
        "channel_name": "Lead Vocal",
        "strip_type": "input",
        "parameter": "fader",
        "requested_value": -4.0,
        "unit": "db",
        "safety_status": "allowed",
        "dry_run": True,
        "reason": "replay",
    }
    log_path.write_text(json.dumps(row) + "\n", encoding="utf-8")

    corrections = load_replay_corrections(log_path, dry_run=True)

    assert len(corrections) == 1
    assert corrections[0].parameter == "fader"
    assert corrections[0].value == -4.0
    assert corrections[0].dry_run is True
