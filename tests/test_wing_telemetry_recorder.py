"""Tests for persistent WING telemetry dataset recording."""

import json

from wing_telemetry_recorder import WingTelemetryRecorder


def test_wing_telemetry_recorder_persists_dataset(tmp_path):
    recorder = WingTelemetryRecorder(
        base_dir=str(tmp_path),
        session_label="live_probe",
        metadata={"ip": "192.168.1.42", "observation_only": True},
    )

    started = recorder.start()
    recorder.record_event(
        "query_sent",
        address="/ch/1/fdr",
        source="send",
    )
    recorder.record_event(
        "console_observed",
        address="/ch/1/fdr",
        values=(-10.0,),
        source="incoming_osc",
        normalized_value=-10.0,
    )
    stopped = recorder.stop(reason="unit_test")

    assert started["active"] is True
    assert stopped["active"] is False
    assert stopped["event_count"] == 2

    with open(stopped["metadata_path"], "r", encoding="utf-8") as handle:
        metadata = json.load(handle)
    with open(stopped["events_path"], "r", encoding="utf-8") as handle:
        events = [json.loads(line) for line in handle if line.strip()]

    assert metadata["metadata"]["ip"] == "192.168.1.42"
    assert metadata["metadata"]["observation_only"] is True
    assert metadata["metadata"]["stop_reason"] == "unit_test"
    assert [event["event_type"] for event in events] == ["query_sent", "console_observed"]
