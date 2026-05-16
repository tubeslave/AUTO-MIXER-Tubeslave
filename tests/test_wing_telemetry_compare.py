"""Tests for offline WING telemetry summary/comparison helpers."""

import json

from wing_telemetry_compare import compare_sessions, summarize_session


def _write_session(base_dir, name, events, metadata=None):
    session_dir = base_dir / name
    session_dir.mkdir()
    with (session_dir / "metadata.json").open("w", encoding="utf-8") as handle:
        json.dump(
            {
                "session_id": name,
                "session_label": name,
                "metadata": metadata or {},
            },
            handle,
        )
    with (session_dir / "events.jsonl").open("w", encoding="utf-8") as handle:
        for event in events:
            handle.write(json.dumps(event) + "\n")
    return session_dir


def test_summarize_session_detects_fallback_only_capture(tmp_path):
    session_dir = _write_session(
        tmp_path,
        "fallback_only",
        [
            {"seq": 1, "timestamp": 1.0, "event_type": "query_sent", "address": "/ch/1/fdr"},
            {"seq": 2, "timestamp": 2.0, "event_type": "console_observed", "address": "/ch/1/fdr", "values": [0.5]},
            {"seq": 3, "timestamp": 3.0, "event_type": "blocked_write", "address": "/ch/1/fdr", "values": [0.7]},
        ],
        metadata={"observation_only": True},
    )

    summary = summarize_session(session_dir)

    assert summary["event_count"] == 3
    assert summary["fallback_only_observation"] is True
    assert summary["blocked_write_count"] == 1
    assert summary["observed_address_classes"]["fallback"] == ["/ch/1/fdr"]


def test_compare_sessions_reports_address_differences(tmp_path):
    left_dir = _write_session(
        tmp_path,
        "live",
        [
            {"seq": 1, "timestamp": 1.0, "event_type": "console_observed", "address": "/$meters/ch/1/in", "values": [0.1]},
            {"seq": 2, "timestamp": 2.0, "event_type": "console_observed", "address": "/ch/1/fdr", "values": [0.5]},
        ],
    )
    right_dir = _write_session(
        tmp_path,
        "replay",
        [
            {"seq": 1, "timestamp": 1.0, "event_type": "console_observed", "address": "/$meters/ch/1/in", "values": [0.1]},
            {"seq": 2, "timestamp": 2.0, "event_type": "console_observed", "address": "/$meters/ch/2/in", "values": [0.2]},
            {"seq": 3, "timestamp": 3.0, "event_type": "write_sent", "address": "/ch/2/fdr", "values": [0.9]},
        ],
    )

    comparison = compare_sessions(left_dir, right_dir)

    assert comparison["observed_address_only_in_left"] == ["/ch/1/fdr"]
    assert comparison["observed_address_only_in_right"] == ["/$meters/ch/2/in"]
    assert comparison["safety_flags"]["right_has_write_sent"] is True
    assert comparison["event_type_delta"]["write_sent"] == -1
