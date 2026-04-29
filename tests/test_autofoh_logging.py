"""Tests for AutoFOH JSONL structured logging."""

import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "backend"))

from autofoh_logging import (
    AutoFOHStructuredLogger,
    build_session_report_from_jsonl,
    render_session_report_summary,
)


def test_structured_logger_writes_jsonl_events(tmp_path):
    log_path = tmp_path / "autofoh.jsonl"
    logger = AutoFOHStructuredLogger(path=log_path, queue_maxsize=8)

    logger.start()
    logger.log_event(
        "action_decision",
        channel_id=3,
        action={"type": "ChannelEQMove", "gain_db": -1.0},
        message="sent",
    )
    logger.stop()

    lines = log_path.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 1
    payload = json.loads(lines[0])
    assert payload["event_type"] == "action_decision"
    assert payload["channel_id"] == 3
    assert payload["action"]["gain_db"] == -1.0


def test_session_report_summarizes_phase_target_guard_blocks(tmp_path):
    log_path = tmp_path / "autofoh.jsonl"
    logger = AutoFOHStructuredLogger(path=log_path, queue_maxsize=16)

    logger.start()
    logger.log_event(
        "phase_target_guard_blocked",
        channel_id=2,
        runtime_state="SNAPSHOT_LOCK",
        action={"action_type": "ChannelEQMove"},
        message="phase target guard blocked EQ cut inside learned green corridor",
        metadata={"phase_name": "SNAPSHOT_LOCK"},
    )
    logger.log_event(
        "action_decision",
        channel_id=2,
        requested_action={"action_type": "ChannelEQMove"},
        applied_action={"action_type": "ChannelEQMove"},
        requested_runtime_state="SNAPSHOT_LOCK",
        sent=False,
        allowed=False,
        supported=True,
        bounded=False,
        rate_limited=False,
        message="phase target guard blocked EQ cut inside learned green corridor",
    )
    logger.stop()

    report = build_session_report_from_jsonl(log_path)

    assert report.total_events == 2
    assert report.guard_block_count == 1
    assert report.action_blocked_count == 1
    assert report.guard_blocks_by_action_type["ChannelEQMove"] == 1
    assert report.guard_blocks_by_phase["SNAPSHOT_LOCK"] == 1
    assert report.channels_with_guard_blocks == [2]
    summary = render_session_report_summary(report)
    assert "guard_blocks=1" in summary
    assert "channels=[2]" in summary
