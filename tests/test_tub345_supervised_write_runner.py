import argparse
from pathlib import Path

import pytest

from tub345_supervised_write_runner import (
    build_no_clipping_evidence,
    build_run_summary,
    extract_request,
    make_artifact_path,
    make_session_label,
    numeric_values_match,
    require_real_write_confirmation,
    should_ignore_message,
    summarize_manual_result,
    summarize_rollback_result,
)


def test_should_ignore_message_filters_background_updates():
    assert should_ignore_message("mixer_update") is True
    assert should_ignore_message("state_update") is True
    assert should_ignore_message("manual_write_applied") is False


def test_make_session_label_uses_prefix_when_unspecified(monkeypatch):
    monkeypatch.setattr("time.strftime", lambda _fmt: "20260514_210000")
    assert make_session_label(None) == "tub350_supervised_live_20260514_210000"
    assert make_session_label("custom_label") == "custom_label"


def test_make_artifact_path_sanitizes_session_label():
    path = make_artifact_path("artifacts/tub345", "label with spaces/#1")
    assert path == Path("artifacts/tub345/label_with_spaces__1_summary.json")


def test_require_real_write_confirmation_fails_without_flag():
    args = argparse.Namespace(confirm_real_write=False)
    with pytest.raises(SystemExit) as exc_info:
        require_real_write_confirmation(args)
    assert exc_info.value.code == 2


def test_summarize_manual_result_extracts_request_fields():
    payload = {
        "type": "manual_write_applied",
        "status": "applied",
        "request": {
            "channel": 7,
            "value_type": "fader",
            "effective_value": -12.0,
            "readback_value": -12.0,
            "old_value": -12.1,
            "approval_id": "approval-7",
            "rollback": None,
        },
    }
    summary = summarize_manual_result(payload)
    assert summary["message_type"] == "manual_write_applied"
    assert summary["status"] == "applied"
    assert summary["channel"] == 7
    assert summary["effective_value"] == -12.0
    assert summary["old_value"] == -12.1
    assert summary["approval_id"] == "approval-7"


def test_summarize_rollback_result_extracts_first_step():
    payload = {
        "type": "manual_write_rollback_applied",
        "status": "applied",
        "reason": "rollback_applied",
        "mode": "rollback_all",
        "rolled_back_count": 3,
        "remaining_step_count": 0,
        "rolled_back_steps": [
            {
                "original_step": {"channel": 7, "value_type": "fader", "old_value": -5.0},
                "rollback_result": {
                    "channel": 7,
                    "value_type": "fader",
                    "effective_value": -5.0,
                    "readback_value": -5.0,
                },
            }
        ],
    }
    summary = summarize_rollback_result(payload)
    assert summary["message_type"] == "manual_write_rollback_applied"
    assert summary["mode"] == "rollback_all"
    assert summary["rolled_back_count"] == 3
    assert summary["readback_value"] == -5.0
    assert summary["restore_target_value"] == -5.0


def test_extract_request_requires_request_payload():
    with pytest.raises(RuntimeError):
        extract_request({"type": "manual_write_applied"})


def test_numeric_values_match_handles_tolerance_and_bad_values():
    assert numeric_values_match(-10.0, -10.005) is True
    assert numeric_values_match(-10.0, -10.2) is False
    assert numeric_values_match("same", "same") is True


def test_build_no_clipping_evidence_marks_missing_measurement_explicitly():
    args = argparse.Namespace(
        no_clipping_threshold_dbtp=-3.0,
        max_true_peak_dbtp=None,
        no_clipping_sample_count=None,
    )
    payload = build_no_clipping_evidence(args)
    assert payload["measured"] is False
    assert payload["clear"] is False
    assert payload["threshold_dbtp"] == -3.0


def test_build_no_clipping_evidence_marks_clear_measurement():
    args = argparse.Namespace(
        no_clipping_threshold_dbtp=-3.0,
        max_true_peak_dbtp=-8.4,
        no_clipping_sample_count=24,
    )
    payload = build_no_clipping_evidence(args)
    assert payload["measured"] is True
    assert payload["clear"] is True
    assert payload["max_true_peak_dbtp"] == -8.4
    assert payload["sample_count"] == 24


def test_build_no_clipping_evidence_marks_peak_over_threshold_as_not_clear():
    args = argparse.Namespace(
        no_clipping_threshold_dbtp=-3.0,
        max_true_peak_dbtp=-1.2,
        no_clipping_sample_count=24,
    )
    payload = build_no_clipping_evidence(args)
    assert payload["measured"] is True
    assert payload["clear"] is False
    assert payload["max_true_peak_dbtp"] == -1.2


def test_build_run_summary_includes_key_live_test_results():
    args = argparse.Namespace(
        channel=9,
        fader_target_db=-30.0,
        gain_target_db=1.5,
        cooldown_fader_target_db=-30.5,
        no_clipping_threshold_dbtp=-3.0,
        max_true_peak_dbtp=-8.4,
        no_clipping_sample_count=24,
    )
    summary = build_run_summary(
        args=args,
        session_label="tub350_supervised_live_20260514_210000",
        connect_payload={
            "connected": True,
            "mode": "wing",
            "ip": "192.168.1.102",
            "observation_only": False,
            "telemetry_capture": {"active": True},
        },
        supervision_before={"armed": False},
        supervision_after_arm={"armed": True},
        session_before_start={"active": False},
        session_after_start={"active": True, "step_count": 0},
        steps=[
            {"step_label": "negative_control", "request": {}, "response": {}},
            {"step_label": "unsupported_control", "request": {}, "response": {}},
            {"step_label": "positive_fader_1", "request": {}, "response": {}},
        ],
        negative_control={"type": "manual_write_blocked", "reason": "approval_required", "request": {"channel": 9}},
        unsupported_control={"type": "manual_write_blocked", "reason": "unsupported_value_type", "request": {"channel": 9, "value_type": "eq"}},
        unsupported_compressor={"type": "manual_write_blocked", "reason": "unsupported_value_type", "request": {"channel": 9, "value_type": "compressor"}},
        positive_fader_1={"type": "manual_write_applied", "request": {"channel": 9, "effective_value": -30.0}},
        positive_gain_1={"type": "manual_write_applied", "request": {"channel": 9, "effective_value": 1.5}},
        positive_fader_2={"type": "manual_write_applied", "request": {"channel": 9, "effective_value": -30.25}},
        cooldown_check={"type": "manual_write_blocked", "reason": "cooldown", "request": {"channel": 9}},
        post_cooldown_check={"type": "manual_write_applied", "status": "applied", "request": {"channel": 9, "effective_value": -30.5}},
        rollback_last={
            "type": "manual_write_rollback_applied",
            "status": "applied",
            "reason": "rollback_applied",
            "mode": "rollback_last",
            "rolled_back_count": 1,
            "remaining_step_count": 3,
            "rolled_back_steps": [
                {
                    "original_step": {"channel": 9, "value_type": "fader", "old_value": -30.25},
                    "rollback_result": {"channel": 9, "value_type": "fader", "readback_value": -30.25},
                }
            ],
        },
        rollback_all={
            "type": "manual_write_rollback_applied",
            "status": "applied",
            "reason": "rollback_applied",
            "mode": "rollback_all",
            "rolled_back_count": 3,
            "remaining_step_count": 0,
            "rolled_back_steps": [
                {
                    "original_step": {"channel": 9, "value_type": "fader", "old_value": -35.0},
                    "rollback_result": {"channel": 9, "value_type": "fader", "readback_value": -35.0},
                }
            ],
        },
        session_after_rollback_last={"active": True, "step_count": 3},
        session_after_rollback_all={"active": False, "step_count": 0},
        emergency_stop={"type": "emergency_stop_applied"},
        blocked_after_stop={"type": "manual_write_blocked", "reason": "manual_path_not_armed", "request": {"channel": 9}},
        blocked_after_emergency_mid={"type": "manual_write_blocked", "reason": "manual_path_not_armed", "request": {"channel": 9}},
        telemetry_stop={"telemetry_capture": {"events_path": "/tmp/events.jsonl", "metadata_path": "/tmp/metadata.json"}},
        transcript_path=Path("artifacts/tub345_supervised_live/run.json"),
    )

    assert summary["issue"] == "TUB-350"
    assert summary["runner"] == "TUB-350"
    assert summary["safe_channel"] == 9
    assert summary["manual_write_session"]["after_start"]["active"] is True
    assert summary["results"]["negative_control"]["reason"] == "approval_required"
    assert summary["results"]["unsupported_control"]["reason"] == "unsupported_value_type"
    assert summary["results"]["unsupported_compressor"]["reason"] == "unsupported_value_type"
    assert summary["results"]["post_cooldown_check"]["status"] == "applied"
    assert summary["results"]["rollback_last"]["readback_value"] == -30.25
    assert summary["results"]["rollback_all"]["readback_value"] == -35.0
    assert summary["results"]["cooldown_check"]["reason"] == "cooldown"
    assert summary["results"]["blocked_after_stop"]["reason"] == "manual_path_not_armed"
    assert summary["results"]["blocked_after_emergency_mid"]["reason"] == "manual_path_not_armed"
    assert summary["telemetry_capture_stop"]["events_path"] == "/tmp/events.jsonl"
    assert summary["no_clipping"]["measured"] is True
    assert summary["no_clipping"]["clear"] is True
    assert summary["checks"]["no_clipping_evidence"]["max_true_peak_dbtp"] == -8.4
    assert len(summary["sequence_steps"]) == 3
