import json
from pathlib import Path

from tools import review_wing_readonly_preflight_artifacts as review_tool


def _valid_summary() -> dict:
    return {
        "run_id": "20260606T120000Z_wing_readonly_preflight",
        "status": "passed",
        "reason": "wing_readonly_preflight_passed",
        "proof_artifact_schema": "wing_readonly_preflight.v1",
        "proof_runner": "wing_readonly_preflight_runner",
        "connection_mode": "wing",
        "channels": [3],
        "value_types": ["fader", "gain"],
        "require_physical_readback": True,
        "read_only": True,
        "live_write": False,
        "write_request_sent": False,
        "proposal_request_sent": False,
        "sent_request_types": [
            "get_operator_mode",
            "get_state",
            "query_mixer_parameter",
            "query_mixer_parameter",
        ],
        "forbidden_request_types_sent": [],
        "readback_count": 2,
        "physical_readback_count": 2,
        "readbacks": [
            {
                "type": "mixer_parameter_value",
                "status": "verified_from_physical_readback",
                "source": "physical_readback",
                "channel": 3,
                "value_type": "fader",
                "address": "/ch/3/fdr",
                "value": -12.0,
            },
            {
                "type": "mixer_parameter_value",
                "status": "verified_from_physical_readback",
                "source": "physical_readback",
                "channel": 3,
                "value_type": "gain",
                "address": "/ch/3/in/set/trim",
                "value": -0.25,
            },
        ],
    }


def _event(seq: int, event_type: str, payload: dict) -> dict:
    return {
        "seq": seq,
        "timestamp": "2026-06-06T12:00:00.000Z",
        "event_type": event_type,
        "payload": payload,
    }


def _valid_events(summary: dict) -> list[dict]:
    events = [
        _event(1, "send", {"type": "get_operator_mode"}),
        _event(2, "recv", {"type": "operator_mode_status", "mode": "assist", "connection_mode": "wing"}),
        _event(3, "send", {"type": "get_state"}),
        _event(4, "recv", {"type": "state_update", "mode": "wing", "state": {"/ch/3/fdr": -12.0}}),
    ]
    seq = 5
    for row in summary["readbacks"]:
        events.append(
            _event(
                seq,
                "send",
                {
                    "type": "query_mixer_parameter",
                    "channel": row["channel"],
                    "value_type": row["value_type"],
                    "address": row["address"],
                },
            )
        )
        events.append(_event(seq + 1, "recv", row))
        seq += 2
    return events


def _write_artifact(run_dir: Path, *, tamper_events: bool = False) -> Path:
    run_dir.mkdir(parents=True)
    summary = _valid_summary()
    summary_path = run_dir / "summary.json"
    events_path = run_dir / "events.jsonl"
    summary["summary_path"] = str(summary_path)
    summary["events_path"] = str(events_path)
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    events = _valid_events(summary)
    if tamper_events:
        events.insert(2, _event(99, "send", {"type": "set_gain", "channel": 3, "value": 1.0}))
    events_path.write_text("".join(json.dumps(row) + "\n" for row in events), encoding="utf-8")
    return summary_path


def test_review_report_accepts_clean_artifact_without_live_authority(tmp_path: Path):
    summary_path = _write_artifact(tmp_path / "preflight" / "run")

    report = review_tool.build_review_report(artifact_roots=(tmp_path / "preflight",))

    assert report["status"] == "candidate_ready_for_manual_review"
    assert report["readiness_update_candidate_available"] is True
    assert report["execution_allowed_by_review"] is False
    assert report["live_authority_granted"] is False
    assert report["full_autonomous_ready"] is False
    assert report["artifact_sources"] == [str(summary_path)]
    assert review_tool.exit_code_for_report(report) == 0
    markdown = review_tool.render_markdown(report)
    assert "candidate_ready_for_manual_review" in markdown
    assert "does not contact WING" in markdown


def test_review_report_blocks_missing_artifacts(tmp_path: Path):
    report = review_tool.build_review_report(artifact_roots=(tmp_path / "missing",))

    assert report["status"] == "blocked"
    assert report["artifact_count"] == 0
    assert "operator_must_run_wing_readonly_preflight_runner" in report["blocked_by"]
    assert review_tool.exit_code_for_report(report) == 2


def test_review_report_blocks_tampered_event_log(tmp_path: Path):
    _write_artifact(tmp_path / "preflight" / "run", tamper_events=True)

    report = review_tool.build_review_report(artifact_roots=(tmp_path / "preflight",))

    assert report["status"] == "blocked"
    assert report["artifact_count"] == 1
    assert report["invalid_candidate_count"] == 1
    assert "event_log_forbidden_request_type_sent" in report["violations"]
    assert review_tool.exit_code_for_report(report) == 3


def test_review_tool_main_writes_output_and_returns_gate_status(tmp_path: Path):
    _write_artifact(tmp_path / "preflight" / "run")
    output = tmp_path / "review.md"

    exit_code = review_tool.main(
        [
            "--artifact-root",
            str(tmp_path / "preflight"),
            "--output",
            str(output),
            "--quiet",
        ]
    )

    assert exit_code == 0
    assert "WING Read-Only Preflight Artifact Review" in output.read_text(encoding="utf-8")


def test_review_tool_is_transport_free():
    source = Path("tools/review_wing_readonly_preflight_artifacts.py").read_text(encoding="utf-8")

    forbidden = [
        "WingClient",
        "EnhancedOSCClient",
        "websockets",
        "socket",
        ".send(",
        "mixer_client",
        "set_fader(",
        "set_gain(",
        "apply_operator_proposal(",
    ]
    for token in forbidden:
        assert token not in source
