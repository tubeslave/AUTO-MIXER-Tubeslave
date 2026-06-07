import hashlib
import json
from pathlib import Path

from backend.wing_readonly_preflight_review_decision import REQUIRED_REVIEW_CHECKLIST_IDS
from tools import review_wing_readonly_preflight_decisions as decision_tool


def _valid_summary() -> dict:
    return {
        "run_id": "20260606T120000Z_wing_readonly_preflight",
        "status": "passed",
        "reason": "wing_readonly_preflight_passed",
        "proof_artifact_schema": "wing_readonly_preflight.v1",
        "proof_runner": "wing_readonly_preflight_runner",
        "connection_mode": "wing",
        "channels": [3],
        "operator_channel_confirmation_required": True,
        "operator_confirmed_channels": [3],
        "value_types": ["fader", "gain"],
        "require_physical_readback": True,
        "request_id_policy": "required_for_every_query_mixer_parameter_response",
        "expected_plan_path": "artifacts/wing_readonly_preflight_plan/latest_plan.json",
        "expected_plan_sha256": "0" * 64,
        "expected_plan_match": True,
        "expected_plan_violations": [],
        "expected_plan_required_for_real_run": True,
        "physical_preflight_requires_existing_backend_connection": True,
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
                "request_id": "wing-readonly-preflight:ch3:fader",
                "status": "verified_from_physical_readback",
                "source": "physical_readback",
                "channel": 3,
                "value_type": "fader",
                "address": "/ch/3/fdr",
                "value": -12.0,
            },
            {
                "type": "mixer_parameter_value",
                "request_id": "wing-readonly-preflight:ch3:gain",
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
                    "request_id": row["request_id"],
                    "channel": row["channel"],
                    "value_type": row["value_type"],
                    "address": row["address"],
                },
            )
        )
        events.append(_event(seq + 1, "recv", row))
        seq += 2
    return events


def _expected_plan_request_sequence(summary: dict) -> list[dict]:
    return [
        {"type": "get_operator_mode"},
        {"type": "get_state"},
        *[
            {
                "type": "query_mixer_parameter",
                "request_id": row["request_id"],
                "channel": row["channel"],
                "value_type": row["value_type"],
                "address": row["address"],
            }
            for row in summary["readbacks"]
        ],
    ]


def _valid_expected_plan(summary: dict) -> dict:
    return {
        "type": "wing_readonly_preflight_offline_plan",
        "status": "ready_for_operator_review",
        "proof_artifact_schema": summary["proof_artifact_schema"],
        "proof_runner": summary["proof_runner"],
        "channels": summary["channels"],
        "operator_channel_confirmation_required": summary[
            "operator_channel_confirmation_required"
        ],
        "operator_confirmed_channels": summary["operator_confirmed_channels"],
        "value_types": summary["value_types"],
        "require_physical_readback": summary["require_physical_readback"],
        "read_only": summary["read_only"],
        "live_write": summary["live_write"],
        "expected_plan_required_for_real_run": summary[
            "expected_plan_required_for_real_run"
        ],
        "physical_preflight_requires_existing_backend_connection": summary[
            "physical_preflight_requires_existing_backend_connection"
        ],
        "manual_review_required_before_readiness_update": True,
        "contacts_backend": False,
        "queries_console": False,
        "execution_allowed_by_plan": False,
        "live_authority_granted": False,
        "request_id_policy": summary["request_id_policy"],
        "request_sequence": _expected_plan_request_sequence(summary),
    }


def _write_artifact(run_dir: Path) -> Path:
    run_dir.mkdir(parents=True)
    summary = _valid_summary()
    summary_path = run_dir / "summary.json"
    events_path = run_dir / "events.jsonl"
    plan_path = run_dir / "expected_plan.json"
    summary["summary_path"] = str(summary_path)
    summary["events_path"] = str(events_path)
    plan_bytes = json.dumps(_valid_expected_plan(summary), sort_keys=True).encode("utf-8")
    plan_path.write_bytes(plan_bytes)
    summary["expected_plan_path"] = str(plan_path)
    summary["expected_plan_sha256"] = hashlib.sha256(plan_bytes).hexdigest()
    summary_path.write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    events_path.write_text(
        "".join(json.dumps(row) + "\n" for row in _valid_events(summary)),
        encoding="utf-8",
    )
    return summary_path


def _decision_record(summary_path: Path, **overrides) -> dict:
    record = {
        "review_id": "review-wing-readonly-1",
        "artifact_source_id": str(summary_path),
        "run_id": "20260606T120000Z_wing_readonly_preflight",
        "summary_path": str(summary_path),
        "events_path": str(summary_path.parent / "events.jsonl"),
        "decision": "accepted",
        "reviewed_by": "operator",
        "reviewed_at": "2026-06-07T01:05:00Z",
        "checklist_ids": list(REQUIRED_REVIEW_CHECKLIST_IDS),
    }
    record.update(overrides)
    return record


def _write_decision(path: Path, payload: dict) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path


def test_decision_review_report_accepts_operator_decision_without_live_authority(tmp_path: Path):
    summary_path = _write_artifact(tmp_path / "preflight" / "run")
    decision_path = _write_decision(
        tmp_path / "decisions" / "accepted.json",
        _decision_record(summary_path),
    )

    report = decision_tool.build_decision_review_report(
        artifact_roots=(tmp_path / "preflight",),
        decision_paths=(decision_path,),
    )

    assert report["type"] == "wing_readonly_preflight_review_decision_report"
    assert report["status"] == "accepted_decision_ready"
    assert report["ledger_status"] == "operator_review_decision_records_available"
    assert report["read_only"] is True
    assert report["live_write"] is False
    assert report["queries_console"] is False
    assert report["executes_tests"] is False
    assert report["execution_allowed_by_decision"] is False
    assert report["readiness_matrix_update_allowed_by_decision"] is False
    assert report["live_authority_granted"] is False
    assert report["full_autonomous_ready"] is False
    assert report["accepted_record_count"] == 1
    assert report["manual_readiness_update_candidate_available"] is True
    assert report["accepted_source_ids"] == [str(summary_path)]
    assert decision_tool.exit_code_for_report(report) == 0
    markdown = decision_tool.render_markdown(report)
    assert "accepted_decision_ready" in markdown
    assert "live_authority_granted: `false`" in markdown
    assert str(summary_path) in markdown
    assert "does not contact WING" in markdown


def test_decision_review_report_rejects_pending_template(tmp_path: Path):
    summary_path = _write_artifact(tmp_path / "preflight" / "run")
    template_path = _write_decision(
        tmp_path / "decisions" / "pending_template.json",
        _decision_record(
            summary_path,
            decision="pending_operator_decision",
            reviewed_by="",
            reviewed_at="",
        ),
    )

    report = decision_tool.build_decision_review_report(
        artifact_roots=(tmp_path / "preflight",),
        decision_paths=(template_path,),
    )

    assert report["status"] == "blocked"
    assert report["invalid_record_count"] == 1
    assert report["manual_readiness_update_candidate_available"] is False
    assert decision_tool.exit_code_for_report(report) == 3
    record = report["ledger"]["records"][0]
    assert "review_decision_unknown" in record["violations"]
    assert "review_decision_reviewer_missing" in record["violations"]
    assert "review_decision_reviewed_at_missing" in record["violations"]


def test_decision_review_report_treats_rejection_as_nonready(tmp_path: Path):
    summary_path = _write_artifact(tmp_path / "preflight" / "run")
    decision_path = _write_decision(
        tmp_path / "decisions" / "rejected.json",
        _decision_record(summary_path, decision="rejected"),
    )

    report = decision_tool.build_decision_review_report(
        artifact_roots=(tmp_path / "preflight",),
        decision_paths=(decision_path,),
    )

    assert report["status"] == "candidate_rejected"
    assert report["accepted_record_count"] == 0
    assert report["rejected_record_count"] == 1
    assert report["manual_readiness_update_candidate_available"] is False
    assert decision_tool.exit_code_for_report(report) == 4


def test_decision_review_report_requires_preflight_candidate_first(tmp_path: Path):
    report = decision_tool.build_decision_review_report(
        artifact_roots=(tmp_path / "missing",),
        decision_roots=(tmp_path / "decisions",),
    )

    assert report["status"] == "blocked"
    assert report["review_candidate_count"] == 0
    assert decision_tool.exit_code_for_report(report) == 2


def test_decision_review_tool_main_writes_output(tmp_path: Path):
    summary_path = _write_artifact(tmp_path / "preflight" / "run")
    decision_path = _write_decision(
        tmp_path / "decisions" / "accepted.json",
        _decision_record(summary_path),
    )
    output = tmp_path / "decision-review.md"

    exit_code = decision_tool.main(
        [
            "--artifact-root",
            str(tmp_path / "preflight"),
            "--decision-path",
            str(decision_path),
            "--output",
            str(output),
            "--quiet",
        ]
    )

    assert exit_code == 0
    assert "WING Read-Only Preflight Decision Review" in output.read_text(encoding="utf-8")


def test_decision_review_tool_is_transport_free():
    source = Path("tools/review_wing_readonly_preflight_decisions.py").read_text(
        encoding="utf-8"
    )

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
