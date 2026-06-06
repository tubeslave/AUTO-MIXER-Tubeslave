import json
from pathlib import Path

from backend.wing_readonly_preflight_review_decision import (
    REQUIRED_REVIEW_CHECKLIST_IDS,
    build_wing_readonly_preflight_review_decision_ledger,
)
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
        "operator_channel_confirmation_required": True,
        "operator_confirmed_channels": [3],
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
    assert report["manual_review_required_before_readiness_update"] is True
    assert report["manual_review_decision_required"] is True
    assert report["review_decision_template_available"] is True
    assert report["review_decision_template_required_fields"] == [
        "review_id",
        "artifact_source_id",
        "run_id",
        "summary_path",
        "events_path",
        "decision",
        "reviewed_by",
        "reviewed_at",
        "checklist_ids",
    ]
    assert [row["id"] for row in report["manual_review_checklist"]] == [
        "confirm_artifact_paths",
        "confirm_operator_channel_scope",
        "confirm_physical_readbacks",
        "confirm_event_log_readonly_sequence",
        "confirm_no_live_authority",
    ]
    assert report["artifact_sources"] == [str(summary_path)]
    assert report["artifact_reviews"] == [
        {
            "source_id": str(summary_path),
            "status": "valid_readonly_preflight_artifact",
            "run_id": "20260606T120000Z_wing_readonly_preflight",
            "summary_path": str(summary_path),
            "events_path": str(summary_path.parent / "events.jsonl"),
            "channels": [3],
            "operator_confirmed_channels": [3],
            "value_types": ["fader", "gain"],
            "readback_count": 2,
            "physical_readback_count": 2,
            "violations": [],
        }
    ]
    assert review_tool.exit_code_for_report(report) == 0
    markdown = review_tool.render_markdown(report)
    assert "candidate_ready_for_manual_review" in markdown
    assert f"summary_path: `{summary_path}`" in markdown
    assert f"events_path: `{summary_path.parent / 'events.jsonl'}`" in markdown
    assert "operator_confirmed_channels: `[3]`" in markdown
    assert "Manual Review Gate" in markdown
    assert "confirm_event_log_readonly_sequence" in markdown
    assert "manual_review_decision_required: `true`" in markdown
    assert "review_decision_template_available: `true`" in markdown
    assert "Review Decision Template" in markdown
    assert "does not contact WING" in markdown


def test_review_decision_template_links_clean_candidate_but_requires_operator_edit(tmp_path: Path):
    summary_path = _write_artifact(tmp_path / "preflight" / "run")
    report = review_tool.build_review_report(artifact_roots=(tmp_path / "preflight",))

    template = review_tool.build_review_decision_template(
        report,
        generated_at="2026-06-07T02:00:00Z",
    )

    assert template["template_status"] == (
        "operator_must_replace_placeholders_before_ledger_acceptance"
    )
    assert template["template_generated_at"] == "2026-06-07T02:00:00Z"
    assert template["review_id"] == "review-wing-readonly-20260606T120000Z_wing_readonly_preflight"
    assert template["artifact_source_id"] == str(summary_path)
    assert template["run_id"] == "20260606T120000Z_wing_readonly_preflight"
    assert template["summary_path"] == str(summary_path)
    assert template["events_path"] == str(summary_path.parent / "events.jsonl")
    assert template["decision"] == "pending_operator_decision"
    assert template["reviewed_by"] == ""
    assert template["reviewed_at"] == ""
    assert template["checklist_ids"] == list(REQUIRED_REVIEW_CHECKLIST_IDS)
    assert template["live_authority_granted"] is False
    assert template["readiness_matrix_update_allowed_by_decision"] is False

    ledger = build_wing_readonly_preflight_review_decision_ledger(
        preflight_evidence=report["evidence"],
        decision_records=[template],
    )

    assert ledger["status"] == "operator_review_decision_replacement_required"
    assert ledger["summary"]["accepted_record_count"] == 0
    assert ledger["summary"]["manual_readiness_update_candidate_available"] is False
    violations = set(ledger["records"][0]["violations"])
    assert "review_decision_unknown" in violations
    assert "review_decision_reviewer_missing" in violations
    assert "review_decision_reviewed_at_missing" in violations


def test_review_decision_template_requires_explicit_source_for_multiple_candidates(tmp_path: Path):
    first_summary = _write_artifact(tmp_path / "preflight" / "run1")
    second_summary = _write_artifact(tmp_path / "preflight" / "run2")
    report = review_tool.build_review_report(artifact_roots=(tmp_path / "preflight",))

    try:
        review_tool.build_review_decision_template(report)
    except ValueError as exc:
        assert str(exc) == "decision_template_requires_single_candidate_or_source_id"
    else:  # pragma: no cover - defensive assertion
        raise AssertionError("expected explicit source id requirement")

    template = review_tool.build_review_decision_template(
        report,
        artifact_source_id=str(second_summary),
        generated_at="2026-06-07T02:00:00Z",
    )

    assert template["artifact_source_id"] == str(second_summary)
    assert template["summary_path"] == str(second_summary)
    assert template["artifact_source_id"] != str(first_summary)


def test_review_report_blocks_missing_artifacts(tmp_path: Path):
    report = review_tool.build_review_report(artifact_roots=(tmp_path / "missing",))

    assert report["status"] == "blocked"
    assert report["artifact_count"] == 0
    assert report["manual_review_required_before_readiness_update"] is True
    assert report["manual_review_decision_required"] is False
    assert report["review_decision_template_available"] is False
    assert "operator_must_run_wing_readonly_preflight_runner" in report["blocked_by"]
    assert review_tool.exit_code_for_report(report) == 2


def test_review_report_blocks_tampered_event_log(tmp_path: Path):
    _write_artifact(tmp_path / "preflight" / "run", tamper_events=True)

    report = review_tool.build_review_report(artifact_roots=(tmp_path / "preflight",))

    assert report["status"] == "blocked"
    assert report["artifact_count"] == 1
    assert report["invalid_candidate_count"] == 1
    assert report["artifact_reviews"][0]["operator_confirmed_channels"] == [3]
    assert "event_log_forbidden_request_type_sent" in report["violations"]
    assert review_tool.exit_code_for_report(report) == 3


def test_review_tool_main_writes_output_and_returns_gate_status(tmp_path: Path):
    _write_artifact(tmp_path / "preflight" / "run")
    output = tmp_path / "review.md"
    template_output = tmp_path / "decision-template.json"

    exit_code = review_tool.main(
        [
            "--artifact-root",
            str(tmp_path / "preflight"),
            "--output",
            str(output),
            "--decision-template-output",
            str(template_output),
            "--quiet",
        ]
    )

    assert exit_code == 0
    assert "WING Read-Only Preflight Artifact Review" in output.read_text(encoding="utf-8")
    template = json.loads(template_output.read_text(encoding="utf-8"))
    assert template["decision"] == "pending_operator_decision"
    assert template["reviewed_by"] == ""


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
