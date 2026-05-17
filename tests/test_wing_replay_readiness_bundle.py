import json

from wing_replay_readiness_bundle import build_wing_replay_readiness_bundle


def _write_json(path, payload):
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def _write_shadow_report_dir(base_dir, *, ready=True, live_mixer_writes=False, dry_run_only=True):
    report_dir = base_dir / "shadow"
    report_dir.mkdir()
    _write_json(
        report_dir / "shadow_mode_summary.json",
        {
            "generated_at": "2026-05-17T10:00:00+00:00",
            "manifest_id": "manifest::1",
            "selected_proposal_id": "proposal::selected",
        },
    )
    _write_json(
        report_dir / "shadow_mode_comparison.json",
        {
            "selected_proposal_id": "proposal::selected",
            "disagreements": [],
            "disagreement_summary": {"disagreement_rate": 0.0},
            "evidence": {
                "dry_run_only": dry_run_only,
                "live_mixer_writes": live_mixer_writes,
            },
        },
    )
    _write_json(
        report_dir / "promotion_readiness_report.json",
        {
            "ready": ready,
            "dry_run_only": dry_run_only,
            "live_mixer_writes": live_mixer_writes,
            "promotion_readiness_score": 91,
            "validation_highest_severity": "info",
            "promotion_readiness_gates": [{"key": "dry_run_only", "ok": dry_run_only}],
            "aggregation": {
                "disagreement_rate": 0.0,
                "audit_coverage": {"coverage_rate": 1.0, "complete": True},
                "rollback_frequency": {"blocked_event_count": 0},
                "confidence_trend": {"direction": "ai_above_operator"},
            },
        },
    )
    _write_json(report_dir / "rollback_simulation_report.json", {"blocked_reasons": []})
    _write_json(report_dir / "confidence_trajectory.json", {"trend_summary": {"direction": "ai_above_operator"}})
    return report_dir


def _write_telemetry_session(base_dir, name, *, observation_only=True, events=None):
    session_dir = base_dir / name
    session_dir.mkdir()
    _write_json(
        session_dir / "metadata.json",
        {
            "session_id": name,
            "session_label": name,
            "metadata": {"observation_only": observation_only},
        },
    )
    rows = events or []
    with (session_dir / "events.jsonl").open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row) + "\n")
    return session_dir


def _write_supervised_summary(base_dir, telemetry_dir, *, no_clipping=None):
    path = base_dir / "supervised_summary.json"
    _write_json(
        path,
        {
            "issue": "TUB-350",
            "runner": "TUB-350",
            "session_label": "tub350_live_safe",
            "safe_channel": 9,
            "connection_status": {
                "telemetry_capture": {
                    "events_path": str(telemetry_dir / "events.jsonl"),
                    "metadata_path": str(telemetry_dir / "metadata.json"),
                }
            },
            "manual_write_session": {"after_start": {"active": True}},
            "results": {
                "negative_control": {"reason": "approval_required"},
                "blocked_after_stop": {"reason": "manual_path_not_armed"},
                "emergency_stop": {"type": "emergency_stop_applied"},
            },
            "telemetry_capture_stop": {
                "events_path": str(telemetry_dir / "events.jsonl"),
                "metadata_path": str(telemetry_dir / "metadata.json"),
            },
            "no_clipping": no_clipping,
        },
    )
    return path


def test_bundle_marks_ready_when_shadow_and_telemetry_are_safe(tmp_path):
    report_dir = _write_shadow_report_dir(tmp_path)
    live_session = _write_telemetry_session(
        tmp_path,
        "live",
        events=[
            {"seq": 1, "timestamp": 1.0, "event_type": "query_sent", "address": "/ch/1/fdr"},
            {"seq": 2, "timestamp": 2.0, "event_type": "console_observed", "address": "/ch/1/fdr", "values": [0.5]},
            {"seq": 3, "timestamp": 3.0, "event_type": "blocked_write", "address": "/ch/1/fdr", "values": [0.7]},
        ],
    )
    replay_session = _write_telemetry_session(
        tmp_path,
        "replay",
        events=[
            {"seq": 1, "timestamp": 1.0, "event_type": "query_sent", "address": "/$meters/ch/1/in"},
            {"seq": 2, "timestamp": 2.0, "event_type": "console_observed", "address": "/$meters/ch/1/in", "values": [0.1]},
        ],
    )
    supervised_summary = _write_supervised_summary(tmp_path, live_session)

    result = build_wing_replay_readiness_bundle(
        output_dir=tmp_path / "bundle",
        shadow_report_dir=report_dir,
        live_telemetry_session=live_session,
        replay_telemetry_session=replay_session,
        supervised_summary_path=supervised_summary,
        generated_at="2026-05-17T10:30:00+00:00",
    )

    payload = json.loads(result["json"].read_text(encoding="utf-8"))
    assert payload["ready"] is True
    assert payload["blockers"] == []
    assert payload["telemetry"]["dangerous_write_detected"] is False
    assert payload["telemetry"]["blocked_write_total"] == 1
    assert payload["supervised"]["telemetry_artifacts_present"] is True
    assert payload["shadow"]["promotion_readiness_score"] == 91
    assert payload["no_clipping"]["measured"] is False
    assert payload["no_clipping"]["clear"] is False
    assert "No measured no-clipping evidence" in payload["no_clipping"]["reason"]
    assert payload["evidence"]["no_clipping_measured"] is False
    assert payload["evidence"]["no_clipping_clear"] is False


def test_bundle_fails_closed_when_observation_only_telemetry_contains_write_sent(tmp_path):
    report_dir = _write_shadow_report_dir(tmp_path)
    live_session = _write_telemetry_session(
        tmp_path,
        "live",
        events=[
            {"seq": 1, "timestamp": 1.0, "event_type": "query_sent", "address": "/ch/1/fdr"},
            {"seq": 2, "timestamp": 2.0, "event_type": "write_sent", "address": "/ch/1/fdr", "values": [0.7]},
        ],
    )
    supervised_summary = _write_supervised_summary(tmp_path, live_session)

    result = build_wing_replay_readiness_bundle(
        output_dir=tmp_path / "bundle",
        shadow_report_dir=report_dir,
        live_telemetry_session=live_session,
        supervised_summary_path=supervised_summary,
    )

    payload = json.loads(result["json"].read_text(encoding="utf-8"))
    assert payload["ready"] is False
    assert "observation_only_telemetry_contains_write_sent" in payload["blockers"]
    assert payload["telemetry"]["dangerous_write_detected"] is True
    assert payload["telemetry"]["dangerous_write_sessions"][0]["write_sent_count"] == 1


def test_bundle_surfaces_explicit_no_clipping_evidence_from_supervised_summary(tmp_path):
    report_dir = _write_shadow_report_dir(tmp_path)
    live_session = _write_telemetry_session(
        tmp_path,
        "live",
        events=[
            {"seq": 1, "timestamp": 1.0, "event_type": "console_observed", "address": "/$meters/ch/1/in", "values": [0.25]},
        ],
    )
    supervised_summary = _write_supervised_summary(
        tmp_path,
        live_session,
        no_clipping={
            "measured": True,
            "clear": True,
            "reason": "Peak telemetry stayed below the approved threshold.",
            "threshold_dbtp": -3.0,
            "max_true_peak_dbtp": -8.4,
            "sample_count": 24,
        },
    )

    result = build_wing_replay_readiness_bundle(
        output_dir=tmp_path / "bundle",
        shadow_report_dir=report_dir,
        live_telemetry_session=live_session,
        supervised_summary_path=supervised_summary,
    )

    payload = json.loads(result["json"].read_text(encoding="utf-8"))
    markdown = result["markdown"].read_text(encoding="utf-8")

    assert payload["no_clipping"]["measured"] is True
    assert payload["no_clipping"]["clear"] is True
    assert payload["no_clipping"]["source"] == "supervised_summary.no_clipping"
    assert payload["no_clipping"]["threshold_dbtp"] == -3.0
    assert payload["no_clipping"]["max_true_peak_dbtp"] == -8.4
    assert payload["evidence"]["no_clipping_measured"] is True
    assert payload["evidence"]["no_clipping_clear"] is True
    assert "## No-Clipping Evidence" in markdown
    assert "- measured: True" in markdown
