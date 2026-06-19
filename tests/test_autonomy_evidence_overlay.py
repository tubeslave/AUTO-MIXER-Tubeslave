from pathlib import Path

from backend.autonomy_evidence_overlay import build_autonomy_evidence_overlay


def proof_report(*rows):
    return {
        "type": "supervised_proof_artifact_report",
        "summary": {"invalid_summary_count": 0},
        "rows": list(rows),
        "invalid_summaries": [],
    }


def physical_candidate(proof_id: str, value_type: str) -> dict:
    return {
        "proof_id": proof_id,
        "value_type": value_type,
        "proof_mode": "physical",
        "status": "supervised_candidate",
        "promotion_allowed": True,
        "autonomous_ready": False,
        "blockers": [],
    }


def dry_run_ready(proof_id: str, value_type: str) -> dict:
    return {
        "proof_id": proof_id,
        "value_type": value_type,
        "proof_mode": "dry_run",
        "status": "dry_run_ready",
        "promotion_allowed": False,
        "autonomous_ready": False,
        "blockers": ["physical_proof_required_for_promotion"],
    }


def legacy_blocked(proof_id: str, value_type: str) -> dict:
    return {
        "proof_id": proof_id,
        "value_type": value_type,
        "channel": 4,
        "proof_mode": "legacy_ambiguous",
        "status": "blocked",
        "promotion_allowed": False,
        "autonomous_ready": False,
        "artifact_path": "artifacts/wing_telemetry/legacy/summary.json",
        "blockers": [
            "legacy_physical_schema_missing",
            "physical_proof_required_for_promotion",
        ],
        "evidence": {"delta_db": -0.25},
    }


def test_autonomy_evidence_overlay_module_is_transport_free():
    source = Path("backend/autonomy_evidence_overlay.py").read_text(encoding="utf-8")

    forbidden = [
        "WingClient",
        "EnhancedOSCClient",
        "websockets",
        "socket",
        ".send(",
        "mixer_client",
        "set_fader",
        "set_gain",
        "apply_operator_proposal",
        "requested_change",
        "requested_changes",
    ]
    for token in forbidden:
        assert token not in source


def test_overlay_maps_physical_fader_candidate_and_legacy_gain_remediation_to_targets():
    overlay = build_autonomy_evidence_overlay(
        proof_report(
            physical_candidate("physical-fader-proof", "fader"),
            legacy_blocked("legacy-gain-proof", "gain"),
        )
    )
    rows = {row["value_type"]: row for row in overlay["rows"]}

    assert overlay["type"] == "autonomy_evidence_overlay"
    assert overlay["status"] == "manual_remediation_required"
    assert overlay["safety"] == {
        "read_only": True,
        "live_write": False,
        "live_widening_allowed": False,
        "automatic_safety_update_allowed": False,
        "full_autonomous_ready": False,
        "operator_required": True,
    }
    assert overlay["summary"]["autonomous_ready_target_ids"] == []
    assert overlay["summary"]["manual_safety_review_candidate_target_ids"] == [
        "fader_supervised_promotion"
    ]
    assert overlay["summary"]["remediation_required_target_ids"] == [
        "gain_trim_supervised_promotion"
    ]
    assert overlay["summary"]["no_current_evidence_target_ids"] == []
    assert "pan" not in rows

    fader = rows["fader"]
    assert fader["status"] == "manual_safety_review_candidate"
    assert fader["target_id"] == "fader_supervised_promotion"
    assert fader["safety_review_candidate"] is True
    assert fader["supervised_live_allowed_static"] is True
    assert fader["physical_candidate_proof_ids"] == ["physical-fader-proof"]
    assert fader["autonomous_ready"] is False
    assert "automatic_promotion_forbidden" in fader["blockers"]

    gain = rows["gain"]
    assert gain["status"] == "remediation_required"
    assert gain["target_id"] == "gain_trim_supervised_promotion"
    assert gain["blocked_proof_ids"] == ["legacy-gain-proof"]
    assert gain["remediation_action_count"] == 1
    assert gain["remediation_action_ids"] == ["rerun_legacy-gain-proof"]
    assert gain["autonomous_ready"] is False


def test_overlay_reports_dry_run_only_and_no_current_evidence_without_promotion():
    overlay = build_autonomy_evidence_overlay(proof_report(dry_run_ready("dry-gain-proof", "gain")))
    rows = {row["value_type"]: row for row in overlay["rows"]}

    assert overlay["status"] == "dry_run_only"
    assert overlay["summary"]["dry_run_only_target_ids"] == ["gain_trim_supervised_promotion"]
    assert overlay["summary"]["no_current_evidence_target_ids"] == [
        "fader_supervised_promotion",
    ]
    assert rows["gain"]["status"] == "dry_run_only"
    assert rows["gain"]["dry_run_ready_proof_ids"] == ["dry-gain-proof"]
    assert rows["fader"]["status"] == "no_current_evidence"
    assert "proof_artifact_missing" in rows["fader"]["blockers"]
    assert "pan" not in rows
