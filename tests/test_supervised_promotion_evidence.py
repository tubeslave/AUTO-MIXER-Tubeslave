from pathlib import Path

from backend.supervised_promotion_evidence import build_supervised_promotion_evidence_snapshot


def proof_report(*rows, invalid_summary_count=0):
    return {
        "type": "supervised_proof_artifact_report",
        "status": "ok" if invalid_summary_count == 0 else "blocked",
        "summary": {
            "invalid_summary_count": invalid_summary_count,
        },
        "rows": list(rows),
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


def blocked_artifact(proof_id: str, value_type: str) -> dict:
    return {
        "proof_id": proof_id,
        "value_type": value_type,
        "proof_mode": "physical",
        "status": "blocked",
        "promotion_allowed": False,
        "autonomous_ready": False,
        "blockers": ["rollback_readback_missing"],
    }


def test_supervised_promotion_evidence_is_read_only_and_transport_free():
    source = Path("backend/supervised_promotion_evidence.py").read_text(encoding="utf-8")

    forbidden = [
        "WingClient",
        "EnhancedOSCClient",
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


def test_dry_run_only_gain_evidence_cannot_update_safety_row():
    snapshot = build_supervised_promotion_evidence_snapshot(
        proof_report(dry_run_ready("dry-gain-proof", "gain"))
    )
    rows = {row["value_type"]: row for row in snapshot["rows"]}
    gain = rows["gain"]

    assert snapshot["safety"]["live_widening_allowed"] is False
    assert snapshot["summary"]["autonomous_ready_row_ids"] == []
    assert gain["status"] == "dry_run_only"
    assert gain["safety_row_id"] == "gain_supervised"
    assert gain["current_safety_status"] == "hardware_proven_narrow"
    assert gain["current_live_allowed"] is False
    assert gain["safety_row_review_candidate"] is False
    assert gain["automatic_safety_update_allowed"] is False
    assert "physical_candidate_missing" in gain["blockers"]


def test_physical_gain_candidate_allows_manual_safety_review_but_not_live_widening():
    snapshot = build_supervised_promotion_evidence_snapshot(
        proof_report(physical_candidate("physical-gain-proof", "gain"))
    )
    rows = {row["value_type"]: row for row in snapshot["rows"]}
    gain = rows["gain"]

    assert gain["status"] == "safety_review_candidate"
    assert gain["physical_candidate_count"] == 1
    assert gain["physical_candidate_proof_ids"] == ["physical-gain-proof"]
    assert gain["safety_row_review_candidate"] is True
    assert gain["automatic_safety_update_allowed"] is False
    assert gain["autonomous_ready"] is False
    assert gain["safety_required_evidence_components"] == [
        "physical_pre_read",
        "bounded_gain_down_apply",
        "apply_readback",
        "rollback_readback",
        "emergency_stop_block",
        "unsafe_gain_up_followup_blocked",
        "gain_up_operator_authored_opt_in",
    ]
    assert snapshot["safety"]["live_widening_allowed"] is False
    assert snapshot["summary"]["safety_row_review_candidate_ids"] == ["gain_supervised_evidence"]


def test_blocked_artifact_for_same_value_type_blocks_review_candidate():
    snapshot = build_supervised_promotion_evidence_snapshot(
        proof_report(
            physical_candidate("physical-gain-proof", "gain"),
            blocked_artifact("blocked-gain-proof", "gain"),
        )
    )
    rows = {row["value_type"]: row for row in snapshot["rows"]}
    gain = rows["gain"]

    assert gain["status"] == "evidence_blocked"
    assert gain["safety_row_review_candidate"] is False
    assert "blocked_artifacts_for_value_type" in gain["blockers"]
    assert gain["blocked_proof_ids"] == ["blocked-gain-proof"]


def test_invalid_summary_blocks_all_safety_review_candidates():
    snapshot = build_supervised_promotion_evidence_snapshot(
        proof_report(physical_candidate("physical-fader-proof", "fader"), invalid_summary_count=1)
    )

    for row in snapshot["rows"]:
        assert row["status"] == "evidence_blocked"
        assert row["safety_row_review_candidate"] is False
        assert "invalid_proof_summaries_present" in row["blockers"]


def test_current_artifact_shape_has_no_autonomous_or_automatic_update_path():
    report = proof_report(
        dry_run_ready("dry-gain-proof", "gain"),
        dry_run_ready("dry-fader-proof", "fader"),
    )

    snapshot = build_supervised_promotion_evidence_snapshot(report)

    assert snapshot["type"] == "supervised_promotion_evidence"
    assert snapshot["safety"] == {
        "read_only": True,
        "live_write": False,
        "live_widening_allowed": False,
        "automatic_safety_update_allowed": False,
        "full_autonomous_ready": False,
        "supervised_only": True,
    }
    assert snapshot["summary"]["safety_row_review_candidate_ids"] == []
    assert set(snapshot["summary"]["blocked_evidence_row_ids"]) == {
        "fader_supervised_evidence",
        "gain_supervised_evidence",
    }
    assert [row["value_type"] for row in snapshot["rows"]] == ["fader", "gain"]


def test_pan_artifacts_do_not_enter_current_wing_promotion_evidence():
    snapshot = build_supervised_promotion_evidence_snapshot(
        proof_report(
            physical_candidate("physical-pan-proof", "pan"),
            dry_run_ready("dry-gain-proof", "gain"),
        )
    )

    rows = {row["value_type"]: row for row in snapshot["rows"]}
    assert set(rows) == {"fader", "gain"}
    assert snapshot["summary"]["row_count"] == 2
    assert snapshot["summary"]["safety_row_review_candidate_ids"] == []
    assert set(snapshot["summary"]["blocked_evidence_row_ids"]) == {
        "fader_supervised_evidence",
        "gain_supervised_evidence",
    }
    assert "pan_supervised_evidence" not in str(snapshot)
