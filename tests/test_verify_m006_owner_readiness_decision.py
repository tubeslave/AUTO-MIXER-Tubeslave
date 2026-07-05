"""Offline tests for the M006 owner readiness decision verifier."""

from __future__ import annotations

import copy
import importlib.util
import json
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
TOOL_PATH = PROJECT_ROOT / "tools" / "verify_m006_owner_readiness_decision.py"
PACKET_SUMMARY = (
    PROJECT_ROOT
    / "artifacts"
    / "wing_owner_readiness_review"
    / "20260704T191101Z_m006_owner_readiness_review_packet"
    / "summary.json"
)
DECISION_TEMPLATE = PACKET_SUMMARY.with_name("owner_decision_template.json")
PACKET_SUMMARY_REL = (
    "artifacts/wing_owner_readiness_review/"
    "20260704T191101Z_m006_owner_readiness_review_packet/summary.json"
)
DECISION_TEMPLATE_REL = (
    "artifacts/wing_owner_readiness_review/"
    "20260704T191101Z_m006_owner_readiness_review_packet/owner_decision_template.json"
)
PACKET_VERIFIER_SUMMARY_REL = (
    "artifacts/wing_owner_readiness_review_verifier/"
    "unit_m006_owner_readiness_packet_verifier/summary.json"
)
M005_SUMMARY_REL = (
    "artifacts/wing_timeout_disconnect_proof/"
    "unit_m005_disconnect_timeout_stale_proof/summary.json"
)

SPEC = importlib.util.spec_from_file_location("verify_m006_owner_readiness_decision", TOOL_PATH)
assert SPEC is not None
verifier = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(verifier)


def load_packet() -> dict:
    return json.loads(PACKET_SUMMARY.read_text(encoding="utf-8"))


def load_template() -> dict:
    return json.loads(DECISION_TEMPLATE.read_text(encoding="utf-8"))


def ready_packet_verifier_report() -> dict:
    packet = load_packet()
    return {
        "schema": "automixer.m006_owner_readiness_packet_verifier.v1",
        "status": "ready_for_owner_review",
        "reason": "m006_owner_review_inputs_verified",
        "packet_summary": PACKET_SUMMARY_REL,
        "m005_summary": M005_SUMMARY_REL,
        "proof_artifact_summaries": list(packet["input_proof_artifacts"]),
        "owner_review_input_ready": True,
        "live_authority_granted": False,
        "readiness_closure_allowed": False,
        "readiness_matrix_update_allowed": False,
        "contacts_wing": False,
        "contacts_backend": False,
        "live_write": False,
        "checks": {
            "unit_ready": True,
            "m005_events_stale_readback_payload_present": True,
            "m005_events_stale_readback_matches_summary": True,
            "m005_events_stale_readback_unavailable": True,
        },
        "failed_checks": [],
    }


def accepted_decision() -> dict:
    decision = load_template()
    decision.update(
        {
            "status": "accepted_for_readiness_matrix_update",
            "decision": "accepted",
            "reviewed_by": "owner",
            "reviewed_at_utc": "2026-07-04T23:35:00Z",
            "packet_verifier_summary": PACKET_VERIFIER_SUMMARY_REL,
            "m005_summary": M005_SUMMARY_REL,
            "accepted_safety_row_ids": list(verifier.EXPECTED_ALLOWED_ROW_IDS),
            "accepted_live_write_kinds": list(verifier.EXPECTED_ALLOWED_KINDS),
            "accepted_proof_artifacts": list(load_packet()["input_proof_artifacts"]),
            "readiness_matrix_update_candidate": True,
            "m005_acceptance": dict(verifier.EXPECTED_M005_ACCEPTANCE),
            "scope_guardrails": dict(verifier.SCOPE_GUARDRAILS),
        }
    )
    return decision


def verify(decision: dict, *, packet_verifier: dict | None = None) -> dict:
    return verifier.verify_decision(
        decision,
        packet=load_packet(),
        packet_verifier=packet_verifier,
        decision_summary_path=DECISION_TEMPLATE_REL,
        packet_summary_path=PACKET_SUMMARY_REL,
        packet_verifier_summary_path=PACKET_VERIFIER_SUMMARY_REL if packet_verifier else None,
        m005_summary_path=M005_SUMMARY_REL,
    )


def test_pending_owner_decision_template_fails_closed() -> None:
    result = verify(load_template())

    assert result["status"] == "blocked"
    assert result["reason"] == "m006_owner_decision_pending"
    assert result["owner_decision_accepted"] is False
    assert result["readiness_matrix_update_candidate_ready"] is False
    assert result["live_authority_granted"] is False
    assert result["readiness_closure_allowed"] is False
    assert result["readiness_matrix_update_allowed"] is False
    assert result["contacts_wing"] is False
    assert result["contacts_backend"] is False
    assert result["live_write"] is False
    assert "decision_is_accepted" in result["failed_checks"]
    assert "packet_verifier_summary_supplied" in result["failed_checks"]


def test_accepted_decision_allows_matrix_update_candidate_only() -> None:
    result = verify(accepted_decision(), packet_verifier=ready_packet_verifier_report())

    assert result["status"] == "ready_for_readiness_matrix_update"
    assert result["reason"] == "m006_owner_decision_accepts_narrow_readiness_rows"
    assert result["owner_decision_accepted"] is True
    assert result["readiness_matrix_update_candidate_ready"] is True
    assert result["accepted_safety_row_ids"] == ["fader_supervised", "gain_supervised"]
    assert result["accepted_live_write_kinds"] == ["fader", "gain"]
    assert result["failed_checks"] == []
    assert all(result["checks"].values())
    assert result["live_authority_granted"] is False
    assert result["readiness_closure_allowed"] is False
    assert result["readiness_matrix_update_allowed"] is False
    assert result["contacts_wing"] is False
    assert result["contacts_backend"] is False
    assert result["live_write"] is False


def test_rejected_owner_decision_fails_closed() -> None:
    decision = accepted_decision()
    decision["decision"] = "rejected"
    decision["status"] = "rejected"

    result = verify(decision, packet_verifier=ready_packet_verifier_report())

    assert result["status"] == "blocked"
    assert result["reason"] == "m006_owner_decision_rejected"
    assert result["owner_decision_accepted"] is False
    assert "decision_is_accepted" in result["failed_checks"]
    assert result["live_authority_granted"] is False
    assert result["readiness_matrix_update_allowed"] is False


def test_extra_forbidden_row_or_kind_blocks_decision() -> None:
    decision = accepted_decision()
    decision["accepted_safety_row_ids"].append("eq_autoeq")
    decision["accepted_live_write_kinds"].append("gain_up")

    result = verify(decision, packet_verifier=ready_packet_verifier_report())

    assert result["status"] == "blocked"
    assert result["reason"] == "m006_owner_decision_failed_closed"
    assert "decision_accepted_rows_match_allowed" in result["failed_checks"]
    assert "decision_no_forbidden_rows_accepted" in result["failed_checks"]
    assert "decision_accepted_kinds_match_allowed" in result["failed_checks"]
    assert "decision_no_forbidden_kinds_accepted" in result["failed_checks"]
    assert result["readiness_matrix_update_candidate_ready"] is False


def test_packet_verifier_must_be_ready_and_clean() -> None:
    packet_verifier = ready_packet_verifier_report()
    packet_verifier["status"] = "blocked"
    packet_verifier["failed_checks"] = ["m005_summary_missing"]
    packet_verifier["owner_review_input_ready"] = False

    result = verify(accepted_decision(), packet_verifier=packet_verifier)

    assert result["status"] == "blocked"
    assert "packet_verifier_status_ready" in result["failed_checks"]
    assert "packet_verifier_failed_checks_empty" in result["failed_checks"]
    assert "packet_verifier_owner_review_input_ready" in result["failed_checks"]
    assert result["owner_decision_accepted"] is False
    assert result["readiness_matrix_update_allowed"] is False


def test_packet_verifier_must_include_m005_stale_event_checks() -> None:
    packet_verifier = ready_packet_verifier_report()
    packet_verifier["checks"].pop("m005_events_stale_readback_matches_summary")

    result = verify(accepted_decision(), packet_verifier=packet_verifier)

    assert result["status"] == "blocked"
    assert result["reason"] == "m006_owner_decision_failed_closed"
    assert (
        "packet_verifier_m005_events_stale_readback_matches_summary"
        in result["failed_checks"]
    )
    assert result["owner_decision_accepted"] is False
    assert result["readiness_matrix_update_candidate_ready"] is False
    assert result["live_authority_granted"] is False
    assert result["readiness_matrix_update_allowed"] is False


def test_m005_path_mismatch_blocks_decision_binding() -> None:
    decision = accepted_decision()
    decision["m005_summary"] = (
        "artifacts/wing_timeout_disconnect_proof/"
        "different_m005_disconnect_timeout_stale_proof/summary.json"
    )

    result = verify(decision, packet_verifier=ready_packet_verifier_report())

    assert result["status"] == "blocked"
    assert "decision_m005_summary_matches" in result["failed_checks"]
    assert result["readiness_matrix_update_candidate_ready"] is False


def test_decision_authority_claims_block_even_with_ready_packet_verifier() -> None:
    decision = accepted_decision()
    decision["live_authority_granted"] = True
    decision["readiness_matrix_update_allowed"] = True
    decision["live_write"] = True
    decision["scope_guardrails"]["direct_write_allowed"] = True

    result = verify(decision, packet_verifier=ready_packet_verifier_report())

    assert result["status"] == "blocked"
    assert "decision_live_authority_granted_denied" in result["failed_checks"]
    assert "decision_readiness_matrix_update_allowed_denied" in result["failed_checks"]
    assert "decision_live_write_denied" in result["failed_checks"]
    assert "decision_scope_guardrail_direct_write_allowed_denied" in result["failed_checks"]


def test_write_artifact_preserves_denied_authority(tmp_path: Path) -> None:
    blocked = verify(load_template())

    saved = verifier.write_verifier_artifact(
        blocked,
        artifact_root=tmp_path / "m006_owner_decision_verifier",
        run_id="unit_owner_decision",
    )

    saved_path = Path(saved["summary_path"])
    assert saved_path.is_file()
    loaded = json.loads(saved_path.read_text(encoding="utf-8"))
    assert loaded["run_id"] == "unit_owner_decision"
    assert loaded["status"] == "blocked"
    assert loaded["live_authority_granted"] is False
    assert loaded["readiness_matrix_update_allowed"] is False
