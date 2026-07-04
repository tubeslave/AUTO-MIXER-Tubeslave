"""Offline tests for the M006 owner readiness packet verifier."""

from __future__ import annotations

import copy
import importlib.util
import json
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
TOOL_PATH = PROJECT_ROOT / "tools" / "verify_m006_owner_readiness_packet.py"
PACKET_SUMMARY = (
    PROJECT_ROOT
    / "artifacts"
    / "wing_owner_readiness_review"
    / "20260704T191101Z_m006_owner_readiness_review_packet"
    / "summary.json"
)
PARTIAL_M005_SUMMARY = (
    PROJECT_ROOT
    / "artifacts"
    / "wing_timeout_disconnect_proof"
    / "20260704T074537Z_wing_m005_disconnect_timeout_stale_proof"
    / "summary.json"
)

SPEC = importlib.util.spec_from_file_location("verify_m006_owner_readiness_packet", TOOL_PATH)
assert SPEC is not None
verifier = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(verifier)


def load_packet() -> dict:
    return json.loads(PACKET_SUMMARY.read_text(encoding="utf-8"))


def load_packet_proofs(packet: dict | None = None) -> dict:
    return verifier.load_packet_proof_summaries(packet or load_packet())


def complete_m005_summary() -> dict:
    return {
        "proof_artifact_schema": "wing_timeout_disconnect_proof.v1",
        "status": "passed",
        "disconnect_mode": "physical-cable-loss",
        "live_authority_granted": False,
        "readiness_closure_allowed": False,
        "readiness_matrix_update_allowed": False,
        "direct_live_write_request_sent": False,
        "forbidden_request_types_sent": [],
        "required_m005_evidence": list(verifier.EXPECTED_M005_EVIDENCE),
        "completed_m005_evidence": list(verifier.EXPECTED_M005_EVIDENCE),
        "missing_m005_evidence": [],
        "remaining_m005_evidence": [],
        "phases": {
            "disconnect": {
                "status": "operator_confirmed_physical_cable_loss",
            },
            "stale_readback": {
                "status": "passed",
            },
            "blocked_write_while_disconnected": {
                "status": "passed",
            },
            "reconnect_recovery": {
                "status": "passed",
                "unchanged_from_baseline": True,
            },
        },
    }


def test_packet_without_m005_summary_fails_closed() -> None:
    result = verifier.verify_packet(load_packet())

    assert result["status"] == "blocked"
    assert result["reason"] == "m005_summary_required_for_owner_review"
    assert result["owner_review_input_ready"] is False
    assert result["live_authority_granted"] is False
    assert result["readiness_closure_allowed"] is False
    assert result["readiness_matrix_update_allowed"] is False
    assert result["contacts_wing"] is False
    assert result["contacts_backend"] is False
    assert result["live_write"] is False
    assert "m005_summary_supplied" in result["failed_checks"]


def test_partial_m005_summary_keeps_owner_review_blocked() -> None:
    partial = copy.deepcopy(complete_m005_summary())
    partial["m005_complete"] = False
    partial["completed_m005_evidence"] = ["physical_baseline_readback"]
    partial["missing_m005_evidence"] = [
        "operator_confirmed_physical_cable_loss",
        "stale_readback_unavailable",
        "blocked_write_while_disconnected",
        "reconnect_recovery_verified",
    ]
    partial["remaining_m005_evidence"] = [
        "operator-confirmed physical cable-loss phase still requires supervised WING cable pull"
    ]

    result = verifier.verify_packet(load_packet(), m005_summary=partial)

    assert result["status"] == "blocked"
    assert result["reason"] == "m006_owner_review_inputs_failed"
    assert result["owner_review_input_ready"] is False
    assert "m005_summary_complete_true" in result["failed_checks"]
    assert "m005_summary_completed_evidence_matches" in result["failed_checks"]
    assert "m005_summary_missing_evidence_empty" in result["failed_checks"]
    assert "m005_summary_remaining_evidence_empty" in result["failed_checks"]
    assert result["live_authority_granted"] is False
    assert result["readiness_matrix_update_allowed"] is False


def test_current_partial_m005_artifact_keeps_owner_review_blocked() -> None:
    packet = load_packet()
    partial_m005 = json.loads(PARTIAL_M005_SUMMARY.read_text(encoding="utf-8"))

    result = verifier.verify_packet(
        packet,
        m005_summary=partial_m005,
        proof_summaries_by_artifact=load_packet_proofs(packet),
        m005_summary_path=str(PARTIAL_M005_SUMMARY.relative_to(PROJECT_ROOT)),
    )

    assert result["status"] == "blocked"
    assert result["reason"] == "m006_owner_review_inputs_failed"
    assert result["owner_review_input_ready"] is False
    assert result["m005_summary"] == (
        "artifacts/wing_timeout_disconnect_proof/"
        "20260704T074537Z_wing_m005_disconnect_timeout_stale_proof/summary.json"
    )
    assert "m005_summary_schema_matches" in result["failed_checks"]
    assert "m005_summary_status_passed" in result["failed_checks"]
    assert "m005_summary_complete_true" in result["failed_checks"]
    assert "m005_summary_missing_evidence_empty" in result["failed_checks"]
    assert "m005_disconnect_phase_operator_confirmed_physical_loss" in result["failed_checks"]
    assert result["live_authority_granted"] is False
    assert result["readiness_closure_allowed"] is False
    assert result["readiness_matrix_update_allowed"] is False


def test_complete_m005_summary_allows_owner_review_input_only() -> None:
    m005 = complete_m005_summary()
    m005["m005_complete"] = True
    packet = load_packet()

    result = verifier.verify_packet(
        packet,
        m005_summary=m005,
        proof_summaries_by_artifact=load_packet_proofs(packet),
    )

    assert result["status"] == "ready_for_owner_review"
    assert result["reason"] == "m006_owner_review_inputs_verified"
    assert result["owner_review_input_ready"] is True
    assert result["failed_checks"] == []
    assert all(result["checks"].values())
    assert result["proof_artifact_summaries"] == packet["input_proof_artifacts"]
    assert result["live_authority_granted"] is False
    assert result["readiness_closure_allowed"] is False
    assert result["readiness_matrix_update_allowed"] is False
    assert result["contacts_wing"] is False
    assert result["contacts_backend"] is False
    assert result["live_write"] is False


def test_packet_attempting_to_grant_authority_fails_closed() -> None:
    packet = load_packet()
    packet["live_authority_granted"] = True
    m005 = complete_m005_summary()
    m005["m005_complete"] = True

    result = verifier.verify_packet(
        packet,
        m005_summary=m005,
        proof_summaries_by_artifact=load_packet_proofs(packet),
    )

    assert result["status"] == "blocked"
    assert result["reason"] == "m006_owner_review_inputs_failed"
    assert "packet_live_authority_granted_denied" in result["failed_checks"]
    assert result["live_authority_granted"] is False
    assert result["readiness_closure_allowed"] is False


def test_missing_referenced_proof_artifact_blocks_owner_review() -> None:
    packet = load_packet()
    packet["proofs"][0]["artifact"] = (
        "artifacts/wing_operator_proposal_validation/"
        "missing_wing_operator_proposal_ch3_fader/summary.json"
    )
    packet["input_proof_artifacts"][0] = packet["proofs"][0]["artifact"]
    m005 = complete_m005_summary()
    m005["m005_complete"] = True

    result = verifier.verify_packet(
        packet,
        m005_summary=m005,
        proof_summaries_by_artifact=load_packet_proofs(packet),
    )

    assert result["status"] == "blocked"
    assert result["reason"] == "m006_owner_review_inputs_failed"
    assert "fader_supervised_artifact_summary_loaded" in result["failed_checks"]
    assert result["owner_review_input_ready"] is False
    assert result["live_authority_granted"] is False
    assert result["readiness_matrix_update_allowed"] is False


def test_proof_artifact_summary_mismatch_blocks_owner_review() -> None:
    packet = load_packet()
    proofs = dict(load_packet_proofs(packet))
    first_artifact = packet["proofs"][0]["artifact"]
    proofs[first_artifact] = copy.deepcopy(proofs[first_artifact])
    proofs[first_artifact]["reason"] = "stale_or_mutated_summary"
    m005 = complete_m005_summary()
    m005["m005_complete"] = True

    result = verifier.verify_packet(
        packet,
        m005_summary=m005,
        proof_summaries_by_artifact=proofs,
    )

    assert result["status"] == "blocked"
    assert result["reason"] == "m006_owner_review_inputs_failed"
    assert "fader_supervised_artifact_reason_complete" in result["failed_checks"]
    assert result["owner_review_input_ready"] is False
    assert result["live_authority_granted"] is False
    assert result["readiness_closure_allowed"] is False


def test_write_artifact_preserves_blocked_authority_denial(tmp_path: Path) -> None:
    result = verifier.verify_packet(load_packet())

    output = verifier.write_verifier_artifact(
        result,
        artifact_root=tmp_path / "m006_verifier",
        run_id="unit_m006_verifier",
    )

    summary_path = Path(output["summary_path"])
    assert summary_path.exists()
    saved = json.loads(summary_path.read_text(encoding="utf-8"))
    assert saved["artifact_written"] is True
    assert saved["run_id"] == "unit_m006_verifier"
    assert saved["status"] == "blocked"
    assert saved["reason"] == "m005_summary_required_for_owner_review"
    assert saved["owner_review_input_ready"] is False
    assert saved["live_authority_granted"] is False
    assert saved["readiness_closure_allowed"] is False
    assert saved["readiness_matrix_update_allowed"] is False
    assert saved["contacts_wing"] is False
    assert saved["contacts_backend"] is False
    assert saved["live_write"] is False
