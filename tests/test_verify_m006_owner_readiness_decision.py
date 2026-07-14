"""Offline tests for the M006 owner readiness decision verifier."""

from __future__ import annotations

import copy
import importlib.util
import json
from pathlib import Path

import pytest


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
M005_EVENTS_REL = (
    "artifacts/wing_timeout_disconnect_proof/"
    "unit_m005_disconnect_timeout_stale_proof/events.jsonl"
)
M005_MIGRATION_SUMMARY_REL = (
    "artifacts/wing_m005_current_release_migration/"
    "unit-migration/summary.json"
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
        "m005_events": M005_EVENTS_REL,
        "proof_artifact_summaries": list(packet["input_proof_artifacts"]),
        "owner_review_input_ready": True,
        "m005_m006_input_accepted": True,
        "m005_acceptance_route": verifier.M005_ROUTE_NATIVE,
        "m005_behavioral_proof_complete": True,
        "m005_owner_review_shape_ready": True,
        "m005_provenance_ready": True,
        "m005_current_release_accepted": True,
        "m005_migration_summary": None,
        "m005_migration_ready": False,
        "m005_native_failed_checks": [],
        "m005_migration_failed_checks": [],
        "m005_migration_checks": {},
        "live_authority_granted": False,
        "readiness_closure_allowed": False,
        "readiness_matrix_update_allowed": False,
        "contacts_wing": False,
        "contacts_backend": False,
        "live_write": False,
        "checks": {
            "unit_ready": True,
            "m005_provenance_unit_binding": True,
            "m005_events_stale_readback_payload_present": True,
            "m005_events_stale_readback_matches_summary": True,
            "m005_events_stale_readback_target_matches": True,
            "m005_events_stale_readback_unavailable": True,
            "m005_events_reconnect_readback_payload_present": True,
            "m005_events_reconnect_readback_matches_summary": True,
            "m005_events_reconnect_readback_physical": True,
            "m005_events_reconnect_readback_target_matches": True,
        },
        "failed_checks": [],
    }


def ready_migration_packet_verifier_report() -> dict:
    report = ready_packet_verifier_report()
    report.update(
        {
            "reason": verifier.EXPECTED_PACKET_VERIFIER_MIGRATION_REASON,
            "m005_acceptance_route": verifier.M005_ROUTE_MIGRATION,
            "m005_owner_review_shape_ready": False,
            "m005_provenance_ready": False,
            "m005_current_release_accepted": False,
            "m005_migration_summary": M005_MIGRATION_SUMMARY_REL,
            "m005_summary": verifier.EXPECTED_HISTORICAL_SUMMARY_LOGICAL_PATH,
            "m005_events": verifier.EXPECTED_HISTORICAL_EVENTS_LOGICAL_PATH,
            "m005_migration_ready": True,
            "m005_native_failed_checks": ["m005_provenance_schema_matches"],
            "m005_migration_failed_checks": [],
            "m005_migration_checks": {"migration_unit_ready": True},
            "checks": {"packet_unit_ready": True, "migration_unit_ready": True},
        }
    )
    return report


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


def verify(
    decision: dict,
    *,
    packet_verifier: dict | None = None,
    migration_summary_path: str | None = None,
    m005_summary_path: str = M005_SUMMARY_REL,
) -> dict:
    return verifier.verify_decision(
        decision,
        packet=load_packet(),
        packet_verifier=packet_verifier,
        decision_summary_path=DECISION_TEMPLATE_REL,
        packet_summary_path=PACKET_SUMMARY_REL,
        packet_verifier_summary_path=PACKET_VERIFIER_SUMMARY_REL if packet_verifier else None,
        m005_summary_path=m005_summary_path,
        m005_migration_summary_path=migration_summary_path,
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


def test_migration_decision_requires_and_accepts_explicit_route_binding() -> None:
    decision = accepted_decision()
    decision["m005_summary"] = verifier.EXPECTED_HISTORICAL_SUMMARY_LOGICAL_PATH
    decision["m005_acceptance_route"] = verifier.M005_ROUTE_MIGRATION
    decision["m005_migration_summary"] = M005_MIGRATION_SUMMARY_REL

    result = verify(
        decision,
        packet_verifier=ready_migration_packet_verifier_report(),
        migration_summary_path=M005_MIGRATION_SUMMARY_REL,
        m005_summary_path=verifier.EXPECTED_HISTORICAL_SUMMARY_LOGICAL_PATH,
    )

    assert result["status"] == "ready_for_readiness_matrix_update"
    assert result["m005_acceptance_route"] == verifier.M005_ROUTE_MIGRATION
    assert result["m005_m006_input_accepted"] is True
    assert result["m005_migration_summary"] == M005_MIGRATION_SUMMARY_REL
    assert result["failed_checks"] == []


def test_migration_decision_rejects_noncanonical_legacy_summary() -> None:
    decision = accepted_decision()
    decision["m005_acceptance_route"] = verifier.M005_ROUTE_MIGRATION
    decision["m005_migration_summary"] = M005_MIGRATION_SUMMARY_REL

    result = verify(
        decision,
        packet_verifier=ready_migration_packet_verifier_report(),
        migration_summary_path=M005_MIGRATION_SUMMARY_REL,
    )

    assert result["status"] == "blocked"
    assert (
        "packet_verifier_migration_legacy_paths_exact"
        in result["failed_checks"]
    )
    assert (
        "decision_m005_migration_legacy_summary_exact"
        in result["failed_checks"]
    )


@pytest.mark.parametrize(
    "mutation",
    [
        lambda decision: decision.pop("m005_acceptance_route", None),
        lambda decision: decision.pop("m005_migration_summary", None),
        lambda decision: decision.update(
            {"m005_migration_summary": "artifacts/other/summary.json"}
        ),
    ],
)
def test_migration_decision_missing_or_mismatched_binding_fails_closed(
    mutation,
) -> None:
    decision = accepted_decision()
    decision["m005_summary"] = verifier.EXPECTED_HISTORICAL_SUMMARY_LOGICAL_PATH
    decision["m005_acceptance_route"] = verifier.M005_ROUTE_MIGRATION
    decision["m005_migration_summary"] = M005_MIGRATION_SUMMARY_REL
    mutation(decision)

    result = verify(
        decision,
        packet_verifier=ready_migration_packet_verifier_report(),
        migration_summary_path=M005_MIGRATION_SUMMARY_REL,
        m005_summary_path=verifier.EXPECTED_HISTORICAL_SUMMARY_LOGICAL_PATH,
    )

    assert result["status"] == "blocked"
    assert result["owner_decision_accepted"] is False


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


def test_path_verifier_recomputes_packet_evidence_instead_of_trusting_saved_ready_report(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    decision_path = PACKET_SUMMARY.parent / "unit_owner_decision_recompute.json"
    packet_verifier_path = (
        PROJECT_ROOT
        / "artifacts"
        / "wing_owner_readiness_review_verifier"
        / "unit_m006_owner_readiness_packet_verifier"
        / "summary.json"
    )
    packet_verifier_path.parent.mkdir(parents=True, exist_ok=True)
    decision_path.write_text(json.dumps(accepted_decision()), encoding="utf-8")
    packet_verifier_path.write_text(
        json.dumps(ready_packet_verifier_report()),
        encoding="utf-8",
    )
    recomputed = ready_packet_verifier_report()
    recomputed.update({
        "status": "blocked",
        "reason": "m005_current_release_provenance_missing_or_mismatch",
        "owner_review_input_ready": False,
        "m005_provenance_ready": False,
        "m005_current_release_accepted": False,
        "failed_checks": ["m005_provenance_source_clean_before_run"],
    })
    monkeypatch.setattr(verifier, "verify_packet_paths", lambda *_args, **_kwargs: recomputed)
    try:
        result = verifier.verify_decision_paths(
            decision_path,
            PACKET_SUMMARY,
            packet_verifier_path,
            Path(M005_SUMMARY_REL),
        )
    finally:
        decision_path.unlink(missing_ok=True)
        packet_verifier_path.unlink(missing_ok=True)
        try:
            packet_verifier_path.parent.rmdir()
        except OSError:
            pass

    assert result["status"] == "blocked"
    assert result["packet_verifier_recomputed_from_current_artifacts"] is True
    assert result["saved_packet_verifier_status"] == "ready_for_owner_review"
    assert "packet_verifier_status_ready" in result["failed_checks"]
    assert "packet_verifier_m005_provenance_ready" in result["failed_checks"]
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


def test_packet_verifier_must_include_m005_reconnect_event_checks() -> None:
    packet_verifier = ready_packet_verifier_report()
    packet_verifier["checks"].pop("m005_events_reconnect_readback_matches_summary")

    result = verify(accepted_decision(), packet_verifier=packet_verifier)

    assert result["status"] == "blocked"
    assert result["reason"] == "m006_owner_decision_failed_closed"
    assert (
        "packet_verifier_m005_events_reconnect_readback_matches_summary"
        in result["failed_checks"]
    )
    assert result["owner_decision_accepted"] is False
    assert result["readiness_matrix_update_candidate_ready"] is False
    assert result["live_authority_granted"] is False
    assert result["readiness_matrix_update_allowed"] is False


def test_packet_verifier_m005_events_path_must_match_m005_summary() -> None:
    packet_verifier = ready_packet_verifier_report()
    packet_verifier["m005_events"] = (
        "artifacts/wing_timeout_disconnect_proof/"
        "different_m005_disconnect_timeout_stale_proof/events.jsonl"
    )

    result = verify(accepted_decision(), packet_verifier=packet_verifier)

    assert result["status"] == "blocked"
    assert result["reason"] == "m006_owner_decision_failed_closed"
    assert "packet_verifier_m005_events_matches_summary_sibling" in result["failed_checks"]
    assert "packet_verifier_m005_events_path_repo_relative" not in result["failed_checks"]
    assert result["owner_decision_accepted"] is False
    assert result["readiness_matrix_update_candidate_ready"] is False
    assert result["live_authority_granted"] is False
    assert result["readiness_matrix_update_allowed"] is False


def test_packet_verifier_m005_events_path_required() -> None:
    packet_verifier = ready_packet_verifier_report()
    packet_verifier.pop("m005_events")

    result = verify(accepted_decision(), packet_verifier=packet_verifier)

    assert result["status"] == "blocked"
    assert result["reason"] == "m006_owner_decision_failed_closed"
    assert "packet_verifier_m005_events_path_repo_relative" in result["failed_checks"]
    assert "packet_verifier_m005_events_matches_summary_sibling" in result["failed_checks"]
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
