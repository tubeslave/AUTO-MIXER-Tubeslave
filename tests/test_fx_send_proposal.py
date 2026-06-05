from pathlib import Path

from backend.fx_send_proposal import (
    FXSendAnalysisInput,
    analyze_fx_send_need,
    build_fx_send_proposals,
    classify_source_family,
    fx_parameter_schema_from_snapshot,
    tempo_delay_ms,
)
from backend.processing_proposal_adapters import build_proposal_only_method_contracts
from backend.processing_registry import get_pipeline_profile, get_processing_method
from backend.project_goal_capability_matrix import get_project_goal_capability


def test_fx_send_proposal_module_is_transport_free():
    source = Path("backend/fx_send_proposal.py").read_text()

    forbidden = [
        "WingClient",
        "EnhancedOSCClient",
        ".send(",
        "set_fx",
        "set_channel_send",
        "set_send_level",
        "mixer_client",
    ]
    for token in forbidden:
        assert token not in source


def test_lead_vocal_creates_proposal_only_fx_send_recommendation():
    recommendation = analyze_fx_send_need(
        FXSendAnalysisInput(
            channel_id=6,
            channel_label="Lead Vox",
            source_role="lead_vocal",
            current_send_db=-28.0,
            tempo_bpm=120.0,
            arrangement_role="lead_space",
            contribution_share=0.85,
            return_headroom_db=6.0,
            feedback_risk=0.1,
            intelligibility_risk=0.2,
            source_confidence=0.92,
            persistence_cycles=3,
            current_state_cache_freshness={
                "status": "fresh",
                "field_count": 6,
                "fresh_field_count": 6,
                "read_only": True,
                "live_write": False,
                "queries_console": False,
            },
            current_fx_parameter_schema={
                "status": "mapped",
                "schema_family": "reverb",
                "schema_source": "automixer_generic_fx_parameter_hints",
                "mapped_parameter_count": 6,
                "generic_parameter_count": 0,
                "physical_model_schema_verified": False,
                "live_write_authority": False,
            },
        )
    )

    assert recommendation.status == "proposal_only"
    assert recommendation.action_type == "fx_send_proposal"
    assert recommendation.source_family == "lead_vocal"
    assert recommendation.fx_type == "reverb"
    assert -36.0 <= recommendation.target_send_db <= -12.0
    assert recommendation.delta_send_db > 0
    assert recommendation.return_level_db <= -3.0
    assert recommendation.ducking_db > 0.0
    assert recommendation.blockers == ()
    assert recommendation.evidence["transport_free"] is True
    assert recommendation.evidence["current_state_cache_freshness_status"] == "fresh"
    assert recommendation.evidence["current_state_cache_freshness"]["fresh_field_count"] == 6
    assert recommendation.evidence["current_fx_parameter_schema_status"] == "mapped"
    assert recommendation.evidence["current_fx_parameter_schema"]["schema_family"] == "reverb"
    assert recommendation.evidence["expected_fx_schema_family"] == "reverb"
    assert recommendation.evidence["source_identity_downstream_trust"]["status"] == "legacy_unverified"

    proposal = recommendation.to_operator_proposal()
    assert proposal["method_id"] == "fx_send_proposal"
    assert proposal["can_import_to_queue"] is False
    assert proposal["can_apply_live"] is False
    assert "requested_change" not in proposal
    assert "unsupported_value_type:fx_send" in proposal["blockers"]
    assert proposal["evidence"]["live_write"] is False
    assert proposal["evidence"]["current_state_cache_freshness_status"] == "fresh"
    assert proposal["evidence"]["current_fx_parameter_schema_status"] == "mapped"


def test_fx_send_can_derive_schema_evidence_from_fx_inserts_snapshot():
    snapshot = {
        "type": "fx_inserts_snapshot",
        "safety": {
            "queries_console": False,
            "live_write": False,
        },
        "summary": {
            "read_coverage_status": "partial",
            "cache_freshness_status": "fresh",
        },
        "cache_freshness": {"status": "fresh"},
        "parameter_schema": {
            "status": "mapped",
            "mapped_parameter_count": 6,
            "generic_parameter_count": 0,
            "physical_model_schema_verified": False,
            "live_write_authority": False,
        },
        "fx_slots": [
            {
                "slot": "FX1",
                "model": "PLATE",
                "parameter_schema": {
                    "status": "mapped",
                    "schema_family": "reverb",
                    "schema_source": "automixer_generic_fx_parameter_hints",
                    "mapped_parameter_count": 6,
                    "generic_parameter_count": 0,
                    "cached_parameter_count": 6,
                    "physical_model_schema_verified": False,
                    "live_write_authority": False,
                },
            },
        ],
    }

    schema = fx_parameter_schema_from_snapshot(snapshot, "reverb")
    recommendation = analyze_fx_send_need(
        FXSendAnalysisInput(
            channel_id=6,
            channel_label="Lead Vox",
            source_role="lead_vocal",
            current_send_db=-28.0,
            arrangement_role="lead_space",
            contribution_share=0.85,
            return_headroom_db=6.0,
            feedback_risk=0.1,
            intelligibility_risk=0.2,
            source_confidence=0.92,
            persistence_cycles=3,
            current_state_cache_freshness={"status": "fresh"},
            current_fx_inserts_snapshot=snapshot,
        )
    )

    assert schema["status"] == "mapped"
    assert schema["schema_family"] == "reverb"
    assert schema["fx_slot"] == "FX1"
    assert recommendation.status == "proposal_only"
    assert recommendation.evidence["current_fx_parameter_schema"]["fx_slot"] == "FX1"
    assert recommendation.evidence["current_fx_parameter_schema"]["fx_model"] == "PLATE"
    assert recommendation.evidence["current_fx_parameter_schema"]["snapshot_cache_freshness_status"] == "fresh"
    assert recommendation.evidence["current_fx_parameter_schema"]["snapshot_queries_console"] is False
    assert recommendation.evidence["current_fx_parameter_schema"]["snapshot_live_write"] is False


def test_requested_tempo_delay_uses_tempo_and_remains_non_applyable():
    recommendation = analyze_fx_send_need(
        FXSendAnalysisInput(
            channel_id=6,
            source_role="lead_vocal",
            current_send_db=-34.0,
            tempo_bpm=100.0,
            requested_fx_type="delay",
            return_headroom_db=7.0,
            feedback_risk=0.1,
            intelligibility_risk=0.15,
            source_confidence=0.9,
            persistence_cycles=3,
        )
    )

    assert recommendation.status == "proposal_only"
    assert recommendation.fx_type == "delay"
    assert recommendation.delay_ms == 450.0
    assert recommendation.to_operator_proposal()["can_apply_live"] is False


def test_dry_anchor_source_is_analysis_only():
    recommendation = analyze_fx_send_need(
        FXSendAnalysisInput(
            channel_id=1,
            source_role="kick",
            current_send_db=-36.0,
            return_headroom_db=8.0,
            feedback_risk=0.1,
            intelligibility_risk=0.1,
        )
    )

    assert recommendation.status == "analysis_only"
    assert "dry_anchor_source" in recommendation.blockers


def test_feedback_or_headroom_blocks_fx_send_proposal():
    recommendation = analyze_fx_send_need(
        FXSendAnalysisInput(
            channel_id=6,
            source_role="lead_vocal",
            current_send_db=-30.0,
            return_headroom_db=1.0,
            feedback_risk=0.7,
            intelligibility_risk=0.2,
            source_confidence=0.9,
            persistence_cycles=3,
        )
    )

    assert recommendation.status == "analysis_only"
    assert "insufficient_fx_return_headroom" in recommendation.blockers
    assert "feedback_risk_high" in recommendation.blockers


def test_stale_current_fx_state_blocks_proposal_only_recommendation():
    recommendation = analyze_fx_send_need(
        FXSendAnalysisInput(
            channel_id=6,
            channel_label="Lead Vox",
            source_role="lead_vocal",
            current_send_db=-28.0,
            tempo_bpm=120.0,
            arrangement_role="lead_space",
            contribution_share=0.85,
            return_headroom_db=6.0,
            feedback_risk=0.1,
            intelligibility_risk=0.2,
            source_confidence=0.92,
            persistence_cycles=3,
            current_state_cache_freshness={
                "status": "stale",
                "field_count": 6,
                "fresh_field_count": 3,
                "stale_field_count": 3,
                "stale_fields": ["send_db", "return_level_db", "fx_slot_type"],
                "read_only": True,
                "live_write": False,
                "queries_console": False,
            },
        )
    )

    assert recommendation.status == "analysis_only"
    assert "fx_current_state_cache_not_fresh" in recommendation.blockers
    assert recommendation.evidence["current_state_cache_freshness_status"] == "stale"
    assert recommendation.evidence["current_state_cache_freshness"]["stale_fields"] == [
        "send_db",
        "return_level_db",
        "fx_slot_type",
    ]
    assert recommendation.to_operator_proposal()["can_apply_live"] is False


def test_generic_fx_parameter_schema_blocks_fx_send_proposal():
    recommendation = analyze_fx_send_need(
        FXSendAnalysisInput(
            channel_id=6,
            channel_label="Lead Vox",
            source_role="lead_vocal",
            current_send_db=-28.0,
            tempo_bpm=120.0,
            arrangement_role="lead_space",
            contribution_share=0.85,
            return_headroom_db=6.0,
            feedback_risk=0.1,
            intelligibility_risk=0.2,
            source_confidence=0.92,
            persistence_cycles=3,
            current_state_cache_freshness={"status": "fresh"},
            current_fx_parameter_schema={
                "status": "partial",
                "schema_family": "unknown",
                "schema_source": "automixer_generic_fx_parameter_hints",
                "mapped_parameter_count": 0,
                "generic_parameter_count": 1,
                "physical_model_schema_verified": False,
                "live_write_authority": False,
            },
        )
    )

    assert recommendation.status == "analysis_only"
    assert "fx_parameter_schema_not_mapped" in recommendation.blockers
    assert recommendation.severity == "medium"
    assert recommendation.evidence["current_fx_parameter_schema_status"] == "partial"
    assert recommendation.evidence["current_fx_parameter_schema"]["generic_parameter_count"] == 1
    assert recommendation.to_operator_proposal()["can_apply_live"] is False


def test_generic_fx_inserts_snapshot_schema_blocks_fx_send_proposal():
    snapshot = {
        "type": "fx_inserts_snapshot",
        "safety": {
            "queries_console": False,
            "live_write": False,
        },
        "summary": {
            "read_coverage_status": "partial",
            "cache_freshness_status": "fresh",
        },
        "cache_freshness": {"status": "fresh"},
        "parameter_schema": {
            "status": "partial",
            "mapped_parameter_count": 0,
            "generic_parameter_count": 1,
            "physical_model_schema_verified": False,
            "live_write_authority": False,
        },
        "fx_slots": [
            {
                "slot": "FX1",
                "model": "CUSTOM-FX",
                "parameter_schema": {
                    "status": "partial",
                    "schema_family": "unknown",
                    "mapped_parameter_count": 0,
                    "generic_parameter_count": 1,
                    "cached_parameter_count": 1,
                    "physical_model_schema_verified": False,
                    "live_write_authority": False,
                },
            },
        ],
    }

    recommendation = analyze_fx_send_need(
        FXSendAnalysisInput(
            channel_id=6,
            channel_label="Lead Vox",
            source_role="lead_vocal",
            current_send_db=-28.0,
            arrangement_role="lead_space",
            contribution_share=0.85,
            return_headroom_db=6.0,
            feedback_risk=0.1,
            intelligibility_risk=0.2,
            source_confidence=0.92,
            persistence_cycles=3,
            current_state_cache_freshness={"status": "fresh"},
            current_fx_inserts_snapshot=snapshot,
        )
    )

    assert recommendation.status == "analysis_only"
    assert "fx_parameter_schema_not_mapped" in recommendation.blockers
    assert recommendation.evidence["current_fx_parameter_schema_status"] == "partial"
    assert recommendation.evidence["current_fx_parameter_schema"]["generic_parameter_count"] == 1
    assert recommendation.evidence["current_fx_parameter_schema"]["snapshot_cache_freshness_status"] == "fresh"
    assert recommendation.to_operator_proposal()["can_apply_live"] is False


def test_mismatched_fx_parameter_schema_family_blocks_fx_send_proposal():
    recommendation = analyze_fx_send_need(
        FXSendAnalysisInput(
            channel_id=6,
            channel_label="Lead Vox",
            source_role="lead_vocal",
            current_send_db=-28.0,
            tempo_bpm=120.0,
            requested_fx_type="delay",
            return_headroom_db=6.0,
            feedback_risk=0.1,
            intelligibility_risk=0.2,
            source_confidence=0.92,
            persistence_cycles=3,
            current_state_cache_freshness={"status": "fresh"},
            current_fx_parameter_schema={
                "status": "mapped",
                "schema_family": "reverb",
                "mapped_parameter_count": 6,
                "generic_parameter_count": 0,
                "physical_model_schema_verified": False,
                "live_write_authority": False,
            },
        )
    )

    assert recommendation.status == "analysis_only"
    assert recommendation.fx_type == "delay"
    assert "fx_parameter_schema_family_mismatch" in recommendation.blockers
    assert recommendation.evidence["expected_fx_schema_family"] == "delay"
    assert recommendation.evidence["current_fx_parameter_schema"]["schema_family"] == "reverb"


def test_source_identity_hold_blocks_downstream_fx_send_proposal():
    recommendation = analyze_fx_send_need(
        FXSendAnalysisInput(
            channel_id=6,
            channel_label="Lead Vox",
            source_role="lead_vocal",
            current_send_db=-28.0,
            tempo_bpm=120.0,
            arrangement_role="lead_space",
            contribution_share=0.85,
            return_headroom_db=6.0,
            feedback_risk=0.1,
            intelligibility_risk=0.2,
            source_confidence=0.92,
            persistence_cycles=3,
            source_identity_trust_state="hold_for_operator_review",
            source_identity_trust_confidence=0.41,
            source_identity_replay_outcome_count=4,
            source_identity_calibration_freshness={
                "status": "fresh",
                "field_count": 4,
                "fresh_field_count": 4,
                "read_only": True,
                "live_write": False,
                "queries_console": False,
            },
        )
    )

    assert recommendation.status == "analysis_only"
    assert "source_identity_not_trusted_for_downstream" in recommendation.blockers
    assert recommendation.severity == "medium"
    trust = recommendation.evidence["source_identity_downstream_trust"]
    assert trust["status"] == "held_for_operator_review"
    assert trust["usable_for_downstream"] is False
    assert trust["evidence"]["queue_mutation"] is False
    assert trust["evidence"]["model_mutation"] is False
    assert recommendation.to_operator_proposal()["can_apply_live"] is False


def test_stale_source_identity_replay_context_blocks_downstream_fx_send_proposal():
    recommendation = analyze_fx_send_need(
        FXSendAnalysisInput(
            channel_id=6,
            channel_label="Lead Vox",
            source_role="lead_vocal",
            current_send_db=-28.0,
            tempo_bpm=120.0,
            arrangement_role="lead_space",
            contribution_share=0.85,
            return_headroom_db=6.0,
            feedback_risk=0.1,
            intelligibility_risk=0.2,
            source_confidence=0.92,
            persistence_cycles=3,
            source_identity_trust_state="trusted_for_downstream_priority",
            source_identity_trust_confidence=0.9,
            source_identity_replay_outcome_count=5,
            source_identity_calibration_freshness={
                "status": "stale",
                "field_count": 4,
                "fresh_field_count": 2,
                "stale_field_count": 2,
                "stale_fields": ["accepted_labels", "replay_outcomes"],
                "read_only": True,
                "live_write": False,
                "queries_console": False,
            },
        )
    )

    assert recommendation.status == "analysis_only"
    assert "source_identity_replay_context_not_fresh" in recommendation.blockers
    assert recommendation.severity == "high"
    trust = recommendation.evidence["source_identity_downstream_trust"]
    assert trust["status"] == "trusted"
    assert trust["usable_for_downstream"] is False
    assert trust["evidence"]["current_state_cache_freshness"]["stale_fields"] == [
        "accepted_labels",
        "replay_outcomes",
    ]
    assert recommendation.to_operator_proposal()["can_apply_live"] is False


def test_fx_helpers_classify_roles_and_tempo_notes():
    assert classify_source_family("Lead Vocal") == "lead_vocal"
    assert classify_source_family("Snare Top") == "snare"
    assert classify_source_family("Bass DI") == "bass"
    assert tempo_delay_ms(100.0, "dotted_eighth") == 450.0
    assert tempo_delay_ms(120.0, "quarter") == 500.0


def test_build_fx_send_proposals_filters_to_proposal_only_items():
    proposals = build_fx_send_proposals(
        [
            FXSendAnalysisInput(
                channel_id=6,
                source_role="lead_vocal",
                current_send_db=-30.0,
                return_headroom_db=6.0,
                feedback_risk=0.1,
                intelligibility_risk=0.2,
                source_confidence=0.9,
                persistence_cycles=3,
            ),
            FXSendAnalysisInput(
                channel_id=1,
                source_role="kick",
                current_send_db=-36.0,
            ),
        ]
    )

    assert len(proposals) == 1
    assert proposals[0]["channel"] == 6


def test_build_fx_send_proposals_filters_stale_state_items():
    proposals = build_fx_send_proposals(
        [
            FXSendAnalysisInput(
                channel_id=6,
                source_role="lead_vocal",
                current_send_db=-30.0,
                return_headroom_db=6.0,
                feedback_risk=0.1,
                intelligibility_risk=0.2,
                source_confidence=0.9,
                persistence_cycles=3,
                current_state_cache_freshness={"status": "stale"},
            ),
        ]
    )

    assert proposals == ()


def test_build_fx_send_proposals_uses_shared_fx_snapshot_schema():
    snapshot = {
        "type": "fx_inserts_snapshot",
        "safety": {
            "queries_console": False,
            "live_write": False,
        },
        "summary": {
            "read_coverage_status": "partial",
            "cache_freshness_status": "fresh",
        },
        "cache_freshness": {"status": "fresh"},
        "parameter_schema": {
            "status": "mapped",
            "mapped_parameter_count": 6,
            "generic_parameter_count": 0,
            "physical_model_schema_verified": False,
            "live_write_authority": False,
        },
        "fx_slots": [
            {
                "slot": "FX1",
                "model": "PLATE",
                "parameter_schema": {
                    "status": "mapped",
                    "schema_family": "reverb",
                    "schema_source": "automixer_generic_fx_parameter_hints",
                    "mapped_parameter_count": 6,
                    "generic_parameter_count": 0,
                    "cached_parameter_count": 6,
                    "physical_model_schema_verified": False,
                    "live_write_authority": False,
                },
            },
        ],
    }

    proposals = build_fx_send_proposals(
        [
            FXSendAnalysisInput(
                channel_id=6,
                source_role="lead_vocal",
                current_send_db=-30.0,
                arrangement_role="lead_space",
                return_headroom_db=6.0,
                feedback_risk=0.1,
                intelligibility_risk=0.2,
                source_confidence=0.9,
                persistence_cycles=3,
                current_state_cache_freshness={"status": "fresh"},
            ),
            FXSendAnalysisInput(
                channel_id=1,
                source_role="kick",
                current_send_db=-36.0,
                current_state_cache_freshness={"status": "fresh"},
            ),
        ],
        current_fx_inserts_snapshot=snapshot,
    )

    assert len(proposals) == 1
    assert proposals[0]["channel"] == 6
    assert proposals[0]["evidence"]["current_fx_parameter_schema_status"] == "mapped"
    assert proposals[0]["evidence"]["current_fx_parameter_schema"]["fx_slot"] == "FX1"
    assert proposals[0]["evidence"]["current_fx_parameter_schema"]["snapshot_live_write"] is False


def test_build_fx_send_proposals_blocks_stale_shared_fx_snapshot_schema():
    snapshot = {
        "type": "fx_inserts_snapshot",
        "safety": {
            "queries_console": False,
            "live_write": False,
        },
        "summary": {
            "read_coverage_status": "partial",
            "cache_freshness_status": "stale",
        },
        "cache_freshness": {"status": "stale"},
        "parameter_schema": {
            "status": "mapped",
            "mapped_parameter_count": 6,
            "generic_parameter_count": 0,
            "physical_model_schema_verified": False,
            "live_write_authority": False,
        },
        "fx_slots": [
            {
                "slot": "FX1",
                "model": "PLATE",
                "parameter_schema": {
                    "status": "mapped",
                    "schema_family": "reverb",
                    "schema_source": "automixer_generic_fx_parameter_hints",
                    "mapped_parameter_count": 6,
                    "generic_parameter_count": 0,
                    "cached_parameter_count": 6,
                    "physical_model_schema_verified": False,
                    "live_write_authority": False,
                },
            },
        ],
    }
    recommendation = analyze_fx_send_need(
        FXSendAnalysisInput(
            channel_id=6,
            channel_label="Lead Vox",
            source_role="lead_vocal",
            current_send_db=-30.0,
            arrangement_role="lead_space",
            return_headroom_db=6.0,
            feedback_risk=0.1,
            intelligibility_risk=0.2,
            source_confidence=0.9,
            persistence_cycles=3,
            current_fx_inserts_snapshot=snapshot,
        )
    )
    proposals = build_fx_send_proposals(
        [
            FXSendAnalysisInput(
                channel_id=6,
                source_role="lead_vocal",
                current_send_db=-30.0,
                arrangement_role="lead_space",
                return_headroom_db=6.0,
                feedback_risk=0.1,
                intelligibility_risk=0.2,
                source_confidence=0.9,
                persistence_cycles=3,
            ),
        ],
        current_fx_inserts_snapshot=snapshot,
    )

    assert recommendation.status == "analysis_only"
    assert recommendation.severity == "medium"
    assert "fx_inserts_snapshot_cache_not_fresh" in recommendation.blockers
    assert recommendation.evidence["current_fx_parameter_schema_status"] == "mapped"
    assert (
        recommendation.evidence["current_fx_parameter_schema"]["snapshot_cache_freshness_status"]
        == "stale"
    )
    assert proposals == ()


def test_build_fx_send_proposals_filters_generic_shared_fx_snapshot_schema():
    snapshot = {
        "type": "fx_inserts_snapshot",
        "safety": {
            "queries_console": False,
            "live_write": False,
        },
        "summary": {
            "read_coverage_status": "partial",
            "cache_freshness_status": "fresh",
        },
        "cache_freshness": {"status": "fresh"},
        "parameter_schema": {
            "status": "partial",
            "mapped_parameter_count": 0,
            "generic_parameter_count": 1,
            "physical_model_schema_verified": False,
            "live_write_authority": False,
        },
        "fx_slots": [
            {
                "slot": "FX1",
                "model": "CUSTOM-FX",
                "parameter_schema": {
                    "status": "partial",
                    "schema_family": "unknown",
                    "mapped_parameter_count": 0,
                    "generic_parameter_count": 1,
                    "cached_parameter_count": 1,
                    "physical_model_schema_verified": False,
                    "live_write_authority": False,
                },
            },
        ],
    }

    proposals = build_fx_send_proposals(
        [
            FXSendAnalysisInput(
                channel_id=6,
                source_role="lead_vocal",
                current_send_db=-30.0,
                arrangement_role="lead_space",
                return_headroom_db=6.0,
                feedback_risk=0.1,
                intelligibility_risk=0.2,
                source_confidence=0.9,
                persistence_cycles=3,
                current_state_cache_freshness={"status": "fresh"},
            ),
        ],
        current_fx_inserts_snapshot=snapshot,
    )

    assert proposals == ()


def test_fx_send_proposal_is_registered_as_goal_capability():
    method = get_processing_method("fx_send_proposal")
    main = get_pipeline_profile("main_live_assist")
    effects = get_project_goal_capability("effects_management")
    sends = get_project_goal_capability("effect_sends")
    console = get_project_goal_capability("console_fx_control")
    contracts = {item["method_id"]: item for item in build_proposal_only_method_contracts()}

    assert method is not None
    assert method.status == "proposal_only"
    assert method.can_write_live is False
    assert "schema hints" in method.summary
    assert "fx_snapshot_freshness_gate" in method.notes
    assert "fx_parameter_schema_gate" in method.notes
    assert "fx_inserts_snapshot_schema_bridge" in method.notes
    assert "batch_fx_snapshot_schema_bridge" in method.notes
    assert main is not None
    assert "fx_send_proposal" in main.proposal_method_ids
    assert effects is not None and effects.status == "proposal_only"
    assert sends is not None and sends.status == "proposal_only"
    assert console is not None and console.status == "proposal_only"
    assert "backend/source_identity_downstream_gate.py" in effects.support_paths
    assert "backend/source_identity_downstream_gate.py" in sends.support_paths
    assert "backend/source_identity_downstream_gate.py" in console.support_paths
    assert contracts["fx_send_proposal"]["proposal_contract"]["value_type"] == "fx_send"
    assert (
        "current_fx_send_return_cache_freshness"
        in contracts["fx_send_proposal"]["proposal_contract"]["safety_evidence_required"]
    )
    assert (
        "current_fx_parameter_schema_hints"
        in contracts["fx_send_proposal"]["proposal_contract"]["safety_evidence_required"]
    )
    assert (
        "source_identity_downstream_trust"
        in contracts["fx_send_proposal"]["proposal_contract"]["safety_evidence_required"]
    )
