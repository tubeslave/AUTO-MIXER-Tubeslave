from pathlib import Path

from backend.pan_spatial_proposal import (
    PanAnalysisInput,
    analyze_pan_need,
    build_pan_proposals,
    classify_source_family,
)
from backend.processing_proposal_adapters import build_proposal_only_method_contracts
from backend.processing_registry import get_pipeline_profile, get_processing_method
from backend.project_goal_capability_matrix import get_project_goal_capability


def test_pan_proposal_module_is_transport_free():
    source = Path("backend/pan_spatial_proposal.py").read_text()

    forbidden = [
        "WingClient",
        "EnhancedOSCClient",
        ".send(",
        "set_pan",
        "set_channel_pan",
        "mixer_client",
    ]
    for token in forbidden:
        assert token not in source


def test_center_anchor_can_create_proposal_only_pan_recommendation():
    recommendation = analyze_pan_need(
        PanAnalysisInput(
            channel_id=4,
            channel_label="Lead Vox",
            source_role="lead_vocal",
            current_pan_percent=-24.0,
            mono_compatibility_risk=0.15,
            contribution_share=0.9,
            role_confidence=0.95,
            persistence_cycles=3,
            current_state_cache_freshness={
                "status": "fresh",
                "field_count": 4,
                "fresh_field_count": 4,
                "read_only": True,
                "live_write": False,
                "queries_console": False,
            },
        )
    )

    assert recommendation.status == "proposal_only"
    assert recommendation.action_type == "pan_proposal"
    assert recommendation.target_pan_percent == 0.0
    assert recommendation.delta_pan_percent > 0
    assert recommendation.confidence >= 0.45
    assert recommendation.blockers == ()
    assert recommendation.evidence["transport_free"] is True
    assert recommendation.evidence["current_state_cache_freshness_status"] == "fresh"
    assert recommendation.evidence["current_state_cache_freshness"]["fresh_field_count"] == 4
    assert recommendation.evidence["source_identity_downstream_trust"]["status"] == "legacy_unverified"

    proposal = recommendation.to_operator_proposal()
    assert proposal["method_id"] == "pan_spatial_proposal"
    assert proposal["can_import_to_queue"] is False
    assert proposal["can_apply_live"] is False
    assert "requested_change" not in proposal
    assert "unsupported_value_type:pan" in proposal["blockers"]
    assert proposal["evidence"]["live_write"] is False
    assert proposal["evidence"]["current_state_cache_freshness_status"] == "fresh"


def test_stereo_pair_side_can_create_spatial_pan_proposal():
    recommendation = analyze_pan_need(
        PanAnalysisInput(
            channel_id=9,
            channel_label="GTR L",
            source_role="guitar",
            current_pan_percent=0.0,
            stereo_pair_id="gtr_pair",
            stereo_pair_side="left",
            mono_compatibility_risk=0.2,
            contribution_share=0.65,
            role_confidence=0.8,
            persistence_cycles=3,
        )
    )

    assert recommendation.status == "proposal_only"
    assert recommendation.source_family == "guitar"
    assert recommendation.target_pan_percent < 0.0
    assert recommendation.evidence["decision_source"] == "stereo_pair_side"
    assert recommendation.evidence["stereo_pair_side"] == "left"


def test_high_mono_risk_blocks_non_center_pan_proposal():
    recommendation = analyze_pan_need(
        PanAnalysisInput(
            channel_id=10,
            source_role="keys",
            current_pan_percent=0.0,
            stage_position=0.8,
            mono_compatibility_risk=0.9,
            role_confidence=0.9,
            persistence_cycles=3,
        )
    )

    assert recommendation.status == "analysis_only"
    assert "mono_compatibility_risk_high" in recommendation.blockers


def test_stale_current_pan_state_blocks_proposal_only_recommendation():
    recommendation = analyze_pan_need(
        PanAnalysisInput(
            channel_id=9,
            channel_label="GTR L",
            source_role="guitar",
            current_pan_percent=0.0,
            stereo_pair_id="gtr_pair",
            stereo_pair_side="left",
            mono_compatibility_risk=0.2,
            contribution_share=0.65,
            role_confidence=0.8,
            persistence_cycles=3,
            current_state_cache_freshness={
                "status": "stale",
                "field_count": 4,
                "fresh_field_count": 2,
                "stale_field_count": 2,
                "stale_fields": ["pan_percent", "stereo_link"],
                "read_only": True,
                "live_write": False,
                "queries_console": False,
            },
        )
    )

    assert recommendation.status == "analysis_only"
    assert "pan_current_state_cache_not_fresh" in recommendation.blockers
    assert recommendation.evidence["current_state_cache_freshness_status"] == "stale"
    assert recommendation.evidence["current_state_cache_freshness"]["stale_fields"] == [
        "pan_percent",
        "stereo_link",
    ]
    assert recommendation.to_operator_proposal()["can_apply_live"] is False


def test_source_identity_hold_blocks_downstream_pan_proposal():
    recommendation = analyze_pan_need(
        PanAnalysisInput(
            channel_id=9,
            channel_label="GTR L",
            source_role="guitar",
            current_pan_percent=0.0,
            stereo_pair_id="gtr_pair",
            stereo_pair_side="left",
            mono_compatibility_risk=0.2,
            contribution_share=0.65,
            role_confidence=0.8,
            persistence_cycles=3,
            source_identity_trust_state="hold_for_operator_review",
            source_identity_trust_confidence=0.42,
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


def test_stale_source_identity_replay_context_blocks_downstream_pan_proposal():
    recommendation = analyze_pan_need(
        PanAnalysisInput(
            channel_id=9,
            channel_label="GTR L",
            source_role="guitar",
            current_pan_percent=0.0,
            stereo_pair_id="gtr_pair",
            stereo_pair_side="left",
            mono_compatibility_risk=0.2,
            contribution_share=0.65,
            role_confidence=0.8,
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


def test_non_anchor_source_without_spatial_context_is_analysis_only():
    recommendation = analyze_pan_need(
        PanAnalysisInput(
            channel_id=11,
            source_role="guitar",
            current_pan_percent=0.0,
            mono_compatibility_risk=0.1,
            role_confidence=0.8,
        )
    )

    assert recommendation.status == "analysis_only"
    assert "stage_or_stereo_pair_context_missing" in recommendation.blockers


def test_pan_helpers_classify_roles():
    assert classify_source_family("Lead Vocal") == "lead_vocal"
    assert classify_source_family("Bass DI") == "bass"
    assert classify_source_family("GTR 57") == "guitar"


def test_build_pan_proposals_filters_to_proposal_only_items():
    proposals = build_pan_proposals(
        [
            PanAnalysisInput(
                channel_id=9,
                source_role="guitar",
                stereo_pair_id="gtr_pair",
                stereo_pair_side="right",
                current_pan_percent=0.0,
                mono_compatibility_risk=0.1,
                role_confidence=0.85,
                persistence_cycles=3,
            ),
            PanAnalysisInput(
                channel_id=11,
                source_role="guitar",
                current_pan_percent=0.0,
            ),
        ]
    )

    assert len(proposals) == 1
    assert proposals[0]["channel"] == 9


def test_build_pan_proposals_filters_stale_state_items():
    proposals = build_pan_proposals(
        [
            PanAnalysisInput(
                channel_id=9,
                source_role="guitar",
                stereo_pair_id="gtr_pair",
                stereo_pair_side="right",
                current_pan_percent=0.0,
                mono_compatibility_risk=0.1,
                role_confidence=0.85,
                persistence_cycles=3,
                current_state_cache_freshness={"status": "stale"},
            ),
        ]
    )

    assert proposals == ()


def test_pan_proposal_is_registered_as_goal_capability():
    method = get_processing_method("pan_spatial_proposal")
    main = get_pipeline_profile("main_live_assist")
    capability = get_project_goal_capability("panning")
    console = get_project_goal_capability("console_pan_control")
    contracts = {item["method_id"]: item for item in build_proposal_only_method_contracts()}

    assert method is not None
    assert method.status == "proposal_only"
    assert method.can_write_live is False
    assert main is not None
    assert "pan_spatial_proposal" in main.proposal_method_ids
    assert capability is not None
    assert capability.status == "proposal_only"
    assert "pan_spatial_proposal" in capability.method_ids
    assert console is not None
    assert console.status == "proposal_only"
    assert "backend/source_identity_downstream_gate.py" in capability.support_paths
    assert "backend/source_identity_downstream_gate.py" in console.support_paths
    assert contracts["pan_spatial_proposal"]["proposal_contract"]["value_type"] == "pan"
    assert (
        "current_pan_cache_freshness"
        in contracts["pan_spatial_proposal"]["proposal_contract"]["safety_evidence_required"]
    )
    assert (
        "source_identity_downstream_trust"
        in contracts["pan_spatial_proposal"]["proposal_contract"]["safety_evidence_required"]
    )
