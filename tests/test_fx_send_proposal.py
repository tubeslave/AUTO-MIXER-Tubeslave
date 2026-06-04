from pathlib import Path

from backend.fx_send_proposal import (
    FXSendAnalysisInput,
    analyze_fx_send_need,
    build_fx_send_proposals,
    classify_source_family,
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

    proposal = recommendation.to_operator_proposal()
    assert proposal["method_id"] == "fx_send_proposal"
    assert proposal["can_import_to_queue"] is False
    assert proposal["can_apply_live"] is False
    assert "requested_change" not in proposal
    assert "unsupported_value_type:fx_send" in proposal["blockers"]
    assert proposal["evidence"]["live_write"] is False


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
    assert main is not None
    assert "fx_send_proposal" in main.proposal_method_ids
    assert effects is not None and effects.status == "proposal_only"
    assert sends is not None and sends.status == "proposal_only"
    assert console is not None and console.status == "proposal_only"
    assert contracts["fx_send_proposal"]["proposal_contract"]["value_type"] == "fx_send"
