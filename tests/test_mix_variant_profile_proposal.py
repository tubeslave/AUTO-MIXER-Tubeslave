from pathlib import Path

from backend.mix_variant_profile_proposal import (
    MixVariantProfileAnalysisInput,
    analyze_mix_variant_profile_need,
    build_mix_variant_profile_proposals,
    normalize_genre,
    normalize_task,
    resolve_mix_variant_profile,
)
from backend.processing_proposal_adapters import build_proposal_only_method_contracts
from backend.processing_registry import get_pipeline_profile, get_processing_method
from backend.project_goal_capability_matrix import get_project_goal_capability


def test_mix_variant_profile_module_is_transport_free():
    source = Path("backend/mix_variant_profile_proposal.py").read_text()

    forbidden = [
        "WingClient",
        "EnhancedOSCClient",
        ".send(",
        "mixer_client",
        "set_fader",
        "set_gain",
        "set_profile",
        "load_snap",
        "save_snap",
        "recall_scene",
    ]
    for token in forbidden:
        assert token not in source


def test_live_pop_rock_variant_creates_proposal_only_profile():
    recommendation = analyze_mix_variant_profile_need(
        MixVariantProfileAnalysisInput(
            variant_id="variant:pop-rock-live",
            genre="pop rock",
            task="live",
            current_profile_id="custom:live_foh",
            reference_id="ref:house_pop_rock",
            reference_lufs=-17.8,
            reference_metrics_available=True,
            arrangement_context_available=True,
            arrangement_roles=("lead_vocal", "drums", "bass", "guitar", "keys"),
            detected_source_count=18,
            stem_count=5,
            operator_intent="operator_variant_request",
            mix_window="soundcheck",
            current_integrated_lufs=-18.5,
            current_true_peak_dbfs=-5.5,
            main_bus_headroom_db=4.5,
            feedback_risk=0.18,
            genre_confidence=0.92,
            task_confidence=0.9,
            reference_confidence=0.84,
            arrangement_confidence=0.88,
            persistence_cycles=3,
        )
    )

    assert recommendation.status == "proposal_only"
    assert recommendation.action_type == "mix_variant_profile_proposal"
    assert recommendation.genre == "pop_rock"
    assert recommendation.task == "live_foh"
    assert recommendation.blockers == ()
    assert recommendation.target_profile["target_planes_lufs"]["lead_main"] > -22.1
    assert recommendation.target_profile["mix_intent"]["drum_punch"] > 0.7
    assert "live_feedback_reserve" in recommendation.target_profile["notes"]
    assert recommendation.evidence["transport_free"] is True
    assert recommendation.evidence["no_named_artist_style_clone"] is True
    assert recommendation.evidence["ayaic_level_planes"] is True

    proposal = recommendation.to_operator_proposal()
    assert proposal["method_id"] == "mix_variant_profile_proposal"
    assert proposal["kind"] == "mix_variant_profile"
    assert proposal["can_import_to_queue"] is False
    assert proposal["can_apply_live"] is False
    assert "requested_change" not in proposal
    assert "requested_changes" not in proposal
    assert "unsupported_value_type:mix_variant_profile" in proposal["blockers"]
    assert "bus_group_balance_proposal" in proposal["proposed_action"]["followup_proposal_methods"]
    assert proposal["evidence"]["live_write"] is False


def test_style_clone_and_active_performance_block_mix_variant():
    recommendation = analyze_mix_variant_profile_need(
        MixVariantProfileAnalysisInput(
            variant_id="variant:clone",
            genre="rock",
            task="concert",
            reference_id="artist:specific_mix",
            reference_lufs=-12.0,
            reference_metrics_available=True,
            arrangement_context_available=True,
            arrangement_roles=("lead_vocal", "drums", "bass"),
            detected_source_count=12,
            operator_intent="operator_variant_request",
            mix_window="show",
            active_performance=True,
            style_clone_requested=True,
            reference_similarity_target=0.93,
            current_integrated_lufs=-18.0,
            main_bus_headroom_db=5.0,
            genre_confidence=0.9,
            task_confidence=0.9,
            reference_confidence=0.9,
            arrangement_confidence=0.9,
            persistence_cycles=3,
        )
    )

    assert recommendation.status == "analysis_only"
    assert "style_clone_request_blocked" in recommendation.blockers
    assert "active_performance_variant_blocked" in recommendation.blockers
    assert recommendation.severity == "high"
    assert recommendation.to_operator_proposal()["can_apply_live"] is False


def test_missing_reference_metrics_or_arrangement_context_blocks_variant():
    recommendation = analyze_mix_variant_profile_need(
        MixVariantProfileAnalysisInput(
            variant_id="variant:missing-evidence",
            genre="acoustic",
            task="studio",
            reference_id="ref:acoustic_target",
            reference_metrics_available=False,
            arrangement_context_available=False,
            operator_intent="operator_variant_request",
            current_integrated_lufs=-18.0,
            main_bus_headroom_db=5.0,
            genre_confidence=0.9,
            task_confidence=0.9,
            reference_confidence=0.9,
            arrangement_confidence=0.9,
        )
    )

    assert recommendation.status == "analysis_only"
    assert "reference_metrics_missing" in recommendation.blockers
    assert "arrangement_context_missing" in recommendation.blockers


def test_loudness_up_requires_true_peak_headroom_evidence():
    recommendation = analyze_mix_variant_profile_need(
        MixVariantProfileAnalysisInput(
            variant_id="variant:streaming-loud",
            genre="electronic",
            task="streaming",
            reference_lufs=-13.5,
            reference_metrics_available=True,
            arrangement_context_available=True,
            arrangement_roles=("drums", "bass", "synth"),
            detected_source_count=20,
            operator_intent="operator_variant_request",
            mix_window="studio",
            current_integrated_lufs=-18.5,
            current_true_peak_dbfs=-0.8,
            main_bus_headroom_db=0.8,
            genre_confidence=0.95,
            task_confidence=0.95,
            reference_confidence=0.9,
            arrangement_confidence=0.9,
            persistence_cycles=3,
        )
    )

    assert recommendation.status == "analysis_only"
    assert "variant_loudness_headroom_insufficient" in recommendation.blockers


def test_mix_variant_helpers_normalize_and_resolve_profile_safely():
    assert normalize_genre("pop rock") == "pop_rock"
    assert normalize_genre("EDM") == "electronic"
    assert normalize_task("concert") == "live_foh"
    assert normalize_task("streaming") == "streaming_reference"

    profile = resolve_mix_variant_profile(
        "worship",
        "live",
        reference_lufs=-10.0,
        lead_role_present=True,
        dense_arrangement=True,
        feedback_risk=0.65,
    )

    assert profile.target_integrated_lufs == -12.0
    assert profile.ambience <= 0.22
    assert "feedback_risk_ambience_limited" in profile.notes
    assert profile.target_planes()["lead_main"] > -22.1


def test_build_mix_variant_profile_proposals_filters_to_proposal_only_items():
    proposals = build_mix_variant_profile_proposals(
        [
            MixVariantProfileAnalysisInput(
                variant_id="variant:ok",
                genre="jazz",
                task="studio",
                reference_metrics_available=True,
                arrangement_context_available=True,
                arrangement_roles=("piano", "bass", "drums"),
                detected_source_count=8,
                operator_intent="operator_variant_request",
                mix_window="studio",
                current_integrated_lufs=-18.0,
                main_bus_headroom_db=5.0,
                genre_confidence=0.9,
                task_confidence=0.9,
                reference_confidence=0.8,
                arrangement_confidence=0.85,
                persistence_cycles=3,
            ),
            MixVariantProfileAnalysisInput(
                variant_id="variant:blocked",
                genre="jazz",
                task="studio",
                reference_id="ref:jazz",
                reference_metrics_available=False,
                arrangement_context_available=True,
                operator_intent="operator_variant_request",
                current_integrated_lufs=-18.0,
                main_bus_headroom_db=5.0,
                genre_confidence=0.9,
                task_confidence=0.9,
                reference_confidence=0.8,
                arrangement_confidence=0.85,
            ),
        ]
    )

    assert len(proposals) == 1
    assert proposals[0]["target"] == "variant:ok"
    assert proposals[0]["can_import_to_queue"] is False


def test_mix_variant_profile_proposal_is_registered_as_goal_capability():
    method = get_processing_method("mix_variant_profile_proposal")
    main = get_pipeline_profile("main_live_assist")
    offline = get_pipeline_profile("offline_mix")
    capability = get_project_goal_capability("genre_variants")
    contracts = {item["method_id"]: item for item in build_proposal_only_method_contracts()}

    assert method is not None
    assert method.status == "proposal_only"
    assert method.domain == "mix_variant"
    assert method.can_write_live is False
    assert main is not None
    assert "mix_variant_profile_proposal" in main.proposal_method_ids
    assert offline is not None
    assert "mix_variant_profile_proposal" in offline.method_ids
    assert capability is not None
    assert capability.status == "proposal_only"
    assert capability.autonomy_mode_ceiling == "assisted"
    assert "mix_variant_profile_proposal" in capability.method_ids
    assert contracts["mix_variant_profile_proposal"]["proposal_contract"]["value_type"] == "mix_variant_profile"
