from soundcheck_recommendation_workflow import (
    KIND_TO_MODE,
    NoWriteSoundcheckRecommendationWorkflow,
    RecommendationMode,
)
from soundcheck_autonomy_contract import AutonomyMode


def test_soundcheck_recommendation_workflow_builds_dry_run_bundle_for_all_kinds():
    workflow = NoWriteSoundcheckRecommendationWorkflow(
        context_metadata={"scene_id": "soundcheck_scene_1", "engine_state": "analyzing"}
    )

    workflow.recommend_input_gain(
        channel=1,
        current_trim_db=0.0,
        target_trim_db=-6.0,
        metadata={"channel_name": "Kick In", "preset": "kick"},
    )
    workflow.recommend_hpf(
        channel=1,
        current_freq_hz=40.0,
        current_enabled=False,
        target_freq_hz=60.0,
        rationale={"sub_ratio": 0.04},
    )
    workflow.recommend_eq(
        channel=1,
        current_bands=[{"band": 1, "freq_hz": 60.0, "gain_db": 0.0, "q": 1.0}],
        target_bands=[{"band": 1, "freq_hz": 60.0, "gain_db": 3.0, "q": 1.0}],
    )
    workflow.recommend_compressor(
        channel=1,
        current_settings={"enabled": False},
        target_settings={"enabled": True, "threshold_db": -20.0, "ratio": 4.0},
    )
    workflow.recommend_pan(channel=2, current_pan=0.0, target_pan=-25.0)
    workflow.recommend_fader(channel=1, current_fader_db=-12.0, target_fader_db=-7.0)
    workflow.recommend_fx_send(
        channel=1,
        bus=1,
        current_level_db=None,
        target_level_db=-18.0,
    )
    workflow.recommend_feedback_notch(
        channel=3,
        frequency_hz=3150.0,
        rationale={"detected_magnitude_db": -6.0},
    )
    workflow.recommend_feedback_fader_reduce(
        channel=3,
        current_fader_db=-8.0,
        target_fader_db=-11.0,
    )

    bundle = workflow.build_bundle()

    assert bundle["schema_version"] == "soundcheck_recommendation_bundle/v1"
    assert bundle["dry_run_only"] is True
    assert bundle["live_mixer_writes"] is False
    assert bundle["context_metadata"]["scene_id"] == "soundcheck_scene_1"
    assert len(bundle["recommendations"]) == 9
    assert len(bundle["proposals"]) == 9
    assert bundle["ranking"]["selected_proposal_id"]
    assert bundle["timeline_artifact"]["dry_run_only"] is True
    assert bundle["timeline_artifact"]["live_mixer_writes"] is False
    assert bundle["graph_checkpoint"]["timeline_artifact"]["dry_run_only"] is True
    assert bundle["mode_summary"]["approval_required"] == 2
    assert bundle["mode_summary"]["assisted"] == 5
    assert bundle["mode_summary"]["observe_only"] == 2
    assert bundle["autonomy_summary"]["semi_autonomous_calibration"] == 2
    assert bundle["autonomy_summary"]["assisted"] == 5
    assert bundle["autonomy_summary"]["observe_only"] == 2

    for proposal in bundle["proposals"]:
        assert proposal["transport_policy"] == "dry_run_only"
        assert proposal["live_mixer_writes"] is False
        assert proposal["auto_apply_blocked"] is True
        assert proposal["safety"]["workflow"] == "soundcheck_no_write"


def test_soundcheck_recommendation_workflow_mode_mapping_prioritizes_console_safety():
    assert KIND_TO_MODE["input_gain"] == RecommendationMode.APPROVAL_REQUIRED
    assert KIND_TO_MODE["fader"] == RecommendationMode.APPROVAL_REQUIRED
    assert KIND_TO_MODE["hpf"] == RecommendationMode.ASSISTED
    assert KIND_TO_MODE["eq"] == RecommendationMode.ASSISTED
    assert KIND_TO_MODE["compressor"] == RecommendationMode.ASSISTED
    assert KIND_TO_MODE["pan"] == RecommendationMode.ASSISTED
    assert KIND_TO_MODE["fx_send"] == RecommendationMode.ASSISTED
    assert KIND_TO_MODE["feedback_notch"] == RecommendationMode.OBSERVE_ONLY
    assert KIND_TO_MODE["feedback_fader_reduce"] == RecommendationMode.OBSERVE_ONLY


def test_soundcheck_recommendation_workflow_preserves_payload_shape():
    workflow = NoWriteSoundcheckRecommendationWorkflow(
        context_metadata={"scene_id": "scene_a", "engine_state": "running"}
    )

    workflow.recommend_fader(
        channel=7,
        current_fader_db=-10.0,
        target_fader_db=-6.5,
        metadata={"channel_name": "Lead Vox"},
        rationale={"gain_correction_db": 3.5, "lufs": -23.0},
    )

    bundle = workflow.build_bundle()
    recommendation = bundle["recommendations"][0]
    proposal = bundle["proposals"][0]

    assert recommendation["mode"] == "approval_required"
    assert recommendation["autonomy_mode"] == "semi_autonomous_calibration"
    assert recommendation["requested_state"]["fader_db"] == -6.5
    assert recommendation["source_payload"]["metadata"]["channel_name"] == "Lead Vox"
    assert recommendation["source_payload"]["rationale"]["gain_correction_db"] == 3.5
    assert recommendation["source_payload"]["action_envelope"]["operator_summary"]
    assert recommendation["action_envelope"]["required_approval_level"] == "operator_required"
    assert proposal["target"]["mixer_channel"] == 7
    assert proposal["requested_state"]["fader_db"] == -6.5
    assert proposal["source_payload"]["mode"] == "approval_required"
    assert proposal["source_payload"]["autonomy_mode"] == "semi_autonomous_calibration"
    assert proposal["source_payload"]["safety"]["requires_confirmation"] is True


def test_soundcheck_recommendation_workflow_downgrades_to_emergency_mode_when_explicit_blockers_exist():
    workflow = NoWriteSoundcheckRecommendationWorkflow(
        context_metadata={
            "scene_id": "scene_blocked",
            "rollback_path_available": False,
            "telemetry_readback_available": False,
        }
    )

    workflow.recommend_input_gain(
        channel=2,
        current_trim_db=0.0,
        target_trim_db=-3.0,
        confidence=0.95,
        metadata={"channel_name": "Lead Vox", "venue_familiarity": "known_venue"},
    )

    recommendation = workflow.build_bundle()["recommendations"][0]

    assert recommendation["mode"] == "approval_required"
    assert recommendation["autonomy_mode"] == AutonomyMode.EMERGENCY_HUMAN_ONLY_FALLBACK.value
    assert recommendation["action_envelope"]["required_approval_level"] == "human_required"
    assert recommendation["action_envelope"]["trust_gates"]["summary"]["blocking_gate_names"] == [
        "telemetry_readback_available",
        "rollback_path_available",
    ]
