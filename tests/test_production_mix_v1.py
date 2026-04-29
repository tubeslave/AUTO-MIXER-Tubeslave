from __future__ import annotations

from pathlib import Path

import numpy as np

from automixer.production_mix_v1 import ProductionMixPipeline, load_production_mix_config
from automixer.production_mix_v1.analyzers import analyze_session
from automixer.production_mix_v1.audio_io import align_stems, generate_smoke_stems, load_audio_stems
from automixer.production_mix_v1.candidates import StagedCandidateGenerator
from automixer.production_mix_v1.critics import ProductionCriticSuite
from automixer.production_mix_v1.decision import ProductionMixDecisionEngine
from automixer.production_mix_v1.models import (
    ACTION_COMPRESSION,
    ACTION_FX_SEND,
    ACTION_GAIN,
    AudioStem,
    CandidateEvaluation,
    MixAction,
    MixCandidate,
    RoleDetectionResult,
    SafetyResult,
)
from automixer.production_mix_v1.osc_queue import OSCCommandQueue
from automixer.production_mix_v1.rendering import render_candidate, render_candidate_with_status
from automixer.production_mix_v1.safety_adapter import ProductionSafetyGovernor


def test_production_mix_v1_config_loads_separate_pipeline_config():
    config = load_production_mix_config()

    assert config.name == "production_mix_v1"
    assert config.baseline_candidate_name == "candidate_000_no_change"
    assert config.mode_config("live")["max_db_step"] == 0.7
    assert config.safety_limits["max_true_peak_dbfs"] == -1.0
    assert config.reference_recipe["inherited_reference_recipe"] == "FX15_SECTION_MOD_SPACE_AIR_MINUS_080"


def test_staged_candidate_generator_keeps_baseline_first():
    config = load_production_mix_config()
    stems = align_stems(generate_smoke_stems())
    baseline = MixCandidate(config.baseline_candidate_name, "baseline")
    mix, _processed = render_candidate(stems, baseline)
    analysis = analyze_session(stems, mix, config)

    candidates = StagedCandidateGenerator(config).generate(stems, analysis, mode="offline")

    assert candidates[0].name == "candidate_000_no_change"
    assert candidates[0].actions == []
    assert len(candidates) <= config.max_candidates("offline")
    assert any(candidate.actions for candidate in candidates[1:])
    assert any(candidate.name == "candidate_065_fx15_air_minus_080_reference" for candidate in candidates)


def test_safety_governor_blocks_true_peak_and_fader_overshoot():
    config = load_production_mix_config()
    action = MixAction(
        action_type=ACTION_GAIN,
        target="Lead",
        channel_id=1,
        parameters={"gain_db": 2.0, "current_fader_db": -1.0},
        rationale="unsafe boost",
    )
    candidate = MixCandidate("unsafe", "test", actions=[action])
    baseline = {
        "mix": {"true_peak_dbfs": -3.0, "phase_correlation": 1.0, "dimensions": {}},
        "stems": {"1": {"metrics": {"true_peak_dbfs": -3.0}}},
    }
    candidate_analysis = {
        "mix": {"true_peak_dbfs": -0.2, "phase_correlation": 1.0, "dimensions": {}},
        "stems": {"1": {"metrics": {"true_peak_dbfs": -0.2}}},
    }

    result = ProductionSafetyGovernor(config).evaluate(
        candidate,
        candidate_analysis=candidate_analysis,
        baseline_analysis=baseline,
        mode="live",
        dry_run=True,
    )

    assert not result.passed
    reasons = {item["reason"] for item in result.blocked_actions}
    assert "true_peak_limit_exceeded" in reasons
    assert "fader_would_exceed_0_dbfs" in reasons


def test_safety_governor_allows_negative_offline_mix_bus_trim():
    config = load_production_mix_config()
    action = MixAction(
        action_type=ACTION_GAIN,
        target="mix_bus",
        channel_id=None,
        parameters={"gain_db": -12.0, "true_peak_before_dbfs": 10.5},
        rationale="offline final safety trim",
    )
    candidate = MixCandidate("trim", "final_safety_trim", actions=[action])
    baseline = {
        "mix": {"true_peak_dbfs": 10.5, "phase_correlation": 1.0, "dimensions": {}},
        "stems": {},
    }
    candidate_analysis = {
        "mix": {"true_peak_dbfs": -1.5, "phase_correlation": 1.0, "dimensions": {}},
        "stems": {},
    }

    result = ProductionSafetyGovernor(config).evaluate(
        candidate,
        candidate_analysis=candidate_analysis,
        baseline_analysis=baseline,
        mode="offline",
        dry_run=True,
    )

    assert result.passed
    assert result.allowed_actions == [action]


def test_decision_engine_rejects_proxy_only_candidate():
    config = load_production_mix_config()
    baseline_candidate = MixCandidate(config.baseline_candidate_name, "baseline")
    proxy_candidate = MixCandidate("proxy_winner", "proxy")
    safety = SafetyResult("proxy_winner", passed=True)
    baseline = CandidateEvaluation(
        candidate=baseline_candidate,
        analysis={},
        dimensions={},
        critic_scores={},
        aggregate_score=0.50,
        primary_evidence=["rules"],
    )
    proxy = CandidateEvaluation(
        candidate=proxy_candidate,
        analysis={},
        dimensions={},
        critic_scores={"clap_proxy": {"score": 1.0}},
        aggregate_score=0.90,
        score_delta=0.40,
        primary_evidence=[],
        proxy_evidence=["clap_proxy"],
        safety=safety,
    )

    decision = ProductionMixDecisionEngine(config).decide([baseline, proxy], mode="offline")

    assert not decision.accepted
    assert decision.selected.candidate.name == config.baseline_candidate_name
    assert "proxy_only_winner" in decision.rejected[0]["reasons"]


def test_final_trim_cannot_win_alone():
    config = load_production_mix_config()
    baseline_candidate = MixCandidate(config.baseline_candidate_name, "baseline")
    trim_action = MixAction(
        action_type=ACTION_GAIN,
        target="mix_bus",
        channel_id=None,
        parameters={"gain_db": -12.0},
        rationale="utility trim",
        stage="final_safety_trim",
    )
    trim_candidate = MixCandidate("technical_trim_only", "utility", actions=[trim_action])
    baseline = CandidateEvaluation(
        candidate=baseline_candidate,
        analysis={},
        dimensions={},
        critic_scores={},
        aggregate_score=0.40,
        musical_score=0.40,
        technical_safety_score=0.10,
        final_score=0.35,
        primary_evidence=["rules"],
    )
    trim = CandidateEvaluation(
        candidate=trim_candidate,
        analysis={},
        dimensions={},
        critic_scores={},
        aggregate_score=0.80,
        musical_score=0.40,
        technical_safety_score=1.0,
        final_score=0.80,
        score_delta=0.45,
        musical_score_delta=0.0,
        technical_score_delta=0.90,
        final_score_delta=0.45,
        primary_evidence=["rules"],
        safety=SafetyResult("technical_trim_only", passed=True, allowed_actions=[trim_action]),
    )

    decision = ProductionMixDecisionEngine(config).decide([baseline, trim], mode="offline")

    assert not decision.accepted
    assert decision.selected.candidate.name == config.baseline_candidate_name
    assert "utility_only_candidate" in decision.rejected[0]["reasons"]


def test_musical_score_cannot_increase_from_action_labels_only():
    config = load_production_mix_config()
    stems = align_stems(generate_smoke_stems())
    baseline = MixCandidate(config.baseline_candidate_name, "baseline")
    fx_action = MixAction(
        action_type=ACTION_FX_SEND,
        target="Lead Vox",
        channel_id=4,
        parameters=_fx_params(),
        rationale="space",
        stage="fx15_section_mod_space",
    )
    fx_candidate = MixCandidate(
        "space",
        "musical",
        actions=[fx_action],
        metadata={"inherited_reference_recipe": "FX15_SECTION_MOD_SPACE_AIR_MINUS_080"},
    )
    base_mix, _ = render_candidate(stems, baseline)
    base_analysis = analyze_session(stems, base_mix, config)
    safety = SafetyResult("safe", passed=True)
    critics = ProductionCriticSuite(config, enable_optional_critics=False)

    dry_eval = critics.evaluate(
        candidate=baseline,
        analysis=base_analysis,
        baseline_audio=base_mix,
        candidate_audio=base_mix,
        sample_rate=48000,
        safety=safety,
        critical_degradations=[],
        baseline_analysis=base_analysis,
        musical_analysis=base_analysis,
    )
    space_eval = critics.evaluate(
        candidate=fx_candidate,
        analysis=base_analysis,
        baseline_audio=base_mix,
        candidate_audio=base_mix,
        sample_rate=48000,
        safety=safety,
        critical_degradations=[],
        baseline_analysis=base_analysis,
        render_verification={
            "actual_offline_render": False,
            "required_actions_rendered": False,
            "candidate_is_not_dry": False,
            "balance_before_after": {},
        },
    )

    assert space_eval.musical_score <= dry_eval.musical_score


def test_dynamics_candidate_not_rejected_only_for_small_delta():
    config = load_production_mix_config()
    baseline_candidate = MixCandidate(config.baseline_candidate_name, "baseline")
    comp_action = MixAction(
        action_type=ACTION_COMPRESSION,
        target="Lead Vox",
        channel_id=4,
        parameters=_compression_params(),
        rationale="vocal leveling",
        stage="track_compression_recipe",
    )
    dynamics_candidate = MixCandidate("small_dynamics", "dynamics", actions=[comp_action])
    baseline = CandidateEvaluation(
        candidate=baseline_candidate,
        analysis={},
        dimensions={},
        critic_scores={},
        aggregate_score=0.50,
        musical_score=0.50,
        technical_safety_score=0.50,
        final_score=0.50,
        primary_evidence=["rules"],
    )
    dynamics = CandidateEvaluation(
        candidate=dynamics_candidate,
        analysis={},
        dimensions={},
        critic_scores={},
        aggregate_score=0.501,
        musical_score=0.501,
        technical_safety_score=0.50,
        final_score=0.501,
        score_delta=0.001,
        musical_score_delta=0.001,
        final_score_delta=0.001,
        primary_evidence=["rules"],
        safety=SafetyResult("small_dynamics", passed=True, allowed_actions=[comp_action]),
        render_verification={
            "actual_offline_render": True,
            "required_actions_rendered": True,
            "required_actions_verified": True,
            "candidate_is_not_overdynamic": True,
            "vocal_balance_ok": True,
            "stem_balance_ok": True,
            "metrics_improved": True,
        },
    )
    dynamics = _candidate_eval(
        config,
        dynamics_candidate,
        final_delta=0.003,
        render_verification=dynamics.render_verification,
    )

    decision = ProductionMixDecisionEngine(config).decide([baseline, dynamics], mode="offline")

    assert decision.accepted
    assert decision.selected.candidate.name == "small_dynamics"


def test_proxy_fallback_muq_triggers_review_for_technical_only_improvement():
    config = load_production_mix_config()
    baseline_candidate = MixCandidate(config.baseline_candidate_name, "baseline")
    trim_action = MixAction(
        action_type=ACTION_GAIN,
        target="mix_bus",
        channel_id=None,
        parameters={"gain_db": -10.0},
        rationale="technical utility",
        stage="final_safety_trim",
    )
    candidate = MixCandidate("technical_only_with_muq_fallback", "utility", actions=[trim_action])
    baseline = CandidateEvaluation(
        candidate=baseline_candidate,
        analysis={},
        dimensions={},
        critic_scores={},
        aggregate_score=0.50,
        musical_score=0.50,
        technical_safety_score=0.20,
        final_score=0.45,
        primary_evidence=["rules"],
    )
    technical = CandidateEvaluation(
        candidate=candidate,
        analysis={},
        dimensions={},
        critic_scores={"muq_eval": {"score": 0.9, "primary": False, "proxy": True}},
        aggregate_score=0.70,
        musical_score=0.50,
        technical_safety_score=0.95,
        final_score=0.70,
        score_delta=0.25,
        musical_score_delta=0.0,
        technical_score_delta=0.75,
        final_score_delta=0.25,
        primary_evidence=["rules"],
        proxy_evidence=["muq_eval"],
        warnings=["muq_eval_fallback_excluded_from_final_score"],
        safety=SafetyResult("technical_only_with_muq_fallback", passed=True, allowed_actions=[trim_action]),
    )

    decision = ProductionMixDecisionEngine(config).decide([baseline, technical], mode="offline")

    assert not decision.accepted
    assert "review_required_proxy_fallback_technical_only" in decision.rejected[0]["reasons"]


def test_osc_queue_dry_run_does_not_send():
    action = MixAction(
        action_type=ACTION_GAIN,
        target="Lead",
        channel_id=1,
        parameters={"gain_db": -1.0},
        rationale="safe cut",
    )
    candidate = MixCandidate("winner", "test", actions=[action])
    safety = SafetyResult("winner", passed=True, allowed_actions=[action])
    queue = OSCCommandQueue(dry_run=True)

    queue.enqueue_candidate(candidate, safety)
    result = queue.flush()

    assert result.dry_run is True
    assert result.sent == []
    assert len(result.would_send) == 1
    assert result.warnings == ["dry_run_enabled_no_osc_sent"]


def test_production_mix_v1_smoke_run_writes_reports(tmp_path: Path):
    config = load_production_mix_config()
    result = ProductionMixPipeline(config, enable_optional_critics=False).run(
        output_dir=tmp_path,
        mode="offline",
        dry_run=True,
        smoke=True,
    )

    assert result.dry_run is True
    assert result.queue_result.sent == []
    assert Path(result.artifacts["json"]).exists()
    assert Path(result.artifacts["markdown"]).exists()
    assert Path(result.artifacts["selected_mix_wav"]).exists()
    assert result.decision.selected.candidate.name


def test_candidate_with_only_planned_fx_send_cannot_be_accepted_rendered():
    config = load_production_mix_config()
    baseline = CandidateEvaluation(
        candidate=MixCandidate(config.baseline_candidate_name, "baseline"),
        analysis={},
        dimensions={},
        critic_scores={},
        aggregate_score=0.50,
        musical_score=0.50,
        technical_safety_score=0.50,
        final_score=0.50,
        primary_evidence=["rules"],
    )
    fx = MixAction(
        action_type=ACTION_FX_SEND,
        target="Lead Vox",
        channel_id=4,
        parameters=_fx_params(),
        rationale="planned only",
        stage="fx15_section_mod_space",
    )
    candidate = CandidateEvaluation(
        candidate=MixCandidate("planned_fx_only", "space", actions=[fx]),
        analysis={},
        dimensions={},
        critic_scores={},
        aggregate_score=0.60,
        musical_score=0.60,
        technical_safety_score=0.50,
        final_score=0.60,
        score_delta=0.10,
        musical_score_delta=0.10,
        final_score_delta=0.10,
        primary_evidence=["rules"],
        safety=SafetyResult("planned_fx_only", passed=True, allowed_actions=[fx]),
        render_verification={
            "actual_offline_render": True,
            "required_actions_rendered": False,
            "required_actions_verified": False,
            "candidate_is_not_dry": False,
            "vocal_balance_ok": True,
            "stem_balance_ok": True,
            "metrics_improved": False,
        },
    )

    decision = ProductionMixDecisionEngine(config).decide([baseline, candidate], mode="offline")

    assert decision.decision_state != "accepted_rendered"
    assert "required_actions_not_rendered" in decision.rejected[0]["reasons"]


def test_compression_without_parameters_is_rejected():
    config = load_production_mix_config()
    action = MixAction(
        action_type=ACTION_COMPRESSION,
        target="Lead Vox",
        channel_id=4,
        parameters={"threshold_db": -21.0, "ratio": 1.8},
        rationale="missing params",
        stage="track_compression_recipe",
    )
    candidate = MixCandidate("bad_comp", "dynamics", actions=[action])
    baseline = {"mix": {"true_peak_dbfs": -3.0, "phase_correlation": 1.0, "dimensions": {}}, "stems": {}}
    result = ProductionSafetyGovernor(config).evaluate(
        candidate,
        candidate_analysis=baseline,
        baseline_analysis=baseline,
        mode="offline",
        dry_run=True,
    )

    assert not result.passed
    assert any(item["reason"] == "compression_parameters_missing" for item in result.blocked_actions)


def test_fx_send_without_return_route_is_rejected():
    config = load_production_mix_config()
    params = _fx_params()
    params["route_to_main"] = False
    action = MixAction(
        action_type=ACTION_FX_SEND,
        target="Lead Vox",
        channel_id=4,
        parameters=params,
        rationale="bad route",
        stage="fx15_section_mod_space",
    )
    candidate = MixCandidate("bad_fx", "space", actions=[action])
    baseline = {"mix": {"true_peak_dbfs": -3.0, "phase_correlation": 1.0, "dimensions": {}}, "stems": {}}
    result = ProductionSafetyGovernor(config).evaluate(
        candidate,
        candidate_analysis=baseline,
        baseline_analysis=baseline,
        mode="offline",
        dry_run=True,
    )

    assert not result.passed
    assert any(item["reason"] == "fx_send_return_not_routed_to_main" for item in result.blocked_actions)


def test_combined_candidate_without_fader_balance_is_rejected():
    config = load_production_mix_config()
    action = MixAction(
        action_type=ACTION_FX_SEND,
        target="Lead Vox",
        channel_id=4,
        parameters=_fx_params(),
        rationale="space",
        stage="fx15_section_mod_space",
    )
    candidate = MixCandidate("candidate_075_combined_musical_mix", "combined", actions=[action])
    baseline = {"mix": {"true_peak_dbfs": -3.0, "phase_correlation": 1.0, "dimensions": {}}, "stems": {}}
    result = ProductionSafetyGovernor(config).evaluate(
        candidate,
        candidate_analysis=baseline,
        baseline_analysis=baseline,
        mode="offline",
        dry_run=True,
    )

    assert not result.passed
    assert any(
        item["reason"] == "combined_candidate_missing_fader_balance"
        for item in result.blocked_actions
    )


def test_dry_mix_penalty_rejects_fake_fx_candidate():
    config = load_production_mix_config()
    baseline_candidate = MixCandidate(config.baseline_candidate_name, "baseline")
    fx = MixAction(
        action_type=ACTION_FX_SEND,
        target="Lead Vox",
        channel_id=4,
        parameters=_fx_params(),
        rationale="fake fx",
        stage="fx15_section_mod_space",
    )
    baseline = CandidateEvaluation(
        candidate=baseline_candidate,
        analysis={},
        dimensions={},
        critic_scores={},
        aggregate_score=0.50,
        musical_score=0.50,
        technical_safety_score=0.50,
        final_score=0.50,
        primary_evidence=["rules"],
    )
    fake = CandidateEvaluation(
        candidate=MixCandidate("fake_fx", "space", actions=[fx]),
        analysis={},
        dimensions={},
        critic_scores={},
        aggregate_score=0.70,
        musical_score=0.70,
        technical_safety_score=0.50,
        final_score=0.70,
        score_delta=0.20,
        musical_score_delta=0.20,
        final_score_delta=0.20,
        primary_evidence=["rules"],
        safety=SafetyResult("fake_fx", passed=True, allowed_actions=[fx]),
        render_verification={
            "actual_offline_render": True,
            "required_actions_rendered": True,
            "required_actions_verified": True,
            "candidate_is_not_dry": False,
            "vocal_balance_ok": True,
            "stem_balance_ok": True,
            "metrics_improved": False,
        },
    )

    decision = ProductionMixDecisionEngine(config).decide([baseline, fake], mode="offline")

    assert not decision.accepted
    assert "dry_mix_penalty_fx_wetness_below_threshold" in decision.rejected[0]["reasons"]


def test_overdynamic_mix_penalty_rejects_fake_compression_candidate():
    config = load_production_mix_config()
    comp = MixAction(
        action_type=ACTION_COMPRESSION,
        target="Lead Vox",
        channel_id=4,
        parameters=_compression_params(),
        rationale="fake comp",
        stage="track_compression_recipe",
    )
    baseline = CandidateEvaluation(
        candidate=MixCandidate(config.baseline_candidate_name, "baseline"),
        analysis={},
        dimensions={},
        critic_scores={},
        aggregate_score=0.50,
        musical_score=0.50,
        technical_safety_score=0.50,
        final_score=0.50,
        primary_evidence=["rules"],
    )
    fake = CandidateEvaluation(
        candidate=MixCandidate("fake_comp", "dynamics", actions=[comp]),
        analysis={},
        dimensions={},
        critic_scores={},
        aggregate_score=0.70,
        musical_score=0.70,
        technical_safety_score=0.50,
        final_score=0.70,
        score_delta=0.20,
        musical_score_delta=0.20,
        final_score_delta=0.20,
        primary_evidence=["rules"],
        safety=SafetyResult("fake_comp", passed=True, allowed_actions=[comp]),
        render_verification={
            "actual_offline_render": True,
            "required_actions_rendered": True,
            "required_actions_verified": True,
            "candidate_is_not_overdynamic": False,
            "vocal_balance_ok": True,
            "stem_balance_ok": True,
            "metrics_improved": False,
        },
    )

    decision = ProductionMixDecisionEngine(config).decide([baseline, fake], mode="offline")

    assert not decision.accepted
    assert "overdynamic_mix_compression_proxy_below_threshold" in decision.rejected[0]["reasons"]


def test_render_status_verifies_real_fx_and_compression():
    config = load_production_mix_config()
    stems = align_stems(generate_smoke_stems())
    actions = [
        MixAction(
            action_type=ACTION_FX_SEND,
            target="Lead Vox",
            channel_id=4,
            parameters=_fx_params(),
            rationale="space",
            stage="fx15_section_mod_space",
        ),
        MixAction(
            action_type=ACTION_COMPRESSION,
            target="Lead Vox",
            channel_id=4,
            parameters=_compression_params(),
            rationale="leveling",
            stage="track_compression_recipe",
        ),
    ]
    result = render_candidate_with_status(stems, MixCandidate("rendered", "test", actions=actions))

    assert result.rendered_count == 2
    assert result.verified_count == 2
    assert result.failed_count == 0


def test_dry_run_report_requires_render_verification_for_audio_improved(tmp_path: Path):
    config = load_production_mix_config()
    result = ProductionMixPipeline(config, enable_optional_critics=False).run(
        output_dir=tmp_path,
        mode="offline",
        dry_run=True,
        smoke=True,
    )
    fields = result.report_fields

    if result.decision.decision_state == "accepted_rendered":
        assert fields["rendered_actions_count"] > 0
        assert fields["verified_actions_count"] > 0
        assert result.decision.selected.render_verification["metrics_improved"] is True


def test_ambiguous_role_blocks_role_specific_processing():
    config = load_production_mix_config()
    action = MixAction(
        action_type=ACTION_GAIN,
        target="Mystery Source",
        channel_id=1,
        parameters={
            "gain_db": 1.0,
            "role": "unknown",
            "role_confidence": 0.42,
            "role_allowed_action_level": "no_op",
        },
        rationale="role specific gain should be blocked",
        stage="ayaic_balance",
    )
    candidate = MixCandidate("ambiguous_role", "test", actions=[action])
    analysis = {"mix": {"true_peak_dbfs": -6.0, "phase_correlation": 1.0, "dimensions": {}}, "stems": {}}

    result = ProductionSafetyGovernor(config).evaluate(
        candidate,
        candidate_analysis=analysis,
        baseline_analysis=analysis,
        mode="offline",
        dry_run=True,
    )

    assert not result.passed
    assert any(
        item["reason"] == "role_specific_action_blocked_by_confidence_gate"
        for item in result.blocked_actions
    )


def test_large_drum_boost_rejected_without_human_approval():
    config = load_production_mix_config()
    action = MixAction(
        action_type=ACTION_GAIN,
        target="KICK",
        channel_id=8,
        parameters={
            "gain_db": 8.0,
            "current_fader_db": -12.0,
            "role": "kick",
            "role_confidence": 1.0,
            "role_allowed_action_level": "role_specific",
        },
        rationale="unsafe old recipe boost",
        stage="role_group_balance_from_report",
    )
    candidate = MixCandidate("old_recipe_boost", "test", actions=[action])
    analysis = {"mix": {"true_peak_dbfs": -6.0, "phase_correlation": 1.0, "dimensions": {}}, "stems": {}}

    result = ProductionSafetyGovernor(config).evaluate(
        candidate,
        candidate_analysis=analysis,
        baseline_analysis=analysis,
        mode="offline",
        dry_run=True,
    )

    reasons = {item["reason"] for item in result.blocked_actions}
    assert "large_gain_change_rejected_without_human_approval" in reasons
    assert "fixed_reference_group_targets_not_allowed_for_new_song" in reasons


def test_room_cut_triggers_dryness_review():
    config = load_production_mix_config()
    baseline = _baseline_eval(config)
    action = MixAction(
        action_type=ACTION_GAIN,
        target="ROOM DR L",
        channel_id=14,
        parameters={"gain_db": -10.0, "role": "drum_room", "role_allowed_action_level": "role_specific"},
        rationale="bad room cut",
        stage="ayaic_balance",
    )
    candidate = _candidate_eval(
        config,
        MixCandidate("dry_room_cut", "test", actions=[action]),
        final_delta=0.05,
        render_verification={
            **_verified_render(),
            "dryness_penalty": 0.5,
            "candidate_is_not_dry": False,
            "natural_ambience_preservation_score": 0.4,
        },
    )

    decision = ProductionMixDecisionEngine(config).decide([baseline, candidate], mode="offline")

    assert not decision.accepted
    assert decision.decision_state == "review_required"
    assert "natural_ambience_destroyed_dryness_penalty" in decision.rejected[0]["reasons"]


def test_muq_excluded_prohibits_large_remix():
    config = load_production_mix_config()
    baseline = _baseline_eval(config)
    actions = [
        MixAction(
            action_type=ACTION_GAIN,
            target=f"Source {idx}",
            channel_id=idx,
            parameters={"gain_db": 1.6, "role": "lead_vocal", "role_allowed_action_level": "role_specific"},
            rationale="too much without MuQ",
            stage="ayaic_balance",
        )
        for idx in range(1, 3)
    ]
    candidate = _candidate_eval(
        config,
        MixCandidate("large_without_muq", "test", actions=actions),
        final_delta=0.05,
        warnings=["muq_eval_fallback_excluded_from_final_score"],
    )

    decision = ProductionMixDecisionEngine(config).decide([baseline, candidate], mode="offline")

    assert not decision.accepted
    assert "review_required_muq_excluded_large_remix" in decision.rejected[0]["reasons"]


def test_fixed_song_repa_group_targets_not_applied_to_different_song():
    config = load_production_mix_config()
    action = MixAction(
        action_type=ACTION_GAIN,
        target="TOM",
        channel_id=4,
        parameters={
            "gain_db": 3.0,
            "reason": "role_group_balance_from_report",
            "role": "toms",
            "role_allowed_action_level": "role_specific",
        },
        rationale="fixed SONG_REPA target",
        stage="role_group_balance_from_report",
    )
    candidate = MixCandidate("fixed_song_repa_template", "test", actions=[action])
    analysis = {"mix": {"true_peak_dbfs": -6.0, "phase_correlation": 1.0, "dimensions": {}}, "stems": {}}

    result = ProductionSafetyGovernor(config).evaluate(
        candidate,
        candidate_analysis=analysis,
        baseline_analysis=analysis,
        mode="offline",
        dry_run=True,
    )

    assert any(
        item["reason"] == "fixed_reference_group_targets_not_allowed_for_new_song"
        for item in result.blocked_actions
    )


def test_role_override_map_forces_correct_role(tmp_path: Path):
    import soundfile as sf

    sample_rate = 48000
    audio = np.zeros(sample_rate // 10, dtype=np.float32)
    path = tmp_path / "Playback L.wav"
    sf.write(path, audio, sample_rate)

    stems = load_audio_stems(
        tmp_path,
        role_overrides={
            "patterns": {
                "Playback L.wav": {
                    "role": "music_stem",
                    "confidence": 1.0,
                    "allowed_action_level": "universal_safe",
                }
            }
        },
    )

    assert stems[0].role == "music_stem"
    assert stems[0].role_detection is not None
    assert stems[0].role_detection.confidence == 1.0
    assert stems[0].role_detection.allowed_action_level == "universal_safe"


def test_dry_run_cannot_set_console_actions_accepted_true():
    config = load_production_mix_config()
    baseline = _baseline_eval(config)
    action = MixAction(
        action_type=ACTION_COMPRESSION,
        target="Lead Vox",
        channel_id=4,
        parameters=_compression_params(),
        rationale="small accepted render",
        stage="track_compression_recipe",
    )
    candidate = _candidate_eval(
        config,
        MixCandidate("accepted_render_only", "test", actions=[action]),
        final_delta=0.05,
    )

    decision = ProductionMixDecisionEngine(config).decide(
        [baseline, candidate],
        mode="offline",
        dry_run=True,
    )

    assert decision.accepted
    assert decision.decision_state == "offline_render_accepted"
    assert decision.offline_render_accepted is True
    assert decision.console_actions_accepted is False
    assert decision.live_ready is False


def test_candidate_with_small_final_delta_and_many_actions_goes_to_review():
    config = load_production_mix_config()
    baseline = _baseline_eval(config)
    actions = [
        MixAction(
            action_type=ACTION_FX_SEND,
            target=f"Stem {idx}",
            channel_id=idx,
            parameters=_fx_params(),
            rationale="many small actions",
            stage="fx15_section_mod_space",
        )
        for idx in range(1, 12)
    ]
    candidate = _candidate_eval(
        config,
        MixCandidate("tiny_delta_many_actions", "test", actions=actions),
        final_delta=0.006,
    )

    decision = ProductionMixDecisionEngine(config).decide([baseline, candidate], mode="offline")

    assert not decision.accepted
    assert decision.decision_state == "review_required"
    assert "review_required_small_final_delta_many_actions" in decision.rejected[0]["reasons"]


def test_fx_wetness_alone_cannot_mark_not_dry_when_ambience_destroyed():
    config = load_production_mix_config()
    baseline = _baseline_eval(config)
    fx = MixAction(
        action_type=ACTION_FX_SEND,
        target="Lead Vox",
        channel_id=4,
        parameters=_fx_params(),
        rationale="wet but dry mix",
        stage="fx15_section_mod_space",
    )
    candidate = _candidate_eval(
        config,
        MixCandidate("wet_fx_destroyed_room", "test", actions=[fx]),
        final_delta=0.05,
        render_verification={
            **_verified_render(),
            "fx_return_present": True,
            "fx_wetness_proxy": 0.10,
            "candidate_is_not_dry": False,
            "dryness_penalty": 0.4,
            "natural_ambience_preservation_score": 0.45,
        },
    )

    decision = ProductionMixDecisionEngine(config).decide([baseline, candidate], mode="offline")

    assert not decision.accepted
    reasons = decision.rejected[0]["reasons"]
    assert "dry_mix_penalty_fx_wetness_below_threshold" in reasons
    assert "natural_ambience_destroyed_dryness_penalty" in reasons


def test_muq_unavailable_micro_final_delta_defaults_to_no_change():
    config = load_production_mix_config()
    baseline = _baseline_eval(config)
    action = MixAction(
        action_type=ACTION_COMPRESSION,
        target="Ride",
        channel_id=16,
        parameters=_compression_params(),
        rationale="micro dynamics",
        stage="dynamics_control",
    )
    candidate = _candidate_eval(
        config,
        MixCandidate(
            "candidate_005_bounded_crest_control",
            "dynamics_control",
            actions=[action],
            metadata={"candidate_type": "single_fix"},
        ),
        final_delta=0.0011,
        warnings=["muq_eval_fallback_excluded_from_final_score"],
    )

    decision = ProductionMixDecisionEngine(config).decide([baseline, candidate], mode="offline")

    assert not decision.accepted
    assert decision.selected.candidate.name == config.baseline_candidate_name
    assert "review_required_muq_unavailable_below_material_final_delta" in decision.rejected[0]["reasons"]


def test_low_signal_channel_compression_is_rejected():
    config = load_production_mix_config()
    action = MixAction(
        action_type=ACTION_COMPRESSION,
        target="Ride",
        channel_id=16,
        parameters={**_compression_params(), "rms_db": -58.0, "lufs_integrated_approx": -58.7},
        rationale="do not compress silence",
        stage="dynamics_control",
    )
    candidate = MixCandidate("low_signal_comp", "dynamics", actions=[action])
    baseline = {
        "mix": {"true_peak_dbfs": -6.0, "phase_correlation": 1.0, "dimensions": {}},
        "stems": {
            "16": {
                "name": "Ride",
                "role": "ride",
                "metrics": {"rms_db": -58.0, "lufs_integrated_approx": -58.7, "crest_factor_db": 35.0},
            }
        },
    }

    result = ProductionSafetyGovernor(config).evaluate(
        candidate,
        candidate_analysis=baseline,
        baseline_analysis=baseline,
        mode="offline",
        dry_run=True,
    )

    assert not result.passed
    assert any(item["reason"] == "low_signal_action_blocked" for item in result.blocked_actions)


def test_crest_factor_conflict_causes_review():
    config = load_production_mix_config()
    baseline = _baseline_eval(config)
    action = MixAction(
        action_type=ACTION_COMPRESSION,
        target="Ride",
        channel_id=16,
        parameters={**_compression_params(), "crest_factor_db": 35.7},
        rationale="conflicting crest",
        stage="dynamics_control",
    )
    candidate = _candidate_eval(
        config,
        MixCandidate("crest_conflict", "dynamics", actions=[action], metadata={"candidate_type": "single_fix"}),
        final_delta=0.05,
        render_verification={**_verified_render(), "metrics_conflicts": [{"delta_db": 13.3}]},
    )

    decision = ProductionMixDecisionEngine(config).decide([baseline, candidate], mode="offline")

    assert not decision.accepted
    assert "metrics_conflict" in decision.rejected[0]["reasons"]


def test_fx_wetness_report_text_uses_correct_less_than_comparison():
    config = load_production_mix_config()
    stems = align_stems(generate_smoke_stems())
    baseline = MixCandidate(config.baseline_candidate_name, "baseline")
    baseline_mix, _ = render_candidate(stems, baseline)
    analysis = analyze_session(stems, baseline_mix, config)
    fx = MixAction(
        action_type=ACTION_FX_SEND,
        target="Lead Vox",
        channel_id=4,
        parameters=_fx_params(),
        rationale="space",
        stage="fx15_section_mod_space",
    )
    verification = ProductionMixPipeline(config, enable_optional_critics=False)._render_verification(
        final_candidate=MixCandidate("fake_fx", "space", actions=[fx]),
        stems=stems,
        baseline_analysis=analysis,
        candidate_analysis=analysis,
        action_statuses=[
            {
                "action": fx.to_dict(),
                "status": "verified",
                "rendered": True,
                "verified": True,
                "metrics": {"fx_wetness_proxy": 0.0, "offline_fx_return_present": True},
            }
        ],
        status_counts={
            "planned_actions_count": 1,
            "rendered_actions_count": 1,
            "verified_actions_count": 1,
            "failed_actions_count": 0,
        },
        mode="offline",
        dry_run=True,
    )
    text = verification["why_candidate_is_not_dry"]

    assert "FX wetness proxy 0.0000 < threshold 0.0020" in text
    assert "0.0000 >= threshold 0.0020" not in text


def test_reference_candidate_is_not_marked_large_remix():
    config = load_production_mix_config()
    baseline = _baseline_eval(config)
    actions = [
        MixAction(
            action_type=ACTION_FX_SEND,
            target=f"Stem {idx}",
            channel_id=idx,
            parameters=_fx_params(),
            rationale="reference taste",
            stage="fx15_section_mod_space",
        )
        for idx in range(1, 12)
    ]
    candidate = _candidate_eval(
        config,
        MixCandidate(
            "candidate_015_accepted_fx15_space_baseline",
            "musical_recipe_fx15_space",
            actions=actions,
            metadata={"candidate_type": "reference_taste"},
        ),
        final_delta=-0.003,
        warnings=["muq_eval_fallback_excluded_from_final_score"],
    )

    decision = ProductionMixDecisionEngine(config).decide([baseline, candidate], mode="offline")

    reasons = decision.rejected[0]["reasons"]
    assert "review_required_muq_excluded_large_remix" not in reasons
    assert set(reasons).issubset(
        {"farther_from_current_song", "reference_not_suitable_for_song", "below_min_score_improvement"}
    )
    assert "farther_from_current_song" in reasons


def test_metrics_improved_without_material_improvement_cannot_accept():
    config = load_production_mix_config()
    baseline = _baseline_eval(config)
    action = MixAction(
        action_type=ACTION_COMPRESSION,
        target="Lead Vox",
        channel_id=4,
        parameters=_compression_params(),
        rationale="numeric only",
        stage="dynamics_control",
    )
    candidate = _candidate_eval(
        config,
        MixCandidate("numeric_only", "dynamics", actions=[action], metadata={"candidate_type": "single_fix"}),
        final_delta=0.0011,
        warnings=["muq_eval_fallback_excluded_from_final_score"],
        render_verification={**_verified_render(), "metrics_improved": True},
    )

    decision = ProductionMixDecisionEngine(config).decide([baseline, candidate], mode="offline")

    assert not decision.accepted
    assert decision.selected.candidate.name == config.baseline_candidate_name
    assert decision.rejected[0]["material_improvement"] is False


def test_all_meaningful_candidates_blocked_selects_no_change():
    config = load_production_mix_config()
    baseline = _baseline_eval(config)
    material_action = MixAction(
        action_type=ACTION_GAIN,
        target="Kick",
        channel_id=8,
        parameters={"gain_db": 8.0, "role": "kick", "role_allowed_action_level": "role_specific"},
        rationale="unsafe material change",
        stage="ayaic_balance",
    )
    material = _candidate_eval(
        config,
        MixCandidate("unsafe_material", "balance", actions=[material_action], metadata={"candidate_type": "large_remix"}),
        final_delta=0.05,
    )
    material = CandidateEvaluation(
        candidate=material.candidate,
        analysis=material.analysis,
        dimensions=material.dimensions,
        critic_scores=material.critic_scores,
        aggregate_score=material.aggregate_score,
        musical_score=material.musical_score,
        technical_safety_score=material.technical_safety_score,
        final_score=material.final_score,
        score_delta=material.score_delta,
        musical_score_delta=material.musical_score_delta,
        final_score_delta=material.final_score_delta,
        primary_evidence=material.primary_evidence,
        safety=SafetyResult("unsafe_material", passed=False, blocked_actions=[{"reason": "unsafe"}]),
        render_verification=material.render_verification,
    )
    micro_action = MixAction(
        action_type=ACTION_COMPRESSION,
        target="Ride",
        channel_id=16,
        parameters=_compression_params(),
        rationale="micro",
        stage="dynamics_control",
    )
    micro = _candidate_eval(
        config,
        MixCandidate("safe_micro", "dynamics", actions=[micro_action], metadata={"candidate_type": "single_fix"}),
        final_delta=0.001,
        warnings=["muq_eval_fallback_excluded_from_final_score"],
    )

    decision = ProductionMixDecisionEngine(config).decide([baseline, material, micro], mode="offline")

    assert not decision.accepted
    assert decision.selected.candidate.name == config.baseline_candidate_name


def test_playback_defaults_to_universal_safe(tmp_path: Path):
    import soundfile as sf

    sample_rate = 48000
    audio = np.zeros(sample_rate // 10, dtype=np.float32)
    path = tmp_path / "Playback R.wav"
    sf.write(path, audio, sample_rate)
    stems = load_audio_stems(tmp_path, role_overrides={"patterns": {}})

    assert stems[0].role == "music_stem"
    assert stems[0].role_detection is not None
    assert stems[0].role_detection.allowed_action_level == "universal_safe"


def _compression_params():
    return {
        "threshold_db": -21.0,
        "ratio": 1.8,
        "attack_ms": 18.0,
        "release_ms": 180.0,
        "knee_db": 3.0,
        "makeup_gain_db": 0.0,
        "wet_mix_percent": 100.0,
        "target_gain_reduction_db": 0.2,
        "reason": "test compression",
        "expected_metric_change": "gain_reduction_proxy_above_threshold",
    }


def _fx_params():
    return {
        "recipe": "fx15_section_mod_space",
        "fx_slot": "FX15",
        "fx_model": "SECTION_MOD_SPACE",
        "send_level_db": -18.0,
        "return_level_db": -6.0,
        "send_mode": "post_fader",
        "route_to_main": True,
        "wet_target_percent": 6.0,
        "pre_delay_ms": 18.0,
        "decay_ms": 920.0,
        "width": 0.72,
        "reason": "test fx send",
    }


def _verified_render():
    return {
        "actual_offline_render": True,
        "required_actions_rendered": True,
        "required_actions_verified": True,
        "candidate_is_not_dry": True,
        "candidate_is_not_overdynamic": True,
        "vocal_balance_ok": True,
        "stem_balance_ok": True,
        "kick_bass_balance_ok": True,
        "metrics_improved": True,
        "fx_return_present": True,
    }


def _baseline_eval(config):
    return CandidateEvaluation(
        candidate=MixCandidate(config.baseline_candidate_name, "baseline"),
        analysis={},
        dimensions={},
        critic_scores={},
        aggregate_score=0.50,
        musical_score=0.50,
        technical_safety_score=0.50,
        final_score=0.50,
        primary_evidence=["rules"],
        render_verification={"actual_offline_render": True},
    )


def _candidate_eval(
    config,
    candidate: MixCandidate,
    *,
    final_delta: float,
    warnings: list[str] | None = None,
    render_verification: dict | None = None,
):
    del config
    return CandidateEvaluation(
        candidate=candidate,
        analysis={},
        dimensions={},
        critic_scores={},
        aggregate_score=0.50 + final_delta,
        musical_score=0.50 + final_delta,
        technical_safety_score=0.50,
        final_score=0.50 + final_delta,
        score_delta=final_delta,
        musical_score_delta=final_delta,
        final_score_delta=final_delta,
        primary_evidence=["rules"],
        warnings=list(warnings or []),
        safety=SafetyResult(candidate.name, passed=True, allowed_actions=candidate.actions),
        render_verification=render_verification or _verified_render(),
    )
