from __future__ import annotations

import numpy as np

from automixer.production_mix_v1.analyzers import analyze_audio, analyzer_issues, guitar_forwardness_from_analysis
from automixer.production_mix_v1.candidates import StagedCandidateGenerator
from automixer.production_mix_v1.config import load_production_mix_config
from automixer.production_mix_v1.decision import ProductionMixDecisionEngine
from automixer.production_mix_v1.models import (
    ACTION_FX_SEND,
    ACTION_GAIN,
    ACTION_PARALLEL_COMPRESSION,
    ROLE_ACTION_ROLE_SPECIFIC,
    ROLE_ACTION_UNIVERSAL_SAFE,
    AudioStem,
    CandidateEvaluation,
    MixAction,
    MixCandidate,
    RoleDetectionResult,
    SafetyResult,
)
from automixer.production_mix_v1.pipeline import ProductionMixPipeline
from automixer.production_mix_v1.rendering import render_candidate_with_status
from automixer.production_mix_v1.role_detection import detect_role
from automixer.production_mix_v1.safety_adapter import ProductionSafetyGovernor


def test_best_candidate_beats_reference_by_noise_reference_wins():
    config = load_production_mix_config()
    baseline = _eval(config.baseline_candidate_name, "utility", 0.30, 0.30)
    reference = _eval(config.accepted_reference_candidate_name, "reference_taste", 0.4600, 0.3900)
    best = _eval("candidate_045_parallel_compression", "drum_glue", 0.4601, 0.3901)

    decision = ProductionMixDecisionEngine(config).decide([baseline, reference, best], mode="offline")

    assert decision.selected.candidate.name == config.accepted_reference_candidate_name
    assert decision.selected_by_tiebreak is True
    assert "accepted_reference_preferred_within_epsilon" in decision.tiebreak_reason


def test_muq_unavailable_requires_001_win_margin_over_reference():
    config = load_production_mix_config()
    baseline = _eval(config.baseline_candidate_name, "utility", 0.40, 0.40)
    reference = _eval(config.accepted_reference_candidate_name, "reference_taste", 0.460, 0.390)
    candidate = _eval("candidate_045_parallel_compression", "drum_glue", 0.468, 0.398)

    decision = ProductionMixDecisionEngine(config).decide([baseline, reference, candidate], mode="offline")

    rejected = next(item for item in decision.rejected if item["candidate"] == "candidate_045_parallel_compression")
    assert "review_required_no_material_improvement_vs_reference" in rejected["reasons"]


def test_verified_reference_fx_allows_small_offline_delta_without_optional_critics():
    config = load_production_mix_config()
    baseline = _eval(config.baseline_candidate_name, "utility", 0.5000, 0.5000)
    reference_action = MixAction(
        ACTION_FX_SEND,
        "Vocal",
        1,
        {"role": "lead_vocal", "role_allowed_action_level": ROLE_ACTION_ROLE_SPECIFIC},
        "verified reference fx",
        "fx15_section_mod_space",
    )
    reference = CandidateEvaluation(
        candidate=MixCandidate(config.accepted_reference_candidate_name, "test", [reference_action], metadata={"candidate_type": "reference_taste"}),
        analysis={},
        dimensions={},
        critic_scores={},
        aggregate_score=0.5012,
        final_score=0.5012,
        musical_score=0.5011,
        technical_safety_score=0.5,
        primary_evidence=["rules"],
        warnings=["optional_critics_disabled"],
        safety=SafetyResult(config.accepted_reference_candidate_name, passed=True, allowed_actions=[reference_action]),
        render_verification={
            "actual_offline_render": True,
            "offline_fx_return_present": True,
            "spatial_fx_verified": True,
            "modulation_fx_verified": True,
            "pre_trim_headroom_rejection": "none",
            "metrics_conflicts": [],
        },
    )

    decision = ProductionMixDecisionEngine(config).decide([baseline, reference], mode="offline")

    assert decision.selected.candidate.name == config.accepted_reference_candidate_name
    assert decision.accepted is True


def test_parallel_compression_on_overheads_triggers_review():
    config = load_production_mix_config()
    action = MixAction(
        ACTION_PARALLEL_COMPRESSION,
        "OH L",
        9,
        {**_comp_params(), "role": "overheads", "role_allowed_action_level": "role_specific"},
        "bad overhead compression",
        "parallel_compression",
    )
    candidate = MixCandidate("bad_oh", "test", [action], metadata={"candidate_type": "drum_glue"})
    analysis = {"mix": {"true_peak_dbfs": -6.0}, "stems": {"9": {"metrics": {}}}}

    result = ProductionSafetyGovernor(config).evaluate(candidate, candidate_analysis=analysis, baseline_analysis=analysis, mode="offline", dry_run=True)

    assert not result.passed
    assert any(item["reason"] == "review_required_parallel_compression_on_overheads" for item in result.blocked_actions)


def test_drum_glue_with_bass_requires_rhythm_section_glue():
    config = load_production_mix_config()
    action = MixAction(
        ACTION_PARALLEL_COMPRESSION,
        "Bass",
        4,
        {**_comp_params(), "role": "bass", "role_allowed_action_level": "role_specific"},
        "bass in drum glue",
        "parallel_compression",
    )
    candidate = MixCandidate("bad_bass_glue", "test", [action], metadata={"candidate_type": "drum_glue"})
    baseline = {"mix": {"true_peak_dbfs": -6.0}, "stems": {"4": {"metrics": {}}}}

    result = ProductionSafetyGovernor(config).evaluate(candidate, candidate_analysis=baseline, baseline_analysis=baseline, mode="offline", dry_run=True)

    assert not result.passed
    assert any(item["reason"] == "drum_glue_contains_bass_requires_rhythm_section_glue" for item in result.blocked_actions)


def test_pre_trim_true_peak_plus_18_hard_rejects_candidate():
    config = load_production_mix_config()
    baseline = _eval(config.baseline_candidate_name, "utility", 0.40, 0.40)
    reference = _eval(config.accepted_reference_candidate_name, "reference_taste", 0.43, 0.43)
    candidate = _eval(
        "overshoot",
        "drum_glue",
        0.60,
        0.60,
        verification={"pre_trim_true_peak_dbfs": 18.12, "pre_trim_headroom_rejection": "hard_reject"},
    )

    decision = ProductionMixDecisionEngine(config).decide([baseline, reference, candidate], mode="offline")

    rejected = next(item for item in decision.rejected if item["candidate"] == "overshoot")
    assert "internal_pre_trim_true_peak_hard_reject" in rejected["reasons"]


def test_material_vs_no_change_but_not_reference_cannot_auto_accept():
    config = load_production_mix_config()
    baseline = _eval(config.baseline_candidate_name, "utility", 0.40, 0.40)
    reference = _eval(config.accepted_reference_candidate_name, "reference_taste", 0.55, 0.55)
    candidate = _eval("beats_no_change_only", "drum_glue", 0.51, 0.51)

    decision = ProductionMixDecisionEngine(config).decide([baseline, reference, candidate], mode="offline")

    rejected = next(item for item in decision.rejected if item["candidate"] == "beats_no_change_only")
    assert "review_required_no_material_improvement_vs_reference" in rejected["reasons"]


def test_spectral_centroid_zero_with_non_silent_rms_triggers_analyzer_failure():
    metrics = {
        "rms_db": -24.0,
        "crest_factor_db": 24.0,
        "spectral_centroid_hz": 0.0,
        "band_ratios": {"sub": 0.0, "punch": 0.0, "mud": 0.0},
    }

    issues = analyzer_issues(metrics, role="music_stem")

    assert "analyzer_failure_or_silence_mismatch" in issues
    assert "analyzer_failure" in issues
    assert "low_signal_or_metric_error" in issues


def test_analyzer_uses_active_audio_not_trailing_silence():
    config = load_production_mix_config()
    sample_rate = 48000
    t = np.linspace(0, 1.0, sample_rate, endpoint=False, dtype=np.float32)
    tone = np.sin(2 * np.pi * 440.0 * t).astype(np.float32) * 0.1
    trailing_silence = np.zeros(sample_rate * 2, dtype=np.float32)
    audio = np.concatenate([tone, trailing_silence])

    metrics = analyze_audio(audio, sample_rate, config)

    assert metrics["spectral_centroid_hz"] > 0.0
    assert "analyzer_failure_or_silence_mismatch" not in metrics["analyzer_issues"]


def test_analyzer_failure_blocks_role_specific_action():
    config = load_production_mix_config()
    action = MixAction(
        ACTION_PARALLEL_COMPRESSION,
        "Vocal",
        1,
        {**_comp_params(), "role": "lead_vocal", "role_allowed_action_level": "role_specific"},
        "blocked",
        "parallel_compression",
    )
    candidate = MixCandidate("bad_analyzer", "test", [action], metadata={"candidate_type": "vocal_depth"})
    analysis = {
        "mix": {"true_peak_dbfs": -6.0},
        "stems": {"1": {"metrics": {"analyzer_issues": ["analyzer_failure"], "rms_db": -24.0}}},
    }

    result = ProductionSafetyGovernor(config).evaluate(candidate, candidate_analysis=analysis, baseline_analysis=analysis, mode="offline", dry_run=True)

    assert not result.passed
    assert any(item["reason"] == "analyzer_failure_blocks_role_specific_action" for item in result.blocked_actions)


def test_playback_defaults_to_universal_safe_role_level():
    config = load_production_mix_config()
    audio = np.sin(np.linspace(0, 10, 4800, dtype=np.float32)) * 0.1

    result = detect_role(filename="Playback L.wav", channel_name="Playback L", audio=audio, sample_rate=48000, config=config, overrides={"patterns": {}})

    assert result.role == "music_stem"
    assert result.allowed_action_level == "universal_safe"


def test_fx_send_renders_spatial_and_modulation_metrics():
    sample_rate = 48000
    audio = np.sin(np.linspace(0, 4 * np.pi, sample_rate, dtype=np.float32)) * 0.1
    detection = RoleDetectionResult("lead_vocal", 1.0, allowed_action_level=ROLE_ACTION_ROLE_SPECIFIC)
    stem = AudioStem("Vocal", audio=audio, sample_rate=sample_rate, role="lead_vocal", channel_id=1, role_detection=detection)
    action = MixAction(
        ACTION_FX_SEND,
        "Vocal",
        1,
        {
            "send_level_db": -18.0,
            "return_level_db": -6.0,
            "pre_delay_ms": 18.0,
            "decay_ms": 920.0,
            "width": 0.72,
            "modulation_depth_ms": 4.5,
            "modulation_rate_hz": 0.32,
            "role": "lead_vocal",
            "role_allowed_action_level": ROLE_ACTION_ROLE_SPECIFIC,
        },
        "test fx",
        "fx15_section_mod_space",
    )

    result = render_candidate_with_status([stem], MixCandidate("fx", "test", [action], metadata={"candidate_type": "reference_taste"}))

    metrics = result.action_statuses[0].metrics
    assert result.action_statuses[0].verified
    assert metrics["offline_fx_return_present"] is True
    assert metrics["spatial_reflection_energy_proxy"] > 0.001
    assert metrics["modulation_energy_proxy"] > 0.00015
    assert metrics["stereo_width_delta_proxy"] > 0.0


def test_reference_fx_allows_universal_safe_backing_vocal_send():
    config = load_production_mix_config()
    sample_rate = 48000
    audio = np.sin(np.linspace(0, 4 * np.pi, sample_rate, dtype=np.float32)) * 0.1
    detection = RoleDetectionResult("backing_vocal", 0.84, is_ambiguous=True, allowed_action_level=ROLE_ACTION_UNIVERSAL_SAFE)
    stem = AudioStem("Back Vox L", audio=audio, sample_rate=sample_rate, role="backing_vocal", channel_id=1, role_detection=detection)
    baseline_analysis = {"mix": {"true_peak_dbfs": -6.0, "band_ratios": {}}, "stems": {"1": {"metrics": {"rms_db": -20.0, "lufs_integrated_approx": -20.0, "crest_factor_db": 8.0, "analyzer_issues": []}}}}

    candidates = StagedCandidateGenerator(config).generate([stem], baseline_analysis, mode="offline")

    reference = next(item for item in candidates if item.name == config.accepted_reference_candidate_name)
    fx_targets = [action.target for action in reference.actions if action.action_type == ACTION_FX_SEND]
    assert fx_targets == ["Back Vox L"]


def test_guitars_too_forward_generates_candidate_066():
    config = load_production_mix_config()
    stems = [_stem("Guitar L", "electric_guitar", 1), _stem("Guitar R", "electric_guitar", 2), _stem("Vocal", "lead_vocal", 3)]
    analysis = _guitar_masking_analysis()

    candidates = StagedCandidateGenerator(config).generate(stems, analysis, mode="offline")

    assert guitar_forwardness_from_analysis(analysis, config)["guitars_too_forward"] is True
    assert any(candidate.name == "candidate_066_fx15_reference_with_guitar_control" for candidate in candidates)


def test_candidate_066_inherits_fx15_air_reference():
    config = load_production_mix_config()
    stems = [_stem("Guitar L", "electric_guitar", 1), _stem("Guitar R", "electric_guitar", 2), _stem("Vocal", "lead_vocal", 3)]
    candidate = next(candidate for candidate in StagedCandidateGenerator(config).generate(stems, _guitar_masking_analysis(), mode="offline") if candidate.name == "candidate_066_fx15_reference_with_guitar_control")

    stages = [action.stage for action in candidate.actions]
    assert candidate.metadata["inherits_reference_candidate"] == config.accepted_reference_candidate_name
    assert "fx15_section_mod_space" in stages
    assert "final_air_minus_080" in stages
    assert any(action.parameters.get("guitar_control") for action in candidate.actions)


def test_guitar_trim_capped_at_minus_1_5_db_in_reference_mode():
    config = load_production_mix_config()
    config.reference_recipe.setdefault("guitar_control", {})["default_trim_db"] = -3.0
    stems = [_stem("Guitar L", "electric_guitar", 1), _stem("Guitar R", "electric_guitar", 2), _stem("Vocal", "lead_vocal", 3)]
    candidate = next(candidate for candidate in StagedCandidateGenerator(config).generate(stems, _guitar_masking_analysis(), mode="offline") if candidate.name == "candidate_066_fx15_reference_with_guitar_control")

    trims = [action.parameters["gain_db"] for action in candidate.actions if action.parameters.get("guitar_control")]

    assert trims == [-1.5, -1.5]


def test_no_lead_vocal_means_no_guitar_ducking():
    config = load_production_mix_config()
    stems = [_stem("Guitar L", "electric_guitar", 1), _stem("Guitar R", "electric_guitar", 2)]
    analysis = _guitar_masking_analysis(include_vocal=False)

    candidates = StagedCandidateGenerator(config).generate(stems, analysis, mode="offline")

    assert not any(candidate.name == "candidate_066_fx15_reference_with_guitar_control" for candidate in candidates)


def test_low_guitar_confidence_blocks_guitar_specific_action():
    config = load_production_mix_config()
    stems = [_stem("Guitar L", "electric_guitar", 1, confidence=0.80), _stem("Guitar R", "electric_guitar", 2), _stem("Vocal", "lead_vocal", 3)]

    candidates = StagedCandidateGenerator(config).generate(stems, _guitar_masking_analysis(), mode="offline")

    assert not any(candidate.name == "candidate_066_fx15_reference_with_guitar_control" for candidate in candidates)


def test_candidate_066_wins_over_065_when_masking_improves():
    config = load_production_mix_config()
    baseline = _eval(config.baseline_candidate_name, "utility", 0.300, 0.300)
    reference = _eval(config.accepted_reference_candidate_name, "reference_taste", 0.500, 0.500)
    control = _eval(
        "candidate_066_fx15_reference_with_guitar_control",
        "reference_taste_guitar_control",
        0.497,
        0.497,
        verification={"guitar_forwardness": _guitar_verification(before=0.82, after=0.72)},
    )

    decision = ProductionMixDecisionEngine(config).decide([baseline, reference, control], mode="offline")

    assert decision.selected.candidate.name == "candidate_066_fx15_reference_with_guitar_control"
    assert "guitar_control_preferred_over_reference" in decision.tiebreak_reason


def test_candidate_066_rejected_if_guitar_power_loss_too_high():
    config = load_production_mix_config()
    baseline = _eval(config.baseline_candidate_name, "utility", 0.300, 0.300)
    reference = _eval(config.accepted_reference_candidate_name, "reference_taste", 0.500, 0.500)
    control = _eval(
        "candidate_066_fx15_reference_with_guitar_control",
        "reference_taste_guitar_control",
        0.600,
        0.600,
        verification={"guitar_forwardness": _guitar_verification(before=0.82, after=0.50, power_loss=True)},
    )

    decision = ProductionMixDecisionEngine(config).decide([baseline, reference, control], mode="offline")

    rejected = next(item for item in decision.rejected if item["candidate"] == "candidate_066_fx15_reference_with_guitar_control")
    assert "guitar_power_loss_detected" in rejected["reasons"]


def test_smoke_report_contains_tiebreak_fields(tmp_path):
    config = load_production_mix_config()
    result = ProductionMixPipeline(config, enable_optional_critics=False).run(output_dir=tmp_path, smoke=True, dry_run=True)

    assert "win_margin_over_reference_final" in result.report_fields
    assert "pre_trim_true_peak_dbfs" in result.report_fields
    assert result.queue_result.sent == []


def _eval(name: str, candidate_type: str, final_score: float, musical_score: float, *, verification=None) -> CandidateEvaluation:
    action = MixAction(
        ACTION_PARALLEL_COMPRESSION,
        "Kick",
        1,
        {**_comp_params(), "role": "kick", "role_allowed_action_level": "role_specific"},
        "test",
        "parallel_compression",
    )
    actions = [] if name == "candidate_000_no_change" else [action]
    candidate = MixCandidate(name, "test", actions, metadata={"candidate_type": candidate_type})
    return CandidateEvaluation(
        candidate=candidate,
        analysis={},
        dimensions={},
        critic_scores={},
        aggregate_score=final_score,
        final_score=final_score,
        musical_score=musical_score,
        technical_safety_score=0.5,
        primary_evidence=["rules"],
        warnings=["optional_critics_disabled"],
        safety=SafetyResult(name, passed=True, allowed_actions=actions),
        render_verification={"actual_offline_render": True, "kick_bass_balance_ok": True, **(verification or {})},
    )


def _stem(name: str, role: str, channel_id: int, *, confidence: float = 0.92) -> AudioStem:
    detection = RoleDetectionResult(role, confidence, allowed_action_level=ROLE_ACTION_ROLE_SPECIFIC)
    return AudioStem(name, audio=np.zeros(2048, dtype=np.float32), sample_rate=48000, role=role, channel_id=channel_id, role_detection=detection)


def _guitar_masking_analysis(*, include_vocal: bool = True) -> dict:
    stems = {
        "1": _analysis_stem("Guitar L", "electric_guitar", intelligibility=55.0, presence=56.0),
        "2": _analysis_stem("Guitar R", "electric_guitar", intelligibility=55.0, presence=56.0),
    }
    if include_vocal:
        stems["3"] = _analysis_stem("Vocal", "lead_vocal", intelligibility=63.0, presence=46.0)
    return {"mix": {"true_peak_dbfs": -6.0, "band_ratios": {}}, "stems": stems}


def _analysis_stem(name: str, role: str, *, intelligibility: float, presence: float) -> dict:
    return {
        "name": name,
        "role": role,
        "metrics": {
            "rms_db": -24.0,
            "lufs_integrated_approx": -24.7,
            "crest_factor_db": 12.0,
            "analyzer_issues": [],
            "band_energy_db": {
                "intelligibility": intelligibility,
                "presence": presence,
            },
        },
    }


def _guitar_verification(*, before: float, after: float, power_loss: bool = False) -> dict:
    return {
        "guitars_too_forward": True,
        "guitar_vocal_masking_index_before": before,
        "guitar_vocal_masking_index_after": after,
        "guitar_masking_improved": after < before,
        "reference_preserved": True,
        "guitar_power_loss_detected": power_loss,
    }


def _comp_params():
    return {
        "threshold_db": -24.0,
        "ratio": 3.2,
        "attack_ms": 10.0,
        "release_ms": 220.0,
        "knee_db": 4.0,
        "makeup_gain_db": 0.0,
        "wet_mix_percent": 22.0,
        "target_gain_reduction_db": 1.0,
        "reason": "test compression",
        "expected_metric_change": "density",
    }
