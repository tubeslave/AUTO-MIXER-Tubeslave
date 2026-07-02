from pathlib import Path

import numpy as np

from backend.smart_dsp_evidence_extractor import (
    SMART_DSP_EVIDENCE_EXTRACTOR_VERSION,
    analyze_smart_dsp_audio,
    build_current_wing_state_from_readonly_preflight,
    build_smart_dsp_analysis_input_from_audio,
    extract_smart_dsp_audio_evidence,
)
from backend.smart_dsp_strategy_engine import (
    WING_DE_ESSER_GATE_MODEL,
    WING_DE_ESSER_GATE_SLOT_CONFLICT_POLICY,
    WING_DYNAMIC_EQ_GATE_MODEL,
    WING_DYNAMICS_MODEL_READBACK_BLOCKER,
    WING_DYNAMICS_REQUIRED_MODEL,
    WING_EQ_FILTER_OPERATOR_OWNED_BLOCKER,
    WING_EQ_MODEL_READBACK_BLOCKER,
    WING_EQ_OPERATOR_OWNED_MODEL_BLOCKER,
    WING_EQ_MIX_BASELINE_PERCENT,
    WING_EQ_REQUIRED_MODEL,
    WING_FX_LEVEL_OPERATOR_OWNED_BLOCKER,
    WING_FX_OPERATOR_OWNED_MODEL_BLOCKER,
    WING_GAIN_UP_FADER_CEILING_BLOCKER,
    WING_GAIN_UP_FADER_READBACK_BLOCKER,
    WING_GAIN_OPERATOR_OWNED_BLOCKER,
    WING_GATE_REQUIRED_MODEL,
    WING_GATE_SLOT_MODEL_CONFLICT_BLOCKER,
    WING_GATE_SLOT_OPERATOR_OWNED_MODEL_BLOCKER,
    analyze_smart_dsp_strategy,
)


SR = 48000


def _fresh(fields=12):
    return {
        "status": "fresh",
        "field_count": fields,
        "fresh_field_count": fields,
        "read_only": True,
        "live_write": False,
        "queries_console": False,
    }


def _tone(frequency_hz, *, amplitude=0.25, duration_sec=2.0, phase=0.0):
    time = np.arange(int(SR * duration_sec), dtype=np.float64) / SR
    return amplitude * np.sin(2.0 * np.pi * frequency_hz * time + phase)


def _band_limited_noise(
    low_hz,
    high_hz,
    *,
    amplitude=0.1,
    duration_sec=2.0,
    seed=4,
):
    rng = np.random.default_rng(seed)
    size = int(SR * duration_sec)
    samples = rng.normal(size=size)
    spectrum = np.fft.rfft(samples)
    freqs = np.fft.rfftfreq(size, 1.0 / SR)
    spectrum[(freqs < low_hz) | (freqs > high_hz)] = 0.0
    filtered = np.fft.irfft(spectrum, n=size)
    filtered = filtered / max(float(np.max(np.abs(filtered))), 1e-12)
    return amplitude * filtered


def _snare_transient_train(*, duration_sec=2.4):
    rng = np.random.default_rng(12)
    samples = np.zeros(int(SR * duration_sec), dtype=np.float64)
    for start_sec in np.arange(0.16, duration_sec - 0.12, 0.34):
        start = int(SR * start_sec)
        length = int(SR * 0.075)
        envelope = np.exp(-np.linspace(0.0, 7.0, length))
        shell = _tone(190.0, amplitude=0.52, duration_sec=length / SR)
        snap = rng.normal(scale=0.16, size=length)
        samples[start : start + length] += envelope * (shell + snap)
    return np.clip(samples, -0.95, 0.95)


def _snare_ring_train(*, duration_sec=2.4):
    rng = np.random.default_rng(31)
    samples = np.zeros(int(SR * duration_sec), dtype=np.float64)
    for start_sec in np.arange(0.12, duration_sec - 0.35, 0.55):
        start = int(SR * start_sec)
        length = int(SR * 0.42)
        local_time = np.arange(length, dtype=np.float64) / SR
        envelope = np.exp(-np.linspace(0.0, 3.2, length))
        ring = np.sin(2.0 * np.pi * 420.0 * local_time)
        body = 0.25 * np.sin(2.0 * np.pi * 210.0 * local_time)
        snap = np.zeros(length, dtype=np.float64)
        snap_length = int(SR * 0.025)
        snap[:snap_length] = rng.normal(scale=0.08, size=snap_length)
        samples[start : start + length] += envelope * (0.28 * ring + 0.06 * body) + snap
    return np.clip(samples, -0.95, 0.95)


def _kick_burst_train(*, duration_sec=2.4):
    samples = np.zeros(int(SR * duration_sec), dtype=np.float64)
    for start_sec in np.arange(0.12, duration_sec - 0.14, 0.46):
        start = int(SR * start_sec)
        length = int(SR * 0.13)
        local_time = np.arange(length, dtype=np.float64) / SR
        envelope = np.exp(-np.linspace(0.0, 8.0, length))
        thump = np.sin(2.0 * np.pi * 74.0 * local_time)
        body = 0.35 * np.sin(2.0 * np.pi * 118.0 * local_time)
        samples[start : start + length] += 0.7 * envelope * (thump + body)
    return np.clip(samples, -0.95, 0.95)


def test_smart_dsp_evidence_extractor_module_is_transport_free():
    source = Path("backend/smart_dsp_evidence_extractor.py").read_text()

    forbidden = [
        "WingClient",
        "EnhancedOSCClient",
        ".send(",
        "mixer_client",
        "connect_wing",
        "connect_mixer",
        "apply_operator_proposal",
    ]
    for token in forbidden:
        assert token not in source


def test_muddy_vocal_audio_builds_evidence_and_selects_low_mid_cut():
    samples = _tone(260.0, amplitude=0.12) + _tone(1800.0, amplitude=0.015)

    evidence = extract_smart_dsp_audio_evidence(
        samples,
        target_id="ch:5",
        source_role="lead_vocal",
        source_label="Lead Vox",
        sample_rate=SR,
    )

    assert evidence.version == SMART_DSP_EVIDENCE_EXTRACTOR_VERSION
    assert evidence.read_only is True
    assert evidence.live_write is False
    assert evidence.transport_free is True
    assert evidence.muddiness_score >= 0.45
    assert "muddy" in evidence.problem_hints
    assert evidence.true_peak_headroom_db > 1.0
    assert evidence.band_energy_share["low_mid"] > evidence.band_energy_share["presence"]

    analysis = build_smart_dsp_analysis_input_from_audio(
        samples,
        target_id="ch:5",
        source_role="lead_vocal",
        source_label="Lead Vox",
        sample_rate=SR,
        source_identity_confirmed=True,
        source_identity_confidence=0.91,
        current_state_cache_freshness=_fresh(),
        current_state_readback_available=True,
        rollback_plan_available=True,
    )
    recommendation = analyze_smart_dsp_strategy(analysis)

    assert recommendation.status == "proposal_only"
    assert recommendation.evidence["detected_problems"] == ["muddy"]
    assert recommendation.selected_strategy.strategy_id == "low_mid_corrective_cut"
    assert recommendation.selected_strategy.parameters["gain_db"] < 0.0


def test_audio_analysis_default_freshness_counts_all_extracted_evidence_fields():
    samples = _tone(260.0, amplitude=0.12) + _tone(1800.0, amplitude=0.015)

    evidence = extract_smart_dsp_audio_evidence(
        samples,
        target_id="ch:5",
        source_role="lead_vocal",
        source_label="Lead Vox",
        sample_rate=SR,
    )
    analysis = build_smart_dsp_analysis_input_from_audio(
        samples,
        target_id="ch:5",
        source_role="lead_vocal",
        source_label="Lead Vox",
        sample_rate=SR,
        source_identity_confirmed=True,
        source_identity_confidence=0.91,
        current_state_readback_available=True,
        rollback_plan_available=True,
    )

    freshness = analysis.current_state_cache_freshness
    assert freshness["status"] == "fresh"
    assert freshness["source"] == SMART_DSP_EVIDENCE_EXTRACTOR_VERSION
    assert freshness["field_count"] == len(evidence.to_dict())
    assert freshness["fresh_field_count"] == freshness["field_count"]
    assert freshness["field_count"] > 12
    assert freshness["stale_field_count"] == 0
    assert freshness["missing_field_count"] == 0
    assert freshness["read_only"] is True
    assert freshness["live_write"] is False
    assert freshness["queries_console"] is False


def test_measured_low_mid_peak_drives_muddy_eq_frequency():
    samples = _tone(340.0, amplitude=0.12) + _tone(1800.0, amplitude=0.018)

    recommendation = analyze_smart_dsp_audio(
        samples,
        target_id="ch:5",
        source_role="lead_vocal",
        source_label="Lead Vox",
        sample_rate=SR,
        source_identity_confirmed=True,
        source_identity_confidence=0.91,
        current_state_cache_freshness=_fresh(),
        current_state_readback_available=True,
        rollback_plan_available=True,
        target_plane_lufs=-22.0,
    )

    audio_evidence = recommendation.evidence["audio_evidence"]
    assert audio_evidence["muddiness_score"] >= 0.45
    assert 335.0 <= audio_evidence["mud_frequency_hz"] <= 345.0
    assert recommendation.selected_strategy.strategy_id == "low_mid_corrective_cut"
    params = recommendation.selected_strategy.parameters
    assert params["frequency_source"] == "measured_low_mid_peak"
    assert 335.0 <= params["frequency_hz"] <= 345.0
    assert params["mud_frequency_hz"] == audio_evidence["mud_frequency_hz"]
    assert recommendation.evidence["mud_frequency_hz"] == audio_evidence["mud_frequency_hz"]
    assert recommendation.to_operator_proposal()["live_apply_allowed"] is False


def test_measured_sibilance_band_drives_wing_native_ds902_de_esser_parameters():
    duration_sec = 2.4
    time = np.arange(int(SR * duration_sec), dtype=np.float64) / SR
    sibilance_envelope = np.zeros_like(time)
    for start_sec in (0.28, 0.82, 1.36, 1.9):
        start = int(SR * start_sec)
        length = int(SR * 0.065)
        sibilance_envelope[start : start + length] += np.hanning(length)
    samples = (
        0.026 * np.sin(2.0 * np.pi * 1400.0 * time)
        + 0.012 * np.sin(2.0 * np.pi * 260.0 * time)
        + 0.19 * sibilance_envelope * np.sin(2.0 * np.pi * 7600.0 * time)
    )

    recommendation = analyze_smart_dsp_audio(
        samples,
        target_id="ch:5",
        source_role="lead_vocal",
        source_label="Lead Vox",
        sample_rate=SR,
        source_identity_confirmed=True,
        source_identity_confidence=0.91,
        current_state_cache_freshness=_fresh(),
        current_state_readback_available=True,
        rollback_plan_available=True,
        target_plane_lufs=-22.0,
    )

    audio_evidence = recommendation.evidence["audio_evidence"]
    assert audio_evidence["sibilance_score"] >= 0.45
    assert 7550.0 <= audio_evidence["sibilance_frequency_hz"] <= 7650.0
    assert audio_evidence["sibilance_bandwidth_hz"] > 0.0
    assert audio_evidence["sibilance_event_peak_dbfs"] < 0.0
    assert audio_evidence["sibilance_event_contrast_db"] >= 3.0
    assert audio_evidence["sibilance_event_rise_time_ms"] > 0.0
    assert audio_evidence["sibilance_event_recovery_ms"] > 0.0
    assert recommendation.selected_strategy.strategy_id == "vocal_de_esser"
    assert recommendation.selected_strategy.method_family == "de_esser"
    params = recommendation.selected_strategy.parameters
    assert params["frequency_source"] == "measured_sibilance_peak"
    assert params["bandwidth_source"] == "measured_sibilance_bandwidth"
    assert params["band_start_hz"] < params["frequency_hz"] < params["band_end_hz"]
    assert params["bandwidth_hz"] >= 900.0
    assert params["range_db"] == params["max_gain_reduction_db"]
    assert params["threshold_source"] == "measured_sibilance_event"
    assert params["de_esser_parameter_source"] == "measured_sibilance_event"
    assert params["wing_de_esser_model_osc_value"] == WING_DE_ESSER_GATE_MODEL
    assert params["wing_de_esser_frequency_osc_value"] == round(
        audio_evidence["sibilance_frequency_hz"],
        3,
    )
    assert params["wing_de_esser_range_osc_value"] == params["range_db"]
    assert params["wing_de_esser_mode_osc_value"] == "HF"
    assert params["wing_de_esser_gate_slot_conflict_policy"] == (
        WING_DE_ESSER_GATE_SLOT_CONFLICT_POLICY
    )
    assert params["measured_sibilance_bandwidth_hz"] == audio_evidence[
        "sibilance_bandwidth_hz"
    ]
    assert params["sibilance_event_peak_dbfs"] == audio_evidence[
        "sibilance_event_peak_dbfs"
    ]
    assert params["sibilance_event_contrast_db"] == audio_evidence[
        "sibilance_event_contrast_db"
    ]
    alternatives = {
        candidate.strategy_id: candidate for candidate in recommendation.alternatives
    }
    static_eq = alternatives["static_sibilance_band_cut"]
    assert static_eq.method_family == "eq"
    assert static_eq.parameters["static_fallback_for_unmapped_de_esser"] is True
    assert static_eq.parameters["post_static_cut_sibilance_recheck_required"] is True
    assert static_eq.parameters["sibilance_parameter_source"] == (
        "measured_sibilance_event"
    )
    readback = recommendation.selected_strategy.future_wing_readback_contract
    assert "eq_band_readback" not in readback
    assert "dynamics_readback" not in readback
    assert readback["de_esser_readback"]["gate_model_readback_must_equal"] == (
        WING_DE_ESSER_GATE_MODEL
    )
    assert recommendation.evidence["sibilance_bandwidth_hz"] == audio_evidence[
        "sibilance_bandwidth_hz"
    ]
    assert recommendation.evidence["sibilance_event_peak_dbfs"] == audio_evidence[
        "sibilance_event_peak_dbfs"
    ]
    assert recommendation.to_operator_proposal()["live_apply_allowed"] is False


def test_rumble_dominant_vocal_audio_selects_high_pass_before_low_mid_cut():
    samples = (
        _tone(82.0, amplitude=0.085)
        + _tone(270.0, amplitude=0.01)
        + _tone(1400.0, amplitude=0.03)
        + _tone(3200.0, amplitude=0.03)
        + _tone(12000.0, amplitude=0.075)
    )

    recommendation = analyze_smart_dsp_audio(
        samples,
        target_id="ch:5",
        source_role="lead_vocal",
        source_label="Lead Vox",
        sample_rate=SR,
        source_identity_confirmed=True,
        source_identity_confidence=0.91,
        current_state_cache_freshness=_fresh(),
        current_state_readback_available=True,
        rollback_plan_available=True,
        target_plane_lufs=-22.0,
    )

    audio_evidence = recommendation.evidence["audio_evidence"]
    assert audio_evidence["low_end_excess_score"] >= 0.78
    assert audio_evidence["source_identity_mismatch_score"] == 0.0
    assert audio_evidence["problem_hints"] == ["muddy"]
    assert recommendation.selected_strategy.strategy_id == "rumble_dominant_high_pass_cleanup"
    assert recommendation.selected_strategy.parameters["filter_type"] == "high_pass"
    assert recommendation.selected_strategy.parameters["hpf_before_low_mid_cut"] is True
    alternatives = {candidate.strategy_id for candidate in recommendation.alternatives}
    assert "low_mid_corrective_cut" in alternatives


def test_plosive_vocal_audio_selects_wing_native_dynamic_deq_hpf():
    duration_sec = 3.0
    time = np.arange(int(SR * duration_sec), dtype=np.float64) / SR
    samples = (
        0.055 * np.sin(2.0 * np.pi * 220.0 * time)
        + 0.05 * np.sin(2.0 * np.pi * 1350.0 * time)
        + 0.055 * np.sin(2.0 * np.pi * 3200.0 * time)
        + 0.025 * np.sin(2.0 * np.pi * 11500.0 * time)
    )
    for start_sec in (0.35, 1.05, 1.75, 2.45):
        start = int(start_sec * SR)
        length = int(0.07 * SR)
        local_time = np.arange(length, dtype=np.float64) / SR
        envelope = np.exp(-np.linspace(0.0, 8.0, length))
        burst = np.sin(2.0 * np.pi * 82.0 * local_time)
        burst += 0.4 * np.sin(2.0 * np.pi * 130.0 * local_time)
        samples[start : start + length] += 0.35 * envelope * burst
    samples = np.clip(samples, -0.95, 0.95)

    recommendation = analyze_smart_dsp_audio(
        samples,
        target_id="ch:5",
        source_role="lead_vocal",
        source_label="Lead Vox",
        sample_rate=SR,
        source_identity_confirmed=True,
        source_identity_confidence=0.91,
        current_state_cache_freshness=_fresh(),
        current_state_readback_available=True,
        rollback_plan_available=True,
        target_plane_lufs=-22.0,
    )

    audio_evidence = recommendation.evidence["audio_evidence"]
    assert audio_evidence["plosive_risk_score"] >= 0.55
    assert audio_evidence["plosive_frequency_hz"] is not None
    assert audio_evidence["plosive_event_peak_dbfs"] < -1.0
    assert audio_evidence["plosive_event_contrast_db"] >= 6.0
    assert audio_evidence["plosive_event_rise_time_ms"] > 0.0
    assert audio_evidence["plosive_event_recovery_ms"] > 0.0
    assert audio_evidence["source_identity_mismatch_score"] == 0.0
    assert "plosive_control" in audio_evidence["problem_hints"]
    assert recommendation.selected_strategy.strategy_id == "vocal_plosive_dynamic_hpf_control"
    assert recommendation.selected_strategy.method_family == "dynamic_eq"
    params = recommendation.selected_strategy.parameters
    assert params["filter_type"] == "dynamic_high_pass"
    assert params["frequency_hz"] >= 75.0
    assert params["slope_db_per_octave"] == 12
    assert params["wing_dynamic_eq_model_osc_value"] == WING_DYNAMIC_EQ_GATE_MODEL
    assert params["wing_dynamic_eq_filter_osc_value"] == "HP12"
    assert params["wing_dynamic_eq_gain_osc_value"] < 0.0
    assert params["wing_dynamic_eq_q_osc_value"] == 0.707
    assert params["plosive_risk_score"] == audio_evidence["plosive_risk_score"]
    assert recommendation.evidence["operator_explanation_sections"]["selection_policy"] == (
        "weighted_strategy_score"
    )
    selected_wing_osc = recommendation.selected_strategy.future_wing_readback_contract[
        "wing_osc_applicability"
    ]
    assert selected_wing_osc["native_osc_supported"] is True
    assert "/ch/{channel}/gate/mdl" in selected_wing_osc["native_osc_path_templates"]
    assert "/ch/{channel}/gate/filt" in selected_wing_osc["native_osc_path_templates"]
    dynamic_eq_readback = recommendation.selected_strategy.future_wing_readback_contract[
        "dynamic_eq_readback"
    ]
    assert dynamic_eq_readback["wing_dynamic_eq_model_readback_must_equal"] == (
        WING_DYNAMIC_EQ_GATE_MODEL
    )
    assert "eq_band_readback" not in recommendation.selected_strategy.future_wing_readback_contract
    alternatives = {candidate.strategy_id: candidate for candidate in recommendation.alternatives}
    static_hpf = alternatives["static_plosive_high_pass_cleanup"]
    static_wing_osc = static_hpf.future_wing_readback_contract["wing_osc_applicability"]
    assert static_wing_osc["native_osc_supported"] is True
    assert "/ch/{channel}/flt/lcf" in static_wing_osc["native_osc_path_templates"]
    dynamic_params = params
    assert dynamic_params["filter_type"] == "dynamic_high_pass"
    assert dynamic_params["plosive_parameter_source"] == "measured_plosive_burst"
    assert dynamic_params["plosive_event_peak_dbfs"] == audio_evidence["plosive_event_peak_dbfs"]
    assert dynamic_params["plosive_event_contrast_db"] == audio_evidence[
        "plosive_event_contrast_db"
    ]
    assert dynamic_params["plosive_event_rise_time_ms"] == audio_evidence[
        "plosive_event_rise_time_ms"
    ]
    assert dynamic_params["plosive_event_recovery_ms"] == audio_evidence[
        "plosive_event_recovery_ms"
    ]
    assert dynamic_params["attack_ms"] != 3.0
    assert dynamic_params["release_ms"] != 130.0
    assert dynamic_params["makeup_gain_db"] == 0.0
    assert dynamic_params["preserve_vocal_body_between_plosives"] is True
    assert "low_mid_corrective_cut" in alternatives


def test_current_wing_state_operator_owned_low_cut_blocks_extracted_static_hpf():
    duration_sec = 3.0
    time = np.arange(int(SR * duration_sec), dtype=np.float64) / SR
    samples = (
        0.055 * np.sin(2.0 * np.pi * 220.0 * time)
        + 0.05 * np.sin(2.0 * np.pi * 1350.0 * time)
        + 0.055 * np.sin(2.0 * np.pi * 3200.0 * time)
        + 0.025 * np.sin(2.0 * np.pi * 11500.0 * time)
    )
    for start_sec in (0.35, 1.05, 1.75, 2.45):
        start = int(start_sec * SR)
        length = int(0.07 * SR)
        local_time = np.arange(length, dtype=np.float64) / SR
        envelope = np.exp(-np.linspace(0.0, 8.0, length))
        burst = np.sin(2.0 * np.pi * 82.0 * local_time)
        burst += 0.4 * np.sin(2.0 * np.pi * 130.0 * local_time)
        samples[start : start + length] += 0.35 * envelope * burst
    samples = np.clip(samples, -0.95, 0.95)

    analysis = build_smart_dsp_analysis_input_from_audio(
        samples,
        target_id="ch:5",
        source_role="lead_vocal",
        source_label="Lead Vox",
        sample_rate=SR,
        source_identity_confirmed=True,
        source_identity_confidence=0.91,
        current_state_cache_freshness=_fresh(),
        current_state_readback_available=True,
        rollback_plan_available=True,
        target_plane_lufs=-22.0,
        current_wing_state={
            "current_wing_eq_model": WING_EQ_REQUIRED_MODEL,
            "current_wing_eq_enabled": True,
            "current_wing_eq_model_readback_available": True,
            "current_wing_gate_model": WING_DYNAMIC_EQ_GATE_MODEL,
            "current_wing_gate_enabled": True,
            "current_wing_gate_slot_readback_available": True,
            "current_wing_gate_slot_operator_owned": True,
            "operator_owned_osc_paths": (
                "/ch/5/flt/lc",
                "/ch/5/flt/lcf",
                "/ch/5/flt/lcs",
            ),
        },
    )
    recommendation = analyze_smart_dsp_strategy(analysis)

    assert analysis.current_wing_eq_filter_operator_owned is True
    assert analysis.current_wing_eq_model == WING_EQ_REQUIRED_MODEL
    assert analysis.current_wing_gate_model == WING_DYNAMIC_EQ_GATE_MODEL
    assert recommendation.selected_strategy.strategy_id not in {
        "static_plosive_high_pass_cleanup",
        "vocal_plosive_dynamic_hpf_control",
    }
    alternatives = {candidate.strategy_id: candidate for candidate in recommendation.alternatives}
    dynamic_hpf = alternatives["vocal_plosive_dynamic_hpf_control"]
    assert dynamic_hpf.parameters["wing_gate_slot_current_operator_owned"] is True
    static_hpf = alternatives["static_plosive_high_pass_cleanup"]
    assert static_hpf.blockers == (WING_EQ_FILTER_OPERATOR_OWNED_BLOCKER,)
    assert static_hpf.parameters["wing_eq_filter_family"] == "channel_low_cut"
    assert static_hpf.parameters["wing_eq_filter_current_operator_owned"] is True
    applicability = static_hpf.future_wing_readback_contract[
        "wing_osc_applicability"
    ]
    assert applicability["native_osc_supported"] is False
    assert applicability["native_osc_support_level"] == (
        "eq_filter_operator_owned_edit_requires_review"
    )
    assert applicability["unsupported_strategy_reason"] == (
        "eq_filter_has_native_osc_paths_but_current_wing_filter_is_operator_owned"
    )


def test_current_wing_state_operator_owned_flags_reach_strategy_input():
    samples = _tone(900.0, amplitude=0.03)

    analysis = build_smart_dsp_analysis_input_from_audio(
        samples,
        target_id="ch:12",
        source_role="lead_vocal",
        source_label="Lead Vox",
        sample_rate=SR,
        source_identity_confirmed=True,
        source_identity_confidence=0.91,
        current_state_cache_freshness=_fresh(),
        current_state_readback_available=True,
        rollback_plan_available=True,
        current_wing_state={
            "current_wing_dynamics_model": WING_DYNAMICS_REQUIRED_MODEL,
            "current_wing_dynamics_enabled": True,
            "current_wing_dynamics_model_readback_available": True,
            "current_wing_dynamics_operator_owned": True,
            "current_wing_fx_model": "HALL",
            "current_wing_fx_model_readback_available": True,
            "current_wing_fx_operator_owned": True,
            "current_wing_fx_level_operator_owned": True,
            "current_wing_pan_width_operator_owned": True,
            "current_wing_mute_operator_owned": True,
            "current_wing_gain_operator_owned": True,
        },
    )

    assert analysis.current_wing_dynamics_model == WING_DYNAMICS_REQUIRED_MODEL
    assert analysis.current_wing_dynamics_enabled is True
    assert analysis.current_wing_dynamics_model_readback_available is True
    assert analysis.current_wing_dynamics_operator_owned is True
    assert analysis.current_wing_fx_model == "HALL"
    assert analysis.current_wing_fx_model_readback_available is True
    assert analysis.current_wing_fx_operator_owned is True
    assert analysis.current_wing_fx_level_operator_owned is True
    assert analysis.current_wing_pan_width_operator_owned is True
    assert analysis.current_wing_mute_operator_owned is True
    assert analysis.current_wing_gain_operator_owned is True


def test_current_wing_state_operator_owned_paths_reach_non_eq_strategy_input():
    samples = _tone(900.0, amplitude=0.03)

    analysis = build_smart_dsp_analysis_input_from_audio(
        samples,
        target_id="ch:12",
        source_role="lead_vocal",
        source_label="Lead Vox",
        sample_rate=SR,
        source_identity_confirmed=True,
        source_identity_confidence=0.91,
        current_state_cache_freshness=_fresh(),
        current_state_readback_available=True,
        rollback_plan_available=True,
        fx_return_target_id="fx:1",
        current_wing_state={
            "operator_owned_osc_paths": {
                "channel": (
                    "/ch/12/dyn/mdl",
                    "/ch/12/gate/mdl",
                    "/ch/12/send/2/lvl",
                    "/ch/12/pan",
                    "/ch/12/wid",
                    "/ch/12/mute",
                    "/ch/12/in/set/trim",
                ),
                "fx_return": ("/fx/1/fxmix",),
            },
        },
    )

    assert analysis.current_wing_dynamics_operator_owned is True
    assert analysis.current_wing_gate_slot_operator_owned is True
    assert analysis.current_wing_fx_level_operator_owned is True
    assert analysis.current_wing_pan_width_operator_owned is True
    assert analysis.current_wing_mute_operator_owned is True
    assert analysis.current_wing_gain_operator_owned is True


def test_current_wing_state_operator_owned_main_alias_paths_reach_strategy_input():
    samples = _tone(1000.0, amplitude=0.18)

    analysis = build_smart_dsp_analysis_input_from_audio(
        samples,
        target_id="main:lr",
        source_role="master",
        source_label="Main LR",
        sample_rate=SR,
        source_identity_confirmed=True,
        source_identity_confidence=0.95,
        current_state_cache_freshness=_fresh(),
        current_state_readback_available=True,
        rollback_plan_available=True,
        current_wing_state={
            "operator_owned_osc_paths": (
                "/main/1/dyn/thr36",
                "/main/1/fdr",
            ),
        },
    )

    assert analysis.current_wing_dynamics_operator_owned is True
    assert analysis.current_wing_gain_operator_owned is True


def test_current_wing_state_operator_owned_matrix_dynamics_threshold_reaches_strategy_input():
    samples = _tone(1000.0, amplitude=0.18)

    analysis = build_smart_dsp_analysis_input_from_audio(
        samples,
        target_id="mtx:1",
        source_role="matrix_output",
        source_label="Matrix 1",
        sample_rate=SR,
        source_identity_confirmed=True,
        source_identity_confidence=0.92,
        current_state_cache_freshness=_fresh(),
        current_state_readback_available=True,
        rollback_plan_available=True,
        current_wing_state={"operator_owned_osc_paths": ("/mtx/1/dyn/thr40",)},
    )

    assert analysis.current_wing_dynamics_operator_owned is True


def test_current_wing_state_operator_owned_eq_path_reaches_strategy_input():
    samples = _tone(900.0, amplitude=0.03)

    analysis = build_smart_dsp_analysis_input_from_audio(
        samples,
        target_id="ch:12",
        source_role="lead_vocal",
        source_label="Lead Vox",
        sample_rate=SR,
        source_identity_confirmed=True,
        source_identity_confidence=0.91,
        current_state_cache_freshness=_fresh(),
        current_state_readback_available=True,
        rollback_plan_available=True,
        current_wing_state={
            "operator_owned_osc_paths": (
                "/ch/12/eq/mdl",
                "/ch/12/eq/3g",
            ),
        },
    )

    assert analysis.current_wing_eq_operator_owned is True


def test_current_wing_state_operator_owned_fx_model_path_reaches_strategy_input():
    samples = _tone(900.0, amplitude=0.03)

    analysis = build_smart_dsp_analysis_input_from_audio(
        samples,
        target_id="ch:12",
        source_role="lead_vocal",
        source_label="Lead Vox",
        sample_rate=SR,
        source_identity_confirmed=True,
        source_identity_confidence=0.91,
        current_state_cache_freshness=_fresh(),
        current_state_readback_available=True,
        rollback_plan_available=True,
        fx_return_target_id="fx:2",
        current_wing_state={
            "operator_owned_osc_paths": (
                "/fx/2/mdl",
                "/fx/2/dcy",
            ),
        },
    )

    assert analysis.current_wing_fx_operator_owned is True


def test_current_wing_state_operator_owned_explicit_false_overrides_path():
    samples = _tone(900.0, amplitude=0.03)

    analysis = build_smart_dsp_analysis_input_from_audio(
        samples,
        target_id="ch:10",
        source_role="lead_vocal",
        source_label="Lead Vox",
        sample_rate=SR,
        source_identity_confirmed=True,
        source_identity_confidence=0.91,
        current_state_cache_freshness=_fresh(),
        current_state_readback_available=True,
        rollback_plan_available=True,
        current_wing_state={
            "current_wing_gain_operator_owned": False,
            "operator_owned_osc_paths": ("/ch/10/fdr",),
        },
    )

    assert analysis.current_wing_gain_operator_owned is False


def test_current_wing_state_operator_owned_gain_blocks_extracted_gain_candidate():
    duration_sec = 2.0
    time = np.arange(int(SR * duration_sec), dtype=np.float64) / SR
    samples = 0.003 * np.sin(2.0 * np.pi * 900.0 * time)

    analysis = build_smart_dsp_analysis_input_from_audio(
        samples,
        target_id="ch:10",
        source_role="lead_vocal",
        source_label="Lead Vox",
        sample_rate=SR,
        source_identity_confirmed=True,
        source_identity_confidence=0.91,
        current_state_cache_freshness=_fresh(),
        current_state_readback_available=True,
        rollback_plan_available=True,
        target_plane_lufs=-22.0,
        extra_problem_hints=("level_plane_balance",),
        current_wing_state={"current_wing_gain_operator_owned": True},
    )
    recommendation = analyze_smart_dsp_strategy(analysis)

    assert analysis.current_wing_gain_operator_owned is True
    candidates = {
        candidate.strategy_id: candidate
        for candidate in (recommendation.selected_strategy, *recommendation.alternatives)
    }
    gain_up = candidates["soundcheck_bounded_gain_up"]
    assert WING_GAIN_OPERATOR_OWNED_BLOCKER in gain_up.blockers
    assert gain_up.parameters["wing_gain_current_operator_owned"] is True
    assert gain_up.parameters["wing_gain_operator_owned_edit_requires_review"] is True
    assert gain_up.parameters["wing_gain_operator_owned_blocker"] == (
        WING_GAIN_OPERATOR_OWNED_BLOCKER
    )
    applicability = gain_up.future_wing_readback_contract["wing_osc_applicability"]
    assert applicability["native_osc_supported"] is False
    assert applicability["native_osc_support_level"] == (
        "gain_operator_owned_edit_requires_review"
    )


def test_current_wing_state_operator_owned_gain_path_blocks_extracted_gain_candidate():
    duration_sec = 2.0
    time = np.arange(int(SR * duration_sec), dtype=np.float64) / SR
    samples = 0.003 * np.sin(2.0 * np.pi * 900.0 * time)

    analysis = build_smart_dsp_analysis_input_from_audio(
        samples,
        target_id="ch:10",
        source_role="lead_vocal",
        source_label="Lead Vox",
        sample_rate=SR,
        source_identity_confirmed=True,
        source_identity_confidence=0.91,
        current_state_cache_freshness=_fresh(),
        current_state_readback_available=True,
        rollback_plan_available=True,
        target_plane_lufs=-22.0,
        extra_problem_hints=("level_plane_balance",),
        current_wing_state={"operator_owned_osc_paths": ("/ch/10/fdr",)},
    )
    recommendation = analyze_smart_dsp_strategy(analysis)

    assert analysis.current_wing_gain_operator_owned is True
    candidates = {
        candidate.strategy_id: candidate
        for candidate in (recommendation.selected_strategy, *recommendation.alternatives)
    }
    gain_up = candidates["soundcheck_bounded_gain_up"]
    assert WING_GAIN_OPERATOR_OWNED_BLOCKER in gain_up.blockers
    assert gain_up.parameters["wing_gain_current_operator_owned"] is True
    assert gain_up.parameters["wing_gain_operator_owned_edit_requires_review"] is True
    assert gain_up.future_wing_readback_contract["wing_osc_applicability"][
        "native_osc_supported"
    ] is False


def test_current_wing_state_operator_owned_eq_path_blocks_extracted_static_eq():
    duration_sec = 3.0
    time = np.arange(int(SR * duration_sec), dtype=np.float64) / SR
    samples = (
        0.055 * np.sin(2.0 * np.pi * 220.0 * time)
        + 0.05 * np.sin(2.0 * np.pi * 1350.0 * time)
        + 0.055 * np.sin(2.0 * np.pi * 3200.0 * time)
        + 0.025 * np.sin(2.0 * np.pi * 11500.0 * time)
    )
    for start_sec in (0.35, 1.05, 1.75, 2.45):
        start = int(start_sec * SR)
        length = int(0.07 * SR)
        local_time = np.arange(length, dtype=np.float64) / SR
        envelope = np.exp(-np.linspace(0.0, 8.0, length))
        burst = np.sin(2.0 * np.pi * 82.0 * local_time)
        burst += 0.4 * np.sin(2.0 * np.pi * 130.0 * local_time)
        samples[start : start + length] += 0.35 * envelope * burst
    samples = np.clip(samples, -0.95, 0.95)

    analysis = build_smart_dsp_analysis_input_from_audio(
        samples,
        target_id="ch:5",
        source_role="lead_vocal",
        source_label="Lead Vox",
        sample_rate=SR,
        source_identity_confirmed=True,
        source_identity_confidence=0.91,
        current_state_cache_freshness=_fresh(),
        current_state_readback_available=True,
        rollback_plan_available=True,
        target_plane_lufs=-22.0,
        current_wing_state={
            "current_wing_eq_model": WING_EQ_REQUIRED_MODEL,
            "current_wing_eq_enabled": True,
            "current_wing_eq_model_readback_available": True,
            "operator_owned_osc_paths": (
                "/ch/5/eq/mdl",
                "/ch/5/eq/3g",
            ),
        },
    )
    recommendation = analyze_smart_dsp_strategy(analysis)

    assert analysis.current_wing_eq_operator_owned is True
    candidates = {
        candidate.strategy_id: candidate
        for candidate in (recommendation.selected_strategy, *recommendation.alternatives)
    }
    static_hpf = candidates["static_plosive_high_pass_cleanup"]
    assert static_hpf.blockers == (WING_EQ_OPERATOR_OWNED_MODEL_BLOCKER,)
    assert static_hpf.parameters["wing_eq_current_operator_owned"] is True
    assert (
        static_hpf.parameters["wing_eq_operator_owned_model_edit_requires_review"]
        is True
    )
    assert static_hpf.future_wing_readback_contract["wing_osc_applicability"][
        "native_osc_supported"
    ] is False


def test_readonly_preflight_gate_readback_reaches_smart_dsp_state():
    wing_state = build_current_wing_state_from_readonly_preflight(
        target_id="ch:3",
        summary={
            "run_id": "20260702T075151Z_wing_readonly_preflight",
            "connection_mode": "wing",
            "read_only": True,
            "live_write": False,
            "write_request_sent": False,
            "proposal_request_sent": False,
            "live_authority_granted": False,
            "readiness_closure_allowed": False,
            "readbacks": [
                {
                    "type": "mixer_parameter_value",
                    "request_id": "wing-readonly-preflight:ch3:gate_model",
                    "status": "verified_from_physical_readback",
                    "source": "physical_readback",
                    "channel": 3,
                    "value_type": "gate_model",
                    "address": "/ch/3/gate/mdl",
                    "value": "GATE",
                },
                {
                    "type": "mixer_parameter_value",
                    "request_id": "wing-readonly-preflight:ch3:gate_on",
                    "status": "verified_from_physical_readback",
                    "source": "physical_readback",
                    "channel": 3,
                    "value_type": "gate_on",
                    "address": "/ch/3/gate/on",
                    "value": 0,
                },
                {
                    "type": "mixer_parameter_value",
                    "request_id": "wing-readonly-preflight:ch3:gate_threshold",
                    "status": "verified_from_physical_readback",
                    "source": "physical_readback",
                    "channel": 3,
                    "value_type": "gate_threshold",
                    "address": "/ch/3/gate/thr",
                    "value": -40.0,
                },
            ],
        },
    )

    assert wing_state["current_wing_gate_model"] == WING_GATE_REQUIRED_MODEL
    assert wing_state["current_wing_gate_enabled"] is False
    assert wing_state["current_wing_gate_slot_readback_available"] is True
    assert wing_state["wing_readonly_gate_threshold_dbfs"] == -40.0
    assert wing_state["current_wing_state_live_authority_granted"] is False

    duration_sec = 3.0
    time = np.arange(int(SR * duration_sec), dtype=np.float64) / SR
    samples = (
        0.055 * np.sin(2.0 * np.pi * 220.0 * time)
        + 0.05 * np.sin(2.0 * np.pi * 1350.0 * time)
        + 0.055 * np.sin(2.0 * np.pi * 3200.0 * time)
    )
    for start_sec in (0.35, 1.05, 1.75, 2.45):
        start = int(start_sec * SR)
        length = int(0.07 * SR)
        local_time = np.arange(length, dtype=np.float64) / SR
        envelope = np.exp(-np.linspace(0.0, 8.0, length))
        burst = np.sin(2.0 * np.pi * 82.0 * local_time)
        samples[start : start + length] += 0.35 * envelope * burst

    analysis = build_smart_dsp_analysis_input_from_audio(
        np.clip(samples, -0.95, 0.95),
        target_id="ch:3",
        source_role="lead_vocal",
        source_label="Lead Vox",
        sample_rate=SR,
        source_identity_confirmed=True,
        source_identity_confidence=0.91,
        current_wing_state=wing_state,
    )
    recommendation = analyze_smart_dsp_strategy(analysis)
    candidates = {
        candidate.strategy_id: candidate
        for candidate in (recommendation.selected_strategy, *recommendation.alternatives)
    }
    dynamic_hpf = candidates["vocal_plosive_dynamic_hpf_control"]

    assert analysis.current_wing_gate_model == WING_GATE_REQUIRED_MODEL
    assert analysis.current_wing_gate_enabled is False
    assert analysis.current_wing_gate_slot_readback_available is True
    assert analysis.current_state_readback_available is True
    assert analysis.rollback_plan_available is False
    assert recommendation.evidence["current_state_cache_freshness"]["source"] == (
        "wing_readonly_preflight_artifact"
    )
    assert recommendation.evidence["current_state_cache_freshness"]["status"] == "fresh"
    assert recommendation.evidence[
        "current_smart_dsp_context_cache_no_live_authority"
    ] is True
    assert "current_state_readback_missing" not in recommendation.blockers
    assert "rollback_plan_missing" in recommendation.blockers
    assert dynamic_hpf.blockers == (WING_GATE_SLOT_MODEL_CONFLICT_BLOCKER,)
    assert dynamic_hpf.parameters["wing_gate_slot_current_model"] == (
        WING_GATE_REQUIRED_MODEL
    )
    assert dynamic_hpf.parameters["wing_gate_slot_current_enabled"] is False
    assert dynamic_hpf.future_wing_readback_contract["wing_osc_applicability"][
        "native_osc_supported"
    ] is False


def test_readonly_preflight_basic_controls_reach_smart_dsp_state():
    wing_state = build_current_wing_state_from_readonly_preflight(
        target_id="ch:3",
        summary={
            "run_id": "20260702T075151Z_wing_readonly_preflight",
            "connection_mode": "wing",
            "read_only": True,
            "live_write": False,
            "write_request_sent": False,
            "proposal_request_sent": False,
            "live_authority_granted": False,
            "readiness_closure_allowed": False,
            "readbacks": [
                {
                    "type": "mixer_parameter_value",
                    "status": "verified_from_physical_readback",
                    "source": "physical_readback",
                    "channel": 3,
                    "value_type": "fader",
                    "address": "/ch/3/fdr",
                    "value": -8.125,
                },
                {
                    "type": "mixer_parameter_value",
                    "status": "verified_from_physical_readback",
                    "source": "physical_readback",
                    "channel": 3,
                    "value_type": "gain",
                    "address": "/ch/3/in/set/trim",
                    "value": 1.25,
                },
                {
                    "type": "mixer_parameter_value",
                    "status": "verified_from_physical_readback",
                    "source": "physical_readback",
                    "channel": 3,
                    "value_type": "pan",
                    "address": "/ch/3/pan",
                    "value": 125.0,
                },
            ],
        },
    )

    assert wing_state["current_wing_fader_db"] == -8.125
    assert wing_state["wing_readonly_fader_db"] == -8.125
    assert wing_state["current_wing_input_trim_db"] == 1.25
    assert wing_state["wing_readonly_input_trim_db"] == 1.25
    assert wing_state["current_wing_pan_percent"] == 100.0
    assert wing_state["wing_readonly_pan_percent"] == 100.0
    assert wing_state["current_wing_state_basic_controls_readback_available"] is True
    assert wing_state["current_wing_state_live_authority_granted"] is False

    analysis = build_smart_dsp_analysis_input_from_audio(
        _tone(440.0, amplitude=0.12, duration_sec=1.0),
        target_id="ch:3",
        source_role="lead_vocal",
        source_label="Lead Vox",
        sample_rate=SR,
        source_identity_confirmed=True,
        source_identity_confidence=0.91,
        current_wing_state=wing_state,
    )

    assert analysis.current_wing_fader_db == -8.125
    assert analysis.current_wing_input_trim_db == 1.25
    assert analysis.current_wing_pan_percent == 100.0
    assert analysis.current_wing_control_readback_available is True
    assert analysis.current_state_readback_available is True

    recommendation = analyze_smart_dsp_strategy(analysis)
    controls = recommendation.evidence["current_wing_controls"]
    assert controls["readback_available"] is True
    assert controls["fader_db"] == -8.125
    assert controls["input_trim_db"] == 1.25
    assert controls["pan_percent"] == 100.0
    assert controls["source"] == "wing_readonly_preflight_artifact"
    assert controls["live_write"] is False
    assert controls["live_authority_granted"] is False


def test_readonly_preflight_eq_dynamics_readbacks_keep_model_guards_closed():
    wing_state = build_current_wing_state_from_readonly_preflight(
        target_id="ch:3",
        summary={
            "run_id": "20260702T075151Z_wing_readonly_preflight",
            "connection_mode": "wing",
            "read_only": True,
            "live_write": False,
            "write_request_sent": False,
            "proposal_request_sent": False,
            "live_authority_granted": False,
            "readiness_closure_allowed": False,
            "readbacks": [
                {
                    "type": "mixer_parameter_value",
                    "status": "verified_from_physical_readback",
                    "source": "physical_readback",
                    "channel": 3,
                    "value_type": "eq_on",
                    "address": "/ch/3/eq/on",
                    "value": 1,
                },
                {
                    "type": "mixer_parameter_value",
                    "status": "verified_from_physical_readback",
                    "source": "physical_readback",
                    "channel": 3,
                    "value_type": "eq_mix",
                    "address": "/ch/3/eq/mix",
                    "value": 100.0,
                },
                {
                    "type": "mixer_parameter_value",
                    "status": "verified_from_physical_readback",
                    "source": "physical_readback",
                    "channel": 3,
                    "value_type": "eq_low_gain",
                    "address": "/ch/3/eq/lg",
                    "value": -1.25,
                },
                {
                    "type": "mixer_parameter_value",
                    "status": "verified_from_physical_readback",
                    "source": "physical_readback",
                    "channel": 3,
                    "value_type": "eq_high_gain",
                    "address": "/ch/3/eq/hg",
                    "value": 0.5,
                },
                {
                    "type": "mixer_parameter_value",
                    "status": "verified_from_physical_readback",
                    "source": "physical_readback",
                    "channel": 3,
                    "value_type": "dyn_on",
                    "address": "/ch/3/dyn/on",
                    "value": 1,
                },
                {
                    "type": "mixer_parameter_value",
                    "status": "verified_from_physical_readback",
                    "source": "physical_readback",
                    "channel": 3,
                    "value_type": "dyn_threshold",
                    "address": "/ch/3/dyn/thr",
                    "value": -22.5,
                },
            ],
        },
    )

    assert wing_state["current_wing_eq_enabled"] is True
    assert wing_state["current_wing_eq_readback_available"] is True
    assert wing_state["wing_readonly_eq_mix_percent"] == 100.0
    assert wing_state["wing_readonly_eq_low_gain_db"] == -1.25
    assert wing_state["wing_readonly_eq_high_gain_db"] == 0.5
    assert wing_state["current_wing_dynamics_enabled"] is True
    assert wing_state["current_wing_dynamics_readback_available"] is True
    assert wing_state["wing_readonly_dynamics_threshold_dbfs"] == -22.5
    assert "current_wing_eq_model" not in wing_state
    assert "current_wing_dynamics_model" not in wing_state

    duration_sec = 3.0
    time = np.arange(int(SR * duration_sec), dtype=np.float64) / SR
    envelope = np.where((time % 0.6) < 0.18, 0.22, 0.025)
    samples = envelope * np.sin(2.0 * np.pi * 900.0 * time)
    for start_sec in (0.35, 1.05, 1.75, 2.45):
        start = int(start_sec * SR)
        length = int(0.07 * SR)
        local_time = np.arange(length, dtype=np.float64) / SR
        burst_envelope = np.exp(-np.linspace(0.0, 8.0, length))
        burst = np.sin(2.0 * np.pi * 82.0 * local_time)
        samples[start : start + length] += 0.35 * burst_envelope * burst

    analysis = build_smart_dsp_analysis_input_from_audio(
        np.clip(samples, -0.95, 0.95),
        target_id="ch:3",
        source_role="lead_vocal",
        source_label="Lead Vox",
        sample_rate=SR,
        source_identity_confirmed=True,
        source_identity_confidence=0.91,
        current_wing_state=wing_state,
        extra_problem_hints=("dynamic_control",),
        target_plane_lufs=-22.0,
    )
    recommendation = analyze_smart_dsp_strategy(analysis)
    candidates = {
        candidate.strategy_id: candidate
        for candidate in (recommendation.selected_strategy, *recommendation.alternatives)
    }
    static_hpf = candidates["static_plosive_high_pass_cleanup"]
    compressor = candidates["vocal_phrase_leveling_compression"]

    assert analysis.current_wing_eq_enabled is True
    assert analysis.current_wing_eq_model == ""
    assert analysis.current_wing_eq_model_readback_available is False
    assert analysis.current_wing_dynamics_enabled is True
    assert analysis.current_wing_dynamics_model == ""
    assert analysis.current_wing_dynamics_model_readback_available is False
    assert analysis.current_state_readback_available is True
    assert analysis.rollback_plan_available is False
    assert WING_EQ_MODEL_READBACK_BLOCKER in static_hpf.blockers
    assert static_hpf.parameters["wing_eq_current_enabled"] is True
    assert (
        static_hpf.parameters["wing_eq_model_readback_required_for_active_slot"]
        is True
    )
    assert WING_DYNAMICS_MODEL_READBACK_BLOCKER in compressor.blockers
    assert compressor.parameters["wing_dynamics_current_enabled"] is True
    assert (
        compressor.parameters[
            "wing_dynamics_model_readback_required_for_active_slot"
        ]
        is True
    )
    assert "current_state_readback_missing" not in recommendation.blockers
    assert "rollback_plan_missing" in recommendation.blockers


def test_readonly_preflight_basic_controls_feed_pan_decision_without_authority():
    wing_state = build_current_wing_state_from_readonly_preflight(
        target_id="ch:3",
        summary={
            "run_id": "20260702T220146Z_wing_readonly_preflight",
            "connection_mode": "wing",
            "read_only": True,
            "live_write": False,
            "write_request_sent": False,
            "proposal_request_sent": False,
            "live_authority_granted": False,
            "readiness_closure_allowed": False,
            "readbacks": [
                {
                    "type": "mixer_parameter_value",
                    "status": "verified_from_physical_readback",
                    "source": "physical_readback",
                    "channel": 3,
                    "value_type": "fader",
                    "address": "/ch/3/fdr",
                    "value": -12.0,
                },
                {
                    "type": "mixer_parameter_value",
                    "status": "verified_from_physical_readback",
                    "source": "physical_readback",
                    "channel": 3,
                    "value_type": "gain",
                    "address": "/ch/3/in/set/trim",
                    "value": 1.5,
                },
                {
                    "type": "mixer_parameter_value",
                    "status": "verified_from_physical_readback",
                    "source": "physical_readback",
                    "channel": 3,
                    "value_type": "pan",
                    "address": "/ch/3/pan",
                    "value": 18.0,
                },
            ],
        },
    )

    assert wing_state["current_wing_fader_db"] == -12.0
    assert wing_state["current_wing_input_trim_db"] == 1.5
    assert wing_state["current_wing_pan_percent"] == 18.0
    assert wing_state["current_wing_state_basic_controls_readback_available"] is True
    assert wing_state["current_wing_state_live_authority_granted"] is False

    samples = _tone(900.0, amplitude=0.12)
    analysis = build_smart_dsp_analysis_input_from_audio(
        samples,
        target_id="ch:3",
        source_role="lead_vocal",
        source_label="Lead Vox",
        sample_rate=SR,
        source_identity_confirmed=True,
        source_identity_confidence=0.91,
        current_wing_state=wing_state,
        rollback_plan_available=True,
        target_plane_lufs=-22.0,
    )
    recommendation = analyze_smart_dsp_strategy(analysis)
    controls = recommendation.evidence["current_wing_controls"]

    assert analysis.current_wing_control_readback_available is True
    assert analysis.current_wing_fader_db == -12.0
    assert analysis.current_wing_input_trim_db == 1.5
    assert analysis.current_wing_pan_percent == 18.0
    assert analysis.current_pan_percent == 18.0
    assert controls["readback_available"] is True
    assert controls["fader_db"] == -12.0
    assert controls["input_trim_db"] == 1.5
    assert controls["pan_percent"] == 18.0
    assert controls["live_write"] is False
    assert controls["live_authority_granted"] is False
    candidates = {
        candidate.strategy_id: candidate
        for candidate in (recommendation.selected_strategy, *recommendation.alternatives)
    }
    pan_review = candidates["center_anchor_recenter_review"]
    assert pan_review.parameters["current_pan_percent"] == 18.0
    assert pan_review.parameters["target_pan_percent"] == 0.0
    assert recommendation.to_operator_proposal()["live_apply_allowed"] is False


def test_readonly_preflight_state_fails_closed_on_unsafe_artifact_flags():
    wing_state = build_current_wing_state_from_readonly_preflight(
        target_id="ch:3",
        summary={
            "run_id": "unsafe_wing_readonly_preflight",
            "read_only": False,
            "live_write": True,
            "write_request_sent": True,
            "proposal_request_sent": True,
            "live_authority_granted": True,
            "readiness_closure_allowed": True,
            "readbacks": [
                {
                    "type": "mixer_parameter_value",
                    "status": "verified_from_physical_readback",
                    "source": "physical_readback",
                    "channel": 3,
                    "value_type": "gate_model",
                    "address": "/ch/3/gate/mdl",
                    "value": "DEQ",
                },
                {
                    "type": "mixer_parameter_value",
                    "status": "verified_from_physical_readback",
                    "source": "physical_readback",
                    "channel": 3,
                    "value_type": "fader",
                    "address": "/ch/3/fdr",
                    "value": -6.0,
                },
                {
                    "type": "mixer_parameter_value",
                    "status": "verified_from_physical_readback",
                    "source": "physical_readback",
                    "channel": 3,
                    "value_type": "pan",
                    "address": "/ch/3/pan",
                    "value": 24.0,
                }
            ],
        },
    )

    assert wing_state["current_wing_state_rejected"] is True
    assert "read_only_not_true" in wing_state["current_wing_state_violations"]
    assert "live_write_not_false" in wing_state["current_wing_state_violations"]
    assert "current_wing_gate_model" not in wing_state
    assert "current_wing_fader_db" not in wing_state
    assert "current_wing_pan_percent" not in wing_state
    assert wing_state["current_wing_state_live_authority_granted"] is False

    analysis = build_smart_dsp_analysis_input_from_audio(
        _tone(900.0, amplitude=0.04),
        target_id="ch:3",
        source_role="lead_vocal",
        source_label="Lead Vox",
        sample_rate=SR,
        source_identity_confirmed=True,
        source_identity_confidence=0.91,
        current_wing_state=wing_state,
    )
    recommendation = analyze_smart_dsp_strategy(analysis)

    assert analysis.current_wing_gate_model == ""
    assert analysis.current_wing_fader_db is None
    assert analysis.current_wing_pan_percent is None
    assert analysis.current_wing_control_readback_available is False
    assert analysis.current_state_readback_available is False
    assert analysis.rollback_plan_available is False
    assert recommendation.evidence["current_state_cache_freshness"]["status"] == (
        "rejected"
    )
    assert "current_state_readback_missing" in recommendation.blockers
    assert "rollback_plan_missing" in recommendation.blockers


def test_current_wing_state_operator_owned_gate_path_blocks_extracted_dynamic_hpf():
    duration_sec = 3.0
    time = np.arange(int(SR * duration_sec), dtype=np.float64) / SR
    samples = (
        0.055 * np.sin(2.0 * np.pi * 220.0 * time)
        + 0.05 * np.sin(2.0 * np.pi * 1350.0 * time)
        + 0.055 * np.sin(2.0 * np.pi * 3200.0 * time)
        + 0.025 * np.sin(2.0 * np.pi * 11500.0 * time)
    )
    for start_sec in (0.35, 1.05, 1.75, 2.45):
        start = int(start_sec * SR)
        length = int(0.07 * SR)
        local_time = np.arange(length, dtype=np.float64) / SR
        envelope = np.exp(-np.linspace(0.0, 8.0, length))
        burst = np.sin(2.0 * np.pi * 82.0 * local_time)
        burst += 0.4 * np.sin(2.0 * np.pi * 130.0 * local_time)
        samples[start : start + length] += 0.35 * envelope * burst
    samples = np.clip(samples, -0.95, 0.95)

    analysis = build_smart_dsp_analysis_input_from_audio(
        samples,
        target_id="ch:5",
        source_role="lead_vocal",
        source_label="Lead Vox",
        sample_rate=SR,
        source_identity_confirmed=True,
        source_identity_confidence=0.91,
        current_state_cache_freshness=_fresh(),
        current_state_readback_available=True,
        rollback_plan_available=True,
        target_plane_lufs=-22.0,
        current_wing_state={
            "current_wing_gate_model": WING_DYNAMIC_EQ_GATE_MODEL,
            "current_wing_gate_enabled": True,
            "current_wing_gate_slot_readback_available": True,
            "operator_owned_osc_paths": (
                "/ch/5/gate/mdl",
                "/ch/5/gate/filt",
            ),
        },
    )
    recommendation = analyze_smart_dsp_strategy(analysis)

    assert analysis.current_wing_gate_slot_operator_owned is True
    candidates = {
        candidate.strategy_id: candidate
        for candidate in (recommendation.selected_strategy, *recommendation.alternatives)
    }
    dynamic_hpf = candidates["vocal_plosive_dynamic_hpf_control"]
    assert dynamic_hpf.blockers == (WING_GATE_SLOT_OPERATOR_OWNED_MODEL_BLOCKER,)
    assert dynamic_hpf.parameters["wing_gate_slot_current_operator_owned"] is True
    assert dynamic_hpf.parameters[
        "wing_gate_slot_operator_owned_model_edit_requires_review"
    ] is True
    assert dynamic_hpf.future_wing_readback_contract["wing_osc_applicability"][
        "native_osc_supported"
    ] is False


def test_live_plosive_vocal_audio_holds_before_hpf_processing():
    duration_sec = 3.0
    time = np.arange(int(SR * duration_sec), dtype=np.float64) / SR
    samples = (
        0.055 * np.sin(2.0 * np.pi * 220.0 * time)
        + 0.05 * np.sin(2.0 * np.pi * 1350.0 * time)
        + 0.055 * np.sin(2.0 * np.pi * 3200.0 * time)
        + 0.025 * np.sin(2.0 * np.pi * 11500.0 * time)
    )
    for start_sec in (0.35, 1.05, 1.75, 2.45):
        start = int(start_sec * SR)
        length = int(0.07 * SR)
        local_time = np.arange(length, dtype=np.float64) / SR
        envelope = np.exp(-np.linspace(0.0, 8.0, length))
        burst = np.sin(2.0 * np.pi * 82.0 * local_time)
        burst += 0.4 * np.sin(2.0 * np.pi * 130.0 * local_time)
        samples[start : start + length] += 0.35 * envelope * burst
    samples = np.clip(samples, -0.95, 0.95)

    recommendation = analyze_smart_dsp_audio(
        samples,
        target_id="ch:5",
        source_role="lead_vocal",
        source_label="Lead Vox",
        mode="live",
        sample_rate=SR,
        source_identity_confirmed=True,
        source_identity_confidence=0.91,
        current_state_cache_freshness=_fresh(),
        current_state_readback_available=True,
        rollback_plan_available=True,
        target_plane_lufs=-22.0,
        active_performance=True,
    )

    audio_evidence = recommendation.evidence["audio_evidence"]
    assert audio_evidence["plosive_risk_score"] >= 0.55
    assert "plosive_control" in audio_evidence["problem_hints"]
    assert recommendation.selected_strategy.strategy_id == (
        "manual_plosive_mic_technique_review"
    )
    assert recommendation.evidence["operator_explanation_sections"][
        "selection_policy"
    ] == "live_plosive_review_priority"
    params = recommendation.selected_strategy.parameters
    assert params["live_plosive_review_priority"] is True
    assert params["forbidden_without_wing_proof"] == "dynamic_hpf_static_hpf"
    alternatives = {
        candidate.strategy_id: candidate for candidate in recommendation.alternatives
    }
    assert "live_plosive_hpf_requires_operator_review_and_wing_proof" in (
        alternatives["vocal_plosive_dynamic_hpf_control"].blockers
    )
    assert recommendation.to_operator_proposal()["live_apply_allowed"] is False


def test_quiet_vocal_phrase_noise_audio_reviews_gain_stage_before_expander():
    rng = np.random.default_rng(61)
    duration_sec = 3.0
    samples = rng.normal(scale=10 ** (-56.0 / 20.0), size=int(SR * duration_sec))
    for start_sec in (0.2, 1.1, 2.0):
        start = int(start_sec * SR)
        length = int(0.48 * SR)
        local_time = np.arange(length, dtype=np.float64) / SR
        envelope = np.sin(np.linspace(0.0, np.pi, length)) ** 0.75
        phrase = (
            0.08 * np.sin(2.0 * np.pi * 780.0 * local_time)
            + 0.06 * np.sin(2.0 * np.pi * 1450.0 * local_time)
            + 0.035 * np.sin(2.0 * np.pi * 3100.0 * local_time)
        )
        samples[start : start + length] += envelope * phrase
    samples = np.clip(samples, -0.95, 0.95)

    recommendation = analyze_smart_dsp_audio(
        samples,
        target_id="ch:5",
        source_role="lead_vocal",
        source_label="Lead Vox",
        sample_rate=SR,
        source_identity_confirmed=True,
        source_identity_confidence=0.91,
        current_state_cache_freshness=_fresh(),
        current_state_readback_available=True,
        rollback_plan_available=True,
        target_plane_lufs=-22.0,
    )

    audio_evidence = recommendation.evidence["audio_evidence"]
    assert audio_evidence["vocal_noise_floor_score"] >= 0.55
    assert audio_evidence["phrase_gap_noise_floor_dbfs"] > -62.0
    assert audio_evidence["phrase_gap_median_ms"] > 0.0
    assert audio_evidence["phrase_gap_coverage_ratio"] > 0.0
    assert "vocal_noise_floor" in audio_evidence["problem_hints"]
    assert "level_plane_balance" in audio_evidence["problem_hints"]
    assert "vocal_noise_floor" in recommendation.evidence["detected_problems"]
    assert recommendation.selected_strategy.strategy_id == (
        "mic_gain_stage_noise_floor_review"
    )
    assert recommendation.evidence["operator_explanation_sections"][
        "selection_policy"
    ] == "soundcheck_quiet_noisy_vocal_gain_stage_priority"
    params = recommendation.selected_strategy.parameters
    assert params["read_only_review"] is True
    assert params["gain_db"] == 0.0
    assert params["phrase_gap_noise_floor_dbfs"] == round(
        audio_evidence["phrase_gap_noise_floor_dbfs"],
        1,
    )

    alternatives = {
        item.strategy_id: item for item in recommendation.alternatives
    }
    expander = alternatives["vocal_phrase_noise_downward_expander"]
    assert expander.parameters["breath_preserve"] is True
    assert expander.parameters["phrase_tail_preserve"] is True
    assert expander.parameters["vocal_noise_parameter_source"] == (
        "measured_phrase_gap_envelope"
    )
    assert expander.parameters["threshold_dbfs"] > audio_evidence[
        "phrase_gap_noise_floor_dbfs"
    ]
    assert expander.parameters["phrase_gap_median_ms"] == audio_evidence[
        "phrase_gap_median_ms"
    ]
    assert expander.parameters["phrase_gap_coverage_ratio"] == audio_evidence[
        "phrase_gap_coverage_ratio"
    ]
    assert expander.parameters["attack_ms"] != 8.0
    assert expander.parameters["hold_ms"] != 115.0
    assert expander.parameters["release_ms"] != 260.0
    assert "noise_floor_path_review_should_precede_gain_up" in alternatives[
        "soundcheck_bounded_gain_up"
    ].blockers
    hard_gate = alternatives["vocal_hard_gate_review"]
    assert "hard_gate_can_clip_vocal_breaths_and_phrase_tails" in hard_gate.blockers
    assert recommendation.to_operator_proposal()["live_apply_allowed"] is False


def test_live_vocal_phrase_noise_audio_prioritizes_noise_path_review():
    rng = np.random.default_rng(61)
    duration_sec = 3.0
    samples = rng.normal(scale=10 ** (-56.0 / 20.0), size=int(SR * duration_sec))
    for start_sec in (0.2, 1.1, 2.0):
        start = int(start_sec * SR)
        length = int(0.48 * SR)
        local_time = np.arange(length, dtype=np.float64) / SR
        envelope = np.sin(np.linspace(0.0, np.pi, length)) ** 0.75
        phrase = (
            0.08 * np.sin(2.0 * np.pi * 780.0 * local_time)
            + 0.06 * np.sin(2.0 * np.pi * 1450.0 * local_time)
            + 0.035 * np.sin(2.0 * np.pi * 3100.0 * local_time)
        )
        samples[start : start + length] += envelope * phrase
    samples = np.clip(samples, -0.95, 0.95)

    recommendation = analyze_smart_dsp_audio(
        samples,
        target_id="ch:5",
        source_role="lead_vocal",
        source_label="Lead Vox",
        sample_rate=SR,
        mode="live",
        active_performance=True,
        source_identity_confirmed=True,
        source_identity_confidence=0.91,
        current_state_cache_freshness=_fresh(),
        current_state_readback_available=True,
        rollback_plan_available=True,
        target_plane_lufs=-22.0,
    )

    audio_evidence = recommendation.evidence["audio_evidence"]
    assert audio_evidence["vocal_noise_floor_score"] >= 0.55
    assert audio_evidence["phrase_gap_noise_floor_dbfs"] > -62.0
    assert "vocal_noise_floor" in audio_evidence["problem_hints"]
    assert recommendation.selected_strategy.strategy_id == (
        "mic_gain_stage_noise_floor_review"
    )
    assert recommendation.selected_strategy.method_family == "analysis_only"
    assert recommendation.evidence["operator_explanation_sections"][
        "selection_policy"
    ] == "live_vocal_noise_review_priority"
    params = recommendation.selected_strategy.parameters
    assert params["live_vocal_noise_review_priority"] is True
    assert params["forbidden_without_wing_proof"] == "expander_gate_broad_eq"
    alternatives = {
        candidate.strategy_id: candidate for candidate in recommendation.alternatives
    }
    expander = alternatives["vocal_phrase_noise_downward_expander"]
    assert (
        "live_vocal_noise_floor_expander_requires_operator_review_and_wing_proof"
        in expander.blockers
    )
    assert expander.parameters["vocal_noise_parameter_source"] == (
        "measured_phrase_gap_envelope"
    )
    tradeoffs = {
        row["strategy_id"]: row
        for row in recommendation.evidence["strategy_search"]["alternative_tradeoffs"]
    }
    assert tradeoffs["vocal_phrase_noise_downward_expander"]["why_not_selected"] == (
        "live_vocal_noise_review_priority:"
        "noise_path_check_precedes_expander_gate_or_eq"
    )
    assert recommendation.to_operator_proposal()["live_apply_allowed"] is False


def test_line_noise_audio_selects_notch_stack_before_hpf_or_broad_cut():
    duration_sec = 3.0
    time = np.arange(int(SR * duration_sec), dtype=np.float64) / SR
    guitar = (
        0.06 * np.sin(2.0 * np.pi * 440.0 * time)
        + 0.045 * np.sin(2.0 * np.pi * 880.0 * time)
        + 0.035 * np.sin(2.0 * np.pi * 1760.0 * time)
    )
    hum = (
        0.045 * np.sin(2.0 * np.pi * 60.0 * time)
        + 0.025 * np.sin(2.0 * np.pi * 120.0 * time)
        + 0.016 * np.sin(2.0 * np.pi * 180.0 * time)
    )
    samples = np.clip(guitar + hum, -0.95, 0.95)

    recommendation = analyze_smart_dsp_audio(
        samples,
        target_id="ch:11",
        source_role="electric_guitar",
        source_label="GTR DI",
        sample_rate=SR,
        source_identity_confirmed=True,
        source_identity_confidence=0.9,
        current_state_cache_freshness=_fresh(),
        current_state_readback_available=True,
        rollback_plan_available=True,
        target_plane_lufs=-22.0,
    )

    audio_evidence = recommendation.evidence["audio_evidence"]
    assert audio_evidence["line_noise_score"] >= 0.55
    assert audio_evidence["line_noise_frequency_hz"] == 60.0
    assert audio_evidence["line_noise_harmonic_count"] >= 2
    assert "line_noise" in audio_evidence["problem_hints"]
    assert "line_noise" in recommendation.evidence["detected_problems"]
    assert recommendation.selected_strategy.strategy_id == "mains_hum_notch_stack"
    params = recommendation.selected_strategy.parameters
    assert params["filter_type"] == "notch_stack"
    assert params["frequency_hz"] == 60.0
    assert params["line_noise_notch_parameter_source"] == "measured_line_noise_harmonics"
    assert params["harmonic_notches"][0]["frequency_hz"] == 60.0
    assert params["harmonic_notches"][0]["prominence_db"] > 0.0
    assert params["harmonic_notches"][0]["gain_db"] < -3.5
    assert params["harmonic_notches"][0]["q"] >= 7.0
    broad_cut = [
        item
        for item in recommendation.alternatives
        if item.strategy_id == "broad_low_cut_hum_decoy"
    ][0]
    assert "broad_low_cut_does_not_target_mains_hum" in broad_cut.blockers
    assert recommendation.to_operator_proposal()["live_apply_allowed"] is False


def test_dynamic_muddy_vocal_audio_selects_wing_native_deq_low_mid_cleanup():
    duration_sec = 2.5
    time = np.arange(int(SR * duration_sec), dtype=np.float64) / SR
    phrase_envelope = 0.5 * (
        0.035 + 0.22 * (np.sin(2.0 * np.pi * 1.4 * time) > 0.25).astype(np.float64)
    )
    samples = phrase_envelope * (
        np.sin(2.0 * np.pi * 260.0 * time)
        + 0.22 * np.sin(2.0 * np.pi * 1800.0 * time)
    )

    recommendation = analyze_smart_dsp_audio(
        samples,
        target_id="ch:5",
        source_role="lead_vocal",
        source_label="Lead Vox",
        sample_rate=SR,
        source_identity_confirmed=True,
        source_identity_confidence=0.91,
        current_state_cache_freshness=_fresh(),
        current_state_readback_available=True,
        rollback_plan_available=True,
        target_plane_lufs=-24.0,
    )

    audio_evidence = recommendation.evidence["audio_evidence"]
    assert audio_evidence["muddiness_score"] >= 0.62
    assert audio_evidence["low_mid_bloom_frequency_hz"] is not None
    assert 250.0 <= audio_evidence["low_mid_bloom_frequency_hz"] <= 270.0
    assert audio_evidence["low_mid_bloom_db"] >= 5.5
    assert audio_evidence["low_mid_bloom_peak_dbfs"] < 0.0
    assert audio_evidence["low_mid_bloom_rise_time_ms"] > 0.0
    assert audio_evidence["low_mid_bloom_recovery_ms"] > 0.0
    assert audio_evidence["dynamics_variance_score"] >= 0.42
    assert "muddy" in audio_evidence["problem_hints"]
    assert recommendation.selected_strategy.strategy_id == "vocal_dynamic_low_mid_de_mud"
    assert recommendation.selected_strategy.method_family == "dynamic_eq"
    params = recommendation.selected_strategy.parameters
    assert params["filter_type"] == "dynamic_peaking"
    assert params["frequency_source"] == "measured_low_mid_bloom_peak"
    assert params["frequency_hz"] == audio_evidence["low_mid_bloom_frequency_hz"]
    assert params["low_mid_bloom_frequency_hz"] == audio_evidence[
        "low_mid_bloom_frequency_hz"
    ]
    assert params["gain_db"] < 0.0
    assert params["max_gain_reduction_db"] > abs(params["gain_db"])
    assert params["wing_dynamic_eq_model_osc_value"] == WING_DYNAMIC_EQ_GATE_MODEL
    assert params["wing_dynamic_eq_filter_osc_value"] == "BP"
    assert params["wing_dynamic_eq_gain_osc_value"] < 0.0
    assert recommendation.selected_strategy.future_wing_readback_contract[
        "dynamic_eq_readback"
    ]["wing_dynamic_eq_model_readback_must_equal"] == WING_DYNAMIC_EQ_GATE_MODEL
    assert "eq_band_readback" not in recommendation.selected_strategy.future_wing_readback_contract

    alternatives = {candidate.strategy_id: candidate for candidate in recommendation.alternatives}
    assert "low_mid_corrective_cut" in alternatives
    assert "vocal_low_mid_multiband_control" in alternatives
    static_params = alternatives["low_mid_corrective_cut"].parameters
    assert static_params["filter_type"] == "peaking"
    assert static_params["frequency_source"] == "measured_low_mid_bloom_peak"
    assert static_params["frequency_hz"] == audio_evidence["low_mid_bloom_frequency_hz"]
    assert static_params["gain_db"] < 0.0
    dynamic_params = params
    assert dynamic_params["muddy_dynamic_parameter_source"] == "measured_low_mid_bloom"
    assert dynamic_params["low_mid_bloom_db"] == audio_evidence["low_mid_bloom_db"]
    assert dynamic_params["low_mid_bloom_peak_dbfs"] == audio_evidence[
        "low_mid_bloom_peak_dbfs"
    ]
    assert dynamic_params["low_mid_bloom_rise_time_ms"] == audio_evidence[
        "low_mid_bloom_rise_time_ms"
    ]
    assert dynamic_params["low_mid_bloom_recovery_ms"] == audio_evidence[
        "low_mid_bloom_recovery_ms"
    ]
    multiband_params = alternatives["vocal_low_mid_multiband_control"].parameters
    assert multiband_params["frequency_source"] == "measured_low_mid_bloom_peak"
    assert multiband_params["band_center_hz"] == audio_evidence[
        "low_mid_bloom_frequency_hz"
    ]
    assert "multiband_change_requires_operator_review" in alternatives[
        "vocal_low_mid_multiband_control"
    ].blockers


def test_live_dynamic_muddy_vocal_audio_prefers_static_cut_before_dynamic_eq():
    duration_sec = 2.5
    time = np.arange(int(SR * duration_sec), dtype=np.float64) / SR
    phrase_envelope = 0.5 * (
        0.035 + 0.22 * (np.sin(2.0 * np.pi * 1.4 * time) > 0.25).astype(np.float64)
    )
    samples = phrase_envelope * (
        np.sin(2.0 * np.pi * 260.0 * time)
        + 0.22 * np.sin(2.0 * np.pi * 1800.0 * time)
    )

    recommendation = analyze_smart_dsp_audio(
        samples,
        target_id="ch:5",
        source_role="lead_vocal",
        source_label="Lead Vox",
        sample_rate=SR,
        mode="live",
        active_performance=True,
        source_identity_confirmed=True,
        source_identity_confidence=0.91,
        current_state_cache_freshness=_fresh(),
        current_state_readback_available=True,
        rollback_plan_available=True,
        target_plane_lufs=-24.0,
    )

    audio_evidence = recommendation.evidence["audio_evidence"]
    assert audio_evidence["muddiness_score"] >= 0.62
    assert audio_evidence["low_mid_bloom_frequency_hz"] is not None
    assert audio_evidence["low_mid_bloom_db"] >= 5.5
    assert audio_evidence["dynamics_variance_score"] >= 0.42
    assert recommendation.selected_strategy.strategy_id == "low_mid_corrective_cut"
    assert recommendation.selected_strategy.method_family == "eq"
    params = recommendation.selected_strategy.parameters
    assert params["frequency_source"] == "measured_low_mid_bloom_peak"
    assert params["frequency_hz"] == audio_evidence["low_mid_bloom_frequency_hz"]
    assert params["gain_db"] == -2.6
    assert params["live_eq_global_on_mix_write_forbidden"] is True
    assert params["live_eq_band_operation_write_forbidden"] is True
    assert params["live_eq_band_operation_readback_required"] is True
    assert params["live_eq_requires_concrete_band_readback"] is True
    assert (
        params["live_eq_band_operation_apply_blocked_without_concrete_band_readback"]
        is True
    )
    assert params["live_eq_mix_review_only_value"] == WING_EQ_MIX_BASELINE_PERCENT
    assert "wing_eq_on_osc_value" not in params
    assert "wing_eq_mix_osc_value" not in params
    assert "wing_eq_band_operations" not in params
    assert params["live_eq_band_operation_review_only_values"] == [
        {
            "band_readback_required": True,
            "filter_family": "parametric_band",
            "filter_type": "peaking",
            "frequency_hz": audio_evidence["low_mid_bloom_frequency_hz"],
            "gain_db": -2.6,
            "path_suffixes": {
                "frequency": "eq/{band}f",
                "gain": "eq/{band}g",
                "q": "eq/{band}q",
            },
            "protected_band_policy": (
                "do_not_overwrite_feedback_notch_or_operator_locked_band"
            ),
            "q": 1.35,
            "value_type": "float",
        }
    ]
    proposal = recommendation.to_operator_proposal()
    assert proposal["selected_wing_osc_summary"][
        "wing_osc_apply_parameter_values"
    ] == {}
    assert proposal["selected_wing_osc_readback_summary"][
        "wing_osc_readback_apply_parameter_values"
    ] == {}
    alternatives = {candidate.strategy_id: candidate for candidate in recommendation.alternatives}
    dynamic = alternatives["vocal_dynamic_low_mid_de_mud"]
    assert "live_vocal_dynamic_low_mid_requires_operator_review" in dynamic.blockers
    assert dynamic.parameters["muddy_dynamic_parameter_source"] == "measured_low_mid_bloom"
    tradeoffs = {
        row["strategy_id"]: row
        for row in recommendation.evidence["strategy_search"]["alternative_tradeoffs"]
    }
    assert tradeoffs["vocal_dynamic_low_mid_de_mud"]["why_not_selected"] == (
        "live_vocal_low_mid_static_cut_priority:"
        "bounded_static_cut_precedes_dynamic_eq"
    )
    assert recommendation.evidence["operator_explanation_sections"][
        "selection_policy"
    ] == "live_vocal_low_mid_static_cut_priority"
    assert recommendation.to_operator_proposal()["live_apply_allowed"] is False


def test_live_dynamic_vocal_audio_prefers_manual_ride_before_compression():
    duration_sec = 3.0
    time = np.arange(int(SR * duration_sec), dtype=np.float64) / SR
    phrase_gate = (np.sin(2.0 * np.pi * 1.25 * time) > -0.2).astype(np.float64)
    phrase_envelope = 0.018 + 0.2 * phrase_gate * (
        0.55 + 0.45 * np.sin(2.0 * np.pi * 0.55 * time) ** 2
    )
    samples = phrase_envelope * (
        0.46 * np.sin(2.0 * np.pi * 720.0 * time)
        + 0.28 * np.sin(2.0 * np.pi * 1450.0 * time)
        + 0.28 * np.sin(2.0 * np.pi * 4200.0 * time)
    )
    samples = np.clip(samples, -0.95, 0.95)

    recommendation = analyze_smart_dsp_audio(
        samples,
        target_id="ch:5",
        source_role="lead_vocal",
        source_label="Lead Vox",
        sample_rate=SR,
        mode="live",
        active_performance=True,
        source_identity_confirmed=True,
        source_identity_confidence=0.91,
        current_state_cache_freshness=_fresh(),
        current_state_readback_available=True,
        rollback_plan_available=True,
        target_plane_lufs=-24.0,
        extra_problem_hints=("dynamic_control",),
    )

    audio_evidence = recommendation.evidence["audio_evidence"]
    assert audio_evidence["dynamics_variance_score"] >= 0.55
    assert audio_evidence["lra_proxy_db"] >= 10.0
    assert audio_evidence["dynamics_peak_to_median_db"] >= 10.0
    assert "dynamic_control" in audio_evidence["problem_hints"]
    assert recommendation.selected_strategy.strategy_id == (
        "manual_level_ride_before_compression"
    )
    assert recommendation.selected_strategy.method_family == "analysis_only"
    assert recommendation.evidence["operator_explanation_sections"][
        "selection_policy"
    ] == "live_dynamic_harsh_vocal_ride_priority"
    params = recommendation.selected_strategy.parameters
    assert params["live_vocal_dynamics_ride_priority"] is True
    assert params["live_dynamic_harsh_vocal_ride_priority"] is True
    assert params["live_dynamic_harshness_tone_processing_deferred"] is True
    assert params["live_dynamic_harshness_remeasure_after_ride_required"] is True
    assert params["forbidden_without_wing_proof"] == (
        "compressor_dynamic_eq_static_presence_cut_until_ride_remeasure"
    )
    alternatives = {
        candidate.strategy_id: candidate for candidate in recommendation.alternatives
    }
    assert (
        "live_vocal_leveling_compression_requires_operator_review_and_wing_proof"
        in alternatives["vocal_phrase_leveling_compression"].blockers
    )
    assert alternatives["vocal_phrase_leveling_compression"].parameters[
        "dynamics_parameter_source"
    ] == "measured_envelope"
    tradeoffs = {
        row["strategy_id"]: row
        for row in recommendation.evidence["strategy_search"]["alternative_tradeoffs"]
    }
    assert tradeoffs["vocal_phrase_leveling_compression"]["why_not_selected"] == (
        "live_dynamic_harsh_vocal_ride_priority:"
        "ride_and_remeasure_before_tone_or_compressor_change"
    )
    assert recommendation.to_operator_proposal()["live_apply_allowed"] is False


def test_live_sibilant_vocal_audio_selects_wing_native_ds902_de_esser():
    samples = (
        _tone(220.0, amplitude=0.045)
        + _tone(1300.0, amplitude=0.03)
        + _tone(7600.0, amplitude=0.09)
    )

    recommendation = analyze_smart_dsp_audio(
        samples,
        target_id="ch:6",
        source_role="lead_vocal",
        source_label="Lead Vox",
        mode="live",
        sample_rate=SR,
        source_identity_confirmed=True,
        source_identity_confidence=0.9,
        current_state_cache_freshness=_fresh(),
        current_state_readback_available=True,
        rollback_plan_available=True,
        target_plane_lufs=-23.0,
        active_performance=True,
    )

    audio_evidence = recommendation.evidence["audio_evidence"]
    assert audio_evidence["sibilance_score"] >= 0.45
    assert 7000.0 <= audio_evidence["sibilance_frequency_hz"] <= 8200.0
    assert recommendation.selected_strategy.strategy_id == "vocal_de_esser"
    assert recommendation.selected_strategy.method_family == "de_esser"
    params = recommendation.selected_strategy.parameters
    assert params["makeup_gain_db"] == 0.0
    assert params["de_esser_parameter_source"] == "sibilance_score_fallback"
    assert "wing_de_esser_model_osc_value" not in params
    assert params["live_de_esser_gate_slot_model_write_forbidden"] is True
    assert params["live_de_esser_requires_existing_ds902_model_readback"] is True
    assert params["live_de_esser_model_review_only_value"] == WING_DE_ESSER_GATE_MODEL
    assert "wing_de_esser_on_osc_value" not in params
    assert "wing_de_esser_mode_osc_value" not in params
    assert params["live_de_esser_on_mode_write_forbidden"] is True
    assert params["live_de_esser_on_mode_readback_required"] is True
    assert params["live_de_esser_on_review_only_value"] == 1
    assert params["live_de_esser_mode_review_only_value"] == "HF"
    assert params["live_de_esser_on_mode_review_only_values"] == {
        "wing_de_esser_on_osc_value": 1,
        "wing_de_esser_mode_osc_value": "HF",
    }
    assert "wing_de_esser_frequency_osc_value" not in params
    assert "wing_de_esser_range_osc_value" not in params
    assert params["live_de_esser_parameter_write_forbidden"] is True
    assert params["live_de_esser_parameter_readback_required"] is True
    assert params["live_de_esser_parameter_apply_blocked_without_wing_proof"] is True
    parameter_review_values = params["live_de_esser_parameter_review_only_values"]
    assert parameter_review_values["wing_de_esser_on_osc_value"] == 1
    assert parameter_review_values["wing_de_esser_mode_osc_value"] == "HF"
    assert 7000.0 <= parameter_review_values["wing_de_esser_frequency_osc_value"] <= 8200.0
    assert parameter_review_values["wing_de_esser_range_osc_value"] == params["range_db"]
    assert params["live_de_esser_parameter_review_only_reason"] == (
        "live_de_esser_must_read_existing_ds902_state_and_operator_confirm_before_parameter_apply"
    )
    assert params["wing_de_esser_gate_slot_conflict_policy"] == (
        WING_DE_ESSER_GATE_SLOT_CONFLICT_POLICY
    )
    assert recommendation.selected_strategy.future_wing_readback_contract[
        "de_esser_readback"
    ]["gate_model_readback_must_equal"] == WING_DE_ESSER_GATE_MODEL
    proposal = recommendation.to_operator_proposal()
    assert (
        "wing_de_esser_model_osc_value"
        not in proposal["selected_wing_osc_summary"][
            "wing_osc_apply_parameter_values"
        ]
    )
    assert (
        "wing_de_esser_on_osc_value"
        not in proposal["selected_wing_osc_summary"][
            "wing_osc_apply_parameter_values"
        ]
    )
    assert (
        "wing_de_esser_mode_osc_value"
        not in proposal["selected_wing_osc_summary"][
            "wing_osc_apply_parameter_values"
        ]
    )
    assert proposal["selected_wing_osc_summary"]["wing_osc_apply_parameter_values"] == {}
    assert proposal["selected_wing_osc_readback_summary"][
        "wing_osc_readback_apply_parameter_values"
    ] == {}
    alternatives = {
        candidate.strategy_id: candidate for candidate in recommendation.alternatives
    }
    assert alternatives["static_sibilance_band_cut"].parameters[
        "static_fallback_for_unmapped_de_esser"
    ] is True
    assert recommendation.evidence["audio_evidence_source"] == (
        SMART_DSP_EVIDENCE_EXTRACTOR_VERSION
    )
    assert proposal["live_apply_allowed"] is False


def test_soundcheck_harsh_sibilant_audio_selects_wing_native_ds902_de_esser():
    samples = (
        _tone(220.0, amplitude=0.025)
        + _tone(1300.0, amplitude=0.025)
        + _tone(3400.0, amplitude=0.11)
        + _tone(7600.0, amplitude=0.095)
    )

    recommendation = analyze_smart_dsp_audio(
        samples,
        target_id="ch:6",
        source_role="lead_vocal",
        source_label="Lead Vox",
        sample_rate=SR,
        source_identity_confirmed=True,
        source_identity_confidence=0.92,
        current_state_cache_freshness=_fresh(),
        current_state_readback_available=True,
        rollback_plan_available=True,
        target_plane_lufs=-16.0,
    )

    audio_evidence = recommendation.evidence["audio_evidence"]
    assert audio_evidence["harshness_score"] >= 0.65
    assert audio_evidence["presence_harshness_score"] >= 0.45
    assert audio_evidence["sibilance_score"] >= 0.65
    assert recommendation.selected_strategy.strategy_id == "vocal_de_esser"
    assert recommendation.selected_strategy.method_family == "de_esser"
    params = recommendation.selected_strategy.parameters
    assert params["frequency_source"] == "measured_sibilance_peak"
    assert params["bandwidth_source"] == "measured_sibilance_bandwidth"
    assert params["range_db"] > 0.0
    assert params["max_gain_reduction_db"] == params["range_db"]
    assert params["wing_de_esser_model_osc_value"] == WING_DE_ESSER_GATE_MODEL
    assert params["wing_de_esser_mode_osc_value"] == "HF"
    assert params["wing_de_esser_range_osc_value"] > 0.0
    assert params["wing_de_esser_frequency_osc_value"] == params["frequency_hz"]
    assert params["makeup_gain_db"] == 0.0
    readback = recommendation.selected_strategy.future_wing_readback_contract
    assert readback["de_esser_readback"]["gate_model_readback_must_equal"] == (
        WING_DE_ESSER_GATE_MODEL
    )
    assert readback["wing_osc_applicability"]["native_osc_supported"] is True
    assert "eq_band_readback" not in readback
    alternatives = {
        candidate.strategy_id: candidate for candidate in recommendation.alternatives
    }
    assert alternatives["static_sibilance_band_cut"].parameters[
        "static_fallback_for_unmapped_de_esser"
    ] is True
    combo_params = alternatives["vocal_de_esser_dynamic_presence_combo"].parameters
    assert combo_params["processing_chain"] == "de_esser_then_dynamic_presence_control"
    assert combo_params["de_esser_max_gain_reduction_db"] > 0.0
    assert combo_params["dynamic_eq_max_gain_reduction_db"] > 0.0
    assert recommendation.to_operator_proposal()["can_apply_live"] is False


def test_harsh_dynamic_vocal_audio_selects_compressor_envelope_smoothing():
    duration_sec = 2.5
    time = np.arange(int(SR * duration_sec), dtype=np.float64) / SR
    pulse = (np.sin(2.0 * np.pi * 2.2 * time) > 0.72).astype(np.float64)
    envelope = 0.25 * (0.04 + 0.35 * pulse)
    samples = envelope * (
        np.sin(2.0 * np.pi * 3300.0 * time)
        + 0.8 * np.sin(2.0 * np.pi * 1500.0 * time)
        + 0.6 * np.sin(2.0 * np.pi * 700.0 * time)
    )

    recommendation = analyze_smart_dsp_audio(
        samples,
        target_id="ch:8",
        source_role="lead_vocal",
        source_label="Harsh Lead",
        sample_rate=SR,
        source_identity_confirmed=True,
        source_identity_confidence=0.92,
        current_state_cache_freshness=_fresh(),
        current_state_readback_available=True,
        rollback_plan_available=True,
        target_plane_lufs=-16.0,
    )

    audio_evidence = recommendation.evidence["audio_evidence"]
    assert audio_evidence["harshness_score"] >= 0.55
    assert audio_evidence["sibilance_score"] < 0.45
    assert audio_evidence["dynamics_variance_score"] >= 0.45
    assert recommendation.selected_strategy.strategy_id == (
        "vocal_compressor_envelope_smoothing"
    )
    params = recommendation.selected_strategy.parameters
    assert params["vocal_envelope_parameter_source"] == "measured_vocal_dynamics_envelope"
    assert params["dynamics_peak_to_median_db"] == 19.8
    assert params["dynamics_rise_time_ms"] == 120.0
    assert params["dynamics_recovery_ms"] == 60.0
    assert params["attack_ms"] == 38.0
    assert params["release_ms"] == 130.0
    assert params["makeup_gain_db"] == 0.0

    presence_eq = [
        candidate
        for candidate in recommendation.alternatives
        if candidate.strategy_id == "presence_dynamic_eq_cut"
    ][0]
    assert presence_eq.parameters["frequency_hz"] == 3300.0
    assert presence_eq.parameters["frequency_source"] == "measured_narrow_presence_peak"
    assert presence_eq.parameters["vocal_presence_parameter_source"] == (
        "measured_narrow_presence_peak"
    )
    assert presence_eq.parameters["q"] == 4.0
    assert presence_eq.parameters["max_gain_reduction_db"] == 3.9

    deesser = [
        candidate
        for candidate in recommendation.alternatives
        if candidate.strategy_id == "vocal_de_esser"
    ][0]
    assert "sibilance_evidence_low_for_de_esser" in deesser.blockers
    assert recommendation.to_operator_proposal()["live_apply_allowed"] is False


def test_soundcheck_feedback_ring_audio_selects_measured_narrow_notch():
    samples = (
        _tone(850.0, amplitude=0.035)
        + _tone(1450.0, amplitude=0.026)
        + _tone(3150.0, amplitude=0.16)
    )

    recommendation = analyze_smart_dsp_audio(
        samples,
        target_id="ch:5",
        source_role="lead_vocal",
        source_label="Lead Vox",
        sample_rate=SR,
        source_identity_confirmed=True,
        source_identity_confidence=0.92,
        current_state_cache_freshness=_fresh(),
        current_state_readback_available=True,
        rollback_plan_available=True,
        target_plane_lufs=-16.0,
        monitor_feedback_risk=0.42,
    )

    audio_evidence = recommendation.evidence["audio_evidence"]
    assert audio_evidence["feedback_risk"] >= 0.62
    assert 3100.0 <= audio_evidence["feedback_frequency_hz"] <= 3200.0
    assert audio_evidence["feedback_frequency_confidence"] >= 0.6
    assert audio_evidence["feedback_frequency_source"] == "measured_feedback_ring_peak"
    assert "feedback_risk" in recommendation.evidence["detected_problems"]
    assert recommendation.selected_strategy.strategy_id == "narrow_feedback_notch_review"
    assert 3100.0 <= recommendation.selected_strategy.parameters["frequency_hz"] <= 3200.0
    assert recommendation.selected_strategy.parameters[
        "frequency_source"
    ] == "measured_feedback_ring_peak"
    assert recommendation.to_operator_proposal()["live_apply_allowed"] is False


def test_unknown_vocal_audio_inference_keeps_source_identity_guard_without_opt_in():
    samples = (
        _tone(220.0, amplitude=0.045)
        + _tone(1300.0, amplitude=0.03)
        + _tone(7600.0, amplitude=0.09)
    )

    recommendation = analyze_smart_dsp_audio(
        samples,
        target_id="ch:6",
        source_role="unknown",
        source_label="",
        sample_rate=SR,
        current_state_cache_freshness=_fresh(),
        current_state_readback_available=True,
        rollback_plan_available=True,
        target_plane_lufs=-23.0,
    )

    audio_evidence = recommendation.evidence["audio_evidence"]
    assert audio_evidence["declared_source_role"] == "unknown"
    assert audio_evidence["inferred_source_family"] == "vocal"
    assert audio_evidence["source_identity_inference_confidence"] >= 0.6
    assert audio_evidence["source_identity_inference_used"] is True
    assert recommendation.source_family == "vocal"
    assert recommendation.status == "analysis_only"
    assert "source_identity_not_confirmed" in recommendation.blockers
    assert (
        recommendation.selected_strategy.strategy_id
        == "hold_for_source_identity_confirmation"
    )
    assert recommendation.selected_strategy.method_family == "analysis_only"
    assert (
        recommendation.selected_strategy.parameters["required_context"]
        == "operator_patch_label_pfl_source_role_confirmation"
    )
    assert any(
        candidate.strategy_id == "vocal_de_esser"
        for candidate in recommendation.alternatives
    )
    assert recommendation.to_operator_proposal()["live_apply_allowed"] is False


def test_unknown_vocal_audio_inference_can_drive_offline_strategy_with_opt_in():
    samples = (
        _tone(220.0, amplitude=0.045)
        + _tone(1300.0, amplitude=0.03)
        + _tone(7600.0, amplitude=0.09)
    )

    recommendation = analyze_smart_dsp_audio(
        samples,
        target_id="ch:6",
        source_role="unknown",
        source_label="",
        sample_rate=SR,
        current_state_cache_freshness=_fresh(),
        current_state_readback_available=True,
        rollback_plan_available=True,
        target_plane_lufs=-23.0,
        allow_inferred_source_identity=True,
    )

    assert recommendation.status == "proposal_only"
    assert recommendation.source_family == "vocal"
    assert recommendation.evidence["source_identity"]["confirmed"] is True
    assert recommendation.evidence["source_identity_inference_used"] is True
    assert recommendation.evidence["source_identity_inference_confidence"] >= 0.6
    assert recommendation.selected_strategy.strategy_id == "vocal_de_esser"
    assert recommendation.selected_strategy.method_family == "de_esser"
    assert recommendation.selected_strategy.parameters["wing_de_esser_model_osc_value"] == (
        WING_DE_ESSER_GATE_MODEL
    )
    assert recommendation.selected_strategy.parameters[
        "wing_de_esser_gate_slot_conflict_policy"
    ] == WING_DE_ESSER_GATE_SLOT_CONFLICT_POLICY
    assert recommendation.selected_strategy.parameters["makeup_gain_db"] == 0.0
    alternatives = {
        candidate.strategy_id: candidate for candidate in recommendation.alternatives
    }
    assert alternatives["static_sibilance_band_cut"].parameters[
        "static_fallback_for_unmapped_de_esser"
    ] is True
    assert recommendation.to_operator_proposal()["can_apply_live"] is False


def test_dynamic_vocal_audio_selects_phrase_leveling_compression():
    base = _tone(950.0, amplitude=1.0, duration_sec=2.4)
    frame = np.arange(base.size)
    envelope = np.where((frame // int(SR * 0.3)) % 2 == 0, 0.06, 0.42)
    samples = base * envelope

    recommendation = analyze_smart_dsp_audio(
        samples,
        target_id="ch:7",
        source_role="lead_vocal",
        source_label="Dynamic Lead",
        sample_rate=SR,
        source_identity_confirmed=True,
        source_identity_confidence=0.9,
        current_state_cache_freshness=_fresh(),
        current_state_readback_available=True,
        rollback_plan_available=True,
        target_plane_lufs=-16.0,
    )

    audio_evidence = recommendation.evidence["audio_evidence"]
    assert audio_evidence["dynamics_variance_score"] >= 0.45
    assert audio_evidence["dynamics_peak_to_median_db"] >= 10.0
    assert audio_evidence["dynamics_rise_time_ms"] > 0.0
    assert audio_evidence["dynamics_recovery_ms"] > 0.0
    assert "dynamic_control" in audio_evidence["problem_hints"]
    assert recommendation.selected_strategy.strategy_id == "vocal_phrase_leveling_compression"
    params = recommendation.selected_strategy.parameters
    assert params["source_family"] == "vocal"
    assert params["dynamics_parameter_source"] == "measured_envelope"
    assert params["dynamics_peak_to_median_db"] == audio_evidence[
        "dynamics_peak_to_median_db"
    ]
    assert params["dynamics_recovery_ms"] == audio_evidence["dynamics_recovery_ms"]
    assert params["threshold_dbfs"] < 0.0
    assert params["ratio"] > 1.0
    assert params["release_ms"] > params["attack_ms"]
    assert params["detector"] == "program_rms_peak_blend"
    assert params["preserve_words_and_consonants"] is True
    assert params["de_esser_after_compression_check"] is True


def test_dynamic_guitar_audio_selects_pick_sustain_compression():
    base = (
        0.75 * _tone(440.0, amplitude=1.0, duration_sec=2.4)
        + 0.25 * _tone(880.0, amplitude=1.0, duration_sec=2.4)
    )
    frame = np.arange(base.size)
    envelope = np.where((frame // int(SR * 0.28)) % 2 == 0, 0.055, 0.36)
    samples = base * envelope

    recommendation = analyze_smart_dsp_audio(
        samples,
        target_id="ch:11",
        source_role="electric_guitar",
        source_label="GTR L",
        sample_rate=SR,
        source_identity_confirmed=True,
        source_identity_confidence=0.9,
        current_state_cache_freshness=_fresh(),
        current_state_readback_available=True,
        rollback_plan_available=True,
        target_plane_lufs=-16.0,
    )

    audio_evidence = recommendation.evidence["audio_evidence"]
    assert audio_evidence["dynamics_variance_score"] >= 0.45
    assert audio_evidence["dynamics_peak_to_median_db"] >= 10.0
    assert "dynamic_control" in audio_evidence["problem_hints"]
    assert recommendation.selected_strategy.strategy_id == (
        "guitar_pick_sustain_compression"
    )
    assert recommendation.selected_strategy.method_family == "dynamics"
    params = recommendation.selected_strategy.parameters
    assert params["source_family"] == "guitar"
    assert params["dynamics_parameter_source"] == "measured_envelope"
    assert params["dynamics_peak_to_median_db"] == audio_evidence[
        "dynamics_peak_to_median_db"
    ]
    assert params["preserve_pick_attack"] is True
    assert params["makeup_gain_db"] == 0.0
    assert recommendation.to_operator_proposal()["live_apply_allowed"] is False


def test_noisy_tom_audio_selects_gentle_expander_over_gate():
    rng = np.random.default_rng(7)
    duration_sec = 2.5
    samples = rng.normal(scale=10 ** (-47.0 / 20.0), size=int(SR * duration_sec))
    for start_sec in (0.2, 0.8, 1.4, 2.0):
        start = int(SR * start_sec)
        length = int(SR * 0.22)
        envelope = np.exp(-np.linspace(0.0, 5.0, length))
        samples[start : start + length] += 0.55 * envelope * _tone(
            145.0,
            amplitude=1.0,
            duration_sec=length / SR,
        )

    recommendation = analyze_smart_dsp_audio(
        samples,
        target_id="ch:8",
        source_role="floor_tom",
        source_label="Floor Tom",
        sample_rate=SR,
        source_identity_confirmed=True,
        source_identity_confidence=0.86,
        current_state_cache_freshness=_fresh(),
        current_state_readback_available=True,
        rollback_plan_available=True,
        target_plane_lufs=-23.0,
    )

    assert recommendation.evidence["detected_problems"]
    assert "noisy_drum" in recommendation.evidence["detected_problems"]
    assert recommendation.selected_strategy.strategy_id == "gentle_expander_bleed_control"
    params = recommendation.selected_strategy.parameters
    assert params["expander_parameter_source"] == "measured_hit_tail_bleed_expander"
    assert params["gate_hit_peak_dbfs"] == -10.0
    assert params["gate_hit_to_bleed_margin_db"] == 37.1
    assert params["gate_tail_decay_ms"] == 800.0
    assert params["release_ms"] == 420.0
    assert params["range_db"] > 0.0
    assert params["ratio"] == "1:2"
    assert params["post_gate_hysteresis_review_db"] == 2.4
    blocked_gate = [
        item for item in recommendation.alternatives if item.strategy_id == "fast_drum_gate"
    ][0]
    assert "gate_tail_or_chatter_risk_high" in blocked_gate.blockers


def test_clean_tom_audio_selects_bleed_aware_gate():
    rng = np.random.default_rng(43)
    duration_sec = 2.5
    samples = rng.normal(scale=10 ** (-56.0 / 20.0), size=int(SR * duration_sec))
    for start_sec in (0.3, 1.2, 2.1):
        start = int(SR * start_sec)
        length = int(SR * 0.055)
        envelope = np.exp(-np.linspace(0.0, 10.0, length))
        local_time = np.arange(length, dtype=np.float64) / SR
        hit = np.sin(2.0 * np.pi * 145.0 * local_time)
        hit += 0.2 * np.sin(2.0 * np.pi * 220.0 * local_time)
        samples[start : start + length] += 0.45 * envelope * hit

    recommendation = analyze_smart_dsp_audio(
        samples,
        target_id="ch:8",
        source_role="floor_tom",
        source_label="Floor Tom",
        sample_rate=SR,
        source_identity_confirmed=True,
        source_identity_confidence=0.86,
        current_state_cache_freshness=_fresh(),
        current_state_readback_available=True,
        rollback_plan_available=True,
        target_plane_lufs=-35.0,
    )

    audio_evidence = recommendation.evidence["audio_evidence"]
    assert 0.35 <= audio_evidence["bleed_risk"] < 0.72
    assert audio_evidence["transient_strength"] >= 0.7
    assert audio_evidence["chatter_risk"] < 0.35
    assert audio_evidence["gate_hit_peak_dbfs"] < 0.0
    assert audio_evidence["gate_hit_to_bleed_margin_db"] > 20.0
    assert audio_evidence["gate_tail_decay_ms"] > 0.0
    assert "noisy_drum" in recommendation.evidence["detected_problems"]
    assert recommendation.selected_strategy.strategy_id == "bleed_aware_clean_hit_gate"
    params = recommendation.selected_strategy.parameters
    assert params["threshold_dbfs"] > audio_evidence["noise_floor_dbfs"]
    assert params["gate_parameter_source"] == "measured_hit_tail"
    assert params["gate_hit_peak_dbfs"] == audio_evidence["gate_hit_peak_dbfs"]
    assert params["gate_hit_to_bleed_margin_db"] == audio_evidence[
        "gate_hit_to_bleed_margin_db"
    ]
    assert params["gate_tail_decay_ms"] == audio_evidence["gate_tail_decay_ms"]
    assert params["release_ms"] > params["hold_ms"]
    assert params["range_db"] > 0.0
    assert params["ratio"] == "GATE"
    assert params["clean_hit_gate_ready"] is True
    assert params["requires_clean_hit_readback"] is True
    assert recommendation.to_operator_proposal()["live_apply_allowed"] is False


def test_snare_transient_audio_selects_peak_aware_transient_tamer():
    samples = _snare_transient_train()

    evidence = extract_smart_dsp_audio_evidence(
        samples,
        target_id="ch:9",
        source_role="snare_top",
        source_label="Snare Top",
        sample_rate=SR,
        target_plane_lufs=-25.0,
    )

    assert evidence.transient_strength >= 0.45
    assert evidence.transient_density >= 2.0
    assert evidence.transient_peak_dbfs < 0.0
    assert evidence.transient_rise_time_ms > 0.0
    assert evidence.transient_decay_ms > evidence.transient_rise_time_ms
    assert "transient_control" in evidence.problem_hints
    assert "drum_ring_resonance" not in evidence.problem_hints
    assert "noisy_drum" not in evidence.problem_hints

    recommendation = analyze_smart_dsp_audio(
        samples,
        target_id="ch:9",
        source_role="snare_top",
        source_label="Snare Top",
        sample_rate=SR,
        source_identity_confirmed=True,
        source_identity_confidence=0.9,
        current_state_cache_freshness=_fresh(),
        current_state_readback_available=True,
        rollback_plan_available=True,
        target_plane_lufs=evidence.estimated_lufs,
    )

    audio_evidence = recommendation.evidence["audio_evidence"]
    assert audio_evidence["transient_strength"] >= 0.45
    assert audio_evidence["transient_rise_time_ms"] > 0.0
    assert audio_evidence["transient_decay_ms"] > audio_evidence["transient_rise_time_ms"]
    assert "transient_control" in recommendation.evidence["detected_problems"]
    assert recommendation.selected_strategy.strategy_id == "drum_transient_tamer_compression"
    params = recommendation.selected_strategy.parameters
    assert params["detector"] == "peak"
    assert params["makeup_gain_db"] == 0.0
    assert params["transient_strength"] == audio_evidence["transient_strength"]
    assert params["transient_parameter_source"] == "measured_transient_envelope"
    assert params["transient_peak_dbfs"] == audio_evidence["transient_peak_dbfs"]
    assert params["transient_rise_time_ms"] == audio_evidence["transient_rise_time_ms"]
    assert params["transient_decay_ms"] == audio_evidence["transient_decay_ms"]
    assert recommendation.to_operator_proposal()["live_apply_allowed"] is False


def test_snare_ring_audio_selects_wing_native_deq_dynamic_notch():
    recommendation = analyze_smart_dsp_audio(
        _snare_ring_train(),
        target_id="ch:9",
        source_role="snare_top",
        source_label="Snare Top",
        sample_rate=SR,
        source_identity_confirmed=True,
        source_identity_confidence=0.9,
        current_state_cache_freshness=_fresh(),
        current_state_readback_available=True,
        rollback_plan_available=True,
        target_plane_lufs=-25.0,
    )

    audio_evidence = recommendation.evidence["audio_evidence"]
    assert audio_evidence["drum_ring_resonance_score"] >= 0.75
    assert 390.0 <= audio_evidence["drum_ring_frequency_hz"] <= 450.0
    assert audio_evidence["drum_ring_bandwidth_hz"] > 0.0
    assert audio_evidence["drum_ring_peak_dbfs"] < -1.0
    assert audio_evidence["drum_ring_decay_ms"] > 0.0
    assert "drum_ring_resonance" in audio_evidence["problem_hints"]
    assert recommendation.selected_strategy.strategy_id == "drum_dynamic_ring_notch_control"
    assert recommendation.selected_strategy.method_family == "dynamic_eq"
    params = recommendation.selected_strategy.parameters
    assert params["source_family"] == "snare"
    assert 390.0 <= params["frequency_hz"] <= 450.0
    assert params["drum_ring_parameter_source"] == "measured_drum_ring_decay"
    assert params["drum_ring_bandwidth_hz"] == audio_evidence["drum_ring_bandwidth_hz"]
    assert params["drum_ring_peak_dbfs"] == audio_evidence["drum_ring_peak_dbfs"]
    assert params["drum_ring_decay_ms"] == audio_evidence["drum_ring_decay_ms"]
    assert params["q"] != 4.0
    assert params["max_gain_reduction_db"] > 0.0
    assert params["wing_dynamic_eq_model_osc_value"] == WING_DYNAMIC_EQ_GATE_MODEL
    assert params["wing_dynamic_eq_filter_osc_value"] == "BP"
    assert params["wing_dynamic_eq_gain_osc_value"] < 0.0
    assert params["wing_dynamic_eq_frequency_osc_value"] == params["frequency_hz"]
    assert recommendation.selected_strategy.future_wing_readback_contract[
        "wing_osc_applicability"
    ]["native_osc_supported"] is True
    assert recommendation.selected_strategy.future_wing_readback_contract[
        "dynamic_eq_readback"
    ]["wing_dynamic_eq_model_readback_must_equal"] == WING_DYNAMIC_EQ_GATE_MODEL
    assert recommendation.evidence["operator_explanation_sections"][
        "selection_policy"
    ] == "weighted_strategy_score"
    static = [
        candidate
        for candidate in recommendation.alternatives
        if candidate.strategy_id == "drum_static_ring_notch_cut"
    ][0]
    assert static.parameters["gain_db"] < 0.0
    assert static.parameters["no_boost"] is True
    assert static.future_wing_readback_contract["wing_osc_applicability"][
        "native_osc_supported"
    ] is True
    assert params["release_ms"] != 180.0
    assert params["makeup_gain_db"] == 0.0
    assert recommendation.to_operator_proposal()["live_apply_allowed"] is False


def test_live_snare_ring_audio_selects_static_notch():
    recommendation = analyze_smart_dsp_audio(
        _snare_ring_train(),
        target_id="ch:9",
        source_role="snare_top",
        source_label="Snare Top",
        mode="live",
        sample_rate=SR,
        source_identity_confirmed=True,
        source_identity_confidence=0.9,
        current_state_cache_freshness=_fresh(),
        current_state_readback_available=True,
        rollback_plan_available=True,
        target_plane_lufs=-25.0,
        active_performance=True,
    )

    audio_evidence = recommendation.evidence["audio_evidence"]
    assert audio_evidence["drum_ring_resonance_score"] >= 0.75
    assert audio_evidence["drum_ring_bandwidth_hz"] > 0.0
    assert audio_evidence["drum_ring_peak_dbfs"] < -1.0
    assert audio_evidence["drum_ring_decay_ms"] > 0.0
    assert "drum_ring_resonance" in audio_evidence["problem_hints"]
    assert recommendation.selected_strategy.strategy_id == "drum_static_ring_notch_cut"
    assert recommendation.selected_strategy.method_family == "eq"
    params = recommendation.selected_strategy.parameters
    assert params["gain_db"] < 0.0
    assert params["no_boost"] is True
    assert params["drum_ring_parameter_source"] == "measured_drum_ring_decay"
    assert params["drum_ring_bandwidth_hz"] == audio_evidence["drum_ring_bandwidth_hz"]
    assert params["drum_ring_peak_dbfs"] == audio_evidence["drum_ring_peak_dbfs"]
    assert params["drum_ring_decay_ms"] == audio_evidence["drum_ring_decay_ms"]
    assert params["q"] != 3.4
    assert recommendation.to_operator_proposal()["live_apply_allowed"] is False


def test_extreme_noisy_tom_audio_selects_manual_mute_review():
    rng = np.random.default_rng(9)
    duration_sec = 2.5
    samples = rng.normal(scale=10 ** (-42.0 / 20.0), size=int(SR * duration_sec))
    for index, start_sec in enumerate(np.linspace(0.08, 2.35, 24)):
        start = int(SR * start_sec)
        length = int(SR * 0.025)
        envelope = np.exp(-np.linspace(0.0, 6.0, length))
        frequency_hz = 140.0 + 20.0 * (index % 3)
        samples[start : start + length] += 0.28 * envelope * _tone(
            frequency_hz,
            amplitude=1.0,
            duration_sec=length / SR,
        )

    recommendation = analyze_smart_dsp_audio(
        samples,
        target_id="ch:8",
        source_role="floor_tom",
        source_label="Floor Tom",
        sample_rate=SR,
        source_identity_confirmed=True,
        source_identity_confidence=0.86,
        current_state_cache_freshness=_fresh(),
        current_state_readback_available=True,
        rollback_plan_available=True,
        target_plane_lufs=-23.0,
    )

    audio_evidence = recommendation.evidence["audio_evidence"]
    assert audio_evidence["bleed_risk"] >= 0.8
    assert audio_evidence["tail_risk"] >= 0.72
    assert audio_evidence["chatter_risk"] >= 0.72
    assert recommendation.selected_strategy.strategy_id == "manual_tom_mute_between_hits"
    assert recommendation.selected_strategy.method_family == "mute"
    assert recommendation.selected_strategy.parameters["proposed_muted"] is True
    assert recommendation.to_operator_proposal()["live_apply_allowed"] is False

    expander = [
        item
        for item in recommendation.alternatives
        if item.strategy_id == "gentle_expander_bleed_control"
    ][0]
    assert "extreme_tail_chatter_prefers_manual_mute_review" in expander.blockers


def test_dull_vocal_audio_selects_bounded_presence_air_lift():
    samples = _tone(900.0, amplitude=0.10) + _tone(1300.0, amplitude=0.08)

    recommendation = analyze_smart_dsp_audio(
        samples,
        target_id="ch:5",
        source_role="lead_vocal",
        source_label="Lead Vox",
        sample_rate=SR,
        source_identity_confirmed=True,
        source_identity_confidence=0.9,
        current_state_cache_freshness=_fresh(),
        current_state_readback_available=True,
        rollback_plan_available=True,
        target_plane_lufs=-22.0,
    )

    audio_evidence = recommendation.evidence["audio_evidence"]
    assert audio_evidence["presence_deficit_score"] >= 0.55
    assert audio_evidence["presence_lift_frequency_hz"] is not None
    assert audio_evidence["presence_lift_frequency_hz"] != 3400.0
    assert audio_evidence["presence_lift_bandwidth_hz"] > 0.0
    assert (
        audio_evidence["presence_lift_parameter_source"]
        == "measured_presence_air_deficit"
    )
    assert "dull_presence" in audio_evidence["problem_hints"]
    assert recommendation.selected_strategy.strategy_id == "soundcheck_presence_air_lift"
    assert recommendation.selected_strategy.method_family == "eq"
    params = recommendation.selected_strategy.parameters
    assert params["frequency_hz"] == audio_evidence["presence_lift_frequency_hz"]
    assert params["frequency_source"] == "measured_presence_air_deficit"
    assert params["presence_lift_bandwidth_hz"] == audio_evidence[
        "presence_lift_bandwidth_hz"
    ]
    assert params["gain_db"] > 0.0
    assert params["bounded_boost_db"] <= 2.4
    assert recommendation.to_operator_proposal()["live_apply_allowed"] is False


def test_backing_vocal_audio_preserves_support_role_for_presence_lift():
    samples = _tone(900.0, amplitude=0.10) + _tone(1300.0, amplitude=0.08)

    recommendation = analyze_smart_dsp_audio(
        samples,
        target_id="ch:9",
        source_role="backing_vocal",
        source_label="BGV 1",
        sample_rate=SR,
        source_identity_confirmed=True,
        source_identity_confidence=0.9,
        current_state_cache_freshness=_fresh(),
        current_state_readback_available=True,
        rollback_plan_available=True,
        target_plane_lufs=-22.0,
    )

    audio_evidence = recommendation.evidence["audio_evidence"]
    assert audio_evidence["musical_role"] == "support"
    assert recommendation.evidence["musical_role"] == "support"
    assert recommendation.selected_strategy.strategy_id == "soundcheck_presence_air_lift"
    params = recommendation.selected_strategy.parameters
    assert params["musical_role"] == "support"
    assert params["role_intent"] == "preserve_support_tuck"
    assert params["bounded_boost_db"] == 1.2
    assert params["frequency_source"] == "measured_presence_air_deficit"
    assert params["presence_lift_bandwidth_hz"] > 0.0


def test_phase_unsafe_stereo_audio_reduces_width_for_soundcheck_request():
    left = _tone(440.0, amplitude=0.28)
    right = -left
    stereo = np.column_stack([left, right])

    recommendation = analyze_smart_dsp_audio(
        stereo,
        target_id="ch:12",
        source_role="electric_guitar",
        source_label="GTR L/R",
        sample_rate=SR,
        desired_stereo_width_request=45.0,
        source_identity_confirmed=True,
        source_identity_confidence=0.9,
        current_state_cache_freshness=_fresh(),
        current_state_readback_available=True,
        rollback_plan_available=True,
        target_plane_lufs=-16.0,
    )

    audio_evidence = recommendation.evidence["audio_evidence"]
    assert audio_evidence["phase_correlation"] < -0.95
    assert audio_evidence["mono_sum_delta_db"] <= -40.0
    assert recommendation.selected_strategy.strategy_id == "mono_compatibility_width_reduction"
    params = recommendation.selected_strategy.parameters
    assert params["target_width_percent"] == 0.0
    assert params["target_pan_percent"] == 0.0
    assert params["severe_cancellation"] is True
    assert recommendation.evidence["operator_explanation_sections"][
        "selection_policy"
    ] == "soundcheck_phase_mono_safety_priority"


def test_harsh_guitar_audio_selects_wing_native_deq_presence_tame():
    duration_sec = 2.0
    time = np.arange(int(SR * duration_sec), dtype=np.float64) / SR
    harsh_envelope = np.zeros_like(time)
    for start_sec in (0.18, 0.78, 1.34):
        start = int(SR * start_sec)
        length = int(SR * 0.18)
        harsh_envelope[start : start + length] = np.exp(
            -np.linspace(0.0, 4.8, length)
        )
    samples = (
        _tone(700.0, amplitude=0.025, duration_sec=duration_sec)
        + _tone(1250.0, amplitude=0.018, duration_sec=duration_sec)
        + 0.14 * harsh_envelope * np.sin(2.0 * np.pi * 3100.0 * time)
    )

    recommendation = analyze_smart_dsp_audio(
        samples,
        target_id="ch:17",
        source_role="electric_guitar",
        source_label="GTR L",
        sample_rate=SR,
        source_identity_confirmed=True,
        source_identity_confidence=0.9,
        current_state_cache_freshness=_fresh(),
        current_state_readback_available=True,
        rollback_plan_available=True,
        target_plane_lufs=-16.0,
    )

    audio_evidence = recommendation.evidence["audio_evidence"]
    assert audio_evidence["declared_source_family"] == "guitar"
    assert audio_evidence["presence_harshness_score"] >= 0.55
    assert audio_evidence["instrument_harsh_frequency_hz"] is not None
    assert 2900.0 <= audio_evidence["instrument_harsh_frequency_hz"] <= 3300.0
    assert audio_evidence["instrument_harsh_bandwidth_hz"] > 0.0
    assert audio_evidence["instrument_harsh_event_peak_dbfs"] < -1.0
    assert audio_evidence["instrument_harsh_event_contrast_db"] > 0.0
    assert audio_evidence["instrument_harsh_event_recovery_ms"] >= 0.0
    assert "harsh" in audio_evidence["problem_hints"]
    assert recommendation.selected_strategy.strategy_id == "instrument_dynamic_presence_tame"
    assert recommendation.selected_strategy.method_family == "dynamic_eq"
    params = recommendation.selected_strategy.parameters
    assert params["source_family"] == "guitar"
    assert params["instrument_harsh_parameter_source"] == (
        "measured_instrument_harshness_event"
    )
    assert params["frequency_hz"] == audio_evidence["instrument_harsh_frequency_hz"]
    assert params["instrument_harsh_bandwidth_hz"] == audio_evidence[
        "instrument_harsh_bandwidth_hz"
    ]
    assert params["instrument_harsh_event_peak_dbfs"] == audio_evidence[
        "instrument_harsh_event_peak_dbfs"
    ]
    assert params["instrument_harsh_event_contrast_db"] == audio_evidence[
        "instrument_harsh_event_contrast_db"
    ]
    assert params["instrument_harsh_event_recovery_ms"] == audio_evidence[
        "instrument_harsh_event_recovery_ms"
    ]
    assert params["max_gain_reduction_db"] > 0.0
    assert params["wing_dynamic_eq_model_osc_value"] == WING_DYNAMIC_EQ_GATE_MODEL
    assert params["wing_dynamic_eq_filter_osc_value"] == "BP"
    assert params["wing_dynamic_eq_gain_osc_value"] < 0.0
    assert params["wing_dynamic_eq_frequency_osc_value"] == params["frequency_hz"]
    assert params["makeup_gain_db"] == 0.0
    assert recommendation.selected_strategy.future_wing_readback_contract[
        "wing_osc_applicability"
    ]["native_osc_supported"] is True
    assert recommendation.selected_strategy.future_wing_readback_contract[
        "dynamic_eq_readback"
    ]["wing_dynamic_eq_model_readback_must_equal"] == WING_DYNAMIC_EQ_GATE_MODEL
    assert recommendation.evidence["operator_explanation_sections"][
        "selection_policy"
    ] == "weighted_strategy_score"
    static = [
        candidate
        for candidate in recommendation.alternatives
        if candidate.strategy_id == "instrument_broad_presence_cut"
    ][0]
    assert static.parameters["gain_db"] < 0.0
    assert static.future_wing_readback_contract["wing_osc_applicability"][
        "native_osc_supported"
    ] is True


def test_harsh_overhead_audio_selects_wing_native_static_de_edge():
    samples = (
        _band_limited_noise(3500.0, 8500.0, amplitude=0.14)
        + _tone(900.0, amplitude=0.02)
        + _tone(250.0, amplitude=0.015)
    )

    recommendation = analyze_smart_dsp_audio(
        samples,
        target_id="ch:10",
        source_role="overhead_cymbal",
        source_label="OH L",
        sample_rate=SR,
        source_identity_confirmed=True,
        source_identity_confidence=0.9,
        current_state_cache_freshness=_fresh(),
        current_state_readback_available=True,
        rollback_plan_available=True,
        target_plane_lufs=-16.0,
    )

    audio_evidence = recommendation.evidence["audio_evidence"]
    assert audio_evidence["declared_source_family"] == "drum"
    assert audio_evidence["source_identity_mismatch_score"] == 0.0
    assert audio_evidence["harshness_score"] >= 0.65
    assert audio_evidence["drum_harsh_frequency_hz"] is not None
    assert audio_evidence["drum_harsh_bandwidth_hz"] > 0.0
    assert audio_evidence["drum_harsh_event_peak_dbfs"] < -1.0
    assert audio_evidence["drum_harsh_event_contrast_db"] > 0.0
    assert audio_evidence["drum_harsh_event_recovery_ms"] >= 0.0
    assert "harsh" in audio_evidence["problem_hints"]
    assert recommendation.selected_strategy.strategy_id == "drum_cymbal_static_air_cut"
    assert recommendation.selected_strategy.method_family == "eq"
    params = recommendation.selected_strategy.parameters
    assert params["filter_type"] == "high_shelf"
    assert params["frequency_hz"] >= 6000.0
    assert params["drum_harsh_parameter_source"] == "measured_drum_harshness_event"
    assert params["drum_harsh_bandwidth_hz"] == audio_evidence["drum_harsh_bandwidth_hz"]
    assert params["drum_harsh_event_peak_dbfs"] == audio_evidence[
        "drum_harsh_event_peak_dbfs"
    ]
    assert params["drum_harsh_event_contrast_db"] == audio_evidence[
        "drum_harsh_event_contrast_db"
    ]
    assert params["drum_harsh_event_recovery_ms"] == audio_evidence[
        "drum_harsh_event_recovery_ms"
    ]
    assert params["gain_db"] < 0.0
    assert recommendation.selected_strategy.future_wing_readback_contract[
        "wing_osc_applicability"
    ]["native_osc_supported"] is True
    assert recommendation.evidence["operator_explanation_sections"][
        "selection_policy"
    ] == "weighted_strategy_score"
    dynamic = [
        candidate
        for candidate in recommendation.alternatives
        if candidate.strategy_id == "drum_cymbal_dynamic_de_edge"
    ][0]
    assert dynamic.parameters["max_gain_reduction_db"] > 0.0
    assert dynamic.parameters["makeup_gain_db"] == 0.0
    assert dynamic.future_wing_readback_contract["wing_osc_applicability"][
        "native_osc_supported"
    ] is False
    assert recommendation.to_operator_proposal()["live_apply_allowed"] is False


def test_phase_unsafe_stereo_audio_without_width_request_selects_width_reduction():
    left = _tone(440.0, amplitude=0.28)
    right = -left
    stereo = np.column_stack([left, right])

    recommendation = analyze_smart_dsp_audio(
        stereo,
        target_id="ch:12",
        source_role="electric_guitar",
        source_label="GTR L/R",
        sample_rate=SR,
        source_identity_confirmed=True,
        source_identity_confidence=0.9,
        current_state_cache_freshness=_fresh(),
        current_state_readback_available=True,
        rollback_plan_available=True,
        target_plane_lufs=-16.0,
    )

    audio_evidence = recommendation.evidence["audio_evidence"]
    assert audio_evidence["phase_correlation"] < -0.95
    assert audio_evidence["mono_sum_delta_db"] <= -40.0
    assert "phase_mono_risk" in audio_evidence["problem_hints"]
    assert "phase_mono_risk" in recommendation.evidence["detected_problems"]
    assert recommendation.selected_strategy.strategy_id == "mono_compatibility_width_reduction"
    params = recommendation.selected_strategy.parameters
    assert params["target_width_percent"] == 0.0
    assert params["severe_cancellation"] is True
    assert recommendation.to_operator_proposal()["live_apply_allowed"] is False


def test_phase_safe_stereo_audio_calculates_pan_delta():
    left = (
        _tone(280.0, amplitude=0.03)
        + _tone(900.0, amplitude=0.08)
        + _tone(1800.0, amplitude=0.10)
        + _tone(3400.0, amplitude=0.06)
    )
    right = (
        _tone(280.0, amplitude=0.028, phase=0.04)
        + _tone(900.0, amplitude=0.075, phase=0.04)
        + _tone(1800.0, amplitude=0.094, phase=0.04)
        + _tone(3400.0, amplitude=0.056, phase=0.04)
    )
    stereo = np.column_stack([left, right])

    recommendation = analyze_smart_dsp_audio(
        stereo,
        target_id="ch:12",
        source_role="electric_guitar",
        source_label="GTR L/R",
        sample_rate=SR,
        desired_stereo_width_request=35.0,
        current_pan_percent=5.0,
        source_identity_confirmed=True,
        source_identity_confidence=0.9,
        current_state_cache_freshness=_fresh(),
        current_state_readback_available=True,
        rollback_plan_available=True,
        target_plane_lufs=-21.0,
    )

    audio_evidence = recommendation.evidence["audio_evidence"]
    assert audio_evidence["phase_correlation"] >= 0.65
    assert audio_evidence["mono_sum_delta_db"] > -3.0
    assert recommendation.selected_strategy.strategy_id == "phase_safe_pan_target"
    params = recommendation.selected_strategy.parameters
    assert params["target_pan_percent"] == 35.0
    assert params["current_pan_percent"] == 5.0
    assert params["pan_delta_percent"] == 30.0
    assert recommendation.to_operator_proposal()["can_apply_live"] is False


def test_measured_stereo_balance_audio_calculates_pan_target_without_request():
    left = _tone(440.0, amplitude=0.15) + _tone(880.0, amplitude=0.04)
    right = _tone(440.0, amplitude=0.08, phase=0.02) + _tone(
        880.0,
        amplitude=0.022,
        phase=0.02,
    )
    stereo = np.column_stack([left, right])

    recommendation = analyze_smart_dsp_audio(
        stereo,
        target_id="ch:12",
        source_role="electric_guitar",
        source_label="GTR L/R",
        sample_rate=SR,
        current_pan_percent=0.0,
        source_identity_confirmed=True,
        source_identity_confidence=0.9,
        current_state_cache_freshness=_fresh(),
        current_state_readback_available=True,
        rollback_plan_available=True,
        target_plane_lufs=-22.0,
    )

    audio_evidence = recommendation.evidence["audio_evidence"]
    assert audio_evidence["phase_correlation"] >= 0.65
    assert audio_evidence["mono_sum_delta_db"] > -3.0
    assert audio_evidence["stereo_balance_db"] < -4.0
    assert audio_evidence["stereo_balance_confidence"] >= 0.35
    assert audio_evidence["measured_pan_target_percent"] > 20.0
    assert "pan_width" in audio_evidence["problem_hints"]
    assert recommendation.selected_strategy.strategy_id == "phase_safe_pan_target"
    params = recommendation.selected_strategy.parameters
    assert params["pan_target_source"] == "measured_stereo_balance"
    assert params["target_pan_percent"] == audio_evidence["measured_pan_target_percent"]
    assert params["pan_delta_percent"] == audio_evidence["measured_pan_delta_percent"]
    assert recommendation.to_operator_proposal()["live_apply_allowed"] is False


def test_phase_unsafe_measured_stereo_balance_audio_reduces_width_before_pan():
    left = _tone(440.0, amplitude=0.15) + _tone(880.0, amplitude=0.04)
    right = -(
        _tone(440.0, amplitude=0.08)
        + _tone(880.0, amplitude=0.022)
    )
    stereo = np.column_stack([left, right])

    recommendation = analyze_smart_dsp_audio(
        stereo,
        target_id="ch:12",
        source_role="electric_guitar",
        source_label="GTR L/R",
        sample_rate=SR,
        current_pan_percent=0.0,
        source_identity_confirmed=True,
        source_identity_confidence=0.9,
        current_state_cache_freshness=_fresh(),
        current_state_readback_available=True,
        rollback_plan_available=True,
        target_plane_lufs=-22.0,
    )

    audio_evidence = recommendation.evidence["audio_evidence"]
    assert audio_evidence["phase_correlation"] < -0.95
    assert audio_evidence["mono_sum_delta_db"] <= -6.0
    assert audio_evidence["stereo_balance_confidence"] >= 0.35
    assert audio_evidence["measured_pan_target_percent"] > 20.0
    assert "pan_width" in audio_evidence["problem_hints"]
    assert "phase_mono_risk" in audio_evidence["problem_hints"]
    assert recommendation.selected_strategy.strategy_id == "mono_compatibility_width_reduction"
    params = recommendation.selected_strategy.parameters
    assert params["target_width_percent"] == 0.0
    assert params["severe_cancellation"] is True
    assert recommendation.evidence["operator_explanation_sections"][
        "selection_policy"
    ] == "soundcheck_phase_mono_safety_priority"

    pan_target = [
        candidate
        for candidate in recommendation.alternatives
        if candidate.strategy_id == "phase_safe_pan_target"
    ][0]
    assert pan_target.parameters["pan_target_source"] == "measured_stereo_balance"
    assert "phase_correlation_blocks_width" in pan_target.blockers
    assert recommendation.to_operator_proposal()["live_apply_allowed"] is False


def test_master_overload_audio_blocks_live_apply_and_selects_trim():
    samples = _tone(1000.0, amplitude=0.98)

    recommendation = analyze_smart_dsp_audio(
        samples,
        target_id="main:lr",
        source_role="master",
        source_label="Main LR",
        mode="live",
        sample_rate=SR,
        source_identity_confirmed=True,
        source_identity_confidence=0.95,
        current_state_cache_freshness=_fresh(),
        current_state_readback_available=True,
        rollback_plan_available=True,
        active_performance=True,
    )

    assert recommendation.status == "analysis_only"
    assert "true_peak_headroom_low" in recommendation.blockers
    assert recommendation.selected_strategy.strategy_id == "master_trim_for_true_peak_headroom"
    assert recommendation.selected_strategy.method_family == "gain_fader"
    assert (
        recommendation.selected_strategy.parameters[
            "live_master_true_peak_pullback_proposal"
        ]
        is True
    )
    assert (
        recommendation.selected_strategy.parameters[
            "live_master_true_peak_native_gain_osc_only_required"
        ]
        is True
    )
    assert (
        recommendation.selected_strategy.parameters[
            "live_master_true_peak_no_limiter_compression_or_eq_write_required"
        ]
        is True
    )
    assert (
        recommendation.selected_strategy.future_wing_readback_contract[
            "gain_fader_readback"
        ]["live_master_true_peak_live_apply_blocked_without_wing_proof"]
        is True
    )
    assert recommendation.to_operator_proposal()["can_apply_live"] is False


def test_master_reference_audio_holds_instead_of_master_bus_processing():
    samples = _tone(1000.0, amplitude=0.98)

    recommendation = analyze_smart_dsp_audio(
        samples,
        target_id="dante:63",
        source_role="master_reference_feed",
        source_label="Dante Reference Capture",
        sample_rate=SR,
        source_identity_confirmed=True,
        source_identity_confidence=0.95,
        current_state_cache_freshness=_fresh(),
        current_state_readback_available=True,
        rollback_plan_available=True,
    )

    assert recommendation.status == "analysis_only"
    assert recommendation.selected_strategy.strategy_id == (
        "hold_master_reference_feed_no_pa_write"
    )
    assert "master_reference_or_capture_feed_not_pa_output" in recommendation.blockers
    assert recommendation.evidence["master_feed_identity"][
        "reference_or_capture_feed"
    ] is True
    assert recommendation.to_operator_proposal()["live_apply_allowed"] is False


def test_master_low_end_audio_selects_broad_eq_cleanup_not_channel_eq():
    duration_sec = 2.5
    time = np.arange(int(SR * duration_sec), dtype=np.float64) / SR
    samples = 0.12 * np.sin(2.0 * np.pi * 70.0 * time)
    samples += 0.03 * np.sin(2.0 * np.pi * 140.0 * time)
    samples += 0.006 * np.sin(2.0 * np.pi * 1200.0 * time)

    recommendation = analyze_smart_dsp_audio(
        samples,
        target_id="main:lr",
        source_role="master",
        source_label="Main LR",
        sample_rate=SR,
        source_identity_confirmed=True,
        source_identity_confidence=0.95,
        current_state_cache_freshness=_fresh(),
        current_state_readback_available=True,
        rollback_plan_available=True,
    )

    audio_evidence = recommendation.evidence["audio_evidence"]
    assert audio_evidence["low_end_excess_score"] >= 0.9
    assert audio_evidence["master_low_end_frequency_hz"] is not None
    assert audio_evidence["master_low_end_bandwidth_hz"] >= 0.0
    assert audio_evidence["true_peak_headroom_db"] > 1.0
    assert recommendation.evidence["detected_problems"] == ["master_overload"]
    assert recommendation.selected_strategy.strategy_id == "master_low_end_broad_eq_cleanup"
    assert recommendation.selected_strategy.method_family == "eq"
    params = recommendation.selected_strategy.parameters
    assert params["filter_type"] == "low_shelf"
    assert params["frequency_source"] == "measured_master_low_end_peak"
    assert params["master_low_end_frequency_hz"] == audio_evidence[
        "master_low_end_frequency_hz"
    ]
    assert params["frequency_hz"] <= 160.0
    assert params["wing_eq_mapping_required"] is True
    assert params["soundcheck_master_processing_review_only_proposal"] is True
    assert params["soundcheck_master_processing_native_osc_only_required"] is True
    assert (
        params["soundcheck_master_processing_live_apply_blocked_without_wing_proof"]
        is True
    )
    assert recommendation.to_operator_proposal()["live_apply_allowed"] is False


def test_master_presence_harsh_audio_selects_broad_eq_cleanup_not_bus_comp():
    duration_sec = 2.5
    time = np.arange(int(SR * duration_sec), dtype=np.float64) / SR
    samples = 0.02 * np.sin(2.0 * np.pi * 900.0 * time)
    samples += 0.11 * np.sin(2.0 * np.pi * 3400.0 * time)
    samples += 0.018 * np.sin(2.0 * np.pi * 7600.0 * time)

    recommendation = analyze_smart_dsp_audio(
        samples,
        target_id="main:lr",
        source_role="master",
        source_label="Main LR",
        sample_rate=SR,
        source_identity_confirmed=True,
        source_identity_confidence=0.95,
        current_state_cache_freshness=_fresh(),
        current_state_readback_available=True,
        rollback_plan_available=True,
    )

    audio_evidence = recommendation.evidence["audio_evidence"]
    assert audio_evidence["presence_harshness_score"] >= 0.55
    assert audio_evidence["master_presence_frequency_hz"] is not None
    assert audio_evidence["master_presence_bandwidth_hz"] >= 0.0
    assert audio_evidence["low_end_excess_score"] < 0.55
    assert audio_evidence["true_peak_headroom_db"] > 1.0
    assert recommendation.selected_strategy.strategy_id == "master_presence_broad_eq_cleanup"
    assert recommendation.selected_strategy.method_family == "eq"
    params = recommendation.selected_strategy.parameters
    assert params["filter_type"] == "peaking"
    assert params["frequency_source"] == "measured_master_presence_peak"
    assert params["frequency_hz"] == audio_evidence["master_presence_frequency_hz"]
    assert params["master_presence_frequency_hz"] == audio_evidence[
        "master_presence_frequency_hz"
    ]
    assert params["gain_db"] < 0.0
    assert params["wing_eq_mapping_required"] is True
    assert params["soundcheck_master_processing_review_only_proposal"] is True
    assert params["soundcheck_master_processing_native_osc_only_required"] is True
    assert (
        params["soundcheck_master_processing_live_apply_blocked_without_wing_proof"]
        is True
    )
    assert recommendation.to_operator_proposal()["live_apply_allowed"] is False


def test_master_dynamic_range_audio_selects_gentle_bus_compression():
    duration_sec = 3.0
    time = np.arange(int(SR * duration_sec), dtype=np.float64) / SR
    envelope = np.where((time % 0.8) < 0.22, 0.28, 0.03)
    samples = envelope * (
        0.45 * np.sin(2.0 * np.pi * 240.0 * time)
        + 0.35 * np.sin(2.0 * np.pi * 720.0 * time)
        + 0.2 * np.sin(2.0 * np.pi * 1800.0 * time)
    )

    recommendation = analyze_smart_dsp_audio(
        samples,
        target_id="main:lr",
        source_role="master",
        source_label="Main LR",
        sample_rate=SR,
        source_identity_confirmed=True,
        source_identity_confidence=0.95,
        current_state_cache_freshness=_fresh(),
        current_state_readback_available=True,
        rollback_plan_available=True,
    )

    audio_evidence = recommendation.evidence["audio_evidence"]
    assert audio_evidence["lra_proxy_db"] >= 10.0
    assert audio_evidence["dynamics_variance_score"] >= 0.55
    assert audio_evidence["dynamics_peak_to_median_db"] >= 8.0
    assert audio_evidence["dynamics_rise_time_ms"] > 0.0
    assert audio_evidence["dynamics_recovery_ms"] > 0.0
    assert audio_evidence["true_peak_headroom_db"] > 1.0
    assert audio_evidence["low_end_excess_score"] < 0.55
    assert audio_evidence["presence_harshness_score"] < 0.55
    assert recommendation.selected_strategy.strategy_id == "gentle_bus_compression"
    params = recommendation.selected_strategy.parameters
    assert params["makeup_gain_db"] == 0.0
    assert params["true_peak_headroom_required_before_compression"] is True
    assert params["dynamic_pressure_score"] >= 0.55
    assert params["master_dynamics_parameter_source"] == "measured_master_envelope"
    assert params["dynamics_peak_to_median_db"] == audio_evidence[
        "dynamics_peak_to_median_db"
    ]
    assert params["dynamics_rise_time_ms"] == audio_evidence["dynamics_rise_time_ms"]
    assert params["dynamics_recovery_ms"] == audio_evidence["dynamics_recovery_ms"]
    assert params["attack_ms"] != 32.0
    assert params["release_ms"] >= 180.0
    assert params["target_gain_reduction_db"] <= 2.6
    assert recommendation.to_operator_proposal()["live_apply_allowed"] is False


def test_guitar_low_mid_bloom_audio_selects_wing_native_deq_dynamic_cut():
    duration_sec = 3.0
    time = np.arange(int(SR * duration_sec), dtype=np.float64) / SR
    samples = 0.035 * np.sin(2.0 * np.pi * 900.0 * time)
    samples += 0.025 * np.sin(2.0 * np.pi * 1800.0 * time)
    samples += 0.015 * np.sin(2.0 * np.pi * 2400.0 * time)
    for start_sec in np.arange(0.25, duration_sec - 0.3, 0.55):
        start = int(SR * start_sec)
        length = int(SR * 0.24)
        local_time = np.arange(length, dtype=np.float64) / SR
        envelope = np.exp(-np.linspace(0.0, 2.2, length))
        samples[start : start + length] += (
            0.18 * envelope * np.sin(2.0 * np.pi * 310.0 * local_time)
        )
        samples[start : start + length] += (
            0.08 * envelope * np.sin(2.0 * np.pi * 230.0 * local_time)
        )
    samples = np.clip(samples, -0.95, 0.95)

    recommendation = analyze_smart_dsp_audio(
        samples,
        target_id="ch:14",
        source_role="electric_guitar",
        source_label="GTR L",
        sample_rate=SR,
        source_identity_confirmed=True,
        source_identity_confidence=0.92,
        current_state_cache_freshness=_fresh(),
        current_state_readback_available=True,
        rollback_plan_available=True,
    )

    audio_evidence = recommendation.evidence["audio_evidence"]
    assert audio_evidence["muddiness_score"] >= 0.6
    assert audio_evidence["mud_frequency_hz"] is not None
    assert audio_evidence["low_mid_bloom_frequency_hz"] is not None
    assert audio_evidence["low_mid_bloom_db"] >= 5.5
    assert audio_evidence["low_mid_bloom_peak_dbfs"] < -1.0
    assert audio_evidence["low_mid_bloom_recovery_ms"] > 0.0
    assert audio_evidence["low_end_excess_score"] < 0.78
    assert recommendation.selected_strategy.strategy_id == "instrument_dynamic_low_mid_de_mud"
    assert recommendation.selected_strategy.method_family == "dynamic_eq"
    params = recommendation.selected_strategy.parameters
    assert params["frequency_source"] == "measured_low_mid_bloom_peak"
    assert params["frequency_hz"] == audio_evidence["low_mid_bloom_frequency_hz"]
    assert params["mud_frequency_hz"] == audio_evidence["mud_frequency_hz"]
    assert params["low_mid_bloom_frequency_hz"] == audio_evidence[
        "low_mid_bloom_frequency_hz"
    ]
    assert params["max_gain_reduction_db"] > 0.0
    assert params["wing_dynamic_eq_model_osc_value"] == WING_DYNAMIC_EQ_GATE_MODEL
    assert params["wing_dynamic_eq_filter_osc_value"] == "BP"
    assert params["wing_dynamic_eq_gain_osc_value"] < 0.0
    assert params["wing_dynamic_eq_frequency_osc_value"] == params["frequency_hz"]
    assert recommendation.selected_strategy.future_wing_readback_contract[
        "dynamic_eq_readback"
    ]["wing_dynamic_eq_model_readback_must_equal"] == WING_DYNAMIC_EQ_GATE_MODEL
    alternatives = {
        candidate.strategy_id: candidate for candidate in recommendation.alternatives
    }
    static_params = alternatives["low_mid_corrective_cut"].parameters
    assert static_params["gain_db"] < 0.0
    assert static_params["frequency_hz"] == audio_evidence["low_mid_bloom_frequency_hz"]
    dynamic_params = params
    assert dynamic_params["instrument_low_mid_parameter_source"] == "measured_low_mid_bloom"
    assert dynamic_params["low_mid_bloom_db"] == audio_evidence["low_mid_bloom_db"]
    assert dynamic_params["low_mid_bloom_peak_dbfs"] == audio_evidence[
        "low_mid_bloom_peak_dbfs"
    ]
    assert dynamic_params["low_mid_bloom_recovery_ms"] == audio_evidence[
        "low_mid_bloom_recovery_ms"
    ]
    assert dynamic_params["makeup_gain_db"] == 0.0
    tradeoffs = {
        row["strategy_id"]: row
        for row in recommendation.evidence["strategy_search"]["alternative_tradeoffs"]
    }
    assert tradeoffs["low_mid_corrective_cut"]["why_not_selected"].startswith("lower_")
    assert recommendation.to_operator_proposal()["live_apply_allowed"] is False


def test_masking_reference_audio_sets_non_vocal_conflict_and_selects_subtractive_eq():
    target = _tone(1000.0, amplitude=0.28) + _tone(2400.0, amplitude=0.035)
    reference = _tone(1000.0, amplitude=0.3) + _tone(180.0, amplitude=0.05)

    analysis = build_smart_dsp_analysis_input_from_audio(
        target,
        target_id="ch:15",
        source_role="keys",
        source_label="Keys",
        sample_rate=SR,
        reference_samples=reference,
        source_identity_confirmed=True,
        source_identity_confidence=0.88,
        current_state_cache_freshness=_fresh(),
        current_state_readback_available=True,
        rollback_plan_available=True,
        target_plane_lufs=-16.0,
        reference_role="electric_guitar",
    )
    recommendation = analyze_smart_dsp_strategy(analysis)

    assert analysis.masking_conflict_role == "electric_guitar"
    assert 950.0 <= analysis.masking_frequency_hz <= 1050.0
    assert "masking" in recommendation.evidence["detected_problems"]
    assert recommendation.selected_strategy.strategy_id == "subtractive_unmasking_eq_split"
    params = recommendation.selected_strategy.parameters
    assert params["linked_counter_boost_db"] == 0.0
    assert params["gain_db"] < 0.0
    assert params["q"] == 3.0
    assert params["q"] != 1.1
    assert params["static_eq_q_source"] == "measured_masking_overlap_band"
    assert params["masking_parameter_source"] == "measured_masking_overlap_band"
    assert params["masking_bandwidth_hz"] == analysis.masking_bandwidth_hz
    assert params["masking_overlap_peak_share"] == analysis.masking_overlap_peak_share
    assert recommendation.selected_strategy.future_wing_readback_contract[
        "wing_osc_applicability"
    ]["native_osc_supported"] is True
    dynamic = [
        candidate
        for candidate in recommendation.alternatives
        if candidate.strategy_id == "dynamic_ducking_proposal"
    ][0]
    assert dynamic.parameters["filter_type"] == "dynamic_peaking"
    assert dynamic.parameters["masking_parameter_source"] == "measured_masking_overlap_band"
    assert dynamic.parameters["masking_bandwidth_hz"] == analysis.masking_bandwidth_hz
    assert dynamic.parameters["masking_overlap_peak_share"] == analysis.masking_overlap_peak_share
    assert dynamic.parameters["bandwidth_q"] == 1.54
    assert dynamic.parameters["dynamic_eq_q"] == 1.54
    assert dynamic.parameters["max_gain_reduction_db"] != 2.5
    assert dynamic.parameters["attack_ms"] != 12.0
    assert dynamic.parameters["release_ms"] != 130.0
    assert dynamic.future_wing_readback_contract["wing_osc_applicability"][
        "native_osc_supported"
    ] is False


def test_vocal_masking_reference_audio_selects_wing_native_static_unmasking():
    target = _tone(2400.0, amplitude=0.26) + _tone(900.0, amplitude=0.04)
    reference = _tone(2400.0, amplitude=0.3) + _tone(180.0, amplitude=0.03)

    analysis = build_smart_dsp_analysis_input_from_audio(
        target,
        target_id="ch:15",
        source_role="keys",
        source_label="Keys",
        sample_rate=SR,
        reference_samples=reference,
        reference_role="lead_vocal",
        source_identity_confirmed=True,
        source_identity_confidence=0.88,
        current_state_cache_freshness=_fresh(),
        current_state_readback_available=True,
        rollback_plan_available=True,
        target_plane_lufs=-16.0,
    )
    recommendation = analyze_smart_dsp_strategy(analysis)

    assert analysis.masking_conflict_role == "lead_vocal"
    assert 2350.0 <= analysis.masking_frequency_hz <= 2450.0
    assert analysis.masking_score >= 0.45
    assert analysis.masking_bandwidth_hz > 0.0
    assert analysis.masking_overlap_peak_share > 0.0
    assert recommendation.selected_strategy.strategy_id == (
        "vocal_intelligibility_static_pocket_eq"
    )
    assert recommendation.selected_strategy.method_family == "eq"
    params = recommendation.selected_strategy.parameters
    assert params["current_target_is_masker"] is True
    assert params["linked_counter_boost_db"] == 0.0
    assert params["gain_db"] < 0.0
    assert params["q"] > 0.0
    assert params["q"] != 1.15
    assert params["static_eq_q_source"] == "measured_masking_overlap_band"
    assert params["masking_parameter_source"] == "measured_masking_overlap_band"
    assert params["masking_bandwidth_hz"] == analysis.masking_bandwidth_hz
    assert params["masking_overlap_peak_share"] == analysis.masking_overlap_peak_share
    assert recommendation.selected_strategy.future_wing_readback_contract[
        "wing_osc_applicability"
    ]["native_osc_supported"] is True
    assert recommendation.evidence["operator_explanation_sections"][
        "selection_policy"
    ] == "weighted_strategy_score"
    dynamic = [
        candidate
        for candidate in recommendation.alternatives
        if candidate.strategy_id == "vocal_intelligibility_dynamic_unmasking"
    ][0]
    assert dynamic.parameters["sidechain_role"] == "lead_vocal"
    assert dynamic.parameters["masking_parameter_source"] == "measured_masking_overlap_band"
    assert dynamic.parameters["masking_bandwidth_hz"] == analysis.masking_bandwidth_hz
    assert dynamic.parameters["masking_overlap_peak_share"] == analysis.masking_overlap_peak_share
    assert dynamic.parameters["bandwidth_q"] != 1.35
    assert dynamic.parameters["attack_ms"] != 8.0
    assert dynamic.future_wing_readback_contract["wing_osc_applicability"][
        "native_osc_supported"
    ] is False
    assert recommendation.to_operator_proposal()["live_apply_allowed"] is False


def test_quiet_vocal_audio_selects_soundcheck_bounded_gain_up():
    samples = _tone(900.0, amplitude=0.035)

    recommendation = analyze_smart_dsp_audio(
        samples,
        target_id="ch:21",
        source_role="lead_vocal",
        source_label="Quiet Lead",
        sample_rate=SR,
        source_identity_confirmed=True,
        source_identity_confidence=0.92,
        current_state_cache_freshness=_fresh(),
        current_state_readback_available=True,
        rollback_plan_available=True,
        target_plane_lufs=-22.0,
    )

    audio_evidence = recommendation.evidence["audio_evidence"]
    assert audio_evidence["level_plane_error_lu"] < -2.0
    assert "level_plane_balance" in audio_evidence["problem_hints"]
    assert recommendation.selected_strategy.strategy_id == "soundcheck_bounded_gain_up"
    assert recommendation.selected_strategy.parameters["delta_db"] > 0.0
    assert recommendation.selected_strategy.parameters["future_fader_ceiling_dbfs"] == 0.0


def test_quiet_vocal_audio_blocks_gain_up_when_readonly_fader_near_ceiling():
    samples = _tone(900.0, amplitude=0.035)
    wing_state = build_current_wing_state_from_readonly_preflight(
        target_id="ch:21",
        summary={
            "run_id": "20260702T075151Z_wing_readonly_preflight",
            "connection_mode": "wing",
            "read_only": True,
            "live_write": False,
            "write_request_sent": False,
            "proposal_request_sent": False,
            "live_authority_granted": False,
            "readiness_closure_allowed": False,
            "readbacks": [
                {
                    "type": "mixer_parameter_value",
                    "status": "verified_from_physical_readback",
                    "source": "physical_readback",
                    "channel": 21,
                    "value_type": "fader",
                    "address": "/ch/21/fdr",
                    "value": -1.0,
                },
            ],
        },
    )

    recommendation = analyze_smart_dsp_audio(
        samples,
        target_id="ch:21",
        source_role="lead_vocal",
        source_label="Quiet Lead",
        sample_rate=SR,
        source_identity_confirmed=True,
        source_identity_confidence=0.92,
        current_state_cache_freshness=_fresh(),
        current_state_readback_available=True,
        rollback_plan_available=True,
        current_wing_state=wing_state,
        target_plane_lufs=-22.0,
    )

    candidates = {
        candidate.strategy_id: candidate
        for candidate in (recommendation.selected_strategy, *recommendation.alternatives)
    }
    gain_up = candidates["soundcheck_bounded_gain_up"]
    assert WING_GAIN_UP_FADER_CEILING_BLOCKER in gain_up.blockers
    assert gain_up.parameters["wing_current_fader_db"] == -1.0
    assert gain_up.parameters["wing_gain_fader_headroom_db"] == 1.0
    assert gain_up.parameters["wing_gain_fader_ceiling_limited_delta_db"] == 1.0
    assert gain_up.parameters["wing_gain_fader_ceiling_exceeded"] is True
    assert recommendation.evidence["current_wing_controls"]["fader_db"] == -1.0
    assert recommendation.to_operator_proposal()["live_apply_allowed"] is False


def test_quiet_vocal_audio_requires_fader_readback_before_wing_gain_up_ceiling_check():
    samples = _tone(900.0, amplitude=0.035)
    wing_state = build_current_wing_state_from_readonly_preflight(
        target_id="ch:21",
        summary={
            "run_id": "20260702T075151Z_wing_readonly_preflight",
            "connection_mode": "wing",
            "read_only": True,
            "live_write": False,
            "write_request_sent": False,
            "proposal_request_sent": False,
            "live_authority_granted": False,
            "readiness_closure_allowed": False,
            "readbacks": [
                {
                    "type": "mixer_parameter_value",
                    "status": "verified_from_physical_readback",
                    "source": "physical_readback",
                    "channel": 21,
                    "value_type": "gain",
                    "address": "/ch/21/in/set/trim",
                    "value": 1.0,
                },
            ],
        },
    )

    recommendation = analyze_smart_dsp_audio(
        samples,
        target_id="ch:21",
        source_role="lead_vocal",
        source_label="Quiet Lead",
        sample_rate=SR,
        source_identity_confirmed=True,
        source_identity_confidence=0.92,
        current_state_cache_freshness=_fresh(),
        current_state_readback_available=True,
        rollback_plan_available=True,
        current_wing_state=wing_state,
        target_plane_lufs=-22.0,
    )

    candidates = {
        candidate.strategy_id: candidate
        for candidate in (recommendation.selected_strategy, *recommendation.alternatives)
    }
    gain_up = candidates["soundcheck_bounded_gain_up"]
    assert WING_GAIN_UP_FADER_READBACK_BLOCKER in gain_up.blockers
    assert gain_up.parameters["wing_current_fader_readback_available"] is False
    assert gain_up.parameters["wing_current_fader_readback_required"] is True
    assert gain_up.parameters["wing_gain_fader_ceiling_blocker"] == (
        WING_GAIN_UP_FADER_READBACK_BLOCKER
    )
    controls = recommendation.evidence["current_wing_controls"]
    assert controls["readback_available"] is True
    assert controls["fader_db"] is None
    assert controls["input_trim_db"] == 1.0
    assert recommendation.to_operator_proposal()["live_apply_allowed"] is False


def test_quiet_vocal_audio_with_hot_bus_rebalances_bus_before_gain_up():
    samples = _tone(900.0, amplitude=0.035)
    summed_bus = _tone(180.0, amplitude=0.92) + _tone(900.0, amplitude=0.18)

    recommendation = analyze_smart_dsp_audio(
        samples,
        target_id="ch:21",
        source_role="lead_vocal",
        source_label="Quiet Lead",
        sample_rate=SR,
        source_identity_confirmed=True,
        source_identity_confidence=0.92,
        current_state_cache_freshness=_fresh(),
        current_state_readback_available=True,
        rollback_plan_available=True,
        target_plane_lufs=-22.0,
        summed_bus_samples=summed_bus,
        summed_bus_role="vocal_bus",
        summed_bus_target_id="bus:1",
        summed_bus_target_plane_lufs=-25.0,
    )

    audio_evidence = recommendation.evidence["audio_evidence"]
    assert audio_evidence["level_plane_error_lu"] < -2.0
    assert audio_evidence["summed_bus_evidence_available"] is True
    assert audio_evidence["summed_bus_true_peak_headroom_db"] < 1.0
    assert audio_evidence["summed_bus_level_plane_error_lu"] > 2.0
    assert audio_evidence["summed_bus_target_id"] == "bus:1"
    assert recommendation.selected_strategy.strategy_id == (
        "summed_bus_rebalance_before_source_gain"
    )
    assert recommendation.selected_strategy.parameters["bus_delta_db"] < 0.0
    assert recommendation.selected_strategy.parameters["gain_db"] == 0.0
    assert recommendation.selected_strategy.parameters["summed_bus_target_id"] == (
        "bus:1"
    )
    assert recommendation.selected_strategy.parameters["wing_osc_effective_target_id"] == (
        "bus:1"
    )
    assert recommendation.evidence["summed_bus"]["evidence_available"] is True
    assert recommendation.evidence["summed_bus"]["target_id"] == "bus:1"
    assert recommendation.to_operator_proposal()["live_apply_allowed"] is False


def test_hot_drum_bus_audio_selects_group_trim_from_bus_evidence():
    duration_sec = 2.4
    time = np.arange(int(SR * duration_sec), dtype=np.float64) / SR
    kick = 0.34 * np.sin(2.0 * np.pi * 74.0 * time)
    snare_body = 0.16 * np.sin(2.0 * np.pi * 190.0 * time)
    cymbal = 0.05 * np.sin(2.0 * np.pi * 4200.0 * time)
    samples = kick + snare_body + cymbal

    recommendation = analyze_smart_dsp_audio(
        samples,
        target_id="bus:1",
        source_role="drum_bus",
        source_label="Drum Bus",
        sample_rate=SR,
        source_identity_confirmed=True,
        source_identity_confidence=0.9,
        current_state_cache_freshness=_fresh(),
        current_state_readback_available=True,
        rollback_plan_available=True,
        target_plane_lufs=-25.0,
    )

    audio_evidence = recommendation.evidence["audio_evidence"]
    assert audio_evidence["level_plane_error_lu"] > 2.0
    assert "bus_group_impact" in audio_evidence["problem_hints"]
    assert recommendation.selected_strategy.strategy_id == "group_bus_trim_for_headroom"
    params = recommendation.selected_strategy.parameters
    assert params["target_scope"] == "group_bus"
    assert params["group_delta_db"] < 0.0
    assert params["gain_db"] == 0.0
    assert recommendation.to_operator_proposal()["live_apply_allowed"] is False


def test_live_loud_source_audio_selects_gain_down():
    samples = _tone(110.0, amplitude=0.42)

    recommendation = analyze_smart_dsp_audio(
        samples,
        target_id="ch:2",
        source_role="bass",
        source_label="Bass DI",
        mode="live",
        sample_rate=SR,
        source_identity_confirmed=True,
        source_identity_confidence=0.9,
        current_state_cache_freshness=_fresh(),
        current_state_readback_available=True,
        rollback_plan_available=True,
        target_plane_lufs=-22.5,
        active_performance=True,
    )

    audio_evidence = recommendation.evidence["audio_evidence"]
    assert audio_evidence["level_plane_error_lu"] > 2.0
    assert "level_plane_balance" in recommendation.evidence["detected_problems"]
    assert recommendation.selected_strategy.strategy_id == "level_plane_gain_down"
    assert recommendation.selected_strategy.parameters["delta_db"] < 0.0
    assert recommendation.to_operator_proposal()["live_apply_allowed"] is False


def test_bass_low_end_resonance_audio_selects_wing_native_deq_dynamic_cut():
    samples = (
        _tone(118.0, amplitude=0.11, duration_sec=2.4)
        + _tone(236.0, amplitude=0.033, duration_sec=2.4)
        + _tone(900.0, amplitude=0.0088, duration_sec=2.4)
    )

    recommendation = analyze_smart_dsp_audio(
        samples,
        target_id="ch:2",
        source_role="bass",
        source_label="Bass DI",
        sample_rate=SR,
        source_identity_confirmed=True,
        source_identity_confidence=0.9,
        current_state_cache_freshness=_fresh(),
        current_state_readback_available=True,
        rollback_plan_available=True,
        target_plane_lufs=-22.5,
    )

    audio_evidence = recommendation.evidence["audio_evidence"]
    assert audio_evidence["low_end_excess_score"] >= 0.9
    assert audio_evidence["low_end_resonance_frequency_hz"] is not None
    assert audio_evidence["low_end_resonance_bandwidth_hz"] > 0.0
    assert audio_evidence["low_end_resonance_peak_dbfs"] < -1.0
    assert audio_evidence["low_end_resonance_sustain_ms"] > 0.0
    assert "low_end_resonance" in audio_evidence["problem_hints"]
    assert "masking" not in audio_evidence["problem_hints"]
    assert recommendation.selected_strategy.strategy_id == "low_end_dynamic_resonance_control"
    assert recommendation.selected_strategy.method_family == "dynamic_eq"
    params = recommendation.selected_strategy.parameters
    assert params["source_family"] == "bass"
    assert params["low_end_parameter_source"] == "measured_low_end_resonance"
    assert params["frequency_hz"] == audio_evidence["low_end_resonance_frequency_hz"]
    assert params["low_end_resonance_bandwidth_hz"] == audio_evidence[
        "low_end_resonance_bandwidth_hz"
    ]
    assert params["max_gain_reduction_db"] > 0.0
    assert params["wing_dynamic_eq_model_osc_value"] == WING_DYNAMIC_EQ_GATE_MODEL
    assert params["wing_dynamic_eq_filter_osc_value"] == "BP"
    assert params["wing_dynamic_eq_gain_osc_value"] < 0.0
    assert params["wing_dynamic_eq_frequency_osc_value"] == params["frequency_hz"]
    assert params["q"] != 1.2
    assert recommendation.selected_strategy.future_wing_readback_contract[
        "wing_osc_applicability"
    ]["native_osc_supported"] is True
    assert recommendation.selected_strategy.future_wing_readback_contract[
        "dynamic_eq_readback"
    ]["wing_dynamic_eq_model_readback_must_equal"] == WING_DYNAMIC_EQ_GATE_MODEL
    assert recommendation.evidence["operator_explanation_sections"][
        "selection_policy"
    ] == "weighted_strategy_score"
    static = [
        candidate
        for candidate in recommendation.alternatives
        if candidate.strategy_id == "low_end_static_pocket_cut"
    ][0]
    assert static.parameters["gain_db"] < 0.0
    assert static.parameters["no_boost"] is True
    assert static.future_wing_readback_contract["wing_osc_applicability"][
        "native_osc_supported"
    ] is True
    assert params["low_end_resonance_peak_dbfs"] == audio_evidence[
        "low_end_resonance_peak_dbfs"
    ]
    assert params["low_end_resonance_sustain_ms"] == audio_evidence[
        "low_end_resonance_sustain_ms"
    ]
    assert params["release_ms"] != 180.0
    assert params["makeup_gain_db"] == 0.0
    assert recommendation.to_operator_proposal()["live_apply_allowed"] is False


def test_live_bass_low_end_resonance_audio_selects_static_cut():
    samples = (
        _tone(118.0, amplitude=0.11, duration_sec=2.4)
        + _tone(236.0, amplitude=0.033, duration_sec=2.4)
        + _tone(900.0, amplitude=0.0088, duration_sec=2.4)
    )

    recommendation = analyze_smart_dsp_audio(
        samples,
        target_id="ch:2",
        source_role="bass",
        source_label="Bass DI",
        mode="live",
        sample_rate=SR,
        source_identity_confirmed=True,
        source_identity_confidence=0.9,
        current_state_cache_freshness=_fresh(),
        current_state_readback_available=True,
        rollback_plan_available=True,
        target_plane_lufs=-22.5,
        active_performance=True,
    )

    audio_evidence = recommendation.evidence["audio_evidence"]
    assert audio_evidence["low_end_excess_score"] >= 0.9
    assert audio_evidence["low_end_resonance_bandwidth_hz"] > 0.0
    assert "low_end_resonance" in audio_evidence["problem_hints"]
    assert recommendation.selected_strategy.strategy_id == "low_end_static_pocket_cut"
    params = recommendation.selected_strategy.parameters
    assert params["gain_db"] < 0.0
    assert params["low_end_parameter_source"] == "measured_low_end_resonance"
    assert params["low_end_resonance_bandwidth_hz"] == audio_evidence[
        "low_end_resonance_bandwidth_hz"
    ]
    assert params["no_boost"] is True
    assert recommendation.to_operator_proposal()["live_apply_allowed"] is False


def test_kick_bass_audio_overlap_selects_wing_native_static_eq_split():
    duration_sec = 2.4
    time = np.arange(int(SR * duration_sec), dtype=np.float64) / SR
    bass = 0.08 * np.sin(2.0 * np.pi * 74.0 * time)
    bass += 0.0144 * np.sin(2.0 * np.pi * 148.0 * time)
    kick_reference = 0.07 * np.sin(2.0 * np.pi * 74.0 * time)
    kick_reference += 0.35 * _kick_burst_train(duration_sec=duration_sec)

    recommendation = analyze_smart_dsp_audio(
        bass,
        target_id="ch:2",
        source_role="bass",
        source_label="Bass DI",
        sample_rate=SR,
        source_identity_confirmed=True,
        source_identity_confidence=0.9,
        current_state_cache_freshness=_fresh(),
        current_state_readback_available=True,
        rollback_plan_available=True,
        reference_samples=kick_reference,
        reference_role="kick",
        reference_target_id="ch:1",
        target_plane_lufs=-25.0,
    )

    audio_evidence = recommendation.evidence["audio_evidence"]
    assert audio_evidence["masking_score"] >= 0.9
    assert 70.0 <= audio_evidence["masking_frequency_hz"] <= 78.0
    assert audio_evidence["masking_bandwidth_hz"] > 0.0
    assert audio_evidence["masking_overlap_peak_share"] > 0.0
    assert audio_evidence["masking_conflict_role"] == "kick"
    assert audio_evidence["masking_conflict_target_id"] == "ch:1"
    assert "masking" in audio_evidence["problem_hints"]
    assert audio_evidence["arrangement_overlap_score"] < 0.68
    assert abs(audio_evidence["level_plane_error_lu"]) < 2.0
    assert recommendation.selected_strategy.strategy_id == "kick_bass_pocket_eq_split"
    assert recommendation.selected_strategy.method_family == "eq"
    params = recommendation.selected_strategy.parameters
    assert params["relationship"] == "kick_masks_bass"
    assert params["target_role"] == "bass"
    assert params["counterpart_role"] == "kick"
    assert params["gain_db"] < 0.0
    assert params["q"] != 1.25
    assert params["static_eq_q_source"] == "measured_masking_overlap_band"
    assert params["masking_parameter_source"] == "measured_masking_overlap_band"
    assert params["masking_bandwidth_hz"] == audio_evidence["masking_bandwidth_hz"]
    assert params["masking_overlap_peak_share"] == audio_evidence[
        "masking_overlap_peak_share"
    ]
    assert params["target_cut_db"] < 0.0
    assert params["requires_pair_readback"] is True
    assert recommendation.selected_strategy.future_wing_readback_contract[
        "wing_osc_applicability"
    ]["native_osc_supported"] is True
    assert recommendation.evidence["operator_explanation_sections"][
        "selection_policy"
    ] == "weighted_strategy_score"
    dynamic = [
        candidate
        for candidate in recommendation.alternatives
        if candidate.strategy_id == "kick_bass_dynamic_pocket_ducking"
    ][0]
    assert dynamic.parameters["sidechain_role"] == "kick"
    assert dynamic.parameters["masking_parameter_source"] == "measured_masking_overlap_band"
    assert dynamic.parameters["masking_bandwidth_hz"] == audio_evidence["masking_bandwidth_hz"]
    assert dynamic.parameters["masking_overlap_peak_share"] == audio_evidence[
        "masking_overlap_peak_share"
    ]
    assert dynamic.parameters["wing_dynamic_eq_external_key_source_target_id"] == "ch:1"
    assert (
        dynamic.parameters["wing_dynamic_eq_external_key_source_target_surface"]
        == "channel"
    )
    assert (
        dynamic.parameters["wing_dynamic_eq_external_key_source_target_parse_blocker"]
        == ""
    )
    assert (
        dynamic.parameters["wing_dynamic_eq_external_key_required_source_alias_format"]
        == "CH.n, BUS.n, MAIN.n, MTX.n or AUX.n"
    )
    assert (
        dynamic.parameters["wing_dynamic_eq_external_key_candidate_source_alias"]
        == "CH.1"
    )
    assert (
        dynamic.parameters["wing_dynamic_eq_external_key_concrete_source_available"]
        is True
    )
    assert (
        dynamic.parameters["wing_dynamic_eq_external_key_verified_osc_mapping_available"]
        is False
    )
    assert (
        dynamic.parameters[
            "wing_dynamic_eq_external_key_apply_blocked_without_concrete_source"
        ]
        is False
    )
    assert (
        dynamic.parameters[
            "wing_dynamic_eq_external_key_apply_blocked_even_with_concrete_source"
        ]
        is True
    )
    assert dynamic.parameters["bandwidth_q"] != 1.1
    assert dynamic.parameters["attack_ms"] != 6.0
    assert dynamic.future_wing_readback_contract["wing_osc_applicability"][
        "native_osc_supported"
    ] is False
    assert recommendation.to_operator_proposal()["live_apply_allowed"] is False


def test_kick_bass_audio_hot_summed_bus_rebalances_before_ducking():
    duration_sec = 2.4
    time = np.arange(int(SR * duration_sec), dtype=np.float64) / SR
    bass = 0.08 * np.sin(2.0 * np.pi * 74.0 * time)
    bass += 0.0144 * np.sin(2.0 * np.pi * 148.0 * time)
    kick_reference = 0.07 * np.sin(2.0 * np.pi * 74.0 * time)
    kick_reference += 0.35 * _kick_burst_train(duration_sec=duration_sec)
    summed_bus = np.clip(2.8 * bass + 1.6 * kick_reference, -0.95, 0.95)

    recommendation = analyze_smart_dsp_audio(
        bass,
        target_id="ch:2",
        source_role="bass",
        source_label="Bass DI",
        sample_rate=SR,
        source_identity_confirmed=True,
        source_identity_confidence=0.9,
        current_state_cache_freshness=_fresh(),
        current_state_readback_available=True,
        rollback_plan_available=True,
        reference_samples=kick_reference,
        reference_role="kick",
        target_plane_lufs=-25.0,
        summed_bus_samples=summed_bus,
        summed_bus_role="low_end_bus",
        summed_bus_target_id="bus:3",
        summed_bus_target_plane_lufs=-25.0,
    )

    audio_evidence = recommendation.evidence["audio_evidence"]
    assert audio_evidence["masking_score"] >= 0.9
    assert audio_evidence["summed_bus_evidence_available"] is True
    assert audio_evidence["summed_bus_level_plane_error_lu"] >= 3.0
    assert audio_evidence["summed_bus_target_id"] == "bus:3"
    assert recommendation.selected_strategy.strategy_id == (
        "kick_bass_summed_bus_rebalance_before_ducking"
    )
    params = recommendation.selected_strategy.parameters
    assert params["target_scope"] == "summed_low_end_bus_before_ducking"
    assert params["bus_delta_db"] < 0.0
    assert params["source_delta_db"] == 0.0
    assert params["summed_bus_target_id"] == "bus:3"
    assert params["wing_osc_effective_target_id"] == "bus:3"
    assert recommendation.evidence["operator_explanation_sections"][
        "selection_policy"
    ] == "soundcheck_kick_bass_summed_bus_rebalance_priority"
    duck = [
        candidate
        for candidate in recommendation.alternatives
        if candidate.strategy_id == "kick_bass_dynamic_pocket_ducking"
    ][0]
    assert "summed_bus_rebalance_should_precede_dynamic_ducking" in duck.blockers
    assert recommendation.to_operator_proposal()["live_apply_allowed"] is False


def test_sustained_kick_bass_overlap_selects_arrangement_hold():
    duration_sec = 2.4
    time = np.arange(int(SR * duration_sec), dtype=np.float64) / SR
    bass = 0.08 * np.sin(2.0 * np.pi * 74.0 * time)
    bass += 0.012 * np.sin(2.0 * np.pi * 148.0 * time)
    kick_reference = 0.072 * np.sin(2.0 * np.pi * 74.0 * time)
    kick_reference += 0.010 * np.sin(2.0 * np.pi * 148.0 * time)

    recommendation = analyze_smart_dsp_audio(
        bass,
        target_id="ch:2",
        source_role="bass",
        source_label="Bass DI",
        sample_rate=SR,
        source_identity_confirmed=True,
        source_identity_confidence=0.9,
        current_state_cache_freshness=_fresh(),
        current_state_readback_available=True,
        rollback_plan_available=True,
        reference_samples=kick_reference,
        reference_role="kick",
        target_plane_lufs=-25.0,
    )

    audio_evidence = recommendation.evidence["audio_evidence"]
    assert audio_evidence["masking_score"] >= 0.9
    assert audio_evidence["arrangement_overlap_score"] >= 0.68
    assert "arrangement_overlap" in audio_evidence["problem_hints"]
    assert (
        recommendation.selected_strategy.strategy_id
        == "arrangement_aware_low_end_separation_hold"
    )
    params = recommendation.selected_strategy.parameters
    assert params["dsp_change"] == "hold"
    assert params["requires_arrangement_or_register_review"] is True
    assert recommendation.to_operator_proposal()["can_apply_live"] is False


def test_unknown_low_end_audio_infers_bass_for_kick_bass_strategy():
    duration_sec = 2.4
    time = np.arange(int(SR * duration_sec), dtype=np.float64) / SR
    bass = 0.08 * np.sin(2.0 * np.pi * 74.0 * time)
    bass += 0.0144 * np.sin(2.0 * np.pi * 148.0 * time)
    kick_reference = 0.07 * np.sin(2.0 * np.pi * 74.0 * time)
    kick_reference += 0.35 * _kick_burst_train(duration_sec=duration_sec)

    recommendation = analyze_smart_dsp_audio(
        bass,
        target_id="ch:2",
        source_role="unknown",
        source_label="",
        sample_rate=SR,
        current_state_cache_freshness=_fresh(),
        current_state_readback_available=True,
        rollback_plan_available=True,
        reference_samples=kick_reference,
        reference_role="kick",
        target_plane_lufs=-25.0,
        allow_inferred_source_identity=True,
    )

    audio_evidence = recommendation.evidence["audio_evidence"]
    assert audio_evidence["declared_source_family"] == "unknown"
    assert audio_evidence["inferred_source_family"] == "bass"
    assert audio_evidence["source_identity_inference_confidence"] >= 0.6
    assert recommendation.status == "proposal_only"
    assert recommendation.source_family == "bass"
    assert recommendation.selected_strategy.strategy_id == "kick_bass_pocket_eq_split"
    assert recommendation.to_operator_proposal()["can_apply_live"] is False


def test_declared_vocal_with_low_end_transient_audio_holds_for_identity_recheck():
    samples = _kick_burst_train()

    recommendation = analyze_smart_dsp_audio(
        samples,
        target_id="ch:1",
        source_role="lead_vocal",
        source_label="Lead Vox",
        sample_rate=SR,
        source_identity_confirmed=True,
        source_identity_confidence=0.95,
        current_state_cache_freshness=_fresh(),
        current_state_readback_available=True,
        rollback_plan_available=True,
    )

    audio_evidence = recommendation.evidence["audio_evidence"]
    assert audio_evidence["declared_source_family"] == "vocal"
    assert audio_evidence["source_identity_mismatch_score"] >= 0.68
    assert audio_evidence["source_identity_mismatch_family"] in {"kick", "bass"}
    assert "source_identity_mismatch" in audio_evidence["problem_hints"]
    assert recommendation.status == "analysis_only"
    assert recommendation.selected_strategy.strategy_id == "hold_for_source_identity_recheck"
    assert recommendation.evidence["detected_problems"] == ["source_identity_mismatch"]
    assert "source_identity_evidence_conflict" in recommendation.blockers
    assert recommendation.to_operator_proposal()["live_apply_allowed"] is False


def test_long_fx_tail_audio_selects_wing_native_level_pullback():
    duration_sec = 3.0
    time = np.arange(int(SR * duration_sec), dtype=np.float64) / SR
    envelope = np.exp(-time / 1.45)
    samples = 0.22 * envelope * (
        np.sin(2.0 * np.pi * 4200.0 * time)
        + 0.35 * np.sin(2.0 * np.pi * 1800.0 * time)
    )

    recommendation = analyze_smart_dsp_audio(
        samples,
        target_id="fx:1",
        source_role="fx_return",
        source_label="Vocal Verb",
        sample_rate=SR,
        source_identity_confirmed=True,
        source_identity_confidence=0.9,
        current_state_cache_freshness=_fresh(),
        current_state_readback_available=True,
        rollback_plan_available=True,
        target_plane_lufs=-22.5,
        fx_return_headroom_db=6.0,
        monitor_feedback_risk=0.05,
    )

    audio_evidence = recommendation.evidence["audio_evidence"]
    assert audio_evidence["fx_decay_risk_score"] >= 0.45
    assert audio_evidence["fx_tail_decay_seconds"] >= 2.5
    assert "fx_intelligibility" in audio_evidence["problem_hints"]
    assert recommendation.selected_strategy.strategy_id == "reduce_fx_send_or_return"
    params = recommendation.selected_strategy.parameters
    assert params["send_delta_db"] == 0.0
    assert params["return_delta_db"] < 0.0
    assert params["post_pullback_time_edit_review_required"] is True
    assert recommendation.evidence["operator_explanation_sections"][
        "selection_policy"
    ] == "soundcheck_wing_native_osc_fx_level_pullback_priority"
    assert recommendation.selected_strategy.future_wing_readback_contract[
        "wing_osc_applicability"
    ]["native_osc_supported"] is True
    decay_edit = {
        candidate.strategy_id: candidate for candidate in recommendation.alternatives
    }["shorten_fx_decay_filter"]
    decay_params = decay_edit.parameters
    assert decay_params["measured_tail_decay_s"] == audio_evidence[
        "fx_tail_decay_seconds"
    ]
    assert decay_params["decay_s"] < decay_params["measured_tail_decay_s"]
    assert decay_params["fx_decay_parameter_source"] == "measured_tail_decay_role_target"
    assert decay_params["fx_target_context"] == "vocal_reverb"
    assert decay_params["pre_delay_ms"] >= 45.0
    assert decay_params["fx_tail_density_score"] > 0.55
    assert decay_params["return_lpf_hz"] < 6800.0
    assert decay_edit.future_wing_readback_contract["wing_osc_applicability"][
        "native_osc_supported"
    ] is False
    assert recommendation.to_operator_proposal()["live_apply_allowed"] is False


def test_current_wing_state_operator_owned_fx_model_path_blocks_decay_edit():
    duration_sec = 3.0
    time = np.arange(int(SR * duration_sec), dtype=np.float64) / SR
    envelope = np.exp(-time / 1.45)
    samples = 0.22 * envelope * (
        np.sin(2.0 * np.pi * 4200.0 * time)
        + 0.35 * np.sin(2.0 * np.pi * 1800.0 * time)
    )

    recommendation = analyze_smart_dsp_audio(
        samples,
        target_id="fx:2",
        source_role="fx_return",
        source_label="Vocal Verb",
        sample_rate=SR,
        source_identity_confirmed=True,
        source_identity_confidence=0.91,
        current_state_cache_freshness=_fresh(),
        current_state_readback_available=True,
        rollback_plan_available=True,
        target_plane_lufs=-22.5,
        fx_return_headroom_db=6.0,
        fx_return_target_id="fx:2",
        monitor_feedback_risk=0.05,
        current_wing_state={
            "current_wing_fx_model": "HALL",
            "current_wing_fx_model_readback_available": True,
            "operator_owned_osc_paths": (
                "/fx/2/mdl",
                "/fx/2/dcy",
            ),
        },
    )

    assert recommendation.selected_strategy.strategy_id == "reduce_fx_send_or_return"
    alternatives = {
        candidate.strategy_id: candidate for candidate in recommendation.alternatives
    }
    decay_edit = alternatives["shorten_fx_decay_filter"]
    assert decay_edit.blockers == (WING_FX_OPERATOR_OWNED_MODEL_BLOCKER,)
    assert decay_edit.parameters["current_wing_fx_operator_owned"] is True
    assert decay_edit.parameters["wing_fx_model_parameter_current_model"] == "HALL"
    assert (
        decay_edit.parameters[
            "wing_fx_model_parameter_current_model_operator_owned_blocks_apply"
        ]
        is True
    )
    assert decay_edit.parameters["wing_fx_model_parameter_blocker"] == (
        WING_FX_OPERATOR_OWNED_MODEL_BLOCKER
    )
    assert decay_edit.future_wing_readback_contract["wing_osc_applicability"][
        "native_osc_supported"
    ] is False


def test_current_wing_state_operator_owned_fx_level_blocks_extracted_pullback():
    duration_sec = 3.0
    time = np.arange(int(SR * duration_sec), dtype=np.float64) / SR
    envelope = np.exp(-time / 1.45)
    samples = 0.22 * envelope * (
        np.sin(2.0 * np.pi * 4200.0 * time)
        + 0.35 * np.sin(2.0 * np.pi * 1800.0 * time)
    )

    recommendation = analyze_smart_dsp_audio(
        samples,
        target_id="fx:1",
        source_role="fx_return",
        source_label="Vocal Verb",
        sample_rate=SR,
        source_identity_confirmed=True,
        source_identity_confidence=0.9,
        current_state_cache_freshness=_fresh(),
        current_state_readback_available=True,
        rollback_plan_available=True,
        target_plane_lufs=-22.5,
        fx_return_headroom_db=6.0,
        monitor_feedback_risk=0.05,
        current_wing_state={"current_wing_fx_level_operator_owned": True},
    )

    assert recommendation.selected_strategy.strategy_id == "shorten_fx_decay_filter"
    alternatives = {
        candidate.strategy_id: candidate for candidate in recommendation.alternatives
    }
    pullback = alternatives["reduce_fx_send_or_return"]
    assert pullback.blockers == (WING_FX_LEVEL_OPERATOR_OWNED_BLOCKER,)
    assert pullback.parameters["wing_fx_level_current_operator_owned"] is True
    assert pullback.parameters["wing_fx_level_operator_owned_edit_requires_review"] is True
    assert pullback.parameters["wing_fx_level_operator_owned_blocker"] == (
        WING_FX_LEVEL_OPERATOR_OWNED_BLOCKER
    )
    wing_osc = pullback.future_wing_readback_contract["wing_osc_applicability"]
    assert wing_osc["native_osc_supported"] is False
    assert wing_osc["native_osc_support_level"] == (
        "fx_level_operator_owned_edit_requires_review"
    )
    assert wing_osc["unsupported_strategy_reason"] == (
        "fx_level_has_native_osc_paths_but_current_wing_fx_send_or_return_level_is_operator_owned"
    )


def test_current_wing_state_operator_owned_fx_level_path_blocks_extracted_pullback():
    duration_sec = 3.0
    time = np.arange(int(SR * duration_sec), dtype=np.float64) / SR
    envelope = np.exp(-time / 1.45)
    samples = 0.22 * envelope * (
        np.sin(2.0 * np.pi * 4200.0 * time)
        + 0.35 * np.sin(2.0 * np.pi * 1800.0 * time)
    )

    recommendation = analyze_smart_dsp_audio(
        samples,
        target_id="fx:1",
        source_role="fx_return",
        source_label="Vocal Verb",
        sample_rate=SR,
        source_identity_confirmed=True,
        source_identity_confidence=0.9,
        current_state_cache_freshness=_fresh(),
        current_state_readback_available=True,
        rollback_plan_available=True,
        target_plane_lufs=-22.5,
        fx_return_headroom_db=6.0,
        monitor_feedback_risk=0.05,
        current_wing_state={"operator_owned_osc_paths": ("/fx/1/fxmix",)},
    )

    assert recommendation.selected_strategy.strategy_id == "shorten_fx_decay_filter"
    alternatives = {
        candidate.strategy_id: candidate for candidate in recommendation.alternatives
    }
    pullback = alternatives["reduce_fx_send_or_return"]
    assert pullback.blockers == (WING_FX_LEVEL_OPERATOR_OWNED_BLOCKER,)
    assert pullback.parameters["wing_fx_level_current_operator_owned"] is True
    assert pullback.parameters["wing_fx_level_operator_owned_edit_requires_review"] is True
    assert pullback.future_wing_readback_contract["wing_osc_applicability"][
        "native_osc_supported"
    ] is False


def test_delay_smear_fx_audio_selects_wing_native_level_pullback():
    duration_sec = 2.5
    samples = np.zeros(int(SR * duration_sec), dtype=np.float64)
    width = int(0.012 * SR)
    envelope = np.hanning(width)
    carrier = (
        np.sin(2.0 * np.pi * 1800.0 * np.arange(width) / SR)
        * envelope
        * 0.18
    )
    for start_sec in (0.1, 0.6, 1.1, 1.6, 2.1):
        start = int(start_sec * SR)
        samples[start : start + width] += carrier
        for delay_sec, gain in ((0.13, 0.42), (0.26, 0.22)):
            delayed = start + int(delay_sec * SR)
            if delayed + width < samples.size:
                samples[delayed : delayed + width] += carrier * gain

    recommendation = analyze_smart_dsp_audio(
        samples,
        target_id="fx:1",
        source_role="fx_return",
        source_label="Vocal Delay",
        sample_rate=SR,
        source_identity_confirmed=True,
        source_identity_confidence=0.95,
        current_state_cache_freshness=_fresh(),
        current_state_readback_available=True,
        rollback_plan_available=True,
        target_plane_lufs=-22.5,
        fx_return_headroom_db=5.0,
        monitor_feedback_risk=0.1,
    )

    audio_evidence = recommendation.evidence["audio_evidence"]
    assert audio_evidence["fx_delay_smear_score"] >= 0.7
    assert 120.0 <= audio_evidence["fx_delay_repeat_ms"] <= 140.0
    assert "fx_intelligibility" in audio_evidence["problem_hints"]
    assert recommendation.selected_strategy.strategy_id == "reduce_fx_send_or_return"
    params = recommendation.selected_strategy.parameters
    assert params["send_delta_db"] == 0.0
    assert params["return_delta_db"] < 0.0
    assert params["post_pullback_time_edit_review_required"] is True
    assert recommendation.evidence["operator_explanation_sections"][
        "selection_policy"
    ] == "soundcheck_wing_native_osc_fx_level_pullback_priority"
    assert recommendation.selected_strategy.future_wing_readback_contract[
        "wing_osc_applicability"
    ]["native_osc_supported"] is True
    delay_ducking = {
        candidate.strategy_id: candidate for candidate in recommendation.alternatives
    }["duck_delay_return_cleanup"]
    delay_params = delay_ducking.parameters
    assert delay_params["fx_delay_parameter_source"] == "measured_delay_repeat"
    assert delay_params["delay_time_ms"] == audio_evidence["fx_delay_repeat_ms"]
    assert delay_params["measured_delay_repeat_ms"] == audio_evidence[
        "fx_delay_repeat_ms"
    ]
    assert delay_params["measured_tail_decay_s"] == audio_evidence[
        "fx_tail_decay_seconds"
    ]
    assert delay_params["attack_ms"] > 0.0
    assert delay_params["release_ms"] >= delay_params["delay_time_ms"]
    assert delay_params["delay_low_pass_hz"] < 6200.0
    assert delay_params["ducking_depth_db"] > 0.0
    assert delay_params["delay_feedback_percent"] < 12.0
    assert delay_ducking.future_wing_readback_contract["wing_osc_applicability"][
        "native_osc_supported"
    ] is False
    assert recommendation.to_operator_proposal()["live_apply_allowed"] is False


def test_low_headroom_delay_fx_audio_restores_return_headroom_before_ducking():
    duration_sec = 2.5
    samples = np.zeros(int(SR * duration_sec), dtype=np.float64)
    width = int(0.012 * SR)
    envelope = np.hanning(width)
    carrier = (
        np.sin(2.0 * np.pi * 1800.0 * np.arange(width) / SR)
        * envelope
        * 0.82
    )
    for start_sec in (0.1, 0.6, 1.1, 1.6, 2.1):
        start = int(start_sec * SR)
        samples[start : start + width] += carrier
        for delay_sec, gain in ((0.13, 0.42), (0.26, 0.22)):
            delayed = start + int(delay_sec * SR)
            if delayed + width < samples.size:
                samples[delayed : delayed + width] += carrier * gain
    samples = np.clip(samples, -0.98, 0.98)

    recommendation = analyze_smart_dsp_audio(
        samples,
        target_id="fx:1",
        source_role="fx_return",
        source_label="Vocal Delay",
        sample_rate=SR,
        source_identity_confirmed=True,
        source_identity_confidence=0.95,
        current_state_cache_freshness=_fresh(),
        current_state_readback_available=True,
        rollback_plan_available=True,
        target_plane_lufs=-22.5,
        monitor_feedback_risk=0.1,
    )

    audio_evidence = recommendation.evidence["audio_evidence"]
    assert audio_evidence["true_peak_headroom_db"] < 1.0
    assert audio_evidence["fx_return_headroom_db"] == audio_evidence[
        "true_peak_headroom_db"
    ]
    assert audio_evidence["fx_delay_smear_score"] >= 0.7
    assert "fx_intelligibility" in audio_evidence["problem_hints"]
    assert recommendation.status == "analysis_only"
    assert "true_peak_headroom_low" in recommendation.blockers
    assert (
        recommendation.selected_strategy.strategy_id
        == "fx_return_headroom_pullback_before_time_edits"
    )
    assert recommendation.evidence["operator_explanation_sections"][
        "selection_policy"
    ] == "fx_return_headroom_restoration_priority"
    params = recommendation.selected_strategy.parameters
    assert params["fx_return_headroom_parameter_source"] == "measured_fx_return_headroom"
    assert params["fx_return_headroom_db"] == audio_evidence["fx_return_headroom_db"]
    assert params["fx_return_headroom_deficit_db"] > 1.0
    assert params["return_delta_db"] < 0.0
    assert params["return_headroom_restoration_first"] is True
    delay_ducking = {
        candidate.strategy_id: candidate for candidate in recommendation.alternatives
    }["duck_delay_return_cleanup"]
    assert "fx_return_headroom_pullback_should_precede_delay_ducking" in (
        delay_ducking.blockers
    )
    assert recommendation.to_operator_proposal()["live_apply_allowed"] is False


def test_source_fx_audio_uses_bound_return_target_for_fxmix_pullback():
    duration_sec = 2.0
    time = np.arange(int(SR * duration_sec), dtype=np.float64) / SR
    samples = (
        0.08 * np.sin(2.0 * np.pi * 1200.0 * time)
        + 0.03 * np.sin(2.0 * np.pi * 260.0 * time)
    )

    recommendation = analyze_smart_dsp_audio(
        samples,
        target_id="ch:5",
        source_role="lead_vocal",
        source_label="Lead Vox",
        sample_rate=SR,
        source_identity_confirmed=True,
        source_identity_confidence=0.91,
        current_state_cache_freshness=_fresh(),
        current_state_readback_available=True,
        rollback_plan_available=True,
        target_plane_lufs=-25.0,
        fx_return_headroom_db=1.2,
        fx_return_target_id="fx:1",
        monitor_feedback_risk=0.1,
        extra_problem_hints=("fx_intelligibility",),
    )

    audio_evidence = recommendation.evidence["audio_evidence"]
    assert audio_evidence["fx_return_headroom_db"] == 1.2
    assert audio_evidence["fx_return_target_id"] == "fx:1"
    assert abs(audio_evidence["level_plane_error_lu"]) < 0.3
    assert recommendation.selected_strategy.strategy_id == (
        "fx_return_headroom_pullback_before_time_edits"
    )
    params = recommendation.selected_strategy.parameters
    assert params["send_delta_db"] == 0.0
    assert params["return_delta_db"] == -1.3
    assert params["fx_return_target_id"] == "fx:1"
    assert params["fx_return_target_surface"] == "fx"
    assert params["wing_fx_return_level_target_surface"] == "fx"
    assert params["wing_fx_return_source_target_surface"] == "channel"
    assert params["wing_fx_return_delta_db"] == -1.3
    assert params["wing_fx_return_fx_slot_binding_required"] is False
    assert "wing_fx_send_level_delta_db" not in params
    assert params["soundcheck_fx_level_move_kinds"] == ["return_level"]
    assert params["soundcheck_fx_selected_write_keys"] == [
        "wing_fx_return_delta_db"
    ]
    proposal = recommendation.to_operator_proposal()
    assert proposal["selected_wing_osc_summary"][
        "wing_osc_absolute_value_readback_parameter_values"
    ] == {"wing_fx_return_delta_db": -1.3}
    assert proposal["live_apply_allowed"] is False


def test_extreme_fx_monitor_risk_audio_selects_safety_pullback():
    duration_sec = 2.5
    time = np.arange(int(SR * duration_sec), dtype=np.float64) / SR
    envelope = np.exp(-time / 1.1)
    samples = 0.18 * envelope * (
        np.sin(2.0 * np.pi * 3800.0 * time)
        + 0.4 * np.sin(2.0 * np.pi * 1200.0 * time)
    )

    recommendation = analyze_smart_dsp_audio(
        samples,
        target_id="fx:1",
        source_role="fx_return",
        source_label="Vocal Verb",
        sample_rate=SR,
        source_identity_confirmed=True,
        source_identity_confidence=0.9,
        current_state_cache_freshness=_fresh(),
        current_state_readback_available=True,
        rollback_plan_available=True,
        target_plane_lufs=-22.5,
        fx_return_headroom_db=0.6,
        monitor_feedback_risk=0.9,
    )

    audio_evidence = recommendation.evidence["audio_evidence"]
    assert audio_evidence["monitor_feedback_risk"] == 0.9
    assert audio_evidence["fx_return_headroom_db"] == 0.6
    assert recommendation.selected_strategy.strategy_id == "fx_monitor_feedback_safety_pullback"
    assert recommendation.selected_strategy.parameters["return_mute_review_required"] is True
    assert recommendation.selected_strategy.parameters["send_delta_db"] == 0.0
    assert recommendation.selected_strategy.parameters["return_delta_db"] < -3.0
    assert recommendation.to_operator_proposal()["live_apply_allowed"] is False
