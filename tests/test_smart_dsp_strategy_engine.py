from pathlib import Path

from backend.smart_dsp_strategy_engine import (
    LIVE_DE_ESSER_MAX_RANGE_DB,
    LIVE_DE_ESSER_MAX_RANGE_STEP_DB,
    LIVE_DYNAMICS_MAX_TARGET_GAIN_REDUCTION_DB,
    LIVE_DYNAMICS_MAX_THRESHOLD_STEP_DB,
    LIVE_EQ_MAX_BOOST_DB,
    LIVE_EQ_MAX_CUT_DB,
    LIVE_FX_MAX_BOOST_DB,
    LIVE_FX_MAX_PULLBACK_DB,
    LIVE_MAX_SINGLE_STEP_DB,
    LIVE_MIN_STEP_INTERVAL_MS,
    LIVE_STRATEGY_SCORE_WEIGHTS,
    SOUNDCHECK_FX_MAX_BOOST_DB,
    SOUNDCHECK_FX_MAX_PULLBACK_DB,
    SOUNDCHECK_MASTER_MAX_LIFT_DB,
    SOUNDCHECK_MASTER_MAX_PULLBACK_DB,
    SOUNDCHECK_MASTER_MIN_HEADROOM_MARGIN_DB,
    SOUNDCHECK_MASTER_BUS_COMPRESSION_MAX_TARGET_GR_DB,
    SOUNDCHECK_MASTER_EQ_MAX_BOOST_DB,
    SOUNDCHECK_MASTER_EQ_MAX_CUT_DB,
    WING_SOUNDCHECK_MASTER_OUTPUT_PROCESSING_REVIEW_ONLY_FIELDS,
    SOUNDCHECK_DYNAMICS_MAX_MAKEUP_GAIN_DB,
    SOUNDCHECK_DYNAMICS_MAX_TARGET_GAIN_REDUCTION_DB,
    SOUNDCHECK_DYNAMICS_MIN_HEADROOM_MARGIN_DB,
    SOUNDCHECK_EQ_MAX_BOOST_DB,
    SOUNDCHECK_EQ_MIN_HEADROOM_MARGIN_DB,
    SOUNDCHECK_STRATEGY_SCORE_WEIGHTS,
    SOUNDCHECK_GATE_EXPANDER_MAX_HYSTERESIS_REVIEW_DB,
    SOUNDCHECK_GATE_EXPANDER_MAX_RANGE_DB,
    SOUNDCHECK_GATE_EXPANDER_MIN_HIT_BLEED_MARGIN_DB,
    SOUNDCHECK_PAN_MAX_DELTA_PERCENT,
    SOUNDCHECK_WIDTH_MAX_REDUCTION_PERCENT,
    SOUNDCHECK_WIDTH_MAX_WIDEN_PERCENT,
    SOUNDCHECK_WIDTH_MIN_TARGET_PERCENT,
    SMART_DSP_METHOD_ID,
    WING_ACTIONABLE_PARAMETER_BLOCKER,
    WING_COMPRESSOR_ATTACK_MAX_MS,
    WING_COMPRESSOR_ATTACK_MIN_MS,
    WING_COMPRESSOR_AUTO_BLOCKER,
    WING_COMPRESSOR_AUTO_MAX,
    WING_COMPRESSOR_AUTO_MIN,
    WING_COMPRESSOR_DETECTOR_BLOCKER,
    WING_COMPRESSOR_ENVELOPE_CURVE_BLOCKER,
    WING_COMPRESSOR_ENVELOPE_CURVE_VALUES,
    WING_COMPRESSOR_ON_BLOCKER,
    WING_COMPRESSOR_ON_MAX,
    WING_COMPRESSOR_ON_MIN,
    WING_COMPRESSOR_HOLD_MAX_MS,
    WING_COMPRESSOR_HOLD_MIN_MS,
    WING_COMPRESSOR_MAKEUP_GAIN_MAX_DB,
    WING_COMPRESSOR_MAKEUP_GAIN_MIN_DB,
    WING_COMPRESSOR_MIX_MAX_PERCENT,
    WING_COMPRESSOR_MIX_MIN_PERCENT,
    WING_COMPRESSOR_RELEASE_MAX_MS,
    WING_COMPRESSOR_RELEASE_MIN_MS,
    WING_COMPRESSOR_CROSSOVER_DEPTH_MAX_DB,
    WING_COMPRESSOR_CROSSOVER_DEPTH_MIN_DB,
    WING_COMPRESSOR_CROSSOVER_FREQ_MAX_HZ,
    WING_COMPRESSOR_CROSSOVER_FREQ_MIN_HZ,
    WING_COMPRESSOR_CROSSOVER_TYPE_BLOCKER,
    WING_COMPRESSOR_CROSSOVER_TYPE_VALUES,
    WING_COMPRESSOR_SIDECHAIN_FREQ_MAX_HZ,
    WING_COMPRESSOR_SIDECHAIN_FREQ_MIN_HZ,
    WING_COMPRESSOR_SIDECHAIN_Q_MAX,
    WING_COMPRESSOR_SIDECHAIN_Q_MIN,
    WING_COMPRESSOR_SIDECHAIN_SOURCE_VALUES,
    WING_COMPRESSOR_SIDECHAIN_TAP_VALUES,
    WING_COMPRESSOR_SIDECHAIN_TYPE_VALUES,
    WING_CHANNEL_HIGH_CUT_FREQ_MAX_HZ,
    WING_CHANNEL_HIGH_CUT_FREQ_MIN_HZ,
    WING_CHANNEL_HIGH_CUT_SLOPES_DB_PER_OCTAVE,
    WING_DE_ESSER_FREQ_MAX_HZ,
    WING_DE_ESSER_FREQ_MIN_HZ,
    WING_DE_ESSER_GATE_MODEL,
    WING_DE_ESSER_GATE_SLOT_CONFLICT_POLICY,
    WING_DE_ESSER_MODE_VALUES,
    WING_DE_ESSER_RANGE_MAX_DB,
    WING_DE_ESSER_RANGE_MIN_DB,
    WING_DE_ESSER_WRITE_KEY_SELECTION_POLICY,
    WING_DE_ESSER_WRITE_PARAMETER_KEYS,
    WING_DYNAMIC_EQ_GATE_MODEL,
    WING_DYNAMIC_EQ_GATE_SLOT_CONFLICT_POLICY,
    WING_DYNAMIC_EQ_VALUE_MAPPING_BLOCKER,
    WING_DYNAMIC_EQ_WRITE_KEY_SELECTION_POLICY,
    WING_DYNAMIC_EQ_WRITE_PARAMETER_KEYS,
    WING_AUX_DYNAMICS_OPERATOR_OWNED_BLOCKER,
    WING_DYNAMICS_MODEL_CONFLICT_BLOCKER,
    WING_DYNAMICS_OPERATOR_OWNED_MODEL_BLOCKER,
    WING_DYNAMICS_MODEL_READBACK_BLOCKER,
    WING_DYNAMICS_GAIN_REDUCTION_OBJECTIVE_TOLERANCE_DB,
    WING_DYNAMICS_REVIEW_ONLY_PARAMETER_REASONS,
    WING_DYNAMICS_REQUIRED_MODEL,
    WING_DYNAMICS_THRESHOLD_PATH_SUFFIX_BY_SURFACE,
    WING_DYNAMICS_WRITE_KEY_SELECTION_POLICY,
    WING_DYNAMICS_WRITE_PARAMETER_KEYS,
    WING_GATE_HYSTERESIS_REVIEW_TOLERANCE_DB,
    WING_EQ_MODEL_CONFLICT_BLOCKER,
    WING_EQ_FILTER_OPERATOR_OWNED_BLOCKER,
    WING_EQ_OPERATOR_OWNED_MODEL_BLOCKER,
    WING_EQ_MODEL_READBACK_BLOCKER,
    WING_EQ_MIX_BASELINE_PERCENT,
    WING_EQ_MIX_MAX_PERCENT_BY_SURFACE,
    WING_EQ_MIX_MIN_PERCENT,
    WING_EQ_REQUIRED_MODEL,
    WING_EQ_WRITE_KEY_SELECTION_POLICY,
    WING_EQ_WRITE_PARAMETER_KEYS,
    WING_FADER_MAX_DB,
    WING_FADER_MIN_DB,
    WING_GAIN_UP_FADER_CEILING_BLOCKER,
    WING_GAIN_UP_FADER_READBACK_BLOCKER,
    WING_GAIN_WRITE_PATH_SELECTION_POLICY,
    WING_GAIN_WRITE_PATH_SUFFIXES,
    WING_GAIN_OPERATOR_OWNED_BLOCKER,
    WING_FX_ACTIONABLE_PARAMETER_BLOCKER,
    WING_FX_MIX_MAX_PERCENT,
    WING_FX_MIX_MIN_PERCENT,
    WING_FX_LEVEL_OPERATOR_OWNED_BLOCKER,
    WING_FX_MODEL_PARAMETER_BLOCKER,
    WING_FX_OPERATOR_OWNED_MODEL_BLOCKER,
    WING_FX_MODEL_PARAMETER_PATH_PATTERN,
    WING_FX_POST_PULLBACK_MODEL_TIME_REVIEW_FIELDS,
    WING_FX_MODEL_SPECIFIC_PARAMETER_KEYS,
    WING_FX_RETURN_SLOT_BINDING_BLOCKER,
    WING_SOUNDCHECK_FX_MODEL_TIME_REVIEW_ONLY_FIELDS,
    WING_FX_SEND_LEVEL_MAX_DB,
    WING_FX_SEND_LEVEL_MIN_DB,
    WING_FX_SEND_SOURCE_SURFACE_BLOCKER,
    WING_GATE_ACCENT_MAX_PERCENT,
    WING_GATE_ACCENT_MIN_PERCENT,
    WING_GATE_ON_BLOCKER,
    WING_GATE_ON_MAX,
    WING_GATE_ON_MIN,
    WING_GATE_REQUIRED_MODEL,
    WING_GATE_SLOT_SINGLE_MODEL_COMBO_BLOCKER,
    WING_GATE_SLOT_SINGLE_MODEL_COMBO_POLICY,
    WING_GATE_SLOT_MODEL_CONFLICT_BLOCKER,
    WING_GATE_SLOT_MODEL_READBACK_BLOCKER,
    WING_GATE_SLOT_OPERATOR_OWNED_MODEL_BLOCKER,
    WING_GATE_SIDECHAIN_FREQ_MAX_HZ,
    WING_GATE_SIDECHAIN_FREQ_MIN_HZ,
    WING_GATE_SIDECHAIN_Q_MAX,
    WING_GATE_SIDECHAIN_Q_MIN,
    WING_GATE_SIDECHAIN_SOURCE_VALUES,
    WING_GATE_SIDECHAIN_TAP_VALUES,
    WING_GATE_SIDECHAIN_TYPE_VALUES,
    WING_GATE_THRESHOLD_PATH_SUFFIX_BY_SURFACE,
    WING_GATE_WRITE_KEY_SELECTION_POLICY,
    WING_GATE_WRITE_PARAMETER_KEYS,
    WING_INPUT_TRIM_MAX_DB,
    WING_INPUT_TRIM_MIN_DB,
    WING_MUTE_OFF_VALUE,
    WING_MUTE_ON_VALUE,
    WING_MUTE_OPERATOR_OWNED_BLOCKER,
    WING_MUTE_WRITE_KEY_SELECTION_POLICY,
    WING_MUTE_WRITE_PARAMETER_KEYS,
    WING_MUTE_VALUE_BLOCKER,
    WING_NATIVE_EQ_PARAMETRIC_BAND_COUNT_BY_SURFACE,
    WING_OSC_UNRESOLVED_TARGET_BLOCKER,
    WING_PAN_MAX_PERCENT,
    WING_PAN_WIDTH_OSC_PROTOCOL_SOURCE,
    WING_PAN_WIDTH_OSC_PROTOCOL_VERSION,
    WING_PAN_WIDTH_OPERATOR_OWNED_BLOCKER,
    WING_RACK_OSC_FIRMWARE,
    WING_RACK_OSC_TARGET_CONSOLE,
    WING_NATIVE_STRATEGY_FIT_MIN_ADVANTAGE,
    WING_NATIVE_STRATEGY_FIT_SCORE_TOLERANCE,
    WING_STRIP_LOW_CUT_TYPE_ALLOWED_VALUES_BY_SURFACE,
    WING_WIDTH_MIN_PERCENT,
    DSPStrategyCandidate,
    SmartDSPAnalysisInput,
    _candidate_wing_osc_tradeoff_summary,
    _future_readback_contract,
    _make_candidate,
    _select_candidate,
    _selection_policy,
    _with_wing_native_parameter_mappings,
    _wing_target_path_templates,
    analyze_smart_dsp_strategy,
    build_smart_dsp_strategy_proposals,
    classify_musical_role,
    classify_source_family,
    normalize_mode,
)
from backend.processing_proposal_adapters import build_proposal_only_method_contracts
from backend.processing_registry import get_pipeline_profile, get_processing_method


def _fresh(fields=12):
    return {
        "status": "fresh",
        "field_count": fields,
        "fresh_field_count": fields,
        "read_only": True,
        "live_write": False,
        "queries_console": False,
    }


def _analysis(**overrides):
    payload = {
        "target_id": "ch:5",
        "source_role": "lead_vocal",
        "source_label": "Lead Vox",
        "mode": "soundcheck",
        "problem_hints": ("muddy",),
        "operator_intent": "soundcheck lead vocal fast start",
        "source_identity_confirmed": True,
        "source_identity_confidence": 0.88,
        "evidence_confidence": 0.84,
        "current_state_cache_freshness": _fresh(),
        "current_state_readback_available": True,
        "rollback_plan_available": True,
        "true_peak_headroom_db": 5.0,
        "feedback_risk": 0.12,
    }
    payload.update(overrides)
    return SmartDSPAnalysisInput(**payload)


def _candidate_scores():
    return {
        "speed": 0.8,
        "precision": 0.8,
        "musical": 0.8,
        "safety": 0.8,
        "reversibility": 0.8,
        "operator_clarity": 0.8,
    }


def _wing_native_fit_candidate(
    *,
    strategy_id,
    method_family,
    total_score,
    native_supported,
    support_level,
    path_templates=(),
    actionable=(),
    apply_keys=(),
    unsupported=(),
    review_only=(),
    absolute=(),
    blockers=(),
    safety=0.8,
):
    parameters = {key: -2.0 for key in set(actionable) | set(apply_keys)}
    return DSPStrategyCandidate(
        strategy_id=strategy_id,
        method_family=method_family,
        problem="same_problem",
        parameters=parameters,
        scores={**_candidate_scores(), "safety": safety},
        total_score=total_score,
        confidence=0.8,
        reason="fit test candidate",
        operator_check="operator review before live promotion",
        rollback_expectation={
            "required_before_live_promotion": True,
            "live_write": False,
        },
        future_wing_readback_contract={
            "required_before_live_promotion": True,
            "requires_real_wing_proof": True,
            "requires_operator_approval": True,
            "live_authority_granted": False,
            "live_write": False,
            "allowed_live_write_scope_before_wing_proof": [],
            "wing_osc_applicability": {
                "target_console": WING_RACK_OSC_TARGET_CONSOLE,
                "firmware": WING_RACK_OSC_FIRMWARE,
                "protocol": "OSC UDP 2223",
                "applicable_corrections_policy": (
                    "Only corrections backed by WING Rack fw3.0.5 native OSC "
                    "paths may be considered WING-applicable; all other "
                    "corrections remain operator-review-only."
                ),
                "correction_exists_on_console_required": True,
                "correction_exists_on_console_policy": (
                    "wing_rack_fw_3_0_5_console_native_osc_corrections_only"
                ),
                "native_osc_required_for_wing_applicable_candidate": True,
                "osc_path_must_match_firmware": WING_RACK_OSC_FIRMWARE,
                "target_id": "ch:5",
                "target_surface_kind": "channel",
                "target_index_surface": "channel",
                "target_index": 5,
                "target_index_valid": True,
                "target_index_range": [1, 48],
                "target_requires_concrete_console_index": True,
                "native_osc_supported": native_supported,
                "native_osc_support_level": support_level,
                "native_osc_path_templates": list(path_templates),
                "unsupported_parameter_keys": list(unsupported),
                "review_only_parameter_keys": list(review_only),
                "actionable_parameter_keys": list(actionable),
                "absolute_value_readback_parameter_keys": list(absolute),
                "direct_apply_parameter_keys": list(apply_keys),
                "requires_real_wing_readback": True,
                "requires_operator_approval": True,
                "live_authority_granted": False,
                "live_write": False,
                "allowed_live_write_scope_before_wing_proof": [],
            },
        },
        blockers=tuple(blockers),
    )


def _assert_weighted_score_summary(summary, *, mode, expected_total_score=None):
    expected_weights = (
        LIVE_STRATEGY_SCORE_WEIGHTS
        if mode == "live"
        else SOUNDCHECK_STRATEGY_SCORE_WEIGHTS
    )
    assert summary["mode"] == mode
    assert summary["score_weight_profile"] == f"{mode}_weighted_strategy_search"
    assert set(item["metric"] for item in summary["weighted_score_contributions"]) == (
        set(expected_weights)
    )
    for item in summary["weighted_score_contributions"]:
        assert item["weight"] == expected_weights[item["metric"]]
        assert item["weighted_contribution"] == round(
            item["score"] * item["weight"],
            3,
        )
    expected_after_blockers = round(
        max(
            0.0,
            min(
                1.0,
                summary["weighted_score_before_blockers"]
                - summary["blocker_score_penalty"],
            ),
        ),
        3,
    )
    assert summary["weighted_score_after_blockers"] == expected_after_blockers
    assert summary["weighted_score_matches_total"] is True
    if expected_total_score is not None:
        assert summary["total_score"] == expected_total_score


def _assert_dsp_parameter_calculation_summary(
    summary,
    *,
    strategy_id=None,
    method_family=None,
    expect_calculated=True,
):
    if strategy_id is not None:
        assert summary["strategy_id"] == strategy_id
    if method_family is not None:
        assert summary["method_family"] == method_family
    assert summary["parameter_status"]
    assert isinstance(summary["calculated_parameter_keys"], list)
    assert isinstance(summary["calculated_parameter_values"], dict)
    assert set(summary["calculated_parameter_values"]) == set(
        summary["calculated_parameter_keys"]
    )
    assert isinstance(summary["calculation_source_keys"], list)
    assert isinstance(summary["calculation_source_values"], dict)
    assert set(summary["calculation_source_values"]).issubset(
        set(summary["calculation_source_keys"])
    )
    assert isinstance(summary["wing_apply_parameter_keys"], list)
    assert isinstance(summary["wing_review_only_or_readback_parameter_keys"], list)
    assert 0.0 <= summary["wing_native_strategy_fit_score"] <= 1.0
    assert summary["wing_native_strategy_fit_status"]
    assert summary["proposal_only"] is True
    assert summary["readback_required_before_promotion"] is True
    assert summary["rollback_required_before_promotion"] is True
    assert summary["live_write"] is False
    assert summary["live_authority_granted"] is False
    if expect_calculated:
        assert summary["has_real_dsp_parameter_calculation"] is True
        assert (
            summary["calculated_parameter_keys"]
            or summary["wing_apply_parameter_keys"]
            or summary["method_family"] in {"analysis", "analysis_only"}
        )


def _assert_problem_action_decision_summary(
    summary,
    *,
    strategy_id=None,
    method_family=None,
    expected_action_intent=None,
    expected_method_families=(),
):
    assert summary["schema"] == "smart_dsp.problem_action_decision.v1"
    assert summary["decision_policy"] == "problem_first_strategy_search_not_fixed_method"
    if strategy_id is not None:
        assert summary["selected_strategy_id"] == strategy_id
        assert summary["selected"]["strategy_id"] == strategy_id
    if method_family is not None:
        assert summary["selected_method_family"] == method_family
        assert summary["selected"]["method_family"] == method_family
    if expected_action_intent is not None:
        assert summary["selected_action_intent"] == expected_action_intent
        assert summary["selected"]["action_intent"] == expected_action_intent
    for family in expected_method_families:
        assert family in summary["considered_method_families"]
    assert summary["considered_strategy_count"] >= 1
    assert summary["compared_alternative_count"] >= 0
    assert isinstance(summary["considered_action_intents"], list)
    assert isinstance(summary["considered_problem_variants"], list)
    assert isinstance(summary["method_family_score_leaders"], list)
    assert summary["selection_basis"]
    assert summary["operator_review_required_before_live_apply"] is True
    assert summary["readback_required_before_live_promotion"] is True
    assert summary["rollback_required_before_live_promotion"] is True
    assert summary["proposal_only"] is True
    assert summary["live_write"] is False
    assert summary["live_authority_granted"] is False


def _assert_proposal_denial(payload):
    assert payload["method_id"] == SMART_DSP_METHOD_ID
    assert payload["queue_action"] == "not_queueable_until_promoted"
    assert payload["can_import_to_queue"] is False
    assert payload["can_apply_live"] is False
    assert payload["queue_import_allowed"] is False
    assert payload["live_apply_allowed"] is False
    assert payload["read_only_or_dry_run_only"] is True
    assert payload["live_authority_granted"] is False
    assert payload["readiness_closure_allowed"] is False
    assert payload["live_write"] is False
    assert payload["queue_mutation"] is False
    assert payload["mutates_mixer"] is False
    assert "smart_dsp_live_apply_quarantined" in payload["safety_blockers"]
    assert "real_wing_proof_required_before_live_apply" in payload["safety_blockers"]
    correction = payload["operator_correction"]
    assert correction["read_only_or_dry_run_only"] is True
    assert correction["live_apply_allowed"] is False
    assert correction["queue_import_allowed"] is False
    assert correction["future_wing_readback_contract"][
        "required_before_live_promotion"
    ] is True


def test_strategy_score_weights_shift_between_soundcheck_and_live():
    fast_precise = {
        "speed": 0.98,
        "precision": 0.95,
        "musical": 0.95,
        "safety": 0.55,
        "reversibility": 0.55,
        "operator_clarity": 0.84,
    }
    safer_reversible = {
        "speed": 0.55,
        "precision": 0.7,
        "musical": 0.68,
        "safety": 0.95,
        "reversibility": 0.9,
        "operator_clarity": 0.82,
    }

    soundcheck_fast = _make_candidate(
        strategy_id="fast_precise_soundcheck_probe",
        method_family="eq",
        problem="weighting_probe",
        parameters={},
        scores=fast_precise,
        reason="Probe fast precise scoring in soundcheck.",
        operator_check="No mixer write; scoring probe only.",
        analysis=_analysis(mode="soundcheck"),
    )
    soundcheck_safe = _make_candidate(
        strategy_id="safe_reversible_soundcheck_probe",
        method_family="eq",
        problem="weighting_probe",
        parameters={},
        scores=safer_reversible,
        reason="Probe safe reversible scoring in soundcheck.",
        operator_check="No mixer write; scoring probe only.",
        analysis=_analysis(mode="soundcheck"),
    )
    live_fast = _make_candidate(
        strategy_id="fast_precise_live_probe",
        method_family="eq",
        problem="weighting_probe",
        parameters={},
        scores=fast_precise,
        reason="Probe fast precise scoring in live mode.",
        operator_check="No mixer write; scoring probe only.",
        analysis=_analysis(mode="live"),
    )
    live_safe = _make_candidate(
        strategy_id="safe_reversible_live_probe",
        method_family="eq",
        problem="weighting_probe",
        parameters={},
        scores=safer_reversible,
        reason="Probe safe reversible scoring in live mode.",
        operator_check="No mixer write; scoring probe only.",
        analysis=_analysis(mode="live"),
    )

    assert soundcheck_fast.total_score > soundcheck_safe.total_score
    assert live_safe.total_score > live_fast.total_score


def _assert_common_readback_contract(contract):
    assert contract["required"] is True
    assert contract["required_before_live_promotion"] is True
    assert contract["requires_real_wing_proof"] is True
    assert contract["requires_operator_approval"] is True
    assert contract["requires_pre_promotion_dry_run"] is True
    assert contract["fader_ceiling_dbfs"] == 0.0
    assert contract["no_routing_scene_snapshot_write"] is True
    assert contract["allowed_live_write_scope_before_wing_proof"] == []
    assert contract["live_write"] is False
    wing_osc = contract["wing_osc_applicability"]
    assert wing_osc["target_console"] == "Behringer WING Rack"
    assert wing_osc["firmware"] == "3.0.5"
    assert wing_osc["protocol"] == "OSC UDP 2223"
    assert wing_osc["protocol_reference"] == (
        "Docs/manuals/wing/WING Remote Protocols v3.0.5.pdf"
    )
    assert wing_osc["correction_exists_on_console_required"] is True
    assert wing_osc["correction_exists_on_console_policy"] == (
        "wing_rack_fw_3_0_5_console_native_osc_corrections_only"
    )
    assert wing_osc["native_osc_required_for_wing_applicable_candidate"] is True
    assert wing_osc["osc_path_must_match_firmware"] == "3.0.5"
    assert wing_osc["requires_real_wing_readback"] is True
    assert wing_osc["requires_operator_approval"] is True
    assert wing_osc["live_authority_granted"] is False
    assert wing_osc["live_write"] is False
    assert wing_osc["allowed_live_write_scope_before_wing_proof"] == []
    assert "Only corrections backed by WING Rack fw3.0.5 native OSC paths" in (
        wing_osc["applicable_corrections_policy"]
    )


def _assert_tradeoff_wing_osc_summary(
    row,
    *,
    native_supported=None,
    support_level=None,
):
    assert row["wing_osc_target_console"] == "Behringer WING Rack"
    assert row["wing_osc_firmware"] == "3.0.5"
    assert row["wing_osc_protocol"] == "OSC UDP 2223"
    assert "Only corrections backed by WING Rack fw3.0.5 native OSC paths" in (
        row["wing_osc_applicable_corrections_policy"]
    )
    assert row["wing_osc_correction_exists_on_console_required"] is True
    assert row["wing_osc_correction_exists_on_console_policy"] == (
        "wing_rack_fw_3_0_5_console_native_osc_corrections_only"
    )
    assert row["wing_osc_native_osc_required_for_wing_applicable_candidate"] is True
    assert row["wing_osc_path_must_match_firmware"] == "3.0.5"
    assert isinstance(row["wing_osc_target_id"], str)
    assert isinstance(row["wing_osc_target_surface_kind"], str)
    assert isinstance(row["wing_osc_target_index_surface"], str)
    assert row["wing_osc_target_index"] is None or isinstance(
        row["wing_osc_target_index"], int
    )
    assert isinstance(row["wing_osc_target_index_valid"], bool)
    assert isinstance(row["wing_osc_target_index_range"], list)
    if row["wing_osc_target_index_range"]:
        assert len(row["wing_osc_target_index_range"]) == 2
    assert isinstance(row["wing_osc_target_requires_concrete_console_index"], bool)
    if row["wing_osc_native_supported"]:
        assert row["wing_osc_target_index_valid"] is True
    assert row["wing_osc_requires_real_wing_readback"] is True
    assert row["wing_osc_requires_operator_approval"] is True
    assert row["wing_osc_live_authority_granted"] is False
    assert row["wing_osc_live_write"] is False
    assert row["wing_osc_allowed_live_write_scope_before_wing_proof"] == []
    assert isinstance(row["wing_osc_path_templates"], list)
    assert row["wing_osc_path_template_count"] == len(row["wing_osc_path_templates"])
    assert isinstance(row["wing_osc_unsupported_parameter_keys"], list)
    assert isinstance(row["wing_osc_review_only_parameter_keys"], list)
    assert isinstance(row["wing_osc_review_only_parameter_values"], dict)
    assert set(row["wing_osc_review_only_parameter_values"]) == set(
        row["wing_osc_review_only_parameter_keys"]
    )
    assert isinstance(row["wing_osc_review_only_parameter_reason_by_key"], dict)
    assert isinstance(row["wing_osc_actionable_parameter_keys"], list)
    assert row["wing_osc_actionable_parameter_count"] == len(
        row["wing_osc_actionable_parameter_keys"]
    )
    assert isinstance(row["wing_osc_absolute_value_readback_parameter_keys"], list)
    assert isinstance(row["wing_osc_absolute_value_readback_parameter_values"], dict)
    assert set(row["wing_osc_absolute_value_readback_parameter_values"]) == set(
        row["wing_osc_absolute_value_readback_parameter_keys"]
    )
    assert row["wing_osc_absolute_value_readback_parameter_count"] == len(
        row["wing_osc_absolute_value_readback_parameter_keys"]
    )
    assert isinstance(row["wing_osc_absolute_value_readback_parameter_reason_by_key"], dict)
    assert row["wing_osc_absolute_value_readback_required_for_apply"] is bool(
        row["wing_osc_absolute_value_readback_parameter_keys"]
    )
    assert set(row["wing_osc_apply_parameter_keys"]).issubset(
        set(row["wing_osc_actionable_parameter_keys"])
    )
    assert not (
        set(row["wing_osc_apply_parameter_keys"])
        & set(row["wing_osc_absolute_value_readback_parameter_keys"])
    )
    assert isinstance(row["wing_osc_apply_parameter_values"], dict)
    assert set(row["wing_osc_apply_parameter_values"]) == set(
        row["wing_osc_apply_parameter_keys"]
    )
    assert not (
        set(row["wing_osc_apply_parameter_values"])
        & set(row["wing_osc_review_only_parameter_keys"])
    )
    assert row["wing_osc_apply_parameter_count"] == len(
        row["wing_osc_apply_parameter_keys"]
    )
    assert row["wing_osc_correction_apply_scope"] == (
        "direct_wing_fw_3_0_5_osc_values_only_after_required_readback"
    )
    assert row["wing_osc_review_only_parameters_are_apply_values"] is False
    if row["wing_osc_review_only_parameter_keys"]:
        assert row["wing_osc_review_only_parameters_apply_ready"] is False
    else:
        assert row["wing_osc_review_only_parameters_apply_ready"] is None
    assert row["wing_osc_candidate_apply_status"]
    assert isinstance(row["wing_native_strategy_fit"], dict)
    fit = row["wing_native_strategy_fit"]
    assert row["wing_native_strategy_fit_score"] == fit["fit_score"]
    assert row["wing_native_strategy_fit_status"] == fit["fit_status"]
    assert row["wing_native_strategy_fit_preferable"] == (
        fit["selection_preferable_for_fast_soundcheck_or_live"]
    )
    assert 0.0 <= fit["fit_score"] <= 1.0
    assert isinstance(fit["fit_status"], str)
    assert isinstance(fit["path_template_count"], int)
    assert isinstance(fit["apply_parameter_count"], int)
    assert isinstance(fit["readback_burden_score"], int)
    if native_supported is not None:
        assert row["wing_osc_native_supported"] is native_supported
    if support_level is not None:
        assert row["wing_osc_support_level"] == support_level


def _assert_selected_wing_osc_readback_summary(
    row,
    *,
    native_supported=None,
    expected_before_paths=None,
    expected_templates_include=(),
):
    assert row["wing_osc_readback_target_console"] == "Behringer WING Rack"
    assert row["wing_osc_readback_firmware"] == "3.0.5"
    assert row["wing_osc_readback_protocol"] == "OSC UDP 2223"
    assert row["wing_osc_readback_correction_exists_on_console_required"] is True
    assert row["wing_osc_readback_correction_exists_on_console_policy"] == (
        "wing_rack_fw_3_0_5_console_native_osc_corrections_only"
    )
    assert (
        row["wing_osc_readback_native_osc_required_for_wing_applicable_candidate"]
        is True
    )
    assert row["wing_osc_readback_path_must_match_firmware"] == "3.0.5"
    assert isinstance(row["wing_osc_readback_target_id"], str)
    assert isinstance(row["wing_osc_readback_target_surface_kind"], str)
    assert isinstance(row["wing_osc_readback_target_index_surface"], str)
    assert row["wing_osc_readback_target_index"] is None or isinstance(
        row["wing_osc_readback_target_index"], int
    )
    assert isinstance(row["wing_osc_readback_target_index_valid"], bool)
    assert isinstance(row["wing_osc_readback_target_index_range"], list)
    if row["wing_osc_readback_target_index_range"]:
        assert len(row["wing_osc_readback_target_index_range"]) == 2
    assert isinstance(
        row["wing_osc_readback_target_requires_concrete_console_index"],
        bool,
    )
    if row["wing_osc_readback_native_supported"]:
        assert row["wing_osc_readback_target_index_valid"] is True
    assert row["wing_osc_readback_required_before_live_promotion"] is True
    assert row["wing_osc_readback_requires_real_wing_proof"] is True
    assert row["wing_osc_readback_requires_operator_approval"] is True
    assert row["wing_osc_readback_live_authority_granted"] is False
    assert row["wing_osc_readback_live_write"] is False
    assert row["wing_osc_readback_allowed_live_write_scope_before_wing_proof"] == []
    assert isinstance(row["wing_osc_readback_actionable_parameter_keys"], list)
    assert row["wing_osc_readback_actionable_parameter_count"] == len(
        row["wing_osc_readback_actionable_parameter_keys"]
    )
    assert isinstance(row["wing_osc_readback_absolute_value_parameter_keys"], list)
    assert isinstance(row["wing_osc_readback_absolute_value_parameter_values"], dict)
    assert set(row["wing_osc_readback_absolute_value_parameter_values"]) == set(
        row["wing_osc_readback_absolute_value_parameter_keys"]
    )
    assert row["wing_osc_readback_absolute_value_parameter_count"] == len(
        row["wing_osc_readback_absolute_value_parameter_keys"]
    )
    assert isinstance(row["wing_osc_readback_absolute_value_parameter_reason_by_key"], dict)
    assert row["wing_osc_readback_absolute_value_required_for_apply"] is bool(
        row["wing_osc_readback_absolute_value_parameter_keys"]
    )
    assert set(row["wing_osc_readback_apply_parameter_keys"]).issubset(
        set(row["wing_osc_readback_actionable_parameter_keys"])
    )
    assert not (
        set(row["wing_osc_readback_apply_parameter_keys"])
        & set(row["wing_osc_readback_absolute_value_parameter_keys"])
    )
    assert isinstance(row["wing_osc_readback_apply_parameter_values"], dict)
    assert set(row["wing_osc_readback_apply_parameter_values"]) == set(
        row["wing_osc_readback_apply_parameter_keys"]
    )
    assert not (
        set(row["wing_osc_readback_apply_parameter_values"])
        & set(row["wing_osc_readback_review_only_parameter_keys"])
    )
    assert row["wing_osc_readback_apply_parameter_count"] == len(
        row["wing_osc_readback_apply_parameter_keys"]
    )
    assert isinstance(row["wing_osc_readback_review_only_parameter_keys"], list)
    assert isinstance(row["wing_osc_readback_review_only_parameter_values"], dict)
    assert set(row["wing_osc_readback_review_only_parameter_values"]) == set(
        row["wing_osc_readback_review_only_parameter_keys"]
    )
    assert isinstance(
        row["wing_osc_readback_review_only_parameter_reason_by_key"],
        dict,
    )
    assert row["wing_osc_readback_review_only_parameters_are_apply_values"] is False
    assert row["wing_osc_readback_correction_apply_scope"] == (
        "direct_wing_fw_3_0_5_osc_values_only_after_required_readback"
    )
    assert isinstance(row["wing_osc_readback_sections"], list)
    assert isinstance(row["wing_osc_read_before_paths"], list)
    assert isinstance(row["wing_osc_read_after_paths"], list)
    assert isinstance(row["wing_osc_readback_path_templates"], list)
    assert row["wing_osc_readback_path_template_count"] == len(
        row["wing_osc_readback_path_templates"]
    )
    assert row["wing_osc_readback_status"]
    if native_supported is not None:
        assert row["wing_osc_readback_native_supported"] is native_supported
    if expected_before_paths is not None:
        assert row["wing_osc_read_before_paths"] == expected_before_paths
        assert row["wing_osc_read_after_paths"] == expected_before_paths
    for template in expected_templates_include:
        assert template in row["wing_osc_readback_path_templates"]


def test_wing_native_strategy_fit_can_select_close_score_native_candidate():
    leader = _wing_native_fit_candidate(
        strategy_id="precise_external_processing",
        method_family="external_processing",
        total_score=0.82,
        native_supported=False,
        support_level="external_processing_mapping_required",
    )
    native = _wing_native_fit_candidate(
        strategy_id="native_fast_osc_method",
        method_family="native_channel_processing",
        total_score=0.77,
        native_supported=True,
        support_level="native_osc_path_templates_declared",
        path_templates=("/ch/{channel}/native/fast",),
        actionable=("wing_native_fast_value",),
        apply_keys=("wing_native_fast_value",),
    )

    selected = _select_candidate(
        [leader, native],
        [],
        _analysis(),
        "soundcheck",
    )

    assert selected.strategy_id == "native_fast_osc_method"
    policy = _selection_policy(
        selected=selected,
        alternatives=(leader,),
        analysis=_analysis(),
        mode="soundcheck",
        blockers=(),
    )
    assert policy == "soundcheck_wing_native_strategy_fit_priority"
    leader_summary = _candidate_wing_osc_tradeoff_summary(leader)
    native_summary = _candidate_wing_osc_tradeoff_summary(native)
    assert native_summary["wing_native_strategy_fit_preferable"] is True
    assert leader_summary["wing_native_strategy_fit_preferable"] is False
    assert (
        native_summary["wing_native_strategy_fit_score"]
        - leader_summary["wing_native_strategy_fit_score"]
    ) >= WING_NATIVE_STRATEGY_FIT_MIN_ADVANTAGE["soundcheck"]
    assert (
        leader.total_score - native.total_score
    ) <= WING_NATIVE_STRATEGY_FIT_SCORE_TOLERANCE["soundcheck"]


def _assert_gain_delta_readback_contract(
    strategy,
    *,
    source,
    target_surface,
    expected_delta,
    path_prefix,
):
    params = strategy.parameters
    assert params["wing_gain_delta_source"] == source
    assert params["wing_gain_target_surface"] == target_surface
    assert params["wing_gain_delta_db"] == expected_delta
    assert params["wing_gain_primary_osc_path_suffix"] == "fdr"
    assert params["wing_gain_secondary_osc_path_suffix"] == "in/set/trim"
    assert params["wing_gain_candidate_write_path_suffixes"] == list(
        WING_GAIN_WRITE_PATH_SUFFIXES
    )
    assert params["wing_gain_write_path_selection_policy"] == (
        WING_GAIN_WRITE_PATH_SELECTION_POLICY
    )
    assert params["wing_gain_apply_requires_single_selected_write_path"] is True
    assert (
        params["wing_gain_apply_requires_current_readback_before_path_selection"]
        is True
    )
    assert params["wing_gain_pre_readback_write_path_suffixes_forbidden"] == list(
        WING_GAIN_WRITE_PATH_SUFFIXES
    )
    assert params["wing_gain_unselected_write_path_suffixes_forbidden"] is True
    assert params["wing_gain_selected_write_path_suffix"] is None
    assert params["wing_gain_delta_value_type"] == (
        "relative_db_delta_requires_readback_to_resolve_absolute_value"
    )
    assert params["wing_gain_absolute_value_requires_readback"] is True
    assert params["wing_gain_no_absolute_osc_value_without_readback"] is True

    gain_readback = strategy.future_wing_readback_contract["gain_fader_readback"]
    assert gain_readback["read_before_paths"] == ["fdr", "in/set/trim"]
    assert gain_readback["read_after_paths"] == ["fdr", "in/set/trim"]
    assert f"{path_prefix}/fdr" in gain_readback["wing_osc_path_templates"]
    assert f"{path_prefix}/in/set/trim" in gain_readback["wing_osc_path_templates"]
    assert gain_readback["wing_gain_delta_mapping_required"] is True
    assert gain_readback["wing_gain_delta_db"] == expected_delta
    assert gain_readback["wing_gain_delta_abs_db"] == abs(expected_delta)
    assert gain_readback["wing_gain_primary_osc_path_suffix"] == "fdr"
    assert gain_readback["wing_gain_secondary_osc_path_suffix"] == "in/set/trim"
    assert gain_readback["wing_gain_candidate_write_path_suffixes"] == list(
        WING_GAIN_WRITE_PATH_SUFFIXES
    )
    assert gain_readback["wing_gain_write_path_selection_policy"] == (
        WING_GAIN_WRITE_PATH_SELECTION_POLICY
    )
    assert gain_readback["wing_gain_apply_requires_single_selected_write_path"] is True
    assert (
        gain_readback[
            "wing_gain_apply_requires_current_readback_before_path_selection"
        ]
        is True
    )
    assert gain_readback[
        "wing_gain_pre_readback_write_path_suffixes_forbidden"
    ] == list(WING_GAIN_WRITE_PATH_SUFFIXES)
    assert gain_readback["wing_gain_unselected_write_path_suffixes_forbidden"] is True
    assert gain_readback["wing_gain_selected_write_path_suffix"] is None
    assert gain_readback["wing_fader_min_db"] == WING_FADER_MIN_DB
    assert gain_readback["wing_fader_max_db"] == WING_FADER_MAX_DB
    assert gain_readback["wing_input_trim_min_db"] == WING_INPUT_TRIM_MIN_DB
    assert gain_readback["wing_input_trim_max_db"] == WING_INPUT_TRIM_MAX_DB
    assert gain_readback["absolute_gain_value_requires_current_readback"] is True
    assert gain_readback["wing_gain_no_absolute_osc_value_without_readback"] is True


def _assert_master_output_target_override(strategy, *, original_target_id):
    params = strategy.parameters
    gain_readback = strategy.future_wing_readback_contract["gain_fader_readback"]
    expected_fields = {
        "wing_osc_original_target_id": original_target_id,
        "wing_osc_effective_target_id": "main",
        "wing_osc_target_id_override": "main",
        "soundcheck_master_output_source_triggered": True,
        "soundcheck_master_output_target_override_reason": (
            "master_output_problem_detected_from_source_context"
        ),
    }
    for field, expected in expected_fields.items():
        assert params[field] == expected
        assert gain_readback[field] == expected
    assert "/ch/{channel}/fdr" not in gain_readback["wing_osc_path_templates"]
    assert "/main/{main}/fdr" in gain_readback["wing_osc_path_templates"]


def _assert_soundcheck_master_output_guard(
    strategy,
    *,
    move_kind,
    followup_review,
):
    params = strategy.parameters
    gain_readback = strategy.future_wing_readback_contract["gain_fader_readback"]
    expected_fields = {
        "soundcheck_master_output_bounded_move_proposal": True,
        "soundcheck_master_output_native_gain_osc_only_required": True,
        "soundcheck_master_output_target_surface": "main",
        "soundcheck_master_output_wing_apply_parameter_scope": (
            "main_fdr_or_input_trim_relative_delta_only"
        ),
        "soundcheck_master_output_max_pullback_db": (
            SOUNDCHECK_MASTER_MAX_PULLBACK_DB
        ),
        "soundcheck_master_output_max_lift_db": SOUNDCHECK_MASTER_MAX_LIFT_DB,
        "soundcheck_master_output_min_headroom_margin_db": (
            SOUNDCHECK_MASTER_MIN_HEADROOM_MARGIN_DB
        ),
        "soundcheck_master_output_move_kind": move_kind,
        "soundcheck_master_output_pa_feed_identity_required": True,
        "soundcheck_master_output_true_peak_recheck_required": True,
        "soundcheck_master_output_feedback_recheck_required": True,
        "soundcheck_master_output_summed_bus_recheck_required": True,
        "soundcheck_master_output_readback_required": True,
        "soundcheck_master_output_operator_confirmation_required": True,
        "soundcheck_master_output_rollback_ready": True,
        "soundcheck_master_output_no_limiter_compression_or_eq_write_required": True,
        "soundcheck_master_output_followup_processing_review_required": (
            followup_review
        ),
        "soundcheck_master_output_live_promotion_status": (
            "blocked_until_real_wing_main_gain_readback_pa_feed_identity_true_peak_headroom_feedback_recheck_and_operator_approval"
        ),
        "soundcheck_master_output_live_apply_blocked_without_wing_proof": True,
    }
    for field, expected in expected_fields.items():
        assert params[field] == expected
        assert gain_readback[field] == expected
    assert (
        params["soundcheck_master_output_processing_review_only_fields"]
        == gain_readback["soundcheck_master_output_processing_review_only_fields"]
    )
    assert "limiter_ceiling_dbtp" in params[
        "soundcheck_master_output_processing_review_only_fields"
    ]


def _assert_live_master_true_peak_output_guard(strategy):
    params = strategy.parameters
    gain_readback = strategy.future_wing_readback_contract["gain_fader_readback"]
    expected_fields = {
        "live_master_true_peak_pullback_proposal": True,
        "live_master_true_peak_native_gain_osc_only_required": True,
        "live_master_true_peak_target_surface": "main",
        "live_master_true_peak_wing_apply_parameter_scope": (
            "main_fdr_or_input_trim_relative_delta_only"
        ),
        "live_master_true_peak_move_kind": "pullback",
        "live_master_true_peak_pa_feed_identity_required": True,
        "live_master_true_peak_previous_gain_readback_required": True,
        "live_master_true_peak_true_peak_recheck_required": True,
        "live_master_true_peak_feedback_recheck_required": True,
        "live_master_true_peak_summed_bus_recheck_required": True,
        "live_master_true_peak_readback_required": True,
        "live_master_true_peak_rollback_ready": True,
        "live_master_true_peak_operator_confirmation_required": True,
        "live_master_true_peak_no_limiter_compression_or_eq_write_required": True,
        "live_master_true_peak_live_promotion_status": (
            "blocked_until_real_wing_main_gain_readback_pa_feed_identity_true_peak_headroom_feedback_recheck_and_operator_approval"
        ),
        "live_master_true_peak_live_apply_blocked_without_wing_proof": True,
        "forbidden_without_wing_proof": (
            "live_master_true_peak_main_gain_limiter_eq_compression"
        ),
    }
    for field, expected in expected_fields.items():
        assert params[field] == expected
        assert gain_readback[field] == expected
    assert params["live_stepped_change_required"] is True
    assert params["live_readback_after_each_step_required"] is True
    assert params["live_max_single_step_db"] == LIVE_MAX_SINGLE_STEP_DB
    assert (
        params["live_master_true_peak_processing_review_only_fields"]
        == gain_readback["live_master_true_peak_processing_review_only_fields"]
    )
    assert "limiter_ceiling_dbtp" in params[
        "live_master_true_peak_processing_review_only_fields"
    ]


def _assert_master_output_processing_followup_review_contract(
    strategy,
    *,
    expected_values,
    expected_native_apply_fields=("wing_gain_delta_db",),
):
    gain_readback = strategy.future_wing_readback_contract["gain_fader_readback"]
    assert (
        gain_readback["wing_master_output_processing_followup_review_required"]
        is True
    )
    assert (
        gain_readback["wing_master_output_processing_followup_apply_ready"]
        is False
    )
    assert (
        gain_readback[
            "wing_master_output_processing_followup_live_apply_blocked_without_wing_proof"
        ]
        is True
    )
    assert (
        gain_readback[
            "wing_master_output_processing_followup_after_native_gain_move_required"
        ]
        is True
    )
    assert (
        gain_readback[
            "wing_master_output_processing_followup_requires_real_wing_processing_readback"
        ]
        is True
    )
    assert gain_readback[
        "wing_master_output_processing_followup_review_only_fields"
    ] == list(WING_SOUNDCHECK_MASTER_OUTPUT_PROCESSING_REVIEW_ONLY_FIELDS)
    assert gain_readback["wing_master_output_processing_followup_values"] == (
        expected_values
    )
    assert gain_readback[
        "wing_master_output_processing_followup_forbidden_apply_fields"
    ] == sorted(expected_values)
    assert gain_readback[
        "wing_master_output_processing_followup_native_gain_apply_fields"
    ] == list(expected_native_apply_fields)
    forbidden_paths = gain_readback[
        "wing_master_output_processing_followup_forbidden_path_patterns"
    ]
    assert "/main/{main}/dyn/*" in forbidden_paths
    assert "/main/{main}/eq/*" in forbidden_paths
    assert "main/limiter" in forbidden_paths
    assert "/main/{main}/dyn/*" not in gain_readback["wing_osc_path_templates"]
    assert "/main/{main}/eq/*" not in gain_readback["wing_osc_path_templates"]
    assert gain_readback[
        "wing_master_output_processing_followup_control_surface"
    ] == "operator_console_ui_only_until_verified_main_processing_osc_mapping"
    assert gain_readback["wing_master_output_processing_followup_path_source"] == (
        "no_selected_apply_path_without_verified_wing_fw_3_0_5_main_processing_readback"
    )
    assert "WING Rack fw3.0.5" in gain_readback[
        "wing_master_output_processing_followup_operator_check"
    ]
    assert "do not write main EQ" in gain_readback[
        "wing_master_output_processing_followup_operator_check"
    ]


def _assert_soundcheck_master_processing_guard(
    strategy,
    *,
    processing_types,
):
    params = strategy.parameters
    if strategy.method_family == "eq":
        processing_readback = strategy.future_wing_readback_contract[
            "eq_band_readback"
        ]
    else:
        processing_readback = strategy.future_wing_readback_contract[
            "dynamics_readback"
        ]
    expected_fields = {
        "soundcheck_master_processing_review_only_proposal": True,
        "soundcheck_master_processing_native_osc_only_required": True,
        "soundcheck_master_processing_target_surface": "main",
        "soundcheck_master_processing_types": list(processing_types),
        "soundcheck_master_processing_wing_apply_parameter_scope": (
            "main_eq_or_dyn_native_osc_review_only_no_scene_routing"
        ),
        "soundcheck_master_processing_pa_feed_identity_required": True,
        "soundcheck_master_processing_true_peak_headroom_recheck_required": True,
        "soundcheck_master_processing_feedback_recheck_required": True,
        "soundcheck_master_processing_bypass_level_match_required": True,
        "soundcheck_master_processing_readback_required": True,
        "soundcheck_master_processing_operator_confirmation_required": True,
        "soundcheck_master_processing_rollback_ready": True,
        "soundcheck_master_processing_no_routing_scene_snapshot_write_required": True,
        "soundcheck_master_processing_live_promotion_status": (
            "blocked_until_real_wing_main_eq_or_dyn_readback_pa_feed_identity_bypass_level_match_headroom_feedback_recheck_and_operator_approval"
        ),
        "soundcheck_master_processing_live_apply_blocked_without_wing_proof": True,
    }
    if "broad_eq" in processing_types:
        expected_fields.update(
            {
                "soundcheck_master_processing_eq_cut_only_required": True,
                "soundcheck_master_processing_eq_max_cut_db": (
                    SOUNDCHECK_MASTER_EQ_MAX_CUT_DB
                ),
                "soundcheck_master_processing_eq_max_boost_db": (
                    SOUNDCHECK_MASTER_EQ_MAX_BOOST_DB
                ),
            }
        )
    if "bus_compression" in processing_types:
        expected_fields[
            "soundcheck_master_processing_bus_compression_max_target_gr_db"
        ] = SOUNDCHECK_MASTER_BUS_COMPRESSION_MAX_TARGET_GR_DB
    for field, expected in expected_fields.items():
        assert params[field] == expected
        assert processing_readback[field] == expected


def _assert_selected_gain_delta_apply_values(recommendation, expected_delta):
    proposal = recommendation.to_operator_proposal()
    expected_readback_values = {"wing_gain_delta_db": expected_delta}
    assert proposal["selected_wing_osc_summary"][
        "wing_osc_apply_parameter_values"
    ] == {}
    assert proposal["selected_wing_osc_readback_summary"][
        "wing_osc_readback_apply_parameter_values"
    ] == {}
    assert proposal["selected_wing_osc_summary"][
        "wing_osc_absolute_value_readback_parameter_values"
    ] == expected_readback_values
    assert proposal["selected_wing_osc_readback_summary"][
        "wing_osc_readback_absolute_value_parameter_values"
    ] == expected_readback_values
    assert proposal["selected_wing_osc_summary"][
        "wing_osc_candidate_apply_status"
    ] == "wing_native_osc_candidate_requires_absolute_value_readback_before_apply"
    assert proposal["live_apply_allowed"] is False
    assert proposal["live_authority_granted"] is False
    assert proposal["live_write"] is False


def _assert_selected_mute_apply_values(recommendation, expected_value):
    proposal = recommendation.to_operator_proposal()
    expected_values = {"wing_mute_osc_value": expected_value}
    assert proposal["selected_wing_osc_summary"][
        "wing_osc_apply_parameter_values"
    ] == expected_values
    assert proposal["selected_wing_osc_readback_summary"][
        "wing_osc_readback_apply_parameter_values"
    ] == expected_values
    assert proposal["live_apply_allowed"] is False
    assert proposal["live_authority_granted"] is False
    assert proposal["live_write"] is False


def _assert_selected_eq_apply_values(recommendation, expected_values):
    proposal = recommendation.to_operator_proposal()
    assert proposal["selected_wing_osc_summary"][
        "wing_osc_apply_parameter_values"
    ] == expected_values
    assert proposal["selected_wing_osc_readback_summary"][
        "wing_osc_readback_apply_parameter_values"
    ] == expected_values
    assert proposal["live_apply_allowed"] is False
    assert proposal["live_authority_granted"] is False
    assert proposal["live_write"] is False


def _assert_eq_write_key_contract(strategy, *, selected_write_keys):
    params = strategy.parameters
    readback = strategy.future_wing_readback_contract["eq_band_readback"]
    mapping = readback["wing_eq_value_mapping"]
    forbidden_write_keys = [
        key for key in WING_EQ_WRITE_PARAMETER_KEYS if key not in selected_write_keys
    ]
    for source in (params, readback, mapping):
        assert source["wing_eq_write_key_selection_policy"] == (
            WING_EQ_WRITE_KEY_SELECTION_POLICY
        )
        assert source["wing_eq_apply_only_selected_write_keys"] is True
        assert source["wing_eq_selected_write_keys"] == selected_write_keys
        assert source["wing_eq_forbidden_write_keys"] == forbidden_write_keys


def _assert_selected_write_key_contract(
    *,
    params,
    readback,
    prefix,
    selected_write_keys,
    write_parameter_keys,
    selection_policy,
):
    forbidden_write_keys = [
        key for key in write_parameter_keys if key not in selected_write_keys
    ]
    assert params[f"{prefix}_write_key_selection_policy"] == selection_policy
    assert readback[f"{prefix}_write_key_selection_policy"] == selection_policy
    assert params[f"{prefix}_apply_only_selected_write_keys"] is True
    assert readback[f"{prefix}_apply_only_selected_write_keys"] is True
    assert params[f"{prefix}_selected_write_keys"] == selected_write_keys
    assert readback[f"{prefix}_selected_write_keys"] == selected_write_keys
    assert params[f"{prefix}_forbidden_write_keys"] == forbidden_write_keys
    assert readback[f"{prefix}_forbidden_write_keys"] == forbidden_write_keys


def _assert_selected_de_esser_apply_values(recommendation, expected_values):
    proposal = recommendation.to_operator_proposal()
    assert proposal["selected_wing_osc_summary"][
        "wing_osc_apply_parameter_values"
    ] == expected_values
    assert proposal["selected_wing_osc_readback_summary"][
        "wing_osc_readback_apply_parameter_values"
    ] == expected_values
    assert proposal["live_apply_allowed"] is False
    assert proposal["live_authority_granted"] is False
    assert proposal["live_write"] is False


def _assert_selected_dynamic_eq_apply_values(recommendation, expected_values):
    proposal = recommendation.to_operator_proposal()
    assert proposal["selected_wing_osc_summary"][
        "wing_osc_apply_parameter_values"
    ] == expected_values
    assert proposal["selected_wing_osc_readback_summary"][
        "wing_osc_readback_apply_parameter_values"
    ] == expected_values
    assert proposal["live_apply_allowed"] is False
    assert proposal["live_authority_granted"] is False
    assert proposal["live_write"] is False


def _assert_selected_dynamics_apply_values(recommendation, expected_values):
    assert recommendation.selected_strategy.method_family == "dynamics"
    proposal = recommendation.to_operator_proposal()
    assert proposal["selected_wing_osc_summary"][
        "wing_osc_apply_parameter_values"
    ] == expected_values
    assert proposal["selected_wing_osc_readback_summary"][
        "wing_osc_readback_apply_parameter_values"
    ] == expected_values
    assert proposal["live_apply_allowed"] is False
    assert proposal["live_authority_granted"] is False
    assert proposal["live_write"] is False


def _assert_selected_gate_apply_values(recommendation, expected_values):
    assert recommendation.selected_strategy.method_family in {"gate", "expander"}
    proposal = recommendation.to_operator_proposal()
    assert proposal["selected_wing_osc_summary"][
        "wing_osc_apply_parameter_values"
    ] == expected_values
    assert proposal["selected_wing_osc_readback_summary"][
        "wing_osc_readback_apply_parameter_values"
    ] == expected_values
    assert proposal["live_apply_allowed"] is False
    assert proposal["live_authority_granted"] is False
    assert proposal["live_write"] is False


def _assert_selected_has_no_apply_values(recommendation):
    proposal = recommendation.to_operator_proposal()
    assert proposal["selected_wing_osc_summary"][
        "wing_osc_apply_parameter_values"
    ] == {}
    assert proposal["selected_wing_osc_readback_summary"][
        "wing_osc_readback_apply_parameter_values"
    ] == {}
    assert proposal["live_apply_allowed"] is False
    assert proposal["live_authority_granted"] is False
    assert proposal["live_write"] is False


def _assert_mute_readback_contract(strategy, *, target_surface, path_prefix):
    params = strategy.parameters
    assert params["wing_mute_target_surface"] == target_surface
    assert params["wing_mute_osc_path_suffix"] == "mute"
    assert params["wing_mute_osc_value"] == WING_MUTE_ON_VALUE
    assert params["wing_mute_value_type"] == "integer_bool"
    assert params["wing_mute_min"] == WING_MUTE_OFF_VALUE
    assert params["wing_mute_max"] == WING_MUTE_ON_VALUE
    assert params["wing_mute_readback_must_match_osc_value"] is True

    mute_readback = strategy.future_wing_readback_contract["mute_readback"]
    _assert_selected_write_key_contract(
        params=params,
        readback=mute_readback,
        prefix="wing_mute",
        selected_write_keys=list(WING_MUTE_WRITE_PARAMETER_KEYS),
        write_parameter_keys=WING_MUTE_WRITE_PARAMETER_KEYS,
        selection_policy=WING_MUTE_WRITE_KEY_SELECTION_POLICY,
    )
    assert mute_readback["read_before_paths"] == ["mute"]
    assert mute_readback["read_after_paths"] == ["mute"]
    assert f"{path_prefix}/mute" in mute_readback["wing_osc_path_templates"]
    assert mute_readback["wing_mute_mapping_required"] is True
    assert mute_readback["wing_mute_osc_path_suffix"] == "mute"
    assert mute_readback["wing_mute_osc_value"] == WING_MUTE_ON_VALUE
    assert mute_readback["wing_mute_value_type"] == "integer_bool"
    assert mute_readback["wing_mute_readback_must_match_osc_value"] is True
    assert mute_readback["previous_mute_state_required_for_rollback"] is True
    assert mute_readback["manual_mute_authority_preserved"] is True


def _assert_fx_level_write_key_contract(
    params,
    fx_readback,
    *,
    prefix,
    selected_write_keys,
    forbidden_write_keys,
):
    assert params[f"{prefix}_fx_apply_only_selected_write_keys"] is True
    assert params[f"{prefix}_fx_selected_write_keys"] == selected_write_keys
    assert params[f"{prefix}_fx_forbidden_write_keys"] == forbidden_write_keys
    assert fx_readback[f"{prefix}_fx_apply_only_selected_write_keys"] is True
    assert fx_readback[f"{prefix}_fx_selected_write_keys"] == selected_write_keys
    assert fx_readback[f"{prefix}_fx_forbidden_write_keys"] == forbidden_write_keys


def _assert_fx_send_level_readback_contract(
    strategy,
    *,
    expected_delta,
    target_surface,
    source_binding_required,
):
    params = strategy.parameters
    assert params["wing_fx_send_level_target_surface"] == target_surface
    assert params["wing_fx_send_level_osc_path_suffix"] == "send/{send}/lvl"
    assert params["wing_fx_send_level_delta_db"] == expected_delta
    assert params["wing_fx_send_level_value_type"] == (
        "relative_db_delta_requires_send_index_and_current_level_readback"
    )
    assert params["wing_fx_send_level_min_db"] == WING_FX_SEND_LEVEL_MIN_DB
    assert params["wing_fx_send_level_max_db"] == WING_FX_SEND_LEVEL_MAX_DB
    assert params["wing_fx_send_level_absolute_value_requires_readback"] is True
    assert params["wing_fx_send_index_required"] is True
    assert params["wing_fx_send_no_absolute_osc_value_without_readback"] is True
    assert params["wing_fx_send_source_binding_required"] is source_binding_required

    fx_applicability = strategy.future_wing_readback_contract["wing_osc_applicability"]
    assert "wing_fx_send_level_delta_db" in fx_applicability[
        "actionable_parameter_keys"
    ]
    fx_readback = strategy.future_wing_readback_contract["fx_readback"]
    assert "/ch/{channel}/send/{send}/lvl" in fx_readback["wing_osc_path_templates"]
    assert "/bus/{bus}/send/{send}/lvl" in fx_readback["wing_osc_path_templates"]
    assert fx_readback["wing_fx_send_level_mapping_required"] is True
    assert fx_readback["wing_fx_send_level_delta_db"] == expected_delta
    assert fx_readback["wing_fx_send_level_osc_path_suffix"] == "send/{send}/lvl"
    assert "send/{send}/on" in fx_readback["read_before_paths"]
    assert "send/{send}/lvl" in fx_readback["read_before_paths"]
    assert "send/{send}/lvl" in fx_readback["read_after_paths"]
    assert "fx/send" not in fx_readback["read_before_paths"]
    assert fx_readback["wing_fx_readback_paths_are_wing_fw_3_0_5_suffixes"] is True
    assert fx_readback["wing_fx_send_level_absolute_value_requires_readback"] is True
    assert fx_readback["wing_fx_send_index_required"] is True
    assert fx_readback["wing_fx_send_no_absolute_osc_value_without_readback"] is True
    assert fx_readback["wing_fx_send_source_binding_required"] is source_binding_required
    if "soundcheck_fx_level_move_kinds" in params:
        _assert_fx_level_write_key_contract(
            params,
            fx_readback,
            prefix="soundcheck",
            selected_write_keys=["wing_fx_send_level_delta_db"],
            forbidden_write_keys=["wing_fx_return_delta_db"],
        )
    if "live_fx_level_move_kinds" in params:
        _assert_fx_level_write_key_contract(
            params,
            fx_readback,
            prefix="live",
            selected_write_keys=["wing_fx_send_level_delta_db"],
            forbidden_write_keys=["wing_fx_return_delta_db"],
        )


def _assert_fx_return_fxmix_readback_contract(
    strategy,
    *,
    expected_delta,
    target_surface,
    fx_slot_binding_required,
):
    params = strategy.parameters
    assert params["wing_fx_return_level_target_surface"] == target_surface
    assert params["wing_fx_return_level_osc_path_suffix"] == "fxmix"
    assert params["wing_fx_return_delta_db"] == expected_delta
    assert 0.0 < params["wing_fx_return_fxmix_multiplier"] < 1.0
    assert params["wing_fx_return_fxmix_delta_percent_at_100"] < 0.0
    assert params["wing_fx_return_value_type"] == (
        "relative_fxmix_percent_requires_readback_to_resolve_absolute_value"
    )
    assert params["wing_fx_return_min_percent"] == WING_FX_MIX_MIN_PERCENT
    assert params["wing_fx_return_max_percent"] == WING_FX_MIX_MAX_PERCENT
    assert params["wing_fx_return_absolute_value_requires_readback"] is True
    assert params["wing_fx_return_no_db_return_fader_mapping"] is True
    assert params["wing_fx_return_no_absolute_osc_value_without_readback"] is True
    assert params["wing_fx_return_fx_slot_required"] is True
    assert params["wing_fx_return_fx_slot_binding_required"] is fx_slot_binding_required

    fx_applicability = strategy.future_wing_readback_contract["wing_osc_applicability"]
    assert "wing_fx_return_delta_db" in fx_applicability["actionable_parameter_keys"]
    fx_readback = strategy.future_wing_readback_contract["fx_readback"]
    assert "/fx/{fx}/mdl" in fx_readback["wing_osc_path_templates"]
    assert "/fx/{fx}/fxmix" in fx_readback["wing_osc_path_templates"]
    assert fx_readback["wing_fx_return_level_mapping_required"] is True
    assert fx_readback["wing_fx_return_delta_db"] == expected_delta
    assert fx_readback["wing_fx_return_level_osc_path_suffix"] == "fxmix"
    assert "mdl" in fx_readback["read_before_paths"]
    assert "fxmix" in fx_readback["read_before_paths"]
    assert "fxmix" in fx_readback["read_after_paths"]
    assert "fx/return" not in fx_readback["read_before_paths"]
    assert fx_readback["wing_fx_readback_paths_are_wing_fw_3_0_5_suffixes"] is True
    assert fx_readback["wing_fx_return_absolute_value_requires_readback"] is True
    assert fx_readback["wing_fx_return_no_db_return_fader_mapping"] is True
    assert fx_readback["wing_fx_return_no_absolute_osc_value_without_readback"] is True
    assert fx_readback["wing_fx_return_fx_slot_required"] is True
    assert fx_readback["wing_fx_return_fx_slot_binding_required"] is fx_slot_binding_required
    if "soundcheck_fx_level_move_kinds" in params:
        _assert_fx_level_write_key_contract(
            params,
            fx_readback,
            prefix="soundcheck",
            selected_write_keys=["wing_fx_return_delta_db"],
            forbidden_write_keys=["wing_fx_send_level_delta_db"],
        )
    if "live_fx_level_move_kinds" in params:
        _assert_fx_level_write_key_contract(
            params,
            fx_readback,
            prefix="live",
            selected_write_keys=["wing_fx_return_delta_db"],
            forbidden_write_keys=["wing_fx_send_level_delta_db"],
        )


def _assert_fx_model_time_followup_review_contract(
    strategy,
    *,
    expected_values,
    expected_native_apply_fields,
    expected_review_fields,
):
    fx_readback = strategy.future_wing_readback_contract["fx_readback"]
    assert fx_readback["wing_fx_model_time_followup_review_required"] is True
    assert fx_readback["wing_fx_model_time_followup_apply_ready"] is False
    assert (
        fx_readback[
            "wing_fx_model_time_followup_live_apply_blocked_without_wing_proof"
        ]
        is True
    )
    assert (
        fx_readback["wing_fx_model_time_followup_after_native_level_pullback_required"]
        is True
    )
    assert (
        fx_readback["wing_fx_model_time_followup_requires_real_wing_model_readback"]
        is True
    )
    assert fx_readback["wing_fx_model_time_followup_review_only_fields"] == list(
        expected_review_fields
    )
    assert fx_readback["wing_fx_model_time_followup_values"] == expected_values
    assert fx_readback["wing_fx_model_time_followup_forbidden_apply_fields"] == sorted(
        expected_values
    )
    assert fx_readback["wing_fx_model_time_followup_native_level_apply_fields"] == list(
        expected_native_apply_fields
    )
    assert WING_FX_MODEL_PARAMETER_PATH_PATTERN in fx_readback[
        "wing_fx_model_time_followup_forbidden_path_patterns"
    ]
    assert "fx/model" in fx_readback[
        "wing_fx_model_time_followup_forbidden_path_patterns"
    ]
    assert WING_FX_MODEL_PARAMETER_PATH_PATTERN not in fx_readback[
        "wing_osc_path_templates"
    ]
    assert WING_FX_MODEL_PARAMETER_PATH_PATTERN not in fx_readback[
        "read_before_paths"
    ]
    assert WING_FX_MODEL_PARAMETER_PATH_PATTERN not in fx_readback["read_after_paths"]
    assert fx_readback["wing_fx_model_time_followup_control_surface"] == (
        "operator_console_ui_only_until_verified_model_specific_osc_mapping"
    )
    assert fx_readback["wing_fx_model_time_followup_path_source"] == (
        "no_selected_apply_path_without_verified_wing_fw_3_0_5_model_mapping"
    )
    assert "WING Rack fw3.0.5" in fx_readback[
        "wing_fx_model_time_followup_operator_check"
    ]


def _assert_selected_fx_apply_values(recommendation, expected_values):
    proposal = recommendation.to_operator_proposal()
    relative_expected_values = {
        key: value
        for key, value in expected_values.items()
        if key in {"wing_fx_send_level_delta_db", "wing_fx_return_delta_db"}
    }
    direct_expected_values = {
        key: value for key, value in expected_values.items() if key not in relative_expected_values
    }
    assert proposal["selected_wing_osc_summary"][
        "wing_osc_apply_parameter_values"
    ] == direct_expected_values
    assert proposal["selected_wing_osc_readback_summary"][
        "wing_osc_readback_apply_parameter_values"
    ] == direct_expected_values
    assert proposal["selected_wing_osc_summary"][
        "wing_osc_absolute_value_readback_parameter_values"
    ] == relative_expected_values
    assert proposal["selected_wing_osc_readback_summary"][
        "wing_osc_readback_absolute_value_parameter_values"
    ] == relative_expected_values
    if relative_expected_values:
        assert proposal["selected_wing_osc_summary"][
            "wing_osc_candidate_apply_status"
        ] == "wing_native_osc_candidate_requires_absolute_value_readback_before_apply"
    assert not (
        set(WING_FX_MODEL_SPECIFIC_PARAMETER_KEYS)
        & set(proposal["selected_wing_osc_summary"]["wing_osc_apply_parameter_values"])
    )


def _assert_selected_pan_width_apply_values(recommendation, expected_values):
    proposal = recommendation.to_operator_proposal()
    assert proposal["selected_wing_osc_summary"][
        "wing_osc_apply_parameter_values"
    ] == expected_values
    assert proposal["selected_wing_osc_readback_summary"][
        "wing_osc_readback_apply_parameter_values"
    ] == expected_values
    assert proposal["live_apply_allowed"] is False
    assert proposal["live_authority_granted"] is False
    assert proposal["live_write"] is False


def _assert_fx_model_specific_guard(strategy, expected_keys):
    params = strategy.parameters
    assert params["wing_fx_model_parameter_binding_required"] is True
    assert params["wing_fx_model_parameter_path_pattern"] == (
        WING_FX_MODEL_PARAMETER_PATH_PATTERN
    )
    assert params["wing_fx_model_parameter_path_pattern_apply_ready"] is False
    assert params["wing_fx_model_parameter_requires_fx_model_readback"] is True
    assert params["wing_fx_model_parameter_requires_effects_appendix_mapping"] is True
    assert params["wing_fx_model_parameter_live_apply_blocked"] is True
    assert params["wing_fx_model_parameter_blocker"] == WING_FX_MODEL_PARAMETER_BLOCKER
    assert set(expected_keys) <= set(params["wing_fx_model_parameter_keys"])
    assert params["wing_fx_model_parameter_effects_appendix_mapping_available"] is True
    assert params["wing_fx_model_parameter_model_readback_osc_path_suffix"] == "mdl"
    assert params["wing_fx_model_parameter_values_require_model_specific_readback"] is True
    assert params["wing_fx_model_parameter_apply_ready_after_model_readback"] is False
    assert params["wing_fx_model_parameter_single_model_full_coverage_available"] is False
    assert params["wing_fx_model_parameter_partial_mapping_only"] is True
    assert params[
        "wing_fx_model_parameter_full_coverage_required_before_single_proposal_apply"
    ] is True
    suffixes = set(params["wing_fx_model_parameter_verified_suffixes"])
    if "decay_s" in expected_keys:
        assert "dcy" in suffixes
        assert {"HALL", "ROOM", "PLATE"} <= set(
            params["wing_fx_model_parameter_supported_models"]
        )
    if "pre_delay_ms" in expected_keys:
        assert "pdel" in suffixes
    if "return_hpf_hz" in expected_keys:
        assert {"lc", "rlc"} & suffixes
    if "return_lpf_hz" in expected_keys:
        assert "hc" in suffixes
    if "delay_time_ms" in expected_keys:
        assert "time" in suffixes
        assert {"ST-DL", "DEL/REV"} <= set(
            params["wing_fx_model_parameter_supported_models"]
        )
    if "delay_feedback_percent" in expected_keys:
        assert "feed" in suffixes
    if "delay_low_pass_hz" in expected_keys:
        assert "fhc" in suffixes
    if "ducking_depth_db" in expected_keys:
        assert "ducking_depth_db" in params["wing_fx_model_parameter_unmapped_keys"]
        assert "no verified WING Rack fw3.0.5 FX model ducking-depth parameter" in (
            params["wing_fx_model_parameter_unmapped_reason_by_key"][
                "ducking_depth_db"
            ]
        )
    assert WING_FX_MODEL_PARAMETER_BLOCKER in strategy.blockers
    assert not any(key.startswith("wing_compressor_") for key in params)

    readback = strategy.future_wing_readback_contract
    wing_osc = readback["wing_osc_applicability"]
    assert wing_osc["native_osc_supported"] is False
    assert wing_osc["native_osc_support_level"] == (
        "partial_model_specific_fx_mapping_required"
    )
    assert WING_FX_MODEL_PARAMETER_PATH_PATTERN not in wing_osc[
        "native_osc_path_templates"
    ]
    assert "/fx/{fx}/mdl" in wing_osc["native_osc_path_templates"]
    assert set(expected_keys) <= set(wing_osc["fx_model_specific_parameter_keys"])
    assert wing_osc["fx_model_specific_path_pattern"] == WING_FX_MODEL_PARAMETER_PATH_PATTERN
    assert wing_osc["fx_model_specific_path_pattern_apply_ready"] is False
    assert wing_osc["fx_model_specific_requires_fx_model_readback"] is True
    assert wing_osc["fx_model_specific_requires_effects_appendix_mapping"] is True
    assert wing_osc["fx_model_specific_effects_appendix_mapping_available"] is True
    assert set(params["wing_fx_model_parameter_verified_path_templates"]) == set(
        wing_osc["fx_model_specific_verified_path_templates"]
    )

    fx_readback = readback["fx_readback"]
    assert WING_FX_MODEL_PARAMETER_PATH_PATTERN not in fx_readback[
        "wing_osc_path_templates"
    ]
    assert "/fx/{fx}/mdl" in fx_readback["wing_osc_path_templates"]
    assert fx_readback["wing_fx_model_parameter_mapping_required"] is True
    assert fx_readback["wing_fx_model_parameter_path_pattern"] == (
        WING_FX_MODEL_PARAMETER_PATH_PATTERN
    )
    assert fx_readback["wing_fx_model_parameter_path_pattern_apply_ready"] is False
    assert fx_readback["wing_fx_model_parameter_live_apply_blocked"] is True
    assert fx_readback["wing_fx_model_parameter_blocker"] == (
        WING_FX_MODEL_PARAMETER_BLOCKER
    )
    assert set(expected_keys) <= set(fx_readback["wing_fx_model_parameter_keys"])
    assert "mdl" in fx_readback["read_before_paths"]
    assert "mdl" in fx_readback["read_after_paths"]
    assert "fx/model" not in fx_readback["read_before_paths"]
    assert fx_readback["wing_fx_readback_paths_are_wing_fw_3_0_5_suffixes"] is True
    assert fx_readback[
        "wing_fx_model_parameter_effects_appendix_mapping_available"
    ] is True
    assert fx_readback[
        "wing_fx_model_parameter_single_model_full_coverage_available"
    ] is False
    assert fx_readback["wing_fx_model_parameter_partial_mapping_only"] is True
    assert set(fx_readback["wing_fx_model_parameter_verified_suffixes"]) == suffixes
    assert fx_readback["wing_fx_model_parameter_binding_candidates"] == (
        params["wing_fx_model_parameter_binding_candidates"]
    )


def test_wing_fw_3_0_5_dynamics_threshold_paths_are_surface_specific():
    assert WING_DYNAMICS_THRESHOLD_PATH_SUFFIX_BY_SURFACE == {
        "channel": "thr24",
        "bus": "thr31",
        "main": "thr36",
        "matrix": "thr40",
    }
    expected_threshold_paths = {
        "ch:5": "/ch/{channel}/dyn/thr24",
        "bus:1": "/bus/{bus}/dyn/thr31",
        "main": "/main/{main}/dyn/thr36",
        "mtx:1": "/mtx/{matrix}/dyn/thr40",
    }
    for target_id, expected_path in expected_threshold_paths.items():
        paths = _wing_target_path_templates(target_id, "dynamics")
        assert [path for path in paths if "/dyn/thr" in path] == [expected_path]
        assert [path for path in paths if path.endswith("/dyn/hld")] == [
            expected_path.rsplit("/", 1)[0] + "/hld"
        ]
    assert _wing_target_path_templates("fx:1", "dynamics") == ()


def test_wing_fw_3_0_5_target_ranges_reject_nonexistent_console_surfaces():
    invalid_cases = {
        "ch:0": ("eq", "channel", 0, [1, 40]),
        "ch:41": ("eq", "channel", 41, [1, 40]),
        "aux:9": ("pan", "aux", 9, [1, 8]),
        "bus:17": ("gain_fader", "bus", 17, [1, 16]),
        "main:5": ("dynamics", "main", 5, [1, 4]),
        "mtx:9": ("mute", "matrix", 9, [1, 8]),
        "fx:17": ("fx", "fx", 17, [1, 16]),
    }
    valid_boundary_cases = {
        "ch:40": "eq",
        "aux:8": "pan",
        "bus:16": "gain_fader",
        "main:4": "dynamics",
        "mtx:8": "mute",
        "fx:16": "fx",
    }
    parameters_by_family = {
        "eq": {"filter_type": "peaking", "frequency_hz": 340.0, "gain_db": -2.4},
        "pan": {"target_pan_percent": 12.0},
        "gain_fader": {"delta_db": -1.0},
        "dynamics": {"threshold_dbfs": -24.0, "ratio": 2.0},
        "mute": {"proposed_muted": True},
        "fx": {"return_delta_db": -2.5},
    }

    for target_id, method_family in valid_boundary_cases.items():
        assert _wing_target_path_templates(target_id, method_family)

    for target_id, (
        method_family,
        expected_surface,
        expected_index,
        expected_range,
    ) in invalid_cases.items():
        mapped = _with_wing_native_parameter_mappings(
            parameters_by_family[method_family],
            method_family=method_family,
            target_id=target_id,
        )
        readback = _future_readback_contract(method_family, target_id, mapped)
        applicability = readback["wing_osc_applicability"]

        assert _wing_target_path_templates(target_id, method_family) == ()
        assert applicability["target_surface_kind"] == expected_surface
        assert applicability["target_index_surface"] == expected_surface
        assert applicability["target_index"] == expected_index
        assert applicability["target_index_valid"] is False
        assert applicability["target_index_range"] == expected_range
        assert applicability["target_requires_concrete_console_index"] is True
        assert applicability["native_osc_supported"] is False
        assert applicability["native_osc_support_level"] == "unresolved_wing_osc_target"
        assert applicability["unsupported_strategy_reason"] == (
            "target_id_does_not_resolve_to_wing_fw_3_0_5_osc_surface"
        )


def test_wing_fw_3_0_5_strip_low_cut_maps_bus_main_matrix_eq_type():
    cases = {
        "bus:1": ("bus", 12, "BW12", "/bus/{bus}"),
        "main": ("main", 24, "BW24", "/main/{main}"),
        "mtx:1": ("matrix", 12, "CUT", "/mtx/{matrix}"),
    }

    for target_id, (surface, slope, expected_type, path_prefix) in cases.items():
        params = _with_wing_native_parameter_mappings(
            {
                "filter_type": "high_pass",
                "frequency_hz": 93.0,
                "q": 0.82,
                "slope_db_per_octave": slope,
            },
            method_family="eq",
            target_id=target_id,
        )
        assert params["wing_eq_filter_family"] == "strip_low_cut"
        assert params["wing_eq_target_surface"] == surface
        assert params["wing_eq_filter_osc_path_suffixes"] == {
            "gain": "eq/lg",
            "frequency": "eq/lf",
            "q": "eq/lq",
            "type": "eq/leq",
        }
        assert params["wing_eq_strip_low_cut_frequency_hz"] == 93.0
        assert params["wing_eq_strip_low_cut_q"] == 0.82
        assert params["wing_eq_strip_low_cut_type_osc_value"] == expected_type
        assert expected_type in WING_STRIP_LOW_CUT_TYPE_ALLOWED_VALUES_BY_SURFACE[
            surface
        ]
        readback = _future_readback_contract("eq", target_id, params)[
            "eq_band_readback"
        ]
        assert f"{path_prefix}/eq/leq" in readback["wing_osc_path_templates"]
        assert "flt/lc" not in readback["read_before_paths"]
        assert "flt/lcf" not in readback["read_before_paths"]
        assert "flt/lcs" not in readback["read_before_paths"]
        for suffix in ("eq/lg", "eq/lf", "eq/lq", "eq/leq"):
            assert suffix in readback["read_before_paths"]
            assert suffix in readback["read_after_paths"]


def test_wing_fw_3_0_5_channel_high_cut_mapping_is_native_osc_value():
    params = _with_wing_native_parameter_mappings(
        {
            "filter_type": "low_pass",
            "frequency_hz": 26000.0,
            "slope_db_per_octave": 18,
        },
        method_family="eq",
        target_id="ch:5",
    )

    assert WING_CHANNEL_HIGH_CUT_FREQ_MIN_HZ == 50.0
    assert WING_CHANNEL_HIGH_CUT_FREQ_MAX_HZ == 20000.0
    assert WING_CHANNEL_HIGH_CUT_SLOPES_DB_PER_OCTAVE == (6, 12)
    assert params["wing_eq_filter_family"] == "channel_high_cut"
    assert params["wing_eq_filter_osc_path_suffixes"] == {
        "enabled": "flt/hc",
        "frequency": "flt/hcf",
        "slope": "flt/hcs",
    }
    assert params["wing_eq_high_cut_enabled_osc_value"] == 1
    assert params["wing_eq_high_cut_frequency_hz"] == 20000.0
    assert params["wing_eq_high_cut_slope_db_per_octave"] == 12
    assert params["wing_eq_high_cut_slope_allowed_values"] == [6, 12]
    readback = _future_readback_contract("eq", "ch:5", params)
    eq_readback = readback["eq_band_readback"]
    assert "/ch/{channel}/flt/hc" in eq_readback["wing_osc_path_templates"]
    assert "/ch/{channel}/flt/hcf" in eq_readback["wing_osc_path_templates"]
    assert "/ch/{channel}/flt/hcs" in eq_readback["wing_osc_path_templates"]
    assert eq_readback["read_before_paths"] == [
        "eq/on",
        "eq/mdl",
        "eq/mix",
        "flt/hc",
        "flt/hcf",
        "flt/hcs",
    ]
    assert eq_readback["wing_eq_value_mapping"]["wing_eq_filter_family"] == (
        "channel_high_cut"
    )


def test_wing_fw_3_0_5_high_cut_does_not_apply_to_bus_main_matrix_eq():
    for target_id in ("bus:1", "main", "mtx:1"):
        params = _with_wing_native_parameter_mappings(
            {
                "filter_type": "high_cut",
                "frequency_hz": 12000.0,
                "slope_db_per_octave": 6,
            },
            method_family="eq",
            target_id=target_id,
        )
        readback = _future_readback_contract("eq", target_id, params)

        assert params["wing_eq_value_mapping_supported"] is False
        assert params["wing_eq_mapping_unsupported_reason"] == (
            "channel_high_cut_requires_channel_filter_surface"
        )
        assert readback["wing_osc_applicability"]["native_osc_supported"] is False
        assert "flt/hc" not in readback["eq_band_readback"]["read_before_paths"]
        assert "flt/hcf" not in readback["eq_band_readback"]["read_before_paths"]
        assert "flt/hcs" not in readback["eq_band_readback"]["read_before_paths"]


def test_wing_fw_3_0_5_de_esser_maps_to_channel_ds902_gate_slot():
    params = _with_wing_native_parameter_mappings(
        {
            "frequency_hz": 9200.0,
            "range_db": 1.8,
            "de_esser_mode": "hf only",
            "threshold_dbfs": -24.0,
            "attack_ms": 5.0,
            "release_ms": 120.0,
        },
        method_family="de_esser",
        target_id="ch:6",
    )

    assert WING_DE_ESSER_GATE_MODEL == "DS902"
    assert WING_DE_ESSER_FREQ_MIN_HZ == 800.0
    assert WING_DE_ESSER_FREQ_MAX_HZ == 8000.0
    assert WING_DE_ESSER_RANGE_MIN_DB == 3.0
    assert WING_DE_ESSER_RANGE_MAX_DB == 12.0
    assert WING_DE_ESSER_MODE_VALUES == ("FULL", "HF")
    assert params["wing_de_esser_value_mapping_supported"] is True
    assert params["wing_de_esser_model_osc_path_suffix"] == "gate/mdl"
    assert params["wing_de_esser_model_osc_value"] == WING_DE_ESSER_GATE_MODEL
    assert params["wing_de_esser_on_osc_path_suffix"] == "gate/on"
    assert params["wing_de_esser_on_osc_value"] == 1
    assert params["wing_de_esser_filter_osc_path_suffixes"] == {
        "frequency": "gate/f",
        "range": "gate/range",
        "mode": "gate/mode",
    }
    assert params["wing_de_esser_frequency_osc_value"] == 8000.0
    assert params["wing_de_esser_range_osc_value"] == 3.0
    assert params["wing_de_esser_mode_osc_value"] == "HF"
    assert params["wing_de_esser_uses_gate_slot"] is True
    assert params["wing_de_esser_gate_slot_conflict_policy"] == (
        WING_DE_ESSER_GATE_SLOT_CONFLICT_POLICY
    )
    assert "wing_compressor_model_osc_path_suffix" not in params

    readback = _future_readback_contract("de_esser", "ch:6", params)
    wing_osc = readback["wing_osc_applicability"]
    de_esser_readback = readback["de_esser_readback"]
    _assert_selected_write_key_contract(
        params=params,
        readback=de_esser_readback,
        prefix="wing_de_esser",
        selected_write_keys=list(WING_DE_ESSER_WRITE_PARAMETER_KEYS),
        write_parameter_keys=WING_DE_ESSER_WRITE_PARAMETER_KEYS,
        selection_policy=WING_DE_ESSER_WRITE_KEY_SELECTION_POLICY,
    )
    assert wing_osc["native_osc_supported"] is True
    assert wing_osc["native_osc_support_level"] == (
        "native_channel_gate_ds902_de_esser_osc_path_templates_declared"
    )
    assert wing_osc["actionable_parameter_required"] is True
    assert "wing_de_esser_frequency_osc_value" in wing_osc["actionable_parameter_keys"]
    assert de_esser_readback["gate_model_readback_must_equal"] == (
        WING_DE_ESSER_GATE_MODEL
    )
    expected_shared_gate_models = [
        WING_GATE_REQUIRED_MODEL,
        WING_DE_ESSER_GATE_MODEL,
        WING_DYNAMIC_EQ_GATE_MODEL,
    ]
    assert de_esser_readback["wing_gate_slot_model_precondition_required"] is True
    assert de_esser_readback[
        "wing_gate_slot_model_precondition_required_model"
    ] == WING_DE_ESSER_GATE_MODEL
    assert de_esser_readback[
        "wing_gate_slot_model_precondition_read_before_paths"
    ] == ["gate/mdl", "gate/on"]
    assert de_esser_readback[
        "wing_gate_slot_model_precondition_parameter_apply_requires_model_match"
    ] is True
    assert de_esser_readback[
        "wing_gate_slot_model_precondition_model_write_requires_operator_review"
    ] is True
    assert de_esser_readback[
        "wing_gate_slot_model_precondition_live_apply_blocked_without_wing_proof"
    ] is True
    assert de_esser_readback[
        "wing_gate_slot_model_precondition_no_hidden_model_change"
    ] is True
    assert de_esser_readback[
        "wing_gate_slot_model_precondition_shared_by_models"
    ] == expected_shared_gate_models
    assert de_esser_readback["read_before_paths"] == [
        "gate/mdl",
        "gate/on",
        "gate/f",
        "gate/range",
        "gate/mode",
    ]
    assert "/ch/{channel}/gate/f" in wing_osc["native_osc_path_templates"]
    assert "eq_band_readback" not in readback
    assert "dynamics_readback" not in readback


def test_wing_fw_3_0_5_de_esser_existing_ds902_omits_model_write():
    params = _with_wing_native_parameter_mappings(
        {
            "frequency_hz": 7550.0,
            "range_db": 5.4,
            "de_esser_mode": "hf only",
            "wing_gate_slot_current_model": WING_DE_ESSER_GATE_MODEL,
            "wing_gate_slot_current_enabled": True,
            "wing_gate_slot_model_readback_available": True,
            "wing_gate_slot_current_operator_owned": False,
        },
        method_family="de_esser",
        target_id="ch:6",
    )

    assert params["wing_de_esser_gate_slot_parameter_apply_ready"] is True
    assert params["wing_de_esser_gate_slot_current_model"] == WING_DE_ESSER_GATE_MODEL
    assert (
        params["wing_de_esser_model_write_suppressed_by_existing_readback"] is True
    )
    assert "wing_de_esser_model_osc_value" not in params
    assert params["wing_de_esser_selected_write_keys"] == [
        "wing_de_esser_on_osc_value",
        "wing_de_esser_frequency_osc_value",
        "wing_de_esser_range_osc_value",
        "wing_de_esser_mode_osc_value",
    ]
    assert params["wing_de_esser_forbidden_write_keys"] == [
        "wing_de_esser_model_osc_value"
    ]

    readback = _future_readback_contract("de_esser", "ch:6", params)
    wing_osc = readback["wing_osc_applicability"]
    de_esser_readback = readback["de_esser_readback"]

    assert wing_osc["native_osc_supported"] is True
    assert "wing_de_esser_model_osc_value" not in wing_osc["direct_apply_parameter_keys"]
    assert wing_osc["direct_apply_parameter_keys"] == [
        "wing_de_esser_on_osc_value",
        "wing_de_esser_frequency_osc_value",
        "wing_de_esser_range_osc_value",
        "wing_de_esser_mode_osc_value",
    ]
    assert de_esser_readback["wing_de_esser_model_osc_value"] is None
    assert "gate/mdl" in de_esser_readback["read_before_paths"]
    assert "gate/mdl" in de_esser_readback["read_after_paths"]


def test_wing_fw_3_0_5_de_esser_rejects_non_channel_targets():
    params = _with_wing_native_parameter_mappings(
        {"frequency_hz": 6500.0, "range_db": 5.0},
        method_family="de_esser",
        target_id="bus:1",
    )
    readback = _future_readback_contract("de_esser", "bus:1", params)

    assert params["wing_de_esser_value_mapping_supported"] is False
    assert params["wing_de_esser_mapping_unsupported_reason"] == (
        "de_esser_requires_channel_gate_slot_surface"
    )
    assert _wing_target_path_templates("bus:1", "de_esser") == ()
    assert readback["wing_osc_applicability"]["native_osc_supported"] is False
    assert readback["wing_osc_applicability"]["unsupported_strategy_reason"] == (
        "target_id_does_not_resolve_to_wing_fw_3_0_5_osc_surface"
    )


def test_active_gate_slot_conflict_blocks_de_esser_selection():
    recommendation = analyze_smart_dsp_strategy(
        _analysis(
            problem_hints=("sibilant",),
            sibilance_score=0.86,
            sibilance_frequency_hz=7600.0,
            sibilance_bandwidth_hz=900.0,
            sibilance_event_peak_dbfs=-19.4,
            sibilance_event_contrast_db=7.8,
            sibilance_event_rise_time_ms=4.0,
            sibilance_event_recovery_ms=82.0,
            harshness_score=0.18,
            presence_harshness_score=0.12,
            current_wing_gate_model=WING_GATE_REQUIRED_MODEL,
            current_wing_gate_enabled=True,
            current_wing_gate_slot_readback_available=True,
        )
    )

    assert recommendation.status == "proposal_only"
    assert recommendation.selected_strategy.strategy_id == "static_sibilance_band_cut"
    assert recommendation.selected_strategy.method_family == "eq"
    alternatives = {candidate.strategy_id: candidate for candidate in recommendation.alternatives}
    de_esser = alternatives["vocal_de_esser"]
    assert WING_GATE_SLOT_MODEL_CONFLICT_BLOCKER in de_esser.blockers
    assert de_esser.parameters["wing_gate_slot_required_model"] == WING_DE_ESSER_GATE_MODEL
    assert de_esser.parameters["wing_gate_slot_current_model"] == WING_GATE_REQUIRED_MODEL
    assert de_esser.parameters["wing_gate_slot_model_conflict"] is True
    assert de_esser.parameters[
        "wing_gate_slot_model_precondition_required_model"
    ] == WING_DE_ESSER_GATE_MODEL
    assert de_esser.parameters[
        "wing_gate_slot_model_precondition_no_hidden_model_change"
    ] is True
    assert de_esser.future_wing_readback_contract["de_esser_readback"][
        "gate_slot_model_conflict_blocker"
    ] == WING_GATE_SLOT_MODEL_CONFLICT_BLOCKER
    assert de_esser.future_wing_readback_contract["de_esser_readback"][
        "wing_gate_slot_model_precondition_required_model"
    ] == WING_DE_ESSER_GATE_MODEL
    tradeoffs = {
        row["strategy_id"]: row
        for row in recommendation.evidence["strategy_search"]["alternative_tradeoffs"]
    }
    assert WING_GATE_SLOT_MODEL_CONFLICT_BLOCKER in tradeoffs["vocal_de_esser"][
        "blockers"
    ]


def test_operator_owned_ds902_gate_slot_blocks_de_esser_parameter_apply():
    recommendation = analyze_smart_dsp_strategy(
        _analysis(
            problem_hints=("sibilant",),
            sibilance_score=0.86,
            sibilance_frequency_hz=7600.0,
            sibilance_bandwidth_hz=900.0,
            sibilance_event_peak_dbfs=-19.4,
            sibilance_event_contrast_db=7.8,
            sibilance_event_rise_time_ms=4.0,
            sibilance_event_recovery_ms=82.0,
            harshness_score=0.18,
            presence_harshness_score=0.12,
            current_wing_gate_model=WING_DE_ESSER_GATE_MODEL,
            current_wing_gate_enabled=True,
            current_wing_gate_slot_readback_available=True,
            current_wing_gate_slot_operator_owned=True,
        )
    )

    assert recommendation.status == "proposal_only"
    assert recommendation.selected_strategy.strategy_id == "static_sibilance_band_cut"
    assert recommendation.selected_strategy.method_family == "eq"
    alternatives = {candidate.strategy_id: candidate for candidate in recommendation.alternatives}
    de_esser = alternatives["vocal_de_esser"]
    assert de_esser.blockers == (WING_GATE_SLOT_OPERATOR_OWNED_MODEL_BLOCKER,)
    assert de_esser.parameters["wing_gate_slot_required_model"] == WING_DE_ESSER_GATE_MODEL
    assert de_esser.parameters["wing_gate_slot_current_model"] == WING_DE_ESSER_GATE_MODEL
    assert de_esser.parameters["wing_gate_slot_current_operator_owned"] is True
    assert (
        de_esser.parameters["wing_gate_slot_operator_owned_model_edit_requires_review"]
        is True
    )
    assert de_esser.parameters["wing_gate_slot_operator_owned_model_blocker"] == (
        WING_GATE_SLOT_OPERATOR_OWNED_MODEL_BLOCKER
    )
    assert de_esser.parameters["wing_de_esser_gate_slot_parameter_apply_ready"] is False
    assert de_esser.parameters["wing_de_esser_gate_slot_apply_ready_reason"] == (
        "gate_slot_operator_owned_model_edit_requires_operator_review"
    )
    applicability = de_esser.future_wing_readback_contract["wing_osc_applicability"]
    assert applicability["native_osc_supported"] is False
    assert applicability["native_osc_support_level"] == (
        "gate_slot_operator_owned_model_edit_requires_review"
    )
    assert applicability["gate_slot_operator_owned_model_edit_requires_review"] is True
    assert applicability["gate_slot_operator_owned_model_blocker"] == (
        WING_GATE_SLOT_OPERATOR_OWNED_MODEL_BLOCKER
    )
    assert applicability["unsupported_strategy_reason"] == (
        "gate_slot_model_matches_but_current_wing_gate_slot_is_operator_owned"
    )
    tradeoffs = {
        row["strategy_id"]: row
        for row in recommendation.evidence["strategy_search"]["alternative_tradeoffs"]
    }
    assert WING_GATE_SLOT_OPERATOR_OWNED_MODEL_BLOCKER in tradeoffs[
        "vocal_de_esser"
    ]["blockers"]
    assert tradeoffs["vocal_de_esser"]["wing_osc_native_supported"] is False
    assert tradeoffs["vocal_de_esser"]["wing_osc_support_level"] == (
        "gate_slot_operator_owned_model_edit_requires_review"
    )


def test_active_eq_model_conflict_blocks_static_eq_fallback_selection():
    recommendation = analyze_smart_dsp_strategy(
        _analysis(
            problem_hints=("plosive_control", "muddy"),
            muddiness_score=0.72,
            low_end_excess_score=0.84,
            spectral_tilt_score=0.58,
            plosive_risk_score=0.82,
            plosive_frequency_hz=82.0,
            plosive_event_peak_dbfs=-18.4,
            plosive_event_contrast_db=17.2,
            plosive_event_rise_time_ms=18.0,
            plosive_event_recovery_ms=82.0,
            estimated_lufs=-24.0,
            level_plane_error_lu=0.2,
            current_wing_gate_model=WING_DE_ESSER_GATE_MODEL,
            current_wing_gate_enabled=True,
            current_wing_gate_slot_readback_available=True,
            current_wing_eq_model="GEQ",
            current_wing_eq_enabled=True,
            current_wing_eq_model_readback_available=True,
            current_wing_eq_operator_owned=True,
        )
    )

    assert recommendation.status == "proposal_only"
    assert recommendation.selected_strategy.strategy_id == (
        "manual_plosive_mic_technique_review"
    )
    assert recommendation.selected_strategy.method_family == "analysis_only"
    alternatives = {
        candidate.strategy_id: candidate for candidate in recommendation.alternatives
    }
    dynamic_hpf = alternatives["vocal_plosive_dynamic_hpf_control"]
    static_hpf = alternatives["static_plosive_high_pass_cleanup"]
    assert WING_GATE_SLOT_MODEL_CONFLICT_BLOCKER in dynamic_hpf.blockers
    assert WING_EQ_MODEL_CONFLICT_BLOCKER in static_hpf.blockers
    assert static_hpf.parameters["wing_eq_required_model"] == WING_EQ_REQUIRED_MODEL
    assert static_hpf.parameters["wing_eq_current_model"] == "GEQ"
    assert static_hpf.parameters["wing_eq_model_conflict"] is True
    assert static_hpf.future_wing_readback_contract["eq_band_readback"][
        "wing_eq_model_conflict_blocker"
    ] == WING_EQ_MODEL_CONFLICT_BLOCKER
    tradeoffs = {
        row["strategy_id"]: row
        for row in recommendation.evidence["strategy_search"]["alternative_tradeoffs"]
    }
    assert WING_EQ_MODEL_CONFLICT_BLOCKER in tradeoffs[
        "static_plosive_high_pass_cleanup"
    ]["blockers"]
    assert tradeoffs["static_plosive_high_pass_cleanup"][
        "wing_osc_candidate_apply_status"
    ] == "not_wing_applicable_until_eq_model_readback_or_operator_review"


def test_operator_owned_std_eq_blocks_static_eq_parameter_apply():
    recommendation = analyze_smart_dsp_strategy(
        _analysis(
            problem_hints=("plosive_control", "muddy"),
            muddiness_score=0.72,
            low_end_excess_score=0.84,
            spectral_tilt_score=0.58,
            plosive_risk_score=0.82,
            plosive_frequency_hz=82.0,
            plosive_event_peak_dbfs=-18.4,
            plosive_event_contrast_db=17.2,
            plosive_event_rise_time_ms=18.0,
            plosive_event_recovery_ms=82.0,
            estimated_lufs=-24.0,
            level_plane_error_lu=0.2,
            current_wing_gate_model=WING_DE_ESSER_GATE_MODEL,
            current_wing_gate_enabled=True,
            current_wing_gate_slot_readback_available=True,
            current_wing_eq_model=WING_EQ_REQUIRED_MODEL,
            current_wing_eq_enabled=True,
            current_wing_eq_model_readback_available=True,
            current_wing_eq_operator_owned=True,
        )
    )

    assert recommendation.status == "proposal_only"
    assert recommendation.selected_strategy.strategy_id == (
        "manual_plosive_mic_technique_review"
    )
    assert recommendation.selected_strategy.method_family == "analysis_only"
    alternatives = {
        candidate.strategy_id: candidate for candidate in recommendation.alternatives
    }
    dynamic_hpf = alternatives["vocal_plosive_dynamic_hpf_control"]
    static_hpf = alternatives["static_plosive_high_pass_cleanup"]
    assert WING_GATE_SLOT_MODEL_CONFLICT_BLOCKER in dynamic_hpf.blockers
    assert static_hpf.blockers == (WING_EQ_OPERATOR_OWNED_MODEL_BLOCKER,)
    assert static_hpf.parameters["wing_eq_required_model"] == WING_EQ_REQUIRED_MODEL
    assert static_hpf.parameters["wing_eq_current_model"] == WING_EQ_REQUIRED_MODEL
    assert static_hpf.parameters["wing_eq_current_operator_owned"] is True
    assert (
        static_hpf.parameters["wing_eq_operator_owned_model_edit_requires_review"]
        is True
    )
    assert static_hpf.parameters["wing_eq_operator_owned_model_blocker"] == (
        WING_EQ_OPERATOR_OWNED_MODEL_BLOCKER
    )
    assert static_hpf.parameters["wing_eq_model_change_requires_operator_review"] is True
    eq_readback = static_hpf.future_wing_readback_contract["eq_band_readback"]
    assert eq_readback["wing_eq_operator_owned_model_blocker"] == (
        WING_EQ_OPERATOR_OWNED_MODEL_BLOCKER
    )
    applicability = static_hpf.future_wing_readback_contract[
        "wing_osc_applicability"
    ]
    assert applicability["native_osc_supported"] is False
    assert applicability["native_osc_support_level"] == (
        "eq_operator_owned_model_edit_requires_review"
    )
    assert applicability["eq_operator_owned_model_edit_requires_review"] is True
    assert applicability["eq_operator_owned_model_blocker"] == (
        WING_EQ_OPERATOR_OWNED_MODEL_BLOCKER
    )
    assert applicability["unsupported_strategy_reason"] == (
        "eq_model_matches_but_current_wing_eq_is_operator_owned"
    )
    tradeoffs = {
        row["strategy_id"]: row
        for row in recommendation.evidence["strategy_search"]["alternative_tradeoffs"]
    }
    assert WING_EQ_OPERATOR_OWNED_MODEL_BLOCKER in tradeoffs[
        "static_plosive_high_pass_cleanup"
    ]["blockers"]
    assert tradeoffs["static_plosive_high_pass_cleanup"][
        "wing_osc_native_supported"
    ] is False
    assert tradeoffs["static_plosive_high_pass_cleanup"][
        "wing_osc_support_level"
    ] == "eq_operator_owned_model_edit_requires_review"
    assert tradeoffs["static_plosive_high_pass_cleanup"][
        "wing_osc_candidate_apply_status"
    ] == "not_wing_applicable_until_eq_model_readback_or_operator_review"


def test_operator_owned_channel_filter_blocks_static_hpf_apply():
    recommendation = analyze_smart_dsp_strategy(
        _analysis(
            problem_hints=("plosive_control", "muddy"),
            muddiness_score=0.72,
            low_end_excess_score=0.84,
            spectral_tilt_score=0.58,
            plosive_risk_score=0.82,
            plosive_frequency_hz=82.0,
            plosive_event_peak_dbfs=-18.4,
            plosive_event_contrast_db=17.2,
            plosive_event_rise_time_ms=18.0,
            plosive_event_recovery_ms=82.0,
            estimated_lufs=-24.0,
            level_plane_error_lu=0.2,
            current_wing_gate_model=WING_DE_ESSER_GATE_MODEL,
            current_wing_gate_enabled=True,
            current_wing_gate_slot_readback_available=True,
            current_wing_eq_model=WING_EQ_REQUIRED_MODEL,
            current_wing_eq_enabled=True,
            current_wing_eq_model_readback_available=True,
            current_wing_eq_filter_operator_owned=True,
        )
    )

    assert recommendation.status == "proposal_only"
    assert recommendation.selected_strategy.strategy_id == "low_mid_corrective_cut"
    assert recommendation.selected_strategy.method_family == "eq"
    alternatives = {
        candidate.strategy_id: candidate for candidate in recommendation.alternatives
    }
    static_hpf = alternatives["static_plosive_high_pass_cleanup"]
    assert static_hpf.blockers == (WING_EQ_FILTER_OPERATOR_OWNED_BLOCKER,)
    assert static_hpf.parameters["wing_eq_filter_family"] == "channel_low_cut"
    assert static_hpf.parameters["wing_eq_filter_family_operator_owned_scope"] == (
        "channel_low_cut"
    )
    assert static_hpf.parameters["wing_eq_filter_current_operator_owned"] is True
    assert (
        static_hpf.parameters["wing_eq_filter_operator_owned_edit_requires_review"]
        is True
    )
    assert static_hpf.parameters["wing_eq_filter_operator_owned_blocker"] == (
        WING_EQ_FILTER_OPERATOR_OWNED_BLOCKER
    )
    eq_readback = static_hpf.future_wing_readback_contract["eq_band_readback"]
    assert eq_readback["wing_eq_filter_current_operator_owned"] is True
    assert eq_readback["wing_eq_filter_operator_owned_blocker"] == (
        WING_EQ_FILTER_OPERATOR_OWNED_BLOCKER
    )
    applicability = static_hpf.future_wing_readback_contract[
        "wing_osc_applicability"
    ]
    assert applicability["native_osc_supported"] is False
    assert applicability["native_osc_support_level"] == (
        "eq_filter_operator_owned_edit_requires_review"
    )
    assert applicability["eq_filter_operator_owned_edit_requires_review"] is True
    assert applicability["eq_filter_operator_owned_blocker"] == (
        WING_EQ_FILTER_OPERATOR_OWNED_BLOCKER
    )
    assert applicability["unsupported_strategy_reason"] == (
        "eq_filter_has_native_osc_paths_but_current_wing_filter_is_operator_owned"
    )
    tradeoffs = {
        row["strategy_id"]: row
        for row in recommendation.evidence["strategy_search"]["alternative_tradeoffs"]
    }
    assert WING_EQ_FILTER_OPERATOR_OWNED_BLOCKER in tradeoffs[
        "static_plosive_high_pass_cleanup"
    ]["blockers"]
    assert tradeoffs["static_plosive_high_pass_cleanup"][
        "wing_osc_candidate_apply_status"
    ] == "not_wing_applicable_until_eq_filter_readback_or_operator_review"


def test_operator_owned_eq_filter_blocks_high_cut_and_strip_low_cut_apply():
    high_cut = _make_candidate(
        strategy_id="future_operator_owned_high_cut_probe",
        method_family="eq",
        problem="harsh",
        parameters={
            "filter_type": "high_cut",
            "frequency_hz": 12000.0,
            "slope_db_per_octave": 6,
        },
        scores=_candidate_scores(),
        reason="Probe WING channel high-cut ownership gating.",
        operator_check="Review the current WING high-cut before changing it.",
        analysis=_analysis(
            target_id="ch:10",
            source_role="overhead_cymbal",
            current_wing_eq_model=WING_EQ_REQUIRED_MODEL,
            current_wing_eq_enabled=True,
            current_wing_eq_model_readback_available=True,
            current_wing_eq_filter_operator_owned=True,
        ),
    )
    strip_low_cut = _make_candidate(
        strategy_id="future_operator_owned_strip_low_cut_probe",
        method_family="eq",
        problem="rumble",
        parameters={
            "filter_type": "high_pass",
            "frequency_hz": 93.0,
            "slope_db_per_octave": 12,
            "q": 0.82,
        },
        scores=_candidate_scores(),
        reason="Probe WING strip low-cut ownership gating.",
        operator_check="Review the current WING strip low-cut before changing it.",
        analysis=_analysis(
            target_id="bus:1",
            source_role="drum_bus",
            current_wing_eq_model=WING_EQ_REQUIRED_MODEL,
            current_wing_eq_enabled=True,
            current_wing_eq_model_readback_available=True,
            current_wing_eq_filter_operator_owned=True,
        ),
    )

    expected_families = (
        (high_cut, "channel_high_cut"),
        (strip_low_cut, "strip_low_cut"),
    )
    for candidate, expected_family in expected_families:
        assert candidate.blockers == (WING_EQ_FILTER_OPERATOR_OWNED_BLOCKER,)
        assert candidate.parameters["wing_eq_filter_family"] == expected_family
        assert candidate.parameters["wing_eq_filter_family_operator_owned_scope"] == (
            expected_family
        )
        assert candidate.parameters["wing_eq_filter_current_operator_owned"] is True
        assert (
            candidate.parameters["wing_eq_filter_operator_owned_edit_requires_review"]
            is True
        )
        assert candidate.parameters["wing_eq_filter_operator_owned_blocker"] == (
            WING_EQ_FILTER_OPERATOR_OWNED_BLOCKER
        )
        applicability = candidate.future_wing_readback_contract[
            "wing_osc_applicability"
        ]
        assert applicability["native_osc_supported"] is False
        assert applicability["native_osc_support_level"] == (
            "eq_filter_operator_owned_edit_requires_review"
        )
        assert applicability["unsupported_strategy_reason"] == (
            "eq_filter_has_native_osc_paths_but_current_wing_filter_is_operator_owned"
        )


def test_active_unknown_eq_model_requires_readback_before_eq_candidate():
    candidate = _make_candidate(
        strategy_id="future_eq_active_unknown_model_probe",
        method_family="eq",
        problem="muddy",
        parameters={
            "filter_type": "peaking",
            "frequency_hz": 340.0,
            "gain_db": -3.0,
            "q": 1.2,
        },
        scores=_candidate_scores(),
        reason="Probe active WING EQ slot model readback gating.",
        operator_check="Read eq/mdl before changing EQ.",
        analysis=_analysis(
            current_wing_eq_model="",
            current_wing_eq_enabled=True,
            current_wing_eq_model_readback_available=False,
            current_wing_eq_operator_owned=True,
        ),
    )

    assert WING_EQ_MODEL_READBACK_BLOCKER in candidate.blockers
    assert candidate.parameters["wing_eq_current_model"] == ""
    assert candidate.parameters["wing_eq_current_enabled"] is True
    assert candidate.parameters["wing_eq_model_readback_available"] is False
    assert (
        candidate.parameters["wing_eq_model_readback_required_for_active_slot"]
        is True
    )
    assert (
        candidate.future_wing_readback_contract["eq_band_readback"][
            "wing_eq_model_readback_blocker"
        ]
        == WING_EQ_MODEL_READBACK_BLOCKER
    )


def test_active_unknown_dynamics_model_requires_readback_before_compressor_candidate():
    candidate = _make_candidate(
        strategy_id="future_dynamics_active_unknown_model_probe",
        method_family="dynamics",
        problem="dynamic_control",
        parameters={
            "threshold_dbfs": -24.0,
            "ratio": 2.0,
            "attack_ms": 15.0,
            "release_ms": 140.0,
            "makeup_gain_db": 0.5,
        },
        scores=_candidate_scores(),
        reason="Probe active WING dynamics slot model readback gating.",
        operator_check="Read dyn/mdl before changing compressor.",
        analysis=_analysis(
            problem_hints=("dynamic_control",),
            current_wing_dynamics_model="",
            current_wing_dynamics_enabled=True,
            current_wing_dynamics_model_readback_available=False,
            current_wing_dynamics_operator_owned=True,
        ),
    )

    assert WING_DYNAMICS_MODEL_READBACK_BLOCKER in candidate.blockers
    assert candidate.parameters["wing_dynamics_current_model"] == ""
    assert candidate.parameters["wing_dynamics_current_enabled"] is True
    assert candidate.parameters["wing_dynamics_model_readback_available"] is False
    assert (
        candidate.parameters[
            "wing_dynamics_model_readback_required_for_active_slot"
        ]
        is True
    )
    assert (
        candidate.future_wing_readback_contract["dynamics_readback"][
            "wing_dynamics_model_readback_blocker"
        ]
        == WING_DYNAMICS_MODEL_READBACK_BLOCKER
    )


def test_active_unknown_gate_slot_model_requires_readback_before_gate_candidate():
    candidate = _make_candidate(
        strategy_id="future_gate_active_unknown_model_probe",
        method_family="gate",
        problem="drum_bleed",
        parameters={
            "threshold_dbfs": -45.0,
            "range_db": 10.0,
            "attack_ms": 2.0,
            "hold_ms": 60.0,
            "release_ms": 140.0,
        },
        scores=_candidate_scores(),
        reason="Probe active WING gate slot model readback gating.",
        operator_check="Read gate/mdl before changing gate-slot processing.",
        analysis=_analysis(
            source_role="rack_tom",
            problem_hints=("drum_bleed",),
            current_wing_gate_model="",
            current_wing_gate_enabled=True,
            current_wing_gate_slot_readback_available=False,
            current_wing_gate_slot_operator_owned=True,
        ),
    )

    assert WING_GATE_SLOT_MODEL_READBACK_BLOCKER in candidate.blockers
    assert candidate.parameters["wing_gate_slot_current_model"] == ""
    assert candidate.parameters["wing_gate_slot_current_enabled"] is True
    assert candidate.parameters["wing_gate_slot_model_readback_available"] is False
    assert (
        candidate.parameters[
            "wing_gate_slot_model_readback_required_for_active_slot"
        ]
        is True
    )
    assert (
        candidate.future_wing_readback_contract["gate_expander_readback"][
            "gate_slot_model_readback_blocker"
        ]
        == WING_GATE_SLOT_MODEL_READBACK_BLOCKER
    )


def test_wing_fw_3_0_5_eq_paths_do_not_apply_to_fx_returns():
    assert _wing_target_path_templates("fx:1", "eq") == ()

    params = _with_wing_native_parameter_mappings(
        {
            "filter_type": "peaking",
            "frequency_hz": 1200.0,
            "gain_db": -2.0,
            "q": 1.4,
        },
        method_family="eq",
        target_id="fx:1",
    )
    readback = _future_readback_contract("eq", "fx:1", params)
    wing_osc = readback["wing_osc_applicability"]

    assert params["wing_eq_value_mapping_supported"] is False
    assert params["wing_eq_mapping_unsupported_reason"] == (
        "eq_requires_channel_aux_bus_main_or_matrix_surface"
    )
    assert wing_osc["native_osc_supported"] is False
    assert wing_osc["native_osc_support_level"] == "unresolved_wing_osc_target"
    assert wing_osc["native_osc_path_templates"] == []
    assert wing_osc["native_eq_band_capacity"] == 0
    assert readback["eq_band_readback"]["wing_native_eq_band_capacity"] == 0


def test_wing_fw_3_0_5_eq_band_capacity_is_surface_specific():
    assert WING_NATIVE_EQ_PARAMETRIC_BAND_COUNT_BY_SURFACE == {
        "channel": 4,
        "aux": 4,
        "bus": 6,
        "main": 6,
        "matrix": 6,
    }
    harmonics = [
        {
            "frequency_hz": 60.0 * band_number,
            "gain_db": -3.0,
            "q": 8.0,
            "filter_type": "peaking",
        }
        for band_number in range(1, 7)
    ]

    channel_params = _with_wing_native_parameter_mappings(
        {
            "filter_type": "notch_stack",
            "frequency_hz": 60.0,
            "gain_db": -3.0,
            "q": 8.0,
            "harmonic_notches": harmonics[:5],
            "wing_required_eq_band_count": 5,
        },
        method_family="eq",
        target_id="ch:5",
    )
    channel_wing_osc = _future_readback_contract("eq", "ch:5", channel_params)[
        "wing_osc_applicability"
    ]
    assert channel_wing_osc["native_osc_supported"] is False
    assert channel_wing_osc["native_eq_band_capacity"] == 4
    assert channel_wing_osc["native_osc_support_level"] == (
        "eq_band_count_exceeds_wing_fw_3_0_5_native_capacity"
    )

    for target_id, expected_prefix in {
        "bus:1": "/bus/{bus}",
        "main": "/main/{main}",
        "mtx:1": "/mtx/{matrix}",
    }.items():
        params = _with_wing_native_parameter_mappings(
            {
                "filter_type": "notch_stack",
                "frequency_hz": 60.0,
                "gain_db": -3.0,
                "q": 8.0,
                "harmonic_notches": harmonics,
                "wing_required_eq_band_count": 6,
            },
            method_family="eq",
            target_id=target_id,
        )
        readback = _future_readback_contract("eq", target_id, params)
        wing_osc = readback["wing_osc_applicability"]

        assert params["wing_eq_native_parametric_band_count"] == 6
        assert wing_osc["native_osc_supported"] is True
        assert wing_osc["native_eq_band_capacity"] == 6
        assert f"{expected_prefix}/eq/{{band}}f" in wing_osc[
            "native_osc_path_templates"
        ]
        assert readback["eq_band_readback"]["wing_native_eq_band_capacity"] == 6


def test_wing_fw_3_0_5_aux_strip_surfaces_are_native_osc_mapped():
    aux_eq = _with_wing_native_parameter_mappings(
        {
            "filter_type": "peaking",
            "frequency_hz": 2400.0,
            "gain_db": -2.0,
            "q": 1.4,
        },
        method_family="eq",
        target_id="aux:1",
    )
    aux_eq_readback = _future_readback_contract("eq", "aux:1", aux_eq)
    aux_eq_applicability = aux_eq_readback["wing_osc_applicability"]

    assert aux_eq["wing_eq_target_surface"] == "aux"
    assert aux_eq["wing_eq_value_mapping_supported"] is True
    assert aux_eq["wing_eq_mix_max_percent"] == WING_EQ_MIX_MAX_PERCENT_BY_SURFACE["aux"]
    assert aux_eq_applicability["native_osc_supported"] is True
    assert aux_eq_applicability["target_surface_kind"] == "aux"
    assert aux_eq_applicability["native_eq_band_capacity"] == 4
    assert "/aux/{aux}/eq/on" in aux_eq_applicability["native_osc_path_templates"]
    assert "/aux/{aux}/eq/{band}f" in aux_eq_applicability["native_osc_path_templates"]
    assert not any(
        path.startswith("/aux/{aux}/flt/")
        for path in aux_eq_applicability["native_osc_path_templates"]
    )

    aux_low_cut = _with_wing_native_parameter_mappings(
        {"filter_type": "high_pass", "frequency_hz": 95.0, "slope_db_per_octave": 24.0},
        method_family="eq",
        target_id="aux:1",
    )
    assert aux_low_cut["wing_eq_filter_family"] == "strip_low_cut"
    assert aux_low_cut["wing_eq_strip_low_cut_type_osc_value"] == "CUT"
    assert aux_low_cut["wing_eq_strip_low_cut_type_allowed_values"] == [
        "SHV",
        "PEQ",
        "CUT",
    ]
    assert "wing_eq_low_cut_frequency_hz" not in aux_low_cut

    aux_gain = _with_wing_native_parameter_mappings(
        {"delta_db": -1.0},
        method_family="gain_fader",
        target_id="aux:1",
    )
    aux_gain_readback = _future_readback_contract("gain_fader", "aux:1", aux_gain)
    assert aux_gain["wing_gain_target_surface"] == "aux"
    assert aux_gain["wing_gain_delta_db"] == -1.0
    assert aux_gain_readback["wing_osc_applicability"]["native_osc_supported"] is True
    assert "/aux/{aux}/fdr" in aux_gain_readback["wing_osc_applicability"][
        "native_osc_path_templates"
    ]
    assert "/aux/{aux}/in/set/trim" in aux_gain_readback["wing_osc_applicability"][
        "native_osc_path_templates"
    ]

    aux_mute = _with_wing_native_parameter_mappings(
        {"proposed_muted": True},
        method_family="mute",
        target_id="aux:1",
    )
    assert aux_mute["wing_mute_target_surface"] == "aux"
    assert aux_mute["wing_mute_osc_value"] == WING_MUTE_ON_VALUE
    assert _future_readback_contract("mute", "aux:1", aux_mute)[
        "wing_osc_applicability"
    ]["native_osc_supported"] is True

    aux_pan = _with_wing_native_parameter_mappings(
        {"target_pan_percent": -12.0, "target_width_percent": 88.0},
        method_family="pan",
        target_id="aux:1",
    )
    aux_pan_readback = _future_readback_contract("pan", "aux:1", aux_pan)
    assert aux_pan["wing_pan_osc_value"] == -12.0
    assert aux_pan["wing_width_osc_value"] == 88.0
    assert "/aux/{aux}/pan" in aux_pan_readback["wing_osc_applicability"][
        "native_osc_path_templates"
    ]
    assert "/aux/{aux}/wid" in aux_pan_readback["wing_osc_applicability"][
        "native_osc_path_templates"
    ]

    aux_fx_send = _with_wing_native_parameter_mappings(
        {"send_delta_db": -1.5},
        method_family="fx",
        target_id="aux:1",
    )
    aux_fx_readback = _future_readback_contract("fx", "aux:1", aux_fx_send)
    assert aux_fx_send["wing_fx_send_level_target_surface"] == "aux"
    assert aux_fx_send["wing_fx_send_source_surface_supported"] is True
    assert aux_fx_send["wing_fx_send_source_binding_required"] is False
    assert aux_fx_readback["wing_osc_applicability"]["native_osc_supported"] is True
    assert "/aux/{aux}/send/{send}/lvl" in aux_fx_readback["wing_osc_applicability"][
        "native_osc_path_templates"
    ]


def test_wing_fw_3_0_5_aux_dynamics_uses_native_pse_la_combo_mapping():
    mapped = _with_wing_native_parameter_mappings(
        {
            "threshold_dbfs": -28.0,
            "target_gain_reduction_db": 4.2,
            "attack_ms": 8.0,
            "detector": "PEAK",
            "compressor_enabled": True,
        },
        method_family="dynamics",
        target_id="aux:1",
    )
    readback = _future_readback_contract("dynamics", "aux:1", mapped)
    applicability = readback["wing_osc_applicability"]
    dynamics_readback = readback["dynamics_readback"]

    assert mapped["wing_aux_dynamics_target_surface"] == "aux"
    assert mapped["wing_aux_dynamics_no_channel_compressor_model_slot"] is True
    assert mapped["wing_aux_dynamics_no_dyn_mdl_path"] is True
    assert mapped["wing_aux_dynamics_on_osc_path_suffix"] == "dyn/on"
    assert mapped["wing_aux_dynamics_on_osc_value"] == 1
    assert mapped["wing_aux_dynamics_threshold_osc_path_suffix"] == "dyn/thr"
    assert mapped["wing_aux_dynamics_threshold_osc_value"] == -28.0
    assert mapped["wing_aux_dynamics_threshold_min"] == -36.0
    assert mapped["wing_aux_dynamics_threshold_max"] == 12.0
    assert mapped["wing_aux_dynamics_depth_osc_path_suffix"] == "dyn/depth"
    assert mapped["wing_aux_dynamics_depth_osc_value"] == 4.2
    assert mapped["wing_aux_dynamics_depth_parameter_source"] == (
        "bounded_starting_depth_from_target_gain_reduction_db"
    )
    assert mapped["wing_aux_dynamics_fast_osc_path_suffix"] == "dyn/fast"
    assert mapped["wing_aux_dynamics_fast_osc_value"] == 1
    assert mapped["wing_aux_dynamics_peak_osc_path_suffix"] == "dyn/peak"
    assert mapped["wing_aux_dynamics_peak_osc_value"] == 1
    assert "wing_compressor_model_osc_path_suffix" not in mapped
    assert "wing_compressor_ratio_osc_value" not in mapped
    assert "wing_compressor_attack_osc_value" not in mapped
    assert applicability["target_surface_kind"] == "aux"
    assert applicability["native_osc_supported"] is True
    assert applicability["native_osc_support_level"] == (
        "native_osc_path_templates_declared"
    )
    assert applicability["actionable_parameter_keys"] == [
        "wing_aux_dynamics_on_osc_value",
        "wing_aux_dynamics_threshold_osc_value",
        "wing_aux_dynamics_depth_osc_value",
        "wing_aux_dynamics_fast_osc_value",
        "wing_aux_dynamics_peak_osc_value",
    ]
    assert "/aux/{aux}/dyn/thr" in applicability["native_osc_path_templates"]
    assert "/aux/{aux}/dyn/depth" in applicability["native_osc_path_templates"]
    assert "/aux/{aux}/dyn/fast" in applicability["native_osc_path_templates"]
    assert "/aux/{aux}/dyn/peak" in applicability["native_osc_path_templates"]
    assert not any(
        path.endswith("/dyn/mdl")
        for path in applicability["native_osc_path_templates"]
    )
    assert dynamics_readback["dynamics_model_readback_required"] is False
    assert dynamics_readback["dynamics_model_readback_must_equal"] is None
    assert dynamics_readback["wing_aux_dynamics_no_dyn_mdl_path"] is True
    assert dynamics_readback["read_before_paths"] == [
        "dyn/on",
        "dyn/thr",
        "dyn/depth",
        "dyn/fast",
        "dyn/peak",
        "dyn/ingain",
        "dyn/cpeak",
        "dyn/cmode",
    ]
    assert "dyn/mdl" not in dynamics_readback["read_before_paths"]
    assert dynamics_readback["wing_aux_dynamics_threshold_osc_value"] == -28.0
    assert dynamics_readback["wing_aux_dynamics_depth_osc_value"] == 4.2


def test_operator_owned_aux_dynamics_blocks_pse_la_apply():
    recommendation = analyze_smart_dsp_strategy(
        _analysis(
            target_id="aux:1",
            source_role="playback",
            source_label="Playback Aux",
            problem_hints=("dynamic_control",),
            muddiness_score=0.0,
            rms_dbfs=-24.0,
            estimated_lufs=-24.5,
            lra_proxy_db=15.0,
            dynamics_variance_score=0.74,
            crest_factor_db=18.0,
            dynamics_peak_to_median_db=13.0,
            dynamics_rise_time_ms=95.0,
            dynamics_recovery_ms=190.0,
            true_peak_headroom_db=4.0,
            feedback_risk=0.1,
            current_wing_dynamics_operator_owned=True,
        )
    )

    assert recommendation.status == "proposal_only"
    assert recommendation.selected_strategy.strategy_id == (
        "hold_group_bus_for_member_evidence"
    )
    assert recommendation.selected_strategy.method_family == "analysis_only"

    alternatives = {
        candidate.strategy_id: candidate for candidate in recommendation.alternatives
    }
    aux_dynamics = alternatives["group_bus_compression_review"]
    assert aux_dynamics.blockers == (WING_AUX_DYNAMICS_OPERATOR_OWNED_BLOCKER,)
    params = aux_dynamics.parameters
    assert params["wing_aux_dynamics_target_surface"] == "aux"
    assert params["wing_aux_dynamics_on_osc_value"] == 1
    assert params["wing_aux_dynamics_threshold_osc_value"] == -20.3
    assert params["wing_aux_dynamics_depth_osc_value"] == 2.9
    assert params["wing_aux_dynamics_fast_osc_value"] == 0
    assert params["wing_aux_dynamics_current_operator_owned"] is True
    assert params["wing_aux_dynamics_operator_owned_edit_requires_review"] is True
    assert (
        params["wing_aux_dynamics_operator_owned_blocker"]
        == WING_AUX_DYNAMICS_OPERATOR_OWNED_BLOCKER
    )

    applicability = aux_dynamics.future_wing_readback_contract[
        "wing_osc_applicability"
    ]
    assert applicability["native_osc_supported"] is False
    assert applicability["native_osc_support_level"] == (
        "aux_dynamics_operator_owned_edit_requires_review"
    )
    assert applicability["unsupported_strategy_reason"] == (
        "aux_dynamics_has_native_osc_paths_but_current_wing_aux_dynamics_is_operator_owned"
    )
    assert applicability["aux_dynamics_operator_owned_edit_requires_review"] is True
    assert applicability["aux_dynamics_operator_owned_blocker"] == (
        WING_AUX_DYNAMICS_OPERATOR_OWNED_BLOCKER
    )
    assert "/aux/{aux}/dyn/thr" in applicability["native_osc_path_templates"]
    assert "/aux/{aux}/dyn/depth" in applicability["native_osc_path_templates"]

    dynamics_readback = aux_dynamics.future_wing_readback_contract[
        "dynamics_readback"
    ]
    assert dynamics_readback["wing_aux_dynamics_current_operator_owned"] is True
    assert (
        dynamics_readback["wing_aux_dynamics_operator_owned_edit_requires_review"]
        is True
    )
    assert dynamics_readback["wing_aux_dynamics_operator_owned_blocker"] == (
        WING_AUX_DYNAMICS_OPERATOR_OWNED_BLOCKER
    )
    assert dynamics_readback["wing_aux_dynamics_operator_owned_policy"] == (
        "do_not_overwrite_operator_owned_aux_dynamics_without_review"
    )

    tradeoffs = {
        row["strategy_id"]: row
        for row in recommendation.evidence["strategy_search"]["alternative_tradeoffs"]
    }
    assert tradeoffs["group_bus_compression_review"][
        "wing_osc_candidate_apply_status"
    ] == "not_wing_applicable_until_aux_dynamics_readback_or_operator_review"
    assert tradeoffs["group_bus_compression_review"]["why_not_selected"] == (
        "blocked_by:wing_aux_dynamics_operator_owned_edit_requires_operator_review"
    )
    proposal = recommendation.to_operator_proposal()
    assert proposal["selected_wing_osc_summary"]["wing_osc_apply_parameter_values"] == {}
    assert proposal["can_apply_live"] is False


def test_wing_fw_3_0_5_aux_gate_and_expander_stay_blocked_until_mapped():
    for method_family in ("gate", "expander"):
        mapped = _with_wing_native_parameter_mappings(
            {"threshold_dbfs": -32.0},
            method_family=method_family,
            target_id="aux:1",
        )
        readback = _future_readback_contract(method_family, "aux:1", mapped)
        applicability = readback["wing_osc_applicability"]

        assert applicability["target_surface_kind"] == "aux"
        assert applicability["native_osc_supported"] is False
        assert applicability["native_osc_path_templates"] == []
        assert applicability["unsupported_strategy_reason"] == (
            "target_id_does_not_resolve_to_wing_fw_3_0_5_osc_surface"
        )
        assert "eq_band_readback" not in readback
        assert readback["requires_operator_approval"] is True
        assert applicability["wing_mvp_live_write_scope_approved"] is False


def test_wing_fw_3_0_5_eq_metadata_does_not_create_compressor_mapping():
    params = _with_wing_native_parameter_mappings(
        {
            "filter_type": "peaking",
            "frequency_hz": 340.0,
            "gain_db": -2.4,
            "q": 1.2,
            "makeup_gain_db": 0.0,
        },
        method_family="eq",
        target_id="ch:5",
    )
    readback = _future_readback_contract("eq", "ch:5", params)

    assert params["wing_eq_value_mapping_supported"] is True
    assert params["wing_eq_filter_family"] == "parametric_band"
    assert "dynamics_readback" not in readback
    assert not any(key.startswith("wing_compressor_") for key in params)


def test_wing_fw_3_0_5_gate_threshold_path_is_channel_specific():
    assert WING_GATE_THRESHOLD_PATH_SUFFIX_BY_SURFACE == {
        "channel": "thr21",
    }
    channel_paths = _wing_target_path_templates("ch:5", "gate")
    assert "/ch/{channel}/gate/thr21" in channel_paths
    assert "/ch/{channel}/gatesc/type" in channel_paths
    assert "/ch/{channel}/gatesc/f" in channel_paths
    assert "/ch/{channel}/gatesc/q" in channel_paths
    assert "/ch/{channel}/gatesc/src" in channel_paths
    assert "/ch/{channel}/gatesc/tap" in channel_paths
    assert "/ch/{channel}/gate/thr" not in channel_paths
    assert _wing_target_path_templates("ch:5", "expander") == channel_paths
    for target_id in ("bus:1", "main", "mtx:1", "fx:1"):
        assert _wing_target_path_templates(target_id, "gate") == ()
        assert _wing_target_path_templates(target_id, "expander") == ()


def test_wing_fw_3_0_5_pan_width_paths_and_mappings_are_native():
    expected_paths = {
        "ch:5": ("/ch/{channel}/pan", "/ch/{channel}/wid"),
        "bus:1": ("/bus/{bus}/pan", "/bus/{bus}/wid"),
        "main": ("/main/{main}/pan", "/main/{main}/wid"),
        "mtx:1": ("/mtx/{matrix}/pan", "/mtx/{matrix}/wid"),
    }
    for target_id, expected in expected_paths.items():
        assert _wing_target_path_templates(target_id, "pan") == expected
    assert _wing_target_path_templates("fx:1", "pan") == ()

    mapped = _with_wing_native_parameter_mappings(
        {
            "target_pan_percent": 125.0,
            "target_width_percent": -180.0,
        },
        target_id="ch:12",
        method_family="pan",
    )

    assert mapped["wing_pan_osc_path_suffix"] == "pan"
    assert mapped["wing_pan_osc_value"] == WING_PAN_MAX_PERCENT
    assert mapped["wing_pan_osc_protocol_version"] == WING_PAN_WIDTH_OSC_PROTOCOL_VERSION
    assert mapped["wing_pan_osc_protocol_source"] == WING_PAN_WIDTH_OSC_PROTOCOL_SOURCE
    assert mapped["wing_pan_quantization_error"] == 25.0
    assert mapped["wing_width_osc_path_suffix"] == "wid"
    assert mapped["wing_width_osc_value"] == WING_WIDTH_MIN_PERCENT
    assert (
        mapped["wing_width_osc_protocol_version"]
        == WING_PAN_WIDTH_OSC_PROTOCOL_VERSION
    )
    assert mapped["wing_width_osc_protocol_source"] == WING_PAN_WIDTH_OSC_PROTOCOL_SOURCE
    assert mapped["wing_width_quantization_error"] == 30.0


def test_width_only_widening_is_blocked_by_phase_mono_evidence():
    analysis = _analysis(
        target_id="ch:12",
        source_role="electric_guitar",
        problem_hints=("pan_width",),
        phase_correlation=0.22,
        mono_sum_delta_db=-5.2,
    )
    scores = {
        "speed": 0.8,
        "precision": 0.8,
        "musical": 0.8,
        "safety": 0.7,
        "reversibility": 0.8,
        "operator_clarity": 0.8,
    }

    widening = _make_candidate(
        strategy_id="future_width_only_widening_probe",
        method_family="pan",
        problem="pan_width",
        parameters={
            "target_width_percent": 125.0,
            "width_delta_percent": 25.0,
        },
        scores=scores,
        reason="Probe future width-only widening safety.",
        operator_check="Check mono compatibility.",
        analysis=analysis,
    )
    assert "phase_correlation_blocks_width" in widening.blockers
    assert "mono_sum_loss_blocks_width" in widening.blockers
    assert widening.parameters["wing_width_osc_path_suffix"] == "wid"
    assert widening.parameters["wing_width_osc_value"] == 125.0

    reduction = _make_candidate(
        strategy_id="future_width_only_reduction_probe",
        method_family="pan",
        problem="phase_mono_risk",
        parameters={
            "target_width_percent": 35.0,
            "width_delta_percent": -65.0,
        },
        scores=scores,
        reason="Probe future width-only reduction safety.",
        operator_check="Check mono compatibility.",
        analysis=analysis,
    )
    assert "phase_correlation_blocks_width" not in reduction.blockers
    assert "mono_sum_loss_blocks_width" not in reduction.blockers
    assert reduction.parameters["wing_width_osc_path_suffix"] == "wid"
    assert reduction.parameters["wing_width_osc_value"] == 35.0


def test_wing_fw_3_0_5_gain_fader_paths_include_trim_for_all_surfaces():
    expected_paths = {
        "ch:5": ("/ch/{channel}/fdr", "/ch/{channel}/in/set/trim"),
        "bus:1": ("/bus/{bus}/fdr", "/bus/{bus}/in/set/trim"),
        "main": ("/main/{main}/fdr", "/main/{main}/in/set/trim"),
        "mtx:1": ("/mtx/{matrix}/fdr", "/mtx/{matrix}/in/set/trim"),
    }
    for target_id, expected in expected_paths.items():
        assert _wing_target_path_templates(target_id, "gain_fader") == expected
        assert _wing_target_path_templates(target_id, "master_bus") == expected
    assert _wing_target_path_templates("fx:1", "gain_fader") == ()

    mapped = _with_wing_native_parameter_mappings(
        {"delta_db": -2.25, "bounded_delta_db": 2.25},
        target_id="bus:1",
        method_family="gain_fader",
    )

    assert mapped["wing_gain_delta_source"] == "delta_db"
    assert mapped["wing_gain_target_surface"] == "bus"
    assert mapped["wing_gain_primary_osc_path_suffix"] == "fdr"
    assert mapped["wing_gain_secondary_osc_path_suffix"] == "in/set/trim"
    assert mapped["wing_gain_candidate_write_path_suffixes"] == list(
        WING_GAIN_WRITE_PATH_SUFFIXES
    )
    assert mapped["wing_gain_preferred_write_path_suffix"] == "readback_dependent"
    assert mapped["wing_gain_write_path_selection_policy"] == (
        WING_GAIN_WRITE_PATH_SELECTION_POLICY
    )
    assert mapped["wing_gain_apply_requires_single_selected_write_path"] is True
    assert (
        mapped["wing_gain_apply_requires_current_readback_before_path_selection"]
        is True
    )
    assert mapped["wing_gain_pre_readback_write_path_suffixes_forbidden"] == list(
        WING_GAIN_WRITE_PATH_SUFFIXES
    )
    assert mapped["wing_gain_unselected_write_path_suffixes_forbidden"] is True
    assert mapped["wing_gain_selected_write_path_suffix"] is None
    assert mapped["wing_gain_delta_db"] == -2.25
    assert mapped["wing_gain_delta_value_type"] == (
        "relative_db_delta_requires_readback_to_resolve_absolute_value"
    )
    assert mapped["wing_gain_fader_min_db"] == WING_FADER_MIN_DB
    assert mapped["wing_gain_fader_max_db"] == WING_FADER_MAX_DB
    assert mapped["wing_gain_input_trim_min_db"] == WING_INPUT_TRIM_MIN_DB
    assert mapped["wing_gain_input_trim_max_db"] == WING_INPUT_TRIM_MAX_DB
    assert mapped["wing_gain_absolute_value_requires_readback"] is True
    assert mapped["wing_gain_no_absolute_osc_value_without_readback"] is True

    trim_mapped = _with_wing_native_parameter_mappings(
        {"trim_delta_db": -1.0},
        target_id="main",
        method_family="gain_fader",
    )
    assert trim_mapped["wing_gain_preferred_write_path_suffix"] == "in/set/trim"

    group_mapped = _with_wing_native_parameter_mappings(
        {"group_delta_db": -1.0},
        target_id="bus:1",
        method_family="gain_fader",
    )
    assert group_mapped["wing_gain_preferred_write_path_suffix"] == "fdr"


def test_wing_fw_3_0_5_mute_paths_and_mappings_are_surface_specific():
    expected_paths = {
        "ch:5": ("/ch/{channel}/mute",),
        "bus:1": ("/bus/{bus}/mute",),
        "main": ("/main/{main}/mute",),
        "mtx:1": ("/mtx/{matrix}/mute",),
    }
    for target_id, expected in expected_paths.items():
        assert _wing_target_path_templates(target_id, "mute") == expected
    assert _wing_target_path_templates("fx:1", "mute") == ()

    mapped = _with_wing_native_parameter_mappings(
        {"proposed_muted": True},
        target_id="ch:12",
        method_family="mute",
    )

    assert mapped["wing_mute_source"] == "WING Rack fw3.0.5 strip mute I range 0..1"
    assert mapped["wing_mute_target_surface"] == "channel"
    assert mapped["wing_mute_osc_path_suffix"] == "mute"
    assert mapped["wing_mute_osc_value"] == WING_MUTE_ON_VALUE
    assert mapped["wing_mute_value_type"] == "integer_bool"
    assert mapped["wing_mute_min"] == WING_MUTE_OFF_VALUE
    assert mapped["wing_mute_max"] == WING_MUTE_ON_VALUE
    assert mapped["wing_mute_readback_must_match_osc_value"] is True
    assert mapped["wing_mute_write_key_selection_policy"] == (
        WING_MUTE_WRITE_KEY_SELECTION_POLICY
    )
    assert mapped["wing_mute_apply_only_selected_write_keys"] is True
    assert mapped["wing_mute_selected_write_keys"] == list(WING_MUTE_WRITE_PARAMETER_KEYS)
    assert mapped["wing_mute_forbidden_write_keys"] == []


def test_wing_fw_3_0_5_mute_value_parsing_is_explicit_and_fail_closed():
    false_cases = (False, 0, 0.0, "false", "0", "off", "unmute", "unmuted")
    true_cases = (True, 1, 1.0, "true", "1", "on", "mute", "muted")

    for value in false_cases:
        mapped = _with_wing_native_parameter_mappings(
            {"proposed_muted": value},
            target_id="ch:12",
            method_family="mute",
        )
        assert mapped["wing_mute_osc_value"] == WING_MUTE_OFF_VALUE
        assert mapped["wing_mute_value_type"] == "integer_bool"

    for value in true_cases:
        mapped = _with_wing_native_parameter_mappings(
            {"proposed_muted": value},
            target_id="ch:12",
            method_family="mute",
        )
        assert mapped["wing_mute_osc_value"] == WING_MUTE_ON_VALUE
        assert mapped["wing_mute_value_type"] == "integer_bool"

    ambiguous = _with_wing_native_parameter_mappings(
        {"proposed_muted": "maybe"},
        target_id="ch:12",
        method_family="mute",
    )
    readback = _future_readback_contract("mute", "ch:12", ambiguous)
    applicability = readback["wing_osc_applicability"]

    assert "wing_mute_osc_value" not in ambiguous
    assert ambiguous["wing_mute_value_parse_blocker"] == WING_MUTE_VALUE_BLOCKER
    assert applicability["native_osc_supported"] is False
    assert (
        applicability["native_osc_support_level"]
        == "no_actionable_wing_osc_parameter_value"
    )
    assert applicability["actionable_parameter_keys"] == []
    assert applicability["has_actionable_parameter_value"] is False
    assert applicability["actionable_parameter_blocker"] == (
        WING_ACTIONABLE_PARAMETER_BLOCKER
    )


def test_wing_fw_3_0_5_compressor_detector_parsing_is_explicit_and_fail_closed():
    explicit_cases = {
        "peak": "PEAK",
        "fast_peak": "PEAK",
        "transient_peak": "PEAK",
        "rms": "RMS",
        "program": "RMS",
        "program_rms_peak_blend": "RMS",
    }

    for value, expected in explicit_cases.items():
        mapped = _with_wing_native_parameter_mappings(
            {"threshold_dbfs": -24.0, "detector": value},
            target_id="ch:12",
            method_family="dynamics",
        )
        assert mapped["wing_compressor_detector_osc_path_suffix"] == "dyn/det"
        assert mapped["wing_compressor_detector_osc_value"] == expected
        assert mapped["wing_compressor_detector_value_type"] == "string_enum"

    ambiguous = _with_wing_native_parameter_mappings(
        {"threshold_dbfs": -24.0, "detector": "smart_auto_blend"},
        target_id="ch:12",
        method_family="dynamics",
    )
    readback = _future_readback_contract("dynamics", "ch:12", ambiguous)
    applicability = readback["wing_osc_applicability"]

    assert "wing_compressor_detector_osc_value" not in ambiguous
    assert ambiguous["wing_compressor_detector_parse_blocker"] == (
        WING_COMPRESSOR_DETECTOR_BLOCKER
    )
    assert applicability["native_osc_supported"] is False
    assert applicability["native_osc_support_level"] == (
        "partial_native_mapping_with_unsupported_parameters"
    )
    assert applicability["unsupported_parameter_keys"] == ["detector"]
    assert applicability["unsupported_strategy_reason"] == (
        "strategy_parameters_exceed_wing_fw_3_0_5_native_osc_mapping"
    )
    assert "wing_compressor_detector_mapping_required" not in (
        readback["dynamics_readback"]
    )


def test_wing_fw_3_0_5_dynamics_gain_reduction_targets_are_review_only():
    mapped = _with_wing_native_parameter_mappings(
        {
            "threshold_dbfs": -24.0,
            "ratio": 2.6,
            "target_gain_reduction_db": 3.0,
            "max_gain_reduction_db": 4.2,
        },
        target_id="ch:12",
        method_family="dynamics",
    )
    readback = _future_readback_contract("dynamics", "ch:12", mapped)
    applicability = readback["wing_osc_applicability"]
    dynamics_readback = readback["dynamics_readback"]

    assert applicability["native_osc_supported"] is True
    assert "wing_compressor_threshold_osc_value" in (
        applicability["actionable_parameter_keys"]
    )
    assert "wing_compressor_ratio_osc_value" in (
        applicability["actionable_parameter_keys"]
    )
    assert "target_gain_reduction_db" not in (
        applicability["actionable_parameter_keys"]
    )
    assert "max_gain_reduction_db" not in (
        applicability["actionable_parameter_keys"]
    )
    assert applicability["review_only_parameter_keys"] == [
        "max_gain_reduction_db",
        "target_gain_reduction_db",
    ]
    assert applicability["review_only_parameters_apply_ready"] is False
    assert applicability["review_only_parameter_reason_by_key"] == {
        key: WING_DYNAMICS_REVIEW_ONLY_PARAMETER_REASONS[key]
        for key in ("max_gain_reduction_db", "target_gain_reduction_db")
    }
    assert dynamics_readback["review_only_parameter_keys"] == [
        "max_gain_reduction_db",
        "target_gain_reduction_db",
    ]
    assert dynamics_readback["wing_dynamics_review_only_values"] == {
        "max_gain_reduction_db": 4.2,
        "target_gain_reduction_db": 3.0,
    }
    assert (
        dynamics_readback["wing_dynamics_review_only_parameter_reason_by_key"]
        == applicability["review_only_parameter_reason_by_key"]
    )
    assert dynamics_readback["wing_dynamics_review_only_no_direct_osc_path"] is True
    assert dynamics_readback["wing_dynamics_review_only_not_apply_ready"] is True
    assert (
        dynamics_readback["wing_dynamics_gain_reduction_objective_required"]
        is True
    )
    assert dynamics_readback[
        "wing_dynamics_gain_reduction_objective_parameter_keys"
    ] == ["max_gain_reduction_db", "target_gain_reduction_db"]
    assert dynamics_readback["wing_dynamics_gain_reduction_objective_values"] == {
        "max_gain_reduction_db": 4.2,
        "target_gain_reduction_db": 3.0,
    }
    assert dynamics_readback["wing_dynamics_gain_reduction_target_db"] == 3.0
    assert dynamics_readback["wing_dynamics_gain_reduction_max_db"] == 4.2
    assert dynamics_readback[
        "wing_dynamics_gain_reduction_objective_tolerance_db"
    ] == WING_DYNAMICS_GAIN_REDUCTION_OBJECTIVE_TOLERANCE_DB
    assert dynamics_readback["wing_dynamics_gain_reduction_acceptance_min_db"] == 2.0
    assert dynamics_readback["wing_dynamics_gain_reduction_acceptance_max_db"] == 4.0
    assert dynamics_readback["wing_dynamics_gain_reduction_must_not_exceed_db"] == 4.2
    assert (
        dynamics_readback["wing_dynamics_gain_reduction_meter_readback_path_suffix"]
        == "dyn/gr"
    )
    assert dynamics_readback[
        "wing_dynamics_gain_reduction_meter_readback_path_templates"
    ] == ["/ch/{channel}/dyn/gr"]
    assert dynamics_readback[
        "wing_dynamics_gain_reduction_meter_fallback_path_templates"
    ] == ["/ch/{channel}/dyn/$gr"]
    assert (
        dynamics_readback["wing_dynamics_gain_reduction_meter_value_type"]
        == "read_only_meter_db"
    )
    assert (
        dynamics_readback["wing_dynamics_gain_reduction_meter_is_not_apply_value"]
        is True
    )
    assert dynamics_readback[
        "wing_dynamics_gain_reduction_meter_path_source"
    ] == "wing_client_cached_state_and_operator_product_state"
    assert "dyn/gr" in dynamics_readback["read_before_paths"]
    assert "dyn/gr" in dynamics_readback["read_after_paths"]
    assert (
        dynamics_readback["wing_dynamics_gain_reduction_objective_is_review_only"]
        is True
    )
    assert (
        dynamics_readback["wing_dynamics_gain_reduction_objective_is_not_osc_write"]
        is True
    )
    assert (
        dynamics_readback[
            "wing_dynamics_gain_reduction_objective_no_direct_osc_write_path"
        ]
        is True
    )
    assert (
        dynamics_readback["wing_dynamics_gain_reduction_objective_apply_ready"]
        is False
    )
    assert (
        dynamics_readback[
            "wing_dynamics_gain_reduction_objective_requires_post_apply_meter_readback"
        ]
        is True
    )
    assert (
        dynamics_readback[
            "wing_dynamics_gain_reduction_objective_requires_operator_confirmation"
        ]
        is True
    )
    assert dynamics_readback[
        "wing_dynamics_gain_reduction_objective_live_promotion_status"
    ] == "blocked_until_actual_gain_reduction_meter_readback_and_operator_review"
    assert "actual WING compressor gain-reduction meter" in dynamics_readback[
        "wing_dynamics_gain_reduction_objective_operator_check"
    ]
    assert dynamics_readback[
        "wing_dynamics_gain_reduction_objective_direct_osc_write_keys_excluded"
    ] == ["max_gain_reduction_db", "target_gain_reduction_db"]


def test_wing_fw_3_0_5_fx_level_paths_and_mappings_are_explicit_readback_moves():
    assert _wing_target_path_templates("fx:1", "fx") == (
        "/ch/{channel}/send/{send}/on",
        "/ch/{channel}/send/{send}/lvl",
        "/aux/{aux}/send/{send}/on",
        "/aux/{aux}/send/{send}/lvl",
        "/bus/{bus}/send/{send}/on",
        "/bus/{bus}/send/{send}/lvl",
        "/fx/{fx}/mdl",
        "/fx/{fx}/fxmix",
        "/fx/{fx}/{model_parameter}",
    )

    channel_send = _with_wing_native_parameter_mappings(
        {"send_delta_db": -3.5, "return_delta_db": 0.0},
        target_id="ch:5",
        method_family="fx",
    )
    assert channel_send["wing_fx_send_level_target_surface"] == "channel"
    assert channel_send["wing_fx_send_level_osc_path_suffix"] == "send/{send}/lvl"
    assert channel_send["wing_fx_send_level_delta_db"] == -3.5
    assert channel_send["wing_fx_send_level_value_type"] == (
        "relative_db_delta_requires_send_index_and_current_level_readback"
    )
    assert channel_send["wing_fx_send_level_min_db"] == WING_FX_SEND_LEVEL_MIN_DB
    assert channel_send["wing_fx_send_level_max_db"] == WING_FX_SEND_LEVEL_MAX_DB
    assert channel_send["wing_fx_send_index_required"] is True
    assert channel_send["wing_fx_send_source_binding_required"] is False
    assert "wing_fx_return_delta_db" not in channel_send
    channel_send_contract = _future_readback_contract("fx", "ch:5", channel_send)
    assert channel_send_contract["wing_osc_applicability"][
        "actionable_parameter_keys"
    ] == ["wing_fx_send_level_delta_db"]

    bus_send = _with_wing_native_parameter_mappings(
        {"send_delta_db": -2.0},
        target_id="bus:1",
        method_family="fx",
    )
    assert bus_send["wing_fx_send_level_target_surface"] == "bus"
    assert bus_send["wing_fx_send_source_binding_required"] is False

    aux_send = _with_wing_native_parameter_mappings(
        {"send_delta_db": -2.5},
        target_id="aux:1",
        method_family="fx",
    )
    assert aux_send["wing_fx_send_level_target_surface"] == "aux"
    assert aux_send["wing_fx_send_level_delta_db"] == -2.5
    assert aux_send["wing_fx_send_source_surface_supported"] is True
    assert aux_send["wing_fx_send_source_binding_required"] is False

    fx_return = _with_wing_native_parameter_mappings(
        {"send_delta_db": 0.0, "return_delta_db": -1.0},
        target_id="fx:1",
        method_family="fx",
    )
    assert fx_return["wing_fx_return_level_target_surface"] == "fx"
    assert fx_return["wing_fx_return_target_id"] == "fx:1"
    assert fx_return["wing_fx_return_target_surface"] == "fx"
    assert fx_return["wing_fx_return_level_osc_path_suffix"] == "fxmix"
    assert fx_return["wing_fx_return_delta_db"] == -1.0
    assert 0.0 < fx_return["wing_fx_return_fxmix_multiplier"] < 1.0
    assert fx_return["wing_fx_return_value_type"] == (
        "relative_fxmix_percent_requires_readback_to_resolve_absolute_value"
    )
    assert fx_return["wing_fx_return_min_percent"] == WING_FX_MIX_MIN_PERCENT
    assert fx_return["wing_fx_return_max_percent"] == WING_FX_MIX_MAX_PERCENT
    assert fx_return["wing_fx_return_no_db_return_fader_mapping"] is True
    assert fx_return["wing_fx_return_fx_slot_binding_required"] is False
    assert fx_return["wing_fx_return_fx_slot_apply_ready"] is True
    assert fx_return["wing_fx_return_fx_slot_blocker"] is None
    fx_return_contract = _future_readback_contract("fx", "fx:1", fx_return)
    assert fx_return_contract["wing_osc_applicability"][
        "actionable_parameter_keys"
    ] == ["wing_fx_return_delta_db"]
    assert fx_return_contract["fx_readback"]["wing_fx_return_target_id"] == "fx:1"
    assert fx_return_contract["fx_readback"]["wing_fx_return_target_surface"] == "fx"

    unresolved_send = _with_wing_native_parameter_mappings(
        {"send_delta_db": -4.0},
        target_id="fx:1",
        method_family="fx",
    )
    assert unresolved_send["wing_fx_send_level_target_surface"] == "fx"
    assert unresolved_send["wing_fx_send_source_surface_supported"] is False
    assert unresolved_send["wing_fx_send_source_surface_apply_ready"] is False
    assert unresolved_send["wing_fx_send_source_binding_required"] is True
    assert (
        unresolved_send["wing_fx_send_source_surface_blocker"]
        == WING_FX_SEND_SOURCE_SURFACE_BLOCKER
    )

    for target_id, expected_surface in {
        "fx:1": "fx",
        "main": "main",
        "mtx:1": "matrix",
    }.items():
        mapped = _with_wing_native_parameter_mappings(
            {"send_delta_db": -2.0},
            target_id=target_id,
            method_family="fx",
        )
        readback = _future_readback_contract("fx", target_id, mapped)
        applicability = readback["wing_osc_applicability"]

        assert mapped["wing_fx_send_level_target_surface"] == expected_surface
        assert mapped["wing_fx_send_source_surface_supported"] is False
        assert mapped["wing_fx_send_source_surface_apply_ready"] is False
        assert mapped["wing_fx_send_source_binding_required"] is True
        assert (
            mapped["wing_fx_send_source_surface_blocker"]
            == WING_FX_SEND_SOURCE_SURFACE_BLOCKER
        )
        assert applicability["native_osc_supported"] is False
        assert (
            applicability["native_osc_support_level"]
            == "fx_send_source_surface_unsupported"
        )
        assert applicability["unsupported_strategy_reason"] == (
            "fx_send_level_requires_channel_aux_or_bus_source_surface_on_wing_fw_3_0_5"
        )
        assert readback["fx_readback"]["wing_fx_send_source_surface_supported"] is False
        assert readback["fx_readback"]["wing_fx_send_source_binding_required"] is True


def test_wing_fw_3_0_5_fx_return_requires_fx_slot_surface():
    for target_id, expected_surface in {
        "ch:5": "channel",
        "bus:1": "bus",
        "main": "main",
        "mtx:1": "matrix",
    }.items():
        mapped = _with_wing_native_parameter_mappings(
            {"send_delta_db": 0.0, "return_delta_db": -1.0},
            target_id=target_id,
            method_family="fx",
        )
        readback = _future_readback_contract("fx", target_id, mapped)
        applicability = readback["wing_osc_applicability"]

        assert mapped["wing_fx_return_level_target_surface"] == expected_surface
        assert mapped["wing_fx_return_level_osc_path_suffix"] == "fxmix"
        assert mapped["wing_fx_return_fx_slot_binding_required"] is True
        assert mapped["wing_fx_return_fx_slot_apply_ready"] is False
        assert (
            mapped["wing_fx_return_fx_slot_blocker"]
            == WING_FX_RETURN_SLOT_BINDING_BLOCKER
        )
        assert applicability["native_osc_supported"] is False
        assert (
            applicability["native_osc_support_level"]
            == "fx_return_slot_binding_required"
        )
        assert applicability["unsupported_strategy_reason"] == (
            "fx_return_level_requires_fx_slot_target_surface_on_wing_fw_3_0_5"
        )
        fx_readback = readback["fx_readback"]
        assert fx_readback["wing_fx_return_fx_slot_binding_required"] is True
        assert fx_readback["wing_fx_return_fx_slot_apply_ready"] is False
        assert (
            fx_readback["wing_fx_return_fx_slot_blocker"]
            == WING_FX_RETURN_SLOT_BINDING_BLOCKER
        )

    fx_return = _with_wing_native_parameter_mappings(
        {"send_delta_db": 0.0, "return_delta_db": -1.0},
        target_id="fx:1",
        method_family="fx",
    )
    fx_readback = _future_readback_contract("fx", "fx:1", fx_return)
    assert fx_readback["wing_osc_applicability"]["native_osc_supported"] is True
    assert fx_return["wing_fx_return_target_id"] == "fx:1"
    assert fx_return["wing_fx_return_target_surface"] == "fx"
    assert fx_return["wing_fx_return_fx_slot_binding_required"] is False
    assert fx_return["wing_fx_return_fx_slot_apply_ready"] is True


def test_wing_fw_3_0_5_fx_requires_nonzero_level_delta_or_model_parameter():
    for target_id in ("ch:5", "bus:1", "main", "mtx:1", "fx:1"):
        mapped = _with_wing_native_parameter_mappings(
            {"send_delta_db": 0.0, "return_delta_db": 0.0},
            target_id=target_id,
            method_family="fx",
        )
        readback = _future_readback_contract("fx", target_id, mapped)
        applicability = readback["wing_osc_applicability"]

        assert "wing_fx_send_level_delta_db" not in mapped
        assert "wing_fx_return_delta_db" not in mapped
        assert applicability["native_osc_supported"] is False
        assert (
            applicability["native_osc_support_level"]
            == "fx_no_actionable_wing_osc_parameter"
        )
        assert applicability["unsupported_strategy_reason"] == (
            "fx_strategy_has_no_nonzero_wing_fw_3_0_5_send_return_or_model_parameter"
        )
        assert (
            applicability["fx_requires_nonzero_level_delta_or_model_parameter"]
            is True
        )
        assert (
            applicability["fx_has_nonzero_level_delta_or_model_parameter"]
            is False
        )
        assert (
            applicability["fx_actionable_parameter_blocker"]
            == WING_FX_ACTIONABLE_PARAMETER_BLOCKER
        )
        fx_readback = readback["fx_readback"]
        assert (
            fx_readback["wing_fx_requires_nonzero_level_delta_or_model_parameter"]
            is True
        )
        assert (
            fx_readback["wing_fx_has_nonzero_level_delta_or_model_parameter"]
            is False
        )
        assert (
            fx_readback["wing_fx_actionable_parameter_blocker"]
            == WING_FX_ACTIONABLE_PARAMETER_BLOCKER
        )


def test_wing_fw_3_0_5_path_only_candidates_are_not_actionable_native_corrections():
    cases = (
        ("dynamics", "ch:5", {}, ("wing_compressor_threshold_osc_value",)),
        ("dynamics", "ch:5", {"threshold_dbfs": None}, ("wing_compressor_threshold_osc_value",)),
        ("gate", "ch:5", {}, ("wing_gate_threshold_osc_value", "wing_gate_on_osc_value")),
        ("gate", "ch:5", {"threshold_dbfs": None}, ("wing_gate_threshold_osc_value", "wing_gate_on_osc_value")),
        ("pan", "ch:5", {}, ("wing_pan_osc_value", "wing_width_osc_value")),
        ("gain_fader", "ch:5", {}, ("wing_gain_delta_db",)),
        ("gain_fader", "ch:5", {"delta_db": 0.0}, ("wing_gain_delta_db",)),
        ("mute", "ch:5", {}, ("wing_mute_osc_value",)),
    )

    for method_family, target_id, parameters, forbidden_keys in cases:
        mapped = _with_wing_native_parameter_mappings(
            parameters,
            target_id=target_id,
            method_family=method_family,
        )
        readback = _future_readback_contract(method_family, target_id, mapped)
        applicability = readback["wing_osc_applicability"]

        for key in forbidden_keys:
            assert key not in mapped
        assert applicability["native_osc_supported"] is False
        assert (
            applicability["native_osc_support_level"]
            == "no_actionable_wing_osc_parameter_value"
        )
        assert applicability["unsupported_strategy_reason"] == (
            "strategy_has_no_actionable_wing_fw_3_0_5_parameter_value"
        )
        assert applicability["actionable_parameter_required"] is True
        assert applicability["actionable_parameter_keys"] == []
        assert applicability["has_actionable_parameter_value"] is False
        assert (
            applicability["actionable_parameter_blocker"]
            == WING_ACTIONABLE_PARAMETER_BLOCKER
        )


def test_wing_fw_3_0_5_actionable_value_candidates_remain_native_mapped():
    cases = (
        ("dynamics", "ch:5", {"threshold_dbfs": -20.0}, "wing_compressor_threshold_osc_value"),
        ("dynamics", "ch:5", {"compressor_enabled": True}, "wing_compressor_on_osc_value"),
        ("dynamics", "ch:5", {"compression_mix_percent": 82.0}, "wing_compressor_mix_osc_value"),
        ("dynamics", "ch:5", {"compressor_envelope_curve": "linear"}, "wing_compressor_envelope_curve_osc_value"),
        ("dynamics", "ch:5", {"compressor_auto_timing_enabled": False}, "wing_compressor_auto_osc_value"),
        ("gate", "ch:5", {"threshold_dbfs": -60.0}, "wing_gate_threshold_osc_value"),
        ("gate", "ch:5", {"gate_enabled": True}, "wing_gate_on_osc_value"),
        ("gate", "ch:5", {"accent_percent": 45.0}, "wing_gate_accent_osc_value"),
        ("pan", "ch:5", {"target_pan_percent": 0.0}, "wing_pan_osc_value"),
        ("pan", "ch:5", {"target_width_percent": 0.0}, "wing_width_osc_value"),
        ("gain_fader", "ch:5", {"delta_db": -1.0}, "wing_gain_delta_db"),
        ("gain_fader", "ch:5", {"bounded_delta_db": -1.0}, "wing_gain_delta_db"),
        ("mute", "ch:5", {"proposed_muted": False}, "wing_mute_osc_value"),
    )

    for method_family, target_id, parameters, expected_key in cases:
        mapped = _with_wing_native_parameter_mappings(
            parameters,
            target_id=target_id,
            method_family=method_family,
        )
        readback = _future_readback_contract(method_family, target_id, mapped)
        applicability = readback["wing_osc_applicability"]

        assert expected_key in mapped
        assert applicability["native_osc_supported"] is True
        assert applicability["actionable_parameter_required"] is True
        assert expected_key in applicability["actionable_parameter_keys"]
        assert applicability["has_actionable_parameter_value"] is True
        assert applicability["actionable_parameter_blocker"] is None


def test_wing_fw_3_0_5_gate_accent_mapping_is_bounded_native_osc_value():
    mapped = _with_wing_native_parameter_mappings(
        {"accent_percent": 112.0},
        target_id="ch:5",
        method_family="gate",
    )
    readback = _future_readback_contract("gate", "ch:5", mapped)
    applicability = readback["wing_osc_applicability"]
    gate_readback = readback["gate_expander_readback"]

    assert mapped["wing_gate_accent_source"] == (
        "WING Rack fw3.0.5 gate/acc F range 0..100 percent"
    )
    assert mapped["wing_gate_accent_osc_path_suffix"] == "gate/acc"
    assert mapped["wing_gate_accent_osc_value"] == WING_GATE_ACCENT_MAX_PERCENT
    assert mapped["wing_gate_accent_value_type"] == "float"
    assert mapped["wing_gate_accent_min"] == WING_GATE_ACCENT_MIN_PERCENT
    assert mapped["wing_gate_accent_max"] == WING_GATE_ACCENT_MAX_PERCENT
    assert mapped["wing_gate_accent_quantized_from"] == 112.0
    assert mapped["wing_gate_accent_quantization_error"] == 12.0
    assert mapped["gate_enabled"] is True
    assert mapped["wing_gate_on_osc_path_suffix"] == "gate/on"
    assert mapped["wing_gate_on_osc_value"] == WING_GATE_ON_MAX
    assert mapped["wing_gate_on_value_type"] == "integer_bool"
    assert mapped["wing_gate_on_min"] == WING_GATE_ON_MIN
    assert mapped["wing_gate_on_max"] == WING_GATE_ON_MAX
    assert mapped["wing_gate_on_original_value"] is True
    assert applicability["native_osc_supported"] is True
    assert "wing_gate_on_osc_value" in applicability["actionable_parameter_keys"]
    assert "wing_gate_accent_osc_value" in applicability["actionable_parameter_keys"]
    assert gate_readback["wing_gate_on_mapping_required"] is True
    assert gate_readback["wing_gate_on_osc_path_suffix"] == "gate/on"
    assert gate_readback["wing_gate_on_osc_value"] == WING_GATE_ON_MAX
    assert gate_readback["wing_gate_on_readback_must_match_osc_value"] is True
    assert gate_readback["wing_gate_accent_mapping_required"] is True
    assert gate_readback["wing_gate_accent_osc_path_suffix"] == "gate/acc"
    assert gate_readback["wing_gate_accent_osc_value"] == WING_GATE_ACCENT_MAX_PERCENT
    assert gate_readback["wing_gate_accent_readback_must_match_osc_value"] is True


def test_wing_fw_3_0_5_gate_sidechain_filter_mapping_is_native_osc_value():
    mapped = _with_wing_native_parameter_mappings(
        {
            "gate_sidechain_filter_type": "bandpass",
            "gate_sidechain_frequency_hz": 160.0,
            "gate_sidechain_q": 2.75,
            "gate_sidechain_source": "self",
            "gate_sidechain_tap": "input",
        },
        target_id="ch:5",
        method_family="gate",
    )
    readback = _future_readback_contract("gate", "ch:5", mapped)
    applicability = readback["wing_osc_applicability"]
    gate_readback = readback["gate_expander_readback"]

    assert mapped["gate_enabled"] is True
    assert mapped["wing_gate_sidechain_type_osc_path_suffix"] == "gatesc/type"
    assert mapped["wing_gate_sidechain_type_osc_value"] == "BP"
    assert mapped["wing_gate_sidechain_type_value_type"] == "string_enum"
    assert mapped["wing_gate_sidechain_type_allowed_values"] == list(
        WING_GATE_SIDECHAIN_TYPE_VALUES
    )
    assert mapped["wing_gate_sidechain_frequency_osc_path_suffix"] == "gatesc/f"
    assert mapped["wing_gate_sidechain_frequency_osc_value"] == 160.0
    assert mapped["wing_gate_sidechain_frequency_min_hz"] == (
        WING_GATE_SIDECHAIN_FREQ_MIN_HZ
    )
    assert mapped["wing_gate_sidechain_frequency_max_hz"] == (
        WING_GATE_SIDECHAIN_FREQ_MAX_HZ
    )
    assert mapped["wing_gate_sidechain_q_osc_path_suffix"] == "gatesc/q"
    assert mapped["wing_gate_sidechain_q_osc_value"] == 2.75
    assert mapped["wing_gate_sidechain_q_min"] == WING_GATE_SIDECHAIN_Q_MIN
    assert mapped["wing_gate_sidechain_q_max"] == WING_GATE_SIDECHAIN_Q_MAX
    assert mapped["wing_gate_sidechain_src_osc_path_suffix"] == "gatesc/src"
    assert mapped["wing_gate_sidechain_src_osc_value"] == "SELF"
    assert mapped["wing_gate_sidechain_src_allowed_values"] == list(
        WING_GATE_SIDECHAIN_SOURCE_VALUES
    )
    assert mapped["wing_gate_sidechain_tap_osc_path_suffix"] == "gatesc/tap"
    assert mapped["wing_gate_sidechain_tap_osc_value"] == "IN"
    assert mapped["wing_gate_sidechain_tap_allowed_values"] == list(
        WING_GATE_SIDECHAIN_TAP_VALUES
    )
    assert "wing_gate_sidechain_type_osc_value" in applicability[
        "actionable_parameter_keys"
    ]
    assert "wing_gate_sidechain_frequency_osc_value" in applicability[
        "actionable_parameter_keys"
    ]
    assert "wing_gate_sidechain_q_osc_value" in applicability[
        "actionable_parameter_keys"
    ]
    assert "wing_gate_sidechain_src_osc_value" in applicability[
        "actionable_parameter_keys"
    ]
    assert "wing_gate_sidechain_tap_osc_value" in applicability[
        "actionable_parameter_keys"
    ]
    assert gate_readback["wing_gate_sidechain_mapping_required"] is True
    assert gate_readback["wing_gate_sidechain_type_osc_path_suffix"] == "gatesc/type"
    assert gate_readback["wing_gate_sidechain_type_osc_value"] == "BP"
    assert gate_readback["wing_gate_sidechain_frequency_osc_path_suffix"] == "gatesc/f"
    assert gate_readback["wing_gate_sidechain_frequency_osc_value"] == 160.0
    assert gate_readback["wing_gate_sidechain_q_osc_path_suffix"] == "gatesc/q"
    assert gate_readback["wing_gate_sidechain_q_osc_value"] == 2.75
    assert gate_readback["wing_gate_sidechain_src_osc_path_suffix"] == "gatesc/src"
    assert gate_readback["wing_gate_sidechain_src_osc_value"] == "SELF"
    assert gate_readback["wing_gate_sidechain_tap_osc_path_suffix"] == "gatesc/tap"
    assert gate_readback["wing_gate_sidechain_tap_osc_value"] == "IN"
    assert (
        gate_readback["wing_gate_sidechain_readback_must_match_osc_value"]
        is True
    )
    assert "gatesc/type" in gate_readback["read_before_paths"]
    assert "gatesc/f" in gate_readback["read_after_paths"]


def test_wing_fw_3_0_5_gate_enabled_requires_explicit_bool_or_alias():
    mapped = _with_wing_native_parameter_mappings(
        {"gate_enabled": "maybe", "threshold_dbfs": -48.0},
        target_id="ch:5",
        method_family="gate",
    )
    readback = _future_readback_contract("gate", "ch:5", mapped)
    applicability = readback["wing_osc_applicability"]

    assert "wing_gate_on_osc_value" not in mapped
    assert mapped["wing_gate_on_parse_blocker"] == WING_GATE_ON_BLOCKER
    assert mapped["wing_gate_on_osc_path_suffix"] == "gate/on"
    assert mapped["wing_gate_on_value_type"] == "integer_bool"
    assert applicability["native_osc_supported"] is False
    assert applicability["unsupported_parameter_keys"] == ["gate_enabled"]
    assert applicability["unsupported_strategy_reason"] == (
        "strategy_parameters_exceed_wing_fw_3_0_5_native_osc_mapping"
    )


def test_wing_fw_3_0_5_gate_sidechain_type_requires_verified_enum():
    mapped = _with_wing_native_parameter_mappings(
        {
            "gate_sidechain_filter_type": "sharp_magic_filter",
            "gate_sidechain_frequency_hz": 160.0,
            "gate_sidechain_q": 2.2,
        },
        target_id="ch:5",
        method_family="gate",
    )
    readback = _future_readback_contract("gate", "ch:5", mapped)
    applicability = readback["wing_osc_applicability"]

    assert "wing_gate_sidechain_type_osc_value" not in mapped
    assert mapped["wing_gate_sidechain_type_parse_blocker"] == (
        "wing_gate_sidechain_filter_type_requires_verified_wing_enum"
    )
    assert mapped["wing_gate_sidechain_type_osc_path_suffix"] == "gatesc/type"
    assert applicability["native_osc_supported"] is False
    assert applicability["unsupported_parameter_keys"] == [
        "gate_sidechain_filter_type"
    ]


def test_wing_fw_3_0_5_gate_hysteresis_advisory_is_review_only():
    mapped = _with_wing_native_parameter_mappings(
        {
            "threshold_dbfs": -48.0,
            "range_db": 8.0,
            "ratio": "1:2",
            "post_gate_hysteresis_review_db": 3.0,
        },
        target_id="ch:5",
        method_family="expander",
    )
    readback = _future_readback_contract("expander", "ch:5", mapped)
    applicability = readback["wing_osc_applicability"]
    gate_readback = readback["gate_expander_readback"]

    assert applicability["native_osc_supported"] is True
    assert "wing_gate_threshold_osc_value" in applicability["actionable_parameter_keys"]
    assert "wing_gate_range_osc_value" in applicability["actionable_parameter_keys"]
    assert "wing_gate_ratio_osc_value" in applicability["actionable_parameter_keys"]
    assert "post_gate_hysteresis_review_db" not in (
        applicability["actionable_parameter_keys"]
    )
    assert applicability["review_only_parameter_keys"] == [
        "post_gate_hysteresis_review_db"
    ]
    assert applicability["review_only_parameters_apply_ready"] is False
    assert (
        "no verified gate hysteresis write path"
        in applicability["review_only_parameter_reason_by_key"][
            "post_gate_hysteresis_review_db"
        ]
    )
    assert gate_readback["hysteresis_is_review_only_without_verified_wing_path"] is True
    assert gate_readback["review_only_parameter_keys"] == [
        "post_gate_hysteresis_review_db"
    ]
    assert gate_readback["wing_gate_hysteresis_review_only_value_db"] == 3.0
    assert gate_readback["wing_gate_hysteresis_review_only_no_osc_path"] is True
    assert gate_readback["wing_gate_hysteresis_review_only_not_apply_ready"] is True
    assert (
        gate_readback["wing_gate_hysteresis_review_tolerance_db"]
        == WING_GATE_HYSTERESIS_REVIEW_TOLERANCE_DB
    )
    assert gate_readback["wing_gate_hysteresis_review_acceptance_min_db"] == 2.5
    assert gate_readback["wing_gate_hysteresis_review_acceptance_max_db"] == 3.5
    assert gate_readback["wing_gate_hysteresis_review_must_not_exceed_db"] == (
        SOUNDCHECK_GATE_EXPANDER_MAX_HYSTERESIS_REVIEW_DB
    )
    assert gate_readback["wing_gate_hysteresis_review_value_type"] == (
        "operator_review_db"
    )
    assert gate_readback["wing_gate_hysteresis_review_is_not_apply_value"] is True
    assert gate_readback["wing_gate_hysteresis_review_control_surface"] == (
        "operator_console_ui_only"
    )
    assert gate_readback["wing_gate_hysteresis_review_path_source"] == (
        "no_verified_wing_fw_3_0_5_osc_path"
    )
    assert gate_readback[
        "wing_gate_hysteresis_review_forbidden_path_suffixes"
    ] == ["gate/hysteresis", "gate/hyst"]
    assert gate_readback["wing_gate_hysteresis_review_requires_audio_recheck"] is True
    assert (
        gate_readback["wing_gate_hysteresis_review_requires_tail_chatter_recheck"]
        is True
    )
    assert "do not write a hysteresis OSC value" in gate_readback[
        "wing_gate_hysteresis_review_operator_check"
    ]
    assert "gate/hysteresis" not in gate_readback["read_before_paths"]
    assert "gate/hysteresis" not in gate_readback["read_after_paths"]
    assert "gate/hyst" not in gate_readback["read_before_paths"]
    assert "gate/hyst" not in gate_readback["read_after_paths"]


def test_wing_fw_3_0_5_dynamics_timing_and_gain_mappings_are_bounded():
    mapped = _with_wing_native_parameter_mappings(
        {
            "threshold_dbfs": -20.0,
            "attack_ms": -5.0,
            "hold_ms": 250.0,
            "release_ms": 2.0,
            "makeup_gain_db": 18.0,
            "compression_mix_percent": 140.0,
            "compressor_envelope_curve": "smooth_log",
            "compressor_auto_timing_enabled": False,
        },
        target_id="bus:1",
        method_family="dynamics",
    )

    assert mapped["wing_compressor_attack_osc_path_suffix"] == "dyn/att"
    assert mapped["wing_compressor_model_osc_path_suffix"] == "dyn/mdl"
    assert (
        mapped["wing_compressor_model_readback_must_equal"]
        == WING_DYNAMICS_REQUIRED_MODEL
    )
    assert mapped["wing_compressor_model_readback_required"] is True
    assert mapped["compressor_enabled"] is True
    assert mapped["wing_compressor_on_osc_path_suffix"] == "dyn/on"
    assert mapped["wing_compressor_on_osc_value"] == WING_COMPRESSOR_ON_MAX
    assert mapped["wing_compressor_on_value_type"] == "integer_bool"
    assert mapped["wing_compressor_on_min"] == WING_COMPRESSOR_ON_MIN
    assert mapped["wing_compressor_on_max"] == WING_COMPRESSOR_ON_MAX
    assert mapped["wing_compressor_on_original_value"] is True
    assert mapped["wing_compressor_on_mapping_policy"] == (
        "enable_compressor_when_strategy_writes_native_wing_dynamics"
    )
    assert mapped["wing_compressor_attack_osc_value"] == WING_COMPRESSOR_ATTACK_MIN_MS
    assert mapped["wing_compressor_attack_quantization_error"] == 5.0
    assert mapped["wing_compressor_hold_osc_path_suffix"] == "dyn/hld"
    assert mapped["wing_compressor_hold_osc_value"] == WING_COMPRESSOR_HOLD_MAX_MS
    assert mapped["wing_compressor_hold_quantization_error"] == 50.0
    assert mapped["wing_compressor_release_osc_path_suffix"] == "dyn/rel"
    assert mapped["wing_compressor_release_osc_value"] == WING_COMPRESSOR_RELEASE_MIN_MS
    assert mapped["wing_compressor_release_quantization_error"] == 2.0
    assert mapped["wing_compressor_makeup_gain_osc_path_suffix"] == "dyn/gain"
    assert (
        mapped["wing_compressor_makeup_gain_osc_value"]
        == WING_COMPRESSOR_MAKEUP_GAIN_MAX_DB
    )
    assert mapped["wing_compressor_makeup_gain_quantization_error"] == 6.0
    assert mapped["wing_compressor_mix_osc_path_suffix"] == "dyn/mix"
    assert mapped["wing_compressor_mix_osc_value"] == WING_COMPRESSOR_MIX_MAX_PERCENT
    assert mapped["wing_compressor_mix_value_type"] == "float"
    assert mapped["wing_compressor_mix_min"] == WING_COMPRESSOR_MIX_MIN_PERCENT
    assert mapped["wing_compressor_mix_max"] == WING_COMPRESSOR_MIX_MAX_PERCENT
    assert mapped["wing_compressor_mix_quantized_from"] == 140.0
    assert mapped["wing_compressor_mix_quantization_error"] == 40.0
    assert mapped["wing_compressor_envelope_curve_osc_path_suffix"] == "dyn/env"
    assert mapped["wing_compressor_envelope_curve_osc_value"] == "LOG"
    assert mapped["wing_compressor_envelope_curve_value_type"] == "string_enum"
    assert (
        mapped["wing_compressor_envelope_curve_allowed_values"]
        == list(WING_COMPRESSOR_ENVELOPE_CURVE_VALUES)
    )
    assert mapped["wing_compressor_envelope_curve_original_value"] == "smooth_log"
    assert mapped["wing_compressor_auto_osc_path_suffix"] == "dyn/auto"
    assert mapped["wing_compressor_auto_osc_value"] == WING_COMPRESSOR_AUTO_MIN
    assert mapped["wing_compressor_auto_value_type"] == "integer_bool"
    assert mapped["wing_compressor_auto_min"] == WING_COMPRESSOR_AUTO_MIN
    assert mapped["wing_compressor_auto_max"] == WING_COMPRESSOR_AUTO_MAX
    assert mapped["wing_compressor_auto_original_value"] is False
    assert mapped["wing_compressor_auto_mapping_policy"] == (
        "manual_timing_disables_wing_auto_when_engine_calculates_attack_release"
    )
    assert mapped["wing_compressor_attack_max"] == WING_COMPRESSOR_ATTACK_MAX_MS
    assert mapped["wing_compressor_hold_min"] == WING_COMPRESSOR_HOLD_MIN_MS
    assert mapped["wing_compressor_release_max"] == WING_COMPRESSOR_RELEASE_MAX_MS
    assert (
        mapped["wing_compressor_makeup_gain_min"]
        == WING_COMPRESSOR_MAKEUP_GAIN_MIN_DB
    )


def test_wing_fw_3_0_5_compressor_crossover_mapping_is_native_osc_value():
    mapped = _with_wing_native_parameter_mappings(
        {
            "compressor_crossover_type": "full-band",
            "compressor_crossover_depth_db": -3.5,
            "compressor_crossover_frequency_hz": 1000.0,
        },
        target_id="ch:5",
        method_family="dynamics",
    )
    readback = _future_readback_contract("dynamics", "ch:5", mapped)
    applicability = readback["wing_osc_applicability"]
    dynamics_readback = readback["dynamics_readback"]

    assert mapped["compressor_enabled"] is True
    assert mapped["wing_compressor_crossover_type_osc_path_suffix"] == "dynxo/type"
    assert mapped["wing_compressor_crossover_type_osc_value"] == "OFF"
    assert mapped["wing_compressor_crossover_type_value_type"] == "string_enum"
    assert mapped["wing_compressor_crossover_type_allowed_values"] == list(
        WING_COMPRESSOR_CROSSOVER_TYPE_VALUES
    )
    assert mapped["wing_compressor_crossover_depth_osc_path_suffix"] == "dynxo/depth"
    assert mapped["wing_compressor_crossover_depth_osc_value"] == 0.0
    assert mapped["wing_compressor_crossover_depth_min_db"] == (
        WING_COMPRESSOR_CROSSOVER_DEPTH_MIN_DB
    )
    assert mapped["wing_compressor_crossover_depth_max_db"] == (
        WING_COMPRESSOR_CROSSOVER_DEPTH_MAX_DB
    )
    assert mapped["wing_compressor_crossover_depth_quantized_from"] == -3.5
    assert mapped["wing_compressor_crossover_frequency_osc_path_suffix"] == "dynxo/f"
    assert mapped["wing_compressor_crossover_frequency_osc_value"] == 1000.0
    assert mapped["wing_compressor_crossover_frequency_min_hz"] == (
        WING_COMPRESSOR_CROSSOVER_FREQ_MIN_HZ
    )
    assert mapped["wing_compressor_crossover_frequency_max_hz"] == (
        WING_COMPRESSOR_CROSSOVER_FREQ_MAX_HZ
    )
    assert "wing_compressor_crossover_type_osc_value" in applicability[
        "actionable_parameter_keys"
    ]
    assert "wing_compressor_crossover_depth_osc_value" in applicability[
        "actionable_parameter_keys"
    ]
    assert "wing_compressor_crossover_frequency_osc_value" in applicability[
        "actionable_parameter_keys"
    ]
    assert dynamics_readback["wing_compressor_crossover_mapping_required"] is True
    assert dynamics_readback["wing_compressor_crossover_type_osc_path_suffix"] == (
        "dynxo/type"
    )
    assert dynamics_readback["wing_compressor_crossover_type_osc_value"] == "OFF"
    assert dynamics_readback["wing_compressor_crossover_depth_osc_path_suffix"] == (
        "dynxo/depth"
    )
    assert dynamics_readback["wing_compressor_crossover_depth_osc_value"] == 0.0
    assert dynamics_readback[
        "wing_compressor_crossover_frequency_osc_path_suffix"
    ] == "dynxo/f"
    assert dynamics_readback[
        "wing_compressor_crossover_frequency_osc_value"
    ] == 1000.0
    assert (
        dynamics_readback[
            "wing_compressor_crossover_readback_must_match_osc_value"
        ]
        is True
    )
    assert "dynxo/type" in dynamics_readback["read_before_paths"]
    assert "dynxo/f" in dynamics_readback["read_after_paths"]


def test_wing_fw_3_0_5_compressor_crossover_type_requires_verified_enum():
    mapped = _with_wing_native_parameter_mappings(
        {
            "compressor_crossover_type": "magic_xover",
            "compressor_crossover_depth_db": 6.0,
            "compressor_crossover_frequency_hz": 1000.0,
        },
        target_id="ch:5",
        method_family="dynamics",
    )
    readback = _future_readback_contract("dynamics", "ch:5", mapped)
    applicability = readback["wing_osc_applicability"]

    assert "wing_compressor_crossover_type_osc_value" not in mapped
    assert (
        mapped["wing_compressor_crossover_type_parse_blocker"]
        == WING_COMPRESSOR_CROSSOVER_TYPE_BLOCKER
    )
    assert mapped["wing_compressor_crossover_type_osc_path_suffix"] == "dynxo/type"
    assert applicability["native_osc_supported"] is False
    assert applicability["unsupported_parameter_keys"] == [
        "compressor_crossover_type"
    ]


def test_wing_fw_3_0_5_compressor_sidechain_filter_mapping_is_native_osc_value():
    mapped = _with_wing_native_parameter_mappings(
        {
            "compressor_sidechain_filter_type": "hpf",
            "compressor_sidechain_frequency_hz": 120.0,
            "compressor_sidechain_q": 1.0,
            "compressor_sidechain_source": "self",
            "compressor_sidechain_tap": "input",
        },
        target_id="ch:5",
        method_family="dynamics",
    )
    readback = _future_readback_contract("dynamics", "ch:5", mapped)
    applicability = readback["wing_osc_applicability"]
    dynamics_readback = readback["dynamics_readback"]

    assert mapped["compressor_enabled"] is True
    assert mapped["wing_compressor_sidechain_type_osc_path_suffix"] == "dynsc/type"
    assert mapped["wing_compressor_sidechain_type_osc_value"] == "HP12"
    assert mapped["wing_compressor_sidechain_type_value_type"] == "string_enum"
    assert mapped["wing_compressor_sidechain_type_allowed_values"] == list(
        WING_COMPRESSOR_SIDECHAIN_TYPE_VALUES
    )
    assert mapped["wing_compressor_sidechain_frequency_osc_path_suffix"] == "dynsc/f"
    assert mapped["wing_compressor_sidechain_frequency_osc_value"] == 120.0
    assert mapped["wing_compressor_sidechain_frequency_min_hz"] == (
        WING_COMPRESSOR_SIDECHAIN_FREQ_MIN_HZ
    )
    assert mapped["wing_compressor_sidechain_frequency_max_hz"] == (
        WING_COMPRESSOR_SIDECHAIN_FREQ_MAX_HZ
    )
    assert mapped["wing_compressor_sidechain_q_osc_path_suffix"] == "dynsc/q"
    assert mapped["wing_compressor_sidechain_q_osc_value"] == 1.0
    assert mapped["wing_compressor_sidechain_q_min"] == (
        WING_COMPRESSOR_SIDECHAIN_Q_MIN
    )
    assert mapped["wing_compressor_sidechain_q_max"] == (
        WING_COMPRESSOR_SIDECHAIN_Q_MAX
    )
    assert mapped["wing_compressor_sidechain_src_osc_path_suffix"] == "dynsc/src"
    assert mapped["wing_compressor_sidechain_src_osc_value"] == "SELF"
    assert mapped["wing_compressor_sidechain_tap_osc_path_suffix"] == "dynsc/tap"
    assert mapped["wing_compressor_sidechain_tap_osc_value"] == "IN"
    assert "wing_compressor_sidechain_type_osc_value" in applicability[
        "actionable_parameter_keys"
    ]
    assert "wing_compressor_sidechain_frequency_osc_value" in applicability[
        "actionable_parameter_keys"
    ]
    assert "wing_compressor_sidechain_q_osc_value" in applicability[
        "actionable_parameter_keys"
    ]
    assert "wing_compressor_sidechain_src_osc_value" in applicability[
        "actionable_parameter_keys"
    ]
    assert "wing_compressor_sidechain_tap_osc_value" in applicability[
        "actionable_parameter_keys"
    ]
    assert dynamics_readback["wing_compressor_sidechain_mapping_required"] is True
    assert dynamics_readback["wing_compressor_sidechain_type_osc_path_suffix"] == (
        "dynsc/type"
    )
    assert dynamics_readback["wing_compressor_sidechain_type_osc_value"] == "HP12"
    assert dynamics_readback[
        "wing_compressor_sidechain_frequency_osc_path_suffix"
    ] == "dynsc/f"
    assert dynamics_readback["wing_compressor_sidechain_frequency_osc_value"] == 120.0
    assert dynamics_readback["wing_compressor_sidechain_q_osc_path_suffix"] == (
        "dynsc/q"
    )
    assert dynamics_readback["wing_compressor_sidechain_q_osc_value"] == 1.0
    assert dynamics_readback["wing_compressor_sidechain_src_osc_path_suffix"] == (
        "dynsc/src"
    )
    assert dynamics_readback["wing_compressor_sidechain_src_osc_value"] == "SELF"
    assert dynamics_readback["wing_compressor_sidechain_tap_osc_path_suffix"] == (
        "dynsc/tap"
    )
    assert dynamics_readback["wing_compressor_sidechain_tap_osc_value"] == "IN"
    assert (
        dynamics_readback[
            "wing_compressor_sidechain_readback_must_match_osc_value"
        ]
        is True
    )
    assert "dynsc/type" in dynamics_readback["read_before_paths"]
    assert "dynsc/tap" in dynamics_readback["read_after_paths"]


def test_wing_fw_3_0_5_compressor_sidechain_source_and_tap_follow_surface():
    mapped = _with_wing_native_parameter_mappings(
        {
            "compressor_sidechain_filter_type": "bp",
            "compressor_sidechain_frequency_hz": 250.0,
            "compressor_sidechain_q": 2.0,
            "compressor_sidechain_source": "aux3",
            "compressor_sidechain_tap": "eq",
        },
        target_id="bus:1",
        method_family="dynamics",
    )

    assert mapped["wing_compressor_sidechain_target_surface"] == "bus"
    assert mapped["wing_compressor_sidechain_type_osc_value"] == "BP"
    assert mapped["wing_compressor_sidechain_src_osc_value"] == "AUX.3"
    assert mapped["wing_compressor_sidechain_tap_osc_value"] == "EQ"
    assert mapped["wing_compressor_sidechain_src_osc_value"] in (
        WING_COMPRESSOR_SIDECHAIN_SOURCE_VALUES
    )
    assert mapped["wing_compressor_sidechain_tap_osc_value"] in (
        WING_COMPRESSOR_SIDECHAIN_TAP_VALUES
    )
    assert "CH.1" not in mapped["wing_compressor_sidechain_src_allowed_values"]
    assert "IN" not in mapped["wing_compressor_sidechain_tap_allowed_values"]


def test_wing_fw_3_0_5_compressor_sidechain_type_requires_verified_enum():
    mapped = _with_wing_native_parameter_mappings(
        {
            "compressor_sidechain_filter_type": "sharp_magic_filter",
            "compressor_sidechain_frequency_hz": 120.0,
            "compressor_sidechain_q": 1.0,
        },
        target_id="ch:5",
        method_family="dynamics",
    )
    readback = _future_readback_contract("dynamics", "ch:5", mapped)
    applicability = readback["wing_osc_applicability"]

    assert "wing_compressor_sidechain_type_osc_value" not in mapped
    assert mapped["wing_compressor_sidechain_type_parse_blocker"] == (
        "wing_compressor_sidechain_filter_type_requires_verified_wing_enum"
    )
    assert mapped["wing_compressor_sidechain_type_osc_path_suffix"] == "dynsc/type"
    assert applicability["native_osc_supported"] is False
    assert applicability["unsupported_parameter_keys"] == [
        "compressor_sidechain_filter_type"
    ]


def test_wing_fw_3_0_5_dynamics_envelope_curve_requires_explicit_lin_or_log_alias():
    mapped = _with_wing_native_parameter_mappings(
        {"compressor_envelope_curve": "musical"},
        target_id="ch:5",
        method_family="dynamics",
    )
    readback = _future_readback_contract("dynamics", "ch:5", mapped)
    applicability = readback["wing_osc_applicability"]

    assert "wing_compressor_envelope_curve_osc_value" not in mapped
    assert (
        mapped["wing_compressor_envelope_curve_parse_blocker"]
        == WING_COMPRESSOR_ENVELOPE_CURVE_BLOCKER
    )
    assert applicability["native_osc_supported"] is False
    assert applicability["unsupported_parameter_keys"] == [
        "compressor_envelope_curve"
    ]
    assert applicability["unsupported_strategy_reason"] == (
        "strategy_parameters_exceed_wing_fw_3_0_5_native_osc_mapping"
    )


def test_wing_fw_3_0_5_compressor_enabled_requires_explicit_bool_or_alias():
    mapped = _with_wing_native_parameter_mappings(
        {"compressor_enabled": "maybe", "threshold_dbfs": -20.0},
        target_id="ch:5",
        method_family="dynamics",
    )
    readback = _future_readback_contract("dynamics", "ch:5", mapped)
    applicability = readback["wing_osc_applicability"]

    assert "wing_compressor_on_osc_value" not in mapped
    assert mapped["wing_compressor_on_parse_blocker"] == WING_COMPRESSOR_ON_BLOCKER
    assert mapped["wing_compressor_on_osc_path_suffix"] == "dyn/on"
    assert mapped["wing_compressor_on_value_type"] == "integer_bool"
    assert applicability["native_osc_supported"] is False
    assert applicability["unsupported_parameter_keys"] == ["compressor_enabled"]
    assert applicability["unsupported_strategy_reason"] == (
        "strategy_parameters_exceed_wing_fw_3_0_5_native_osc_mapping"
    )


def test_wing_fw_3_0_5_compressor_auto_timing_requires_explicit_bool_or_alias():
    mapped = _with_wing_native_parameter_mappings(
        {"compressor_auto_timing_enabled": "maybe"},
        target_id="ch:5",
        method_family="dynamics",
    )
    readback = _future_readback_contract("dynamics", "ch:5", mapped)
    applicability = readback["wing_osc_applicability"]

    assert "wing_compressor_auto_osc_value" not in mapped
    assert mapped["wing_compressor_auto_parse_blocker"] == WING_COMPRESSOR_AUTO_BLOCKER
    assert mapped["wing_compressor_auto_osc_path_suffix"] == "dyn/auto"
    assert mapped["wing_compressor_auto_value_type"] == "integer_bool"
    assert applicability["native_osc_supported"] is False
    assert applicability["unsupported_parameter_keys"] == [
        "compressor_auto_timing_enabled"
    ]
    assert applicability["unsupported_strategy_reason"] == (
        "strategy_parameters_exceed_wing_fw_3_0_5_native_osc_mapping"
    )


def test_smart_dsp_strategy_module_is_transport_free():
    source = Path("backend/smart_dsp_strategy_engine.py").read_text()

    forbidden = [
        "WingClient",
        "EnhancedOSCClient",
        ".send(",
        "mixer_client",
        "set_fader",
        "set_gain",
        "set_eq",
        "set_compressor",
        "apply_operator_proposal",
    ]
    for token in forbidden:
        assert token not in source


def test_normalizers_classify_modes_and_sources():
    assert normalize_mode("concert") == "live"
    assert normalize_mode("sound check") == "soundcheck"
    assert classify_source_family("Lead Vox") == "vocal"
    assert classify_source_family("KICK IN") == "kick"
    assert classify_source_family("Main PA") == "master"
    assert classify_musical_role("Lead Vox", source_family="vocal") == "lead"
    assert classify_musical_role("backing_vocal", "BGV 1", "vocal") == "support"
    assert classify_musical_role("kick_in", source_family="kick") == "low_end_anchor"


def test_muddy_vocal_soundcheck_selects_corrective_cut_with_alternatives():
    recommendation = analyze_smart_dsp_strategy(
        _analysis(muddiness_score=0.82, level_plane_error_lu=1.0)
    )

    assert recommendation.status == "proposal_only"
    assert recommendation.mode == "soundcheck"
    assert recommendation.source_family == "vocal"
    assert recommendation.selected_strategy.strategy_id == "low_mid_corrective_cut"
    assert recommendation.selected_strategy.method_family == "eq"
    assert recommendation.selected_strategy.parameters["gain_db"] < 0.0
    assert recommendation.selected_strategy.parameters["frequency_hz"] == 260.0
    expected_eq_apply_values = {
        "wing_eq_on_osc_value": 1,
        "wing_eq_mix_osc_value": WING_EQ_MIX_BASELINE_PERCENT,
        "wing_eq_band_operations": [
            {
                "band_readback_required": True,
                "filter_family": "parametric_band",
                "filter_type": "peaking",
                "frequency_hz": 260.0,
                "gain_db": -3.2,
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
        ],
    }
    assert recommendation.alternatives
    assert "high_pass_cleanup" in {
        candidate.strategy_id for candidate in recommendation.alternatives
    }
    assert recommendation.evidence["strategy_search"]["candidate_count"] >= 2
    assert recommendation.evidence["current_smart_dsp_context_cache_no_live_authority"] is True
    assert "Alternatives scored:" in recommendation.operator_explanation
    assert "Alternative tradeoffs:" in recommendation.operator_explanation
    assert "Selected WING OSC: Behringer WING Rack fw3.0.5" in (
        recommendation.operator_explanation
    )
    assert "native_supported=True" in recommendation.operator_explanation
    assert "live_write=False" in recommendation.operator_explanation
    expected_eq_readback_paths = [
        "eq/on",
        "eq/mdl",
        "eq/mix",
        "eq/{band}f",
        "eq/{band}g",
        "eq/{band}q",
    ]
    assert (
        "Selected readback paths: "
        "eq/on, eq/mdl, eq/mix, eq/{band}f, eq/{band}g, eq/{band}q."
    ) in recommendation.operator_explanation
    assert (
        "Selected readback status: "
        "wing_osc_readback_paths_declared_before_live_promotion."
    ) in recommendation.operator_explanation
    search = recommendation.evidence["strategy_search"]
    assert search["selected_rationale"]["strategy_id"] == "low_mid_corrective_cut"
    _assert_weighted_score_summary(
        search["selected_rationale"]["weighted_score_summary"],
        mode="soundcheck",
        expected_total_score=recommendation.selected_strategy.total_score,
    )
    assert search["selected_rationale"]["wing_native_strategy_fit"]["fit_score"] >= 0.0
    assert search["selected_rationale"]["wing_native_strategy_fit"]["fit_status"]
    _assert_dsp_parameter_calculation_summary(
        search["selected_rationale"]["dsp_parameter_calculation_summary"],
        strategy_id="low_mid_corrective_cut",
        method_family="eq",
    )
    _assert_problem_action_decision_summary(
        search["selected_rationale"]["problem_action_decision_summary"],
        strategy_id="low_mid_corrective_cut",
        method_family="eq",
        expected_action_intent="static_spectral_correction",
        expected_method_families=("eq",),
    )
    assert "gain_db" in search["selected_rationale"][
        "dsp_parameter_calculation_summary"
    ]["calculated_parameter_keys"]
    _assert_tradeoff_wing_osc_summary(
        search["selected_rationale"],
        native_supported=True,
        support_level="native_osc_path_templates_declared",
    )
    assert search["alternative_tradeoffs"]
    assert all(row["why_not_selected"] for row in search["alternative_tradeoffs"])
    for row in search["alternative_tradeoffs"]:
        _assert_tradeoff_wing_osc_summary(row)
        _assert_weighted_score_summary(
            row["weighted_score_summary"],
            mode="soundcheck",
            expected_total_score=row["total_score"],
        )
        assert row["weighted_score_gaps"]
        assert row["problem_action_summary"]["strategy_id"] == row["strategy_id"]
        assert row["problem_action_summary"]["live_write"] is False
        _assert_dsp_parameter_calculation_summary(
            row["dsp_parameter_calculation_summary"]
        )
        assert row["wing_native_strategy_fit"]["fit_status"]
        assert row["wing_native_strategy_fit_gaps"]

    proposal = recommendation.to_operator_proposal()
    _assert_proposal_denial(proposal)
    assert proposal["selected_strategy"]["strategy_id"] == "low_mid_corrective_cut"
    assert proposal["proposed_action"]["parameters"]["gain_db"] < 0.0
    _assert_tradeoff_wing_osc_summary(
        proposal["selected_wing_osc_summary"],
        native_supported=True,
        support_level="native_osc_path_templates_declared",
    )
    _assert_dsp_parameter_calculation_summary(
        proposal["selected_dsp_parameter_calculation_summary"],
        strategy_id="low_mid_corrective_cut",
        method_family="eq",
    )
    _assert_problem_action_decision_summary(
        proposal["problem_action_decision_summary"],
        strategy_id="low_mid_corrective_cut",
        method_family="eq",
        expected_action_intent="static_spectral_correction",
        expected_method_families=("eq",),
    )
    _assert_dsp_parameter_calculation_summary(
        proposal["proposed_action"]["dsp_parameter_calculation_summary"],
        strategy_id="low_mid_corrective_cut",
        method_family="eq",
    )
    _assert_problem_action_decision_summary(
        proposal["proposed_action"]["problem_action_decision_summary"],
        strategy_id="low_mid_corrective_cut",
        method_family="eq",
        expected_action_intent="static_spectral_correction",
        expected_method_families=("eq",),
    )
    _assert_selected_wing_osc_readback_summary(
        proposal["selected_wing_osc_readback_summary"],
        native_supported=True,
        expected_before_paths=expected_eq_readback_paths,
        expected_templates_include=(
            "/ch/{channel}/eq/{band}f",
            "/ch/{channel}/eq/{band}g",
            "/ch/{channel}/eq/{band}q",
        ),
    )
    _assert_tradeoff_wing_osc_summary(
        proposal["proposed_action"],
        native_supported=True,
        support_level="native_osc_path_templates_declared",
    )
    _assert_selected_wing_osc_readback_summary(
        proposal["proposed_action"],
        native_supported=True,
        expected_before_paths=expected_eq_readback_paths,
        expected_templates_include=("/ch/{channel}/eq/{band}g",),
    )
    _assert_tradeoff_wing_osc_summary(
        proposal["proposed_action"]["selected_wing_osc_summary"],
        native_supported=True,
        support_level="native_osc_path_templates_declared",
    )
    _assert_selected_wing_osc_readback_summary(
        proposal["proposed_action"]["selected_wing_osc_readback_summary"],
        native_supported=True,
        expected_before_paths=expected_eq_readback_paths,
        expected_templates_include=("/ch/{channel}/eq/{band}q",),
    )
    _assert_tradeoff_wing_osc_summary(
        proposal["operator_correction"],
        native_supported=True,
        support_level="native_osc_path_templates_declared",
    )
    _assert_selected_wing_osc_readback_summary(
        proposal["operator_correction"],
        native_supported=True,
        expected_before_paths=expected_eq_readback_paths,
        expected_templates_include=("/ch/{channel}/eq/{band}f",),
    )
    _assert_tradeoff_wing_osc_summary(
        proposal["operator_correction"]["selected_wing_osc_summary"],
        native_supported=True,
        support_level="native_osc_path_templates_declared",
    )
    _assert_selected_wing_osc_readback_summary(
        proposal["operator_correction"]["selected_wing_osc_readback_summary"],
        native_supported=True,
        expected_before_paths=expected_eq_readback_paths,
        expected_templates_include=("/ch/{channel}/eq/{band}g",),
    )
    _assert_tradeoff_wing_osc_summary(
        proposal["operator_correction"]["proposed_correction"],
        native_supported=True,
        support_level="native_osc_path_templates_declared",
    )
    _assert_selected_wing_osc_readback_summary(
        proposal["operator_correction"]["proposed_correction"],
        native_supported=True,
        expected_before_paths=expected_eq_readback_paths,
        expected_templates_include=("/ch/{channel}/eq/{band}q",),
    )
    _assert_tradeoff_wing_osc_summary(
        proposal["operator_correction"]["proposed_correction"][
            "selected_wing_osc_summary"
        ],
        native_supported=True,
        support_level="native_osc_path_templates_declared",
    )
    _assert_selected_wing_osc_readback_summary(
        proposal["operator_correction"]["proposed_correction"][
            "selected_wing_osc_readback_summary"
        ],
        native_supported=True,
        expected_before_paths=expected_eq_readback_paths,
        expected_templates_include=("/ch/{channel}/eq/{band}f",),
    )
    assert proposal["evidence"]["strategy_search"] is True
    assert proposal["strategy_search_details"]["selected_rationale"][
        "strategy_id"
    ] == "low_mid_corrective_cut"
    _assert_weighted_score_summary(
        proposal["strategy_search_details"]["selected_rationale"][
            "weighted_score_summary"
        ],
        mode="soundcheck",
        expected_total_score=recommendation.selected_strategy.total_score,
    )
    _assert_tradeoff_wing_osc_summary(
        proposal["strategy_search_details"]["selected_rationale"],
        native_supported=True,
        support_level="native_osc_path_templates_declared",
    )
    assert proposal["proposed_action"]["alternative_tradeoffs"]
    assert all(
        row["why_not_selected"]
        for row in proposal["proposed_action"]["alternatives_considered"]
    )
    for row in proposal["proposed_action"]["alternatives_considered"]:
        _assert_tradeoff_wing_osc_summary(row)
        assert row["problem_action_summary"]["strategy_id"] == row["strategy_id"]
        assert row["problem_action_summary"]["proposal_only"] is True
        _assert_dsp_parameter_calculation_summary(
            row["dsp_parameter_calculation_summary"]
        )
    sections = proposal["operator_explanation_sections"]
    assert sections["problem_found"] == "muddy"
    assert sections["selected_processing_type"] == "eq"
    assert sections["selected_strategy_id"] == "low_mid_corrective_cut"
    _assert_weighted_score_summary(
        sections["selected_weighted_score_summary"],
        mode="soundcheck",
        expected_total_score=recommendation.selected_strategy.total_score,
    )
    assert sections["selected_wing_native_strategy_fit"]["fit_status"]
    _assert_dsp_parameter_calculation_summary(
        sections["selected_dsp_parameter_calculation_summary"],
        strategy_id="low_mid_corrective_cut",
        method_family="eq",
    )
    _assert_problem_action_decision_summary(
        sections["selected_problem_action_decision_summary"],
        strategy_id="low_mid_corrective_cut",
        method_family="eq",
        expected_action_intent="static_spectral_correction",
        expected_method_families=("eq",),
    )
    _assert_tradeoff_wing_osc_summary(
        sections["selected_wing_osc_summary"],
        native_supported=True,
        support_level="native_osc_path_templates_declared",
    )
    _assert_selected_wing_osc_readback_summary(
        sections["selected_wing_osc_readback_summary"],
        native_supported=True,
        expected_before_paths=expected_eq_readback_paths,
        expected_templates_include=(
            "/ch/{channel}/eq/{band}f",
            "/ch/{channel}/eq/{band}g",
            "/ch/{channel}/eq/{band}q",
        ),
    )
    assert sections["selected_wing_osc_operator_status"] == (
        "wing_native_osc_candidate_requires_readback_before_promotion"
    )
    assert sections["selected_wing_osc_readback_status"] == (
        "wing_osc_readback_paths_declared_before_live_promotion"
    )
    assert sections["why_selected"]
    assert sections["proposed_parameters"]["gain_db"] < 0.0
    assert sections["alternatives_worse"]
    assert all(row["why_not_selected"] for row in sections["alternatives_worse"])
    for row in sections["alternatives_worse"]:
        _assert_tradeoff_wing_osc_summary(row)
        _assert_weighted_score_summary(
            row["weighted_score_summary"],
            mode="soundcheck",
            expected_total_score=row["total_score"],
        )
        assert row["weighted_score_gaps"]
        assert row["problem_action_summary"]["strategy_id"] == row["strategy_id"]
        _assert_dsp_parameter_calculation_summary(
            row["dsp_parameter_calculation_summary"]
        )
        assert row["wing_native_strategy_fit"]["fit_status"]
        assert row["wing_native_strategy_fit_gaps"]
    assert sections["remaining_risk"] == proposal["residual_risk"]
    assert sections["operator_verification"]
    assert sections["rollback_readback"]["required_before_live_promotion"] is True
    assert sections["rollback_readback"]["live_authority_granted"] is False
    assert proposal["evidence"]["operator_explanation_sections"] == sections
    contract = recommendation.evidence["decision_engine_contract"]
    assert contract["contract_id"] == "smart_dsp_decision_engine_contract_v1"
    assert contract["method_id"] == SMART_DSP_METHOD_ID
    assert contract["mode_policy"]["mode"] == "soundcheck"
    assert contract["mode_policy"]["supported_modes"] == ["live", "soundcheck"]
    assert contract["mode_policy"]["soundcheck_strategy_search_allowed"] is True
    assert contract["mode_policy"]["live_conservative_policy"] is False
    assert contract["mode_policy"]["live_apply_allowed"] is False
    assert contract["mode_policy"]["live_authority_granted"] is False
    assert contract["evidence_extraction"]["source_identity"]["source_family"] == "vocal"
    assert contract["evidence_extraction"]["source_identity"]["musical_role"] == "lead"
    assert contract["evidence_extraction"]["source_identity"]["confirmed"] is True
    assert contract["evidence_extraction"]["spectral_balance"]["muddiness_score"] == 0.82
    assert contract["evidence_extraction"]["level"]["true_peak_headroom_db"] == 5.0
    assert "phase_mono_safety" in contract["evidence_extraction"]
    assert "feedback_risk" in contract["evidence_extraction"]
    assert "bus_group_master_impact" in contract["evidence_extraction"]
    assert contract["strategy_search"]["candidate_count"] == 1 + len(
        recommendation.alternatives
    )
    assert contract["strategy_search"]["alternative_count"] == len(
        recommendation.alternatives
    )
    assert contract["strategy_search"]["selected_strategy_id"] == "low_mid_corrective_cut"
    assert contract["strategy_search"]["selected_processing_type"] == "eq"
    assert contract["strategy_search"]["scoring_dimensions"] == [
        "speed",
        "precision",
        "musical",
        "safety",
        "reversibility",
        "operator_clarity",
    ]
    assert contract["strategy_search"]["score_weights"] == (
        SOUNDCHECK_STRATEGY_SCORE_WEIGHTS
    )
    assert contract["strategy_search"]["score_weight_profile"] == (
        "soundcheck_weighted_strategy_search"
    )
    assert contract["strategy_search"]["score_weights_equal"] is False
    assert contract["strategy_search"]["mode_specific_score_weights"] is True
    _assert_weighted_score_summary(
        contract["strategy_search"]["selected_weighted_score_summary"],
        mode="soundcheck",
        expected_total_score=recommendation.selected_strategy.total_score,
    )
    assert contract["strategy_search"]["selected_wing_native_strategy_fit"][
        "fit_status"
    ]
    _assert_dsp_parameter_calculation_summary(
        contract["strategy_search"]["selected_dsp_parameter_calculation_summary"],
        strategy_id="low_mid_corrective_cut",
        method_family="eq",
    )
    _assert_problem_action_decision_summary(
        contract["strategy_search"]["selected_problem_action_decision_summary"],
        strategy_id="low_mid_corrective_cut",
        method_family="eq",
        expected_action_intent="static_spectral_correction",
        expected_method_families=("eq",),
    )
    assert contract["strategy_search"]["alternatives_compared"]
    assert all(
        row["why_not_selected"]
        for row in contract["strategy_search"]["alternatives_compared"]
    )
    for row in contract["strategy_search"]["alternatives_compared"]:
        _assert_weighted_score_summary(
            row["weighted_score_summary"],
            mode="soundcheck",
        )
        assert row["weighted_score_gaps"]
        assert row["problem_action_summary"]["strategy_id"] == row["strategy_id"]
        _assert_dsp_parameter_calculation_summary(
            row["dsp_parameter_calculation_summary"]
        )
        assert row["wing_native_strategy_fit"]["fit_status"]
        assert row["wing_native_strategy_fit_gaps"]
    _assert_dsp_parameter_calculation_summary(
        contract["parameter_calculation"][
            "selected_dsp_parameter_calculation_summary"
        ],
        strategy_id="low_mid_corrective_cut",
        method_family="eq",
    )
    assert contract["parameter_calculation"][
        "has_real_dsp_parameter_calculation"
    ] is True
    _assert_problem_action_decision_summary(
        contract["problem_action_selection"],
        strategy_id="low_mid_corrective_cut",
        method_family="eq",
        expected_action_intent="static_spectral_correction",
        expected_method_families=("eq",),
    )
    assert "gain_db" in contract["parameter_calculation"]["parameter_keys"]
    assert "wing_eq_band_operations" in contract["parameter_calculation"][
        "wing_osc_apply_parameter_keys"
    ]
    assert contract["safety_layer"]["true_peak_headroom_guard"] is True
    assert contract["safety_layer"]["feedback_guard"] is True
    assert contract["safety_layer"]["no_hidden_live_authority_widening"] is True
    assert contract["safety_layer"]["live_write"] is False
    assert contract["safety_layer"]["queue_mutation"] is False
    assert contract["safety_layer"]["live_authority_granted"] is False
    assert contract["operator_explanation"]["problem_found"] == "muddy"
    assert contract["operator_explanation"]["rollback_readback"][
        "required_before_live_promotion"
    ] is True
    assert contract["wing_readback_contract"]["target_console"] == (
        WING_RACK_OSC_TARGET_CONSOLE
    )
    assert contract["wing_readback_contract"]["firmware"] == WING_RACK_OSC_FIRMWARE
    assert contract["wing_readback_contract"]["selected_read_before_paths"] == (
        expected_eq_readback_paths
    )
    assert contract["wing_readback_contract"]["live_write"] is False
    evidence_contract = recommendation.evidence["evidence_extraction_contract"]
    assert evidence_contract["contract_id"] == (
        "smart_dsp_evidence_extraction_contract_v1"
    )
    assert evidence_contract["required_components"] == [
        "source_identity",
        "musical_role",
        "level_lufs_peak_true_peak",
        "crest_lra",
        "spectral_balance",
        "masking_conflict",
        "transient_behavior",
        "noise_floor_bleed",
        "phase_mono_safety",
        "feedback_risk",
        "bus_group_master_impact",
    ]
    assert evidence_contract["all_required_components_present"] is True
    assert evidence_contract["missing_components"] == []
    assert evidence_contract["component_coverage"]["source_identity"][
        "available"
    ] is True
    assert evidence_contract["component_coverage"]["bus_group_master_impact"][
        "required_for_strategy_search"
    ] is True
    assert evidence_contract["transport_free"] is True
    assert evidence_contract["live_write"] is False
    assert evidence_contract["live_authority_granted"] is False
    assert proposal["decision_engine_contract"] == contract
    assert proposal["proposed_action"]["decision_engine_contract"] == contract
    assert proposal["evidence"]["decision_engine_contract"] == contract
    assert proposal["evidence_extraction_contract"] == evidence_contract
    assert proposal["proposed_action"]["evidence_extraction_contract"] == (
        evidence_contract
    )
    assert proposal["evidence"]["evidence_extraction_contract"] == evidence_contract
    readback = recommendation.selected_strategy.future_wing_readback_contract
    _assert_common_readback_contract(readback)
    assert readback["eq_band_readback"]["protected_band_policy"] == (
        "do_not_overwrite_feedback_notch_or_operator_locked_band"
    )
    assert readback["eq_band_readback"]["eq_on_readback_required"] is True
    wing_osc = readback["wing_osc_applicability"]
    assert wing_osc["native_osc_supported"] is True
    assert wing_osc["native_osc_support_level"] == "native_osc_path_templates_declared"
    assert "/ch/{channel}/eq/{band}f" in wing_osc["native_osc_path_templates"]
    assert "/ch/{channel}/eq/{band}g" in wing_osc["native_osc_path_templates"]
    assert "/ch/{channel}/eq/{band}q" in wing_osc["native_osc_path_templates"]
    assert "/ch/{channel}/eq/mix" in wing_osc["native_osc_path_templates"]
    assert "/ch/{channel}/flt/lcf" in wing_osc["native_osc_path_templates"]
    eq_readback = readback["eq_band_readback"]
    assert eq_readback["eq_model_readback_must_equal"] == WING_EQ_REQUIRED_MODEL
    assert eq_readback["read_before_paths"] == [
        "eq/on",
        "eq/mdl",
        "eq/mix",
        "eq/{band}f",
        "eq/{band}g",
        "eq/{band}q",
    ]
    assert eq_readback["wing_eq_mix_osc_path_suffix"] == "eq/mix"
    assert eq_readback["wing_eq_mix_osc_value"] == WING_EQ_MIX_BASELINE_PERCENT
    assert eq_readback["wing_eq_mix_min_percent"] == WING_EQ_MIX_MIN_PERCENT
    assert eq_readback["wing_eq_mix_max_percent"] == (
        WING_EQ_MIX_MAX_PERCENT_BY_SURFACE["channel"]
    )
    eq_mapping = eq_readback["wing_eq_value_mapping"]
    assert eq_mapping["wing_eq_value_mapping_supported"] is True
    assert eq_mapping["wing_eq_filter_family"] == "parametric_band"
    assert eq_mapping["wing_eq_model_readback_must_equal"] == WING_EQ_REQUIRED_MODEL
    assert eq_mapping["wing_eq_readback_must_match_osc_value"] is True
    assert eq_mapping["wing_eq_mix_osc_path_suffix"] == "eq/mix"
    assert eq_mapping["wing_eq_mix_osc_value"] == WING_EQ_MIX_BASELINE_PERCENT
    assert eq_mapping["wing_eq_band_operations"][0]["path_suffixes"] == {
        "frequency": "eq/{band}f",
        "gain": "eq/{band}g",
        "q": "eq/{band}q",
    }
    assert sections["rollback_readback"]["wing_osc_applicability"] == wing_osc
    assert readback["eq_band_readback"]["wing_rack_fw_3_0_5_native_osc_supported"] is True
    assert "gain_fader_readback" not in readback
    _assert_eq_write_key_contract(
        recommendation.selected_strategy,
        selected_write_keys=[
            "wing_eq_on_osc_value",
            "wing_eq_mix_osc_value",
            "wing_eq_band_operations",
        ],
    )
    _assert_selected_eq_apply_values(recommendation, expected_eq_apply_values)


def test_rumble_dominant_vocal_prefers_hpf_before_low_mid_cut():
    recommendation = analyze_smart_dsp_strategy(
        _analysis(
            muddiness_score=0.62,
            low_end_excess_score=0.86,
            spectral_tilt_score=0.45,
            level_plane_error_lu=0.4,
        )
    )

    assert recommendation.status == "proposal_only"
    assert recommendation.selected_strategy.strategy_id == "rumble_dominant_high_pass_cleanup"
    assert recommendation.selected_strategy.method_family == "eq"
    params = recommendation.selected_strategy.parameters
    assert params["filter_type"] == "high_pass"
    assert params["frequency_hz"] == 125.0
    assert params["slope_db_per_octave"] == 18
    assert params["hpf_before_low_mid_cut"] is True
    assert params["wing_eq_filter_family"] == "channel_low_cut"
    assert params["wing_eq_filter_osc_path_suffixes"] == {
        "enabled": "flt/lc",
        "frequency": "flt/lcf",
        "slope": "flt/lcs",
    }
    assert params["wing_eq_low_cut_frequency_hz"] == 125.0
    assert params["wing_eq_low_cut_slope_db_per_octave"] == 18
    readback = recommendation.selected_strategy.future_wing_readback_contract
    eq_readback = readback["eq_band_readback"]
    assert eq_readback["read_before_paths"] == [
        "eq/on",
        "eq/mdl",
        "eq/mix",
        "flt/lc",
        "flt/lcf",
        "flt/lcs",
    ]
    assert eq_readback["wing_eq_value_mapping"]["wing_eq_filter_family"] == (
        "channel_low_cut"
    )
    _assert_eq_write_key_contract(
        recommendation.selected_strategy,
        selected_write_keys=[
            "wing_eq_on_osc_value",
            "wing_eq_mix_osc_value",
            "wing_eq_low_cut_enabled_osc_value",
            "wing_eq_low_cut_frequency_hz",
            "wing_eq_low_cut_slope_db_per_octave",
        ],
    )
    alternatives = {candidate.strategy_id: candidate for candidate in recommendation.alternatives}
    assert "low_mid_corrective_cut" in alternatives
    assert "high_pass_cleanup" in alternatives
    tradeoffs = {
        row["strategy_id"]: row
        for row in recommendation.evidence["strategy_search"]["alternative_tradeoffs"]
    }
    assert tradeoffs["low_mid_corrective_cut"]["why_not_selected"].startswith("lower_")
    assert recommendation.to_operator_proposal()["can_apply_live"] is False


def test_plosive_vocal_prefers_wing_native_deq_hpf_when_available():
    recommendation = analyze_smart_dsp_strategy(
        _analysis(
            problem_hints=("plosive_control", "muddy"),
            muddiness_score=0.72,
            low_end_excess_score=0.84,
            spectral_tilt_score=0.58,
            plosive_risk_score=0.82,
            plosive_frequency_hz=82.0,
            plosive_event_peak_dbfs=-18.4,
            plosive_event_contrast_db=17.2,
            plosive_event_rise_time_ms=18.0,
            plosive_event_recovery_ms=82.0,
            dynamics_variance_score=0.62,
            crest_factor_db=18.0,
            lra_proxy_db=14.0,
            estimated_lufs=-24.0,
            level_plane_error_lu=0.2,
        )
    )

    assert recommendation.status == "proposal_only"
    assert recommendation.selected_strategy.strategy_id == "vocal_plosive_dynamic_hpf_control"
    assert recommendation.selected_strategy.method_family == "dynamic_eq"
    params = recommendation.selected_strategy.parameters
    assert params["filter_type"] == "dynamic_high_pass"
    assert params["frequency_hz"] == 100.0
    assert params["slope_db_per_octave"] == 12
    assert params["threshold_dbfs"] == -26.5
    assert params["max_low_band_reduction_db"] == 5.7
    assert params["max_gain_reduction_db"] == 5.7
    assert params["attack_ms"] == 4.3
    assert params["release_ms"] == 113.4
    expected_apply_values = {
        "wing_dynamic_eq_model_osc_value": WING_DYNAMIC_EQ_GATE_MODEL,
        "wing_dynamic_eq_on_osc_value": 1,
        "wing_dynamic_eq_threshold_osc_value": -26.5,
        "wing_dynamic_eq_ratio_osc_value": 2.0,
        "wing_dynamic_eq_attack_osc_value": 4.3,
        "wing_dynamic_eq_release_osc_value": 113.4,
        "wing_dynamic_eq_filter_osc_value": "HP12",
        "wing_dynamic_eq_gain_osc_value": -5.7,
        "wing_dynamic_eq_frequency_osc_value": 100.0,
        "wing_dynamic_eq_q_osc_value": 0.707,
        "wing_dynamic_eq_mode_osc_value": "high",
    }
    assert params["wing_dynamic_eq_model_osc_value"] == WING_DYNAMIC_EQ_GATE_MODEL
    assert params["wing_dynamic_eq_on_osc_value"] == 1
    assert params["wing_dynamic_eq_threshold_osc_value"] == -26.5
    assert params["wing_dynamic_eq_ratio_osc_value"] == 2.0
    assert params["wing_dynamic_eq_attack_osc_value"] == 4.3
    assert params["wing_dynamic_eq_release_osc_value"] == 113.4
    assert params["wing_dynamic_eq_filter_osc_value"] == "HP12"
    assert params["wing_dynamic_eq_gain_osc_value"] == -5.7
    assert params["wing_dynamic_eq_frequency_osc_value"] == 100.0
    assert params["wing_dynamic_eq_q_osc_value"] == 0.707
    assert params["wing_dynamic_eq_mode_osc_value"] == "high"
    assert params["wing_dynamic_eq_gate_slot_conflict_policy"] == (
        WING_DYNAMIC_EQ_GATE_SLOT_CONFLICT_POLICY
    )
    assert params["wing_gate_slot_model_precondition_required"] is True
    assert params[
        "wing_gate_slot_model_precondition_required_model"
    ] == WING_DYNAMIC_EQ_GATE_MODEL
    assert params[
        "wing_gate_slot_model_precondition_read_before_paths"
    ] == ["gate/mdl", "gate/on"]
    assert params[
        "wing_gate_slot_model_precondition_parameter_apply_requires_model_match"
    ] is True
    assert params[
        "wing_gate_slot_model_precondition_model_write_requires_operator_review"
    ] is True
    assert params[
        "wing_gate_slot_model_precondition_no_hidden_model_change"
    ] is True
    assert params["plosive_risk_score"] == 0.82
    assert params["preserve_vocal_body_between_plosives"] is True
    assert recommendation.evidence["plosive_event_peak_dbfs"] == -18.4
    assert recommendation.evidence["plosive_event_contrast_db"] == 17.2
    assert recommendation.evidence["operator_explanation_sections"][
        "selection_policy"
    ] == "weighted_strategy_score"
    alternatives = {candidate.strategy_id: candidate for candidate in recommendation.alternatives}
    assert "static_plosive_high_pass_cleanup" in alternatives
    assert "low_mid_corrective_cut" in alternatives
    assert "rumble_dominant_high_pass_cleanup" not in alternatives
    tradeoffs = {
        row["strategy_id"]: row
        for row in recommendation.evidence["strategy_search"]["alternative_tradeoffs"]
    }
    assert tradeoffs["static_plosive_high_pass_cleanup"][
        "why_not_selected"
    ].startswith("lower_")
    assert tradeoffs["low_mid_corrective_cut"]["why_not_selected"].startswith(
        "lower_"
    )
    readback = recommendation.selected_strategy.future_wing_readback_contract
    _assert_common_readback_contract(readback)
    wing_osc = readback["wing_osc_applicability"]
    assert wing_osc["native_osc_supported"] is True
    assert wing_osc["native_osc_support_level"] == (
        "native_channel_gate_deq_dynamic_eq_osc_path_templates_declared"
    )
    assert wing_osc["unsupported_strategy_reason"] is None
    assert "/ch/{channel}/gate/mdl" in wing_osc["native_osc_path_templates"]
    assert "/ch/{channel}/gate/filt" in wing_osc["native_osc_path_templates"]
    assert "/ch/{channel}/gate/g" in wing_osc["native_osc_path_templates"]
    dynamic_eq_readback = readback["dynamic_eq_readback"]
    _assert_selected_write_key_contract(
        params=params,
        readback=dynamic_eq_readback,
        prefix="wing_dynamic_eq",
        selected_write_keys=list(WING_DYNAMIC_EQ_WRITE_PARAMETER_KEYS),
        write_parameter_keys=WING_DYNAMIC_EQ_WRITE_PARAMETER_KEYS,
        selection_policy=WING_DYNAMIC_EQ_WRITE_KEY_SELECTION_POLICY,
    )
    assert dynamic_eq_readback["wing_rack_fw_3_0_5_native_osc_supported"] is True
    assert dynamic_eq_readback["wing_dynamic_eq_model_readback_must_equal"] == (
        WING_DYNAMIC_EQ_GATE_MODEL
    )
    assert dynamic_eq_readback[
        "wing_gate_slot_model_precondition_required_model"
    ] == WING_DYNAMIC_EQ_GATE_MODEL
    assert dynamic_eq_readback[
        "wing_gate_slot_model_precondition_shared_by_models"
    ] == [
        WING_GATE_REQUIRED_MODEL,
        WING_DE_ESSER_GATE_MODEL,
        WING_DYNAMIC_EQ_GATE_MODEL,
    ]
    assert dynamic_eq_readback[
        "wing_gate_slot_model_precondition_live_apply_blocked_without_wing_proof"
    ] is True
    assert dynamic_eq_readback["wing_dynamic_eq_value_mapping"][
        "wing_dynamic_eq_on_osc_value"
    ] == 1
    assert dynamic_eq_readback["wing_dynamic_eq_threshold_osc_value"] == -26.5
    assert dynamic_eq_readback["wing_dynamic_eq_ratio_osc_value"] == 2.0
    assert dynamic_eq_readback["wing_dynamic_eq_attack_osc_value"] == 4.3
    assert dynamic_eq_readback["wing_dynamic_eq_release_osc_value"] == 113.4
    assert dynamic_eq_readback["wing_dynamic_eq_mode_osc_value"] == "high"
    assert dynamic_eq_readback["read_before_paths"] == [
        "gate/mdl",
        "gate/on",
        "gate/thr",
        "gate/ratio",
        "gate/att",
        "gate/rel",
        "gate/filt",
        "gate/g",
        "gate/f",
        "gate/q",
        "gate/mode",
    ]
    assert "eq_band_readback" not in readback
    assert "dynamics_readback" not in readback
    _assert_selected_dynamic_eq_apply_values(recommendation, expected_apply_values)
    assert recommendation.to_operator_proposal()["can_apply_live"] is False


def test_soundcheck_plosive_sibilant_vocal_prefers_de_esser_before_dynamic_hpf():
    recommendation = analyze_smart_dsp_strategy(
        _analysis(
            problem_hints=("plosive_control", "sibilant", "harsh"),
            operator_intent="soundcheck vocal plosives and sibilance",
            plosive_risk_score=0.84,
            plosive_frequency_hz=82.0,
            plosive_event_peak_dbfs=-18.4,
            plosive_event_contrast_db=17.2,
            plosive_event_rise_time_ms=18.0,
            plosive_event_recovery_ms=82.0,
            sibilance_score=0.82,
            sibilance_frequency_hz=7700.0,
            sibilance_bandwidth_hz=1300.0,
            sibilance_event_peak_dbfs=-12.5,
            sibilance_event_contrast_db=9.5,
            sibilance_event_rise_time_ms=6.0,
            sibilance_event_recovery_ms=92.0,
            harshness_score=0.66,
            presence_harshness_score=0.55,
            level_plane_error_lu=0.4,
            true_peak_headroom_db=5.5,
        )
    )

    assert recommendation.status == "proposal_only"
    assert recommendation.selected_strategy.strategy_id == "vocal_de_esser"
    assert recommendation.selected_strategy.method_family == "de_esser"
    assert (
        recommendation.evidence["operator_explanation_sections"]["selection_policy"]
        == "soundcheck_wing_native_de_esser_before_plosive_dynamic_hpf_priority"
    )
    params = recommendation.selected_strategy.parameters
    assert params["processing_chain"] == (
        "ds902_de_esser_first_then_plosive_mic_or_low_cut_recheck"
    )
    assert params["plosive_deesser_gate_slot_conflict"] is True
    assert params["post_de_esser_plosive_recheck_required"] is True
    assert params["wing_gate_slot_plosive_deesser_combo_blocker"] == (
        WING_GATE_SLOT_SINGLE_MODEL_COMBO_BLOCKER
    )
    dynamic_hpf = [
        candidate
        for candidate in recommendation.alternatives
        if candidate.strategy_id == "vocal_plosive_dynamic_hpf_control"
    ][0]
    assert WING_GATE_SLOT_SINGLE_MODEL_COMBO_BLOCKER in dynamic_hpf.blockers
    tradeoffs = {
        row["strategy_id"]: row["why_not_selected"]
        for row in recommendation.evidence["strategy_search"][
            "alternative_tradeoffs"
        ]
    }
    assert tradeoffs["vocal_plosive_dynamic_hpf_control"] == (
        "soundcheck_wing_native_de_esser_before_plosive_dynamic_hpf_priority:"
        "ds902_and_dynamic_hpf_share_wing_gate_slot_so_de_esser_recheck_precedes_deq_hpf"
    )
    proposal = recommendation.to_operator_proposal()
    assert proposal["live_apply_allowed"] is False
    apply_values = proposal["selected_wing_osc_summary"][
        "wing_osc_apply_parameter_values"
    ]
    assert apply_values == {
        "wing_de_esser_model_osc_value": WING_DE_ESSER_GATE_MODEL,
        "wing_de_esser_on_osc_value": 1,
        "wing_de_esser_frequency_osc_value": 7700.0,
        "wing_de_esser_range_osc_value": 5.0,
        "wing_de_esser_mode_osc_value": "HF",
    }
    assert not any(key.startswith("wing_dynamic_eq_") for key in apply_values)


def test_live_plosive_vocal_holds_before_dynamic_or_static_hpf():
    recommendation = analyze_smart_dsp_strategy(
        _analysis(
            mode="live",
            active_performance=True,
            problem_hints=("plosive_control", "muddy"),
            muddiness_score=0.72,
            low_end_excess_score=0.84,
            spectral_tilt_score=0.58,
            plosive_risk_score=0.82,
            plosive_frequency_hz=82.0,
            plosive_event_peak_dbfs=-18.4,
            plosive_event_contrast_db=17.2,
            plosive_event_rise_time_ms=18.0,
            plosive_event_recovery_ms=82.0,
            dynamics_variance_score=0.62,
            crest_factor_db=18.0,
            lra_proxy_db=14.0,
            estimated_lufs=-24.0,
            level_plane_error_lu=0.2,
        )
    )

    assert recommendation.status == "proposal_only"
    assert recommendation.selected_strategy.strategy_id == (
        "manual_plosive_mic_technique_review"
    )
    assert recommendation.selected_strategy.method_family == "analysis_only"
    assert recommendation.evidence["operator_explanation_sections"][
        "selection_policy"
    ] == "live_plosive_review_priority"
    params = recommendation.selected_strategy.parameters
    assert params["dsp_change"] == "hold_or_manual_mic_technique"
    assert params["live_plosive_review_priority"] is True
    assert params["plosive_parameter_source"] == "measured_plosive_burst"
    assert params["plosive_frequency_hz"] == 82.0
    assert params["plosive_event_peak_dbfs"] == -18.4
    assert params["plosive_event_contrast_db"] == 17.2
    assert params["live_plosive_hpf_hold_proposal"] is True
    assert params["live_plosive_hpf_target_surface"] == "channel"
    assert params["live_plosive_source_identity_required"] is True
    assert params["live_plosive_mic_technique_review_required"] is True
    assert params["live_plosive_previous_hpf_readback_required"] is True
    assert params["live_plosive_dynamic_hpf_gate_slot_readback_required"] is True
    assert params["live_plosive_static_low_cut_readback_required"] is True
    assert params["live_plosive_body_loss_audition_required"] is True
    assert params["live_plosive_feedback_recheck_required"] is True
    assert params["live_plosive_rollback_ready"] is True
    assert (
        params["live_plosive_no_dynamic_or_static_hpf_without_operator_wing_proof"]
        is True
    )
    assert params["forbidden_without_wing_proof"] == "dynamic_hpf_static_hpf"
    assert "wing_dynamic_eq_on_osc_value" not in params
    assert "wing_eq_low_cut_enabled_osc_value" not in params
    plosive_readback = recommendation.selected_strategy.future_wing_readback_contract[
        "live_plosive_hpf_hold_readback"
    ]
    assert plosive_readback["read_only"] is True
    assert plosive_readback["live_plosive_hpf_target_surface"] == "channel"
    assert plosive_readback["live_plosive_previous_hpf_readback_required"] is True
    assert (
        plosive_readback["live_plosive_dynamic_hpf_gate_slot_readback_required"]
        is True
    )
    assert plosive_readback["live_plosive_static_low_cut_readback_required"] is True
    assert (
        plosive_readback[
            "live_plosive_no_dynamic_or_static_hpf_without_operator_wing_proof"
        ]
        is True
    )
    assert plosive_readback["allowed_live_write_scope_before_wing_proof"] == []
    assert plosive_readback["wing_dynamic_hpf_filter_osc_path_suffix"] == "gate/filt"
    assert plosive_readback["wing_static_low_cut_path_suffixes"] == [
        "flt/lc",
        "flt/lcf",
        "flt/lcs",
    ]
    assert "/ch/{channel}/gate/filt" in plosive_readback[
        "wing_osc_path_templates"
    ]
    assert "/ch/{channel}/flt/lc" in plosive_readback["wing_osc_path_templates"]
    assert "/ch/{channel}/flt/lcf" in plosive_readback["wing_osc_path_templates"]
    assert "/ch/{channel}/flt/lcs" in plosive_readback["wing_osc_path_templates"]
    alternatives = {
        candidate.strategy_id: candidate for candidate in recommendation.alternatives
    }
    assert "live_plosive_hpf_requires_operator_review_and_wing_proof" in (
        alternatives["vocal_plosive_dynamic_hpf_control"].blockers
    )
    assert "live_plosive_hpf_requires_operator_review_and_wing_proof" in (
        alternatives["static_plosive_high_pass_cleanup"].blockers
    )
    tradeoffs = {
        row["strategy_id"]: row
        for row in recommendation.evidence["strategy_search"]["alternative_tradeoffs"]
    }
    assert tradeoffs["vocal_plosive_dynamic_hpf_control"]["why_not_selected"] == (
        "live_plosive_review_priority:hpf_change_can_thin_vocal_mid_show"
    )
    _assert_selected_has_no_apply_values(recommendation)
    _assert_proposal_denial(recommendation.to_operator_proposal())


def test_active_de_esser_gate_slot_blocks_dynamic_eq_plosive_hpf_selection():
    recommendation = analyze_smart_dsp_strategy(
        _analysis(
            problem_hints=("plosive_control", "muddy"),
            muddiness_score=0.72,
            low_end_excess_score=0.84,
            spectral_tilt_score=0.58,
            plosive_risk_score=0.82,
            plosive_frequency_hz=82.0,
            plosive_event_peak_dbfs=-18.4,
            plosive_event_contrast_db=17.2,
            plosive_event_rise_time_ms=18.0,
            plosive_event_recovery_ms=82.0,
            dynamics_variance_score=0.62,
            crest_factor_db=18.0,
            lra_proxy_db=14.0,
            estimated_lufs=-24.0,
            level_plane_error_lu=0.2,
            current_wing_gate_model=WING_DE_ESSER_GATE_MODEL,
            current_wing_gate_enabled=True,
            current_wing_gate_slot_readback_available=True,
        )
    )

    assert recommendation.status == "proposal_only"
    assert recommendation.selected_strategy.strategy_id == (
        "static_plosive_high_pass_cleanup"
    )
    assert recommendation.selected_strategy.method_family == "eq"
    alternatives = {candidate.strategy_id: candidate for candidate in recommendation.alternatives}
    dynamic_hpf = alternatives["vocal_plosive_dynamic_hpf_control"]
    assert WING_GATE_SLOT_MODEL_CONFLICT_BLOCKER in dynamic_hpf.blockers
    assert dynamic_hpf.parameters["wing_gate_slot_required_model"] == (
        WING_DYNAMIC_EQ_GATE_MODEL
    )
    assert dynamic_hpf.parameters["wing_gate_slot_current_model"] == (
        WING_DE_ESSER_GATE_MODEL
    )
    assert dynamic_hpf.parameters["wing_gate_slot_model_conflict"] is True
    assert dynamic_hpf.future_wing_readback_contract["dynamic_eq_readback"][
        "gate_slot_model_conflict_blocker"
    ] == WING_GATE_SLOT_MODEL_CONFLICT_BLOCKER
    tradeoffs = {
        row["strategy_id"]: row
        for row in recommendation.evidence["strategy_search"]["alternative_tradeoffs"]
    }
    assert WING_GATE_SLOT_MODEL_CONFLICT_BLOCKER in tradeoffs[
        "vocal_plosive_dynamic_hpf_control"
    ]["blockers"]


def test_real_wing_gate_readback_blocks_deq_dynamic_hpf_model_switch():
    recommendation = analyze_smart_dsp_strategy(
        _analysis(
            target_id="ch:3",
            problem_hints=("plosive_control", "muddy"),
            muddiness_score=0.72,
            low_end_excess_score=0.84,
            spectral_tilt_score=0.58,
            plosive_risk_score=0.82,
            plosive_frequency_hz=82.0,
            plosive_event_peak_dbfs=-18.4,
            plosive_event_contrast_db=17.2,
            plosive_event_rise_time_ms=18.0,
            plosive_event_recovery_ms=82.0,
            dynamics_variance_score=0.62,
            crest_factor_db=18.0,
            lra_proxy_db=14.0,
            estimated_lufs=-24.0,
            level_plane_error_lu=0.2,
            current_wing_gate_model=WING_GATE_REQUIRED_MODEL,
            current_wing_gate_enabled=False,
            current_wing_gate_slot_readback_available=True,
        )
    )

    assert recommendation.status == "proposal_only"
    assert recommendation.selected_strategy.strategy_id == (
        "static_plosive_high_pass_cleanup"
    )
    assert recommendation.selected_strategy.method_family == "eq"
    alternatives = {
        candidate.strategy_id: candidate for candidate in recommendation.alternatives
    }
    dynamic_hpf = alternatives["vocal_plosive_dynamic_hpf_control"]
    assert dynamic_hpf.blockers == (WING_GATE_SLOT_MODEL_CONFLICT_BLOCKER,)
    assert dynamic_hpf.parameters["wing_gate_slot_required_model"] == (
        WING_DYNAMIC_EQ_GATE_MODEL
    )
    assert dynamic_hpf.parameters["wing_gate_slot_current_model"] == (
        WING_GATE_REQUIRED_MODEL
    )
    assert dynamic_hpf.parameters["wing_gate_slot_current_enabled"] is False
    assert dynamic_hpf.parameters["wing_gate_slot_model_readback_available"] is True
    assert dynamic_hpf.parameters["wing_gate_slot_model_conflict"] is True
    assert (
        dynamic_hpf.parameters["wing_dynamic_eq_gate_slot_parameter_apply_ready"]
        is False
    )
    assert dynamic_hpf.parameters["wing_dynamic_eq_gate_slot_apply_ready_reason"] == (
        "requires_gate_mdl_readback_equal_deq_before_parameter_apply"
    )
    applicability = dynamic_hpf.future_wing_readback_contract[
        "wing_osc_applicability"
    ]
    assert applicability["native_osc_supported"] is False
    assert applicability["native_osc_support_level"] == (
        "gate_slot_model_conflict_requires_operator_review"
    )
    assert applicability["gate_slot_model_conflict_blocker"] == (
        WING_GATE_SLOT_MODEL_CONFLICT_BLOCKER
    )
    assert applicability["unsupported_strategy_reason"] == (
        "current_wing_gate_slot_model_readback_does_not_match_required_processor_model"
    )
    readback = dynamic_hpf.future_wing_readback_contract["dynamic_eq_readback"]
    assert readback["gate_slot_current_model"] == WING_GATE_REQUIRED_MODEL
    assert readback["gate_slot_current_enabled"] is False
    assert readback["gate_slot_model_conflict"] is True
    assert readback["gate_model_readback_must_equal"] == WING_DYNAMIC_EQ_GATE_MODEL
    assert readback["wing_dynamic_eq_gate_slot_parameter_apply_ready"] is False


def test_operator_owned_deq_gate_slot_blocks_dynamic_eq_parameter_apply():
    recommendation = analyze_smart_dsp_strategy(
        _analysis(
            problem_hints=("plosive_control", "muddy"),
            muddiness_score=0.72,
            low_end_excess_score=0.84,
            spectral_tilt_score=0.58,
            plosive_risk_score=0.82,
            plosive_frequency_hz=82.0,
            plosive_event_peak_dbfs=-18.4,
            plosive_event_contrast_db=17.2,
            plosive_event_rise_time_ms=18.0,
            plosive_event_recovery_ms=82.0,
            dynamics_variance_score=0.62,
            crest_factor_db=18.0,
            lra_proxy_db=14.0,
            estimated_lufs=-24.0,
            level_plane_error_lu=0.2,
            current_wing_gate_model=WING_DYNAMIC_EQ_GATE_MODEL,
            current_wing_gate_enabled=True,
            current_wing_gate_slot_readback_available=True,
            current_wing_gate_slot_operator_owned=True,
        )
    )

    assert recommendation.status == "proposal_only"
    assert recommendation.selected_strategy.strategy_id == (
        "static_plosive_high_pass_cleanup"
    )
    assert recommendation.selected_strategy.method_family == "eq"
    alternatives = {candidate.strategy_id: candidate for candidate in recommendation.alternatives}
    dynamic_hpf = alternatives["vocal_plosive_dynamic_hpf_control"]
    assert dynamic_hpf.blockers == (WING_GATE_SLOT_OPERATOR_OWNED_MODEL_BLOCKER,)
    assert dynamic_hpf.parameters["wing_gate_slot_required_model"] == (
        WING_DYNAMIC_EQ_GATE_MODEL
    )
    assert dynamic_hpf.parameters["wing_gate_slot_current_model"] == (
        WING_DYNAMIC_EQ_GATE_MODEL
    )
    assert dynamic_hpf.parameters["wing_gate_slot_current_operator_owned"] is True
    assert (
        dynamic_hpf.parameters[
            "wing_gate_slot_operator_owned_model_edit_requires_review"
        ]
        is True
    )
    assert (
        dynamic_hpf.parameters["wing_gate_slot_operator_owned_model_blocker"]
        == WING_GATE_SLOT_OPERATOR_OWNED_MODEL_BLOCKER
    )
    assert dynamic_hpf.parameters["wing_dynamic_eq_gate_slot_parameter_apply_ready"] is False
    assert dynamic_hpf.parameters["wing_dynamic_eq_gate_slot_apply_ready_reason"] == (
        "gate_slot_operator_owned_model_edit_requires_operator_review"
    )
    applicability = dynamic_hpf.future_wing_readback_contract[
        "wing_osc_applicability"
    ]
    assert applicability["native_osc_supported"] is False
    assert applicability["native_osc_support_level"] == (
        "gate_slot_operator_owned_model_edit_requires_review"
    )
    assert applicability["gate_slot_operator_owned_model_edit_requires_review"] is True
    assert applicability["gate_slot_operator_owned_model_blocker"] == (
        WING_GATE_SLOT_OPERATOR_OWNED_MODEL_BLOCKER
    )
    assert applicability["unsupported_strategy_reason"] == (
        "gate_slot_model_matches_but_current_wing_gate_slot_is_operator_owned"
    )
    tradeoffs = {
        row["strategy_id"]: row
        for row in recommendation.evidence["strategy_search"]["alternative_tradeoffs"]
    }
    assert WING_GATE_SLOT_OPERATOR_OWNED_MODEL_BLOCKER in tradeoffs[
        "vocal_plosive_dynamic_hpf_control"
    ]["blockers"]
    assert tradeoffs["vocal_plosive_dynamic_hpf_control"][
        "wing_osc_native_supported"
    ] is False
    assert tradeoffs["vocal_plosive_dynamic_hpf_control"][
        "wing_osc_support_level"
    ] == "gate_slot_operator_owned_model_edit_requires_review"


def test_vocal_phrase_noise_prefers_downward_expander_over_gate_eq_or_compression():
    recommendation = analyze_smart_dsp_strategy(
        _analysis(
            problem_hints=("vocal_noise_floor", "dynamic_control"),
            vocal_noise_floor_score=0.76,
            phrase_gap_noise_floor_dbfs=-55.2,
            phrase_gap_noise_flatness=0.32,
            phrase_gap_contrast_db=15.8,
            phrase_gap_median_ms=420.0,
            phrase_gap_coverage_ratio=0.38,
            noise_floor_dbfs=-55.2,
            lra_proxy_db=15.8,
            dynamics_variance_score=0.52,
            crest_factor_db=17.0,
            estimated_lufs=-25.0,
            level_plane_error_lu=0.3,
        )
    )

    assert recommendation.status == "proposal_only"
    assert recommendation.selected_strategy.strategy_id == "vocal_phrase_noise_downward_expander"
    assert recommendation.selected_strategy.method_family == "expander"
    params = recommendation.selected_strategy.parameters
    assert params["vocal_noise_parameter_source"] == "measured_phrase_gap_envelope"
    assert params["threshold_dbfs"] == -48.8
    assert params["measured_ratio"] == 1.83
    assert params["ratio"] == "1:2"
    assert params["range_db"] == 8.3
    assert params["attack_ms"] == 4.3
    assert params["hold_ms"] == 140.6
    assert params["release_ms"] == 323.0
    assert params["post_gate_hysteresis_review_db"] == 3.2
    assert params["phrase_gap_median_ms"] == 420.0
    assert params["phrase_gap_coverage_ratio"] == 0.38
    assert params["breath_preserve"] is True
    assert params["phrase_tail_preserve"] is True
    readback = recommendation.selected_strategy.future_wing_readback_contract
    wing_osc = readback["wing_osc_applicability"]
    assert wing_osc["native_osc_supported"] is True
    assert "/ch/{channel}/gate/thr21" in wing_osc["native_osc_path_templates"]
    assert "/ch/{channel}/gate/thr" not in wing_osc["native_osc_path_templates"]
    assert params["wing_gate_threshold_osc_path_suffix"] == "gate/thr21"
    assert params["wing_gate_model_osc_path_suffix"] == "gate/mdl"
    assert params["wing_gate_model_readback_must_equal"] == WING_GATE_REQUIRED_MODEL
    assert params["wing_gate_model_readback_required"] is True
    assert params["wing_gate_threshold_osc_value"] == -48.8
    assert params["wing_gate_range_osc_path_suffix"] == "gate/range"
    assert params["wing_gate_attack_osc_path_suffix"] == "gate/att"
    assert params["wing_gate_hold_osc_path_suffix"] == "gate/hld"
    assert params["wing_gate_release_osc_path_suffix"] == "gate/rel"
    assert params["wing_gate_ratio_osc_path_suffix"] == "gate/ratio"
    assert params["wing_gate_ratio_osc_value"] == "1:2"
    assert readback["gate_expander_readback"][
        "wing_gate_threshold_mapping_required"
    ] is True
    assert readback["gate_expander_readback"][
        "wing_gate_threshold_osc_path_suffix"
    ] == "gate/thr21"
    assert "gate/thr21" in readback["gate_expander_readback"]["read_after_paths"]
    assert "gate/mdl" in readback["gate_expander_readback"]["read_after_paths"]
    assert readback["gate_expander_readback"]["gate_model_readback_must_equal"] == (
        WING_GATE_REQUIRED_MODEL
    )
    assert "gate/thr" not in readback["gate_expander_readback"]["read_after_paths"]
    assert "gate/hysteresis" not in readback["gate_expander_readback"][
        "read_after_paths"
    ]
    alternatives = {candidate.strategy_id: candidate for candidate in recommendation.alternatives}
    assert "vocal_hard_gate_review" in alternatives
    assert "noise_floor_eq_cleanup_decoy" in alternatives
    assert "gentle_channel_compression" in alternatives
    assert "hard_gate_can_clip_vocal_breaths_and_phrase_tails" in alternatives[
        "vocal_hard_gate_review"
    ].blockers
    assert "broad_eq_does_not_solve_vocal_phrase_noise_floor" in alternatives[
        "noise_floor_eq_cleanup_decoy"
    ].blockers
    assert recommendation.evidence["vocal_noise_floor_score"] == 0.76
    assert "vocal_noise_floor" in recommendation.evidence["detected_problems"]
    assert recommendation.to_operator_proposal()["can_apply_live"] is False


def test_vocal_phrase_noise_expander_precedes_de_esser_or_compressor_order():
    recommendation = analyze_smart_dsp_strategy(
        _analysis(
            problem_hints=("vocal_noise_floor", "sibilant", "dynamic_control"),
            vocal_noise_floor_score=0.6,
            phrase_gap_noise_floor_dbfs=-52.0,
            phrase_gap_noise_flatness=0.38,
            phrase_gap_contrast_db=12.0,
            phrase_gap_median_ms=480.0,
            phrase_gap_coverage_ratio=0.4,
            noise_floor_dbfs=-52.0,
            sibilance_score=0.75,
            sibilance_frequency_hz=7600.0,
            sibilance_event_peak_dbfs=-20.0,
            sibilance_event_contrast_db=7.0,
            dynamics_variance_score=0.75,
            lra_proxy_db=14.0,
            dynamics_peak_to_median_db=12.0,
            true_peak_headroom_db=4.0,
            level_plane_error_lu=0.0,
        )
    )

    assert recommendation.status == "proposal_only"
    assert (
        recommendation.selected_strategy.strategy_id
        == "vocal_phrase_noise_downward_expander"
    )
    assert recommendation.selected_strategy.method_family == "expander"
    assert recommendation.evidence["operator_explanation_sections"][
        "selection_policy"
    ] == "soundcheck_vocal_phrase_noise_expander_priority"
    params = recommendation.selected_strategy.parameters
    assert params["vocal_noise_parameter_source"] == "measured_phrase_gap_envelope"
    assert params["phrase_gap_noise_floor_dbfs"] == -52.0
    assert params["phrase_gap_median_ms"] == 480.0
    assert params["phrase_gap_coverage_ratio"] == 0.4
    assert params["breath_preserve"] is True
    assert params["phrase_tail_preserve"] is True

    alternatives = {
        candidate.strategy_id: candidate for candidate in recommendation.alternatives
    }
    assert "vocal_sibilance_compressor_order_review" in alternatives
    assert "vocal_de_esser" in alternatives
    assert "gentle_channel_compression" in alternatives
    assert alternatives["vocal_phrase_leveling_compression"].blockers == (
        "vocal_phrase_noise_expander_should_precede_leveling_compression",
    )
    tradeoffs = {
        row["strategy_id"]: row
        for row in recommendation.evidence["strategy_search"]["alternative_tradeoffs"]
    }
    assert tradeoffs["vocal_sibilance_compressor_order_review"][
        "why_not_selected"
    ] == (
        "soundcheck_vocal_phrase_noise_priority:"
        "clean_phrase_gaps_before_de_essing_or_compression"
    )
    assert tradeoffs["vocal_de_esser"]["why_not_selected"] == (
        "soundcheck_vocal_phrase_noise_priority:"
        "clean_phrase_gaps_before_de_essing_or_compression"
    )
    assert tradeoffs["gentle_channel_compression"]["why_not_selected"] == (
        "soundcheck_vocal_phrase_noise_priority:"
        "clean_phrase_gaps_before_de_essing_or_compression"
    )
    assert recommendation.to_operator_proposal()["live_apply_allowed"] is False


def test_live_active_vocal_noise_floor_holds_before_expander_change():
    recommendation = analyze_smart_dsp_strategy(
        _analysis(
            mode="live",
            problem_hints=("vocal_noise_floor",),
            vocal_noise_floor_score=0.8,
            phrase_gap_noise_floor_dbfs=-53.0,
            phrase_gap_noise_flatness=0.35,
            phrase_gap_contrast_db=17.0,
            phrase_gap_median_ms=500.0,
            phrase_gap_coverage_ratio=0.42,
            noise_floor_dbfs=-53.0,
            lra_proxy_db=17.0,
            active_performance=True,
            level_plane_error_lu=0.2,
        )
    )

    assert recommendation.status == "proposal_only"
    assert recommendation.selected_strategy.strategy_id == (
        "mic_gain_stage_noise_floor_review"
    )
    assert recommendation.selected_strategy.method_family == "analysis_only"
    assert recommendation.evidence["operator_explanation_sections"][
        "selection_policy"
    ] == "live_vocal_noise_review_priority"
    params = recommendation.selected_strategy.parameters
    assert params["live_vocal_noise_review_priority"] is True
    assert params["phrase_gap_noise_flatness"] == 0.35
    assert params["phrase_gap_contrast_db"] == 17.0
    assert params["phrase_gap_median_ms"] == 500.0
    assert params["phrase_gap_coverage_ratio"] == 0.42
    assert params["vocal_noise_parameter_source"] == "measured_phrase_gap_envelope"
    assert params["live_vocal_noise_hold_proposal"] is True
    assert params["live_vocal_noise_target_surface"] == "channel"
    assert params["live_vocal_noise_source_identity_required"] is True
    assert params["live_vocal_noise_mic_preamp_review_required"] is True
    assert params["live_vocal_noise_room_noise_review_required"] is True
    assert params["live_vocal_noise_monitor_bleed_review_required"] is True
    assert params["live_vocal_noise_rf_check_required"] is True
    assert params["live_vocal_noise_previous_gate_expander_readback_required"] is True
    assert params["live_vocal_noise_gate_model_readback_required"] is True
    assert params["live_vocal_noise_gate_parameters_readback_required"] is True
    assert params["live_vocal_noise_gate_sidechain_readback_required"] is True
    assert params["live_vocal_noise_breath_tail_audition_required"] is True
    assert params["live_vocal_noise_noise_floor_remeasure_required"] is True
    assert params["live_vocal_noise_rollback_ready"] is True
    assert (
        params["live_vocal_noise_no_expander_gate_broad_eq_without_operator_wing_proof"]
        is True
    )
    assert params["required_context"] == (
        "mic_preamp_room_noise_monitor_bleed_rf_previous_expander_readback"
    )
    assert params["forbidden_without_wing_proof"] == "expander_gate_broad_eq"
    assert "wing_gate_on_osc_value" not in params
    assert "wing_gate_threshold_osc_value" not in params
    vocal_noise_readback = recommendation.selected_strategy.future_wing_readback_contract[
        "live_vocal_noise_hold_readback"
    ]
    assert vocal_noise_readback["read_only"] is True
    assert vocal_noise_readback["live_vocal_noise_target_surface"] == "channel"
    assert vocal_noise_readback[
        "live_vocal_noise_previous_gate_expander_readback_required"
    ] is True
    assert vocal_noise_readback["live_vocal_noise_gate_model_readback_required"] is True
    assert vocal_noise_readback[
        "live_vocal_noise_gate_sidechain_readback_required"
    ] is True
    assert (
        vocal_noise_readback[
            "live_vocal_noise_no_expander_gate_broad_eq_without_operator_wing_proof"
        ]
        is True
    )
    assert vocal_noise_readback["allowed_live_write_scope_before_wing_proof"] == []
    assert vocal_noise_readback["wing_gate_threshold_osc_path_suffix"] == "gate/thr21"
    assert "/ch/{channel}/gate/thr21" in vocal_noise_readback[
        "wing_osc_path_templates"
    ]
    assert "/ch/{channel}/gatesc/type" in vocal_noise_readback[
        "wing_osc_path_templates"
    ]
    expander = [
        candidate
        for candidate in recommendation.alternatives
        if candidate.strategy_id == "vocal_phrase_noise_downward_expander"
    ][0]
    assert (
        "live_vocal_noise_floor_expander_requires_operator_review_and_wing_proof"
        in expander.blockers
    )
    assert expander.parameters["vocal_noise_parameter_source"] == (
        "measured_phrase_gap_envelope"
    )
    assert expander.parameters["release_ms"] > 340.0
    tradeoffs = {
        row["strategy_id"]: row
        for row in recommendation.evidence["strategy_search"]["alternative_tradeoffs"]
    }
    assert tradeoffs["vocal_phrase_noise_downward_expander"]["why_not_selected"] == (
        "live_vocal_noise_review_priority:"
        "noise_path_check_precedes_expander_gate_or_eq"
    )
    _assert_selected_has_no_apply_values(recommendation)
    assert recommendation.to_operator_proposal()["can_apply_live"] is False


def test_soundcheck_quiet_noisy_vocal_reviews_gain_stage_before_gain_up_or_expander():
    recommendation = analyze_smart_dsp_strategy(
        _analysis(
            problem_hints=("level_plane_balance", "vocal_noise_floor"),
            estimated_lufs=-31.0,
            level_plane_error_lu=-5.0,
            true_peak_headroom_db=8.0,
            feedback_risk=0.1,
            bleed_risk=0.12,
            vocal_noise_floor_score=0.82,
            phrase_gap_noise_floor_dbfs=-51.5,
            phrase_gap_noise_flatness=0.44,
            phrase_gap_contrast_db=9.0,
            phrase_gap_median_ms=520.0,
            phrase_gap_coverage_ratio=0.5,
            noise_floor_dbfs=-51.5,
            lra_proxy_db=14.0,
        )
    )

    assert recommendation.status == "proposal_only"
    assert recommendation.selected_strategy.strategy_id == (
        "mic_gain_stage_noise_floor_review"
    )
    assert recommendation.selected_strategy.method_family == "analysis_only"
    assert recommendation.evidence["operator_explanation_sections"][
        "selection_policy"
    ] == "soundcheck_quiet_noisy_vocal_gain_stage_priority"
    params = recommendation.selected_strategy.parameters
    assert params["recommended_check"] == "mic_preamp_room_noise_monitor_bleed"
    assert params["phrase_gap_noise_floor_dbfs"] == -51.5
    assert params["vocal_noise_floor_score"] == 0.82
    assert params["required_context"] == "mic_preamp_room_noise_monitor_bleed"
    assert params["gain_db"] == 0.0
    assert params["read_only_review"] is True

    alternatives = {candidate.strategy_id: candidate for candidate in recommendation.alternatives}
    assert "noise_floor_path_review_should_precede_gain_up" in alternatives[
        "soundcheck_bounded_gain_up"
    ].blockers
    expander = alternatives["vocal_phrase_noise_downward_expander"]
    assert expander.parameters["vocal_noise_parameter_source"] == (
        "measured_phrase_gap_envelope"
    )
    assert expander.parameters["phrase_gap_noise_floor_dbfs"] == -51.5
    assert expander.parameters["makeup_gain_db"] == 0.0

    tradeoffs = {
        row["strategy_id"]: row
        for row in recommendation.evidence["strategy_search"]["alternative_tradeoffs"]
    }
    assert tradeoffs["soundcheck_bounded_gain_up"]["why_not_selected"] == (
        "soundcheck_quiet_noisy_vocal_priority:"
        "noise_path_review_precedes_gain_up"
    )
    assert tradeoffs["vocal_phrase_noise_downward_expander"]["why_not_selected"] == (
        "soundcheck_quiet_noisy_vocal_priority:"
        "fix_gain_stage_noise_path_before_expander"
    )
    assert recommendation.to_operator_proposal()["can_apply_live"] is False


def test_line_noise_prefers_notch_stack_over_hpf_or_broad_low_cut():
    recommendation = analyze_smart_dsp_strategy(
        _analysis(
            source_role="electric_guitar",
            source_label="GTR DI",
            problem_hints=("line_noise", "muddy"),
            line_noise_score=0.84,
            line_noise_frequency_hz=60.0,
            line_noise_prominence_db=28.0,
            line_noise_harmonic_count=3,
            line_noise_harmonics=(
                {"frequency_hz": 60.0, "prominence_db": 28.0},
                {"frequency_hz": 120.0, "prominence_db": 22.0},
                {"frequency_hz": 180.0, "prominence_db": 18.0},
            ),
            low_end_excess_score=0.72,
            muddiness_score=0.58,
            spectral_tilt_score=0.5,
            level_plane_error_lu=0.2,
        )
    )

    assert recommendation.status == "proposal_only"
    assert recommendation.selected_strategy.strategy_id == "mains_hum_notch_stack"
    assert recommendation.selected_strategy.method_family == "eq"
    params = recommendation.selected_strategy.parameters
    assert params["filter_type"] == "notch_stack"
    assert params["frequency_hz"] == 60.0
    assert params["line_noise_notch_parameter_source"] == "measured_line_noise_harmonics"
    assert params["gain_db"] == -6.7
    assert params["q"] == 14.0
    assert len(params["harmonic_notches"]) == 3
    assert params["wing_eq_band_capacity"] == 4
    assert params["wing_required_eq_band_count"] == 3
    assert params["wing_eq_band_allocation_required"] is True
    assert params["wing_eq_band_overflow_review_required"] is False
    assert params["harmonic_notches"][0] == {
        "frequency_hz": 60.0,
        "gain_db": -6.7,
        "q": 14.0,
        "filter_type": "peaking",
        "prominence_db": 28.0,
    }
    assert params["harmonic_notches"][1]["gain_db"] == -5.7
    assert params["harmonic_notches"][2]["q"] == 9.9
    alternatives = {candidate.strategy_id: candidate for candidate in recommendation.alternatives}
    assert "line_noise_high_pass_cleanup" in alternatives
    assert "broad_low_cut_hum_decoy" in alternatives
    assert "di_ground_lift_line_check" in alternatives
    assert "broad_low_cut_does_not_target_mains_hum" in alternatives[
        "broad_low_cut_hum_decoy"
    ].blockers
    assert "line_noise" in recommendation.evidence["detected_problems"]
    assert recommendation.evidence["line_noise_harmonic_count"] == 3
    readback = recommendation.selected_strategy.future_wing_readback_contract
    eq_readback = readback["eq_band_readback"]
    assert eq_readback["wing_native_eq_band_capacity"] == 4
    assert eq_readback["required_band_count"] == 3
    assert eq_readback["native_notch_band_numbers"] == [1, 2, 3]
    assert eq_readback["apply_only_native_harmonic_notches"] is True
    assert readback["wing_osc_applicability"]["native_eq_band_capacity"] == 4
    assert recommendation.to_operator_proposal()["can_apply_live"] is False


def test_soundcheck_line_noise_precedes_low_end_resonance_eq_when_hum_overlaps_bass_resonance():
    recommendation = analyze_smart_dsp_strategy(
        _analysis(
            target_id="ch:12",
            source_role="bass_di",
            source_label="Bass DI",
            problem_hints=("line_noise", "low_end_resonance"),
            line_noise_score=0.78,
            line_noise_frequency_hz=60.0,
            line_noise_prominence_db=22.0,
            line_noise_harmonic_count=3,
            line_noise_harmonics=(
                {"frequency_hz": 60.0, "prominence_db": 22.0},
                {"frequency_hz": 120.0, "prominence_db": 18.0},
                {"frequency_hz": 180.0, "prominence_db": 14.0},
            ),
            low_end_resonance_frequency_hz=62.0,
            low_end_resonance_bandwidth_hz=35.0,
            low_end_resonance_peak_dbfs=-18.0,
            low_end_resonance_sustain_ms=540.0,
            low_end_excess_score=0.82,
            level_plane_error_lu=0.0,
        )
    )

    assert recommendation.status == "proposal_only"
    assert recommendation.source_family == "bass"
    assert "line_noise" in recommendation.evidence["detected_problems"]
    assert "low_end_resonance" in recommendation.evidence["detected_problems"]
    assert recommendation.selected_strategy.strategy_id == "mains_hum_notch_stack"
    assert recommendation.selected_strategy.method_family == "eq"
    assert recommendation.evidence["operator_explanation_sections"][
        "selection_policy"
    ] == "soundcheck_line_noise_low_end_resonance_priority"
    params = recommendation.selected_strategy.parameters
    assert params["line_noise_low_end_resonance_conflict"] is True
    assert params["line_noise_low_end_resonance_priority"] is True
    assert params["frequency_hz"] == 60.0
    assert params["low_end_resonance_frequency_hz"] == 62.0
    assert params["low_end_excess_score"] == 0.82
    assert params["wing_eq_band_capacity"] == 4
    assert params["wing_required_eq_band_count"] == 3
    readback = recommendation.selected_strategy.future_wing_readback_contract
    assert readback["wing_osc_applicability"]["native_osc_supported"] is True
    assert readback["eq_band_readback"]["apply_only_native_harmonic_notches"] is True

    alternatives = {
        candidate.strategy_id: candidate for candidate in recommendation.alternatives
    }
    assert "low_end_dynamic_resonance_control" in alternatives
    assert "low_end_static_pocket_cut" in alternatives
    tradeoffs = {
        row["strategy_id"]: row["why_not_selected"]
        for row in recommendation.evidence["strategy_search"]["alternative_tradeoffs"]
    }
    assert tradeoffs["low_end_dynamic_resonance_control"] == (
        "soundcheck_line_noise_low_end_resonance_priority:"
        "measured_hum_notch_precedes_low_end_tone_eq"
    )
    assert tradeoffs["low_end_static_pocket_cut"] == (
        "soundcheck_line_noise_low_end_resonance_priority:"
        "measured_hum_notch_precedes_low_end_tone_eq"
    )
    assert recommendation.to_operator_proposal()["live_apply_allowed"] is False


def test_bass_di_line_noise_blocks_gain_up_before_hum_cleanup():
    recommendation = analyze_smart_dsp_strategy(
        _analysis(
            target_id="ch:12",
            source_role="bass_di",
            source_label="Bass DI",
            problem_hints=("line_noise", "level_plane_balance"),
            line_noise_score=0.86,
            line_noise_frequency_hz=50.0,
            line_noise_prominence_db=22.0,
            line_noise_harmonic_count=4,
            line_noise_harmonics=(
                {"frequency_hz": 50.0, "prominence_db": 22.0},
                {"frequency_hz": 100.0, "prominence_db": 17.0},
                {"frequency_hz": 150.0, "prominence_db": 11.0},
                {"frequency_hz": 200.0, "prominence_db": 8.0},
            ),
            estimated_lufs=-32.0,
            level_plane_error_lu=-5.2,
            noise_floor_dbfs=-53.0,
        )
    )

    assert recommendation.source_family == "bass"
    assert recommendation.selected_strategy.strategy_id == "mains_hum_notch_stack"
    assert recommendation.selected_strategy.method_family == "eq"
    assert recommendation.evidence["operator_explanation_sections"][
        "selection_policy"
    ] == "soundcheck_line_noise_before_gain_up_priority"
    assert "line_noise" in recommendation.evidence["detected_problems"]
    params = recommendation.selected_strategy.parameters
    assert params["measured_line_noise_harmonic_count"] == 4
    assert len(params["harmonic_notches"]) == 4
    assert params["harmonic_notches"][3] == {
        "frequency_hz": 200.0,
        "gain_db": -3.9,
        "q": 7.0,
        "filter_type": "peaking",
        "prominence_db": 8.0,
    }
    assert params["wing_eq_band_capacity"] == 4
    assert params["wing_required_eq_band_count"] == 4
    assert params["wing_eq_band_allocation_required"] is True
    assert params["wing_eq_band_overflow_review_required"] is False
    assert params["wing_eq_band_overflow_review_only_count"] == 0
    readback = recommendation.selected_strategy.future_wing_readback_contract
    eq_readback = readback["eq_band_readback"]
    assert eq_readback["wing_native_eq_band_capacity"] == 4
    assert eq_readback["required_band_count"] == 4
    assert eq_readback["native_notch_band_numbers"] == [1, 2, 3, 4]
    assert eq_readback["wing_eq_band_overflow_review_required"] is False
    wing_osc = readback["wing_osc_applicability"]
    assert wing_osc["native_osc_supported"] is True
    assert wing_osc["native_eq_band_capacity"] == 4
    alternatives = {candidate.strategy_id: candidate for candidate in recommendation.alternatives}
    assert "line_noise_path_review_should_precede_gain_up" in alternatives[
        "soundcheck_bounded_gain_up"
    ].blockers
    tradeoffs = {
        row["strategy_id"]: row
        for row in recommendation.evidence["strategy_search"]["alternative_tradeoffs"]
    }
    assert tradeoffs["soundcheck_bounded_gain_up"]["why_not_selected"] == (
        "soundcheck_line_noise_before_gain_up_priority:"
        "remove_or_verify_hum_before_adding_level"
    )


def test_line_noise_notch_stack_caps_apply_to_wing_four_band_eq_capacity():
    recommendation = analyze_smart_dsp_strategy(
        _analysis(
            target_id="ch:18",
            source_role="keys_di",
            source_label="Keys DI",
            problem_hints=("line_noise",),
            line_noise_score=0.9,
            line_noise_frequency_hz=50.0,
            line_noise_prominence_db=24.0,
            line_noise_harmonic_count=6,
            line_noise_harmonics=(
                {"frequency_hz": 50.0, "prominence_db": 24.0},
                {"frequency_hz": 100.0, "prominence_db": 20.0},
                {"frequency_hz": 150.0, "prominence_db": 16.0},
                {"frequency_hz": 200.0, "prominence_db": 12.0},
                {"frequency_hz": 250.0, "prominence_db": 10.0},
                {"frequency_hz": 300.0, "prominence_db": 8.0},
            ),
        )
    )

    assert recommendation.selected_strategy.strategy_id == "mains_hum_notch_stack"
    params = recommendation.selected_strategy.parameters
    assert params["wing_eq_band_capacity"] == 4
    assert params["wing_required_eq_band_count"] == 4
    assert len(params["harmonic_notches"]) == 4
    assert params["harmonic_notches"][-1]["frequency_hz"] == 200.0
    assert params["wing_eq_band_overflow_review_required"] is True
    assert params["wing_eq_band_overflow_review_only_count"] == 2
    assert params["line_noise_harmonics_review_only"] == [
        {"frequency_hz": 250.0, "prominence_db": 10.0},
        {"frequency_hz": 300.0, "prominence_db": 8.0},
    ]
    readback = recommendation.selected_strategy.future_wing_readback_contract
    eq_readback = readback["eq_band_readback"]
    assert eq_readback["required_band_count"] == 4
    assert eq_readback["wing_native_eq_band_capacity"] == 4
    assert eq_readback["wing_eq_band_overflow_review_required"] is True
    assert eq_readback["wing_eq_band_overflow_review_only_count"] == 2
    assert readback["wing_osc_applicability"]["native_osc_supported"] is True


def test_line_noise_notch_stack_uses_six_native_bands_on_wing_bus_eq():
    recommendation = analyze_smart_dsp_strategy(
        _analysis(
            target_id="bus:1",
            source_role="keys_group",
            source_label="Keys Group",
            problem_hints=("line_noise",),
            line_noise_score=0.9,
            line_noise_frequency_hz=50.0,
            line_noise_prominence_db=24.0,
            line_noise_harmonic_count=6,
            line_noise_harmonics=(
                {"frequency_hz": 50.0, "prominence_db": 24.0},
                {"frequency_hz": 100.0, "prominence_db": 20.0},
                {"frequency_hz": 150.0, "prominence_db": 16.0},
                {"frequency_hz": 200.0, "prominence_db": 12.0},
                {"frequency_hz": 250.0, "prominence_db": 10.0},
                {"frequency_hz": 300.0, "prominence_db": 8.0},
            ),
        )
    )

    assert recommendation.selected_strategy.strategy_id == "mains_hum_notch_stack"
    params = recommendation.selected_strategy.parameters
    assert params["wing_eq_band_capacity"] == 6
    assert params["wing_required_eq_band_count"] == 6
    assert len(params["harmonic_notches"]) == 6
    assert params["harmonic_notches"][-1]["frequency_hz"] == 260.0
    assert params["wing_eq_band_overflow_review_required"] is False
    assert params["wing_eq_band_overflow_review_only_count"] == 0
    assert params["line_noise_harmonics_review_only"] == []
    readback = recommendation.selected_strategy.future_wing_readback_contract
    eq_readback = readback["eq_band_readback"]
    assert eq_readback["required_band_count"] == 6
    assert eq_readback["wing_native_eq_band_capacity"] == 6
    assert "B=1-6" in eq_readback["wing_native_eq_band_capacity_source"]
    wing_osc = readback["wing_osc_applicability"]
    assert wing_osc["native_osc_supported"] is True
    assert wing_osc["native_eq_band_capacity"] == 6


def test_live_line_noise_prefers_physical_path_review_before_notch_or_dynamics():
    recommendation = analyze_smart_dsp_strategy(
        _analysis(
            target_id="ch:14",
            source_role="keys_di",
            source_label="Keys DI",
            mode="live",
            problem_hints=("line_noise", "dynamic_control"),
            active_performance=True,
            line_noise_score=0.8,
            line_noise_frequency_hz=60.0,
            line_noise_prominence_db=20.0,
            line_noise_harmonic_count=2,
            line_noise_harmonics=(
                {"frequency_hz": 60.0, "prominence_db": 20.0},
                {"frequency_hz": 120.0, "prominence_db": 14.0},
            ),
            dynamics_variance_score=0.7,
            lra_proxy_db=13.0,
        )
    )

    assert recommendation.selected_strategy.strategy_id == "di_ground_lift_line_check"
    assert recommendation.selected_strategy.method_family == "analysis_only"
    assert recommendation.evidence["operator_explanation_sections"][
        "selection_policy"
    ] == "live_line_noise_physical_path_review_priority"
    params = recommendation.selected_strategy.parameters
    assert params["live_line_noise_review_priority"] is True
    assert params["required_context"] == (
        "di_cable_ground_lift_power_source_check_before_live_eq"
    )
    assert params["forbidden_without_wing_proof"] == "live_line_noise_notch_hpf_eq"
    assert params["live_line_noise_hold_proposal"] is True
    assert params["live_line_noise_previous_eq_readback_required"] is True
    assert params["live_line_noise_previous_low_cut_readback_required"] is True
    assert params["live_line_noise_previous_dynamics_readback_required"] is True
    assert (
        params["live_line_noise_no_notch_hpf_dynamics_without_operator_wing_proof"]
        is True
    )
    line_noise_readback = recommendation.selected_strategy.future_wing_readback_contract[
        "live_line_noise_hold_readback"
    ]
    assert line_noise_readback["read_only"] is True
    assert line_noise_readback["wing_rack_fw_3_0_5_native_osc_supported"] is True
    assert "/ch/{channel}/eq/{band}f" in line_noise_readback[
        "wing_eq_notch_osc_path_templates"
    ]
    assert "/ch/{channel}/flt/lc" in line_noise_readback[
        "wing_static_low_cut_osc_path_templates"
    ]
    assert "/ch/{channel}/dyn/on" in line_noise_readback[
        "wing_dynamics_osc_path_templates"
    ]
    assert "eq/{band}g" in line_noise_readback["read_before_paths"]
    assert "flt/lcf" in line_noise_readback["read_before_paths"]
    assert "dyn/thr24" in line_noise_readback["read_before_paths"]
    assert line_noise_readback["allowed_live_write_scope_before_wing_proof"] == []

    alternatives = {candidate.strategy_id: candidate for candidate in recommendation.alternatives}
    assert alternatives["mains_hum_notch_stack"].blockers == (
        "live_line_noise_eq_requires_operator_review_and_wing_proof",
        "live_eq_cut_exceeds_conservative_limit",
    )
    assert alternatives["line_noise_high_pass_cleanup"].blockers == (
        "live_line_noise_eq_requires_operator_review_and_wing_proof",
    )
    tradeoffs = {
        row["strategy_id"]: row
        for row in recommendation.evidence["strategy_search"]["alternative_tradeoffs"]
    }
    assert tradeoffs["mains_hum_notch_stack"]["why_not_selected"] == (
        "live_line_noise_review_priority:"
        "physical_path_check_precedes_live_notch_or_hpf"
    )
    assert tradeoffs["manual_level_ride_before_compression"]["why_not_selected"] == (
        "live_line_noise_review_priority:"
        "hum_path_check_precedes_compression_or_level_ride"
    )


def test_live_vocal_plosive_review_precedes_line_noise_path_review():
    recommendation = analyze_smart_dsp_strategy(
        _analysis(
            target_id="ch:15",
            source_role="lead_vocal",
            source_label="Lead Vox",
            mode="live",
            problem_hints=("plosive_control", "line_noise"),
            active_performance=True,
            plosive_risk_score=0.84,
            plosive_frequency_hz=82.0,
            plosive_event_peak_dbfs=-16.0,
            plosive_event_contrast_db=18.0,
            plosive_event_rise_time_ms=18.0,
            plosive_event_recovery_ms=90.0,
            line_noise_score=0.84,
            line_noise_frequency_hz=60.0,
            line_noise_prominence_db=24.0,
            line_noise_harmonic_count=3,
            line_noise_harmonics=(
                {"frequency_hz": 60.0, "prominence_db": 24.0},
                {"frequency_hz": 120.0, "prominence_db": 18.0},
                {"frequency_hz": 180.0, "prominence_db": 12.0},
            ),
        )
    )

    assert recommendation.selected_strategy.strategy_id == (
        "manual_plosive_mic_technique_review"
    )
    assert recommendation.selected_strategy.method_family == "analysis_only"
    assert recommendation.evidence["operator_explanation_sections"][
        "selection_policy"
    ] == "live_plosive_review_priority"
    params = recommendation.selected_strategy.parameters
    assert params["live_plosive_review_priority"] is True
    assert params["forbidden_without_wing_proof"] == "dynamic_hpf_static_hpf"
    alternatives = {candidate.strategy_id: candidate for candidate in recommendation.alternatives}
    assert alternatives["di_ground_lift_line_check"].parameters[
        "live_line_noise_review_priority"
    ] is True
    assert alternatives["mains_hum_notch_stack"].blockers == (
        "live_line_noise_eq_requires_operator_review_and_wing_proof",
        "live_eq_cut_exceeds_conservative_limit",
    )
    tradeoffs = {
        row["strategy_id"]: row
        for row in recommendation.evidence["strategy_search"]["alternative_tradeoffs"]
    }
    assert tradeoffs["di_ground_lift_line_check"]["why_not_selected"] == (
        "live_plosive_review_priority:"
        "mic_technique_review_precedes_hum_path_when_plosives_are_active"
    )
    assert tradeoffs["mains_hum_notch_stack"]["why_not_selected"] == (
        "live_plosive_review_priority:"
        "mic_technique_review_precedes_hum_path_when_plosives_are_active"
    )


def test_dynamic_muddy_vocal_prefers_wing_native_deq_low_mid_cut():
    recommendation = analyze_smart_dsp_strategy(
        _analysis(
            target_id="ch:5",
            source_role="lead_vocal",
            problem_hints=("muddy",),
            muddiness_score=0.86,
            low_mid_bloom_frequency_hz=310.0,
            low_mid_bloom_db=9.4,
            low_mid_bloom_peak_dbfs=-16.0,
            low_mid_bloom_rise_time_ms=20.0,
            low_mid_bloom_recovery_ms=120.0,
            dynamics_variance_score=0.64,
            crest_factor_db=18.0,
            lra_proxy_db=15.0,
            estimated_lufs=-24.0,
            level_plane_error_lu=0.8,
        )
    )

    assert recommendation.status == "proposal_only"
    assert recommendation.selected_strategy.strategy_id == "vocal_dynamic_low_mid_de_mud"
    assert recommendation.selected_strategy.method_family == "dynamic_eq"
    params = recommendation.selected_strategy.parameters
    assert params["filter_type"] == "dynamic_peaking"
    assert params["frequency_hz"] == 310.0
    assert params["frequency_source"] == "measured_low_mid_bloom_peak"
    assert params["low_mid_bloom_frequency_hz"] == 310.0
    assert params["static_gain_db"] == -0.6
    assert params["gain_db"] == -0.6
    assert params["max_gain_reduction_db"] == 3.5
    assert params["q"] == 1.6
    assert params["threshold_dbfs"] == -21.7
    assert params["attack_ms"] == 16.0
    assert params["release_ms"] == 150.0
    assert params["wing_dynamic_eq_model_osc_value"] == WING_DYNAMIC_EQ_GATE_MODEL
    assert params["wing_dynamic_eq_filter_osc_value"] == "BP"
    assert params["wing_dynamic_eq_gain_osc_value"] == -3.5
    assert params["wing_dynamic_eq_frequency_osc_value"] == 310.0
    assert params["wing_dynamic_eq_q_osc_value"] == 1.6
    assert recommendation.evidence["operator_explanation_sections"][
        "selection_policy"
    ] == "weighted_strategy_score"

    alternatives = {candidate.strategy_id: candidate for candidate in recommendation.alternatives}
    assert "low_mid_corrective_cut" in alternatives
    assert "vocal_low_mid_multiband_control" in alternatives
    assert params["max_gain_reduction_db"] > abs(params["static_gain_db"])
    assert params["makeup_gain_db"] == 0.0
    assert params["muddy_dynamic_parameter_source"] == "measured_low_mid_bloom"
    assert params["low_mid_bloom_db"] == 9.4
    assert params["low_mid_bloom_peak_dbfs"] == -16.0
    assert params["low_mid_bloom_rise_time_ms"] == 20.0
    assert params["low_mid_bloom_recovery_ms"] == 120.0
    static = alternatives["low_mid_corrective_cut"]
    assert static.method_family == "eq"
    assert static.parameters["filter_type"] == "peaking"
    assert static.parameters["gain_db"] == -3.3
    assert static.future_wing_readback_contract["wing_osc_applicability"][
        "native_osc_supported"
    ] is True
    multiband = alternatives["vocal_low_mid_multiband_control"]
    assert "multiband_change_requires_operator_review" in multiband.blockers
    multiband_params = multiband.parameters
    assert multiband_params["multiband_parameter_source"] == "measured_low_mid_bloom"
    assert multiband_params["frequency_source"] == "measured_low_mid_bloom_peak"
    assert multiband_params["band_center_hz"] == 310.0
    assert multiband_params["low_mid_bloom_frequency_hz"] == 310.0
    assert multiband_params["band_low_hz"] < 310.0 < multiband_params["band_high_hz"]
    assert multiband_params["band_low_hz"] != 160.0
    assert multiband_params["band_high_hz"] != 450.0
    assert multiband_params["threshold_dbfs"] == -19.3
    assert multiband_params["ratio"] == 2.3
    assert multiband_params["attack_ms"] == 22.0
    assert multiband_params["release_ms"] == 198.0
    assert multiband_params["low_mid_bloom_db"] == 9.4
    assert multiband_params["low_mid_bloom_peak_dbfs"] == -16.0
    assert multiband_params["low_mid_bloom_rise_time_ms"] == 20.0
    assert multiband_params["low_mid_bloom_recovery_ms"] == 120.0
    tradeoffs = {
        row["strategy_id"]: row
        for row in recommendation.evidence["strategy_search"]["alternative_tradeoffs"]
    }
    assert "multiband_change_requires_operator_review" in tradeoffs[
        "vocal_low_mid_multiband_control"
    ]["blockers"]
    assert tradeoffs["low_mid_corrective_cut"]["why_not_selected"].startswith(
        "lower_"
    )
    readback = recommendation.selected_strategy.future_wing_readback_contract
    assert readback["wing_osc_applicability"]["native_osc_supported"] is True
    assert readback["wing_osc_applicability"]["native_osc_support_level"] == (
        "native_channel_gate_deq_dynamic_eq_osc_path_templates_declared"
    )
    assert readback["dynamic_eq_readback"][
        "wing_dynamic_eq_model_readback_must_equal"
    ] == WING_DYNAMIC_EQ_GATE_MODEL
    assert "eq_band_readback" not in readback
    assert "dynamics_readback" not in readback
    assert recommendation.to_operator_proposal()["can_apply_live"] is False


def test_live_dynamic_muddy_vocal_prefers_bounded_static_cut_before_dynamic_eq():
    recommendation = analyze_smart_dsp_strategy(
        _analysis(
            target_id="ch:5",
            source_role="lead_vocal",
            mode="live",
            problem_hints=("muddy",),
            operator_intent="live vocal low-mid mud cleanup",
            active_performance=True,
            muddiness_score=0.86,
            low_mid_bloom_frequency_hz=310.0,
            low_mid_bloom_db=9.4,
            low_mid_bloom_peak_dbfs=-16.0,
            low_mid_bloom_rise_time_ms=20.0,
            low_mid_bloom_recovery_ms=120.0,
            dynamics_variance_score=0.64,
            crest_factor_db=18.0,
            lra_proxy_db=15.0,
            estimated_lufs=-24.0,
            level_plane_error_lu=0.8,
            true_peak_headroom_db=5.5,
            feedback_risk=0.1,
        )
    )

    assert recommendation.status == "proposal_only"
    assert recommendation.selected_strategy.strategy_id == "low_mid_corrective_cut"
    assert recommendation.selected_strategy.method_family == "eq"
    params = recommendation.selected_strategy.parameters
    assert params["filter_type"] == "peaking"
    assert params["frequency_hz"] == 310.0
    assert params["frequency_source"] == "measured_low_mid_bloom_peak"
    assert params["low_mid_bloom_frequency_hz"] == 310.0
    assert params["gain_db"] == -2.6
    assert params["q"] == 1.35
    assert params["live_eq_conservative_cut_proposal"] is True
    assert params["live_eq_cut_only_required"] is True
    assert params["live_eq_max_cut_db"] == LIVE_EQ_MAX_CUT_DB
    assert params["live_eq_max_boost_db"] == LIVE_EQ_MAX_BOOST_DB
    assert params["live_eq_protected_band_readback_required"] is True
    assert params["live_eq_model_readback_required"] is True
    assert params["live_eq_global_on_mix_write_forbidden"] is True
    assert params["live_eq_global_on_mix_readback_required"] is True
    assert params["live_eq_band_apply_requires_existing_eq_enabled"] is True
    assert params["live_eq_band_apply_requires_mix_readback"] is True
    assert params["live_eq_band_operation_write_forbidden"] is True
    assert params["live_eq_band_operation_readback_required"] is True
    assert params["live_eq_requires_concrete_band_readback"] is True
    assert (
        params["live_eq_band_operation_apply_blocked_without_concrete_band_readback"]
        is True
    )
    assert params["live_eq_band_operation_review_only_reason"] == (
        "live_eq_must_resolve_concrete_unprotected_band_before_parameter_apply"
    )
    assert params["live_eq_on_review_only_value"] == 1
    assert params["live_eq_mix_review_only_value"] == WING_EQ_MIX_BASELINE_PERCENT
    assert params["live_eq_global_on_mix_review_only_reason"] == (
        "live_eq_must_read_existing_eq_on_and_mix_before_band_parameter_apply"
    )
    assert "wing_eq_on_osc_value" not in params
    assert "wing_eq_mix_osc_value" not in params
    assert "wing_eq_band_operations" not in params
    assert params["live_eq_band_operation_review_only_values"] == [
        {
            "band_readback_required": True,
            "filter_family": "parametric_band",
            "filter_type": "peaking",
            "frequency_hz": 310.0,
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
    assert params["live_eq_feedback_recheck_after_apply_required"] is True
    assert params["live_eq_no_boost_without_operator_wing_proof"] is True
    assert params["live_eq_operator_rehearsal_required"] is True
    assert params["live_eq_live_promotion_status"] == (
        "blocked_until_real_wing_eq_readback_feedback_recheck_and_rollback"
    )
    assert params["live_eq_live_apply_blocked_without_wing_proof"] is True
    eq_readback = recommendation.selected_strategy.future_wing_readback_contract[
        "eq_band_readback"
    ]
    assert eq_readback["live_eq_conservative_cut_proposal"] is True
    assert eq_readback["live_eq_cut_only_required"] is True
    assert eq_readback["live_eq_max_cut_db"] == LIVE_EQ_MAX_CUT_DB
    assert eq_readback["live_eq_max_boost_db"] == LIVE_EQ_MAX_BOOST_DB
    assert eq_readback["live_eq_protected_band_readback_required"] is True
    assert eq_readback["live_eq_model_readback_required"] is True
    assert eq_readback["live_eq_global_on_mix_write_forbidden"] is True
    assert eq_readback["live_eq_global_on_mix_readback_required"] is True
    assert eq_readback["live_eq_band_apply_requires_existing_eq_enabled"] is True
    assert eq_readback["live_eq_band_apply_requires_mix_readback"] is True
    assert eq_readback["live_eq_band_operation_write_forbidden"] is True
    assert eq_readback["live_eq_band_operation_readback_required"] is True
    assert eq_readback["live_eq_requires_concrete_band_readback"] is True
    assert (
        eq_readback[
            "live_eq_band_operation_apply_blocked_without_concrete_band_readback"
        ]
        is True
    )
    assert eq_readback["live_eq_band_operation_review_only_values"] == (
        params["live_eq_band_operation_review_only_values"]
    )
    assert eq_readback["live_eq_band_operation_review_only_reason"] == (
        "live_eq_must_resolve_concrete_unprotected_band_before_parameter_apply"
    )
    assert eq_readback["live_eq_on_review_only_value"] == 1
    assert eq_readback["live_eq_mix_review_only_value"] == (
        WING_EQ_MIX_BASELINE_PERCENT
    )
    assert eq_readback["live_eq_global_on_mix_review_only_reason"] == (
        "live_eq_must_read_existing_eq_on_and_mix_before_band_parameter_apply"
    )
    assert eq_readback["live_eq_feedback_recheck_after_apply_required"] is True
    assert eq_readback["live_eq_no_boost_without_operator_wing_proof"] is True
    assert eq_readback["live_eq_operator_rehearsal_required"] is True
    assert eq_readback["live_eq_live_promotion_status"] == (
        "blocked_until_real_wing_eq_readback_feedback_recheck_and_rollback"
    )
    assert eq_readback["live_eq_live_apply_blocked_without_wing_proof"] is True
    alternatives = {candidate.strategy_id: candidate for candidate in recommendation.alternatives}
    dynamic = alternatives["vocal_dynamic_low_mid_de_mud"]
    assert "live_vocal_dynamic_low_mid_requires_operator_review" in dynamic.blockers
    assert dynamic.parameters["muddy_dynamic_parameter_source"] == "measured_low_mid_bloom"
    assert dynamic.parameters["max_gain_reduction_db"] == 3.5
    assert dynamic.parameters["static_gain_db"] == 0.0
    assert dynamic.parameters["gain_db"] == 0.0
    multiband = alternatives["vocal_low_mid_multiband_control"]
    assert "multiband_change_requires_operator_review" in multiband.blockers
    tradeoffs = {
        row["strategy_id"]: row
        for row in recommendation.evidence["strategy_search"]["alternative_tradeoffs"]
    }
    assert tradeoffs["vocal_dynamic_low_mid_de_mud"]["why_not_selected"] == (
        "live_vocal_low_mid_static_cut_priority:"
        "bounded_static_cut_precedes_dynamic_eq"
    )
    assert tradeoffs["vocal_low_mid_multiband_control"]["why_not_selected"] == (
        "live_vocal_low_mid_static_cut_priority:"
        "bounded_static_cut_precedes_multiband"
    )
    assert recommendation.evidence["operator_explanation_sections"][
        "selection_policy"
    ] == "live_vocal_low_mid_static_cut_priority"
    live_eq_write_keys = []
    _assert_eq_write_key_contract(
        recommendation.selected_strategy,
        selected_write_keys=live_eq_write_keys,
    )
    _assert_selected_eq_apply_values(
        recommendation,
        {},
    )
    assert recommendation.to_operator_proposal()["can_apply_live"] is False


def test_live_muddy_vocal_hot_summed_bus_prefers_bus_headroom_pullback_before_eq():
    recommendation = analyze_smart_dsp_strategy(
        _analysis(
            target_id="ch:5",
            source_role="lead_vocal",
            mode="live",
            problem_hints=("muddy",),
            operator_intent="live vocal low-mid mud cleanup",
            active_performance=True,
            muddiness_score=0.86,
            low_mid_bloom_frequency_hz=310.0,
            low_mid_bloom_db=9.4,
            low_mid_bloom_peak_dbfs=-16.0,
            low_mid_bloom_rise_time_ms=20.0,
            low_mid_bloom_recovery_ms=120.0,
            dynamics_variance_score=0.64,
            crest_factor_db=18.0,
            lra_proxy_db=15.0,
            estimated_lufs=-24.0,
            level_plane_error_lu=0.8,
            true_peak_headroom_db=5.5,
            feedback_risk=0.1,
            summed_bus_evidence_available=True,
            summed_bus_role="vocal_bus",
            summed_bus_target_id="bus:5",
            summed_bus_true_peak_headroom_db=0.4,
            summed_bus_level_plane_error_lu=0.0,
        )
    )

    assert recommendation.status == "proposal_only"
    assert recommendation.selected_strategy.strategy_id == (
        "live_summed_bus_headroom_pullback_before_source_processing"
    )
    assert recommendation.selected_strategy.method_family == "gain_fader"
    params = recommendation.selected_strategy.parameters
    assert params["target_scope"] == "summed_bus_headroom_before_source_processing"
    assert params["source_family"] == "vocal"
    assert params["bus_delta_db"] == -1.1
    assert params["delta_db"] == -1.1
    assert params["bounded_delta_db"] == 1.1
    assert params["source_delta_db"] == 0.0
    assert params["summed_bus_role"] == "vocal_bus"
    assert params["summed_bus_target_id"] == "bus:5"
    assert params["wing_osc_target_id_override"] == "bus:5"
    assert params["wing_osc_original_target_id"] == "ch:5"
    assert params["wing_osc_effective_target_id"] == "bus:5"
    assert params["requires_summed_bus_readback"] is True
    assert params["live_summed_bus_pullback_only_proposal"] is True
    assert params["live_source_processing_deferred_until_bus_recheck"] is True
    assert params["live_summed_bus_true_peak_recheck_required"] is True
    assert params["live_summed_bus_feedback_recheck_required"] is True
    assert params["live_stepped_change_required"] is True
    assert params["live_readback_after_each_step_required"] is True
    assert params["live_max_single_step_db"] == 0.75
    assert recommendation.evidence["operator_explanation_sections"][
        "selection_policy"
    ] == "live_summed_bus_headroom_pullback_priority"
    _assert_gain_delta_readback_contract(
        recommendation.selected_strategy,
        source="bus_delta_db",
        target_surface="bus",
        expected_delta=-1.1,
        path_prefix="/bus/{bus}",
    )
    _assert_selected_gain_delta_apply_values(recommendation, -1.1)
    alternatives = {
        candidate.strategy_id: candidate for candidate in recommendation.alternatives
    }
    assert "low_mid_corrective_cut" in alternatives
    assert "vocal_dynamic_low_mid_de_mud" in alternatives
    tradeoffs = {
        row["strategy_id"]: row["why_not_selected"]
        for row in recommendation.evidence["strategy_search"]["alternative_tradeoffs"]
    }
    assert tradeoffs["low_mid_corrective_cut"] == (
        "summed_bus_headroom_priority:"
        "restore_bus_headroom_before_source_processing"
    )
    proposal = recommendation.to_operator_proposal()
    assert proposal["live_apply_allowed"] is False
    assert proposal["live_authority_granted"] is False
    assert proposal["live_write"] is False


def test_measured_guitar_low_mid_bloom_prefers_wing_native_deq_cut():
    soundcheck = analyze_smart_dsp_strategy(
        _analysis(
            target_id="ch:14",
            source_role="electric_guitar",
            source_label="GTR L",
            problem_hints=("muddy",),
            operator_intent="soundcheck guitar low-mid bloom cleanup",
            muddiness_score=0.78,
            mud_frequency_hz=310.0,
            low_mid_bloom_frequency_hz=295.0,
            low_mid_bloom_db=8.8,
            low_mid_bloom_peak_dbfs=-17.4,
            low_mid_bloom_rise_time_ms=26.0,
            low_mid_bloom_recovery_ms=180.0,
            dynamics_variance_score=0.5,
            crest_factor_db=17.0,
            estimated_lufs=-24.0,
            level_plane_error_lu=0.6,
            true_peak_headroom_db=5.5,
            feedback_risk=0.11,
            low_end_excess_score=0.42,
            spectral_tilt_score=0.5,
        )
    )

    assert soundcheck.status == "proposal_only"
    assert soundcheck.selected_strategy.strategy_id == "instrument_dynamic_low_mid_de_mud"
    assert soundcheck.selected_strategy.method_family == "dynamic_eq"
    params = soundcheck.selected_strategy.parameters
    assert params["filter_type"] == "dynamic_peaking"
    assert params["frequency_hz"] == 295.0
    assert params["frequency_source"] == "measured_low_mid_bloom_peak"
    assert params["mud_frequency_hz"] == 310.0
    assert params["low_mid_bloom_frequency_hz"] == 295.0
    assert params["static_gain_db"] == -0.4
    assert params["gain_db"] == -0.4
    assert params["q"] == 1.5
    assert params["threshold_dbfs"] == -21.8
    assert params["max_gain_reduction_db"] == 3.2
    assert params["attack_ms"] == 23.4
    assert params["release_ms"] == 243.0
    assert params["wing_dynamic_eq_model_osc_value"] == WING_DYNAMIC_EQ_GATE_MODEL
    assert params["wing_dynamic_eq_filter_osc_value"] == "BP"
    assert params["wing_dynamic_eq_gain_osc_value"] == -3.2
    assert params["wing_dynamic_eq_frequency_osc_value"] == 295.0
    assert params["wing_dynamic_eq_q_osc_value"] == 1.5
    assert soundcheck.evidence["operator_explanation_sections"][
        "selection_policy"
    ] == "soundcheck_instrument_low_mid_bloom_priority"
    alternatives = {candidate.strategy_id: candidate for candidate in soundcheck.alternatives}
    assert "guitar_pick_sustain_compression" in alternatives
    assert "low_mid_corrective_cut" in alternatives
    assert params["instrument_low_mid_parameter_source"] == "measured_low_mid_bloom"
    assert params["makeup_gain_db"] == 0.0
    tradeoffs = {
        row["strategy_id"]: row
        for row in soundcheck.evidence["strategy_search"]["alternative_tradeoffs"]
    }
    assert tradeoffs["low_mid_corrective_cut"]["why_not_selected"].startswith(
        "lower_"
    )
    readback = soundcheck.selected_strategy.future_wing_readback_contract
    assert readback["wing_osc_applicability"]["native_osc_supported"] is True
    assert readback["wing_osc_applicability"]["native_osc_support_level"] == (
        "native_channel_gate_deq_dynamic_eq_osc_path_templates_declared"
    )
    assert readback["dynamic_eq_readback"][
        "wing_dynamic_eq_model_readback_must_equal"
    ] == WING_DYNAMIC_EQ_GATE_MODEL
    assert "eq_band_readback" not in readback
    assert "dynamics_readback" not in readback
    assert soundcheck.to_operator_proposal()["can_apply_live"] is False

    live = analyze_smart_dsp_strategy(
        _analysis(
            target_id="ch:14",
            source_role="electric_guitar",
            source_label="GTR L",
            mode="live",
            problem_hints=("muddy",),
            operator_intent="live guitar low-mid bloom cleanup",
            muddiness_score=0.78,
            mud_frequency_hz=310.0,
            low_mid_bloom_frequency_hz=295.0,
            low_mid_bloom_db=8.8,
            low_mid_bloom_peak_dbfs=-17.4,
            low_mid_bloom_rise_time_ms=26.0,
            low_mid_bloom_recovery_ms=180.0,
            dynamics_variance_score=0.5,
            crest_factor_db=17.0,
            estimated_lufs=-24.0,
            level_plane_error_lu=0.6,
            true_peak_headroom_db=5.5,
            feedback_risk=0.11,
            low_end_excess_score=0.42,
            spectral_tilt_score=0.5,
        )
    )

    assert live.status == "proposal_only"
    assert live.selected_strategy.strategy_id == "low_mid_corrective_cut"
    assert live.selected_strategy.parameters["frequency_hz"] == 295.0
    assert live.selected_strategy.parameters["frequency_source"] == (
        "measured_low_mid_bloom_peak"
    )
    assert live.selected_strategy.parameters["gain_db"] == -2.6
    assert live.evidence["operator_explanation_sections"][
        "selection_policy"
    ] == "live_instrument_low_mid_static_cut_priority"
    dynamic_live = [
        candidate
        for candidate in live.alternatives
        if candidate.strategy_id == "instrument_dynamic_low_mid_de_mud"
    ][0]
    assert (
        "live_dynamic_instrument_low_mid_requires_operator_review"
        in dynamic_live.blockers
    )
    assert live.to_operator_proposal()["can_apply_live"] is False


def test_soundcheck_fx_wash_muddy_vocal_prefers_fx_cleanup_before_low_mid_eq():
    recommendation = analyze_smart_dsp_strategy(
        _analysis(
            target_id="ch:5",
            source_role="lead_vocal",
            source_label="Lead Vox",
            problem_hints=("muddy", "fx_intelligibility"),
            operator_intent="soundcheck vocal muddy from reverb wash",
            muddiness_score=0.62,
            low_end_excess_score=0.22,
            spectral_tilt_score=0.22,
            level_plane_error_lu=0.3,
            fx_wetness_score=0.82,
            fx_decay_risk_score=0.78,
            fx_delay_smear_score=0.42,
            fx_return_headroom_db=4.5,
            monitor_feedback_risk=0.14,
        )
    )

    assert recommendation.status == "proposal_only"
    assert recommendation.selected_strategy.strategy_id == "reduce_fx_send_or_return"
    assert recommendation.selected_strategy.method_family == "fx"
    params = recommendation.selected_strategy.parameters
    assert params["send_delta_db"] < 0.0
    assert params["return_delta_db"] == 0.0
    assert params["fx_level_pullback_parameter_source"] == "fx_wet_decay_score_fallback"
    assert params["native_fx_level_pullback_db"] == 4.0
    assert params["native_fx_level_pullback_legacy_db"] == 4.0
    assert params["measured_tail_decay_s"] == 0.0
    assert params["measured_delay_repeat_ms"] == 0.0
    assert params["post_pullback_time_edit_review_required"] is True
    _assert_fx_send_level_readback_contract(
        recommendation.selected_strategy,
        expected_delta=params["send_delta_db"],
        target_surface="channel",
        source_binding_required=False,
    )
    _assert_selected_fx_apply_values(
        recommendation,
        {"wing_fx_send_level_delta_db": -4.0},
    )
    assert recommendation.evidence["operator_explanation_sections"][
        "selection_policy"
    ] == "soundcheck_wing_native_osc_fx_level_pullback_priority"
    wing_osc = recommendation.selected_strategy.future_wing_readback_contract[
        "wing_osc_applicability"
    ]
    assert wing_osc["native_osc_supported"] is True
    assert wing_osc["native_osc_support_level"] == (
        "native_fx_level_osc_path_templates_declared"
    )
    alternatives = {candidate.strategy_id: candidate for candidate in recommendation.alternatives}
    decay_edit = alternatives["shorten_fx_decay_filter"]
    assert decay_edit.parameters["decay_s"] < 2.2
    assert decay_edit.parameters["pre_delay_ms"] >= 45.0
    assert decay_edit.parameters["fx_decay_parameter_source"] == (
        "fx_decay_risk_role_target_fallback"
    )
    assert decay_edit.parameters["fx_target_context"] == "vocal_reverb"
    assert decay_edit.parameters["fx_tail_density_score"] > 0.5
    assert decay_edit.future_wing_readback_contract["wing_osc_applicability"][
        "native_osc_supported"
    ] is False
    _assert_fx_model_specific_guard(
        decay_edit,
        {
            "decay_s",
            "pre_delay_ms",
            "return_hpf_hz",
            "return_lpf_hz",
            "delay_feedback_percent",
        },
    )
    assert "low_mid_corrective_cut" in alternatives
    tradeoffs = {
        row["strategy_id"]: row
        for row in recommendation.evidence["strategy_search"]["alternative_tradeoffs"]
    }
    assert tradeoffs["shorten_fx_decay_filter"]["why_not_selected"] == (
        "soundcheck_wing_native_osc_priority:"
        "native_fx_level_pullback_has_verified_wing_fw_3_0_5_osc_path_"
        "alternative_requires_model_mapping"
    )
    _assert_tradeoff_wing_osc_summary(
        tradeoffs["shorten_fx_decay_filter"],
        native_supported=False,
        support_level="partial_model_specific_fx_mapping_required",
    )
    assert tradeoffs["shorten_fx_decay_filter"][
        "wing_osc_unsupported_strategy_reason"
    ] == "fx_strategy_requires_wing_fx_slot_model_and_parameter_mapping_before_apply"
    assert set(
        tradeoffs["shorten_fx_decay_filter"]["wing_osc_unsupported_parameter_keys"]
    ) >= {
        "decay_s",
        "pre_delay_ms",
        "return_hpf_hz",
        "return_lpf_hz",
        "delay_feedback_percent",
    }
    assert tradeoffs["shorten_fx_decay_filter"]["wing_osc_candidate_apply_status"] == (
        "not_wing_applicable_until_mapping_or_operator_review"
    )
    assert (
        tradeoffs["low_mid_corrective_cut"]["why_not_selected"]
        == "fx_wash_cleanup_priority:clean_fx_return_before_channel_tone_cut"
    )
    _assert_tradeoff_wing_osc_summary(
        tradeoffs["low_mid_corrective_cut"],
        native_supported=True,
        support_level="native_osc_path_templates_declared",
    )
    assert tradeoffs["low_mid_corrective_cut"]["score_delta_vs_selected"] < 0.0
    assert recommendation.to_operator_proposal()["can_apply_live"] is False


def test_operator_owned_fx_level_blocks_soundcheck_fx_pullback_apply():
    recommendation = analyze_smart_dsp_strategy(
        _analysis(
            target_id="ch:5",
            source_role="lead_vocal",
            source_label="Lead Vox",
            problem_hints=("fx_intelligibility",),
            operator_intent="soundcheck vocal wet FX from operator-owned FX level",
            muddiness_score=0.05,
            low_end_excess_score=0.0,
            spectral_tilt_score=0.0,
            level_plane_error_lu=0.0,
            fx_wetness_score=0.95,
            fx_decay_risk_score=0.4,
            fx_delay_smear_score=0.0,
            fx_return_headroom_db=4.5,
            monitor_feedback_risk=0.1,
            current_wing_fx_level_operator_owned=True,
        )
    )

    assert recommendation.status == "proposal_only"
    selected = recommendation.selected_strategy
    assert selected.strategy_id == "hold_soundcheck_operator_owned_fx_for_review"
    assert selected.method_family == "analysis_only"
    assert selected.blockers == ()
    assert recommendation.evidence["operator_explanation_sections"][
        "selection_policy"
    ] == "soundcheck_operator_owned_fx_review_priority"

    params = selected.parameters
    assert params["reason"] == "soundcheck_operator_owned_fx_requires_operator_review"
    assert set(params["blocked_fx_strategy_ids"]) >= {
        "reduce_fx_send_or_return",
        "shorten_fx_decay_filter",
        "increase_fx_depth",
    }
    assert WING_FX_LEVEL_OPERATOR_OWNED_BLOCKER in (
        params["blocked_fx_operator_owned_blockers"]
    )
    assert params["blocked_fx_send_delta_db"] == -4.3
    assert params["blocked_fx_return_delta_db"] == 0.0
    assert params["wing_fx_level_current_operator_owned"] is True
    assert params["wing_fx_level_operator_owned_edit_requires_review"] is True
    assert (
        params["wing_fx_level_operator_owned_blocker"]
        == WING_FX_LEVEL_OPERATOR_OWNED_BLOCKER
    )
    assert params["soundcheck_operator_owned_fx_hold_proposal"] is True
    assert params["soundcheck_operator_owned_fx_target_surface"] == "channel"
    assert params["soundcheck_operator_owned_fx_level_readback_required"] is True
    assert params["soundcheck_operator_owned_fx_model_readback_required"] is True
    assert (
        params["soundcheck_operator_owned_fx_monitor_feedback_recheck_required"]
        is True
    )
    assert (
        params["soundcheck_operator_owned_fx_return_headroom_recheck_required"]
        is True
    )
    assert params["soundcheck_operator_owned_fx_rollback_required"] is True
    assert (
        params[
            "soundcheck_operator_owned_fx_no_level_model_time_or_ducking_without_operator_review"
        ]
        is True
    )
    assert params["required_context"] == (
        "operator_review_current_wing_fx_send_return_model_owner_readback_rollback"
    )
    assert params["forbidden_until_reviewed"] == (
        "soundcheck_fx_send_return_model_time_ducking_queue_import"
    )

    readback = selected.future_wing_readback_contract[
        "soundcheck_operator_owned_fx_review_readback"
    ]
    assert readback["read_only"] is True
    assert readback["allowed_live_write_scope_before_wing_proof"] == []
    assert readback["soundcheck_operator_owned_fx_hold_proposal"] is True
    assert readback["wing_fx_level_current_operator_owned"] is True
    assert readback["wing_fx_level_operator_owned_blocker"] == (
        WING_FX_LEVEL_OPERATOR_OWNED_BLOCKER
    )
    assert "send/{send}/lvl" in readback["read_before_paths"]
    assert "fxmix" in readback["read_before_paths"]

    alternatives = {
        candidate.strategy_id: candidate for candidate in recommendation.alternatives
    }
    fx_pullback = alternatives["reduce_fx_send_or_return"]
    assert fx_pullback.blockers == (WING_FX_LEVEL_OPERATOR_OWNED_BLOCKER,)
    assert fx_pullback.parameters["wing_fx_send_level_delta_db"] == -4.3
    tradeoffs = {
        row["strategy_id"]: row
        for row in recommendation.evidence["strategy_search"]["alternative_tradeoffs"]
    }
    assert tradeoffs["reduce_fx_send_or_return"]["why_not_selected"] == (
        "soundcheck_operator_owned_fx_review_priority:"
        "manual_fx_level_or_model_owner_review_precedes_soundcheck_fx_change"
    )
    _assert_selected_has_no_apply_values(recommendation)


def test_live_fx_wash_muddy_vocal_prefers_native_fx_pullback_before_eq():
    recommendation = analyze_smart_dsp_strategy(
        _analysis(
            target_id="ch:5",
            source_role="lead_vocal",
            source_label="Lead Vox",
            mode="live",
            problem_hints=("muddy", "fx_intelligibility"),
            operator_intent="live vocal muddy from reverb wash",
            muddiness_score=0.72,
            low_mid_bloom_frequency_hz=310.0,
            low_mid_bloom_db=8.6,
            low_mid_bloom_peak_dbfs=-17.0,
            low_mid_bloom_rise_time_ms=24.0,
            low_mid_bloom_recovery_ms=130.0,
            low_end_excess_score=0.2,
            spectral_tilt_score=0.24,
            level_plane_error_lu=0.2,
            true_peak_headroom_db=4.2,
            feedback_risk=0.12,
            monitor_feedback_risk=0.18,
            dynamics_variance_score=0.62,
            crest_factor_db=18.0,
            lra_proxy_db=14.0,
            fx_wetness_score=0.84,
            fx_decay_risk_score=0.72,
            fx_delay_smear_score=0.36,
            fx_return_headroom_db=4.4,
        )
    )

    assert recommendation.status == "proposal_only"
    assert recommendation.selected_strategy.strategy_id == "reduce_fx_send_or_return"
    assert recommendation.selected_strategy.method_family == "fx"
    params = recommendation.selected_strategy.parameters
    assert params["send_delta_db"] == -3.0
    assert params["return_delta_db"] == 0.0
    assert params["fx_level_pullback_parameter_source"] == "fx_wet_decay_score_fallback"
    assert params["native_fx_level_pullback_db"] == 3.0
    assert params["native_fx_level_pullback_legacy_db"] == 4.0
    assert params["live_fx_level_pullback_conservative_cap_db"] == 3.0
    assert params["live_fx_pullback_only_proposal"] is True
    assert params["live_fx_level_reduction_only_required"] is True
    assert params["live_fx_no_model_or_time_parameter_write_required"] is True
    assert params["live_fx_live_apply_blocked_without_wing_proof"] is True
    assert params["live_stepped_change_required"] is True
    assert params["live_max_single_step_db"] == LIVE_MAX_SINGLE_STEP_DB
    assert params["live_readback_after_each_step_required"] is True
    _assert_fx_send_level_readback_contract(
        recommendation.selected_strategy,
        expected_delta=params["send_delta_db"],
        target_surface="channel",
        source_binding_required=False,
    )
    _assert_selected_fx_apply_values(
        recommendation,
        {"wing_fx_send_level_delta_db": -3.0},
    )
    assert recommendation.evidence["operator_explanation_sections"][
        "selection_policy"
    ] == "live_fx_wash_cleanup_priority"
    alternatives = {candidate.strategy_id: candidate for candidate in recommendation.alternatives}
    assert "low_mid_corrective_cut" in alternatives
    assert "vocal_dynamic_low_mid_de_mud" in alternatives
    assert "live_vocal_dynamic_low_mid_requires_operator_review" in alternatives[
        "vocal_dynamic_low_mid_de_mud"
    ].blockers
    tradeoffs = {
        row["strategy_id"]: row
        for row in recommendation.evidence["strategy_search"]["alternative_tradeoffs"]
    }
    assert tradeoffs["low_mid_corrective_cut"]["why_not_selected"] == (
        "live_fx_wash_cleanup_priority:reduce_wet_level_before_channel_tone_cut"
    )
    assert tradeoffs["vocal_dynamic_low_mid_de_mud"]["why_not_selected"] == (
        "live_fx_wash_cleanup_priority:reduce_wet_level_before_channel_tone_cut"
    )
    _assert_tradeoff_wing_osc_summary(
        tradeoffs["low_mid_corrective_cut"],
        native_supported=True,
        support_level="native_osc_path_templates_declared",
    )
    proposal = recommendation.to_operator_proposal()
    assert proposal["can_apply_live"] is False
    assert proposal["live_authority_granted"] is False
    assert proposal["live_write"] is False


def test_live_operator_owned_fx_level_holds_instead_of_channel_eq_fallback():
    recommendation = analyze_smart_dsp_strategy(
        _analysis(
            target_id="ch:5",
            source_role="lead_vocal",
            source_label="Lead Vox",
            mode="live",
            problem_hints=("muddy", "fx_intelligibility"),
            operator_intent="live vocal muddy from operator-owned reverb wash",
            muddiness_score=0.72,
            low_mid_bloom_frequency_hz=310.0,
            low_mid_bloom_db=8.6,
            low_mid_bloom_peak_dbfs=-17.0,
            low_mid_bloom_rise_time_ms=24.0,
            low_mid_bloom_recovery_ms=130.0,
            low_end_excess_score=0.2,
            spectral_tilt_score=0.24,
            level_plane_error_lu=0.2,
            true_peak_headroom_db=4.2,
            feedback_risk=0.12,
            monitor_feedback_risk=0.18,
            dynamics_variance_score=0.62,
            crest_factor_db=18.0,
            lra_proxy_db=14.0,
            fx_wetness_score=0.84,
            fx_decay_risk_score=0.72,
            fx_delay_smear_score=0.36,
            fx_return_headroom_db=4.4,
            current_wing_fx_level_operator_owned=True,
        )
    )

    assert recommendation.selected_strategy.strategy_id == (
        "hold_live_operator_owned_fx_for_review"
    )
    assert recommendation.selected_strategy.method_family == "analysis_only"
    assert recommendation.evidence["operator_explanation_sections"][
        "selection_policy"
    ] == "live_operator_owned_fx_review_priority"
    params = recommendation.selected_strategy.parameters
    assert params["dsp_change"] == "hold"
    assert params["blocked_fx_strategy_ids"] == [
        "reduce_fx_send_or_return",
        "shorten_fx_decay_filter",
        "increase_fx_depth",
    ]
    assert params["blocked_fx_operator_owned_blockers"] == [
        WING_FX_LEVEL_OPERATOR_OWNED_BLOCKER
    ]
    assert params["blocked_fx_send_delta_db"] == -3.0
    assert params["blocked_fx_return_delta_db"] == 0.0
    assert params["wing_fx_level_current_operator_owned"] is True
    assert params["wing_fx_level_operator_owned_edit_requires_review"] is True
    assert (
        params["wing_fx_level_operator_owned_blocker"]
        == WING_FX_LEVEL_OPERATOR_OWNED_BLOCKER
    )
    assert params["wing_rack_fw305_verified_osc_mapping_required"] is True
    assert params["wing_rack_fw305_fx_send_return_readback_required"] is True
    assert params["wing_rack_fw305_rollback_required"] is True
    assert params["live_authority_granted"] is False
    assert params["live_write"] is False
    assert params["readiness_closure_allowed"] is False

    alternatives = {candidate.strategy_id: candidate for candidate in recommendation.alternatives}
    assert WING_FX_LEVEL_OPERATOR_OWNED_BLOCKER in alternatives[
        "reduce_fx_send_or_return"
    ].blockers
    assert "low_mid_corrective_cut" in alternatives
    tradeoffs = {
        row["strategy_id"]: row
        for row in recommendation.evidence["strategy_search"]["alternative_tradeoffs"]
    }
    assert tradeoffs["reduce_fx_send_or_return"]["why_not_selected"] == (
        "live_operator_owned_fx_review_priority:"
        "manual_fx_level_or_model_owner_review_precedes_live_fx_change"
    )
    assert tradeoffs["low_mid_corrective_cut"]["why_not_selected"] == (
        "live_operator_owned_fx_review_priority:"
        "manual_fx_level_review_precedes_channel_tone_cut"
    )
    proposal = recommendation.to_operator_proposal()
    selected_summary = proposal["selected_wing_osc_summary"]
    assert selected_summary["wing_osc_candidate_apply_status"] == (
        "analysis_only_no_console_write"
    )
    assert selected_summary["wing_osc_apply_parameter_values"] == {}
    assert selected_summary["wing_osc_live_authority_granted"] is False
    assert selected_summary["wing_osc_live_write"] is False
    assert proposal["can_apply_live"] is False
    assert proposal["live_authority_granted"] is False
    assert proposal["live_write"] is False


def test_soundcheck_dull_vocal_selects_bounded_presence_air_lift():
    recommendation = analyze_smart_dsp_strategy(
        _analysis(
            problem_hints=("dull_presence",),
            muddiness_score=0.0,
            harshness_score=0.0,
            sibilance_score=0.0,
            presence_deficit_score=0.82,
            presence_lift_frequency_hz=3150.0,
            presence_lift_bandwidth_hz=3900.0,
            presence_lift_parameter_source="measured_presence_air_deficit",
            true_peak_headroom_db=6.0,
            feedback_risk=0.12,
        )
    )

    assert recommendation.status == "proposal_only"
    assert recommendation.selected_strategy.strategy_id == "soundcheck_presence_air_lift"
    assert recommendation.selected_strategy.method_family == "eq"
    params = recommendation.selected_strategy.parameters
    assert params["filter_type"] == "high_shelf"
    assert params["frequency_hz"] == 3150.0
    assert params["frequency_source"] == "measured_presence_air_deficit"
    assert params["presence_lift_bandwidth_hz"] == 3900.0
    assert params["presence_lift_parameter_source"] == "measured_presence_air_deficit"
    assert params["q"] == 0.81
    assert params["gain_db"] > 0.0
    assert params["bounded_boost_db"] <= 2.4
    assert params["requires_true_peak_headroom_db"] > params["gain_db"]
    assert params["soundcheck_eq_bounded_boost_proposal"] is True
    assert params["soundcheck_eq_boost_max_db"] == SOUNDCHECK_EQ_MAX_BOOST_DB
    assert (
        params["soundcheck_eq_boost_min_headroom_margin_db"]
        == SOUNDCHECK_EQ_MIN_HEADROOM_MARGIN_DB
    )
    assert params["soundcheck_eq_boost_required_true_peak_headroom_db"] == 3.3
    assert (
        params["soundcheck_eq_boost_required_summed_bus_true_peak_headroom_db"]
        == 3.3
    )
    assert params["soundcheck_eq_boost_requires_true_peak_headroom"] is True
    assert params["soundcheck_eq_boost_requires_summed_bus_headroom"] is True
    assert params["soundcheck_eq_boost_feedback_recheck_required"] is True
    assert params["soundcheck_eq_boost_monitor_feedback_recheck_required"] is True
    assert params["soundcheck_eq_boost_operator_confirmation_required"] is True
    assert params["soundcheck_eq_boost_readback_required"] is True
    assert params["soundcheck_eq_boost_rollback_ready"] is True
    assert params["soundcheck_eq_boost_live_promotion_status"] == (
        "blocked_until_real_wing_eq_readback_headroom_feedback_recheck_and_operator_approval"
    )
    assert params["soundcheck_eq_live_apply_blocked_without_wing_proof"] is True
    assert params["wing_eq_filter_family"] == "high_shelf"
    assert params["wing_eq_shelf_osc_path_suffixes"] == {
        "gain": "eq/hg",
        "frequency": "eq/hf",
        "q": "eq/hq",
        "type": "eq/heq",
    }
    assert params["wing_eq_high_shelf_type_osc_value"] == "SHV"
    assert params["wing_eq_high_shelf_frequency_hz"] == 3150.0
    readback = recommendation.selected_strategy.future_wing_readback_contract
    eq_readback = readback["eq_band_readback"]
    assert eq_readback["read_before_paths"] == [
        "eq/on",
        "eq/mdl",
        "eq/mix",
        "eq/hg",
        "eq/hf",
        "eq/hq",
        "eq/heq",
    ]
    assert eq_readback["wing_eq_value_mapping"]["wing_eq_filter_family"] == (
        "high_shelf"
    )
    assert eq_readback["soundcheck_eq_bounded_boost_proposal"] is True
    assert eq_readback["soundcheck_eq_boost_max_db"] == SOUNDCHECK_EQ_MAX_BOOST_DB
    assert (
        eq_readback["soundcheck_eq_boost_min_headroom_margin_db"]
        == SOUNDCHECK_EQ_MIN_HEADROOM_MARGIN_DB
    )
    assert eq_readback["soundcheck_eq_boost_required_true_peak_headroom_db"] == 3.3
    assert (
        eq_readback["soundcheck_eq_boost_required_summed_bus_true_peak_headroom_db"]
        == 3.3
    )
    assert eq_readback["soundcheck_eq_boost_requires_true_peak_headroom"] is True
    assert eq_readback["soundcheck_eq_boost_requires_summed_bus_headroom"] is True
    assert eq_readback["soundcheck_eq_boost_feedback_recheck_required"] is True
    assert (
        eq_readback["soundcheck_eq_boost_monitor_feedback_recheck_required"]
        is True
    )
    assert eq_readback["soundcheck_eq_boost_operator_confirmation_required"] is True
    assert eq_readback["soundcheck_eq_boost_readback_required"] is True
    assert eq_readback["soundcheck_eq_boost_rollback_ready"] is True
    assert eq_readback["soundcheck_eq_boost_live_promotion_status"] == (
        "blocked_until_real_wing_eq_readback_headroom_feedback_recheck_and_operator_approval"
    )
    assert eq_readback["soundcheck_eq_live_apply_blocked_without_wing_proof"] is True
    assert recommendation.evidence["presence_deficit_score"] == 0.82
    assert recommendation.evidence["presence_lift_frequency_hz"] == 3150.0
    assert recommendation.to_operator_proposal()["live_apply_allowed"] is False


def test_backing_vocal_presence_lift_is_role_aware_and_keeps_support_tucked():
    recommendation = analyze_smart_dsp_strategy(
        _analysis(
            source_role="backing_vocal",
            source_label="BGV 1",
            problem_hints=("dull_presence",),
            muddiness_score=0.0,
            harshness_score=0.0,
            sibilance_score=0.0,
            presence_deficit_score=0.82,
            presence_lift_frequency_hz=3600.0,
            presence_lift_bandwidth_hz=5200.0,
            presence_lift_parameter_source="measured_presence_air_deficit",
            true_peak_headroom_db=6.0,
            feedback_risk=0.12,
        )
    )

    assert recommendation.status == "proposal_only"
    assert recommendation.evidence["musical_role"] == "support"
    assert recommendation.selected_strategy.strategy_id == "soundcheck_presence_air_lift"
    params = recommendation.selected_strategy.parameters
    assert params["musical_role"] == "support"
    assert params["gain_db"] == 1.2
    assert params["role_aware_max_boost_db"] == 1.2
    assert params["role_intent"] == "preserve_support_tuck"
    assert params["frequency_source"] == "measured_presence_air_deficit"
    assert params["presence_lift_bandwidth_hz"] == 5200.0


def test_live_dull_vocal_holds_presence_boost_for_operator_review():
    recommendation = analyze_smart_dsp_strategy(
        _analysis(
            mode="live",
            problem_hints=("dull_presence",),
            muddiness_score=0.0,
            harshness_score=0.0,
            sibilance_score=0.0,
            presence_deficit_score=0.82,
            presence_lift_frequency_hz=3180.0,
            presence_lift_bandwidth_hz=3800.0,
            presence_lift_parameter_source="measured_presence_air_deficit",
            true_peak_headroom_db=6.0,
            feedback_risk=0.12,
            active_performance=True,
        )
    )

    assert recommendation.status == "proposal_only"
    assert recommendation.selected_strategy.strategy_id == "hold_presence_boost_for_live_safety"
    assert recommendation.selected_strategy.method_family == "analysis_only"
    params = recommendation.selected_strategy.parameters
    assert params["required_context"] == (
        "source_identity_eq_band_headroom_feedback_readback_rollback"
    )
    assert params["forbidden_without_wing_proof"] == (
        "presence_boost_eq_gain_up_or_dynamic_presence_change"
    )
    assert params["presence_boost_hold_proposal"] is True
    assert params["live_presence_boost_hold_proposal"] is True
    assert params["presence_boost_target_surface"] == "channel"
    assert params["presence_boost_source_identity_required"] is True
    assert params["presence_boost_eq_band_readback_required"] is True
    assert params["presence_boost_previous_eq_readback_required"] is True
    assert params["presence_boost_feedback_recheck_required"] is True
    assert params["presence_boost_monitor_feedback_recheck_required"] is True
    assert params["presence_boost_true_peak_headroom_recheck_required"] is True
    assert params["presence_boost_summed_bus_headroom_recheck_required"] is True
    assert params["presence_boost_rollback_ready"] is True
    assert params["presence_boost_no_boost_without_operator_wing_proof"] is True
    assert params["live_presence_boost_live_apply_blocked_without_wing_proof"] is True
    assert "wing_eq_on_osc_value" not in params
    assert "wing_eq_mix_osc_value" not in params
    readback = recommendation.selected_strategy.future_wing_readback_contract[
        "presence_boost_hold_readback"
    ]
    assert readback["read_only"] is True
    assert readback["presence_boost_target_surface"] == "channel"
    assert readback["presence_boost_eq_band_readback_required"] is True
    assert readback["presence_boost_no_boost_without_operator_wing_proof"] is True
    assert readback["allowed_live_write_scope_before_wing_proof"] == []
    assert "/ch/{channel}/eq/on" in readback["wing_osc_path_templates"]
    assert "/ch/{channel}/eq/{band}f" in readback["wing_osc_path_templates"]
    assert "/ch/{channel}/eq/on" in readback["wing_eq_osc_path_templates"]
    assert "/ch/{channel}/eq/{band}f" in readback["wing_eq_osc_path_templates"]
    boost = [
        candidate
        for candidate in recommendation.alternatives
        if candidate.strategy_id == "soundcheck_presence_air_lift"
    ][0]
    assert "live_eq_boost_requires_operator_review_and_wing_proof" in boost.blockers
    assert boost.parameters["frequency_hz"] == 3180.0
    assert boost.parameters["frequency_source"] == "measured_presence_air_deficit"


def test_soundcheck_presence_boost_holds_when_true_peak_headroom_is_low():
    recommendation = analyze_smart_dsp_strategy(
        _analysis(
            problem_hints=("dull_presence",),
            muddiness_score=0.0,
            harshness_score=0.0,
            sibilance_score=0.0,
            presence_deficit_score=0.82,
            true_peak_headroom_db=2.0,
            feedback_risk=0.12,
        )
    )

    assert recommendation.status == "proposal_only"
    assert recommendation.selected_strategy.strategy_id == "hold_presence_boost_for_live_safety"
    boost = [
        candidate
        for candidate in recommendation.alternatives
        if candidate.strategy_id == "soundcheck_presence_air_lift"
    ][0]
    assert "insufficient_true_peak_headroom_for_boost" in boost.blockers


def test_soundcheck_presence_boost_holds_when_summed_bus_headroom_is_low():
    recommendation = analyze_smart_dsp_strategy(
        _analysis(
            problem_hints=("dull_presence",),
            muddiness_score=0.0,
            harshness_score=0.0,
            sibilance_score=0.0,
            presence_deficit_score=0.82,
            true_peak_headroom_db=6.0,
            feedback_risk=0.12,
            summed_bus_evidence_available=True,
            summed_bus_true_peak_headroom_db=1.2,
            summed_bus_level_plane_error_lu=0.0,
        )
    )

    assert recommendation.status == "proposal_only"
    assert recommendation.selected_strategy.strategy_id == (
        "hold_presence_boost_for_live_safety"
    )
    assert recommendation.selected_strategy.method_family == "analysis_only"
    assert recommendation.evidence["operator_explanation_sections"][
        "selection_policy"
    ] == "soundcheck_summed_bus_headroom_presence_boost_hold_priority"
    params = recommendation.selected_strategy.parameters
    assert params["summed_bus_headroom_review_required"] is True
    assert params["summed_bus_true_peak_headroom_db"] == 1.2
    assert params["requires_summed_bus_true_peak_headroom_db"] == 3.3
    boost = [
        candidate
        for candidate in recommendation.alternatives
        if candidate.strategy_id == "soundcheck_presence_air_lift"
    ][0]
    assert "summed_bus_headroom_blocks_presence_boost" in boost.blockers
    assert "summed_bus_headroom_blocks_gain_up" in boost.blockers
    assert boost.parameters["summed_bus_true_peak_headroom_db"] == 1.2
    assert boost.parameters["requires_summed_bus_true_peak_headroom_db"] == 3.3
    tradeoff = [
        row
        for row in recommendation.evidence["strategy_search"][
            "alternative_tradeoffs"
        ]
        if row["strategy_id"] == "soundcheck_presence_air_lift"
    ][0]
    assert tradeoff["why_not_selected"] == (
        "summed_bus_presence_boost_priority:"
        "restore_bus_headroom_before_air_lift"
    )
    assert recommendation.to_operator_proposal()["live_apply_allowed"] is False


def test_soundcheck_presence_boost_holds_when_monitor_feedback_risk_is_high():
    recommendation = analyze_smart_dsp_strategy(
        _analysis(
            problem_hints=("dull_presence",),
            muddiness_score=0.0,
            harshness_score=0.0,
            sibilance_score=0.0,
            presence_deficit_score=0.82,
            presence_lift_frequency_hz=3150.0,
            presence_lift_bandwidth_hz=3900.0,
            presence_lift_parameter_source="measured_presence_air_deficit",
            true_peak_headroom_db=6.0,
            feedback_risk=0.12,
            monitor_feedback_risk=0.72,
        )
    )

    assert recommendation.status == "proposal_only"
    assert recommendation.selected_strategy.strategy_id == (
        "hold_presence_boost_for_live_safety"
    )
    assert recommendation.selected_strategy.method_family == "analysis_only"
    assert recommendation.evidence["operator_explanation_sections"][
        "selection_policy"
    ] == "soundcheck_monitor_feedback_presence_boost_hold_priority"
    params = recommendation.selected_strategy.parameters
    assert params["monitor_path_review_required"] is True
    assert params["monitor_feedback_risk"] == 0.72
    boost = [
        candidate
        for candidate in recommendation.alternatives
        if candidate.strategy_id == "soundcheck_presence_air_lift"
    ][0]
    assert "monitor_feedback_risk_blocks_presence_boost" in boost.blockers
    assert boost.parameters["monitor_feedback_risk"] == 0.72
    tradeoff = [
        row
        for row in recommendation.evidence["strategy_search"][
            "alternative_tradeoffs"
        ]
        if row["strategy_id"] == "soundcheck_presence_air_lift"
    ][0]
    assert tradeoff["why_not_selected"] == (
        "monitor_feedback_presence_boost_priority:"
        "check_monitor_path_before_air_lift"
    )


def test_soundcheck_presence_boost_holds_when_feedback_path_risk_is_elevated():
    recommendation = analyze_smart_dsp_strategy(
        _analysis(
            problem_hints=("dull_presence",),
            muddiness_score=0.0,
            harshness_score=0.0,
            sibilance_score=0.0,
            presence_deficit_score=0.82,
            presence_lift_frequency_hz=3150.0,
            presence_lift_bandwidth_hz=3900.0,
            presence_lift_parameter_source="measured_presence_air_deficit",
            true_peak_headroom_db=6.0,
            feedback_risk=0.12,
            monitor_feedback_risk=0.5,
        )
    )

    assert recommendation.status == "proposal_only"
    assert recommendation.selected_strategy.strategy_id == (
        "hold_presence_boost_for_live_safety"
    )
    assert recommendation.selected_strategy.method_family == "analysis_only"
    assert recommendation.evidence["operator_explanation_sections"][
        "selection_policy"
    ] == "soundcheck_presence_boost_feedback_path_hold_priority"
    params = recommendation.selected_strategy.parameters
    assert params["feedback_path_review_required"] is True
    assert params["monitor_path_review_required"] is False
    assert params["monitor_feedback_risk"] == 0.5
    assert params["monitor_feedback_risk_review_threshold"] == 0.45
    boost = [
        candidate
        for candidate in recommendation.alternatives
        if candidate.strategy_id == "soundcheck_presence_air_lift"
    ][0]
    assert "feedback_path_review_blocks_presence_boost" in boost.blockers
    assert "monitor_feedback_risk_blocks_presence_boost" not in boost.blockers
    tradeoff = [
        row
        for row in recommendation.evidence["strategy_search"][
            "alternative_tradeoffs"
        ]
        if row["strategy_id"] == "soundcheck_presence_air_lift"
    ][0]
    assert tradeoff["why_not_selected"] == (
        "presence_boost_feedback_path_priority:"
        "check_feedback_path_before_air_lift"
    )
    assert recommendation.to_operator_proposal()["live_apply_allowed"] is False


def test_live_feedback_risk_selects_bounded_gain_pullback_with_notch_alternative():
    recommendation = analyze_smart_dsp_strategy(
        _analysis(
            mode="live",
            problem_hints=("feedback_risk",),
            muddiness_score=0.0,
            feedback_risk=0.82,
            active_performance=True,
        )
    )

    assert recommendation.status == "proposal_only"
    assert recommendation.selected_strategy.strategy_id == "feedback_gain_pullback"
    assert recommendation.selected_strategy.method_family == "gain_fader"
    params = recommendation.selected_strategy.parameters
    assert params["delta_db"] < 0.0
    assert params["bounded_delta_db"] <= 1.5
    assert params["feedback_risk"] == 0.82
    notch = [
        candidate
        for candidate in recommendation.alternatives
        if candidate.strategy_id == "narrow_feedback_notch_review"
    ][0]
    assert notch.parameters["filter_type"] == "peaking"
    assert notch.parameters["gain_db"] < 0.0
    assert "feedback_frequency_requires_operator_confirmation" in notch.blockers
    assert recommendation.to_operator_proposal()["live_apply_allowed"] is False


def test_live_level_and_fx_moves_include_stepwise_readback_guard():
    feedback_pullback = analyze_smart_dsp_strategy(
        _analysis(
            mode="live",
            problem_hints=("feedback_risk",),
            muddiness_score=0.0,
            feedback_risk=0.82,
            active_performance=True,
        )
    )
    loud_bass = analyze_smart_dsp_strategy(
        _analysis(
            target_id="ch:14",
            source_role="bass",
            mode="live",
            problem_hints=("level_plane_balance",),
            muddiness_score=0.0,
            estimated_lufs=-19.5,
            level_plane_error_lu=5.5,
            true_peak_headroom_db=3.5,
            active_performance=True,
        )
    )
    fx_pullback = analyze_smart_dsp_strategy(
        _analysis(
            target_id="fx:1",
            source_role="fx_return",
            mode="live",
            problem_hints=("fx_intelligibility",),
            muddiness_score=0.0,
            fx_wetness_score=0.5,
            fx_decay_risk_score=0.2,
            fx_delay_smear_score=0.95,
            fx_return_headroom_db=5.0,
            monitor_feedback_risk=0.1,
            active_performance=True,
        )
    )

    for recommendation in (feedback_pullback, loud_bass, fx_pullback):
        params = recommendation.selected_strategy.parameters
        assert recommendation.status == "proposal_only"
        assert params["live_no_sudden_jump_guard"] is True
        assert params["live_stepped_change_required"] is True
        assert params["live_readback_after_each_step_required"] is True
        assert params["live_max_single_step_db"] == 0.75
        assert params["live_min_step_interval_ms"] == 750
        assert params["live_rate_limit_reason"] == "avoid_abrupt_live_mix_jump"
        assert recommendation.to_operator_proposal()["live_apply_allowed"] is False
    fx_params = fx_pullback.selected_strategy.parameters
    assert fx_params["live_fx_pullback_only_proposal"] is True
    assert fx_params["live_fx_level_reduction_only_required"] is True
    assert fx_params["live_fx_max_pullback_db"] == LIVE_FX_MAX_PULLBACK_DB
    assert fx_params["live_fx_max_boost_db"] == LIVE_FX_MAX_BOOST_DB
    assert fx_params["live_fx_native_level_osc_only_required"] is True
    assert fx_params["live_fx_no_model_or_time_parameter_write_required"] is True
    assert fx_params["live_fx_absolute_level_readback_required"] is True
    assert fx_params["live_fx_monitor_feedback_recheck_after_apply_required"] is True
    assert fx_params["live_fx_return_headroom_recheck_after_apply_required"] is True
    assert fx_params["live_fx_operator_rehearsal_required"] is True
    assert fx_params["live_fx_live_promotion_status"] == (
        "blocked_until_real_wing_fx_level_readback_monitor_recheck_and_rollback"
    )
    assert fx_params["live_fx_live_apply_blocked_without_wing_proof"] is True


def test_soundcheck_measured_feedback_ring_prefers_narrow_notch_over_gain_pullback():
    recommendation = analyze_smart_dsp_strategy(
        _analysis(
            target_id="ch:5",
            source_role="lead_vocal",
            problem_hints=("feedback_risk",),
            muddiness_score=0.0,
            feedback_risk=0.84,
            feedback_frequency_hz=3150.0,
            feedback_frequency_confidence=0.88,
            feedback_frequency_source="measured_feedback_ring_peak",
            true_peak_headroom_db=6.0,
        )
    )

    assert recommendation.status == "proposal_only"
    assert recommendation.selected_strategy.strategy_id == "narrow_feedback_notch_review"
    assert recommendation.selected_strategy.method_family == "eq"
    params = recommendation.selected_strategy.parameters
    assert params["frequency_hz"] == 3150.0
    assert params["frequency_source"] == "measured_feedback_ring_peak"
    assert params["feedback_frequency_confidence"] == 0.88
    assert "feedback_frequency_requires_operator_confirmation" not in (
        recommendation.selected_strategy.blockers
    )
    gain_pullback = [
        candidate
        for candidate in recommendation.alternatives
        if candidate.strategy_id == "feedback_gain_pullback"
    ][0]
    assert gain_pullback.method_family == "gain_fader"
    assert recommendation.evidence["feedback_frequency_hz"] == 3150.0
    assert recommendation.evidence["feedback_frequency_source"] == "measured_feedback_ring_peak"
    assert recommendation.to_operator_proposal()["can_apply_live"] is False


def test_soundcheck_measured_feedback_ring_low_headroom_pullback_beats_notch():
    recommendation = analyze_smart_dsp_strategy(
        _analysis(
            target_id="ch:5",
            source_role="lead_vocal",
            problem_hints=("feedback_risk",),
            muddiness_score=0.0,
            feedback_risk=0.84,
            feedback_frequency_hz=3150.0,
            feedback_frequency_confidence=0.88,
            feedback_frequency_source="measured_feedback_ring_peak",
            true_peak_headroom_db=0.6,
        )
    )

    assert recommendation.status == "analysis_only"
    assert "true_peak_headroom_low" in recommendation.blockers
    assert recommendation.selected_strategy.strategy_id == "feedback_gain_pullback"
    assert recommendation.selected_strategy.method_family == "gain_fader"
    params = recommendation.selected_strategy.parameters
    assert params["target_scope"] == "source_feedback_and_true_peak_headroom_reduction"
    assert params["true_peak_headroom_db"] == 0.6
    assert params["target_true_peak_headroom_db"] == 1.75
    assert params["expected_post_pullback_true_peak_headroom_db"] == 2.7
    assert params["feedback_pullback_also_restores_headroom"] is True
    notch = [
        candidate
        for candidate in recommendation.alternatives
        if candidate.strategy_id == "narrow_feedback_notch_review"
    ][0]
    assert notch.parameters["frequency_hz"] == 3150.0
    assert notch.parameters["frequency_source"] == "measured_feedback_ring_peak"
    assert "feedback_frequency_requires_operator_confirmation" not in notch.blockers
    tradeoff = [
        item
        for item in recommendation.evidence["strategy_search"][
            "alternative_tradeoffs"
        ]
        if item["strategy_id"] == "narrow_feedback_notch_review"
    ][0]
    assert tradeoff["why_not_selected"] == (
        "feedback_headroom_priority:"
        "level_pullback_restores_headroom_before_measured_notch"
    )
    _assert_proposal_denial(recommendation.to_operator_proposal())


def test_feedback_risk_blocks_presence_boost_and_prefers_pullback():
    recommendation = analyze_smart_dsp_strategy(
        _analysis(
            problem_hints=("dull_presence",),
            muddiness_score=0.0,
            harshness_score=0.0,
            sibilance_score=0.0,
            presence_deficit_score=0.82,
            true_peak_headroom_db=6.0,
            feedback_risk=0.82,
        )
    )

    assert recommendation.status == "proposal_only"
    assert recommendation.selected_strategy.strategy_id == "feedback_gain_pullback"
    boost = [
        candidate
        for candidate in recommendation.alternatives
        if candidate.strategy_id == "soundcheck_presence_air_lift"
    ][0]
    assert "feedback_risk_blocks_boost" in boost.blockers


def test_soundcheck_vocal_dynamics_feedback_path_holds_before_makeup_compression():
    for overrides in (
        {"monitor_feedback_risk": 0.7, "feedback_risk": 0.1},
        {"monitor_feedback_risk": 0.1, "feedback_risk": 0.5},
    ):
        recommendation = analyze_smart_dsp_strategy(
            _analysis(
                problem_hints=("dynamic_control",),
                muddiness_score=0.0,
                estimated_lufs=-24.0,
                lra_proxy_db=14.0,
                crest_factor_db=19.0,
                dynamics_variance_score=0.72,
                dynamics_peak_to_median_db=13.0,
                dynamics_rise_time_ms=30.0,
                dynamics_recovery_ms=190.0,
                true_peak_headroom_db=4.0,
                level_plane_error_lu=0.2,
                **overrides,
            )
        )

        assert recommendation.status == "proposal_only"
        assert (
            recommendation.selected_strategy.strategy_id
            == "hold_dynamics_for_feedback_path_review"
        )
        assert recommendation.selected_strategy.method_family == "analysis_only"
        params = recommendation.selected_strategy.parameters
        assert params["dsp_change"] == "hold"
        assert params["reason"] == "feedback_path_review_before_compression_or_makeup"
        assert params["feedback_path_review_threshold"] == 0.45
        assert params["candidate_makeup_gain_db"] == 1.0
        assert params["required_context"] == (
            "pa_monitor_feedback_path_source_bus_headroom_previous_"
            "compressor_readback"
        )
        assert params["forbidden_without_wing_proof"] == (
            "compressor_makeup_gain_until_feedback_path_review"
        )
        assert recommendation.evidence["operator_explanation_sections"][
            "selection_policy"
        ] == "soundcheck_dynamics_feedback_path_before_compression_priority"
        alternatives = {
            candidate.strategy_id: candidate for candidate in recommendation.alternatives
        }
        assert "feedback_path_review_should_precede_vocal_compression" in (
            alternatives["vocal_phrase_leveling_compression"].blockers
        )
        assert "feedback_path_review_should_precede_compression_makeup" in (
            alternatives["gentle_channel_compression"].blockers
        )
        tradeoffs = {
            row["strategy_id"]: row
            for row in recommendation.evidence["strategy_search"]["alternative_tradeoffs"]
        }
        assert tradeoffs["vocal_phrase_leveling_compression"]["why_not_selected"] == (
            "soundcheck_dynamics_feedback_path_priority:"
            "feedback_path_review_precedes_compression_or_makeup"
        )
        assert recommendation.to_operator_proposal()["live_apply_allowed"] is False


def test_soundcheck_high_feedback_beats_dynamic_compression_without_feedback_hint():
    recommendation = analyze_smart_dsp_strategy(
        _analysis(
            problem_hints=("dynamic_control",),
            muddiness_score=0.0,
            estimated_lufs=-24.0,
            lra_proxy_db=14.0,
            crest_factor_db=19.0,
            dynamics_variance_score=0.72,
            dynamics_peak_to_median_db=13.0,
            dynamics_rise_time_ms=30.0,
            dynamics_recovery_ms=190.0,
            true_peak_headroom_db=4.0,
            level_plane_error_lu=0.2,
            feedback_risk=0.82,
            monitor_feedback_risk=0.1,
        )
    )

    assert recommendation.status == "proposal_only"
    assert recommendation.selected_strategy.strategy_id == "feedback_gain_pullback"
    assert recommendation.selected_strategy.method_family == "gain_fader"
    assert recommendation.selected_strategy.problem == "feedback_risk"
    assert recommendation.evidence["operator_explanation_sections"][
        "selection_policy"
    ] == "feedback_risk_priority"
    alternatives = {
        candidate.strategy_id: candidate for candidate in recommendation.alternatives
    }
    assert "vocal_phrase_leveling_compression" in alternatives
    tradeoffs = {
        row["strategy_id"]: row
        for row in recommendation.evidence["strategy_search"]["alternative_tradeoffs"]
    }
    assert tradeoffs["vocal_phrase_leveling_compression"]["why_not_selected"] == (
        "feedback_risk_priority:feedback_reduction_precedes_musical_correction"
    )
    assert recommendation.to_operator_proposal()["can_apply_live"] is False


def test_live_sibilant_vocal_prefers_wing_native_ds902_de_esser():
    recommendation = analyze_smart_dsp_strategy(
        _analysis(
            mode="live",
            problem_hints=("sibilant", "harsh"),
            muddiness_score=0.0,
            harshness_score=0.72,
            sibilance_score=0.91,
            sibilance_frequency_hz=7800.0,
            sibilance_bandwidth_hz=520.0,
            sibilance_event_peak_dbfs=-14.8,
            sibilance_event_contrast_db=11.5,
            sibilance_event_rise_time_ms=10.0,
            sibilance_event_recovery_ms=85.0,
            active_performance=True,
        )
    )

    assert recommendation.status == "proposal_only"
    assert recommendation.selected_strategy.strategy_id == "vocal_de_esser"
    assert recommendation.selected_strategy.method_family == "de_esser"
    params = recommendation.selected_strategy.parameters
    assert 5500.0 <= params["frequency_hz"] <= 10000.0
    assert params["frequency_source"] == "measured_sibilance_peak"
    assert params["bandwidth_source"] == "measured_sibilance_bandwidth"
    assert params["band_start_hz"] < params["frequency_hz"] < params["band_end_hz"]
    assert params["bandwidth_hz"] == 900.0
    assert params["q"] > 4.0
    assert params["range_db"] == 5.4
    assert params["max_gain_reduction_db"] == 5.4
    assert params["threshold_dbfs"] == -23.1
    assert params["attack_ms"] == 4.5
    assert params["release_ms"] == 120.0
    assert "wing_de_esser_model_osc_value" not in params
    assert "wing_de_esser_on_osc_value" not in params
    assert "wing_de_esser_frequency_osc_value" not in params
    assert "wing_de_esser_range_osc_value" not in params
    assert "wing_de_esser_mode_osc_value" not in params
    assert params["live_de_esser_parameter_write_forbidden"] is True
    assert params["live_de_esser_parameter_readback_required"] is True
    assert params["live_de_esser_parameter_apply_blocked_without_wing_proof"] is True
    assert params["live_de_esser_parameter_review_only_values"] == {
        "wing_de_esser_on_osc_value": 1,
        "wing_de_esser_frequency_osc_value": 7800.0,
        "wing_de_esser_range_osc_value": 5.4,
        "wing_de_esser_mode_osc_value": "HF",
    }
    assert params["live_de_esser_parameter_review_only_reason"] == (
        "live_de_esser_must_read_existing_ds902_state_and_operator_confirm_before_parameter_apply"
    )
    assert params["wing_de_esser_gate_slot_conflict_policy"] == (
        WING_DE_ESSER_GATE_SLOT_CONFLICT_POLICY
    )
    assert params["measured_sibilance_bandwidth_hz"] == 520.0
    assert params["de_esser_parameter_source"] == "measured_sibilance_event"
    assert params["sibilance_event_peak_dbfs"] == -14.8
    assert params["sibilance_event_contrast_db"] == 11.5
    assert params["makeup_gain_db"] == 0.0
    assert params["live_de_esser_conservative_proposal"] is True
    assert params["live_de_esser_max_range_db"] == LIVE_DE_ESSER_MAX_RANGE_DB
    assert params["live_de_esser_stepwise_range_review_required"] is True
    assert (
        params["live_de_esser_max_range_step_db"]
        == LIVE_DE_ESSER_MAX_RANGE_STEP_DB
    )
    assert params["live_de_esser_sibilance_recheck_after_apply_required"] is True
    assert (
        params["live_de_esser_gate_slot_readback_before_model_change_required"]
        is True
    )
    assert params["live_de_esser_gate_slot_model_write_forbidden"] is True
    assert params["live_de_esser_requires_existing_ds902_model_readback"] is True
    assert (
        params["live_de_esser_model_review_only_value"]
        == WING_DE_ESSER_GATE_MODEL
    )
    assert params["live_de_esser_model_write_review_only_reason"] == (
        "live_de_esser_must_read_existing_ds902_gate_slot_model_before_parameter_apply"
    )
    assert params["live_de_esser_on_mode_write_forbidden"] is True
    assert params["live_de_esser_on_mode_readback_required"] is True
    assert params["live_de_esser_on_write_forbidden"] is True
    assert params["live_de_esser_mode_write_forbidden"] is True
    assert params["live_de_esser_requires_existing_enabled_readback"] is True
    assert params["live_de_esser_requires_existing_mode_readback"] is True
    assert params["live_de_esser_on_review_only_value"] == 1
    assert params["live_de_esser_mode_review_only_value"] == "HF"
    assert params["live_de_esser_on_mode_review_only_values"] == {
        "wing_de_esser_on_osc_value": 1,
        "wing_de_esser_mode_osc_value": "HF",
    }
    assert params["live_de_esser_on_mode_review_only_reason"] == (
        "live_de_esser_must_read_existing_gate_on_and_mode_before_parameter_apply"
    )
    assert params["live_de_esser_no_fx_rack_insert_required"] is True
    assert params["live_de_esser_operator_rehearsal_required"] is True
    assert params["live_de_esser_live_promotion_status"] == (
        "blocked_until_real_wing_ds902_readback_sibilance_recheck_and_rollback"
    )
    assert params["live_de_esser_live_apply_blocked_without_wing_proof"] is True
    assert recommendation.evidence["sibilance_bandwidth_hz"] == 520.0
    assert recommendation.evidence["sibilance_event_peak_dbfs"] == -14.8
    assert recommendation.evidence["sibilance_event_contrast_db"] == 11.5
    assert recommendation.evidence["operator_explanation_sections"][
        "selection_policy"
    ] == "weighted_strategy_score"
    readback = recommendation.selected_strategy.future_wing_readback_contract
    _assert_common_readback_contract(readback)
    assert "eq_band_readback" not in readback
    assert "dynamics_readback" not in readback
    assert readback["de_esser_readback"]["gate_model_readback_must_equal"] == (
        WING_DE_ESSER_GATE_MODEL
    )
    live_de_esser_write_keys = []
    _assert_selected_write_key_contract(
        params=params,
        readback=readback["de_esser_readback"],
        prefix="wing_de_esser",
        selected_write_keys=live_de_esser_write_keys,
        write_parameter_keys=WING_DE_ESSER_WRITE_PARAMETER_KEYS,
        selection_policy=WING_DE_ESSER_WRITE_KEY_SELECTION_POLICY,
    )
    assert readback["de_esser_readback"]["live_de_esser_conservative_proposal"] is True
    assert (
        readback["de_esser_readback"]["live_de_esser_max_range_db"]
        == LIVE_DE_ESSER_MAX_RANGE_DB
    )
    assert (
        readback["de_esser_readback"][
            "live_de_esser_stepwise_range_review_required"
        ]
        is True
    )
    assert (
        readback["de_esser_readback"]["live_de_esser_max_range_step_db"]
        == LIVE_DE_ESSER_MAX_RANGE_STEP_DB
    )
    assert (
        readback["de_esser_readback"][
            "live_de_esser_sibilance_recheck_after_apply_required"
        ]
        is True
    )
    assert (
        readback["de_esser_readback"][
            "live_de_esser_gate_slot_readback_before_model_change_required"
        ]
        is True
    )
    assert (
        readback["de_esser_readback"]["live_de_esser_gate_slot_model_write_forbidden"]
        is True
    )
    assert (
        readback["de_esser_readback"][
            "live_de_esser_requires_existing_ds902_model_readback"
        ]
        is True
    )
    assert readback["de_esser_readback"]["live_de_esser_model_review_only_value"] == (
        WING_DE_ESSER_GATE_MODEL
    )
    assert readback["de_esser_readback"]["live_de_esser_model_write_review_only_reason"] == (
        "live_de_esser_must_read_existing_ds902_gate_slot_model_before_parameter_apply"
    )
    assert (
        readback["de_esser_readback"]["live_de_esser_on_mode_write_forbidden"]
        is True
    )
    assert (
        readback["de_esser_readback"]["live_de_esser_on_mode_readback_required"]
        is True
    )
    assert readback["de_esser_readback"]["live_de_esser_on_write_forbidden"] is True
    assert readback["de_esser_readback"]["live_de_esser_mode_write_forbidden"] is True
    assert (
        readback["de_esser_readback"][
            "live_de_esser_requires_existing_enabled_readback"
        ]
        is True
    )
    assert (
        readback["de_esser_readback"]["live_de_esser_requires_existing_mode_readback"]
        is True
    )
    assert readback["de_esser_readback"]["live_de_esser_on_review_only_value"] == 1
    assert readback["de_esser_readback"]["live_de_esser_mode_review_only_value"] == (
        "HF"
    )
    assert readback["de_esser_readback"]["live_de_esser_on_mode_review_only_values"] == {
        "wing_de_esser_on_osc_value": 1,
        "wing_de_esser_mode_osc_value": "HF",
    }
    assert (
        readback["de_esser_readback"]["live_de_esser_on_mode_review_only_reason"]
        == "live_de_esser_must_read_existing_gate_on_and_mode_before_parameter_apply"
    )
    assert (
        readback["de_esser_readback"]["live_de_esser_no_fx_rack_insert_required"]
        is True
    )
    assert (
        readback["de_esser_readback"]["live_de_esser_operator_rehearsal_required"]
        is True
    )
    assert readback["de_esser_readback"]["live_de_esser_live_promotion_status"] == (
        "blocked_until_real_wing_ds902_readback_sibilance_recheck_and_rollback"
    )
    assert (
        readback["de_esser_readback"][
            "live_de_esser_live_apply_blocked_without_wing_proof"
        ]
        is True
    )
    assert readback["de_esser_readback"]["read_before_paths"] == [
        "gate/mdl",
        "gate/on",
        "gate/f",
        "gate/range",
        "gate/mode",
    ]
    assert (
        readback["de_esser_readback"]["live_de_esser_parameter_review_only_values"]
        == params["live_de_esser_parameter_review_only_values"]
    )
    assert readback["wing_osc_applicability"]["native_osc_supported"] is True
    assert readback["wing_osc_applicability"]["wing_mvp_live_write_scope_approved"] is False
    _assert_selected_de_esser_apply_values(
        recommendation,
        {},
    )
    alternatives = {candidate.strategy_id: candidate for candidate in recommendation.alternatives}
    assert alternatives["static_sibilance_band_cut"].method_family == "eq"
    assert recommendation.to_operator_proposal()["can_apply_live"] is False


def test_soundcheck_vocal_level_pressure_precedes_moderate_de_essing():
    recommendation = analyze_smart_dsp_strategy(
        _analysis(
            problem_hints=("sibilant", "harsh"),
            operator_intent="soundcheck lead vocal is sharp and too loud in the mix",
            muddiness_score=0.0,
            harshness_score=0.58,
            presence_harshness_score=0.55,
            sibilance_score=0.52,
            sibilance_frequency_hz=7400.0,
            sibilance_bandwidth_hz=520.0,
            sibilance_event_peak_dbfs=-18.0,
            sibilance_event_contrast_db=5.5,
            sibilance_event_rise_time_ms=12.0,
            sibilance_event_recovery_ms=90.0,
            level_plane_error_lu=3.4,
            estimated_lufs=-18.8,
            true_peak_headroom_db=4.5,
            feedback_risk=0.1,
        )
    )

    assert recommendation.status == "proposal_only"
    assert recommendation.selected_strategy.strategy_id == "level_plane_gain_down"
    assert recommendation.selected_strategy.method_family == "gain_fader"
    assert recommendation.evidence["operator_explanation_sections"][
        "selection_policy"
    ] == "soundcheck_vocal_level_pressure_before_tone_priority"
    params = recommendation.selected_strategy.parameters
    assert params["delta_db"] < 0.0
    assert params["current_minus_target_lu"] == 3.4
    assert params["target_scope"] == "source_and_summed_bus"
    alternatives = {
        candidate.strategy_id: candidate for candidate in recommendation.alternatives
    }
    assert alternatives["vocal_de_esser"].parameters["de_esser_parameter_source"] == (
        "measured_sibilance_event"
    )
    tradeoffs = {
        row["strategy_id"]: row
        for row in recommendation.evidence["strategy_search"]["alternative_tradeoffs"]
    }
    assert tradeoffs["vocal_de_esser"]["why_not_selected"] == (
        "soundcheck_vocal_level_pressure_priority:"
        "restore_level_before_tone_processing"
    )
    assert tradeoffs["presence_dynamic_eq_cut"]["why_not_selected"] == (
        "soundcheck_vocal_level_pressure_priority:"
        "restore_level_before_tone_processing"
    )
    assert recommendation.to_operator_proposal()["live_apply_allowed"] is False


def test_live_feedback_risk_beats_sibilance_correction():
    recommendation = analyze_smart_dsp_strategy(
        _analysis(
            mode="live",
            problem_hints=("feedback_risk", "sibilant", "harsh"),
            muddiness_score=0.0,
            feedback_risk=0.86,
            feedback_frequency_hz=3150.0,
            feedback_frequency_confidence=0.7,
            feedback_frequency_source="measured_feedback_ring_peak",
            harshness_score=0.72,
            sibilance_score=0.91,
            sibilance_frequency_hz=7800.0,
            active_performance=True,
        )
    )

    assert recommendation.status == "proposal_only"
    assert recommendation.selected_strategy.strategy_id == "feedback_gain_pullback"
    assert recommendation.selected_strategy.method_family == "gain_fader"
    assert recommendation.selected_strategy.problem == "feedback_risk"
    alternatives = {candidate.strategy_id for candidate in recommendation.alternatives}
    assert "vocal_de_esser" in alternatives
    assert "narrow_feedback_notch_review" in alternatives
    proposal = recommendation.to_operator_proposal()
    _assert_proposal_denial(proposal)
    assert proposal["operator_explanation_sections"]["problem_found"] == "feedback_risk"
    assert proposal["operator_explanation_sections"]["selection_policy"] == (
        "live_feedback_risk_priority"
    )
    assert proposal["strategy_search_details"]["selected_rationale"][
        "selection_policy"
    ] == "live_feedback_risk_priority"


def test_live_feedback_risk_beats_true_peak_headroom_pullback():
    recommendation = analyze_smart_dsp_strategy(
        _analysis(
            mode="live",
            problem_hints=("feedback_risk",),
            muddiness_score=0.0,
            feedback_risk=0.78,
            feedback_frequency_hz=2800.0,
            feedback_frequency_confidence=0.8,
            feedback_frequency_source="measured_feedback_ring_peak",
            true_peak_headroom_db=0.6,
            active_performance=True,
        )
    )

    assert recommendation.status == "analysis_only"
    assert "true_peak_headroom_low" in recommendation.blockers
    assert recommendation.selected_strategy.strategy_id == "feedback_gain_pullback"
    assert recommendation.selected_strategy.method_family == "gain_fader"
    assert recommendation.selected_strategy.problem == "feedback_risk"
    assert recommendation.evidence["operator_explanation_sections"][
        "selection_policy"
    ] == "live_feedback_risk_priority"
    params = recommendation.selected_strategy.parameters
    assert params["delta_db"] < 0.0
    assert params["feedback_risk"] == 0.78
    assert params["target_scope"] == "source_feedback_and_true_peak_headroom_reduction"
    assert params["true_peak_headroom_db"] == 0.6
    assert params["minimum_true_peak_headroom_db"] == 1.0
    assert params["target_true_peak_headroom_db"] == 1.5
    assert params["expected_post_pullback_true_peak_headroom_db"] == 2.1
    assert params["feedback_pullback_also_restores_headroom"] is True
    assert params["post_pullback_headroom_recheck_required"] is True
    assert params["live_stepped_change_required"] is True
    assert params["live_max_single_step_db"] == 0.75
    source_pullback = [
        candidate
        for candidate in recommendation.alternatives
        if candidate.strategy_id == "source_true_peak_headroom_pullback"
    ][0]
    assert source_pullback.parameters["headroom_restoration_first"] is True
    tradeoff = [
        item
        for item in recommendation.evidence["strategy_search"][
            "alternative_tradeoffs"
        ]
        if item["strategy_id"] == "source_true_peak_headroom_pullback"
    ][0]
    assert tradeoff["why_not_selected"] == (
        "feedback_risk_priority:"
        "feedback_gain_reduction_precedes_general_headroom_pullback"
    )
    _assert_proposal_denial(recommendation.to_operator_proposal())


def test_live_harsh_vocal_prefers_static_presence_cut_over_dynamic_eq():
    soundcheck = analyze_smart_dsp_strategy(
        _analysis(
            problem_hints=("harsh",),
            muddiness_score=0.0,
            harshness_score=0.82,
            sibilance_score=0.1,
            presence_harshness_score=0.76,
            dynamics_variance_score=0.2,
            feedback_frequency_hz=3350.0,
            feedback_frequency_confidence=0.82,
            feedback_frequency_source="narrow_peak_without_feedback_context",
            true_peak_headroom_db=5.0,
        )
    )
    recommendation = analyze_smart_dsp_strategy(
        _analysis(
            mode="live",
            operator_intent="reduce harsh vocal live",
            problem_hints=("harsh",),
            muddiness_score=0.0,
            harshness_score=0.82,
            sibilance_score=0.1,
            presence_harshness_score=0.76,
            dynamics_variance_score=0.2,
            feedback_frequency_hz=3350.0,
            feedback_frequency_confidence=0.82,
            feedback_frequency_source="narrow_peak_without_feedback_context",
            true_peak_headroom_db=5.0,
            active_performance=True,
        )
    )

    assert soundcheck.selected_strategy.strategy_id == "broad_presence_cut"
    assert recommendation.status == "proposal_only"
    assert recommendation.selected_strategy.strategy_id == "broad_presence_cut"
    assert recommendation.selected_strategy.method_family == "eq"
    params = recommendation.selected_strategy.parameters
    assert params["frequency_hz"] == 3350.0
    assert params["frequency_source"] == "measured_narrow_presence_peak"
    assert params["vocal_presence_parameter_source"] == "measured_narrow_presence_peak"
    assert params["gain_db"] == -2.8
    assert params["q"] == 1.54
    assert params["presence_harshness_score"] == 0.76
    assert params["feedback_frequency_hz"] == 3350.0
    assert recommendation.evidence["operator_explanation_sections"][
        "selection_policy"
    ] == "live_static_vocal_harshness_cut_priority"
    dynamic = [
        candidate
        for candidate in recommendation.alternatives
        if candidate.strategy_id == "presence_dynamic_eq_cut"
    ][0]
    assert "live_dynamic_presence_eq_requires_operator_review" in dynamic.blockers
    assert dynamic.parameters["frequency_hz"] == 3350.0
    assert dynamic.parameters["frequency_source"] == "measured_narrow_presence_peak"
    assert dynamic.parameters["q"] == 3.64
    assert dynamic.parameters["max_gain_reduction_db"] == 3.3
    dynamic_tradeoff = [
        row
        for row in recommendation.evidence["strategy_search"]["alternative_tradeoffs"]
        if row["strategy_id"] == "presence_dynamic_eq_cut"
    ][0]
    assert (
        dynamic_tradeoff["why_not_selected"]
        == "live_static_vocal_harshness_cut_priority:"
        "static_no_boost_cut_precedes_dynamic_eq"
    )
    assert recommendation.to_operator_proposal()["can_apply_live"] is False


def test_live_operator_owned_eq_holds_instead_of_weak_de_esser_fallback():
    recommendation = analyze_smart_dsp_strategy(
        _analysis(
            mode="live",
            operator_intent="reduce harsh vocal live",
            problem_hints=("harsh",),
            muddiness_score=0.0,
            harshness_score=0.88,
            sibilance_score=0.2,
            presence_harshness_score=0.82,
            dynamics_variance_score=0.2,
            current_wing_eq_model=WING_EQ_REQUIRED_MODEL,
            current_wing_eq_enabled=True,
            current_wing_eq_model_readback_available=True,
            current_wing_eq_operator_owned=True,
            true_peak_headroom_db=7.0,
            active_performance=True,
        )
    )

    assert recommendation.status == "proposal_only"
    assert recommendation.selected_strategy.strategy_id == (
        "hold_live_operator_owned_eq_for_review"
    )
    assert recommendation.selected_strategy.method_family == "analysis_only"
    assert recommendation.evidence["operator_explanation_sections"][
        "selection_policy"
    ] == "live_operator_owned_eq_review_priority"
    params = recommendation.selected_strategy.parameters
    assert params["dsp_change"] == "hold"
    assert params["reason"] == "live_operator_owned_eq_requires_operator_review"
    assert "broad_presence_cut" in params["blocked_eq_strategy_ids"]
    assert params["blocked_eq_problem"] == "harsh"
    assert params["blocked_eq_filter_type"] == "peaking"
    assert params["blocked_eq_gain_db"] == -2.8
    assert params["wing_eq_current_operator_owned"] is True
    assert params["wing_eq_operator_owned_model_edit_requires_review"] is True
    assert params["wing_eq_operator_owned_model_blocker"] == (
        WING_EQ_OPERATOR_OWNED_MODEL_BLOCKER
    )
    assert params["wing_rack_fw305_verified_osc_mapping_required"] is True
    assert params["wing_rack_fw305_eq_band_readback_required"] is True
    assert params["wing_rack_fw305_rollback_required"] is True
    assert params["live_authority_granted"] is False
    assert params["live_write"] is False
    assert params["readiness_closure_allowed"] is False
    alternatives = {
        candidate.strategy_id: candidate for candidate in recommendation.alternatives
    }
    assert alternatives["broad_presence_cut"].blockers == (
        WING_EQ_OPERATOR_OWNED_MODEL_BLOCKER,
    )
    assert "sibilance_evidence_low_for_de_esser" in alternatives[
        "vocal_de_esser"
    ].blockers
    tradeoffs = {
        row["strategy_id"]: row["why_not_selected"]
        for row in recommendation.evidence["strategy_search"][
            "alternative_tradeoffs"
        ]
    }
    assert tradeoffs["broad_presence_cut"] == (
        "live_operator_owned_eq_review_priority:"
        "manual_eq_slot_review_precedes_live_eq_change"
    )
    assert recommendation.to_operator_proposal()["can_apply_live"] is False


def test_soundcheck_operator_owned_eq_holds_instead_of_weak_de_esser_fallback():
    recommendation = analyze_smart_dsp_strategy(
        _analysis(
            operator_intent="soundcheck harsh vocal cleanup",
            problem_hints=("harsh",),
            muddiness_score=0.0,
            harshness_score=0.84,
            sibilance_score=0.2,
            presence_harshness_score=0.82,
            dynamics_variance_score=0.2,
            current_wing_eq_model=WING_EQ_REQUIRED_MODEL,
            current_wing_eq_enabled=True,
            current_wing_eq_model_readback_available=True,
            current_wing_eq_operator_owned=True,
            true_peak_headroom_db=7.0,
        )
    )

    assert recommendation.status == "proposal_only"
    assert recommendation.selected_strategy.strategy_id == (
        "hold_soundcheck_operator_owned_eq_for_review"
    )
    assert recommendation.selected_strategy.method_family == "analysis_only"
    assert recommendation.evidence["operator_explanation_sections"][
        "selection_policy"
    ] == "soundcheck_operator_owned_eq_review_priority"
    params = recommendation.selected_strategy.parameters
    assert params["dsp_change"] == "hold"
    assert params["reason"] == "soundcheck_operator_owned_eq_requires_operator_review"
    assert "broad_presence_cut" in params["blocked_eq_strategy_ids"]
    assert params["blocked_eq_operator_owned_blockers"] == [
        WING_EQ_OPERATOR_OWNED_MODEL_BLOCKER
    ]
    assert params["blocked_eq_problem"] == "harsh"
    assert params["blocked_eq_filter_type"] == "peaking"
    assert params["blocked_eq_gain_db"] == -2.8
    assert params["soundcheck_operator_owned_eq_hold_proposal"] is True
    assert params["soundcheck_operator_owned_eq_target_surface"] == "channel"
    assert params["soundcheck_operator_owned_eq_model_readback_required"] is True
    assert params["soundcheck_operator_owned_eq_band_readback_required"] is True
    assert (
        params["soundcheck_operator_owned_eq_protected_band_review_required"]
        is True
    )
    assert params["soundcheck_operator_owned_eq_audio_recheck_required"] is True
    assert params["soundcheck_operator_owned_eq_rollback_required"] is True
    assert (
        params["soundcheck_operator_owned_eq_no_eq_change_without_operator_review"]
        is True
    )
    assert params["wing_eq_current_model"] == WING_EQ_REQUIRED_MODEL
    assert params["wing_eq_current_enabled"] is True
    assert params["wing_eq_current_operator_owned"] is True
    assert params["wing_eq_model_readback_available"] is True
    assert params["wing_eq_operator_owned_model_edit_requires_review"] is True
    assert params["wing_eq_operator_owned_model_blocker"] == (
        WING_EQ_OPERATOR_OWNED_MODEL_BLOCKER
    )
    assert params["wing_rack_fw305_verified_osc_mapping_required"] is True
    assert params["wing_rack_fw305_eq_band_readback_required"] is True
    assert params["wing_rack_fw305_rollback_required"] is True
    assert params["required_context"] == (
        "operator_review_current_wing_eq_model_band_owner_readback_rollback"
    )
    assert params["forbidden_until_reviewed"] == (
        "soundcheck_eq_boost_cut_notch_filter_queue_import"
    )
    readback = recommendation.selected_strategy.future_wing_readback_contract[
        "soundcheck_operator_owned_eq_review_readback"
    ]
    assert readback["read_only"] is True
    assert readback["soundcheck_operator_owned_eq_current_model"] == (
        WING_EQ_REQUIRED_MODEL
    )
    assert readback["soundcheck_operator_owned_eq_current_enabled"] is True
    assert readback["soundcheck_operator_owned_eq_current_operator_owned"] is True
    assert readback["soundcheck_operator_owned_eq_operator_owned_blockers"] == [
        WING_EQ_OPERATOR_OWNED_MODEL_BLOCKER
    ]
    assert "eq/mdl" in readback["read_before_paths"]
    assert "eq/on" in readback["read_before_paths"]
    assert "eq/{band}f" in readback["read_before_paths"]
    assert "/ch/{channel}/eq/{band}f" in readback["wing_osc_path_templates"]
    assert readback["allowed_live_write_scope_before_wing_proof"] == []
    alternatives = {
        candidate.strategy_id: candidate for candidate in recommendation.alternatives
    }
    assert alternatives["broad_presence_cut"].blockers == (
        WING_EQ_OPERATOR_OWNED_MODEL_BLOCKER,
    )
    assert "sibilance_evidence_low_for_de_esser" in alternatives[
        "vocal_de_esser"
    ].blockers
    tradeoffs = {
        row["strategy_id"]: row["why_not_selected"]
        for row in recommendation.evidence["strategy_search"][
            "alternative_tradeoffs"
        ]
    }
    assert tradeoffs["broad_presence_cut"] == (
        "soundcheck_operator_owned_eq_review_priority:"
        "manual_eq_slot_owner_review_precedes_soundcheck_eq_change"
    )
    assert tradeoffs["vocal_de_esser"] == (
        "soundcheck_operator_owned_eq_review_priority:"
        "manual_eq_owner_review_precedes_weak_fallback_tone_processor"
    )
    _assert_selected_has_no_apply_values(recommendation)


def test_soundcheck_harsh_sibilant_vocal_prefers_wing_native_ds902_de_esser():
    recommendation = analyze_smart_dsp_strategy(
        _analysis(
            target_id="ch:6",
            source_role="lead_vocal",
            problem_hints=("sibilant", "harsh"),
            muddiness_score=0.0,
            harshness_score=0.86,
            presence_harshness_score=0.82,
            sibilance_score=0.84,
            sibilance_frequency_hz=7600.0,
            sibilance_bandwidth_hz=1400.0,
            sibilance_event_peak_dbfs=-16.5,
            sibilance_event_contrast_db=10.2,
            sibilance_event_rise_time_ms=12.0,
            sibilance_event_recovery_ms=95.0,
            feedback_frequency_hz=3650.0,
            feedback_frequency_confidence=0.8,
            feedback_frequency_source="narrow_peak_without_feedback_context",
            true_peak_headroom_db=6.0,
            feedback_risk=0.12,
        )
    )

    assert recommendation.status == "proposal_only"
    assert recommendation.selected_strategy.strategy_id == (
        "vocal_de_esser"
    )
    assert recommendation.selected_strategy.method_family == "de_esser"
    params = recommendation.selected_strategy.parameters
    assert params["frequency_hz"] == 7600.0
    assert params["frequency_source"] == "measured_sibilance_peak"
    assert params["bandwidth_source"] == "measured_sibilance_bandwidth"
    assert params["band_start_hz"] < params["frequency_hz"]
    assert params["band_end_hz"] > params["frequency_hz"]
    assert params["bandwidth_hz"] == 1890.0
    assert params["q"] > 3.0
    assert params["threshold_dbfs"] == -24.1
    assert params["range_db"] == 5.2
    assert params["max_gain_reduction_db"] == 5.2
    assert params["attack_ms"] == 5.4
    assert params["release_ms"] == 130.0
    assert params["de_esser_parameter_source"] == "measured_sibilance_event"
    assert params["sibilance_event_contrast_db"] == 10.2
    assert params["wing_de_esser_model_osc_value"] == WING_DE_ESSER_GATE_MODEL
    assert params["wing_de_esser_frequency_osc_value"] == 7600.0
    assert params["wing_de_esser_range_osc_value"] == 5.2
    assert params["wing_de_esser_mode_osc_value"] == "HF"
    assert params["wing_de_esser_gate_slot_conflict_policy"] == (
        WING_DE_ESSER_GATE_SLOT_CONFLICT_POLICY
    )
    assert recommendation.evidence["operator_explanation_sections"][
        "selection_policy"
    ] == "weighted_strategy_score"
    alternatives = {candidate.strategy_id: candidate for candidate in recommendation.alternatives}
    assert "static_sibilance_band_cut" in alternatives
    combo = alternatives["vocal_de_esser_dynamic_presence_combo"]
    combo_params = combo.parameters
    assert combo_params["processing_chain"] == "de_esser_then_dynamic_presence_control"
    assert combo_params["de_esser_frequency_hz"] == 7600.0
    assert combo_params["de_esser_threshold_dbfs"] == -24.1
    assert combo_params["de_esser_attack_ms"] == 5.4
    assert combo_params["de_esser_release_ms"] == 130.0
    assert combo_params["dynamic_eq_frequency_hz"] == 3650.0
    assert combo_params["dynamic_eq_max_gain_reduction_db"] == 3.5
    assert combo_params["wing_gate_slot_mutually_exclusive_processor_chain"] is True
    assert combo_params["wing_gate_slot_required_models"] == [
        WING_DE_ESSER_GATE_MODEL,
        WING_DYNAMIC_EQ_GATE_MODEL,
    ]
    assert combo_params["wing_gate_slot_single_model_only"] is True
    assert combo_params["wing_gate_slot_combo_policy"] == (
        WING_GATE_SLOT_SINGLE_MODEL_COMBO_POLICY
    )
    assert combo_params["wing_gate_slot_combo_apply_ready"] is False
    assert combo_params["wing_gate_slot_combo_live_apply_blocked"] is True
    assert combo_params["wing_gate_slot_combo_review_required"] is True
    assert combo_params["wing_gate_slot_combo_requires_separate_passes"] is True
    assert combo_params["wing_gate_slot_combo_not_direct_osc_chain"] is True
    assert combo_params["wing_gate_slot_combo_blocker"] == (
        WING_GATE_SLOT_SINGLE_MODEL_COMBO_BLOCKER
    )
    assert combo_params["wing_gate_slot_combo_safe_fallback_strategy"] == (
        "apply_ds902_de_esser_first_then_recheck_before_deq"
    )
    assert WING_GATE_SLOT_SINGLE_MODEL_COMBO_BLOCKER in combo.blockers
    combo_readback = combo.future_wing_readback_contract["dynamic_eq_readback"]
    assert combo_readback["wing_gate_slot_mutually_exclusive_processor_chain"] is True
    assert combo_readback["wing_gate_slot_required_models"] == [
        WING_DE_ESSER_GATE_MODEL,
        WING_DYNAMIC_EQ_GATE_MODEL,
    ]
    assert combo_readback["wing_gate_slot_combo_apply_ready"] is False
    assert combo_readback["wing_gate_slot_combo_live_apply_blocked"] is True
    assert combo_readback["wing_gate_slot_combo_review_required"] is True
    assert combo_readback["wing_gate_slot_combo_requires_separate_passes"] is True
    assert combo_readback["wing_gate_slot_combo_not_direct_osc_chain"] is True
    assert combo_readback["wing_gate_slot_combo_blocker"] == (
        WING_GATE_SLOT_SINGLE_MODEL_COMBO_BLOCKER
    )
    assert "presence_dynamic_eq_cut" in alternatives
    tradeoffs = {
        row["strategy_id"]: row
        for row in recommendation.evidence["strategy_search"]["alternative_tradeoffs"]
    }
    assert tradeoffs["static_sibilance_band_cut"]["why_not_selected"].startswith(
        "lower_"
    )
    assert tradeoffs["vocal_de_esser_dynamic_presence_combo"][
        "why_not_selected"
    ].startswith("blocked_by:")
    assert "wing_dynamic_eq_value_mapping_required" in tradeoffs[
        "vocal_de_esser_dynamic_presence_combo"
    ]["blockers"]
    assert WING_GATE_SLOT_SINGLE_MODEL_COMBO_BLOCKER in tradeoffs[
        "vocal_de_esser_dynamic_presence_combo"
    ]["blockers"]
    readback = recommendation.selected_strategy.future_wing_readback_contract
    assert readback["wing_osc_applicability"]["native_osc_supported"] is True
    assert readback["wing_osc_applicability"]["native_osc_support_level"] == (
        "native_channel_gate_ds902_de_esser_osc_path_templates_declared"
    )
    assert readback["de_esser_readback"]["wing_de_esser_model_readback_must_equal"] == (
        WING_DE_ESSER_GATE_MODEL
    )
    _assert_selected_write_key_contract(
        params=params,
        readback=readback["de_esser_readback"],
        prefix="wing_de_esser",
        selected_write_keys=list(WING_DE_ESSER_WRITE_PARAMETER_KEYS),
        write_parameter_keys=WING_DE_ESSER_WRITE_PARAMETER_KEYS,
        selection_policy=WING_DE_ESSER_WRITE_KEY_SELECTION_POLICY,
    )
    assert "eq_band_readback" not in readback
    assert "dynamics_readback" not in readback
    _assert_selected_de_esser_apply_values(
        recommendation,
        {
            "wing_de_esser_model_osc_value": WING_DE_ESSER_GATE_MODEL,
            "wing_de_esser_on_osc_value": 1,
            "wing_de_esser_frequency_osc_value": 7600.0,
            "wing_de_esser_range_osc_value": 5.2,
            "wing_de_esser_mode_osc_value": "HF",
        },
    )
    assert recommendation.to_operator_proposal()["live_apply_allowed"] is False


def test_soundcheck_sibilant_dynamic_vocal_orders_de_esser_before_compression():
    recommendation = analyze_smart_dsp_strategy(
        _analysis(
            target_id="ch:7",
            source_role="lead_vocal",
            problem_hints=("sibilant", "harsh", "dynamic_control"),
            operator_intent="soundcheck lead vocal sibilance sharpens on loud words",
            muddiness_score=0.0,
            harshness_score=0.78,
            presence_harshness_score=0.72,
            sibilance_score=0.86,
            sibilance_frequency_hz=7700.0,
            sibilance_bandwidth_hz=980.0,
            sibilance_event_peak_dbfs=-15.8,
            sibilance_event_contrast_db=10.8,
            sibilance_event_rise_time_ms=9.0,
            sibilance_event_recovery_ms=82.0,
            dynamics_variance_score=0.76,
            crest_factor_db=19.0,
            lra_proxy_db=15.0,
            dynamics_peak_to_median_db=15.0,
            dynamics_rise_time_ms=32.0,
            dynamics_recovery_ms=170.0,
            estimated_lufs=-24.5,
            true_peak_headroom_db=5.0,
            feedback_risk=0.1,
        )
    )

    assert recommendation.status == "proposal_only"
    assert recommendation.selected_strategy.strategy_id == (
        "vocal_de_esser_before_compression_recheck"
    )
    assert recommendation.selected_strategy.method_family == "de_esser"
    assert recommendation.evidence["operator_explanation_sections"][
        "selection_policy"
    ] == "soundcheck_wing_native_de_esser_first_before_compression_priority"
    params = recommendation.selected_strategy.parameters
    assert params["processing_chain"] == (
        "ds902_de_esser_first_then_compressor_recheck"
    )
    assert params["decision_type"] == (
        "wing_native_de_esser_first_before_compression_review"
    )
    assert params["frequency_hz"] == 7700.0
    assert params["bandwidth_source"] == "measured_sibilance_bandwidth"
    assert params["bandwidth_hz"] == 1323.0
    assert params["threshold_dbfs"] == -23.7
    assert params["attack_ms"] == 4.0
    assert params["release_ms"] == 117.0
    assert params["range_db"] == 4.8
    assert params["max_gain_reduction_db"] == 4.8
    assert params["wing_de_esser_model_osc_value"] == WING_DE_ESSER_GATE_MODEL
    assert params["wing_de_esser_frequency_osc_value"] == 7700.0
    assert params["wing_de_esser_range_osc_value"] == 4.8
    assert params["wing_de_esser_mode_osc_value"] == "HF"
    assert params["de_esser_parameter_source"] == "measured_sibilance_event"
    assert params["sibilance_compressor_interaction_first_step"] is True
    assert params["followup_compressor_threshold_dbfs"] == -19.7
    assert params["followup_compressor_ratio"] == 2.8
    assert params["followup_compressor_attack_ms"] == 22.4
    assert params["followup_compressor_release_ms"] == 195.5
    assert params["followup_compressor_target_gain_reduction_db"] == 3.1
    assert params["followup_compressor_parameter_source"] == (
        "measured_vocal_dynamics_envelope"
    )
    assert params["makeup_gain_db"] == 0.0
    assert params["followup_compressor_makeup_gain_db"] == 0.0
    assert params["followup_compressor_stage"] == (
        "phrase_leveling_after_de_essing"
    )
    assert params["followup_compressor_order"] == "after_ds902_de_esser_recheck"
    assert params["followup_dynamic_eq_frequency_hz"] == 3400.0
    assert params["followup_dynamic_eq_max_gain_reduction_db"] == 3.0
    assert params["post_de_esser_compressor_recheck_required"] is True
    assert params["post_de_esser_presence_recheck_required"] is True
    assert params["followup_processing_review_only"] is True
    assert params["followup_processing_apply_ready"] is False
    assert (
        params["followup_processing_live_apply_blocked_without_wing_proof"]
        is True
    )
    alternatives = {
        candidate.strategy_id: candidate for candidate in recommendation.alternatives
    }
    assert "vocal_sibilance_compressor_order_review" in alternatives
    assert "vocal_de_esser_dynamic_presence_combo" in alternatives
    assert "vocal_compressor_envelope_smoothing" in alternatives
    tradeoffs = {
        row["strategy_id"]: row
        for row in recommendation.evidence["strategy_search"]["alternative_tradeoffs"]
    }
    assert tradeoffs["vocal_sibilance_compressor_order_review"][
        "why_not_selected"
    ] == (
        "soundcheck_wing_native_de_esser_first_priority:"
        "verified_ds902_first_then_recheck_compressor_or_deq"
    )
    assert tradeoffs["vocal_de_esser_dynamic_presence_combo"][
        "why_not_selected"
    ] == (
        "soundcheck_wing_native_de_esser_first_priority:"
        "verified_ds902_first_then_recheck_compressor_or_deq"
    )
    _assert_selected_de_esser_apply_values(
        recommendation,
        {
            "wing_de_esser_model_osc_value": WING_DE_ESSER_GATE_MODEL,
            "wing_de_esser_on_osc_value": 1,
            "wing_de_esser_frequency_osc_value": 7700.0,
            "wing_de_esser_range_osc_value": 4.8,
            "wing_de_esser_mode_osc_value": "HF",
        },
    )
    readback = recommendation.selected_strategy.future_wing_readback_contract
    assert "eq_band_readback" not in readback
    assert "dynamics_readback" not in readback
    assert readback["de_esser_readback"]["read_before_paths"] == [
        "gate/mdl",
        "gate/on",
        "gate/f",
        "gate/range",
        "gate/mode",
    ]
    assert recommendation.to_operator_proposal()["live_apply_allowed"] is False


def test_live_sibilant_dynamic_vocal_keeps_compressor_order_review_blocked():
    recommendation = analyze_smart_dsp_strategy(
        _analysis(
            mode="live",
            target_id="ch:7",
            source_role="lead_vocal",
            problem_hints=("sibilant", "harsh", "dynamic_control"),
            operator_intent="live lead vocal sibilance sharpens on loud words",
            active_performance=True,
            muddiness_score=0.0,
            harshness_score=0.78,
            presence_harshness_score=0.72,
            sibilance_score=0.86,
            sibilance_frequency_hz=7700.0,
            sibilance_bandwidth_hz=980.0,
            sibilance_event_peak_dbfs=-15.8,
            sibilance_event_contrast_db=10.8,
            sibilance_event_rise_time_ms=9.0,
            sibilance_event_recovery_ms=82.0,
            dynamics_variance_score=0.76,
            crest_factor_db=19.0,
            lra_proxy_db=15.0,
            dynamics_peak_to_median_db=15.0,
            dynamics_rise_time_ms=32.0,
            dynamics_recovery_ms=170.0,
            estimated_lufs=-24.5,
            true_peak_headroom_db=5.0,
            feedback_risk=0.1,
        )
    )

    assert recommendation.status == "proposal_only"
    assert recommendation.selected_strategy.strategy_id == (
        "manual_level_ride_before_compression"
    )
    assert recommendation.evidence["operator_explanation_sections"][
        "selection_policy"
    ] == "live_vocal_dynamics_ride_priority"
    alternatives = {
        candidate.strategy_id: candidate for candidate in recommendation.alternatives
    }
    order_review = alternatives["vocal_sibilance_compressor_order_review"]
    assert (
        "live_sibilance_compressor_order_change_requires_operator_review_and_wing_proof"
        in order_review.blockers
    )
    assert order_review.parameters["post_compression_sibilance_recheck_required"] is True
    assert recommendation.to_operator_proposal()["live_apply_allowed"] is False


def test_soundcheck_harsh_guitar_prefers_wing_native_deq_presence_tame():
    recommendation = analyze_smart_dsp_strategy(
        _analysis(
            target_id="ch:17",
            source_role="electric_guitar",
            source_label="GTR L",
            problem_hints=("harsh",),
            muddiness_score=0.0,
            harshness_score=0.82,
            presence_harshness_score=0.76,
            instrument_harsh_frequency_hz=3100.0,
            instrument_harsh_bandwidth_hz=720.0,
            instrument_harsh_event_peak_dbfs=-18.5,
            instrument_harsh_event_contrast_db=8.5,
            instrument_harsh_event_recovery_ms=160.0,
            level_plane_error_lu=0.2,
            true_peak_headroom_db=5.0,
        )
    )

    assert recommendation.status == "proposal_only"
    assert recommendation.selected_strategy.strategy_id == "instrument_dynamic_presence_tame"
    assert recommendation.selected_strategy.method_family == "dynamic_eq"
    params = recommendation.selected_strategy.parameters
    assert params["source_family"] == "guitar"
    assert params["filter_type"] == "dynamic_peaking"
    assert params["frequency_hz"] == 3100.0
    assert params["max_gain_reduction_db"] == 2.9
    assert params["q"] == 3.59
    assert params["threshold_dbfs"] == -25.5
    assert params["attack_ms"] == 4.4
    assert params["release_ms"] == 180.0
    assert params["makeup_gain_db"] == 0.0
    assert params["wing_dynamic_eq_model_osc_value"] == WING_DYNAMIC_EQ_GATE_MODEL
    assert params["wing_dynamic_eq_filter_osc_value"] == "BP"
    assert params["wing_dynamic_eq_gain_osc_value"] == -2.9
    assert params["wing_dynamic_eq_frequency_osc_value"] == 3100.0
    assert params["wing_dynamic_eq_q_osc_value"] == 3.59
    assert params["instrument_harsh_parameter_source"] == (
        "measured_instrument_harshness_event"
    )
    assert params["instrument_harsh_bandwidth_hz"] == 720.0
    assert params["instrument_harsh_event_peak_dbfs"] == -18.5
    assert params["instrument_harsh_event_contrast_db"] == 8.5
    assert params["instrument_harsh_event_recovery_ms"] == 160.0
    alternatives = {candidate.strategy_id: candidate for candidate in recommendation.alternatives}
    broad = alternatives["instrument_broad_presence_cut"]
    assert broad.parameters["filter_type"] == "peaking"
    assert broad.parameters["gain_db"] == -2.1
    assert broad.parameters["q"] == 2.97
    assert broad.future_wing_readback_contract["wing_osc_applicability"][
        "native_osc_supported"
    ] is True
    assert "instrument_level_check_before_harsh_eq" in alternatives
    assert recommendation.evidence["operator_explanation_sections"]["selection_policy"] == (
        "weighted_strategy_score"
    )
    tradeoffs = {
        row["strategy_id"]: row["why_not_selected"]
        for row in recommendation.evidence["strategy_search"]["alternative_tradeoffs"]
    }
    assert tradeoffs["instrument_broad_presence_cut"].startswith("lower_")
    readback = recommendation.selected_strategy.future_wing_readback_contract
    assert readback["wing_osc_applicability"]["native_osc_supported"] is True
    assert readback["wing_osc_applicability"]["native_osc_support_level"] == (
        "native_channel_gate_deq_dynamic_eq_osc_path_templates_declared"
    )
    assert readback["dynamic_eq_readback"][
        "wing_dynamic_eq_model_readback_must_equal"
    ] == WING_DYNAMIC_EQ_GATE_MODEL
    assert "eq_band_readback" not in readback
    assert "dynamics_readback" not in readback


def test_live_harsh_guitar_prefers_conservative_presence_cut():
    recommendation = analyze_smart_dsp_strategy(
        _analysis(
            target_id="ch:17",
            source_role="electric_guitar",
            source_label="GTR L",
            mode="live",
            problem_hints=("harsh",),
            muddiness_score=0.0,
            harshness_score=0.82,
            presence_harshness_score=0.76,
            level_plane_error_lu=0.2,
            true_peak_headroom_db=5.0,
            active_performance=True,
        )
    )

    assert recommendation.status == "proposal_only"
    assert recommendation.selected_strategy.strategy_id == "instrument_broad_presence_cut"
    assert recommendation.selected_strategy.method_family == "eq"
    dynamic = [
        candidate
        for candidate in recommendation.alternatives
        if candidate.strategy_id == "instrument_dynamic_presence_tame"
    ][0]
    assert "live_dynamic_presence_tame_requires_operator_review" in dynamic.blockers
    assert recommendation.to_operator_proposal()["can_apply_live"] is False


def test_harsh_guitar_above_level_plane_prefers_balance_before_eq():
    recommendation = analyze_smart_dsp_strategy(
        _analysis(
            target_id="ch:17",
            source_role="electric_guitar",
            source_label="GTR L",
            problem_hints=("harsh",),
            muddiness_score=0.0,
            harshness_score=0.82,
            presence_harshness_score=0.76,
            level_plane_error_lu=1.8,
            true_peak_headroom_db=5.0,
        )
    )

    assert recommendation.status == "proposal_only"
    assert recommendation.selected_strategy.strategy_id == (
        "instrument_level_check_before_harsh_eq"
    )
    assert recommendation.selected_strategy.method_family == "gain_fader"
    params = recommendation.selected_strategy.parameters
    assert params["delta_db"] < 0.0
    assert params["target_scope"] == "source_level_before_tone"


def test_soundcheck_overhead_harshness_prefers_wing_native_static_de_edge():
    recommendation = analyze_smart_dsp_strategy(
        _analysis(
            target_id="ch:10",
            source_role="overhead_cymbal",
            source_label="OH L",
            problem_hints=("harsh",),
            muddiness_score=0.0,
            harshness_score=0.82,
            presence_harshness_score=0.86,
            drum_harsh_frequency_hz=6800.0,
            drum_harsh_bandwidth_hz=1600.0,
            drum_harsh_event_peak_dbfs=-17.0,
            drum_harsh_event_contrast_db=9.0,
            drum_harsh_event_recovery_ms=130.0,
            estimated_lufs=-24.0,
            true_peak_headroom_db=5.0,
        )
    )

    assert recommendation.status == "proposal_only"
    assert recommendation.selected_strategy.strategy_id == "drum_cymbal_static_air_cut"
    assert recommendation.selected_strategy.method_family == "eq"
    params = recommendation.selected_strategy.parameters
    assert params["source_family"] == "drum"
    assert params["filter_type"] == "high_shelf"
    assert params["frequency_hz"] == 6800.0
    assert params["drum_harsh_parameter_source"] == "measured_drum_harshness_event"
    assert params["gain_db"] == -2.2
    assert params["q"] == 0.59
    assert params["drum_harsh_bandwidth_hz"] == 1600.0
    assert params["drum_harsh_event_peak_dbfs"] == -17.0
    assert params["drum_harsh_event_contrast_db"] == 9.0
    assert params["drum_harsh_event_recovery_ms"] == 130.0
    alternatives = {candidate.strategy_id: candidate for candidate in recommendation.alternatives}
    assert {
        "drum_cymbal_dynamic_de_edge",
        "drum_cymbal_mic_level_review",
    } <= set(alternatives)
    dynamic = alternatives["drum_cymbal_dynamic_de_edge"]
    assert dynamic.parameters["threshold_dbfs"] == -24.4
    assert dynamic.parameters["attack_ms"] == 2.6
    assert dynamic.parameters["release_ms"] == 149.8
    assert dynamic.future_wing_readback_contract["wing_osc_applicability"][
        "native_osc_supported"
    ] is False
    assert recommendation.evidence["operator_explanation_sections"]["selection_policy"] == (
        "weighted_strategy_score"
    )
    tradeoffs = {
        row["strategy_id"]: row["why_not_selected"]
        for row in recommendation.evidence["strategy_search"]["alternative_tradeoffs"]
    }
    assert tradeoffs["drum_cymbal_dynamic_de_edge"] == (
        "soundcheck_wing_native_osc_priority:"
        "native_eq_has_verified_wing_fw_3_0_5_osc_path_alternative_requires_mapping"
    )
    assert recommendation.selected_strategy.future_wing_readback_contract[
        "wing_osc_applicability"
    ]["native_osc_supported"] is True
    assert recommendation.to_operator_proposal()["live_apply_allowed"] is False


def test_soundcheck_overhead_top_octave_harshness_prefers_wing_native_high_cut():
    recommendation = analyze_smart_dsp_strategy(
        _analysis(
            target_id="ch:10",
            source_role="overhead_cymbal",
            source_label="OH L",
            problem_hints=("harsh",),
            muddiness_score=0.0,
            harshness_score=0.86,
            presence_harshness_score=0.88,
            drum_harsh_frequency_hz=12500.0,
            drum_harsh_bandwidth_hz=4200.0,
            drum_harsh_event_peak_dbfs=-17.0,
            drum_harsh_event_contrast_db=9.0,
            drum_harsh_event_recovery_ms=130.0,
            estimated_lufs=-24.0,
            true_peak_headroom_db=5.0,
        )
    )

    assert recommendation.status == "proposal_only"
    assert recommendation.selected_strategy.strategy_id == (
        "drum_cymbal_high_cut_air_limit"
    )
    assert recommendation.selected_strategy.method_family == "eq"
    params = recommendation.selected_strategy.parameters
    assert params["filter_type"] == "high_cut"
    assert params["frequency_hz"] == 12000.0
    assert params["slope_db_per_octave"] == 6
    assert params["wing_eq_filter_family"] == "channel_high_cut"
    assert params["wing_eq_filter_osc_path_suffixes"] == {
        "enabled": "flt/hc",
        "frequency": "flt/hcf",
        "slope": "flt/hcs",
    }
    assert params["wing_eq_high_cut_frequency_hz"] == 12000.0
    assert params["wing_eq_high_cut_slope_db_per_octave"] == 6
    readback = recommendation.selected_strategy.future_wing_readback_contract
    assert readback["eq_band_readback"]["read_before_paths"] == [
        "eq/on",
        "eq/mdl",
        "eq/mix",
        "flt/hc",
        "flt/hcf",
        "flt/hcs",
    ]
    assert "/ch/{channel}/flt/hcf" in readback["wing_osc_applicability"][
        "native_osc_path_templates"
    ]
    alternatives = {candidate.strategy_id: candidate for candidate in recommendation.alternatives}
    assert "drum_cymbal_static_air_cut" in alternatives
    assert "drum_cymbal_dynamic_de_edge" in alternatives
    assert recommendation.to_operator_proposal()["live_apply_allowed"] is False


def test_live_overhead_harshness_prefers_mic_level_review_before_eq():
    recommendation = analyze_smart_dsp_strategy(
        _analysis(
            target_id="ch:10",
            source_role="overhead_cymbal",
            source_label="OH L",
            mode="live",
            active_performance=True,
            problem_hints=("harsh",),
            muddiness_score=0.0,
            harshness_score=0.82,
            presence_harshness_score=0.86,
            drum_harsh_frequency_hz=6800.0,
            drum_harsh_bandwidth_hz=1600.0,
            drum_harsh_event_peak_dbfs=-17.0,
            drum_harsh_event_contrast_db=9.0,
            drum_harsh_event_recovery_ms=130.0,
            estimated_lufs=-24.0,
            true_peak_headroom_db=5.0,
        )
    )

    assert recommendation.status == "proposal_only"
    assert recommendation.selected_strategy.strategy_id == "drum_cymbal_mic_level_review"
    assert recommendation.selected_strategy.method_family == "analysis_only"
    params = recommendation.selected_strategy.parameters
    assert params["dsp_change"] == "hold"
    assert params["live_cymbal_harshness_review_priority"] is True
    assert params["frequency_hz"] == 6800.0
    assert params["drum_harsh_bandwidth_hz"] == 1600.0
    assert params["drum_harsh_event_peak_dbfs"] == -17.0
    assert params["drum_harsh_event_contrast_db"] == 9.0
    assert params["drum_harsh_event_recovery_ms"] == 130.0
    assert params["operator_action"] == (
        "check_mic_angle_overhead_balance_and_drum_bus_before_more_dsp"
    )
    assert params["forbidden_without_wing_proof"] == "live_cymbal_eq_dynamic_de_edge"
    assert params["operator_confirmation_required"] is True
    alternatives = {candidate.strategy_id: candidate for candidate in recommendation.alternatives}
    assert "drum_cymbal_static_air_cut" in alternatives
    dynamic = [
        candidate
        for candidate in recommendation.alternatives
        if candidate.strategy_id == "drum_cymbal_dynamic_de_edge"
    ][0]
    assert "live_dynamic_cymbal_de_edge_requires_operator_review" in dynamic.blockers
    assert "aggressive_live_strategy_requires_operator_review" not in dynamic.blockers
    proposal = recommendation.to_operator_proposal()
    assert (
        proposal["operator_explanation_sections"]["selection_policy"]
        == "live_cymbal_harshness_review_priority"
    )
    assert any(
        item["strategy_id"] == "drum_cymbal_static_air_cut"
        and item["why_not_selected"]
        == (
            "live_cymbal_harshness_review_priority:"
            "operator_overhead_balance_review_precedes_live_cymbal_eq"
        )
        for item in proposal["proposed_action"]["alternative_tradeoffs"]
    )
    assert recommendation.to_operator_proposal()["can_apply_live"] is False


def test_soundcheck_snare_ring_prefers_wing_native_deq_notch():
    recommendation = analyze_smart_dsp_strategy(
        _analysis(
            target_id="ch:9",
            source_role="snare_top",
            source_label="Snare Top",
            problem_hints=("drum_ring_resonance", "muddy", "harsh"),
            muddiness_score=0.66,
            harshness_score=0.52,
            presence_harshness_score=0.0,
            drum_ring_resonance_score=0.78,
            drum_ring_frequency_hz=420.0,
            drum_ring_decay_score=0.62,
            drum_ring_bandwidth_hz=48.0,
            drum_ring_peak_dbfs=-18.2,
            drum_ring_decay_ms=280.0,
            transient_strength=0.38,
            transient_density=1.2,
            estimated_lufs=-24.0,
            true_peak_headroom_db=5.0,
        )
    )

    assert recommendation.status == "proposal_only"
    assert recommendation.selected_strategy.strategy_id == (
        "drum_dynamic_ring_notch_control"
    )
    assert recommendation.selected_strategy.method_family == "dynamic_eq"
    params = recommendation.selected_strategy.parameters
    assert params["source_family"] == "snare"
    assert params["filter_type"] == "dynamic_peaking"
    assert params["frequency_hz"] == 420.0
    assert params["drum_ring_parameter_source"] == "measured_drum_ring_decay"
    assert params["static_gain_db"] == -0.3
    assert params["gain_db"] == -0.3
    assert params["max_gain_reduction_db"] == 4.4
    assert params["q"] == 7.5
    assert params["threshold_dbfs"] == -24.2
    assert params["attack_ms"] == 6.4
    assert params["release_ms"] == 219.8
    assert params["wing_dynamic_eq_model_osc_value"] == WING_DYNAMIC_EQ_GATE_MODEL
    assert params["wing_dynamic_eq_filter_osc_value"] == "BP"
    assert params["wing_dynamic_eq_gain_osc_value"] == -4.4
    assert params["wing_dynamic_eq_frequency_osc_value"] == 420.0
    assert params["wing_dynamic_eq_q_osc_value"] == 7.5
    assert params["drum_ring_bandwidth_hz"] == 48.0
    assert params["drum_ring_peak_dbfs"] == -18.2
    assert params["drum_ring_decay_ms"] == 280.0
    alternatives = {candidate.strategy_id: candidate for candidate in recommendation.alternatives}
    assert {
        "drum_tuning_mic_damping_review",
        "avoid_broad_drum_de_edge_for_ring",
    } <= set(alternatives)
    static = alternatives["drum_static_ring_notch_cut"]
    assert static.parameters["q"] == 6.48
    assert static.parameters["gain_db"] == -3.4
    assert static.parameters["no_boost"] is True
    assert static.future_wing_readback_contract["wing_osc_applicability"][
        "native_osc_supported"
    ] is True
    assert "drum_cymbal_dynamic_de_edge" not in alternatives
    assert recommendation.evidence["operator_explanation_sections"]["selection_policy"] == (
        "weighted_strategy_score"
    )
    tradeoffs = {
        row["strategy_id"]: row["why_not_selected"]
        for row in recommendation.evidence["strategy_search"]["alternative_tradeoffs"]
    }
    assert tradeoffs["drum_static_ring_notch_cut"].startswith("lower_")
    readback = recommendation.selected_strategy.future_wing_readback_contract
    assert readback["wing_osc_applicability"]["native_osc_supported"] is True
    assert readback["wing_osc_applicability"]["native_osc_support_level"] == (
        "native_channel_gate_deq_dynamic_eq_osc_path_templates_declared"
    )
    assert readback["dynamic_eq_readback"][
        "wing_dynamic_eq_model_readback_must_equal"
    ] == WING_DYNAMIC_EQ_GATE_MODEL
    assert "eq_band_readback" not in readback
    assert "dynamics_readback" not in readback
    assert recommendation.to_operator_proposal()["live_apply_allowed"] is False


def test_live_snare_ring_prefers_static_notch_and_blocks_dynamic_ring_control():
    recommendation = analyze_smart_dsp_strategy(
        _analysis(
            target_id="ch:9",
            source_role="snare_top",
            source_label="Snare Top",
            mode="live",
            active_performance=True,
            problem_hints=("drum_ring_resonance", "muddy", "harsh"),
            muddiness_score=0.66,
            harshness_score=0.52,
            presence_harshness_score=0.0,
            drum_ring_resonance_score=0.78,
            drum_ring_frequency_hz=420.0,
            drum_ring_decay_score=0.62,
            drum_ring_bandwidth_hz=48.0,
            drum_ring_peak_dbfs=-18.2,
            drum_ring_decay_ms=280.0,
            transient_strength=0.38,
            transient_density=1.2,
            estimated_lufs=-24.0,
            true_peak_headroom_db=5.0,
        )
    )

    assert recommendation.status == "proposal_only"
    assert recommendation.selected_strategy.strategy_id == "drum_static_ring_notch_cut"
    assert recommendation.selected_strategy.method_family == "eq"
    params = recommendation.selected_strategy.parameters
    assert params["gain_db"] < 0.0
    assert params["no_boost"] is True
    assert params["drum_ring_parameter_source"] == "measured_drum_ring_decay"
    assert params["q"] == 6.48
    assert params["drum_ring_bandwidth_hz"] == 48.0
    dynamic = [
        candidate
        for candidate in recommendation.alternatives
        if candidate.strategy_id == "drum_dynamic_ring_notch_control"
    ][0]
    assert "live_dynamic_drum_ring_control_requires_operator_review" in dynamic.blockers
    assert recommendation.to_operator_proposal()["live_apply_allowed"] is False


def test_harsh_dynamic_vocal_prefers_compressor_envelope_over_de_esser():
    recommendation = analyze_smart_dsp_strategy(
        _analysis(
            target_id="ch:8",
            source_role="lead_vocal",
            problem_hints=("harsh",),
            muddiness_score=0.0,
            harshness_score=0.82,
            sibilance_score=0.12,
            dynamics_variance_score=0.72,
            crest_factor_db=19.0,
            lra_proxy_db=16.0,
            dynamics_peak_to_median_db=14.0,
            dynamics_rise_time_ms=36.0,
            dynamics_recovery_ms=180.0,
            estimated_lufs=-25.0,
            true_peak_headroom_db=5.0,
        )
    )

    assert recommendation.status == "proposal_only"
    assert recommendation.selected_strategy.strategy_id == (
        "vocal_compressor_envelope_smoothing"
    )
    assert recommendation.selected_strategy.method_family == "dynamics"
    params = recommendation.selected_strategy.parameters
    assert params["envelope_change"] == "slower_attack_smoother_release"
    assert params["vocal_envelope_parameter_source"] == "measured_vocal_dynamics_envelope"
    assert params["dynamics_peak_to_median_db"] == 14.0
    assert params["dynamics_rise_time_ms"] == 36.0
    assert params["dynamics_recovery_ms"] == 180.0
    assert params["threshold_dbfs"] == -20.5
    assert params["ratio"] == 2.7
    assert params["attack_ms"] == 25.2
    assert params["release_ms"] == 207.0
    assert params["knee_db"] == 5.0
    assert params["wing_compressor_knee_osc_value"] == 5
    assert params["wing_compressor_knee_value_type"] == "integer"
    assert params["target_gain_reduction_db"] == 3.0
    assert params["max_attack_delta_ms"] == 15.2
    assert params["max_release_delta_ms"] == 107.0
    assert params["makeup_gain_db"] == 0.0
    assert "harsh" in recommendation.evidence["detected_problems"]

    deesser = [
        candidate
        for candidate in recommendation.alternatives
        if candidate.strategy_id == "vocal_de_esser"
    ][0]
    assert "sibilance_evidence_low_for_de_esser" in deesser.blockers


def test_live_dynamic_harsh_vocal_rides_before_tone_or_compressor_changes():
    recommendation = analyze_smart_dsp_strategy(
        _analysis(
            mode="live",
            target_id="ch:8",
            source_role="lead_vocal",
            problem_hints=("harsh", "dynamic_control"),
            muddiness_score=0.0,
            harshness_score=0.82,
            presence_harshness_score=0.72,
            sibilance_score=0.12,
            dynamics_variance_score=0.72,
            crest_factor_db=19.0,
            lra_proxy_db=16.0,
            dynamics_peak_to_median_db=14.0,
            dynamics_rise_time_ms=36.0,
            dynamics_recovery_ms=180.0,
            estimated_lufs=-25.0,
            true_peak_headroom_db=5.0,
            active_performance=True,
        )
    )

    assert recommendation.status == "proposal_only"
    assert recommendation.selected_strategy.strategy_id == (
        "manual_level_ride_before_compression"
    )
    assert recommendation.selected_strategy.method_family == "analysis_only"
    assert recommendation.evidence["operator_explanation_sections"][
        "selection_policy"
    ] == "live_dynamic_harsh_vocal_ride_priority"
    params = recommendation.selected_strategy.parameters
    assert params["live_dynamic_harsh_vocal_ride_priority"] is True
    assert params["live_dynamic_harshness_score"] == 0.82
    assert params["live_dynamic_harshness_sibilance_score"] == 0.12
    assert params["live_dynamic_harshness_tone_processing_deferred"] is True
    assert params["live_dynamic_harshness_compressor_change_deferred"] is True
    assert params["live_dynamic_harshness_remeasure_after_ride_required"] is True
    assert (
        params[
            "live_dynamic_harshness_no_compressor_eq_or_deq_without_operator_wing_proof"
        ]
        is True
    )
    assert params["live_dynamics_ride_profile"] == "lead_vocal_dynamic_harshness"
    assert params["forbidden_without_wing_proof"] == (
        "compressor_dynamic_eq_static_presence_cut_until_ride_remeasure"
    )
    alternatives = {
        candidate.strategy_id: candidate for candidate in recommendation.alternatives
    }
    assert "broad_presence_cut" in alternatives
    assert "presence_dynamic_eq_cut" in alternatives
    assert "vocal_compressor_envelope_smoothing" in alternatives
    assert "vocal_phrase_leveling_compression" in alternatives
    tradeoffs = {
        row["strategy_id"]: row
        for row in recommendation.evidence["strategy_search"]["alternative_tradeoffs"]
    }
    expected_reason = (
        "live_dynamic_harsh_vocal_ride_priority:"
        "ride_and_remeasure_before_tone_or_compressor_change"
    )
    assert tradeoffs["broad_presence_cut"]["why_not_selected"] == expected_reason
    assert tradeoffs["presence_dynamic_eq_cut"]["why_not_selected"] == expected_reason
    assert (
        tradeoffs["vocal_compressor_envelope_smoothing"]["why_not_selected"]
        == expected_reason
    )
    assert (
        tradeoffs["vocal_phrase_leveling_compression"]["why_not_selected"]
        == expected_reason
    )
    proposal = recommendation.to_operator_proposal()
    assert proposal["can_apply_live"] is False
    assert proposal["live_apply_allowed"] is False


def test_soundcheck_vocal_masking_prefers_wing_native_static_unmasking():
    recommendation = analyze_smart_dsp_strategy(
        _analysis(
            target_id="ch:12",
            source_role="keys",
            source_label="Keys Pad",
            problem_hints=("masking",),
            masking_conflict_role="lead_vocal",
            masking_frequency_hz=2400.0,
            masking_score=0.84,
            masking_bandwidth_hz=760.0,
            masking_overlap_peak_share=0.31,
            rms_dbfs=-24.0,
            true_peak_headroom_db=5.0,
        )
    )

    assert recommendation.status == "proposal_only"
    assert (
        recommendation.selected_strategy.strategy_id
        == "vocal_intelligibility_static_pocket_eq"
    )
    assert recommendation.selected_strategy.method_family == "eq"
    params = recommendation.selected_strategy.parameters
    assert params["masked_role"] == "lead_vocal"
    assert params["masker_role"] == "keys"
    assert params["current_target_is_masker"] is True
    assert params["linked_counter_boost_db"] == 0.0
    assert params["requires_pair_readback"] is True
    assert params["filter_type"] == "peaking"
    assert params["frequency_hz"] == 2400.0
    assert params["gain_db"] == -2.7
    assert params["q"] == 2.18
    assert params["static_eq_q_source"] == "measured_masking_overlap_band"
    assert params["masking_parameter_source"] == "measured_masking_overlap_band"
    assert params["masking_bandwidth_hz"] == 760.0
    assert params["masking_overlap_peak_share"] == 0.31
    assert recommendation.evidence["masking_bandwidth_hz"] == 760.0
    assert recommendation.evidence["masking_overlap_peak_share"] == 0.31
    alternatives = {candidate.strategy_id: candidate for candidate in recommendation.alternatives}
    assert {
        "vocal_intelligibility_dynamic_unmasking",
        "vocal_presence_boost_unmasking",
    } <= set(alternatives)
    dynamic = alternatives["vocal_intelligibility_dynamic_unmasking"]
    assert dynamic.parameters["sidechain_role"] == "lead_vocal"
    assert dynamic.parameters["makeup_gain_db"] == 0.0
    assert dynamic.parameters["masking_parameter_source"] == "measured_masking_overlap_band"
    assert dynamic.parameters["bandwidth_q"] == 2.63
    assert dynamic.parameters["threshold_dbfs"] == -20.6
    assert dynamic.parameters["attack_ms"] == 8.7
    assert dynamic.parameters["release_ms"] == 121.8
    assert dynamic.future_wing_readback_contract["wing_osc_applicability"][
        "native_osc_supported"
    ] is False
    boost = alternatives["vocal_presence_boost_unmasking"]
    assert "boosting_masked_vocal_should_not_precede_masker_pocket" in boost.blockers
    assert recommendation.evidence["operator_explanation_sections"]["selection_policy"] == (
        "weighted_strategy_score"
    )
    tradeoffs = {
        row["strategy_id"]: row["why_not_selected"]
        for row in recommendation.evidence["strategy_search"]["alternative_tradeoffs"]
    }
    assert tradeoffs["vocal_intelligibility_dynamic_unmasking"] == (
        "soundcheck_wing_native_osc_priority:"
        "native_eq_has_verified_wing_fw_3_0_5_osc_path_alternative_requires_mapping"
    )
    assert recommendation.selected_strategy.future_wing_readback_contract[
        "wing_osc_applicability"
    ]["native_osc_supported"] is True
    proposal = recommendation.to_operator_proposal()
    _assert_proposal_denial(proposal)
    assert proposal["selected_strategy"]["parameters"]["processing_target"] == (
        "current_masker_channel"
    )


def test_live_vocal_masking_prefers_static_pocket_over_dynamic_ducking():
    recommendation = analyze_smart_dsp_strategy(
        _analysis(
            target_id="ch:12",
            source_role="keys",
            source_label="Keys Pad",
            mode="live",
            active_performance=True,
            problem_hints=("masking",),
            masking_conflict_role="lead_vocal",
            masking_frequency_hz=2400.0,
            masking_score=0.84,
            true_peak_headroom_db=5.0,
        )
    )

    assert recommendation.status == "proposal_only"
    assert (
        recommendation.selected_strategy.strategy_id
        == "vocal_intelligibility_static_pocket_eq"
    )
    assert recommendation.selected_strategy.parameters["linked_counter_boost_db"] == 0.0
    dynamic = [
        candidate
        for candidate in recommendation.alternatives
        if candidate.strategy_id == "vocal_intelligibility_dynamic_unmasking"
    ][0]
    assert "live_dynamic_unmasking_requires_operator_review" in dynamic.blockers
    assert "aggressive_live_strategy_requires_operator_review" in dynamic.blockers
    assert recommendation.to_operator_proposal()["live_apply_allowed"] is False


def test_soundcheck_vocal_masking_rebalances_over_level_masker_before_eq():
    recommendation = analyze_smart_dsp_strategy(
        _analysis(
            target_id="ch:12",
            source_role="keys",
            source_label="Keys Pad",
            problem_hints=("masking",),
            masking_conflict_role="lead_vocal",
            masking_frequency_hz=2400.0,
            masking_score=0.84,
            masking_bandwidth_hz=760.0,
            masking_overlap_peak_share=0.31,
            level_plane_error_lu=3.6,
            true_peak_headroom_db=5.0,
        )
    )

    assert recommendation.status == "proposal_only"
    assert (
        recommendation.selected_strategy.strategy_id
        == "vocal_masking_level_plane_rebalance"
    )
    params = recommendation.selected_strategy.parameters
    assert params["masked_role"] == "lead_vocal"
    assert params["masker_role"] == "keys"
    assert params["delta_db"] == -2.5
    assert params["current_minus_target_lu"] == 3.6
    assert params["reason"] == "balance_may_solve_masking_before_tone"
    assert recommendation.evidence["operator_explanation_sections"][
        "selection_policy"
    ] == (
        "soundcheck_vocal_masking_level_plane_priority"
    )
    problem_action = recommendation.evidence["strategy_search"][
        "selected_rationale"
    ]["problem_action_decision_summary"]
    _assert_problem_action_decision_summary(
        problem_action,
        strategy_id="vocal_masking_level_plane_rebalance",
        method_family="gain_fader",
        expected_action_intent="level_plane_rebalance_before_tone_change",
        expected_method_families=("gain_fader", "eq", "dynamic_eq"),
    )
    assert problem_action["problem_first_not_method_locked"] is True
    assert problem_action["selection_basis"] == (
        "balance_or_headroom_before_tone_processing"
    )
    proposal = recommendation.to_operator_proposal()
    _assert_problem_action_decision_summary(
        proposal["problem_action_decision_summary"],
        strategy_id="vocal_masking_level_plane_rebalance",
        method_family="gain_fader",
        expected_action_intent="level_plane_rebalance_before_tone_change",
        expected_method_families=("gain_fader", "eq", "dynamic_eq"),
    )
    tradeoffs = {
        item["strategy_id"]: item["why_not_selected"]
        for item in proposal["alternative_tradeoffs"]
    }
    assert tradeoffs["vocal_intelligibility_dynamic_unmasking"] == (
        "vocal_masker_level_plane_priority:"
        "rebalance_masker_level_before_pocket_eq_or_ducking"
    )


def test_live_vocal_masking_rebalances_over_level_masker_before_static_pocket():
    recommendation = analyze_smart_dsp_strategy(
        _analysis(
            target_id="ch:12",
            source_role="keys",
            source_label="Keys Pad",
            mode="live",
            active_performance=True,
            problem_hints=("masking",),
            masking_conflict_role="lead_vocal",
            masking_frequency_hz=2400.0,
            masking_score=0.84,
            masking_bandwidth_hz=760.0,
            masking_overlap_peak_share=0.31,
            level_plane_error_lu=3.6,
            true_peak_headroom_db=5.0,
        )
    )

    assert recommendation.status == "proposal_only"
    assert (
        recommendation.selected_strategy.strategy_id
        == "vocal_masking_level_plane_rebalance"
    )
    params = recommendation.selected_strategy.parameters
    assert params["delta_db"] == -2.5
    assert params["live_stepped_change_required"] is True
    assert params["live_max_single_step_db"] == 0.75
    assert params["live_readback_after_each_step_required"] is True
    assert recommendation.evidence["operator_explanation_sections"][
        "selection_policy"
    ] == (
        "live_vocal_masking_level_plane_priority"
    )
    static_pocket = [
        candidate
        for candidate in recommendation.alternatives
        if candidate.strategy_id == "vocal_intelligibility_static_pocket_eq"
    ][0]
    assert static_pocket.total_score > recommendation.selected_strategy.total_score
    proposal = recommendation.to_operator_proposal()
    tradeoffs = {
        item["strategy_id"]: item["why_not_selected"]
        for item in proposal["alternative_tradeoffs"]
    }
    assert tradeoffs["vocal_intelligibility_static_pocket_eq"] == (
        "vocal_masker_level_plane_priority:"
        "rebalance_masker_level_before_pocket_eq_or_ducking"
    )
    assert proposal["live_apply_allowed"] is False


def test_kick_bass_masking_low_headroom_prefers_headroom_pullback_before_eq():
    recommendation = analyze_smart_dsp_strategy(
        _analysis(
            target_id="ch:1",
            source_role="kick_in",
            problem_hints=("masking",),
            masking_conflict_role="bass",
            masking_frequency_hz=82.0,
            masking_score=0.78,
            level_plane_error_lu=-3.0,
            true_peak_headroom_db=0.6,
        )
    )

    assert recommendation.status == "analysis_only"
    assert "true_peak_headroom_low" in recommendation.blockers
    assert recommendation.selected_strategy.strategy_id == (
        "source_true_peak_headroom_pullback"
    )
    assert recommendation.selected_strategy.method_family == "gain_fader"
    assert recommendation.selected_strategy.parameters["delta_db"] < 0.0
    assert recommendation.selected_strategy.parameters["bounded_delta_db"] > 0.0
    assert recommendation.selected_strategy.parameters["headroom_restoration_first"] is True
    assert recommendation.evidence["operator_explanation_sections"][
        "selection_policy"
    ] == "true_peak_headroom_safety_priority"
    _assert_gain_delta_readback_contract(
        recommendation.selected_strategy,
        source="delta_db",
        target_surface="channel",
        expected_delta=-1.1,
        path_prefix="/ch/{channel}",
    )
    _assert_selected_gain_delta_apply_values(recommendation, -1.1)
    eq_split = [
        candidate
        for candidate in recommendation.alternatives
        if candidate.strategy_id == "kick_bass_pocket_eq_split"
    ][0]
    assert eq_split.parameters["linked_counter_boost_db"] == 0.0
    assert eq_split.parameters["gain_db"] < 0.0
    eq_tradeoff = [
        row
        for row in recommendation.evidence["strategy_search"]["alternative_tradeoffs"]
        if row["strategy_id"] == "kick_bass_pocket_eq_split"
    ][0]
    assert (
        eq_tradeoff["why_not_selected"]
        == "true_peak_headroom_priority:"
        "restore_headroom_before_tone_dynamics_or_gain_up"
    )
    gain_up = [
        candidate
        for candidate in recommendation.alternatives
        if candidate.strategy_id == "kick_bass_level_plane_rebalance"
    ][0]
    assert "gain_up_requires_operator_confirmation" in gain_up.blockers
    proposal = recommendation.to_operator_proposal()
    _assert_proposal_denial(proposal)
    assert "true_peak_headroom_low" in proposal["safety_blockers"]


def test_live_source_true_peak_headroom_pullback_beats_tone_change_but_not_evidence_guard():
    recommendation = analyze_smart_dsp_strategy(
        _analysis(
            mode="live",
            operator_intent="clean muddy vocal live but the channel is near clipping",
            problem_hints=("muddy",),
            muddiness_score=0.8,
            true_peak_headroom_db=0.6,
            active_performance=True,
        )
    )

    assert recommendation.status == "analysis_only"
    assert "true_peak_headroom_low" in recommendation.blockers
    assert recommendation.selected_strategy.strategy_id == (
        "source_true_peak_headroom_pullback"
    )
    assert recommendation.selected_strategy.method_family == "gain_fader"
    params = recommendation.selected_strategy.parameters
    assert params["delta_db"] < 0.0
    assert params["bounded_delta_db"] <= 1.5
    assert params["live_stepped_change_required"] is True
    assert params["live_max_single_step_db"] == 0.75
    assert recommendation.evidence["operator_explanation_sections"][
        "selection_policy"
    ] == "true_peak_headroom_safety_priority"
    _assert_gain_delta_readback_contract(
        recommendation.selected_strategy,
        source="delta_db",
        target_surface="channel",
        expected_delta=-0.9,
        path_prefix="/ch/{channel}",
    )
    _assert_selected_gain_delta_apply_values(recommendation, -0.9)

    low_evidence = analyze_smart_dsp_strategy(
        _analysis(
            mode="live",
            operator_intent="clean muddy vocal live but the channel is near clipping",
            problem_hints=("muddy",),
            muddiness_score=0.8,
            true_peak_headroom_db=0.6,
            evidence_confidence=0.4,
            active_performance=True,
        )
    )

    assert low_evidence.status == "analysis_only"
    assert "audio_evidence_confidence_low" in low_evidence.blockers
    assert "true_peak_headroom_low" in low_evidence.blockers
    assert low_evidence.selected_strategy.strategy_id == "hold_for_audio_evidence_recheck"


def test_soundcheck_kick_bass_overlap_prefers_wing_native_eq_split():
    recommendation = analyze_smart_dsp_strategy(
        _analysis(
            target_id="ch:2",
            source_role="bass",
            source_label="Bass DI",
            problem_hints=("masking",),
            muddiness_score=0.0,
            masking_conflict_role="kick",
            masking_frequency_hz=74.0,
            masking_score=0.92,
            masking_bandwidth_hz=28.0,
            masking_overlap_peak_share=0.42,
            rms_dbfs=-23.0,
            estimated_lufs=-24.0,
            true_peak_headroom_db=6.0,
        )
    )

    assert recommendation.status == "proposal_only"
    assert recommendation.selected_strategy.strategy_id == "kick_bass_pocket_eq_split"
    assert recommendation.selected_strategy.method_family == "eq"
    params = recommendation.selected_strategy.parameters
    assert params["relationship"] == "kick_masks_bass"
    assert params["target_role"] == "bass"
    assert params["counterpart_role"] == "kick"
    assert 55.0 <= params["frequency_hz"] <= 130.0
    assert params["target_cut_db"] == -3.3
    assert params["counterpart_cut_db"] == -2.1
    assert params["gain_db"] == -3.3
    assert params["q"] == 1.47
    assert params["static_eq_q_source"] == "measured_masking_overlap_band"
    assert params["masking_parameter_source"] == "measured_masking_overlap_band"
    assert params["masking_bandwidth_hz"] == 28.0
    assert params["masking_overlap_peak_share"] == 0.42
    assert params["requires_pair_readback"] is True
    assert recommendation.evidence["masking_bandwidth_hz"] == 28.0
    assert recommendation.evidence["masking_overlap_peak_share"] == 0.42
    alternatives = {candidate.strategy_id: candidate for candidate in recommendation.alternatives}
    assert {
        "kick_bass_dynamic_pocket_ducking",
        "arrangement_aware_low_end_separation_hold",
    } <= set(alternatives)
    dynamic = alternatives["kick_bass_dynamic_pocket_ducking"]
    assert dynamic.parameters["sidechain_role"] == "kick"
    assert dynamic.parameters["masking_parameter_source"] == "measured_masking_overlap_band"
    assert dynamic.parameters["bandwidth_q"] == 1.82
    assert dynamic.parameters["dynamic_eq_q"] == 1.82
    assert dynamic.parameters["dynamic_eq_frequency_hz"] == dynamic.parameters["frequency_hz"]
    assert dynamic.parameters["dynamic_eq_filter_type"] == "dynamic_peaking"
    assert dynamic.parameters["dynamic_eq_ratio"] == 2.0
    assert dynamic.parameters["threshold_dbfs"] == -19.4
    assert dynamic.parameters["dynamic_eq_threshold_dbfs"] == -19.4
    assert dynamic.parameters["dynamic_eq_max_gain_reduction_db"] == (
        dynamic.parameters["max_gain_reduction_db"]
    )
    assert dynamic.parameters["attack_ms"] == 4.3
    assert dynamic.parameters["release_ms"] == 117.8
    assert dynamic.parameters["wing_dynamic_eq_external_key_review_only"] is True
    assert dynamic.parameters["wing_dynamic_eq_external_key_source_role"] == "kick"
    assert dynamic.parameters["wing_dynamic_eq_external_key_source_target_id"] == ""
    assert (
        dynamic.parameters["wing_dynamic_eq_external_key_source_target_surface"]
        == "unresolved"
    )
    assert dynamic.parameters[
        "wing_dynamic_eq_external_key_source_target_parse_blocker"
    ] == "wing_dynamic_eq_external_key_source_requires_reference_target_id"
    assert dynamic.parameters["wing_dynamic_eq_external_key_candidate_source_alias"] == ""
    assert (
        dynamic.parameters["wing_dynamic_eq_external_key_concrete_source_available"]
        is False
    )
    assert (
        dynamic.parameters[
            "wing_dynamic_eq_external_key_apply_blocked_without_concrete_source"
        ]
        is True
    )
    assert (
        dynamic.parameters[
            "wing_dynamic_eq_external_key_apply_blocked_even_with_concrete_source"
        ]
        is False
    )
    assert dynamic.parameters[
        "wing_dynamic_eq_external_key_requires_concrete_source_channel"
    ] is True
    assert dynamic.parameters["wing_dynamic_eq_keyed_ducking_live_apply_blocked"] is True
    assert dynamic.parameters["wing_dynamic_eq_mapping_unsupported_reason"] == (
        "dynamic_eq_sidechain_or_paired_ducking_requires_verified_wing_sidechain_mapping"
    )
    assert dynamic.future_wing_readback_contract["wing_osc_applicability"][
        "native_osc_supported"
    ] is False
    assert dynamic.future_wing_readback_contract["wing_osc_applicability"][
        "native_osc_support_level"
    ] == "dynamic_eq_value_mapping_required"
    assert recommendation.evidence["operator_explanation_sections"]["selection_policy"] == (
        "weighted_strategy_score"
    )
    tradeoffs = {
        row["strategy_id"]: row["why_not_selected"]
        for row in recommendation.evidence["strategy_search"]["alternative_tradeoffs"]
    }
    assert tradeoffs["kick_bass_dynamic_pocket_ducking"] == (
        "soundcheck_wing_native_osc_priority:"
        "native_eq_has_verified_wing_fw_3_0_5_osc_path_alternative_requires_mapping"
    )
    assert recommendation.selected_strategy.future_wing_readback_contract[
        "wing_osc_applicability"
    ]["native_osc_supported"] is True


def test_soundcheck_kick_bass_concrete_key_transient_overlap_prefers_dynamic_ducking_review_only():
    recommendation = analyze_smart_dsp_strategy(
        _analysis(
            target_id="ch:2",
            source_role="bass",
            source_label="Bass DI",
            problem_hints=("masking",),
            muddiness_score=0.0,
            masking_conflict_role="kick",
            masking_conflict_target_id="ch:1",
            masking_frequency_hz=74.0,
            masking_score=0.94,
            masking_bandwidth_hz=28.0,
            masking_overlap_peak_share=0.48,
            arrangement_overlap_score=0.12,
            level_plane_error_lu=0.0,
            rms_dbfs=-23.0,
            estimated_lufs=-24.0,
            true_peak_headroom_db=6.0,
            transient_strength=0.82,
            transient_rise_time_ms=7.0,
            dynamics_peak_to_median_db=16.0,
        )
    )

    assert recommendation.status == "analysis_only"
    assert recommendation.blockers == (WING_DYNAMIC_EQ_VALUE_MAPPING_BLOCKER,)
    assert (
        recommendation.selected_strategy.strategy_id
        == "kick_bass_dynamic_pocket_ducking"
    )
    assert recommendation.selected_strategy.method_family == "dynamic_eq"
    assert recommendation.evidence["operator_explanation_sections"][
        "selection_policy"
    ] == "soundcheck_kick_bass_dynamic_ducking_priority"
    params = recommendation.selected_strategy.parameters
    assert params["relationship"] == "kick_masks_bass"
    assert params["processing_target"] == "current_bass_channel"
    assert params["sidechain_role"] == "kick"
    assert params["wing_dynamic_eq_external_key_source_target_id"] == "ch:1"
    assert params["wing_dynamic_eq_external_key_candidate_source_alias"] == "CH.1"
    assert params["wing_dynamic_eq_external_key_concrete_source_available"] is True
    assert (
        params["wing_dynamic_eq_external_key_verified_osc_mapping_available"]
        is False
    )
    assert (
        params["wing_dynamic_eq_external_key_apply_blocked_without_concrete_source"]
        is False
    )
    assert (
        params["wing_dynamic_eq_external_key_apply_blocked_even_with_concrete_source"]
        is True
    )
    assert params["wing_dynamic_eq_keyed_ducking_live_apply_blocked"] is True
    assert params["soundcheck_dynamic_ducking_priority_ready"] is True
    assert params["soundcheck_dynamic_ducking_review_only"] is True
    assert params["static_eq_native_fallback_strategy_id"] == "kick_bass_pocket_eq_split"
    assert params["transient_masking_score"] >= 0.88
    assert params["dynamic_eq_filter_type"] == "dynamic_peaking"
    assert params["dynamic_eq_frequency_hz"] == 74.0
    assert params["dynamic_eq_q"] == 1.82
    assert params["dynamic_eq_max_gain_reduction_db"] == 3.5
    assert params["dynamic_eq_threshold_dbfs"] == -19.5
    assert params["dynamic_eq_attack_ms"] == 4.2
    assert params["dynamic_eq_release_ms"] == 115.0

    alternatives = {
        candidate.strategy_id: candidate for candidate in recommendation.alternatives
    }
    assert alternatives["kick_bass_pocket_eq_split"].method_family == "eq"
    assert alternatives["kick_bass_pocket_eq_split"].future_wing_readback_contract[
        "wing_osc_applicability"
    ]["native_osc_supported"] is True
    tradeoffs = {
        row["strategy_id"]: row["why_not_selected"]
        for row in recommendation.evidence["strategy_search"]["alternative_tradeoffs"]
    }
    assert tradeoffs["kick_bass_pocket_eq_split"] == (
        "kick_bass_dynamic_ducking_priority:"
        "concrete_counterpart_key_and_transient_overlap_make_ducking_more_precise_"
        "than_static_eq_review_only"
    )
    assert recommendation.selected_strategy.future_wing_readback_contract[
        "wing_osc_applicability"
    ]["native_osc_supported"] is False
    assert recommendation.selected_strategy.future_wing_readback_contract[
        "wing_osc_applicability"
    ]["native_osc_support_level"] == "dynamic_eq_value_mapping_required"
    contract = recommendation.evidence["decision_engine_contract"]
    assert contract["strategy_search"]["selected_processing_type"] == "dynamic_eq"
    assert contract["wing_readback_contract"]["live_write"] is False
    assert contract["wing_readback_contract"]["live_authority_granted"] is False
    _assert_selected_has_no_apply_values(recommendation)
    proposal = recommendation.to_operator_proposal()
    _assert_proposal_denial(proposal)
    assert proposal["decision_engine_contract"] == contract


def test_kick_bass_external_key_abstract_source_target_stays_unresolved_review_only():
    recommendation = analyze_smart_dsp_strategy(
        _analysis(
            target_id="ch:2",
            source_role="bass",
            source_label="Bass DI",
            problem_hints=("masking",),
            muddiness_score=0.0,
            masking_conflict_role="kick",
            masking_conflict_target_id="stagebox:kick",
            masking_frequency_hz=74.0,
            masking_score=0.92,
            masking_bandwidth_hz=28.0,
            masking_overlap_peak_share=0.42,
            rms_dbfs=-23.0,
            estimated_lufs=-24.0,
            true_peak_headroom_db=6.0,
        )
    )

    alternatives = {
        candidate.strategy_id: candidate for candidate in recommendation.alternatives
    }
    dynamic = alternatives["kick_bass_dynamic_pocket_ducking"]
    assert dynamic.parameters["wing_dynamic_eq_external_key_source_target_id"] == (
        "stagebox:kick"
    )
    assert (
        dynamic.parameters["wing_dynamic_eq_external_key_source_target_surface"]
        == "unresolved"
    )
    assert dynamic.parameters[
        "wing_dynamic_eq_external_key_source_target_parse_blocker"
    ] == (
        "wing_dynamic_eq_external_key_source_target_requires_concrete_wing_surface"
    )
    assert dynamic.parameters["wing_dynamic_eq_external_key_candidate_source_alias"] == ""
    assert (
        dynamic.parameters["wing_dynamic_eq_external_key_concrete_source_available"]
        is False
    )
    assert (
        dynamic.parameters[
            "wing_dynamic_eq_external_key_apply_blocked_without_concrete_source"
        ]
        is True
    )
    assert dynamic.future_wing_readback_contract["wing_osc_applicability"][
        "native_osc_supported"
    ] is False


def test_soundcheck_kick_bass_hot_bus_rebalances_before_dynamic_ducking():
    recommendation = analyze_smart_dsp_strategy(
        _analysis(
            target_id="ch:2",
            source_role="bass",
            source_label="Bass DI",
            problem_hints=("masking",),
            muddiness_score=0.0,
            masking_conflict_role="kick",
            masking_frequency_hz=74.0,
            masking_score=0.92,
            masking_bandwidth_hz=28.0,
            masking_overlap_peak_share=0.42,
            level_plane_error_lu=0.3,
            summed_bus_evidence_available=True,
            summed_bus_role="low_end_bus",
            summed_bus_target_id="bus:3",
            summed_bus_true_peak_headroom_db=5.5,
            summed_bus_level_plane_error_lu=4.2,
            true_peak_headroom_db=6.0,
        )
    )

    assert recommendation.status == "proposal_only"
    assert recommendation.selected_strategy.strategy_id == (
        "kick_bass_summed_bus_rebalance_before_ducking"
    )
    assert recommendation.selected_strategy.method_family == "gain_fader"
    assert recommendation.evidence["operator_explanation_sections"][
        "selection_policy"
    ] == "soundcheck_kick_bass_summed_bus_rebalance_priority"
    params = recommendation.selected_strategy.parameters
    assert params["target_scope"] == "summed_low_end_bus_before_ducking"
    assert params["bus_delta_db"] < 0.0
    assert params["source_delta_db"] == 0.0
    assert params["summed_bus_role"] == "low_end_bus"
    assert params["summed_bus_target_id"] == "bus:3"
    assert params["summed_bus_level_plane_error_lu"] == 4.2
    assert params["wing_osc_target_id_override"] == "bus:3"
    assert params["wing_osc_original_target_id"] == "ch:2"
    assert params["wing_osc_effective_target_id"] == "bus:3"
    _assert_gain_delta_readback_contract(
        recommendation.selected_strategy,
        source="bus_delta_db",
        target_surface="bus",
        expected_delta=params["bus_delta_db"],
        path_prefix="/bus/{bus}",
    )
    gain_readback = recommendation.selected_strategy.future_wing_readback_contract[
        "gain_fader_readback"
    ]
    assert "/ch/{channel}/fdr" not in gain_readback["wing_osc_path_templates"]

    duck = [
        candidate
        for candidate in recommendation.alternatives
        if candidate.strategy_id == "kick_bass_dynamic_pocket_ducking"
    ][0]
    assert "summed_bus_rebalance_should_precede_dynamic_ducking" in duck.blockers
    tradeoff = [
        item
        for item in recommendation.evidence["strategy_search"][
            "alternative_tradeoffs"
        ]
        if item["strategy_id"] == "kick_bass_dynamic_pocket_ducking"
    ][0]
    assert tradeoff["why_not_selected"] == (
        "low_end_bus_level_priority:"
        "rebalance_summed_bus_before_ducking_or_eq"
    )
    assert recommendation.to_operator_proposal()["live_apply_allowed"] is False


def test_live_kick_bass_hot_bus_prefers_wing_native_bus_pullback_before_eq_or_ducking():
    recommendation = analyze_smart_dsp_strategy(
        _analysis(
            target_id="ch:2",
            source_role="bass",
            source_label="Bass DI",
            mode="live",
            problem_hints=("masking",),
            muddiness_score=0.0,
            masking_conflict_role="kick",
            masking_frequency_hz=74.0,
            masking_score=0.92,
            masking_bandwidth_hz=28.0,
            masking_overlap_peak_share=0.42,
            level_plane_error_lu=0.3,
            active_performance=True,
            summed_bus_evidence_available=True,
            summed_bus_role="low_end_bus",
            summed_bus_target_id="bus:3",
            summed_bus_true_peak_headroom_db=5.5,
            summed_bus_level_plane_error_lu=4.2,
            true_peak_headroom_db=6.0,
        )
    )

    assert recommendation.status == "proposal_only"
    assert recommendation.selected_strategy.strategy_id == (
        "kick_bass_summed_bus_rebalance_before_ducking"
    )
    assert recommendation.selected_strategy.method_family == "gain_fader"
    assert recommendation.evidence["operator_explanation_sections"][
        "selection_policy"
    ] == "live_kick_bass_summed_bus_pullback_priority"
    params = recommendation.selected_strategy.parameters
    assert params["target_scope"] == "summed_low_end_bus_before_ducking"
    assert params["bus_delta_db"] == -1.5
    assert params["delta_db"] == -1.5
    assert params["bounded_delta_db"] == 1.5
    assert params["source_delta_db"] == 0.0
    assert params["summed_bus_target_id"] == "bus:3"
    assert params["wing_osc_target_id_override"] == "bus:3"
    assert params["wing_osc_original_target_id"] == "ch:2"
    assert params["wing_osc_effective_target_id"] == "bus:3"
    assert params["live_kick_bass_bus_pullback_only_proposal"] is True
    assert params["live_kick_bass_bus_pullback_max_delta_db"] == 1.5
    assert params["live_kick_bass_source_eq_deferred_until_bus_recheck"] is True
    assert params["live_kick_bass_ducking_deferred_until_bus_recheck"] is True
    assert (
        params["live_kick_bass_rebalance_live_apply_blocked_without_wing_proof"]
        is True
    )
    assert params["live_stepped_change_required"] is True
    assert params["live_readback_after_each_step_required"] is True
    assert params["live_max_single_step_db"] == 0.75
    _assert_gain_delta_readback_contract(
        recommendation.selected_strategy,
        source="bus_delta_db",
        target_surface="bus",
        expected_delta=-1.5,
        path_prefix="/bus/{bus}",
    )
    _assert_selected_gain_delta_apply_values(recommendation, -1.5)
    alternatives = {
        candidate.strategy_id: candidate for candidate in recommendation.alternatives
    }
    assert "kick_bass_pocket_eq_split" in alternatives
    assert "kick_bass_dynamic_pocket_ducking" in alternatives
    assert (
        "summed_bus_rebalance_should_precede_dynamic_ducking"
        in alternatives["kick_bass_dynamic_pocket_ducking"].blockers
    )
    tradeoffs = {
        row["strategy_id"]: row["why_not_selected"]
        for row in recommendation.evidence["strategy_search"]["alternative_tradeoffs"]
    }
    assert tradeoffs["kick_bass_pocket_eq_split"] == (
        "low_end_bus_level_priority:"
        "rebalance_summed_bus_before_ducking_or_eq"
    )
    proposal = recommendation.to_operator_proposal()
    assert proposal["live_apply_allowed"] is False
    assert proposal["live_authority_granted"] is False
    assert proposal["live_write"] is False


def test_soundcheck_kick_target_avoids_ducking_kick_from_bass():
    recommendation = analyze_smart_dsp_strategy(
        _analysis(
            target_id="ch:1",
            source_role="kick_in",
            source_label="Kick In",
            problem_hints=("masking",),
            operator_intent="soundcheck kick bass low end overlap",
            muddiness_score=0.0,
            masking_conflict_role="bass",
            masking_frequency_hz=82.0,
            masking_score=0.92,
            masking_bandwidth_hz=34.0,
            masking_overlap_peak_share=0.44,
            rms_dbfs=-22.0,
            estimated_lufs=-23.0,
            true_peak_headroom_db=6.0,
        )
    )

    assert recommendation.status == "proposal_only"
    assert recommendation.selected_strategy.strategy_id == "kick_bass_pocket_eq_split"
    assert recommendation.selected_strategy.method_family == "eq"
    params = recommendation.selected_strategy.parameters
    assert params["relationship"] == "bass_masks_kick"
    assert params["target_role"] == "kick"
    assert params["counterpart_role"] == "bass"
    assert params["linked_counter_boost_db"] == 0.0
    assert params["gain_db"] < 0.0
    duck = [
        candidate
        for candidate in recommendation.alternatives
        if candidate.strategy_id == "kick_bass_dynamic_pocket_ducking"
    ][0]
    assert duck.parameters["processing_target"] == "counterpart_bass_channel"
    assert duck.parameters["sidechain_role"] == "bass"
    assert duck.parameters["wing_dynamic_eq_external_key_source_role"] == "bass"
    assert duck.parameters["wing_dynamic_eq_external_key_target_role"] == "bass"
    assert duck.parameters["wing_dynamic_eq_keyed_ducking_live_apply_blocked"] is True
    assert duck.parameters["current_target_ducking_safe"] is False
    assert duck.parameters["current_target_is_kick_anchor"] is True
    assert duck.parameters["recommended_counterpart_role"] == "bass"
    assert "dynamic_ducking_current_kick_can_kill_punch" in duck.blockers
    assert "analyze_counterpart_bass_for_kick_keyed_ducking" in duck.blockers
    tradeoff = [
        item
        for item in recommendation.evidence["strategy_search"][
            "alternative_tradeoffs"
        ]
        if item["strategy_id"] == "kick_bass_dynamic_pocket_ducking"
    ][0]
    assert tradeoff["score_delta_vs_selected"] > 0.0
    _assert_proposal_denial(recommendation.to_operator_proposal())


def test_kick_bass_arrangement_overlap_holds_before_eq_or_ducking():
    recommendation = analyze_smart_dsp_strategy(
        _analysis(
            target_id="ch:2",
            source_role="bass",
            source_label="Bass DI",
            problem_hints=("masking", "arrangement_overlap"),
            masking_conflict_role="kick",
            masking_frequency_hz=74.0,
            masking_score=0.88,
            arrangement_overlap_score=0.84,
            level_plane_error_lu=0.3,
            true_peak_headroom_db=6.0,
        )
    )

    assert recommendation.status == "proposal_only"
    assert (
        recommendation.selected_strategy.strategy_id
        == "arrangement_aware_low_end_separation_hold"
    )
    assert recommendation.selected_strategy.method_family == "analysis_only"
    params = recommendation.selected_strategy.parameters
    assert params["dsp_change"] == "hold"
    assert params["arrangement_overlap_score"] == 0.84
    assert params["requires_arrangement_or_register_review"] is True
    duck = [
        candidate
        for candidate in recommendation.alternatives
        if candidate.strategy_id == "kick_bass_dynamic_pocket_ducking"
    ][0]
    assert "arrangement_overlap_should_precede_dynamic_ducking" in duck.blockers
    assert recommendation.to_operator_proposal()["live_apply_allowed"] is False


def test_arrangement_overlap_without_counterpart_role_holds_for_pair_context():
    recommendation = analyze_smart_dsp_strategy(
        _analysis(
            target_id="ch:2",
            source_role="bass_di",
            source_label="Bass DI",
            problem_hints=("arrangement_overlap",),
            masking_conflict_role="",
            masking_frequency_hz=72.0,
            masking_bandwidth_hz=48.0,
            masking_score=0.0,
            arrangement_overlap_score=0.86,
            level_plane_error_lu=0.0,
            true_peak_headroom_db=6.0,
        )
    )

    assert recommendation.status == "proposal_only"
    assert recommendation.selected_strategy.strategy_id == "arrangement_overlap_pair_context_hold"
    assert recommendation.selected_strategy.method_family == "analysis_only"
    assert recommendation.selected_strategy.problem == "masking"
    assert "arrangement_overlap" in recommendation.evidence["detected_problems"]
    assert "masking" in recommendation.evidence["detected_problems"]
    assert (
        recommendation.evidence["operator_explanation_sections"]["selection_policy"]
        == "arrangement_overlap_pair_context_priority"
    )
    params = recommendation.selected_strategy.parameters
    assert params["dsp_change"] == "hold"
    assert params["missing_masking_conflict_role"] is True
    assert params["required_counterpart_family"] == "kick"
    assert params["requires_counterpart_source_identity"] is True
    assert params["requires_pair_readback"] is True
    assert params["requires_arrangement_or_register_review"] is True
    proposal = recommendation.to_operator_proposal()
    assert proposal["live_apply_allowed"] is False
    assert (
        proposal["selected_wing_osc_summary"]["wing_osc_apply_parameter_values"]
        == {}
    )


def test_live_kick_bass_overlap_prefers_no_boost_eq_split_over_ducking():
    recommendation = analyze_smart_dsp_strategy(
        _analysis(
            target_id="ch:2",
            source_role="bass",
            mode="live",
            source_label="Bass DI",
            problem_hints=("masking",),
            muddiness_score=0.0,
            masking_conflict_role="kick",
            masking_frequency_hz=74.0,
            masking_score=0.92,
            true_peak_headroom_db=6.0,
            active_performance=True,
        )
    )

    assert recommendation.status == "proposal_only"
    assert recommendation.selected_strategy.strategy_id == "kick_bass_pocket_eq_split"
    assert recommendation.selected_strategy.parameters["linked_counter_boost_db"] == 0.0
    duck = [
        candidate
        for candidate in recommendation.alternatives
        if candidate.strategy_id == "kick_bass_dynamic_pocket_ducking"
    ][0]
    assert "aggressive_live_strategy_requires_operator_review" in duck.blockers


def test_soundcheck_bass_low_end_resonance_prefers_wing_native_deq_cut():
    recommendation = analyze_smart_dsp_strategy(
        _analysis(
            target_id="ch:2",
            source_role="bass",
            source_label="Bass DI",
            problem_hints=("low_end_resonance",),
            muddiness_score=0.78,
            low_end_excess_score=0.72,
            spectral_tilt_score=0.66,
            low_end_resonance_frequency_hz=118.0,
            low_end_resonance_bandwidth_hz=72.0,
            low_end_resonance_peak_dbfs=-19.6,
            low_end_resonance_sustain_ms=360.0,
            dynamics_variance_score=0.52,
            crest_factor_db=17.5,
            estimated_lufs=-23.0,
            level_plane_error_lu=0.2,
            true_peak_headroom_db=5.0,
        )
    )

    assert recommendation.status == "proposal_only"
    assert recommendation.selected_strategy.strategy_id == (
        "low_end_dynamic_resonance_control"
    )
    assert recommendation.selected_strategy.method_family == "dynamic_eq"
    params = recommendation.selected_strategy.parameters
    assert params["source_family"] == "bass"
    assert params["filter_type"] == "dynamic_peaking"
    assert params["frequency_hz"] == 118.0
    assert params["low_end_parameter_source"] == "measured_low_end_resonance"
    assert params["low_end_resonance_bandwidth_hz"] == 72.0
    assert params["static_gain_db"] == -0.4
    assert params["gain_db"] == -0.4
    assert params["max_gain_reduction_db"] == 3.9
    assert params["q"] == 1.37
    assert params["threshold_dbfs"] == -26.7
    assert params["attack_ms"] == 15.0
    assert params["release_ms"] == 215.0
    assert params["wing_dynamic_eq_model_osc_value"] == WING_DYNAMIC_EQ_GATE_MODEL
    assert params["wing_dynamic_eq_filter_osc_value"] == "BP"
    assert params["wing_dynamic_eq_gain_osc_value"] == -3.9
    assert params["wing_dynamic_eq_frequency_osc_value"] == 118.0
    assert params["wing_dynamic_eq_q_osc_value"] == 1.37
    assert recommendation.evidence["low_end_resonance_frequency_hz"] == 118.0
    assert recommendation.evidence["low_end_resonance_bandwidth_hz"] == 72.0
    alternatives = {candidate.strategy_id: candidate for candidate in recommendation.alternatives}
    assert {
        "low_end_level_plane_rebalance_before_eq",
        "hold_low_end_resonance_for_context",
    }.issubset(set(alternatives))
    assert "low_end_static_pocket_cut" in alternatives
    static = alternatives["low_end_static_pocket_cut"]
    assert static.parameters["gain_db"] == -2.8
    assert static.parameters["q"] == 1.21
    assert static.parameters["no_boost"] is True
    assert static.future_wing_readback_contract["wing_osc_applicability"][
        "native_osc_supported"
    ] is True
    assert recommendation.evidence["operator_explanation_sections"]["selection_policy"] == (
        "weighted_strategy_score"
    )
    tradeoffs = {
        row["strategy_id"]: row["why_not_selected"]
        for row in recommendation.evidence["strategy_search"]["alternative_tradeoffs"]
    }
    assert tradeoffs["low_end_static_pocket_cut"].startswith("lower_")
    readback = recommendation.selected_strategy.future_wing_readback_contract
    assert readback["wing_osc_applicability"]["native_osc_supported"] is True
    assert readback["wing_osc_applicability"]["native_osc_support_level"] == (
        "native_channel_gate_deq_dynamic_eq_osc_path_templates_declared"
    )
    assert readback["dynamic_eq_readback"][
        "wing_dynamic_eq_model_readback_must_equal"
    ] == WING_DYNAMIC_EQ_GATE_MODEL
    assert "eq_band_readback" not in readback
    assert "dynamics_readback" not in readback
    assert recommendation.to_operator_proposal()["live_apply_allowed"] is False


def test_soundcheck_bass_low_end_resonance_rebalances_hot_bus_before_source_deq():
    recommendation = analyze_smart_dsp_strategy(
        _analysis(
            target_id="ch:2",
            source_role="bass",
            source_label="Bass DI",
            problem_hints=("low_end_resonance",),
            muddiness_score=0.78,
            low_end_excess_score=0.72,
            spectral_tilt_score=0.66,
            low_end_resonance_frequency_hz=118.0,
            low_end_resonance_bandwidth_hz=72.0,
            low_end_resonance_peak_dbfs=-19.6,
            low_end_resonance_sustain_ms=360.0,
            dynamics_variance_score=0.52,
            crest_factor_db=17.5,
            estimated_lufs=-23.0,
            level_plane_error_lu=0.2,
            true_peak_headroom_db=5.0,
            summed_bus_evidence_available=True,
            summed_bus_role="low_end_bus",
            summed_bus_target_id="bus:3",
            summed_bus_true_peak_headroom_db=0.4,
            summed_bus_level_plane_error_lu=3.4,
        )
    )

    assert recommendation.status == "proposal_only"
    assert recommendation.selected_strategy.strategy_id == (
        "low_end_summed_bus_rebalance_before_resonance_eq"
    )
    assert recommendation.selected_strategy.method_family == "gain_fader"
    params = recommendation.selected_strategy.parameters
    assert params["target_scope"] == "summed_low_end_bus_before_resonance_eq"
    assert params["source_family"] == "bass"
    assert params["bus_delta_db"] == -2.5
    assert params["delta_db"] == -2.5
    assert params["gain_db"] == 0.0
    assert params["summed_bus_target_id"] == "bus:3"
    assert params["wing_osc_effective_target_id"] == "bus:3"
    assert params["wing_osc_original_target_id"] == "ch:2"
    assert params["wing_gain_target_override_reason"] == (
        "low_end_summed_bus_rebalance_before_resonance_eq"
    )
    assert params["requires_summed_bus_readback"] is True
    assert params["low_end_resonance_frequency_hz"] == 118.0
    assert recommendation.evidence["operator_explanation_sections"][
        "selection_policy"
    ] == "soundcheck_low_end_resonance_summed_bus_rebalance_priority"
    _assert_gain_delta_readback_contract(
        recommendation.selected_strategy,
        source="bus_delta_db",
        target_surface="bus",
        expected_delta=-2.5,
        path_prefix="/bus/{bus}",
    )
    _assert_selected_gain_delta_apply_values(recommendation, -2.5)
    alternatives = {
        candidate.strategy_id: candidate for candidate in recommendation.alternatives
    }
    assert "low_end_dynamic_resonance_control" in alternatives
    assert "low_end_static_pocket_cut" in alternatives
    tradeoffs = {
        row["strategy_id"]: row["why_not_selected"]
        for row in recommendation.evidence["strategy_search"]["alternative_tradeoffs"]
    }
    assert tradeoffs["low_end_dynamic_resonance_control"] == (
        "low_end_bus_level_priority:"
        "rebalance_summed_bus_before_source_resonance_eq"
    )
    assert recommendation.to_operator_proposal()["live_apply_allowed"] is False


def test_live_bass_low_end_resonance_prefers_static_no_boost_cut():
    recommendation = analyze_smart_dsp_strategy(
        _analysis(
            target_id="ch:2",
            source_role="bass",
            source_label="Bass DI",
            mode="live",
            problem_hints=("low_end_resonance",),
            muddiness_score=0.78,
            low_end_excess_score=0.72,
            spectral_tilt_score=0.66,
            low_end_resonance_frequency_hz=118.0,
            low_end_resonance_bandwidth_hz=72.0,
            low_end_resonance_peak_dbfs=-19.6,
            low_end_resonance_sustain_ms=360.0,
            dynamics_variance_score=0.52,
            crest_factor_db=17.5,
            estimated_lufs=-23.0,
            level_plane_error_lu=0.2,
            true_peak_headroom_db=5.0,
            active_performance=True,
        )
    )

    assert recommendation.status == "proposal_only"
    assert recommendation.selected_strategy.strategy_id == "low_end_static_pocket_cut"
    assert recommendation.selected_strategy.method_family == "eq"
    params = recommendation.selected_strategy.parameters
    assert params["source_family"] == "bass"
    assert params["gain_db"] < 0.0
    assert params["q"] == 1.21
    assert params["low_end_parameter_source"] == "measured_low_end_resonance"
    assert params["low_end_resonance_bandwidth_hz"] == 72.0
    assert params["no_boost"] is True
    dynamic_candidate = [
        candidate
        for candidate in recommendation.alternatives
        if candidate.strategy_id == "low_end_dynamic_resonance_control"
    ][0]
    assert (
        "live_dynamic_low_end_resonance_requires_operator_review"
        in dynamic_candidate.blockers
    )
    assert recommendation.to_operator_proposal()["live_apply_allowed"] is False


def test_live_bass_low_end_hot_bus_prefers_wing_native_bus_pullback_before_source_eq():
    recommendation = analyze_smart_dsp_strategy(
        _analysis(
            target_id="ch:2",
            source_role="bass",
            source_label="Bass DI",
            mode="live",
            problem_hints=("low_end_resonance",),
            muddiness_score=0.78,
            low_end_excess_score=0.72,
            spectral_tilt_score=0.66,
            low_end_resonance_frequency_hz=118.0,
            low_end_resonance_bandwidth_hz=72.0,
            low_end_resonance_peak_dbfs=-19.6,
            low_end_resonance_sustain_ms=360.0,
            dynamics_variance_score=0.52,
            crest_factor_db=17.5,
            estimated_lufs=-23.0,
            level_plane_error_lu=0.2,
            true_peak_headroom_db=5.0,
            active_performance=True,
            summed_bus_evidence_available=True,
            summed_bus_role="low_end_bus",
            summed_bus_target_id="bus:3",
            summed_bus_true_peak_headroom_db=0.4,
            summed_bus_level_plane_error_lu=3.4,
        )
    )

    assert recommendation.status == "proposal_only"
    assert recommendation.selected_strategy.strategy_id == (
        "low_end_summed_bus_rebalance_before_resonance_eq"
    )
    assert recommendation.selected_strategy.method_family == "gain_fader"
    params = recommendation.selected_strategy.parameters
    assert params["target_scope"] == "summed_low_end_bus_before_resonance_eq"
    assert params["source_family"] == "bass"
    assert params["bus_delta_db"] == -1.5
    assert params["delta_db"] == -1.5
    assert params["bounded_delta_db"] == 1.5
    assert params["gain_db"] == 0.0
    assert params["summed_bus_target_id"] == "bus:3"
    assert params["wing_osc_effective_target_id"] == "bus:3"
    assert params["wing_osc_original_target_id"] == "ch:2"
    assert params["wing_gain_target_override_reason"] == (
        "low_end_summed_bus_rebalance_before_resonance_eq"
    )
    assert params["requires_summed_bus_readback"] is True
    assert params["live_low_end_bus_pullback_only_proposal"] is True
    assert params["live_low_end_bus_pullback_max_delta_db"] == 1.5
    assert params["live_source_eq_deferred_until_bus_recheck"] is True
    assert (
        params["live_low_end_rebalance_live_apply_blocked_without_wing_proof"]
        is True
    )
    assert params["live_stepped_change_required"] is True
    assert params["live_readback_after_each_step_required"] is True
    assert params["live_max_single_step_db"] == 0.75
    assert recommendation.evidence["operator_explanation_sections"][
        "selection_policy"
    ] == "live_low_end_resonance_summed_bus_pullback_priority"
    _assert_gain_delta_readback_contract(
        recommendation.selected_strategy,
        source="bus_delta_db",
        target_surface="bus",
        expected_delta=-1.5,
        path_prefix="/bus/{bus}",
    )
    _assert_selected_gain_delta_apply_values(recommendation, -1.5)
    alternatives = {
        candidate.strategy_id: candidate for candidate in recommendation.alternatives
    }
    assert "low_end_static_pocket_cut" in alternatives
    assert "low_end_dynamic_resonance_control" in alternatives
    assert (
        "live_dynamic_low_end_resonance_requires_operator_review"
        in alternatives["low_end_dynamic_resonance_control"].blockers
    )
    tradeoffs = {
        row["strategy_id"]: row["why_not_selected"]
        for row in recommendation.evidence["strategy_search"]["alternative_tradeoffs"]
    }
    assert tradeoffs["low_end_static_pocket_cut"] == (
        "low_end_bus_level_priority:"
        "rebalance_summed_bus_before_source_resonance_eq"
    )
    proposal = recommendation.to_operator_proposal()
    assert proposal["live_apply_allowed"] is False
    assert proposal["live_authority_granted"] is False
    assert proposal["live_write"] is False


def test_kick_bass_level_plane_skew_blocks_dynamic_ducking_first():
    recommendation = analyze_smart_dsp_strategy(
        _analysis(
            target_id="ch:2",
            source_role="bass",
            source_label="Bass DI",
            problem_hints=("masking", "level_plane_balance"),
            muddiness_score=0.0,
            masking_conflict_role="kick",
            masking_frequency_hz=74.0,
            masking_score=0.92,
            level_plane_error_lu=4.2,
            true_peak_headroom_db=6.0,
        )
    )

    assert recommendation.status == "proposal_only"
    assert recommendation.selected_strategy.strategy_id == "level_plane_gain_down"
    duck = [
        candidate
        for candidate in recommendation.alternatives
        if candidate.strategy_id == "kick_bass_dynamic_pocket_ducking"
    ][0]
    assert "level_plane_rebalance_should_precede_dynamic_ducking" in duck.blockers
    assert "wing_dynamic_eq_value_mapping_required" in duck.blockers


def test_noisy_tom_tail_risk_selects_gentle_expander_over_hard_gate():
    recommendation = analyze_smart_dsp_strategy(
        _analysis(
            target_id="ch:8",
            source_role="floor_tom",
            problem_hints=("noisy_drum",),
            muddiness_score=0.0,
            noise_floor_dbfs=-49.0,
            bleed_risk=0.82,
            tail_risk=0.78,
            chatter_risk=0.61,
        )
    )

    assert recommendation.status == "proposal_only"
    assert recommendation.selected_strategy.strategy_id == "gentle_expander_bleed_control"
    assert recommendation.selected_strategy.method_family == "expander"
    hard_gate = [
        candidate
        for candidate in recommendation.alternatives
        if candidate.strategy_id == "fast_drum_gate"
    ][0]
    assert "gate_tail_or_chatter_risk_high" in hard_gate.blockers
    assert recommendation.selected_strategy.parameters["range_db"] == 8.0
    assert recommendation.selected_strategy.parameters["ratio"] == "1:2"
    assert (
        recommendation.selected_strategy.parameters["post_gate_hysteresis_review_db"]
        == 3.0
    )
    readback = recommendation.selected_strategy.future_wing_readback_contract
    assert readback["wing_osc_applicability"]["native_osc_supported"] is True
    assert "/ch/{channel}/gate/thr21" in readback["gate_expander_readback"][
        "wing_osc_path_templates"
    ]
    assert "/ch/{channel}/gate/thr" not in readback["gate_expander_readback"][
        "wing_osc_path_templates"
    ]
    assert recommendation.selected_strategy.parameters[
        "wing_gate_threshold_osc_path_suffix"
    ] == "gate/thr21"
    assert recommendation.selected_strategy.parameters[
        "wing_gate_model_readback_must_equal"
    ] == WING_GATE_REQUIRED_MODEL
    assert recommendation.selected_strategy.parameters[
        "wing_gate_range_osc_path_suffix"
    ] == "gate/range"
    assert recommendation.selected_strategy.parameters[
        "wing_gate_ratio_osc_value"
    ] == "1:2"
    assert recommendation.selected_strategy.parameters[
        "expander_parameter_source"
    ] == "risk_score_fallback"


def test_measured_tail_risk_tom_expander_uses_hit_tail_parameters():
    recommendation = analyze_smart_dsp_strategy(
        _analysis(
            target_id="ch:8",
            source_role="floor_tom",
            problem_hints=("noisy_drum",),
            muddiness_score=0.0,
            noise_floor_dbfs=-50.0,
            bleed_risk=0.78,
            tail_risk=0.64,
            chatter_risk=0.42,
            transient_strength=0.62,
            transient_density=2.1,
            gate_hit_peak_dbfs=-14.0,
            gate_hit_to_bleed_margin_db=18.0,
            gate_tail_decay_ms=190.0,
        )
    )

    assert recommendation.status == "proposal_only"
    assert recommendation.selected_strategy.strategy_id == "gentle_expander_bleed_control"
    assert recommendation.selected_strategy.method_family == "expander"
    params = recommendation.selected_strategy.parameters
    assert params["expander_parameter_source"] == "measured_hit_tail_bleed_expander"
    assert params["threshold_dbfs"] == -43.3
    assert params["range_db"] == 10.0
    assert params["hold_ms"] == 123.5
    assert params["release_ms"] == 374.5
    assert params["ratio"] == "1:2"
    assert params["post_gate_hysteresis_review_db"] == 2.9
    assert params["gate_hit_peak_dbfs"] == -14.0
    assert params["gate_hit_to_bleed_margin_db"] == 18.0
    assert params["gate_tail_decay_ms"] == 190.0
    assert params["soundcheck_gate_expander_bounded_proposal"] is True
    assert params["soundcheck_gate_expander_native_gate_osc_only_required"] is True
    assert (
        params["soundcheck_gate_expander_max_range_db"]
        == SOUNDCHECK_GATE_EXPANDER_MAX_RANGE_DB
    )
    assert (
        params["soundcheck_gate_expander_max_hysteresis_review_db"]
        == SOUNDCHECK_GATE_EXPANDER_MAX_HYSTERESIS_REVIEW_DB
    )
    assert (
        params["soundcheck_gate_expander_min_hit_bleed_margin_db"]
        == SOUNDCHECK_GATE_EXPANDER_MIN_HIT_BLEED_MARGIN_DB
    )
    assert params["soundcheck_gate_expander_hysteresis_review_only"] is True
    assert (
        params["soundcheck_gate_expander_hysteresis_no_verified_wing_osc_path"]
        is True
    )
    assert params["soundcheck_gate_expander_gate_model_readback_required"] is True
    assert params["soundcheck_gate_expander_gate_parameters_readback_required"] is True
    assert params["soundcheck_gate_expander_sidechain_readback_required"] is True
    assert params["soundcheck_gate_expander_audio_recheck_required"] is True
    assert params["soundcheck_gate_expander_tail_chatter_recheck_required"] is True
    assert params["soundcheck_gate_expander_operator_confirmation_required"] is True
    assert params["soundcheck_gate_expander_readback_required"] is True
    assert params["soundcheck_gate_expander_rollback_ready"] is True
    assert (
        params["soundcheck_gate_expander_live_promotion_status"]
        == "blocked_until_real_wing_gate_readback_clean_hit_bleed_tail_chatter_recheck_and_operator_approval"
    )
    assert params["soundcheck_gate_expander_live_apply_blocked_without_wing_proof"] is True
    assert params["soundcheck_gate_expander_clean_hit_bleed_margin_db"] == 18.0
    assert (
        params["soundcheck_gate_expander_clean_hit_bleed_evidence_status"]
        == "measured_margin_ready"
    )
    assert (
        params["soundcheck_gate_expander_wing_apply_parameter_scope"]
        == "gate_on_threshold_range_timing_accent_ratio_sidechain_only"
    )
    assert (
        params["soundcheck_expander_allows_risk_score_fallback_with_operator_recheck"]
        is True
    )
    hard_gate = [
        candidate
        for candidate in recommendation.alternatives
        if candidate.strategy_id == "fast_drum_gate"
    ][0]
    assert "gate_tail_or_chatter_risk_high" in hard_gate.blockers
    clean_gate = [
        candidate
        for candidate in recommendation.alternatives
        if candidate.strategy_id == "bleed_aware_clean_hit_gate"
    ][0]
    assert "clean_hit_gate_requires_low_tail_chatter_and_strong_hits" in (
        clean_gate.blockers
    )
    _assert_selected_gate_apply_values(
        recommendation,
        {
            "wing_gate_on_osc_value": WING_GATE_ON_MAX,
            "wing_gate_threshold_osc_value": -43.3,
            "wing_gate_range_osc_value": 10.0,
            "wing_gate_attack_osc_value": 4.0,
            "wing_gate_hold_osc_value": 123.5,
            "wing_gate_release_osc_value": 374.5,
            "wing_gate_ratio_osc_value": "1:2",
            "wing_gate_sidechain_type_osc_value": "BP",
            "wing_gate_sidechain_frequency_osc_value": 160.0,
            "wing_gate_sidechain_q_osc_value": 2.99,
            "wing_gate_sidechain_src_osc_value": "SELF",
            "wing_gate_sidechain_tap_osc_value": "IN",
        },
    )
    gate_readback = recommendation.selected_strategy.future_wing_readback_contract[
        "gate_expander_readback"
    ]
    assert gate_readback["soundcheck_gate_expander_bounded_proposal"] is True
    assert (
        gate_readback["soundcheck_gate_expander_native_gate_osc_only_required"]
        is True
    )
    assert gate_readback["soundcheck_gate_expander_gate_model_readback_required"] is True
    assert (
        gate_readback["soundcheck_gate_expander_live_apply_blocked_without_wing_proof"]
        is True
    )


def test_clean_tom_hits_select_bleed_aware_gate_before_expander():
    recommendation = analyze_smart_dsp_strategy(
        _analysis(
            target_id="ch:8",
            source_role="floor_tom",
            problem_hints=("noisy_drum",),
            muddiness_score=0.0,
            noise_floor_dbfs=-55.0,
            bleed_risk=0.55,
            tail_risk=0.22,
            chatter_risk=0.12,
            transient_strength=0.72,
            transient_density=1.8,
            gate_hit_peak_dbfs=-15.0,
            gate_hit_to_bleed_margin_db=40.0,
            gate_tail_decay_ms=80.0,
        )
    )

    assert recommendation.status == "proposal_only"
    assert recommendation.selected_strategy.strategy_id == "bleed_aware_clean_hit_gate"
    assert recommendation.selected_strategy.method_family == "gate"
    params = recommendation.selected_strategy.parameters
    assert params["clean_hit_gate_ready"] is True
    assert params["threshold_dbfs"] > params["noise_floor_dbfs"]
    assert params["gate_parameter_source"] == "measured_hit_tail"
    assert params["gate_hit_peak_dbfs"] == -15.0
    assert params["gate_hit_to_bleed_margin_db"] == 40.0
    assert params["gate_tail_decay_ms"] == 80.0
    assert params["hold_ms"] >= 60.0
    assert params["release_ms"] > params["hold_ms"]
    assert params["range_db"] > 0.0
    assert params["accent_percent"] == 51.0
    assert params["accent_parameter_source"] == "measured_hit_transient_bleed_margin"
    assert params["ratio"] == "GATE"
    assert 3.0 <= params["post_gate_hysteresis_review_db"] <= 5.5
    assert params["requires_clean_hit_readback"] is True
    assert params["soundcheck_gate_expander_bounded_proposal"] is True
    assert params["soundcheck_gate_expander_native_gate_osc_only_required"] is True
    assert params["soundcheck_gate_expander_max_range_db"] == (
        SOUNDCHECK_GATE_EXPANDER_MAX_RANGE_DB
    )
    assert params["soundcheck_gate_expander_max_hysteresis_review_db"] == (
        SOUNDCHECK_GATE_EXPANDER_MAX_HYSTERESIS_REVIEW_DB
    )
    assert params["soundcheck_gate_expander_min_hit_bleed_margin_db"] == (
        SOUNDCHECK_GATE_EXPANDER_MIN_HIT_BLEED_MARGIN_DB
    )
    assert params["soundcheck_gate_expander_clean_hit_bleed_margin_db"] == 40.0
    assert params["soundcheck_gate_expander_clean_hit_bleed_evidence_status"] == (
        "measured_margin_ready"
    )
    assert params["soundcheck_gate_requires_measured_clean_hit_margin"] is True
    assert params["soundcheck_gate_expander_live_apply_blocked_without_wing_proof"] is True
    readback = recommendation.selected_strategy.future_wing_readback_contract
    assert readback["wing_osc_applicability"]["native_osc_supported"] is True
    assert "/ch/{channel}/gate/thr21" in readback["gate_expander_readback"][
        "wing_osc_path_templates"
    ]
    assert "gate/acc" in readback["gate_expander_readback"]["read_after_paths"]
    assert "gate/mdl" in readback["gate_expander_readback"]["read_after_paths"]
    assert readback["gate_expander_readback"]["gate_model_readback_must_equal"] == (
        WING_GATE_REQUIRED_MODEL
    )
    assert params["wing_gate_ratio_osc_value"] == "GATE"
    assert params["wing_gate_threshold_osc_path_suffix"] == "gate/thr21"
    assert params["wing_gate_accent_osc_path_suffix"] == "gate/acc"
    assert params["wing_gate_accent_osc_value"] == 51.0
    assert params["gate_sidechain_filter_type"] == "BP"
    assert params["gate_sidechain_frequency_hz"] == 175.4
    assert params["gate_sidechain_q"] == 2.75
    assert params["gate_sidechain_source"] == "SELF"
    assert params["gate_sidechain_tap"] == "IN"
    assert params["gate_sidechain_parameter_source"] == (
        "measured_clean_hit_detector_band"
    )
    assert params["gate_sidechain_evidence_frequency_hz"] == 0.0
    assert params["gate_sidechain_hit_to_bleed_margin_db"] == 40.0
    assert params["gate_sidechain_tail_decay_ms"] == 80.0
    assert params["gate_sidechain_transient_strength"] == 0.72
    assert params["wing_gate_sidechain_type_osc_path_suffix"] == "gatesc/type"
    assert params["wing_gate_sidechain_type_osc_value"] == "BP"
    assert params["wing_gate_sidechain_frequency_osc_path_suffix"] == "gatesc/f"
    assert params["wing_gate_sidechain_frequency_osc_value"] == 175.4
    assert params["wing_gate_sidechain_q_osc_path_suffix"] == "gatesc/q"
    assert params["wing_gate_sidechain_q_osc_value"] == 2.75
    assert params["wing_gate_sidechain_src_osc_path_suffix"] == "gatesc/src"
    assert params["wing_gate_sidechain_src_osc_value"] == "SELF"
    assert params["wing_gate_sidechain_tap_osc_path_suffix"] == "gatesc/tap"
    assert params["wing_gate_sidechain_tap_osc_value"] == "IN"
    assert "wing_gate_accent_osc_value" in readback["wing_osc_applicability"][
        "actionable_parameter_keys"
    ]
    assert "wing_gate_sidechain_frequency_osc_value" in readback[
        "wing_osc_applicability"
    ]["actionable_parameter_keys"]
    assert readback["gate_expander_readback"]["wing_gate_accent_osc_value"] == 51.0
    assert readback["gate_expander_readback"][
        "wing_gate_sidechain_mapping_required"
    ] is True
    assert readback["gate_expander_readback"][
        "wing_gate_sidechain_frequency_osc_value"
    ] == 175.4
    assert readback["gate_expander_readback"][
        "wing_gate_sidechain_q_osc_value"
    ] == 2.75
    assert "gatesc/type" in readback["gate_expander_readback"]["read_after_paths"]
    assert readback["gate_expander_readback"][
        "wing_gate_accent_readback_must_match_osc_value"
    ] is True
    assert readback["gate_expander_readback"][
        "soundcheck_gate_expander_bounded_proposal"
    ] is True
    assert readback["gate_expander_readback"][
        "soundcheck_gate_expander_live_apply_blocked_without_wing_proof"
    ] is True
    assert readback["gate_expander_readback"][
        "soundcheck_gate_requires_measured_clean_hit_margin"
    ] is True
    _assert_selected_write_key_contract(
        params=params,
        readback=readback["gate_expander_readback"],
        prefix="wing_gate",
        selected_write_keys=list(WING_GATE_WRITE_PARAMETER_KEYS),
        write_parameter_keys=WING_GATE_WRITE_PARAMETER_KEYS,
        selection_policy=WING_GATE_WRITE_KEY_SELECTION_POLICY,
    )
    proposal = recommendation.to_operator_proposal()
    expected_apply_values = {
        "wing_gate_on_osc_value": WING_GATE_ON_MAX,
        "wing_gate_threshold_osc_value": -45.1,
        "wing_gate_range_osc_value": 10.9,
        "wing_gate_attack_osc_value": 2.0,
        "wing_gate_hold_osc_value": 60.0,
        "wing_gate_release_osc_value": 133.0,
        "wing_gate_accent_osc_value": 51.0,
        "wing_gate_ratio_osc_value": "GATE",
        "wing_gate_sidechain_type_osc_value": "BP",
        "wing_gate_sidechain_frequency_osc_value": 175.4,
        "wing_gate_sidechain_q_osc_value": 2.75,
        "wing_gate_sidechain_src_osc_value": "SELF",
        "wing_gate_sidechain_tap_osc_value": "IN",
    }
    _assert_selected_gate_apply_values(recommendation, expected_apply_values)
    assert "post_gate_hysteresis_review_db" not in expected_apply_values
    assert proposal["selected_wing_osc_summary"][
        "wing_osc_review_only_parameter_values"
    ] == {
        "post_gate_hysteresis_review_db": params["post_gate_hysteresis_review_db"]
    }
    expander = [
        candidate
        for candidate in recommendation.alternatives
        if candidate.strategy_id == "gentle_expander_bleed_control"
    ][0]
    assert expander.total_score < recommendation.selected_strategy.total_score
    assert recommendation.to_operator_proposal()["live_apply_allowed"] is False


def test_soundcheck_operator_owned_gate_slot_holds_before_gate_or_expander():
    recommendation = analyze_smart_dsp_strategy(
        _analysis(
            target_id="ch:8",
            source_role="floor_tom",
            problem_hints=("noisy_drum",),
            muddiness_score=0.0,
            noise_floor_dbfs=-55.0,
            bleed_risk=0.55,
            tail_risk=0.22,
            chatter_risk=0.08,
            transient_strength=0.82,
            transient_density=1.1,
            gate_hit_peak_dbfs=-18.0,
            gate_hit_to_bleed_margin_db=22.0,
            gate_tail_decay_ms=145.0,
            current_wing_gate_model="GATE",
            current_wing_gate_enabled=True,
            current_wing_gate_slot_readback_available=True,
            current_wing_gate_slot_operator_owned=True,
        )
    )

    assert recommendation.status == "proposal_only"
    assert recommendation.selected_strategy.strategy_id == (
        "hold_soundcheck_operator_owned_gate_slot_for_review"
    )
    assert recommendation.selected_strategy.method_family == "analysis_only"
    assert recommendation.evidence["operator_explanation_sections"][
        "selection_policy"
    ] == "soundcheck_operator_owned_gate_slot_review_priority"
    params = recommendation.selected_strategy.parameters
    assert params["dsp_change"] == "hold"
    assert params["soundcheck_operator_owned_gate_slot_hold_proposal"] is True
    assert params["soundcheck_operator_owned_gate_slot_target_surface"] == "channel"
    assert params["soundcheck_operator_owned_gate_slot_previous_model_readback_required"] is True
    assert (
        params[
            "soundcheck_operator_owned_gate_slot_previous_gate_parameters_readback_required"
        ]
        is True
    )
    assert params["soundcheck_operator_owned_gate_slot_sidechain_readback_required"] is True
    assert params["soundcheck_operator_owned_gate_slot_audio_recheck_required"] is True
    assert params["soundcheck_operator_owned_gate_slot_rollback_required"] is True
    assert (
        params[
            "soundcheck_operator_owned_gate_slot_no_gate_or_expander_without_operator_review"
        ]
        is True
    )
    assert params["blocked_gate_method_family"] == "gate"
    assert params["blocked_gate_threshold_dbfs"] == -46.9
    assert params["blocked_gate_range_db"] == 11.6
    assert params["blocked_gate_sidechain_frequency_hz"] == 160.0
    assert params["blocked_gate_operator_owned_blockers"] == [
        WING_GATE_SLOT_OPERATOR_OWNED_MODEL_BLOCKER
    ]
    assert params["wing_gate_slot_current_model"] == "GATE"
    assert params["wing_gate_slot_current_enabled"] is True
    assert params["wing_gate_slot_current_operator_owned"] is True
    assert params["wing_gate_slot_model_readback_available"] is True
    assert params["wing_gate_slot_operator_owned_model_edit_requires_review"] is True
    assert (
        params["wing_gate_slot_operator_owned_model_blocker"]
        == WING_GATE_SLOT_OPERATOR_OWNED_MODEL_BLOCKER
    )
    assert params["required_context"] == (
        "operator_review_current_wing_gate_slot_owner_model_parameters_sidechain_readback_rollback"
    )
    assert params["forbidden_until_reviewed"] == (
        "soundcheck_gate_expander_gate_slot_queue_import"
    )
    readback = recommendation.selected_strategy.future_wing_readback_contract[
        "soundcheck_operator_owned_gate_slot_review_readback"
    ]
    assert readback["read_only"] is True
    assert readback["soundcheck_operator_owned_gate_slot_current_model"] == "GATE"
    assert readback[
        "soundcheck_operator_owned_gate_slot_current_operator_owned"
    ] is True
    assert readback[
        "soundcheck_operator_owned_gate_slot_operator_owned_blockers"
    ] == [WING_GATE_SLOT_OPERATOR_OWNED_MODEL_BLOCKER]
    assert "gate/mdl" in readback["read_before_paths"]
    assert "gate/on" in readback["read_before_paths"]
    assert "gate/thr21" in readback["read_before_paths"]
    assert "gatesc/type" in readback["read_before_paths"]
    assert "/ch/{channel}/gate/thr21" in readback["wing_osc_path_templates"]
    assert readback["allowed_live_write_scope_before_wing_proof"] == []

    gate = [
        candidate
        for candidate in recommendation.alternatives
        if candidate.strategy_id == "bleed_aware_clean_hit_gate"
    ][0]
    expander = [
        candidate
        for candidate in recommendation.alternatives
        if candidate.strategy_id == "gentle_expander_bleed_control"
    ][0]
    assert WING_GATE_SLOT_OPERATOR_OWNED_MODEL_BLOCKER in gate.blockers
    assert WING_GATE_SLOT_OPERATOR_OWNED_MODEL_BLOCKER in expander.blockers
    tradeoffs = {
        row["strategy_id"]: row["why_not_selected"]
        for row in recommendation.evidence["strategy_search"]["alternative_tradeoffs"]
    }
    assert tradeoffs["bleed_aware_clean_hit_gate"] == (
        "soundcheck_operator_owned_gate_slot_review_priority:"
        "manual_gate_slot_owner_review_precedes_gate_or_expander_change"
    )
    assert tradeoffs["gentle_expander_bleed_control"] == (
        "soundcheck_operator_owned_gate_slot_review_priority:"
        "manual_gate_slot_owner_review_precedes_gate_or_expander_change"
    )
    _assert_selected_has_no_apply_values(recommendation)


def test_low_hit_bleed_margin_tom_holds_before_gate_or_expander():
    recommendation = analyze_smart_dsp_strategy(
        _analysis(
            target_id="ch:8",
            source_role="floor_tom",
            problem_hints=("noisy_drum",),
            muddiness_score=0.0,
            noise_floor_dbfs=-50.0,
            bleed_risk=0.55,
            tail_risk=0.2,
            chatter_risk=0.1,
            transient_strength=0.85,
            transient_density=1.5,
            gate_hit_peak_dbfs=-14.0,
            gate_hit_to_bleed_margin_db=3.0,
            gate_tail_decay_ms=190.0,
        )
    )

    assert recommendation.status == "analysis_only"
    assert (
        recommendation.selected_strategy.strategy_id
        == "hold_noisy_drum_for_clean_hit_bleed_evidence"
    )
    assert recommendation.selected_strategy.method_family == "analysis_only"
    assert recommendation.evidence["operator_explanation_sections"][
        "selection_policy"
    ] == "soundcheck_noisy_drum_clean_hit_bleed_evidence_priority"
    params = recommendation.selected_strategy.parameters
    assert params["dsp_change"] == "hold"
    assert params["gate_hit_to_bleed_margin_db"] == 3.0
    assert params["minimum_hit_to_bleed_margin_db"] == 6.0
    assert params["forbidden_without_clean_hit_separation"] == (
        "gate_expander_hard_mute"
    )
    assert "clean_hit_bleed_margin_required_before_drum_dynamics" in (
        recommendation.selected_strategy.blockers
    )

    gate = [
        candidate
        for candidate in recommendation.alternatives
        if candidate.strategy_id == "bleed_aware_clean_hit_gate"
    ][0]
    expander = [
        candidate
        for candidate in recommendation.alternatives
        if candidate.strategy_id == "gentle_expander_bleed_control"
    ][0]
    assert "hit_bleed_margin_too_low_for_gate" in gate.blockers
    assert "hit_bleed_margin_too_low_for_expander" in expander.blockers
    tradeoff = [
        item
        for item in recommendation.evidence["strategy_search"][
            "alternative_tradeoffs"
        ]
        if item["strategy_id"] == "bleed_aware_clean_hit_gate"
    ][0]
    assert tradeoff["why_not_selected"] == (
        "soundcheck_noisy_drum_clean_hit_priority:"
        "separate_hits_from_bleed_before_gate_or_expander"
    )
    _assert_proposal_denial(recommendation.to_operator_proposal())


def test_extreme_noisy_tom_prefers_manual_mute_over_unstable_gate_or_expander():
    recommendation = analyze_smart_dsp_strategy(
        _analysis(
            target_id="ch:8",
            source_role="floor_tom",
            problem_hints=("noisy_drum",),
            muddiness_score=0.0,
            noise_floor_dbfs=-44.0,
            bleed_risk=0.9,
            tail_risk=0.86,
            chatter_risk=0.82,
        )
    )

    assert recommendation.status == "proposal_only"
    assert recommendation.selected_strategy.strategy_id == "manual_tom_mute_between_hits"
    assert recommendation.selected_strategy.method_family == "mute"
    params = recommendation.selected_strategy.parameters
    assert params["proposed_muted"] is True
    assert params["manual_mute_scope"] == "between_tom_hits_only"
    assert params["manual_mute_authority_preserved"] is True
    assert params["rollback_requires_previous_mute_readback"] is True
    _assert_mute_readback_contract(
        recommendation.selected_strategy,
        target_surface="channel",
        path_prefix="/ch/{channel}",
    )
    _assert_selected_mute_apply_values(recommendation, WING_MUTE_ON_VALUE)
    assert recommendation.to_operator_proposal()["can_apply_live"] is False

    expander = [
        candidate
        for candidate in recommendation.alternatives
        if candidate.strategy_id == "gentle_expander_bleed_control"
    ][0]
    hard_gate = [
        candidate
        for candidate in recommendation.alternatives
        if candidate.strategy_id == "fast_drum_gate"
    ][0]
    hold = [
        candidate
        for candidate in recommendation.alternatives
        if candidate.strategy_id == "block_gate_until_clean_hit_detection"
    ][0]
    assert "extreme_tail_chatter_prefers_manual_mute_review" in expander.blockers
    assert "gate_tail_or_chatter_risk_high" in hard_gate.blockers
    assert "clean_hit_detection_required_before_gate" in hold.blockers


def test_operator_owned_mute_blocks_soundcheck_manual_mute_apply():
    recommendation = analyze_smart_dsp_strategy(
        _analysis(
            target_id="ch:8",
            source_role="floor_tom",
            problem_hints=("noisy_drum",),
            muddiness_score=0.0,
            noise_floor_dbfs=-44.0,
            bleed_risk=0.9,
            tail_risk=0.86,
            chatter_risk=0.82,
            current_wing_mute_operator_owned=True,
        )
    )

    assert recommendation.status == "analysis_only"
    assert recommendation.selected_strategy.strategy_id == (
        "hold_soundcheck_operator_owned_mute_for_review"
    )
    assert recommendation.selected_strategy.method_family == "analysis_only"
    assert recommendation.selected_strategy.blockers == (
        "soundcheck_operator_owned_mute_requires_operator_review",
    )
    assert recommendation.evidence["operator_explanation_sections"][
        "selection_policy"
    ] == "soundcheck_operator_owned_mute_review_priority"
    params = recommendation.selected_strategy.parameters
    assert params["dsp_change"] == "hold"
    assert params["reason"] == "soundcheck_operator_owned_mute_requires_operator_review"
    assert params["blocked_mute_strategy_id"] == "manual_tom_mute_between_hits"
    assert params["blocked_mute_proposed_muted"] is True
    assert params["blocked_mute_scope"] == "between_tom_hits_only"
    assert params["wing_mute_current_operator_owned"] is True
    assert params["wing_mute_operator_owned_edit_requires_review"] is True
    assert params["wing_mute_operator_owned_blocker"] == WING_MUTE_OPERATOR_OWNED_BLOCKER
    assert params["soundcheck_operator_owned_mute_hold_proposal"] is True
    assert params["soundcheck_operator_owned_mute_target_surface"] == "channel"
    assert params["soundcheck_operator_owned_mute_selected_write_keys"] == []
    assert params["soundcheck_operator_owned_mute_forbidden_write_keys"] == [
        "wing_mute_osc_value"
    ]
    assert "wing_mute_osc_value" not in params

    manual_mute = [
        candidate
        for candidate in recommendation.alternatives
        if candidate.strategy_id == "manual_tom_mute_between_hits"
    ][0]
    assert manual_mute.parameters["wing_mute_osc_value"] == 1
    assert WING_MUTE_OPERATOR_OWNED_BLOCKER in manual_mute.blockers
    tradeoffs = {
        row["strategy_id"]: row["why_not_selected"]
        for row in recommendation.evidence["strategy_search"][
            "alternative_tradeoffs"
        ]
    }
    assert tradeoffs["manual_tom_mute_between_hits"] == (
        "soundcheck_operator_owned_mute_review_priority:"
        "manual_mute_owner_review_precedes_soundcheck_mute_change"
    )

    mute_readback = recommendation.selected_strategy.future_wing_readback_contract[
        "operator_owned_mute_readback"
    ]
    assert mute_readback["read_only"] is True
    assert mute_readback["wing_osc_path_templates"] == ["/ch/{channel}/mute"]
    assert mute_readback["read_before_paths"] == ["mute"]
    assert mute_readback["read_after_paths"] == ["mute"]
    assert mute_readback["blocked_mute_strategy_id"] == "manual_tom_mute_between_hits"
    assert mute_readback["blocked_mute_proposed_muted"] is True
    assert mute_readback["wing_mute_forbidden_apply_fields"] == [
        "wing_mute_osc_value"
    ]
    assert mute_readback["allowed_live_write_scope_before_wing_proof"] == []
    assert mute_readback["wing_mute_current_operator_owned"] is True
    assert mute_readback["wing_mute_operator_owned_edit_requires_review"] is True
    assert (
        mute_readback["wing_mute_operator_owned_blocker"]
        == WING_MUTE_OPERATOR_OWNED_BLOCKER
    )

    proposal = recommendation.to_operator_proposal()
    assert proposal["selected_wing_osc_summary"][
        "wing_osc_candidate_apply_status"
    ] == "analysis_only_no_console_write"
    assert proposal["selected_wing_osc_summary"][
        "wing_osc_apply_parameter_values"
    ] == {}
    assert proposal["can_apply_live"] is False
    assert proposal["live_authority_granted"] is False
    assert proposal["live_write"] is False


def test_live_noisy_tom_holds_before_gate_or_expander():
    recommendation = analyze_smart_dsp_strategy(
        _analysis(
            target_id="ch:8",
            source_role="floor_tom",
            mode="live",
            problem_hints=("noisy_drum",),
            muddiness_score=0.0,
            noise_floor_dbfs=-55.0,
            bleed_risk=0.55,
            tail_risk=0.22,
            chatter_risk=0.12,
            transient_strength=0.72,
            transient_density=1.8,
            active_performance=True,
        )
    )

    assert recommendation.status == "proposal_only"
    assert recommendation.selected_strategy.strategy_id == (
        "hold_live_noisy_drum_processing_for_operator"
    )
    assert recommendation.selected_strategy.method_family == "analysis_only"
    assert recommendation.evidence["operator_explanation_sections"][
        "selection_policy"
    ] == "live_noisy_drum_review_priority"
    params = recommendation.selected_strategy.parameters
    assert params["dsp_change"] == "hold"
    assert params["live_noisy_drum_hold_proposal"] is True
    assert params["live_noisy_drum_target_surface"] == "channel"
    assert params["live_noisy_drum_source_identity_required"] is True
    assert params["live_noisy_drum_clean_hit_detection_required"] is True
    assert params["live_noisy_drum_tail_map_required"] is True
    assert params["live_noisy_drum_bleed_map_required"] is True
    assert params["live_noisy_drum_previous_gate_expander_readback_required"] is True
    assert params["live_noisy_drum_previous_mute_readback_required"] is True
    assert params["live_noisy_drum_gate_model_readback_required"] is True
    assert params["live_noisy_drum_gate_parameters_readback_required"] is True
    assert params["live_noisy_drum_gate_sidechain_readback_required"] is True
    assert params["live_noisy_drum_mute_readback_required"] is True
    assert params["live_noisy_drum_audio_recheck_required"] is True
    assert params["live_noisy_drum_tail_chatter_recheck_required"] is True
    assert params["live_noisy_drum_rollback_ready"] is True
    assert (
        params["live_noisy_drum_no_gate_expander_mute_without_operator_wing_proof"]
        is True
    )
    assert params["forbidden_without_wing_proof"] == "gate_expander_manual_mute"
    assert "wing_gate_on_osc_value" not in params
    assert "wing_mute_osc_value" not in params
    noisy_drum_readback = recommendation.selected_strategy.future_wing_readback_contract[
        "live_noisy_drum_hold_readback"
    ]
    assert noisy_drum_readback["read_only"] is True
    assert noisy_drum_readback["live_noisy_drum_target_surface"] == "channel"
    assert noisy_drum_readback["live_noisy_drum_gate_model_readback_required"] is True
    assert noisy_drum_readback["live_noisy_drum_mute_readback_required"] is True
    assert (
        noisy_drum_readback[
            "live_noisy_drum_no_gate_expander_mute_without_operator_wing_proof"
        ]
        is True
    )
    assert noisy_drum_readback["allowed_live_write_scope_before_wing_proof"] == []
    assert noisy_drum_readback["wing_gate_threshold_osc_path_suffix"] == "gate/thr21"
    assert "/ch/{channel}/gate/thr21" in noisy_drum_readback[
        "wing_osc_path_templates"
    ]
    assert "/ch/{channel}/mute" in noisy_drum_readback["wing_osc_path_templates"]
    gate = [
        candidate
        for candidate in recommendation.alternatives
        if candidate.strategy_id == "bleed_aware_clean_hit_gate"
    ][0]
    expander = [
        candidate
        for candidate in recommendation.alternatives
        if candidate.strategy_id == "gentle_expander_bleed_control"
    ][0]
    assert "live_gate_change_requires_operator_review_and_wing_proof" in gate.blockers
    assert "live_expander_change_requires_operator_review_and_wing_proof" in (
        expander.blockers
    )
    tradeoff = [
        item
        for item in recommendation.evidence["strategy_search"][
            "alternative_tradeoffs"
        ]
        if item["strategy_id"] == "bleed_aware_clean_hit_gate"
    ][0]
    assert tradeoff["why_not_selected"] == (
        "live_noisy_drum_review_priority:gate_expander_or_mute_can_cut_hits"
    )
    _assert_proposal_denial(recommendation.to_operator_proposal())


def test_live_extreme_noisy_tom_holds_before_manual_mute_review():
    recommendation = analyze_smart_dsp_strategy(
        _analysis(
            target_id="ch:8",
            source_role="floor_tom",
            mode="live",
            problem_hints=("noisy_drum",),
            muddiness_score=0.0,
            noise_floor_dbfs=-44.0,
            bleed_risk=0.9,
            tail_risk=0.86,
            chatter_risk=0.82,
            active_performance=True,
        )
    )

    assert recommendation.status == "proposal_only"
    assert recommendation.selected_strategy.strategy_id == (
        "hold_live_noisy_drum_processing_for_operator"
    )
    assert recommendation.evidence["operator_explanation_sections"][
        "selection_policy"
    ] == "live_noisy_drum_review_priority"
    mute = [
        candidate
        for candidate in recommendation.alternatives
        if candidate.strategy_id == "manual_tom_mute_between_hits"
    ][0]
    assert "live_manual_mute_requires_operator_review_and_wing_proof" in (
        mute.blockers
    )
    _assert_selected_has_no_apply_values(recommendation)
    params = recommendation.selected_strategy.parameters
    assert params["live_noisy_drum_hold_proposal"] is True
    assert params["live_noisy_drum_previous_mute_readback_required"] is True
    assert (
        params["live_noisy_drum_no_gate_expander_mute_without_operator_wing_proof"]
        is True
    )
    noisy_drum_readback = recommendation.selected_strategy.future_wing_readback_contract[
        "live_noisy_drum_hold_readback"
    ]
    assert noisy_drum_readback["read_only"] is True
    assert "/ch/{channel}/gate/thr21" in noisy_drum_readback[
        "wing_osc_path_templates"
    ]
    assert "/ch/{channel}/mute" in noisy_drum_readback["wing_osc_path_templates"]
    assert recommendation.to_operator_proposal()["can_apply_live"] is False


def test_soundcheck_snare_transients_select_gentle_peak_aware_compression():
    recommendation = analyze_smart_dsp_strategy(
        _analysis(
            target_id="ch:9",
            source_role="snare_top",
            problem_hints=("transient_control",),
            muddiness_score=0.0,
            estimated_lufs=-23.5,
            transient_density=5.2,
            transient_strength=0.78,
            transient_peak_dbfs=-5.8,
            transient_rise_time_ms=6.0,
            transient_decay_ms=42.0,
            crest_factor_db=22.0,
            true_peak_headroom_db=5.0,
        )
    )

    assert recommendation.status == "proposal_only"
    assert "transient_control" in recommendation.evidence["detected_problems"]
    assert recommendation.selected_strategy.strategy_id == "drum_transient_tamer_compression"
    assert recommendation.selected_strategy.method_family == "dynamics"
    params = recommendation.selected_strategy.parameters
    assert params["detector"] == "peak"
    assert params["wing_compressor_detector_osc_value"] == "PEAK"
    assert params["wing_compressor_detector_value_type"] == "string_enum"
    assert params["wing_compressor_detector_original_value"] == "peak"
    assert params["knee_db"] == 2.5
    assert params["wing_compressor_knee_osc_value"] == 3
    assert params["wing_compressor_knee_value_type"] == "integer"
    assert params["source_family"] == "snare"
    assert params["makeup_gain_db"] == 0.0
    assert params["target_gain_reduction_db"] <= 3.2
    assert params["attack_ms"] < params["release_ms"]
    assert params["transient_strength"] == 0.78
    assert params["transient_parameter_source"] == "measured_transient_envelope"
    assert params["transient_peak_dbfs"] == -5.8
    assert params["transient_rise_time_ms"] == 6.0
    assert params["transient_decay_ms"] == 42.0

    hold = [
        candidate
        for candidate in recommendation.alternatives
        if candidate.strategy_id == "manual_transient_hold_for_operator"
    ][0]
    clamp = [
        candidate
        for candidate in recommendation.alternatives
        if candidate.strategy_id == "fast_transient_clamp_review"
    ][0]
    assert hold.method_family == "analysis_only"
    assert "transient_clamp_requires_operator_review" in clamp.blockers
    assert recommendation.to_operator_proposal()["live_apply_allowed"] is False


def test_live_snare_transients_hold_when_true_peak_headroom_is_low():
    recommendation = analyze_smart_dsp_strategy(
        _analysis(
            target_id="ch:9",
            source_role="snare_top",
            mode="live",
            problem_hints=("transient_control",),
            muddiness_score=0.0,
            estimated_lufs=-20.0,
            transient_density=4.4,
            transient_strength=0.72,
            crest_factor_db=21.0,
            true_peak_headroom_db=0.6,
            active_performance=True,
        )
    )

    assert recommendation.status == "analysis_only"
    assert recommendation.selected_strategy.strategy_id == (
        "source_true_peak_headroom_pullback"
    )
    assert recommendation.selected_strategy.method_family == "gain_fader"
    assert "true_peak_headroom_low" in recommendation.blockers
    assert recommendation.selected_strategy.parameters["delta_db"] < 0.0
    assert recommendation.selected_strategy.parameters["headroom_restoration_first"] is True
    assert recommendation.evidence["operator_explanation_sections"][
        "selection_policy"
    ] == "true_peak_headroom_safety_priority"
    hold = [
        candidate
        for candidate in recommendation.alternatives
        if candidate.strategy_id == "manual_transient_hold_for_operator"
    ][0]
    assert hold.parameters["headroom_watch_required"] is True
    tamer = [
        candidate
        for candidate in recommendation.alternatives
        if candidate.strategy_id == "drum_transient_tamer_compression"
    ][0]
    assert tamer.parameters["makeup_gain_db"] == 0.0
    assert recommendation.to_operator_proposal()["live_apply_allowed"] is False


def test_dynamic_vocal_selects_phrase_leveling_compression_with_bounded_makeup():
    recommendation = analyze_smart_dsp_strategy(
        _analysis(
            problem_hints=("dynamic_control",),
            muddiness_score=0.0,
            rms_dbfs=-26.0,
            estimated_lufs=-26.8,
            lra_proxy_db=18.0,
            dynamics_variance_score=0.78,
            crest_factor_db=20.0,
            dynamics_peak_to_median_db=16.0,
            dynamics_rise_time_ms=180.0,
            dynamics_recovery_ms=160.0,
            true_peak_headroom_db=4.0,
        )
    )

    assert recommendation.status == "proposal_only"
    assert recommendation.selected_strategy.strategy_id == "vocal_phrase_leveling_compression"
    assert recommendation.selected_strategy.method_family == "dynamics"
    params = recommendation.selected_strategy.parameters
    assert params["source_family"] == "vocal"
    assert params["compressor_enabled"] is True
    assert params["wing_compressor_on_osc_path_suffix"] == "dyn/on"
    assert params["wing_compressor_on_osc_value"] == WING_COMPRESSOR_ON_MAX
    assert params["wing_compressor_on_value_type"] == "integer_bool"
    assert params["threshold_dbfs"] < 0.0
    assert params["wing_compressor_threshold_osc_path_suffix"] == "dyn/thr24"
    assert params["wing_compressor_threshold_osc_value"] == params["threshold_dbfs"]
    assert params["wing_compressor_threshold_value_type"] == "float"
    assert params["dynamics_parameter_source"] == "measured_envelope"
    assert params["dynamics_peak_to_median_db"] == 16.0
    assert params["dynamics_recovery_ms"] == 160.0
    assert 1.6 <= params["ratio"] <= 2.9
    assert params["wing_compressor_ratio_osc_value"] == "2.5"
    assert params["wing_compressor_ratio_value_type"] == "string_enum"
    assert params["wing_compressor_ratio_numeric_value"] == 2.5
    assert params["wing_compressor_ratio_target_surface"] == "channel"
    assert params["wing_compressor_ratio_osc_path_suffix"] == "dyn/ratio"
    assert params["attack_ms"] > 0.0
    assert params["release_ms"] > params["attack_ms"]
    assert params["knee_db"] == 4.1
    assert params["compressor_knee_parameter_source"] == (
        "measured_envelope_variance_peak_to_median"
    )
    assert params["wing_compressor_knee_osc_value"] == 4
    assert params["wing_compressor_knee_quantized_from"] == 4.1
    assert params["wing_compressor_knee_value_type"] == "integer"
    assert params["wing_compressor_knee_osc_path_suffix"] == "dyn/knee"
    assert 0.0 <= params["makeup_gain_db"] <= 0.8
    assert params["requires_true_peak_headroom_db"] == 1.8
    assert params["requires_summed_bus_true_peak_headroom_db"] == 1.8
    assert params["soundcheck_dynamics_bounded_compression_proposal"] is True
    assert params["soundcheck_dynamics_native_compressor_osc_only_required"] is True
    assert (
        params["soundcheck_dynamics_max_target_gain_reduction_db"]
        == SOUNDCHECK_DYNAMICS_MAX_TARGET_GAIN_REDUCTION_DB
    )
    assert (
        params["soundcheck_dynamics_max_makeup_gain_db"]
        == SOUNDCHECK_DYNAMICS_MAX_MAKEUP_GAIN_DB
    )
    assert (
        params["soundcheck_dynamics_min_headroom_margin_db"]
        == SOUNDCHECK_DYNAMICS_MIN_HEADROOM_MARGIN_DB
    )
    assert params["soundcheck_dynamics_required_true_peak_headroom_db"] == 1.8
    assert (
        params["soundcheck_dynamics_required_summed_bus_true_peak_headroom_db"]
        == 1.8
    )
    assert params["soundcheck_dynamics_target_gr_review_only"] is True
    assert params["soundcheck_dynamics_target_gr_readback_required"] is True
    assert params["soundcheck_dynamics_makeup_gain_headroom_required"] is True
    assert params["soundcheck_dynamics_summed_bus_headroom_required"] is True
    assert params["soundcheck_dynamics_compressor_model_readback_required"] is True
    assert params["soundcheck_dynamics_gain_reduction_meter_readback_required"] is True
    assert params["soundcheck_dynamics_feedback_recheck_required"] is True
    assert params["soundcheck_dynamics_operator_confirmation_required"] is True
    assert params["soundcheck_dynamics_readback_required"] is True
    assert params["soundcheck_dynamics_rollback_ready"] is True
    assert params["soundcheck_dynamics_live_promotion_status"] == (
        "blocked_until_real_wing_compressor_readback_gr_meter_headroom_feedback_recheck_and_operator_approval"
    )
    assert params["soundcheck_dynamics_live_apply_blocked_without_wing_proof"] is True
    assert params["compression_mix_percent"] == 83.5
    assert params["wing_compressor_mix_osc_path_suffix"] == "dyn/mix"
    assert params["wing_compressor_mix_osc_value"] == 83.5
    assert params["wing_compressor_mix_value_type"] == "float"
    assert params["compressor_envelope_curve"] == "smooth_log"
    assert params["compressor_envelope_curve_source"] == "source_family_envelope_policy"
    assert params["wing_compressor_envelope_curve_osc_path_suffix"] == "dyn/env"
    assert params["wing_compressor_envelope_curve_osc_value"] == "LOG"
    assert params["wing_compressor_envelope_curve_value_type"] == "string_enum"
    assert params["compressor_auto_timing_enabled"] is False
    assert params["compressor_auto_timing_source"] == (
        "engine_calculated_source_dynamics_timing_policy"
    )
    assert params["wing_compressor_auto_osc_path_suffix"] == "dyn/auto"
    assert params["wing_compressor_auto_osc_value"] == WING_COMPRESSOR_AUTO_MIN
    assert params["wing_compressor_auto_value_type"] == "integer_bool"
    assert params["detector"] == "program_rms_peak_blend"
    assert params["wing_compressor_detector_osc_value"] == "RMS"
    assert params["wing_compressor_detector_value_type"] == "string_enum"
    assert params["wing_compressor_detector_original_value"] == (
        "program_rms_peak_blend"
    )
    assert params["wing_compressor_detector_osc_path_suffix"] == "dyn/det"
    assert params["compressor_sidechain_frequency_hz"] == 120.0
    assert params["compressor_sidechain_q"] == 1.0
    assert params["compressor_sidechain_parameter_source"] == (
        "wing_dynsc_source_family_detector_filter_policy"
    )
    assert params["compressor_sidechain_evidence_frequency_hz"] == 0.0
    assert params["compressor_sidechain_low_end_pressure_score"] == 0.0
    assert params["compressor_sidechain_plosive_risk_score"] == 0.0
    assert params["preserve_words_and_consonants"] is True
    assert params["de_esser_after_compression_check"] is True
    assert params["requires_source_bus_readback"] is True
    readback = recommendation.selected_strategy.future_wing_readback_contract
    _assert_common_readback_contract(readback)
    assert readback["dynamics_readback"]["makeup_gain_must_respect_headroom"] is True
    dynamics_readback = readback["dynamics_readback"]
    assert dynamics_readback["soundcheck_dynamics_bounded_compression_proposal"] is True
    assert (
        dynamics_readback["soundcheck_dynamics_native_compressor_osc_only_required"]
        is True
    )
    assert (
        dynamics_readback["soundcheck_dynamics_max_target_gain_reduction_db"]
        == SOUNDCHECK_DYNAMICS_MAX_TARGET_GAIN_REDUCTION_DB
    )
    assert (
        dynamics_readback["soundcheck_dynamics_max_makeup_gain_db"]
        == SOUNDCHECK_DYNAMICS_MAX_MAKEUP_GAIN_DB
    )
    assert (
        dynamics_readback["soundcheck_dynamics_min_headroom_margin_db"]
        == SOUNDCHECK_DYNAMICS_MIN_HEADROOM_MARGIN_DB
    )
    assert dynamics_readback["soundcheck_dynamics_required_true_peak_headroom_db"] == 1.8
    assert (
        dynamics_readback[
            "soundcheck_dynamics_required_summed_bus_true_peak_headroom_db"
        ]
        == 1.8
    )
    assert dynamics_readback["soundcheck_dynamics_target_gr_review_only"] is True
    assert dynamics_readback["soundcheck_dynamics_target_gr_readback_required"] is True
    assert dynamics_readback["soundcheck_dynamics_makeup_gain_headroom_required"] is True
    assert dynamics_readback["soundcheck_dynamics_summed_bus_headroom_required"] is True
    assert (
        dynamics_readback["soundcheck_dynamics_compressor_model_readback_required"]
        is True
    )
    assert (
        dynamics_readback["soundcheck_dynamics_gain_reduction_meter_readback_required"]
        is True
    )
    assert dynamics_readback["soundcheck_dynamics_readback_required"] is True
    assert dynamics_readback["soundcheck_dynamics_rollback_ready"] is True
    assert (
        dynamics_readback["soundcheck_dynamics_live_apply_blocked_without_wing_proof"]
        is True
    )
    assert readback["dynamics_readback"]["dynamics_model_readback_must_equal"] == (
        WING_DYNAMICS_REQUIRED_MODEL
    )
    assert readback["dynamics_readback"]["wing_compressor_model_osc_path_suffix"] == (
        "dyn/mdl"
    )
    assert readback["dynamics_readback"][
        "wing_compressor_on_mapping_required"
    ] is True
    assert readback["dynamics_readback"][
        "wing_compressor_on_osc_path_suffix"
    ] == "dyn/on"
    assert readback["dynamics_readback"][
        "wing_compressor_on_osc_value"
    ] == WING_COMPRESSOR_ON_MAX
    assert readback["dynamics_readback"][
        "wing_compressor_on_readback_must_match_osc_value"
    ] is True
    assert readback["dynamics_readback"]["read_before_paths"] == [
        "dyn/on",
        "dyn/mdl",
        "dyn/thr24",
        "dyn/ratio",
        "dyn/det",
        "dyn/att",
        "dyn/hld",
        "dyn/rel",
        "dyn/env",
        "dyn/auto",
        "dyn/knee",
        "dyn/mix",
        "dyn/gain",
        "dyn/gr",
        "dynxo/depth",
        "dynxo/type",
        "dynxo/f",
        "dynsc/type",
        "dynsc/f",
        "dynsc/q",
        "dynsc/src",
        "dynsc/tap",
    ]
    assert readback["dynamics_readback"]["read_after_paths"] == readback[
        "dynamics_readback"
    ]["read_before_paths"]
    assert readback["dynamics_readback"][
        "wing_compressor_ratio_mapping_required"
    ] is True
    assert readback["dynamics_readback"]["wing_compressor_ratio_osc_value"] == "2.5"
    assert readback["dynamics_readback"][
        "wing_compressor_ratio_readback_must_match_osc_value"
    ] is True
    assert readback["dynamics_readback"][
        "wing_compressor_detector_mapping_required"
    ] is True
    assert readback["dynamics_readback"][
        "wing_compressor_detector_osc_value"
    ] == "RMS"
    assert readback["dynamics_readback"][
        "wing_compressor_detector_readback_must_match_osc_value"
    ] is True
    assert readback["dynamics_readback"][
        "wing_compressor_knee_mapping_required"
    ] is True
    assert readback["dynamics_readback"]["wing_compressor_knee_osc_value"] == 4
    assert readback["dynamics_readback"][
        "wing_compressor_knee_readback_must_match_osc_value"
    ] is True
    assert readback["dynamics_readback"][
        "wing_compressor_threshold_mapping_required"
    ] is True
    assert readback["dynamics_readback"][
        "wing_compressor_threshold_osc_path_suffix"
    ] == "dyn/thr24"
    assert readback["dynamics_readback"][
        "wing_compressor_threshold_osc_value"
    ] == params["wing_compressor_threshold_osc_value"]
    assert readback["dynamics_readback"][
        "wing_compressor_threshold_readback_must_match_osc_path"
    ] is True
    assert readback["dynamics_readback"][
        "wing_compressor_threshold_readback_must_match_osc_value"
    ] is True
    assert readback["dynamics_readback"][
        "wing_compressor_attack_mapping_required"
    ] is True
    assert readback["dynamics_readback"]["wing_compressor_attack_osc_path_suffix"] == (
        "dyn/att"
    )
    assert readback["dynamics_readback"]["wing_compressor_attack_osc_value"] == (
        params["wing_compressor_attack_osc_value"]
    )
    assert readback["dynamics_readback"][
        "wing_compressor_release_mapping_required"
    ] is True
    assert readback["dynamics_readback"]["wing_compressor_release_osc_path_suffix"] == (
        "dyn/rel"
    )
    assert readback["dynamics_readback"]["wing_compressor_release_osc_value"] == (
        params["wing_compressor_release_osc_value"]
    )
    assert readback["dynamics_readback"][
        "wing_compressor_makeup_gain_mapping_required"
    ] is True
    assert readback["dynamics_readback"][
        "wing_compressor_makeup_gain_osc_path_suffix"
    ] == "dyn/gain"
    assert readback["dynamics_readback"][
        "wing_compressor_makeup_gain_osc_value"
    ] == params["wing_compressor_makeup_gain_osc_value"]
    assert readback["dynamics_readback"][
        "wing_compressor_mix_mapping_required"
    ] is True
    assert readback["dynamics_readback"][
        "wing_compressor_mix_osc_path_suffix"
    ] == "dyn/mix"
    assert readback["dynamics_readback"][
        "wing_compressor_mix_osc_value"
    ] == params["wing_compressor_mix_osc_value"]
    assert readback["dynamics_readback"][
        "wing_compressor_mix_readback_must_match_osc_value"
    ] is True
    assert readback["dynamics_readback"][
        "wing_compressor_envelope_curve_mapping_required"
    ] is True
    assert readback["dynamics_readback"][
        "wing_compressor_envelope_curve_osc_path_suffix"
    ] == "dyn/env"
    assert readback["dynamics_readback"][
        "wing_compressor_envelope_curve_osc_value"
    ] == "LOG"
    assert readback["dynamics_readback"][
        "wing_compressor_envelope_curve_readback_must_match_osc_value"
    ] is True
    assert readback["dynamics_readback"][
        "wing_compressor_auto_mapping_required"
    ] is True
    assert readback["dynamics_readback"][
        "wing_compressor_auto_osc_path_suffix"
    ] == "dyn/auto"
    assert readback["dynamics_readback"][
        "wing_compressor_auto_osc_value"
    ] == WING_COMPRESSOR_AUTO_MIN
    assert readback["dynamics_readback"][
        "wing_compressor_auto_readback_must_match_osc_value"
    ] is True
    _assert_selected_write_key_contract(
        params=params,
        readback=readback["dynamics_readback"],
        prefix="wing_dynamics",
        selected_write_keys=list(WING_DYNAMICS_WRITE_PARAMETER_KEYS),
        write_parameter_keys=WING_DYNAMICS_WRITE_PARAMETER_KEYS,
        selection_policy=WING_DYNAMICS_WRITE_KEY_SELECTION_POLICY,
    )
    wing_paths = readback["wing_osc_applicability"]["native_osc_path_templates"]
    assert "/ch/{channel}/dyn/on" in wing_paths
    assert "/ch/{channel}/dyn/thr24" in wing_paths
    assert "/ch/{channel}/dyn/mix" in wing_paths
    assert "/ch/{channel}/dyn/env" in wing_paths
    assert "/ch/{channel}/dyn/auto" in wing_paths
    assert "/ch/{channel}/dyn/thr26" not in wing_paths
    assert "/ch/{channel}/dyn/thr37" not in wing_paths
    assert "/ch/{channel}/dyn/det" in wing_paths
    assert "eq_band_readback" not in readback
    _assert_selected_dynamics_apply_values(
        recommendation,
        {
            "wing_compressor_on_osc_value": WING_COMPRESSOR_ON_MAX,
            "wing_compressor_threshold_osc_value": params[
                "wing_compressor_threshold_osc_value"
            ],
            "wing_compressor_attack_osc_value": params[
                "wing_compressor_attack_osc_value"
            ],
            "wing_compressor_hold_osc_value": params["wing_compressor_hold_osc_value"],
            "wing_compressor_release_osc_value": params[
                "wing_compressor_release_osc_value"
            ],
            "wing_compressor_makeup_gain_osc_value": params[
                "wing_compressor_makeup_gain_osc_value"
            ],
            "wing_compressor_mix_osc_value": 83.5,
            "wing_compressor_ratio_osc_value": "2.5",
            "wing_compressor_detector_osc_value": "RMS",
            "wing_compressor_envelope_curve_osc_value": "LOG",
            "wing_compressor_auto_osc_value": WING_COMPRESSOR_AUTO_MIN,
            "wing_compressor_knee_osc_value": 4,
            "wing_compressor_crossover_type_osc_value": "OFF",
            "wing_compressor_crossover_depth_osc_value": 0.0,
            "wing_compressor_crossover_frequency_osc_value": 1000.0,
            "wing_compressor_sidechain_type_osc_value": "HP12",
            "wing_compressor_sidechain_frequency_osc_value": 120.0,
            "wing_compressor_sidechain_q_osc_value": 1.0,
            "wing_compressor_sidechain_src_osc_value": "SELF",
            "wing_compressor_sidechain_tap_osc_value": "IN",
        },
    )
    alternatives = {candidate.strategy_id for candidate in recommendation.alternatives}
    assert "gentle_channel_compression" in alternatives
    peak_clamp = [
        candidate
        for candidate in recommendation.alternatives
        if candidate.strategy_id == "fast_peak_clamp_compression"
    ][0]
    assert "limiter_style_channel_compression_requires_operator_review" in peak_clamp.blockers


def test_vocal_dynamics_sidechain_hpf_uses_measured_low_end_pressure():
    recommendation = analyze_smart_dsp_strategy(
        _analysis(
            problem_hints=("dynamic_control",),
            muddiness_score=0.0,
            rms_dbfs=-26.0,
            estimated_lufs=-26.8,
            lra_proxy_db=18.0,
            dynamics_variance_score=0.78,
            crest_factor_db=20.0,
            dynamics_peak_to_median_db=16.0,
            dynamics_rise_time_ms=180.0,
            dynamics_recovery_ms=160.0,
            true_peak_headroom_db=4.0,
            low_end_excess_score=0.72,
            low_end_resonance_frequency_hz=96.0,
        )
    )

    assert recommendation.status == "proposal_only"
    assert recommendation.selected_strategy.strategy_id == "vocal_phrase_leveling_compression"
    params = recommendation.selected_strategy.parameters
    assert params["compressor_sidechain_filter_type"] == "HP12"
    assert params["compressor_sidechain_frequency_hz"] == 138.5
    assert params["compressor_sidechain_q"] == 1.0
    assert params["compressor_sidechain_parameter_source"] == (
        "measured_low_end_pressure_detector_high_pass"
    )
    assert params["compressor_sidechain_evidence_frequency_hz"] == 96.0
    assert params["compressor_sidechain_low_end_pressure_score"] == 0.72
    assert params["compressor_sidechain_plosive_risk_score"] == 0.0
    assert params["wing_compressor_sidechain_frequency_osc_path_suffix"] == "dynsc/f"
    assert params["wing_compressor_sidechain_frequency_osc_value"] == 138.5
    assert params["wing_compressor_sidechain_q_osc_path_suffix"] == "dynsc/q"
    assert params["wing_compressor_sidechain_q_osc_value"] == 1.0
    readback = recommendation.selected_strategy.future_wing_readback_contract[
        "dynamics_readback"
    ]
    assert readback["wing_compressor_sidechain_frequency_osc_value"] == 138.5
    assert readback["wing_compressor_sidechain_q_osc_value"] == 1.0
    assert recommendation.to_operator_proposal()["live_apply_allowed"] is False


def test_active_dynamics_model_conflict_blocks_compressor_selection():
    recommendation = analyze_smart_dsp_strategy(
        _analysis(
            problem_hints=("dynamic_control",),
            muddiness_score=0.0,
            rms_dbfs=-26.0,
            estimated_lufs=-26.8,
            lra_proxy_db=18.0,
            dynamics_variance_score=0.78,
            crest_factor_db=20.0,
            dynamics_peak_to_median_db=16.0,
            dynamics_rise_time_ms=180.0,
            dynamics_recovery_ms=160.0,
            true_peak_headroom_db=4.0,
            current_wing_dynamics_model="LEV",
            current_wing_dynamics_enabled=True,
            current_wing_dynamics_model_readback_available=True,
            current_wing_dynamics_operator_owned=True,
        )
    )

    assert recommendation.status == "proposal_only"
    assert recommendation.selected_strategy.strategy_id == (
        "manual_level_ride_before_compression"
    )
    assert recommendation.selected_strategy.method_family == "analysis_only"
    alternatives = {
        candidate.strategy_id: candidate for candidate in recommendation.alternatives
    }
    compressor = alternatives["vocal_phrase_leveling_compression"]
    assert WING_DYNAMICS_MODEL_CONFLICT_BLOCKER in compressor.blockers
    assert compressor.parameters["wing_dynamics_required_model"] == (
        WING_DYNAMICS_REQUIRED_MODEL
    )
    assert compressor.parameters["wing_dynamics_current_model"] == "LEV"
    assert compressor.parameters["wing_dynamics_model_conflict"] is True
    assert compressor.future_wing_readback_contract["dynamics_readback"][
        "wing_dynamics_model_conflict_blocker"
    ] == WING_DYNAMICS_MODEL_CONFLICT_BLOCKER
    tradeoffs = {
        row["strategy_id"]: row
        for row in recommendation.evidence["strategy_search"]["alternative_tradeoffs"]
    }
    assert WING_DYNAMICS_MODEL_CONFLICT_BLOCKER in tradeoffs[
        "vocal_phrase_leveling_compression"
    ]["blockers"]
    assert tradeoffs["vocal_phrase_leveling_compression"][
        "wing_osc_candidate_apply_status"
    ] == "not_wing_applicable_until_dynamics_model_readback_or_operator_review"


def test_operator_owned_comp_dynamics_blocks_compressor_parameter_apply():
    recommendation = analyze_smart_dsp_strategy(
        _analysis(
            problem_hints=("dynamic_control",),
            muddiness_score=0.0,
            rms_dbfs=-26.0,
            estimated_lufs=-26.8,
            lra_proxy_db=18.0,
            dynamics_variance_score=0.78,
            crest_factor_db=20.0,
            dynamics_peak_to_median_db=16.0,
            dynamics_rise_time_ms=180.0,
            dynamics_recovery_ms=160.0,
            true_peak_headroom_db=4.0,
            current_wing_dynamics_model=WING_DYNAMICS_REQUIRED_MODEL,
            current_wing_dynamics_enabled=True,
            current_wing_dynamics_model_readback_available=True,
            current_wing_dynamics_operator_owned=True,
        )
    )

    assert recommendation.status == "proposal_only"
    assert recommendation.selected_strategy.strategy_id == (
        "manual_level_ride_before_compression"
    )
    assert recommendation.selected_strategy.method_family == "analysis_only"
    alternatives = {
        candidate.strategy_id: candidate for candidate in recommendation.alternatives
    }
    compressor = alternatives["vocal_phrase_leveling_compression"]
    assert compressor.blockers == (WING_DYNAMICS_OPERATOR_OWNED_MODEL_BLOCKER,)
    assert compressor.parameters["wing_dynamics_required_model"] == (
        WING_DYNAMICS_REQUIRED_MODEL
    )
    assert compressor.parameters["wing_dynamics_current_model"] == (
        WING_DYNAMICS_REQUIRED_MODEL
    )
    assert compressor.parameters["wing_dynamics_current_operator_owned"] is True
    assert (
        compressor.parameters[
            "wing_dynamics_operator_owned_model_edit_requires_review"
        ]
        is True
    )
    assert compressor.parameters["wing_dynamics_operator_owned_model_blocker"] == (
        WING_DYNAMICS_OPERATOR_OWNED_MODEL_BLOCKER
    )
    assert (
        compressor.parameters["wing_dynamics_model_change_requires_operator_review"]
        is True
    )
    dynamics_readback = compressor.future_wing_readback_contract[
        "dynamics_readback"
    ]
    assert dynamics_readback["wing_dynamics_operator_owned_model_blocker"] == (
        WING_DYNAMICS_OPERATOR_OWNED_MODEL_BLOCKER
    )
    applicability = compressor.future_wing_readback_contract[
        "wing_osc_applicability"
    ]
    assert applicability["native_osc_supported"] is False
    assert applicability["native_osc_support_level"] == (
        "dynamics_operator_owned_model_edit_requires_review"
    )
    assert (
        applicability["dynamics_operator_owned_model_edit_requires_review"] is True
    )
    assert applicability["dynamics_operator_owned_model_blocker"] == (
        WING_DYNAMICS_OPERATOR_OWNED_MODEL_BLOCKER
    )
    assert applicability["unsupported_strategy_reason"] == (
        "dynamics_model_matches_but_current_wing_dynamics_is_operator_owned"
    )
    tradeoffs = {
        row["strategy_id"]: row
        for row in recommendation.evidence["strategy_search"]["alternative_tradeoffs"]
    }
    assert WING_DYNAMICS_OPERATOR_OWNED_MODEL_BLOCKER in tradeoffs[
        "vocal_phrase_leveling_compression"
    ]["blockers"]
    assert tradeoffs["vocal_phrase_leveling_compression"][
        "wing_osc_native_supported"
    ] is False
    assert tradeoffs["vocal_phrase_leveling_compression"][
        "wing_osc_support_level"
    ] == "dynamics_operator_owned_model_edit_requires_review"
    assert tradeoffs["vocal_phrase_leveling_compression"][
        "wing_osc_candidate_apply_status"
    ] == "not_wing_applicable_until_dynamics_model_readback_or_operator_review"


def test_dynamic_vocal_holds_compressor_makeup_when_summed_bus_headroom_low():
    recommendation = analyze_smart_dsp_strategy(
        _analysis(
            problem_hints=("dynamic_control",),
            muddiness_score=0.0,
            rms_dbfs=-26.0,
            estimated_lufs=-26.8,
            lra_proxy_db=18.0,
            dynamics_variance_score=0.78,
            crest_factor_db=20.0,
            dynamics_peak_to_median_db=16.0,
            dynamics_rise_time_ms=180.0,
            dynamics_recovery_ms=160.0,
            true_peak_headroom_db=4.0,
            summed_bus_evidence_available=True,
            summed_bus_role="lead_vocal_bus",
            summed_bus_true_peak_headroom_db=1.2,
            summed_bus_level_plane_error_lu=0.1,
        )
    )

    assert recommendation.status == "proposal_only"
    assert (
        recommendation.selected_strategy.strategy_id
        == "hold_compressor_makeup_for_summed_bus_headroom"
    )
    assert recommendation.selected_strategy.method_family == "analysis_only"
    params = recommendation.selected_strategy.parameters
    assert params["dsp_change"] == "hold"
    assert params["reason"] == "summed_bus_headroom_before_compressor_makeup"
    assert params["candidate_makeup_gain_db"] == 1.0
    assert params["true_peak_headroom_db"] == 4.0
    assert params["summed_bus_role"] == "lead_vocal_bus"
    assert params["summed_bus_true_peak_headroom_db"] == 1.2
    assert params["requires_summed_bus_true_peak_headroom_db"] == 2.0
    assert params["blocked_wing_osc_path_suffix"] == "dyn/gain"
    assert params["forbidden_without_wing_proof"] == (
        "compressor_makeup_gain_dyn_gain_until_summed_bus_headroom_verified"
    )
    alternatives = {
        candidate.strategy_id: candidate for candidate in recommendation.alternatives
    }
    assert "summed_bus_headroom_blocks_compressor_makeup" in alternatives[
        "vocal_phrase_leveling_compression"
    ].blockers
    assert "summed_bus_headroom_blocks_compressor_makeup" in alternatives[
        "gentle_channel_compression"
    ].blockers
    assert alternatives["vocal_phrase_leveling_compression"].parameters[
        "makeup_gain_db"
    ] == 0.8
    assert alternatives["gentle_channel_compression"].parameters[
        "makeup_gain_db"
    ] == 1.0
    tradeoffs = {
        item["strategy_id"]: item
        for item in recommendation.to_operator_proposal()["proposed_action"][
            "alternatives_considered"
        ]
    }
    assert tradeoffs["vocal_phrase_leveling_compression"]["why_not_selected"] == (
        "soundcheck_summed_bus_compressor_makeup_priority:"
        "verify_summed_bus_headroom_before_wing_dyn_gain"
    )
    assert tradeoffs["gentle_channel_compression"]["why_not_selected"] == (
        "soundcheck_summed_bus_compressor_makeup_priority:"
        "verify_summed_bus_headroom_before_wing_dyn_gain"
    )
    _assert_selected_has_no_apply_values(recommendation)


def test_soundcheck_dynamic_vocal_restores_level_plane_before_compression():
    recommendation = analyze_smart_dsp_strategy(
        _analysis(
            problem_hints=("dynamic_control",),
            muddiness_score=0.0,
            estimated_lufs=-30.5,
            level_plane_error_lu=-4.2,
            lra_proxy_db=18.0,
            dynamics_variance_score=0.78,
            crest_factor_db=20.0,
            dynamics_peak_to_median_db=16.0,
            dynamics_rise_time_ms=180.0,
            dynamics_recovery_ms=160.0,
            true_peak_headroom_db=7.0,
            feedback_risk=0.1,
            bleed_risk=0.1,
        )
    )

    assert recommendation.status == "proposal_only"
    assert recommendation.selected_strategy.strategy_id == "soundcheck_bounded_gain_up"
    assert recommendation.selected_strategy.method_family == "gain_fader"
    assert recommendation.evidence["operator_explanation_sections"][
        "selection_policy"
    ] == "soundcheck_level_plane_before_dynamics_priority"
    params = recommendation.selected_strategy.parameters
    assert params["target_plane"] == "ayaic_mix_monolith"
    assert params["target_scope"] == "source_and_summed_bus"
    assert params["current_minus_target_lu"] == -4.2
    assert params["bounded_delta_db"] == 3.0
    assert params["operator_confirmation_required"] is True
    alternatives = {
        candidate.strategy_id: candidate for candidate in recommendation.alternatives
    }
    assert alternatives["vocal_phrase_leveling_compression"].parameters[
        "dynamics_parameter_source"
    ] == "measured_envelope"
    tradeoffs = {
        row["strategy_id"]: row
        for row in recommendation.evidence["strategy_search"]["alternative_tradeoffs"]
    }
    assert tradeoffs["vocal_phrase_leveling_compression"]["why_not_selected"] == (
        "soundcheck_level_plane_priority:"
        "restore_level_plane_before_compression"
    )
    assert tradeoffs["gentle_channel_compression"]["why_not_selected"] == (
        "soundcheck_level_plane_priority:"
        "restore_level_plane_before_compression"
    )
    assert recommendation.to_operator_proposal()["live_apply_allowed"] is False


def test_live_active_vocal_dynamics_prefers_manual_ride_before_compression():
    recommendation = analyze_smart_dsp_strategy(
        _analysis(
            mode="live",
            active_performance=True,
            problem_hints=("dynamic_control",),
            muddiness_score=0.0,
            estimated_lufs=-25.0,
            lra_proxy_db=17.0,
            dynamics_variance_score=0.76,
            crest_factor_db=20.0,
            dynamics_peak_to_median_db=15.0,
            dynamics_rise_time_ms=160.0,
            dynamics_recovery_ms=150.0,
            true_peak_headroom_db=4.0,
        )
    )

    assert recommendation.status == "proposal_only"
    assert recommendation.selected_strategy.strategy_id == (
        "manual_level_ride_before_compression"
    )
    assert recommendation.selected_strategy.method_family == "analysis_only"
    assert recommendation.evidence["operator_explanation_sections"][
        "selection_policy"
    ] == "live_vocal_dynamics_ride_priority"
    params = recommendation.selected_strategy.parameters
    assert params["source_family"] == "vocal"
    assert params["target_plane"] == "ayaic_mix_monolith"
    assert params["max_ride_delta_db"] == 1.5
    assert params["live_vocal_dynamics_ride_priority"] is True
    assert params["required_context"] == (
        "lead_vocal_phrase_manual_ride_bus_headroom_compressor_readback"
    )
    assert params["forbidden_without_wing_proof"] == (
        "compressor_peak_clamp_makeup_gain"
    )
    assert params["live_dynamics_ride_hold_proposal"] is True
    assert params["live_dynamics_ride_profile"] == "lead_vocal_phrase"
    assert params["live_dynamics_ride_previous_gain_readback_required"] is True
    assert params["live_dynamics_ride_previous_compressor_readback_required"] is True
    assert params["live_dynamics_ride_gain_reduction_meter_readback_required"] is True
    assert (
        params["live_dynamics_ride_no_compressor_or_automation_without_operator_wing_proof"]
        is True
    )
    ride_readback = recommendation.selected_strategy.future_wing_readback_contract[
        "live_dynamics_ride_hold_readback"
    ]
    assert ride_readback["read_only"] is True
    assert ride_readback["live_dynamics_ride_profile"] == "lead_vocal_phrase"
    assert ride_readback["allowed_live_write_scope_before_wing_proof"] == []
    assert "/ch/{channel}/fdr" in ride_readback["wing_gain_osc_path_templates"]
    assert "/ch/{channel}/in/set/trim" in ride_readback[
        "wing_gain_osc_path_templates"
    ]
    assert "/ch/{channel}/dyn/thr24" in ride_readback[
        "wing_dynamics_osc_path_templates"
    ]
    assert "fdr" in ride_readback["read_before_paths"]
    assert "dyn/thr24" in ride_readback["read_before_paths"]
    alternatives = {
        candidate.strategy_id: candidate for candidate in recommendation.alternatives
    }
    assert (
        "live_vocal_leveling_compression_requires_operator_review_and_wing_proof"
        in alternatives["vocal_phrase_leveling_compression"].blockers
    )
    assert (
        "live_vocal_compression_requires_operator_review_and_wing_proof"
        in alternatives["gentle_channel_compression"].blockers
    )
    assert alternatives["vocal_phrase_leveling_compression"].parameters[
        "dynamics_parameter_source"
    ] == "measured_envelope"
    tradeoffs = {
        row["strategy_id"]: row
        for row in recommendation.evidence["strategy_search"]["alternative_tradeoffs"]
    }
    assert tradeoffs["vocal_phrase_leveling_compression"]["why_not_selected"] == (
        "live_vocal_dynamics_ride_priority:"
        "manual_level_review_precedes_compressor_change"
    )
    assert tradeoffs["gentle_channel_compression"]["why_not_selected"] == (
        "live_vocal_dynamics_ride_priority:"
        "manual_level_review_precedes_compressor_change"
    )
    assert recommendation.to_operator_proposal()["live_apply_allowed"] is False


def test_live_operator_owned_dynamics_prefers_manual_ride_before_compression():
    recommendation = analyze_smart_dsp_strategy(
        _analysis(
            target_id="ch:8",
            source_role="lead_vocal",
            source_label="Lead Vox",
            mode="live",
            problem_hints=("dynamic_control",),
            muddiness_score=0.0,
            estimated_lufs=-25.0,
            lra_proxy_db=13.0,
            dynamics_variance_score=0.82,
            crest_factor_db=20.0,
            dynamics_peak_to_median_db=11.0,
            dynamics_rise_time_ms=18.0,
            dynamics_recovery_ms=190.0,
            true_peak_headroom_db=5.0,
            feedback_risk=0.12,
            monitor_feedback_risk=0.12,
            current_wing_dynamics_model=WING_DYNAMICS_REQUIRED_MODEL,
            current_wing_dynamics_model_readback_available=True,
            current_wing_dynamics_enabled=True,
            current_wing_dynamics_operator_owned=True,
        )
    )

    assert recommendation.status == "proposal_only"
    assert recommendation.selected_strategy.strategy_id == (
        "manual_level_ride_before_compression"
    )
    assert recommendation.selected_strategy.method_family == "analysis_only"
    assert recommendation.evidence["operator_explanation_sections"][
        "selection_policy"
    ] == "live_operator_owned_dynamics_ride_priority"
    params = recommendation.selected_strategy.parameters
    assert params["live_operator_owned_dynamics_ride_priority"] is True
    assert params["live_operator_owned_dynamics_review_reason"] == (
        "current_wing_dynamics_operator_owned"
    )
    assert params["live_operator_owned_dynamics_current_operator_owned"] is True
    assert params["live_dynamics_ride_hold_proposal"] is True
    assert params["live_dynamics_ride_profile"] == "operator_owned_compressor"
    assert params["live_dynamics_ride_previous_gain_readback_required"] is True
    assert params["live_dynamics_ride_previous_compressor_readback_required"] is True
    assert params["live_dynamics_ride_gain_reduction_meter_readback_required"] is True
    assert params["live_dynamics_ride_feedback_recheck_required"] is True
    assert params["live_dynamics_ride_rollback_ready"] is True
    assert params["required_context"] == (
        "operator_owned_compressor_manual_ride_dynamics_readback"
    )
    assert params["forbidden_without_wing_proof"] == (
        "operator_owned_compressor_change"
    )
    ride_readback = recommendation.selected_strategy.future_wing_readback_contract[
        "live_dynamics_ride_hold_readback"
    ]
    assert ride_readback["read_only"] is True
    assert ride_readback["allowed_live_write_scope_before_wing_proof"] == []
    assert "/ch/{channel}/dyn/thr24" in ride_readback[
        "wing_dynamics_osc_path_templates"
    ]
    alternatives = {
        candidate.strategy_id: candidate for candidate in recommendation.alternatives
    }
    assert alternatives["vocal_phrase_leveling_compression"].blockers == (
        WING_DYNAMICS_OPERATOR_OWNED_MODEL_BLOCKER,
    )
    assert WING_DYNAMICS_OPERATOR_OWNED_MODEL_BLOCKER in alternatives[
        "gentle_channel_compression"
    ].blockers
    tradeoffs = {
        row["strategy_id"]: row
        for row in recommendation.evidence["strategy_search"]["alternative_tradeoffs"]
    }
    assert tradeoffs["vocal_phrase_leveling_compression"]["why_not_selected"] == (
        "live_operator_owned_dynamics_ride_priority:"
        "manual_level_ride_precedes_operator_owned_compressor_change"
    )
    assert tradeoffs["gentle_channel_compression"]["why_not_selected"] == (
        "live_operator_owned_dynamics_ride_priority:"
        "manual_level_ride_precedes_operator_owned_compressor_change"
    )
    proposal = recommendation.to_operator_proposal()
    assert proposal["can_apply_live"] is False
    assert proposal["live_authority_granted"] is False
    assert proposal["live_write"] is False


def test_soundcheck_operator_owned_dynamics_prefers_bounded_ride_review():
    recommendation = analyze_smart_dsp_strategy(
        _analysis(
            target_id="ch:8",
            source_role="lead_vocal",
            source_label="Lead Vox",
            mode="soundcheck",
            problem_hints=("dynamic_control",),
            muddiness_score=0.0,
            estimated_lufs=-25.0,
            lra_proxy_db=13.0,
            dynamics_variance_score=0.82,
            crest_factor_db=20.0,
            dynamics_peak_to_median_db=11.0,
            dynamics_rise_time_ms=18.0,
            dynamics_recovery_ms=190.0,
            true_peak_headroom_db=5.0,
            feedback_risk=0.12,
            monitor_feedback_risk=0.12,
            current_wing_dynamics_model=WING_DYNAMICS_REQUIRED_MODEL,
            current_wing_dynamics_model_readback_available=True,
            current_wing_dynamics_enabled=True,
            current_wing_dynamics_operator_owned=True,
        )
    )

    assert recommendation.status == "proposal_only"
    assert recommendation.selected_strategy.strategy_id == (
        "manual_level_ride_before_compression"
    )
    assert recommendation.selected_strategy.method_family == "analysis_only"
    assert recommendation.evidence["operator_explanation_sections"][
        "selection_policy"
    ] == "soundcheck_operator_owned_dynamics_ride_priority"
    params = recommendation.selected_strategy.parameters
    assert params["soundcheck_operator_owned_dynamics_ride_priority"] is True
    assert params["soundcheck_operator_owned_dynamics_review_reason"] == (
        "current_wing_dynamics_operator_owned"
    )
    assert (
        params["soundcheck_operator_owned_dynamics_current_operator_owned"]
        is True
    )
    assert params["soundcheck_dynamics_ride_hold_proposal"] is True
    assert params["soundcheck_dynamics_ride_profile"] == "operator_owned_compressor"
    assert params["soundcheck_dynamics_ride_target_surface"] == "channel"
    assert params["soundcheck_dynamics_ride_previous_gain_readback_required"] is True
    assert (
        params["soundcheck_dynamics_ride_previous_compressor_readback_required"]
        is True
    )
    assert (
        params["soundcheck_dynamics_ride_gain_reduction_meter_readback_required"]
        is True
    )
    assert params["soundcheck_dynamics_ride_feedback_recheck_required"] is True
    assert params["soundcheck_dynamics_ride_rollback_ready"] is True
    assert (
        params["soundcheck_dynamics_ride_no_compressor_or_automation_without_operator_review"]
        is True
    )
    assert (
        params["soundcheck_dynamics_ride_live_apply_blocked_without_wing_proof"]
        is True
    )
    assert params["required_context"] == (
        "soundcheck_operator_owned_compressor_manual_ride_dynamics_readback"
    )
    assert params["forbidden_without_wing_proof"] == (
        "soundcheck_operator_owned_compressor_change"
    )
    ride_readback = recommendation.selected_strategy.future_wing_readback_contract[
        "soundcheck_operator_owned_dynamics_ride_readback"
    ]
    assert ride_readback["read_only"] is True
    assert ride_readback["allowed_live_write_scope_before_wing_proof"] == []
    assert ride_readback["soundcheck_dynamics_ride_hold_proposal"] is True
    assert ride_readback["soundcheck_dynamics_ride_profile"] == (
        "operator_owned_compressor"
    )
    assert (
        ride_readback["soundcheck_operator_owned_dynamics_current_operator_owned"]
        is True
    )
    assert "/ch/{channel}/fdr" in ride_readback["wing_gain_osc_path_templates"]
    assert "/ch/{channel}/in/set/trim" in ride_readback[
        "wing_gain_osc_path_templates"
    ]
    assert "/ch/{channel}/dyn/thr24" in ride_readback[
        "wing_dynamics_osc_path_templates"
    ]
    assert "fdr" in ride_readback["read_before_paths"]
    assert "dyn/thr24" in ride_readback["read_before_paths"]
    alternatives = {
        candidate.strategy_id: candidate for candidate in recommendation.alternatives
    }
    assert alternatives["vocal_phrase_leveling_compression"].blockers == (
        WING_DYNAMICS_OPERATOR_OWNED_MODEL_BLOCKER,
    )
    assert WING_DYNAMICS_OPERATOR_OWNED_MODEL_BLOCKER in alternatives[
        "gentle_channel_compression"
    ].blockers
    tradeoffs = {
        row["strategy_id"]: row
        for row in recommendation.evidence["strategy_search"]["alternative_tradeoffs"]
    }
    assert tradeoffs["vocal_phrase_leveling_compression"]["why_not_selected"] == (
        "soundcheck_operator_owned_dynamics_ride_priority:"
        "manual_level_ride_precedes_operator_owned_compressor_change"
    )
    assert tradeoffs["gentle_channel_compression"]["why_not_selected"] == (
        "soundcheck_operator_owned_dynamics_ride_priority:"
        "manual_level_ride_precedes_operator_owned_compressor_change"
    )
    _assert_selected_has_no_apply_values(recommendation)


def test_bass_dynamics_prefers_sustain_leveling_over_generic_compression():
    recommendation = analyze_smart_dsp_strategy(
        _analysis(
            target_id="ch:2",
            source_role="bass",
            source_label="Bass DI",
            problem_hints=("dynamic_control",),
            muddiness_score=0.0,
            estimated_lufs=-24.0,
            lra_proxy_db=13.0,
            dynamics_variance_score=0.72,
            crest_factor_db=20.0,
            dynamics_peak_to_median_db=15.0,
            dynamics_rise_time_ms=220.0,
            dynamics_recovery_ms=180.0,
            true_peak_headroom_db=5.0,
        )
    )

    assert recommendation.status == "proposal_only"
    assert recommendation.selected_strategy.strategy_id == (
        "bass_sustain_leveling_compression"
    )
    assert recommendation.selected_strategy.method_family == "dynamics"
    params = recommendation.selected_strategy.parameters
    assert params["source_family"] == "bass"
    assert params["dynamics_parameter_source"] == "measured_envelope"
    assert params["ratio"] == 2.8
    assert params["wing_compressor_ratio_osc_value"] == "3.0"
    assert params["wing_compressor_ratio_value_type"] == "string_enum"
    assert params["wing_compressor_ratio_numeric_value"] == 3.0
    assert params["detector"] == "rms"
    assert params["wing_compressor_detector_osc_value"] == "RMS"
    assert params["wing_compressor_detector_value_type"] == "string_enum"
    assert params["wing_compressor_detector_original_value"] == "rms"
    assert params["knee_db"] == 4.4
    assert params["compressor_knee_parameter_source"] == (
        "measured_envelope_variance_peak_to_median"
    )
    assert params["wing_compressor_knee_osc_value"] == 4
    assert params["wing_compressor_knee_quantized_from"] == 4.4
    assert params["wing_compressor_knee_value_type"] == "integer"
    assert params["sidechain_hpf_hz"] == 65.0
    assert params["preserve_fundamental"] is True
    assert params["makeup_gain_db"] == 0.0
    alternatives = {candidate.strategy_id for candidate in recommendation.alternatives}
    assert "gentle_channel_compression" in alternatives
    assert recommendation.to_operator_proposal()["live_apply_allowed"] is False


def test_soundcheck_bass_hot_bus_rebalances_before_sustain_compression():
    recommendation = analyze_smart_dsp_strategy(
        _analysis(
            target_id="ch:2",
            source_role="bass",
            source_label="Bass DI",
            problem_hints=("dynamic_control",),
            muddiness_score=0.0,
            estimated_lufs=-24.0,
            level_plane_error_lu=0.2,
            lra_proxy_db=13.0,
            dynamics_variance_score=0.72,
            crest_factor_db=20.0,
            dynamics_peak_to_median_db=15.0,
            dynamics_rise_time_ms=220.0,
            dynamics_recovery_ms=180.0,
            true_peak_headroom_db=5.0,
            summed_bus_evidence_available=True,
            summed_bus_role="low_end_bus",
            summed_bus_target_id="bus:3",
            summed_bus_true_peak_headroom_db=5.0,
            summed_bus_level_plane_error_lu=3.1,
        )
    )

    assert recommendation.status == "proposal_only"
    assert recommendation.selected_strategy.strategy_id == (
        "bass_sustain_summed_bus_rebalance_before_compression"
    )
    assert recommendation.selected_strategy.method_family == "gain_fader"
    assert recommendation.evidence["operator_explanation_sections"][
        "selection_policy"
    ] == "soundcheck_bass_sustain_summed_bus_rebalance_priority"
    params = recommendation.selected_strategy.parameters
    assert params["target_scope"] == "summed_low_end_bus_before_bass_compression"
    assert params["compression_candidate_strategy"] == (
        "bass_sustain_leveling_compression"
    )
    assert params["post_rebalance_compression_recheck_required"] is True
    assert params["bus_delta_db"] < 0.0
    assert params["source_delta_db"] == 0.0
    assert params["summed_bus_target_id"] == "bus:3"
    assert params["wing_osc_target_id_override"] == "bus:3"
    assert params["wing_osc_original_target_id"] == "ch:2"
    assert params["wing_osc_effective_target_id"] == "bus:3"
    _assert_gain_delta_readback_contract(
        recommendation.selected_strategy,
        source="bus_delta_db",
        target_surface="bus",
        expected_delta=params["bus_delta_db"],
        path_prefix="/bus/{bus}",
    )

    alternatives = {
        candidate.strategy_id: candidate for candidate in recommendation.alternatives
    }
    assert "summed_bus_rebalance_should_precede_bass_sustain_compression" in (
        alternatives["bass_sustain_leveling_compression"].blockers
    )
    tradeoffs = {
        row["strategy_id"]: row["why_not_selected"]
        for row in recommendation.evidence["strategy_search"]["alternative_tradeoffs"]
    }
    assert tradeoffs["bass_sustain_leveling_compression"] == (
        "low_end_bus_level_priority:"
        "rebalance_summed_bus_before_bass_compression"
    )
    assert recommendation.to_operator_proposal()["live_apply_allowed"] is False


def test_live_guitar_dynamics_prefers_pick_sustain_compression_without_makeup():
    recommendation = analyze_smart_dsp_strategy(
        _analysis(
            target_id="ch:11",
            source_role="electric_guitar",
            source_label="GTR L",
            mode="live",
            active_performance=True,
            problem_hints=("dynamic_control",),
            muddiness_score=0.0,
            estimated_lufs=-24.0,
            lra_proxy_db=13.0,
            dynamics_variance_score=0.72,
            crest_factor_db=20.0,
            dynamics_peak_to_median_db=16.0,
            dynamics_rise_time_ms=260.0,
            dynamics_recovery_ms=80.0,
            true_peak_headroom_db=5.0,
        )
    )

    assert recommendation.status == "proposal_only"
    assert recommendation.selected_strategy.strategy_id == (
        "guitar_pick_sustain_compression"
    )
    assert recommendation.selected_strategy.method_family == "dynamics"
    params = recommendation.selected_strategy.parameters
    assert params["source_family"] == "guitar"
    assert params["dynamics_parameter_source"] == "measured_envelope"
    assert params["ratio"] == 2.7
    parameter_review_values = params["live_dynamics_parameter_review_only_values"]
    assert parameter_review_values["wing_compressor_ratio_osc_value"] == "2.5"
    assert "wing_compressor_ratio_osc_value" not in params
    assert params["wing_compressor_ratio_value_type"] == "string_enum"
    assert params["wing_compressor_ratio_numeric_value"] == 2.5
    assert params["detector"] == "program"
    assert parameter_review_values["wing_compressor_detector_osc_value"] == "RMS"
    assert "wing_compressor_detector_osc_value" not in params
    assert params["wing_compressor_detector_value_type"] == "string_enum"
    assert params["wing_compressor_detector_original_value"] == "program"
    assert params["knee_db"] == 3.1
    assert params["compressor_knee_parameter_source"] == (
        "measured_envelope_variance_peak_to_median"
    )
    assert parameter_review_values["wing_compressor_knee_osc_value"] == 3
    assert "wing_compressor_knee_osc_value" not in params
    assert params["wing_compressor_knee_quantized_from"] == 3.1
    assert params["wing_compressor_knee_value_type"] == "integer"
    assert params["preserve_pick_attack"] is True
    assert params["makeup_gain_db"] == 0.0
    assert params["live_dynamics_conservative_proposal"] is True
    assert params["live_dynamics_parameter_write_forbidden"] is True
    assert params["live_dynamics_parameter_readback_required"] is True
    assert params["live_dynamics_parameter_apply_blocked_without_wing_proof"] is True
    assert params["live_dynamics_parameter_review_only_reason"] == (
        "live_dynamics_must_read_existing_compressor_state_and_operator_confirm_before_parameter_apply"
    )
    assert not any(key in params for key in WING_DYNAMICS_WRITE_PARAMETER_KEYS)
    assert params["wing_dynamics_selected_write_keys"] == []
    assert params["live_dynamics_stepwise_threshold_review_required"] is True
    assert (
        params["live_dynamics_max_threshold_step_db"]
        == LIVE_DYNAMICS_MAX_THRESHOLD_STEP_DB
    )
    assert (
        params["live_dynamics_max_target_gain_reduction_db"]
        == LIVE_DYNAMICS_MAX_TARGET_GAIN_REDUCTION_DB
    )
    assert params["live_dynamics_no_makeup_gain_required"] is True
    assert params["live_dynamics_makeup_gain_db_limit"] == 0.0
    assert params["live_dynamics_makeup_gain_write_forbidden"] is True
    assert params["live_dynamics_makeup_gain_review_only_db"] == 0.0
    assert "wing_compressor_makeup_gain_osc_value" not in params
    assert "wing_compressor_makeup_gain_osc_path_suffix" not in params
    assert "wing_compressor_makeup_gain_source" not in params
    assert "wing_compressor_makeup_gain_osc_value" not in (
        params["wing_dynamics_selected_write_keys"]
    )
    forbidden_live_dynamics_prefixes = (
        "wing_compressor_crossover_",
        "wing_compressor_sidechain_",
    )
    assert params["live_dynamics_minimal_write_set_required"] is True
    assert params["live_dynamics_crossover_write_forbidden"] is True
    assert params["live_dynamics_sidechain_write_forbidden"] is True
    assert params["live_dynamics_crossover_review_only_values"] == {
        "compressor_crossover_type": "OFF",
        "compressor_crossover_depth_db": 0.0,
        "compressor_crossover_frequency_hz": 1000.0,
        "compressor_crossover_parameter_source": (
            "wing_dynxo_full_band_compression_baseline_policy"
        ),
    }
    assert params["live_dynamics_sidechain_review_only_values"] == {
        "compressor_sidechain_filter_type": "HP12",
        "compressor_sidechain_frequency_hz": 90.0,
        "compressor_sidechain_q": 1.0,
        "compressor_sidechain_source": "SELF",
        "compressor_sidechain_tap": "IN",
        "compressor_sidechain_parameter_source": (
            "wing_dynsc_source_family_detector_filter_policy"
        ),
        "compressor_sidechain_evidence_frequency_hz": 0.0,
        "compressor_sidechain_low_end_pressure_score": 0.0,
        "compressor_sidechain_plosive_risk_score": 0.0,
        "compressor_sidechain_low_mid_bloom_score": 0.0,
    }
    assert not any(
        key.startswith(forbidden_live_dynamics_prefixes) for key in params
    )
    assert not any(
        key.startswith(forbidden_live_dynamics_prefixes)
        for key in params["wing_dynamics_selected_write_keys"]
    )
    assert (
        params["live_dynamics_gain_reduction_meter_readback_after_each_step_required"]
        is True
    )
    assert params["live_dynamics_previous_comp_model_readback_required"] is True
    assert params["live_dynamics_global_on_mix_write_forbidden"] is True
    assert params["live_dynamics_on_mix_readback_required"] is True
    assert params["live_dynamics_compressor_enable_write_forbidden"] is True
    assert params["live_dynamics_compressor_mix_write_forbidden"] is True
    assert (
        params["live_dynamics_requires_existing_compressor_enabled_readback"]
        is True
    )
    assert params["live_dynamics_compressor_on_review_only_value"] == (
        WING_COMPRESSOR_ON_MAX
    )
    assert params["live_dynamics_compressor_mix_review_only_value"] == 78.5
    assert params["live_dynamics_global_on_mix_review_only_reason"] == (
        "live_dynamics_must_read_existing_compressor_on_and_mix_before_parameter_apply"
    )
    assert "wing_compressor_on_osc_value" not in params
    assert "wing_compressor_mix_osc_value" not in params
    assert "wing_compressor_on_osc_value" not in params["wing_dynamics_selected_write_keys"]
    assert "wing_compressor_mix_osc_value" not in params["wing_dynamics_selected_write_keys"]
    assert params["live_dynamics_operator_rehearsal_required"] is True
    assert params["live_dynamics_live_promotion_status"] == (
        "blocked_until_real_wing_readback_operator_rehearsal_and_rollback"
    )
    assert params["live_dynamics_live_apply_blocked_without_wing_proof"] is True
    assert params["release_ms"] > params["attack_ms"]
    readback = recommendation.selected_strategy.future_wing_readback_contract[
        "dynamics_readback"
    ]
    assert readback["live_dynamics_conservative_proposal"] is True
    assert readback["live_dynamics_parameter_write_forbidden"] is True
    assert readback["live_dynamics_parameter_readback_required"] is True
    assert readback["live_dynamics_parameter_apply_blocked_without_wing_proof"] is True
    assert readback["live_dynamics_parameter_review_only_values"] == (
        parameter_review_values
    )
    assert readback["live_dynamics_parameter_review_only_reason"] == (
        params["live_dynamics_parameter_review_only_reason"]
    )
    assert not any(key in readback for key in WING_DYNAMICS_WRITE_PARAMETER_KEYS)
    assert readback["wing_dynamics_selected_write_keys"] == []
    assert readback["live_dynamics_stepwise_threshold_review_required"] is True
    assert (
        readback["live_dynamics_max_threshold_step_db"]
        == LIVE_DYNAMICS_MAX_THRESHOLD_STEP_DB
    )
    assert (
        readback["live_dynamics_max_target_gain_reduction_db"]
        == LIVE_DYNAMICS_MAX_TARGET_GAIN_REDUCTION_DB
    )
    assert readback["live_dynamics_no_makeup_gain_required"] is True
    assert readback["live_dynamics_makeup_gain_db_limit"] == 0.0
    assert readback["live_dynamics_makeup_gain_write_forbidden"] is True
    assert readback["live_dynamics_makeup_gain_review_only_db"] == 0.0
    assert "wing_compressor_makeup_gain_osc_value" not in readback
    assert "wing_compressor_makeup_gain_osc_value" not in (
        readback["wing_dynamics_selected_write_keys"]
    )
    assert readback["live_dynamics_minimal_write_set_required"] is True
    assert readback["live_dynamics_crossover_write_forbidden"] is True
    assert readback["live_dynamics_sidechain_write_forbidden"] is True
    assert readback["live_dynamics_crossover_review_only_values"] == (
        params["live_dynamics_crossover_review_only_values"]
    )
    assert readback["live_dynamics_sidechain_review_only_values"] == (
        params["live_dynamics_sidechain_review_only_values"]
    )
    assert not any(
        key.startswith(forbidden_live_dynamics_prefixes) for key in readback
    )
    assert not any(
        key.startswith(forbidden_live_dynamics_prefixes)
        for key in readback["wing_dynamics_selected_write_keys"]
    )
    assert (
        readback[
            "live_dynamics_gain_reduction_meter_readback_after_each_step_required"
        ]
        is True
    )
    assert readback["live_dynamics_previous_comp_model_readback_required"] is True
    assert readback["live_dynamics_global_on_mix_write_forbidden"] is True
    assert readback["live_dynamics_on_mix_readback_required"] is True
    assert readback["live_dynamics_compressor_enable_write_forbidden"] is True
    assert readback["live_dynamics_compressor_mix_write_forbidden"] is True
    assert (
        readback["live_dynamics_requires_existing_compressor_enabled_readback"]
        is True
    )
    assert readback["live_dynamics_compressor_on_review_only_value"] == (
        WING_COMPRESSOR_ON_MAX
    )
    assert readback["live_dynamics_compressor_mix_review_only_value"] == 78.5
    assert readback["live_dynamics_global_on_mix_review_only_reason"] == (
        "live_dynamics_must_read_existing_compressor_on_and_mix_before_parameter_apply"
    )
    assert "wing_compressor_on_osc_value" not in readback
    assert "wing_compressor_mix_osc_value" not in readback
    assert "wing_compressor_on_osc_value" not in (
        readback["wing_dynamics_selected_write_keys"]
    )
    assert "wing_compressor_mix_osc_value" not in (
        readback["wing_dynamics_selected_write_keys"]
    )
    assert readback["live_dynamics_operator_rehearsal_required"] is True
    assert readback["live_dynamics_live_promotion_status"] == (
        "blocked_until_real_wing_readback_operator_rehearsal_and_rollback"
    )
    assert readback["live_dynamics_live_apply_blocked_without_wing_proof"] is True
    proposal = recommendation.to_operator_proposal()
    assert proposal["selected_wing_osc_summary"]["wing_osc_apply_parameter_values"] == {}
    assert proposal["selected_wing_osc_readback_summary"][
        "wing_osc_readback_apply_parameter_values"
    ] == {}
    assert not any(
        key in proposal["selected_wing_osc_summary"]["wing_osc_apply_parameter_values"]
        for key in WING_DYNAMICS_WRITE_PARAMETER_KEYS
    )
    assert "wing_compressor_makeup_gain_osc_value" not in proposal[
        "selected_wing_osc_summary"
    ]["wing_osc_apply_parameter_values"]
    assert "wing_compressor_on_osc_value" not in proposal[
        "selected_wing_osc_summary"
    ]["wing_osc_apply_parameter_values"]
    assert "wing_compressor_mix_osc_value" not in proposal[
        "selected_wing_osc_summary"
    ]["wing_osc_apply_parameter_values"]
    assert not any(
        key.startswith(forbidden_live_dynamics_prefixes)
        for key in proposal["selected_wing_osc_summary"][
            "wing_osc_apply_parameter_values"
        ]
    )
    assert "wing_compressor_makeup_gain_osc_value" not in proposal[
        "selected_wing_osc_readback_summary"
    ]["wing_osc_readback_apply_parameter_values"]
    assert "wing_compressor_on_osc_value" not in proposal[
        "selected_wing_osc_readback_summary"
    ]["wing_osc_readback_apply_parameter_values"]
    assert "wing_compressor_mix_osc_value" not in proposal[
        "selected_wing_osc_readback_summary"
    ]["wing_osc_readback_apply_parameter_values"]
    assert not any(
        key.startswith(forbidden_live_dynamics_prefixes)
        for key in proposal["selected_wing_osc_readback_summary"][
            "wing_osc_readback_apply_parameter_values"
        ]
    )
    assert proposal["live_apply_allowed"] is False


def test_live_lead_guitar_dynamics_prefers_manual_ride_before_compression():
    recommendation = analyze_smart_dsp_strategy(
        _analysis(
            target_id="ch:11",
            source_role="electric_guitar",
            source_label="Lead GTR Solo",
            musical_role="lead",
            operator_intent="preserve lead guitar solo phrasing",
            mode="live",
            active_performance=True,
            problem_hints=("dynamic_control",),
            muddiness_score=0.0,
            estimated_lufs=-24.0,
            lra_proxy_db=13.0,
            dynamics_variance_score=0.72,
            crest_factor_db=20.0,
            dynamics_peak_to_median_db=16.0,
            dynamics_rise_time_ms=260.0,
            dynamics_recovery_ms=80.0,
            true_peak_headroom_db=5.0,
        )
    )

    assert recommendation.status == "proposal_only"
    assert recommendation.selected_strategy.strategy_id == (
        "manual_level_ride_before_compression"
    )
    assert recommendation.selected_strategy.method_family == "analysis_only"
    assert recommendation.evidence["operator_explanation_sections"][
        "selection_policy"
    ] == "live_guitar_lead_ride_priority"
    params = recommendation.selected_strategy.parameters
    assert params["source_family"] == "guitar"
    assert params["max_ride_delta_db"] == 1.5
    assert params["live_guitar_lead_ride_priority"] is True
    assert params["live_keys_swell_ride_priority"] is False
    assert params["live_vocal_dynamics_ride_priority"] is False
    assert params["required_context"] == (
        "lead_guitar_solo_manual_ride_bus_headroom_compressor_readback"
    )
    assert params["forbidden_without_wing_proof"] == (
        "live_guitar_lead_compressor_change"
    )
    assert params["live_dynamics_ride_hold_proposal"] is True
    assert params["live_dynamics_ride_profile"] == "lead_guitar_solo"
    assert params["live_dynamics_ride_previous_gain_readback_required"] is True
    assert params["live_dynamics_ride_previous_compressor_readback_required"] is True
    ride_readback = recommendation.selected_strategy.future_wing_readback_contract[
        "live_dynamics_ride_hold_readback"
    ]
    assert ride_readback["read_only"] is True
    assert ride_readback["live_dynamics_ride_profile"] == "lead_guitar_solo"
    assert ride_readback["allowed_live_write_scope_before_wing_proof"] == []
    assert "/ch/{channel}/fdr" in ride_readback["wing_gain_osc_path_templates"]
    assert "/ch/{channel}/dyn/thr24" in ride_readback[
        "wing_dynamics_osc_path_templates"
    ]
    alternatives = {
        candidate.strategy_id: candidate for candidate in recommendation.alternatives
    }
    assert alternatives["guitar_pick_sustain_compression"].blockers == (
        "live_guitar_lead_compression_requires_operator_review_and_wing_proof",
    )
    assert alternatives["gentle_channel_compression"].blockers == (
        "live_guitar_lead_compression_requires_operator_review_and_wing_proof",
        "live_dynamics_target_gr_exceeds_conservative_limit",
    )
    tradeoffs = {
        row["strategy_id"]: row
        for row in recommendation.evidence["strategy_search"]["alternative_tradeoffs"]
    }
    assert tradeoffs["guitar_pick_sustain_compression"]["why_not_selected"] == (
        "live_guitar_lead_ride_priority:"
        "preserve_solo_phrasing_before_compressor_change"
    )
    assert tradeoffs["gentle_channel_compression"]["why_not_selected"] == (
        "live_guitar_lead_ride_priority:"
        "preserve_solo_phrasing_before_compressor_change"
    )
    assert recommendation.to_operator_proposal()["live_apply_allowed"] is False


def test_keys_dynamics_prefers_slow_swell_leveling():
    recommendation = analyze_smart_dsp_strategy(
        _analysis(
            target_id="ch:15",
            source_role="keys",
            source_label="Keys L",
            problem_hints=("dynamic_control",),
            muddiness_score=0.0,
            estimated_lufs=-24.0,
            lra_proxy_db=13.0,
            dynamics_variance_score=0.72,
            crest_factor_db=20.0,
            true_peak_headroom_db=5.0,
        )
    )

    assert recommendation.status == "proposal_only"
    assert recommendation.selected_strategy.strategy_id == "keys_swell_leveling_compression"
    assert recommendation.selected_strategy.method_family == "dynamics"
    params = recommendation.selected_strategy.parameters
    assert params["source_family"] == "keys"
    assert params["preserve_pad_swell"] is True
    assert params["release_ms"] >= 300.0
    assert params["makeup_gain_db"] == 0.0
    assert recommendation.to_operator_proposal()["live_apply_allowed"] is False


def test_live_keys_swell_dynamics_prefers_manual_ride_before_compression():
    recommendation = analyze_smart_dsp_strategy(
        _analysis(
            target_id="ch:15",
            source_role="keys",
            source_label="Keys L",
            mode="live",
            active_performance=True,
            problem_hints=("dynamic_control",),
            muddiness_score=0.0,
            estimated_lufs=-24.0,
            lra_proxy_db=13.0,
            dynamics_variance_score=0.72,
            crest_factor_db=20.0,
            dynamics_peak_to_median_db=10.0,
            dynamics_rise_time_ms=380.0,
            dynamics_recovery_ms=900.0,
            true_peak_headroom_db=5.0,
        )
    )

    assert recommendation.status == "proposal_only"
    assert (
        recommendation.selected_strategy.strategy_id
        == "manual_level_ride_before_compression"
    )
    assert recommendation.selected_strategy.method_family == "analysis_only"
    assert recommendation.evidence["operator_explanation_sections"][
        "selection_policy"
    ] == "live_keys_swell_ride_priority"
    params = recommendation.selected_strategy.parameters
    assert params["source_family"] == "keys"
    assert params["live_keys_swell_ride_priority"] is True
    assert params["live_vocal_dynamics_ride_priority"] is False
    assert params["max_ride_delta_db"] == 1.5
    assert params["required_context"] == (
        "keys_swell_manual_ride_bus_headroom_compressor_readback"
    )
    assert params["forbidden_without_wing_proof"] == "keys_swell_compressor_change"
    assert params["live_dynamics_ride_hold_proposal"] is True
    assert params["live_dynamics_ride_profile"] == "keys_swell"
    assert params["live_dynamics_ride_previous_gain_readback_required"] is True
    assert params["live_dynamics_ride_previous_compressor_readback_required"] is True
    ride_readback = recommendation.selected_strategy.future_wing_readback_contract[
        "live_dynamics_ride_hold_readback"
    ]
    assert ride_readback["read_only"] is True
    assert ride_readback["live_dynamics_ride_profile"] == "keys_swell"
    assert ride_readback["allowed_live_write_scope_before_wing_proof"] == []
    assert "/ch/{channel}/fdr" in ride_readback["wing_gain_osc_path_templates"]
    assert "/ch/{channel}/dyn/thr24" in ride_readback[
        "wing_dynamics_osc_path_templates"
    ]

    alternatives = {
        candidate.strategy_id: candidate for candidate in recommendation.alternatives
    }
    assert alternatives["keys_swell_leveling_compression"].blockers == (
        "live_keys_swell_compression_requires_operator_review_and_wing_proof",
    )
    assert alternatives["gentle_channel_compression"].blockers == (
        "live_keys_swell_compression_requires_operator_review_and_wing_proof",
    )
    tradeoffs = {
        row["strategy_id"]: row
        for row in recommendation.evidence["strategy_search"]["alternative_tradeoffs"]
    }
    assert tradeoffs["keys_swell_leveling_compression"]["why_not_selected"] == (
        "live_keys_swell_ride_priority:preserve_swell_before_compressor_change"
    )
    assert tradeoffs["gentle_channel_compression"]["why_not_selected"] == (
        "live_keys_swell_ride_priority:preserve_swell_before_compressor_change"
    )
    assert recommendation.to_operator_proposal()["live_apply_allowed"] is False


def test_extreme_fx_monitor_risk_selects_safety_pullback_and_blocks_complex_edits():
    recommendation = analyze_smart_dsp_strategy(
        _analysis(
            target_id="fx:1",
            source_role="fx_return",
            problem_hints=("fx_intelligibility",),
            muddiness_score=0.0,
            fx_wetness_score=0.88,
            fx_return_headroom_db=0.8,
            monitor_feedback_risk=0.82,
        )
    )

    assert recommendation.status == "proposal_only"
    assert recommendation.selected_strategy.strategy_id == "fx_monitor_feedback_safety_pullback"
    assert recommendation.selected_strategy.parameters["send_delta_db"] == 0.0
    assert recommendation.selected_strategy.parameters["return_delta_db"] < 0.0
    assert recommendation.selected_strategy.parameters["return_mute_review_required"] is True
    assert recommendation.selected_strategy.parameters["monitor_feedback_risk"] == 0.82
    _assert_fx_return_fxmix_readback_contract(
        recommendation.selected_strategy,
        expected_delta=recommendation.selected_strategy.parameters["return_delta_db"],
        target_surface="fx",
        fx_slot_binding_required=False,
    )
    _assert_selected_fx_apply_values(
        recommendation,
        {"wing_fx_return_delta_db": -5.5},
    )

    reduce = [
        candidate
        for candidate in recommendation.alternatives
        if candidate.strategy_id == "reduce_fx_send_or_return"
    ][0]
    assert reduce.parameters["send_delta_db"] == 0.0
    assert reduce.parameters["return_delta_db"] < 0.0
    blocked_increase = [
        candidate
        for candidate in recommendation.alternatives
        if candidate.strategy_id == "increase_fx_depth"
    ][0]
    blocked_duck = [
        candidate
        for candidate in recommendation.alternatives
        if candidate.strategy_id == "duck_fx_return"
    ][0]
    blocked_delay = [
        candidate
        for candidate in recommendation.alternatives
        if candidate.strategy_id == "duck_delay_return_cleanup"
    ][0]
    assert "fx_increase_does_not_match_problem" in blocked_increase.blockers
    assert "fx_increase_monitor_or_return_risk" in blocked_increase.blockers
    assert "monitor_feedback_risk_blocks_fx_ducking" in blocked_duck.blockers
    assert "monitor_feedback_risk_blocks_delay_ducking" in blocked_delay.blockers


def test_fx_return_low_headroom_selects_pullback_without_wetness_or_tail_problem():
    recommendation = analyze_smart_dsp_strategy(
        _analysis(
            target_id="fx:1",
            source_role="fx_return",
            source_label="Vocal Verb Return",
            problem_hints=(),
            muddiness_score=0.0,
            fx_wetness_score=0.1,
            fx_decay_risk_score=0.1,
            fx_delay_smear_score=0.1,
            fx_return_headroom_db=1.2,
            monitor_feedback_risk=0.1,
        )
    )

    assert recommendation.status == "proposal_only"
    assert "fx_intelligibility" in recommendation.evidence["detected_problems"]
    assert recommendation.selected_strategy.strategy_id == (
        "fx_return_headroom_pullback_before_time_edits"
    )
    assert recommendation.selected_strategy.problem == "fx_return_headroom_risk"
    assert recommendation.evidence["operator_explanation_sections"][
        "selection_policy"
    ] == "fx_return_headroom_restoration_priority"
    params = recommendation.selected_strategy.parameters
    assert params["fx_return_headroom_db"] == 1.2
    assert params["target_fx_return_headroom_db"] == 2.5
    assert params["fx_return_headroom_deficit_db"] == 1.3
    assert params["return_delta_db"] < 0.0
    assert params["send_delta_db"] == 0.0
    assert params["return_headroom_restoration_first"] is True
    _assert_fx_return_fxmix_readback_contract(
        recommendation.selected_strategy,
        expected_delta=params["return_delta_db"],
        target_surface="fx",
        fx_slot_binding_required=False,
    )
    _assert_selected_fx_apply_values(
        recommendation,
        {"wing_fx_return_delta_db": -1.3},
    )
    assert recommendation.to_operator_proposal()["live_apply_allowed"] is False

    duck = [
        candidate
        for candidate in recommendation.alternatives
        if candidate.strategy_id == "duck_delay_return_cleanup"
    ][0]
    assert "fx_return_headroom_pullback_should_precede_delay_ducking" in duck.blockers


def test_source_fx_return_headroom_uses_send_only_until_return_slot_bound():
    recommendation = analyze_smart_dsp_strategy(
        _analysis(
            target_id="ch:5",
            source_role="lead_vocal",
            source_label="Lead Vox",
            problem_hints=("fx_intelligibility",),
            muddiness_score=0.0,
            fx_wetness_score=0.8,
            fx_decay_risk_score=0.7,
            fx_delay_smear_score=0.1,
            fx_return_headroom_db=1.2,
            monitor_feedback_risk=0.1,
        )
    )

    assert recommendation.status == "proposal_only"
    assert recommendation.selected_strategy.strategy_id == (
        "fx_return_headroom_pullback_before_time_edits"
    )
    assert recommendation.selected_strategy.method_family == "fx"
    assert recommendation.evidence["operator_explanation_sections"][
        "selection_policy"
    ] == "fx_return_headroom_restoration_priority"
    params = recommendation.selected_strategy.parameters
    assert params["send_delta_db"] == -1.3
    assert params["return_delta_db"] == 0.0
    assert params["fx_level_target_surface"] == "channel"
    assert params["fx_level_targeting_policy"] == (
        "source_send_only_until_fx_return_slot_resolved"
    )
    assert (
        params["fx_return_slot_binding_required_before_return_level_move"] is True
    )
    assert params["post_pullback_fx_return_level_review_db"] == -1.5
    assert params["post_pullback_fx_return_level_review_target_id_required"] is True
    assert params["post_pullback_fx_return_level_review_required_surface"] == "fx"
    assert params["wing_native_source_send_before_fx_return_slot_binding"] is True
    assert "wing_fx_return_delta_db" not in params
    _assert_fx_send_level_readback_contract(
        recommendation.selected_strategy,
        expected_delta=params["send_delta_db"],
        target_surface="channel",
        source_binding_required=False,
    )
    _assert_selected_fx_apply_values(
        recommendation,
        {"wing_fx_send_level_delta_db": -1.3},
    )
    proposal = recommendation.to_operator_proposal()
    assert "wing_fx_return_delta_db" not in proposal["selected_wing_osc_summary"][
        "wing_osc_absolute_value_readback_parameter_values"
    ]
    wing_osc = recommendation.selected_strategy.future_wing_readback_contract[
        "wing_osc_applicability"
    ]
    assert wing_osc["native_osc_supported"] is True
    assert wing_osc["native_osc_support_level"] == (
        "native_fx_level_osc_path_templates_declared"
    )
    fx_readback = recommendation.selected_strategy.future_wing_readback_contract[
        "fx_readback"
    ]
    assert fx_readback["fx_return_target_id"] == ""
    assert fx_readback["fx_return_target_surface"] == "unresolved"
    assert fx_readback["fx_return_slot_binding_required_before_return_level_move"] is True
    assert fx_readback["post_pullback_fx_return_level_review_db"] == -1.5
    assert fx_readback["post_pullback_fx_return_level_review_target_id_required"] is True
    assert fx_readback["post_pullback_fx_return_level_review_required_surface"] == "fx"
    assert fx_readback["wing_native_source_send_before_fx_return_slot_binding"] is True
    assert "wing_fx_return_level_mapping_required" not in fx_readback


def test_source_fx_return_headroom_uses_bound_return_fxmix_when_slot_known():
    recommendation = analyze_smart_dsp_strategy(
        _analysis(
            target_id="ch:5",
            fx_return_target_id="fx:1",
            source_role="lead_vocal",
            source_label="Lead Vox",
            problem_hints=("fx_intelligibility",),
            muddiness_score=0.0,
            fx_wetness_score=0.8,
            fx_decay_risk_score=0.7,
            fx_delay_smear_score=0.1,
            fx_return_headroom_db=1.2,
            monitor_feedback_risk=0.1,
        )
    )

    assert recommendation.status == "proposal_only"
    assert recommendation.selected_strategy.strategy_id == (
        "fx_return_headroom_pullback_before_time_edits"
    )
    assert recommendation.selected_strategy.method_family == "fx"
    params = recommendation.selected_strategy.parameters
    assert params["send_delta_db"] == -1.3
    assert params["return_delta_db"] == -1.5
    assert params["fx_return_target_id"] == "fx:1"
    assert params["fx_return_target_surface"] == "fx"
    assert params["fx_level_target_surface"] == "channel"
    assert params["fx_level_targeting_policy"] == (
        "source_send_uses_channel_aux_or_bus_send_level"
    )
    assert params["wing_fx_send_level_delta_db"] == -1.3
    assert params["wing_fx_send_level_target_surface"] == "channel"
    assert params["wing_fx_return_delta_db"] == -1.5
    assert params["wing_fx_return_level_target_surface"] == "fx"
    assert params["wing_fx_return_source_target_surface"] == "channel"
    assert params["wing_fx_return_target_id"] == "fx:1"
    assert params["wing_fx_return_target_surface"] == "fx"
    assert params["wing_fx_return_fx_slot_binding_required"] is False
    assert params["wing_fx_return_fx_slot_apply_ready"] is True
    assert params["soundcheck_fx_level_move_kinds"] == [
        "send_level",
        "return_level",
    ]
    assert params["soundcheck_fx_selected_write_keys"] == [
        "wing_fx_send_level_delta_db",
        "wing_fx_return_delta_db",
    ]
    assert params["soundcheck_fx_forbidden_write_keys"] == []
    assert "post_pullback_fx_return_level_review_db" not in params

    readback = recommendation.selected_strategy.future_wing_readback_contract
    wing_osc = readback["wing_osc_applicability"]
    assert wing_osc["native_osc_supported"] is True
    assert wing_osc["native_osc_support_level"] == (
        "native_fx_level_osc_path_templates_declared"
    )
    fx_readback = readback["fx_readback"]
    assert fx_readback["fx_return_target_id"] == "fx:1"
    assert fx_readback["fx_return_target_surface"] == "fx"
    assert fx_readback["wing_fx_send_level_mapping_required"] is True
    assert fx_readback["wing_fx_send_level_delta_db"] == -1.3
    assert fx_readback["wing_fx_return_level_mapping_required"] is True
    assert fx_readback["wing_fx_return_delta_db"] == -1.5
    assert fx_readback["wing_fx_return_level_target_surface"] == "fx"
    assert fx_readback["wing_fx_return_source_target_surface"] == "channel"
    assert fx_readback["wing_fx_return_target_id"] == "fx:1"
    assert fx_readback["wing_fx_return_target_surface"] == "fx"
    assert fx_readback["wing_fx_return_fx_slot_binding_required"] is False
    assert fx_readback["soundcheck_fx_level_move_kinds"] == [
        "send_level",
        "return_level",
    ]
    assert fx_readback["soundcheck_fx_selected_write_keys"] == [
        "wing_fx_send_level_delta_db",
        "wing_fx_return_delta_db",
    ]
    assert fx_readback["soundcheck_fx_forbidden_write_keys"] == []
    _assert_selected_fx_apply_values(
        recommendation,
        {
            "wing_fx_send_level_delta_db": -1.3,
            "wing_fx_return_delta_db": -1.5,
        },
    )


def test_long_fx_tail_selects_decay_filter_cleanup_with_delay_alternative():
    recommendation = analyze_smart_dsp_strategy(
        _analysis(
            target_id="fx:1",
            source_role="fx_return",
            problem_hints=("fx_intelligibility",),
            muddiness_score=0.0,
            fx_wetness_score=0.62,
            fx_decay_risk_score=0.92,
            fx_delay_smear_score=0.58,
            fx_tail_decay_seconds=2.4,
            fx_return_headroom_db=5.0,
            monitor_feedback_risk=0.18,
        )
    )

    assert recommendation.status == "proposal_only"
    assert recommendation.selected_strategy.strategy_id == "reduce_fx_send_or_return"
    assert recommendation.selected_strategy.method_family == "fx"
    params = recommendation.selected_strategy.parameters
    assert params["send_delta_db"] == 0.0
    assert params["return_delta_db"] < 0.0
    assert params["post_pullback_time_edit_review_required"] is True
    assert params["fx_decay_native_pullback_first"] is True
    assert params["fx_decay_parameter_source"] == "measured_tail_decay_role_target"
    assert params["measured_tail_decay_s"] == 2.4
    assert params["measured_delay_repeat_ms"] == 0.0
    assert (
        params["fx_level_pullback_parameter_source"]
        == "measured_tail_decay_native_level_pullback"
    )
    assert params["native_fx_level_pullback_db"] == 4.7
    assert params["native_fx_level_pullback_legacy_db"] == 4.3
    assert params["post_pullback_decay_review_s"] < 2.2
    assert params["post_pullback_return_hpf_review_hz"] > 220.0
    assert params["post_pullback_return_lpf_review_hz"] < 6800.0
    assert params["post_pullback_delay_feedback_review_percent"] < 12.0
    assert params["wing_native_fx_level_before_model_specific_decay_edit"] is True
    assert params["soundcheck_fx_bounded_level_proposal"] is True
    assert params["soundcheck_fx_native_level_osc_only_required"] is True
    assert params["soundcheck_fx_max_pullback_db"] == SOUNDCHECK_FX_MAX_PULLBACK_DB
    assert params["soundcheck_fx_max_boost_db"] == SOUNDCHECK_FX_MAX_BOOST_DB
    assert params["soundcheck_fx_level_move_kinds"] == ["return_level"]
    assert params["soundcheck_fx_no_model_or_time_parameter_write_required"] is True
    assert params["soundcheck_fx_followup_model_time_review_required"] is True
    assert params["soundcheck_fx_absolute_level_readback_required"] is True
    assert (
        params["soundcheck_fx_monitor_feedback_recheck_after_apply_required"]
        is True
    )
    assert (
        params["soundcheck_fx_return_headroom_recheck_after_apply_required"]
        is True
    )
    assert params["soundcheck_fx_operator_confirmation_required"] is True
    assert params["soundcheck_fx_readback_required"] is True
    assert params["soundcheck_fx_rollback_ready"] is True
    assert params["soundcheck_fx_live_promotion_status"] == (
        "blocked_until_real_wing_fx_level_readback_monitor_return_headroom_recheck_and_operator_approval"
    )
    assert params["soundcheck_fx_live_apply_blocked_without_wing_proof"] is True
    _assert_fx_return_fxmix_readback_contract(
        recommendation.selected_strategy,
        expected_delta=params["return_delta_db"],
        target_surface="fx",
        fx_slot_binding_required=False,
    )
    _assert_selected_fx_apply_values(
        recommendation,
        {"wing_fx_return_delta_db": -4.7},
    )
    assert recommendation.evidence["operator_explanation_sections"][
        "selection_policy"
    ] == "soundcheck_wing_native_osc_fx_level_pullback_priority"
    readback = recommendation.selected_strategy.future_wing_readback_contract
    _assert_common_readback_contract(readback)
    assert readback["wing_osc_applicability"]["native_osc_supported"] is True
    assert readback["wing_osc_applicability"]["native_osc_support_level"] == (
        "native_fx_level_osc_path_templates_declared"
    )
    assert readback["fx_readback"]["monitor_send_readback_required"] is True
    assert readback["fx_readback"]["fx_return_headroom_readback_required"] is True
    _assert_fx_model_time_followup_review_contract(
        recommendation.selected_strategy,
        expected_values={
            "post_pullback_return_hpf_review_hz": params[
                "post_pullback_return_hpf_review_hz"
            ],
            "post_pullback_return_lpf_review_hz": params[
                "post_pullback_return_lpf_review_hz"
            ],
            "post_pullback_decay_review_s": params[
                "post_pullback_decay_review_s"
            ],
            "post_pullback_delay_feedback_review_percent": params[
                "post_pullback_delay_feedback_review_percent"
            ],
        },
        expected_native_apply_fields=["wing_fx_return_delta_db"],
        expected_review_fields=WING_SOUNDCHECK_FX_MODEL_TIME_REVIEW_ONLY_FIELDS,
    )
    assert readback["fx_readback"]["soundcheck_fx_bounded_level_proposal"] is True
    assert (
        readback["fx_readback"]["soundcheck_fx_native_level_osc_only_required"]
        is True
    )
    assert readback["fx_readback"]["soundcheck_fx_level_move_kinds"] == [
        "return_level"
    ]
    assert (
        readback["fx_readback"]["soundcheck_fx_live_apply_blocked_without_wing_proof"]
        is True
    )
    assert {
        candidate.strategy_id for candidate in recommendation.alternatives
    } >= {"shorten_fx_decay_filter", "duck_delay_return_cleanup"}
    decay_edit = [
        candidate
        for candidate in recommendation.alternatives
        if candidate.strategy_id == "shorten_fx_decay_filter"
    ][0]
    assert decay_edit.parameters["decay_s"] < 2.2
    assert decay_edit.parameters["pre_delay_ms"] == 50.0
    assert decay_edit.parameters["fx_decay_parameter_source"] == (
        "measured_tail_decay_role_target"
    )
    assert decay_edit.parameters["fx_target_context"] == "vocal_reverb"
    assert decay_edit.parameters["fx_tail_density_score"] > 0.7
    assert decay_edit.parameters["decay_reduction_ratio"] > 0.3
    assert decay_edit.parameters["return_hpf_hz"] > 220.0
    assert decay_edit.parameters["return_lpf_hz"] < 6800.0
    assert decay_edit.parameters["delay_feedback_percent"] < 12.0
    assert decay_edit.future_wing_readback_contract["wing_osc_applicability"][
        "native_osc_supported"
    ] is False
    _assert_fx_model_specific_guard(
        decay_edit,
        {
            "decay_s",
            "pre_delay_ms",
            "return_hpf_hz",
            "return_lpf_hz",
            "delay_feedback_percent",
        },
    )
    fx_duck = [
        candidate
        for candidate in recommendation.alternatives
        if candidate.strategy_id == "duck_fx_return"
    ][0]
    fx_duck_params = fx_duck.parameters
    assert fx_duck_params["fx_ducking_parameter_source"] == "measured_fx_tail_decay"
    assert fx_duck_params["ducking_depth_db"] > 4.0
    assert fx_duck_params["release_ms"] > 600.0
    assert fx_duck_params["return_lpf_hz"] < 6800.0
    assert fx_duck_params["return_hpf_hz"] > 220.0
    assert fx_duck_params["measured_tail_decay_s"] == 2.4
    assert fx_duck_params["fx_decay_risk_score"] == 0.92
    _assert_fx_model_specific_guard(
        fx_duck,
        {
            "ducking_depth_db",
            "return_hpf_hz",
            "return_lpf_hz",
        },
    )


def test_operator_owned_hall_fx_model_prefers_level_pullback_before_model_edit():
    recommendation = analyze_smart_dsp_strategy(
        _analysis(
            target_id="ch:5",
            source_role="lead_vocal",
            source_label="Lead Vox",
            problem_hints=("muddy", "fx_intelligibility"),
            operator_intent=(
                "soundcheck vocal muddy from operator-owned HALL reverb wash"
            ),
            muddiness_score=0.62,
            low_end_excess_score=0.22,
            spectral_tilt_score=0.22,
            level_plane_error_lu=0.3,
            fx_wetness_score=0.82,
            fx_decay_risk_score=0.86,
            fx_delay_smear_score=0.22,
            fx_tail_decay_seconds=4.8,
            fx_return_headroom_db=4.5,
            fx_return_target_id="fx:2",
            current_wing_fx_model="HALL",
            current_wing_fx_model_readback_available=True,
            current_wing_fx_operator_owned=True,
            monitor_feedback_risk=0.14,
        )
    )

    assert recommendation.status == "proposal_only"
    assert recommendation.selected_strategy.strategy_id == "reduce_fx_send_or_return"
    assert recommendation.selected_strategy.method_family == "fx"
    assert recommendation.evidence["operator_explanation_sections"][
        "selection_policy"
    ] == "soundcheck_wing_native_osc_fx_level_pullback_priority"
    _assert_selected_fx_apply_values(
        recommendation,
        {"wing_fx_send_level_delta_db": -4.8},
    )

    decay_edit = [
        candidate
        for candidate in recommendation.alternatives
        if candidate.strategy_id == "shorten_fx_decay_filter"
    ][0]
    params = decay_edit.parameters
    assert decay_edit.blockers == (WING_FX_OPERATOR_OWNED_MODEL_BLOCKER,)
    assert params["fx_model_specific_parameter_policy"] == (
        "current_wing_fx_model_operator_owned_review_only"
    )
    assert params["wing_fx_model_parameter_current_model"] == "HALL"
    assert params["wing_fx_model_parameter_current_model_full_coverage"] is True
    assert (
        params["wing_fx_model_parameter_current_model_operator_owned_blocks_apply"]
        is True
    )
    assert params["wing_fx_model_parameter_operator_owned_model_blocks_apply"] is True
    assert params["wing_fx_model_parameter_apply_ready_after_model_readback"] is False
    assert params["wing_fx_model_parameter_path_pattern_apply_ready"] is False
    assert params["wing_fx_model_parameter_blocker"] == (
        WING_FX_OPERATOR_OWNED_MODEL_BLOCKER
    )
    assert params["wing_fx_model_parameter_operator_owned_blocker"] == (
        WING_FX_OPERATOR_OWNED_MODEL_BLOCKER
    )
    assert params["wing_fx_model_parameter_operations"] == []
    applicability = decay_edit.future_wing_readback_contract[
        "wing_osc_applicability"
    ]
    assert applicability["native_osc_supported"] is False
    assert applicability["native_osc_support_level"] == (
        "fx_operator_owned_model_edit_requires_review"
    )
    assert applicability["fx_model_specific_operator_owned_blocks_apply"] is True
    assert applicability["fx_model_specific_operator_owned_blocker"] == (
        WING_FX_OPERATOR_OWNED_MODEL_BLOCKER
    )
    assert applicability["unsupported_strategy_reason"] == (
        "fx_model_parameters_are_mapped_but_current_wing_fx_slot_is_operator_owned"
    )
    assert recommendation.to_operator_proposal()["live_apply_allowed"] is False


def test_soundcheck_fx_monitor_risk_prefers_pullback_before_decay_or_ducking():
    recommendation = analyze_smart_dsp_strategy(
        _analysis(
            target_id="fx:1",
            source_role="fx_return",
            problem_hints=("fx_intelligibility",),
            muddiness_score=0.0,
            fx_wetness_score=0.62,
            fx_decay_risk_score=0.92,
            fx_delay_smear_score=0.58,
            fx_tail_decay_seconds=2.4,
            fx_return_headroom_db=5.0,
            monitor_feedback_risk=0.62,
        )
    )

    assert recommendation.status == "proposal_only"
    assert recommendation.selected_strategy.strategy_id == "reduce_fx_send_or_return"
    assert recommendation.selected_strategy.method_family == "fx"
    assert recommendation.evidence["operator_explanation_sections"][
        "selection_policy"
    ] == "soundcheck_fx_monitor_feedback_pullback_priority"
    params = recommendation.selected_strategy.parameters
    assert params["send_delta_db"] == 0.0
    assert params["return_delta_db"] < 0.0
    assert params["monitor_feedback_risk"] == 0.62
    assert params["monitor_safety_pullback_first"] is True
    assert (
        params["fx_level_pullback_parameter_source"]
        == "measured_tail_decay_native_level_pullback"
    )
    assert params["native_fx_level_pullback_db"] == 4.7
    assert params["native_fx_level_pullback_legacy_db"] == 4.3
    assert params["measured_tail_decay_s"] == 2.4
    assert params["measured_delay_repeat_ms"] == 0.0
    _assert_selected_fx_apply_values(
        recommendation,
        {"wing_fx_return_delta_db": -4.7},
    )

    decay_edit = [
        candidate
        for candidate in recommendation.alternatives
        if candidate.strategy_id == "shorten_fx_decay_filter"
    ][0]
    fx_ducking = [
        candidate
        for candidate in recommendation.alternatives
        if candidate.strategy_id == "duck_fx_return"
    ][0]
    delay_ducking = [
        candidate
        for candidate in recommendation.alternatives
        if candidate.strategy_id == "duck_delay_return_cleanup"
    ][0]
    assert "monitor_feedback_risk_pullback_should_precede_decay_edit" in (
        decay_edit.blockers
    )
    assert "monitor_feedback_risk_blocks_fx_ducking" in fx_ducking.blockers
    assert "monitor_feedback_risk_blocks_delay_ducking" in delay_ducking.blockers
    _assert_fx_model_specific_guard(
        decay_edit,
        {
            "decay_s",
            "pre_delay_ms",
            "return_hpf_hz",
            "return_lpf_hz",
            "delay_feedback_percent",
        },
    )
    _assert_fx_model_specific_guard(
        fx_ducking,
        {
            "ducking_depth_db",
            "return_hpf_hz",
            "return_lpf_hz",
        },
    )
    _assert_fx_model_specific_guard(
        delay_ducking,
        {
            "ducking_depth_db",
            "delay_feedback_percent",
            "delay_low_pass_hz",
        },
    )
    tradeoff_reasons = {
        row["strategy_id"]: row["why_not_selected"]
        for row in recommendation.evidence["strategy_search"][
            "alternative_tradeoffs"
        ]
    }
    assert tradeoff_reasons["shorten_fx_decay_filter"] == (
        "fx_safety_pullback_priority:fx_reduction_precedes_ducking_or_time_edits"
    )
    _assert_proposal_denial(recommendation.to_operator_proposal())


def test_delay_smear_fx_selects_delay_ducking_over_wet_pullback():
    recommendation = analyze_smart_dsp_strategy(
        _analysis(
            target_id="fx:1",
            source_role="fx_return",
            problem_hints=("fx_intelligibility",),
            muddiness_score=0.0,
            fx_wetness_score=0.5,
            fx_decay_risk_score=0.2,
            fx_delay_smear_score=0.95,
            fx_return_headroom_db=5.0,
            monitor_feedback_risk=0.1,
        )
    )

    assert recommendation.status == "proposal_only"
    assert recommendation.selected_strategy.strategy_id == "reduce_fx_send_or_return"
    assert recommendation.selected_strategy.method_family == "fx"
    params = recommendation.selected_strategy.parameters
    assert params["send_delta_db"] == 0.0
    assert params["return_delta_db"] < 0.0
    assert params["post_pullback_time_edit_review_required"] is True
    assert params["fx_delay_smear_native_pullback_first"] is True
    assert params["fx_delay_smear_parameter_source"] == "fx_delay_smear_score_fallback"
    assert params["measured_delay_repeat_ms"] == 0.0
    assert params["measured_tail_decay_s"] == 0.0
    assert (
        params["fx_level_pullback_parameter_source"]
        == "delay_smear_score_native_level_pullback"
    )
    assert params["native_fx_level_pullback_db"] == 3.0
    assert params["native_fx_level_pullback_legacy_db"] == 3.0
    assert params["post_pullback_delay_time_review_ms"] == 250.0
    assert params["post_pullback_delay_feedback_review_percent"] == 9.5
    assert params["post_pullback_delay_low_pass_review_hz"] == 5800.0
    assert params["post_pullback_delay_ducking_review_db"] == 4.5
    assert params["post_pullback_delay_ducking_release_review_ms"] == 387.5
    assert params["wing_native_fx_level_before_model_specific_delay_edit"] is True
    _assert_selected_fx_apply_values(
        recommendation,
        {"wing_fx_return_delta_db": -3.0},
    )
    _assert_fx_model_time_followup_review_contract(
        recommendation.selected_strategy,
        expected_values={
            "post_pullback_delay_time_review_ms": 250.0,
            "post_pullback_delay_feedback_review_percent": 9.5,
            "post_pullback_delay_low_pass_review_hz": 5800.0,
            "post_pullback_delay_ducking_review_db": 4.5,
            "post_pullback_delay_ducking_release_review_ms": 387.5,
            "post_pullback_return_hpf_review_hz": 180.0,
            "post_pullback_return_lpf_review_hz": 7600.0,
        },
        expected_native_apply_fields=["wing_fx_return_delta_db"],
        expected_review_fields=WING_SOUNDCHECK_FX_MODEL_TIME_REVIEW_ONLY_FIELDS,
    )
    assert recommendation.evidence["operator_explanation_sections"][
        "selection_policy"
    ] == "soundcheck_wing_native_osc_fx_level_pullback_priority"
    ducking = [
        candidate
        for candidate in recommendation.alternatives
        if candidate.strategy_id == "duck_delay_return_cleanup"
    ][0]
    ducking_params = ducking.parameters
    assert ducking_params["fx_delay_parameter_source"] == "fx_delay_smear_score_fallback"
    assert ducking_params["ducking_depth_db"] > 0.0
    assert ducking_params["delay_feedback_percent"] < 12.0
    assert ducking_params["delay_low_pass_hz"] == 5800.0
    assert ducking.scores["precision"] > recommendation.selected_strategy.scores["precision"]
    assert ducking.future_wing_readback_contract["wing_osc_applicability"][
        "native_osc_supported"
    ] is False
    _assert_fx_model_specific_guard(
        ducking,
        {
            "ducking_depth_db",
            "delay_feedback_percent",
            "delay_low_pass_hz",
        },
    )


def test_fx_return_low_headroom_prefers_pullback_before_time_or_ducking_edits():
    recommendation = analyze_smart_dsp_strategy(
        _analysis(
            target_id="fx:1",
            source_role="fx_return",
            problem_hints=("fx_intelligibility",),
            muddiness_score=0.0,
            fx_wetness_score=0.52,
            fx_decay_risk_score=0.2,
            fx_delay_smear_score=0.95,
            fx_tail_decay_seconds=1.2,
            fx_delay_repeat_ms=130.0,
            fx_return_headroom_db=1.7,
            monitor_feedback_risk=0.2,
        )
    )

    assert recommendation.status == "proposal_only"
    assert (
        recommendation.selected_strategy.strategy_id
        == "fx_return_headroom_pullback_before_time_edits"
    )
    assert recommendation.selected_strategy.method_family == "fx"
    params = recommendation.selected_strategy.parameters
    assert params["return_delta_db"] == -1.0
    assert params["send_delta_db"] == 0.0
    assert params["fx_return_headroom_db"] == 1.7
    assert params["target_fx_return_headroom_db"] == 2.5
    assert params["fx_return_headroom_deficit_db"] == 0.8
    assert params["fx_return_headroom_parameter_source"] == "measured_fx_return_headroom"
    assert params["return_headroom_restoration_first"] is True
    assert params["post_pullback_delay_ducking_review_required"] is True
    assert params["post_pullback_decay_filter_review_required"] is True
    assert params["post_pullback_fx_ducking_review_required"] is True
    assert params["post_pullback_return_hpf_review_hz"] > 180.0
    assert params["post_pullback_return_lpf_review_hz"] < 7600.0
    _assert_fx_return_fxmix_readback_contract(
        recommendation.selected_strategy,
        expected_delta=params["return_delta_db"],
        target_surface="fx",
        fx_slot_binding_required=False,
    )
    _assert_selected_fx_apply_values(
        recommendation,
        {"wing_fx_return_delta_db": -1.0},
    )
    assert recommendation.evidence["operator_explanation_sections"][
        "selection_policy"
    ] == "fx_return_headroom_restoration_priority"
    readback = recommendation.selected_strategy.future_wing_readback_contract
    _assert_common_readback_contract(readback)
    assert readback["fx_readback"]["fx_return_headroom_readback_required"] is True

    delay_ducking = [
        candidate
        for candidate in recommendation.alternatives
        if candidate.strategy_id == "duck_delay_return_cleanup"
    ][0]
    decay_edit = [
        candidate
        for candidate in recommendation.alternatives
        if candidate.strategy_id == "shorten_fx_decay_filter"
    ][0]
    fx_ducking = [
        candidate
        for candidate in recommendation.alternatives
        if candidate.strategy_id == "duck_fx_return"
    ][0]
    assert "fx_return_headroom_pullback_should_precede_delay_ducking" in (
        delay_ducking.blockers
    )
    assert "fx_return_headroom_pullback_should_precede_decay_edit" in (
        decay_edit.blockers
    )
    assert "fx_return_headroom_pullback_should_precede_fx_ducking" in (
        fx_ducking.blockers
    )
    tradeoff_reasons = {
        row["strategy_id"]: row["why_not_selected"]
        for row in recommendation.evidence["strategy_search"][
            "alternative_tradeoffs"
        ]
    }
    assert tradeoff_reasons["duck_delay_return_cleanup"] == (
        "fx_return_headroom_priority:"
        "restore_return_headroom_before_ducking_decay_or_delay_edits"
    )


def test_live_delay_smear_fx_prefers_pullback_before_ducking():
    recommendation = analyze_smart_dsp_strategy(
        _analysis(
            target_id="fx:1",
            source_role="fx_return",
            mode="live",
            problem_hints=("fx_intelligibility",),
            muddiness_score=0.0,
            fx_wetness_score=0.5,
            fx_decay_risk_score=0.2,
            fx_delay_smear_score=0.95,
            fx_return_headroom_db=5.0,
            monitor_feedback_risk=0.1,
            active_performance=True,
        )
    )

    assert recommendation.status == "proposal_only"
    assert recommendation.selected_strategy.strategy_id == "reduce_fx_send_or_return"
    assert recommendation.selected_strategy.method_family == "fx"
    assert recommendation.evidence["operator_explanation_sections"][
        "selection_policy"
    ] == "live_fx_safety_pullback_priority"
    params = recommendation.selected_strategy.parameters
    assert params["send_delta_db"] == 0.0
    assert params["return_delta_db"] < 0.0
    assert (
        params["fx_level_pullback_parameter_source"]
        == "delay_smear_score_native_level_pullback"
    )
    assert params["native_fx_level_pullback_db"] == 3.0
    assert params["native_fx_level_pullback_legacy_db"] == 3.0
    assert params["measured_tail_decay_s"] == 0.0
    assert params["measured_delay_repeat_ms"] == 0.0
    assert params["live_fx_level_pullback_conservative_cap_db"] == 3.0
    assert params["live_fx_pullback_only_proposal"] is True
    assert params["live_fx_level_reduction_only_required"] is True
    assert params["live_fx_max_pullback_db"] == LIVE_FX_MAX_PULLBACK_DB
    assert params["live_fx_max_boost_db"] == LIVE_FX_MAX_BOOST_DB
    assert params["live_fx_native_level_osc_only_required"] is True
    assert params["live_fx_no_model_or_time_parameter_write_required"] is True
    assert params["live_fx_absolute_level_readback_required"] is True
    assert params["live_fx_monitor_feedback_recheck_after_apply_required"] is True
    assert params["live_fx_return_headroom_recheck_after_apply_required"] is True
    assert params["live_fx_operator_rehearsal_required"] is True
    assert params["live_fx_live_promotion_status"] == (
        "blocked_until_real_wing_fx_level_readback_monitor_recheck_and_rollback"
    )
    assert params["live_fx_live_apply_blocked_without_wing_proof"] is True
    fx_readback = recommendation.selected_strategy.future_wing_readback_contract[
        "fx_readback"
    ]
    assert fx_readback["live_fx_pullback_only_proposal"] is True
    assert fx_readback["live_fx_level_reduction_only_required"] is True
    assert fx_readback["live_fx_max_pullback_db"] == LIVE_FX_MAX_PULLBACK_DB
    assert fx_readback["live_fx_max_boost_db"] == LIVE_FX_MAX_BOOST_DB
    assert fx_readback["live_fx_native_level_osc_only_required"] is True
    assert fx_readback["live_fx_no_model_or_time_parameter_write_required"] is True
    assert fx_readback["live_fx_absolute_level_readback_required"] is True
    assert fx_readback["live_fx_monitor_feedback_recheck_after_apply_required"] is True
    assert fx_readback["live_fx_return_headroom_recheck_after_apply_required"] is True
    assert fx_readback["live_fx_operator_rehearsal_required"] is True
    assert fx_readback["live_fx_live_promotion_status"] == (
        "blocked_until_real_wing_fx_level_readback_monitor_recheck_and_rollback"
    )
    assert fx_readback["live_fx_live_apply_blocked_without_wing_proof"] is True
    _assert_fx_model_time_followup_review_contract(
        recommendation.selected_strategy,
        expected_values={
            "post_pullback_delay_time_review_ms": 250.0,
            "post_pullback_delay_feedback_review_percent": 9.5,
            "post_pullback_delay_low_pass_review_hz": 5800.0,
            "post_pullback_delay_ducking_review_db": 4.5,
            "post_pullback_delay_ducking_release_review_ms": 387.5,
            "post_pullback_return_hpf_review_hz": 180.0,
            "post_pullback_return_lpf_review_hz": 7600.0,
        },
        expected_native_apply_fields=["wing_fx_return_delta_db"],
        expected_review_fields=WING_FX_POST_PULLBACK_MODEL_TIME_REVIEW_FIELDS,
    )
    _assert_selected_fx_apply_values(
        recommendation,
        {"wing_fx_return_delta_db": -3.0},
    )
    ducking = [
        candidate
        for candidate in recommendation.alternatives
        if candidate.strategy_id == "duck_delay_return_cleanup"
    ][0]
    assert ducking.scores["precision"] > recommendation.selected_strategy.scores["precision"]
    _assert_fx_model_specific_guard(
        ducking,
        {
            "ducking_depth_db",
            "delay_feedback_percent",
            "delay_low_pass_hz",
        },
    )
    tradeoff = [
        item
        for item in recommendation.evidence["strategy_search"][
            "alternative_tradeoffs"
        ]
        if item["strategy_id"] == "duck_delay_return_cleanup"
    ][0]
    assert tradeoff["why_not_selected"] == (
        "fx_safety_pullback_priority:"
        "fx_reduction_precedes_ducking_or_time_edits"
    )
    _assert_proposal_denial(recommendation.to_operator_proposal())


def test_soundcheck_pan_phase_failure_reduces_width_instead_of_center_hold():
    recommendation = analyze_smart_dsp_strategy(
        _analysis(
            target_id="ch:12",
            source_role="electric_guitar",
            problem_hints=("pan_width",),
            muddiness_score=0.0,
            stereo_width_request=45.0,
            phase_correlation=0.21,
            mono_sum_delta_db=-5.5,
        )
    )

    assert recommendation.status == "proposal_only"
    assert recommendation.selected_strategy.strategy_id == "mono_compatibility_width_reduction"
    assert recommendation.selected_strategy.problem == "phase_mono_risk"
    assert recommendation.evidence["operator_explanation_sections"][
        "selection_policy"
    ] == "soundcheck_phase_mono_safety_priority"
    params = recommendation.selected_strategy.parameters
    assert params["target_width_percent"] == 35.0
    assert recommendation.selected_strategy.parameters["target_pan_percent"] == 0.0
    assert params["wing_pan_osc_path_suffix"] == "pan"
    assert params["wing_pan_osc_value"] == 0.0
    assert params["wing_width_osc_path_suffix"] == "wid"
    assert params["wing_width_osc_value"] == 35.0
    assert params["soundcheck_pan_width_bounded_move_proposal"] is True
    assert params["soundcheck_pan_width_move_kinds"] == ["pan", "width"]
    assert params["soundcheck_pan_width_phase_mono_recheck_required"] is True
    assert params["soundcheck_pan_width_target_surface"] == "channel"
    assert params["soundcheck_pan_width_identity_confirmation_scope"] == "source"
    assert params["soundcheck_pan_width_source_identity_required"] is True
    assert params["soundcheck_pan_width_output_feed_identity_required"] is False
    assert params["soundcheck_pan_width_readback_required"] is True
    assert params["soundcheck_pan_width_rollback_ready"] is True
    assert params["soundcheck_pan_width_live_apply_blocked_without_wing_proof"] is True
    readback = recommendation.selected_strategy.future_wing_readback_contract
    assert readback["pan_width_readback"]["wing_pan_mapping_required"] is True
    assert readback["pan_width_readback"]["wing_width_mapping_required"] is True
    assert (
        readback["pan_width_readback"]["soundcheck_pan_width_bounded_move_proposal"]
        is True
    )
    assert (
        readback["pan_width_readback"]["soundcheck_pan_width_move_kinds"]
        == ["pan", "width"]
    )
    assert (
        readback["pan_width_readback"][
            "soundcheck_pan_width_live_apply_blocked_without_wing_proof"
        ]
        is True
    )
    assert readback["pan_width_readback"]["read_before_paths"] == ["pan", "wid"]
    assert readback["pan_width_readback"]["read_after_paths"] == ["pan", "wid"]
    _assert_selected_pan_width_apply_values(
        recommendation,
        {
            "wing_pan_osc_value": 0.0,
            "wing_width_osc_value": 35.0,
        },
    )
    unsafe_width = [
        candidate
        for candidate in recommendation.alternatives
        if candidate.strategy_id == "phase_safe_pan_target"
    ][0]
    assert "phase_correlation_blocks_width" in unsafe_width.blockers
    assert "mono_sum_loss_blocks_width" in unsafe_width.blockers
    hold = [
        candidate
        for candidate in recommendation.alternatives
        if candidate.strategy_id == "center_hold_for_phase_safety"
    ][0]
    assert hold.problem == "pan_width"


def test_soundcheck_pan_request_calculates_bounded_delta_when_phase_safe():
    recommendation = analyze_smart_dsp_strategy(
        _analysis(
            target_id="ch:12",
            source_role="electric_guitar",
            problem_hints=("pan_width",),
            muddiness_score=0.0,
            stereo_width_request=35.0,
            current_pan_percent=5.0,
            phase_correlation=0.92,
            mono_sum_delta_db=-0.6,
        )
    )

    assert recommendation.status == "proposal_only"
    assert recommendation.selected_strategy.strategy_id == "phase_safe_pan_target"
    params = recommendation.selected_strategy.parameters
    assert params["target_pan_percent"] == 35.0
    assert params["current_pan_percent"] == 5.0
    assert params["pan_delta_percent"] == 30.0
    assert params["bounded_delta_percent"] == 30.0
    assert params["pan_scope"] == "source_role_stereo_placement"
    assert params["soundcheck_pan_width_bounded_move_proposal"] is True
    assert params["soundcheck_pan_width_native_osc_only_required"] is True
    assert params["soundcheck_pan_max_delta_percent"] == SOUNDCHECK_PAN_MAX_DELTA_PERCENT
    assert (
        params["soundcheck_width_max_widen_percent"]
        == SOUNDCHECK_WIDTH_MAX_WIDEN_PERCENT
    )
    assert (
        params["soundcheck_width_max_reduction_percent"]
        == SOUNDCHECK_WIDTH_MAX_REDUCTION_PERCENT
    )
    assert (
        params["soundcheck_width_min_target_percent"]
        == SOUNDCHECK_WIDTH_MIN_TARGET_PERCENT
    )
    assert params["wing_pan_width_osc_protocol_version"] == WING_PAN_WIDTH_OSC_PROTOCOL_VERSION
    assert params["wing_pan_width_osc_protocol_source"] == WING_PAN_WIDTH_OSC_PROTOCOL_SOURCE
    assert params["wing_pan_width_requires_real_wing_fw_3_0_5_readback"] is True
    assert params["soundcheck_pan_width_move_kinds"] == ["pan"]
    assert params["soundcheck_pan_width_apply_only_selected_write_keys"] is True
    assert params["soundcheck_pan_width_selected_write_keys"] == ["wing_pan_osc_value"]
    assert params["soundcheck_pan_width_forbidden_write_keys"] == [
        "wing_width_osc_value"
    ]
    assert params["soundcheck_pan_width_phase_mono_recheck_required"] is True
    assert params["soundcheck_pan_width_source_identity_required"] is True
    assert params["soundcheck_pan_width_operator_confirmation_required"] is True
    assert params["soundcheck_pan_width_readback_required"] is True
    assert params["soundcheck_pan_width_rollback_ready"] is True
    assert params["soundcheck_pan_width_live_promotion_status"] == (
        "blocked_until_real_wing_pan_width_readback_phase_mono_recheck_and_operator_approval"
    )
    assert params["soundcheck_pan_width_live_apply_blocked_without_wing_proof"] is True
    assert params["wing_pan_osc_path_suffix"] == "pan"
    assert params["wing_pan_osc_value"] == 35.0
    readback = recommendation.selected_strategy.future_wing_readback_contract
    assert readback["wing_osc_applicability"]["native_osc_supported"] is True
    assert "/ch/{channel}/pan" in readback["pan_width_readback"][
        "wing_osc_path_templates"
    ]
    assert "/ch/{channel}/wid" in readback["pan_width_readback"][
        "wing_osc_path_templates"
    ]
    assert readback["pan_width_readback"]["wing_pan_mapping_required"] is True
    assert readback["pan_width_readback"]["wing_pan_osc_path_suffix"] == "pan"
    assert readback["pan_width_readback"]["wing_pan_osc_value"] == 35.0
    assert (
        readback["pan_width_readback"]["wing_pan_width_osc_protocol_version"]
        == WING_PAN_WIDTH_OSC_PROTOCOL_VERSION
    )
    assert (
        readback["pan_width_readback"]["wing_pan_width_osc_protocol_source"]
        == WING_PAN_WIDTH_OSC_PROTOCOL_SOURCE
    )
    assert (
        readback["pan_width_readback"][
            "wing_pan_width_requires_real_wing_fw_3_0_5_readback"
        ]
        is True
    )
    assert (
        readback["pan_width_readback"]["soundcheck_pan_width_bounded_move_proposal"]
        is True
    )
    assert (
        readback["pan_width_readback"]["soundcheck_pan_width_native_osc_only_required"]
        is True
    )
    assert (
        readback["pan_width_readback"]["soundcheck_pan_max_delta_percent"]
        == SOUNDCHECK_PAN_MAX_DELTA_PERCENT
    )
    assert (
        readback["pan_width_readback"]["soundcheck_width_max_widen_percent"]
        == SOUNDCHECK_WIDTH_MAX_WIDEN_PERCENT
    )
    assert (
        readback["pan_width_readback"]["soundcheck_width_max_reduction_percent"]
        == SOUNDCHECK_WIDTH_MAX_REDUCTION_PERCENT
    )
    assert readback["pan_width_readback"]["soundcheck_pan_width_move_kinds"] == ["pan"]
    assert (
        readback["pan_width_readback"][
            "soundcheck_pan_width_apply_only_selected_write_keys"
        ]
        is True
    )
    assert readback["pan_width_readback"]["soundcheck_pan_width_selected_write_keys"] == [
        "wing_pan_osc_value"
    ]
    assert readback["pan_width_readback"]["soundcheck_pan_width_forbidden_write_keys"] == [
        "wing_width_osc_value"
    ]
    assert (
        readback["pan_width_readback"][
            "soundcheck_pan_width_live_apply_blocked_without_wing_proof"
        ]
        is True
    )
    assert readback["pan_width_readback"]["read_after_paths"] == ["pan", "wid"]
    _assert_selected_pan_width_apply_values(
        recommendation,
        {"wing_pan_osc_value": 35.0},
    )
    assert recommendation.to_operator_proposal()["live_apply_allowed"] is False


def test_operator_owned_pan_width_blocks_soundcheck_pan_parameter_apply():
    recommendation = analyze_smart_dsp_strategy(
        _analysis(
            target_id="ch:12",
            source_role="electric_guitar",
            problem_hints=("pan_width",),
            muddiness_score=0.0,
            stereo_width_request=35.0,
            current_pan_percent=5.0,
            phase_correlation=0.92,
            mono_sum_delta_db=-0.6,
            current_wing_pan_width_operator_owned=True,
        )
    )

    assert recommendation.status == "analysis_only"
    assert recommendation.selected_strategy.strategy_id == (
        "hold_soundcheck_operator_owned_pan_width_for_review"
    )
    assert recommendation.selected_strategy.method_family == "analysis_only"
    selected_params = recommendation.selected_strategy.parameters
    assert selected_params["dsp_change"] == "hold"
    assert selected_params["blocked_pan_width_target_pan_percent"] == 35.0
    assert selected_params["blocked_pan_width_current_pan_percent"] == 5.0
    assert selected_params["blocked_pan_width_pan_delta_percent"] == 30.0
    assert selected_params["soundcheck_operator_owned_pan_width_hold_proposal"] is True
    assert selected_params["soundcheck_operator_owned_pan_width_selected_write_keys"] == []
    assert selected_params["soundcheck_operator_owned_pan_width_forbidden_write_keys"] == [
        "wing_pan_osc_value",
        "wing_width_osc_value",
    ]
    assert "wing_pan_osc_value" not in selected_params
    assert "wing_width_osc_value" not in selected_params
    selected_readback = recommendation.selected_strategy.future_wing_readback_contract[
        "pan_width_readback"
    ]
    assert selected_readback["soundcheck_operator_owned_pan_width_hold_proposal"] is True
    assert selected_readback["soundcheck_operator_owned_pan_width_selected_write_keys"] == []
    assert selected_readback["soundcheck_operator_owned_pan_width_forbidden_write_keys"] == [
        "wing_pan_osc_value",
        "wing_width_osc_value",
    ]
    _assert_selected_pan_width_apply_values(recommendation, {})
    assert recommendation.to_operator_proposal()["live_apply_allowed"] is False

    alternatives = {
        candidate.strategy_id: candidate for candidate in recommendation.alternatives
    }
    pan_target = alternatives["phase_safe_pan_target"]
    assert pan_target.blockers == (WING_PAN_WIDTH_OPERATOR_OWNED_BLOCKER,)
    assert pan_target.parameters["wing_pan_width_current_operator_owned"] is True
    assert (
        pan_target.parameters["wing_pan_width_operator_owned_edit_requires_review"]
        is True
    )
    assert pan_target.parameters["wing_pan_width_operator_owned_blocker"] == (
        WING_PAN_WIDTH_OPERATOR_OWNED_BLOCKER
    )
    applicability = pan_target.future_wing_readback_contract[
        "wing_osc_applicability"
    ]
    assert applicability["native_osc_supported"] is False
    assert applicability["native_osc_support_level"] == (
        "pan_width_operator_owned_edit_requires_review"
    )
    assert applicability["pan_width_operator_owned_edit_requires_review"] is True
    assert applicability["pan_width_operator_owned_blocker"] == (
        WING_PAN_WIDTH_OPERATOR_OWNED_BLOCKER
    )
    assert applicability["unsupported_strategy_reason"] == (
        "pan_width_has_native_osc_paths_but_current_wing_pan_width_is_operator_owned"
    )
    pan_readback = pan_target.future_wing_readback_contract["pan_width_readback"]
    assert pan_readback["wing_pan_width_operator_owned_blocker"] == (
        WING_PAN_WIDTH_OPERATOR_OWNED_BLOCKER
    )
    tradeoffs = {
        row["strategy_id"]: row
        for row in recommendation.evidence["strategy_search"]["alternative_tradeoffs"]
    }
    assert WING_PAN_WIDTH_OPERATOR_OWNED_BLOCKER in tradeoffs[
        "phase_safe_pan_target"
    ]["blockers"]
    assert tradeoffs["phase_safe_pan_target"]["wing_osc_native_supported"] is False
    assert tradeoffs["phase_safe_pan_target"]["wing_osc_support_level"] == (
        "pan_width_operator_owned_edit_requires_review"
    )
    assert tradeoffs["phase_safe_pan_target"]["wing_osc_candidate_apply_status"] == (
        "not_wing_applicable_until_pan_width_readback_or_operator_review"
    )


def test_soundcheck_matrix_phase_width_requires_output_feed_identity():
    recommendation = analyze_smart_dsp_strategy(
        _analysis(
            target_id="mtx:1",
            source_role="stereo_guitar_bus",
            source_label="Matrix 1 Fill",
            problem_hints=("pan_width",),
            muddiness_score=0.0,
            stereo_width_request=45.0,
            current_pan_percent=0.0,
            phase_correlation=0.2,
            mono_sum_delta_db=-5.6,
        )
    )

    assert recommendation.status == "proposal_only"
    assert recommendation.selected_strategy.strategy_id == (
        "mono_compatibility_width_reduction"
    )
    params = recommendation.selected_strategy.parameters
    assert params["wing_pan_osc_path_suffix"] == "pan"
    assert params["wing_width_osc_path_suffix"] == "wid"
    assert params["wing_pan_osc_value"] == 0.0
    assert params["wing_width_osc_value"] == 35.0
    assert params["soundcheck_pan_width_target_surface"] == "matrix"
    assert params["soundcheck_pan_width_identity_confirmation_scope"] == (
        "output_feed"
    )
    assert params["soundcheck_pan_width_source_identity_required"] is False
    assert params["soundcheck_pan_width_output_feed_identity_required"] is True
    assert params["soundcheck_pan_width_live_promotion_status"] == (
        "blocked_until_real_wing_pan_width_readback_output_feed_identity_"
        "phase_mono_recheck_and_operator_approval"
    )
    readback = recommendation.selected_strategy.future_wing_readback_contract
    assert readback["wing_osc_applicability"]["target_surface_kind"] == "matrix"
    assert readback["wing_osc_applicability"]["native_osc_supported"] is True
    assert readback["pan_width_readback"]["wing_osc_path_templates"] == [
        "/mtx/{matrix}/pan",
        "/mtx/{matrix}/wid",
    ]
    assert (
        readback["pan_width_readback"][
            "soundcheck_pan_width_output_feed_identity_required"
        ]
        is True
    )
    _assert_selected_pan_width_apply_values(
        recommendation,
        {
            "wing_pan_osc_value": 0.0,
            "wing_width_osc_value": 35.0,
        },
    )
    assert recommendation.to_operator_proposal()["live_apply_allowed"] is False


def test_soundcheck_measured_stereo_balance_calculates_pan_target_without_request():
    recommendation = analyze_smart_dsp_strategy(
        _analysis(
            target_id="ch:12",
            source_role="electric_guitar",
            problem_hints=(),
            muddiness_score=0.0,
            stereo_width_request=0.0,
            current_pan_percent=0.0,
            stereo_balance_db=-5.2,
            stereo_balance_confidence=0.62,
            measured_pan_target_percent=26.0,
            measured_pan_delta_percent=26.0,
            phase_correlation=0.92,
            mono_sum_delta_db=-0.6,
        )
    )

    assert recommendation.status == "proposal_only"
    assert "pan_width" in recommendation.evidence["detected_problems"]
    assert recommendation.selected_strategy.strategy_id == "phase_safe_pan_target"
    params = recommendation.selected_strategy.parameters
    assert params["target_pan_percent"] == 26.0
    assert params["pan_delta_percent"] == 26.0
    assert params["pan_target_source"] == "measured_stereo_balance"
    assert params["stereo_balance_db"] == -5.2
    assert params["stereo_balance_confidence"] == 0.62
    assert params["measured_pan_target_percent"] == 26.0
    assert recommendation.evidence["measured_pan_delta_percent"] == 26.0
    _assert_selected_pan_width_apply_values(
        recommendation,
        {"wing_pan_osc_value": 26.0},
    )


def test_soundcheck_center_anchor_source_recenters_before_measured_pan():
    recommendation = analyze_smart_dsp_strategy(
        _analysis(
            target_id="ch:5",
            source_role="lead_vocal",
            source_label="Lead Vox",
            problem_hints=(),
            muddiness_score=0.0,
            stereo_width_request=0.0,
            current_pan_percent=18.0,
            stereo_balance_db=4.8,
            stereo_balance_confidence=0.72,
            measured_pan_target_percent=-24.0,
            measured_pan_delta_percent=-42.0,
            phase_correlation=0.82,
            mono_sum_delta_db=-0.8,
        )
    )

    assert recommendation.status == "proposal_only"
    assert "pan_width" in recommendation.evidence["detected_problems"]
    assert recommendation.selected_strategy.strategy_id == "center_anchor_recenter_review"
    assert recommendation.selected_strategy.method_family == "pan"
    assert recommendation.evidence["operator_explanation_sections"][
        "selection_policy"
    ] == "center_anchor_pan_priority"
    params = recommendation.selected_strategy.parameters
    assert params["target_pan_percent"] == 0.0
    assert params["current_pan_percent"] == 18.0
    assert params["pan_delta_percent"] == -18.0
    assert params["bounded_delta_percent"] == 18.0
    assert params["pan_target_source"] == "center_anchor_override"
    assert params["center_anchor_recenter_required"] is True
    assert params["measured_stereo_balance_ignored_for_center_anchor"] is True
    assert params["dsp_change"] == "recenter_pan"
    _assert_selected_pan_width_apply_values(
        recommendation,
        {"wing_pan_osc_value": 0.0},
    )

    measured_pan = [
        candidate
        for candidate in recommendation.alternatives
        if candidate.strategy_id == "phase_safe_pan_target"
    ][0]
    assert measured_pan.parameters["pan_target_source"] == "center_anchor_override"
    assert "center_anchor_source_not_panned" in measured_pan.blockers
    tradeoff = [
        item
        for item in recommendation.evidence["strategy_search"][
            "alternative_tradeoffs"
        ]
        if item["strategy_id"] == "phase_safe_pan_target"
    ][0]
    assert tradeoff["why_not_selected"] == (
        "center_anchor_pan_priority:"
        "center_source_role_precedes_measured_stereo_balance"
    )


def test_soundcheck_center_anchor_phase_mono_risk_holds_without_pan_write():
    recommendation = analyze_smart_dsp_strategy(
        _analysis(
            target_id="ch:5",
            source_role="lead_vocal",
            source_label="Lead Vox",
            problem_hints=(),
            muddiness_score=0.0,
            stereo_width_request=0.0,
            current_pan_percent=18.0,
            stereo_balance_db=4.8,
            stereo_balance_confidence=0.72,
            measured_pan_target_percent=-24.0,
            measured_pan_delta_percent=-42.0,
            phase_correlation=0.18,
            mono_sum_delta_db=-5.4,
        )
    )

    assert recommendation.status == "proposal_only"
    assert "pan_width" in recommendation.evidence["detected_problems"]
    assert "phase_mono_risk" in recommendation.evidence["detected_problems"]
    assert (
        recommendation.selected_strategy.strategy_id
        == "center_anchor_phase_mono_review_hold"
    )
    assert recommendation.selected_strategy.method_family == "analysis_only"
    assert recommendation.selected_strategy.problem == "phase_mono_risk"
    assert recommendation.evidence["operator_explanation_sections"][
        "selection_policy"
    ] == "soundcheck_phase_mono_safety_priority"
    params = recommendation.selected_strategy.parameters
    assert params["dsp_change"] == "hold"
    assert params["current_pan_percent"] == 18.0
    assert params["proposed_center_pan_percent"] == 0.0
    assert params["center_anchor_recenter_deferred"] is True
    assert params["phase_risk"] is True
    assert params["mono_risk"] is True
    assert params["center_anchor_phase_mono_selected_write_keys"] == []
    assert set(params["center_anchor_phase_mono_forbidden_write_keys"]) == {
        "wing_pan_osc_value",
        "wing_width_osc_value",
    }
    assert params["wing_pan_width_requires_real_wing_fw_3_0_5_readback"] is True
    assert "wing_pan_osc_value" not in params
    assert "wing_width_osc_value" not in params
    readback = recommendation.selected_strategy.future_wing_readback_contract[
        "pan_width_readback"
    ]
    assert readback["read_before_paths"] == ["pan", "wid"]
    assert readback["read_after_paths"] == ["pan", "wid"]
    assert readback["center_anchor_phase_mono_hold_proposal"] is True
    assert readback["center_anchor_phase_mono_selected_write_keys"] == []
    assert set(readback["center_anchor_phase_mono_forbidden_write_keys"]) == {
        "wing_pan_osc_value",
        "wing_width_osc_value",
    }
    _assert_selected_pan_width_apply_values(recommendation, {})

    center_write = [
        candidate
        for candidate in recommendation.alternatives
        if candidate.strategy_id == "center_hold_for_phase_safety"
    ][0]
    assert center_write.method_family == "pan"
    assert center_write.parameters["wing_pan_osc_value"] == 0.0
    tradeoff = [
        item
        for item in recommendation.evidence["strategy_search"][
            "alternative_tradeoffs"
        ]
        if item["strategy_id"] == "center_hold_for_phase_safety"
    ][0]
    assert tradeoff["why_not_selected"] == (
        "center_anchor_phase_mono_hold_priority:"
        "operator_audition_precedes_pan_or_width_write"
    )


def test_soundcheck_phase_mono_risk_beats_measured_pan_target():
    recommendation = analyze_smart_dsp_strategy(
        _analysis(
            target_id="ch:12",
            source_role="electric_guitar",
            problem_hints=(),
            muddiness_score=0.0,
            stereo_width_request=0.0,
            current_pan_percent=12.0,
            stereo_balance_db=-5.2,
            stereo_balance_confidence=0.62,
            measured_pan_target_percent=26.0,
            measured_pan_delta_percent=14.0,
            phase_correlation=0.18,
            mono_sum_delta_db=-5.4,
        )
    )

    assert recommendation.status == "proposal_only"
    assert "pan_width" in recommendation.evidence["detected_problems"]
    assert "phase_mono_risk" in recommendation.evidence["detected_problems"]
    assert recommendation.selected_strategy.strategy_id == "mono_compatibility_width_reduction"
    assert recommendation.selected_strategy.problem == "phase_mono_risk"
    assert recommendation.evidence["operator_explanation_sections"][
        "selection_policy"
    ] == "soundcheck_phase_mono_safety_priority"
    params = recommendation.selected_strategy.parameters
    assert params["target_width_percent"] == 35.0
    assert params["phase_risk"] is True
    assert params["mono_risk"] is True
    _assert_selected_pan_width_apply_values(
        recommendation,
        {
            "wing_pan_osc_value": 0.0,
            "wing_width_osc_value": 35.0,
        },
    )

    pan_target = [
        candidate
        for candidate in recommendation.alternatives
        if candidate.strategy_id == "phase_safe_pan_target"
    ][0]
    assert pan_target.parameters["pan_target_source"] == "measured_stereo_balance"
    assert "phase_correlation_blocks_width" in pan_target.blockers
    assert "mono_sum_loss_blocks_width" in pan_target.blockers
    tradeoff = [
        item
        for item in recommendation.evidence["strategy_search"][
            "alternative_tradeoffs"
        ]
        if item["strategy_id"] == "phase_safe_pan_target"
    ][0]
    assert tradeoff["why_not_selected"] == (
        "phase_mono_safety_priority:"
        "mono_compatibility_precedes_tone_or_width_changes"
    )


def test_live_pan_request_blocks_non_center_change_for_operator_review():
    recommendation = analyze_smart_dsp_strategy(
        _analysis(
            target_id="ch:12",
            source_role="electric_guitar",
            mode="live",
            problem_hints=("pan_width",),
            muddiness_score=0.0,
            stereo_width_request=35.0,
            current_pan_percent=5.0,
            phase_correlation=0.92,
            mono_sum_delta_db=-0.6,
            active_performance=True,
        )
    )

    assert recommendation.status == "proposal_only"
    assert (
        recommendation.selected_strategy.strategy_id
        == "hold_live_pan_change_for_operator_review"
    )
    assert recommendation.selected_strategy.method_family == "analysis_only"
    assert recommendation.evidence["operator_explanation_sections"][
        "selection_policy"
    ] == "live_pan_change_review_priority"
    params = recommendation.selected_strategy.parameters
    assert params["target_pan_percent"] == 35.0
    assert params["pan_delta_percent"] == 30.0
    assert params["reason"] == "live_pan_change_requires_operator_review"
    assert params["forbidden_without_wing_proof"] == "live_pan_width_change"
    pan_target = [
        candidate
        for candidate in recommendation.alternatives
        if candidate.strategy_id == "phase_safe_pan_target"
    ][0]
    assert "live_pan_change_requires_operator_review_and_wing_proof" in pan_target.blockers
    tradeoff = [
        item
        for item in recommendation.evidence["strategy_search"][
            "alternative_tradeoffs"
        ]
        if item["strategy_id"] == "phase_safe_pan_target"
    ][0]
    assert tradeoff["why_not_selected"] == (
        "live_pan_change_review_priority:"
        "operator_review_precedes_live_pan_or_width_move"
    )


def test_live_measured_stereo_balance_holds_pan_change_for_operator_review():
    recommendation = analyze_smart_dsp_strategy(
        _analysis(
            target_id="ch:12",
            source_role="electric_guitar",
            mode="live",
            problem_hints=(),
            muddiness_score=0.0,
            stereo_width_request=0.0,
            current_pan_percent=0.0,
            stereo_balance_db=6.4,
            stereo_balance_confidence=0.68,
            measured_pan_target_percent=-32.0,
            measured_pan_delta_percent=-32.0,
            phase_correlation=0.92,
            mono_sum_delta_db=-0.6,
            active_performance=True,
        )
    )

    assert recommendation.status == "proposal_only"
    assert "pan_width" in recommendation.evidence["detected_problems"]
    assert (
        recommendation.selected_strategy.strategy_id
        == "hold_live_pan_change_for_operator_review"
    )
    assert recommendation.evidence["operator_explanation_sections"][
        "selection_policy"
    ] == "live_pan_change_review_priority"
    params = recommendation.selected_strategy.parameters
    assert params["target_pan_percent"] == -32.0
    assert params["current_pan_percent"] == 0.0
    assert params["pan_delta_percent"] == -32.0
    assert params["pan_target_source"] == "measured_stereo_balance"
    assert params["stereo_balance_confidence"] == 0.68
    assert params["required_context"] == (
        "source_identity_stereo_role_phase_mono_readback_rollback"
    )
    pan_target = [
        candidate
        for candidate in recommendation.alternatives
        if candidate.strategy_id == "phase_safe_pan_target"
    ][0]
    assert pan_target.parameters["pan_target_source"] == "measured_stereo_balance"
    assert "live_pan_change_requires_operator_review_and_wing_proof" in pan_target.blockers
    assert recommendation.to_operator_proposal()["live_apply_allowed"] is False


def test_live_matrix_pan_review_hold_requires_output_feed_identity():
    recommendation = analyze_smart_dsp_strategy(
        _analysis(
            target_id="mtx:1",
            source_role="stereo_guitar_bus",
            source_label="Matrix 1 Fill",
            mode="live",
            problem_hints=(),
            muddiness_score=0.0,
            stereo_width_request=0.0,
            current_pan_percent=0.0,
            stereo_balance_db=6.4,
            stereo_balance_confidence=0.68,
            measured_pan_target_percent=-32.0,
            measured_pan_delta_percent=-32.0,
            phase_correlation=0.92,
            mono_sum_delta_db=-0.6,
            active_performance=True,
        )
    )

    assert recommendation.status == "proposal_only"
    assert (
        recommendation.selected_strategy.strategy_id
        == "hold_live_pan_change_for_operator_review"
    )
    assert recommendation.selected_strategy.method_family == "analysis_only"
    params = recommendation.selected_strategy.parameters
    assert params["target_pan_percent"] == -32.0
    assert params["pan_target_source"] == "measured_stereo_balance"
    assert params["required_context"] == (
        "output_feed_identity_phase_mono_readback_rollback"
    )
    assert params["wing_pan_width_osc_protocol_version"] == (
        WING_PAN_WIDTH_OSC_PROTOCOL_VERSION
    )
    assert params["wing_pan_width_requires_real_wing_fw_3_0_5_readback"] is True
    assert params["live_pan_width_hold_proposal"] is True
    assert params["live_pan_width_native_osc_only_required"] is True
    assert params["live_pan_width_move_kinds"] == ["pan"]
    assert params["live_pan_width_apply_only_selected_write_keys"] is True
    assert params["live_pan_width_selected_write_keys"] == []
    assert set(params["live_pan_width_forbidden_write_keys"]) == {
        "wing_pan_osc_value",
        "wing_width_osc_value",
    }
    assert params["live_pan_width_target_surface"] == "matrix"
    assert params["live_pan_width_identity_confirmation_scope"] == "output_feed"
    assert params["live_pan_width_source_identity_required"] is False
    assert params["live_pan_width_output_feed_identity_required"] is True
    assert params["live_pan_width_phase_mono_recheck_required"] is True
    assert params["live_pan_width_readback_required"] is True
    assert params["live_pan_width_rollback_ready"] is True
    assert (
        params["live_pan_width_live_promotion_status"]
        == "blocked_until_real_wing_live_pan_width_readback_output_feed_identity_"
        "phase_mono_recheck_and_operator_approval"
    )
    assert params["live_pan_width_live_apply_blocked_without_wing_proof"] is True
    assert "wing_pan_osc_value" not in params
    assert "wing_width_osc_value" not in params
    readback = recommendation.selected_strategy.future_wing_readback_contract[
        "pan_width_readback"
    ]
    assert readback["live_pan_width_target_surface"] == "matrix"
    assert readback["live_pan_width_identity_confirmation_scope"] == "output_feed"
    assert readback["live_pan_width_output_feed_identity_required"] is True
    assert readback["live_pan_width_selected_write_keys"] == []
    assert recommendation.to_operator_proposal()["live_apply_allowed"] is False


def test_soundcheck_phase_mono_risk_reduces_width_without_pan_request():
    recommendation = analyze_smart_dsp_strategy(
        _analysis(
            target_id="ch:12",
            source_role="electric_guitar",
            problem_hints=(),
            muddiness_score=0.0,
            stereo_width_request=0.0,
            current_pan_percent=18.0,
            phase_correlation=-0.42,
            mono_sum_delta_db=-8.0,
        )
    )

    assert recommendation.status == "proposal_only"
    assert "phase_mono_risk" in recommendation.evidence["detected_problems"]
    assert recommendation.selected_strategy.strategy_id == "mono_compatibility_width_reduction"
    assert recommendation.selected_strategy.method_family == "pan"
    params = recommendation.selected_strategy.parameters
    assert params["target_width_percent"] == 0.0
    assert params["width_delta_percent"] < 0.0
    assert params["target_pan_percent"] == 0.0
    assert params["wing_pan_osc_path_suffix"] == "pan"
    assert params["wing_pan_osc_value"] == 0.0
    assert params["wing_width_osc_path_suffix"] == "wid"
    assert params["wing_width_osc_value"] == 0.0
    assert params["phase_risk"] is True
    assert params["mono_risk"] is True
    readback = recommendation.selected_strategy.future_wing_readback_contract
    _assert_common_readback_contract(readback)
    assert readback["pan_width_readback"]["read_before_paths"] == ["pan", "wid"]
    assert readback["pan_width_readback"]["read_after_paths"] == ["pan", "wid"]
    assert readback["pan_width_readback"]["wing_pan_mapping_required"] is True
    assert readback["pan_width_readback"]["wing_width_mapping_required"] is True
    assert readback["pan_width_readback"][
        "phase_alignment_changes_are_review_only_without_verified_wing_path"
    ] is True
    assert readback["pan_width_readback"]["mono_phase_recheck_required"] is True
    _assert_selected_pan_width_apply_values(
        recommendation,
        {
            "wing_pan_osc_value": 0.0,
            "wing_width_osc_value": 0.0,
        },
    )

    alignment = [
        candidate
        for candidate in recommendation.alternatives
        if candidate.strategy_id == "phase_polarity_alignment_review"
    ][0]
    hold = [
        candidate
        for candidate in recommendation.alternatives
        if candidate.strategy_id == "hold_phase_mono_processing_for_operator"
    ][0]
    assert "phase_alignment_requires_stereo_pair_identity_and_readback" in alignment.blockers
    assert "soundcheck_width_reduction_safe_to_compare" in hold.blockers
    assert recommendation.to_operator_proposal()["live_apply_allowed"] is False


def test_live_phase_mono_risk_holds_width_reduction_for_operator_review():
    recommendation = analyze_smart_dsp_strategy(
        _analysis(
            target_id="ch:12",
            source_role="electric_guitar",
            mode="live",
            problem_hints=(),
            muddiness_score=0.0,
            stereo_width_request=0.0,
            current_pan_percent=18.0,
            phase_correlation=0.18,
            mono_sum_delta_db=-5.4,
            active_performance=True,
        )
    )

    assert recommendation.status == "proposal_only"
    assert recommendation.selected_strategy.strategy_id == "hold_phase_mono_processing_for_operator"
    assert recommendation.selected_strategy.method_family == "analysis_only"
    params = recommendation.selected_strategy.parameters
    assert params["required_context"] == (
        "source_identity_stereo_role_phase_mono_readback_rollback"
    )
    assert params["forbidden_without_wing_proof"] == (
        "live_phase_mono_pan_width_change"
    )
    assert params["wing_pan_width_requires_real_wing_fw_3_0_5_readback"] is True
    assert params["live_phase_mono_hold_proposal"] is True
    assert params["live_phase_mono_target_surface"] == "channel"
    assert params["live_phase_mono_identity_confirmation_scope"] == "source"
    assert params["live_phase_mono_source_identity_required"] is True
    assert params["live_phase_mono_output_feed_identity_required"] is False
    assert params["live_phase_mono_selected_write_keys"] == []
    assert set(params["live_phase_mono_forbidden_write_keys"]) == {
        "wing_pan_osc_value",
        "wing_width_osc_value",
    }
    width_reduction = [
        candidate
        for candidate in recommendation.alternatives
        if candidate.strategy_id == "mono_compatibility_width_reduction"
    ][0]
    assert "live_stereo_width_reduction_requires_operator_review_and_wing_proof" in (
        width_reduction.blockers
    )
    assert recommendation.to_operator_proposal()["can_apply_live"] is False


def test_live_matrix_phase_mono_risk_holds_without_pan_width_apply_values():
    recommendation = analyze_smart_dsp_strategy(
        _analysis(
            target_id="mtx:1",
            source_role="stereo_guitar_bus",
            source_label="Matrix 1 Fill",
            mode="live",
            problem_hints=(),
            muddiness_score=0.0,
            stereo_width_request=45.0,
            current_pan_percent=0.0,
            phase_correlation=0.2,
            mono_sum_delta_db=-5.6,
            active_performance=True,
        )
    )

    assert recommendation.status == "proposal_only"
    assert recommendation.selected_strategy.strategy_id == (
        "hold_phase_mono_processing_for_operator"
    )
    assert recommendation.selected_strategy.method_family == "analysis_only"
    assert recommendation.evidence["operator_explanation_sections"][
        "selection_policy"
    ] == "live_phase_mono_safety_priority"
    params = recommendation.selected_strategy.parameters
    assert params["required_context"] == (
        "output_feed_identity_phase_mono_readback_rollback"
    )
    assert params["forbidden_without_wing_proof"] == (
        "live_phase_mono_pan_width_change"
    )
    assert params["wing_pan_width_osc_protocol_version"] == (
        WING_PAN_WIDTH_OSC_PROTOCOL_VERSION
    )
    assert params["wing_pan_width_requires_real_wing_fw_3_0_5_readback"] is True
    assert params["live_phase_mono_hold_proposal"] is True
    assert params["live_phase_mono_native_osc_only_required"] is True
    assert params["live_phase_mono_apply_only_selected_write_keys"] is True
    assert params["live_phase_mono_selected_write_keys"] == []
    assert set(params["live_phase_mono_forbidden_write_keys"]) == {
        "wing_pan_osc_value",
        "wing_width_osc_value",
    }
    assert params["live_phase_mono_target_surface"] == "matrix"
    assert params["live_phase_mono_identity_confirmation_scope"] == "output_feed"
    assert params["live_phase_mono_source_identity_required"] is False
    assert params["live_phase_mono_output_feed_identity_required"] is True
    assert params["live_phase_mono_phase_mono_recheck_required"] is True
    assert params["live_phase_mono_readback_required"] is True
    assert params["live_phase_mono_rollback_ready"] is True
    assert "wing_pan_osc_value" not in params
    assert "wing_width_osc_value" not in params
    readback = recommendation.selected_strategy.future_wing_readback_contract[
        "pan_width_readback"
    ]
    assert readback["live_phase_mono_target_surface"] == "matrix"
    assert readback["live_phase_mono_identity_confirmation_scope"] == "output_feed"
    assert readback["live_phase_mono_output_feed_identity_required"] is True
    assert readback["live_phase_mono_selected_write_keys"] == []
    center_hold = [
        candidate
        for candidate in recommendation.alternatives
        if candidate.strategy_id == "center_hold_for_phase_safety"
    ][0]
    assert center_hold.method_family == "pan"
    assert center_hold.parameters["wing_pan_osc_value"] == 0.0
    tradeoff = [
        item
        for item in recommendation.evidence["strategy_search"][
            "alternative_tradeoffs"
        ]
        if item["strategy_id"] == "center_hold_for_phase_safety"
    ][0]
    assert tradeoff["why_not_selected"] == (
        "live_phase_mono_hold_priority:"
        "analysis_only_hold_precedes_wing_pan_center_value"
    )
    assert recommendation.to_operator_proposal()["live_apply_allowed"] is False


def test_soundcheck_phase_mono_risk_beats_harshness_eq():
    recommendation = analyze_smart_dsp_strategy(
        _analysis(
            target_id="stereo:gtr",
            source_role="electric_guitar",
            source_label="GTR Stereo",
            problem_hints=("harsh",),
            muddiness_score=0.0,
            harshness_score=0.84,
            presence_harshness_score=0.78,
            stereo_width_request=0.0,
            current_pan_percent=0.0,
            phase_correlation=0.18,
            mono_sum_delta_db=-5.4,
        )
    )

    assert recommendation.status == "proposal_only"
    assert recommendation.selected_strategy.strategy_id == (
        "hold_phase_mono_processing_for_operator"
    )
    assert recommendation.selected_strategy.problem == "phase_mono_risk"
    assert recommendation.selected_strategy.method_family == "analysis_only"
    params = recommendation.selected_strategy.parameters
    assert params["target_resolution_required"] is True
    assert params["required_context"] == (
        "concrete_wing_channel_or_stereo_pair_target_before_width_change"
    )
    assert params["forbidden_without_wing_proof"] == (
        "pan_width_change_on_unresolved_stereo_target"
    )
    assert recommendation.evidence["operator_explanation_sections"][
        "selection_policy"
    ] == "soundcheck_phase_mono_safety_priority"
    width_reduction = [
        candidate
        for candidate in recommendation.alternatives
        if candidate.strategy_id == "mono_compatibility_width_reduction"
    ][0]
    assert "wing_osc_target_requires_concrete_console_surface" in (
        width_reduction.blockers
    )
    wing_osc = width_reduction.future_wing_readback_contract[
        "wing_osc_applicability"
    ]
    assert wing_osc["native_osc_supported"] is False
    assert wing_osc["native_osc_support_level"] == "unresolved_wing_osc_target"
    assert wing_osc["target_surface_kind"] == "unresolved"
    assert wing_osc["unsupported_strategy_reason"] == (
        "target_id_does_not_resolve_to_wing_fw_3_0_5_osc_surface"
    )
    dynamic_eq = [
        candidate
        for candidate in recommendation.alternatives
        if candidate.strategy_id == "instrument_dynamic_presence_tame"
    ][0]
    assert dynamic_eq.problem == "harsh_instrument"
    tradeoff = [
        item
        for item in recommendation.evidence["strategy_search"][
            "alternative_tradeoffs"
        ]
        if item["strategy_id"] == "instrument_dynamic_presence_tame"
    ][0]
    assert tradeoff["why_not_selected"] == (
        "phase_mono_safety_priority:"
        "mono_compatibility_precedes_tone_or_width_changes"
    )


def test_live_phase_mono_risk_beats_harshness_eq_with_hold():
    recommendation = analyze_smart_dsp_strategy(
        _analysis(
            target_id="stereo:gtr",
            source_role="electric_guitar",
            source_label="GTR Stereo",
            mode="live",
            problem_hints=("harsh",),
            muddiness_score=0.0,
            harshness_score=0.84,
            presence_harshness_score=0.78,
            stereo_width_request=0.0,
            current_pan_percent=0.0,
            phase_correlation=0.18,
            mono_sum_delta_db=-5.4,
            active_performance=True,
        )
    )

    assert recommendation.status == "proposal_only"
    assert recommendation.selected_strategy.strategy_id == (
        "hold_phase_mono_processing_for_operator"
    )
    assert recommendation.selected_strategy.problem == "phase_mono_risk"
    assert recommendation.selected_strategy.method_family == "analysis_only"
    assert recommendation.evidence["operator_explanation_sections"][
        "selection_policy"
    ] == "live_phase_mono_safety_priority"
    broad_eq = [
        candidate
        for candidate in recommendation.alternatives
        if candidate.strategy_id == "instrument_broad_presence_cut"
    ][0]
    assert broad_eq.problem == "harsh_instrument"
    tradeoff = [
        item
        for item in recommendation.evidence["strategy_search"][
            "alternative_tradeoffs"
        ]
        if item["strategy_id"] == "instrument_broad_presence_cut"
    ][0]
    assert tradeoff["why_not_selected"] == (
        "phase_mono_safety_priority:"
        "mono_compatibility_precedes_tone_or_width_changes"
    )
    _assert_proposal_denial(recommendation.to_operator_proposal())


def test_soundcheck_hot_drum_bus_selects_group_trim_before_compression_or_eq():
    recommendation = analyze_smart_dsp_strategy(
        _analysis(
            target_id="bus:1",
            source_role="drum_bus",
            source_label="Drum Bus",
            problem_hints=("bus_group_impact",),
            muddiness_score=0.0,
            estimated_lufs=-20.5,
            level_plane_error_lu=4.5,
            true_peak_headroom_db=1.4,
            lra_proxy_db=12.0,
            dynamics_variance_score=0.62,
            low_end_excess_score=0.72,
            spectral_tilt_score=0.38,
            low_end_resonance_frequency_hz=124.0,
            low_end_resonance_bandwidth_hz=70.0,
        )
    )

    assert recommendation.status == "proposal_only"
    assert "bus_group_impact" in recommendation.evidence["detected_problems"]
    assert recommendation.selected_strategy.strategy_id == "group_bus_trim_for_headroom"
    assert recommendation.selected_strategy.method_family == "gain_fader"
    params = recommendation.selected_strategy.parameters
    assert params["target_scope"] == "group_bus"
    assert params["group_delta_db"] < 0.0
    assert params["gain_db"] == 0.0
    assert params["future_fader_ceiling_dbfs"] == 0.0
    _assert_gain_delta_readback_contract(
        recommendation.selected_strategy,
        source="group_delta_db",
        target_surface="bus",
        expected_delta=params["group_delta_db"],
        path_prefix="/bus/{bus}",
    )

    member_rebalance = [
        candidate
        for candidate in recommendation.alternatives
        if candidate.strategy_id == "upstream_member_rebalance_before_group_move"
    ][0]
    compression = [
        candidate
        for candidate in recommendation.alternatives
        if candidate.strategy_id == "group_bus_compression_review"
    ][0]
    tone = [
        candidate
        for candidate in recommendation.alternatives
        if candidate.strategy_id == "group_bus_broad_tone_cleanup_review"
    ][0]
    tone_params = tone.parameters
    assert "group_membership_readback_required" in member_rebalance.blockers
    assert "group_bus_compression_requires_operator_review" in compression.blockers
    assert "group_bus_eq_requires_member_and_pa_review" in tone.blockers
    assert tone_params["group_tone_parameter_source"] == "measured_low_end_resonance"
    assert tone_params["filter_type"] == "peaking"
    assert tone_params["frequency_hz"] == 124.0
    assert tone_params["q"] == 1.27
    assert tone_params["gain_db"] < 0.0
    assert tone_params["low_end_resonance_bandwidth_hz"] == 70.0
    assert tone_params["low_end_excess_score"] == 0.72
    assert recommendation.to_operator_proposal()["live_apply_allowed"] is False


def test_soundcheck_dynamic_group_bus_selects_bounded_compression_when_headroom_safe():
    recommendation = analyze_smart_dsp_strategy(
        _analysis(
            target_id="bus:2",
            source_role="vocal_bus",
            source_label="Vocal Bus",
            problem_hints=("bus_group_impact",),
            muddiness_score=0.0,
            estimated_lufs=-25.5,
            level_plane_error_lu=0.2,
            true_peak_headroom_db=5.5,
            lra_proxy_db=15.0,
            dynamics_variance_score=0.78,
            dynamics_peak_to_median_db=11.0,
            dynamics_rise_time_ms=32.0,
            dynamics_recovery_ms=240.0,
            crest_factor_db=19.0,
            feedback_risk=0.1,
        )
    )

    assert recommendation.status == "proposal_only"
    assert recommendation.selected_strategy.strategy_id == "group_bus_compression_review"
    assert recommendation.selected_strategy.method_family == "dynamics"
    params = recommendation.selected_strategy.parameters
    assert params["target_scope"] == "group_bus"
    assert params["ratio"] <= 2.6
    assert params["wing_compressor_ratio_osc_value"] == 2.5
    assert params["wing_compressor_ratio_value_type"] == "float"
    assert params["wing_compressor_ratio_target_surface"] == "bus"
    assert params["target_gain_reduction_db"] <= 3.0
    assert params["group_bus_compression_parameter_source"] == "measured_group_envelope"
    assert params["threshold_dbfs"] == -22.0
    assert params["wing_compressor_threshold_osc_path_suffix"] == "dyn/thr31"
    assert params["wing_compressor_threshold_osc_value"] == -22.0
    assert params["wing_compressor_threshold_value_type"] == "float"
    assert params["attack_ms"] == 20.8
    assert params["release_ms"] == 228.0
    assert params["knee_db"] == 4.4
    assert params["wing_compressor_knee_osc_value"] == 4
    assert params["wing_compressor_knee_value_type"] == "integer"
    assert params["wing_compressor_knee_quantization_error"] == 0.4
    assert params["dynamics_peak_to_median_db"] == 11.0
    assert params["dynamics_rise_time_ms"] == 32.0
    assert params["dynamics_recovery_ms"] == 240.0
    assert params["makeup_gain_db"] == 0.0
    assert params["true_peak_headroom_required_before_compression"] is True
    assert params["dynamic_pressure_score"] >= 0.55
    readback = recommendation.selected_strategy.future_wing_readback_contract
    wing_paths = readback["wing_osc_applicability"]["native_osc_path_templates"]
    assert "/bus/{bus}/dyn/thr31" in wing_paths
    assert "/bus/{bus}/dyn/thr26" not in wing_paths
    assert "/bus/{bus}/dyn/thr37" not in wing_paths
    assert "/bus/{bus}/dyn/thr31" in readback["dynamics_readback"][
        "wing_osc_path_templates"
    ]
    assert readback["dynamics_readback"][
        "wing_compressor_threshold_mapping_required"
    ] is True
    assert readback["dynamics_readback"][
        "wing_compressor_threshold_osc_path_suffix"
    ] == "dyn/thr31"
    assert readback["dynamics_readback"][
        "wing_compressor_threshold_osc_value"
    ] == params["wing_compressor_threshold_osc_value"]
    assert readback["dynamics_readback"][
        "wing_compressor_threshold_readback_must_match_osc_value"
    ] is True
    assert "dyn/thr31" in readback["dynamics_readback"]["read_before_paths"]
    assert "dyn/thr31" in readback["dynamics_readback"]["read_after_paths"]
    assert "dyn/thr" not in readback["dynamics_readback"]["read_before_paths"]
    assert "dyn/thr24" not in readback["dynamics_readback"]["read_before_paths"]
    assert readback["dynamics_readback"][
        "wing_compressor_attack_mapping_required"
    ] is True
    assert readback["dynamics_readback"]["wing_compressor_attack_osc_path_suffix"] == (
        "dyn/att"
    )
    assert readback["dynamics_readback"]["wing_compressor_attack_osc_value"] == (
        params["wing_compressor_attack_osc_value"]
    )
    assert readback["dynamics_readback"][
        "wing_compressor_release_mapping_required"
    ] is True
    assert readback["dynamics_readback"]["wing_compressor_release_osc_path_suffix"] == (
        "dyn/rel"
    )
    assert readback["dynamics_readback"]["wing_compressor_release_osc_value"] == (
        params["wing_compressor_release_osc_value"]
    )
    assert readback["dynamics_readback"][
        "wing_compressor_makeup_gain_mapping_required"
    ] is True
    assert readback["dynamics_readback"][
        "wing_compressor_makeup_gain_osc_path_suffix"
    ] == "dyn/gain"
    assert readback["dynamics_readback"][
        "wing_compressor_makeup_gain_osc_value"
    ] == params["wing_compressor_makeup_gain_osc_value"]
    assert recommendation.selected_strategy.blockers == ()
    hold = [
        candidate
        for candidate in recommendation.alternatives
        if candidate.strategy_id == "hold_group_bus_for_member_evidence"
    ][0]
    assert hold.method_family == "analysis_only"
    assert recommendation.to_operator_proposal()["live_apply_allowed"] is False


def test_named_group_bus_target_holds_until_concrete_bus_index_resolved():
    recommendation = analyze_smart_dsp_strategy(
        _analysis(
            target_id="bus:vocals",
            source_role="vocal_bus",
            source_label="Vocal Bus",
            problem_hints=("bus_group_impact",),
            muddiness_score=0.0,
            estimated_lufs=-25.5,
            level_plane_error_lu=0.2,
            true_peak_headroom_db=5.5,
            lra_proxy_db=15.0,
            dynamics_variance_score=0.78,
            dynamics_peak_to_median_db=11.0,
            dynamics_rise_time_ms=32.0,
            dynamics_recovery_ms=240.0,
            crest_factor_db=19.0,
            feedback_risk=0.1,
        )
    )

    assert recommendation.status == "proposal_only"
    assert recommendation.selected_strategy.strategy_id == (
        "hold_group_bus_for_member_evidence"
    )
    assert recommendation.selected_strategy.method_family == "analysis_only"
    compression = [
        candidate
        for candidate in recommendation.alternatives
        if candidate.strategy_id == "group_bus_compression_review"
    ][0]
    assert WING_OSC_UNRESOLVED_TARGET_BLOCKER in compression.blockers
    applicability = compression.future_wing_readback_contract[
        "wing_osc_applicability"
    ]
    assert applicability["target_surface_kind"] == "bus"
    assert applicability["target_index"] is None
    assert applicability["target_index_valid"] is False
    assert applicability["target_index_range"] == [1, 16]
    assert applicability["native_osc_supported"] is False
    assert applicability["native_osc_support_level"] == "unresolved_wing_osc_target"
    assert applicability["unsupported_strategy_reason"] == (
        "target_id_does_not_resolve_to_wing_fw_3_0_5_osc_surface"
    )


def test_live_quiet_vocal_bus_holds_group_gain_up_for_operator_review():
    recommendation = analyze_smart_dsp_strategy(
        _analysis(
            target_id="bus:2",
            source_role="vocal_bus",
            source_label="Vocal Bus",
            mode="live",
            problem_hints=("bus_group_impact",),
            muddiness_score=0.0,
            estimated_lufs=-31.0,
            level_plane_error_lu=-6.0,
            true_peak_headroom_db=7.0,
            feedback_risk=0.16,
            active_performance=True,
        )
    )

    assert recommendation.status == "proposal_only"
    assert recommendation.selected_strategy.strategy_id == "hold_group_bus_for_member_evidence"
    assert recommendation.selected_strategy.method_family == "analysis_only"
    params = recommendation.selected_strategy.parameters
    assert params["target_scope"] == "group_bus_and_members"
    assert params["required_context"] == (
        "group_membership_bus_output_identity_summed_bus_readback_rollback"
    )
    assert params["forbidden_without_wing_proof"] == (
        "group_bus_gain_eq_dynamics_pan_mute_or_fx_change"
    )
    assert params["group_bus_hold_proposal"] is True
    assert params["live_group_bus_hold_proposal"] is True
    assert params["group_bus_target_surface"] == "bus"
    assert params["group_bus_output_feed_identity_required"] is True
    assert params["group_bus_source_member_identity_required"] is True
    assert params["group_bus_member_channel_evidence_required"] is True
    assert params["group_bus_membership_readback_required"] is True
    assert params["group_bus_previous_bus_readback_required"] is True
    assert params["group_bus_summed_bus_evidence_required"] is True
    assert params["group_bus_true_peak_headroom_recheck_required"] is True
    assert params["group_bus_feedback_recheck_required"] is True
    assert params["group_bus_rollback_ready"] is True
    assert params["live_group_bus_live_apply_blocked_without_wing_proof"] is True
    assert "wing_gain_delta_db" not in params
    assert "wing_eq_osc_value" not in params
    readback = recommendation.selected_strategy.future_wing_readback_contract[
        "group_bus_hold_readback"
    ]
    assert readback["read_only"] is True
    assert readback["group_bus_target_surface"] == "bus"
    assert readback["group_bus_output_feed_identity_required"] is True
    assert readback["group_bus_source_member_identity_required"] is True
    assert readback["group_bus_summed_bus_evidence_required"] is True
    assert readback["allowed_live_write_scope_before_wing_proof"] == []
    assert "/bus/{bus}/fdr" in readback["wing_osc_path_templates"]
    assert "/bus/{bus}/eq/{band}f" in readback["wing_osc_path_templates"]
    assert "/bus/{bus}/dyn/thr31" in readback["wing_osc_path_templates"]
    assert "/bus/{bus}/fdr" in readback["wing_gain_osc_path_templates"]
    assert "/bus/{bus}/eq/on" in readback["wing_eq_osc_path_templates"]
    assert "/bus/{bus}/eq/{band}f" in readback["wing_eq_osc_path_templates"]
    assert "/bus/{bus}/dyn/thr31" in readback["wing_dynamics_osc_path_templates"]
    gain_up = [
        candidate
        for candidate in recommendation.alternatives
        if candidate.strategy_id == "soundcheck_bounded_group_gain_up"
    ][0]
    assert "live_group_gain_up_requires_operator_review_and_wing_proof" in gain_up.blockers
    assert recommendation.to_operator_proposal()["can_apply_live"] is False


def test_live_quiet_source_gain_up_selects_analysis_hold_not_gain_fader():
    recommendation = analyze_smart_dsp_strategy(
        _analysis(
            target_id="ch:7",
            source_role="lead_vocal",
            source_label="Lead Vox",
            mode="live",
            problem_hints=("level_plane_balance",),
            muddiness_score=0.0,
            estimated_lufs=-34.0,
            level_plane_error_lu=-6.0,
            true_peak_headroom_db=6.0,
            feedback_risk=0.1,
            monitor_feedback_risk=0.1,
            bleed_risk=0.12,
            active_performance=True,
        )
    )

    assert recommendation.status == "proposal_only"
    assert recommendation.selected_strategy.strategy_id == (
        "hold_live_quiet_source_gain_up_for_operator_review"
    )
    assert recommendation.selected_strategy.method_family == "analysis_only"
    assert recommendation.evidence["operator_explanation_sections"][
        "selection_policy"
    ] == "live_quiet_source_gain_up_review_priority"
    params = recommendation.selected_strategy.parameters
    assert params["dsp_change"] == "hold"
    assert params["reason"] == "live_quiet_source_gain_up_requires_operator_review"
    assert params["calculated_gain_up_db"] == 1.5
    assert params["gain_db"] == 0.0
    assert params["live_gain_up_from_short_signal_or_bleed_blocked"] is True
    assert params["wing_rack_fw305_correction_exists_on_console_required"] is True
    assert params["wing_rack_fw305_verified_osc_mapping_required"] is True
    assert params["wing_rack_fw305_readback_required"] is True
    assert params["wing_rack_fw305_rollback_required"] is True
    assert params["live_authority_granted"] is False
    assert params["live_write"] is False
    assert params["readiness_closure_allowed"] is False
    gain_up = [
        candidate
        for candidate in recommendation.alternatives
        if candidate.strategy_id == "live_hold_gain_up_for_operator"
    ][0]
    assert gain_up.method_family == "gain_fader"
    assert gain_up.parameters["gain_db"] == 1.5
    assert gain_up.blockers == (
        "live_gain_up_requires_operator_review_and_wing_proof",
    )
    tradeoffs = {
        row["strategy_id"]: row["why_not_selected"]
        for row in recommendation.evidence["strategy_search"][
            "alternative_tradeoffs"
        ]
    }
    assert tradeoffs["live_hold_gain_up_for_operator"] == (
        "live_quiet_source_gain_up_review_priority:"
        "live_source_gain_up_requires_operator_review_and_wing_proof"
    )
    proposal = recommendation.to_operator_proposal()
    assert proposal["can_apply_live"] is False
    assert proposal["live_authority_granted"] is False
    assert proposal["live_write"] is False


def test_soundcheck_quiet_group_feedback_holds_member_review_before_gain_up():
    recommendation = analyze_smart_dsp_strategy(
        _analysis(
            target_id="bus:vocals",
            source_role="vocal_bus",
            source_label="Vocal Bus",
            problem_hints=("level_balance",),
            muddiness_score=0.0,
            estimated_lufs=-31.0,
            level_plane_error_lu=-5.0,
            true_peak_headroom_db=12.0,
            feedback_risk=0.82,
            monitor_feedback_risk=0.0,
            summed_bus_evidence_available=True,
            summed_bus_role="vocal_bus",
            summed_bus_estimated_lufs=-31.0,
            summed_bus_target_plane_lufs=-25.0,
            summed_bus_level_plane_error_lu=-6.0,
            summed_bus_true_peak_headroom_db=12.0,
        )
    )

    assert recommendation.status == "proposal_only"
    assert "bus_group_feedback_risk" in recommendation.evidence["detected_problems"]
    assert recommendation.selected_strategy.strategy_id == (
        "hold_group_bus_feedback_member_review"
    )
    assert recommendation.selected_strategy.problem == "feedback_risk"
    assert recommendation.evidence["operator_explanation_sections"][
        "selection_policy"
    ] == "feedback_risk_priority"
    params = recommendation.selected_strategy.parameters
    assert params["feedback_risk"] == 0.82
    assert params["monitor_feedback_risk"] == 0.0
    assert params["target_scope"] == "group_members_before_bus_processing"
    assert params["live_authority_granted"] is False
    assert params["live_write"] is False
    assert params["readiness_closure_allowed"] is False

    gain_up = [
        candidate
        for candidate in recommendation.alternatives
        if candidate.strategy_id == "soundcheck_bounded_group_gain_up"
    ][0]
    assert "feedback_risk_blocks_boost" in gain_up.blockers
    tradeoff = [
        item
        for item in recommendation.evidence["strategy_search"][
            "alternative_tradeoffs"
        ]
        if item["strategy_id"] == "soundcheck_bounded_group_gain_up"
    ][0]
    assert tradeoff["why_not_selected"] == (
        "group_feedback_review_priority:identify_feedback_member_before_group_processing"
    )


def test_live_group_feedback_holds_member_review_before_transient_or_bus_dsp():
    recommendation = analyze_smart_dsp_strategy(
        _analysis(
            target_id="bus:drums",
            source_role="drum_group_bus",
            source_label="Drum Bus",
            mode="live",
            problem_hints=("bus_group_impact", "dynamic_control", "feedback_risk"),
            muddiness_score=0.0,
            estimated_lufs=-25.0,
            level_plane_error_lu=0.2,
            true_peak_headroom_db=5.0,
            dynamics_variance_score=0.7,
            dynamics_peak_to_median_db=10.0,
            dynamics_rise_time_ms=24.0,
            dynamics_recovery_ms=180.0,
            lra_proxy_db=12.0,
            crest_factor_db=18.0,
            transient_strength=0.72,
            transient_density=4.4,
            feedback_risk=0.78,
            monitor_feedback_risk=0.24,
            active_performance=True,
        )
    )

    assert recommendation.status == "proposal_only"
    assert "bus_group_feedback_risk" in recommendation.evidence["detected_problems"]
    assert recommendation.selected_strategy.strategy_id == (
        "hold_group_bus_feedback_member_review"
    )
    assert recommendation.selected_strategy.method_family == "analysis_only"
    assert recommendation.selected_strategy.problem == "feedback_risk"
    assert recommendation.evidence["operator_explanation_sections"][
        "selection_policy"
    ] == "live_feedback_risk_priority"
    params = recommendation.selected_strategy.parameters
    assert params["target_scope"] == "group_members_before_bus_processing"
    assert params["feedback_risk"] == 0.78
    assert params["monitor_feedback_risk"] == 0.24
    assert params["live_authority_granted"] is False
    assert params["live_write"] is False
    assert params["readiness_closure_allowed"] is False

    transient = [
        candidate
        for candidate in recommendation.alternatives
        if candidate.strategy_id == "drum_transient_tamer_compression"
    ][0]
    assert transient.total_score > 0.0
    group_compression = [
        candidate
        for candidate in recommendation.alternatives
        if candidate.strategy_id == "group_bus_compression_review"
    ][0]
    assert "feedback_risk_requires_group_member_review" in group_compression.blockers
    tradeoffs = {
        item["strategy_id"]: item["why_not_selected"]
        for item in recommendation.evidence["strategy_search"][
            "alternative_tradeoffs"
        ]
    }
    assert tradeoffs["drum_transient_tamer_compression"] == (
        "group_feedback_review_priority:"
        "identify_feedback_member_before_group_processing"
    )
    assert tradeoffs["group_bus_compression_review"] == (
        "group_feedback_review_priority:"
        "identify_feedback_member_before_group_processing"
    )
    _assert_proposal_denial(recommendation.to_operator_proposal())


def test_source_identity_mismatch_holds_before_dsp_selection():
    recommendation = analyze_smart_dsp_strategy(
        _analysis(
            target_id="ch:1",
            source_role="lead_vocal",
            source_label="Lead Vox",
            declared_source_role="lead_vocal",
            declared_source_family="vocal",
            problem_hints=("source_identity_mismatch", "sibilant", "harsh"),
            muddiness_score=0.0,
            harshness_score=0.72,
            sibilance_score=0.84,
            source_identity_confirmed=True,
            source_identity_confidence=0.95,
            source_identity_mismatch_score=0.86,
            source_identity_mismatch_family="kick",
            source_identity_mismatch_reason=(
                "declared_vocal_conflicts_with_audio_kick:"
                "low_end_transient_dominant"
            ),
        )
    )

    assert recommendation.status == "analysis_only"
    assert recommendation.selected_strategy.strategy_id == "hold_for_source_identity_recheck"
    assert recommendation.selected_strategy.method_family == "analysis_only"
    assert recommendation.evidence["detected_problems"] == ["source_identity_mismatch"]
    assert "source_identity_evidence_conflict" in recommendation.blockers
    assert recommendation.evidence["source_identity_mismatch"]["active"] is True
    assert recommendation.evidence["source_identity_mismatch"]["audio_evidence_family"] == "kick"
    assert recommendation.evidence["strategy_search"]["candidate_count"] == 3
    assert {
        candidate.strategy_id for candidate in recommendation.alternatives
    } == {
        "reroute_to_acoustic_family_after_confirmation",
        "use_declared_identity_after_operator_confirmation",
    }
    proposal = recommendation.to_operator_proposal()
    _assert_proposal_denial(proposal)
    assert proposal["selected_strategy"]["strategy_id"] == "hold_for_source_identity_recheck"


def test_unconfirmed_or_low_confidence_source_identity_holds_before_dsp():
    inferred = analyze_smart_dsp_strategy(
        _analysis(
            target_id="ch:6",
            source_role="lead_vocal_inferred",
            source_label="",
            problem_hints=("sibilant", "harsh"),
            muddiness_score=0.0,
            harshness_score=0.62,
            sibilance_score=0.88,
            sibilance_frequency_hz=7600.0,
            declared_source_role="unknown",
            declared_source_family="unknown",
            inferred_source_role="lead_vocal_inferred",
            inferred_source_family="vocal",
            source_identity_inference_confidence=0.74,
            source_identity_inference_reason="presence_sibilance_vocal_signature",
            source_identity_inference_used=True,
            source_identity_confirmed=False,
            source_identity_confidence=0.0,
        )
    )
    low_confidence = analyze_smart_dsp_strategy(
        _analysis(
            muddiness_score=0.8,
            source_identity_confirmed=True,
            source_identity_confidence=0.42,
        )
    )

    for recommendation in (inferred, low_confidence):
        assert recommendation.status == "analysis_only"
        assert (
            recommendation.selected_strategy.strategy_id
            == "hold_for_source_identity_confirmation"
        )
        assert recommendation.selected_strategy.method_family == "analysis_only"
        assert recommendation.selected_strategy.parameters["dsp_change"] == "hold"
        assert recommendation.selected_strategy.parameters[
            "required_context"
        ] == "operator_patch_label_pfl_source_role_confirmation"
        assert recommendation.evidence["operator_explanation_sections"][
            "selection_policy"
        ] == "source_identity_confirmation_guard_priority"
        assert recommendation.to_operator_proposal()["can_apply_live"] is False

    assert "source_identity_not_confirmed" in inferred.blockers
    assert "source_identity_confidence_low" in inferred.blockers
    assert inferred.selected_strategy.parameters["source_identity_inference_used"] is True
    assert inferred.selected_strategy.parameters["inferred_source_family"] == "vocal"
    deesser_tradeoff = [
        row
        for row in inferred.evidence["strategy_search"]["alternative_tradeoffs"]
        if row["strategy_id"] == "vocal_de_esser"
    ][0]
    assert (
        deesser_tradeoff["why_not_selected"]
        == "source_identity_guard_priority:confirm_source_before_dsp"
    )

    assert "source_identity_confidence_low" in low_confidence.blockers
    assert low_confidence.selected_strategy.parameters["source_identity_confirmed"] is True
    low_mid_tradeoff = [
        row
        for row in low_confidence.evidence["strategy_search"]["alternative_tradeoffs"]
        if row["strategy_id"] == "low_mid_corrective_cut"
    ][0]
    assert (
        low_mid_tradeoff["why_not_selected"]
        == "source_identity_guard_priority:confirm_source_before_dsp"
    )


def test_stale_or_live_authority_context_fails_closed():
    stale = analyze_smart_dsp_strategy(
        _analysis(
            muddiness_score=0.7,
            current_state_cache_freshness={
                "status": "stale",
                "read_only": True,
                "live_write": False,
                "queries_console": False,
            },
        )
    )
    live_authority = analyze_smart_dsp_strategy(
        _analysis(
            muddiness_score=0.7,
            current_state_cache_freshness={
                "status": "fresh",
                "read_only": False,
                "live_write": True,
                "queries_console": True,
            },
        )
    )

    assert stale.status == "analysis_only"
    assert "smart_dsp_context_cache_not_fresh" in stale.blockers
    assert (
        stale.selected_strategy.strategy_id
        == "hold_for_context_readback_or_rollback_recheck"
    )
    assert stale.selected_strategy.method_family == "analysis_only"
    assert stale.evidence["operator_explanation_sections"]["selection_policy"] == (
        "context_readback_rollback_guard_priority"
    )
    assert stale.selected_strategy.parameters["freshness_status"] == "stale"
    assert stale.selected_strategy.parameters["global_blockers"] == [
        "smart_dsp_context_cache_not_fresh"
    ]
    low_mid_tradeoff = [
        row
        for row in stale.evidence["strategy_search"]["alternative_tradeoffs"]
        if row["strategy_id"] == "low_mid_corrective_cut"
    ][0]
    assert (
        low_mid_tradeoff["why_not_selected"]
        == "context_guard_priority:refresh_readback_and_rollback_before_dsp"
    )
    assert stale.to_operator_proposal()["can_apply_live"] is False
    assert live_authority.status == "analysis_only"
    assert "smart_dsp_context_cache_live_authority_present" in live_authority.blockers
    assert (
        live_authority.selected_strategy.strategy_id
        == "hold_for_context_readback_or_rollback_recheck"
    )
    assert live_authority.evidence["operator_explanation_sections"][
        "selection_policy"
    ] == "context_readback_rollback_guard_priority"
    assert live_authority.evidence["current_smart_dsp_context_cache_no_live_authority"] is False


def test_missing_readback_or_rollback_context_selects_hold_not_eq():
    missing_readback = analyze_smart_dsp_strategy(
        _analysis(muddiness_score=0.7, current_state_readback_available=False)
    )
    missing_rollback = analyze_smart_dsp_strategy(
        _analysis(muddiness_score=0.7, rollback_plan_available=False)
    )

    for recommendation, blocker in (
        (missing_readback, "current_state_readback_missing"),
        (missing_rollback, "rollback_plan_missing"),
    ):
        assert recommendation.status == "analysis_only"
        assert blocker in recommendation.blockers
        assert (
            recommendation.selected_strategy.strategy_id
            == "hold_for_context_readback_or_rollback_recheck"
        )
        assert recommendation.selected_strategy.method_family == "analysis_only"
        assert recommendation.selected_strategy.parameters["dsp_change"] == "hold"
        assert blocker in recommendation.selected_strategy.parameters["global_blockers"]
        assert recommendation.evidence["operator_explanation_sections"][
            "selection_policy"
        ] == "context_readback_rollback_guard_priority"
        assert recommendation.to_operator_proposal()["can_apply_live"] is False


def test_low_confidence_audio_evidence_holds_before_dsp_selection():
    recommendation = analyze_smart_dsp_strategy(
        _analysis(
            muddiness_score=0.78,
            evidence_confidence=0.42,
            level_plane_error_lu=0.6,
        )
    )

    assert recommendation.status == "analysis_only"
    assert "audio_evidence_confidence_low" in recommendation.blockers
    assert recommendation.selected_strategy.strategy_id == "hold_for_audio_evidence_recheck"
    assert recommendation.selected_strategy.method_family == "analysis_only"
    assert recommendation.selected_strategy.parameters["dsp_change"] == "hold"
    assert recommendation.selected_strategy.parameters["evidence_confidence"] == 0.42
    assert recommendation.selected_strategy.parameters[
        "required_context"
    ] == "repeat_audio_capture_or_add_stronger_evidence"
    assert recommendation.evidence["operator_explanation_sections"][
        "selection_policy"
    ] == "audio_evidence_confidence_guard_priority"
    low_mid_tradeoff = [
        row
        for row in recommendation.evidence["strategy_search"]["alternative_tradeoffs"]
        if row["strategy_id"] == "low_mid_corrective_cut"
    ][0]
    assert (
        low_mid_tradeoff["why_not_selected"]
        == "audio_evidence_guard_priority:repeat_capture_before_dsp"
    )
    assert recommendation.to_operator_proposal()["can_apply_live"] is False


def test_stale_live_audio_evidence_holds_before_dsp_selection():
    soundcheck_recent_enough = analyze_smart_dsp_strategy(
        _analysis(
            muddiness_score=0.78,
            audio_evidence_age_seconds=6.5,
        )
    )
    live_stale = analyze_smart_dsp_strategy(
        _analysis(
            mode="live",
            muddiness_score=0.78,
            audio_evidence_age_seconds=6.5,
        )
    )
    soundcheck_stale = analyze_smart_dsp_strategy(
        _analysis(
            muddiness_score=0.78,
            audio_evidence_age_seconds=31.0,
        )
    )

    assert "audio_evidence_stale" not in soundcheck_recent_enough.blockers
    assert soundcheck_recent_enough.selected_strategy.strategy_id != (
        "hold_for_audio_evidence_recheck"
    )

    assert live_stale.status == "analysis_only"
    assert "audio_evidence_stale" in live_stale.blockers
    assert live_stale.selected_strategy.strategy_id == "hold_for_audio_evidence_recheck"
    assert live_stale.selected_strategy.method_family == "analysis_only"
    assert live_stale.selected_strategy.parameters["dsp_change"] == "hold"
    assert live_stale.selected_strategy.parameters["audio_evidence_age_seconds"] == 6.5
    assert live_stale.selected_strategy.parameters["max_audio_evidence_age_seconds"] == 5.0
    assert live_stale.selected_strategy.parameters["audio_evidence_stale"] is True
    assert live_stale.evidence["operator_explanation_sections"][
        "selection_policy"
    ] == "audio_evidence_freshness_guard_priority"
    low_mid_tradeoff = [
        row
        for row in live_stale.evidence["strategy_search"]["alternative_tradeoffs"]
        if row["strategy_id"] == "low_mid_corrective_cut"
    ][0]
    assert (
        low_mid_tradeoff["why_not_selected"]
        == "audio_evidence_guard_priority:repeat_capture_before_dsp"
    )
    assert live_stale.to_operator_proposal()["can_apply_live"] is False

    assert "audio_evidence_stale" in soundcheck_stale.blockers
    assert (
        soundcheck_stale.selected_strategy.parameters["max_audio_evidence_age_seconds"]
        == 30.0
    )


def test_auto_apply_request_selects_operator_review_hold_not_dsp():
    recommendation = analyze_smart_dsp_strategy(
        _analysis(
            muddiness_score=0.8,
            level_plane_error_lu=0.6,
            auto_apply_requested=True,
        )
    )

    assert recommendation.status == "analysis_only"
    assert "auto_apply_not_supported" in recommendation.blockers
    assert (
        recommendation.selected_strategy.strategy_id
        == "hold_for_operator_review_no_auto_apply"
    )
    assert recommendation.selected_strategy.method_family == "analysis_only"
    assert recommendation.selected_strategy.parameters["dsp_change"] == "hold"
    assert recommendation.selected_strategy.parameters["auto_apply_requested"] is True
    assert recommendation.selected_strategy.parameters["live_authority_granted"] is False
    assert recommendation.selected_strategy.parameters["live_write"] is False
    assert recommendation.selected_strategy.parameters[
        "required_context"
    ] == "operator_review_real_wing_proof_and_explicit_promotion"
    assert recommendation.evidence["operator_explanation_sections"][
        "selection_policy"
    ] == "auto_apply_operator_review_guard_priority"
    low_mid_tradeoff = [
        row
        for row in recommendation.evidence["strategy_search"]["alternative_tradeoffs"]
        if row["strategy_id"] == "low_mid_corrective_cut"
    ][0]
    assert (
        low_mid_tradeoff["why_not_selected"]
        == "auto_apply_guard_priority:operator_review_before_any_apply"
    )
    proposal = recommendation.to_operator_proposal()
    assert proposal["can_apply_live"] is False
    assert proposal["queue_import_allowed"] is False
    assert proposal["live_authority_granted"] is False


def test_unsupported_mode_selects_mode_hold_not_dsp():
    recommendation = analyze_smart_dsp_strategy(
        _analysis(
            mode="studio_recording",
            muddiness_score=0.8,
            level_plane_error_lu=0.6,
        )
    )

    assert recommendation.status == "analysis_only"
    assert "unsupported_smart_dsp_mode" in recommendation.blockers
    assert recommendation.selected_strategy.strategy_id == "hold_for_supported_mode_selection"
    assert recommendation.selected_strategy.method_family == "analysis_only"
    params = recommendation.selected_strategy.parameters
    assert params["dsp_change"] == "hold"
    assert params["requested_mode"] == "studio_recording"
    assert params["supported_modes"] == ["live", "soundcheck"]
    assert params["required_context"] == "choose_soundcheck_or_live_mode_before_dsp"
    assert recommendation.evidence["operator_explanation_sections"][
        "selection_policy"
    ] == "unsupported_mode_guard_priority"
    low_mid_tradeoff = [
        row
        for row in recommendation.evidence["strategy_search"]["alternative_tradeoffs"]
        if row["strategy_id"] == "low_mid_corrective_cut"
    ][0]
    assert (
        low_mid_tradeoff["why_not_selected"]
        == "unsupported_mode_guard_priority:choose_soundcheck_or_live_before_dsp"
    )
    assert recommendation.to_operator_proposal()["can_apply_live"] is False


def test_routing_scene_snapshot_intent_selects_hold_not_dsp():
    recommendation = analyze_smart_dsp_strategy(
        _analysis(
            operator_intent="load scene for chorus and fix muddy vocal",
            problem_hints=("muddy", "scene_change"),
            muddiness_score=0.8,
            level_plane_error_lu=0.6,
        )
    )

    assert recommendation.status == "analysis_only"
    assert "forbidden_routing_scene_snapshot_intent" in recommendation.blockers
    assert (
        recommendation.selected_strategy.strategy_id
        == "hold_for_routing_scene_snapshot_review"
    )
    assert recommendation.selected_strategy.method_family == "analysis_only"
    params = recommendation.selected_strategy.parameters
    assert params["dsp_change"] == "hold"
    assert params["live_authority_granted"] is False
    assert params["live_write"] is False
    assert params["readiness_closure_allowed"] is False
    assert "scene" in params["forbidden_intent_tokens"]
    assert "scene_change" in params["forbidden_intent_tokens"]
    assert params["required_context"] == (
        "operator_review_real_wing_proof_and_explicit_routing_scene_tool"
    )
    assert recommendation.evidence["operator_explanation_sections"][
        "selection_policy"
    ] == "routing_scene_snapshot_guard_priority"
    low_mid_tradeoff = [
        row
        for row in recommendation.evidence["strategy_search"]["alternative_tradeoffs"]
        if row["strategy_id"] == "low_mid_corrective_cut"
    ][0]
    assert (
        low_mid_tradeoff["why_not_selected"]
        == "routing_scene_guard_priority:"
        "operator_review_before_routing_scene_or_snapshot"
    )
    proposal = recommendation.to_operator_proposal()
    assert proposal["can_apply_live"] is False
    assert proposal["queue_import_allowed"] is False
    assert proposal["live_authority_granted"] is False


def test_live_master_output_intent_selects_hold_not_channel_eq():
    recommendation = analyze_smart_dsp_strategy(
        _analysis(
            mode="live",
            operator_intent="raise master output 2 dB during live and clean muddy vocal",
            problem_hints=("muddy",),
            muddiness_score=0.8,
            level_plane_error_lu=0.6,
            active_performance=True,
        )
    )

    assert recommendation.status == "analysis_only"
    assert "live_master_output_intent_requires_review" in recommendation.blockers
    assert (
        recommendation.selected_strategy.strategy_id
        == "hold_for_live_master_output_intent_review"
    )
    assert recommendation.selected_strategy.method_family == "analysis_only"
    params = recommendation.selected_strategy.parameters
    assert params["dsp_change"] == "hold"
    assert params["live_authority_granted"] is False
    assert params["live_write"] is False
    assert params["readiness_closure_allowed"] is False
    assert "master" in params["live_master_output_intent_tokens"]
    assert "output" in params["live_master_output_intent_tokens"]
    assert params["required_context"] == (
        "operator_review_real_wing_proof_pa_feed_identity_readback_rollback"
    )
    assert recommendation.evidence["operator_explanation_sections"][
        "selection_policy"
    ] == "live_master_output_intent_guard_priority"
    low_mid_tradeoff = [
        row
        for row in recommendation.evidence["strategy_search"]["alternative_tradeoffs"]
        if row["strategy_id"] == "low_mid_corrective_cut"
    ][0]
    assert (
        low_mid_tradeoff["why_not_selected"]
        == "live_master_output_intent_guard_priority:"
        "operator_review_before_master_output_change"
    )
    proposal = recommendation.to_operator_proposal()
    assert proposal["can_apply_live"] is False
    assert proposal["queue_import_allowed"] is False
    assert proposal["live_authority_granted"] is False


def test_live_source_gain_up_intent_selects_hold_not_channel_eq():
    recommendation = analyze_smart_dsp_strategy(
        _analysis(
            mode="live",
            operator_intent="boost lead vocal 4 dB during live and clean muddiness",
            problem_hints=("muddy",),
            muddiness_score=0.8,
            level_plane_error_lu=0.2,
            active_performance=True,
        )
    )

    assert recommendation.status == "analysis_only"
    assert "live_source_gain_up_intent_requires_review" in recommendation.blockers
    assert (
        recommendation.selected_strategy.strategy_id
        == "hold_for_live_source_gain_up_intent_review"
    )
    assert recommendation.selected_strategy.method_family == "analysis_only"
    params = recommendation.selected_strategy.parameters
    assert params["dsp_change"] == "hold"
    assert params["live_authority_granted"] is False
    assert params["live_write"] is False
    assert params["readiness_closure_allowed"] is False
    assert "boost" in params["live_source_gain_up_intent_tokens"]
    assert params["required_context"] == (
        "operator_review_real_wing_proof_source_identity_headroom_feedback_readback_rollback"
    )
    assert recommendation.evidence["operator_explanation_sections"][
        "selection_policy"
    ] == "live_source_gain_up_intent_guard_priority"
    low_mid_tradeoff = [
        row
        for row in recommendation.evidence["strategy_search"]["alternative_tradeoffs"]
        if row["strategy_id"] == "low_mid_corrective_cut"
    ][0]
    assert (
        low_mid_tradeoff["why_not_selected"]
        == "live_source_gain_up_intent_guard_priority:"
        "operator_review_before_source_gain_up"
    )
    proposal = recommendation.to_operator_proposal()
    assert proposal["can_apply_live"] is False
    assert proposal["queue_import_allowed"] is False
    assert proposal["live_authority_granted"] is False


def test_live_eq_intent_selects_hold_not_channel_eq_or_gain_up_guard():
    recommendation = analyze_smart_dsp_strategy(
        _analysis(
            mode="live",
            operator_intent="add air eq to vocal live and reduce mud",
            problem_hints=("muddy",),
            muddiness_score=0.8,
            level_plane_error_lu=0.2,
            active_performance=True,
        )
    )
    boost_highs = analyze_smart_dsp_strategy(
        _analysis(
            mode="live",
            operator_intent="boost highs on vocal live and clean muddiness",
            problem_hints=("muddy",),
            muddiness_score=0.8,
            level_plane_error_lu=0.2,
            active_performance=True,
        )
    )

    for result in (recommendation, boost_highs):
        assert result.status == "analysis_only"
        assert "live_eq_intent_requires_review" in result.blockers
        assert "live_source_gain_up_intent_requires_review" not in result.blockers
        assert result.selected_strategy.strategy_id == "hold_for_live_eq_intent_review"
        assert result.selected_strategy.method_family == "analysis_only"
        params = result.selected_strategy.parameters
        assert params["dsp_change"] == "hold"
        assert params["live_authority_granted"] is False
        assert params["live_write"] is False
        assert params["readiness_closure_allowed"] is False
        assert params["required_context"] == (
            "operator_review_real_wing_proof_source_identity_eq_band_readback_rollback"
        )
        assert result.evidence["operator_explanation_sections"][
            "selection_policy"
        ] == "live_eq_intent_guard_priority"
        low_mid_tradeoff = [
            row
            for row in result.evidence["strategy_search"]["alternative_tradeoffs"]
            if row["strategy_id"] == "low_mid_corrective_cut"
        ][0]
        assert (
            low_mid_tradeoff["why_not_selected"]
            == "live_eq_intent_guard_priority:"
            "operator_review_before_live_eq_change"
        )
        proposal = result.to_operator_proposal()
        assert proposal["can_apply_live"] is False
        assert proposal["queue_import_allowed"] is False
        assert proposal["live_authority_granted"] is False

    assert "add_air" in recommendation.selected_strategy.parameters[
        "live_eq_intent_tokens"
    ]
    assert "eq" in recommendation.selected_strategy.parameters["live_eq_intent_tokens"]
    assert "boost_highs" in boost_highs.selected_strategy.parameters[
        "live_eq_intent_tokens"
    ]
    assert "highs" in boost_highs.selected_strategy.parameters["live_eq_intent_tokens"]


def test_live_feedback_risk_beats_intent_only_holds_but_not_audio_evidence_guard():
    cases = (
        (
            {
                "operator_intent": "boost highs on vocal live while it is feeding back",
            },
            "live_eq_intent_requires_review",
            "hold_for_live_eq_intent_review",
        ),
        (
            {
                "operator_intent": "raise lead vocal fader live while it is feeding back",
            },
            "live_source_gain_up_intent_requires_review",
            "hold_for_live_source_gain_up_intent_review",
        ),
        (
            {
                "operator_intent": "add more reverb live while vocal is feeding back",
            },
            "live_processing_intent_requires_review",
            "hold_for_live_processing_intent_review",
        ),
        (
            {
                "target_id": "ch:12",
                "source_role": "electric_guitar",
                "source_label": "GTR L",
                "operator_intent": "pan guitar left live while monitor is ringing",
            },
            "live_pan_width_intent_requires_review",
            "hold_for_live_pan_width_intent_review",
        ),
    )

    for overrides, blocker, hold_strategy_id in cases:
        recommendation = analyze_smart_dsp_strategy(
            _analysis(
                mode="live",
                problem_hints=("muddy",),
                muddiness_score=0.7,
                feedback_risk=0.86,
                active_performance=True,
                **overrides,
            )
        )

        assert recommendation.status == "analysis_only"
        assert blocker in recommendation.blockers
        assert recommendation.selected_strategy.strategy_id == "feedback_gain_pullback"
        assert recommendation.selected_strategy.method_family == "gain_fader"
        assert recommendation.selected_strategy.parameters["delta_db"] < 0.0
        assert recommendation.evidence["operator_explanation_sections"][
            "selection_policy"
        ] == "live_feedback_risk_priority"
        hold_tradeoff = [
            row
            for row in recommendation.evidence["strategy_search"]["alternative_tradeoffs"]
            if row["strategy_id"] == hold_strategy_id
        ][0]
        assert (
            hold_tradeoff["why_not_selected"]
            == "live_feedback_risk_priority:"
            "feedback_reduction_precedes_intent_review_hold"
        )
        proposal = recommendation.to_operator_proposal()
        assert proposal["can_apply_live"] is False
        assert proposal["queue_import_allowed"] is False
        assert proposal["live_authority_granted"] is False

    low_evidence = analyze_smart_dsp_strategy(
        _analysis(
            mode="live",
            operator_intent="boost highs on vocal live while it is feeding back",
            problem_hints=("muddy",),
            muddiness_score=0.7,
            feedback_risk=0.86,
            evidence_confidence=0.4,
            active_performance=True,
        )
    )

    assert low_evidence.status == "analysis_only"
    assert "audio_evidence_confidence_low" in low_evidence.blockers
    assert "live_eq_intent_requires_review" in low_evidence.blockers
    assert low_evidence.selected_strategy.strategy_id == "hold_for_audio_evidence_recheck"
    assert low_evidence.evidence["operator_explanation_sections"][
        "selection_policy"
    ] == "audio_evidence_confidence_guard_priority"


def test_live_processing_change_intent_selects_hold_not_channel_eq():
    recommendation = analyze_smart_dsp_strategy(
        _analysis(
            mode="live",
            operator_intent="slam compressor on lead vocal live and clean muddiness",
            problem_hints=("muddy",),
            muddiness_score=0.8,
            level_plane_error_lu=0.2,
            active_performance=True,
        )
    )

    assert recommendation.status == "analysis_only"
    assert "live_processing_intent_requires_review" in recommendation.blockers
    assert (
        recommendation.selected_strategy.strategy_id
        == "hold_for_live_processing_intent_review"
    )
    assert recommendation.selected_strategy.method_family == "analysis_only"
    params = recommendation.selected_strategy.parameters
    assert params["dsp_change"] == "hold"
    assert params["live_authority_granted"] is False
    assert params["live_write"] is False
    assert params["readiness_closure_allowed"] is False
    assert "slam" in params["live_processing_intent_tokens"]
    assert "compressor" in params["live_processing_intent_tokens"]
    assert params["required_context"] == (
        "operator_review_real_wing_proof_source_identity_readback_rollback"
    )
    assert recommendation.evidence["operator_explanation_sections"][
        "selection_policy"
    ] == "live_processing_intent_guard_priority"
    low_mid_tradeoff = [
        row
        for row in recommendation.evidence["strategy_search"]["alternative_tradeoffs"]
        if row["strategy_id"] == "low_mid_corrective_cut"
    ][0]
    assert (
        low_mid_tradeoff["why_not_selected"]
        == "live_processing_intent_guard_priority:"
        "operator_review_before_live_processing_change"
    )
    proposal = recommendation.to_operator_proposal()
    assert proposal["can_apply_live"] is False
    assert proposal["queue_import_allowed"] is False
    assert proposal["live_authority_granted"] is False

    soundcheck = analyze_smart_dsp_strategy(
        _analysis(
            mode="soundcheck",
            operator_intent="slam compressor on lead vocal and clean muddiness",
            problem_hints=("muddy",),
            muddiness_score=0.8,
            level_plane_error_lu=0.2,
        )
    )
    assert "live_processing_intent_requires_review" not in soundcheck.blockers
    assert soundcheck.selected_strategy.strategy_id == "low_mid_corrective_cut"


def test_live_pan_width_intent_selects_hold_not_channel_eq():
    recommendation = analyze_smart_dsp_strategy(
        _analysis(
            target_id="ch:12",
            source_role="electric_guitar",
            source_label="Guitar L",
            mode="live",
            operator_intent="pan guitar left live and clean muddiness",
            problem_hints=("muddy",),
            muddiness_score=0.8,
            level_plane_error_lu=0.2,
            phase_correlation=0.92,
            mono_sum_delta_db=-0.6,
            active_performance=True,
        )
    )

    assert recommendation.status == "analysis_only"
    assert "live_pan_width_intent_requires_review" in recommendation.blockers
    assert (
        recommendation.selected_strategy.strategy_id
        == "hold_for_live_pan_width_intent_review"
    )
    assert recommendation.selected_strategy.method_family == "analysis_only"
    params = recommendation.selected_strategy.parameters
    assert params["dsp_change"] == "hold"
    assert params["live_authority_granted"] is False
    assert params["live_write"] is False
    assert params["readiness_closure_allowed"] is False
    assert "pan" in params["live_pan_width_intent_tokens"]
    assert "left" in params["live_pan_width_intent_tokens"]
    assert params["required_context"] == (
        "operator_review_real_wing_proof_source_identity_phase_mono_readback_rollback"
    )
    assert recommendation.evidence["operator_explanation_sections"][
        "selection_policy"
    ] == "live_pan_width_intent_guard_priority"
    low_mid_tradeoff = [
        row
        for row in recommendation.evidence["strategy_search"]["alternative_tradeoffs"]
        if row["strategy_id"] == "low_mid_corrective_cut"
    ][0]
    assert (
        low_mid_tradeoff["why_not_selected"]
        == "live_pan_width_intent_guard_priority:"
        "operator_review_before_live_pan_width_change"
    )
    proposal = recommendation.to_operator_proposal()
    assert proposal["can_apply_live"] is False
    assert proposal["queue_import_allowed"] is False
    assert proposal["live_authority_granted"] is False

    soundcheck = analyze_smart_dsp_strategy(
        _analysis(
            target_id="ch:12",
            source_role="electric_guitar",
            source_label="Guitar L",
            mode="soundcheck",
            operator_intent="pan guitar left and clean muddiness",
            problem_hints=("muddy",),
            muddiness_score=0.8,
            level_plane_error_lu=0.2,
        )
    )
    assert "live_pan_width_intent_requires_review" not in soundcheck.blockers
    assert soundcheck.selected_strategy.strategy_id == "low_mid_corrective_cut"


def test_master_overload_prefers_trim_before_bus_processing():
    recommendation = analyze_smart_dsp_strategy(
        _analysis(
            target_id="main",
            source_role="master_main",
            problem_hints=("master_overload",),
            muddiness_score=0.0,
            master_true_peak_db=-0.2,
            master_integrated_lufs=-17.0,
            dynamics_peak_to_median_db=14.0,
            dynamics_rise_time_ms=10.0,
            dynamics_recovery_ms=140.0,
            low_end_excess_score=0.4,
        )
    )

    assert recommendation.status == "proposal_only"
    assert recommendation.selected_strategy.strategy_id == "master_trim_for_true_peak_headroom"
    assert recommendation.selected_strategy.method_family == "gain_fader"
    assert recommendation.selected_strategy.parameters["trim_delta_db"] < 0.0
    assert recommendation.selected_strategy.parameters["pa_reference_feed_check_required"] is True
    _assert_gain_delta_readback_contract(
        recommendation.selected_strategy,
        source="trim_delta_db",
        target_surface="main",
        expected_delta=-1.3,
        path_prefix="/main/{main}",
    )
    _assert_soundcheck_master_output_guard(
        recommendation.selected_strategy,
        move_kind="pullback",
        followup_review=True,
    )
    limiter = [
        candidate
        for candidate in recommendation.alternatives
        if candidate.strategy_id == "master_limiter_ceiling_guard"
    ][0]
    assert "limiter_change_requires_pa_feed_safety" in limiter.blockers
    assert limiter.parameters["master_limiter_parameter_source"] == (
        "measured_master_peak_envelope"
    )
    assert limiter.parameters["target_gain_reduction_db"] == 0.8
    assert limiter.parameters["master_true_peak_overshoot_db"] == 0.8
    assert limiter.parameters["attack_ms"] == 1.5
    assert limiter.parameters["release_ms"] == 133.0
    assert limiter.parameters["lookahead_ms"] == 1.6
    assert limiter.parameters["dynamics_peak_to_median_db"] == 14.0
    assert limiter.parameters["dynamics_rise_time_ms"] == 10.0
    assert limiter.parameters["dynamics_recovery_ms"] == 140.0
    assert limiter.parameters["limiter_cannot_restore_static_headroom"] is False
    bus_comp = [
        candidate
        for candidate in recommendation.alternatives
        if candidate.strategy_id == "gentle_bus_compression"
    ][0]
    assert "master_true_peak_trim_should_precede_bus_compression" in bus_comp.blockers
    proposal = recommendation.to_operator_proposal()
    _assert_proposal_denial(proposal)
    assert proposal["proposed_action"]["method_family"] == "gain_fader"


def test_live_master_true_peak_overload_prefers_native_main_pullback_before_processing():
    recommendation = analyze_smart_dsp_strategy(
        _analysis(
            target_id="main",
            source_role="master_main",
            mode="live",
            problem_hints=("master_overload",),
            active_performance=True,
            muddiness_score=0.0,
            master_true_peak_db=-0.2,
            true_peak_headroom_db=4.0,
            master_integrated_lufs=-18.0,
            dynamics_peak_to_median_db=14.0,
            dynamics_rise_time_ms=10.0,
            dynamics_recovery_ms=140.0,
            low_end_excess_score=0.4,
            feedback_risk=0.12,
            monitor_feedback_risk=0.1,
        )
    )

    assert recommendation.status == "proposal_only"
    assert recommendation.selected_strategy.strategy_id == (
        "master_trim_for_true_peak_headroom"
    )
    assert recommendation.selected_strategy.method_family == "gain_fader"
    assert recommendation.evidence["operator_explanation_sections"][
        "selection_policy"
    ] == "master_true_peak_headroom_priority"
    params = recommendation.selected_strategy.parameters
    assert params["trim_delta_db"] == -1.3
    assert params["master_true_peak_pullback_db"] == 1.3
    assert params["headroom_deficit_pullback_db"] == 0.0
    assert params["pa_reference_feed_check_required"] is True
    _assert_gain_delta_readback_contract(
        recommendation.selected_strategy,
        source="trim_delta_db",
        target_surface="main",
        expected_delta=-1.3,
        path_prefix="/main/{main}",
    )
    _assert_live_master_true_peak_output_guard(recommendation.selected_strategy)
    limiter = [
        candidate
        for candidate in recommendation.alternatives
        if candidate.strategy_id == "master_limiter_ceiling_guard"
    ][0]
    assert "limiter_change_requires_pa_feed_safety" in limiter.blockers
    bus_comp = [
        candidate
        for candidate in recommendation.alternatives
        if candidate.strategy_id == "gentle_bus_compression"
    ][0]
    assert "master_true_peak_trim_should_precede_bus_compression" in (
        bus_comp.blockers
    )
    assert (
        "live_master_bus_compression_requires_operator_review_and_wing_proof"
        in bus_comp.blockers
    )
    contract = recommendation.evidence["decision_engine_contract"]
    assert contract["contract_id"] == "smart_dsp_decision_engine_contract_v1"
    assert contract["method_id"] == SMART_DSP_METHOD_ID
    mode_policy = contract["mode_policy"]
    assert mode_policy["mode"] == "live"
    assert mode_policy["soundcheck_strategy_search_allowed"] is False
    assert mode_policy["soundcheck_bounded_exploration_allowed"] is False
    assert mode_policy["live_conservative_policy"] is True
    assert mode_policy["live_prefers_risk_reduction"] is True
    assert mode_policy["live_gain_up_requires_operator_review"] is True
    assert mode_policy["live_aggressive_processing_requires_operator_review"] is True
    assert mode_policy["live_routing_scene_master_requires_supervised_proof"] is True
    assert mode_policy["live_max_single_step_db"] == LIVE_MAX_SINGLE_STEP_DB
    assert mode_policy["live_min_step_interval_ms"] == LIVE_MIN_STEP_INTERVAL_MS
    assert mode_policy["live_eq_max_boost_db"] == LIVE_EQ_MAX_BOOST_DB
    assert mode_policy["live_eq_max_cut_db"] == LIVE_EQ_MAX_CUT_DB
    assert mode_policy["live_fx_max_boost_db"] == LIVE_FX_MAX_BOOST_DB
    assert mode_policy["live_fx_max_pullback_db"] == LIVE_FX_MAX_PULLBACK_DB
    assert mode_policy["live_apply_allowed"] is False
    assert mode_policy["live_authority_granted"] is False
    strategy_search = contract["strategy_search"]
    assert strategy_search["score_weights"] == LIVE_STRATEGY_SCORE_WEIGHTS
    assert strategy_search["score_weight_profile"] == "live_weighted_strategy_search"
    assert strategy_search["score_weights_equal"] is False
    assert strategy_search["mode_specific_score_weights"] is True
    _assert_weighted_score_summary(
        strategy_search["selected_weighted_score_summary"],
        mode="live",
        expected_total_score=recommendation.selected_strategy.total_score,
    )
    parameter_contract = contract["parameter_calculation"]
    assert "master_true_peak_pullback_db" in parameter_contract["parameter_keys"]
    assert "trim_delta_db" in parameter_contract["parameter_keys"]
    assert parameter_contract["live_apply_blocked"] is True
    assert parameter_contract["live_apply_blocker_status"] == "blocked_before_live_apply"
    assert parameter_contract["operator_review_required_before_live_apply"] is True
    assert (
        parameter_contract["wing_osc_candidate_apply_status"]
        == "wing_native_osc_candidate_requires_absolute_value_readback_before_apply"
    )
    safety = contract["safety_layer"]
    assert safety["true_peak_headroom_guard"] is True
    assert safety["feedback_guard"] is True
    assert safety["live_gain_up_blocked_without_operator_review"] is True
    assert safety["live_aggressive_dsp_blocked_without_operator_review"] is True
    assert (
        safety["routing_scene_master_live_write_blocked_without_supervised_proof"]
        is True
    )
    assert safety["wing_fw305_verified_mapping_readback_rollback_required"] is True
    assert "master_output" in safety["unverified_dsp_families_operator_review_only"]
    assert safety["live_write"] is False
    assert safety["queue_mutation"] is False
    assert safety["live_authority_granted"] is False
    readback_contract = contract["wing_readback_contract"]
    assert readback_contract["target_console"] == WING_RACK_OSC_TARGET_CONSOLE
    assert readback_contract["firmware"] == WING_RACK_OSC_FIRMWARE
    assert readback_contract["correction_exists_on_console_required"] is True
    assert readback_contract["verified_osc_mapping_required"] is True
    assert readback_contract["verified_readback_required"] is True
    assert readback_contract["verified_rollback_required"] is True
    assert readback_contract["read_only_until_verified"] is True
    assert "fdr" in readback_contract["selected_read_before_paths"]
    assert "in/set/trim" in readback_contract["selected_read_before_paths"]
    assert readback_contract["live_write"] is False
    assert readback_contract["live_authority_granted"] is False
    proposal = recommendation.to_operator_proposal()
    _assert_proposal_denial(proposal)
    assert proposal["proposed_action"]["method_family"] == "gain_fader"
    assert proposal["decision_engine_contract"] == contract
    assert proposal["proposed_action"]["decision_engine_contract"] == contract
    assert proposal["evidence"]["decision_engine_contract"] == contract


def _assert_operator_owned_master_output_gain_hold(
    recommendation,
    *,
    mode,
    expected_strategy_id,
    expected_policy,
    blocked_strategy_id,
    blocked_delta_db,
    move_kind,
    tradeoff_reason,
):
    assert recommendation.status == "analysis_only"
    selected = recommendation.selected_strategy
    assert selected.strategy_id == expected_strategy_id
    assert selected.method_family == "analysis_only"
    assert selected.blockers == (
        f"{mode}_operator_owned_master_output_gain_requires_operator_review",
    )
    assert recommendation.evidence["operator_explanation_sections"][
        "selection_policy"
    ] == expected_policy

    params = selected.parameters
    assert params["dsp_change"] == "hold"
    assert params["reason"] == (
        f"{mode}_operator_owned_master_output_gain_requires_operator_review"
    )
    assert blocked_strategy_id in params["blocked_master_output_gain_strategy_ids"]
    assert params["blocked_master_output_gain_delta_db"] == blocked_delta_db
    assert params["blocked_master_output_gain_move_kind"] == move_kind
    assert params["wing_gain_current_operator_owned"] is True
    assert params["wing_gain_operator_owned_edit_requires_review"] is True
    assert params["wing_gain_operator_owned_blocker"] == WING_GAIN_OPERATOR_OWNED_BLOCKER
    assert params[f"{mode}_operator_owned_master_output_gain_hold_proposal"] is True
    assert params[f"{mode}_operator_owned_master_output_gain_target_surface"] == "main"
    assert params[f"{mode}_operator_owned_master_output_gain_move_kind"] == move_kind
    assert params[f"{mode}_operator_owned_master_output_gain_readback_required"] is True
    assert params[f"{mode}_operator_owned_master_output_gain_rollback_ready"] is True
    assert params[f"{mode}_operator_owned_master_output_gain_selected_write_keys"] == []
    assert params[f"{mode}_operator_owned_master_output_gain_forbidden_write_keys"] == [
        "wing_gain_delta_db"
    ]
    assert (
        params[
            f"{mode}_operator_owned_master_output_gain_live_apply_blocked_without_wing_proof"
        ]
        is True
    )
    assert params["wing_rack_fw305_correction_exists_on_console_required"] is True
    assert params["wing_rack_fw305_verified_osc_mapping_required"] is True
    assert params["wing_rack_fw305_main_gain_readback_required"] is True
    assert params["wing_rack_fw305_rollback_required"] is True
    assert params["operator_confirmation_required"] is True
    assert params["live_authority_granted"] is False
    assert params["live_write"] is False
    assert params["readiness_closure_allowed"] is False
    assert "wing_gain_delta_db" not in params

    blocked_move = [
        candidate
        for candidate in recommendation.alternatives
        if candidate.strategy_id == blocked_strategy_id
    ][0]
    assert WING_GAIN_OPERATOR_OWNED_BLOCKER in blocked_move.blockers

    tradeoffs = {
        item["strategy_id"]: item["why_not_selected"]
        for item in recommendation.evidence["strategy_search"][
            "alternative_tradeoffs"
        ]
    }
    assert tradeoffs[blocked_strategy_id] == tradeoff_reason

    readback = selected.future_wing_readback_contract[
        "operator_owned_master_output_gain_readback"
    ]
    assert readback["read_only"] is True
    assert readback["read_before_paths"] == ["fdr", "in/set/trim"]
    assert readback["read_after_paths"] == ["fdr", "in/set/trim"]
    assert "/main/{main}/fdr" in readback["wing_osc_path_templates"]
    assert "/main/{main}/in/set/trim" in readback["wing_osc_path_templates"]
    assert readback["allowed_live_write_scope_before_wing_proof"] == []
    assert readback["wing_master_output_gain_forbidden_apply_fields"] == [
        "wing_gain_delta_db"
    ]
    assert readback["blocked_master_output_gain_delta_db"] == blocked_delta_db
    assert readback["blocked_master_output_gain_move_kind"] == move_kind
    assert readback["wing_gain_current_operator_owned"] is True
    assert readback["wing_gain_operator_owned_blocker"] == WING_GAIN_OPERATOR_OWNED_BLOCKER

    proposal = recommendation.to_operator_proposal()
    assert proposal["selected_wing_osc_summary"][
        "wing_osc_candidate_apply_status"
    ] == "analysis_only_no_console_write"
    assert proposal["selected_wing_osc_summary"][
        "wing_osc_apply_parameter_values"
    ] == {}
    assert proposal["can_apply_live"] is False
    assert proposal["live_authority_granted"] is False
    assert proposal["live_write"] is False


def test_soundcheck_operator_owned_master_output_gain_holds_before_true_peak_pullback():
    recommendation = analyze_smart_dsp_strategy(
        _analysis(
            target_id="main",
            source_role="master_main",
            problem_hints=("master_overload",),
            muddiness_score=0.0,
            master_true_peak_db=-0.2,
            true_peak_headroom_db=4.0,
            master_integrated_lufs=-17.0,
            dynamics_peak_to_median_db=14.0,
            dynamics_rise_time_ms=10.0,
            dynamics_recovery_ms=140.0,
            current_wing_gain_operator_owned=True,
        )
    )

    _assert_operator_owned_master_output_gain_hold(
        recommendation,
        mode="soundcheck",
        expected_strategy_id="hold_soundcheck_operator_owned_master_output_gain_for_review",
        expected_policy="soundcheck_operator_owned_master_output_gain_review_priority",
        blocked_strategy_id="master_trim_for_true_peak_headroom",
        blocked_delta_db=-1.3,
        move_kind="pullback",
        tradeoff_reason=(
            "soundcheck_operator_owned_master_output_gain_review_priority:"
            "manual_main_fader_or_trim_owner_review_precedes_soundcheck_master_output_move"
        ),
    )


def test_soundcheck_operator_owned_quiet_master_output_holds_before_lift():
    recommendation = analyze_smart_dsp_strategy(
        _analysis(
            target_id="main",
            source_role="master_main",
            source_label="Main LR",
            problem_hints=("master_overload",),
            operator_intent="soundcheck main too quiet but peaks safe",
            muddiness_score=0.0,
            rms_dbfs=-27.0,
            lra_proxy_db=5.0,
            dynamics_variance_score=0.16,
            dynamics_peak_to_median_db=3.0,
            dynamics_rise_time_ms=0.0,
            dynamics_recovery_ms=0.0,
            master_true_peak_db=-9.0,
            true_peak_headroom_db=8.0,
            master_integrated_lufs=-25.0,
            low_end_excess_score=0.12,
            spectral_tilt_score=0.18,
            presence_harshness_score=0.05,
            current_wing_gain_operator_owned=True,
        )
    )

    _assert_operator_owned_master_output_gain_hold(
        recommendation,
        mode="soundcheck",
        expected_strategy_id="hold_soundcheck_operator_owned_master_output_gain_for_review",
        expected_policy="soundcheck_operator_owned_master_output_gain_review_priority",
        blocked_strategy_id="master_output_lift_for_quiet_soundcheck",
        blocked_delta_db=3.0,
        move_kind="lift",
        tradeoff_reason=(
            "soundcheck_operator_owned_master_output_gain_review_priority:"
            "manual_main_fader_or_trim_owner_review_precedes_soundcheck_master_output_move"
        ),
    )


def test_live_operator_owned_master_output_gain_holds_before_true_peak_pullback():
    recommendation = analyze_smart_dsp_strategy(
        _analysis(
            target_id="main",
            source_role="master_main",
            mode="live",
            problem_hints=("master_overload",),
            active_performance=True,
            muddiness_score=0.0,
            master_true_peak_db=-0.2,
            true_peak_headroom_db=4.0,
            master_integrated_lufs=-18.0,
            dynamics_peak_to_median_db=14.0,
            dynamics_rise_time_ms=10.0,
            dynamics_recovery_ms=140.0,
            low_end_excess_score=0.4,
            feedback_risk=0.12,
            monitor_feedback_risk=0.1,
            current_wing_gain_operator_owned=True,
        )
    )

    _assert_operator_owned_master_output_gain_hold(
        recommendation,
        mode="live",
        expected_strategy_id="hold_live_operator_owned_master_output_gain_for_review",
        expected_policy="live_operator_owned_master_output_gain_review_priority",
        blocked_strategy_id="master_trim_for_true_peak_headroom",
        blocked_delta_db=-1.3,
        move_kind="pullback",
        tradeoff_reason=(
            "live_operator_owned_master_output_gain_review_priority:"
            "manual_main_fader_or_trim_owner_review_precedes_live_master_output_move"
        ),
    )


def test_master_true_peak_trim_beats_source_vocal_compression():
    recommendation = analyze_smart_dsp_strategy(
        _analysis(
            target_id="ch:7",
            source_role="lead_vocal",
            source_label="Lead Vox",
            problem_hints=("dynamic_control", "master_overload"),
            operator_intent="soundcheck vocal jumps while main true peak is unsafe",
            muddiness_score=0.0,
            rms_dbfs=-26.0,
            estimated_lufs=-26.8,
            lra_proxy_db=18.0,
            dynamics_variance_score=0.7,
            crest_factor_db=20.0,
            dynamics_peak_to_median_db=12.0,
            dynamics_rise_time_ms=120.0,
            dynamics_recovery_ms=180.0,
            master_true_peak_db=-0.2,
            true_peak_headroom_db=4.0,
            master_integrated_lufs=-18.0,
        )
    )

    assert recommendation.status == "proposal_only"
    assert recommendation.selected_strategy.strategy_id == "master_trim_for_true_peak_headroom"
    assert recommendation.selected_strategy.method_family == "gain_fader"
    assert recommendation.evidence["operator_explanation_sections"][
        "selection_policy"
    ] == "master_true_peak_headroom_priority"
    compressor = [
        candidate
        for candidate in recommendation.alternatives
        if candidate.strategy_id == "vocal_phrase_leveling_compression"
    ][0]
    assert compressor.problem == "dynamic_control"
    assert compressor.total_score > recommendation.selected_strategy.total_score
    tradeoff = [
        item
        for item in recommendation.evidence["strategy_search"][
            "alternative_tradeoffs"
        ]
        if item["strategy_id"] == "vocal_phrase_leveling_compression"
    ][0]
    assert tradeoff["why_not_selected"] == (
        "master_true_peak_headroom_priority:"
        "restore_master_headroom_before_source_tone_dynamics_fx_or_gain"
    )
    params = recommendation.selected_strategy.parameters
    assert params["trim_delta_db"] < 0.0
    assert params["master_true_peak_pullback_db"] == 1.3
    assert params["pa_reference_feed_check_required"] is True
    _assert_gain_delta_readback_contract(
        recommendation.selected_strategy,
        source="trim_delta_db",
        target_surface="main",
        expected_delta=-1.3,
        path_prefix="/main/{main}",
    )
    _assert_soundcheck_master_output_guard(
        recommendation.selected_strategy,
        move_kind="pullback",
        followup_review=True,
    )
    _assert_master_output_target_override(
        recommendation.selected_strategy,
        original_target_id="ch:7",
    )
    _assert_proposal_denial(recommendation.to_operator_proposal())


def test_master_true_peak_trim_beats_source_pullback_when_global_headroom_low():
    recommendation = analyze_smart_dsp_strategy(
        _analysis(
            target_id="ch:7",
            source_role="lead_vocal",
            source_label="Lead Vox",
            problem_hints=("dynamic_control", "master_overload"),
            operator_intent="soundcheck vocal and main both show low headroom",
            muddiness_score=0.0,
            rms_dbfs=-26.0,
            estimated_lufs=-26.8,
            lra_proxy_db=18.0,
            dynamics_variance_score=0.7,
            crest_factor_db=20.0,
            dynamics_peak_to_median_db=12.0,
            dynamics_rise_time_ms=120.0,
            dynamics_recovery_ms=180.0,
            master_true_peak_db=-0.2,
            true_peak_headroom_db=0.6,
            master_integrated_lufs=-18.0,
        )
    )

    assert recommendation.status == "analysis_only"
    assert "true_peak_headroom_low" in recommendation.blockers
    assert recommendation.selected_strategy.strategy_id == "master_trim_for_true_peak_headroom"
    assert recommendation.selected_strategy.method_family == "gain_fader"
    assert recommendation.evidence["operator_explanation_sections"][
        "selection_policy"
    ] == "master_true_peak_headroom_priority"
    params = recommendation.selected_strategy.parameters
    assert params["trim_delta_db"] == -1.3
    assert params["headroom_deficit_pullback_db"] == 0.9
    assert params["master_true_peak_pullback_db"] == 1.3
    assert params["true_peak_headroom_db"] == 0.6
    _assert_gain_delta_readback_contract(
        recommendation.selected_strategy,
        source="trim_delta_db",
        target_surface="main",
        expected_delta=-1.3,
        path_prefix="/main/{main}",
    )
    _assert_soundcheck_master_output_guard(
        recommendation.selected_strategy,
        move_kind="pullback",
        followup_review=True,
    )
    _assert_master_output_target_override(
        recommendation.selected_strategy,
        original_target_id="ch:7",
    )
    source_pullback = [
        candidate
        for candidate in recommendation.alternatives
        if candidate.strategy_id == "source_true_peak_headroom_pullback"
    ][0]
    assert source_pullback.parameters["headroom_restoration_first"] is True
    tradeoff = [
        item
        for item in recommendation.evidence["strategy_search"][
            "alternative_tradeoffs"
        ]
        if item["strategy_id"] == "source_true_peak_headroom_pullback"
    ][0]
    assert tradeoff["why_not_selected"] == (
        "master_true_peak_headroom_priority:"
        "restore_master_headroom_before_source_tone_dynamics_fx_or_gain"
    )
    proposal = recommendation.to_operator_proposal()
    assert proposal["selected_wing_osc_summary"][
        "wing_osc_apply_parameter_values"
    ] == {}
    assert proposal["selected_wing_osc_readback_summary"][
        "wing_osc_readback_apply_parameter_values"
    ] == {}
    _assert_proposal_denial(recommendation.to_operator_proposal())


def test_master_low_headroom_trim_uses_headroom_deficit_before_tone_or_compression():
    recommendation = analyze_smart_dsp_strategy(
        _analysis(
            target_id="main",
            source_role="master_main",
            source_label="Main LR",
            problem_hints=("master_overload", "muddy", "harsh"),
            operator_intent="soundcheck main low headroom with tone and dynamics pressure",
            muddiness_score=0.8,
            low_end_excess_score=0.82,
            spectral_tilt_score=0.7,
            harshness_score=0.72,
            presence_harshness_score=0.7,
            dynamics_variance_score=0.7,
            lra_proxy_db=12.0,
            rms_dbfs=-15.0,
            master_true_peak_db=-4.0,
            true_peak_headroom_db=0.2,
            master_integrated_lufs=-16.5,
        )
    )

    assert recommendation.status == "analysis_only"
    assert "true_peak_headroom_low" in recommendation.blockers
    assert recommendation.selected_strategy.strategy_id == "master_trim_for_true_peak_headroom"
    assert recommendation.selected_strategy.method_family == "gain_fader"
    params = recommendation.selected_strategy.parameters
    assert params["trim_delta_db"] == -1.3
    assert params["true_peak_headroom_db"] == 0.2
    assert params["target_true_peak_headroom_db"] == 1.5
    assert params["headroom_deficit_pullback_db"] == 1.3
    assert params["master_true_peak_pullback_db"] == 0.0
    _assert_gain_delta_readback_contract(
        recommendation.selected_strategy,
        source="trim_delta_db",
        target_surface="main",
        expected_delta=-1.3,
        path_prefix="/main/{main}",
    )
    _assert_soundcheck_master_output_guard(
        recommendation.selected_strategy,
        move_kind="pullback",
        followup_review=True,
    )
    assert recommendation.evidence["operator_explanation_sections"][
        "selection_policy"
    ] == "master_true_peak_headroom_priority"
    tradeoffs = {
        row["strategy_id"]: row
        for row in recommendation.evidence["strategy_search"]["alternative_tradeoffs"]
    }
    for strategy_id in (
        "master_low_end_broad_eq_cleanup",
        "master_presence_broad_eq_cleanup",
        "master_limiter_ceiling_guard",
        "gentle_bus_compression",
    ):
        assert tradeoffs[strategy_id]["why_not_selected"] == (
            "master_true_peak_headroom_priority:"
            "restore_master_headroom_before_limiter_tone_or_bus_compression"
        )
    limiter = [
        candidate
        for candidate in recommendation.alternatives
        if candidate.strategy_id == "master_limiter_ceiling_guard"
    ][0]
    assert limiter.parameters["master_limiter_parameter_source"] == (
        "true_peak_headroom_fallback"
    )
    assert limiter.parameters["headroom_deficit_pullback_db"] == 1.3
    assert limiter.parameters["master_true_peak_overshoot_db"] == 0.0
    assert limiter.parameters["limiter_cannot_restore_static_headroom"] is True
    _assert_proposal_denial(recommendation.to_operator_proposal())


def test_master_reference_feed_holds_instead_of_master_bus_processing():
    recommendation = analyze_smart_dsp_strategy(
        _analysis(
            target_id="dante:63",
            source_role="master_reference_feed",
            source_label="Dante Reference Capture",
            operator_intent="analyze recording reference feed",
            problem_hints=("master_overload",),
            muddiness_score=0.0,
            master_true_peak_db=-0.2,
            master_integrated_lufs=-17.0,
            true_peak_headroom_db=0.8,
        )
    )

    assert recommendation.status == "analysis_only"
    assert recommendation.selected_strategy.strategy_id == (
        "hold_master_reference_feed_no_pa_write"
    )
    assert recommendation.selected_strategy.method_family == "analysis_only"
    assert "master_reference_or_capture_feed_not_pa_output" in recommendation.blockers
    assert recommendation.evidence["master_feed_identity"][
        "reference_or_capture_feed"
    ] is True
    assert recommendation.selected_strategy.parameters[
        "requires_pa_output_identity"
    ] is True
    alternatives = {
        candidate.strategy_id: candidate for candidate in recommendation.alternatives
    }
    assert set(alternatives) == {
        "blocked_reference_feed_master_trim_candidate",
        "blocked_reference_feed_master_processing_candidate",
    }
    for candidate in alternatives.values():
        assert candidate.method_family == "analysis_only"
        assert candidate.parameters["dsp_change"] == "blocked_candidate"
        assert candidate.parameters["requires_pa_output_identity"] is True
        assert candidate.parameters["operator_confirmation_required"] is True
        assert candidate.parameters["live_authority_granted"] is False
        assert candidate.parameters["live_write"] is False
        assert (
            "master_reference_or_capture_feed_not_pa_output"
            in candidate.blockers
        )
    _assert_proposal_denial(recommendation.to_operator_proposal())


def test_reference_capture_channel_feed_holds_before_source_dsp():
    recommendation = analyze_smart_dsp_strategy(
        _analysis(
            target_id="ch:7",
            source_role="vocal",
            source_label="Dante Reference Capture",
            operator_intent="analyze reference recording feed before vocal eq",
            problem_hints=("muddy",),
            muddiness_score=0.86,
            low_end_excess_score=0.74,
            spectral_tilt_score=0.68,
            true_peak_headroom_db=4.2,
        )
    )

    assert recommendation.status == "analysis_only"
    assert recommendation.selected_strategy.strategy_id == (
        "hold_reference_capture_feed_no_console_write"
    )
    assert recommendation.selected_strategy.method_family == "analysis_only"
    assert (
        "reference_or_capture_feed_requires_console_target_confirmation"
        in recommendation.blockers
    )
    assert recommendation.evidence["master_feed_identity"][
        "reference_or_capture_feed"
    ] is True
    assert recommendation.evidence["operator_explanation_sections"][
        "selection_policy"
    ] == "reference_capture_feed_guard_priority"
    params = recommendation.selected_strategy.parameters
    assert params["requires_concrete_wing_console_target"] is True
    assert params["requires_pa_or_source_output_identity"] is True
    assert params["wing_fw305_osc_target_binding_required"] is True
    assert params["wing_fw305_correction_exists_on_console_required"] is True
    assert params["target_surface_kind"] == "channel"
    assert recommendation.alternatives == ()
    _assert_proposal_denial(recommendation.to_operator_proposal())


def test_master_low_end_overload_prefers_broad_eq_when_true_peak_is_safe():
    recommendation = analyze_smart_dsp_strategy(
        _analysis(
            target_id="main",
            source_role="master_main",
            problem_hints=("master_overload", "muddy"),
            muddiness_score=0.91,
            master_true_peak_db=-4.2,
            true_peak_headroom_db=3.2,
            master_integrated_lufs=-19.5,
            low_end_excess_score=0.88,
            spectral_tilt_score=0.72,
        )
    )

    assert recommendation.status == "proposal_only"
    assert recommendation.evidence["detected_problems"] == ["master_overload"]
    assert recommendation.selected_strategy.strategy_id == "master_low_end_broad_eq_cleanup"
    assert recommendation.selected_strategy.method_family == "eq"
    params = recommendation.selected_strategy.parameters
    assert params["filter_type"] == "low_shelf"
    assert params["gain_db"] < 0.0
    assert params["target_scope"] == "master_and_pa_feed"
    assert params["pa_reference_feed_check_required"] is True
    assert params["true_peak_headroom_required_before_tone"] is True
    assert params["wing_eq_mapping_required"] is True
    assert params["wing_eq_low_shelf_gain_db"] == params["gain_db"]
    _assert_soundcheck_master_processing_guard(
        recommendation.selected_strategy,
        processing_types=("broad_eq",),
    )
    _assert_proposal_denial(recommendation.to_operator_proposal())
    assert recommendation.evidence["low_end_excess_score"] >= 0.8


def test_master_presence_overload_prefers_broad_eq_when_true_peak_is_safe():
    recommendation = analyze_smart_dsp_strategy(
        _analysis(
            target_id="main",
            source_role="master_main",
            problem_hints=("master_overload", "harsh"),
            muddiness_score=0.0,
            harshness_score=0.84,
            presence_harshness_score=0.78,
            master_true_peak_db=-4.5,
            true_peak_headroom_db=3.5,
            master_integrated_lufs=-19.3,
            low_end_excess_score=0.24,
            spectral_tilt_score=0.32,
        )
    )

    assert recommendation.status == "proposal_only"
    assert recommendation.selected_strategy.strategy_id == "master_presence_broad_eq_cleanup"
    assert recommendation.selected_strategy.method_family == "eq"
    params = recommendation.selected_strategy.parameters
    assert params["filter_type"] == "peaking"
    assert params["frequency_hz"] == 3400.0
    assert params["gain_db"] < 0.0
    assert params["q"] == 1.1
    assert params["target_scope"] == "master_and_pa_feed"
    assert params["pa_reference_feed_check_required"] is True
    assert params["true_peak_headroom_required_before_tone"] is True
    assert params["presence_harshness_score"] == 0.78
    assert params["wing_eq_mapping_required"] is True
    assert params["wing_eq_band_operation_count"] == 1
    _assert_soundcheck_master_processing_guard(
        recommendation.selected_strategy,
        processing_types=("broad_eq",),
    )
    _assert_proposal_denial(recommendation.to_operator_proposal())


def test_master_presence_overload_uses_measured_master_tone_band():
    recommendation = analyze_smart_dsp_strategy(
        _analysis(
            target_id="main",
            source_role="master_main",
            problem_hints=("master_overload", "harsh"),
            muddiness_score=0.0,
            harshness_score=0.82,
            presence_harshness_score=0.76,
            master_presence_frequency_hz=3920.0,
            master_presence_bandwidth_hz=820.0,
            master_true_peak_db=-4.5,
            true_peak_headroom_db=3.5,
            master_integrated_lufs=-19.3,
            low_end_excess_score=0.24,
            spectral_tilt_score=0.32,
        )
    )

    assert recommendation.status == "proposal_only"
    assert recommendation.selected_strategy.strategy_id == "master_presence_broad_eq_cleanup"
    params = recommendation.selected_strategy.parameters
    assert params["frequency_hz"] == 3920.0
    assert params["frequency_source"] == "measured_master_presence_peak"
    assert params["q"] == 1.45
    assert params["master_presence_frequency_hz"] == 3920.0
    assert params["master_presence_bandwidth_hz"] == 820.0
    assert params["master_presence_effective_bandwidth_hz"] == 2400.0
    assert recommendation.evidence["master_presence_frequency_hz"] == 3920.0


def test_live_master_presence_holds_before_master_eq_change():
    recommendation = analyze_smart_dsp_strategy(
        _analysis(
            target_id="main",
            source_role="master_main",
            mode="live",
            problem_hints=("master_overload", "harsh"),
            muddiness_score=0.0,
            harshness_score=0.84,
            presence_harshness_score=0.78,
            master_true_peak_db=-4.5,
            true_peak_headroom_db=3.5,
            master_integrated_lufs=-19.3,
            low_end_excess_score=0.24,
            spectral_tilt_score=0.32,
            active_performance=True,
        )
    )

    assert recommendation.status == "proposal_only"
    assert recommendation.selected_strategy.strategy_id == (
        "hold_live_master_output_processing_for_operator"
    )
    assert recommendation.selected_strategy.method_family == "analysis_only"
    assert recommendation.evidence["operator_explanation_sections"][
        "selection_policy"
    ] == "live_master_output_review_priority"
    params = recommendation.selected_strategy.parameters
    assert params["dsp_change"] == "hold"
    assert params["pa_reference_feed_check_required"] is True
    assert "master_eq_master_bus_compression" in params["forbidden_without_wing_proof"]
    assert params["live_master_output_hold_proposal"] is True
    assert params["live_master_output_target_surface"] == "main"
    assert params["live_master_output_previous_gain_readback_required"] is True
    assert params["live_master_output_previous_eq_readback_required"] is True
    assert params["live_master_output_previous_dynamics_readback_required"] is True
    assert params["live_master_output_limiter_ui_review_required"] is True
    assert params["live_master_output_presence_review_required"] is True
    assert params["live_master_output_dynamics_review_required"] is False
    assert params["live_master_output_no_processing_without_operator_wing_proof"] is True
    assert "wing_gain_delta_db" not in params
    assert "wing_eq_band_operations" not in params
    assert "wing_compressor_threshold_osc_value" not in params
    readback = recommendation.selected_strategy.future_wing_readback_contract[
        "live_master_output_hold_readback"
    ]
    assert readback["read_only"] is True
    assert readback["allowed_live_write_scope_before_wing_proof"] == []
    assert readback["live_master_output_target_surface"] == "main"
    assert readback["wing_master_limiter_osc_path_templates"] == []
    assert "/main/{main}/fdr" in readback["wing_master_gain_osc_path_templates"]
    assert "/main/{main}/eq/{band}f" in readback[
        "wing_master_eq_osc_path_templates"
    ]
    assert "/main/{main}/dyn/thr36" in readback[
        "wing_master_dynamics_osc_path_templates"
    ]
    assert "eq/{band}g" in readback["read_before_paths"]
    assert "dyn/thr36" in readback["read_before_paths"]
    eq_candidate = [
        candidate
        for candidate in recommendation.alternatives
        if candidate.strategy_id == "master_presence_broad_eq_cleanup"
    ][0]
    assert eq_candidate.method_family == "analysis_only"
    tradeoff = [
        item
        for item in recommendation.evidence["strategy_search"][
            "alternative_tradeoffs"
        ]
        if item["strategy_id"] == "master_presence_broad_eq_cleanup"
    ][0]
    assert tradeoff["why_not_selected"] == (
        "live_master_output_review_priority:"
        "master_tone_or_dynamics_requires_operator_review"
    )
    _assert_proposal_denial(recommendation.to_operator_proposal())


def test_master_low_end_overload_uses_measured_master_low_band():
    recommendation = analyze_smart_dsp_strategy(
        _analysis(
            target_id="main",
            source_role="master_main",
            problem_hints=("master_overload", "muddy"),
            muddiness_score=0.86,
            master_low_end_frequency_hz=74.0,
            master_low_end_bandwidth_hz=34.0,
            master_true_peak_db=-4.2,
            true_peak_headroom_db=3.2,
            master_integrated_lufs=-19.5,
            low_end_excess_score=0.88,
            spectral_tilt_score=0.72,
        )
    )

    assert recommendation.status == "proposal_only"
    assert recommendation.selected_strategy.strategy_id == "master_low_end_broad_eq_cleanup"
    params = recommendation.selected_strategy.parameters
    assert params["frequency_hz"] == 88.8
    assert params["frequency_source"] == "measured_master_low_end_peak"
    assert params["q"] == 0.75
    assert params["master_low_end_frequency_hz"] == 74.0
    assert params["master_low_end_bandwidth_hz"] == 34.0
    assert recommendation.evidence["master_low_end_frequency_hz"] == 74.0


def test_master_static_loudness_trims_before_low_end_tone_recheck():
    recommendation = analyze_smart_dsp_strategy(
        _analysis(
            target_id="main",
            source_role="master_main",
            source_label="Main LR",
            problem_hints=("master_overload", "low_end_resonance"),
            operator_intent="soundcheck main is loud and low end is heavy",
            muddiness_score=0.0,
            rms_dbfs=-17.2,
            lra_proxy_db=5.4,
            dynamics_variance_score=0.18,
            dynamics_peak_to_median_db=3.4,
            dynamics_rise_time_ms=0.0,
            dynamics_recovery_ms=0.0,
            master_true_peak_db=-4.5,
            true_peak_headroom_db=4.5,
            master_integrated_lufs=-13.8,
            low_end_excess_score=0.86,
            master_low_end_frequency_hz=68.0,
            master_low_end_bandwidth_hz=55.0,
            spectral_tilt_score=0.2,
            presence_harshness_score=0.08,
        )
    )

    assert recommendation.status == "proposal_only"
    assert recommendation.selected_strategy.strategy_id == (
        "master_level_trim_for_static_loudness"
    )
    assert recommendation.selected_strategy.method_family == "gain_fader"
    params = recommendation.selected_strategy.parameters
    assert params["trim_delta_db"] == -3.0
    assert params["master_integrated_lufs"] == -13.8
    assert params["static_loudness_tone_recheck_primary"] is True
    assert params["post_trim_master_tone_recheck_required"] is True
    assert (
        params["candidate_tone_cleanup_after_level_recheck"]
        == "master_low_end_broad_eq_cleanup"
    )
    _assert_gain_delta_readback_contract(
        recommendation.selected_strategy,
        source="trim_delta_db",
        target_surface="main",
        expected_delta=-3.0,
        path_prefix="/main/{main}",
    )
    _assert_soundcheck_master_output_guard(
        recommendation.selected_strategy,
        move_kind="pullback",
        followup_review=True,
    )
    _assert_master_output_processing_followup_review_contract(
        recommendation.selected_strategy,
        expected_values={
            "candidate_tone_cleanup_after_level_recheck": (
                "master_low_end_broad_eq_cleanup"
            ),
            "post_trim_master_tone_recheck_required": True,
        },
    )
    tone_eq = [
        candidate
        for candidate in recommendation.alternatives
        if candidate.strategy_id == "master_low_end_broad_eq_cleanup"
    ][0]
    assert tone_eq.parameters["frequency_source"] == "measured_master_low_end_peak"
    assert tone_eq.total_score < recommendation.selected_strategy.total_score
    tradeoff = [
        item
        for item in recommendation.evidence["strategy_search"][
            "alternative_tradeoffs"
        ]
        if item["strategy_id"] == "master_low_end_broad_eq_cleanup"
    ][0]
    assert tradeoff["why_not_selected"] == (
        "master_static_loudness_trim_priority:"
        "trim_static_level_before_master_tone_recheck"
    )
    _assert_proposal_denial(recommendation.to_operator_proposal())


def test_master_static_loudness_prefers_trim_before_bus_compression():
    recommendation = analyze_smart_dsp_strategy(
        _analysis(
            target_id="main",
            source_role="master_main",
            source_label="Main LR",
            problem_hints=("master_overload",),
            operator_intent="soundcheck main static loud",
            muddiness_score=0.0,
            rms_dbfs=-18.5,
            lra_proxy_db=5.0,
            dynamics_variance_score=0.18,
            dynamics_peak_to_median_db=3.2,
            dynamics_rise_time_ms=0.0,
            dynamics_recovery_ms=0.0,
            master_true_peak_db=-3.2,
            true_peak_headroom_db=3.2,
            master_integrated_lufs=-15.8,
            low_end_excess_score=0.18,
            spectral_tilt_score=0.24,
            presence_harshness_score=0.08,
        )
    )

    assert recommendation.status == "proposal_only"
    assert recommendation.selected_strategy.strategy_id == (
        "master_level_trim_for_static_loudness"
    )
    assert recommendation.selected_strategy.method_family == "gain_fader"
    params = recommendation.selected_strategy.parameters
    assert params["trim_delta_db"] == -2.2
    assert params["target_master_integrated_lufs"] == -18.0
    assert params["master_integrated_lufs"] == -15.8
    assert params["master_loudness_excess_lu"] == 2.2
    assert params["master_level_parameter_source"] == (
        "master_integrated_lufs_static_loudness"
    )
    assert params["true_peak_headroom_required_before_master_level"] is True
    assert params["static_loudness_primary"] is True
    assert params["makeup_gain_db"] == 0.0
    _assert_gain_delta_readback_contract(
        recommendation.selected_strategy,
        source="trim_delta_db",
        target_surface="main",
        expected_delta=-2.2,
        path_prefix="/main/{main}",
    )
    _assert_soundcheck_master_output_guard(
        recommendation.selected_strategy,
        move_kind="pullback",
        followup_review=True,
    )
    _assert_master_output_processing_followup_review_contract(
        recommendation.selected_strategy,
        expected_values={"post_trim_master_tone_recheck_required": False},
    )
    assert recommendation.evidence["operator_explanation_sections"][
        "selection_policy"
    ] == "master_static_loudness_trim_priority"
    bus_comp = [
        candidate
        for candidate in recommendation.alternatives
        if candidate.strategy_id == "gentle_bus_compression"
    ][0]
    assert bus_comp.parameters["master_dynamics_parameter_source"] == (
        "dynamic_pressure_fallback"
    )
    assert bus_comp.parameters["knee_db"] == 5.0
    assert bus_comp.parameters["compression_mix_percent"] == 82.2
    assert bus_comp.parameters["compressor_mix_parameter_source"] == (
        "dynamic_pressure_parallel_bus_blend_fallback"
    )
    assert bus_comp.parameters["compressor_knee_parameter_source"] == (
        "soft_master_bus_knee_fallback"
    )
    assert bus_comp.total_score < recommendation.selected_strategy.total_score
    tradeoff = [
        item
        for item in recommendation.evidence["strategy_search"][
            "alternative_tradeoffs"
        ]
        if item["strategy_id"] == "gentle_bus_compression"
    ][0]
    assert tradeoff["why_not_selected"] == (
        "master_static_loudness_trim_priority:"
        "trim_static_level_before_bus_compression"
    )
    _assert_proposal_denial(recommendation.to_operator_proposal())


def test_master_loud_dynamic_loudness_trims_before_bus_compression():
    recommendation = analyze_smart_dsp_strategy(
        _analysis(
            target_id="main",
            source_role="master_main",
            source_label="Main LR",
            problem_hints=("master_overload",),
            operator_intent="soundcheck main is too loud and still has phrase jumps",
            muddiness_score=0.0,
            rms_dbfs=-15.5,
            lra_proxy_db=14.0,
            dynamics_variance_score=0.7,
            dynamics_peak_to_median_db=13.0,
            dynamics_rise_time_ms=28.0,
            dynamics_recovery_ms=220.0,
            master_true_peak_db=-4.5,
            true_peak_headroom_db=4.5,
            master_integrated_lufs=-14.8,
            low_end_excess_score=0.1,
            spectral_tilt_score=0.1,
            harshness_score=0.1,
            presence_harshness_score=0.1,
        )
    )

    assert recommendation.status == "proposal_only"
    assert recommendation.selected_strategy.strategy_id == (
        "master_loud_dynamic_trim_before_compression"
    )
    assert recommendation.selected_strategy.method_family == "gain_fader"
    params = recommendation.selected_strategy.parameters
    assert params["trim_delta_db"] == -3.0
    assert params["target_master_integrated_lufs"] == -18.0
    assert params["master_integrated_lufs"] == -14.8
    assert params["master_loudness_excess_lu"] == 3.2
    assert params["master_level_parameter_source"] == (
        "master_integrated_lufs_loud_dynamic"
    )
    assert params["dynamic_envelope_pressure_score"] == 0.7
    assert params["bus_compression_after_level_recheck"] is True
    assert params["post_trim_bus_compression_review_required"] is True
    assert params["post_trim_loudness_recheck_required"] is True
    assert params["candidate_bus_compression_threshold_dbfs"] == -9.6
    assert params["candidate_bus_compression_ratio"] == 1.89
    assert params["candidate_bus_compression_knee_db"] == 4.6
    assert params["candidate_bus_compression_mix_percent"] == 86.2
    assert params["candidate_bus_compression_target_gr_db"] == 2.1
    assert params["candidate_bus_compression_detector"] == "rms"
    assert params["candidate_bus_compression_envelope_curve"] == "smooth_log"
    assert params["candidate_bus_compression_auto_timing_enabled"] is False
    assert params["dynamics_peak_to_median_db"] == 13.0
    assert params["dynamics_rise_time_ms"] == 28.0
    assert params["dynamics_recovery_ms"] == 220.0
    assert params["true_peak_headroom_required_before_master_level"] is True
    assert params["loud_dynamic_loudness_primary"] is True
    assert params["makeup_gain_db"] == 0.0
    _assert_gain_delta_readback_contract(
        recommendation.selected_strategy,
        source="trim_delta_db",
        target_surface="main",
        expected_delta=-3.0,
        path_prefix="/main/{main}",
    )
    _assert_soundcheck_master_output_guard(
        recommendation.selected_strategy,
        move_kind="pullback",
        followup_review=True,
    )
    _assert_master_output_processing_followup_review_contract(
        recommendation.selected_strategy,
        expected_values={
            "candidate_bus_compression_threshold_dbfs": -9.6,
            "candidate_bus_compression_ratio": 1.89,
            "candidate_bus_compression_knee_db": 4.6,
            "candidate_bus_compression_mix_percent": 86.2,
            "candidate_bus_compression_target_gr_db": 2.1,
            "candidate_bus_compression_detector": "rms",
            "candidate_bus_compression_envelope_curve": "smooth_log",
            "candidate_bus_compression_auto_timing_enabled": False,
            "post_trim_bus_compression_review_required": True,
            "post_trim_loudness_recheck_required": True,
        },
    )
    assert recommendation.evidence["operator_explanation_sections"][
        "selection_policy"
    ] == "master_loud_dynamic_trim_priority"
    bus_comp = [
        candidate
        for candidate in recommendation.alternatives
        if candidate.strategy_id == "gentle_bus_compression"
    ][0]
    assert bus_comp.parameters["master_dynamics_parameter_source"] == (
        "measured_master_envelope"
    )
    assert bus_comp.parameters["threshold_dbfs"] == -9.6
    assert bus_comp.parameters["knee_db"] == 4.6
    assert bus_comp.parameters["compression_mix_percent"] == 86.2
    assert bus_comp.parameters["compressor_mix_parameter_source"] == (
        "measured_master_envelope_parallel_bus_blend"
    )
    assert bus_comp.parameters["compressor_knee_parameter_source"] == (
        "measured_master_envelope_soft_knee_policy"
    )
    assert bus_comp.parameters["detector"] == "rms"
    assert bus_comp.parameters["compressor_envelope_curve"] == "smooth_log"
    assert bus_comp.parameters["compressor_auto_timing_enabled"] is False
    assert bus_comp.total_score < recommendation.selected_strategy.total_score
    tradeoff = [
        item
        for item in recommendation.evidence["strategy_search"][
            "alternative_tradeoffs"
        ]
        if item["strategy_id"] == "gentle_bus_compression"
    ][0]
    assert tradeoff["why_not_selected"] == (
        "master_loud_dynamic_trim_priority:"
        "restore_master_level_before_bus_compression_recheck"
    )
    _assert_proposal_denial(recommendation.to_operator_proposal())


def test_soundcheck_quiet_master_output_prefers_bounded_lift_before_hold():
    recommendation = analyze_smart_dsp_strategy(
        _analysis(
            target_id="main",
            source_role="master_main",
            source_label="Main LR",
            problem_hints=("master_overload",),
            operator_intent="soundcheck main too quiet but peaks safe",
            muddiness_score=0.0,
            rms_dbfs=-27.0,
            lra_proxy_db=5.0,
            dynamics_variance_score=0.16,
            dynamics_peak_to_median_db=3.0,
            dynamics_rise_time_ms=0.0,
            dynamics_recovery_ms=0.0,
            master_true_peak_db=-9.0,
            true_peak_headroom_db=8.0,
            master_integrated_lufs=-25.0,
            low_end_excess_score=0.12,
            spectral_tilt_score=0.18,
            presence_harshness_score=0.05,
        )
    )

    assert recommendation.status == "proposal_only"
    assert recommendation.selected_strategy.strategy_id == (
        "master_output_lift_for_quiet_soundcheck"
    )
    assert recommendation.selected_strategy.method_family == "gain_fader"
    params = recommendation.selected_strategy.parameters
    assert params["trim_delta_db"] == 3.0
    assert params["bounded_delta_db"] == 3.0
    assert params["gain_db"] == 3.0
    assert params["target_master_integrated_lufs"] == -20.0
    assert params["master_integrated_lufs"] == -25.0
    assert params["master_loudness_deficit_lu"] == 5.0
    assert params["available_true_peak_lift_db"] == 7.0
    assert params["master_level_parameter_source"] == (
        "master_integrated_lufs_quiet_soundcheck"
    )
    assert params["true_peak_headroom_required_before_master_level"] is True
    assert params["soundcheck_master_output_required_true_peak_headroom_db"] == 4.0
    assert (
        params["soundcheck_master_output_required_summed_bus_true_peak_headroom_db"]
        == 4.0
    )
    assert params["soundcheck_only"] is True
    _assert_gain_delta_readback_contract(
        recommendation.selected_strategy,
        source="trim_delta_db",
        target_surface="main",
        expected_delta=3.0,
        path_prefix="/main/{main}",
    )
    _assert_soundcheck_master_output_guard(
        recommendation.selected_strategy,
        move_kind="lift",
        followup_review=False,
    )
    assert recommendation.evidence["operator_explanation_sections"][
        "selection_policy"
    ] == "master_quiet_loudness_lift_priority"
    hold_tradeoff = [
        item
        for item in recommendation.evidence["strategy_search"][
            "alternative_tradeoffs"
        ]
        if item["strategy_id"] == "hold_master_bus_for_more_evidence"
    ][0]
    assert hold_tradeoff["why_not_selected"] == (
        "master_quiet_loudness_lift_priority:"
        "bounded_lift_before_hold_compression_limiter_or_eq"
    )
    _assert_proposal_denial(recommendation.to_operator_proposal())


def test_soundcheck_quiet_master_output_holds_for_feedback_review_before_lift():
    recommendation = analyze_smart_dsp_strategy(
        _analysis(
            target_id="main",
            source_role="master_main",
            source_label="Main LR",
            problem_hints=("master_overload",),
            operator_intent="soundcheck main too quiet but feedback margin is unknown",
            muddiness_score=0.0,
            rms_dbfs=-27.0,
            lra_proxy_db=5.0,
            dynamics_variance_score=0.16,
            dynamics_peak_to_median_db=3.0,
            dynamics_rise_time_ms=0.0,
            dynamics_recovery_ms=0.0,
            master_true_peak_db=-9.0,
            true_peak_headroom_db=8.0,
            master_integrated_lufs=-25.0,
            low_end_excess_score=0.12,
            spectral_tilt_score=0.18,
            presence_harshness_score=0.05,
            feedback_risk=0.72,
            monitor_feedback_risk=0.68,
        )
    )

    assert recommendation.status == "proposal_only"
    assert recommendation.selected_strategy.strategy_id == (
        "hold_master_feedback_or_monitor_risk_before_loudness_lift"
    )
    assert recommendation.selected_strategy.method_family == "analysis_only"
    params = recommendation.selected_strategy.parameters
    assert params["dsp_change"] == "hold"
    assert params["candidate_lift_db"] == 3.0
    assert params["master_loudness_deficit_lu"] == 5.0
    assert params["feedback_risk"] == 0.72
    assert params["monitor_feedback_risk"] == 0.68
    assert params["required_context"] == (
        "pa_and_monitor_feedback_path_review_before_master_loudness_lift"
    )
    assert params["operator_confirmation_required"] is True
    assert recommendation.evidence["operator_explanation_sections"][
        "selection_policy"
    ] == "master_feedback_monitor_loudness_hold_priority"
    blocked_lift = [
        candidate
        for candidate in recommendation.alternatives
        if candidate.strategy_id == "master_output_lift_for_quiet_soundcheck"
    ][0]
    assert "feedback_path_review_blocks_master_loudness_lift" in blocked_lift.blockers
    assert (
        "monitor_feedback_risk_blocks_master_loudness_lift"
        in blocked_lift.blockers
    )
    assert "feedback_risk_blocks_boost" in blocked_lift.blockers
    lift_tradeoff = [
        item
        for item in recommendation.evidence["strategy_search"][
            "alternative_tradeoffs"
        ]
        if item["strategy_id"] == "master_output_lift_for_quiet_soundcheck"
    ][0]
    assert lift_tradeoff["why_not_selected"] == (
        "master_feedback_monitor_loudness_hold_priority:"
        "feedback_path_review_precedes_master_loudness_lift"
    )
    _assert_proposal_denial(recommendation.to_operator_proposal())


def test_soundcheck_quiet_master_output_holds_when_headroom_cannot_support_lift():
    recommendation = analyze_smart_dsp_strategy(
        _analysis(
            target_id="main",
            source_role="master_main",
            source_label="Main LR",
            problem_hints=("master_overload",),
            operator_intent="soundcheck main too quiet with limited headroom",
            muddiness_score=0.0,
            rms_dbfs=-27.0,
            lra_proxy_db=5.0,
            dynamics_variance_score=0.16,
            dynamics_peak_to_median_db=3.0,
            dynamics_rise_time_ms=0.0,
            dynamics_recovery_ms=0.0,
            master_true_peak_db=-9.0,
            true_peak_headroom_db=1.2,
            master_integrated_lufs=-25.0,
            low_end_excess_score=0.12,
            spectral_tilt_score=0.18,
            presence_harshness_score=0.05,
        )
    )

    assert recommendation.status == "proposal_only"
    assert recommendation.selected_strategy.strategy_id == (
        "hold_master_bus_for_more_evidence"
    )
    blocked_lift = [
        candidate
        for candidate in recommendation.alternatives
        if candidate.strategy_id == "master_output_lift_for_quiet_soundcheck"
    ][0]
    assert "insufficient_true_peak_headroom_for_boost" in blocked_lift.blockers
    assert blocked_lift.parameters["gain_db"] == 0.5
    _assert_proposal_denial(recommendation.to_operator_proposal())


def test_master_loudness_pressure_selects_soundcheck_bus_compression_proposal():
    recommendation = analyze_smart_dsp_strategy(
        _analysis(
            target_id="main",
            source_role="master_main",
            problem_hints=("master_overload",),
            muddiness_score=0.0,
            rms_dbfs=-21.0,
            lra_proxy_db=12.0,
            dynamics_variance_score=0.62,
            dynamics_peak_to_median_db=15.0,
            dynamics_rise_time_ms=80.0,
            dynamics_recovery_ms=280.0,
            master_true_peak_db=-3.0,
            true_peak_headroom_db=2.0,
            master_integrated_lufs=-16.5,
        )
    )

    assert recommendation.status == "proposal_only"
    assert recommendation.selected_strategy.strategy_id == "gentle_bus_compression"
    assert recommendation.selected_strategy.method_family == "dynamics"
    params = recommendation.selected_strategy.parameters
    assert params["makeup_gain_db"] == 0.0
    assert params["true_peak_headroom_required_before_compression"] is True
    assert params["dynamic_pressure_score"] >= 0.55
    assert params["target_gain_reduction_db"] <= 2.4
    assert params["master_dynamics_parameter_source"] == "measured_master_envelope"
    assert params["dynamics_peak_to_median_db"] == 15.0
    assert params["dynamics_rise_time_ms"] == 80.0
    assert params["dynamics_recovery_ms"] == 280.0
    assert params["threshold_dbfs"] == -14.5
    assert params["ratio"] == 1.9
    assert params["attack_ms"] == 36.0
    assert params["release_ms"] == 336.0
    assert params["knee_db"] == 4.0
    assert params["target_gain_reduction_db"] == 2.1
    assert params["compression_mix_percent"] == 86.0
    assert params["compressor_mix_parameter_source"] == (
        "measured_master_envelope_parallel_bus_blend"
    )
    assert params["compressor_knee_parameter_source"] == (
        "measured_master_envelope_soft_knee_policy"
    )
    assert params["detector"] == "rms"
    assert params["compressor_envelope_curve"] == "smooth_log"
    assert params["compressor_auto_timing_enabled"] is False
    assert params["wing_compressor_threshold_osc_value"] == -14.5
    assert params["wing_compressor_ratio_osc_value"] == 1.9
    assert params["wing_compressor_knee_osc_value"] == 4
    assert params["wing_compressor_mix_osc_value"] == 86.0
    assert params["wing_compressor_detector_osc_value"] == "RMS"
    assert params["wing_compressor_envelope_curve_osc_value"] == "LOG"
    assert params["wing_compressor_auto_osc_value"] == 0
    _assert_soundcheck_master_processing_guard(
        recommendation.selected_strategy,
        processing_types=("bus_compression",),
    )
    assert recommendation.to_operator_proposal()["live_apply_allowed"] is False


def test_live_master_loudness_pressure_keeps_bus_compression_review_blocked():
    recommendation = analyze_smart_dsp_strategy(
        _analysis(
            target_id="main",
            source_role="master_main",
            mode="live",
            problem_hints=("master_overload",),
            muddiness_score=0.0,
            rms_dbfs=-21.0,
            lra_proxy_db=12.0,
            dynamics_variance_score=0.62,
            dynamics_peak_to_median_db=15.0,
            dynamics_rise_time_ms=80.0,
            dynamics_recovery_ms=280.0,
            master_true_peak_db=-3.0,
            true_peak_headroom_db=2.0,
            master_integrated_lufs=-16.5,
            active_performance=True,
        )
    )

    assert recommendation.status == "proposal_only"
    assert recommendation.selected_strategy.strategy_id == (
        "hold_live_master_output_processing_for_operator"
    )
    assert recommendation.evidence["operator_explanation_sections"][
        "selection_policy"
    ] == "live_master_output_review_priority"
    params = recommendation.selected_strategy.parameters
    assert params["live_master_output_hold_proposal"] is True
    assert params["live_master_output_presence_review_required"] is False
    assert params["live_master_output_dynamics_review_required"] is True
    assert params["live_master_output_previous_dynamics_readback_required"] is True
    readback = recommendation.selected_strategy.future_wing_readback_contract[
        "live_master_output_hold_readback"
    ]
    assert readback["read_only"] is True
    assert readback["allowed_live_write_scope_before_wing_proof"] == []
    assert readback["wing_master_dynamics_threshold_osc_path_suffix"] == "dyn/thr36"
    assert "/main/{main}/dyn/thr36" in readback[
        "wing_master_dynamics_osc_path_templates"
    ]
    assert "wing_compressor_threshold_osc_value" not in params
    bus_comp = [
        candidate
        for candidate in recommendation.alternatives
        if candidate.strategy_id == "gentle_bus_compression"
    ][0]
    assert (
        "live_master_bus_compression_requires_operator_review_and_wing_proof"
        in bus_comp.blockers
    )
    assert (
        bus_comp.parameters["master_dynamics_parameter_source"]
        == "measured_master_envelope"
    )
    assert bus_comp.parameters["attack_ms"] == 36.0
    assert bus_comp.parameters["release_ms"] == 336.0
    assert bus_comp.parameters["knee_db"] == 4.0
    assert bus_comp.parameters["compression_mix_percent"] == 86.0
    assert bus_comp.parameters["detector"] == "rms"
    assert bus_comp.parameters["compressor_envelope_curve"] == "smooth_log"
    assert bus_comp.parameters["compressor_auto_timing_enabled"] is False
    assert recommendation.to_operator_proposal()["can_apply_live"] is False


def test_soundcheck_quiet_source_selects_bounded_level_plane_gain_up():
    recommendation = analyze_smart_dsp_strategy(
        _analysis(
            target_id="ch:10",
            source_role="lead_vocal",
            problem_hints=("level_plane_balance",),
            muddiness_score=0.0,
            estimated_lufs=-30.0,
            level_plane_error_lu=-8.0,
            true_peak_headroom_db=8.0,
            feedback_risk=0.12,
            bleed_risk=0.1,
        )
    )

    assert recommendation.status == "proposal_only"
    assert recommendation.selected_strategy.strategy_id == "soundcheck_bounded_gain_up"
    assert recommendation.selected_strategy.method_family == "gain_fader"
    params = recommendation.selected_strategy.parameters
    assert 0.0 < params["delta_db"] <= 3.0
    assert params["gain_db"] == params["delta_db"]
    assert params["future_fader_ceiling_dbfs"] == 0.0
    assert params["target_scope"] == "source_and_summed_bus"
    _assert_gain_delta_readback_contract(
        recommendation.selected_strategy,
        source="delta_db",
        target_surface="channel",
        expected_delta=params["delta_db"],
        path_prefix="/ch/{channel}",
    )
    assert "level_plane_balance" in recommendation.evidence["detected_problems"]
    assert recommendation.to_operator_proposal()["live_apply_allowed"] is False


def test_soundcheck_gain_up_uses_current_wing_fader_headroom_when_safe():
    recommendation = analyze_smart_dsp_strategy(
        _analysis(
            target_id="ch:10",
            source_role="lead_vocal",
            problem_hints=("level_plane_balance",),
            muddiness_score=0.0,
            estimated_lufs=-30.0,
            level_plane_error_lu=-8.0,
            true_peak_headroom_db=8.0,
            feedback_risk=0.12,
            bleed_risk=0.1,
            current_wing_fader_db=-12.0,
            current_wing_control_readback_available=True,
        )
    )

    assert recommendation.selected_strategy.strategy_id == "soundcheck_bounded_gain_up"
    params = recommendation.selected_strategy.parameters
    assert params["wing_current_fader_db"] == -12.0
    assert params["wing_gain_requested_delta_db"] == params["wing_gain_delta_db"]
    assert params["wing_gain_projected_fader_db"] == -9.0
    assert params["wing_gain_fader_headroom_db"] == 12.0
    assert params["wing_gain_fader_ceiling_limited_delta_db"] == params["delta_db"]
    assert params["wing_gain_fader_ceiling_exceeded"] is False
    assert params["wing_gain_fader_ceiling_blocker"] is None
    assert WING_GAIN_UP_FADER_CEILING_BLOCKER not in recommendation.selected_strategy.blockers


def test_soundcheck_gain_up_blocks_when_current_wing_fader_would_cross_zero():
    recommendation = analyze_smart_dsp_strategy(
        _analysis(
            target_id="ch:10",
            source_role="lead_vocal",
            problem_hints=("level_plane_balance",),
            muddiness_score=0.0,
            estimated_lufs=-30.0,
            level_plane_error_lu=-8.0,
            true_peak_headroom_db=8.0,
            feedback_risk=0.12,
            bleed_risk=0.1,
            current_wing_fader_db=-1.0,
            current_wing_control_readback_available=True,
        )
    )

    candidates = {
        candidate.strategy_id: candidate
        for candidate in (recommendation.selected_strategy, *recommendation.alternatives)
    }
    gain_up = candidates["soundcheck_bounded_gain_up"]
    assert WING_GAIN_UP_FADER_CEILING_BLOCKER in gain_up.blockers
    params = gain_up.parameters
    assert params["wing_current_fader_db"] == -1.0
    assert params["wing_gain_requested_delta_db"] == 3.0
    assert params["wing_gain_projected_fader_db"] == 2.0
    assert params["wing_gain_fader_headroom_db"] == 1.0
    assert params["wing_gain_fader_ceiling_limited_delta_db"] == 1.0
    assert params["wing_gain_fader_ceiling_exceeded"] is True
    assert params["wing_gain_fader_ceiling_blocker"] == WING_GAIN_UP_FADER_CEILING_BLOCKER
    assert recommendation.to_operator_proposal()["live_apply_allowed"] is False


def test_soundcheck_gain_up_requires_current_wing_fader_readback_when_controls_exist():
    recommendation = analyze_smart_dsp_strategy(
        _analysis(
            target_id="ch:10",
            source_role="lead_vocal",
            problem_hints=("level_plane_balance",),
            muddiness_score=0.0,
            estimated_lufs=-30.0,
            level_plane_error_lu=-8.0,
            true_peak_headroom_db=8.0,
            feedback_risk=0.12,
            bleed_risk=0.1,
            current_wing_input_trim_db=1.5,
            current_wing_control_readback_available=True,
        )
    )

    candidates = {
        candidate.strategy_id: candidate
        for candidate in (recommendation.selected_strategy, *recommendation.alternatives)
    }
    gain_up = candidates["soundcheck_bounded_gain_up"]
    assert WING_GAIN_UP_FADER_READBACK_BLOCKER in gain_up.blockers
    params = gain_up.parameters
    assert params["wing_current_fader_readback_available"] is False
    assert params["wing_current_fader_readback_required"] is True
    assert params["wing_gain_requested_delta_db"] == 3.0
    assert params["wing_gain_fader_ceiling_dbfs"] == 0.0
    assert params["wing_gain_fader_ceiling_blocker"] == WING_GAIN_UP_FADER_READBACK_BLOCKER
    assert params["wing_gain_fader_ceiling_policy"] == (
        "require_current_wing_fader_readback_before_gain_up_ceiling_check"
    )
    assert recommendation.to_operator_proposal()["live_apply_allowed"] is False


def test_operator_owned_gain_blocks_soundcheck_level_plane_gain_apply():
    recommendation = analyze_smart_dsp_strategy(
        _analysis(
            target_id="ch:10",
            source_role="lead_vocal",
            problem_hints=("level_plane_balance",),
            muddiness_score=0.0,
            estimated_lufs=-30.0,
            level_plane_error_lu=-8.0,
            true_peak_headroom_db=8.0,
            feedback_risk=0.12,
            bleed_risk=0.1,
            current_wing_gain_operator_owned=True,
        )
    )

    assert recommendation.status == "analysis_only"
    assert recommendation.selected_strategy.strategy_id == (
        "hold_soundcheck_operator_owned_gain_for_review"
    )
    assert recommendation.selected_strategy.method_family == "analysis_only"
    assert recommendation.selected_strategy.blockers == (
        "soundcheck_operator_owned_gain_requires_operator_review",
    )
    assert recommendation.evidence["operator_explanation_sections"][
        "selection_policy"
    ] == "soundcheck_operator_owned_gain_review_priority"
    params = recommendation.selected_strategy.parameters
    assert params["dsp_change"] == "hold"
    assert params["reason"] == "soundcheck_operator_owned_gain_requires_operator_review"
    assert params["calculated_delta_db"] == 3.0
    assert params["calculated_gain_direction"] == "up"
    assert params["blocked_gain_strategy_id"] == "soundcheck_bounded_gain_up"
    assert params["blocked_gain_delta_db"] == 3.0
    assert params["gain_db"] == 0.0
    assert params["wing_gain_current_operator_owned"] is True
    assert params["wing_gain_operator_owned_edit_requires_review"] is True
    assert params["wing_gain_operator_owned_blocker"] == WING_GAIN_OPERATOR_OWNED_BLOCKER
    assert params["soundcheck_operator_owned_gain_hold_proposal"] is True
    assert params["soundcheck_operator_owned_gain_target_surface"] == "channel"
    assert params["soundcheck_operator_owned_gain_move_kind"] == "up"
    assert params["soundcheck_operator_owned_gain_selected_write_keys"] == []
    assert params["soundcheck_operator_owned_gain_forbidden_write_keys"] == [
        "wing_gain_delta_db"
    ]
    assert "wing_gain_delta_db" not in params

    blocked_gain_up = [
        candidate
        for candidate in recommendation.alternatives
        if candidate.strategy_id == "soundcheck_bounded_gain_up"
    ][0]
    assert blocked_gain_up.parameters["wing_gain_delta_db"] == 3.0
    assert WING_GAIN_OPERATOR_OWNED_BLOCKER in blocked_gain_up.blockers
    tradeoffs = {
        row["strategy_id"]: row["why_not_selected"]
        for row in recommendation.evidence["strategy_search"][
            "alternative_tradeoffs"
        ]
    }
    assert tradeoffs["soundcheck_bounded_gain_up"] == (
        "soundcheck_operator_owned_gain_review_priority:"
        "manual_fader_or_trim_owner_review_precedes_soundcheck_gain_move"
    )

    source_gain_readback = recommendation.selected_strategy.future_wing_readback_contract[
        "operator_owned_source_gain_readback"
    ]
    assert source_gain_readback["read_only"] is True
    assert source_gain_readback["read_before_paths"] == ["fdr", "in/set/trim"]
    assert source_gain_readback["read_after_paths"] == ["fdr", "in/set/trim"]
    assert "/ch/{channel}/fdr" in source_gain_readback["wing_osc_path_templates"]
    assert "/ch/{channel}/in/set/trim" in source_gain_readback[
        "wing_osc_path_templates"
    ]
    assert source_gain_readback["allowed_live_write_scope_before_wing_proof"] == []
    assert source_gain_readback["wing_source_gain_forbidden_apply_fields"] == [
        "wing_gain_delta_db"
    ]
    proposal = recommendation.to_operator_proposal()
    assert proposal["selected_wing_osc_summary"][
        "wing_osc_candidate_apply_status"
    ] == "analysis_only_no_console_write"
    assert proposal["selected_wing_osc_summary"][
        "wing_osc_apply_parameter_values"
    ] == {}
    assert proposal["can_apply_live"] is False
    assert proposal["live_authority_granted"] is False
    assert proposal["live_write"] is False


def test_operator_owned_gain_blocks_soundcheck_level_plane_gain_down_apply():
    recommendation = analyze_smart_dsp_strategy(
        _analysis(
            target_id="ch:10",
            source_role="lead_vocal",
            problem_hints=("level_plane_balance",),
            muddiness_score=0.0,
            estimated_lufs=-18.0,
            level_plane_error_lu=4.0,
            true_peak_headroom_db=8.0,
            feedback_risk=0.12,
            bleed_risk=0.1,
            current_wing_gain_operator_owned=True,
        )
    )

    assert recommendation.status == "analysis_only"
    assert recommendation.selected_strategy.strategy_id == (
        "hold_soundcheck_operator_owned_gain_for_review"
    )
    assert recommendation.selected_strategy.method_family == "analysis_only"
    params = recommendation.selected_strategy.parameters
    assert params["calculated_delta_db"] == -3.0
    assert params["calculated_gain_direction"] == "down"
    assert params["blocked_gain_strategy_id"] == "level_plane_gain_down"
    assert params["blocked_gain_delta_db"] == -3.0
    assert params["gain_db"] == 0.0
    assert "wing_gain_delta_db" not in params
    blocked_gain_down = [
        candidate
        for candidate in recommendation.alternatives
        if candidate.strategy_id == "level_plane_gain_down"
    ][0]
    assert blocked_gain_down.parameters["wing_gain_delta_db"] == -3.0
    assert WING_GAIN_OPERATOR_OWNED_BLOCKER in blocked_gain_down.blockers
    tradeoffs = {
        row["strategy_id"]: row["why_not_selected"]
        for row in recommendation.evidence["strategy_search"][
            "alternative_tradeoffs"
        ]
    }
    assert tradeoffs["level_plane_gain_down"] == (
        "soundcheck_operator_owned_gain_review_priority:"
        "manual_fader_or_trim_owner_review_precedes_soundcheck_gain_move"
    )
    source_gain_readback = recommendation.selected_strategy.future_wing_readback_contract[
        "operator_owned_source_gain_readback"
    ]
    assert source_gain_readback["blocked_gain_strategy_id"] == "level_plane_gain_down"
    assert source_gain_readback["blocked_gain_delta_db"] == -3.0
    assert source_gain_readback["soundcheck_operator_owned_gain_move_kind"] == "down"
    assert source_gain_readback["wing_source_gain_forbidden_apply_fields"] == [
        "wing_gain_delta_db"
    ]
    proposal = recommendation.to_operator_proposal()
    assert proposal["selected_wing_osc_summary"][
        "wing_osc_candidate_apply_status"
    ] == "analysis_only_no_console_write"
    assert proposal["selected_wing_osc_summary"][
        "wing_osc_apply_parameter_values"
    ] == {}
    assert proposal["can_apply_live"] is False
    assert proposal["live_authority_granted"] is False
    assert proposal["live_write"] is False


def test_live_operator_owned_gain_selects_hold_not_gain_fader_move():
    recommendation = analyze_smart_dsp_strategy(
        _analysis(
            target_id="ch:8",
            source_role="bass",
            source_label="Bass DI",
            mode="live",
            problem_hints=("level_plane_balance",),
            muddiness_score=0.0,
            estimated_lufs=-18.0,
            level_plane_error_lu=4.8,
            true_peak_headroom_db=7.0,
            feedback_risk=0.08,
            monitor_feedback_risk=0.04,
            current_wing_gain_operator_owned=True,
            active_performance=True,
        )
    )

    assert recommendation.status == "proposal_only"
    assert recommendation.selected_strategy.strategy_id == (
        "hold_live_operator_owned_gain_for_review"
    )
    assert recommendation.selected_strategy.method_family == "analysis_only"
    assert recommendation.evidence["operator_explanation_sections"][
        "selection_policy"
    ] == "live_operator_owned_gain_review_priority"
    params = recommendation.selected_strategy.parameters
    assert params["dsp_change"] == "hold"
    assert params["reason"] == "live_operator_owned_gain_requires_operator_review"
    assert params["calculated_delta_db"] == -1.5
    assert params["calculated_gain_direction"] == "down"
    assert params["gain_db"] == 0.0
    assert params["wing_gain_current_operator_owned"] is True
    assert params["wing_gain_operator_owned_edit_requires_review"] is True
    assert params["wing_gain_operator_owned_blocker"] == WING_GAIN_OPERATOR_OWNED_BLOCKER
    assert params["wing_rack_fw305_verified_osc_mapping_required"] is True
    assert params["wing_rack_fw305_readback_required"] is True
    assert params["wing_rack_fw305_rollback_required"] is True
    assert params["live_authority_granted"] is False
    assert params["live_write"] is False
    assert params["readiness_closure_allowed"] is False
    gain_down = [
        candidate
        for candidate in recommendation.alternatives
        if candidate.strategy_id == "level_plane_gain_down"
    ][0]
    assert gain_down.method_family == "gain_fader"
    assert gain_down.parameters["delta_db"] == -1.5
    assert gain_down.blockers == (WING_GAIN_OPERATOR_OWNED_BLOCKER,)
    tradeoffs = {
        row["strategy_id"]: row["why_not_selected"]
        for row in recommendation.evidence["strategy_search"][
            "alternative_tradeoffs"
        ]
    }
    assert tradeoffs["level_plane_gain_down"] == (
        "live_operator_owned_gain_review_priority:"
        "manual_fader_or_trim_owner_review_precedes_live_gain_move"
    )
    proposal = recommendation.to_operator_proposal()
    assert proposal["selected_wing_osc_summary"][
        "wing_osc_candidate_apply_status"
    ] == "analysis_only_no_console_write"
    assert proposal["can_apply_live"] is False
    assert proposal["live_authority_granted"] is False
    assert proposal["live_write"] is False


def test_soundcheck_quiet_source_feedback_risk_holds_before_gain_up():
    recommendation = analyze_smart_dsp_strategy(
        _analysis(
            target_id="ch:10",
            source_role="lead_vocal",
            problem_hints=("level_plane_balance",),
            muddiness_score=0.0,
            estimated_lufs=-31.0,
            level_plane_error_lu=-5.0,
            true_peak_headroom_db=8.0,
            feedback_risk=0.5,
            monitor_feedback_risk=0.2,
            bleed_risk=0.1,
        )
    )

    assert recommendation.status == "proposal_only"
    assert recommendation.selected_strategy.strategy_id == (
        "hold_gain_up_for_feedback_path_review"
    )
    assert recommendation.selected_strategy.method_family == "analysis_only"
    assert recommendation.evidence["operator_explanation_sections"][
        "selection_policy"
    ] == "soundcheck_feedback_path_before_gain_up_priority"
    params = recommendation.selected_strategy.parameters
    assert params["read_only_review"] is True
    assert params["gain_db"] == 0.0
    assert params["feedback_risk"] == 0.5
    assert params["monitor_feedback_risk"] == 0.2
    assert params["required_context"] == (
        "pa_monitor_feedback_path_before_source_gain_up"
    )

    alternatives = {
        candidate.strategy_id: candidate for candidate in recommendation.alternatives
    }
    assert "feedback_path_review_should_precede_gain_up" in alternatives[
        "soundcheck_bounded_gain_up"
    ].blockers
    tradeoffs = {
        row["strategy_id"]: row
        for row in recommendation.evidence["strategy_search"]["alternative_tradeoffs"]
    }
    assert tradeoffs["soundcheck_bounded_gain_up"]["why_not_selected"] == (
        "soundcheck_feedback_path_gain_up_priority:"
        "feedback_path_review_precedes_gain_up"
    )
    assert recommendation.to_operator_proposal()["live_apply_allowed"] is False


def test_soundcheck_quiet_muddy_vocal_restores_level_before_eq():
    recommendation = analyze_smart_dsp_strategy(
        _analysis(
            target_id="ch:10",
            source_role="lead_vocal",
            problem_hints=("muddy",),
            muddiness_score=0.82,
            estimated_lufs=-31.0,
            level_plane_error_lu=-4.2,
            true_peak_headroom_db=7.0,
            feedback_risk=0.1,
            bleed_risk=0.1,
        )
    )

    assert recommendation.status == "proposal_only"
    assert recommendation.selected_strategy.strategy_id == "soundcheck_bounded_gain_up"
    assert recommendation.selected_strategy.method_family == "gain_fader"
    assert recommendation.evidence["operator_explanation_sections"][
        "selection_policy"
    ] == "soundcheck_quiet_level_plane_priority"
    params = recommendation.selected_strategy.parameters
    assert 0.0 < params["delta_db"] <= 3.0
    assert params["gain_db"] == params["delta_db"]
    assert params["target_scope"] == "source_and_summed_bus"
    low_mid_tradeoff = [
        row
        for row in recommendation.evidence["strategy_search"]["alternative_tradeoffs"]
        if row["strategy_id"] == "low_mid_corrective_cut"
    ][0]
    assert (
        low_mid_tradeoff["why_not_selected"]
        == "quiet_level_plane_priority:restore_level_before_tone_change"
    )
    assert recommendation.to_operator_proposal()["live_apply_allowed"] is False


def test_soundcheck_hot_muddy_vocal_reduces_level_before_low_mid_eq():
    recommendation = analyze_smart_dsp_strategy(
        _analysis(
            target_id="ch:10",
            source_role="lead_vocal",
            problem_hints=("muddy",),
            muddiness_score=0.72,
            mud_frequency_hz=260.0,
            estimated_lufs=-18.5,
            level_plane_error_lu=4.0,
            true_peak_headroom_db=4.5,
            feedback_risk=0.12,
            bleed_risk=0.1,
        )
    )

    assert recommendation.status == "proposal_only"
    assert recommendation.selected_strategy.strategy_id == "level_plane_gain_down"
    assert recommendation.selected_strategy.method_family == "gain_fader"
    assert recommendation.evidence["operator_explanation_sections"][
        "selection_policy"
    ] == "soundcheck_hot_level_plane_before_tone_priority"
    params = recommendation.selected_strategy.parameters
    assert params["delta_db"] < 0.0
    assert params["target_plane"] == "ayaic_mix_monolith"
    assert params["current_minus_target_lu"] == 4.0
    assert params["target_scope"] == "source_and_summed_bus"
    _assert_gain_delta_readback_contract(
        recommendation.selected_strategy,
        source="delta_db",
        target_surface="channel",
        expected_delta=params["delta_db"],
        path_prefix="/ch/{channel}",
    )
    low_mid_tradeoff = [
        row
        for row in recommendation.evidence["strategy_search"][
            "alternative_tradeoffs"
        ]
        if row["strategy_id"] == "low_mid_corrective_cut"
    ][0]
    assert low_mid_tradeoff["why_not_selected"] == (
        "soundcheck_hot_level_plane_priority:"
        "restore_level_before_low_mid_tone_processing"
    )
    assert recommendation.to_operator_proposal()["live_apply_allowed"] is False


def test_soundcheck_quiet_muddy_vocal_hot_bus_rebalances_bus_before_eq():
    recommendation = analyze_smart_dsp_strategy(
        _analysis(
            target_id="ch:10",
            source_role="lead_vocal",
            problem_hints=("muddy",),
            muddiness_score=0.82,
            estimated_lufs=-31.0,
            level_plane_error_lu=-4.2,
            true_peak_headroom_db=7.0,
            feedback_risk=0.1,
            bleed_risk=0.1,
            summed_bus_evidence_available=True,
            summed_bus_role="vocal_bus",
            summed_bus_target_id="bus:1",
            summed_bus_true_peak_headroom_db=0.5,
            summed_bus_level_plane_error_lu=4.0,
        )
    )

    assert recommendation.status == "proposal_only"
    assert (
        recommendation.selected_strategy.strategy_id
        == "summed_bus_rebalance_before_source_gain"
    )
    assert recommendation.selected_strategy.method_family == "gain_fader"
    assert recommendation.evidence["operator_explanation_sections"][
        "selection_policy"
    ] == "soundcheck_quiet_hot_bus_rebalance_priority"
    params = recommendation.selected_strategy.parameters
    assert params["bus_delta_db"] < 0.0
    assert params["source_delta_db"] == 0.0
    assert params["target_scope"] == "summed_bus_before_source_gain"
    assert params["summed_bus_target_id"] == "bus:1"
    assert params["wing_osc_target_id_override"] == "bus:1"
    assert params["wing_osc_effective_target_id"] == "bus:1"
    _assert_gain_delta_readback_contract(
        recommendation.selected_strategy,
        source="bus_delta_db",
        target_surface="bus",
        expected_delta=params["bus_delta_db"],
        path_prefix="/bus/{bus}",
    )
    low_mid_tradeoff = [
        row
        for row in recommendation.evidence["strategy_search"]["alternative_tradeoffs"]
        if row["strategy_id"] == "low_mid_corrective_cut"
    ][0]
    assert (
        low_mid_tradeoff["why_not_selected"]
        == "summed_bus_level_priority:rebalance_bus_before_tone_change"
    )
    blocked_source_gain = [
        candidate
        for candidate in recommendation.alternatives
        if candidate.strategy_id == "soundcheck_bounded_gain_up"
    ][0]
    assert "summed_bus_headroom_blocks_gain_up" in blocked_source_gain.blockers
    assert recommendation.to_operator_proposal()["live_apply_allowed"] is False


def test_soundcheck_quiet_source_hot_bus_rebalances_bus_before_source_gain_up():
    recommendation = analyze_smart_dsp_strategy(
        _analysis(
            target_id="ch:10",
            source_role="lead_vocal",
            problem_hints=("level_plane_balance",),
            muddiness_score=0.0,
            estimated_lufs=-30.0,
            level_plane_error_lu=-8.0,
            true_peak_headroom_db=8.0,
            feedback_risk=0.12,
            bleed_risk=0.1,
            summed_bus_evidence_available=True,
            summed_bus_role="vocal_bus",
            summed_bus_target_id="bus:1",
            summed_bus_true_peak_headroom_db=0.4,
            summed_bus_estimated_lufs=-20.5,
            summed_bus_target_plane_lufs=-25.0,
            summed_bus_level_plane_error_lu=4.5,
        )
    )

    assert recommendation.status == "proposal_only"
    assert recommendation.selected_strategy.strategy_id == (
        "summed_bus_rebalance_before_source_gain"
    )
    assert recommendation.selected_strategy.method_family == "gain_fader"
    params = recommendation.selected_strategy.parameters
    assert params["bus_delta_db"] < 0.0
    assert params["source_delta_db"] == 0.0
    assert params["gain_db"] == 0.0
    assert params["target_scope"] == "summed_bus_before_source_gain"
    assert params["summed_bus_target_id"] == "bus:1"
    assert params["wing_osc_target_id_override"] == "bus:1"
    assert params["wing_osc_effective_target_id"] == "bus:1"
    _assert_gain_delta_readback_contract(
        recommendation.selected_strategy,
        source="bus_delta_db",
        target_surface="bus",
        expected_delta=params["bus_delta_db"],
        path_prefix="/bus/{bus}",
    )
    assert recommendation.evidence["summed_bus"]["evidence_available"] is True
    assert recommendation.evidence["summed_bus"]["target_id"] == "bus:1"
    assert recommendation.evidence["summed_bus"]["true_peak_headroom_db"] == 0.4

    blocked_source_gain = [
        candidate
        for candidate in recommendation.alternatives
        if candidate.strategy_id == "soundcheck_bounded_gain_up"
    ][0]
    assert "summed_bus_headroom_blocks_gain_up" in blocked_source_gain.blockers
    assert "summed_bus_level_plane_rebalance_required" in blocked_source_gain.blockers
    assert recommendation.to_operator_proposal()["live_apply_allowed"] is False


def test_summed_bus_rebalance_named_bus_target_is_not_wing_apply_capable():
    recommendation = analyze_smart_dsp_strategy(
        _analysis(
            target_id="ch:10",
            source_role="lead_vocal",
            problem_hints=("level_plane_balance",),
            muddiness_score=0.0,
            estimated_lufs=-30.0,
            level_plane_error_lu=-8.0,
            true_peak_headroom_db=8.0,
            feedback_risk=0.12,
            bleed_risk=0.1,
            summed_bus_evidence_available=True,
            summed_bus_role="vocal_bus",
            summed_bus_target_id="bus:vocals",
            summed_bus_true_peak_headroom_db=0.4,
            summed_bus_estimated_lufs=-20.5,
            summed_bus_target_plane_lufs=-25.0,
            summed_bus_level_plane_error_lu=4.5,
        )
    )

    assert recommendation.status == "analysis_only"
    assert recommendation.selected_strategy.strategy_id == "hold_for_level_plane_context"
    assert recommendation.selected_strategy.method_family == "analysis_only"
    rebalance = [
        candidate
        for candidate in recommendation.alternatives
        if candidate.strategy_id == "summed_bus_rebalance_before_source_gain"
    ][0]
    params = rebalance.parameters
    assert params["summed_bus_target_id"] == "bus:vocals"
    assert "wing_osc_target_id_override" not in params
    assert params["wing_osc_original_target_id"] == "ch:10"
    assert params["wing_osc_effective_target_id"] == (
        "unresolved:required_wing_gain_target_override"
    )
    assert params["wing_gain_target_override_blocker"] == (
        "summed_bus_rebalance_requires_concrete_wing_bus_target"
    )
    assert WING_OSC_UNRESOLVED_TARGET_BLOCKER in rebalance.blockers
    applicability = rebalance.future_wing_readback_contract[
        "wing_osc_applicability"
    ]
    assert applicability["target_id"] == "unresolved:required_wing_gain_target_override"
    assert applicability["target_surface_kind"] == "unresolved"
    assert applicability["target_index_valid"] is False
    assert applicability["native_osc_supported"] is False
    assert applicability["native_osc_support_level"] == "unresolved_wing_osc_target"
    proposal = recommendation.to_operator_proposal()
    assert proposal["selected_wing_osc_summary"]["wing_osc_native_supported"] is False
    assert proposal["selected_wing_osc_summary"]["wing_osc_apply_parameter_values"] == {}
    assert proposal["live_apply_allowed"] is False


def test_live_loud_source_selects_gain_down_not_tone_change():
    recommendation = analyze_smart_dsp_strategy(
        _analysis(
            target_id="ch:14",
            source_role="bass",
            mode="live",
            problem_hints=("level_plane_balance",),
            muddiness_score=0.0,
            estimated_lufs=-19.5,
            level_plane_error_lu=5.5,
            true_peak_headroom_db=3.5,
            active_performance=True,
        )
    )

    assert recommendation.status == "proposal_only"
    assert recommendation.selected_strategy.strategy_id == "level_plane_gain_down"
    assert recommendation.selected_strategy.method_family == "gain_fader"
    assert recommendation.selected_strategy.parameters["delta_db"] < 0.0
    assert recommendation.selected_strategy.parameters["bounded_delta_db"] <= 1.5
    tone_decoy = [
        candidate
        for candidate in recommendation.alternatives
        if candidate.strategy_id == "avoid_tone_change_for_level_problem"
    ][0]
    assert "tone_change_does_not_match_level_plane_problem" in tone_decoy.blockers


def test_batch_builder_returns_operator_proposals():
    proposals = build_smart_dsp_strategy_proposals(
        [
            _analysis(muddiness_score=0.8),
            _analysis(
                target_id="ch:9",
                source_role="snare",
                problem_hints=("noisy_drum",),
                muddiness_score=0.0,
                noise_floor_dbfs=-48.0,
                bleed_risk=0.72,
            ),
        ]
    )

    assert len(proposals) == 2
    for proposal in proposals:
        _assert_proposal_denial(proposal)
        assert proposal["evidence"]["transport_free"] is True


def test_smart_dsp_strategy_engine_is_registered_as_proposal_only_method():
    method = get_processing_method(SMART_DSP_METHOD_ID)
    assert method is not None
    assert method.status == "proposal_only"
    assert method.domain == "decision_evidence"
    assert method.can_analyze is True
    assert method.can_propose is True
    assert method.module == "backend/smart_dsp_strategy_engine.py"

    profile = get_pipeline_profile("main_live_assist")
    assert profile is not None
    assert SMART_DSP_METHOD_ID in profile.method_ids
    assert SMART_DSP_METHOD_ID in profile.proposal_method_ids
    assert SMART_DSP_METHOD_ID not in profile.executor_method_ids

    contracts = {item["method_id"]: item for item in build_proposal_only_method_contracts()}
    contract = contracts[SMART_DSP_METHOD_ID]
    assert contract["value_type"] == "smart_dsp_strategy"
    assert contract["target_kind"] == "offline_soundcheck_live_strategy_selection"
    assert contract["can_apply_live"] is False
    assert contract["queue_import_allowed"] is False
    assert contract["live_authority_granted"] is False
    assert "selected_strategy" in contract["proposal_contract"]["safety_evidence_required"]
    assert "alternatives" in contract["proposal_contract"]["safety_evidence_required"]
