from copy import deepcopy
import json
from pathlib import Path
from types import SimpleNamespace

from backend.smart_dsp_replay_evaluation import (
    AUTHORITY_FALSE_FLAGS,
    SMART_DSP_REPLAY_EVALUATION_SCHEMA,
    SmartDSPReplayScenario,
    _candidate_readback_contract_failures,
    _readback_contract_failures,
    build_default_smart_dsp_replay_scenarios,
    evaluate_smart_dsp_replay_scenarios,
    render_smart_dsp_replay_evaluation_markdown,
    write_smart_dsp_replay_evaluation_artifacts,
)
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
    SOUNDCHECK_GATE_EXPANDER_MAX_HYSTERESIS_REVIEW_DB,
    SOUNDCHECK_GATE_EXPANDER_MAX_RANGE_DB,
    SOUNDCHECK_GATE_EXPANDER_MIN_HIT_BLEED_MARGIN_DB,
    SOUNDCHECK_PAN_MAX_DELTA_PERCENT,
    SOUNDCHECK_WIDTH_MAX_REDUCTION_PERCENT,
    SOUNDCHECK_WIDTH_MAX_WIDEN_PERCENT,
    SOUNDCHECK_WIDTH_MIN_TARGET_PERCENT,
    WING_PAN_WIDTH_OSC_PROTOCOL_SOURCE,
    WING_PAN_WIDTH_OSC_PROTOCOL_VERSION,
    WING_PAN_WIDTH_OPERATOR_OWNED_BLOCKER,
    WING_ACTIONABLE_PARAMETER_BLOCKER,
    WING_COMPRESSOR_AUTO_MIN,
    WING_COMPRESSOR_CROSSOVER_DEPTH_MAX_DB,
    WING_COMPRESSOR_CROSSOVER_DEPTH_MIN_DB,
    WING_COMPRESSOR_CROSSOVER_FREQ_MAX_HZ,
    WING_COMPRESSOR_CROSSOVER_FREQ_MIN_HZ,
    WING_COMPRESSOR_CROSSOVER_TYPE_VALUES,
    WING_COMPRESSOR_DETECTOR_BLOCKER,
    WING_COMPRESSOR_ON_MAX,
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
    WING_DE_ESSER_GATE_MODEL,
    WING_DE_ESSER_GATE_SLOT_CONFLICT_POLICY,
    WING_DE_ESSER_WRITE_KEY_SELECTION_POLICY,
    WING_DE_ESSER_WRITE_PARAMETER_KEYS,
    WING_DYNAMIC_EQ_GATE_MODEL,
    WING_DYNAMIC_EQ_WRITE_KEY_SELECTION_POLICY,
    WING_DYNAMIC_EQ_WRITE_PARAMETER_KEYS,
    WING_EQ_MIX_BASELINE_PERCENT,
    WING_EQ_MIX_MAX_PERCENT_BY_SURFACE,
    WING_EQ_MIX_MIN_PERCENT,
    WING_EQ_FILTER_OPERATOR_OWNED_BLOCKER,
    WING_GATE_ON_MAX,
    WING_AUX_DYNAMICS_OPERATOR_OWNED_BLOCKER,
    WING_DYNAMICS_GAIN_REDUCTION_OBJECTIVE_TOLERANCE_DB,
    WING_DYNAMICS_REQUIRED_MODEL,
    WING_DYNAMICS_REVIEW_ONLY_PARAMETER_REASONS,
    WING_DYNAMICS_OPERATOR_OWNED_MODEL_BLOCKER,
    WING_DYNAMICS_WRITE_KEY_SELECTION_POLICY,
    WING_DYNAMICS_WRITE_PARAMETER_KEYS,
    WING_EQ_REQUIRED_MODEL,
    WING_FX_ACTIONABLE_PARAMETER_BLOCKER,
    WING_FX_LEVEL_WRITE_PARAMETER_KEYS,
    WING_FX_LEVEL_OPERATOR_OWNED_BLOCKER,
    WING_FX_MODEL_PARAMETER_BLOCKER,
    WING_FX_OPERATOR_OWNED_MODEL_BLOCKER,
    WING_FX_MODEL_PARAMETER_PATH_PATTERN,
    WING_FX_POST_PULLBACK_MODEL_TIME_REVIEW_FIELDS,
    WING_FX_MODEL_SPECIFIC_PARAMETER_KEYS,
    WING_FX_RETURN_SLOT_BINDING_BLOCKER,
    WING_SOUNDCHECK_FX_MODEL_TIME_REVIEW_ONLY_FIELDS,
    WING_FX_SEND_SOURCE_SURFACE_BLOCKER,
    WING_EQ_WRITE_KEY_SELECTION_POLICY,
    WING_EQ_WRITE_PARAMETER_KEYS,
    WING_EQ_OPERATOR_OWNED_MODEL_BLOCKER,
    WING_GAIN_OPERATOR_OWNED_BLOCKER,
    WING_GAIN_WRITE_PATH_SELECTION_POLICY,
    WING_GAIN_WRITE_PATH_SUFFIXES,
    WING_GATE_REQUIRED_MODEL,
    WING_GATE_HYSTERESIS_REVIEW_TOLERANCE_DB,
    WING_GATE_SLOT_OPERATOR_OWNED_MODEL_BLOCKER,
    WING_GATE_SLOT_SINGLE_MODEL_COMBO_BLOCKER,
    WING_GATE_SLOT_SINGLE_MODEL_COMBO_POLICY,
    WING_GATE_SIDECHAIN_FREQ_MAX_HZ,
    WING_GATE_SIDECHAIN_FREQ_MIN_HZ,
    WING_GATE_SIDECHAIN_Q_MAX,
    WING_GATE_SIDECHAIN_Q_MIN,
    WING_GATE_SIDECHAIN_SOURCE_VALUES,
    WING_GATE_SIDECHAIN_TAP_VALUES,
    WING_GATE_SIDECHAIN_TYPE_VALUES,
    WING_GATE_WRITE_KEY_SELECTION_POLICY,
    WING_GATE_WRITE_PARAMETER_KEYS,
    WING_MUTE_OFF_VALUE,
    WING_MUTE_ON_VALUE,
    WING_MUTE_OPERATOR_OWNED_BLOCKER,
    WING_MUTE_WRITE_KEY_SELECTION_POLICY,
    WING_MUTE_WRITE_PARAMETER_KEYS,
    WING_OSC_UNRESOLVED_TARGET_BLOCKER,
    _future_readback_contract,
    _with_wing_native_parameter_mappings,
)
from tools.evaluate_smart_dsp_replay import main as evaluate_smart_dsp_replay_main


_CACHED_DEFAULT_SMART_DSP_REPLAY_REPORT = None


def _default_smart_dsp_replay_report(*, generated_at_utc):
    global _CACHED_DEFAULT_SMART_DSP_REPLAY_REPORT
    if _CACHED_DEFAULT_SMART_DSP_REPLAY_REPORT is None:
        _CACHED_DEFAULT_SMART_DSP_REPLAY_REPORT = evaluate_smart_dsp_replay_scenarios(
            generated_at_utc=generated_at_utc,
        )

    report = deepcopy(_CACHED_DEFAULT_SMART_DSP_REPLAY_REPORT)
    report["generated_at_utc"] = generated_at_utc
    return report


def _assert_replay_tradeoff_wing_osc_summary(
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
    expected_review_only_apply_overlap = sorted(
        set(row["wing_osc_apply_parameter_keys"])
        & set(row["wing_osc_review_only_parameter_keys"])
    )
    expected_unsupported_apply_overlap = sorted(
        set(row["wing_osc_apply_parameter_keys"])
        & set(row["wing_osc_unsupported_parameter_keys"])
    )
    expected_absolute_apply_overlap = sorted(
        set(row["wing_osc_apply_parameter_keys"])
        & set(row["wing_osc_absolute_value_readback_parameter_keys"])
    )
    expected_console_native_only = (
        not row["wing_osc_apply_parameter_keys"]
        or (
            row["wing_osc_native_supported"] is True
            and set(row["wing_osc_apply_parameter_keys"]).issubset(
                set(row["wing_osc_actionable_parameter_keys"])
            )
            and not expected_review_only_apply_overlap
            and not expected_unsupported_apply_overlap
            and not expected_absolute_apply_overlap
        )
    )
    assert row["wing_osc_direct_apply_is_console_native_only"] is (
        expected_console_native_only
    )
    assert row["wing_osc_direct_apply_review_only_overlap_keys"] == (
        expected_review_only_apply_overlap
    )
    assert row["wing_osc_direct_apply_unsupported_overlap_keys"] == (
        expected_unsupported_apply_overlap
    )
    assert row["wing_osc_direct_apply_absolute_readback_overlap_keys"] == (
        expected_absolute_apply_overlap
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
    if native_supported is not None:
        assert row["wing_osc_native_supported"] is native_supported
    if support_level is not None:
        assert row["wing_osc_support_level"] == support_level


def _assert_replay_selected_wing_osc_readback_summary(
    row,
    *,
    native_supported=None,
    require_paths=False,
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
    expected_review_only_apply_overlap = sorted(
        set(row["wing_osc_readback_apply_parameter_keys"])
        & set(row["wing_osc_readback_review_only_parameter_keys"])
    )
    expected_absolute_apply_overlap = sorted(
        set(row["wing_osc_readback_apply_parameter_keys"])
        & set(row["wing_osc_readback_absolute_value_parameter_keys"])
    )
    expected_console_native_only = (
        not row["wing_osc_readback_apply_parameter_keys"]
        or (
            row["wing_osc_readback_native_supported"] is True
            and set(row["wing_osc_readback_apply_parameter_keys"]).issubset(
                set(row["wing_osc_readback_actionable_parameter_keys"])
            )
            and not expected_review_only_apply_overlap
            and not expected_absolute_apply_overlap
            and not row[
                "wing_osc_readback_direct_apply_unsupported_overlap_keys"
            ]
        )
    )
    assert row["wing_osc_readback_direct_apply_is_console_native_only"] is (
        expected_console_native_only
    )
    assert row["wing_osc_readback_direct_apply_review_only_overlap_keys"] == (
        expected_review_only_apply_overlap
    )
    assert row["wing_osc_readback_direct_apply_unsupported_overlap_keys"] == []
    assert row["wing_osc_readback_direct_apply_absolute_readback_overlap_keys"] == (
        expected_absolute_apply_overlap
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
    assert row["wing_osc_readback_required_before_live_promotion"] is True
    assert row["wing_osc_readback_requires_real_wing_proof"] is True
    assert row["wing_osc_readback_requires_operator_approval"] is True
    assert row["wing_osc_readback_live_authority_granted"] is False
    assert row["wing_osc_readback_live_write"] is False
    assert row["wing_osc_readback_allowed_live_write_scope_before_wing_proof"] == []
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
    if require_paths:
        assert row["wing_osc_readback_sections"]
        assert row["wing_osc_read_before_paths"]
        assert row["wing_osc_read_after_paths"]
        assert row["wing_osc_readback_path_templates"]
        assert row["wing_osc_readback_status"] == (
            "wing_osc_readback_paths_declared_before_live_promotion"
        )
    elif row["wing_osc_readback_path_templates"]:
        assert row["wing_osc_readback_status"] == (
            "analysis_review_only_wing_osc_readback_paths_declared_no_live_write"
        )
    else:
        assert row["wing_osc_readback_status"] == "analysis_only_no_console_write"


def _assert_replay_gain_delta_mapping(result, *, source, expected_delta, path_prefix):
    params = result["selected_parameters"]
    assert params["wing_gain_delta_source"] == source
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
    gain_readback = result["selected_future_wing_readback_contract"][
        "gain_fader_readback"
    ]
    assert gain_readback["read_before_paths"] == ["fdr", "in/set/trim"]
    assert gain_readback["read_after_paths"] == ["fdr", "in/set/trim"]
    assert f"{path_prefix}/fdr" in gain_readback["wing_osc_path_templates"]
    assert f"{path_prefix}/in/set/trim" in gain_readback["wing_osc_path_templates"]
    assert gain_readback["wing_gain_delta_mapping_required"] is True
    assert gain_readback["wing_gain_delta_db"] == expected_delta
    assert gain_readback["wing_gain_delta_abs_db"] == abs(expected_delta)
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
    assert gain_readback["absolute_gain_value_requires_current_readback"] is True
    assert gain_readback["wing_gain_no_absolute_osc_value_without_readback"] is True


def _assert_replay_master_output_target_override(result, *, original_target_id):
    params = result["selected_parameters"]
    gain_readback = result["selected_future_wing_readback_contract"][
        "gain_fader_readback"
    ]
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


def _assert_replay_soundcheck_master_output_guard(
    result,
    *,
    move_kind,
    followup_review,
):
    params = result["selected_parameters"]
    gain_readback = result["selected_future_wing_readback_contract"][
        "gain_fader_readback"
    ]
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


def _assert_replay_live_master_true_peak_output_guard(result):
    params = result["selected_parameters"]
    gain_readback = result["selected_future_wing_readback_contract"][
        "gain_fader_readback"
    ]
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


def _assert_replay_master_output_processing_followup_review_contract(
    result,
    *,
    expected_values,
    expected_native_apply_fields=("wing_gain_delta_db",),
):
    gain_readback = result["selected_future_wing_readback_contract"][
        "gain_fader_readback"
    ]
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
    assert not (
        set(expected_values)
        & set(result["selected_rationale"]["wing_osc_apply_parameter_values"])
    )


def _assert_replay_soundcheck_master_processing_guard(
    result,
    *,
    processing_types,
):
    params = result["selected_parameters"]
    contract = result["selected_future_wing_readback_contract"]
    if result["actual_method_family"] == "eq":
        processing_readback = contract["eq_band_readback"]
    else:
        processing_readback = contract["dynamics_readback"]
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


def _assert_replay_selected_gain_delta_apply_values(result, expected_delta):
    expected_readback_values = {"wing_gain_delta_db": expected_delta}
    assert result["selected_rationale"][
        "wing_osc_apply_parameter_values"
    ] == {}
    assert result["proposal_selected_wing_osc_readback_summary"][
        "wing_osc_readback_apply_parameter_values"
    ] == {}
    assert result["selected_rationale"][
        "wing_osc_absolute_value_readback_parameter_values"
    ] == expected_readback_values
    assert result["proposal_selected_wing_osc_readback_summary"][
        "wing_osc_readback_absolute_value_parameter_values"
    ] == expected_readback_values
    assert result["selected_rationale"]["wing_osc_candidate_apply_status"] == (
        "wing_native_osc_candidate_requires_absolute_value_readback_before_apply"
    )
    assert result["proposal_authority"]["live_apply_allowed"] is False
    assert result["proposal_authority"]["live_authority_granted"] is False
    assert result["proposal_authority"]["live_write"] is False


def _assert_replay_analysis_only_gain_has_no_apply_values(result):
    assert result["actual_method_family"] == "analysis_only"
    assert result["selected_rationale"][
        "wing_osc_apply_parameter_values"
    ] == {}
    assert result["proposal_selected_wing_osc_readback_summary"][
        "wing_osc_readback_apply_parameter_values"
    ] == {}
    assert result["proposal_authority"]["live_apply_allowed"] is False
    assert result["proposal_authority"]["live_authority_granted"] is False
    assert result["proposal_authority"]["live_write"] is False


def _assert_replay_selected_mute_apply_values(result, expected_value):
    expected_values = {"wing_mute_osc_value": expected_value}
    assert result["selected_rationale"][
        "wing_osc_apply_parameter_values"
    ] == expected_values
    assert result["proposal_selected_wing_osc_readback_summary"][
        "wing_osc_readback_apply_parameter_values"
    ] == expected_values
    assert result["proposal_authority"]["live_apply_allowed"] is False
    assert result["proposal_authority"]["live_authority_granted"] is False
    assert result["proposal_authority"]["live_write"] is False


def _assert_replay_selected_eq_apply_values(result, expected_values):
    assert result["actual_method_family"] == "eq"
    assert result["selected_rationale"][
        "wing_osc_apply_parameter_values"
    ] == expected_values
    assert result["proposal_selected_wing_osc_readback_summary"][
        "wing_osc_readback_apply_parameter_values"
    ] == expected_values
    assert result["proposal_authority"]["live_apply_allowed"] is False
    assert result["proposal_authority"]["live_authority_granted"] is False
    assert result["proposal_authority"]["live_write"] is False


def _assert_replay_eq_write_key_contract(result, *, selected_write_keys):
    params = result["selected_parameters"]
    readback = result["selected_future_wing_readback_contract"]["eq_band_readback"]
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


def _assert_replay_eq_selected_parameter_rollback_operations(
    result,
    *,
    selected_write_keys,
):
    params = result["selected_parameters"]
    readback = result["selected_future_wing_readback_contract"]["eq_band_readback"]
    operations = readback["wing_eq_selected_parameter_rollback_operations"]
    assert [operation["selected_write_key"] for operation in operations] == (
        selected_write_keys
    )
    operation_by_key = {
        operation["selected_write_key"]: operation for operation in operations
    }
    for key in selected_write_keys:
        operation = operation_by_key[key]
        assert operation["parameter_key"] == key
        assert operation["selected_write_key"] == key
        assert operation["read_before_required"] is True
        assert operation["read_after_required"] is True
        assert operation["readback_must_match_osc_value"] is True
        assert operation["previous_value_readback_required"] is True
        assert operation["rollback_required"] is True
        assert operation["rollback_value_source"] == "previous_value_readback"
        assert operation["rollback_readback_required"] is True
        assert (
            operation["fw305_selected_parameter_rollback_contract"]
            == "read_previous_eq_value_before_write_then_restore_same_path"
        )
        if key != "wing_eq_band_operations":
            base = key[: -len("_osc_value")] if key.endswith("_osc_value") else key
            path_suffix = None
            direct_suffix = params.get(f"{base}_osc_path_suffix")
            if direct_suffix:
                path_suffix = direct_suffix
            else:
                for mapping_key, mapping in params.items():
                    if not str(mapping_key).endswith("_osc_path_suffixes"):
                        continue
                    if not isinstance(mapping, dict):
                        continue
                    for name, suffix in mapping.items():
                        semantic_suffixes = (
                            f"_{name}",
                            f"_{name}_enabled",
                            f"_{name}_hz",
                            f"_{name}_db",
                            f"_{name}_q",
                            f"_{name}_percent",
                            f"_{name}_db_per_octave",
                            f"_{name}_osc_value",
                        )
                        if base.endswith(semantic_suffixes):
                            path_suffix = suffix
                            break
                    if path_suffix:
                        break
            assert operation["osc_path_suffix"] == path_suffix
            assert operation["osc_value"] == params[key]
            assert operation["previous_value_path_suffix"] == path_suffix
            assert operation["rollback_path_suffix"] == path_suffix
            assert path_suffix in readback["read_before_paths"]
            assert path_suffix in readback["read_after_paths"]
            continue

        band_parameter_operations = operation["band_parameter_operations"]
        expected_band_parameter_operations = []
        for operation_index, band_operation in enumerate(
            params["wing_eq_band_operations"]
        ):
            for parameter_name, value_key in (
                ("frequency", "frequency_hz"),
                ("gain", "gain_db"),
                ("q", "q"),
            ):
                path_suffix = band_operation["path_suffixes"][parameter_name]
                expected_band_parameter_operations.append(
                    {
                        "band_operation_index": operation_index,
                        "native_band_number": band_operation.get(
                            "native_band_number"
                        ),
                        "parameter_name": parameter_name,
                        "osc_path_suffix": path_suffix,
                        "osc_value": band_operation[value_key],
                        "value_type": band_operation["value_type"],
                        "read_before_required": True,
                        "read_after_required": True,
                        "readback_must_match_osc_value": True,
                        "previous_value_readback_required": True,
                        "previous_value_path_suffix": path_suffix,
                        "rollback_required": True,
                        "rollback_value_source": "previous_value_readback",
                        "rollback_path_suffix": path_suffix,
                        "rollback_readback_required": True,
                    }
                )
        assert band_parameter_operations == expected_band_parameter_operations
        assert operation["band_operation_count"] == len(
            params["wing_eq_band_operations"]
        )
        assert operation["band_parameter_operation_count"] == len(
            expected_band_parameter_operations
        )
        assert operation["requires_concrete_band_readback_before_apply"] is True


def _assert_replay_selected_write_key_contract(
    result,
    *,
    readback_section,
    prefix,
    selected_write_keys,
    write_parameter_keys,
    selection_policy,
):
    params = result["selected_parameters"]
    readback = result["selected_future_wing_readback_contract"][readback_section]
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


def _assert_replay_selected_parameter_rollback_operations(
    result,
    *,
    readback_section,
    operations_field,
    selected_write_keys,
    contract_marker,
):
    params = result["selected_parameters"]
    readback = result["selected_future_wing_readback_contract"][readback_section]
    operations = readback[operations_field]

    def path_suffix_for_key(key):
        base = key[: -len("_osc_value")] if key.endswith("_osc_value") else key
        direct_suffix = params.get(f"{base}_osc_path_suffix")
        if direct_suffix:
            return direct_suffix
        for mapping_key, mapping in params.items():
            if not str(mapping_key).endswith("_osc_path_suffixes"):
                continue
            if not isinstance(mapping, dict):
                continue
            for name, suffix in mapping.items():
                semantic_suffixes = (
                    f"_{name}",
                    f"_{name}_enabled",
                    f"_{name}_hz",
                    f"_{name}_db",
                    f"_{name}_q",
                    f"_{name}_percent",
                    f"_{name}_db_per_octave",
                    f"_{name}_osc_value",
                )
                if base.endswith(semantic_suffixes):
                    return suffix
        raise AssertionError(f"missing OSC path suffix for {key}")

    assert [operation["selected_write_key"] for operation in operations] == (
        selected_write_keys
    )
    for key, operation in zip(selected_write_keys, operations):
        base = key[: -len("_osc_value")] if key.endswith("_osc_value") else key
        path_suffix = path_suffix_for_key(key)
        assert operation["parameter_key"] == key
        assert operation["selected_write_key"] == key
        assert operation["osc_path_suffix"] == path_suffix
        assert operation["osc_value"] == params[key]
        assert operation["value_type"] == params[f"{base}_value_type"]
        assert operation["read_before_required"] is True
        assert operation["read_after_required"] is True
        assert operation["readback_must_match_osc_value"] is True
        assert operation["previous_value_readback_required"] is True
        assert operation["previous_value_path_suffix"] == path_suffix
        assert operation["rollback_required"] is True
        assert operation["rollback_value_source"] == "previous_value_readback"
        assert operation["rollback_path_suffix"] == path_suffix
        assert operation["rollback_readback_required"] is True
        assert (
            operation["fw305_selected_parameter_rollback_contract"]
            == contract_marker
        )
        assert path_suffix in readback["read_before_paths"]
        assert path_suffix in readback["read_after_paths"]


def _assert_replay_selected_de_esser_apply_values(result, expected_values):
    assert result["selected_rationale"][
        "wing_osc_apply_parameter_values"
    ] == expected_values
    assert result["proposal_selected_wing_osc_readback_summary"][
        "wing_osc_readback_apply_parameter_values"
    ] == expected_values
    assert result["proposal_authority"]["live_apply_allowed"] is False
    assert result["proposal_authority"]["live_authority_granted"] is False
    assert result["proposal_authority"]["live_write"] is False


def _assert_replay_selected_dynamic_eq_apply_values(result, expected_values):
    assert result["actual_method_family"] == "dynamic_eq"
    assert result["selected_rationale"][
        "wing_osc_apply_parameter_values"
    ] == expected_values
    assert result["proposal_selected_wing_osc_readback_summary"][
        "wing_osc_readback_apply_parameter_values"
    ] == expected_values
    assert result["proposal_authority"]["live_apply_allowed"] is False
    assert result["proposal_authority"]["live_authority_granted"] is False
    assert result["proposal_authority"]["live_write"] is False


def _assert_replay_selected_dynamics_apply_values(result, expected_values):
    assert result["actual_method_family"] == "dynamics"
    assert result["selected_rationale"][
        "wing_osc_apply_parameter_values"
    ] == expected_values
    assert result["proposal_selected_wing_osc_readback_summary"][
        "wing_osc_readback_apply_parameter_values"
    ] == expected_values
    assert result["proposal_authority"]["live_apply_allowed"] is False
    assert result["proposal_authority"]["live_authority_granted"] is False
    assert result["proposal_authority"]["live_write"] is False


def _assert_replay_selected_gate_apply_values(result, expected_values):
    assert result["actual_method_family"] in {"gate", "expander"}
    assert result["selected_rationale"][
        "wing_osc_apply_parameter_values"
    ] == expected_values
    assert result["proposal_selected_wing_osc_readback_summary"][
        "wing_osc_readback_apply_parameter_values"
    ] == expected_values
    assert result["proposal_authority"]["live_apply_allowed"] is False
    assert result["proposal_authority"]["live_authority_granted"] is False
    assert result["proposal_authority"]["live_write"] is False


def _assert_replay_analysis_only_has_no_apply_values(result):
    assert result["actual_method_family"] == "analysis_only"
    assert result["selected_rationale"][
        "wing_osc_apply_parameter_values"
    ] == {}
    assert result["proposal_selected_wing_osc_readback_summary"][
        "wing_osc_readback_apply_parameter_values"
    ] == {}
    assert result["proposal_authority"]["live_apply_allowed"] is False
    assert result["proposal_authority"]["live_authority_granted"] is False
    assert result["proposal_authority"]["live_write"] is False


def _assert_replay_operator_owned_master_output_gain_hold(
    result,
    *,
    mode,
    expected_strategy_id,
    expected_policy,
    blocked_strategy_id,
    blocked_delta_db,
    move_kind,
    tradeoff_reason,
):
    assert result["actual_status"] == "analysis_only"
    assert result["actual_selected_strategy_id"] == expected_strategy_id
    assert result["actual_method_family"] == "analysis_only"
    assert result["operator_explanation_sections"]["selection_policy"] == expected_policy

    params = result["selected_parameters"]
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

    assert result["candidate_blockers"][blocked_strategy_id] == [
        WING_GAIN_OPERATOR_OWNED_BLOCKER
    ]
    tradeoffs = {
        item["strategy_id"]: item["why_not_selected"]
        for item in result["alternative_tradeoffs"]
    }
    assert tradeoffs[blocked_strategy_id] == tradeoff_reason

    readback = result["selected_future_wing_readback_contract"][
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

    assert result["selected_rationale"][
        "wing_osc_candidate_apply_status"
    ] == "analysis_only_no_console_write"
    assert result["proposal_selected_wing_osc_summary"][
        "wing_osc_candidate_apply_status"
    ] == "analysis_only_no_console_write"
    _assert_replay_analysis_only_has_no_apply_values(result)


def _assert_replay_mute_mapping(result, *, expected_value, path_prefix):
    params = result["selected_parameters"]
    assert params["wing_mute_osc_path_suffix"] == "mute"
    assert params["wing_mute_osc_value"] == expected_value
    assert params["wing_mute_value_type"] == "integer_bool"
    assert params["wing_mute_readback_must_match_osc_value"] is True
    mute_readback = result["selected_future_wing_readback_contract"]["mute_readback"]
    _assert_replay_selected_write_key_contract(
        result,
        readback_section="mute_readback",
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
    assert mute_readback["wing_mute_osc_value"] == expected_value
    assert mute_readback["wing_mute_readback_must_match_osc_value"] is True
    assert mute_readback["previous_mute_state_required_for_rollback"] is True
    _assert_replay_selected_parameter_rollback_operations(
        result,
        readback_section="mute_readback",
        operations_field="wing_mute_selected_parameter_rollback_operations",
        selected_write_keys=list(WING_MUTE_WRITE_PARAMETER_KEYS),
        contract_marker="read_previous_mute_value_before_write_then_restore_same_path",
    )


def _assert_replay_fx_level_write_key_contract(
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


def _assert_replay_fx_level_selected_parameter_rollback_operations(
    result,
    *,
    selected_write_keys,
):
    params = result["selected_parameters"]
    fx_readback = result["selected_future_wing_readback_contract"]["fx_readback"]
    operations = fx_readback["wing_fx_level_selected_parameter_rollback_operations"]
    assert [operation["selected_write_key"] for operation in operations] == (
        selected_write_keys
    )
    specs = {
        "wing_fx_send_level_delta_db": {
            "path_suffix_field": "wing_fx_send_level_osc_path_suffix",
            "value_type_field": "wing_fx_send_level_value_type",
            "absolute_value_requires_readback_field": (
                "wing_fx_send_level_absolute_value_requires_readback"
            ),
            "no_absolute_value_field": (
                "wing_fx_send_no_absolute_osc_value_without_readback"
            ),
        },
        "wing_fx_return_delta_db": {
            "path_suffix_field": "wing_fx_return_level_osc_path_suffix",
            "value_type_field": "wing_fx_return_value_type",
            "absolute_value_requires_readback_field": (
                "wing_fx_return_absolute_value_requires_readback"
            ),
            "no_absolute_value_field": (
                "wing_fx_return_no_absolute_osc_value_without_readback"
            ),
        },
    }
    for key, operation in zip(selected_write_keys, operations):
        spec = specs[key]
        path_suffix = params[spec["path_suffix_field"]]
        assert key in WING_FX_LEVEL_WRITE_PARAMETER_KEYS
        assert operation["parameter_key"] == key
        assert operation["selected_write_key"] == key
        assert operation["osc_path_suffix"] == path_suffix
        assert "osc_value" not in operation
        assert operation["relative_delta_value"] == params[key]
        assert operation["relative_delta_unit"] == "db"
        assert operation["value_type"] == params[spec["value_type_field"]]
        assert operation["absolute_value_requires_readback"] is True
        assert operation["no_absolute_osc_value_without_readback"] is True
        assert params[spec["absolute_value_requires_readback_field"]] is True
        assert params[spec["no_absolute_value_field"]] is True
        assert operation["read_before_required"] is True
        assert operation["read_after_required"] is True
        assert operation["resolved_osc_value_required_before_apply"] is True
        assert operation["resolved_osc_value_source"] == (
            "previous_value_readback_plus_relative_delta"
        )
        assert operation["readback_must_match_resolved_osc_value"] is True
        assert operation["previous_value_readback_required"] is True
        assert operation["previous_value_path_suffix"] == path_suffix
        assert operation["rollback_required"] is True
        assert operation["rollback_value_source"] == "previous_value_readback"
        assert operation["rollback_path_suffix"] == path_suffix
        assert operation["rollback_readback_required"] is True
        assert operation["fw305_selected_parameter_rollback_contract"] == (
            "read_previous_fx_level_before_relative_delta_then_restore_same_path"
        )
        assert path_suffix in fx_readback["read_before_paths"]
        assert path_suffix in fx_readback["read_after_paths"]


def _assert_replay_fx_send_level_mapping(
    result,
    *,
    expected_delta,
    target_surface,
    source_binding_required,
):
    params = result["selected_parameters"]
    assert params["wing_fx_send_level_target_surface"] == target_surface
    assert params["wing_fx_send_level_osc_path_suffix"] == "send/{send}/lvl"
    assert params["wing_fx_send_level_delta_db"] == expected_delta
    assert params["wing_fx_send_level_value_type"] == (
        "relative_db_delta_requires_send_index_and_current_level_readback"
    )
    assert params["wing_fx_send_level_absolute_value_requires_readback"] is True
    assert params["wing_fx_send_index_required"] is True
    assert params["wing_fx_send_no_absolute_osc_value_without_readback"] is True
    assert params["wing_fx_send_source_binding_required"] is source_binding_required
    fx_applicability = result["selected_future_wing_readback_contract"][
        "wing_osc_applicability"
    ]
    assert "wing_fx_send_level_delta_db" in fx_applicability[
        "actionable_parameter_keys"
    ]
    fx_readback = result["selected_future_wing_readback_contract"]["fx_readback"]
    target_send_level_templates = {
        "channel": "/ch/{channel}/send/{send}/lvl",
        "aux": "/aux/{aux}/send/{send}/lvl",
        "bus": "/bus/{bus}/send/{send}/lvl",
    }
    if target_surface in target_send_level_templates:
        assert (
            target_send_level_templates[target_surface]
            in fx_readback["wing_osc_path_templates"]
        )
    assert "/ch/{channel}/send/{send}/lvl" in fx_readback["wing_osc_path_templates"]
    assert "/aux/{aux}/send/{send}/lvl" in fx_readback["wing_osc_path_templates"]
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
        _assert_replay_fx_level_write_key_contract(
            params,
            fx_readback,
            prefix="soundcheck",
            selected_write_keys=["wing_fx_send_level_delta_db"],
            forbidden_write_keys=["wing_fx_return_delta_db"],
        )
    if "live_fx_level_move_kinds" in params:
        _assert_replay_fx_level_write_key_contract(
            params,
            fx_readback,
            prefix="live",
            selected_write_keys=["wing_fx_send_level_delta_db"],
            forbidden_write_keys=["wing_fx_return_delta_db"],
        )


def _assert_replay_fx_return_fxmix_mapping(
    result,
    *,
    expected_delta,
    target_surface,
    fx_slot_binding_required,
):
    params = result["selected_parameters"]
    assert params["wing_fx_return_level_target_surface"] == target_surface
    if fx_slot_binding_required:
        assert params.get("wing_fx_return_target_surface") != "fx"
    else:
        assert str(params["wing_fx_return_target_id"]).startswith("fx:")
        assert params["wing_fx_return_target_surface"] == "fx"
    assert params["wing_fx_return_level_osc_path_suffix"] == "fxmix"
    assert params["wing_fx_return_delta_db"] == expected_delta
    assert 0.0 < params["wing_fx_return_fxmix_multiplier"] < 1.0
    assert params["wing_fx_return_fxmix_delta_percent_at_100"] < 0.0
    assert params["wing_fx_return_value_type"] == (
        "relative_fxmix_percent_requires_readback_to_resolve_absolute_value"
    )
    assert params["wing_fx_return_absolute_value_requires_readback"] is True
    assert params["wing_fx_return_no_db_return_fader_mapping"] is True
    assert params["wing_fx_return_no_absolute_osc_value_without_readback"] is True
    assert params["wing_fx_return_fx_slot_required"] is True
    assert params["wing_fx_return_fx_slot_binding_required"] is fx_slot_binding_required
    fx_applicability = result["selected_future_wing_readback_contract"][
        "wing_osc_applicability"
    ]
    assert "wing_fx_return_delta_db" in fx_applicability["actionable_parameter_keys"]
    fx_readback = result["selected_future_wing_readback_contract"]["fx_readback"]
    assert "/fx/{fx}/mdl" in fx_readback["wing_osc_path_templates"]
    assert "/fx/{fx}/fxmix" in fx_readback["wing_osc_path_templates"]
    assert fx_readback["wing_fx_return_level_mapping_required"] is True
    if fx_slot_binding_required:
        assert fx_readback.get("wing_fx_return_target_surface") != "fx"
    else:
        assert str(fx_readback["wing_fx_return_target_id"]).startswith("fx:")
        assert fx_readback["wing_fx_return_target_surface"] == "fx"
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
        _assert_replay_fx_level_write_key_contract(
            params,
            fx_readback,
            prefix="soundcheck",
            selected_write_keys=["wing_fx_return_delta_db"],
            forbidden_write_keys=["wing_fx_send_level_delta_db"],
        )
    if "live_fx_level_move_kinds" in params:
        _assert_replay_fx_level_write_key_contract(
            params,
            fx_readback,
            prefix="live",
            selected_write_keys=["wing_fx_return_delta_db"],
            forbidden_write_keys=["wing_fx_send_level_delta_db"],
        )


def _assert_replay_fx_model_time_followup_review_contract(
    result,
    *,
    expected_values,
    expected_native_apply_fields,
    expected_review_fields,
):
    fx_readback = result["selected_future_wing_readback_contract"]["fx_readback"]
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
    assert not (
        set(expected_values)
        & set(result["selected_rationale"]["wing_osc_apply_parameter_values"])
    )


def _assert_replay_selected_fx_apply_values(result, expected_values):
    relative_expected_values = {
        key: value
        for key, value in expected_values.items()
        if key in {"wing_fx_send_level_delta_db", "wing_fx_return_delta_db"}
    }
    direct_expected_values = {
        key: value for key, value in expected_values.items() if key not in relative_expected_values
    }
    assert result["selected_rationale"][
        "wing_osc_apply_parameter_values"
    ] == direct_expected_values
    assert result["proposal_selected_wing_osc_readback_summary"][
        "wing_osc_readback_apply_parameter_values"
    ] == direct_expected_values
    assert result["selected_rationale"][
        "wing_osc_absolute_value_readback_parameter_values"
    ] == relative_expected_values
    assert result["proposal_selected_wing_osc_readback_summary"][
        "wing_osc_readback_absolute_value_parameter_values"
    ] == relative_expected_values
    if relative_expected_values:
        assert result["selected_rationale"]["wing_osc_candidate_apply_status"] == (
            "wing_native_osc_candidate_requires_absolute_value_readback_before_apply"
        )
    assert not (
        set(WING_FX_MODEL_SPECIFIC_PARAMETER_KEYS)
        & set(result["selected_rationale"]["wing_osc_apply_parameter_values"])
    )


def _assert_replay_selected_pan_width_apply_values(result, expected_values):
    assert result["selected_rationale"][
        "wing_osc_apply_parameter_values"
    ] == expected_values
    assert result["proposal_selected_wing_osc_readback_summary"][
        "wing_osc_readback_apply_parameter_values"
    ] == expected_values
    assert result["proposal_authority"]["live_apply_allowed"] is False
    assert result["proposal_authority"]["live_authority_granted"] is False
    assert result["proposal_authority"]["live_write"] is False


def _assert_replay_fx_model_specific_guard(result, candidate_id, expected_keys):
    params = result["candidate_parameters"][candidate_id]
    assert not any(key.startswith("wing_compressor_") for key in params)
    assert params["wing_fx_model_parameter_binding_required"] is True
    assert params["wing_fx_model_parameter_path_pattern"] == (
        WING_FX_MODEL_PARAMETER_PATH_PATTERN
    )
    assert params["wing_fx_model_parameter_path_pattern_apply_ready"] is False
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
    assert WING_FX_MODEL_PARAMETER_BLOCKER in result["candidate_blockers"][candidate_id]

    contract = result["candidate_future_wing_readback_contracts"][candidate_id]
    wing_osc = contract["wing_osc_applicability"]
    assert wing_osc["native_osc_supported"] is False
    assert wing_osc["native_osc_support_level"] == (
        "partial_model_specific_fx_mapping_required"
    )
    assert WING_FX_MODEL_PARAMETER_PATH_PATTERN not in wing_osc[
        "native_osc_path_templates"
    ]
    assert "/fx/{fx}/mdl" in wing_osc["native_osc_path_templates"]
    assert set(expected_keys) <= set(wing_osc["fx_model_specific_parameter_keys"])
    assert wing_osc["fx_model_specific_effects_appendix_mapping_available"] is True
    assert set(params["wing_fx_model_parameter_verified_path_templates"]) == set(
        wing_osc["fx_model_specific_verified_path_templates"]
    )
    fx_readback = contract["fx_readback"]
    assert WING_FX_MODEL_PARAMETER_PATH_PATTERN not in fx_readback[
        "wing_osc_path_templates"
    ]
    assert "/fx/{fx}/mdl" in fx_readback["wing_osc_path_templates"]
    assert fx_readback["wing_fx_model_parameter_mapping_required"] is True
    assert fx_readback["wing_fx_model_parameter_path_pattern_apply_ready"] is False
    assert fx_readback["wing_fx_model_parameter_live_apply_blocked"] is True
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


def test_smart_dsp_replay_evaluation_module_is_transport_free():
    source = Path("backend/smart_dsp_replay_evaluation.py").read_text()

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
        "connect_wing",
        "connect_mixer",
    ]
    for token in forbidden:
        assert token not in source


def test_default_smart_dsp_replay_scenarios_pass_and_deny_authority():
    report = _default_smart_dsp_replay_report(
        generated_at_utc="2026-06-24T20:30:00Z",
    )

    assert report["schema"] == SMART_DSP_REPLAY_EVALUATION_SCHEMA
    assert report["status"] == "passed"
    assert report["failed_checks"] == []
    assert report["scenario_count"] == 187
    assert report["passed_scenario_count"] == 187
    assert report["failed_scenario_count"] == 0
    assert report["pass_rate"] == 1.0
    assert report["read_only_or_dry_run_only"] is True
    assert report["contacts_wing"] is False
    assert report["contacts_backend"] is False
    assert report["queries_console"] is False
    assert report["mutates_mixer"] is False
    assert report["mutates_console_state"] is False
    assert report["live_write"] is False
    assert report["live_authority_granted"] is False
    assert report["readiness_closure_allowed"] is False
    assert report["live_apply_allowed"] is False
    assert report["queue_import_allowed"] is False
    assert report["can_apply_live"] is False
    assert report["can_import_to_queue"] is False
    assert report["operator_review_required"] is True
    assert report["future_wing_readback_required"] is True
    assert report["hardware_proof_required_before_live"] is True
    assert report["selected_unsupported_wing_osc_count"] == 0
    assert report["selected_unsupported_wing_osc_by_family"] == {}
    assert report["selected_unsupported_wing_osc_scenarios"] == []
    assert report["selected_operator_owned_review_only_wing_osc_count"] == 0
    assert report["selected_operator_owned_review_only_wing_osc_by_family"] == {}
    assert report["selected_operator_owned_review_only_wing_osc_scenarios"] == []
    assert report["selected_review_only_wing_osc_parameter_count"] == 13
    assert report["selected_review_only_wing_osc_by_family"] == {
        "dynamics": 8,
        "expander": 4,
        "gate": 1,
    }
    results = {row["scenario_id"]: row for row in report["results"]}
    review_only_by_id = {
        row["scenario_id"]: row
        for row in report["selected_review_only_wing_osc_scenarios"]
    }
    assert review_only_by_id["soundcheck_dynamic_vocal_compression"][
        "review_only_parameter_keys"
    ] == ["target_gain_reduction_db"]
    assert review_only_by_id["soundcheck_dynamic_vocal_compression"][
        "review_only_parameter_values"
    ]["target_gain_reduction_db"] == results["soundcheck_dynamic_vocal_compression"][
        "selected_parameters"
    ]["target_gain_reduction_db"]
    assert "no direct target gain-reduction write path" in review_only_by_id[
        "soundcheck_dynamic_vocal_compression"
    ]["review_only_parameter_reason_by_key"]["target_gain_reduction_db"]
    assert review_only_by_id["soundcheck_dynamic_vocal_compression"][
        "review_only_parameters_apply_ready"
    ] is False
    assert "wing_compressor_threshold_osc_value" in review_only_by_id[
        "soundcheck_dynamic_vocal_compression"
    ]["actionable_parameter_keys"]
    assert review_only_by_id["soundcheck_dynamic_vocal_compression"][
        "apply_parameter_keys"
    ] == review_only_by_id["soundcheck_dynamic_vocal_compression"][
        "actionable_parameter_keys"
    ]
    assert review_only_by_id["soundcheck_dynamic_vocal_compression"][
        "apply_parameter_count"
    ] == len(
        review_only_by_id["soundcheck_dynamic_vocal_compression"][
            "actionable_parameter_keys"
        ]
    )
    assert set(
        review_only_by_id["soundcheck_dynamic_vocal_compression"][
            "apply_parameter_values"
        ]
    ) == set(
        review_only_by_id["soundcheck_dynamic_vocal_compression"][
            "apply_parameter_keys"
        ]
    )
    assert "target_gain_reduction_db" not in review_only_by_id[
        "soundcheck_dynamic_vocal_compression"
    ]["apply_parameter_values"]
    assert review_only_by_id["soundcheck_dynamic_vocal_compression"][
        "apply_parameter_values"
    ]["wing_compressor_threshold_osc_value"] == results[
        "soundcheck_dynamic_vocal_compression"
    ]["selected_parameters"]["wing_compressor_threshold_osc_value"]
    assert review_only_by_id["soundcheck_dynamic_vocal_compression"][
        "correction_apply_scope"
    ] == "direct_wing_fw_3_0_5_osc_values_only_after_required_readback"
    assert review_only_by_id["soundcheck_dynamic_vocal_compression"][
        "review_only_parameters_are_apply_values"
    ] is False
    gr_objective = review_only_by_id["soundcheck_dynamic_vocal_compression"][
        "gain_reduction_objective"
    ]
    assert gr_objective[
        "wing_dynamics_gain_reduction_objective_required"
    ] is True
    assert gr_objective[
        "wing_dynamics_gain_reduction_objective_parameter_keys"
    ] == ["target_gain_reduction_db"]
    assert gr_objective["wing_dynamics_gain_reduction_objective_values"] == {
        "target_gain_reduction_db": results["soundcheck_dynamic_vocal_compression"][
            "selected_parameters"
        ]["target_gain_reduction_db"]
    }
    assert gr_objective[
        "wing_dynamics_gain_reduction_target_db"
    ] == results["soundcheck_dynamic_vocal_compression"]["selected_parameters"][
        "target_gain_reduction_db"
    ]
    assert gr_objective["wing_dynamics_gain_reduction_max_db"] is None
    assert gr_objective[
        "wing_dynamics_gain_reduction_objective_tolerance_db"
    ] == WING_DYNAMICS_GAIN_REDUCTION_OBJECTIVE_TOLERANCE_DB
    target_gr = results["soundcheck_dynamic_vocal_compression"][
        "selected_parameters"
    ]["target_gain_reduction_db"]
    assert gr_objective[
        "wing_dynamics_gain_reduction_acceptance_min_db"
    ] == round(
        max(0.0, target_gr - WING_DYNAMICS_GAIN_REDUCTION_OBJECTIVE_TOLERANCE_DB),
        3,
    )
    assert gr_objective[
        "wing_dynamics_gain_reduction_acceptance_max_db"
    ] == round(target_gr + WING_DYNAMICS_GAIN_REDUCTION_OBJECTIVE_TOLERANCE_DB, 3)
    assert gr_objective["wing_dynamics_gain_reduction_must_not_exceed_db"] is None
    assert (
        gr_objective["wing_dynamics_gain_reduction_meter_readback_path_suffix"]
        == "dyn/gr"
    )
    assert gr_objective[
        "wing_dynamics_gain_reduction_meter_readback_path_templates"
    ] == ["/ch/{channel}/dyn/gr"]
    assert gr_objective[
        "wing_dynamics_gain_reduction_meter_fallback_path_templates"
    ] == ["/ch/{channel}/dyn/$gr"]
    assert (
        gr_objective["wing_dynamics_gain_reduction_meter_value_type"]
        == "read_only_meter_db"
    )
    assert (
        gr_objective["wing_dynamics_gain_reduction_meter_is_not_apply_value"]
        is True
    )
    assert gr_objective[
        "wing_dynamics_gain_reduction_meter_path_source"
    ] == "wing_client_cached_state_and_operator_product_state"
    assert gr_objective[
        "wing_dynamics_gain_reduction_objective_apply_ready"
    ] is False
    assert gr_objective[
        "wing_dynamics_gain_reduction_objective_requires_post_apply_meter_readback"
    ] is True
    assert gr_objective[
        "wing_dynamics_gain_reduction_objective_requires_operator_confirmation"
    ] is True
    assert gr_objective[
        "wing_dynamics_gain_reduction_objective_live_promotion_status"
    ] == "blocked_until_actual_gain_reduction_meter_readback_and_operator_review"
    assert "actual WING compressor gain-reduction meter" in gr_objective[
        "wing_dynamics_gain_reduction_objective_operator_check"
    ]
    aux_review = review_only_by_id["soundcheck_aux_playback_pse_la_dynamics"]
    aux_result = results["soundcheck_aux_playback_pse_la_dynamics"]
    aux_params = aux_result["selected_parameters"]
    aux_contract = aux_result["selected_future_wing_readback_contract"][
        "dynamics_readback"
    ]
    assert aux_result["actual_selected_strategy_id"] == "group_bus_compression_review"
    assert aux_result["actual_method_family"] == "dynamics"
    assert aux_params["wing_aux_dynamics_selected_write_keys"] == [
        "wing_aux_dynamics_on_osc_value",
        "wing_aux_dynamics_threshold_osc_value",
        "wing_aux_dynamics_depth_osc_value",
        "wing_aux_dynamics_fast_osc_value",
    ]
    assert aux_params["wing_aux_dynamics_forbidden_write_keys"] == [
        "wing_aux_dynamics_peak_osc_value"
    ]
    assert aux_params["wing_aux_dynamics_threshold_osc_value"] == -20.3
    assert aux_params["wing_aux_dynamics_depth_osc_value"] == 2.9
    assert aux_params["wing_aux_dynamics_fast_osc_value"] == 0
    assert "wing_aux_dynamics_peak_osc_value" not in aux_params
    assert not any(key.startswith("wing_compressor_") for key in aux_params)
    assert aux_review["actionable_parameter_keys"] == [
        "wing_aux_dynamics_on_osc_value",
        "wing_aux_dynamics_threshold_osc_value",
        "wing_aux_dynamics_depth_osc_value",
        "wing_aux_dynamics_fast_osc_value",
    ]
    assert aux_review["apply_parameter_values"] == {
        "wing_aux_dynamics_depth_osc_value": 2.9,
        "wing_aux_dynamics_fast_osc_value": 0,
        "wing_aux_dynamics_on_osc_value": 1,
        "wing_aux_dynamics_threshold_osc_value": -20.3,
    }
    assert aux_review["review_only_parameter_keys"] == ["target_gain_reduction_db"]
    assert aux_review["review_only_parameter_values"] == {
        "target_gain_reduction_db": 2.9
    }
    assert aux_contract["dynamics_model_readback_required"] is False
    assert aux_contract["wing_compressor_model_readback_required"] is False
    assert aux_contract["wing_aux_dynamics_no_dyn_mdl_path"] is True
    assert aux_contract["read_before_paths"] == [
        "dyn/on",
        "dyn/thr",
        "dyn/depth",
        "dyn/fast",
        "dyn/peak",
        "dyn/ingain",
        "dyn/cpeak",
        "dyn/cmode",
    ]
    assert "dyn/mdl" not in aux_contract["read_before_paths"]
    aux_gr_objective = aux_review["gain_reduction_objective"]
    assert aux_gr_objective[
        "wing_dynamics_gain_reduction_meter_readback_path_templates"
    ] == ["/aux/{aux}/dyn/cpeak", "/aux/{aux}/dyn/cmode"]
    assert aux_gr_objective[
        "wing_dynamics_gain_reduction_meter_fallback_path_templates"
    ] == []
    assert aux_gr_objective["wing_dynamics_gain_reduction_meter_value_type"] == (
        "read_only_aux_compressor_peak_percent_and_mode"
    )
    assert "WING aux PSE/LA cpeak" in aux_gr_objective[
        "wing_dynamics_gain_reduction_objective_operator_check"
    ]
    assert review_only_by_id["master_dynamic_range_bus_compression"][
        "review_only_parameter_keys"
    ] == ["target_gain_reduction_db"]
    assert review_only_by_id["master_dynamic_range_bus_compression"][
        "review_only_parameter_values"
    ]["target_gain_reduction_db"] == results[
        "master_dynamic_range_bus_compression"
    ]["selected_parameters"]["target_gain_reduction_db"]
    assert review_only_by_id["master_dynamic_range_bus_compression"][
        "review_only_parameters_apply_ready"
    ] is False
    assert review_only_by_id["clean_tom_bleed_aware_gate"][
        "review_only_parameter_keys"
    ] == ["post_gate_hysteresis_review_db"]
    assert review_only_by_id["clean_tom_bleed_aware_gate"][
        "review_only_parameter_values"
    ]["post_gate_hysteresis_review_db"] == results["clean_tom_bleed_aware_gate"][
        "selected_parameters"
    ]["post_gate_hysteresis_review_db"]
    assert "no verified gate hysteresis write path" in review_only_by_id[
        "clean_tom_bleed_aware_gate"
    ]["review_only_parameter_reason_by_key"]["post_gate_hysteresis_review_db"]
    gate_hysteresis_objective = review_only_by_id["clean_tom_bleed_aware_gate"][
        "gate_hysteresis_objective"
    ]
    gate_hysteresis_target = results["clean_tom_bleed_aware_gate"][
        "selected_parameters"
    ]["post_gate_hysteresis_review_db"]
    assert gate_hysteresis_objective[
        "wing_gate_hysteresis_review_only_parameter_key"
    ] == "post_gate_hysteresis_review_db"
    assert gate_hysteresis_objective[
        "wing_gate_hysteresis_review_only_value_db"
    ] == gate_hysteresis_target
    assert gate_hysteresis_objective[
        "wing_gate_hysteresis_review_tolerance_db"
    ] == WING_GATE_HYSTERESIS_REVIEW_TOLERANCE_DB
    assert gate_hysteresis_objective[
        "wing_gate_hysteresis_review_acceptance_min_db"
    ] == round(
        max(0.0, gate_hysteresis_target - WING_GATE_HYSTERESIS_REVIEW_TOLERANCE_DB),
        3,
    )
    assert gate_hysteresis_objective[
        "wing_gate_hysteresis_review_acceptance_max_db"
    ] == round(
        min(
            SOUNDCHECK_GATE_EXPANDER_MAX_HYSTERESIS_REVIEW_DB,
            gate_hysteresis_target + WING_GATE_HYSTERESIS_REVIEW_TOLERANCE_DB,
        ),
        3,
    )
    assert gate_hysteresis_objective[
        "wing_gate_hysteresis_review_must_not_exceed_db"
    ] == SOUNDCHECK_GATE_EXPANDER_MAX_HYSTERESIS_REVIEW_DB
    assert gate_hysteresis_objective[
        "wing_gate_hysteresis_review_value_type"
    ] == "operator_review_db"
    assert (
        gate_hysteresis_objective[
            "wing_gate_hysteresis_review_is_not_apply_value"
        ]
        is True
    )
    assert gate_hysteresis_objective[
        "wing_gate_hysteresis_review_control_surface"
    ] == "operator_console_ui_only"
    assert gate_hysteresis_objective[
        "wing_gate_hysteresis_review_path_source"
    ] == "no_verified_wing_fw_3_0_5_osc_path"
    assert gate_hysteresis_objective[
        "wing_gate_hysteresis_review_forbidden_path_suffixes"
    ] == ["gate/hysteresis", "gate/hyst"]
    assert (
        gate_hysteresis_objective[
            "wing_gate_hysteresis_review_requires_audio_recheck"
        ]
        is True
    )
    assert (
        gate_hysteresis_objective[
            "wing_gate_hysteresis_review_requires_tail_chatter_recheck"
        ]
        is True
    )
    assert "do not write a hysteresis OSC value" in gate_hysteresis_objective[
        "wing_gate_hysteresis_review_operator_check"
    ]
    assert review_only_by_id["clean_tom_bleed_aware_gate"][
        "review_only_parameters_apply_ready"
    ] is False
    assert "wing_gate_threshold_osc_value" in review_only_by_id[
        "clean_tom_bleed_aware_gate"
    ]["actionable_parameter_keys"]
    assert review_only_by_id["clean_tom_bleed_aware_gate"][
        "apply_parameter_keys"
    ] == review_only_by_id["clean_tom_bleed_aware_gate"][
        "actionable_parameter_keys"
    ]
    assert set(
        review_only_by_id["clean_tom_bleed_aware_gate"]["apply_parameter_values"]
    ) == set(review_only_by_id["clean_tom_bleed_aware_gate"]["apply_parameter_keys"])
    assert "post_gate_hysteresis_review_db" not in review_only_by_id[
        "clean_tom_bleed_aware_gate"
    ]["apply_parameter_values"]
    assert review_only_by_id["clean_tom_bleed_aware_gate"][
        "apply_parameter_values"
    ]["wing_gate_threshold_osc_value"] == results["clean_tom_bleed_aware_gate"][
        "selected_parameters"
    ]["wing_gate_threshold_osc_value"]
    assert review_only_by_id["clean_tom_bleed_aware_gate"][
        "apply_parameter_values"
    ] == {
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
    _assert_replay_selected_write_key_contract(
        results["clean_tom_bleed_aware_gate"],
        readback_section="gate_expander_readback",
        prefix="wing_gate",
        selected_write_keys=list(WING_GATE_WRITE_PARAMETER_KEYS),
        write_parameter_keys=WING_GATE_WRITE_PARAMETER_KEYS,
        selection_policy=WING_GATE_WRITE_KEY_SELECTION_POLICY,
    )
    _assert_replay_selected_parameter_rollback_operations(
        results["clean_tom_bleed_aware_gate"],
        readback_section="gate_expander_readback",
        operations_field="wing_gate_selected_parameter_rollback_operations",
        selected_write_keys=list(WING_GATE_WRITE_PARAMETER_KEYS),
        contract_marker="read_previous_gate_value_before_write_then_restore_same_path",
    )
    assert review_only_by_id["clean_tom_bleed_aware_gate"][
        "correction_apply_scope"
    ] == "direct_wing_fw_3_0_5_osc_values_only_after_required_readback"
    assert review_only_by_id["clean_tom_bleed_aware_gate"][
        "review_only_parameters_are_apply_values"
    ] is False

    assert results["soundcheck_muddy_vocal"]["actual_selected_strategy_id"] == (
        "low_mid_corrective_cut"
    )
    assert results["soundcheck_muddy_vocal"]["selected_rationale"][
        "strategy_id"
    ] == "low_mid_corrective_cut"
    _assert_replay_tradeoff_wing_osc_summary(
        results["soundcheck_muddy_vocal"]["selected_rationale"],
        native_supported=True,
        support_level="native_osc_path_templates_declared",
    )
    _assert_replay_tradeoff_wing_osc_summary(
        results["soundcheck_muddy_vocal"]["proposed_action_wing_osc_summary"],
        native_supported=True,
        support_level="native_osc_path_templates_declared",
    )
    _assert_replay_tradeoff_wing_osc_summary(
        results["soundcheck_muddy_vocal"][
            "operator_correction_proposed_correction_wing_osc_summary"
        ],
        native_supported=True,
        support_level="native_osc_path_templates_declared",
    )
    assert results["soundcheck_muddy_vocal"]["selected_parameters"][
        "frequency_source"
    ] == "measured_low_mid_peak"
    assert results["soundcheck_muddy_vocal"]["selected_parameters"][
        "frequency_hz"
    ] == 340.0
    _assert_replay_selected_eq_apply_values(
        results["soundcheck_muddy_vocal"],
        {
            "wing_eq_on_osc_value": 1,
            "wing_eq_mix_osc_value": WING_EQ_MIX_BASELINE_PERCENT,
            "wing_eq_band_operations": [
                {
                    "band_readback_required": True,
                    "filter_family": "parametric_band",
                    "filter_type": "peaking",
                    "frequency_hz": 340.0,
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
        },
    )
    _assert_replay_eq_write_key_contract(
        results["soundcheck_muddy_vocal"],
        selected_write_keys=[
            "wing_eq_on_osc_value",
            "wing_eq_mix_osc_value",
            "wing_eq_band_operations",
        ],
    )
    _assert_replay_eq_selected_parameter_rollback_operations(
        results["soundcheck_muddy_vocal"],
        selected_write_keys=[
            "wing_eq_on_osc_value",
            "wing_eq_mix_osc_value",
            "wing_eq_band_operations",
        ],
    )
    muddy_readback = results["soundcheck_muddy_vocal"][
        "selected_future_wing_readback_contract"
    ]
    muddy_wing_osc = muddy_readback["wing_osc_applicability"]
    assert muddy_wing_osc["target_console"] == "Behringer WING Rack"
    assert muddy_wing_osc["firmware"] == "3.0.5"
    assert muddy_wing_osc["protocol"] == "OSC UDP 2223"
    assert muddy_wing_osc["native_osc_supported"] is True
    assert "/ch/{channel}/eq/{band}f" in muddy_wing_osc["native_osc_path_templates"]
    assert "/ch/{channel}/eq/{band}g" in muddy_wing_osc["native_osc_path_templates"]
    assert "/ch/{channel}/eq/{band}q" in muddy_wing_osc["native_osc_path_templates"]
    assert "/ch/{channel}/eq/mix" in muddy_wing_osc["native_osc_path_templates"]
    muddy_eq_readback = muddy_readback["eq_band_readback"]
    assert muddy_eq_readback["eq_model_readback_must_equal"] == WING_EQ_REQUIRED_MODEL
    assert muddy_eq_readback["read_before_paths"] == [
        "eq/on",
        "eq/mdl",
        "eq/mix",
        "eq/{band}f",
        "eq/{band}g",
        "eq/{band}q",
    ]
    assert muddy_eq_readback["wing_eq_mix_osc_path_suffix"] == "eq/mix"
    assert muddy_eq_readback["wing_eq_mix_osc_value"] == WING_EQ_MIX_BASELINE_PERCENT
    assert muddy_eq_readback["wing_eq_mix_min_percent"] == WING_EQ_MIX_MIN_PERCENT
    assert muddy_eq_readback["wing_eq_mix_max_percent"] == (
        WING_EQ_MIX_MAX_PERCENT_BY_SURFACE["channel"]
    )
    assert muddy_eq_readback["wing_eq_value_mapping"]["wing_eq_filter_family"] == (
        "parametric_band"
    )
    assert muddy_eq_readback["wing_eq_value_mapping"]["wing_eq_mix_osc_value"] == (
        WING_EQ_MIX_BASELINE_PERCENT
    )
    assert results["soundcheck_muddy_vocal"]["alternative_tradeoffs"]
    assert all(
        row["why_not_selected"]
        for row in results["soundcheck_muddy_vocal"]["alternative_tradeoffs"]
    )
    assert results["soundcheck_rumble_vocal_hpf"]["actual_selected_strategy_id"] == (
        "rumble_dominant_high_pass_cleanup"
    )
    assert results["soundcheck_rumble_vocal_hpf"]["selected_parameters"][
        "hpf_before_low_mid_cut"
    ] is True
    _assert_replay_eq_selected_parameter_rollback_operations(
        results["soundcheck_rumble_vocal_hpf"],
        selected_write_keys=[
            "wing_eq_on_osc_value",
            "wing_eq_mix_osc_value",
            "wing_eq_low_cut_enabled_osc_value",
            "wing_eq_low_cut_frequency_hz",
            "wing_eq_low_cut_slope_db_per_octave",
        ],
    )
    assert results["soundcheck_plosive_vocal_dynamic_hpf"][
        "actual_selected_strategy_id"
    ] == "vocal_plosive_dynamic_hpf_control"
    assert results["soundcheck_plosive_vocal_dynamic_hpf"][
        "actual_method_family"
    ] == "dynamic_eq"
    assert results["soundcheck_plosive_vocal_dynamic_hpf"]["selected_parameters"][
        "filter_type"
    ] == "dynamic_high_pass"
    assert results["soundcheck_plosive_vocal_dynamic_hpf"]["selected_parameters"][
        "frequency_hz"
    ] == 100.0
    assert results["soundcheck_plosive_vocal_dynamic_hpf"]["selected_parameters"][
        "slope_db_per_octave"
    ] == 12
    assert results["soundcheck_plosive_vocal_dynamic_hpf"]["selected_parameters"][
        "wing_dynamic_eq_model_osc_value"
    ] == WING_DYNAMIC_EQ_GATE_MODEL
    assert results["soundcheck_plosive_vocal_dynamic_hpf"]["selected_parameters"][
        "wing_dynamic_eq_on_osc_value"
    ] == 1
    assert results["soundcheck_plosive_vocal_dynamic_hpf"]["selected_parameters"][
        "wing_dynamic_eq_threshold_osc_value"
    ] == -26.5
    assert results["soundcheck_plosive_vocal_dynamic_hpf"]["selected_parameters"][
        "wing_dynamic_eq_ratio_osc_value"
    ] == 2.0
    assert results["soundcheck_plosive_vocal_dynamic_hpf"]["selected_parameters"][
        "wing_dynamic_eq_attack_osc_value"
    ] == 4.3
    assert results["soundcheck_plosive_vocal_dynamic_hpf"]["selected_parameters"][
        "wing_dynamic_eq_release_osc_value"
    ] == 113.4
    assert results["soundcheck_plosive_vocal_dynamic_hpf"]["selected_parameters"][
        "wing_dynamic_eq_filter_osc_value"
    ] == "HP12"
    assert results["soundcheck_plosive_vocal_dynamic_hpf"]["selected_parameters"][
        "wing_dynamic_eq_gain_osc_value"
    ] == -5.7
    assert results["soundcheck_plosive_vocal_dynamic_hpf"]["selected_parameters"][
        "wing_dynamic_eq_mode_osc_value"
    ] == "high"
    assert results["soundcheck_plosive_vocal_dynamic_hpf"]["selected_parameters"][
        "plosive_risk_score"
    ] == 0.82
    assert results["soundcheck_plosive_vocal_dynamic_hpf"][
        "operator_explanation_sections"
    ]["selection_policy"] == "weighted_strategy_score"
    selected_plosive_params = results["soundcheck_plosive_vocal_dynamic_hpf"][
        "selected_parameters"
    ]
    assert selected_plosive_params["plosive_parameter_source"] == "measured_plosive_burst"
    assert selected_plosive_params["threshold_dbfs"] == -26.5
    assert selected_plosive_params["attack_ms"] == 4.3
    assert selected_plosive_params["preserve_vocal_body_between_plosives"] is True
    plosive_tradeoffs = {
        row["strategy_id"]: row["why_not_selected"]
        for row in results["soundcheck_plosive_vocal_dynamic_hpf"][
            "alternative_tradeoffs"
        ]
    }
    assert plosive_tradeoffs["static_plosive_high_pass_cleanup"].startswith("lower_")
    plosive_readback = results["soundcheck_plosive_vocal_dynamic_hpf"][
        "selected_future_wing_readback_contract"
    ]
    plosive_wing_osc = plosive_readback["wing_osc_applicability"]
    assert plosive_wing_osc["native_osc_supported"] is True
    assert plosive_wing_osc["native_osc_support_level"] == (
        "native_channel_gate_deq_dynamic_eq_osc_path_templates_declared"
    )
    assert plosive_wing_osc["unsupported_strategy_reason"] is None
    assert "/ch/{channel}/gate/filt" in plosive_wing_osc["native_osc_path_templates"]
    assert plosive_readback["dynamic_eq_readback"][
        "wing_rack_fw_3_0_5_native_osc_supported"
    ] is True
    assert "eq_band_readback" not in plosive_readback
    assert "dynamics_readback" not in plosive_readback
    plosive_deq_mapping = plosive_readback["dynamic_eq_readback"][
        "wing_dynamic_eq_value_mapping"
    ]
    assert plosive_deq_mapping["wing_dynamic_eq_on_osc_value"] == 1
    assert plosive_deq_mapping["wing_dynamic_eq_threshold_osc_value"] == -26.5
    assert plosive_deq_mapping["wing_dynamic_eq_ratio_osc_value"] == 2.0
    assert plosive_deq_mapping["wing_dynamic_eq_attack_osc_value"] == 4.3
    assert plosive_deq_mapping["wing_dynamic_eq_release_osc_value"] == 113.4
    assert plosive_deq_mapping["wing_dynamic_eq_filter_osc_value"] == "HP12"
    assert plosive_deq_mapping["wing_dynamic_eq_frequency_osc_value"] == 100.0
    assert plosive_deq_mapping["wing_dynamic_eq_q_osc_value"] == 0.707
    assert plosive_deq_mapping["wing_dynamic_eq_mode_osc_value"] == "high"
    _assert_replay_selected_dynamic_eq_apply_values(
        results["soundcheck_plosive_vocal_dynamic_hpf"],
        {
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
        },
    )
    _assert_replay_selected_write_key_contract(
        results["soundcheck_plosive_vocal_dynamic_hpf"],
        readback_section="dynamic_eq_readback",
        prefix="wing_dynamic_eq",
        selected_write_keys=list(WING_DYNAMIC_EQ_WRITE_PARAMETER_KEYS),
        write_parameter_keys=WING_DYNAMIC_EQ_WRITE_PARAMETER_KEYS,
        selection_policy=WING_DYNAMIC_EQ_WRITE_KEY_SELECTION_POLICY,
    )
    _assert_replay_selected_parameter_rollback_operations(
        results["soundcheck_plosive_vocal_dynamic_hpf"],
        readback_section="dynamic_eq_readback",
        operations_field=(
            "wing_dynamic_eq_selected_parameter_rollback_operations"
        ),
        selected_write_keys=list(WING_DYNAMIC_EQ_WRITE_PARAMETER_KEYS),
        contract_marker=(
            "read_previous_dynamic_eq_value_before_write_then_restore_same_path"
        ),
    )
    plosive_sibilant = results["soundcheck_plosive_sibilant_vocal_de_esser_first"]
    assert plosive_sibilant["actual_selected_strategy_id"] == "vocal_de_esser"
    assert plosive_sibilant["actual_method_family"] == "de_esser"
    assert plosive_sibilant["operator_explanation_sections"][
        "selection_policy"
    ] == "soundcheck_wing_native_de_esser_before_plosive_dynamic_hpf_priority"
    assert plosive_sibilant["selected_parameters"]["processing_chain"] == (
        "ds902_de_esser_first_then_plosive_mic_or_low_cut_recheck"
    )
    assert plosive_sibilant["selected_parameters"][
        "plosive_deesser_gate_slot_conflict"
    ] is True
    assert plosive_sibilant["selected_parameters"][
        "post_de_esser_plosive_recheck_required"
    ] is True
    assert plosive_sibilant["selected_parameters"][
        "wing_gate_slot_plosive_deesser_combo_blocker"
    ] == WING_GATE_SLOT_SINGLE_MODEL_COMBO_BLOCKER
    assert WING_GATE_SLOT_SINGLE_MODEL_COMBO_BLOCKER in plosive_sibilant[
        "candidate_blockers"
    ]["vocal_plosive_dynamic_hpf_control"]
    plosive_sibilant_tradeoffs = {
        row["strategy_id"]: row["why_not_selected"]
        for row in plosive_sibilant["alternative_tradeoffs"]
    }
    assert plosive_sibilant_tradeoffs["vocal_plosive_dynamic_hpf_control"] == (
        "soundcheck_wing_native_de_esser_before_plosive_dynamic_hpf_priority:"
        "ds902_and_dynamic_hpf_share_wing_gate_slot_so_de_esser_recheck_precedes_deq_hpf"
    )
    _assert_replay_selected_de_esser_apply_values(
        plosive_sibilant,
        {
            "wing_de_esser_model_osc_value": WING_DE_ESSER_GATE_MODEL,
            "wing_de_esser_on_osc_value": 1,
            "wing_de_esser_frequency_osc_value": 7700.0,
            "wing_de_esser_range_osc_value": 5.0,
            "wing_de_esser_mode_osc_value": "HF",
        },
    )
    assert not any(
        key.startswith("wing_dynamic_eq_")
        for key in plosive_sibilant["proposal_selected_wing_osc_summary"][
            "wing_osc_apply_parameter_keys"
        ]
    )
    existing_deq = results["soundcheck_plosive_vocal_existing_deq_readback"]
    assert existing_deq["actual_selected_strategy_id"] == (
        "vocal_plosive_dynamic_hpf_control"
    )
    assert existing_deq["actual_method_family"] == "dynamic_eq"
    existing_deq_params = existing_deq["selected_parameters"]
    assert existing_deq_params[
        "wing_dynamic_eq_gate_slot_parameter_apply_ready"
    ] is True
    assert existing_deq_params["wing_dynamic_eq_gate_slot_current_model"] == (
        WING_DYNAMIC_EQ_GATE_MODEL
    )
    assert existing_deq_params[
        "wing_dynamic_eq_model_write_suppressed_by_existing_readback"
    ] is True
    assert "wing_dynamic_eq_model_osc_value" not in existing_deq_params
    existing_deq_expected_apply = {
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
    _assert_replay_selected_dynamic_eq_apply_values(
        existing_deq,
        existing_deq_expected_apply,
    )
    _assert_replay_selected_write_key_contract(
        existing_deq,
        readback_section="dynamic_eq_readback",
        prefix="wing_dynamic_eq",
        selected_write_keys=[
            "wing_dynamic_eq_on_osc_value",
            "wing_dynamic_eq_threshold_osc_value",
            "wing_dynamic_eq_ratio_osc_value",
            "wing_dynamic_eq_attack_osc_value",
            "wing_dynamic_eq_release_osc_value",
            "wing_dynamic_eq_filter_osc_value",
            "wing_dynamic_eq_gain_osc_value",
            "wing_dynamic_eq_frequency_osc_value",
            "wing_dynamic_eq_q_osc_value",
            "wing_dynamic_eq_mode_osc_value",
        ],
        write_parameter_keys=WING_DYNAMIC_EQ_WRITE_PARAMETER_KEYS,
        selection_policy=WING_DYNAMIC_EQ_WRITE_KEY_SELECTION_POLICY,
    )
    _assert_replay_selected_parameter_rollback_operations(
        existing_deq,
        readback_section="dynamic_eq_readback",
        operations_field=(
            "wing_dynamic_eq_selected_parameter_rollback_operations"
        ),
        selected_write_keys=[
            "wing_dynamic_eq_on_osc_value",
            "wing_dynamic_eq_threshold_osc_value",
            "wing_dynamic_eq_ratio_osc_value",
            "wing_dynamic_eq_attack_osc_value",
            "wing_dynamic_eq_release_osc_value",
            "wing_dynamic_eq_filter_osc_value",
            "wing_dynamic_eq_gain_osc_value",
            "wing_dynamic_eq_frequency_osc_value",
            "wing_dynamic_eq_q_osc_value",
            "wing_dynamic_eq_mode_osc_value",
        ],
        contract_marker=(
            "read_previous_dynamic_eq_value_before_write_then_restore_same_path"
        ),
    )
    existing_deq_readback = existing_deq[
        "selected_future_wing_readback_contract"
    ]["dynamic_eq_readback"]
    assert existing_deq_readback["wing_dynamic_eq_model_osc_value"] is None
    assert "gate/mdl" in existing_deq_readback["read_before_paths"]
    assert "gate/mdl" in existing_deq_readback["read_after_paths"]
    assert (
        "wing_dynamic_eq_model_osc_value"
        not in existing_deq["selected_rationale"]["wing_osc_apply_parameter_keys"]
    )
    assert (
        "wing_dynamic_eq_model_osc_value"
        not in existing_deq["proposal_selected_wing_osc_summary"][
            "wing_osc_apply_parameter_keys"
        ]
    )
    assert results["live_plosive_vocal_hold_before_hpf"][
        "actual_selected_strategy_id"
    ] == "manual_plosive_mic_technique_review"
    assert results["live_plosive_vocal_hold_before_hpf"][
        "operator_explanation_sections"
    ]["selection_policy"] == "live_plosive_review_priority"
    live_plosive_hold = results["live_plosive_vocal_hold_before_hpf"]
    assert live_plosive_hold["selected_parameters"]["plosive_parameter_source"] == (
        "measured_plosive_burst"
    )
    assert live_plosive_hold["selected_parameters"]["plosive_frequency_hz"] == 82.0
    assert live_plosive_hold["selected_parameters"][
        "live_plosive_hpf_hold_proposal"
    ] is True
    assert live_plosive_hold["selected_parameters"][
        "live_plosive_hpf_target_surface"
    ] == "channel"
    assert live_plosive_hold["selected_parameters"][
        "live_plosive_previous_hpf_readback_required"
    ] is True
    assert live_plosive_hold["selected_parameters"][
        "live_plosive_no_dynamic_or_static_hpf_without_operator_wing_proof"
    ] is True
    assert "wing_dynamic_eq_on_osc_value" not in live_plosive_hold[
        "selected_parameters"
    ]
    assert "wing_eq_low_cut_enabled_osc_value" not in live_plosive_hold[
        "selected_parameters"
    ]
    live_plosive_readback = live_plosive_hold[
        "selected_future_wing_readback_contract"
    ]["live_plosive_hpf_hold_readback"]
    assert live_plosive_readback["read_only"] is True
    assert live_plosive_readback["live_plosive_hpf_target_surface"] == "channel"
    assert live_plosive_readback[
        "live_plosive_dynamic_hpf_gate_slot_readback_required"
    ] is True
    assert live_plosive_readback[
        "live_plosive_static_low_cut_readback_required"
    ] is True
    assert live_plosive_readback["allowed_live_write_scope_before_wing_proof"] == []
    assert live_plosive_readback["wing_dynamic_hpf_filter_osc_path_suffix"] == (
        "gate/filt"
    )
    assert "/ch/{channel}/gate/filt" in live_plosive_readback[
        "wing_osc_path_templates"
    ]
    assert "/ch/{channel}/flt/lc" in live_plosive_readback[
        "wing_osc_path_templates"
    ]
    assert "/ch/{channel}/flt/lcf" in live_plosive_readback[
        "wing_osc_path_templates"
    ]
    assert "/ch/{channel}/flt/lcs" in live_plosive_readback[
        "wing_osc_path_templates"
    ]
    _assert_replay_analysis_only_has_no_apply_values(live_plosive_hold)
    assert results["live_plosive_vocal_hold_before_hpf"]["candidate_blockers"][
        "vocal_plosive_dynamic_hpf_control"
    ] == ["live_plosive_hpf_requires_operator_review_and_wing_proof"]
    assert results["soundcheck_vocal_noise_floor_expander"][
        "actual_selected_strategy_id"
    ] == "vocal_phrase_noise_downward_expander"
    assert results["soundcheck_vocal_noise_floor_expander"]["selected_parameters"][
        "breath_preserve"
    ] is True
    assert results["soundcheck_vocal_noise_floor_expander"]["selected_parameters"][
        "vocal_noise_parameter_source"
    ] == "measured_phrase_gap_envelope"
    assert results["soundcheck_vocal_noise_floor_expander"]["selected_parameters"][
        "threshold_dbfs"
    ] == -48.8
    assert results["soundcheck_vocal_noise_floor_expander"]["selected_parameters"][
        "hold_ms"
    ] == 140.6
    assert results["soundcheck_vocal_noise_floor_expander"]["selected_parameters"][
        "release_ms"
    ] == 323.0
    assert results[
        "soundcheck_vocal_phrase_noise_before_de_esser_compressor"
    ]["actual_selected_strategy_id"] == "vocal_phrase_noise_downward_expander"
    assert results[
        "soundcheck_vocal_phrase_noise_before_de_esser_compressor"
    ]["operator_explanation_sections"][
        "selection_policy"
    ] == "soundcheck_vocal_phrase_noise_expander_priority"
    assert results[
        "soundcheck_vocal_phrase_noise_before_de_esser_compressor"
    ]["selected_parameters"][
        "vocal_noise_parameter_source"
    ] == "measured_phrase_gap_envelope"
    phrase_noise_tradeoffs = {
        row["strategy_id"]: row["why_not_selected"]
        for row in results[
            "soundcheck_vocal_phrase_noise_before_de_esser_compressor"
        ]["alternative_tradeoffs"]
    }
    assert phrase_noise_tradeoffs["vocal_sibilance_compressor_order_review"] == (
        "soundcheck_vocal_phrase_noise_priority:"
        "clean_phrase_gaps_before_de_essing_or_compression"
    )
    assert phrase_noise_tradeoffs["vocal_de_esser"] == (
        "soundcheck_vocal_phrase_noise_priority:"
        "clean_phrase_gaps_before_de_essing_or_compression"
    )
    assert phrase_noise_tradeoffs["gentle_channel_compression"] == (
        "soundcheck_vocal_phrase_noise_priority:"
        "clean_phrase_gaps_before_de_essing_or_compression"
    )
    assert results["soundcheck_quiet_noisy_vocal_gain_stage_review"][
        "actual_selected_strategy_id"
    ] == "mic_gain_stage_noise_floor_review"
    assert results["soundcheck_quiet_noisy_vocal_gain_stage_review"][
        "actual_method_family"
    ] == "analysis_only"
    assert results["soundcheck_quiet_noisy_vocal_gain_stage_review"][
        "operator_explanation_sections"
    ]["selection_policy"] == "soundcheck_quiet_noisy_vocal_gain_stage_priority"
    assert results["soundcheck_quiet_noisy_vocal_gain_stage_review"][
        "selected_parameters"
    ]["read_only_review"] is True
    assert results["soundcheck_quiet_noisy_vocal_gain_stage_review"][
        "selected_parameters"
    ]["phrase_gap_noise_floor_dbfs"] == -51.5
    assert results["soundcheck_quiet_noisy_vocal_gain_stage_review"][
        "candidate_blockers"
    ]["soundcheck_bounded_gain_up"] == [
        "noise_floor_path_review_should_precede_gain_up"
    ]
    quiet_noisy_tradeoffs = {
        row["strategy_id"]: row["why_not_selected"]
        for row in results["soundcheck_quiet_noisy_vocal_gain_stage_review"][
            "alternative_tradeoffs"
        ]
    }
    assert quiet_noisy_tradeoffs["soundcheck_bounded_gain_up"] == (
        "soundcheck_quiet_noisy_vocal_priority:"
        "noise_path_review_precedes_gain_up"
    )
    assert quiet_noisy_tradeoffs["vocal_phrase_noise_downward_expander"] == (
        "soundcheck_quiet_noisy_vocal_priority:"
        "fix_gain_stage_noise_path_before_expander"
    )
    assert results["live_vocal_noise_floor_review_before_expander"][
        "actual_selected_strategy_id"
    ] == "mic_gain_stage_noise_floor_review"
    assert results["live_vocal_noise_floor_review_before_expander"][
        "actual_method_family"
    ] == "analysis_only"
    assert results["live_vocal_noise_floor_review_before_expander"][
        "operator_explanation_sections"
    ]["selection_policy"] == "live_vocal_noise_review_priority"
    assert results["live_vocal_noise_floor_review_before_expander"][
        "selected_parameters"
    ]["live_vocal_noise_review_priority"] is True
    live_vocal_noise = results["live_vocal_noise_floor_review_before_expander"]
    assert live_vocal_noise["selected_parameters"]["phrase_gap_noise_flatness"] == (
        0.35
    )
    assert live_vocal_noise["selected_parameters"]["phrase_gap_contrast_db"] == 17.0
    assert live_vocal_noise["selected_parameters"]["phrase_gap_median_ms"] == 500.0
    assert live_vocal_noise["selected_parameters"][
        "phrase_gap_coverage_ratio"
    ] == 0.42
    assert live_vocal_noise["selected_parameters"][
        "vocal_noise_parameter_source"
    ] == "measured_phrase_gap_envelope"
    assert live_vocal_noise["selected_parameters"][
        "live_vocal_noise_hold_proposal"
    ] is True
    assert live_vocal_noise["selected_parameters"][
        "live_vocal_noise_target_surface"
    ] == "channel"
    assert live_vocal_noise["selected_parameters"][
        "live_vocal_noise_previous_gate_expander_readback_required"
    ] is True
    assert live_vocal_noise["selected_parameters"][
        "live_vocal_noise_gate_model_readback_required"
    ] is True
    assert live_vocal_noise["selected_parameters"][
        "live_vocal_noise_gate_sidechain_readback_required"
    ] is True
    assert live_vocal_noise["selected_parameters"][
        "live_vocal_noise_no_expander_gate_broad_eq_without_operator_wing_proof"
    ] is True
    assert "wing_gate_on_osc_value" not in live_vocal_noise["selected_parameters"]
    assert "wing_gate_threshold_osc_value" not in live_vocal_noise[
        "selected_parameters"
    ]
    live_vocal_noise_readback = live_vocal_noise[
        "selected_future_wing_readback_contract"
    ]["live_vocal_noise_hold_readback"]
    assert live_vocal_noise_readback["read_only"] is True
    assert live_vocal_noise_readback["live_vocal_noise_target_surface"] == "channel"
    assert live_vocal_noise_readback[
        "live_vocal_noise_gate_sidechain_readback_required"
    ] is True
    assert live_vocal_noise_readback["allowed_live_write_scope_before_wing_proof"] == []
    assert live_vocal_noise_readback["wing_gate_threshold_osc_path_suffix"] == (
        "gate/thr21"
    )
    assert "/ch/{channel}/gate/thr21" in live_vocal_noise_readback[
        "wing_osc_path_templates"
    ]
    assert "/ch/{channel}/gatesc/type" in live_vocal_noise_readback[
        "wing_osc_path_templates"
    ]
    _assert_replay_analysis_only_has_no_apply_values(live_vocal_noise)
    assert results["live_vocal_noise_floor_review_before_expander"][
        "selected_parameters"
    ]["forbidden_without_wing_proof"] == "expander_gate_broad_eq"
    assert results["live_vocal_noise_floor_review_before_expander"][
        "candidate_blockers"
    ]["vocal_phrase_noise_downward_expander"] == [
        "live_vocal_noise_floor_expander_requires_operator_review_and_wing_proof"
    ]
    assert results["soundcheck_line_noise_hum_notch"][
        "actual_selected_strategy_id"
    ] == "mains_hum_notch_stack"
    assert results["soundcheck_line_noise_hum_notch"]["selected_parameters"][
        "frequency_hz"
    ] == 60.0
    assert results["soundcheck_line_noise_hum_notch"]["selected_parameters"][
        "line_noise_notch_parameter_source"
    ] == "measured_line_noise_harmonics"
    assert results["soundcheck_line_noise_hum_notch"]["selected_parameters"][
        "gain_db"
    ] == -6.7
    assert results["soundcheck_line_noise_hum_notch"]["selected_parameters"][
        "q"
    ] == 14.0
    assert results["soundcheck_line_noise_hum_notch"]["selected_parameters"][
        "harmonic_notches"
    ][1]["gain_db"] == -5.7
    assert results["soundcheck_line_noise_hum_notch"]["selected_parameters"][
        "wing_eq_band_capacity"
    ] == 4
    assert results["soundcheck_line_noise_hum_notch"]["selected_parameters"][
        "wing_required_eq_band_count"
    ] == 3
    assert results["soundcheck_line_noise_hum_notch"][
        "selected_future_wing_readback_contract"
    ]["eq_band_readback"]["required_band_count"] == 3
    line_noise_low_end = results["soundcheck_line_noise_before_low_end_resonance_eq"]
    line_noise_low_end_params = line_noise_low_end["selected_parameters"]
    assert line_noise_low_end["actual_selected_strategy_id"] == "mains_hum_notch_stack"
    assert line_noise_low_end["operator_explanation_sections"][
        "selection_policy"
    ] == "soundcheck_line_noise_low_end_resonance_priority"
    assert line_noise_low_end_params["line_noise_low_end_resonance_conflict"] is True
    assert line_noise_low_end_params["line_noise_low_end_resonance_priority"] is True
    assert line_noise_low_end_params["frequency_hz"] == 60.0
    assert line_noise_low_end_params["low_end_resonance_frequency_hz"] == 62.0
    assert line_noise_low_end_params["wing_required_eq_band_count"] == 3
    assert line_noise_low_end["selected_future_wing_readback_contract"][
        "eq_band_readback"
    ]["apply_only_native_harmonic_notches"] is True
    line_noise_low_end_tradeoffs = {
        row["strategy_id"]: row["why_not_selected"]
        for row in line_noise_low_end["alternative_tradeoffs"]
    }
    assert line_noise_low_end_tradeoffs["low_end_dynamic_resonance_control"] == (
        "soundcheck_line_noise_low_end_resonance_priority:"
        "measured_hum_notch_precedes_low_end_tone_eq"
    )
    assert line_noise_low_end_tradeoffs["low_end_static_pocket_cut"] == (
        "soundcheck_line_noise_low_end_resonance_priority:"
        "measured_hum_notch_precedes_low_end_tone_eq"
    )
    assert results["soundcheck_bass_di_line_noise_before_gain_up"][
        "actual_selected_strategy_id"
    ] == "mains_hum_notch_stack"
    assert results["soundcheck_bass_di_line_noise_before_gain_up"][
        "operator_explanation_sections"
    ]["selection_policy"] == "soundcheck_line_noise_before_gain_up_priority"
    assert results["soundcheck_bass_di_line_noise_before_gain_up"][
        "selected_parameters"
    ]["frequency_hz"] == 50.0
    assert results["soundcheck_bass_di_line_noise_before_gain_up"][
        "selected_parameters"
    ]["line_noise_harmonic_count"] == 4
    assert results["soundcheck_bass_di_line_noise_before_gain_up"][
        "selected_parameters"
    ]["measured_line_noise_harmonic_count"] == 4
    assert len(
        results["soundcheck_bass_di_line_noise_before_gain_up"][
            "selected_parameters"
        ]["harmonic_notches"]
    ) == 4
    assert results["soundcheck_bass_di_line_noise_before_gain_up"][
        "selected_parameters"
    ]["harmonic_notches"][3] == {
        "frequency_hz": 200.0,
        "gain_db": -3.9,
        "q": 7.0,
        "filter_type": "peaking",
        "prominence_db": 8.0,
    }
    assert results["soundcheck_bass_di_line_noise_before_gain_up"][
        "selected_parameters"
    ]["wing_eq_band_capacity"] == 4
    assert results["soundcheck_bass_di_line_noise_before_gain_up"][
        "selected_parameters"
    ]["wing_required_eq_band_count"] == 4
    assert results["soundcheck_bass_di_line_noise_before_gain_up"][
        "selected_parameters"
    ]["wing_eq_band_overflow_review_required"] is False
    bass_eq_readback = results["soundcheck_bass_di_line_noise_before_gain_up"][
        "selected_future_wing_readback_contract"
    ]["eq_band_readback"]
    assert bass_eq_readback["wing_native_eq_band_capacity"] == 4
    assert bass_eq_readback["required_band_count"] == 4
    assert bass_eq_readback["native_notch_band_numbers"] == [1, 2, 3, 4]
    assert results["soundcheck_bass_di_line_noise_before_gain_up"][
        "candidate_blockers"
    ]["soundcheck_bounded_gain_up"] == [
        "line_noise_path_review_should_precede_gain_up"
    ]
    bass_hum_tradeoffs = {
        row["strategy_id"]: row["why_not_selected"]
        for row in results["soundcheck_bass_di_line_noise_before_gain_up"][
            "alternative_tradeoffs"
        ]
    }
    assert bass_hum_tradeoffs["soundcheck_bounded_gain_up"] == (
        "soundcheck_line_noise_before_gain_up_priority:"
        "remove_or_verify_hum_before_adding_level"
    )
    assert results["live_line_noise_physical_path_before_notch_or_dynamics"][
        "actual_selected_strategy_id"
    ] == "di_ground_lift_line_check"
    assert results["live_line_noise_physical_path_before_notch_or_dynamics"][
        "operator_explanation_sections"
    ]["selection_policy"] == "live_line_noise_physical_path_review_priority"
    assert results["live_line_noise_physical_path_before_notch_or_dynamics"][
        "selected_parameters"
    ]["live_line_noise_review_priority"] is True
    assert results["live_line_noise_physical_path_before_notch_or_dynamics"][
        "selected_parameters"
    ]["forbidden_without_wing_proof"] == "live_line_noise_notch_hpf_eq"
    live_line_noise_params = results[
        "live_line_noise_physical_path_before_notch_or_dynamics"
    ]["selected_parameters"]
    assert live_line_noise_params["live_line_noise_hold_proposal"] is True
    assert live_line_noise_params[
        "live_line_noise_previous_eq_readback_required"
    ] is True
    assert live_line_noise_params[
        "live_line_noise_previous_low_cut_readback_required"
    ] is True
    assert live_line_noise_params[
        "live_line_noise_previous_dynamics_readback_required"
    ] is True
    assert live_line_noise_params[
        "live_line_noise_no_notch_hpf_dynamics_without_operator_wing_proof"
    ] is True
    live_line_noise_readback = results[
        "live_line_noise_physical_path_before_notch_or_dynamics"
    ]["selected_future_wing_readback_contract"]["live_line_noise_hold_readback"]
    assert live_line_noise_readback["read_only"] is True
    assert live_line_noise_readback[
        "wing_rack_fw_3_0_5_native_osc_supported"
    ] is True
    assert "/ch/{channel}/eq/{band}f" in live_line_noise_readback[
        "wing_eq_notch_osc_path_templates"
    ]
    assert "/ch/{channel}/flt/lc" in live_line_noise_readback[
        "wing_static_low_cut_osc_path_templates"
    ]
    assert "/ch/{channel}/dyn/on" in live_line_noise_readback[
        "wing_dynamics_osc_path_templates"
    ]
    assert "eq/{band}g" in live_line_noise_readback["read_before_paths"]
    assert "flt/lcf" in live_line_noise_readback["read_before_paths"]
    assert "dyn/thr24" in live_line_noise_readback["read_before_paths"]
    assert live_line_noise_readback["allowed_live_write_scope_before_wing_proof"] == []
    _assert_replay_analysis_only_has_no_apply_values(
        results["live_line_noise_physical_path_before_notch_or_dynamics"]
    )
    assert results["live_line_noise_physical_path_before_notch_or_dynamics"][
        "candidate_blockers"
    ]["mains_hum_notch_stack"] == [
        "live_line_noise_eq_requires_operator_review_and_wing_proof",
        "live_eq_cut_exceeds_conservative_limit",
    ]
    live_hum_tradeoffs = {
        row["strategy_id"]: row["why_not_selected"]
        for row in results["live_line_noise_physical_path_before_notch_or_dynamics"][
            "alternative_tradeoffs"
        ]
    }
    assert live_hum_tradeoffs["mains_hum_notch_stack"] == (
        "live_line_noise_review_priority:"
        "physical_path_check_precedes_live_notch_or_hpf"
    )
    assert live_hum_tradeoffs["manual_level_ride_before_compression"] == (
        "live_line_noise_review_priority:"
        "hum_path_check_precedes_compression_or_level_ride"
    )
    assert results["live_vocal_plosive_line_noise_mic_review_before_hum"][
        "actual_selected_strategy_id"
    ] == "manual_plosive_mic_technique_review"
    assert results["live_vocal_plosive_line_noise_mic_review_before_hum"][
        "operator_explanation_sections"
    ]["selection_policy"] == "live_plosive_review_priority"
    assert results["live_vocal_plosive_line_noise_mic_review_before_hum"][
        "selected_parameters"
    ]["live_plosive_review_priority"] is True
    plosive_hum_hold = results["live_vocal_plosive_line_noise_mic_review_before_hum"]
    assert plosive_hum_hold["selected_parameters"][
        "live_plosive_hpf_hold_proposal"
    ] is True
    assert plosive_hum_hold["selected_parameters"][
        "live_plosive_hpf_target_surface"
    ] == "channel"
    assert plosive_hum_hold["selected_parameters"][
        "live_plosive_no_dynamic_or_static_hpf_without_operator_wing_proof"
    ] is True
    plosive_hum_readback = plosive_hum_hold[
        "selected_future_wing_readback_contract"
    ]["live_plosive_hpf_hold_readback"]
    assert plosive_hum_readback["read_only"] is True
    assert "/ch/{channel}/gate/filt" in plosive_hum_readback[
        "wing_osc_path_templates"
    ]
    assert "/ch/{channel}/flt/lc" in plosive_hum_readback[
        "wing_osc_path_templates"
    ]
    assert plosive_hum_readback["allowed_live_write_scope_before_wing_proof"] == []
    _assert_replay_analysis_only_has_no_apply_values(plosive_hum_hold)
    assert results["live_vocal_plosive_line_noise_mic_review_before_hum"][
        "candidate_blockers"
    ]["mains_hum_notch_stack"] == [
        "live_line_noise_eq_requires_operator_review_and_wing_proof",
        "live_eq_cut_exceeds_conservative_limit",
    ]
    plosive_hum_tradeoffs = {
        row["strategy_id"]: row["why_not_selected"]
        for row in results["live_vocal_plosive_line_noise_mic_review_before_hum"][
            "alternative_tradeoffs"
        ]
    }
    assert plosive_hum_tradeoffs["di_ground_lift_line_check"] == (
        "live_plosive_review_priority:"
        "mic_technique_review_precedes_hum_path_when_plosives_are_active"
    )
    assert plosive_hum_tradeoffs["mains_hum_notch_stack"] == (
        "live_plosive_review_priority:"
        "mic_technique_review_precedes_hum_path_when_plosives_are_active"
    )
    assert results["soundcheck_dynamic_muddy_vocal"][
        "actual_selected_strategy_id"
    ] == "vocal_dynamic_low_mid_de_mud"
    assert results["soundcheck_dynamic_muddy_vocal"][
        "actual_method_family"
    ] == "dynamic_eq"
    assert results["soundcheck_dynamic_muddy_vocal"][
        "selected_rationale"
    ]["selection_policy"] == "weighted_strategy_score"
    assert results["soundcheck_dynamic_muddy_vocal"]["selected_parameters"][
        "frequency_source"
    ] == "measured_low_mid_bloom_peak"
    assert results["soundcheck_dynamic_muddy_vocal"]["selected_parameters"][
        "low_mid_bloom_frequency_hz"
    ] == 310.0
    assert results["soundcheck_dynamic_muddy_vocal"]["selected_parameters"][
        "gain_db"
    ] == -0.6
    assert results["soundcheck_dynamic_muddy_vocal"]["selected_parameters"][
        "wing_dynamic_eq_gain_osc_value"
    ] == -3.5
    assert results["soundcheck_dynamic_muddy_vocal"]["selected_parameters"][
        "wing_dynamic_eq_model_osc_value"
    ] == WING_DYNAMIC_EQ_GATE_MODEL
    assert results["soundcheck_dynamic_muddy_vocal"]["selected_parameters"][
        "muddy_dynamic_parameter_source"
    ] == "measured_low_mid_bloom"
    assert results["live_dynamic_muddy_vocal_static_low_mid_priority"][
        "actual_selected_strategy_id"
    ] == "low_mid_corrective_cut"
    assert results["live_dynamic_muddy_vocal_static_low_mid_priority"][
        "operator_explanation_sections"
    ]["selection_policy"] == "live_vocal_low_mid_static_cut_priority"
    assert results["live_dynamic_muddy_vocal_static_low_mid_priority"][
        "selected_parameters"
    ]["gain_db"] == -2.6
    live_eq_params = results["live_dynamic_muddy_vocal_static_low_mid_priority"][
        "selected_parameters"
    ]
    assert live_eq_params["live_eq_conservative_cut_proposal"] is True
    assert live_eq_params["live_eq_cut_only_required"] is True
    assert live_eq_params["live_eq_max_cut_db"] == LIVE_EQ_MAX_CUT_DB
    assert live_eq_params["live_eq_max_boost_db"] == LIVE_EQ_MAX_BOOST_DB
    assert live_eq_params["live_eq_protected_band_readback_required"] is True
    assert live_eq_params["live_eq_model_readback_required"] is True
    assert live_eq_params["live_eq_global_on_mix_write_forbidden"] is True
    assert live_eq_params["live_eq_global_on_mix_readback_required"] is True
    assert live_eq_params["live_eq_band_apply_requires_existing_eq_enabled"] is True
    assert live_eq_params["live_eq_band_apply_requires_mix_readback"] is True
    assert live_eq_params["live_eq_band_operation_write_forbidden"] is True
    assert live_eq_params["live_eq_band_operation_readback_required"] is True
    assert live_eq_params["live_eq_requires_concrete_band_readback"] is True
    assert (
        live_eq_params[
            "live_eq_band_operation_apply_blocked_without_concrete_band_readback"
        ]
        is True
    )
    assert live_eq_params["live_eq_band_operation_review_only_reason"] == (
        "live_eq_must_resolve_concrete_unprotected_band_before_parameter_apply"
    )
    assert live_eq_params["live_eq_on_review_only_value"] == 1
    assert live_eq_params["live_eq_mix_review_only_value"] == (
        WING_EQ_MIX_BASELINE_PERCENT
    )
    assert live_eq_params["live_eq_global_on_mix_review_only_reason"] == (
        "live_eq_must_read_existing_eq_on_and_mix_before_band_parameter_apply"
    )
    assert "wing_eq_on_osc_value" not in live_eq_params
    assert "wing_eq_mix_osc_value" not in live_eq_params
    assert "wing_eq_band_operations" not in live_eq_params
    expected_live_eq_review_operations = [
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
    assert (
        live_eq_params["live_eq_band_operation_review_only_values"]
        == expected_live_eq_review_operations
    )
    assert live_eq_params["live_eq_feedback_recheck_after_apply_required"] is True
    assert live_eq_params["live_eq_no_boost_without_operator_wing_proof"] is True
    assert live_eq_params["live_eq_operator_rehearsal_required"] is True
    assert live_eq_params["live_eq_live_promotion_status"] == (
        "blocked_until_real_wing_eq_readback_feedback_recheck_and_rollback"
    )
    assert live_eq_params["live_eq_live_apply_blocked_without_wing_proof"] is True
    assert results["live_dynamic_muddy_vocal_static_low_mid_priority"][
        "candidate_blockers"
    ]["vocal_dynamic_low_mid_de_mud"] == [
        "live_vocal_dynamic_low_mid_requires_operator_review"
    ]
    _assert_replay_selected_eq_apply_values(
        results["live_dynamic_muddy_vocal_static_low_mid_priority"],
        {},
    )
    _assert_replay_eq_write_key_contract(
        results["live_dynamic_muddy_vocal_static_low_mid_priority"],
        selected_write_keys=[],
    )
    live_hot_bus = results["live_muddy_vocal_hot_summed_bus_headroom_pullback"]
    live_hot_bus_params = live_hot_bus["selected_parameters"]
    assert live_hot_bus["actual_selected_strategy_id"] == (
        "live_summed_bus_headroom_pullback_before_source_processing"
    )
    assert live_hot_bus["actual_method_family"] == "gain_fader"
    assert live_hot_bus["operator_explanation_sections"]["selection_policy"] == (
        "live_summed_bus_headroom_pullback_priority"
    )
    assert live_hot_bus_params["bus_delta_db"] == -1.1
    assert live_hot_bus_params["delta_db"] == -1.1
    assert live_hot_bus_params["bounded_delta_db"] == 1.1
    assert live_hot_bus_params["source_delta_db"] == 0.0
    assert live_hot_bus_params["summed_bus_target_id"] == "bus:5"
    assert live_hot_bus_params["wing_osc_original_target_id"] == "ch:5"
    assert live_hot_bus_params["wing_osc_target_id_override"] == "bus:5"
    assert live_hot_bus_params["wing_osc_effective_target_id"] == "bus:5"
    assert live_hot_bus_params["live_summed_bus_pullback_only_proposal"] is True
    assert (
        live_hot_bus_params["live_source_processing_deferred_until_bus_recheck"]
        is True
    )
    assert live_hot_bus_params["live_summed_bus_true_peak_recheck_required"] is True
    assert live_hot_bus_params["live_summed_bus_feedback_recheck_required"] is True
    assert live_hot_bus_params["live_stepped_change_required"] is True
    assert live_hot_bus_params["live_readback_after_each_step_required"] is True
    assert live_hot_bus_params["live_max_single_step_db"] == 0.75
    _assert_replay_gain_delta_mapping(
        live_hot_bus,
        source="bus_delta_db",
        expected_delta=-1.1,
        path_prefix="/bus/{bus}",
    )
    _assert_replay_selected_gain_delta_apply_values(
        live_hot_bus,
        expected_delta=-1.1,
    )
    assert live_hot_bus["selected_future_wing_readback_contract"][
        "wing_osc_applicability"
    ]["target_surface_kind"] == "bus"
    assert live_hot_bus["proposal_selected_wing_osc_summary"][
        "wing_osc_native_supported"
    ] is True
    assert live_hot_bus["proposal_selected_wing_osc_summary"][
        "wing_osc_candidate_apply_status"
    ] == "wing_native_osc_candidate_requires_absolute_value_readback_before_apply"
    assert live_hot_bus["proposal_authority"]["live_apply_allowed"] is False
    assert live_hot_bus["proposal_authority"]["live_authority_granted"] is False
    assert live_hot_bus["proposal_authority"]["live_write"] is False
    live_hot_bus_tradeoffs = {
        row["strategy_id"]: row["why_not_selected"]
        for row in live_hot_bus["alternative_tradeoffs"]
    }
    assert live_hot_bus_tradeoffs["low_mid_corrective_cut"] == (
        "summed_bus_headroom_priority:"
        "restore_bus_headroom_before_source_processing"
    )
    assert results["soundcheck_dynamic_muddy_guitar_low_mid_bloom"][
        "actual_selected_strategy_id"
    ] == "instrument_dynamic_low_mid_de_mud"
    assert results["soundcheck_dynamic_muddy_guitar_low_mid_bloom"][
        "actual_method_family"
    ] == "dynamic_eq"
    assert results["soundcheck_dynamic_muddy_guitar_low_mid_bloom"][
        "selected_parameters"
    ]["frequency_source"] == "measured_low_mid_bloom_peak"
    assert results["soundcheck_dynamic_muddy_guitar_low_mid_bloom"][
        "selected_parameters"
    ]["low_mid_bloom_frequency_hz"] == 295.0
    assert results["soundcheck_dynamic_muddy_guitar_low_mid_bloom"][
        "selected_rationale"
    ]["selection_policy"] == "soundcheck_instrument_low_mid_bloom_priority"
    assert results["soundcheck_dynamic_muddy_guitar_low_mid_bloom"][
        "selected_parameters"
    ][
        "instrument_low_mid_parameter_source"
    ] == "measured_low_mid_bloom"
    assert results["soundcheck_dynamic_muddy_guitar_low_mid_bloom"][
        "selected_parameters"
    ]["wing_dynamic_eq_gain_osc_value"] == -3.2
    assert results["soundcheck_dynamic_muddy_guitar_low_mid_bloom"][
        "selected_parameters"
    ]["wing_dynamic_eq_model_osc_value"] == WING_DYNAMIC_EQ_GATE_MODEL
    assert results["live_dynamic_muddy_guitar_static_low_mid_priority"][
        "actual_selected_strategy_id"
    ] == "low_mid_corrective_cut"
    assert results["live_dynamic_muddy_guitar_static_low_mid_priority"][
        "selected_rationale"
    ]["selection_policy"] == "live_instrument_low_mid_static_cut_priority"
    assert results["live_dynamic_muddy_guitar_static_low_mid_priority"][
        "selected_parameters"
    ]["frequency_source"] == "measured_low_mid_bloom_peak"
    assert results["live_dynamic_muddy_guitar_static_low_mid_priority"][
        "candidate_blockers"
    ]["instrument_dynamic_low_mid_de_mud"] == [
        "live_dynamic_instrument_low_mid_requires_operator_review"
    ]
    assert results["live_dynamic_muddy_guitar_static_low_mid_priority"][
        "candidate_parameters"
    ]["instrument_dynamic_low_mid_de_mud"]["low_mid_bloom_frequency_hz"] == 295.0
    assert results["soundcheck_fx_wash_muddy_vocal_cleanup"][
        "actual_selected_strategy_id"
    ] == "reduce_fx_send_or_return"
    assert results["soundcheck_fx_wash_muddy_vocal_cleanup"][
        "operator_explanation_sections"
    ]["selection_policy"] == "soundcheck_wing_native_osc_fx_level_pullback_priority"
    fx_wash_params = results["soundcheck_fx_wash_muddy_vocal_cleanup"][
        "selected_parameters"
    ]
    assert fx_wash_params["send_delta_db"] < 0.0
    assert fx_wash_params["fx_level_pullback_parameter_source"] == (
        "fx_wet_decay_score_fallback"
    )
    assert fx_wash_params["native_fx_level_pullback_db"] == 4.0
    assert fx_wash_params["native_fx_level_pullback_legacy_db"] == 4.0
    assert fx_wash_params["measured_tail_decay_s"] == 0.0
    assert fx_wash_params["measured_delay_repeat_ms"] == 0.0
    assert fx_wash_params["post_pullback_time_edit_review_required"] is True
    _assert_replay_fx_send_level_mapping(
        results["soundcheck_fx_wash_muddy_vocal_cleanup"],
        expected_delta=fx_wash_params["send_delta_db"],
        target_surface="channel",
        source_binding_required=False,
    )
    _assert_replay_selected_fx_apply_values(
        results["soundcheck_fx_wash_muddy_vocal_cleanup"],
        {"wing_fx_send_level_delta_db": -4.0},
    )
    fx_wash_tradeoffs = {
        item["strategy_id"]: item["why_not_selected"]
        for item in results["soundcheck_fx_wash_muddy_vocal_cleanup"][
            "alternative_tradeoffs"
        ]
    }
    assert fx_wash_tradeoffs["shorten_fx_decay_filter"] == (
        "soundcheck_wing_native_osc_priority:"
        "native_fx_level_pullback_has_verified_wing_fw_3_0_5_osc_path_"
        "alternative_requires_model_mapping"
    )
    _assert_replay_fx_model_specific_guard(
        results["soundcheck_fx_wash_muddy_vocal_cleanup"],
        "shorten_fx_decay_filter",
        {
            "decay_s",
            "pre_delay_ms",
            "return_hpf_hz",
            "return_lpf_hz",
            "delay_feedback_percent",
        },
    )
    assert results["live_fx_wash_muddy_vocal_native_pullback_before_eq"][
        "actual_selected_strategy_id"
    ] == "reduce_fx_send_or_return"
    assert results["live_fx_wash_muddy_vocal_native_pullback_before_eq"][
        "operator_explanation_sections"
    ]["selection_policy"] == "live_fx_wash_cleanup_priority"
    live_fx_wash_params = results[
        "live_fx_wash_muddy_vocal_native_pullback_before_eq"
    ]["selected_parameters"]
    assert live_fx_wash_params["send_delta_db"] == -3.0
    assert live_fx_wash_params["return_delta_db"] == 0.0
    assert live_fx_wash_params["native_fx_level_pullback_db"] == 3.0
    assert live_fx_wash_params["native_fx_level_pullback_legacy_db"] == 4.0
    assert live_fx_wash_params["live_fx_level_pullback_conservative_cap_db"] == 3.0
    assert live_fx_wash_params["live_fx_pullback_only_proposal"] is True
    assert live_fx_wash_params["live_fx_level_reduction_only_required"] is True
    assert live_fx_wash_params["live_fx_no_model_or_time_parameter_write_required"] is True
    assert live_fx_wash_params["live_fx_live_apply_blocked_without_wing_proof"] is True
    assert live_fx_wash_params["live_stepped_change_required"] is True
    assert live_fx_wash_params["live_readback_after_each_step_required"] is True
    _assert_replay_fx_send_level_mapping(
        results["live_fx_wash_muddy_vocal_native_pullback_before_eq"],
        expected_delta=-3.0,
        target_surface="channel",
        source_binding_required=False,
    )
    _assert_replay_selected_fx_apply_values(
        results["live_fx_wash_muddy_vocal_native_pullback_before_eq"],
        {"wing_fx_send_level_delta_db": -3.0},
    )
    live_fx_wash_tradeoffs = {
        item["strategy_id"]: item["why_not_selected"]
        for item in results["live_fx_wash_muddy_vocal_native_pullback_before_eq"][
            "alternative_tradeoffs"
        ]
    }
    assert live_fx_wash_tradeoffs["low_mid_corrective_cut"] == (
        "live_fx_wash_cleanup_priority:reduce_wet_level_before_channel_tone_cut"
    )
    assert live_fx_wash_tradeoffs["vocal_dynamic_low_mid_de_mud"] == (
        "live_fx_wash_cleanup_priority:reduce_wet_level_before_channel_tone_cut"
    )
    assert results["live_operator_owned_fx_holds_muddy_vocal"][
        "actual_selected_strategy_id"
    ] == "hold_live_operator_owned_fx_for_review"
    assert results["live_operator_owned_fx_holds_muddy_vocal"][
        "actual_method_family"
    ] == "analysis_only"
    assert results["live_operator_owned_fx_holds_muddy_vocal"][
        "operator_explanation_sections"
    ]["selection_policy"] == "live_operator_owned_fx_review_priority"
    live_owned_fx_hold = results["live_operator_owned_fx_holds_muddy_vocal"]
    live_owned_fx_params = live_owned_fx_hold["selected_parameters"]
    assert live_owned_fx_params["dsp_change"] == "hold"
    assert live_owned_fx_params["blocked_fx_strategy_ids"] == [
        "reduce_fx_send_or_return",
        "shorten_fx_decay_filter",
        "increase_fx_depth",
    ]
    assert live_owned_fx_params["blocked_fx_operator_owned_blockers"] == [
        "wing_fx_level_operator_owned_edit_requires_operator_review"
    ]
    assert live_owned_fx_params["blocked_fx_send_delta_db"] == -3.0
    assert live_owned_fx_params["blocked_fx_return_delta_db"] == 0.0
    assert live_owned_fx_params["wing_fx_level_current_operator_owned"] is True
    assert (
        live_owned_fx_params["wing_fx_level_operator_owned_edit_requires_review"]
        is True
    )
    assert live_owned_fx_params["wing_fx_level_operator_owned_blocker"] == (
        "wing_fx_level_operator_owned_edit_requires_operator_review"
    )
    assert (
        live_owned_fx_params["wing_rack_fw305_verified_osc_mapping_required"]
        is True
    )
    assert (
        live_owned_fx_params["wing_rack_fw305_fx_send_return_readback_required"]
        is True
    )
    assert live_owned_fx_params["wing_rack_fw305_rollback_required"] is True
    assert live_owned_fx_params["live_authority_granted"] is False
    assert live_owned_fx_params["live_write"] is False
    assert live_owned_fx_params["readiness_closure_allowed"] is False
    assert live_owned_fx_hold["candidate_blockers"]["reduce_fx_send_or_return"] == [
        "wing_fx_level_operator_owned_edit_requires_operator_review"
    ]
    _assert_replay_analysis_only_has_no_apply_values(live_owned_fx_hold)
    live_owned_fx_tradeoffs = {
        item["strategy_id"]: item["why_not_selected"]
        for item in live_owned_fx_hold["alternative_tradeoffs"]
    }
    assert live_owned_fx_tradeoffs["reduce_fx_send_or_return"] == (
        "live_operator_owned_fx_review_priority:"
        "manual_fx_level_or_model_owner_review_precedes_live_fx_change"
    )
    assert live_owned_fx_tradeoffs["low_mid_corrective_cut"] == (
        "live_operator_owned_fx_review_priority:"
        "manual_fx_level_review_precedes_channel_tone_cut"
    )
    assert results["soundcheck_dull_vocal_presence_lift"][
        "actual_selected_strategy_id"
    ] == "soundcheck_presence_air_lift"
    assert results["soundcheck_dull_vocal_presence_lift"]["selected_parameters"][
        "frequency_hz"
    ] == 3150.0
    assert results["soundcheck_dull_vocal_presence_lift"]["selected_parameters"][
        "frequency_source"
    ] == "measured_presence_air_deficit"
    assert results["soundcheck_dull_vocal_presence_lift"]["selected_parameters"][
        "presence_lift_bandwidth_hz"
    ] == 3900.0
    soundcheck_boost_params = results["soundcheck_dull_vocal_presence_lift"][
        "selected_parameters"
    ]
    assert soundcheck_boost_params["soundcheck_eq_bounded_boost_proposal"] is True
    assert (
        soundcheck_boost_params["soundcheck_eq_boost_max_db"]
        == SOUNDCHECK_EQ_MAX_BOOST_DB
    )
    assert (
        soundcheck_boost_params["soundcheck_eq_boost_min_headroom_margin_db"]
        == SOUNDCHECK_EQ_MIN_HEADROOM_MARGIN_DB
    )
    assert (
        soundcheck_boost_params["soundcheck_eq_boost_required_true_peak_headroom_db"]
        == 3.3
    )
    assert (
        soundcheck_boost_params[
            "soundcheck_eq_boost_required_summed_bus_true_peak_headroom_db"
        ]
        == 3.3
    )
    assert soundcheck_boost_params["soundcheck_eq_boost_readback_required"] is True
    assert soundcheck_boost_params["soundcheck_eq_boost_rollback_ready"] is True
    assert (
        soundcheck_boost_params["soundcheck_eq_live_apply_blocked_without_wing_proof"]
        is True
    )
    _assert_replay_eq_selected_parameter_rollback_operations(
        results["soundcheck_dull_vocal_presence_lift"],
        selected_write_keys=[
            "wing_eq_on_osc_value",
            "wing_eq_mix_osc_value",
            "wing_eq_high_shelf_gain_db",
            "wing_eq_high_shelf_frequency_hz",
            "wing_eq_high_shelf_q",
            "wing_eq_high_shelf_type_osc_value",
        ],
    )
    assert results["soundcheck_dull_vocal_monitor_feedback_hold"][
        "actual_selected_strategy_id"
    ] == "hold_presence_boost_for_live_safety"
    assert results["soundcheck_dull_vocal_monitor_feedback_hold"][
        "operator_explanation_sections"
    ]["selection_policy"] == "soundcheck_monitor_feedback_presence_boost_hold_priority"
    assert results["soundcheck_dull_vocal_monitor_feedback_hold"][
        "selected_parameters"
    ]["monitor_path_review_required"] is True
    assert results["soundcheck_dull_vocal_monitor_feedback_hold"][
        "candidate_blockers"
    ]["soundcheck_presence_air_lift"] == [
        "monitor_feedback_risk_blocks_presence_boost"
    ]
    assert results["soundcheck_dull_vocal_elevated_feedback_path_hold"][
        "actual_selected_strategy_id"
    ] == "hold_presence_boost_for_live_safety"
    assert results["soundcheck_dull_vocal_elevated_feedback_path_hold"][
        "operator_explanation_sections"
    ]["selection_policy"] == "soundcheck_presence_boost_feedback_path_hold_priority"
    assert results["soundcheck_dull_vocal_elevated_feedback_path_hold"][
        "selected_parameters"
    ]["feedback_path_review_required"] is True
    assert results["soundcheck_dull_vocal_elevated_feedback_path_hold"][
        "selected_parameters"
    ]["monitor_path_review_required"] is False
    assert results["soundcheck_dull_vocal_elevated_feedback_path_hold"][
        "candidate_blockers"
    ]["soundcheck_presence_air_lift"] == [
        "feedback_path_review_blocks_presence_boost"
    ]
    elevated_presence_tradeoffs = {
        row["strategy_id"]: row["why_not_selected"]
        for row in results["soundcheck_dull_vocal_elevated_feedback_path_hold"][
            "alternative_tradeoffs"
        ]
    }
    assert elevated_presence_tradeoffs["soundcheck_presence_air_lift"] == (
        "presence_boost_feedback_path_priority:"
        "check_feedback_path_before_air_lift"
    )
    assert results["soundcheck_dull_vocal_summed_bus_headroom_hold"][
        "actual_selected_strategy_id"
    ] == "hold_presence_boost_for_live_safety"
    assert results["soundcheck_dull_vocal_summed_bus_headroom_hold"][
        "operator_explanation_sections"
    ]["selection_policy"] == (
        "soundcheck_summed_bus_headroom_presence_boost_hold_priority"
    )
    assert results["soundcheck_dull_vocal_summed_bus_headroom_hold"][
        "selected_parameters"
    ]["summed_bus_headroom_review_required"] is True
    assert results["soundcheck_dull_vocal_summed_bus_headroom_hold"][
        "selected_parameters"
    ]["summed_bus_true_peak_headroom_db"] == 1.2
    assert results["soundcheck_dull_vocal_summed_bus_headroom_hold"][
        "candidate_blockers"
    ]["soundcheck_presence_air_lift"] == [
        "summed_bus_headroom_blocks_presence_boost",
        "summed_bus_headroom_blocks_gain_up",
    ]
    summed_bus_presence_tradeoffs = {
        row["strategy_id"]: row["why_not_selected"]
        for row in results["soundcheck_dull_vocal_summed_bus_headroom_hold"][
            "alternative_tradeoffs"
        ]
    }
    assert summed_bus_presence_tradeoffs["soundcheck_presence_air_lift"] == (
        "summed_bus_presence_boost_priority:"
        "restore_bus_headroom_before_air_lift"
    )
    assert results["soundcheck_backing_vocal_role_aware_presence_lift"][
        "selected_parameters"
    ]["role_intent"] == "preserve_support_tuck"
    assert results["soundcheck_backing_vocal_role_aware_presence_lift"][
        "selected_parameters"
    ]["presence_lift_bandwidth_hz"] == 5200.0
    assert results["live_dull_vocal_presence_hold"][
        "actual_selected_strategy_id"
    ] == "hold_presence_boost_for_live_safety"
    live_presence_hold = results["live_dull_vocal_presence_hold"]
    assert live_presence_hold["actual_method_family"] == "analysis_only"
    assert live_presence_hold["selected_parameters"]["required_context"] == (
        "source_identity_eq_band_headroom_feedback_readback_rollback"
    )
    assert live_presence_hold["selected_parameters"][
        "forbidden_without_wing_proof"
    ] == "presence_boost_eq_gain_up_or_dynamic_presence_change"
    assert live_presence_hold["selected_parameters"][
        "presence_boost_target_surface"
    ] == "channel"
    assert live_presence_hold["selected_parameters"][
        "presence_boost_eq_band_readback_required"
    ] is True
    assert live_presence_hold["selected_parameters"][
        "presence_boost_no_boost_without_operator_wing_proof"
    ] is True
    assert live_presence_hold["selected_parameters"][
        "live_presence_boost_live_apply_blocked_without_wing_proof"
    ] is True
    assert "wing_eq_on_osc_value" not in live_presence_hold["selected_parameters"]
    presence_readback = live_presence_hold["selected_future_wing_readback_contract"][
        "presence_boost_hold_readback"
    ]
    assert presence_readback["read_only"] is True
    assert presence_readback["presence_boost_target_surface"] == "channel"
    assert presence_readback["presence_boost_eq_band_readback_required"] is True
    assert presence_readback["allowed_live_write_scope_before_wing_proof"] == []
    assert "/ch/{channel}/eq/on" in presence_readback["wing_osc_path_templates"]
    assert "/ch/{channel}/eq/{band}f" in presence_readback[
        "wing_osc_path_templates"
    ]
    assert results["live_dull_vocal_presence_hold"]["candidate_parameters"][
        "soundcheck_presence_air_lift"
    ]["frequency_source"] == "measured_presence_air_deficit"
    assert results["live_vocal_feedback_gain_pullback"][
        "actual_selected_strategy_id"
    ] == "feedback_gain_pullback"
    assert results["live_vocal_feedback_gain_pullback"]["selected_parameters"][
        "live_stepped_change_required"
    ] is True
    assert results["live_vocal_feedback_gain_pullback"]["selected_parameters"][
        "live_max_single_step_db"
    ] == 0.75
    assert results["live_feedback_low_headroom_gain_pullback"][
        "actual_status"
    ] == "analysis_only"
    assert "true_peak_headroom_low" in results[
        "live_feedback_low_headroom_gain_pullback"
    ]["global_blockers"]
    assert results["live_feedback_low_headroom_gain_pullback"][
        "actual_selected_strategy_id"
    ] == "feedback_gain_pullback"
    assert results["live_feedback_low_headroom_gain_pullback"][
        "operator_explanation_sections"
    ]["selection_policy"] == "live_feedback_risk_priority"
    assert results["live_feedback_low_headroom_gain_pullback"][
        "selected_parameters"
    ]["live_stepped_change_required"] is True
    assert results["live_feedback_low_headroom_gain_pullback"][
        "selected_parameters"
    ]["target_scope"] == "source_feedback_and_true_peak_headroom_reduction"
    assert results["live_feedback_low_headroom_gain_pullback"][
        "selected_parameters"
    ]["true_peak_headroom_db"] == 0.6
    assert results["live_feedback_low_headroom_gain_pullback"][
        "selected_parameters"
    ]["target_true_peak_headroom_db"] == 1.5
    assert results["live_feedback_low_headroom_gain_pullback"][
        "selected_parameters"
    ]["expected_post_pullback_true_peak_headroom_db"] == 2.1
    assert results["live_feedback_low_headroom_gain_pullback"][
        "selected_parameters"
    ]["feedback_pullback_also_restores_headroom"] is True
    assert results["live_feedback_low_headroom_gain_pullback"][
        "selected_parameters"
    ]["post_pullback_headroom_recheck_required"] is True
    tradeoffs = {
        item["strategy_id"]: item
        for item in results["live_feedback_low_headroom_gain_pullback"][
            "alternative_tradeoffs"
        ]
    }
    assert tradeoffs["source_true_peak_headroom_pullback"]["why_not_selected"] == (
        "feedback_risk_priority:"
        "feedback_gain_reduction_precedes_general_headroom_pullback"
    )
    assert results["soundcheck_feedback_blocks_presence_boost"][
        "actual_selected_strategy_id"
    ] == "feedback_gain_pullback"
    assert results["soundcheck_measured_feedback_ring_notch"][
        "actual_selected_strategy_id"
    ] == "narrow_feedback_notch_review"
    assert results["soundcheck_measured_feedback_ring_notch"]["selected_parameters"][
        "frequency_source"
    ] == "measured_feedback_ring_peak"
    assert results[
        "soundcheck_measured_feedback_low_headroom_pullback_before_notch"
    ]["actual_status"] == "analysis_only"
    assert results[
        "soundcheck_measured_feedback_low_headroom_pullback_before_notch"
    ]["actual_selected_strategy_id"] == "feedback_gain_pullback"
    assert results[
        "soundcheck_measured_feedback_low_headroom_pullback_before_notch"
    ]["selected_parameters"]["target_scope"] == (
        "source_feedback_and_true_peak_headroom_reduction"
    )
    assert results[
        "soundcheck_measured_feedback_low_headroom_pullback_before_notch"
    ]["selected_parameters"]["expected_post_pullback_true_peak_headroom_db"] == 2.7
    assert results[
        "soundcheck_measured_feedback_low_headroom_pullback_before_notch"
    ]["candidate_parameters"]["narrow_feedback_notch_review"][
        "frequency_source"
    ] == "measured_feedback_ring_peak"
    feedback_tradeoffs = {
        item["strategy_id"]: item
        for item in results[
            "soundcheck_measured_feedback_low_headroom_pullback_before_notch"
        ]["alternative_tradeoffs"]
    }
    assert feedback_tradeoffs["narrow_feedback_notch_review"][
        "why_not_selected"
    ] == (
        "feedback_headroom_priority:"
        "level_pullback_restores_headroom_before_measured_notch"
    )
    assert results["soundcheck_harsh_sibilant_vocal_split_chain"][
        "actual_selected_strategy_id"
    ] == "vocal_de_esser"
    assert results["soundcheck_harsh_sibilant_vocal_split_chain"][
        "actual_method_family"
    ] == "de_esser"
    assert results["soundcheck_harsh_sibilant_vocal_split_chain"][
        "selected_rationale"
    ]["selection_policy"] == "weighted_strategy_score"
    assert results["soundcheck_harsh_sibilant_vocal_split_chain"][
        "selected_parameters"
    ]["bandwidth_source"] == "measured_sibilance_bandwidth"
    assert results["soundcheck_harsh_sibilant_vocal_split_chain"][
        "selected_parameters"
    ]["bandwidth_hz"] == 1890.0
    assert results["soundcheck_harsh_sibilant_vocal_split_chain"][
        "selected_parameters"
    ]["de_esser_parameter_source"] == "measured_sibilance_event"
    assert results["soundcheck_harsh_sibilant_vocal_split_chain"][
        "selected_parameters"
    ]["wing_de_esser_model_osc_value"] == WING_DE_ESSER_GATE_MODEL
    assert results["soundcheck_harsh_sibilant_vocal_split_chain"][
        "selected_parameters"
    ]["wing_de_esser_range_osc_value"] == 5.2
    assert results["soundcheck_harsh_sibilant_vocal_split_chain"][
        "selected_parameters"
    ]["wing_de_esser_mode_osc_value"] == "HF"
    _assert_replay_selected_de_esser_apply_values(
        results["soundcheck_harsh_sibilant_vocal_split_chain"],
        {
            "wing_de_esser_model_osc_value": WING_DE_ESSER_GATE_MODEL,
            "wing_de_esser_on_osc_value": 1,
            "wing_de_esser_frequency_osc_value": 7600.0,
            "wing_de_esser_range_osc_value": 5.2,
            "wing_de_esser_mode_osc_value": "HF",
        },
    )
    _assert_replay_selected_write_key_contract(
        results["soundcheck_harsh_sibilant_vocal_split_chain"],
        readback_section="de_esser_readback",
        prefix="wing_de_esser",
        selected_write_keys=list(WING_DE_ESSER_WRITE_PARAMETER_KEYS),
        write_parameter_keys=WING_DE_ESSER_WRITE_PARAMETER_KEYS,
        selection_policy=WING_DE_ESSER_WRITE_KEY_SELECTION_POLICY,
    )
    _assert_replay_selected_parameter_rollback_operations(
        results["soundcheck_harsh_sibilant_vocal_split_chain"],
        readback_section="de_esser_readback",
        operations_field="wing_de_esser_selected_parameter_rollback_operations",
        selected_write_keys=list(WING_DE_ESSER_WRITE_PARAMETER_KEYS),
        contract_marker=(
            "read_previous_de_esser_value_before_write_then_restore_same_path"
        ),
    )
    assert results["soundcheck_harsh_sibilant_vocal_split_chain"][
        "candidate_parameters"
    ]["vocal_de_esser_dynamic_presence_combo"][
        "processing_chain"
    ] == "de_esser_then_dynamic_presence_control"
    existing_ds902 = results["soundcheck_sibilant_vocal_existing_ds902_readback"]
    assert existing_ds902["actual_selected_strategy_id"] == "vocal_de_esser"
    assert existing_ds902["actual_method_family"] == "de_esser"
    existing_ds902_params = existing_ds902["selected_parameters"]
    assert existing_ds902_params[
        "wing_de_esser_gate_slot_parameter_apply_ready"
    ] is True
    assert existing_ds902_params["wing_de_esser_gate_slot_current_model"] == (
        WING_DE_ESSER_GATE_MODEL
    )
    assert existing_ds902_params[
        "wing_de_esser_model_write_suppressed_by_existing_readback"
    ] is True
    assert "wing_de_esser_model_osc_value" not in existing_ds902_params
    existing_ds902_expected_apply = {
        "wing_de_esser_on_osc_value": 1,
        "wing_de_esser_frequency_osc_value": 7550.0,
        "wing_de_esser_range_osc_value": 5.4,
        "wing_de_esser_mode_osc_value": "HF",
    }
    _assert_replay_selected_de_esser_apply_values(
        existing_ds902,
        existing_ds902_expected_apply,
    )
    _assert_replay_selected_write_key_contract(
        existing_ds902,
        readback_section="de_esser_readback",
        prefix="wing_de_esser",
        selected_write_keys=[
            "wing_de_esser_on_osc_value",
            "wing_de_esser_frequency_osc_value",
            "wing_de_esser_range_osc_value",
            "wing_de_esser_mode_osc_value",
        ],
        write_parameter_keys=WING_DE_ESSER_WRITE_PARAMETER_KEYS,
        selection_policy=WING_DE_ESSER_WRITE_KEY_SELECTION_POLICY,
    )
    _assert_replay_selected_parameter_rollback_operations(
        existing_ds902,
        readback_section="de_esser_readback",
        operations_field="wing_de_esser_selected_parameter_rollback_operations",
        selected_write_keys=[
            "wing_de_esser_on_osc_value",
            "wing_de_esser_frequency_osc_value",
            "wing_de_esser_range_osc_value",
            "wing_de_esser_mode_osc_value",
        ],
        contract_marker=(
            "read_previous_de_esser_value_before_write_then_restore_same_path"
        ),
    )
    existing_ds902_readback = existing_ds902[
        "selected_future_wing_readback_contract"
    ]["de_esser_readback"]
    assert existing_ds902_readback["wing_de_esser_model_osc_value"] is None
    assert "gate/mdl" in existing_ds902_readback["read_before_paths"]
    assert "gate/mdl" in existing_ds902_readback["read_after_paths"]
    assert (
        "wing_de_esser_model_osc_value"
        not in existing_ds902["selected_rationale"]["wing_osc_apply_parameter_keys"]
    )
    assert (
        "wing_de_esser_model_osc_value"
        not in existing_ds902["proposal_selected_wing_osc_summary"][
            "wing_osc_apply_parameter_keys"
        ]
    )
    combo_params = results["soundcheck_harsh_sibilant_vocal_split_chain"][
        "candidate_parameters"
    ]["vocal_de_esser_dynamic_presence_combo"]
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
    assert WING_GATE_SLOT_SINGLE_MODEL_COMBO_BLOCKER in results[
        "soundcheck_harsh_sibilant_vocal_split_chain"
    ]["candidate_blockers"]["vocal_de_esser_dynamic_presence_combo"]
    combo_readback = results["soundcheck_harsh_sibilant_vocal_split_chain"][
        "candidate_future_wing_readback_contracts"
    ]["vocal_de_esser_dynamic_presence_combo"]["dynamic_eq_readback"]
    assert combo_readback["wing_gate_slot_mutually_exclusive_processor_chain"] is True
    assert combo_readback["wing_gate_slot_required_models"] == [
        WING_DE_ESSER_GATE_MODEL,
        WING_DYNAMIC_EQ_GATE_MODEL,
    ]
    assert combo_readback["wing_gate_slot_combo_apply_ready"] is False
    assert combo_readback["wing_gate_slot_combo_blocker"] == (
        WING_GATE_SLOT_SINGLE_MODEL_COMBO_BLOCKER
    )
    assert results["soundcheck_sibilant_dynamic_vocal_order_review"][
        "actual_selected_strategy_id"
    ] == "vocal_de_esser_before_compression_recheck"
    assert results["soundcheck_sibilant_dynamic_vocal_order_review"][
        "actual_method_family"
    ] == "de_esser"
    assert results["soundcheck_sibilant_dynamic_vocal_order_review"][
        "operator_explanation_sections"
    ]["selection_policy"] == (
        "soundcheck_wing_native_de_esser_first_before_compression_priority"
    )
    sibilance_order_readback = results["soundcheck_sibilant_dynamic_vocal_order_review"][
        "proposal_selected_wing_osc_readback_summary"
    ]
    assert sibilance_order_readback["wing_osc_readback_status"] == (
        "wing_osc_readback_paths_declared_before_live_promotion"
    )
    assert "gate/f" in sibilance_order_readback["wing_osc_read_before_paths"]
    assert "gate/range" in sibilance_order_readback["wing_osc_read_before_paths"]
    assert "/ch/{channel}/gate/f" in sibilance_order_readback[
        "wing_osc_readback_path_templates"
    ]
    assert results["soundcheck_sibilant_dynamic_vocal_order_review"][
        "operator_explanation_sections"
    ]["selected_wing_osc_readback_status"] == (
        "wing_osc_readback_paths_declared_before_live_promotion"
    )
    assert (
        "Selected readback status: "
        "wing_osc_readback_paths_declared_before_live_promotion."
    ) in results["soundcheck_sibilant_dynamic_vocal_order_review"][
        "operator_explanation"
    ]
    assert results["soundcheck_sibilant_dynamic_vocal_order_review"][
        "selected_parameters"
    ]["processing_chain"] == "ds902_de_esser_first_then_compressor_recheck"
    assert results["soundcheck_sibilant_dynamic_vocal_order_review"][
        "selected_parameters"
    ]["threshold_dbfs"] == -23.7
    assert results["soundcheck_sibilant_dynamic_vocal_order_review"][
        "selected_parameters"
    ]["range_db"] == 4.8
    assert results["soundcheck_sibilant_dynamic_vocal_order_review"][
        "selected_parameters"
    ]["wing_de_esser_range_osc_value"] == 4.8
    assert results["soundcheck_sibilant_dynamic_vocal_order_review"][
        "selected_parameters"
    ]["followup_compressor_threshold_dbfs"] == -19.7
    assert results["soundcheck_sibilant_dynamic_vocal_order_review"][
        "selected_parameters"
    ]["followup_compressor_attack_ms"] == 22.4
    assert results["soundcheck_sibilant_dynamic_vocal_order_review"][
        "selected_parameters"
    ]["followup_compressor_release_ms"] == 195.5
    assert results["soundcheck_sibilant_dynamic_vocal_order_review"][
        "selected_parameters"
    ]["post_de_esser_compressor_recheck_required"] is True
    assert results["soundcheck_sibilant_dynamic_vocal_order_review"][
        "selected_parameters"
    ]["followup_processing_apply_ready"] is False
    assert "wing_compressor_ratio_osc_value" not in results[
        "soundcheck_sibilant_dynamic_vocal_order_review"
    ]["selected_parameters"]
    _assert_replay_selected_de_esser_apply_values(
        results["soundcheck_sibilant_dynamic_vocal_order_review"],
        {
            "wing_de_esser_model_osc_value": WING_DE_ESSER_GATE_MODEL,
            "wing_de_esser_on_osc_value": 1,
            "wing_de_esser_frequency_osc_value": 7700.0,
            "wing_de_esser_range_osc_value": 4.8,
            "wing_de_esser_mode_osc_value": "HF",
        },
    )
    assert results["live_feedback_risk_beats_sibilance_correction"][
        "actual_selected_strategy_id"
    ] == "feedback_gain_pullback"
    assert results["live_feedback_risk_beats_sibilance_correction"][
        "operator_explanation_sections"
    ]["problem_found"] == "feedback_risk"
    assert results["live_feedback_risk_beats_sibilance_correction"][
        "operator_explanation_sections"
    ]["selection_policy"] == "live_feedback_risk_priority"
    assert results["live_sibilant_vocal"]["actual_selected_strategy_id"] == (
        "vocal_de_esser"
    )
    assert results["live_sibilant_vocal"]["actual_method_family"] == "de_esser"
    assert results["live_sibilant_vocal"]["selected_rationale"][
        "selection_policy"
    ] == "weighted_strategy_score"
    assert results["live_sibilant_vocal"]["selected_parameters"][
        "bandwidth_source"
    ] == "measured_sibilance_bandwidth"
    assert results["live_sibilant_vocal"]["selected_parameters"][
        "bandwidth_hz"
    ] == 900.0
    assert results["live_sibilant_vocal"]["selected_parameters"][
        "de_esser_parameter_source"
    ] == "measured_sibilance_event"
    assert (
        "wing_de_esser_model_osc_value"
        not in results["live_sibilant_vocal"]["selected_parameters"]
    )
    assert (
        "wing_de_esser_on_osc_value"
        not in results["live_sibilant_vocal"]["selected_parameters"]
    )
    assert (
        "wing_de_esser_frequency_osc_value"
        not in results["live_sibilant_vocal"]["selected_parameters"]
    )
    assert (
        "wing_de_esser_range_osc_value"
        not in results["live_sibilant_vocal"]["selected_parameters"]
    )
    assert (
        "wing_de_esser_mode_osc_value"
        not in results["live_sibilant_vocal"]["selected_parameters"]
    )
    assert results["live_sibilant_vocal"]["selected_parameters"][
        "wing_de_esser_gate_slot_conflict_policy"
    ] == WING_DE_ESSER_GATE_SLOT_CONFLICT_POLICY
    live_de_esser_params = results["live_sibilant_vocal"]["selected_parameters"]
    assert live_de_esser_params["live_de_esser_parameter_write_forbidden"] is True
    assert live_de_esser_params["live_de_esser_parameter_readback_required"] is True
    assert (
        live_de_esser_params[
            "live_de_esser_parameter_apply_blocked_without_wing_proof"
        ]
        is True
    )
    assert live_de_esser_params["live_de_esser_parameter_review_only_values"] == {
        "wing_de_esser_on_osc_value": 1,
        "wing_de_esser_frequency_osc_value": 7800.0,
        "wing_de_esser_range_osc_value": 5.4,
        "wing_de_esser_mode_osc_value": "HF",
    }
    assert live_de_esser_params["live_de_esser_parameter_review_only_reason"] == (
        "live_de_esser_must_read_existing_ds902_state_and_operator_confirm_before_parameter_apply"
    )
    assert live_de_esser_params["live_de_esser_conservative_proposal"] is True
    assert live_de_esser_params["live_de_esser_max_range_db"] == (
        LIVE_DE_ESSER_MAX_RANGE_DB
    )
    assert (
        live_de_esser_params["live_de_esser_stepwise_range_review_required"]
        is True
    )
    assert live_de_esser_params["live_de_esser_max_range_step_db"] == (
        LIVE_DE_ESSER_MAX_RANGE_STEP_DB
    )
    assert (
        live_de_esser_params["live_de_esser_sibilance_recheck_after_apply_required"]
        is True
    )
    assert (
        live_de_esser_params[
            "live_de_esser_gate_slot_readback_before_model_change_required"
        ]
        is True
    )
    assert (
        live_de_esser_params["live_de_esser_gate_slot_model_write_forbidden"]
        is True
    )
    assert (
        live_de_esser_params["live_de_esser_requires_existing_ds902_model_readback"]
        is True
    )
    assert live_de_esser_params["live_de_esser_model_review_only_value"] == (
        WING_DE_ESSER_GATE_MODEL
    )
    assert live_de_esser_params["live_de_esser_model_write_review_only_reason"] == (
        "live_de_esser_must_read_existing_ds902_gate_slot_model_before_parameter_apply"
    )
    assert live_de_esser_params["live_de_esser_on_mode_write_forbidden"] is True
    assert live_de_esser_params["live_de_esser_on_mode_readback_required"] is True
    assert live_de_esser_params["live_de_esser_on_write_forbidden"] is True
    assert live_de_esser_params["live_de_esser_mode_write_forbidden"] is True
    assert (
        live_de_esser_params["live_de_esser_requires_existing_enabled_readback"]
        is True
    )
    assert live_de_esser_params["live_de_esser_requires_existing_mode_readback"] is True
    assert live_de_esser_params["live_de_esser_on_review_only_value"] == 1
    assert live_de_esser_params["live_de_esser_mode_review_only_value"] == "HF"
    assert live_de_esser_params["live_de_esser_on_mode_review_only_values"] == {
        "wing_de_esser_on_osc_value": 1,
        "wing_de_esser_mode_osc_value": "HF",
    }
    assert live_de_esser_params["live_de_esser_on_mode_review_only_reason"] == (
        "live_de_esser_must_read_existing_gate_on_and_mode_before_parameter_apply"
    )
    assert live_de_esser_params["live_de_esser_no_fx_rack_insert_required"] is True
    assert live_de_esser_params["live_de_esser_operator_rehearsal_required"] is True
    assert live_de_esser_params["live_de_esser_live_promotion_status"] == (
        "blocked_until_real_wing_ds902_readback_sibilance_recheck_and_rollback"
    )
    assert (
        live_de_esser_params["live_de_esser_live_apply_blocked_without_wing_proof"]
        is True
    )
    _assert_replay_selected_de_esser_apply_values(
        results["live_sibilant_vocal"],
        {},
    )
    live_de_esser_write_keys = []
    _assert_replay_selected_write_key_contract(
        results["live_sibilant_vocal"],
        readback_section="de_esser_readback",
        prefix="wing_de_esser",
        selected_write_keys=live_de_esser_write_keys,
        write_parameter_keys=WING_DE_ESSER_WRITE_PARAMETER_KEYS,
        selection_policy=WING_DE_ESSER_WRITE_KEY_SELECTION_POLICY,
    )
    assert results["live_sibilant_vocal"]["proposal_authority"][
        "can_apply_live"
    ] is False
    assert results["live_sibilant_vocal"]["proposal_authority"][
        "live_apply_allowed"
    ] is False
    assert results["live_sibilant_vocal"]["candidate_parameters"][
        "static_sibilance_band_cut"
    ]["static_sibilance_cut_db"] == -2.8
    assert results["soundcheck_vocal_level_pressure_before_de_essing"][
        "actual_selected_strategy_id"
    ] == "level_plane_gain_down"
    assert results["soundcheck_vocal_level_pressure_before_de_essing"][
        "operator_explanation_sections"
    ]["selection_policy"] == "soundcheck_vocal_level_pressure_before_tone_priority"
    assert results["soundcheck_vocal_level_pressure_before_de_essing"][
        "selected_parameters"
    ]["current_minus_target_lu"] == 3.4
    assert results["live_harsh_vocal_static_presence_cut"][
        "actual_selected_strategy_id"
    ] == "broad_presence_cut"
    assert results["live_harsh_vocal_static_presence_cut"][
        "operator_explanation_sections"
    ]["selection_policy"] == "live_static_vocal_harshness_cut_priority"
    assert "live_dynamic_presence_eq_requires_operator_review" in results[
        "live_harsh_vocal_static_presence_cut"
    ]["candidate_blockers"]["presence_dynamic_eq_cut"]
    assert results["live_operator_owned_eq_holds_harsh_vocal"][
        "actual_selected_strategy_id"
    ] == "hold_live_operator_owned_eq_for_review"
    assert results["live_operator_owned_eq_holds_harsh_vocal"][
        "actual_method_family"
    ] == "analysis_only"
    live_owned_eq_hold = results["live_operator_owned_eq_holds_harsh_vocal"]
    assert live_owned_eq_hold["operator_explanation_sections"][
        "selection_policy"
    ] == "live_operator_owned_eq_review_priority"
    assert live_owned_eq_hold["selected_parameters"]["dsp_change"] == "hold"
    assert "broad_presence_cut" in live_owned_eq_hold["selected_parameters"][
        "blocked_eq_strategy_ids"
    ]
    assert live_owned_eq_hold["selected_parameters"][
        "blocked_eq_problem"
    ] == "harsh"
    assert live_owned_eq_hold["selected_parameters"][
        "blocked_eq_filter_type"
    ] == "peaking"
    assert live_owned_eq_hold["selected_parameters"]["blocked_eq_gain_db"] == -2.8
    assert live_owned_eq_hold["selected_parameters"][
        "wing_eq_current_operator_owned"
    ] is True
    assert live_owned_eq_hold["selected_parameters"][
        "wing_eq_operator_owned_model_edit_requires_review"
    ] is True
    assert live_owned_eq_hold["selected_parameters"][
        "wing_eq_operator_owned_model_blocker"
    ] == WING_EQ_OPERATOR_OWNED_MODEL_BLOCKER
    assert live_owned_eq_hold["selected_parameters"][
        "wing_rack_fw305_verified_osc_mapping_required"
    ] is True
    assert live_owned_eq_hold["selected_parameters"][
        "wing_rack_fw305_eq_band_readback_required"
    ] is True
    assert live_owned_eq_hold["selected_parameters"][
        "wing_rack_fw305_rollback_required"
    ] is True
    assert live_owned_eq_hold["selected_parameters"][
        "live_authority_granted"
    ] is False
    assert live_owned_eq_hold["selected_parameters"]["live_write"] is False
    assert live_owned_eq_hold["candidate_blockers"]["broad_presence_cut"] == [
        WING_EQ_OPERATOR_OWNED_MODEL_BLOCKER
    ]
    _assert_replay_analysis_only_has_no_apply_values(live_owned_eq_hold)
    live_owned_eq_tradeoffs = {
        row["strategy_id"]: row["why_not_selected"]
        for row in live_owned_eq_hold["alternative_tradeoffs"]
    }
    assert live_owned_eq_tradeoffs["broad_presence_cut"] == (
        "live_operator_owned_eq_review_priority:"
        "manual_eq_slot_review_precedes_live_eq_change"
    )
    assert results["soundcheck_operator_owned_eq_holds_harsh_vocal"][
        "actual_selected_strategy_id"
    ] == "hold_soundcheck_operator_owned_eq_for_review"
    assert results["soundcheck_operator_owned_eq_holds_harsh_vocal"][
        "actual_method_family"
    ] == "analysis_only"
    soundcheck_owned_eq_hold = results[
        "soundcheck_operator_owned_eq_holds_harsh_vocal"
    ]
    assert soundcheck_owned_eq_hold["operator_explanation_sections"][
        "selection_policy"
    ] == "soundcheck_operator_owned_eq_review_priority"
    soundcheck_owned_eq_params = soundcheck_owned_eq_hold["selected_parameters"]
    assert soundcheck_owned_eq_params["dsp_change"] == "hold"
    assert soundcheck_owned_eq_params["reason"] == (
        "soundcheck_operator_owned_eq_requires_operator_review"
    )
    assert "broad_presence_cut" in soundcheck_owned_eq_params[
        "blocked_eq_strategy_ids"
    ]
    assert soundcheck_owned_eq_params["blocked_eq_operator_owned_blockers"] == [
        WING_EQ_OPERATOR_OWNED_MODEL_BLOCKER
    ]
    assert soundcheck_owned_eq_params["blocked_eq_problem"] == "harsh"
    assert soundcheck_owned_eq_params["blocked_eq_filter_type"] == "peaking"
    assert soundcheck_owned_eq_params["blocked_eq_gain_db"] == -2.8
    assert soundcheck_owned_eq_params["soundcheck_operator_owned_eq_hold_proposal"] is True
    assert soundcheck_owned_eq_params[
        "soundcheck_operator_owned_eq_target_surface"
    ] == "channel"
    assert soundcheck_owned_eq_params[
        "soundcheck_operator_owned_eq_model_readback_required"
    ] is True
    assert soundcheck_owned_eq_params[
        "soundcheck_operator_owned_eq_band_readback_required"
    ] is True
    assert soundcheck_owned_eq_params[
        "soundcheck_operator_owned_eq_protected_band_review_required"
    ] is True
    assert soundcheck_owned_eq_params[
        "soundcheck_operator_owned_eq_no_eq_change_without_operator_review"
    ] is True
    assert soundcheck_owned_eq_params["wing_eq_current_operator_owned"] is True
    assert soundcheck_owned_eq_params[
        "wing_eq_operator_owned_model_edit_requires_review"
    ] is True
    assert soundcheck_owned_eq_params[
        "wing_eq_operator_owned_model_blocker"
    ] == WING_EQ_OPERATOR_OWNED_MODEL_BLOCKER
    soundcheck_owned_eq_readback = soundcheck_owned_eq_hold[
        "selected_future_wing_readback_contract"
    ]["soundcheck_operator_owned_eq_review_readback"]
    assert soundcheck_owned_eq_readback["read_only"] is True
    assert soundcheck_owned_eq_readback[
        "soundcheck_operator_owned_eq_current_operator_owned"
    ] is True
    assert soundcheck_owned_eq_readback[
        "soundcheck_operator_owned_eq_operator_owned_blockers"
    ] == [WING_EQ_OPERATOR_OWNED_MODEL_BLOCKER]
    assert "eq/mdl" in soundcheck_owned_eq_readback["read_before_paths"]
    assert "eq/on" in soundcheck_owned_eq_readback["read_before_paths"]
    assert "eq/{band}f" in soundcheck_owned_eq_readback["read_before_paths"]
    assert "/ch/{channel}/eq/{band}f" in soundcheck_owned_eq_readback[
        "wing_osc_path_templates"
    ]
    assert soundcheck_owned_eq_hold["candidate_blockers"]["broad_presence_cut"] == [
        WING_EQ_OPERATOR_OWNED_MODEL_BLOCKER
    ]
    _assert_replay_analysis_only_has_no_apply_values(soundcheck_owned_eq_hold)
    soundcheck_owned_eq_tradeoffs = {
        row["strategy_id"]: row["why_not_selected"]
        for row in soundcheck_owned_eq_hold["alternative_tradeoffs"]
    }
    assert soundcheck_owned_eq_tradeoffs["broad_presence_cut"] == (
        "soundcheck_operator_owned_eq_review_priority:"
        "manual_eq_slot_owner_review_precedes_soundcheck_eq_change"
    )
    assert soundcheck_owned_eq_tradeoffs["vocal_de_esser"] == (
        "soundcheck_operator_owned_eq_review_priority:"
        "manual_eq_owner_review_precedes_weak_fallback_tone_processor"
    )
    assert results["soundcheck_harsh_dynamic_vocal_envelope"][
        "actual_selected_strategy_id"
    ] == "vocal_compressor_envelope_smoothing"
    live_dynamic_harsh = results[
        "live_dynamic_harsh_vocal_ride_before_tone_or_compressor"
    ]
    assert live_dynamic_harsh["actual_selected_strategy_id"] == (
        "manual_level_ride_before_compression"
    )
    assert live_dynamic_harsh["actual_method_family"] == "analysis_only"
    assert live_dynamic_harsh["operator_explanation_sections"][
        "selection_policy"
    ] == "live_dynamic_harsh_vocal_ride_priority"
    assert live_dynamic_harsh["selected_parameters"][
        "live_dynamic_harsh_vocal_ride_priority"
    ] is True
    assert live_dynamic_harsh["selected_parameters"][
        "live_dynamic_harshness_tone_processing_deferred"
    ] is True
    assert live_dynamic_harsh["selected_parameters"][
        "live_dynamic_harshness_compressor_change_deferred"
    ] is True
    assert live_dynamic_harsh["selected_parameters"][
        "live_dynamic_harshness_remeasure_after_ride_required"
    ] is True
    assert live_dynamic_harsh["selected_parameters"][
        "live_dynamic_harshness_no_compressor_eq_or_deq_without_operator_wing_proof"
    ] is True
    assert live_dynamic_harsh["selected_parameters"][
        "forbidden_without_wing_proof"
    ] == "compressor_dynamic_eq_static_presence_cut_until_ride_remeasure"
    live_dynamic_harsh_tradeoffs = {
        row["strategy_id"]: row
        for row in live_dynamic_harsh["alternative_tradeoffs"]
    }
    assert live_dynamic_harsh_tradeoffs["broad_presence_cut"][
        "why_not_selected"
    ] == (
        "live_dynamic_harsh_vocal_ride_priority:"
        "ride_and_remeasure_before_tone_or_compressor_change"
    )
    assert results["soundcheck_harsh_guitar_dynamic_presence_tame"][
        "actual_selected_strategy_id"
    ] == "instrument_dynamic_presence_tame"
    assert results["soundcheck_harsh_guitar_dynamic_presence_tame"][
        "actual_method_family"
    ] == "dynamic_eq"
    assert results["soundcheck_harsh_guitar_dynamic_presence_tame"][
        "selected_parameters"
    ]["instrument_harsh_parameter_source"] == "measured_instrument_harshness_event"
    assert results["soundcheck_harsh_guitar_dynamic_presence_tame"][
        "selected_parameters"
    ]["frequency_hz"] == 3100.0
    assert results["soundcheck_harsh_guitar_dynamic_presence_tame"][
        "selected_parameters"
    ]["wing_dynamic_eq_gain_osc_value"] == -2.9
    assert results["soundcheck_harsh_guitar_dynamic_presence_tame"][
        "selected_parameters"
    ]["q"] == 3.59
    assert results["soundcheck_harsh_guitar_dynamic_presence_tame"][
        "selected_parameters"
    ]["threshold_dbfs"] == -25.5
    assert results["soundcheck_harsh_guitar_dynamic_presence_tame"][
        "operator_explanation_sections"
    ]["selection_policy"] == "weighted_strategy_score"
    guitar_dynamic_params = results["soundcheck_harsh_guitar_dynamic_presence_tame"][
        "selected_parameters"
    ]
    assert guitar_dynamic_params["threshold_dbfs"] == -25.5
    assert guitar_dynamic_params["attack_ms"] == 4.4
    assert guitar_dynamic_params["release_ms"] == 180.0
    assert guitar_dynamic_params["wing_dynamic_eq_model_osc_value"] == (
        WING_DYNAMIC_EQ_GATE_MODEL
    )
    assert results["soundcheck_harsh_overhead_cymbal_dynamic_de_edge"][
        "actual_selected_strategy_id"
    ] == "drum_cymbal_static_air_cut"
    assert results["soundcheck_harsh_overhead_cymbal_dynamic_de_edge"][
        "selected_parameters"
    ]["drum_harsh_parameter_source"] == "measured_drum_harshness_event"
    assert results["soundcheck_harsh_overhead_cymbal_dynamic_de_edge"][
        "selected_parameters"
    ]["gain_db"] == -2.2
    assert results["soundcheck_harsh_overhead_cymbal_dynamic_de_edge"][
        "selected_parameters"
    ]["q"] == 0.59
    assert results["soundcheck_harsh_overhead_cymbal_dynamic_de_edge"][
        "operator_explanation_sections"
    ]["selection_policy"] == "weighted_strategy_score"
    overhead_dynamic_params = results[
        "soundcheck_harsh_overhead_cymbal_dynamic_de_edge"
    ]["candidate_parameters"]["drum_cymbal_dynamic_de_edge"]
    assert overhead_dynamic_params["threshold_dbfs"] == -24.4
    assert overhead_dynamic_params["release_ms"] == 149.8
    assert results["soundcheck_harsh_overhead_top_octave_high_cut"][
        "actual_selected_strategy_id"
    ] == "drum_cymbal_high_cut_air_limit"
    assert results["soundcheck_harsh_overhead_top_octave_high_cut"][
        "selected_parameters"
    ]["filter_type"] == "high_cut"
    assert results["soundcheck_harsh_overhead_top_octave_high_cut"][
        "selected_parameters"
    ]["wing_eq_high_cut_frequency_hz"] == 12000.0
    assert results["soundcheck_harsh_overhead_top_octave_high_cut"][
        "selected_parameters"
    ]["wing_eq_high_cut_slope_db_per_octave"] == 6
    assert results["soundcheck_harsh_overhead_top_octave_high_cut"][
        "selected_future_wing_readback_contract"
    ]["eq_band_readback"]["read_before_paths"] == [
        "eq/on",
        "eq/mdl",
        "eq/mix",
        "flt/hc",
        "flt/hcf",
        "flt/hcs",
    ]
    _assert_replay_eq_write_key_contract(
        results["soundcheck_harsh_overhead_top_octave_high_cut"],
        selected_write_keys=[
            "wing_eq_on_osc_value",
            "wing_eq_mix_osc_value",
            "wing_eq_high_cut_enabled_osc_value",
            "wing_eq_high_cut_frequency_hz",
            "wing_eq_high_cut_slope_db_per_octave",
        ],
    )
    _assert_replay_eq_selected_parameter_rollback_operations(
        results["soundcheck_harsh_overhead_top_octave_high_cut"],
        selected_write_keys=[
            "wing_eq_on_osc_value",
            "wing_eq_mix_osc_value",
            "wing_eq_high_cut_enabled_osc_value",
            "wing_eq_high_cut_frequency_hz",
            "wing_eq_high_cut_slope_db_per_octave",
        ],
    )
    assert results["live_harsh_overhead_cymbal_mic_level_review"][
        "actual_selected_strategy_id"
    ] == "drum_cymbal_mic_level_review"
    assert results["live_harsh_overhead_cymbal_mic_level_review"][
        "selected_parameters"
    ]["dsp_change"] == "hold"
    assert results["live_harsh_overhead_cymbal_mic_level_review"][
        "selected_parameters"
    ]["live_cymbal_harshness_review_priority"] is True
    assert results["live_harsh_overhead_cymbal_mic_level_review"][
        "selected_parameters"
    ]["forbidden_without_wing_proof"] == "live_cymbal_eq_dynamic_de_edge"
    assert results["inferred_vocal_identity_requires_confirmation"][
        "actual_status"
    ] == "analysis_only"
    assert results["inferred_vocal_identity_requires_confirmation"][
        "actual_selected_strategy_id"
    ] == "hold_for_source_identity_confirmation"
    assert results["inferred_vocal_identity_requires_confirmation"][
        "operator_explanation_sections"
    ]["selection_policy"] == "source_identity_confirmation_guard_priority"
    assert "source_identity_not_confirmed" in results[
        "inferred_vocal_identity_requires_confirmation"
    ]["global_blockers"]
    for scenario_id in (
        "low_confidence_source_identity_blocks_eq",
        "unconfirmed_source_identity_blocks_eq_without_inference",
    ):
        assert results[scenario_id]["actual_selected_strategy_id"] == (
            "hold_for_source_identity_confirmation"
        )
        assert results[scenario_id]["actual_method_family"] == "analysis_only"
        assert results[scenario_id]["operator_explanation_sections"][
            "selection_policy"
        ] == "source_identity_confirmation_guard_priority"
    assert results["low_confidence_audio_evidence_holds_before_eq"][
        "actual_selected_strategy_id"
    ] == "hold_for_audio_evidence_recheck"
    assert results["low_confidence_audio_evidence_holds_before_eq"][
        "actual_method_family"
    ] == "analysis_only"
    assert results["low_confidence_audio_evidence_holds_before_eq"][
        "operator_explanation_sections"
    ]["selection_policy"] == "audio_evidence_confidence_guard_priority"
    assert "audio_evidence_confidence_low" in results[
        "low_confidence_audio_evidence_holds_before_eq"
    ]["global_blockers"]
    assert results["stale_live_audio_evidence_holds_before_eq"][
        "actual_selected_strategy_id"
    ] == "hold_for_audio_evidence_recheck"
    assert results["stale_live_audio_evidence_holds_before_eq"][
        "actual_method_family"
    ] == "analysis_only"
    assert results["stale_live_audio_evidence_holds_before_eq"][
        "operator_explanation_sections"
    ]["selection_policy"] == "audio_evidence_freshness_guard_priority"
    assert "audio_evidence_stale" in results[
        "stale_live_audio_evidence_holds_before_eq"
    ]["global_blockers"]
    assert results["auto_apply_request_holds_before_dsp"][
        "actual_selected_strategy_id"
    ] == "hold_for_operator_review_no_auto_apply"
    assert results["auto_apply_request_holds_before_dsp"][
        "actual_method_family"
    ] == "analysis_only"
    assert results["auto_apply_request_holds_before_dsp"][
        "operator_explanation_sections"
    ]["selection_policy"] == "auto_apply_operator_review_guard_priority"
    assert "auto_apply_not_supported" in results[
        "auto_apply_request_holds_before_dsp"
    ]["global_blockers"]
    assert results["unsupported_mode_holds_before_dsp"][
        "actual_selected_strategy_id"
    ] == "hold_for_supported_mode_selection"
    assert results["unsupported_mode_holds_before_dsp"][
        "actual_method_family"
    ] == "analysis_only"
    assert results["unsupported_mode_holds_before_dsp"][
        "operator_explanation_sections"
    ]["selection_policy"] == "unsupported_mode_guard_priority"
    assert "unsupported_smart_dsp_mode" in results[
        "unsupported_mode_holds_before_dsp"
    ]["global_blockers"]
    assert results["routing_scene_snapshot_intent_holds_before_dsp"][
        "actual_selected_strategy_id"
    ] == "hold_for_routing_scene_snapshot_review"
    assert results["routing_scene_snapshot_intent_holds_before_dsp"][
        "actual_method_family"
    ] == "analysis_only"
    assert results["routing_scene_snapshot_intent_holds_before_dsp"][
        "operator_explanation_sections"
    ]["selection_policy"] == "routing_scene_snapshot_guard_priority"
    assert "forbidden_routing_scene_snapshot_intent" in results[
        "routing_scene_snapshot_intent_holds_before_dsp"
    ]["global_blockers"]
    assert results["routing_scene_snapshot_intent_holds_before_dsp"][
        "selected_parameters"
    ]["live_write"] is False
    assert results["live_master_output_intent_holds_before_channel_eq"][
        "actual_selected_strategy_id"
    ] == "hold_for_live_master_output_intent_review"
    assert results["live_master_output_intent_holds_before_channel_eq"][
        "actual_method_family"
    ] == "analysis_only"
    assert results["live_master_output_intent_holds_before_channel_eq"][
        "operator_explanation_sections"
    ]["selection_policy"] == "live_master_output_intent_guard_priority"
    assert "live_master_output_intent_requires_review" in results[
        "live_master_output_intent_holds_before_channel_eq"
    ]["global_blockers"]
    assert results["live_master_output_intent_holds_before_channel_eq"][
        "selected_parameters"
    ]["live_write"] is False
    assert results["live_eq_intent_holds_before_channel_eq_or_gain_up_guard"][
        "actual_selected_strategy_id"
    ] == "hold_for_live_eq_intent_review"
    assert results["live_eq_intent_holds_before_channel_eq_or_gain_up_guard"][
        "actual_method_family"
    ] == "analysis_only"
    assert results["live_eq_intent_holds_before_channel_eq_or_gain_up_guard"][
        "operator_explanation_sections"
    ]["selection_policy"] == "live_eq_intent_guard_priority"
    assert "live_eq_intent_requires_review" in results[
        "live_eq_intent_holds_before_channel_eq_or_gain_up_guard"
    ]["global_blockers"]
    assert "live_source_gain_up_intent_requires_review" not in results[
        "live_eq_intent_holds_before_channel_eq_or_gain_up_guard"
    ]["global_blockers"]
    assert results["live_eq_intent_holds_before_channel_eq_or_gain_up_guard"][
        "selected_parameters"
    ]["live_write"] is False
    assert results["live_feedback_risk_beats_eq_intent_hold"][
        "actual_selected_strategy_id"
    ] == "feedback_gain_pullback"
    assert results["live_feedback_risk_beats_eq_intent_hold"][
        "actual_method_family"
    ] == "gain_fader"
    assert results["live_feedback_risk_beats_eq_intent_hold"][
        "operator_explanation_sections"
    ]["selection_policy"] == "live_feedback_risk_priority"
    assert "live_eq_intent_requires_review" in results[
        "live_feedback_risk_beats_eq_intent_hold"
    ]["global_blockers"]
    assert "live_source_gain_up_intent_requires_review" not in results[
        "live_feedback_risk_beats_eq_intent_hold"
    ]["global_blockers"]
    assert results["live_feedback_risk_beats_eq_intent_hold"][
        "selected_parameters"
    ]["delta_db"] < 0.0
    _assert_replay_gain_delta_mapping(
        results["live_feedback_risk_beats_eq_intent_hold"],
        source="delta_db",
        expected_delta=results["live_feedback_risk_beats_eq_intent_hold"][
            "selected_parameters"
        ]["delta_db"],
        path_prefix="/ch/{channel}",
    )
    assert results["live_source_gain_up_intent_holds_before_channel_eq"][
        "actual_selected_strategy_id"
    ] == "hold_for_live_source_gain_up_intent_review"
    assert results["live_source_gain_up_intent_holds_before_channel_eq"][
        "actual_method_family"
    ] == "analysis_only"
    assert results["live_source_gain_up_intent_holds_before_channel_eq"][
        "operator_explanation_sections"
    ]["selection_policy"] == "live_source_gain_up_intent_guard_priority"
    assert "live_source_gain_up_intent_requires_review" in results[
        "live_source_gain_up_intent_holds_before_channel_eq"
    ]["global_blockers"]
    assert results["live_source_gain_up_intent_holds_before_channel_eq"][
        "selected_parameters"
    ]["live_write"] is False
    assert results["live_processing_change_intent_holds_before_channel_eq"][
        "actual_selected_strategy_id"
    ] == "hold_for_live_processing_intent_review"
    assert results["live_processing_change_intent_holds_before_channel_eq"][
        "actual_method_family"
    ] == "analysis_only"
    assert results["live_processing_change_intent_holds_before_channel_eq"][
        "operator_explanation_sections"
    ]["selection_policy"] == "live_processing_intent_guard_priority"
    assert "live_processing_intent_requires_review" in results[
        "live_processing_change_intent_holds_before_channel_eq"
    ]["global_blockers"]
    assert results["live_processing_change_intent_holds_before_channel_eq"][
        "selected_parameters"
    ]["live_write"] is False
    assert results["live_pan_width_intent_holds_before_channel_eq"][
        "actual_selected_strategy_id"
    ] == "hold_for_live_pan_width_intent_review"
    assert results["live_pan_width_intent_holds_before_channel_eq"][
        "actual_method_family"
    ] == "analysis_only"
    assert results["live_pan_width_intent_holds_before_channel_eq"][
        "operator_explanation_sections"
    ]["selection_policy"] == "live_pan_width_intent_guard_priority"
    assert "live_pan_width_intent_requires_review" in results[
        "live_pan_width_intent_holds_before_channel_eq"
    ]["global_blockers"]
    assert results["live_pan_width_intent_holds_before_channel_eq"][
        "selected_parameters"
    ]["live_write"] is False
    assert results["declared_vocal_audio_identity_conflict_holds"][
        "actual_selected_strategy_id"
    ] == "hold_for_source_identity_recheck"
    assert results["declared_vocal_audio_identity_conflict_holds"][
        "actual_status"
    ] == "analysis_only"
    assert "source_identity_evidence_conflict" in results[
        "declared_vocal_audio_identity_conflict_holds"
    ]["global_blockers"]
    assert results["soundcheck_dynamic_vocal_compression"][
        "actual_selected_strategy_id"
    ] == "vocal_phrase_leveling_compression"
    assert results["soundcheck_dynamic_vocal_compression"]["actual_method_family"] == (
        "dynamics"
    )
    assert results["soundcheck_dynamic_vocal_compression"]["selected_parameters"][
        "compressor_enabled"
    ] is True
    assert results["soundcheck_dynamic_vocal_compression"]["selected_parameters"][
        "wing_compressor_on_osc_path_suffix"
    ] == "dyn/on"
    assert results["soundcheck_dynamic_vocal_compression"]["selected_parameters"][
        "wing_compressor_on_osc_value"
    ] == WING_COMPRESSOR_ON_MAX
    assert results["soundcheck_dynamic_vocal_compression"]["selected_parameters"][
        "wing_compressor_on_value_type"
    ] == "integer_bool"
    assert results["soundcheck_dynamic_vocal_compression"]["selected_parameters"][
        "dynamics_parameter_source"
    ] == "measured_envelope"
    assert results["soundcheck_dynamic_vocal_compression"]["selected_parameters"][
        "dynamics_peak_to_median_db"
    ] == 16.0
    assert results["soundcheck_dynamic_vocal_compression"]["selected_parameters"][
        "detector"
    ] == "program_rms_peak_blend"
    assert results["soundcheck_dynamic_vocal_compression"]["selected_parameters"][
        "wing_compressor_detector_osc_value"
    ] == "RMS"
    assert results["soundcheck_dynamic_vocal_compression"]["selected_parameters"][
        "wing_compressor_model_readback_must_equal"
    ] == WING_DYNAMICS_REQUIRED_MODEL
    assert results["soundcheck_dynamic_vocal_compression"]["selected_parameters"][
        "wing_compressor_threshold_osc_path_suffix"
    ] == "dyn/thr24"
    assert results["soundcheck_dynamic_vocal_compression"]["selected_parameters"][
        "wing_compressor_threshold_osc_value"
    ] == results["soundcheck_dynamic_vocal_compression"]["selected_parameters"][
        "threshold_dbfs"
    ]
    assert results["soundcheck_dynamic_vocal_compression"]["selected_parameters"][
        "compression_mix_percent"
    ] == 83.5
    assert results["soundcheck_dynamic_vocal_compression"]["selected_parameters"][
        "wing_compressor_mix_osc_path_suffix"
    ] == "dyn/mix"
    assert results["soundcheck_dynamic_vocal_compression"]["selected_parameters"][
        "wing_compressor_mix_osc_value"
    ] == 83.5
    assert results["soundcheck_dynamic_vocal_compression"]["selected_parameters"][
        "compressor_envelope_curve"
    ] == "smooth_log"
    assert results["soundcheck_dynamic_vocal_compression"]["selected_parameters"][
        "wing_compressor_envelope_curve_osc_path_suffix"
    ] == "dyn/env"
    assert results["soundcheck_dynamic_vocal_compression"]["selected_parameters"][
        "wing_compressor_envelope_curve_osc_value"
    ] == "LOG"
    assert results["soundcheck_dynamic_vocal_compression"]["selected_parameters"][
        "compressor_auto_timing_enabled"
    ] is False
    assert results["soundcheck_dynamic_vocal_compression"]["selected_parameters"][
        "compressor_auto_timing_source"
    ] == "engine_calculated_source_dynamics_timing_policy"
    assert results["soundcheck_dynamic_vocal_compression"]["selected_parameters"][
        "wing_compressor_auto_osc_path_suffix"
    ] == "dyn/auto"
    assert results["soundcheck_dynamic_vocal_compression"]["selected_parameters"][
        "wing_compressor_auto_osc_value"
    ] == WING_COMPRESSOR_AUTO_MIN
    assert results["soundcheck_dynamic_vocal_compression"]["selected_parameters"][
        "wing_compressor_auto_value_type"
    ] == "integer_bool"
    assert results["soundcheck_dynamic_vocal_compression"]["selected_parameters"][
        "compressor_crossover_type"
    ] == "OFF"
    assert results["soundcheck_dynamic_vocal_compression"]["selected_parameters"][
        "compressor_crossover_depth_db"
    ] == 0.0
    assert results["soundcheck_dynamic_vocal_compression"]["selected_parameters"][
        "compressor_crossover_frequency_hz"
    ] == 1000.0
    assert results["soundcheck_dynamic_vocal_compression"]["selected_parameters"][
        "wing_compressor_crossover_type_osc_path_suffix"
    ] == "dynxo/type"
    assert results["soundcheck_dynamic_vocal_compression"]["selected_parameters"][
        "wing_compressor_crossover_type_osc_value"
    ] in WING_COMPRESSOR_CROSSOVER_TYPE_VALUES
    assert results["soundcheck_dynamic_vocal_compression"]["selected_parameters"][
        "wing_compressor_crossover_type_osc_value"
    ] == "OFF"
    assert results["soundcheck_dynamic_vocal_compression"]["selected_parameters"][
        "wing_compressor_crossover_depth_osc_path_suffix"
    ] == "dynxo/depth"
    assert (
        WING_COMPRESSOR_CROSSOVER_DEPTH_MIN_DB
        <= results["soundcheck_dynamic_vocal_compression"]["selected_parameters"][
            "wing_compressor_crossover_depth_osc_value"
        ]
        <= WING_COMPRESSOR_CROSSOVER_DEPTH_MAX_DB
    )
    assert results["soundcheck_dynamic_vocal_compression"]["selected_parameters"][
        "wing_compressor_crossover_depth_osc_value"
    ] == 0.0
    assert results["soundcheck_dynamic_vocal_compression"]["selected_parameters"][
        "wing_compressor_crossover_frequency_osc_path_suffix"
    ] == "dynxo/f"
    assert (
        WING_COMPRESSOR_CROSSOVER_FREQ_MIN_HZ
        <= results["soundcheck_dynamic_vocal_compression"]["selected_parameters"][
            "wing_compressor_crossover_frequency_osc_value"
        ]
        <= WING_COMPRESSOR_CROSSOVER_FREQ_MAX_HZ
    )
    assert results["soundcheck_dynamic_vocal_compression"]["selected_parameters"][
        "wing_compressor_crossover_frequency_osc_value"
    ] == 1000.0
    assert results["soundcheck_dynamic_vocal_compression"]["selected_parameters"][
        "compressor_sidechain_filter_type"
    ] == "HP12"
    assert results["soundcheck_dynamic_vocal_compression"]["selected_parameters"][
        "compressor_sidechain_frequency_hz"
    ] == 138.5
    assert results["soundcheck_dynamic_vocal_compression"]["selected_parameters"][
        "compressor_sidechain_q"
    ] == 1.0
    assert results["soundcheck_dynamic_vocal_compression"]["selected_parameters"][
        "compressor_sidechain_parameter_source"
    ] == "measured_low_end_pressure_detector_high_pass"
    assert results["soundcheck_dynamic_vocal_compression"]["selected_parameters"][
        "compressor_sidechain_evidence_frequency_hz"
    ] == 96.0
    assert results["soundcheck_dynamic_vocal_compression"]["selected_parameters"][
        "compressor_sidechain_low_end_pressure_score"
    ] == 0.72
    assert results["soundcheck_dynamic_vocal_compression"]["selected_parameters"][
        "compressor_sidechain_source"
    ] == "SELF"
    assert results["soundcheck_dynamic_vocal_compression"]["selected_parameters"][
        "compressor_sidechain_tap"
    ] == "IN"
    assert results["soundcheck_dynamic_vocal_compression"]["selected_parameters"][
        "wing_compressor_sidechain_type_osc_path_suffix"
    ] == "dynsc/type"
    assert results["soundcheck_dynamic_vocal_compression"]["selected_parameters"][
        "wing_compressor_sidechain_type_osc_value"
    ] in WING_COMPRESSOR_SIDECHAIN_TYPE_VALUES
    assert results["soundcheck_dynamic_vocal_compression"]["selected_parameters"][
        "wing_compressor_sidechain_type_osc_value"
    ] == "HP12"
    assert results["soundcheck_dynamic_vocal_compression"]["selected_parameters"][
        "wing_compressor_sidechain_frequency_osc_path_suffix"
    ] == "dynsc/f"
    assert (
        WING_COMPRESSOR_SIDECHAIN_FREQ_MIN_HZ
        <= results["soundcheck_dynamic_vocal_compression"]["selected_parameters"][
            "wing_compressor_sidechain_frequency_osc_value"
        ]
        <= WING_COMPRESSOR_SIDECHAIN_FREQ_MAX_HZ
    )
    assert results["soundcheck_dynamic_vocal_compression"]["selected_parameters"][
        "wing_compressor_sidechain_frequency_osc_value"
    ] == 138.5
    assert results["soundcheck_dynamic_vocal_compression"]["selected_parameters"][
        "wing_compressor_sidechain_q_osc_path_suffix"
    ] == "dynsc/q"
    assert (
        WING_COMPRESSOR_SIDECHAIN_Q_MIN
        <= results["soundcheck_dynamic_vocal_compression"]["selected_parameters"][
            "wing_compressor_sidechain_q_osc_value"
        ]
        <= WING_COMPRESSOR_SIDECHAIN_Q_MAX
    )
    assert results["soundcheck_dynamic_vocal_compression"]["selected_parameters"][
        "wing_compressor_sidechain_q_osc_value"
    ] == 1.0
    assert results["soundcheck_dynamic_vocal_compression"]["selected_parameters"][
        "wing_compressor_sidechain_src_osc_path_suffix"
    ] == "dynsc/src"
    assert results["soundcheck_dynamic_vocal_compression"]["selected_parameters"][
        "wing_compressor_sidechain_src_osc_value"
    ] in WING_COMPRESSOR_SIDECHAIN_SOURCE_VALUES
    assert results["soundcheck_dynamic_vocal_compression"]["selected_parameters"][
        "wing_compressor_sidechain_src_osc_value"
    ] == "SELF"
    assert results["soundcheck_dynamic_vocal_compression"]["selected_parameters"][
        "wing_compressor_sidechain_tap_osc_path_suffix"
    ] == "dynsc/tap"
    assert results["soundcheck_dynamic_vocal_compression"]["selected_parameters"][
        "wing_compressor_sidechain_tap_osc_value"
    ] in WING_COMPRESSOR_SIDECHAIN_TAP_VALUES
    assert results["soundcheck_dynamic_vocal_compression"]["selected_parameters"][
        "wing_compressor_sidechain_tap_osc_value"
    ] == "IN"
    dynamic_vocal_readback = results["soundcheck_dynamic_vocal_compression"][
        "selected_future_wing_readback_contract"
    ]["dynamics_readback"]
    dynamic_vocal_wing_osc = results["soundcheck_dynamic_vocal_compression"][
        "selected_rationale"
    ]
    assert dynamic_vocal_wing_osc["wing_osc_candidate_apply_status"] == (
        "wing_native_osc_candidate_has_review_only_parameters_not_apply_ready_"
        "requires_operator_readback"
    )
    assert dynamic_vocal_wing_osc["wing_osc_review_only_parameter_keys"] == [
        "target_gain_reduction_db"
    ]
    assert dynamic_vocal_wing_osc[
        "wing_osc_review_only_parameters_apply_ready"
    ] is False
    dynamic_vocal_params = results["soundcheck_dynamic_vocal_compression"][
        "selected_parameters"
    ]
    assert (
        dynamic_vocal_params["soundcheck_dynamics_bounded_compression_proposal"]
        is True
    )
    assert (
        dynamic_vocal_params["soundcheck_dynamics_native_compressor_osc_only_required"]
        is True
    )
    assert (
        dynamic_vocal_params["soundcheck_dynamics_max_target_gain_reduction_db"]
        == SOUNDCHECK_DYNAMICS_MAX_TARGET_GAIN_REDUCTION_DB
    )
    assert (
        dynamic_vocal_params["soundcheck_dynamics_max_makeup_gain_db"]
        == SOUNDCHECK_DYNAMICS_MAX_MAKEUP_GAIN_DB
    )
    assert (
        dynamic_vocal_params["soundcheck_dynamics_min_headroom_margin_db"]
        == SOUNDCHECK_DYNAMICS_MIN_HEADROOM_MARGIN_DB
    )
    assert (
        dynamic_vocal_params["soundcheck_dynamics_required_true_peak_headroom_db"]
        == 1.8
    )
    assert (
        dynamic_vocal_params[
            "soundcheck_dynamics_required_summed_bus_true_peak_headroom_db"
        ]
        == 1.8
    )
    assert dynamic_vocal_params["soundcheck_dynamics_target_gr_review_only"] is True
    assert (
        dynamic_vocal_params["soundcheck_dynamics_gain_reduction_meter_readback_required"]
        is True
    )
    assert (
        dynamic_vocal_params["soundcheck_dynamics_live_apply_blocked_without_wing_proof"]
        is True
    )
    assert (
        dynamic_vocal_readback["soundcheck_dynamics_bounded_compression_proposal"]
        is True
    )
    assert (
        dynamic_vocal_readback[
            "soundcheck_dynamics_native_compressor_osc_only_required"
        ]
        is True
    )
    assert (
        dynamic_vocal_readback["soundcheck_dynamics_max_target_gain_reduction_db"]
        == SOUNDCHECK_DYNAMICS_MAX_TARGET_GAIN_REDUCTION_DB
    )
    assert (
        dynamic_vocal_readback["soundcheck_dynamics_max_makeup_gain_db"]
        == SOUNDCHECK_DYNAMICS_MAX_MAKEUP_GAIN_DB
    )
    assert (
        dynamic_vocal_readback["soundcheck_dynamics_required_true_peak_headroom_db"]
        == 1.8
    )
    assert (
        dynamic_vocal_readback[
            "soundcheck_dynamics_required_summed_bus_true_peak_headroom_db"
        ]
        == 1.8
    )
    assert (
        dynamic_vocal_readback[
            "soundcheck_dynamics_live_apply_blocked_without_wing_proof"
        ]
        is True
    )
    _assert_replay_selected_write_key_contract(
        results["soundcheck_dynamic_vocal_compression"],
        readback_section="dynamics_readback",
        prefix="wing_dynamics",
        selected_write_keys=list(WING_DYNAMICS_WRITE_PARAMETER_KEYS),
        write_parameter_keys=WING_DYNAMICS_WRITE_PARAMETER_KEYS,
        selection_policy=WING_DYNAMICS_WRITE_KEY_SELECTION_POLICY,
    )
    _assert_replay_selected_parameter_rollback_operations(
        results["soundcheck_dynamic_vocal_compression"],
        readback_section="dynamics_readback",
        operations_field="wing_dynamics_selected_parameter_rollback_operations",
        selected_write_keys=list(WING_DYNAMICS_WRITE_PARAMETER_KEYS),
        contract_marker=(
            "read_previous_dynamics_value_before_write_then_restore_same_path"
        ),
    )
    _assert_replay_selected_dynamics_apply_values(
        results["soundcheck_dynamic_vocal_compression"],
        {
            "wing_compressor_on_osc_value": WING_COMPRESSOR_ON_MAX,
            "wing_compressor_threshold_osc_value": dynamic_vocal_params[
                "wing_compressor_threshold_osc_value"
            ],
            "wing_compressor_attack_osc_value": dynamic_vocal_params[
                "wing_compressor_attack_osc_value"
            ],
            "wing_compressor_hold_osc_value": dynamic_vocal_params[
                "wing_compressor_hold_osc_value"
            ],
            "wing_compressor_release_osc_value": dynamic_vocal_params[
                "wing_compressor_release_osc_value"
            ],
            "wing_compressor_makeup_gain_osc_value": dynamic_vocal_params[
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
            "wing_compressor_sidechain_frequency_osc_value": 138.5,
            "wing_compressor_sidechain_q_osc_value": 1.0,
            "wing_compressor_sidechain_src_osc_value": "SELF",
            "wing_compressor_sidechain_tap_osc_value": "IN",
        },
    )
    assert dynamic_vocal_readback["dynamics_model_readback_must_equal"] == (
        WING_DYNAMICS_REQUIRED_MODEL
    )
    assert "dyn/mdl" in dynamic_vocal_readback["read_before_paths"]
    assert "dyn/on" in dynamic_vocal_readback["read_before_paths"]
    assert "dyn/mix" in dynamic_vocal_readback["read_before_paths"]
    assert "dyn/env" in dynamic_vocal_readback["read_before_paths"]
    assert "dyn/auto" in dynamic_vocal_readback["read_before_paths"]
    assert "dynxo/depth" in dynamic_vocal_readback["read_before_paths"]
    assert "dynxo/type" in dynamic_vocal_readback["read_before_paths"]
    assert "dynxo/f" in dynamic_vocal_readback["read_before_paths"]
    assert "dynsc/type" in dynamic_vocal_readback["read_before_paths"]
    assert "dynsc/f" in dynamic_vocal_readback["read_before_paths"]
    assert "dynsc/q" in dynamic_vocal_readback["read_before_paths"]
    assert "dynsc/src" in dynamic_vocal_readback["read_before_paths"]
    assert "dynsc/tap" in dynamic_vocal_readback["read_before_paths"]
    assert dynamic_vocal_readback["wing_compressor_on_mapping_required"] is True
    assert dynamic_vocal_readback["wing_compressor_on_osc_path_suffix"] == "dyn/on"
    assert dynamic_vocal_readback["wing_compressor_on_osc_value"] == (
        WING_COMPRESSOR_ON_MAX
    )
    assert (
        dynamic_vocal_readback["wing_compressor_on_readback_must_match_osc_value"]
        is True
    )
    assert (
        dynamic_vocal_readback["wing_compressor_threshold_osc_value"]
        == results["soundcheck_dynamic_vocal_compression"]["selected_parameters"][
            "wing_compressor_threshold_osc_value"
        ]
    )
    assert (
        dynamic_vocal_readback[
            "wing_compressor_threshold_readback_must_match_osc_value"
        ]
        is True
    )
    assert dynamic_vocal_readback["wing_compressor_mix_mapping_required"] is True
    assert dynamic_vocal_readback["wing_compressor_mix_osc_path_suffix"] == "dyn/mix"
    assert dynamic_vocal_readback["wing_compressor_mix_osc_value"] == 83.5
    assert (
        dynamic_vocal_readback["wing_compressor_mix_readback_must_match_osc_value"]
        is True
    )
    assert (
        dynamic_vocal_readback["wing_compressor_envelope_curve_mapping_required"]
        is True
    )
    assert (
        dynamic_vocal_readback["wing_compressor_envelope_curve_osc_path_suffix"]
        == "dyn/env"
    )
    assert dynamic_vocal_readback["wing_compressor_envelope_curve_osc_value"] == "LOG"
    assert (
        dynamic_vocal_readback[
            "wing_compressor_envelope_curve_readback_must_match_osc_value"
        ]
        is True
    )
    assert dynamic_vocal_readback["wing_compressor_auto_mapping_required"] is True
    assert dynamic_vocal_readback["wing_compressor_auto_osc_path_suffix"] == "dyn/auto"
    assert dynamic_vocal_readback["wing_compressor_auto_osc_value"] == (
        WING_COMPRESSOR_AUTO_MIN
    )
    assert (
        dynamic_vocal_readback["wing_compressor_auto_readback_must_match_osc_value"]
        is True
    )
    assert dynamic_vocal_readback[
        "wing_compressor_crossover_mapping_required"
    ] is True
    assert dynamic_vocal_readback[
        "wing_compressor_crossover_type_osc_path_suffix"
    ] == "dynxo/type"
    assert dynamic_vocal_readback[
        "wing_compressor_crossover_type_osc_value"
    ] == "OFF"
    assert dynamic_vocal_readback[
        "wing_compressor_crossover_depth_osc_path_suffix"
    ] == "dynxo/depth"
    assert dynamic_vocal_readback[
        "wing_compressor_crossover_depth_osc_value"
    ] == 0.0
    assert dynamic_vocal_readback[
        "wing_compressor_crossover_frequency_osc_path_suffix"
    ] == "dynxo/f"
    assert dynamic_vocal_readback[
        "wing_compressor_crossover_frequency_osc_value"
    ] == 1000.0
    assert (
        dynamic_vocal_readback[
            "wing_compressor_crossover_readback_must_match_osc_value"
        ]
        is True
    )
    assert dynamic_vocal_readback[
        "wing_compressor_sidechain_mapping_required"
    ] is True
    assert dynamic_vocal_readback[
        "wing_compressor_sidechain_type_osc_path_suffix"
    ] == "dynsc/type"
    assert dynamic_vocal_readback[
        "wing_compressor_sidechain_type_osc_value"
    ] == "HP12"
    assert dynamic_vocal_readback[
        "wing_compressor_sidechain_frequency_osc_path_suffix"
    ] == "dynsc/f"
    assert dynamic_vocal_readback[
        "wing_compressor_sidechain_frequency_osc_value"
    ] == 138.5
    assert dynamic_vocal_readback[
        "wing_compressor_sidechain_q_osc_path_suffix"
    ] == "dynsc/q"
    assert dynamic_vocal_readback["wing_compressor_sidechain_q_osc_value"] == 1.0
    assert dynamic_vocal_readback[
        "wing_compressor_sidechain_src_osc_path_suffix"
    ] == "dynsc/src"
    assert dynamic_vocal_readback[
        "wing_compressor_sidechain_src_osc_value"
    ] == "SELF"
    assert dynamic_vocal_readback[
        "wing_compressor_sidechain_tap_osc_path_suffix"
    ] == "dynsc/tap"
    assert dynamic_vocal_readback[
        "wing_compressor_sidechain_tap_osc_value"
    ] == "IN"
    assert (
        dynamic_vocal_readback[
            "wing_compressor_sidechain_readback_must_match_osc_value"
        ]
        is True
    )
    assert results["soundcheck_dynamic_vocal_compression"]["selected_parameters"][
        "wing_compressor_detector_value_type"
    ] == "string_enum"
    assert results["soundcheck_dynamic_vocal_compression"]["selected_parameters"][
        "wing_compressor_knee_osc_value"
    ] == 4
    assert results["soundcheck_dynamic_vocal_compression"]["selected_parameters"][
        "knee_db"
    ] == 4.1
    assert results["soundcheck_dynamic_vocal_compression"]["selected_parameters"][
        "compressor_knee_parameter_source"
    ] == "measured_envelope_variance_peak_to_median"
    assert results["soundcheck_dynamic_vocal_compression"]["selected_parameters"][
        "wing_compressor_knee_quantized_from"
    ] == 4.1
    assert results["soundcheck_dynamic_vocal_compression"]["selected_parameters"][
        "wing_compressor_knee_value_type"
    ] == "integer"
    assert results["soundcheck_dynamic_vocal_compression"]["selected_parameters"][
        "preserve_words_and_consonants"
    ] is True
    assert results["soundcheck_dynamic_vocal_summed_bus_makeup_hold"][
        "actual_selected_strategy_id"
    ] == "hold_compressor_makeup_for_summed_bus_headroom"
    assert results["soundcheck_dynamic_vocal_summed_bus_makeup_hold"][
        "actual_method_family"
    ] == "analysis_only"
    assert results["soundcheck_dynamic_vocal_summed_bus_makeup_hold"][
        "operator_explanation_sections"
    ]["selection_policy"] == (
        "soundcheck_summed_bus_headroom_before_compressor_makeup_priority"
    )
    summed_bus_makeup_params = results[
        "soundcheck_dynamic_vocal_summed_bus_makeup_hold"
    ]["selected_parameters"]
    assert summed_bus_makeup_params["candidate_makeup_gain_db"] == 1.0
    assert summed_bus_makeup_params["true_peak_headroom_db"] == 4.0
    assert summed_bus_makeup_params["summed_bus_role"] == "lead_vocal_bus"
    assert summed_bus_makeup_params["summed_bus_true_peak_headroom_db"] == 1.2
    assert summed_bus_makeup_params[
        "requires_summed_bus_true_peak_headroom_db"
    ] == 2.0
    assert summed_bus_makeup_params["blocked_wing_osc_path_suffix"] == "dyn/gain"
    assert results["soundcheck_dynamic_vocal_summed_bus_makeup_hold"][
        "candidate_blockers"
    ]["vocal_phrase_leveling_compression"] == [
        "summed_bus_headroom_blocks_compressor_makeup"
    ]
    assert results["soundcheck_dynamic_vocal_summed_bus_makeup_hold"][
        "candidate_blockers"
    ]["gentle_channel_compression"] == [
        "summed_bus_headroom_blocks_compressor_makeup"
    ]
    makeup_tradeoffs = {
        item["strategy_id"]: item
        for item in results["soundcheck_dynamic_vocal_summed_bus_makeup_hold"][
            "alternative_tradeoffs"
        ]
    }
    assert makeup_tradeoffs["vocal_phrase_leveling_compression"][
        "why_not_selected"
    ] == (
        "soundcheck_summed_bus_compressor_makeup_priority:"
        "verify_summed_bus_headroom_before_wing_dyn_gain"
    )
    _assert_replay_analysis_only_has_no_apply_values(
        results["soundcheck_dynamic_vocal_summed_bus_makeup_hold"]
    )
    assert results["soundcheck_dynamic_vocal_feedback_path_hold"][
        "actual_selected_strategy_id"
    ] == "hold_dynamics_for_feedback_path_review"
    assert results["soundcheck_dynamic_vocal_feedback_path_hold"][
        "actual_method_family"
    ] == "analysis_only"
    assert results["soundcheck_dynamic_vocal_feedback_path_hold"][
        "operator_explanation_sections"
    ]["selection_policy"] == "soundcheck_dynamics_feedback_path_before_compression_priority"
    assert results["soundcheck_dynamic_vocal_feedback_path_hold"][
        "selected_parameters"
    ]["feedback_path_review_threshold"] == 0.45
    assert results["soundcheck_dynamic_vocal_feedback_path_hold"][
        "selected_parameters"
    ]["candidate_makeup_gain_db"] == 1.0
    assert results["soundcheck_dynamic_vocal_feedback_path_hold"][
        "candidate_blockers"
    ]["vocal_phrase_leveling_compression"] == [
        "feedback_path_review_should_precede_vocal_compression"
    ]
    assert results["soundcheck_dynamic_vocal_feedback_path_hold"][
        "candidate_blockers"
    ]["gentle_channel_compression"] == [
        "feedback_path_review_should_precede_compression_makeup"
    ]
    tradeoffs = {
        item["strategy_id"]: item
        for item in results["soundcheck_dynamic_vocal_feedback_path_hold"][
            "alternative_tradeoffs"
        ]
    }
    assert tradeoffs["vocal_phrase_leveling_compression"]["why_not_selected"] == (
        "soundcheck_dynamics_feedback_path_priority:"
        "feedback_path_review_precedes_compression_or_makeup"
    )
    assert results["soundcheck_master_true_peak_beats_source_vocal_compression"][
        "actual_selected_strategy_id"
    ] == "master_trim_for_true_peak_headroom"
    assert results["soundcheck_master_true_peak_beats_source_vocal_compression"][
        "actual_method_family"
    ] == "gain_fader"
    assert results["soundcheck_master_true_peak_beats_source_vocal_compression"][
        "operator_explanation_sections"
    ]["selection_policy"] == "master_true_peak_headroom_priority"
    assert results["soundcheck_master_true_peak_beats_source_vocal_compression"][
        "selected_parameters"
    ]["master_true_peak_pullback_db"] == 1.3
    _assert_replay_gain_delta_mapping(
        results["soundcheck_master_true_peak_beats_source_vocal_compression"],
        source="trim_delta_db",
        expected_delta=-1.3,
        path_prefix="/main/{main}",
    )
    _assert_replay_soundcheck_master_output_guard(
        results["soundcheck_master_true_peak_beats_source_vocal_compression"],
        move_kind="pullback",
        followup_review=True,
    )
    _assert_replay_master_output_target_override(
        results["soundcheck_master_true_peak_beats_source_vocal_compression"],
        original_target_id="ch:7",
    )
    _assert_replay_selected_gain_delta_apply_values(
        results["soundcheck_master_true_peak_beats_source_vocal_compression"],
        expected_delta=-1.3,
    )
    tradeoffs = {
        item["strategy_id"]: item
        for item in results[
            "soundcheck_master_true_peak_beats_source_vocal_compression"
        ]["alternative_tradeoffs"]
    }
    assert tradeoffs["vocal_phrase_leveling_compression"]["why_not_selected"] == (
        "master_true_peak_headroom_priority:"
        "restore_master_headroom_before_source_tone_dynamics_fx_or_gain"
    )
    assert results["soundcheck_master_true_peak_beats_source_headroom_pullback"][
        "actual_status"
    ] == "analysis_only"
    assert "true_peak_headroom_low" in results[
        "soundcheck_master_true_peak_beats_source_headroom_pullback"
    ]["global_blockers"]
    assert results["soundcheck_master_true_peak_beats_source_headroom_pullback"][
        "actual_selected_strategy_id"
    ] == "master_trim_for_true_peak_headroom"
    assert results["soundcheck_master_true_peak_beats_source_headroom_pullback"][
        "actual_method_family"
    ] == "gain_fader"
    assert results["soundcheck_master_true_peak_beats_source_headroom_pullback"][
        "operator_explanation_sections"
    ]["selection_policy"] == "master_true_peak_headroom_priority"
    assert results["soundcheck_master_true_peak_beats_source_headroom_pullback"][
        "selected_parameters"
    ]["headroom_deficit_pullback_db"] == 0.9
    _assert_replay_gain_delta_mapping(
        results["soundcheck_master_true_peak_beats_source_headroom_pullback"],
        source="trim_delta_db",
        expected_delta=-1.3,
        path_prefix="/main/{main}",
    )
    _assert_replay_soundcheck_master_output_guard(
        results["soundcheck_master_true_peak_beats_source_headroom_pullback"],
        move_kind="pullback",
        followup_review=True,
    )
    _assert_replay_master_output_target_override(
        results["soundcheck_master_true_peak_beats_source_headroom_pullback"],
        original_target_id="ch:7",
    )
    _assert_replay_selected_gain_delta_apply_values(
        results["soundcheck_master_true_peak_beats_source_headroom_pullback"],
        expected_delta=-1.3,
    )
    tradeoffs = {
        item["strategy_id"]: item
        for item in results[
            "soundcheck_master_true_peak_beats_source_headroom_pullback"
        ]["alternative_tradeoffs"]
    }
    assert tradeoffs["source_true_peak_headroom_pullback"]["why_not_selected"] == (
        "master_true_peak_headroom_priority:"
        "restore_master_headroom_before_source_tone_dynamics_fx_or_gain"
    )
    assert results["live_vocal_dynamics_ride_before_compression"][
        "actual_selected_strategy_id"
    ] == "manual_level_ride_before_compression"
    assert results["live_vocal_dynamics_ride_before_compression"][
        "actual_method_family"
    ] == "analysis_only"
    assert results["live_vocal_dynamics_ride_before_compression"][
        "operator_explanation_sections"
    ]["selection_policy"] == "live_vocal_dynamics_ride_priority"
    assert results["live_vocal_dynamics_ride_before_compression"][
        "selected_parameters"
    ]["live_vocal_dynamics_ride_priority"] is True
    assert results["live_vocal_dynamics_ride_before_compression"][
        "selected_parameters"
    ]["forbidden_without_wing_proof"] == "compressor_peak_clamp_makeup_gain"
    live_vocal_ride = results["live_vocal_dynamics_ride_before_compression"]
    assert live_vocal_ride["selected_parameters"][
        "live_dynamics_ride_hold_proposal"
    ] is True
    assert live_vocal_ride["selected_parameters"][
        "live_dynamics_ride_profile"
    ] == "lead_vocal_phrase"
    assert live_vocal_ride["selected_parameters"][
        "live_dynamics_ride_previous_gain_readback_required"
    ] is True
    assert live_vocal_ride["selected_parameters"][
        "live_dynamics_ride_previous_compressor_readback_required"
    ] is True
    assert live_vocal_ride["selected_parameters"][
        "live_dynamics_ride_no_compressor_or_automation_without_operator_wing_proof"
    ] is True
    live_vocal_ride_readback = live_vocal_ride[
        "selected_future_wing_readback_contract"
    ]["live_dynamics_ride_hold_readback"]
    assert live_vocal_ride_readback["read_only"] is True
    assert live_vocal_ride_readback[
        "allowed_live_write_scope_before_wing_proof"
    ] == []
    assert "/ch/{channel}/fdr" in live_vocal_ride_readback[
        "wing_gain_osc_path_templates"
    ]
    assert "/ch/{channel}/in/set/trim" in live_vocal_ride_readback[
        "wing_gain_osc_path_templates"
    ]
    assert "/ch/{channel}/dyn/thr24" in live_vocal_ride_readback[
        "wing_dynamics_osc_path_templates"
    ]
    assert "dyn/thr24" in live_vocal_ride_readback["read_before_paths"]
    assert "wing_gain_delta_db" not in live_vocal_ride["selected_parameters"]
    assert "wing_compressor_threshold_osc_value" not in live_vocal_ride[
        "selected_parameters"
    ]
    _assert_replay_analysis_only_has_no_apply_values(live_vocal_ride)
    assert results["live_vocal_dynamics_ride_before_compression"][
        "candidate_blockers"
    ]["vocal_phrase_leveling_compression"] == [
        "live_vocal_leveling_compression_requires_operator_review_and_wing_proof"
    ]
    assert results["live_vocal_dynamics_ride_before_compression"][
        "candidate_blockers"
    ]["gentle_channel_compression"] == [
        "live_vocal_compression_requires_operator_review_and_wing_proof",
        "live_dynamics_target_gr_exceeds_conservative_limit",
    ]
    assert results["live_operator_owned_dynamics_manual_ride"][
        "actual_selected_strategy_id"
    ] == "manual_level_ride_before_compression"
    assert results["live_operator_owned_dynamics_manual_ride"][
        "actual_method_family"
    ] == "analysis_only"
    assert results["live_operator_owned_dynamics_manual_ride"][
        "operator_explanation_sections"
    ]["selection_policy"] == "live_operator_owned_dynamics_ride_priority"
    live_owned_dynamics_ride = results["live_operator_owned_dynamics_manual_ride"]
    owned_dynamics_params = live_owned_dynamics_ride["selected_parameters"]
    assert owned_dynamics_params["live_operator_owned_dynamics_ride_priority"] is True
    assert owned_dynamics_params["live_operator_owned_dynamics_review_reason"] == (
        "current_wing_dynamics_operator_owned"
    )
    assert (
        owned_dynamics_params["live_operator_owned_dynamics_current_operator_owned"]
        is True
    )
    assert owned_dynamics_params["live_dynamics_ride_hold_proposal"] is True
    assert owned_dynamics_params["live_dynamics_ride_profile"] == (
        "operator_owned_compressor"
    )
    assert (
        owned_dynamics_params["live_dynamics_ride_previous_compressor_readback_required"]
        is True
    )
    assert (
        owned_dynamics_params["live_dynamics_ride_gain_reduction_meter_readback_required"]
        is True
    )
    assert owned_dynamics_params["live_dynamics_ride_rollback_ready"] is True
    assert owned_dynamics_params["required_context"] == (
        "operator_owned_compressor_manual_ride_dynamics_readback"
    )
    assert owned_dynamics_params["forbidden_without_wing_proof"] == (
        "operator_owned_compressor_change"
    )
    assert live_owned_dynamics_ride["candidate_blockers"][
        "vocal_phrase_leveling_compression"
    ] == [WING_DYNAMICS_OPERATOR_OWNED_MODEL_BLOCKER]
    assert WING_DYNAMICS_OPERATOR_OWNED_MODEL_BLOCKER in (
        live_owned_dynamics_ride["candidate_blockers"]["gentle_channel_compression"]
    )
    _assert_replay_analysis_only_has_no_apply_values(live_owned_dynamics_ride)
    owned_dynamics_tradeoffs = {
        item["strategy_id"]: item["why_not_selected"]
        for item in live_owned_dynamics_ride["alternative_tradeoffs"]
    }
    assert owned_dynamics_tradeoffs["vocal_phrase_leveling_compression"] == (
        "live_operator_owned_dynamics_ride_priority:"
        "manual_level_ride_precedes_operator_owned_compressor_change"
    )
    assert owned_dynamics_tradeoffs["gentle_channel_compression"] == (
        "live_operator_owned_dynamics_ride_priority:"
        "manual_level_ride_precedes_operator_owned_compressor_change"
    )
    assert results["soundcheck_operator_owned_dynamics_manual_ride"][
        "actual_selected_strategy_id"
    ] == "manual_level_ride_before_compression"
    assert results["soundcheck_operator_owned_dynamics_manual_ride"][
        "actual_method_family"
    ] == "analysis_only"
    assert results["soundcheck_operator_owned_dynamics_manual_ride"][
        "operator_explanation_sections"
    ]["selection_policy"] == "soundcheck_operator_owned_dynamics_ride_priority"
    soundcheck_owned_dynamics_ride = results[
        "soundcheck_operator_owned_dynamics_manual_ride"
    ]
    soundcheck_owned_params = soundcheck_owned_dynamics_ride["selected_parameters"]
    assert (
        soundcheck_owned_params["soundcheck_operator_owned_dynamics_ride_priority"]
        is True
    )
    assert soundcheck_owned_params[
        "soundcheck_operator_owned_dynamics_review_reason"
    ] == "current_wing_dynamics_operator_owned"
    assert (
        soundcheck_owned_params[
            "soundcheck_operator_owned_dynamics_current_operator_owned"
        ]
        is True
    )
    assert soundcheck_owned_params["soundcheck_dynamics_ride_hold_proposal"] is True
    assert soundcheck_owned_params["soundcheck_dynamics_ride_profile"] == (
        "operator_owned_compressor"
    )
    assert (
        soundcheck_owned_params[
            "soundcheck_dynamics_ride_previous_compressor_readback_required"
        ]
        is True
    )
    assert (
        soundcheck_owned_params[
            "soundcheck_dynamics_ride_gain_reduction_meter_readback_required"
        ]
        is True
    )
    assert soundcheck_owned_params["soundcheck_dynamics_ride_rollback_ready"] is True
    assert soundcheck_owned_params["required_context"] == (
        "soundcheck_operator_owned_compressor_manual_ride_dynamics_readback"
    )
    assert soundcheck_owned_params["forbidden_without_wing_proof"] == (
        "soundcheck_operator_owned_compressor_change"
    )
    assert soundcheck_owned_dynamics_ride["candidate_blockers"][
        "vocal_phrase_leveling_compression"
    ] == [WING_DYNAMICS_OPERATOR_OWNED_MODEL_BLOCKER]
    assert WING_DYNAMICS_OPERATOR_OWNED_MODEL_BLOCKER in (
        soundcheck_owned_dynamics_ride["candidate_blockers"][
            "gentle_channel_compression"
        ]
    )
    soundcheck_owned_readback = soundcheck_owned_dynamics_ride[
        "selected_future_wing_readback_contract"
    ]["soundcheck_operator_owned_dynamics_ride_readback"]
    assert soundcheck_owned_readback["read_only"] is True
    assert soundcheck_owned_readback["allowed_live_write_scope_before_wing_proof"] == []
    assert "/ch/{channel}/fdr" in soundcheck_owned_readback[
        "wing_gain_osc_path_templates"
    ]
    assert "/ch/{channel}/dyn/thr24" in soundcheck_owned_readback[
        "wing_dynamics_osc_path_templates"
    ]
    assert "dyn/thr24" in soundcheck_owned_readback["read_before_paths"]
    _assert_replay_analysis_only_has_no_apply_values(soundcheck_owned_dynamics_ride)
    soundcheck_owned_tradeoffs = {
        item["strategy_id"]: item["why_not_selected"]
        for item in soundcheck_owned_dynamics_ride["alternative_tradeoffs"]
    }
    assert soundcheck_owned_tradeoffs["vocal_phrase_leveling_compression"] == (
        "soundcheck_operator_owned_dynamics_ride_priority:"
        "manual_level_ride_precedes_operator_owned_compressor_change"
    )
    assert soundcheck_owned_tradeoffs["gentle_channel_compression"] == (
        "soundcheck_operator_owned_dynamics_ride_priority:"
        "manual_level_ride_precedes_operator_owned_compressor_change"
    )
    assert results["live_keys_swell_ride_before_compression"][
        "actual_selected_strategy_id"
    ] == "manual_level_ride_before_compression"
    assert results["live_keys_swell_ride_before_compression"][
        "actual_method_family"
    ] == "analysis_only"
    assert results["live_keys_swell_ride_before_compression"][
        "operator_explanation_sections"
    ]["selection_policy"] == "live_keys_swell_ride_priority"
    assert results["live_keys_swell_ride_before_compression"][
        "selected_parameters"
    ]["live_keys_swell_ride_priority"] is True
    assert results["live_keys_swell_ride_before_compression"][
        "selected_parameters"
    ]["live_vocal_dynamics_ride_priority"] is False
    assert results["live_keys_swell_ride_before_compression"][
        "selected_parameters"
    ]["max_ride_delta_db"] == 1.5
    assert results["live_keys_swell_ride_before_compression"][
        "selected_parameters"
    ]["forbidden_without_wing_proof"] == "keys_swell_compressor_change"
    live_keys_ride = results["live_keys_swell_ride_before_compression"]
    assert live_keys_ride["selected_parameters"][
        "live_dynamics_ride_hold_proposal"
    ] is True
    assert live_keys_ride["selected_parameters"][
        "live_dynamics_ride_profile"
    ] == "keys_swell"
    live_keys_ride_readback = live_keys_ride[
        "selected_future_wing_readback_contract"
    ]["live_dynamics_ride_hold_readback"]
    assert live_keys_ride_readback["read_only"] is True
    assert live_keys_ride_readback["live_dynamics_ride_profile"] == "keys_swell"
    assert live_keys_ride_readback[
        "allowed_live_write_scope_before_wing_proof"
    ] == []
    assert "/ch/{channel}/fdr" in live_keys_ride_readback[
        "wing_gain_osc_path_templates"
    ]
    assert "/ch/{channel}/dyn/thr24" in live_keys_ride_readback[
        "wing_dynamics_osc_path_templates"
    ]
    _assert_replay_analysis_only_has_no_apply_values(live_keys_ride)
    assert results["live_keys_swell_ride_before_compression"][
        "candidate_blockers"
    ]["keys_swell_leveling_compression"] == [
        "live_keys_swell_compression_requires_operator_review_and_wing_proof"
    ]
    assert results["live_keys_swell_ride_before_compression"][
        "candidate_blockers"
    ]["gentle_channel_compression"] == [
        "live_keys_swell_compression_requires_operator_review_and_wing_proof"
    ]
    tradeoffs = {
        item["strategy_id"]: item["why_not_selected"]
        for item in results["live_keys_swell_ride_before_compression"][
            "alternative_tradeoffs"
        ]
    }
    assert tradeoffs["keys_swell_leveling_compression"] == (
        "live_keys_swell_ride_priority:preserve_swell_before_compressor_change"
    )
    assert tradeoffs["gentle_channel_compression"] == (
        "live_keys_swell_ride_priority:preserve_swell_before_compressor_change"
    )
    assert results["soundcheck_dynamic_vocal_level_plane_before_compression"][
        "actual_selected_strategy_id"
    ] == "soundcheck_bounded_gain_up"
    assert results["soundcheck_dynamic_vocal_level_plane_before_compression"][
        "operator_explanation_sections"
    ]["selection_policy"] == "soundcheck_level_plane_before_dynamics_priority"
    assert results["soundcheck_dynamic_vocal_level_plane_before_compression"][
        "selected_parameters"
    ]["current_minus_target_lu"] == -4.2
    assert results["soundcheck_dynamic_vocal_level_plane_before_compression"][
        "selected_parameters"
    ]["target_scope"] == "source_and_summed_bus"
    assert results["soundcheck_bass_sustain_leveling_compression"][
        "actual_selected_strategy_id"
    ] == "bass_sustain_leveling_compression"
    assert results["soundcheck_bass_sustain_leveling_compression"][
        "selected_parameters"
    ]["preserve_fundamental"] is True
    assert results["soundcheck_bass_sustain_leveling_compression"][
        "selected_parameters"
    ]["dynamics_parameter_source"] == "measured_envelope"
    assert results["soundcheck_bass_sustain_leveling_compression"][
        "selected_parameters"
    ]["wing_compressor_ratio_osc_value"] == "3.0"
    assert results["soundcheck_bass_sustain_leveling_compression"][
        "selected_parameters"
    ]["wing_compressor_ratio_value_type"] == "string_enum"
    assert results["soundcheck_bass_sustain_leveling_compression"][
        "selected_parameters"
    ]["wing_compressor_detector_osc_value"] == "RMS"
    assert results["soundcheck_bass_sustain_leveling_compression"][
        "selected_parameters"
    ]["wing_compressor_detector_value_type"] == "string_enum"
    assert results["soundcheck_bass_sustain_leveling_compression"][
        "selected_parameters"
    ]["wing_compressor_knee_osc_value"] == 4
    assert results["soundcheck_bass_sustain_leveling_compression"][
        "selected_parameters"
    ]["knee_db"] == 4.4
    assert results["soundcheck_bass_sustain_leveling_compression"][
        "selected_parameters"
    ]["compressor_knee_parameter_source"] == (
        "measured_envelope_variance_peak_to_median"
    )
    assert results["soundcheck_bass_sustain_leveling_compression"][
        "selected_parameters"
    ]["wing_compressor_knee_quantized_from"] == 4.4
    assert results["soundcheck_bass_sustain_leveling_compression"][
        "selected_parameters"
    ]["wing_compressor_knee_value_type"] == "integer"
    assert results["soundcheck_bass_hot_bus_rebalance_before_compression"][
        "actual_selected_strategy_id"
    ] == "bass_sustain_summed_bus_rebalance_before_compression"
    bass_hot_bus = results["soundcheck_bass_hot_bus_rebalance_before_compression"]
    assert bass_hot_bus["operator_explanation_sections"]["selection_policy"] == (
        "soundcheck_bass_sustain_summed_bus_rebalance_priority"
    )
    assert bass_hot_bus["selected_parameters"]["target_scope"] == (
        "summed_low_end_bus_before_bass_compression"
    )
    assert bass_hot_bus["selected_parameters"][
        "post_rebalance_compression_recheck_required"
    ] is True
    assert bass_hot_bus["selected_parameters"]["summed_bus_target_id"] == "bus:3"
    assert bass_hot_bus["selected_parameters"]["wing_osc_target_id_override"] == (
        "bus:3"
    )
    _assert_replay_gain_delta_mapping(
        bass_hot_bus,
        source="bus_delta_db",
        expected_delta=bass_hot_bus["selected_parameters"]["bus_delta_db"],
        path_prefix="/bus/{bus}",
    )
    _assert_replay_selected_gain_delta_apply_values(
        bass_hot_bus,
        expected_delta=bass_hot_bus["selected_parameters"]["bus_delta_db"],
    )
    assert bass_hot_bus["candidate_blockers"]["bass_sustain_leveling_compression"] == [
        "summed_bus_rebalance_should_precede_bass_sustain_compression"
    ]
    assert {
        item["strategy_id"]: item["why_not_selected"]
        for item in bass_hot_bus["alternative_tradeoffs"]
    }["bass_sustain_leveling_compression"] == (
        "low_end_bus_level_priority:"
        "rebalance_summed_bus_before_bass_compression"
    )
    assert results["live_guitar_pick_sustain_compression"][
        "actual_selected_strategy_id"
    ] == "guitar_pick_sustain_compression"
    assert results["live_guitar_pick_sustain_compression"]["selected_parameters"][
        "makeup_gain_db"
    ] == 0.0
    assert results["live_guitar_pick_sustain_compression"]["selected_parameters"][
        "dynamics_parameter_source"
    ] == "measured_envelope"
    live_guitar_dynamics = results["live_guitar_pick_sustain_compression"][
        "selected_parameters"
    ]
    live_guitar_parameter_review_values = live_guitar_dynamics[
        "live_dynamics_parameter_review_only_values"
    ]
    assert live_guitar_parameter_review_values[
        "wing_compressor_ratio_osc_value"
    ] == "2.5"
    assert "wing_compressor_ratio_osc_value" not in live_guitar_dynamics
    assert results["live_guitar_pick_sustain_compression"]["selected_parameters"][
        "wing_compressor_ratio_value_type"
    ] == "string_enum"
    assert live_guitar_parameter_review_values[
        "wing_compressor_detector_osc_value"
    ] == "RMS"
    assert "wing_compressor_detector_osc_value" not in live_guitar_dynamics
    assert results["live_guitar_pick_sustain_compression"]["selected_parameters"][
        "wing_compressor_detector_value_type"
    ] == "string_enum"
    assert live_guitar_parameter_review_values["wing_compressor_knee_osc_value"] == 3
    assert "wing_compressor_knee_osc_value" not in live_guitar_dynamics
    assert results["live_guitar_pick_sustain_compression"]["selected_parameters"][
        "knee_db"
    ] == 3.1
    assert results["live_guitar_pick_sustain_compression"]["selected_parameters"][
        "compressor_knee_parameter_source"
    ] == "measured_envelope_variance_peak_to_median"
    assert results["live_guitar_pick_sustain_compression"]["selected_parameters"][
        "wing_compressor_knee_quantized_from"
    ] == 3.1
    assert results["live_guitar_pick_sustain_compression"]["selected_parameters"][
        "wing_compressor_knee_value_type"
    ] == "integer"
    assert live_guitar_dynamics["live_dynamics_conservative_proposal"] is True
    assert live_guitar_dynamics["live_dynamics_parameter_write_forbidden"] is True
    assert live_guitar_dynamics["live_dynamics_parameter_readback_required"] is True
    assert (
        live_guitar_dynamics[
            "live_dynamics_parameter_apply_blocked_without_wing_proof"
        ]
        is True
    )
    assert live_guitar_dynamics["live_dynamics_parameter_review_only_reason"] == (
        "live_dynamics_must_read_existing_compressor_state_and_operator_confirm_before_parameter_apply"
    )
    assert not any(key in live_guitar_dynamics for key in WING_DYNAMICS_WRITE_PARAMETER_KEYS)
    assert live_guitar_dynamics["wing_dynamics_selected_write_keys"] == []
    assert (
        live_guitar_dynamics["live_dynamics_stepwise_threshold_review_required"]
        is True
    )
    assert live_guitar_dynamics["live_dynamics_max_threshold_step_db"] == (
        LIVE_DYNAMICS_MAX_THRESHOLD_STEP_DB
    )
    assert live_guitar_dynamics["live_dynamics_max_target_gain_reduction_db"] == (
        LIVE_DYNAMICS_MAX_TARGET_GAIN_REDUCTION_DB
    )
    assert live_guitar_dynamics["live_dynamics_no_makeup_gain_required"] is True
    assert live_guitar_dynamics["live_dynamics_makeup_gain_db_limit"] == 0.0
    assert live_guitar_dynamics["live_dynamics_makeup_gain_write_forbidden"] is True
    assert live_guitar_dynamics["live_dynamics_makeup_gain_review_only_db"] == 0.0
    assert "wing_compressor_makeup_gain_osc_value" not in live_guitar_dynamics
    assert "wing_compressor_makeup_gain_osc_value" not in (
        live_guitar_dynamics["wing_dynamics_selected_write_keys"]
    )
    assert live_guitar_dynamics["live_dynamics_global_on_mix_write_forbidden"] is True
    assert live_guitar_dynamics["live_dynamics_on_mix_readback_required"] is True
    assert (
        live_guitar_dynamics["live_dynamics_compressor_enable_write_forbidden"]
        is True
    )
    assert live_guitar_dynamics["live_dynamics_compressor_mix_write_forbidden"] is True
    assert (
        live_guitar_dynamics[
            "live_dynamics_requires_existing_compressor_enabled_readback"
        ]
        is True
    )
    assert live_guitar_dynamics["live_dynamics_compressor_on_review_only_value"] == (
        WING_COMPRESSOR_ON_MAX
    )
    assert live_guitar_dynamics["live_dynamics_compressor_mix_review_only_value"] == (
        78.5
    )
    assert live_guitar_dynamics["live_dynamics_global_on_mix_review_only_reason"] == (
        "live_dynamics_must_read_existing_compressor_on_and_mix_before_parameter_apply"
    )
    assert "wing_compressor_on_osc_value" not in live_guitar_dynamics
    assert "wing_compressor_mix_osc_value" not in live_guitar_dynamics
    assert "wing_compressor_on_osc_value" not in (
        live_guitar_dynamics["wing_dynamics_selected_write_keys"]
    )
    assert "wing_compressor_mix_osc_value" not in (
        live_guitar_dynamics["wing_dynamics_selected_write_keys"]
    )
    forbidden_live_dynamics_prefixes = (
        "wing_compressor_crossover_",
        "wing_compressor_sidechain_",
    )
    assert live_guitar_dynamics["live_dynamics_minimal_write_set_required"] is True
    assert live_guitar_dynamics["live_dynamics_crossover_write_forbidden"] is True
    assert live_guitar_dynamics["live_dynamics_sidechain_write_forbidden"] is True
    assert live_guitar_dynamics["live_dynamics_crossover_review_only_values"] == {
        "compressor_crossover_type": "OFF",
        "compressor_crossover_depth_db": 0.0,
        "compressor_crossover_frequency_hz": 1000.0,
        "compressor_crossover_parameter_source": (
            "wing_dynxo_full_band_compression_baseline_policy"
        ),
    }
    assert live_guitar_dynamics["live_dynamics_sidechain_review_only_values"] == {
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
        key.startswith(forbidden_live_dynamics_prefixes)
        for key in live_guitar_dynamics
    )
    assert not any(
        key.startswith(forbidden_live_dynamics_prefixes)
        for key in live_guitar_dynamics["wing_dynamics_selected_write_keys"]
    )
    live_guitar_dynamics_readback = results["live_guitar_pick_sustain_compression"][
        "selected_future_wing_readback_contract"
    ]["dynamics_readback"]
    assert (
        live_guitar_dynamics_readback["live_dynamics_parameter_write_forbidden"]
        is True
    )
    assert (
        live_guitar_dynamics_readback["live_dynamics_parameter_readback_required"]
        is True
    )
    assert (
        live_guitar_dynamics_readback[
            "live_dynamics_parameter_apply_blocked_without_wing_proof"
        ]
        is True
    )
    assert (
        live_guitar_dynamics_readback["live_dynamics_parameter_review_only_values"]
        == live_guitar_parameter_review_values
    )
    assert (
        live_guitar_dynamics_readback["live_dynamics_parameter_review_only_reason"]
        == live_guitar_dynamics["live_dynamics_parameter_review_only_reason"]
    )
    assert not any(
        key in live_guitar_dynamics_readback for key in WING_DYNAMICS_WRITE_PARAMETER_KEYS
    )
    assert live_guitar_dynamics_readback["wing_dynamics_selected_write_keys"] == []
    assert (
        live_guitar_dynamics_readback["live_dynamics_makeup_gain_write_forbidden"]
        is True
    )
    assert (
        live_guitar_dynamics_readback["live_dynamics_makeup_gain_review_only_db"]
        == 0.0
    )
    assert "wing_compressor_makeup_gain_osc_value" not in live_guitar_dynamics_readback
    assert "wing_compressor_makeup_gain_osc_value" not in (
        live_guitar_dynamics_readback["wing_dynamics_selected_write_keys"]
    )
    assert (
        live_guitar_dynamics_readback["live_dynamics_global_on_mix_write_forbidden"]
        is True
    )
    assert (
        live_guitar_dynamics_readback["live_dynamics_on_mix_readback_required"]
        is True
    )
    assert (
        live_guitar_dynamics_readback[
            "live_dynamics_compressor_enable_write_forbidden"
        ]
        is True
    )
    assert (
        live_guitar_dynamics_readback["live_dynamics_compressor_mix_write_forbidden"]
        is True
    )
    assert (
        live_guitar_dynamics_readback[
            "live_dynamics_requires_existing_compressor_enabled_readback"
        ]
        is True
    )
    assert live_guitar_dynamics_readback[
        "live_dynamics_compressor_on_review_only_value"
    ] == WING_COMPRESSOR_ON_MAX
    assert (
        live_guitar_dynamics_readback[
            "live_dynamics_compressor_mix_review_only_value"
        ]
        == 78.5
    )
    assert live_guitar_dynamics_readback[
        "live_dynamics_global_on_mix_review_only_reason"
    ] == "live_dynamics_must_read_existing_compressor_on_and_mix_before_parameter_apply"
    assert "wing_compressor_on_osc_value" not in live_guitar_dynamics_readback
    assert "wing_compressor_mix_osc_value" not in live_guitar_dynamics_readback
    assert "wing_compressor_on_osc_value" not in (
        live_guitar_dynamics_readback["wing_dynamics_selected_write_keys"]
    )
    assert "wing_compressor_mix_osc_value" not in (
        live_guitar_dynamics_readback["wing_dynamics_selected_write_keys"]
    )
    assert (
        live_guitar_dynamics_readback["live_dynamics_minimal_write_set_required"]
        is True
    )
    assert (
        live_guitar_dynamics_readback["live_dynamics_crossover_write_forbidden"]
        is True
    )
    assert (
        live_guitar_dynamics_readback["live_dynamics_sidechain_write_forbidden"]
        is True
    )
    assert live_guitar_dynamics_readback[
        "live_dynamics_crossover_review_only_values"
    ] == live_guitar_dynamics["live_dynamics_crossover_review_only_values"]
    assert live_guitar_dynamics_readback[
        "live_dynamics_sidechain_review_only_values"
    ] == live_guitar_dynamics["live_dynamics_sidechain_review_only_values"]
    assert not any(
        key.startswith(forbidden_live_dynamics_prefixes)
        for key in live_guitar_dynamics_readback
    )
    assert not any(
        key.startswith(forbidden_live_dynamics_prefixes)
        for key in live_guitar_dynamics_readback["wing_dynamics_selected_write_keys"]
    )
    live_guitar_apply_values = results["live_guitar_pick_sustain_compression"][
        "proposal_selected_wing_osc_summary"
    ]["wing_osc_apply_parameter_values"]
    assert live_guitar_apply_values == {}
    assert results["live_guitar_pick_sustain_compression"][
        "proposal_selected_wing_osc_readback_summary"
    ]["wing_osc_readback_apply_parameter_values"] == {}
    assert not any(key in live_guitar_apply_values for key in WING_DYNAMICS_WRITE_PARAMETER_KEYS)
    assert "wing_compressor_makeup_gain_osc_value" not in live_guitar_apply_values
    assert "wing_compressor_on_osc_value" not in live_guitar_apply_values
    assert "wing_compressor_mix_osc_value" not in live_guitar_apply_values
    assert not any(
        key.startswith(forbidden_live_dynamics_prefixes)
        for key in live_guitar_apply_values
    )
    assert (
        live_guitar_dynamics[
            "live_dynamics_gain_reduction_meter_readback_after_each_step_required"
        ]
        is True
    )
    assert (
        live_guitar_dynamics["live_dynamics_previous_comp_model_readback_required"]
        is True
    )
    assert live_guitar_dynamics["live_dynamics_operator_rehearsal_required"] is True
    assert live_guitar_dynamics["live_dynamics_live_promotion_status"] == (
        "blocked_until_real_wing_readback_operator_rehearsal_and_rollback"
    )
    assert (
        live_guitar_dynamics["live_dynamics_live_apply_blocked_without_wing_proof"]
        is True
    )
    assert results["live_lead_guitar_ride_before_compression"][
        "actual_selected_strategy_id"
    ] == "manual_level_ride_before_compression"
    assert results["live_lead_guitar_ride_before_compression"][
        "actual_method_family"
    ] == "analysis_only"
    assert results["live_lead_guitar_ride_before_compression"][
        "operator_explanation_sections"
    ]["selection_policy"] == "live_guitar_lead_ride_priority"
    assert results["live_lead_guitar_ride_before_compression"][
        "selected_parameters"
    ]["live_guitar_lead_ride_priority"] is True
    assert results["live_lead_guitar_ride_before_compression"][
        "selected_parameters"
    ]["forbidden_without_wing_proof"] == "live_guitar_lead_compressor_change"
    live_guitar_ride = results["live_lead_guitar_ride_before_compression"]
    assert live_guitar_ride["selected_parameters"][
        "live_dynamics_ride_hold_proposal"
    ] is True
    assert live_guitar_ride["selected_parameters"][
        "live_dynamics_ride_profile"
    ] == "lead_guitar_solo"
    live_guitar_ride_readback = live_guitar_ride[
        "selected_future_wing_readback_contract"
    ]["live_dynamics_ride_hold_readback"]
    assert live_guitar_ride_readback["read_only"] is True
    assert live_guitar_ride_readback[
        "allowed_live_write_scope_before_wing_proof"
    ] == []
    assert "/ch/{channel}/fdr" in live_guitar_ride_readback[
        "wing_gain_osc_path_templates"
    ]
    assert "/ch/{channel}/dyn/thr24" in live_guitar_ride_readback[
        "wing_dynamics_osc_path_templates"
    ]
    _assert_replay_analysis_only_has_no_apply_values(live_guitar_ride)
    assert results["live_lead_guitar_ride_before_compression"][
        "candidate_blockers"
    ]["guitar_pick_sustain_compression"] == [
        "live_guitar_lead_compression_requires_operator_review_and_wing_proof"
    ]
    assert results["live_lead_guitar_ride_before_compression"][
        "candidate_blockers"
    ]["gentle_channel_compression"] == [
        "live_guitar_lead_compression_requires_operator_review_and_wing_proof",
        "live_dynamics_target_gr_exceeds_conservative_limit",
    ]
    tradeoffs = {
        item["strategy_id"]: item["why_not_selected"]
        for item in results["live_lead_guitar_ride_before_compression"][
            "alternative_tradeoffs"
        ]
    }
    assert tradeoffs["guitar_pick_sustain_compression"] == (
        "live_guitar_lead_ride_priority:"
        "preserve_solo_phrasing_before_compressor_change"
    )
    assert tradeoffs["gentle_channel_compression"] == (
        "live_guitar_lead_ride_priority:"
        "preserve_solo_phrasing_before_compressor_change"
    )
    assert results["soundcheck_quiet_vocal_level_plane_gain_up"][
        "actual_selected_strategy_id"
    ] == "soundcheck_bounded_gain_up"
    assert results["soundcheck_quiet_vocal_feedback_path_holds_gain_up"][
        "actual_selected_strategy_id"
    ] == "hold_gain_up_for_feedback_path_review"
    assert results["soundcheck_quiet_vocal_feedback_path_holds_gain_up"][
        "actual_method_family"
    ] == "analysis_only"
    assert results["soundcheck_quiet_vocal_feedback_path_holds_gain_up"][
        "operator_explanation_sections"
    ]["selection_policy"] == "soundcheck_feedback_path_before_gain_up_priority"
    assert results["soundcheck_quiet_vocal_feedback_path_holds_gain_up"][
        "selected_parameters"
    ]["read_only_review"] is True
    assert results["soundcheck_quiet_vocal_feedback_path_holds_gain_up"][
        "selected_parameters"
    ]["gain_db"] == 0.0
    assert results["soundcheck_quiet_vocal_feedback_path_holds_gain_up"][
        "candidate_blockers"
    ]["soundcheck_bounded_gain_up"] == [
        "feedback_path_review_should_precede_gain_up"
    ]
    feedback_gain_up_tradeoffs = {
        row["strategy_id"]: row["why_not_selected"]
        for row in results["soundcheck_quiet_vocal_feedback_path_holds_gain_up"][
            "alternative_tradeoffs"
        ]
    }
    assert feedback_gain_up_tradeoffs["soundcheck_bounded_gain_up"] == (
        "soundcheck_feedback_path_gain_up_priority:"
        "feedback_path_review_precedes_gain_up"
    )
    assert results["live_quiet_vocal_source_gain_up_review_hold"][
        "actual_selected_strategy_id"
    ] == "hold_live_quiet_source_gain_up_for_operator_review"
    assert results["live_quiet_vocal_source_gain_up_review_hold"][
        "actual_method_family"
    ] == "analysis_only"
    live_quiet_gain_hold = results["live_quiet_vocal_source_gain_up_review_hold"]
    assert live_quiet_gain_hold["operator_explanation_sections"][
        "selection_policy"
    ] == "live_quiet_source_gain_up_review_priority"
    assert live_quiet_gain_hold["selected_parameters"]["dsp_change"] == "hold"
    assert live_quiet_gain_hold["selected_parameters"]["gain_db"] == 0.0
    assert live_quiet_gain_hold["selected_parameters"][
        "calculated_gain_up_db"
    ] == 1.5
    assert live_quiet_gain_hold["selected_parameters"][
        "live_gain_up_from_short_signal_or_bleed_blocked"
    ] is True
    assert live_quiet_gain_hold["selected_parameters"][
        "wing_rack_fw305_correction_exists_on_console_required"
    ] is True
    assert live_quiet_gain_hold["selected_parameters"][
        "wing_rack_fw305_verified_osc_mapping_required"
    ] is True
    assert live_quiet_gain_hold["selected_parameters"][
        "wing_rack_fw305_readback_required"
    ] is True
    assert live_quiet_gain_hold["selected_parameters"][
        "wing_rack_fw305_rollback_required"
    ] is True
    assert live_quiet_gain_hold["selected_parameters"][
        "live_authority_granted"
    ] is False
    assert live_quiet_gain_hold["selected_parameters"]["live_write"] is False
    assert live_quiet_gain_hold["candidate_blockers"][
        "live_hold_gain_up_for_operator"
    ] == ["live_gain_up_requires_operator_review_and_wing_proof"]
    _assert_replay_analysis_only_has_no_apply_values(live_quiet_gain_hold)
    live_quiet_gain_tradeoffs = {
        row["strategy_id"]: row["why_not_selected"]
        for row in live_quiet_gain_hold["alternative_tradeoffs"]
    }
    assert live_quiet_gain_tradeoffs["live_hold_gain_up_for_operator"] == (
        "live_quiet_source_gain_up_review_priority:"
        "live_source_gain_up_requires_operator_review_and_wing_proof"
    )
    assert results["soundcheck_quiet_vocal_hot_bus_rebalance"][
        "actual_selected_strategy_id"
    ] == "summed_bus_rebalance_before_source_gain"
    quiet_hot_bus = results["soundcheck_quiet_vocal_hot_bus_rebalance"]
    assert quiet_hot_bus["selected_parameters"]["summed_bus_target_id"] == "bus:1"
    assert quiet_hot_bus["selected_parameters"]["wing_osc_target_id_override"] == (
        "bus:1"
    )
    _assert_replay_gain_delta_mapping(
        quiet_hot_bus,
        source="bus_delta_db",
        expected_delta=quiet_hot_bus["selected_parameters"]["bus_delta_db"],
        path_prefix="/bus/{bus}",
    )
    _assert_replay_selected_gain_delta_apply_values(
        quiet_hot_bus,
        expected_delta=quiet_hot_bus["selected_parameters"]["bus_delta_db"],
    )
    assert results["soundcheck_quiet_muddy_vocal_level_first"][
        "actual_selected_strategy_id"
    ] == "soundcheck_bounded_gain_up"
    assert results["soundcheck_quiet_muddy_vocal_level_first"][
        "selected_rationale"
    ]["selection_policy"] == "soundcheck_quiet_level_plane_priority"
    assert results["soundcheck_hot_muddy_vocal_level_down_before_eq"][
        "actual_selected_strategy_id"
    ] == "level_plane_gain_down"
    assert results["soundcheck_hot_muddy_vocal_level_down_before_eq"][
        "selected_rationale"
    ]["selection_policy"] == "soundcheck_hot_level_plane_before_tone_priority"
    assert results["soundcheck_hot_muddy_vocal_level_down_before_eq"][
        "selected_parameters"
    ]["delta_db"] < 0.0
    assert {
        item["strategy_id"]: item["why_not_selected"]
        for item in results["soundcheck_hot_muddy_vocal_level_down_before_eq"][
            "alternative_tradeoffs"
        ]
    }["low_mid_corrective_cut"] == (
        "soundcheck_hot_level_plane_priority:"
        "restore_level_before_low_mid_tone_processing"
    )
    assert results[
        "soundcheck_quiet_muddy_vocal_hot_bus_rebalance_before_eq"
    ]["actual_selected_strategy_id"] == "summed_bus_rebalance_before_source_gain"
    assert results[
        "soundcheck_quiet_muddy_vocal_hot_bus_rebalance_before_eq"
    ]["selected_rationale"]["selection_policy"] == (
        "soundcheck_quiet_hot_bus_rebalance_priority"
    )
    muddy_hot_bus = results[
        "soundcheck_quiet_muddy_vocal_hot_bus_rebalance_before_eq"
    ]
    assert muddy_hot_bus["selected_parameters"]["summed_bus_target_id"] == "bus:1"
    assert muddy_hot_bus["selected_parameters"]["wing_osc_target_id_override"] == (
        "bus:1"
    )
    _assert_replay_gain_delta_mapping(
        muddy_hot_bus,
        source="bus_delta_db",
        expected_delta=muddy_hot_bus["selected_parameters"]["bus_delta_db"],
        path_prefix="/bus/{bus}",
    )
    _assert_replay_selected_gain_delta_apply_values(
        muddy_hot_bus,
        expected_delta=muddy_hot_bus["selected_parameters"]["bus_delta_db"],
    )
    assert results["live_loud_bass_level_plane_gain_down"][
        "actual_selected_strategy_id"
    ] == "level_plane_gain_down"
    assert results["live_loud_bass_level_plane_gain_down"]["selected_parameters"][
        "live_readback_after_each_step_required"
    ] is True
    assert results["live_operator_owned_loud_bass_gain_hold"][
        "actual_selected_strategy_id"
    ] == "hold_live_operator_owned_gain_for_review"
    assert results["live_operator_owned_loud_bass_gain_hold"][
        "actual_method_family"
    ] == "analysis_only"
    live_owned_gain_hold = results["live_operator_owned_loud_bass_gain_hold"]
    assert live_owned_gain_hold["operator_explanation_sections"][
        "selection_policy"
    ] == "live_operator_owned_gain_review_priority"
    assert live_owned_gain_hold["selected_parameters"]["dsp_change"] == "hold"
    assert live_owned_gain_hold["selected_parameters"]["gain_db"] == 0.0
    assert live_owned_gain_hold["selected_parameters"]["calculated_delta_db"] == -1.5
    assert live_owned_gain_hold["selected_parameters"][
        "calculated_gain_direction"
    ] == "down"
    assert live_owned_gain_hold["selected_parameters"][
        "wing_gain_current_operator_owned"
    ] is True
    assert live_owned_gain_hold["selected_parameters"][
        "wing_gain_operator_owned_edit_requires_review"
    ] is True
    assert live_owned_gain_hold["selected_parameters"][
        "wing_gain_operator_owned_blocker"
    ] == WING_GAIN_OPERATOR_OWNED_BLOCKER
    assert live_owned_gain_hold["selected_parameters"][
        "wing_rack_fw305_verified_osc_mapping_required"
    ] is True
    assert live_owned_gain_hold["selected_parameters"][
        "wing_rack_fw305_readback_required"
    ] is True
    assert live_owned_gain_hold["selected_parameters"][
        "wing_rack_fw305_rollback_required"
    ] is True
    assert live_owned_gain_hold["selected_parameters"][
        "live_authority_granted"
    ] is False
    assert live_owned_gain_hold["selected_parameters"]["live_write"] is False
    assert live_owned_gain_hold["candidate_blockers"]["level_plane_gain_down"] == [
        WING_GAIN_OPERATOR_OWNED_BLOCKER
    ]
    _assert_replay_analysis_only_has_no_apply_values(live_owned_gain_hold)
    live_owned_gain_tradeoffs = {
        row["strategy_id"]: row["why_not_selected"]
        for row in live_owned_gain_hold["alternative_tradeoffs"]
    }
    assert live_owned_gain_tradeoffs["level_plane_gain_down"] == (
        "live_operator_owned_gain_review_priority:"
        "manual_fader_or_trim_owner_review_precedes_live_gain_move"
    )
    assert results["soundcheck_hot_drum_bus_group_trim"][
        "actual_selected_strategy_id"
    ] == "group_bus_trim_for_headroom"
    assert results["soundcheck_hot_drum_bus_group_trim"]["selected_parameters"][
        "target_scope"
    ] == "group_bus"
    assert results["soundcheck_dynamic_vocal_bus_group_compression"][
        "actual_selected_strategy_id"
    ] == "group_bus_compression_review"
    assert results["soundcheck_dynamic_vocal_bus_group_compression"][
        "selected_parameters"
    ]["makeup_gain_db"] == 0.0
    assert results["soundcheck_dynamic_vocal_bus_group_compression"][
        "selected_parameters"
    ]["wing_compressor_ratio_osc_value"] == 2.5
    assert results["soundcheck_dynamic_vocal_bus_group_compression"][
        "selected_parameters"
    ]["wing_compressor_ratio_value_type"] == "float"
    assert results["soundcheck_dynamic_vocal_bus_group_compression"][
        "selected_parameters"
    ]["wing_compressor_knee_osc_value"] == 4
    assert results["soundcheck_dynamic_vocal_bus_group_compression"][
        "selected_parameters"
    ]["wing_compressor_knee_value_type"] == "integer"
    assert results["live_quiet_vocal_bus_group_gain_hold"][
        "actual_selected_strategy_id"
    ] == "hold_group_bus_for_member_evidence"
    live_group_hold = results["live_quiet_vocal_bus_group_gain_hold"]
    assert live_group_hold["selected_parameters"]["target_scope"] == (
        "group_bus_and_members"
    )
    assert live_group_hold["selected_parameters"]["required_context"] == (
        "group_membership_bus_output_identity_summed_bus_readback_rollback"
    )
    assert live_group_hold["selected_parameters"]["forbidden_without_wing_proof"] == (
        "group_bus_gain_eq_dynamics_pan_mute_or_fx_change"
    )
    assert live_group_hold["selected_parameters"]["group_bus_target_surface"] == "bus"
    assert live_group_hold["selected_parameters"][
        "group_bus_output_feed_identity_required"
    ] is True
    assert live_group_hold["selected_parameters"][
        "group_bus_source_member_identity_required"
    ] is True
    assert live_group_hold["selected_parameters"][
        "group_bus_membership_readback_required"
    ] is True
    assert live_group_hold["selected_parameters"][
        "group_bus_summed_bus_evidence_required"
    ] is True
    assert live_group_hold["selected_parameters"][
        "live_group_bus_live_apply_blocked_without_wing_proof"
    ] is True
    assert "wing_gain_delta_db" not in live_group_hold["selected_parameters"]
    group_hold_readback = live_group_hold["selected_future_wing_readback_contract"][
        "group_bus_hold_readback"
    ]
    assert group_hold_readback["read_only"] is True
    assert group_hold_readback["group_bus_target_surface"] == "bus"
    assert group_hold_readback["group_bus_output_feed_identity_required"] is True
    assert group_hold_readback["group_bus_summed_bus_evidence_required"] is True
    assert group_hold_readback["allowed_live_write_scope_before_wing_proof"] == []
    assert "/bus/{bus}/fdr" in group_hold_readback["wing_osc_path_templates"]
    assert "/bus/{bus}/eq/{band}f" in group_hold_readback[
        "wing_osc_path_templates"
    ]
    assert "/bus/{bus}/dyn/thr31" in group_hold_readback[
        "wing_osc_path_templates"
    ]
    assert results["soundcheck_quiet_vocal_bus_feedback_member_review"][
        "actual_selected_strategy_id"
    ] == "hold_group_bus_feedback_member_review"
    assert results["soundcheck_quiet_vocal_bus_feedback_member_review"][
        "operator_explanation_sections"
    ]["selection_policy"] == "feedback_risk_priority"
    assert results["soundcheck_quiet_vocal_bus_feedback_member_review"][
        "selected_parameters"
    ]["target_scope"] == "group_members_before_bus_processing"
    assert results["soundcheck_quiet_vocal_bus_feedback_member_review"][
        "candidate_blockers"
    ]["soundcheck_bounded_group_gain_up"] == [
        "feedback_risk_blocks_boost",
        WING_OSC_UNRESOLVED_TARGET_BLOCKER,
    ]
    assert results["live_group_feedback_member_review_before_bus_dsp"][
        "actual_selected_strategy_id"
    ] == "hold_group_bus_feedback_member_review"
    assert results["live_group_feedback_member_review_before_bus_dsp"][
        "actual_method_family"
    ] == "analysis_only"
    assert results["live_group_feedback_member_review_before_bus_dsp"][
        "operator_explanation_sections"
    ]["selection_policy"] == "live_feedback_risk_priority"
    assert results["live_group_feedback_member_review_before_bus_dsp"][
        "selected_parameters"
    ]["target_scope"] == "group_members_before_bus_processing"
    assert "feedback_risk_requires_group_member_review" in results[
        "live_group_feedback_member_review_before_bus_dsp"
    ]["candidate_blockers"]["group_bus_compression_review"]
    feedback_tradeoffs = {
        item["strategy_id"]: item["why_not_selected"]
        for item in results["live_group_feedback_member_review_before_bus_dsp"][
            "alternative_tradeoffs"
        ]
    }
    assert feedback_tradeoffs["drum_transient_tamer_compression"] == (
        "group_feedback_review_priority:"
        "identify_feedback_member_before_group_processing"
    )
    assert results["soundcheck_long_fx_tail_decay_filter"][
        "actual_selected_strategy_id"
    ] == "reduce_fx_send_or_return"
    long_tail_params = results["soundcheck_long_fx_tail_decay_filter"][
        "selected_parameters"
    ]
    assert long_tail_params["send_delta_db"] == 0.0
    assert long_tail_params["return_delta_db"] < 0.0
    assert long_tail_params["post_pullback_time_edit_review_required"] is True
    assert long_tail_params["fx_decay_native_pullback_first"] is True
    assert long_tail_params["fx_decay_parameter_source"] == (
        "measured_tail_decay_role_target"
    )
    assert long_tail_params["measured_tail_decay_s"] == 2.4
    assert long_tail_params["measured_delay_repeat_ms"] == 0.0
    assert long_tail_params["fx_level_pullback_parameter_source"] == (
        "measured_tail_decay_native_level_pullback"
    )
    assert long_tail_params["native_fx_level_pullback_db"] == 4.7
    assert long_tail_params["native_fx_level_pullback_legacy_db"] == 4.3
    assert long_tail_params["post_pullback_decay_review_s"] < 2.2
    assert long_tail_params["post_pullback_return_hpf_review_hz"] > 220.0
    assert long_tail_params["post_pullback_return_lpf_review_hz"] < 6800.0
    assert long_tail_params["post_pullback_delay_feedback_review_percent"] < 12.0
    assert long_tail_params[
        "wing_native_fx_level_before_model_specific_decay_edit"
    ] is True
    assert long_tail_params["soundcheck_fx_bounded_level_proposal"] is True
    assert long_tail_params["soundcheck_fx_native_level_osc_only_required"] is True
    assert long_tail_params["soundcheck_fx_max_pullback_db"] == (
        SOUNDCHECK_FX_MAX_PULLBACK_DB
    )
    assert long_tail_params["soundcheck_fx_max_boost_db"] == (
        SOUNDCHECK_FX_MAX_BOOST_DB
    )
    assert long_tail_params["soundcheck_fx_level_move_kinds"] == ["return_level"]
    assert long_tail_params[
        "soundcheck_fx_no_model_or_time_parameter_write_required"
    ] is True
    assert long_tail_params["soundcheck_fx_followup_model_time_review_required"] is True
    assert long_tail_params["soundcheck_fx_absolute_level_readback_required"] is True
    assert (
        long_tail_params["soundcheck_fx_monitor_feedback_recheck_after_apply_required"]
        is True
    )
    assert (
        long_tail_params["soundcheck_fx_return_headroom_recheck_after_apply_required"]
        is True
    )
    assert long_tail_params["soundcheck_fx_operator_confirmation_required"] is True
    assert long_tail_params["soundcheck_fx_readback_required"] is True
    assert long_tail_params["soundcheck_fx_rollback_ready"] is True
    assert long_tail_params["soundcheck_fx_live_promotion_status"] == (
        "blocked_until_real_wing_fx_level_readback_monitor_return_headroom_recheck_and_operator_approval"
    )
    assert long_tail_params["soundcheck_fx_live_apply_blocked_without_wing_proof"] is True
    long_tail_candidate_params = results["soundcheck_long_fx_tail_decay_filter"][
        "candidate_parameters"
    ]["shorten_fx_decay_filter"]
    assert long_tail_candidate_params["fx_decay_parameter_source"] == (
        "measured_tail_decay_role_target"
    )
    assert long_tail_candidate_params["fx_target_context"] == "vocal_reverb"
    assert long_tail_candidate_params["pre_delay_ms"] == 50.0
    assert long_tail_candidate_params["fx_tail_density_score"] > 0.7
    assert long_tail_candidate_params["delay_feedback_percent"] < 12.0
    _assert_replay_fx_model_specific_guard(
        results["soundcheck_long_fx_tail_decay_filter"],
        "shorten_fx_decay_filter",
        {
            "decay_s",
            "pre_delay_ms",
            "return_hpf_hz",
            "return_lpf_hz",
            "delay_feedback_percent",
        },
    )
    assert results["soundcheck_long_fx_tail_decay_filter"][
        "operator_explanation_sections"
    ]["selection_policy"] == "soundcheck_wing_native_osc_fx_level_pullback_priority"
    long_tail_fx_readback = results["soundcheck_long_fx_tail_decay_filter"][
        "selected_future_wing_readback_contract"
    ]["fx_readback"]
    assert long_tail_fx_readback["soundcheck_fx_bounded_level_proposal"] is True
    assert (
        long_tail_fx_readback["soundcheck_fx_native_level_osc_only_required"]
        is True
    )
    assert long_tail_fx_readback["soundcheck_fx_level_move_kinds"] == [
        "return_level"
    ]
    assert (
        long_tail_fx_readback[
            "soundcheck_fx_live_apply_blocked_without_wing_proof"
        ]
        is True
    )
    _assert_replay_selected_fx_apply_values(
        results["soundcheck_long_fx_tail_decay_filter"],
        {"wing_fx_return_delta_db": -4.7},
    )
    _assert_replay_fx_model_time_followup_review_contract(
        results["soundcheck_long_fx_tail_decay_filter"],
        expected_values={
            "post_pullback_return_hpf_review_hz": long_tail_params[
                "post_pullback_return_hpf_review_hz"
            ],
            "post_pullback_return_lpf_review_hz": long_tail_params[
                "post_pullback_return_lpf_review_hz"
            ],
            "post_pullback_decay_review_s": long_tail_params[
                "post_pullback_decay_review_s"
            ],
            "post_pullback_delay_feedback_review_percent": long_tail_params[
                "post_pullback_delay_feedback_review_percent"
            ],
        },
        expected_native_apply_fields=["wing_fx_return_delta_db"],
        expected_review_fields=WING_SOUNDCHECK_FX_MODEL_TIME_REVIEW_ONLY_FIELDS,
    )
    assert results["soundcheck_delay_smear_fx_ducking_cleanup"][
        "actual_selected_strategy_id"
    ] == "reduce_fx_send_or_return"
    delay_params = results["soundcheck_delay_smear_fx_ducking_cleanup"][
        "selected_parameters"
    ]
    assert delay_params["send_delta_db"] == 0.0
    assert delay_params["return_delta_db"] < 0.0
    assert delay_params["post_pullback_time_edit_review_required"] is True
    assert delay_params["fx_delay_smear_native_pullback_first"] is True
    assert delay_params["fx_delay_smear_parameter_source"] == "measured_delay_repeat"
    assert delay_params["measured_delay_repeat_ms"] == 130.0
    assert delay_params["measured_tail_decay_s"] == 1.2
    assert delay_params["fx_level_pullback_parameter_source"] == (
        "measured_delay_repeat_native_level_pullback"
    )
    assert delay_params["native_fx_level_pullback_db"] == 3.4
    assert delay_params["native_fx_level_pullback_legacy_db"] == 3.0
    assert delay_params["post_pullback_delay_time_review_ms"] == 130.0
    assert delay_params["post_pullback_delay_feedback_review_percent"] == 9.0
    assert delay_params["post_pullback_delay_low_pass_review_hz"] == 5850.0
    assert delay_params["post_pullback_delay_ducking_review_db"] == 4.5
    assert delay_params["post_pullback_delay_ducking_release_review_ms"] == 288.0
    assert delay_params["wing_native_fx_level_before_model_specific_delay_edit"] is True
    delay_candidate_params = results["soundcheck_delay_smear_fx_ducking_cleanup"][
        "candidate_parameters"
    ]["duck_delay_return_cleanup"]
    assert delay_candidate_params["fx_delay_parameter_source"] == "measured_delay_repeat"
    assert delay_candidate_params["delay_time_ms"] == 130.0
    assert delay_candidate_params["measured_delay_repeat_ms"] == 130.0
    assert delay_candidate_params["measured_tail_decay_s"] == 1.2
    assert delay_candidate_params["ducking_depth_db"] == 4.5
    assert delay_candidate_params["attack_ms"] == 7.7
    assert delay_candidate_params["release_ms"] == 288.0
    assert delay_candidate_params["delay_feedback_percent"] == 9.0
    assert delay_candidate_params["delay_low_pass_hz"] == 5850.0
    _assert_replay_fx_model_specific_guard(
        results["soundcheck_delay_smear_fx_ducking_cleanup"],
        "duck_delay_return_cleanup",
        {
            "ducking_depth_db",
            "delay_feedback_percent",
            "delay_low_pass_hz",
        },
    )
    _assert_replay_selected_fx_apply_values(
        results["soundcheck_delay_smear_fx_ducking_cleanup"],
        {"wing_fx_return_delta_db": -3.4},
    )
    _assert_replay_fx_model_time_followup_review_contract(
        results["soundcheck_delay_smear_fx_ducking_cleanup"],
        expected_values={
            "post_pullback_delay_time_review_ms": 130.0,
            "post_pullback_delay_feedback_review_percent": 9.0,
            "post_pullback_delay_low_pass_review_hz": 5850.0,
            "post_pullback_delay_ducking_review_db": 4.5,
            "post_pullback_delay_ducking_release_review_ms": 288.0,
            "post_pullback_return_hpf_review_hz": 180.0,
            "post_pullback_return_lpf_review_hz": 7600.0,
        },
        expected_native_apply_fields=["wing_fx_return_delta_db"],
        expected_review_fields=WING_SOUNDCHECK_FX_MODEL_TIME_REVIEW_ONLY_FIELDS,
    )
    assert results["soundcheck_delay_smear_fx_ducking_cleanup"][
        "operator_explanation_sections"
    ]["selection_policy"] == "soundcheck_wing_native_osc_fx_level_pullback_priority"
    assert results["soundcheck_fx_monitor_risk_pullback_before_time_edits"][
        "actual_selected_strategy_id"
    ] == "reduce_fx_send_or_return"
    fx_monitor_params = results[
        "soundcheck_fx_monitor_risk_pullback_before_time_edits"
    ]["selected_parameters"]
    assert fx_monitor_params["send_delta_db"] == 0.0
    assert fx_monitor_params["return_delta_db"] < 0.0
    assert fx_monitor_params["monitor_feedback_risk"] == 0.62
    assert fx_monitor_params["monitor_safety_pullback_first"] is True
    assert fx_monitor_params["fx_level_pullback_parameter_source"] == (
        "measured_tail_decay_native_level_pullback"
    )
    assert fx_monitor_params["native_fx_level_pullback_db"] == 4.7
    assert fx_monitor_params["native_fx_level_pullback_legacy_db"] == 4.3
    assert fx_monitor_params["measured_tail_decay_s"] == 2.4
    assert fx_monitor_params["measured_delay_repeat_ms"] == 0.0
    assert results["soundcheck_fx_monitor_risk_pullback_before_time_edits"][
        "operator_explanation_sections"
    ]["selection_policy"] == "soundcheck_fx_monitor_feedback_pullback_priority"
    _assert_replay_selected_fx_apply_values(
        results["soundcheck_fx_monitor_risk_pullback_before_time_edits"],
        {"wing_fx_return_delta_db": -4.7},
    )
    fx_monitor_blockers = results[
        "soundcheck_fx_monitor_risk_pullback_before_time_edits"
    ]["candidate_blockers"]
    assert "monitor_feedback_risk_pullback_should_precede_decay_edit" in (
        fx_monitor_blockers["shorten_fx_decay_filter"]
    )
    assert "monitor_feedback_risk_blocks_fx_ducking" in (
        fx_monitor_blockers["duck_fx_return"]
    )
    assert results["soundcheck_fx_return_headroom_only_pullback"][
        "actual_selected_strategy_id"
    ] == "fx_return_headroom_pullback_before_time_edits"
    fx_headroom_only_params = results[
        "soundcheck_fx_return_headroom_only_pullback"
    ]["selected_parameters"]
    assert fx_headroom_only_params["return_delta_db"] == -1.3
    assert fx_headroom_only_params["send_delta_db"] == 0.0
    assert fx_headroom_only_params["fx_return_headroom_db"] == 1.2
    assert fx_headroom_only_params["target_fx_return_headroom_db"] == 2.5
    assert fx_headroom_only_params["fx_return_headroom_deficit_db"] == 1.3
    assert fx_headroom_only_params["return_headroom_restoration_first"] is True
    _assert_replay_fx_return_fxmix_mapping(
        results["soundcheck_fx_return_headroom_only_pullback"],
        expected_delta=fx_headroom_only_params["return_delta_db"],
        target_surface="fx",
        fx_slot_binding_required=False,
    )
    _assert_replay_selected_fx_apply_values(
        results["soundcheck_fx_return_headroom_only_pullback"],
        {"wing_fx_return_delta_db": -1.3},
    )
    assert results["soundcheck_fx_return_headroom_only_pullback"][
        "operator_explanation_sections"
    ]["selection_policy"] == "fx_return_headroom_restoration_priority"
    fx_headroom_only_blockers = results[
        "soundcheck_fx_return_headroom_only_pullback"
    ]["candidate_blockers"]
    assert "fx_return_headroom_pullback_should_precede_delay_ducking" in (
        fx_headroom_only_blockers["duck_delay_return_cleanup"]
    )
    assert results[
        "soundcheck_source_fx_return_headroom_send_only_until_slot_bound"
    ]["actual_selected_strategy_id"] == "fx_return_headroom_pullback_before_time_edits"
    source_fx_headroom_params = results[
        "soundcheck_source_fx_return_headroom_send_only_until_slot_bound"
    ]["selected_parameters"]
    assert source_fx_headroom_params["send_delta_db"] == -1.3
    assert source_fx_headroom_params["return_delta_db"] == 0.0
    assert source_fx_headroom_params["fx_level_target_surface"] == "channel"
    assert source_fx_headroom_params["fx_level_targeting_policy"] == (
        "source_send_only_until_fx_return_slot_resolved"
    )
    assert (
        source_fx_headroom_params[
            "fx_return_slot_binding_required_before_return_level_move"
        ]
        is True
    )
    assert source_fx_headroom_params["post_pullback_fx_return_level_review_db"] == -1.5
    assert (
        source_fx_headroom_params[
            "post_pullback_fx_return_level_review_target_id_required"
        ]
        is True
    )
    assert (
        source_fx_headroom_params[
            "post_pullback_fx_return_level_review_required_surface"
        ]
        == "fx"
    )
    assert (
        source_fx_headroom_params["wing_native_source_send_before_fx_return_slot_binding"]
        is True
    )
    assert "wing_fx_return_delta_db" not in source_fx_headroom_params
    _assert_replay_fx_send_level_mapping(
        results["soundcheck_source_fx_return_headroom_send_only_until_slot_bound"],
        expected_delta=source_fx_headroom_params["send_delta_db"],
        target_surface="channel",
        source_binding_required=False,
    )
    _assert_replay_selected_fx_apply_values(
        results["soundcheck_source_fx_return_headroom_send_only_until_slot_bound"],
        {"wing_fx_send_level_delta_db": -1.3},
    )
    _assert_replay_fx_level_selected_parameter_rollback_operations(
        results["soundcheck_source_fx_return_headroom_send_only_until_slot_bound"],
        selected_write_keys=["wing_fx_send_level_delta_db"],
    )
    source_fx_readback = results[
        "soundcheck_source_fx_return_headroom_send_only_until_slot_bound"
    ]["selected_future_wing_readback_contract"]["fx_readback"]
    assert source_fx_readback["fx_return_target_id"] == ""
    assert source_fx_readback["fx_return_target_surface"] == "unresolved"
    assert (
        source_fx_readback[
            "fx_return_slot_binding_required_before_return_level_move"
        ]
        is True
    )
    assert source_fx_readback["post_pullback_fx_return_level_review_db"] == -1.5
    assert (
        source_fx_readback["post_pullback_fx_return_level_review_target_id_required"]
        is True
    )
    assert source_fx_readback["post_pullback_fx_return_level_review_required_surface"] == "fx"
    assert source_fx_readback["wing_native_source_send_before_fx_return_slot_binding"] is True
    assert "wing_fx_return_level_mapping_required" not in source_fx_readback
    assert results[
        "soundcheck_source_fx_return_headroom_send_only_until_slot_bound"
    ]["selected_rationale"]["selection_policy"] == (
        "fx_return_headroom_restoration_priority"
    )
    assert results[
        "soundcheck_source_fx_return_headroom_bound_slot_uses_send_and_return"
    ]["actual_selected_strategy_id"] == "fx_return_headroom_pullback_before_time_edits"
    bound_source_fx_params = results[
        "soundcheck_source_fx_return_headroom_bound_slot_uses_send_and_return"
    ]["selected_parameters"]
    assert bound_source_fx_params["send_delta_db"] == -1.3
    assert bound_source_fx_params["return_delta_db"] == -1.5
    assert bound_source_fx_params["fx_return_target_id"] == "fx:1"
    assert bound_source_fx_params["fx_return_target_surface"] == "fx"
    assert bound_source_fx_params["wing_fx_send_level_delta_db"] == -1.3
    assert bound_source_fx_params["wing_fx_return_delta_db"] == -1.5
    assert bound_source_fx_params["wing_fx_return_source_target_surface"] == "channel"
    assert bound_source_fx_params["wing_fx_return_target_id"] == "fx:1"
    assert bound_source_fx_params["wing_fx_return_target_surface"] == "fx"
    assert bound_source_fx_params["wing_fx_return_fx_slot_binding_required"] is False
    assert bound_source_fx_params["soundcheck_fx_level_move_kinds"] == [
        "send_level",
        "return_level",
    ]
    assert bound_source_fx_params["soundcheck_fx_selected_write_keys"] == [
        "wing_fx_send_level_delta_db",
        "wing_fx_return_delta_db",
    ]
    assert bound_source_fx_params["soundcheck_fx_forbidden_write_keys"] == []
    assert "post_pullback_fx_return_level_review_db" not in bound_source_fx_params
    bound_source_fx_readback = results[
        "soundcheck_source_fx_return_headroom_bound_slot_uses_send_and_return"
    ]["selected_future_wing_readback_contract"]["fx_readback"]
    assert bound_source_fx_readback["fx_return_target_id"] == "fx:1"
    assert bound_source_fx_readback["fx_return_target_surface"] == "fx"
    assert bound_source_fx_readback["wing_fx_send_level_mapping_required"] is True
    assert bound_source_fx_readback["wing_fx_return_level_mapping_required"] is True
    assert bound_source_fx_readback["wing_fx_return_level_target_surface"] == "fx"
    assert bound_source_fx_readback["wing_fx_return_source_target_surface"] == "channel"
    assert bound_source_fx_readback["wing_fx_return_target_id"] == "fx:1"
    assert bound_source_fx_readback["wing_fx_return_target_surface"] == "fx"
    assert bound_source_fx_readback["soundcheck_fx_level_move_kinds"] == [
        "send_level",
        "return_level",
    ]
    assert bound_source_fx_readback["soundcheck_fx_selected_write_keys"] == [
        "wing_fx_send_level_delta_db",
        "wing_fx_return_delta_db",
    ]
    _assert_replay_selected_fx_apply_values(
        results["soundcheck_source_fx_return_headroom_bound_slot_uses_send_and_return"],
        {
            "wing_fx_send_level_delta_db": -1.3,
            "wing_fx_return_delta_db": -1.5,
        },
    )
    _assert_replay_fx_level_selected_parameter_rollback_operations(
        results["soundcheck_source_fx_return_headroom_bound_slot_uses_send_and_return"],
        selected_write_keys=[
            "wing_fx_send_level_delta_db",
            "wing_fx_return_delta_db",
        ],
    )
    assert results[
        "soundcheck_source_fx_return_headroom_bound_slot_uses_send_and_return"
    ]["selected_rationale"]["selection_policy"] == (
        "fx_return_headroom_restoration_priority"
    )
    assert results["soundcheck_aux_playback_fx_send_pullback"][
        "actual_selected_strategy_id"
    ] == "reduce_fx_send_or_return"
    aux_fx_params = results["soundcheck_aux_playback_fx_send_pullback"][
        "selected_parameters"
    ]
    assert aux_fx_params["send_delta_db"] == -4.1
    assert aux_fx_params["return_delta_db"] == 0.0
    assert aux_fx_params["fx_level_target_surface"] == "aux"
    assert aux_fx_params["fx_level_targeting_policy"] == (
        "source_send_uses_channel_aux_or_bus_send_level"
    )
    assert aux_fx_params["wing_fx_send_level_delta_db"] == -4.1
    assert aux_fx_params["wing_fx_send_level_target_surface"] == "aux"
    assert aux_fx_params["wing_fx_send_source_surface_supported"] is True
    assert aux_fx_params["wing_fx_send_source_surface_apply_ready"] is True
    assert aux_fx_params["wing_fx_send_source_binding_required"] is False
    assert aux_fx_params["soundcheck_fx_level_move_kinds"] == ["send_level"]
    assert aux_fx_params["soundcheck_fx_selected_write_keys"] == [
        "wing_fx_send_level_delta_db"
    ]
    assert aux_fx_params["soundcheck_fx_forbidden_write_keys"] == [
        "wing_fx_return_delta_db"
    ]
    assert aux_fx_params["native_fx_level_pullback_db"] == 4.1
    assert aux_fx_params["native_fx_level_pullback_legacy_db"] == 4.1
    assert aux_fx_params["post_pullback_time_edit_review_required"] is True
    _assert_replay_fx_send_level_mapping(
        results["soundcheck_aux_playback_fx_send_pullback"],
        expected_delta=aux_fx_params["send_delta_db"],
        target_surface="aux",
        source_binding_required=False,
    )
    _assert_replay_selected_fx_apply_values(
        results["soundcheck_aux_playback_fx_send_pullback"],
        {"wing_fx_send_level_delta_db": -4.1},
    )
    _assert_replay_fx_level_selected_parameter_rollback_operations(
        results["soundcheck_aux_playback_fx_send_pullback"],
        selected_write_keys=["wing_fx_send_level_delta_db"],
    )
    aux_fx_readback = results["soundcheck_aux_playback_fx_send_pullback"][
        "selected_future_wing_readback_contract"
    ]["fx_readback"]
    assert "/aux/{aux}/send/{send}/lvl" in aux_fx_readback[
        "wing_osc_path_templates"
    ]
    assert aux_fx_readback["wing_fx_send_level_target_surface"] == "aux"
    assert aux_fx_readback["wing_fx_send_source_surface_apply_ready"] is True
    assert aux_fx_readback["wing_fx_model_time_followup_review_required"] is True
    assert aux_fx_readback["wing_fx_model_time_followup_apply_ready"] is False
    assert results["soundcheck_aux_playback_fx_send_pullback"][
        "selected_rationale"
    ]["selection_policy"] == "soundcheck_wing_native_osc_fx_level_pullback_priority"
    assert results["soundcheck_fx_return_headroom_before_delay_ducking"][
        "actual_selected_strategy_id"
    ] == "fx_return_headroom_pullback_before_time_edits"
    fx_headroom_params = results[
        "soundcheck_fx_return_headroom_before_delay_ducking"
    ]["selected_parameters"]
    assert fx_headroom_params["return_delta_db"] == -1.0
    assert fx_headroom_params["send_delta_db"] == 0.0
    assert fx_headroom_params["fx_return_headroom_db"] == 1.7
    assert fx_headroom_params["target_fx_return_headroom_db"] == 2.5
    assert fx_headroom_params["fx_return_headroom_deficit_db"] == 0.8
    assert fx_headroom_params["return_headroom_restoration_first"] is True
    _assert_replay_fx_return_fxmix_mapping(
        results["soundcheck_fx_return_headroom_before_delay_ducking"],
        expected_delta=fx_headroom_params["return_delta_db"],
        target_surface="fx",
        fx_slot_binding_required=False,
    )
    _assert_replay_selected_fx_apply_values(
        results["soundcheck_fx_return_headroom_before_delay_ducking"],
        {"wing_fx_return_delta_db": -1.0},
    )
    assert results["soundcheck_fx_return_headroom_before_delay_ducking"][
        "selected_rationale"
    ]["selection_policy"] == "fx_return_headroom_restoration_priority"
    fx_headroom_blockers = results[
        "soundcheck_fx_return_headroom_before_delay_ducking"
    ]["candidate_blockers"]
    assert "fx_return_headroom_pullback_should_precede_delay_ducking" in (
        fx_headroom_blockers["duck_delay_return_cleanup"]
    )
    assert results["live_delay_smear_fx_pullback_before_ducking"][
        "actual_selected_strategy_id"
    ] == "reduce_fx_send_or_return"
    assert results["live_delay_smear_fx_pullback_before_ducking"][
        "selected_parameters"
    ]["live_max_single_step_db"] == 0.75
    live_delay_params = results["live_delay_smear_fx_pullback_before_ducking"][
        "selected_parameters"
    ]
    assert live_delay_params["fx_level_pullback_parameter_source"] == (
        "measured_delay_repeat_native_level_pullback"
    )
    assert live_delay_params["native_fx_level_pullback_db"] == 3.0
    assert live_delay_params["native_fx_level_pullback_legacy_db"] == 3.0
    assert live_delay_params["measured_tail_decay_s"] == 1.2
    assert live_delay_params["measured_delay_repeat_ms"] == 130.0
    assert live_delay_params["live_fx_level_pullback_conservative_cap_db"] == 3.0
    assert live_delay_params["live_fx_pullback_only_proposal"] is True
    assert live_delay_params["live_fx_level_reduction_only_required"] is True
    assert live_delay_params["live_fx_max_pullback_db"] == LIVE_FX_MAX_PULLBACK_DB
    assert live_delay_params["live_fx_max_boost_db"] == LIVE_FX_MAX_BOOST_DB
    assert live_delay_params["live_fx_native_level_osc_only_required"] is True
    assert live_delay_params["live_fx_no_model_or_time_parameter_write_required"] is True
    assert live_delay_params["live_fx_absolute_level_readback_required"] is True
    assert (
        live_delay_params["live_fx_monitor_feedback_recheck_after_apply_required"]
        is True
    )
    assert (
        live_delay_params["live_fx_return_headroom_recheck_after_apply_required"]
        is True
    )
    assert live_delay_params["live_fx_operator_rehearsal_required"] is True
    assert live_delay_params["live_fx_live_promotion_status"] == (
        "blocked_until_real_wing_fx_level_readback_monitor_recheck_and_rollback"
    )
    assert live_delay_params["live_fx_live_apply_blocked_without_wing_proof"] is True
    _assert_replay_fx_return_fxmix_mapping(
        results["live_delay_smear_fx_pullback_before_ducking"],
        expected_delta=live_delay_params["return_delta_db"],
        target_surface="fx",
        fx_slot_binding_required=False,
    )
    _assert_replay_selected_fx_apply_values(
        results["live_delay_smear_fx_pullback_before_ducking"],
        {"wing_fx_return_delta_db": -3.0},
    )
    _assert_replay_fx_model_time_followup_review_contract(
        results["live_delay_smear_fx_pullback_before_ducking"],
        expected_values={
            "post_pullback_delay_time_review_ms": live_delay_params[
                "post_pullback_delay_time_review_ms"
            ],
            "post_pullback_delay_feedback_review_percent": live_delay_params[
                "post_pullback_delay_feedback_review_percent"
            ],
            "post_pullback_delay_low_pass_review_hz": live_delay_params[
                "post_pullback_delay_low_pass_review_hz"
            ],
            "post_pullback_delay_ducking_review_db": live_delay_params[
                "post_pullback_delay_ducking_review_db"
            ],
            "post_pullback_delay_ducking_release_review_ms": live_delay_params[
                "post_pullback_delay_ducking_release_review_ms"
            ],
            "post_pullback_return_hpf_review_hz": live_delay_params[
                "post_pullback_return_hpf_review_hz"
            ],
            "post_pullback_return_lpf_review_hz": live_delay_params[
                "post_pullback_return_lpf_review_hz"
            ],
        },
        expected_native_apply_fields=["wing_fx_return_delta_db"],
        expected_review_fields=WING_FX_POST_PULLBACK_MODEL_TIME_REVIEW_FIELDS,
    )
    assert results["live_delay_smear_fx_pullback_before_ducking"][
        "operator_explanation_sections"
    ]["selection_policy"] == "live_fx_safety_pullback_priority"
    assert results["wet_fx_monitor_risk"][
        "actual_selected_strategy_id"
    ] == "fx_monitor_feedback_safety_pullback"
    wet_fx_params = results["wet_fx_monitor_risk"]["selected_parameters"]
    assert wet_fx_params["send_delta_db"] == 0.0
    _assert_replay_fx_return_fxmix_mapping(
        results["wet_fx_monitor_risk"],
        expected_delta=wet_fx_params["return_delta_db"],
        target_surface="fx",
        fx_slot_binding_required=False,
    )
    _assert_replay_selected_fx_apply_values(
        results["wet_fx_monitor_risk"],
        {"wing_fx_return_delta_db": -5.5},
    )
    assert results["extreme_noisy_tom_manual_mute_review"][
        "actual_selected_strategy_id"
    ] == "manual_tom_mute_between_hits"
    _assert_replay_mute_mapping(
        results["extreme_noisy_tom_manual_mute_review"],
        expected_value=1,
        path_prefix="/ch/{channel}",
    )
    _assert_replay_selected_mute_apply_values(
        results["extreme_noisy_tom_manual_mute_review"],
        WING_MUTE_ON_VALUE,
    )
    assert results["measured_tom_tail_risk_expander"][
        "actual_selected_strategy_id"
    ] == "gentle_expander_bleed_control"
    assert results["measured_tom_tail_risk_expander"]["selected_parameters"][
        "expander_parameter_source"
    ] == "measured_hit_tail_bleed_expander"
    assert results["measured_tom_tail_risk_expander"]["selected_parameters"][
        "threshold_dbfs"
    ] == -43.3
    assert results["measured_tom_tail_risk_expander"]["selected_parameters"][
        "release_ms"
    ] == 374.5
    assert results["measured_tom_tail_risk_expander"]["selected_parameters"][
        "wing_gate_model_readback_must_equal"
    ] == WING_GATE_REQUIRED_MODEL
    expected_measured_tom_expander_apply_values = {
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
    }
    _assert_replay_selected_gate_apply_values(
        results["measured_tom_tail_risk_expander"],
        expected_measured_tom_expander_apply_values,
    )
    assert results[
        "measured_tom_tail_risk_expander"
    ]["selected_rationale"]["wing_osc_apply_parameter_values"] == (
        expected_measured_tom_expander_apply_values
    )
    measured_tom_gate_readback = results["measured_tom_tail_risk_expander"][
        "selected_future_wing_readback_contract"
    ]["gate_expander_readback"]
    assert measured_tom_gate_readback["gate_model_readback_must_equal"] == (
        WING_GATE_REQUIRED_MODEL
    )
    assert "gate/mdl" in measured_tom_gate_readback["read_before_paths"]
    measured_tom_params = results["measured_tom_tail_risk_expander"][
        "selected_parameters"
    ]
    assert measured_tom_params["gate_sidechain_frequency_hz"] == 160.0
    assert measured_tom_params["gate_sidechain_q"] == 2.99
    assert measured_tom_params["gate_sidechain_parameter_source"] == (
        "tom_role_default_frequency"
    )
    assert measured_tom_params["gate_sidechain_hit_to_bleed_margin_db"] == 18.0
    assert measured_tom_params["gate_sidechain_tail_decay_ms"] == 190.0
    assert measured_tom_params["gate_sidechain_transient_strength"] == 0.62
    assert measured_tom_params["soundcheck_gate_expander_bounded_proposal"] is True
    assert (
        measured_tom_params["soundcheck_gate_expander_native_gate_osc_only_required"]
        is True
    )
    assert measured_tom_params["soundcheck_gate_expander_max_range_db"] == (
        SOUNDCHECK_GATE_EXPANDER_MAX_RANGE_DB
    )
    assert measured_tom_params[
        "soundcheck_gate_expander_max_hysteresis_review_db"
    ] == SOUNDCHECK_GATE_EXPANDER_MAX_HYSTERESIS_REVIEW_DB
    assert measured_tom_params["soundcheck_gate_expander_min_hit_bleed_margin_db"] == (
        SOUNDCHECK_GATE_EXPANDER_MIN_HIT_BLEED_MARGIN_DB
    )
    assert measured_tom_params["soundcheck_gate_expander_clean_hit_bleed_margin_db"] == 18.0
    assert measured_tom_params[
        "soundcheck_gate_expander_clean_hit_bleed_evidence_status"
    ] == "measured_margin_ready"
    assert (
        measured_tom_params[
            "soundcheck_expander_allows_risk_score_fallback_with_operator_recheck"
        ]
        is True
    )
    assert (
        measured_tom_gate_readback[
            "soundcheck_gate_expander_live_apply_blocked_without_wing_proof"
        ]
        is True
    )
    assert results["clean_tom_bleed_aware_gate"][
        "actual_selected_strategy_id"
    ] == "bleed_aware_clean_hit_gate"
    assert results["clean_tom_bleed_aware_gate"]["selected_parameters"][
        "clean_hit_gate_ready"
    ] is True
    assert results["clean_tom_bleed_aware_gate"]["selected_parameters"][
        "gate_parameter_source"
    ] == "measured_hit_tail"
    assert results["clean_tom_bleed_aware_gate"]["selected_parameters"][
        "gate_enabled"
    ] is True
    assert results["clean_tom_bleed_aware_gate"]["selected_parameters"][
        "wing_gate_on_osc_path_suffix"
    ] == "gate/on"
    assert results["clean_tom_bleed_aware_gate"]["selected_parameters"][
        "wing_gate_on_osc_value"
    ] == WING_GATE_ON_MAX
    assert results["clean_tom_bleed_aware_gate"]["selected_parameters"][
        "wing_gate_on_value_type"
    ] == "integer_bool"
    assert results["clean_tom_bleed_aware_gate"]["selected_parameters"][
        "gate_tail_decay_ms"
    ] == 80.0
    assert results["clean_tom_bleed_aware_gate"]["selected_parameters"][
        "accent_percent"
    ] == 51.0
    assert results["clean_tom_bleed_aware_gate"]["selected_parameters"][
        "wing_gate_accent_osc_path_suffix"
    ] == "gate/acc"
    assert results["clean_tom_bleed_aware_gate"]["selected_parameters"][
        "wing_gate_accent_osc_value"
    ] == 51.0
    assert results["clean_tom_bleed_aware_gate"]["selected_parameters"][
        "gate_sidechain_filter_type"
    ] == "BP"
    assert results["clean_tom_bleed_aware_gate"]["selected_parameters"][
        "gate_sidechain_frequency_hz"
    ] == 175.4
    assert results["clean_tom_bleed_aware_gate"]["selected_parameters"][
        "gate_sidechain_q"
    ] == 2.75
    assert results["clean_tom_bleed_aware_gate"]["selected_parameters"][
        "gate_sidechain_parameter_source"
    ] == "measured_clean_hit_detector_band"
    assert results["clean_tom_bleed_aware_gate"]["selected_parameters"][
        "gate_sidechain_evidence_frequency_hz"
    ] == 0.0
    assert results["clean_tom_bleed_aware_gate"]["selected_parameters"][
        "gate_sidechain_hit_to_bleed_margin_db"
    ] == 40.0
    assert results["clean_tom_bleed_aware_gate"]["selected_parameters"][
        "gate_sidechain_tail_decay_ms"
    ] == 80.0
    assert results["clean_tom_bleed_aware_gate"]["selected_parameters"][
        "gate_sidechain_transient_strength"
    ] == 0.72
    assert results["clean_tom_bleed_aware_gate"]["selected_parameters"][
        "wing_gate_sidechain_type_osc_value"
    ] in WING_GATE_SIDECHAIN_TYPE_VALUES
    assert results["clean_tom_bleed_aware_gate"]["selected_parameters"][
        "wing_gate_sidechain_frequency_osc_value"
    ] == 175.4
    assert (
        WING_GATE_SIDECHAIN_FREQ_MIN_HZ
        <= results["clean_tom_bleed_aware_gate"]["selected_parameters"][
            "wing_gate_sidechain_frequency_osc_value"
        ]
        <= WING_GATE_SIDECHAIN_FREQ_MAX_HZ
    )
    assert results["clean_tom_bleed_aware_gate"]["selected_parameters"][
        "wing_gate_sidechain_q_osc_value"
    ] == 2.75
    assert (
        WING_GATE_SIDECHAIN_Q_MIN
        <= results["clean_tom_bleed_aware_gate"]["selected_parameters"][
            "wing_gate_sidechain_q_osc_value"
        ]
        <= WING_GATE_SIDECHAIN_Q_MAX
    )
    assert results["clean_tom_bleed_aware_gate"]["selected_parameters"][
        "wing_gate_sidechain_src_osc_value"
    ] in WING_GATE_SIDECHAIN_SOURCE_VALUES
    assert results["clean_tom_bleed_aware_gate"]["selected_parameters"][
        "wing_gate_sidechain_tap_osc_value"
    ] in WING_GATE_SIDECHAIN_TAP_VALUES
    clean_tom_gate_readback = results["clean_tom_bleed_aware_gate"][
        "selected_future_wing_readback_contract"
    ]["gate_expander_readback"]
    clean_tom_gate_wing_osc = results["clean_tom_bleed_aware_gate"][
        "selected_rationale"
    ]
    assert clean_tom_gate_wing_osc["wing_osc_candidate_apply_status"] == (
        "wing_native_osc_candidate_has_review_only_parameters_not_apply_ready_"
        "requires_operator_readback"
    )
    assert clean_tom_gate_wing_osc["wing_osc_review_only_parameter_keys"] == [
        "post_gate_hysteresis_review_db"
    ]
    assert clean_tom_gate_wing_osc[
        "wing_osc_review_only_parameters_apply_ready"
    ] is False
    expected_clean_tom_gate_apply_values = {
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
    _assert_replay_selected_gate_apply_values(
        results["clean_tom_bleed_aware_gate"],
        expected_clean_tom_gate_apply_values,
    )
    _assert_replay_selected_parameter_rollback_operations(
        results["clean_tom_bleed_aware_gate"],
        readback_section="gate_expander_readback",
        operations_field="wing_gate_selected_parameter_rollback_operations",
        selected_write_keys=list(WING_GATE_WRITE_PARAMETER_KEYS),
        contract_marker="read_previous_gate_value_before_write_then_restore_same_path",
    )
    assert clean_tom_gate_readback["wing_gate_on_mapping_required"] is True
    assert clean_tom_gate_readback["wing_gate_on_osc_path_suffix"] == "gate/on"
    assert clean_tom_gate_readback["wing_gate_on_osc_value"] == WING_GATE_ON_MAX
    assert (
        clean_tom_gate_readback["wing_gate_on_readback_must_match_osc_value"]
        is True
    )
    assert clean_tom_gate_readback["wing_gate_accent_mapping_required"] is True
    assert clean_tom_gate_readback["wing_gate_accent_osc_path_suffix"] == "gate/acc"
    assert clean_tom_gate_readback["wing_gate_accent_osc_value"] == 51.0
    assert (
        clean_tom_gate_readback["wing_gate_accent_readback_must_match_osc_value"]
        is True
    )
    assert clean_tom_gate_readback["soundcheck_gate_expander_bounded_proposal"] is True
    assert (
        clean_tom_gate_readback[
            "soundcheck_gate_expander_native_gate_osc_only_required"
        ]
        is True
    )
    assert (
        clean_tom_gate_readback[
            "soundcheck_gate_expander_live_apply_blocked_without_wing_proof"
        ]
        is True
    )
    clean_tom_hysteresis_target = results["clean_tom_bleed_aware_gate"][
        "selected_parameters"
    ]["post_gate_hysteresis_review_db"]
    assert clean_tom_gate_readback[
        "wing_gate_hysteresis_review_only_parameter_key"
    ] == "post_gate_hysteresis_review_db"
    assert clean_tom_gate_readback[
        "wing_gate_hysteresis_review_only_value_db"
    ] == clean_tom_hysteresis_target
    assert clean_tom_gate_readback[
        "wing_gate_hysteresis_review_tolerance_db"
    ] == WING_GATE_HYSTERESIS_REVIEW_TOLERANCE_DB
    assert clean_tom_gate_readback[
        "wing_gate_hysteresis_review_acceptance_min_db"
    ] == round(
        max(
            0.0,
            clean_tom_hysteresis_target
            - WING_GATE_HYSTERESIS_REVIEW_TOLERANCE_DB,
        ),
        3,
    )
    assert clean_tom_gate_readback[
        "wing_gate_hysteresis_review_acceptance_max_db"
    ] == round(
        min(
            SOUNDCHECK_GATE_EXPANDER_MAX_HYSTERESIS_REVIEW_DB,
            clean_tom_hysteresis_target
            + WING_GATE_HYSTERESIS_REVIEW_TOLERANCE_DB,
        ),
        3,
    )
    assert clean_tom_gate_readback[
        "wing_gate_hysteresis_review_must_not_exceed_db"
    ] == SOUNDCHECK_GATE_EXPANDER_MAX_HYSTERESIS_REVIEW_DB
    assert clean_tom_gate_readback[
        "wing_gate_hysteresis_review_forbidden_path_suffixes"
    ] == ["gate/hysteresis", "gate/hyst"]
    assert "gate/hysteresis" not in clean_tom_gate_readback["read_before_paths"]
    assert "gate/hysteresis" not in clean_tom_gate_readback["read_after_paths"]
    assert "gate/hyst" not in clean_tom_gate_readback["read_before_paths"]
    assert "gate/hyst" not in clean_tom_gate_readback["read_after_paths"]
    assert clean_tom_gate_readback[
        "wing_gate_hysteresis_review_requires_audio_recheck"
    ] is True
    assert clean_tom_gate_readback[
        "wing_gate_hysteresis_review_requires_tail_chatter_recheck"
    ] is True
    assert "do not write a hysteresis OSC value" in clean_tom_gate_readback[
        "wing_gate_hysteresis_review_operator_check"
    ]
    assert (
        clean_tom_gate_readback["soundcheck_gate_requires_measured_clean_hit_margin"]
        is True
    )
    assert clean_tom_gate_readback["wing_gate_sidechain_mapping_required"] is True
    assert clean_tom_gate_readback["wing_gate_sidechain_type_osc_path_suffix"] == (
        "gatesc/type"
    )
    assert clean_tom_gate_readback["wing_gate_sidechain_type_osc_value"] == "BP"
    assert clean_tom_gate_readback[
        "wing_gate_sidechain_frequency_osc_path_suffix"
    ] == "gatesc/f"
    assert clean_tom_gate_readback["wing_gate_sidechain_frequency_osc_value"] == 175.4
    assert clean_tom_gate_readback["wing_gate_sidechain_q_osc_path_suffix"] == (
        "gatesc/q"
    )
    assert clean_tom_gate_readback["wing_gate_sidechain_q_osc_value"] == 2.75
    assert clean_tom_gate_readback["wing_gate_sidechain_src_osc_path_suffix"] == (
        "gatesc/src"
    )
    assert clean_tom_gate_readback["wing_gate_sidechain_src_osc_value"] == "SELF"
    assert clean_tom_gate_readback["wing_gate_sidechain_tap_osc_path_suffix"] == (
        "gatesc/tap"
    )
    assert clean_tom_gate_readback["wing_gate_sidechain_tap_osc_value"] == "IN"
    assert (
        clean_tom_gate_readback["wing_gate_sidechain_readback_must_match_osc_value"]
        is True
    )
    assert "gatesc/type" in clean_tom_gate_readback["read_before_paths"]
    assert "gatesc/f" in clean_tom_gate_readback["read_after_paths"]
    assert results["soundcheck_operator_owned_gate_slot_hold"][
        "actual_selected_strategy_id"
    ] == "hold_soundcheck_operator_owned_gate_slot_for_review"
    assert results["soundcheck_operator_owned_gate_slot_hold"][
        "actual_method_family"
    ] == "analysis_only"
    assert results["soundcheck_operator_owned_gate_slot_hold"][
        "operator_explanation_sections"
    ]["selection_policy"] == (
        "soundcheck_operator_owned_gate_slot_review_priority"
    )
    owned_gate = results["soundcheck_operator_owned_gate_slot_hold"]
    owned_gate_params = owned_gate["selected_parameters"]
    assert owned_gate_params["soundcheck_operator_owned_gate_slot_hold_proposal"] is True
    assert owned_gate_params["soundcheck_operator_owned_gate_slot_target_surface"] == (
        "channel"
    )
    assert owned_gate_params[
        "soundcheck_operator_owned_gate_slot_previous_model_readback_required"
    ] is True
    assert owned_gate_params[
        "soundcheck_operator_owned_gate_slot_previous_gate_parameters_readback_required"
    ] is True
    assert owned_gate_params[
        "soundcheck_operator_owned_gate_slot_sidechain_readback_required"
    ] is True
    assert owned_gate_params["soundcheck_operator_owned_gate_slot_rollback_required"] is True
    assert owned_gate_params[
        "soundcheck_operator_owned_gate_slot_no_gate_or_expander_without_operator_review"
    ] is True
    assert owned_gate_params["blocked_gate_operator_owned_blockers"] == [
        WING_GATE_SLOT_OPERATOR_OWNED_MODEL_BLOCKER
    ]
    assert owned_gate_params["blocked_gate_method_family"] == "gate"
    assert owned_gate_params["blocked_gate_threshold_dbfs"] == -46.9
    assert owned_gate_params["blocked_gate_range_db"] == 11.6
    assert owned_gate_params["blocked_gate_sidechain_frequency_hz"] == 160.0
    assert owned_gate_params["wing_gate_slot_current_model"] == "GATE"
    assert owned_gate_params["wing_gate_slot_current_enabled"] is True
    assert owned_gate_params["wing_gate_slot_current_operator_owned"] is True
    assert (
        owned_gate_params["wing_gate_slot_operator_owned_model_blocker"]
        == WING_GATE_SLOT_OPERATOR_OWNED_MODEL_BLOCKER
    )
    owned_gate_readback = owned_gate["selected_future_wing_readback_contract"][
        "soundcheck_operator_owned_gate_slot_review_readback"
    ]
    assert owned_gate_readback["read_only"] is True
    assert owned_gate_readback[
        "soundcheck_operator_owned_gate_slot_current_operator_owned"
    ] is True
    assert owned_gate_readback[
        "soundcheck_operator_owned_gate_slot_operator_owned_blockers"
    ] == [WING_GATE_SLOT_OPERATOR_OWNED_MODEL_BLOCKER]
    assert "gate/mdl" in owned_gate_readback["read_before_paths"]
    assert "gate/on" in owned_gate_readback["read_before_paths"]
    assert "gate/thr21" in owned_gate_readback["read_before_paths"]
    assert "gatesc/type" in owned_gate_readback["read_before_paths"]
    assert "/ch/{channel}/gate/thr21" in owned_gate_readback[
        "wing_osc_path_templates"
    ]
    assert owned_gate["candidate_blockers"]["bleed_aware_clean_hit_gate"] == [
        WING_GATE_SLOT_OPERATOR_OWNED_MODEL_BLOCKER
    ]
    assert owned_gate["candidate_blockers"]["gentle_expander_bleed_control"] == [
        WING_GATE_SLOT_OPERATOR_OWNED_MODEL_BLOCKER
    ]
    owned_gate_tradeoffs = {
        row["strategy_id"]: row["why_not_selected"]
        for row in owned_gate["alternative_tradeoffs"]
    }
    assert owned_gate_tradeoffs["bleed_aware_clean_hit_gate"] == (
        "soundcheck_operator_owned_gate_slot_review_priority:"
        "manual_gate_slot_owner_review_precedes_gate_or_expander_change"
    )
    assert owned_gate_tradeoffs["gentle_expander_bleed_control"] == (
        "soundcheck_operator_owned_gate_slot_review_priority:"
        "manual_gate_slot_owner_review_precedes_gate_or_expander_change"
    )
    _assert_replay_analysis_only_has_no_apply_values(owned_gate)
    assert results["low_margin_tom_clean_hit_evidence_hold"][
        "actual_selected_strategy_id"
    ] == "hold_noisy_drum_for_clean_hit_bleed_evidence"
    assert results["low_margin_tom_clean_hit_evidence_hold"]["actual_status"] == (
        "analysis_only"
    )
    assert results["low_margin_tom_clean_hit_evidence_hold"][
        "operator_explanation_sections"
    ]["selection_policy"] == (
        "soundcheck_noisy_drum_clean_hit_bleed_evidence_priority"
    )
    assert results["low_margin_tom_clean_hit_evidence_hold"][
        "selected_parameters"
    ]["minimum_hit_to_bleed_margin_db"] == 6.0
    assert results["low_margin_tom_clean_hit_evidence_hold"][
        "candidate_blockers"
    ]["bleed_aware_clean_hit_gate"] == [
        "hit_bleed_margin_too_low_for_gate",
        "clean_hit_gate_requires_low_tail_chatter_and_strong_hits",
        "soundcheck_gate_expander_range_exceeds_bounded_limit",
        "soundcheck_gate_expander_hysteresis_review_exceeds_bounded_limit",
        "soundcheck_gate_clean_hit_bleed_evidence_required",
    ]
    assert results["live_noisy_tom_hold_before_gate_or_expander"][
        "actual_selected_strategy_id"
    ] == "hold_live_noisy_drum_processing_for_operator"
    assert results["live_noisy_tom_hold_before_gate_or_expander"][
        "operator_explanation_sections"
    ]["selection_policy"] == "live_noisy_drum_review_priority"
    live_noisy_tom = results["live_noisy_tom_hold_before_gate_or_expander"]
    assert live_noisy_tom["selected_parameters"]["live_noisy_drum_hold_proposal"] is True
    assert live_noisy_tom["selected_parameters"]["live_noisy_drum_target_surface"] == (
        "channel"
    )
    assert live_noisy_tom["selected_parameters"][
        "live_noisy_drum_clean_hit_detection_required"
    ] is True
    assert live_noisy_tom["selected_parameters"][
        "live_noisy_drum_previous_gate_expander_readback_required"
    ] is True
    assert live_noisy_tom["selected_parameters"][
        "live_noisy_drum_previous_mute_readback_required"
    ] is True
    assert live_noisy_tom["selected_parameters"][
        "live_noisy_drum_no_gate_expander_mute_without_operator_wing_proof"
    ] is True
    assert "wing_gate_on_osc_value" not in live_noisy_tom["selected_parameters"]
    assert "wing_mute_osc_value" not in live_noisy_tom["selected_parameters"]
    live_noisy_tom_readback = live_noisy_tom["selected_future_wing_readback_contract"][
        "live_noisy_drum_hold_readback"
    ]
    assert live_noisy_tom_readback["read_only"] is True
    assert live_noisy_tom_readback["live_noisy_drum_target_surface"] == "channel"
    assert live_noisy_tom_readback[
        "live_noisy_drum_gate_model_readback_required"
    ] is True
    assert live_noisy_tom_readback[
        "live_noisy_drum_mute_readback_required"
    ] is True
    assert live_noisy_tom_readback["allowed_live_write_scope_before_wing_proof"] == []
    assert live_noisy_tom_readback["wing_gate_threshold_osc_path_suffix"] == (
        "gate/thr21"
    )
    assert "/ch/{channel}/gate/thr21" in live_noisy_tom_readback[
        "wing_osc_path_templates"
    ]
    assert "/ch/{channel}/mute" in live_noisy_tom_readback["wing_osc_path_templates"]
    assert results["live_extreme_noisy_tom_hold_before_manual_mute"][
        "actual_selected_strategy_id"
    ] == "hold_live_noisy_drum_processing_for_operator"
    assert results["live_extreme_noisy_tom_hold_before_manual_mute"][
        "operator_explanation_sections"
    ]["selection_policy"] == "live_noisy_drum_review_priority"
    _assert_replay_analysis_only_has_no_apply_values(
        results["live_extreme_noisy_tom_hold_before_manual_mute"]
    )
    extreme_noisy_tom = results["live_extreme_noisy_tom_hold_before_manual_mute"]
    assert extreme_noisy_tom["selected_parameters"][
        "live_noisy_drum_hold_proposal"
    ] is True
    extreme_noisy_tom_readback = extreme_noisy_tom[
        "selected_future_wing_readback_contract"
    ]["live_noisy_drum_hold_readback"]
    assert extreme_noisy_tom_readback["read_only"] is True
    assert "/ch/{channel}/gate/thr21" in extreme_noisy_tom_readback[
        "wing_osc_path_templates"
    ]
    assert "/ch/{channel}/mute" in extreme_noisy_tom_readback[
        "wing_osc_path_templates"
    ]
    assert results["kick_bass_masking_low_headroom"]["actual_status"] == (
        "analysis_only"
    )
    assert results["kick_bass_masking_low_headroom"][
        "actual_selected_strategy_id"
    ] == "source_true_peak_headroom_pullback"
    assert results["kick_bass_masking_low_headroom"][
        "operator_explanation_sections"
    ]["selection_policy"] == "true_peak_headroom_safety_priority"
    assert "true_peak_headroom_low" in results[
        "kick_bass_masking_low_headroom"
    ]["global_blockers"]
    assert results["kick_bass_masking_low_headroom"]["selected_parameters"][
        "headroom_restoration_first"
    ] is True
    _assert_replay_gain_delta_mapping(
        results["kick_bass_masking_low_headroom"],
        source="delta_db",
        expected_delta=-1.1,
        path_prefix="/ch/{channel}",
    )
    _assert_replay_selected_gain_delta_apply_values(
        results["kick_bass_masking_low_headroom"],
        -1.1,
    )
    assert results["live_source_true_peak_headroom_pullback_before_eq"][
        "actual_selected_strategy_id"
    ] == "source_true_peak_headroom_pullback"
    assert results["live_source_true_peak_headroom_pullback_before_eq"][
        "actual_method_family"
    ] == "gain_fader"
    assert results["live_source_true_peak_headroom_pullback_before_eq"][
        "operator_explanation_sections"
    ]["selection_policy"] == "true_peak_headroom_safety_priority"
    assert "true_peak_headroom_low" in results[
        "live_source_true_peak_headroom_pullback_before_eq"
    ]["global_blockers"]
    assert results["live_source_true_peak_headroom_pullback_before_eq"][
        "selected_parameters"
    ]["live_stepped_change_required"] is True
    _assert_replay_gain_delta_mapping(
        results["live_source_true_peak_headroom_pullback_before_eq"],
        source="delta_db",
        expected_delta=-0.9,
        path_prefix="/ch/{channel}",
    )
    _assert_replay_selected_gain_delta_apply_values(
        results["live_source_true_peak_headroom_pullback_before_eq"],
        -0.9,
    )
    assert results["soundcheck_general_masking_static_unmasking_eq"][
        "actual_selected_strategy_id"
    ] == "subtractive_unmasking_eq_split"
    general_masking_params = results["soundcheck_general_masking_static_unmasking_eq"][
        "selected_parameters"
    ]
    assert general_masking_params["gain_db"] == -3.0
    assert general_masking_params["q"] == 1.79
    assert general_masking_params["q"] != 1.1
    assert general_masking_params["static_eq_q_source"] == (
        "measured_masking_overlap_band"
    )
    assert general_masking_params["masking_parameter_source"] == (
        "measured_masking_overlap_band"
    )
    assert general_masking_params["masking_bandwidth_hz"] == 360.0
    assert general_masking_params["masking_overlap_peak_share"] == 0.34
    assert general_masking_params["linked_counter_boost_db"] == 0.0
    assert results["soundcheck_general_masking_static_unmasking_eq"][
        "operator_explanation_sections"
    ]["selection_policy"] == "weighted_strategy_score"
    general_dynamic_params = results["soundcheck_general_masking_static_unmasking_eq"][
        "candidate_parameters"
    ]["dynamic_ducking_proposal"]
    assert general_dynamic_params["filter_type"] == "dynamic_peaking"
    assert general_dynamic_params["masking_parameter_source"] == (
        "measured_masking_overlap_band"
    )
    assert general_dynamic_params["masking_bandwidth_hz"] == 360.0
    assert general_dynamic_params["masking_overlap_peak_share"] == 0.34
    assert general_dynamic_params["bandwidth_q"] == 1.54
    assert general_dynamic_params["dynamic_eq_q"] == 1.54
    assert general_dynamic_params["max_gain_reduction_db"] == 2.8
    assert general_dynamic_params["dynamic_eq_max_gain_reduction_db"] == 2.8
    assert general_dynamic_params["threshold_dbfs"] == -20.6
    assert general_dynamic_params["dynamic_eq_threshold_dbfs"] == -20.6
    assert general_dynamic_params["attack_ms"] == 10.8
    assert general_dynamic_params["dynamic_eq_attack_ms"] == 10.8
    assert general_dynamic_params["release_ms"] == 112.7
    assert general_dynamic_params["dynamic_eq_release_ms"] == 112.7
    assert general_dynamic_params["makeup_gain_db"] == 0.0
    general_dynamic_wing = results["soundcheck_general_masking_static_unmasking_eq"][
        "candidate_future_wing_readback_contracts"
    ]["dynamic_ducking_proposal"]["wing_osc_applicability"]
    assert general_dynamic_wing["native_osc_supported"] is False
    assert general_dynamic_wing["native_osc_support_level"] == (
        "dynamic_eq_value_mapping_required"
    )
    assert general_dynamic_wing["unsupported_strategy_reason"] == (
        "dynamic_eq_sidechain_or_paired_ducking_requires_verified_wing_sidechain_mapping"
    )
    assert results["soundcheck_vocal_intelligibility_dynamic_unmasking"][
        "actual_selected_strategy_id"
    ] == "vocal_intelligibility_static_pocket_eq"
    assert results["soundcheck_vocal_intelligibility_dynamic_unmasking"][
        "selected_parameters"
    ]["linked_counter_boost_db"] == 0.0
    assert results["soundcheck_vocal_intelligibility_dynamic_unmasking"][
        "selected_parameters"
    ]["gain_db"] == -2.7
    assert results["soundcheck_vocal_intelligibility_dynamic_unmasking"][
        "selected_parameters"
    ]["q"] == 2.18
    assert results["soundcheck_vocal_intelligibility_dynamic_unmasking"][
        "selected_parameters"
    ]["static_eq_q_source"] == "measured_masking_overlap_band"
    assert results["soundcheck_vocal_intelligibility_dynamic_unmasking"][
        "selected_parameters"
    ]["masking_parameter_source"] == "measured_masking_overlap_band"
    assert results["soundcheck_vocal_intelligibility_dynamic_unmasking"][
        "operator_explanation_sections"
    ]["selection_policy"] == "weighted_strategy_score"
    vocal_dynamic_params = results["soundcheck_vocal_intelligibility_dynamic_unmasking"][
        "candidate_parameters"
    ]["vocal_intelligibility_dynamic_unmasking"]
    assert vocal_dynamic_params["masking_parameter_source"] == "measured_masking_overlap_band"
    assert vocal_dynamic_params["masking_bandwidth_hz"] == 760.0
    assert vocal_dynamic_params["bandwidth_q"] == 2.63
    assert results["live_vocal_intelligibility_static_pocket"][
        "actual_selected_strategy_id"
    ] == "vocal_intelligibility_static_pocket_eq"
    assert results["soundcheck_vocal_masking_over_level_rebalance"][
        "actual_selected_strategy_id"
    ] == "vocal_masking_level_plane_rebalance"
    assert results["soundcheck_vocal_masking_over_level_rebalance"][
        "operator_explanation_sections"
    ]["selection_policy"] == "soundcheck_vocal_masking_level_plane_priority"
    assert results["soundcheck_vocal_masking_over_level_rebalance"][
        "selected_parameters"
    ]["current_minus_target_lu"] == 3.6
    assert results["live_vocal_masking_over_level_rebalance"][
        "actual_selected_strategy_id"
    ] == "vocal_masking_level_plane_rebalance"
    assert results["live_vocal_masking_over_level_rebalance"][
        "operator_explanation_sections"
    ]["selection_policy"] == "live_vocal_masking_level_plane_priority"
    assert results["live_vocal_masking_over_level_rebalance"][
        "selected_parameters"
    ]["live_stepped_change_required"] is True
    assert results["live_vocal_masking_over_level_rebalance"][
        "selected_parameters"
    ]["live_max_single_step_db"] == 0.75
    assert results["soundcheck_kick_bass_dynamic_pocket"][
        "actual_selected_strategy_id"
    ] == "kick_bass_pocket_eq_split"
    assert results["soundcheck_kick_bass_dynamic_pocket"][
        "selected_parameters"
    ]["target_cut_db"] == -3.3
    assert results["soundcheck_kick_bass_dynamic_pocket"][
        "selected_parameters"
    ]["counterpart_cut_db"] == -2.1
    assert results["soundcheck_kick_bass_dynamic_pocket"][
        "selected_parameters"
    ]["q"] == 1.47
    assert results["soundcheck_kick_bass_dynamic_pocket"][
        "selected_parameters"
    ]["static_eq_q_source"] == "measured_masking_overlap_band"
    assert results["soundcheck_kick_bass_dynamic_pocket"][
        "selected_parameters"
    ]["masking_parameter_source"] == "measured_masking_overlap_band"
    assert results["soundcheck_kick_bass_dynamic_pocket"][
        "operator_explanation_sections"
    ]["selection_policy"] == "weighted_strategy_score"
    kick_bass_dynamic_params = results["soundcheck_kick_bass_dynamic_pocket"][
        "candidate_parameters"
    ]["kick_bass_dynamic_pocket_ducking"]
    assert kick_bass_dynamic_params["masking_parameter_source"] == (
        "measured_masking_overlap_band"
    )
    assert kick_bass_dynamic_params["masking_bandwidth_hz"] == 28.0
    assert kick_bass_dynamic_params["bandwidth_q"] == 1.82
    assert kick_bass_dynamic_params["dynamic_eq_q"] == 1.82
    assert kick_bass_dynamic_params["dynamic_eq_frequency_hz"] == 74.0
    assert kick_bass_dynamic_params["dynamic_eq_filter_type"] == "dynamic_peaking"
    assert kick_bass_dynamic_params["dynamic_eq_ratio"] == 2.0
    assert kick_bass_dynamic_params["dynamic_eq_threshold_dbfs"] == -19.4
    assert kick_bass_dynamic_params["wing_dynamic_eq_external_key_review_only"] is True
    assert kick_bass_dynamic_params["wing_dynamic_eq_external_key_source_role"] == "kick"
    assert (
        kick_bass_dynamic_params["wing_dynamic_eq_keyed_ducking_live_apply_blocked"]
        is True
    )
    dynamic_wing = results["soundcheck_kick_bass_dynamic_pocket"][
        "candidate_future_wing_readback_contracts"
    ]["kick_bass_dynamic_pocket_ducking"]["wing_osc_applicability"]
    assert dynamic_wing["native_osc_supported"] is False
    assert dynamic_wing["native_osc_support_level"] == (
        "dynamic_eq_value_mapping_required"
    )
    assert dynamic_wing["unsupported_strategy_reason"] == (
        "dynamic_eq_sidechain_or_paired_ducking_requires_verified_wing_sidechain_mapping"
    )
    concrete_key_result = results[
        "soundcheck_kick_bass_concrete_external_key_stays_review_only"
    ]
    assert concrete_key_result["actual_selected_strategy_id"] == (
        "kick_bass_pocket_eq_split"
    )
    concrete_key_duck = concrete_key_result["candidate_parameters"][
        "kick_bass_dynamic_pocket_ducking"
    ]
    assert concrete_key_duck["wing_dynamic_eq_external_key_source_target_id"] == "ch:1"
    assert (
        concrete_key_duck["wing_dynamic_eq_external_key_source_target_surface"]
        == "channel"
    )
    assert (
        concrete_key_duck["wing_dynamic_eq_external_key_source_target_parse_blocker"]
        == ""
    )
    assert (
        concrete_key_duck["wing_dynamic_eq_external_key_required_source_alias_format"]
        == "CH.n, BUS.n, MAIN.n, MTX.n or AUX.n"
    )
    assert (
        concrete_key_duck["wing_dynamic_eq_external_key_candidate_source_alias"]
        == "CH.1"
    )
    assert (
        concrete_key_duck["wing_dynamic_eq_external_key_concrete_source_available"]
        is True
    )
    assert (
        concrete_key_duck[
            "wing_dynamic_eq_external_key_verified_osc_mapping_available"
        ]
        is False
    )
    assert (
        concrete_key_duck[
            "wing_dynamic_eq_external_key_apply_blocked_without_concrete_source"
        ]
        is False
    )
    assert (
        concrete_key_duck[
            "wing_dynamic_eq_external_key_apply_blocked_even_with_concrete_source"
        ]
        is True
    )
    concrete_key_duck_wing = concrete_key_result[
        "candidate_future_wing_readback_contracts"
    ]["kick_bass_dynamic_pocket_ducking"]["wing_osc_applicability"]
    assert concrete_key_duck_wing["native_osc_supported"] is False
    assert concrete_key_duck_wing["native_osc_support_level"] == (
        "dynamic_eq_value_mapping_required"
    )
    assert concrete_key_duck_wing["unsupported_strategy_reason"] == (
        "dynamic_eq_sidechain_or_paired_ducking_requires_verified_wing_sidechain_mapping"
    )
    assert concrete_key_result["proposal_authority"]["live_apply_allowed"] is False
    assert (
        "wing_dynamic_eq_value_mapping_required"
        in concrete_key_result["candidate_blockers"][
            "kick_bass_dynamic_pocket_ducking"
        ]
    )
    assert results["soundcheck_kick_bass_hot_bus_rebalance_before_ducking"][
        "actual_selected_strategy_id"
    ] == "kick_bass_summed_bus_rebalance_before_ducking"
    assert results["soundcheck_kick_bass_hot_bus_rebalance_before_ducking"][
        "operator_explanation_sections"
    ]["selection_policy"] == "soundcheck_kick_bass_summed_bus_rebalance_priority"
    kick_bass_hot_bus = results[
        "soundcheck_kick_bass_hot_bus_rebalance_before_ducking"
    ]
    assert kick_bass_hot_bus["selected_parameters"]["summed_bus_target_id"] == "bus:3"
    assert kick_bass_hot_bus["selected_parameters"]["wing_osc_target_id_override"] == (
        "bus:3"
    )
    _assert_replay_gain_delta_mapping(
        kick_bass_hot_bus,
        source="bus_delta_db",
        expected_delta=kick_bass_hot_bus["selected_parameters"]["bus_delta_db"],
        path_prefix="/bus/{bus}",
    )
    _assert_replay_selected_gain_delta_apply_values(
        kick_bass_hot_bus,
        expected_delta=kick_bass_hot_bus["selected_parameters"]["bus_delta_db"],
    )
    assert results["soundcheck_kick_bass_hot_bus_rebalance_before_ducking"][
        "candidate_blockers"
    ]["kick_bass_dynamic_pocket_ducking"] == [
        "summed_bus_rebalance_should_precede_dynamic_ducking",
        "wing_dynamic_eq_value_mapping_required",
    ]
    live_kick_bass_hot_bus = results[
        "live_kick_bass_hot_bus_pullback_before_eq_or_ducking"
    ]
    live_kick_bass_params = live_kick_bass_hot_bus["selected_parameters"]
    assert live_kick_bass_hot_bus["actual_selected_strategy_id"] == (
        "kick_bass_summed_bus_rebalance_before_ducking"
    )
    assert live_kick_bass_hot_bus["actual_method_family"] == "gain_fader"
    assert live_kick_bass_hot_bus["operator_explanation_sections"][
        "selection_policy"
    ] == "live_kick_bass_summed_bus_pullback_priority"
    assert live_kick_bass_params["bus_delta_db"] == -1.5
    assert live_kick_bass_params["delta_db"] == -1.5
    assert live_kick_bass_params["bounded_delta_db"] == 1.5
    assert live_kick_bass_params["summed_bus_target_id"] == "bus:3"
    assert live_kick_bass_params["wing_osc_original_target_id"] == "ch:2"
    assert live_kick_bass_params["wing_osc_target_id_override"] == "bus:3"
    assert live_kick_bass_params["wing_osc_effective_target_id"] == "bus:3"
    assert live_kick_bass_params["live_kick_bass_bus_pullback_only_proposal"] is True
    assert (
        live_kick_bass_params["live_kick_bass_source_eq_deferred_until_bus_recheck"]
        is True
    )
    assert (
        live_kick_bass_params["live_kick_bass_ducking_deferred_until_bus_recheck"]
        is True
    )
    assert live_kick_bass_params["live_stepped_change_required"] is True
    assert live_kick_bass_params["live_readback_after_each_step_required"] is True
    assert live_kick_bass_params["live_max_single_step_db"] == 0.75
    _assert_replay_gain_delta_mapping(
        live_kick_bass_hot_bus,
        source="bus_delta_db",
        expected_delta=-1.5,
        path_prefix="/bus/{bus}",
    )
    _assert_replay_selected_gain_delta_apply_values(
        live_kick_bass_hot_bus,
        expected_delta=-1.5,
    )
    assert live_kick_bass_hot_bus["selected_future_wing_readback_contract"][
        "wing_osc_applicability"
    ]["target_surface_kind"] == "bus"
    assert live_kick_bass_hot_bus["proposal_selected_wing_osc_summary"][
        "wing_osc_native_supported"
    ] is True
    assert live_kick_bass_hot_bus["proposal_selected_wing_osc_summary"][
        "wing_osc_candidate_apply_status"
    ] == "wing_native_osc_candidate_requires_absolute_value_readback_before_apply"
    assert live_kick_bass_hot_bus["proposal_authority"][
        "live_apply_allowed"
    ] is False
    assert live_kick_bass_hot_bus["proposal_authority"][
        "live_authority_granted"
    ] is False
    assert live_kick_bass_hot_bus["proposal_authority"]["live_write"] is False
    assert (
        "summed_bus_rebalance_should_precede_dynamic_ducking"
        in live_kick_bass_hot_bus["candidate_blockers"][
            "kick_bass_dynamic_pocket_ducking"
        ]
    )
    live_kick_bass_tradeoffs = {
        row["strategy_id"]: row["why_not_selected"]
        for row in live_kick_bass_hot_bus["alternative_tradeoffs"]
    }
    assert live_kick_bass_tradeoffs["kick_bass_pocket_eq_split"] == (
        "low_end_bus_level_priority:"
        "rebalance_summed_bus_before_ducking_or_eq"
    )
    assert results["soundcheck_kick_target_avoids_ducking_kick_from_bass"][
        "actual_selected_strategy_id"
    ] == "kick_bass_pocket_eq_split"
    assert results["soundcheck_kick_target_avoids_ducking_kick_from_bass"][
        "selected_parameters"
    ]["relationship"] == "bass_masks_kick"
    assert (
        "dynamic_ducking_current_kick_can_kill_punch"
        in results["soundcheck_kick_target_avoids_ducking_kick_from_bass"][
            "candidate_blockers"
        ]["kick_bass_dynamic_pocket_ducking"]
    )
    assert (
        results["soundcheck_kick_target_avoids_ducking_kick_from_bass"][
            "candidate_parameters"
        ]["kick_bass_dynamic_pocket_ducking"]["processing_target"]
        == "counterpart_bass_channel"
    )
    assert results["soundcheck_kick_bass_arrangement_overlap_hold"][
        "actual_selected_strategy_id"
    ] == "arrangement_aware_low_end_separation_hold"
    assert results["soundcheck_kick_bass_arrangement_overlap_hold"][
        "actual_method_family"
    ] == "analysis_only"
    assert results["soundcheck_arrangement_overlap_pair_context_hold"][
        "actual_selected_strategy_id"
    ] == "arrangement_overlap_pair_context_hold"
    assert results["soundcheck_arrangement_overlap_pair_context_hold"][
        "selected_parameters"
    ]["missing_masking_conflict_role"] is True
    assert results["live_kick_bass_conservative_eq_split"][
        "actual_selected_strategy_id"
    ] == "kick_bass_pocket_eq_split"
    assert results["soundcheck_bass_low_end_dynamic_resonance"][
        "actual_selected_strategy_id"
    ] == "low_end_dynamic_resonance_control"
    assert results["soundcheck_bass_low_end_dynamic_resonance"][
        "actual_method_family"
    ] == "dynamic_eq"
    assert results["soundcheck_bass_low_end_dynamic_resonance"][
        "selected_parameters"
    ]["low_end_parameter_source"] == "measured_low_end_resonance"
    assert results["soundcheck_bass_low_end_dynamic_resonance"][
        "selected_parameters"
    ]["q"] == 1.37
    assert results["soundcheck_bass_low_end_dynamic_resonance"][
        "selected_parameters"
    ]["wing_dynamic_eq_gain_osc_value"] == -3.9
    assert results["soundcheck_bass_low_end_dynamic_resonance"][
        "selected_parameters"
    ]["wing_dynamic_eq_model_osc_value"] == WING_DYNAMIC_EQ_GATE_MODEL
    assert results["soundcheck_bass_low_end_dynamic_resonance"][
        "operator_explanation_sections"
    ]["selection_policy"] == "weighted_strategy_score"
    bass_dynamic_params = results["soundcheck_bass_low_end_dynamic_resonance"][
        "selected_parameters"
    ]
    assert bass_dynamic_params["release_ms"] == 215.0
    hot_bus = results[
        "soundcheck_bass_low_end_hot_bus_rebalance_before_resonance_deq"
    ]
    hot_bus_params = hot_bus["selected_parameters"]
    assert hot_bus["actual_selected_strategy_id"] == (
        "low_end_summed_bus_rebalance_before_resonance_eq"
    )
    assert hot_bus["actual_method_family"] == "gain_fader"
    assert hot_bus["operator_explanation_sections"]["selection_policy"] == (
        "soundcheck_low_end_resonance_summed_bus_rebalance_priority"
    )
    assert hot_bus_params["target_scope"] == (
        "summed_low_end_bus_before_resonance_eq"
    )
    assert hot_bus_params["bus_delta_db"] == -2.5
    assert hot_bus_params["delta_db"] == -2.5
    assert hot_bus_params["gain_db"] == 0.0
    assert hot_bus_params["summed_bus_target_id"] == "bus:3"
    assert hot_bus_params["wing_osc_original_target_id"] == "ch:2"
    assert hot_bus_params["wing_osc_target_id_override"] == "bus:3"
    assert hot_bus_params["wing_osc_effective_target_id"] == "bus:3"
    assert hot_bus_params["requires_summed_bus_readback"] is True
    _assert_replay_gain_delta_mapping(
        hot_bus,
        source="bus_delta_db",
        expected_delta=-2.5,
        path_prefix="/bus/{bus}",
    )
    assert hot_bus["selected_future_wing_readback_contract"][
        "wing_osc_applicability"
    ]["target_surface_kind"] == "bus"
    assert hot_bus["candidate_parameters"]["low_end_dynamic_resonance_control"][
        "low_end_resonance_bandwidth_hz"
    ] == 72.0
    assert hot_bus["candidate_parameters"]["low_end_static_pocket_cut"][
        "no_boost"
    ] is True
    hot_bus_tradeoffs = {
        row["strategy_id"]: row["why_not_selected"]
        for row in hot_bus["alternative_tradeoffs"]
    }
    assert hot_bus_tradeoffs["low_end_dynamic_resonance_control"] == (
        "low_end_bus_level_priority:"
        "rebalance_summed_bus_before_source_resonance_eq"
    )
    assert results["live_bass_low_end_static_pocket_cut"][
        "actual_selected_strategy_id"
    ] == "low_end_static_pocket_cut"
    assert results["live_bass_low_end_static_pocket_cut"][
        "selected_parameters"
    ]["low_end_parameter_source"] == "measured_low_end_resonance"
    assert results["live_bass_low_end_static_pocket_cut"][
        "selected_parameters"
    ]["q"] == 1.21
    assert "live_dynamic_low_end_resonance_requires_operator_review" in results[
        "live_bass_low_end_static_pocket_cut"
    ]["candidate_blockers"]["low_end_dynamic_resonance_control"]
    live_hot_bus = results[
        "live_bass_low_end_hot_bus_pullback_before_source_eq"
    ]
    live_hot_bus_params = live_hot_bus["selected_parameters"]
    assert live_hot_bus["actual_selected_strategy_id"] == (
        "low_end_summed_bus_rebalance_before_resonance_eq"
    )
    assert live_hot_bus["actual_method_family"] == "gain_fader"
    assert live_hot_bus["operator_explanation_sections"]["selection_policy"] == (
        "live_low_end_resonance_summed_bus_pullback_priority"
    )
    assert live_hot_bus_params["bus_delta_db"] == -1.5
    assert live_hot_bus_params["delta_db"] == -1.5
    assert live_hot_bus_params["bounded_delta_db"] == 1.5
    assert live_hot_bus_params["summed_bus_target_id"] == "bus:3"
    assert live_hot_bus_params["wing_osc_original_target_id"] == "ch:2"
    assert live_hot_bus_params["wing_osc_target_id_override"] == "bus:3"
    assert live_hot_bus_params["wing_osc_effective_target_id"] == "bus:3"
    assert live_hot_bus_params["live_low_end_bus_pullback_only_proposal"] is True
    assert live_hot_bus_params["live_source_eq_deferred_until_bus_recheck"] is True
    assert (
        live_hot_bus_params[
            "live_low_end_rebalance_live_apply_blocked_without_wing_proof"
        ]
        is True
    )
    assert live_hot_bus_params["live_stepped_change_required"] is True
    assert live_hot_bus_params["live_readback_after_each_step_required"] is True
    assert live_hot_bus_params["live_max_single_step_db"] == 0.75
    _assert_replay_gain_delta_mapping(
        live_hot_bus,
        source="bus_delta_db",
        expected_delta=-1.5,
        path_prefix="/bus/{bus}",
    )
    _assert_replay_selected_gain_delta_apply_values(
        live_hot_bus,
        expected_delta=-1.5,
    )
    assert live_hot_bus["selected_future_wing_readback_contract"][
        "wing_osc_applicability"
    ]["target_surface_kind"] == "bus"
    assert live_hot_bus["proposal_selected_wing_osc_summary"][
        "wing_osc_native_supported"
    ] is True
    assert live_hot_bus["proposal_selected_wing_osc_summary"][
        "wing_osc_candidate_apply_status"
    ] == "wing_native_osc_candidate_requires_absolute_value_readback_before_apply"
    assert live_hot_bus["proposal_authority"]["live_apply_allowed"] is False
    assert live_hot_bus["proposal_authority"]["live_authority_granted"] is False
    assert live_hot_bus["proposal_authority"]["live_write"] is False
    assert "live_dynamic_low_end_resonance_requires_operator_review" in live_hot_bus[
        "candidate_blockers"
    ]["low_end_dynamic_resonance_control"]
    live_hot_bus_tradeoffs = {
        row["strategy_id"]: row["why_not_selected"]
        for row in live_hot_bus["alternative_tradeoffs"]
    }
    assert live_hot_bus_tradeoffs["low_end_static_pocket_cut"] == (
        "low_end_bus_level_priority:"
        "rebalance_summed_bus_before_source_resonance_eq"
    )
    assert results["soundcheck_snare_ring_dynamic_notch"][
        "actual_selected_strategy_id"
    ] == "drum_dynamic_ring_notch_control"
    assert results["soundcheck_snare_ring_dynamic_notch"][
        "actual_method_family"
    ] == "dynamic_eq"
    assert results["soundcheck_snare_ring_dynamic_notch"]["selected_parameters"][
        "drum_ring_parameter_source"
    ] == "measured_drum_ring_decay"
    assert results["soundcheck_snare_ring_dynamic_notch"]["selected_parameters"][
        "q"
    ] == 7.5
    assert results["soundcheck_snare_ring_dynamic_notch"]["selected_parameters"][
        "wing_dynamic_eq_gain_osc_value"
    ] == -4.4
    assert results["soundcheck_snare_ring_dynamic_notch"]["selected_parameters"][
        "wing_dynamic_eq_model_osc_value"
    ] == WING_DYNAMIC_EQ_GATE_MODEL
    assert results["soundcheck_snare_ring_dynamic_notch"][
        "operator_explanation_sections"
    ]["selection_policy"] == "weighted_strategy_score"
    snare_dynamic_params = results["soundcheck_snare_ring_dynamic_notch"][
        "selected_parameters"
    ]
    assert snare_dynamic_params["release_ms"] == 219.8
    assert results["live_snare_ring_static_notch"][
        "actual_selected_strategy_id"
    ] == "drum_static_ring_notch_cut"
    assert results["live_snare_ring_static_notch"]["selected_parameters"][
        "drum_ring_parameter_source"
    ] == "measured_drum_ring_decay"
    assert results["live_snare_ring_static_notch"]["selected_parameters"][
        "q"
    ] == 6.48
    assert "live_dynamic_drum_ring_control_requires_operator_review" in results[
        "live_snare_ring_static_notch"
    ]["candidate_blockers"]["drum_dynamic_ring_notch_control"]
    assert results["soundcheck_snare_transient_tamer"][
        "actual_selected_strategy_id"
    ] == "drum_transient_tamer_compression"
    assert results["soundcheck_snare_transient_tamer"]["selected_parameters"][
        "makeup_gain_db"
    ] == 0.0
    assert results["soundcheck_snare_transient_tamer"]["selected_parameters"][
        "wing_compressor_detector_osc_value"
    ] == "PEAK"
    assert results["soundcheck_snare_transient_tamer"]["selected_parameters"][
        "wing_compressor_detector_value_type"
    ] == "string_enum"
    assert results["soundcheck_snare_transient_tamer"]["selected_parameters"][
        "wing_compressor_knee_osc_value"
    ] == 3
    assert results["soundcheck_snare_transient_tamer"]["selected_parameters"][
        "wing_compressor_knee_value_type"
    ] == "integer"
    assert results["soundcheck_snare_transient_tamer"]["selected_parameters"][
        "transient_parameter_source"
    ] == "measured_transient_envelope"
    assert results["live_snare_low_headroom_transient_hold"][
        "actual_selected_strategy_id"
    ] == "source_true_peak_headroom_pullback"
    assert results["live_snare_low_headroom_transient_hold"][
        "operator_explanation_sections"
    ]["selection_policy"] == "true_peak_headroom_safety_priority"
    assert "true_peak_headroom_low" in results[
        "live_snare_low_headroom_transient_hold"
    ]["global_blockers"]
    assert results["live_snare_low_headroom_transient_hold"]["selected_parameters"][
        "headroom_restoration_first"
    ] is True
    assert results["pan_phase_mono_failure"]["actual_selected_strategy_id"] == (
        "mono_compatibility_width_reduction"
    )
    assert results["pan_phase_mono_failure"]["operator_explanation_sections"][
        "selection_policy"
    ] == "soundcheck_phase_mono_safety_priority"
    assert results["pan_phase_mono_failure"]["selected_parameters"][
        "wing_pan_osc_path_suffix"
    ] == "pan"
    assert results["pan_phase_mono_failure"]["selected_parameters"][
        "wing_width_osc_path_suffix"
    ] == "wid"
    assert results["pan_phase_mono_failure"][
        "selected_future_wing_readback_contract"
    ]["pan_width_readback"]["read_after_paths"] == ["pan", "wid"]
    assert results["pan_phase_mono_failure"]["selected_parameters"][
        "soundcheck_pan_width_target_surface"
    ] == "channel"
    assert results["pan_phase_mono_failure"]["selected_parameters"][
        "soundcheck_pan_width_identity_confirmation_scope"
    ] == "source"
    assert results["pan_phase_mono_failure"]["selected_parameters"][
        "soundcheck_pan_width_source_identity_required"
    ] is True
    assert results["pan_phase_mono_failure"]["selected_parameters"][
        "soundcheck_pan_width_output_feed_identity_required"
    ] is False
    _assert_replay_selected_pan_width_apply_values(
        results["pan_phase_mono_failure"],
        {
            "wing_pan_osc_value": 0.0,
            "wing_width_osc_value": 35.0,
        },
    )
    matrix_phase = results["soundcheck_matrix_phase_mono_width_reduction"]
    matrix_params = matrix_phase["selected_parameters"]
    matrix_readback = matrix_phase["selected_future_wing_readback_contract"][
        "pan_width_readback"
    ]
    assert matrix_phase["actual_selected_strategy_id"] == (
        "mono_compatibility_width_reduction"
    )
    assert matrix_params["wing_pan_osc_path_suffix"] == "pan"
    assert matrix_params["wing_width_osc_path_suffix"] == "wid"
    assert matrix_params["wing_pan_osc_value"] == 0.0
    assert matrix_params["wing_width_osc_value"] == 35.0
    assert matrix_params["soundcheck_pan_width_target_surface"] == "matrix"
    assert matrix_params["soundcheck_pan_width_identity_confirmation_scope"] == (
        "output_feed"
    )
    assert matrix_params["soundcheck_pan_width_source_identity_required"] is False
    assert matrix_params["soundcheck_pan_width_output_feed_identity_required"] is True
    assert matrix_params["soundcheck_pan_width_live_promotion_status"] == (
        "blocked_until_real_wing_pan_width_readback_output_feed_identity_"
        "phase_mono_recheck_and_operator_approval"
    )
    assert matrix_readback["wing_osc_path_templates"] == [
        "/mtx/{matrix}/pan",
        "/mtx/{matrix}/wid",
    ]
    assert matrix_readback["read_after_paths"] == ["pan", "wid"]
    assert matrix_readback["soundcheck_pan_width_target_surface"] == "matrix"
    assert matrix_readback["soundcheck_pan_width_output_feed_identity_required"] is True
    _assert_replay_selected_pan_width_apply_values(
        matrix_phase,
        {
            "wing_pan_osc_value": 0.0,
            "wing_width_osc_value": 35.0,
        },
    )
    _assert_replay_selected_parameter_rollback_operations(
        matrix_phase,
        readback_section="pan_width_readback",
        operations_field=(
            "soundcheck_pan_width_selected_parameter_rollback_operations"
        ),
        selected_write_keys=["wing_pan_osc_value", "wing_width_osc_value"],
        contract_marker=(
            "read_previous_pan_width_value_before_write_then_restore_same_path"
        ),
    )
    assert results["soundcheck_phase_safe_pan_delta"][
        "actual_selected_strategy_id"
    ] == "phase_safe_pan_target"
    assert results["soundcheck_phase_safe_pan_delta"]["selected_parameters"][
        "wing_pan_osc_path_suffix"
    ] == "pan"
    assert results["soundcheck_phase_safe_pan_delta"]["selected_parameters"][
        "wing_pan_osc_value"
    ] == 35.0
    pan_params = results["soundcheck_phase_safe_pan_delta"]["selected_parameters"]
    assert pan_params["soundcheck_pan_width_bounded_move_proposal"] is True
    assert pan_params["soundcheck_pan_width_native_osc_only_required"] is True
    assert pan_params["soundcheck_pan_max_delta_percent"] == SOUNDCHECK_PAN_MAX_DELTA_PERCENT
    assert (
        pan_params["soundcheck_width_max_widen_percent"]
        == SOUNDCHECK_WIDTH_MAX_WIDEN_PERCENT
    )
    assert (
        pan_params["soundcheck_width_max_reduction_percent"]
        == SOUNDCHECK_WIDTH_MAX_REDUCTION_PERCENT
    )
    assert (
        pan_params["soundcheck_width_min_target_percent"]
        == SOUNDCHECK_WIDTH_MIN_TARGET_PERCENT
    )
    assert (
        pan_params["wing_pan_width_osc_protocol_version"]
        == WING_PAN_WIDTH_OSC_PROTOCOL_VERSION
    )
    assert (
        pan_params["wing_pan_width_osc_protocol_source"]
        == WING_PAN_WIDTH_OSC_PROTOCOL_SOURCE
    )
    assert pan_params["wing_pan_width_requires_real_wing_fw_3_0_5_readback"] is True
    assert pan_params["soundcheck_pan_width_move_kinds"] == ["pan"]
    assert pan_params["soundcheck_pan_width_apply_only_selected_write_keys"] is True
    assert pan_params["soundcheck_pan_width_selected_write_keys"] == [
        "wing_pan_osc_value"
    ]
    assert pan_params["soundcheck_pan_width_forbidden_write_keys"] == [
        "wing_width_osc_value"
    ]
    assert pan_params["soundcheck_pan_width_phase_mono_recheck_required"] is True
    assert pan_params["soundcheck_pan_width_source_identity_required"] is True
    assert pan_params["soundcheck_pan_width_readback_required"] is True
    assert pan_params["soundcheck_pan_width_rollback_ready"] is True
    assert pan_params["soundcheck_pan_width_live_apply_blocked_without_wing_proof"] is True
    pan_readback = results["soundcheck_phase_safe_pan_delta"][
        "selected_future_wing_readback_contract"
    ]["pan_width_readback"]
    assert (
        pan_readback["wing_pan_width_osc_protocol_version"]
        == WING_PAN_WIDTH_OSC_PROTOCOL_VERSION
    )
    assert (
        pan_readback["wing_pan_width_osc_protocol_source"]
        == WING_PAN_WIDTH_OSC_PROTOCOL_SOURCE
    )
    assert pan_readback["wing_pan_width_requires_real_wing_fw_3_0_5_readback"] is True
    assert pan_readback["soundcheck_pan_width_bounded_move_proposal"] is True
    assert pan_readback["soundcheck_pan_width_move_kinds"] == ["pan"]
    assert pan_readback["soundcheck_pan_width_apply_only_selected_write_keys"] is True
    assert pan_readback["soundcheck_pan_width_selected_write_keys"] == [
        "wing_pan_osc_value"
    ]
    assert pan_readback["soundcheck_pan_width_forbidden_write_keys"] == [
        "wing_width_osc_value"
    ]
    assert (
        pan_readback["soundcheck_pan_width_live_apply_blocked_without_wing_proof"]
        is True
    )
    _assert_replay_selected_pan_width_apply_values(
        results["soundcheck_phase_safe_pan_delta"],
        {"wing_pan_osc_value": 35.0},
    )
    _assert_replay_selected_parameter_rollback_operations(
        results["soundcheck_phase_safe_pan_delta"],
        readback_section="pan_width_readback",
        operations_field=(
            "soundcheck_pan_width_selected_parameter_rollback_operations"
        ),
        selected_write_keys=["wing_pan_osc_value"],
        contract_marker=(
            "read_previous_pan_width_value_before_write_then_restore_same_path"
        ),
    )
    assert results["soundcheck_measured_stereo_balance_pan_target"][
        "actual_selected_strategy_id"
    ] == "phase_safe_pan_target"
    assert results["soundcheck_measured_stereo_balance_pan_target"][
        "selected_parameters"
    ]["pan_target_source"] == "measured_stereo_balance"
    assert results["soundcheck_measured_stereo_balance_pan_target"][
        "selected_parameters"
    ]["target_pan_percent"] == 26.0
    _assert_replay_selected_pan_width_apply_values(
        results["soundcheck_measured_stereo_balance_pan_target"],
        {"wing_pan_osc_value": 26.0},
    )
    assert results["live_measured_stereo_balance_pan_review_hold"][
        "actual_selected_strategy_id"
    ] == "hold_live_pan_change_for_operator_review"
    assert results["live_measured_stereo_balance_pan_review_hold"][
        "operator_explanation_sections"
    ]["selection_policy"] == "live_pan_change_review_priority"
    assert results["live_measured_stereo_balance_pan_review_hold"][
        "selected_parameters"
    ]["pan_target_source"] == "measured_stereo_balance"
    assert results["live_measured_stereo_balance_pan_review_hold"][
        "selected_parameters"
    ]["target_pan_percent"] == -32.0
    live_matrix = results["live_matrix_pan_review_hold_output_feed_identity"]
    assert (
        live_matrix["actual_selected_strategy_id"]
        == "hold_live_pan_change_for_operator_review"
    )
    assert live_matrix["selected_parameters"]["required_context"] == (
        "output_feed_identity_phase_mono_readback_rollback"
    )
    assert live_matrix["selected_parameters"]["live_pan_width_target_surface"] == (
        "matrix"
    )
    assert live_matrix["selected_parameters"][
        "live_pan_width_identity_confirmation_scope"
    ] == "output_feed"
    assert live_matrix["selected_parameters"][
        "live_pan_width_source_identity_required"
    ] is False
    assert live_matrix["selected_parameters"][
        "live_pan_width_output_feed_identity_required"
    ] is True
    assert live_matrix["selected_parameters"]["live_pan_width_selected_write_keys"] == []
    assert set(
        live_matrix["selected_parameters"]["live_pan_width_forbidden_write_keys"]
    ) == {
        "wing_pan_osc_value",
        "wing_width_osc_value",
    }
    assert "wing_pan_osc_value" not in live_matrix["selected_parameters"]
    assert "wing_width_osc_value" not in live_matrix["selected_parameters"]
    live_matrix_readback = live_matrix["selected_future_wing_readback_contract"][
        "pan_width_readback"
    ]
    assert live_matrix_readback["live_pan_width_target_surface"] == "matrix"
    assert (
        live_matrix_readback["live_pan_width_identity_confirmation_scope"]
        == "output_feed"
    )
    assert (
        live_matrix_readback["live_pan_width_output_feed_identity_required"]
        is True
    )
    assert live_matrix_readback["live_pan_width_selected_write_keys"] == []
    assert results["soundcheck_center_anchor_vocal_recenter"][
        "actual_selected_strategy_id"
    ] == "center_anchor_recenter_review"
    assert results["soundcheck_center_anchor_vocal_recenter"][
        "operator_explanation_sections"
    ]["selection_policy"] == "center_anchor_pan_priority"
    assert results["soundcheck_center_anchor_vocal_recenter"]["selected_parameters"][
        "target_pan_percent"
    ] == 0.0
    assert results["soundcheck_center_anchor_vocal_recenter"]["selected_parameters"][
        "measured_stereo_balance_ignored_for_center_anchor"
    ] is True
    _assert_replay_selected_pan_width_apply_values(
        results["soundcheck_center_anchor_vocal_recenter"],
        {"wing_pan_osc_value": 0.0},
    )
    assert results["soundcheck_center_anchor_vocal_recenter"]["candidate_blockers"][
        "phase_safe_pan_target"
    ] == ["center_anchor_source_not_panned"]
    center_anchor_phase_hold = results["soundcheck_center_anchor_phase_mono_hold"]
    center_anchor_phase_params = center_anchor_phase_hold["selected_parameters"]
    assert (
        center_anchor_phase_hold["actual_selected_strategy_id"]
        == "center_anchor_phase_mono_review_hold"
    )
    assert center_anchor_phase_hold["actual_method_family"] == "analysis_only"
    assert center_anchor_phase_hold["operator_explanation_sections"][
        "selection_policy"
    ] == "soundcheck_phase_mono_safety_priority"
    assert center_anchor_phase_params["dsp_change"] == "hold"
    assert center_anchor_phase_params["center_anchor_recenter_deferred"] is True
    assert center_anchor_phase_params["center_anchor_phase_mono_selected_write_keys"] == []
    assert set(
        center_anchor_phase_params["center_anchor_phase_mono_forbidden_write_keys"]
    ) == {
        "wing_pan_osc_value",
        "wing_width_osc_value",
    }
    assert "wing_pan_osc_value" not in center_anchor_phase_params
    assert "wing_width_osc_value" not in center_anchor_phase_params
    center_anchor_phase_readback = center_anchor_phase_hold[
        "selected_future_wing_readback_contract"
    ]["pan_width_readback"]
    assert center_anchor_phase_readback["read_after_paths"] == ["pan", "wid"]
    assert center_anchor_phase_readback["center_anchor_phase_mono_hold_proposal"] is True
    assert (
        center_anchor_phase_readback["center_anchor_phase_mono_selected_write_keys"]
        == []
    )
    _assert_replay_selected_pan_width_apply_values(center_anchor_phase_hold, {})
    center_hold_tradeoff = [
        item
        for item in center_anchor_phase_hold["alternative_tradeoffs"]
        if item["strategy_id"] == "center_hold_for_phase_safety"
    ][0]
    assert center_hold_tradeoff["why_not_selected"] == (
        "center_anchor_phase_mono_hold_priority:"
        "operator_audition_precedes_pan_or_width_write"
    )
    assert results["soundcheck_phase_mono_beats_measured_pan_target"][
        "actual_selected_strategy_id"
    ] == "mono_compatibility_width_reduction"
    assert results["soundcheck_phase_mono_beats_measured_pan_target"][
        "operator_explanation_sections"
    ]["selection_policy"] == "soundcheck_phase_mono_safety_priority"
    assert results["soundcheck_phase_mono_beats_measured_pan_target"][
        "candidate_blockers"
    ]["phase_safe_pan_target"] == [
        "phase_correlation_blocks_width",
        "mono_sum_loss_blocks_width",
    ]
    _assert_replay_selected_pan_width_apply_values(
        results["soundcheck_phase_mono_beats_measured_pan_target"],
        {
            "wing_pan_osc_value": 0.0,
            "wing_width_osc_value": 35.0,
        },
    )
    assert results["live_pan_change_holds_for_operator_review"][
        "actual_selected_strategy_id"
    ] == "hold_live_pan_change_for_operator_review"
    assert results["live_pan_change_holds_for_operator_review"][
        "operator_explanation_sections"
    ]["selection_policy"] == "live_pan_change_review_priority"
    assert results["live_pan_change_holds_for_operator_review"][
        "selected_parameters"
    ]["forbidden_without_wing_proof"] == "live_pan_width_change"
    assert results["soundcheck_phase_mono_risk_width_reduction"][
        "actual_selected_strategy_id"
    ] == "mono_compatibility_width_reduction"
    assert results["soundcheck_phase_mono_risk_width_reduction"][
        "selected_parameters"
    ]["wing_width_osc_value"] == 0.0
    assert results["soundcheck_phase_mono_risk_width_reduction"]["selected_parameters"][
        "severe_cancellation"
    ] is True
    _assert_replay_selected_pan_width_apply_values(
        results["soundcheck_phase_mono_risk_width_reduction"],
        {
            "wing_pan_osc_value": 0.0,
            "wing_width_osc_value": 0.0,
        },
    )
    assert results["live_phase_mono_risk_hold"][
        "actual_selected_strategy_id"
    ] == "hold_phase_mono_processing_for_operator"
    assert results["live_phase_mono_risk_hold"]["selected_parameters"][
        "required_context"
    ] == "source_identity_stereo_role_phase_mono_readback_rollback"
    assert results["live_phase_mono_risk_hold"]["selected_parameters"][
        "forbidden_without_wing_proof"
    ] == "live_phase_mono_pan_width_change"
    live_matrix_phase = results["live_matrix_phase_mono_hold_output_feed_identity"]
    assert (
        live_matrix_phase["actual_selected_strategy_id"]
        == "hold_phase_mono_processing_for_operator"
    )
    assert live_matrix_phase["actual_method_family"] == "analysis_only"
    assert live_matrix_phase["selected_parameters"]["required_context"] == (
        "output_feed_identity_phase_mono_readback_rollback"
    )
    assert live_matrix_phase["selected_parameters"][
        "forbidden_without_wing_proof"
    ] == "live_phase_mono_pan_width_change"
    assert live_matrix_phase["selected_parameters"][
        "live_phase_mono_target_surface"
    ] == "matrix"
    assert live_matrix_phase["selected_parameters"][
        "live_phase_mono_identity_confirmation_scope"
    ] == "output_feed"
    assert live_matrix_phase["selected_parameters"][
        "live_phase_mono_source_identity_required"
    ] is False
    assert live_matrix_phase["selected_parameters"][
        "live_phase_mono_output_feed_identity_required"
    ] is True
    assert live_matrix_phase["selected_parameters"][
        "live_phase_mono_selected_write_keys"
    ] == []
    assert set(
        live_matrix_phase["selected_parameters"][
            "live_phase_mono_forbidden_write_keys"
        ]
    ) == {
        "wing_pan_osc_value",
        "wing_width_osc_value",
    }
    assert "wing_pan_osc_value" not in live_matrix_phase["selected_parameters"]
    assert "wing_width_osc_value" not in live_matrix_phase["selected_parameters"]
    live_matrix_phase_readback = live_matrix_phase[
        "selected_future_wing_readback_contract"
    ]["pan_width_readback"]
    assert live_matrix_phase_readback["live_phase_mono_target_surface"] == "matrix"
    assert (
        live_matrix_phase_readback["live_phase_mono_identity_confirmation_scope"]
        == "output_feed"
    )
    assert (
        live_matrix_phase_readback[
            "live_phase_mono_output_feed_identity_required"
        ]
        is True
    )
    assert live_matrix_phase_readback["live_phase_mono_selected_write_keys"] == []
    assert results["soundcheck_phase_mono_risk_beats_harshness_eq"][
        "actual_selected_strategy_id"
    ] == "hold_phase_mono_processing_for_operator"
    assert results["soundcheck_phase_mono_risk_beats_harshness_eq"][
        "actual_method_family"
    ] == "analysis_only"
    assert results["soundcheck_phase_mono_risk_beats_harshness_eq"][
        "selected_parameters"
    ]["target_resolution_required"] is True
    assert results["soundcheck_phase_mono_risk_beats_harshness_eq"][
        "candidate_blockers"
    ]["mono_compatibility_width_reduction"] == [
        "wing_osc_target_requires_concrete_console_surface"
    ]
    assert results["soundcheck_phase_mono_risk_beats_harshness_eq"][
        "operator_explanation_sections"
    ]["selection_policy"] == "soundcheck_phase_mono_safety_priority"
    assert results["live_phase_mono_risk_beats_harshness_eq"][
        "actual_selected_strategy_id"
    ] == "hold_phase_mono_processing_for_operator"
    assert results["live_phase_mono_risk_beats_harshness_eq"][
        "operator_explanation_sections"
    ]["selection_policy"] == "live_phase_mono_safety_priority"
    assert results["master_true_peak_overload"]["actual_method_family"] == "gain_fader"
    assert results["master_true_peak_overload"][
        "actual_selected_strategy_id"
    ] == "master_trim_for_true_peak_headroom"
    _assert_replay_gain_delta_mapping(
        results["master_true_peak_overload"],
        source="trim_delta_db",
        expected_delta=-1.3,
        path_prefix="/main/{main}",
    )
    _assert_replay_soundcheck_master_output_guard(
        results["master_true_peak_overload"],
        move_kind="pullback",
        followup_review=True,
    )
    _assert_replay_selected_gain_delta_apply_values(
        results["master_true_peak_overload"],
        expected_delta=-1.3,
    )
    live_master_true_peak = results[
        "live_master_true_peak_overload_native_pullback"
    ]
    assert live_master_true_peak["actual_method_family"] == "gain_fader"
    assert live_master_true_peak["actual_selected_strategy_id"] == (
        "master_trim_for_true_peak_headroom"
    )
    assert live_master_true_peak["selected_parameters"]["trim_delta_db"] == -1.3
    assert live_master_true_peak["selected_parameters"][
        "master_true_peak_pullback_db"
    ] == 1.3
    assert live_master_true_peak["selected_parameters"][
        "headroom_deficit_pullback_db"
    ] == 0.0
    assert live_master_true_peak["operator_explanation_sections"][
        "selection_policy"
    ] == "master_true_peak_headroom_priority"
    _assert_replay_gain_delta_mapping(
        live_master_true_peak,
        source="trim_delta_db",
        expected_delta=-1.3,
        path_prefix="/main/{main}",
    )
    _assert_replay_live_master_true_peak_output_guard(live_master_true_peak)
    _assert_replay_selected_gain_delta_apply_values(
        live_master_true_peak,
        expected_delta=-1.3,
    )
    for flag in AUTHORITY_FALSE_FLAGS:
        assert live_master_true_peak["proposal_authority"][flag] is False
    _assert_replay_operator_owned_master_output_gain_hold(
        results[
            "soundcheck_operator_owned_master_output_gain_holds_true_peak_pullback"
        ],
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
    _assert_replay_operator_owned_master_output_gain_hold(
        results["live_operator_owned_master_output_gain_holds_true_peak_pullback"],
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
    assert results["master_low_headroom_trim_before_tone_or_compression"][
        "actual_selected_strategy_id"
    ] == "master_trim_for_true_peak_headroom"
    assert results["master_low_headroom_trim_before_tone_or_compression"][
        "actual_method_family"
    ] == "gain_fader"
    assert results["master_low_headroom_trim_before_tone_or_compression"][
        "selected_parameters"
    ]["trim_delta_db"] == -1.3
    assert results["master_low_headroom_trim_before_tone_or_compression"][
        "operator_explanation_sections"
    ]["selection_policy"] == "master_true_peak_headroom_priority"
    _assert_replay_gain_delta_mapping(
        results["master_low_headroom_trim_before_tone_or_compression"],
        source="trim_delta_db",
        expected_delta=-1.3,
        path_prefix="/main/{main}",
    )
    _assert_replay_soundcheck_master_output_guard(
        results["master_low_headroom_trim_before_tone_or_compression"],
        move_kind="pullback",
        followup_review=True,
    )
    _assert_replay_selected_gain_delta_apply_values(
        results["master_low_headroom_trim_before_tone_or_compression"],
        expected_delta=-1.3,
    )
    assert results["master_reference_feed_holds_processing"][
        "actual_selected_strategy_id"
    ] == "hold_master_reference_feed_no_pa_write"
    assert results["master_reference_feed_holds_processing"]["actual_status"] == (
        "analysis_only"
    )
    reference_channel = results["reference_capture_channel_feed_holds_source_dsp"]
    assert reference_channel["actual_selected_strategy_id"] == (
        "hold_reference_capture_feed_no_console_write"
    )
    assert reference_channel["actual_status"] == "analysis_only"
    assert reference_channel["actual_method_family"] == "analysis_only"
    assert reference_channel["operator_explanation_sections"][
        "selection_policy"
    ] == "reference_capture_feed_guard_priority"
    reference_channel_params = reference_channel["selected_parameters"]
    assert reference_channel_params["requires_concrete_wing_console_target"] is True
    assert reference_channel_params["requires_pa_or_source_output_identity"] is True
    assert reference_channel_params["wing_fw305_osc_target_binding_required"] is True
    assert (
        reference_channel_params["wing_fw305_correction_exists_on_console_required"]
        is True
    )
    assert reference_channel["candidate_blockers"][
        "hold_reference_capture_feed_no_console_write"
    ] == ["reference_or_capture_feed_requires_console_target_confirmation"]
    _assert_replay_analysis_only_has_no_apply_values(reference_channel)
    assert results["master_low_end_broad_eq_cleanup"][
        "actual_selected_strategy_id"
    ] == "master_low_end_broad_eq_cleanup"
    assert results["master_low_end_broad_eq_cleanup"]["actual_method_family"] == "eq"
    assert results["master_low_end_broad_eq_cleanup"]["selected_parameters"][
        "frequency_source"
    ] == "measured_master_low_end_peak"
    assert results["master_low_end_broad_eq_cleanup"]["selected_parameters"][
        "frequency_hz"
    ] == 88.8
    _assert_replay_soundcheck_master_processing_guard(
        results["master_low_end_broad_eq_cleanup"],
        processing_types=("broad_eq",),
    )
    assert results["master_low_end_broad_eq_cleanup"]["selected_rationale"][
        "wing_osc_candidate_apply_status"
    ] == "wing_native_osc_candidate_requires_readback_before_promotion"
    assert results["master_low_end_broad_eq_cleanup"]["proposal_authority"][
        "live_apply_allowed"
    ] is False
    assert results["master_presence_broad_eq_cleanup"][
        "actual_selected_strategy_id"
    ] == "master_presence_broad_eq_cleanup"
    assert results["master_presence_broad_eq_cleanup"]["actual_method_family"] == "eq"
    assert results["master_presence_broad_eq_cleanup"]["selected_parameters"][
        "frequency_source"
    ] == "measured_master_presence_peak"
    assert results["master_presence_broad_eq_cleanup"]["selected_parameters"][
        "frequency_hz"
    ] == 3920.0
    assert results["master_presence_broad_eq_cleanup"]["selected_parameters"][
        "q"
    ] == 1.45
    _assert_replay_soundcheck_master_processing_guard(
        results["master_presence_broad_eq_cleanup"],
        processing_types=("broad_eq",),
    )
    assert results["master_presence_broad_eq_cleanup"]["selected_rationale"][
        "wing_osc_candidate_apply_status"
    ] == "wing_native_osc_candidate_requires_readback_before_promotion"
    assert results["master_static_loudness_trim_before_bus_compression"][
        "actual_selected_strategy_id"
    ] == "master_level_trim_for_static_loudness"
    assert results["master_static_loudness_trim_before_bus_compression"][
        "actual_method_family"
    ] == "gain_fader"
    assert results["master_static_loudness_trim_before_bus_compression"][
        "selected_parameters"
    ]["master_level_parameter_source"] == "master_integrated_lufs_static_loudness"
    assert results["master_static_loudness_trim_before_bus_compression"][
        "selected_parameters"
    ]["trim_delta_db"] == -2.2
    assert results["master_static_loudness_trim_before_bus_compression"][
        "operator_explanation_sections"
    ]["selection_policy"] == "master_static_loudness_trim_priority"
    _assert_replay_gain_delta_mapping(
        results["master_static_loudness_trim_before_bus_compression"],
        source="trim_delta_db",
        expected_delta=-2.2,
        path_prefix="/main/{main}",
    )
    _assert_replay_selected_gain_delta_apply_values(
        results["master_static_loudness_trim_before_bus_compression"],
        expected_delta=-2.2,
    )
    assert results["master_static_loudness_trim_before_low_end_recheck"][
        "actual_selected_strategy_id"
    ] == "master_level_trim_for_static_loudness"
    assert results["master_static_loudness_trim_before_low_end_recheck"][
        "actual_method_family"
    ] == "gain_fader"
    assert results["master_static_loudness_trim_before_low_end_recheck"][
        "selected_parameters"
    ]["trim_delta_db"] == -3.0
    assert results["master_static_loudness_trim_before_low_end_recheck"][
        "selected_parameters"
    ]["static_loudness_tone_recheck_primary"] is True
    assert results["master_static_loudness_trim_before_low_end_recheck"][
        "selected_parameters"
    ]["post_trim_master_tone_recheck_required"] is True
    assert results["master_static_loudness_trim_before_low_end_recheck"][
        "selected_parameters"
    ]["candidate_tone_cleanup_after_level_recheck"] == "master_low_end_broad_eq_cleanup"
    assert results["master_static_loudness_trim_before_low_end_recheck"][
        "candidate_parameters"
    ]["master_low_end_broad_eq_cleanup"]["frequency_source"] == (
        "measured_master_low_end_peak"
    )
    assert results["master_static_loudness_trim_before_low_end_recheck"][
        "operator_explanation_sections"
    ]["selection_policy"] == "master_static_loudness_trim_priority"
    _assert_replay_gain_delta_mapping(
        results["master_static_loudness_trim_before_low_end_recheck"],
        source="trim_delta_db",
        expected_delta=-3.0,
        path_prefix="/main/{main}",
    )
    _assert_replay_soundcheck_master_output_guard(
        results["master_static_loudness_trim_before_low_end_recheck"],
        move_kind="pullback",
        followup_review=True,
    )
    _assert_replay_master_output_processing_followup_review_contract(
        results["master_static_loudness_trim_before_low_end_recheck"],
        expected_values={
            "candidate_tone_cleanup_after_level_recheck": (
                "master_low_end_broad_eq_cleanup"
            ),
            "post_trim_master_tone_recheck_required": True,
        },
    )
    _assert_replay_selected_gain_delta_apply_values(
        results["master_static_loudness_trim_before_low_end_recheck"],
        expected_delta=-3.0,
    )
    assert results["master_loud_dynamic_trim_before_bus_compression"][
        "actual_selected_strategy_id"
    ] == "master_loud_dynamic_trim_before_compression"
    assert results["master_loud_dynamic_trim_before_bus_compression"][
        "actual_method_family"
    ] == "gain_fader"
    assert results["master_loud_dynamic_trim_before_bus_compression"][
        "operator_explanation_sections"
    ]["selection_policy"] == "master_loud_dynamic_trim_priority"
    assert results["master_loud_dynamic_trim_before_bus_compression"][
        "selected_parameters"
    ]["master_level_parameter_source"] == "master_integrated_lufs_loud_dynamic"
    assert results["master_loud_dynamic_trim_before_bus_compression"][
        "selected_parameters"
    ]["trim_delta_db"] == -3.0
    assert results["master_loud_dynamic_trim_before_bus_compression"][
        "selected_parameters"
    ]["dynamic_envelope_pressure_score"] == 0.7
    assert results["master_loud_dynamic_trim_before_bus_compression"][
        "selected_parameters"
    ]["candidate_bus_compression_threshold_dbfs"] == -9.6
    assert results["master_loud_dynamic_trim_before_bus_compression"][
        "selected_parameters"
    ]["candidate_bus_compression_knee_db"] == 4.6
    assert results["master_loud_dynamic_trim_before_bus_compression"][
        "selected_parameters"
    ]["candidate_bus_compression_mix_percent"] == 86.2
    assert results["master_loud_dynamic_trim_before_bus_compression"][
        "selected_parameters"
    ]["candidate_bus_compression_target_gr_db"] == 2.1
    assert results["master_loud_dynamic_trim_before_bus_compression"][
        "selected_parameters"
    ]["candidate_bus_compression_detector"] == "rms"
    assert results["master_loud_dynamic_trim_before_bus_compression"][
        "selected_parameters"
    ]["candidate_bus_compression_envelope_curve"] == "smooth_log"
    assert results["master_loud_dynamic_trim_before_bus_compression"][
        "selected_parameters"
    ]["candidate_bus_compression_auto_timing_enabled"] is False
    assert results["master_loud_dynamic_trim_before_bus_compression"][
        "selected_parameters"
    ]["post_trim_bus_compression_review_required"] is True
    _assert_replay_gain_delta_mapping(
        results["master_loud_dynamic_trim_before_bus_compression"],
        source="trim_delta_db",
        expected_delta=-3.0,
        path_prefix="/main/{main}",
    )
    _assert_replay_soundcheck_master_output_guard(
        results["master_loud_dynamic_trim_before_bus_compression"],
        move_kind="pullback",
        followup_review=True,
    )
    _assert_replay_master_output_processing_followup_review_contract(
        results["master_loud_dynamic_trim_before_bus_compression"],
        expected_values={
            "candidate_bus_compression_threshold_dbfs": -9.6,
            "candidate_bus_compression_ratio": results[
                "master_loud_dynamic_trim_before_bus_compression"
            ]["selected_parameters"]["candidate_bus_compression_ratio"],
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
    _assert_replay_selected_gain_delta_apply_values(
        results["master_loud_dynamic_trim_before_bus_compression"],
        expected_delta=-3.0,
    )
    assert results["master_quiet_loudness_lift_before_hold_or_processing"][
        "actual_selected_strategy_id"
    ] == "master_output_lift_for_quiet_soundcheck"
    assert results["master_quiet_loudness_lift_before_hold_or_processing"][
        "actual_method_family"
    ] == "gain_fader"
    assert results["master_quiet_loudness_lift_before_hold_or_processing"][
        "selected_parameters"
    ]["master_level_parameter_source"] == "master_integrated_lufs_quiet_soundcheck"
    assert results["master_quiet_loudness_lift_before_hold_or_processing"][
        "selected_parameters"
    ]["trim_delta_db"] == 3.0
    _assert_replay_gain_delta_mapping(
        results["master_quiet_loudness_lift_before_hold_or_processing"],
        source="trim_delta_db",
        expected_delta=3.0,
        path_prefix="/main/{main}",
    )
    assert results["master_quiet_loudness_lift_before_hold_or_processing"][
        "selected_parameters"
    ]["soundcheck_master_output_required_true_peak_headroom_db"] == 4.0
    _assert_replay_soundcheck_master_output_guard(
        results["master_quiet_loudness_lift_before_hold_or_processing"],
        move_kind="lift",
        followup_review=False,
    )
    _assert_replay_selected_gain_delta_apply_values(
        results["master_quiet_loudness_lift_before_hold_or_processing"],
        expected_delta=3.0,
    )
    assert results["master_quiet_loudness_lift_before_hold_or_processing"][
        "operator_explanation_sections"
    ]["selection_policy"] == "master_quiet_loudness_lift_priority"
    _assert_replay_operator_owned_master_output_gain_hold(
        results["soundcheck_operator_owned_quiet_master_output_gain_holds_lift"],
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
    assert results["master_quiet_loudness_feedback_hold_before_lift"][
        "actual_selected_strategy_id"
    ] == "hold_master_feedback_or_monitor_risk_before_loudness_lift"
    assert results["master_quiet_loudness_feedback_hold_before_lift"][
        "actual_method_family"
    ] == "analysis_only"
    assert results["master_quiet_loudness_feedback_hold_before_lift"][
        "operator_explanation_sections"
    ]["selection_policy"] == "master_feedback_monitor_loudness_hold_priority"
    assert results["master_quiet_loudness_feedback_hold_before_lift"][
        "selected_parameters"
    ]["candidate_lift_db"] == 3.0
    assert results["master_quiet_loudness_feedback_hold_before_lift"][
        "selected_parameters"
    ]["required_context"] == (
        "pa_and_monitor_feedback_path_review_before_master_loudness_lift"
    )
    assert results["master_quiet_loudness_feedback_hold_before_lift"][
        "candidate_blockers"
    ]["master_output_lift_for_quiet_soundcheck"] == [
        "feedback_path_review_blocks_master_loudness_lift",
        "monitor_feedback_risk_blocks_master_loudness_lift",
        "feedback_risk_blocks_boost",
    ]
    assert results["live_master_presence_holds_before_master_eq"][
        "actual_selected_strategy_id"
    ] == "hold_live_master_output_processing_for_operator"
    assert results["live_master_presence_holds_before_master_eq"][
        "operator_explanation_sections"
    ]["selection_policy"] == "live_master_output_review_priority"
    live_master_presence = results["live_master_presence_holds_before_master_eq"]
    assert live_master_presence["selected_parameters"][
        "live_master_output_hold_proposal"
    ] is True
    assert live_master_presence["selected_parameters"][
        "live_master_output_target_surface"
    ] == "main"
    assert live_master_presence["selected_parameters"][
        "live_master_output_presence_review_required"
    ] is True
    assert live_master_presence["selected_parameters"][
        "live_master_output_dynamics_review_required"
    ] is False
    assert live_master_presence["selected_parameters"][
        "live_master_output_no_processing_without_operator_wing_proof"
    ] is True
    assert "wing_gain_delta_db" not in live_master_presence["selected_parameters"]
    assert "wing_eq_band_operations" not in live_master_presence["selected_parameters"]
    assert "wing_compressor_threshold_osc_value" not in live_master_presence[
        "selected_parameters"
    ]
    live_master_presence_readback = live_master_presence[
        "selected_future_wing_readback_contract"
    ]["live_master_output_hold_readback"]
    assert live_master_presence_readback["read_only"] is True
    assert live_master_presence_readback[
        "allowed_live_write_scope_before_wing_proof"
    ] == []
    assert "/main/{main}/fdr" in live_master_presence_readback[
        "wing_master_gain_osc_path_templates"
    ]
    assert "/main/{main}/eq/{band}f" in live_master_presence_readback[
        "wing_master_eq_osc_path_templates"
    ]
    assert "/main/{main}/dyn/thr36" in live_master_presence_readback[
        "wing_master_dynamics_osc_path_templates"
    ]
    assert live_master_presence_readback["wing_master_limiter_osc_path_templates"] == []
    assert "eq/{band}g" in live_master_presence_readback["read_before_paths"]
    assert "dyn/thr36" in live_master_presence_readback["read_before_paths"]
    _assert_replay_analysis_only_has_no_apply_values(live_master_presence)
    assert results["master_dynamic_range_bus_compression"][
        "actual_selected_strategy_id"
    ] == "gentle_bus_compression"
    assert results["master_dynamic_range_bus_compression"][
        "actual_method_family"
    ] == "dynamics"
    assert results["master_dynamic_range_bus_compression"]["selected_parameters"][
        "makeup_gain_db"
    ] == 0.0
    assert results["master_dynamic_range_bus_compression"]["selected_parameters"][
        "master_dynamics_parameter_source"
    ] == "measured_master_envelope"
    assert results["master_dynamic_range_bus_compression"]["selected_parameters"][
        "threshold_dbfs"
    ] == -14.5
    assert results["master_dynamic_range_bus_compression"]["selected_parameters"][
        "attack_ms"
    ] == 36.0
    assert results["master_dynamic_range_bus_compression"]["selected_parameters"][
        "release_ms"
    ] == 336.0
    assert results["master_dynamic_range_bus_compression"]["selected_parameters"][
        "knee_db"
    ] == 4.0
    assert results["master_dynamic_range_bus_compression"]["selected_parameters"][
        "compression_mix_percent"
    ] == 86.0
    assert results["master_dynamic_range_bus_compression"]["selected_parameters"][
        "compressor_mix_parameter_source"
    ] == "measured_master_envelope_parallel_bus_blend"
    assert results["master_dynamic_range_bus_compression"]["selected_parameters"][
        "compressor_knee_parameter_source"
    ] == "measured_master_envelope_soft_knee_policy"
    assert results["master_dynamic_range_bus_compression"]["selected_parameters"][
        "detector"
    ] == "rms"
    assert results["master_dynamic_range_bus_compression"]["selected_parameters"][
        "compressor_envelope_curve"
    ] == "smooth_log"
    assert results["master_dynamic_range_bus_compression"]["selected_parameters"][
        "compressor_auto_timing_enabled"
    ] is False
    assert results["master_dynamic_range_bus_compression"]["selected_parameters"][
        "dynamics_peak_to_median_db"
    ] == 15.0
    _assert_replay_soundcheck_master_processing_guard(
        results["master_dynamic_range_bus_compression"],
        processing_types=("bus_compression",),
    )
    _assert_replay_selected_dynamics_apply_values(
        results["master_dynamic_range_bus_compression"],
        {
            "wing_compressor_on_osc_value": 1,
            "wing_compressor_threshold_osc_value": -14.5,
            "wing_compressor_attack_osc_value": 36.0,
            "wing_compressor_release_osc_value": 336.0,
            "wing_compressor_makeup_gain_osc_value": 0.0,
            "wing_compressor_mix_osc_value": 86.0,
            "wing_compressor_ratio_osc_value": 1.9,
            "wing_compressor_detector_osc_value": "RMS",
            "wing_compressor_envelope_curve_osc_value": "LOG",
            "wing_compressor_auto_osc_value": 0,
            "wing_compressor_knee_osc_value": 4,
        },
    )
    master_bus_compression_readback = results["master_dynamic_range_bus_compression"][
        "proposal_selected_wing_osc_readback_summary"
    ]
    assert master_bus_compression_readback["wing_osc_readback_status"] == (
        "wing_osc_readback_paths_declared_before_live_promotion"
    )
    assert "dyn/thr36" in master_bus_compression_readback[
        "wing_osc_read_before_paths"
    ]
    assert "dyn/mix" in master_bus_compression_readback["wing_osc_read_before_paths"]
    assert "dyn/det" in master_bus_compression_readback["wing_osc_read_before_paths"]
    assert "dyn/env" in master_bus_compression_readback["wing_osc_read_before_paths"]
    assert "dyn/auto" in master_bus_compression_readback["wing_osc_read_before_paths"]
    assert "/main/{main}/dyn/thr36" in master_bus_compression_readback[
        "wing_osc_readback_path_templates"
    ]
    assert results["master_dynamic_range_bus_compression"][
        "operator_explanation_sections"
    ]["selected_wing_osc_readback_status"] == (
        "wing_osc_readback_paths_declared_before_live_promotion"
    )
    assert (
        "Selected readback status: "
        "wing_osc_readback_paths_declared_before_live_promotion."
    ) in results["master_dynamic_range_bus_compression"]["operator_explanation"]
    assert results["live_master_dynamics_holds_before_bus_compression"][
        "actual_selected_strategy_id"
    ] == "hold_live_master_output_processing_for_operator"
    assert results["live_master_dynamics_holds_before_bus_compression"][
        "operator_explanation_sections"
    ]["selection_policy"] == "live_master_output_review_priority"
    live_master_dynamics = results[
        "live_master_dynamics_holds_before_bus_compression"
    ]
    assert live_master_dynamics["selected_parameters"][
        "live_master_output_hold_proposal"
    ] is True
    assert live_master_dynamics["selected_parameters"][
        "live_master_output_presence_review_required"
    ] is False
    assert live_master_dynamics["selected_parameters"][
        "live_master_output_dynamics_review_required"
    ] is True
    assert live_master_dynamics["selected_parameters"][
        "live_master_output_previous_dynamics_readback_required"
    ] is True
    live_master_dynamics_readback = live_master_dynamics[
        "selected_future_wing_readback_contract"
    ]["live_master_output_hold_readback"]
    assert live_master_dynamics_readback["read_only"] is True
    assert live_master_dynamics_readback[
        "allowed_live_write_scope_before_wing_proof"
    ] == []
    assert live_master_dynamics_readback[
        "wing_master_dynamics_threshold_osc_path_suffix"
    ] == "dyn/thr36"
    assert "/main/{main}/dyn/thr36" in live_master_dynamics_readback[
        "wing_master_dynamics_osc_path_templates"
    ]
    assert "wing_compressor_threshold_osc_value" not in live_master_dynamics[
        "selected_parameters"
    ]
    _assert_replay_analysis_only_has_no_apply_values(live_master_dynamics)
    for scenario_id in (
        "stale_context_blocks_strategy",
        "live_authority_context_blocks_strategy",
        "missing_readback_context_blocks_strategy",
        "missing_rollback_context_blocks_strategy",
    ):
        assert results[scenario_id]["actual_selected_strategy_id"] == (
            "hold_for_context_readback_or_rollback_recheck"
        )
        assert results[scenario_id]["actual_method_family"] == "analysis_only"
        assert results[scenario_id]["operator_explanation_sections"][
            "selection_policy"
        ] == "context_readback_rollback_guard_priority"

    for result in report["results"]:
        for flag in AUTHORITY_FALSE_FLAGS:
            assert result["proposal_authority"][flag] is False
        assert result["read_only_or_dry_run_only"] is True
        assert result["candidate_count"] >= 1
        assert result["selected_total_score"] >= 0.0
        readback = result["selected_future_wing_readback_contract"]
        assert readback["required"] is True
        assert readback["required_before_live_promotion"] is True
        assert readback["requires_real_wing_proof"] is True
        assert readback["requires_operator_approval"] is True
        assert readback["requires_pre_promotion_dry_run"] is True
        assert readback["fader_ceiling_dbfs"] == 0.0
        assert readback["no_routing_scene_snapshot_write"] is True
        assert readback["allowed_live_write_scope_before_wing_proof"] == []
        assert readback["live_write"] is False
        wing_osc = readback["wing_osc_applicability"]
        assert wing_osc["target_console"] == "Behringer WING Rack"
        assert wing_osc["firmware"] == "3.0.5"
        assert wing_osc["protocol"] == "OSC UDP 2223"
        assert wing_osc["protocol_reference"] == (
            "Docs/manuals/wing/WING Remote Protocols v3.0.5.pdf"
        )
        assert wing_osc["requires_real_wing_readback"] is True
        assert wing_osc["requires_operator_approval"] is True
        assert wing_osc["live_authority_granted"] is False
        assert wing_osc["live_write"] is False
        assert wing_osc["allowed_live_write_scope_before_wing_proof"] == []
        selected_blockers = set(
            result["candidate_blockers"].get(
                result["actual_selected_strategy_id"],
                [],
            )
        )
        operator_owned_review_only_selected = bool(
            selected_blockers
            & {
                WING_DYNAMICS_OPERATOR_OWNED_MODEL_BLOCKER,
                WING_EQ_FILTER_OPERATOR_OWNED_BLOCKER,
                WING_EQ_OPERATOR_OWNED_MODEL_BLOCKER,
                WING_FX_LEVEL_OPERATOR_OWNED_BLOCKER,
                WING_GAIN_OPERATOR_OWNED_BLOCKER,
                WING_GATE_SLOT_OPERATOR_OWNED_MODEL_BLOCKER,
                WING_MUTE_OPERATOR_OWNED_BLOCKER,
                WING_PAN_WIDTH_OPERATOR_OWNED_BLOCKER,
            }
        )
        if result["actual_method_family"] not in {"analysis", "analysis_only"}:
            assert wing_osc["native_osc_supported"] is (
                False if operator_owned_review_only_selected else True
            )
            assert isinstance(wing_osc["native_osc_path_templates"], list)
            assert wing_osc["native_osc_path_templates"]
        sections = result["operator_explanation_sections"]
        _assert_replay_tradeoff_wing_osc_summary(result["selected_rationale"])
        _assert_replay_tradeoff_wing_osc_summary(
            sections["selected_wing_osc_summary"]
        )
        require_readback_paths = result["actual_method_family"] not in {
            "analysis",
            "analysis_only",
        }
        expected_wing_native_supported = (
            False
            if operator_owned_review_only_selected
            else True
            if require_readback_paths
            else None
        )
        _assert_replay_selected_wing_osc_readback_summary(
            sections["selected_wing_osc_readback_summary"],
            native_supported=expected_wing_native_supported,
            require_paths=require_readback_paths,
        )
        assert sections["selected_wing_osc_operator_status"] == result[
            "selected_rationale"
        ]["wing_osc_candidate_apply_status"]
        assert sections["selected_wing_osc_readback_status"] == sections[
            "selected_wing_osc_readback_summary"
        ]["wing_osc_readback_status"]
        _assert_replay_tradeoff_wing_osc_summary(
            result["proposal_selected_wing_osc_summary"]
        )
        _assert_replay_selected_wing_osc_readback_summary(
            result["proposal_selected_wing_osc_readback_summary"],
            native_supported=expected_wing_native_supported,
            require_paths=require_readback_paths,
        )
        _assert_replay_tradeoff_wing_osc_summary(
            result["proposed_action_wing_osc_summary"]
        )
        _assert_replay_selected_wing_osc_readback_summary(
            result["proposed_action_wing_osc_readback_summary"],
            native_supported=expected_wing_native_supported,
            require_paths=require_readback_paths,
        )
        _assert_replay_tradeoff_wing_osc_summary(
            result["operator_correction_wing_osc_summary"]
        )
        _assert_replay_selected_wing_osc_readback_summary(
            result["operator_correction_wing_osc_readback_summary"],
            native_supported=expected_wing_native_supported,
            require_paths=require_readback_paths,
        )
        _assert_replay_tradeoff_wing_osc_summary(
            result["operator_correction_proposed_correction_wing_osc_summary"]
        )
        _assert_replay_selected_wing_osc_readback_summary(
            result["operator_correction_proposed_correction_wing_osc_readback_summary"],
            native_supported=expected_wing_native_supported,
            require_paths=require_readback_paths,
        )
        assert sections["problem_found"]
        assert sections["selected_processing_type"] == result["actual_method_family"]
        assert sections["selected_strategy_id"] == result["actual_selected_strategy_id"]
        assert sections["why_selected"]
        assert isinstance(sections["proposed_parameters"], dict)
        assert sections["remaining_risk"]
        assert sections["operator_verification"]
        assert sections["rollback_readback"]["required_before_live_promotion"] is True
        assert sections["rollback_readback"]["live_authority_granted"] is False
        if result["candidate_count"] > 1:
            assert result["alternative_tradeoffs"]
            for row in result["alternative_tradeoffs"]:
                _assert_replay_tradeoff_wing_osc_summary(row)
            assert sections["alternatives_worse"]
            assert all(row["why_not_selected"] for row in sections["alternatives_worse"])
            for row in sections["alternatives_worse"]:
                _assert_replay_tradeoff_wing_osc_summary(row)


def test_selected_console_correction_must_be_wing_native_osc_supported():
    report = _default_smart_dsp_replay_report(
        generated_at_utc="2026-06-25T21:45:00Z",
    )
    muddy_result = {
        row["scenario_id"]: row for row in report["results"]
    }["soundcheck_muddy_vocal"]
    contract = deepcopy(muddy_result["selected_future_wing_readback_contract"])
    contract["wing_osc_applicability"]["native_osc_supported"] = False
    contract["wing_osc_applicability"][
        "unsupported_strategy_reason"
    ] = "tampered_dynamic_eq_requires_non_native_mapping"
    selected = SimpleNamespace(
        method_family="eq",
        parameters=muddy_result["selected_parameters"],
        future_wing_readback_contract=contract,
    )

    failures = _readback_contract_failures(selected, "tampered_non_native_selected")

    failure_codes = {failure["code"] for failure in failures}
    assert (
        "tampered_non_native_selected_readback_contract_wing_osc_selected_not_native"
        in failure_codes
    )


def test_selected_console_correction_rejects_review_only_direct_apply_key():
    report = _default_smart_dsp_replay_report(
        generated_at_utc="2026-06-29T09:10:00Z",
    )
    dynamic_vocal = {
        row["scenario_id"]: row for row in report["results"]
    }["soundcheck_dynamic_vocal_compression"]
    contract = deepcopy(dynamic_vocal["selected_future_wing_readback_contract"])
    applicability = contract["wing_osc_applicability"]
    applicability["actionable_parameter_keys"] = list(
        applicability["actionable_parameter_keys"]
    ) + ["target_gain_reduction_db"]
    applicability["direct_apply_parameter_keys"] = list(
        applicability["direct_apply_parameter_keys"]
    ) + ["target_gain_reduction_db"]
    selected = SimpleNamespace(
        method_family=dynamic_vocal["actual_method_family"],
        parameters=dynamic_vocal["selected_parameters"],
        future_wing_readback_contract=contract,
    )

    failures = _readback_contract_failures(
        selected,
        "tampered_review_only_direct_apply",
    )

    failure_codes = {failure["code"] for failure in failures}
    assert (
        "tampered_review_only_direct_apply_"
        "readback_contract_review_only_keys_marked_direct_apply"
        in failure_codes
    )


def test_selected_de_esser_requires_ds902_gate_slot_readback_paths():
    report = _default_smart_dsp_replay_report(
        generated_at_utc="2026-06-26T20:15:00Z",
    )
    live_sibilant = {
        row["scenario_id"]: row for row in report["results"]
    }["live_sibilant_vocal"]
    contract = deepcopy(live_sibilant["selected_future_wing_readback_contract"])
    contract["de_esser_readback"]["read_before_paths"].remove("gate/f")
    selected = SimpleNamespace(
        method_family="de_esser",
        parameters=live_sibilant["selected_parameters"],
        future_wing_readback_contract=contract,
    )

    failures = _readback_contract_failures(selected, "tampered_de_esser_readback")

    failure_codes = {failure["code"] for failure in failures}
    assert (
        "tampered_de_esser_readback_readback_contract_de_esser_path_set_mismatch"
        in failure_codes
    )

    precondition_contract = deepcopy(
        live_sibilant["selected_future_wing_readback_contract"]
    )
    precondition_contract["de_esser_readback"][
        "wing_gate_slot_model_precondition_no_hidden_model_change"
    ] = False
    precondition_selected = SimpleNamespace(
        method_family="de_esser",
        parameters=live_sibilant["selected_parameters"],
        future_wing_readback_contract=precondition_contract,
    )

    failures = _readback_contract_failures(
        precondition_selected,
        "tampered_de_esser_gate_slot_precondition",
    )

    failure_codes = {failure["code"] for failure in failures}
    assert (
        "tampered_de_esser_gate_slot_precondition_"
        "readback_contract_de_esser_gate_slot_model_precondition_"
        "wing_gate_slot_model_precondition_no_hidden_model_change_mismatch"
        in failure_codes
    )


def test_selected_de_esser_requires_exact_wing_value_readback_contract():
    report = _default_smart_dsp_replay_report(
        generated_at_utc="2026-06-26T15:40:00Z",
    )
    live_sibilant = {
        row["scenario_id"]: row for row in report["results"]
    }["live_sibilant_vocal"]
    contract = deepcopy(live_sibilant["selected_future_wing_readback_contract"])
    contract["de_esser_readback"]["wing_de_esser_range_osc_value"] = 1.0
    selected = SimpleNamespace(
        method_family="de_esser",
        parameters=live_sibilant["selected_parameters"],
        future_wing_readback_contract=contract,
    )

    failures = _readback_contract_failures(selected, "tampered_de_esser_values")

    failure_codes = {failure["code"] for failure in failures}
    assert (
        "tampered_de_esser_values_readback_contract_de_esser_value_mismatch"
        in failure_codes
    )


def test_selected_de_esser_requires_selected_write_key_contract():
    report = _default_smart_dsp_replay_report(
        generated_at_utc="2026-06-28T02:30:00Z",
    )
    live_sibilant = {
        row["scenario_id"]: row for row in report["results"]
    }["live_sibilant_vocal"]
    contract = deepcopy(live_sibilant["selected_future_wing_readback_contract"])
    contract["de_esser_readback"]["wing_de_esser_selected_write_keys"] = [
        "wing_de_esser_frequency_osc_value"
    ]
    selected = SimpleNamespace(
        method_family="de_esser",
        parameters=live_sibilant["selected_parameters"],
        future_wing_readback_contract=contract,
    )

    failures = _readback_contract_failures(selected, "tampered_de_esser_write_keys")

    failure_codes = {failure["code"] for failure in failures}
    assert (
        "tampered_de_esser_write_keys_selected_de_esser_"
        "wing_de_esser_selected_write_keys_mismatch"
        in failure_codes
    )


def test_selected_de_esser_requires_selected_parameter_rollback_operations():
    report = _default_smart_dsp_replay_report(
        generated_at_utc="2026-06-30T09:30:00Z",
    )
    de_esser = {
        row["scenario_id"]: row for row in report["results"]
    }["soundcheck_harsh_sibilant_vocal_split_chain"]
    contract = deepcopy(de_esser["selected_future_wing_readback_contract"])
    contract["de_esser_readback"].pop(
        "wing_de_esser_selected_parameter_rollback_operations"
    )
    selected = SimpleNamespace(
        method_family="de_esser",
        parameters=de_esser["selected_parameters"],
        future_wing_readback_contract=contract,
    )

    failures = _readback_contract_failures(
        selected,
        "tampered_de_esser_rollback_ops",
    )

    failure_codes = {failure["code"] for failure in failures}
    assert (
        "tampered_de_esser_rollback_ops_readback_contract_"
        "wing_de_esser_selected_parameter_rollback_operations_missing"
    ) in failure_codes


def test_selected_live_de_esser_requires_conservative_guard_contract():
    report = _default_smart_dsp_replay_report(
        generated_at_utc="2026-06-26T22:55:00Z",
    )
    live_sibilant = {
        row["scenario_id"]: row for row in report["results"]
    }["live_sibilant_vocal"]
    selected = SimpleNamespace(
        method_family="de_esser",
        parameters=live_sibilant["selected_parameters"],
        future_wing_readback_contract=live_sibilant[
            "selected_future_wing_readback_contract"
        ],
    )

    assert _readback_contract_failures(
        selected,
        "valid_live_de_esser_guard",
    ) == []

    tampered_contract = deepcopy(
        live_sibilant["selected_future_wing_readback_contract"]
    )
    tampered_contract["de_esser_readback"][
        "live_de_esser_live_apply_blocked_without_wing_proof"
    ] = False
    tampered_selected = SimpleNamespace(
        method_family="de_esser",
        parameters=live_sibilant["selected_parameters"],
        future_wing_readback_contract=tampered_contract,
    )

    failures = _readback_contract_failures(
        tampered_selected,
        "tampered_live_de_esser_guard",
    )

    failure_codes = {failure["code"] for failure in failures}
    assert (
        "tampered_live_de_esser_guard_readback_contract_live_de_esser_guard_live_de_esser_live_apply_blocked_without_wing_proof_mismatch"
        in failure_codes
    )

    tampered_parameters = dict(live_sibilant["selected_parameters"])
    tampered_parameters["wing_de_esser_model_osc_value"] = WING_DE_ESSER_GATE_MODEL
    tampered_selected = SimpleNamespace(
        method_family="de_esser",
        parameters=tampered_parameters,
        future_wing_readback_contract=live_sibilant[
            "selected_future_wing_readback_contract"
        ],
    )

    failures = _readback_contract_failures(
        tampered_selected,
        "tampered_live_de_esser_model_write",
    )

    failure_codes = {failure["code"] for failure in failures}
    assert (
        "tampered_live_de_esser_model_write_readback_contract_live_de_esser_guard_model_write_violation"
        in failure_codes
    )

    tampered_parameters = dict(live_sibilant["selected_parameters"])
    tampered_parameters["wing_de_esser_on_osc_value"] = 1
    tampered_parameters["wing_de_esser_mode_osc_value"] = "HF"
    tampered_selected = SimpleNamespace(
        method_family="de_esser",
        parameters=tampered_parameters,
        future_wing_readback_contract=live_sibilant[
            "selected_future_wing_readback_contract"
        ],
    )

    failures = _readback_contract_failures(
        tampered_selected,
        "tampered_live_de_esser_on_mode_write",
    )

    failure_codes = {failure["code"] for failure in failures}
    assert (
        "tampered_live_de_esser_on_mode_write_readback_contract_live_de_esser_guard_on_mode_write_violation"
        in failure_codes
    )


def test_selected_live_eq_requires_conservative_cut_guard_contract():
    report = _default_smart_dsp_replay_report(
        generated_at_utc="2026-06-26T23:20:00Z",
    )
    live_eq = {
        row["scenario_id"]: row for row in report["results"]
    }["live_dynamic_muddy_vocal_static_low_mid_priority"]
    selected = SimpleNamespace(
        method_family="eq",
        parameters=live_eq["selected_parameters"],
        future_wing_readback_contract=live_eq[
            "selected_future_wing_readback_contract"
        ],
    )

    assert _readback_contract_failures(selected, "valid_live_eq_guard") == []

    tampered_contract = deepcopy(live_eq["selected_future_wing_readback_contract"])
    tampered_contract["eq_band_readback"][
        "live_eq_live_apply_blocked_without_wing_proof"
    ] = False
    tampered_selected = SimpleNamespace(
        method_family="eq",
        parameters=live_eq["selected_parameters"],
        future_wing_readback_contract=tampered_contract,
    )

    failures = _readback_contract_failures(
        tampered_selected,
        "tampered_live_eq_guard",
    )

    failure_codes = {failure["code"] for failure in failures}
    assert (
        "tampered_live_eq_guard_readback_contract_live_eq_guard_live_eq_live_apply_blocked_without_wing_proof_mismatch"
        in failure_codes
    )

    tampered_parameters = dict(live_eq["selected_parameters"])
    tampered_parameters["wing_eq_on_osc_value"] = 1
    tampered_parameters["wing_eq_mix_osc_value"] = WING_EQ_MIX_BASELINE_PERCENT
    tampered_selected = SimpleNamespace(
        method_family="eq",
        parameters=tampered_parameters,
        future_wing_readback_contract=live_eq[
            "selected_future_wing_readback_contract"
        ],
    )

    failures = _readback_contract_failures(
        tampered_selected,
        "tampered_live_eq_global_on_mix_write",
    )

    failure_codes = {failure["code"] for failure in failures}
    assert (
        "tampered_live_eq_global_on_mix_write_readback_contract_live_eq_guard_global_on_mix_write_violation"
        in failure_codes
    )

    tampered_parameters = dict(live_eq["selected_parameters"])
    tampered_parameters["wing_eq_band_operations"] = tampered_parameters[
        "live_eq_band_operation_review_only_values"
    ]
    tampered_selected = SimpleNamespace(
        method_family="eq",
        parameters=tampered_parameters,
        future_wing_readback_contract=live_eq[
            "selected_future_wing_readback_contract"
        ],
    )

    failures = _readback_contract_failures(
        tampered_selected,
        "tampered_live_eq_band_operation_write",
    )

    failure_codes = {failure["code"] for failure in failures}
    assert (
        "tampered_live_eq_band_operation_write_readback_contract_live_eq_guard_band_operation_write_violation"
        in failure_codes
    )


def test_selected_soundcheck_eq_boost_requires_bounded_guard_contract():
    report = _default_smart_dsp_replay_report(
        generated_at_utc="2026-06-26T23:55:00Z",
    )
    soundcheck_eq = {
        row["scenario_id"]: row for row in report["results"]
    }["soundcheck_dull_vocal_presence_lift"]
    selected = SimpleNamespace(
        method_family="eq",
        parameters=soundcheck_eq["selected_parameters"],
        future_wing_readback_contract=soundcheck_eq[
            "selected_future_wing_readback_contract"
        ],
    )

    assert _readback_contract_failures(
        selected,
        "valid_soundcheck_eq_boost_guard",
    ) == []

    tampered_contract = deepcopy(
        soundcheck_eq["selected_future_wing_readback_contract"]
    )
    tampered_contract["eq_band_readback"][
        "soundcheck_eq_live_apply_blocked_without_wing_proof"
    ] = False
    tampered_selected = SimpleNamespace(
        method_family="eq",
        parameters=soundcheck_eq["selected_parameters"],
        future_wing_readback_contract=tampered_contract,
    )

    failures = _readback_contract_failures(
        tampered_selected,
        "tampered_soundcheck_eq_boost_guard",
    )

    failure_codes = {failure["code"] for failure in failures}
    assert (
        "tampered_soundcheck_eq_boost_guard_readback_contract_soundcheck_eq_boost_guard_soundcheck_eq_live_apply_blocked_without_wing_proof_mismatch"
        in failure_codes
    )


def test_selected_dynamic_eq_requires_exact_wing_value_readback_contract():
    report = _default_smart_dsp_replay_report(
        generated_at_utc="2026-06-26T15:55:00Z",
    )
    plosive = {
        row["scenario_id"]: row for row in report["results"]
    }["soundcheck_plosive_vocal_dynamic_hpf"]
    contract = deepcopy(plosive["selected_future_wing_readback_contract"])
    contract["dynamic_eq_readback"]["wing_dynamic_eq_gain_osc_value"] = -1.0
    selected = SimpleNamespace(
        method_family="dynamic_eq",
        parameters=plosive["selected_parameters"],
        future_wing_readback_contract=contract,
    )

    failures = _readback_contract_failures(selected, "tampered_dynamic_eq_values")

    failure_codes = {failure["code"] for failure in failures}
    assert (
        "tampered_dynamic_eq_values_readback_contract_dynamic_eq_value_mismatch"
        in failure_codes
    )

    precondition_contract = deepcopy(plosive["selected_future_wing_readback_contract"])
    precondition_contract["dynamic_eq_readback"][
        "wing_gate_slot_model_precondition_required_model"
    ] = WING_DE_ESSER_GATE_MODEL
    precondition_selected = SimpleNamespace(
        method_family="dynamic_eq",
        parameters=plosive["selected_parameters"],
        future_wing_readback_contract=precondition_contract,
    )

    failures = _readback_contract_failures(
        precondition_selected,
        "tampered_dynamic_eq_gate_slot_precondition",
    )

    failure_codes = {failure["code"] for failure in failures}
    assert (
        "tampered_dynamic_eq_gate_slot_precondition_"
        "readback_contract_dynamic_eq_gate_slot_model_precondition_"
        "wing_gate_slot_model_precondition_required_model_mismatch"
        in failure_codes
    )


def test_selected_dynamic_eq_requires_selected_write_key_contract():
    report = _default_smart_dsp_replay_report(
        generated_at_utc="2026-06-28T02:35:00Z",
    )
    plosive = {
        row["scenario_id"]: row for row in report["results"]
    }["soundcheck_plosive_vocal_dynamic_hpf"]
    contract = deepcopy(plosive["selected_future_wing_readback_contract"])
    contract["dynamic_eq_readback"]["wing_dynamic_eq_selected_write_keys"] = []
    selected = SimpleNamespace(
        method_family="dynamic_eq",
        parameters=plosive["selected_parameters"],
        future_wing_readback_contract=contract,
    )

    failures = _readback_contract_failures(selected, "tampered_dynamic_eq_write_keys")

    failure_codes = {failure["code"] for failure in failures}
    assert (
        "tampered_dynamic_eq_write_keys_selected_dynamic_eq_"
        "wing_dynamic_eq_selected_write_keys_mismatch"
        in failure_codes
    )


def test_selected_dynamic_eq_requires_selected_parameter_rollback_operations():
    report = _default_smart_dsp_replay_report(
        generated_at_utc="2026-06-30T09:35:00Z",
    )
    dynamic_eq = {
        row["scenario_id"]: row for row in report["results"]
    }["soundcheck_plosive_vocal_dynamic_hpf"]
    contract = deepcopy(dynamic_eq["selected_future_wing_readback_contract"])
    operations = contract["dynamic_eq_readback"][
        "wing_dynamic_eq_selected_parameter_rollback_operations"
    ]
    operations[2]["rollback_path_suffix"] = "gate/not-a-fw305-deq-path"
    selected = SimpleNamespace(
        method_family="dynamic_eq",
        parameters=dynamic_eq["selected_parameters"],
        future_wing_readback_contract=contract,
    )

    failures = _readback_contract_failures(
        selected,
        "tampered_dynamic_eq_rollback_ops",
    )

    failure_codes = {failure["code"] for failure in failures}
    assert (
        "tampered_dynamic_eq_rollback_ops_readback_contract_"
        "wing_dynamic_eq_selected_parameter_rollback_operations_"
        "rollback_path_suffix_mismatch"
    ) in failure_codes


def test_selected_pan_width_requires_exact_wing_value_readback_contract():
    report = _default_smart_dsp_replay_report(
        generated_at_utc="2026-06-26T14:50:00Z",
    )
    pan_phase = {
        row["scenario_id"]: row for row in report["results"]
    }["pan_phase_mono_failure"]
    contract = deepcopy(pan_phase["selected_future_wing_readback_contract"])
    contract["pan_width_readback"]["wing_width_osc_value"] = 20.0
    selected = SimpleNamespace(
        method_family="pan",
        parameters=pan_phase["selected_parameters"],
        future_wing_readback_contract=contract,
    )

    failures = _readback_contract_failures(selected, "tampered_pan_width_readback")

    failure_codes = {failure["code"] for failure in failures}
    assert (
        "tampered_pan_width_readback_readback_contract_pan_width_value_mismatch"
        in failure_codes
    )


def test_selected_soundcheck_pan_width_requires_bounded_guard_contract():
    report = _default_smart_dsp_replay_report(
        generated_at_utc="2026-06-27T00:15:00Z",
    )
    result = {
        row["scenario_id"]: row for row in report["results"]
    }["soundcheck_phase_safe_pan_delta"]
    contract = deepcopy(result["selected_future_wing_readback_contract"])
    contract["pan_width_readback"][
        "soundcheck_pan_width_live_apply_blocked_without_wing_proof"
    ] = False
    selected = SimpleNamespace(
        method_family=result["actual_method_family"],
        parameters=deepcopy(result["selected_parameters"]),
        future_wing_readback_contract=contract,
    )

    failures = _readback_contract_failures(
        selected,
        "tampered_soundcheck_pan_width_guard",
    )

    failure_codes = {failure["code"] for failure in failures}
    assert (
        "tampered_soundcheck_pan_width_guard_readback_contract_soundcheck_pan_width_guard_soundcheck_pan_width_live_apply_blocked_without_wing_proof_mismatch"
        in failure_codes
    )


def test_selected_soundcheck_pan_width_blocks_unselected_width_write_contract():
    report = _default_smart_dsp_replay_report(
        generated_at_utc="2026-06-27T06:20:00Z",
    )
    result = {
        row["scenario_id"]: row for row in report["results"]
    }["soundcheck_phase_safe_pan_delta"]
    contract = deepcopy(result["selected_future_wing_readback_contract"])
    contract["pan_width_readback"]["soundcheck_pan_width_forbidden_write_keys"] = []
    selected = SimpleNamespace(
        method_family=result["actual_method_family"],
        parameters=deepcopy(result["selected_parameters"]),
        future_wing_readback_contract=contract,
    )

    failures = _readback_contract_failures(
        selected,
        "tampered_soundcheck_pan_width_forbidden_contract",
    )

    failure_codes = {failure["code"] for failure in failures}
    assert (
        "tampered_soundcheck_pan_width_forbidden_contract_readback_contract_soundcheck_pan_width_guard_soundcheck_pan_width_forbidden_write_keys_mismatch"
        in failure_codes
    )


def test_selected_soundcheck_pan_width_rejects_unselected_width_write_value():
    report = _default_smart_dsp_replay_report(
        generated_at_utc="2026-06-27T06:25:00Z",
    )
    result = {
        row["scenario_id"]: row for row in report["results"]
    }["soundcheck_phase_safe_pan_delta"]
    parameters = deepcopy(result["selected_parameters"])
    parameters["wing_width_osc_value"] = 45.0
    selected = SimpleNamespace(
        method_family=result["actual_method_family"],
        parameters=parameters,
        future_wing_readback_contract=deepcopy(
            result["selected_future_wing_readback_contract"]
        ),
    )

    failures = _readback_contract_failures(
        selected,
        "tampered_soundcheck_pan_width_unselected_width",
    )

    failure_codes = {failure["code"] for failure in failures}
    assert (
        "tampered_soundcheck_pan_width_unselected_width_readback_contract_soundcheck_pan_width_guard_forbidden_write_key_present"
        in failure_codes
    )


def test_selected_soundcheck_master_output_requires_bounded_guard_contract():
    report = _default_smart_dsp_replay_report(
        generated_at_utc="2026-06-27T02:15:00Z",
    )
    result = {
        row["scenario_id"]: row for row in report["results"]
    }["master_quiet_loudness_lift_before_hold_or_processing"]
    contract = deepcopy(result["selected_future_wing_readback_contract"])
    contract["gain_fader_readback"][
        "soundcheck_master_output_live_apply_blocked_without_wing_proof"
    ] = False
    selected = SimpleNamespace(
        method_family=result["actual_method_family"],
        parameters=deepcopy(result["selected_parameters"]),
        future_wing_readback_contract=contract,
    )

    failures = _readback_contract_failures(
        selected,
        "tampered_soundcheck_master_output_guard",
    )

    failure_codes = {failure["code"] for failure in failures}
    assert (
        "tampered_soundcheck_master_output_guard_readback_contract_soundcheck_master_output_guard_soundcheck_master_output_live_apply_blocked_without_wing_proof_mismatch"
        in failure_codes
    )

    followup_result = {
        row["scenario_id"]: row for row in report["results"]
    }["master_loud_dynamic_trim_before_bus_compression"]
    followup_contract = deepcopy(
        followup_result["selected_future_wing_readback_contract"]
    )
    followup_contract["gain_fader_readback"][
        "wing_master_output_processing_followup_apply_ready"
    ] = True
    followup_contract["gain_fader_readback"]["read_before_paths"] = [
        *followup_contract["gain_fader_readback"]["read_before_paths"],
        "/main/{main}/dyn/*",
    ]
    followup_selected = SimpleNamespace(
        method_family=followup_result["actual_method_family"],
        parameters=deepcopy(followup_result["selected_parameters"]),
        future_wing_readback_contract=followup_contract,
    )

    followup_failures = _readback_contract_failures(
        followup_selected,
        "tampered_soundcheck_master_output_followup",
    )

    followup_failure_codes = {failure["code"] for failure in followup_failures}
    assert (
        "tampered_soundcheck_master_output_followup_readback_contract_master_output_processing_followup_wing_master_output_processing_followup_apply_ready_mismatch"
        in followup_failure_codes
    )
    assert (
        "tampered_soundcheck_master_output_followup_readback_contract_master_output_processing_followup_forbidden_read_before_paths_present"
        in followup_failure_codes
    )


def test_selected_soundcheck_master_eq_requires_processing_guard_contract():
    report = _default_smart_dsp_replay_report(
        generated_at_utc="2026-06-27T02:55:00Z",
    )
    result = {
        row["scenario_id"]: row for row in report["results"]
    }["master_presence_broad_eq_cleanup"]
    contract = deepcopy(result["selected_future_wing_readback_contract"])
    contract["eq_band_readback"][
        "soundcheck_master_processing_live_apply_blocked_without_wing_proof"
    ] = False
    selected = SimpleNamespace(
        method_family=result["actual_method_family"],
        parameters=deepcopy(result["selected_parameters"]),
        future_wing_readback_contract=contract,
    )

    failures = _readback_contract_failures(
        selected,
        "tampered_soundcheck_master_eq_processing_guard",
    )

    failure_codes = {failure["code"] for failure in failures}
    assert (
        "tampered_soundcheck_master_eq_processing_guard_readback_contract_soundcheck_master_processing_guard_soundcheck_master_processing_live_apply_blocked_without_wing_proof_mismatch"
        in failure_codes
    )


def test_selected_soundcheck_master_dynamics_requires_processing_guard_contract():
    report = _default_smart_dsp_replay_report(
        generated_at_utc="2026-06-27T02:56:00Z",
    )
    result = {
        row["scenario_id"]: row for row in report["results"]
    }["master_dynamic_range_bus_compression"]
    contract = deepcopy(result["selected_future_wing_readback_contract"])
    contract["dynamics_readback"][
        "soundcheck_master_processing_live_apply_blocked_without_wing_proof"
    ] = False
    selected = SimpleNamespace(
        method_family=result["actual_method_family"],
        parameters=deepcopy(result["selected_parameters"]),
        future_wing_readback_contract=contract,
    )

    failures = _readback_contract_failures(
        selected,
        "tampered_soundcheck_master_dynamics_processing_guard",
    )

    failure_codes = {failure["code"] for failure in failures}
    assert (
        "tampered_soundcheck_master_dynamics_processing_guard_readback_contract_soundcheck_master_processing_guard_soundcheck_master_processing_live_apply_blocked_without_wing_proof_mismatch"
        in failure_codes
    )


def test_selected_gain_delta_requires_exact_abs_delta_readback_contract():
    report = _default_smart_dsp_replay_report(
        generated_at_utc="2026-06-26T15:05:00Z",
    )
    low_headroom = {
        row["scenario_id"]: row for row in report["results"]
    }["kick_bass_masking_low_headroom"]
    contract = deepcopy(low_headroom["selected_future_wing_readback_contract"])
    contract["gain_fader_readback"]["wing_gain_delta_abs_db"] = 0.2
    selected = SimpleNamespace(
        method_family="gain_fader",
        parameters=low_headroom["selected_parameters"],
        future_wing_readback_contract=contract,
    )

    failures = _readback_contract_failures(selected, "tampered_gain_delta_readback")

    failure_codes = {failure["code"] for failure in failures}
    assert (
        "tampered_gain_delta_readback_readback_contract_gain_delta_abs_value_mismatch"
        in failure_codes
    )


def test_selected_gain_delta_requires_readback_gated_write_path_contract():
    report = _default_smart_dsp_replay_report(
        generated_at_utc="2026-06-28T06:05:00Z",
    )
    low_headroom = {
        row["scenario_id"]: row for row in report["results"]
    }["kick_bass_masking_low_headroom"]
    contract = deepcopy(low_headroom["selected_future_wing_readback_contract"])
    contract["gain_fader_readback"][
        "wing_gain_pre_readback_write_path_suffixes_forbidden"
    ] = ["in/set/trim"]
    selected = SimpleNamespace(
        method_family="gain_fader",
        parameters=deepcopy(low_headroom["selected_parameters"]),
        future_wing_readback_contract=contract,
    )

    failures = _readback_contract_failures(
        selected,
        "tampered_gain_delta_write_path_contract",
    )

    failure_codes = {failure["code"] for failure in failures}
    assert (
        "tampered_gain_delta_write_path_contract_selected_gain_pre_readback_forbidden_suffixes_mismatch"
        in failure_codes
    )

    selected_path_contract = deepcopy(
        low_headroom["selected_future_wing_readback_contract"]
    )
    selected_path_contract["gain_fader_readback"][
        "wing_gain_selected_write_path_suffix"
    ] = "fdr"
    selected_path = SimpleNamespace(
        method_family="gain_fader",
        parameters=deepcopy(low_headroom["selected_parameters"]),
        future_wing_readback_contract=selected_path_contract,
    )

    selected_path_failures = _readback_contract_failures(
        selected_path,
        "tampered_gain_delta_preselected_path",
    )

    selected_path_failure_codes = {
        failure["code"] for failure in selected_path_failures
    }
    assert (
        "tampered_gain_delta_preselected_path_selected_gain_write_path_selected_before_readback"
        in selected_path_failure_codes
    )


def test_selected_gain_delta_requires_actionable_wing_value():
    report = _default_smart_dsp_replay_report(
        generated_at_utc="2026-06-26T20:30:00Z",
    )
    low_headroom = {
        row["scenario_id"]: row for row in report["results"]
    }["kick_bass_masking_low_headroom"]
    contract = deepcopy(low_headroom["selected_future_wing_readback_contract"])
    contract["wing_osc_applicability"]["actionable_parameter_keys"].remove(
        "wing_gain_delta_db"
    )
    selected = SimpleNamespace(
        method_family="gain_fader",
        parameters=deepcopy(low_headroom["selected_parameters"]),
        future_wing_readback_contract=contract,
    )

    failures = _readback_contract_failures(
        selected,
        "tampered_gain_delta_actionable_value",
    )

    failure_codes = {failure["code"] for failure in failures}
    assert (
        "tampered_gain_delta_actionable_value_selected_gain_delta_actionable_key_missing"
        in failure_codes
    )


def test_selected_gain_delta_cannot_be_direct_apply_without_absolute_readback():
    report = _default_smart_dsp_replay_report(
        generated_at_utc="2026-06-26T21:25:00Z",
    )
    low_headroom = {
        row["scenario_id"]: row for row in report["results"]
    }["kick_bass_masking_low_headroom"]
    contract = deepcopy(low_headroom["selected_future_wing_readback_contract"])
    applicability = contract["wing_osc_applicability"]
    applicability["direct_apply_parameter_keys"] = ["wing_gain_delta_db"]
    applicability["direct_apply_parameter_count"] = 1
    applicability["has_direct_apply_parameter_value"] = True
    selected = SimpleNamespace(
        method_family="gain_fader",
        parameters=deepcopy(low_headroom["selected_parameters"]),
        future_wing_readback_contract=contract,
    )

    failures = _readback_contract_failures(
        selected,
        "tampered_gain_delta_direct_apply",
    )

    failure_codes = {failure["code"] for failure in failures}
    assert (
        "tampered_gain_delta_direct_apply_readback_contract_relative_keys_marked_direct_apply"
        in failure_codes
    )


def test_selected_mute_requires_exact_wing_value_readback_contract():
    report = _default_smart_dsp_replay_report(
        generated_at_utc="2026-06-26T15:20:00Z",
    )
    noisy_tom = {
        row["scenario_id"]: row for row in report["results"]
    }["extreme_noisy_tom_manual_mute_review"]
    contract = deepcopy(noisy_tom["selected_future_wing_readback_contract"])
    contract["mute_readback"]["wing_mute_osc_value"] = WING_MUTE_OFF_VALUE
    selected = SimpleNamespace(
        method_family="mute",
        parameters=noisy_tom["selected_parameters"],
        future_wing_readback_contract=contract,
    )

    failures = _readback_contract_failures(selected, "tampered_mute_readback")

    failure_codes = {failure["code"] for failure in failures}
    assert (
        "tampered_mute_readback_readback_contract_mute_value_mismatch"
        in failure_codes
    )


def test_selected_mute_requires_selected_write_key_contract():
    report = _default_smart_dsp_replay_report(
        generated_at_utc="2026-06-28T02:55:00Z",
    )
    noisy_tom = {
        row["scenario_id"]: row for row in report["results"]
    }["extreme_noisy_tom_manual_mute_review"]
    contract = deepcopy(noisy_tom["selected_future_wing_readback_contract"])
    contract["mute_readback"]["wing_mute_selected_write_keys"] = []
    selected = SimpleNamespace(
        method_family="mute",
        parameters=noisy_tom["selected_parameters"],
        future_wing_readback_contract=contract,
    )

    failures = _readback_contract_failures(selected, "tampered_mute_write_keys")

    failure_codes = {failure["code"] for failure in failures}
    assert (
        "tampered_mute_write_keys_selected_mute_"
        "wing_mute_selected_write_keys_mismatch"
        in failure_codes
    )


def test_selected_fx_requires_level_delta_rollback_operations():
    report = _default_smart_dsp_replay_report(
        generated_at_utc="2026-06-30T09:05:00Z",
    )
    result = {
        row["scenario_id"]: row for row in report["results"]
    }["soundcheck_source_fx_return_headroom_bound_slot_uses_send_and_return"]
    contract = deepcopy(result["selected_future_wing_readback_contract"])
    contract["fx_readback"].pop(
        "wing_fx_level_selected_parameter_rollback_operations"
    )
    selected = SimpleNamespace(
        method_family=result["actual_method_family"],
        parameters=deepcopy(result["selected_parameters"]),
        future_wing_readback_contract=contract,
    )

    failures = _readback_contract_failures(
        selected,
        "tampered_fx_level_rollback_ops",
    )

    failure_codes = {failure["code"] for failure in failures}
    assert (
        "tampered_fx_level_rollback_ops_readback_contract_"
        "wing_fx_level_selected_parameter_rollback_operations_missing"
    ) in failure_codes


def test_selected_fx_level_rollback_rejects_direct_osc_value():
    report = _default_smart_dsp_replay_report(
        generated_at_utc="2026-06-30T09:10:00Z",
    )
    result = {
        row["scenario_id"]: row for row in report["results"]
    }["soundcheck_source_fx_return_headroom_bound_slot_uses_send_and_return"]
    contract = deepcopy(result["selected_future_wing_readback_contract"])
    operation = contract["fx_readback"][
        "wing_fx_level_selected_parameter_rollback_operations"
    ][0]
    operation["osc_value"] = result["selected_parameters"][
        operation["selected_write_key"]
    ]
    selected = SimpleNamespace(
        method_family=result["actual_method_family"],
        parameters=deepcopy(result["selected_parameters"]),
        future_wing_readback_contract=contract,
    )

    failures = _readback_contract_failures(
        selected,
        "tampered_fx_level_rollback_direct_osc",
    )

    failure_codes = {failure["code"] for failure in failures}
    assert (
        "tampered_fx_level_rollback_direct_osc_readback_contract_"
        "wing_fx_level_selected_parameter_rollback_direct_osc_value_present"
    ) in failure_codes


def test_selected_pan_width_requires_selected_parameter_rollback_operations():
    report = _default_smart_dsp_replay_report(
        generated_at_utc="2026-06-30T09:15:00Z",
    )
    result = {
        row["scenario_id"]: row for row in report["results"]
    }["soundcheck_phase_safe_pan_delta"]
    contract = deepcopy(result["selected_future_wing_readback_contract"])
    operations = contract["pan_width_readback"][
        "soundcheck_pan_width_selected_parameter_rollback_operations"
    ]
    operations[0]["rollback_path_suffix"] = "pan/not-a-fw305-path"
    selected = SimpleNamespace(
        method_family=result["actual_method_family"],
        parameters=deepcopy(result["selected_parameters"]),
        future_wing_readback_contract=contract,
    )

    failures = _readback_contract_failures(
        selected,
        "tampered_pan_width_rollback_ops",
    )

    failure_codes = {failure["code"] for failure in failures}
    assert (
        "tampered_pan_width_rollback_ops_readback_contract_"
        "soundcheck_pan_width_selected_parameter_rollback_operations_"
        "rollback_path_suffix_mismatch"
    ) in failure_codes


def test_selected_mute_requires_selected_parameter_rollback_operations():
    report = _default_smart_dsp_replay_report(
        generated_at_utc="2026-06-30T09:20:00Z",
    )
    result = {
        row["scenario_id"]: row for row in report["results"]
    }["extreme_noisy_tom_manual_mute_review"]
    contract = deepcopy(result["selected_future_wing_readback_contract"])
    contract["mute_readback"].pop(
        "wing_mute_selected_parameter_rollback_operations"
    )
    selected = SimpleNamespace(
        method_family=result["actual_method_family"],
        parameters=deepcopy(result["selected_parameters"]),
        future_wing_readback_contract=contract,
    )

    failures = _readback_contract_failures(
        selected,
        "tampered_mute_rollback_ops",
    )

    failure_codes = {failure["code"] for failure in failures}
    assert (
        "tampered_mute_rollback_ops_readback_contract_"
        "wing_mute_selected_parameter_rollback_operations_missing"
    ) in failure_codes


def test_selected_mute_requires_actionable_wing_value():
    report = _default_smart_dsp_replay_report(
        generated_at_utc="2026-06-26T20:35:00Z",
    )
    noisy_tom = {
        row["scenario_id"]: row for row in report["results"]
    }["extreme_noisy_tom_manual_mute_review"]
    contract = deepcopy(noisy_tom["selected_future_wing_readback_contract"])
    contract["wing_osc_applicability"]["actionable_parameter_keys"].remove(
        "wing_mute_osc_value"
    )
    selected = SimpleNamespace(
        method_family="mute",
        parameters=deepcopy(noisy_tom["selected_parameters"]),
        future_wing_readback_contract=contract,
    )

    failures = _readback_contract_failures(
        selected,
        "tampered_mute_actionable_value",
    )

    failure_codes = {failure["code"] for failure in failures}
    assert (
        "tampered_mute_actionable_value_selected_mute_actionable_key_missing"
        in failure_codes
    )


def test_selected_eq_requires_wing_value_mapping_contract():
    report = _default_smart_dsp_replay_report(
        generated_at_utc="2026-06-25T21:45:00Z",
    )
    muddy_result = {
        row["scenario_id"]: row for row in report["results"]
    }["soundcheck_muddy_vocal"]
    contract = deepcopy(muddy_result["selected_future_wing_readback_contract"])
    contract["eq_band_readback"].pop("wing_eq_value_mapping")
    selected = SimpleNamespace(
        method_family="eq",
        parameters=muddy_result["selected_parameters"],
        future_wing_readback_contract=contract,
    )

    failures = _readback_contract_failures(selected, "tampered_eq_value_mapping")

    failure_codes = {failure["code"] for failure in failures}
    assert (
        "tampered_eq_value_mapping_readback_contract_eq_value_mapping_missing"
        in failure_codes
    )


def test_selected_eq_requires_selected_write_key_contract():
    report = _default_smart_dsp_replay_report(
        generated_at_utc="2026-06-28T07:05:00Z",
    )
    muddy_result = {
        row["scenario_id"]: row for row in report["results"]
    }["soundcheck_muddy_vocal"]
    contract = deepcopy(muddy_result["selected_future_wing_readback_contract"])
    contract["eq_band_readback"]["wing_eq_forbidden_write_keys"] = []
    selected = SimpleNamespace(
        method_family="eq",
        parameters=deepcopy(muddy_result["selected_parameters"]),
        future_wing_readback_contract=contract,
    )

    failures = _readback_contract_failures(selected, "tampered_eq_write_keys")

    failure_codes = {failure["code"] for failure in failures}
    assert (
        "tampered_eq_write_keys_selected_eq_wing_eq_forbidden_write_keys_mismatch"
        in failure_codes
    )

    stray_params = deepcopy(muddy_result["selected_parameters"])
    stray_params["wing_eq_high_cut_frequency_hz"] = 12000.0
    stray_selected = SimpleNamespace(
        method_family="eq",
        parameters=stray_params,
        future_wing_readback_contract=deepcopy(
            muddy_result["selected_future_wing_readback_contract"]
        ),
    )

    stray_failures = _readback_contract_failures(
        stray_selected,
        "tampered_eq_stray_forbidden_key",
    )

    stray_failure_codes = {failure["code"] for failure in stray_failures}
    assert (
        "tampered_eq_stray_forbidden_key_selected_eq_forbidden_write_key_present"
        in stray_failure_codes
    )


def test_selected_eq_requires_selected_parameter_rollback_operations():
    report = _default_smart_dsp_replay_report(
        generated_at_utc="2026-06-30T10:20:00Z",
    )
    muddy_result = {
        row["scenario_id"]: row for row in report["results"]
    }["soundcheck_muddy_vocal"]
    contract = deepcopy(muddy_result["selected_future_wing_readback_contract"])
    contract["eq_band_readback"].pop(
        "wing_eq_selected_parameter_rollback_operations"
    )
    selected = SimpleNamespace(
        method_family="eq",
        parameters=muddy_result["selected_parameters"],
        future_wing_readback_contract=contract,
    )

    failures = _readback_contract_failures(
        selected,
        "tampered_eq_rollback_ops",
    )

    failure_codes = {failure["code"] for failure in failures}
    assert (
        "tampered_eq_rollback_ops_readback_contract_"
        "wing_eq_selected_parameter_rollback_operations_missing"
    ) in failure_codes

    tampered_contract = deepcopy(
        muddy_result["selected_future_wing_readback_contract"]
    )
    band_operation = tampered_contract["eq_band_readback"][
        "wing_eq_selected_parameter_rollback_operations"
    ][2]["band_parameter_operations"][1]
    band_operation["rollback_path_suffix"] = "eq/not-a-fw305-band-path"
    tampered_selected = SimpleNamespace(
        method_family="eq",
        parameters=muddy_result["selected_parameters"],
        future_wing_readback_contract=tampered_contract,
    )

    failures = _readback_contract_failures(
        tampered_selected,
        "tampered_eq_band_rollback_ops",
    )

    failure_codes = {failure["code"] for failure in failures}
    assert (
        "tampered_eq_band_rollback_ops_readback_contract_"
        "wing_eq_band_parameter_rollback_operations_mismatch"
    ) in failure_codes


def test_selected_eq_requires_exact_band_operation_readback_contract():
    report = _default_smart_dsp_replay_report(
        generated_at_utc="2026-06-26T16:20:00Z",
    )
    muddy_result = {
        row["scenario_id"]: row for row in report["results"]
    }["soundcheck_muddy_vocal"]
    contract = deepcopy(muddy_result["selected_future_wing_readback_contract"])
    contract["eq_band_readback"]["wing_eq_band_operations"][0]["gain_db"] = -0.5
    selected = SimpleNamespace(
        method_family="eq",
        parameters=muddy_result["selected_parameters"],
        future_wing_readback_contract=contract,
    )

    failures = _readback_contract_failures(selected, "tampered_eq_band_values")

    failure_codes = {failure["code"] for failure in failures}
    assert (
        "tampered_eq_band_values_readback_contract_eq_band_operations_value_mismatch"
        in failure_codes
    )


def test_selected_eq_requires_wing_mix_baseline_contract():
    report = _default_smart_dsp_replay_report(
        generated_at_utc="2026-06-26T19:15:00Z",
    )
    muddy_result = {
        row["scenario_id"]: row for row in report["results"]
    }["soundcheck_muddy_vocal"]
    params = deepcopy(muddy_result["selected_parameters"])
    for key in (
        "wing_eq_mix_source",
        "wing_eq_mix_osc_path_suffix",
        "wing_eq_mix_osc_value",
        "wing_eq_mix_value_type",
        "wing_eq_mix_min_percent",
        "wing_eq_mix_max_percent",
        "wing_eq_mix_baseline_policy",
    ):
        params.pop(key, None)
    contract = deepcopy(muddy_result["selected_future_wing_readback_contract"])
    for key in (
        "wing_eq_mix_source",
        "wing_eq_mix_osc_path_suffix",
        "wing_eq_mix_osc_value",
        "wing_eq_mix_value_type",
        "wing_eq_mix_min_percent",
        "wing_eq_mix_max_percent",
        "wing_eq_mix_baseline_policy",
    ):
        contract["eq_band_readback"].pop(key, None)
        contract["eq_band_readback"]["wing_eq_value_mapping"].pop(key, None)
    contract["eq_band_readback"]["read_before_paths"] = [
        path
        for path in contract["eq_band_readback"]["read_before_paths"]
        if path != "eq/mix"
    ]
    contract["eq_band_readback"]["read_after_paths"] = [
        path
        for path in contract["eq_band_readback"]["read_after_paths"]
        if path != "eq/mix"
    ]
    contract["wing_osc_applicability"]["native_osc_path_templates"] = [
        path
        for path in contract["wing_osc_applicability"]["native_osc_path_templates"]
        if not path.endswith("/eq/mix")
    ]
    selected = SimpleNamespace(
        method_family="eq",
        parameters=params,
        future_wing_readback_contract=contract,
    )

    failures = _readback_contract_failures(selected, "tampered_eq_mix")

    failure_codes = {failure["code"] for failure in failures}
    assert "tampered_eq_mix_selected_eq_mix_mapping_missing" in failure_codes
    assert "tampered_eq_mix_readback_contract_eq_mix_path_mismatch" in failure_codes
    assert "tampered_eq_mix_readback_contract_eq_mix_read_before_missing" in (
        failure_codes
    )
    assert "tampered_eq_mix_readback_contract_eq_mix_native_path_missing" in (
        failure_codes
    )


def test_selected_eq_replay_contracts_do_not_include_compressor_readback():
    report = _default_smart_dsp_replay_report(
        generated_at_utc="2026-06-26T03:35:00Z",
    )

    for row in report["results"]:
        if row["actual_method_family"] != "eq":
            continue
        selected_parameters = row["selected_parameters"]
        readback = row["selected_future_wing_readback_contract"]

        assert "eq_band_readback" in readback
        assert "dynamics_readback" not in readback
        assert not any(key.startswith("wing_compressor_") for key in selected_parameters)


def test_selected_eq_rejects_strip_low_cut_channel_filter_readback_path():
    params = _with_wing_native_parameter_mappings(
        {
            "filter_type": "high_pass",
            "frequency_hz": 95.0,
            "q": 0.8,
            "slope_db_per_octave": 12,
        },
        method_family="eq",
        target_id="bus:1",
    )
    contract = _future_readback_contract("eq", "bus:1", params)
    contract["eq_band_readback"]["read_before_paths"] = [
        "flt/lc" if path == "eq/leq" else path
        for path in contract["eq_band_readback"]["read_before_paths"]
    ]
    selected = SimpleNamespace(
        method_family="eq",
        parameters=params,
        future_wing_readback_contract=contract,
    )

    failures = _readback_contract_failures(
        selected,
        "tampered_strip_low_cut",
    )

    failure_codes = {failure["code"] for failure in failures}
    assert (
        "tampered_strip_low_cut_readback_contract_eq_strip_low_cut_path_missing"
        in failure_codes
    )
    assert (
        "tampered_strip_low_cut_readback_contract_eq_strip_low_cut_channel_filter_path_present"
        in failure_codes
    )


def test_selected_eq_requires_wing_channel_high_cut_readback_path():
    params = _with_wing_native_parameter_mappings(
        {
            "filter_type": "high_cut",
            "frequency_hz": 12000.0,
            "slope_db_per_octave": 6,
        },
        method_family="eq",
        target_id="ch:5",
    )
    contract = _future_readback_contract("eq", "ch:5", params)
    contract["eq_band_readback"]["read_before_paths"] = [
        "eq/heq" if path == "flt/hcf" else path
        for path in contract["eq_band_readback"]["read_before_paths"]
    ]
    selected = SimpleNamespace(
        method_family="eq",
        parameters=params,
        future_wing_readback_contract=contract,
    )

    failures = _readback_contract_failures(
        selected,
        "tampered_high_cut",
    )

    failure_codes = {failure["code"] for failure in failures}
    assert "tampered_high_cut_readback_contract_eq_high_cut_path_missing" in (
        failure_codes
    )


def test_selected_fx_send_rejects_non_channel_bus_source_surface():
    params = _with_wing_native_parameter_mappings(
        {"send_delta_db": -2.0},
        method_family="fx",
        target_id="main",
    )
    contract = _future_readback_contract("fx", "main", params)
    selected = SimpleNamespace(
        method_family="fx",
        parameters=params,
        future_wing_readback_contract=contract,
    )

    failures = _readback_contract_failures(
        selected,
        "tampered_fx_send_surface",
    )

    failure_codes = {failure["code"] for failure in failures}
    assert params["wing_fx_send_source_surface_blocker"] == (
        WING_FX_SEND_SOURCE_SURFACE_BLOCKER
    )
    assert contract["wing_osc_applicability"]["native_osc_supported"] is False
    assert (
        "tampered_fx_send_surface_readback_contract_wing_osc_selected_not_native"
        in failure_codes
    )
    assert (
        "tampered_fx_send_surface_selected_fx_send_source_surface_not_supported"
        in failure_codes
    )
    assert (
        "tampered_fx_send_surface_selected_fx_send_source_binding_required"
        in failure_codes
    )


def test_selected_fx_return_rejects_non_fx_slot_surface():
    params = _with_wing_native_parameter_mappings(
        {"send_delta_db": 0.0, "return_delta_db": -1.0},
        method_family="fx",
        target_id="main",
    )
    contract = _future_readback_contract("fx", "main", params)
    selected = SimpleNamespace(
        method_family="fx",
        parameters=params,
        future_wing_readback_contract=contract,
    )

    failures = _readback_contract_failures(
        selected,
        "tampered_fx_return_surface",
    )

    failure_codes = {failure["code"] for failure in failures}
    assert params["wing_fx_return_fx_slot_blocker"] == (
        WING_FX_RETURN_SLOT_BINDING_BLOCKER
    )
    assert params["wing_fx_return_fx_slot_apply_ready"] is False
    assert contract["wing_osc_applicability"]["native_osc_supported"] is False
    assert (
        contract["wing_osc_applicability"]["native_osc_support_level"]
        == "fx_return_slot_binding_required"
    )
    assert (
        "tampered_fx_return_surface_readback_contract_wing_osc_selected_not_native"
        in failure_codes
    )
    assert (
        "tampered_fx_return_surface_selected_fx_return_slot_binding_required"
        in failure_codes
    )
    assert (
        "tampered_fx_return_surface_selected_fx_return_slot_not_apply_ready"
        in failure_codes
    )


def test_selected_fx_rejects_no_actionable_wing_osc_parameter():
    params = _with_wing_native_parameter_mappings(
        {"send_delta_db": 0.0, "return_delta_db": 0.0},
        method_family="fx",
        target_id="fx:1",
    )
    contract = _future_readback_contract("fx", "fx:1", params)
    selected = SimpleNamespace(
        method_family="fx",
        parameters=params,
        future_wing_readback_contract=contract,
    )

    failures = _readback_contract_failures(
        selected,
        "tampered_fx_noop",
    )

    failure_codes = {failure["code"] for failure in failures}
    assert contract["wing_osc_applicability"]["native_osc_supported"] is False
    assert (
        contract["wing_osc_applicability"]["native_osc_support_level"]
        == "fx_no_actionable_wing_osc_parameter"
    )
    assert (
        contract["wing_osc_applicability"]["fx_actionable_parameter_blocker"]
        == WING_FX_ACTIONABLE_PARAMETER_BLOCKER
    )
    assert (
        "tampered_fx_noop_readback_contract_wing_osc_selected_not_native"
        in failure_codes
    )
    assert (
        "tampered_fx_noop_selected_fx_no_actionable_wing_osc_parameter"
        in failure_codes
    )


def test_confirmed_hall_fx_decay_filter_uses_current_model_osc_operations():
    report = _default_smart_dsp_replay_report(
        generated_at_utc="2026-06-29T07:30:00Z",
    )
    result = {
        row["scenario_id"]: row for row in report["results"]
    }["soundcheck_fx_wash_confirmed_hall_decay_filter"]

    assert result["passed"] is True
    assert result["actual_selected_strategy_id"] == "shorten_fx_decay_filter"
    assert result["actual_method_family"] == "fx"

    params = result["selected_parameters"]
    assert params["wing_fx_model_parameter_current_model"] == "HALL"
    assert params["wing_fx_model_parameter_current_model_readback_available"] is True
    assert params["wing_fx_model_parameter_current_model_target_id"] == "fx:2"
    assert params["wing_fx_model_parameter_current_model_target_bound"] is True
    assert params["wing_fx_model_parameter_current_model_full_coverage"] is True
    assert params["wing_fx_model_parameter_path_pattern_apply_ready"] is True
    assert params["wing_fx_model_parameter_apply_ready_after_model_readback"] is True
    assert params["wing_fx_model_parameter_live_apply_blocked"] is True
    assert params["wing_fx_model_parameter_blocker"] is None
    assert params["wing_fx_model_parameter_unmapped_keys"] == []
    assert params["soundcheck_fx_native_level_osc_only_required"] is False
    assert params["soundcheck_fx_no_model_or_time_parameter_write_required"] is False
    assert params["soundcheck_fx_model_time_edit_review_only_fields"] == []
    assert params["soundcheck_fx_followup_model_time_review_required"] is False

    operations = params["wing_fx_model_parameter_operations"]
    operation_by_key = {operation["parameter_key"]: operation for operation in operations}
    assert set(operation_by_key) == {
        "decay_s",
        "pre_delay_ms",
        "return_hpf_hz",
        "return_lpf_hz",
    }
    assert {
        operation["osc_path_suffix"] for operation in operations
    } == {"dcy", "pdel", "lc", "hc"}
    assert all(operation["model"] == "HALL" for operation in operations)
    assert all(
        operation["osc_path_template"].startswith("/fx/{fx}/")
        for operation in operations
    )
    for operation in operations:
        assert operation["read_before_required"] is True
        assert operation["read_after_required"] is True
        assert operation["previous_value_readback_required"] is True
        assert (
            operation["previous_value_path_template"]
            == operation["osc_path_template"]
        )
        assert operation["rollback_required"] is True
        assert operation["rollback_value_source"] == "previous_value_readback"
        assert operation["rollback_path_template"] == operation["osc_path_template"]
        assert operation["rollback_readback_required"] is True
        assert operation["fw305_effect_parameter_rollback_contract"] == (
            "read_previous_value_before_write_then_restore_same_path"
        )

    selected = result["selected_rationale"]
    assert selected["wing_osc_native_supported"] is True
    assert selected["wing_osc_support_level"] == (
        "native_fx_model_parameter_osc_path_templates_declared"
    )
    assert selected["wing_osc_unsupported_parameter_keys"] == []
    assert selected["wing_osc_apply_parameter_values"] == {
        "wing_fx_model_parameter_operations": operations
    }
    assert selected["wing_osc_live_write"] is False
    assert selected["wing_osc_live_authority_granted"] is False

    readback = result["selected_future_wing_readback_contract"]["fx_readback"]
    assert set(readback["read_before_paths"]) == {"mdl", "dcy", "pdel", "lc", "hc"}
    assert set(readback["read_after_paths"]) == {"mdl", "dcy", "pdel", "lc", "hc"}
    assert readback["wing_fx_model_parameter_blocker"] is None
    assert readback["soundcheck_fx_native_level_osc_only_required"] is False
    assert readback["soundcheck_fx_no_model_or_time_parameter_write_required"] is False
    assert readback["soundcheck_fx_model_time_edit_review_only_fields"] == []
    assert readback["soundcheck_fx_followup_model_time_review_required"] is False
    assert readback.get("wing_fx_model_time_followup_review_required") is None
    assert not any(key.startswith("wing_fx_model_time_followup_") for key in readback)
    assert readback["wing_fx_model_parameter_operations"] == operations

    tampered_contract = deepcopy(result["selected_future_wing_readback_contract"])
    tampered_contract["fx_readback"][
        "wing_fx_model_time_followup_review_required"
    ] = True
    tampered_selected = SimpleNamespace(
        method_family="fx",
        parameters=params,
        future_wing_readback_contract=tampered_contract,
    )
    failures = _readback_contract_failures(
        tampered_selected,
        "tampered_confirmed_hall_fx",
    )
    failure_codes = {failure["code"] for failure in failures}
    assert (
        "tampered_confirmed_hall_fx_readback_contract_fx_model_time_followup_present_for_current_model_apply_ready"
        in failure_codes
    )


def test_operator_owned_hall_fx_decay_filter_stays_review_only_in_replay():
    report = _default_smart_dsp_replay_report(
        generated_at_utc="2026-06-29T10:45:00Z",
    )
    result = {
        row["scenario_id"]: row for row in report["results"]
    }["soundcheck_fx_wash_operator_owned_hall_prefers_level_pullback"]

    assert result["passed"] is True
    assert result["actual_selected_strategy_id"] == "reduce_fx_send_or_return"
    assert result["actual_method_family"] == "fx"

    params = result["selected_parameters"]
    assert params["wing_fx_send_level_delta_db"] == -4.8
    assert params["soundcheck_fx_native_level_osc_only_required"] is True
    assert params["soundcheck_fx_no_model_or_time_parameter_write_required"] is True
    assert params["soundcheck_fx_live_apply_blocked_without_wing_proof"] is True
    assert result["selected_rationale"]["wing_osc_support_level"] == (
        "native_fx_level_osc_path_templates_declared"
    )
    assert result["selected_rationale"]["wing_osc_live_write"] is False
    assert result["selected_rationale"]["wing_osc_live_authority_granted"] is False

    decay_params = result["candidate_parameters"]["shorten_fx_decay_filter"]
    assert result["candidate_blockers"]["shorten_fx_decay_filter"] == [
        WING_FX_OPERATOR_OWNED_MODEL_BLOCKER
    ]
    assert decay_params["current_wing_fx_operator_owned"] is True
    assert decay_params["wing_fx_model_parameter_current_model"] == "HALL"
    assert decay_params["wing_fx_model_parameter_current_model_full_coverage"] is True
    assert (
        decay_params[
            "wing_fx_model_parameter_current_model_operator_owned_blocks_apply"
        ]
        is True
    )
    assert decay_params["wing_fx_model_parameter_apply_ready_after_model_readback"] is False
    assert decay_params["wing_fx_model_parameter_blocker"] == (
        WING_FX_OPERATOR_OWNED_MODEL_BLOCKER
    )
    tradeoffs = {
        row["strategy_id"]: row for row in result["alternative_tradeoffs"]
    }
    assert tradeoffs["shorten_fx_decay_filter"]["wing_osc_native_supported"] is False
    assert tradeoffs["shorten_fx_decay_filter"]["wing_osc_support_level"] == (
        "fx_operator_owned_model_edit_requires_review"
    )
    assert tradeoffs["shorten_fx_decay_filter"][
        "wing_osc_unsupported_strategy_reason"
    ] == "fx_model_parameters_are_mapped_but_current_wing_fx_slot_is_operator_owned"


def test_operator_owned_fx_level_replay_keeps_level_pullback_review_only():
    report = _default_smart_dsp_replay_report(
        generated_at_utc="2026-06-29T13:15:00Z",
    )
    result = {
        row["scenario_id"]: row for row in report["results"]
    }["soundcheck_fx_wash_operator_owned_level_holds_review"]

    assert result["passed"] is True
    assert result["actual_status"] == "proposal_only"
    assert result["actual_selected_strategy_id"] == (
        "hold_soundcheck_operator_owned_fx_for_review"
    )
    assert result["actual_method_family"] == "analysis_only"
    assert result["candidate_blockers"]["reduce_fx_send_or_return"] == [
        WING_FX_LEVEL_OPERATOR_OWNED_BLOCKER
    ]

    params = result["selected_parameters"]
    assert params["reason"] == "soundcheck_operator_owned_fx_requires_operator_review"
    assert "reduce_fx_send_or_return" in params["blocked_fx_strategy_ids"]
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
    assert params["soundcheck_operator_owned_fx_level_readback_required"] is True
    assert params["soundcheck_operator_owned_fx_model_readback_required"] is True
    assert params["soundcheck_operator_owned_fx_rollback_required"] is True
    assert (
        params[
            "soundcheck_operator_owned_fx_no_level_model_time_or_ducking_without_operator_review"
        ]
        is True
    )
    assert result["operator_explanation_sections"]["selection_policy"] == (
        "soundcheck_operator_owned_fx_review_priority"
    )
    readback = result["selected_future_wing_readback_contract"][
        "soundcheck_operator_owned_fx_review_readback"
    ]
    assert readback["read_only"] is True
    assert readback["allowed_live_write_scope_before_wing_proof"] == []
    assert readback["wing_fx_level_current_operator_owned"] is True
    assert readback["wing_fx_level_operator_owned_blocker"] == (
        WING_FX_LEVEL_OPERATOR_OWNED_BLOCKER
    )
    assert "send/{send}/lvl" in readback["read_before_paths"]
    assert "fxmix" in readback["read_before_paths"]
    assert result["candidate_parameters"]["reduce_fx_send_or_return"][
        "wing_fx_send_level_delta_db"
    ] == -4.3
    tradeoffs = {
        row["strategy_id"]: row for row in result["alternative_tradeoffs"]
    }
    assert tradeoffs["reduce_fx_send_or_return"]["why_not_selected"] == (
        "soundcheck_operator_owned_fx_review_priority:"
        "manual_fx_level_or_model_owner_review_precedes_soundcheck_fx_change"
    )
    _assert_replay_analysis_only_has_no_apply_values(result)
    assert result["selected_rationale"]["wing_osc_live_write"] is False
    assert result["selected_rationale"]["wing_osc_live_authority_granted"] is False


def test_operator_owned_fx_level_alternative_keeps_level_pullback_review_only():
    report = _default_smart_dsp_replay_report(
        generated_at_utc="2026-06-29T13:15:00Z",
    )
    result = {
        row["scenario_id"]: row for row in report["results"]
    }["soundcheck_fx_wash_operator_owned_level_holds_review"]

    fx_pullback = result["candidate_parameters"]["reduce_fx_send_or_return"]
    assert fx_pullback["wing_fx_send_level_delta_db"] == -4.3
    assert fx_pullback["wing_fx_level_current_operator_owned"] is True
    assert fx_pullback["wing_fx_level_operator_owned_edit_requires_review"] is True
    assert fx_pullback["wing_fx_level_operator_owned_blocker"] == (
        WING_FX_LEVEL_OPERATOR_OWNED_BLOCKER
    )
    assert result["candidate_blockers"]["reduce_fx_send_or_return"] == [
        WING_FX_LEVEL_OPERATOR_OWNED_BLOCKER
    ]


def test_operator_owned_ds902_replay_keeps_de_esser_review_only():
    report = _default_smart_dsp_replay_report(
        generated_at_utc="2026-06-29T12:30:00Z",
    )
    result = {
        row["scenario_id"]: row for row in report["results"]
    }["soundcheck_sibilant_vocal_operator_owned_ds902_prefers_static_eq"]

    assert result["passed"] is True
    assert result["actual_selected_strategy_id"] == "static_sibilance_band_cut"
    assert result["actual_method_family"] == "eq"
    assert result["candidate_blockers"]["vocal_de_esser"] == [
        WING_GATE_SLOT_OPERATOR_OWNED_MODEL_BLOCKER
    ]

    de_esser_params = result["candidate_parameters"]["vocal_de_esser"]
    assert de_esser_params["wing_gate_slot_current_model"] == WING_DE_ESSER_GATE_MODEL
    assert de_esser_params["wing_gate_slot_current_operator_owned"] is True
    assert (
        de_esser_params["wing_gate_slot_operator_owned_model_edit_requires_review"]
        is True
    )
    assert de_esser_params["wing_gate_slot_operator_owned_model_blocker"] == (
        WING_GATE_SLOT_OPERATOR_OWNED_MODEL_BLOCKER
    )
    assert de_esser_params["wing_de_esser_gate_slot_parameter_apply_ready"] is False
    assert de_esser_params["wing_de_esser_gate_slot_apply_ready_reason"] == (
        "gate_slot_operator_owned_model_edit_requires_operator_review"
    )

    tradeoffs = {
        row["strategy_id"]: row for row in result["alternative_tradeoffs"]
    }
    assert tradeoffs["vocal_de_esser"]["wing_osc_native_supported"] is False
    assert tradeoffs["vocal_de_esser"]["wing_osc_support_level"] == (
        "gate_slot_operator_owned_model_edit_requires_review"
    )
    assert tradeoffs["vocal_de_esser"][
        "wing_osc_unsupported_strategy_reason"
    ] == "gate_slot_model_matches_but_current_wing_gate_slot_is_operator_owned"


def test_operator_owned_aux_dynamics_replay_keeps_pse_la_review_only():
    report = _default_smart_dsp_replay_report(
        generated_at_utc="2026-06-29T14:15:00Z",
    )
    result = {
        row["scenario_id"]: row for row in report["results"]
    }["soundcheck_aux_playback_operator_owned_pse_la_hold"]

    assert result["passed"] is True
    assert result["actual_status"] == "proposal_only"
    assert result["actual_selected_strategy_id"] == "hold_group_bus_for_member_evidence"
    assert result["actual_method_family"] == "analysis_only"
    assert result["candidate_blockers"]["group_bus_compression_review"] == [
        WING_AUX_DYNAMICS_OPERATOR_OWNED_BLOCKER
    ]

    aux_params = result["candidate_parameters"]["group_bus_compression_review"]
    assert aux_params["wing_aux_dynamics_target_surface"] == "aux"
    assert aux_params["wing_aux_dynamics_on_osc_value"] == 1
    assert aux_params["wing_aux_dynamics_threshold_osc_value"] == -20.3
    assert aux_params["wing_aux_dynamics_depth_osc_value"] == 2.9
    assert aux_params["wing_aux_dynamics_fast_osc_value"] == 0
    assert aux_params["wing_aux_dynamics_current_operator_owned"] is True
    assert (
        aux_params["wing_aux_dynamics_operator_owned_edit_requires_review"] is True
    )
    assert aux_params["wing_aux_dynamics_operator_owned_blocker"] == (
        WING_AUX_DYNAMICS_OPERATOR_OWNED_BLOCKER
    )

    candidate_contract = result["candidate_future_wing_readback_contracts"][
        "group_bus_compression_review"
    ]
    applicability = candidate_contract["wing_osc_applicability"]
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

    readback = candidate_contract["dynamics_readback"]
    assert readback["wing_aux_dynamics_current_operator_owned"] is True
    assert readback["wing_aux_dynamics_operator_owned_edit_requires_review"] is True
    assert readback["wing_aux_dynamics_operator_owned_blocker"] == (
        WING_AUX_DYNAMICS_OPERATOR_OWNED_BLOCKER
    )
    assert readback["wing_aux_dynamics_operator_owned_policy"] == (
        "do_not_overwrite_operator_owned_aux_dynamics_without_review"
    )

    tradeoffs = {
        row["strategy_id"]: row for row in result["alternative_tradeoffs"]
    }
    aux_tradeoff = tradeoffs["group_bus_compression_review"]
    assert aux_tradeoff["wing_osc_native_supported"] is False
    assert aux_tradeoff["wing_osc_support_level"] == (
        "aux_dynamics_operator_owned_edit_requires_review"
    )
    assert aux_tradeoff["wing_osc_unsupported_strategy_reason"] == (
        "aux_dynamics_has_native_osc_paths_but_current_wing_aux_dynamics_is_operator_owned"
    )
    assert aux_tradeoff["wing_osc_candidate_apply_status"] == (
        "not_wing_applicable_until_aux_dynamics_readback_or_operator_review"
    )
    assert aux_tradeoff["why_not_selected"] == (
        "blocked_by:wing_aux_dynamics_operator_owned_edit_requires_operator_review"
    )


def test_operator_owned_std_eq_replay_keeps_static_hpf_review_only():
    report = _default_smart_dsp_replay_report(
        generated_at_utc="2026-06-29T13:20:00Z",
    )
    result = {
        row["scenario_id"]: row for row in report["results"]
    }["soundcheck_plosive_vocal_operator_owned_std_eq_hold"]

    assert result["passed"] is True
    assert result["actual_selected_strategy_id"] == (
        "manual_plosive_mic_technique_review"
    )
    assert result["actual_method_family"] == "analysis_only"
    assert result["candidate_blockers"]["static_plosive_high_pass_cleanup"] == [
        WING_EQ_OPERATOR_OWNED_MODEL_BLOCKER
    ]

    static_hpf_params = result["candidate_parameters"][
        "static_plosive_high_pass_cleanup"
    ]
    assert static_hpf_params["wing_eq_current_model"] == WING_EQ_REQUIRED_MODEL
    assert static_hpf_params["wing_eq_current_operator_owned"] is True
    assert (
        static_hpf_params["wing_eq_operator_owned_model_edit_requires_review"]
        is True
    )
    assert static_hpf_params["wing_eq_operator_owned_model_blocker"] == (
        WING_EQ_OPERATOR_OWNED_MODEL_BLOCKER
    )

    tradeoffs = {
        row["strategy_id"]: row for row in result["alternative_tradeoffs"]
    }
    assert tradeoffs["static_plosive_high_pass_cleanup"][
        "wing_osc_native_supported"
    ] is False
    assert tradeoffs["static_plosive_high_pass_cleanup"][
        "wing_osc_support_level"
    ] == "eq_operator_owned_model_edit_requires_review"
    assert tradeoffs["static_plosive_high_pass_cleanup"][
        "wing_osc_unsupported_strategy_reason"
    ] == "eq_model_matches_but_current_wing_eq_is_operator_owned"


def test_operator_owned_low_cut_filter_replay_keeps_static_hpf_review_only():
    report = _default_smart_dsp_replay_report(
        generated_at_utc="2026-06-29T14:25:00Z",
    )
    result = {
        row["scenario_id"]: row for row in report["results"]
    }["soundcheck_plosive_vocal_operator_owned_low_cut_filter_hold"]

    assert result["passed"] is True
    assert result["actual_selected_strategy_id"] == "low_mid_corrective_cut"
    assert result["actual_method_family"] == "eq"
    assert result["candidate_blockers"]["static_plosive_high_pass_cleanup"] == [
        WING_EQ_FILTER_OPERATOR_OWNED_BLOCKER
    ]

    static_hpf_params = result["candidate_parameters"][
        "static_plosive_high_pass_cleanup"
    ]
    assert static_hpf_params["wing_eq_filter_family"] == "channel_low_cut"
    assert static_hpf_params["wing_eq_filter_family_operator_owned_scope"] == (
        "channel_low_cut"
    )
    assert static_hpf_params["wing_eq_filter_current_operator_owned"] is True
    assert (
        static_hpf_params["wing_eq_filter_operator_owned_edit_requires_review"]
        is True
    )
    assert static_hpf_params["wing_eq_filter_operator_owned_blocker"] == (
        WING_EQ_FILTER_OPERATOR_OWNED_BLOCKER
    )

    candidate_contract = result["candidate_future_wing_readback_contracts"][
        "static_plosive_high_pass_cleanup"
    ]
    eq_readback = candidate_contract["eq_band_readback"]
    assert eq_readback["wing_eq_filter_current_operator_owned"] is True
    assert eq_readback["wing_eq_filter_operator_owned_blocker"] == (
        WING_EQ_FILTER_OPERATOR_OWNED_BLOCKER
    )
    applicability = candidate_contract["wing_osc_applicability"]
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
        row["strategy_id"]: row for row in result["alternative_tradeoffs"]
    }
    assert tradeoffs["static_plosive_high_pass_cleanup"][
        "wing_osc_native_supported"
    ] is False
    assert tradeoffs["static_plosive_high_pass_cleanup"][
        "wing_osc_support_level"
    ] == "eq_filter_operator_owned_edit_requires_review"
    assert tradeoffs["static_plosive_high_pass_cleanup"][
        "wing_osc_candidate_apply_status"
    ] == "not_wing_applicable_until_eq_filter_readback_or_operator_review"
    assert tradeoffs["static_plosive_high_pass_cleanup"]["why_not_selected"] == (
        "blocked_by:wing_eq_filter_operator_owned_edit_requires_operator_review"
    )


def test_operator_owned_comp_replay_keeps_compressor_review_only():
    report = _default_smart_dsp_replay_report(
        generated_at_utc="2026-06-29T13:25:00Z",
    )
    result = {
        row["scenario_id"]: row for row in report["results"]
    }["soundcheck_dynamic_vocal_operator_owned_comp_ride"]

    assert result["passed"] is True
    assert result["actual_selected_strategy_id"] == (
        "manual_level_ride_before_compression"
    )
    assert result["actual_method_family"] == "analysis_only"
    assert result["candidate_blockers"]["vocal_phrase_leveling_compression"] == [
        WING_DYNAMICS_OPERATOR_OWNED_MODEL_BLOCKER
    ]

    compressor_params = result["candidate_parameters"][
        "vocal_phrase_leveling_compression"
    ]
    assert compressor_params["wing_dynamics_current_model"] == (
        WING_DYNAMICS_REQUIRED_MODEL
    )
    assert compressor_params["wing_dynamics_current_operator_owned"] is True
    assert (
        compressor_params[
            "wing_dynamics_operator_owned_model_edit_requires_review"
        ]
        is True
    )
    assert compressor_params["wing_dynamics_operator_owned_model_blocker"] == (
        WING_DYNAMICS_OPERATOR_OWNED_MODEL_BLOCKER
    )

    tradeoffs = {
        row["strategy_id"]: row for row in result["alternative_tradeoffs"]
    }
    assert tradeoffs["vocal_phrase_leveling_compression"][
        "wing_osc_native_supported"
    ] is False
    assert tradeoffs["vocal_phrase_leveling_compression"][
        "wing_osc_support_level"
    ] == "dynamics_operator_owned_model_edit_requires_review"
    assert tradeoffs["vocal_phrase_leveling_compression"][
        "wing_osc_unsupported_strategy_reason"
    ] == "dynamics_model_matches_but_current_wing_dynamics_is_operator_owned"


def test_operator_owned_gain_replay_keeps_level_plane_gain_review_only():
    report = _default_smart_dsp_replay_report(
        generated_at_utc="2026-06-29T15:10:00Z",
    )
    results = {
        row["scenario_id"]: row for row in report["results"]
    }

    expected_cases = {
        "soundcheck_quiet_vocal_operator_owned_gain_holds_review": (
            "soundcheck_bounded_gain_up",
            3.0,
            "up",
        ),
        "soundcheck_loud_vocal_operator_owned_gain_down_holds_review": (
            "level_plane_gain_down",
            -3.0,
            "down",
        ),
    }
    for scenario_id, (blocked_strategy_id, expected_delta, move_kind) in (
        expected_cases.items()
    ):
        result = results[scenario_id]
        assert result["passed"] is True
        assert result["actual_status"] == "analysis_only"
        assert result["actual_selected_strategy_id"] == (
            "hold_soundcheck_operator_owned_gain_for_review"
        )
        assert result["actual_method_family"] == "analysis_only"
        assert result["operator_explanation_sections"]["selection_policy"] == (
            "soundcheck_operator_owned_gain_review_priority"
        )
        assert result["candidate_blockers"][blocked_strategy_id] == [
            WING_GAIN_OPERATOR_OWNED_BLOCKER
        ]

        params = result["selected_parameters"]
        assert params["dsp_change"] == "hold"
        assert params["calculated_delta_db"] == expected_delta
        assert params["calculated_gain_direction"] == move_kind
        assert params["blocked_gain_strategy_id"] == blocked_strategy_id
        assert params["blocked_gain_delta_db"] == expected_delta
        assert params["gain_db"] == 0.0
        assert params["wing_gain_current_operator_owned"] is True
        assert params["wing_gain_operator_owned_edit_requires_review"] is True
        assert params["wing_gain_operator_owned_blocker"] == (
            WING_GAIN_OPERATOR_OWNED_BLOCKER
        )
        assert params["soundcheck_operator_owned_gain_selected_write_keys"] == []
        assert params["soundcheck_operator_owned_gain_forbidden_write_keys"] == [
            "wing_gain_delta_db"
        ]
        assert "wing_gain_delta_db" not in params

        source_readback = result["selected_future_wing_readback_contract"][
            "operator_owned_source_gain_readback"
        ]
        assert source_readback["read_only"] is True
        assert source_readback["read_before_paths"] == ["fdr", "in/set/trim"]
        assert source_readback["read_after_paths"] == ["fdr", "in/set/trim"]
        assert "/ch/{channel}/fdr" in source_readback["wing_osc_path_templates"]
        assert "/ch/{channel}/in/set/trim" in source_readback[
            "wing_osc_path_templates"
        ]
        assert source_readback["blocked_gain_strategy_id"] == blocked_strategy_id
        assert source_readback["blocked_gain_delta_db"] == expected_delta
        assert source_readback["soundcheck_operator_owned_gain_move_kind"] == move_kind
        assert source_readback["wing_source_gain_forbidden_apply_fields"] == [
            "wing_gain_delta_db"
        ]
        assert source_readback["allowed_live_write_scope_before_wing_proof"] == []

        tradeoffs = {
            row["strategy_id"]: row["why_not_selected"]
            for row in result["alternative_tradeoffs"]
        }
        assert tradeoffs[blocked_strategy_id] == (
            "soundcheck_operator_owned_gain_review_priority:"
            "manual_fader_or_trim_owner_review_precedes_soundcheck_gain_move"
        )
        assert result["selected_rationale"][
            "wing_osc_candidate_apply_status"
        ] == "analysis_only_no_console_write"
        _assert_replay_analysis_only_has_no_apply_values(result)


def test_operator_owned_mute_replay_keeps_manual_mute_review_only():
    report = _default_smart_dsp_replay_report(
        generated_at_utc="2026-06-29T14:35:00Z",
    )
    result = {
        row["scenario_id"]: row for row in report["results"]
    }["soundcheck_extreme_noisy_tom_operator_owned_mute_holds_review"]

    assert result["passed"] is True
    assert result["actual_status"] == "analysis_only"
    assert result["actual_selected_strategy_id"] == (
        "hold_soundcheck_operator_owned_mute_for_review"
    )
    assert result["actual_method_family"] == "analysis_only"
    assert result["operator_explanation_sections"]["selection_policy"] == (
        "soundcheck_operator_owned_mute_review_priority"
    )
    assert result["candidate_blockers"]["manual_tom_mute_between_hits"] == [
        WING_MUTE_OPERATOR_OWNED_BLOCKER
    ]

    params = result["selected_parameters"]
    assert params["dsp_change"] == "hold"
    assert params["blocked_mute_strategy_id"] == "manual_tom_mute_between_hits"
    assert params["blocked_mute_proposed_muted"] is True
    assert params["blocked_mute_scope"] == "between_tom_hits_only"
    assert params["wing_mute_current_operator_owned"] is True
    assert params["wing_mute_operator_owned_edit_requires_review"] is True
    assert params["wing_mute_operator_owned_blocker"] == (
        WING_MUTE_OPERATOR_OWNED_BLOCKER
    )
    assert params["soundcheck_operator_owned_mute_selected_write_keys"] == []
    assert params["soundcheck_operator_owned_mute_forbidden_write_keys"] == [
        "wing_mute_osc_value"
    ]
    assert "wing_mute_osc_value" not in params

    candidate_params = result["candidate_parameters"]["manual_tom_mute_between_hits"]
    assert candidate_params["wing_mute_osc_value"] == WING_MUTE_ON_VALUE
    mute_readback = result["selected_future_wing_readback_contract"][
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
    assert mute_readback["wing_mute_operator_owned_blocker"] == (
        WING_MUTE_OPERATOR_OWNED_BLOCKER
    )
    tradeoffs = {
        row["strategy_id"]: row["why_not_selected"]
        for row in result["alternative_tradeoffs"]
    }
    assert tradeoffs["manual_tom_mute_between_hits"] == (
        "soundcheck_operator_owned_mute_review_priority:"
        "manual_mute_owner_review_precedes_soundcheck_mute_change"
    )
    assert result["selected_rationale"]["wing_osc_support_level"] == (
        "not_a_console_mutation"
    )
    assert result["selected_rationale"][
        "wing_osc_candidate_apply_status"
    ] == "analysis_only_no_console_write"
    _assert_replay_analysis_only_has_no_apply_values(result)


def test_operator_owned_pan_width_replay_keeps_pan_review_only():
    report = _default_smart_dsp_replay_report(
        generated_at_utc="2026-06-29T14:20:00Z",
    )
    result = {
        row["scenario_id"]: row for row in report["results"]
    }["soundcheck_operator_owned_pan_width_holds_review"]

    assert result["passed"] is True
    assert result["actual_selected_strategy_id"] == (
        "hold_soundcheck_operator_owned_pan_width_for_review"
    )
    assert result["actual_method_family"] == "analysis_only"
    assert result["operator_explanation_sections"]["selection_policy"] == (
        "soundcheck_operator_owned_pan_width_review_priority"
    )
    assert result["candidate_blockers"]["phase_safe_pan_target"] == [
        WING_PAN_WIDTH_OPERATOR_OWNED_BLOCKER
    ]

    selected_params = result["selected_parameters"]
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
    pan_readback = result["selected_future_wing_readback_contract"][
        "pan_width_readback"
    ]
    assert pan_readback["soundcheck_operator_owned_pan_width_hold_proposal"] is True
    assert pan_readback["soundcheck_operator_owned_pan_width_selected_write_keys"] == []
    assert pan_readback["soundcheck_operator_owned_pan_width_forbidden_write_keys"] == [
        "wing_pan_osc_value",
        "wing_width_osc_value",
    ]
    assert pan_readback["soundcheck_operator_owned_pan_width_readback_required"] is True
    assert (
        pan_readback[
            "soundcheck_operator_owned_pan_width_live_apply_blocked_without_wing_proof"
        ]
        is True
    )
    assert result["selected_rationale"]["wing_osc_apply_parameter_values"] == {}
    assert result["proposal_authority"]["live_apply_allowed"] is False
    assert result["proposal_authority"]["live_authority_granted"] is False
    assert result["proposal_authority"]["live_write"] is False

    pan_params = result["candidate_parameters"]["phase_safe_pan_target"]
    assert pan_params["wing_pan_width_current_operator_owned"] is True
    assert (
        pan_params["wing_pan_width_operator_owned_edit_requires_review"] is True
    )
    assert pan_params["wing_pan_width_operator_owned_blocker"] == (
        WING_PAN_WIDTH_OPERATOR_OWNED_BLOCKER
    )

    tradeoffs = {
        row["strategy_id"]: row for row in result["alternative_tradeoffs"]
    }
    assert tradeoffs["phase_safe_pan_target"]["wing_osc_native_supported"] is False
    assert tradeoffs["phase_safe_pan_target"]["wing_osc_support_level"] == (
        "pan_width_operator_owned_edit_requires_review"
    )
    assert tradeoffs["phase_safe_pan_target"][
        "wing_osc_unsupported_strategy_reason"
    ] == (
        "pan_width_has_native_osc_paths_but_current_wing_pan_width_is_operator_owned"
    )


def test_selected_live_fx_requires_pullback_only_guard_contract():
    report = _default_smart_dsp_replay_report(
        generated_at_utc="2026-06-26T23:35:00Z",
    )
    live_fx = {
        row["scenario_id"]: row for row in report["results"]
    }["live_delay_smear_fx_pullback_before_ducking"]
    selected = SimpleNamespace(
        method_family="fx",
        parameters=live_fx["selected_parameters"],
        future_wing_readback_contract=live_fx[
            "selected_future_wing_readback_contract"
        ],
    )

    assert _readback_contract_failures(selected, "valid_live_fx_guard") == []

    tampered_contract = deepcopy(live_fx["selected_future_wing_readback_contract"])
    tampered_contract["fx_readback"][
        "live_fx_live_apply_blocked_without_wing_proof"
    ] = False
    tampered_selected = SimpleNamespace(
        method_family="fx",
        parameters=live_fx["selected_parameters"],
        future_wing_readback_contract=tampered_contract,
    )

    failures = _readback_contract_failures(
        tampered_selected,
        "tampered_live_fx_guard",
    )

    failure_codes = {failure["code"] for failure in failures}
    assert (
        "tampered_live_fx_guard_readback_contract_live_fx_guard_live_fx_live_apply_blocked_without_wing_proof_mismatch"
        in failure_codes
    )

    tampered_write_contract = deepcopy(
        live_fx["selected_future_wing_readback_contract"]
    )
    tampered_write_contract["fx_readback"]["live_fx_forbidden_write_keys"] = []
    tampered_write_selected = SimpleNamespace(
        method_family="fx",
        parameters=live_fx["selected_parameters"],
        future_wing_readback_contract=tampered_write_contract,
    )

    write_failures = _readback_contract_failures(
        tampered_write_selected,
        "tampered_live_fx_write_guard",
    )

    write_failure_codes = {failure["code"] for failure in write_failures}
    assert (
        "tampered_live_fx_write_guard_readback_contract_live_fx_guard_live_fx_forbidden_write_keys_mismatch"
        in write_failure_codes
    )


def test_selected_soundcheck_fx_requires_bounded_level_guard_contract():
    report = _default_smart_dsp_replay_report(
        generated_at_utc="2026-06-27T01:55:00Z",
    )
    soundcheck_fx = {
        row["scenario_id"]: row for row in report["results"]
    }["soundcheck_long_fx_tail_decay_filter"]
    selected = SimpleNamespace(
        method_family="fx",
        parameters=soundcheck_fx["selected_parameters"],
        future_wing_readback_contract=soundcheck_fx[
            "selected_future_wing_readback_contract"
        ],
    )

    assert _readback_contract_failures(selected, "valid_soundcheck_fx_guard") == []

    tampered_contract = deepcopy(soundcheck_fx["selected_future_wing_readback_contract"])
    tampered_contract["fx_readback"][
        "soundcheck_fx_live_apply_blocked_without_wing_proof"
    ] = False
    tampered_selected = SimpleNamespace(
        method_family="fx",
        parameters=soundcheck_fx["selected_parameters"],
        future_wing_readback_contract=tampered_contract,
    )

    failures = _readback_contract_failures(
        tampered_selected,
        "tampered_soundcheck_fx_guard",
    )

    failure_codes = {failure["code"] for failure in failures}
    assert (
        "tampered_soundcheck_fx_guard_readback_contract_soundcheck_fx_guard_soundcheck_fx_live_apply_blocked_without_wing_proof_mismatch"
        in failure_codes
    )

    tampered_write_contract = deepcopy(
        soundcheck_fx["selected_future_wing_readback_contract"]
    )
    tampered_write_contract["fx_readback"][
        "soundcheck_fx_forbidden_write_keys"
    ] = []
    tampered_write_selected = SimpleNamespace(
        method_family="fx",
        parameters=soundcheck_fx["selected_parameters"],
        future_wing_readback_contract=tampered_write_contract,
    )

    write_failures = _readback_contract_failures(
        tampered_write_selected,
        "tampered_soundcheck_fx_write_guard",
    )

    write_failure_codes = {failure["code"] for failure in write_failures}
    assert (
        "tampered_soundcheck_fx_write_guard_readback_contract_soundcheck_fx_guard_soundcheck_fx_forbidden_write_keys_mismatch"
        in write_failure_codes
    )

    tampered_parameters = deepcopy(soundcheck_fx["selected_parameters"])
    tampered_parameters["wing_fx_send_level_delta_db"] = -1.0
    tampered_parameter_selected = SimpleNamespace(
        method_family="fx",
        parameters=tampered_parameters,
        future_wing_readback_contract=soundcheck_fx[
            "selected_future_wing_readback_contract"
        ],
    )

    parameter_failures = _readback_contract_failures(
        tampered_parameter_selected,
        "tampered_soundcheck_fx_unselected_send",
    )

    parameter_failure_codes = {failure["code"] for failure in parameter_failures}
    assert (
        "tampered_soundcheck_fx_unselected_send_readback_contract_soundcheck_fx_guard_forbidden_write_key_present"
        in parameter_failure_codes
    )

    tampered_followup_contract = deepcopy(
        soundcheck_fx["selected_future_wing_readback_contract"]
    )
    tampered_followup_contract["fx_readback"][
        "wing_fx_model_time_followup_apply_ready"
    ] = True
    tampered_followup_contract["fx_readback"]["read_before_paths"] = [
        *tampered_followup_contract["fx_readback"]["read_before_paths"],
        WING_FX_MODEL_PARAMETER_PATH_PATTERN,
    ]
    tampered_followup_selected = SimpleNamespace(
        method_family="fx",
        parameters=soundcheck_fx["selected_parameters"],
        future_wing_readback_contract=tampered_followup_contract,
    )

    followup_failures = _readback_contract_failures(
        tampered_followup_selected,
        "tampered_soundcheck_fx_followup",
    )

    followup_failure_codes = {failure["code"] for failure in followup_failures}
    assert (
        "tampered_soundcheck_fx_followup_readback_contract_fx_model_time_followup_wing_fx_model_time_followup_apply_ready_mismatch"
        in followup_failure_codes
    )
    assert (
        "tampered_soundcheck_fx_followup_readback_contract_fx_model_time_followup_forbidden_read_before_paths_present"
        in followup_failure_codes
    )


def test_selected_path_only_dynamics_rejects_no_actionable_wing_osc_value():
    params = _with_wing_native_parameter_mappings(
        {},
        method_family="dynamics",
        target_id="ch:5",
    )
    contract = _future_readback_contract("dynamics", "ch:5", params)
    selected = SimpleNamespace(
        method_family="dynamics",
        parameters=params,
        future_wing_readback_contract=contract,
    )

    failures = _readback_contract_failures(
        selected,
        "tampered_dynamics_noop",
    )

    failure_codes = {failure["code"] for failure in failures}
    applicability = contract["wing_osc_applicability"]
    assert applicability["native_osc_supported"] is False
    assert (
        applicability["native_osc_support_level"]
        == "no_actionable_wing_osc_parameter_value"
    )
    assert applicability["actionable_parameter_keys"] == []
    assert (
        applicability["actionable_parameter_blocker"]
        == WING_ACTIONABLE_PARAMETER_BLOCKER
    )
    assert (
        "tampered_dynamics_noop_readback_contract_wing_osc_selected_not_native"
        in failure_codes
    )
    assert (
        "tampered_dynamics_noop_selected_no_actionable_wing_osc_parameter_value"
        in failure_codes
    )


def test_selected_dynamics_requires_wing_model_contract():
    report = _default_smart_dsp_replay_report(
        generated_at_utc="2026-06-25T21:45:00Z",
    )
    result = {
        row["scenario_id"]: row for row in report["results"]
    }["soundcheck_dynamic_vocal_compression"]
    params = deepcopy(result["selected_parameters"])
    params.pop("wing_compressor_model_readback_must_equal")
    contract = deepcopy(result["selected_future_wing_readback_contract"])
    contract["dynamics_readback"].pop("dynamics_model_readback_must_equal")
    selected = SimpleNamespace(
        method_family=result["actual_method_family"],
        parameters=params,
        future_wing_readback_contract=contract,
    )

    failures = _readback_contract_failures(selected, "tampered_dynamics_model")

    failure_codes = {failure["code"] for failure in failures}
    assert (
        "tampered_dynamics_model_selected_dynamics_wing_model_mismatch"
        in failure_codes
    )
    assert (
        "tampered_dynamics_model_readback_contract_wing_dynamics_model_mismatch"
        in failure_codes
    )


def test_selected_dynamics_requires_wing_compressor_on_mapping():
    report = _default_smart_dsp_replay_report(
        generated_at_utc="2026-06-26T16:45:00Z",
    )
    result = {
        row["scenario_id"]: row for row in report["results"]
    }["soundcheck_dynamic_vocal_compression"]
    params = deepcopy(result["selected_parameters"])
    for key in (
        "wing_compressor_on_osc_value",
        "wing_compressor_on_value_type",
        "wing_compressor_on_source",
        "wing_compressor_on_osc_path_suffix",
        "wing_compressor_on_original_value",
        "wing_compressor_on_mapping_policy",
        "wing_compressor_on_min",
        "wing_compressor_on_max",
    ):
        params.pop(key, None)
    contract = deepcopy(result["selected_future_wing_readback_contract"])
    contract["dynamics_readback"].pop("wing_compressor_on_mapping_required", None)
    selected = SimpleNamespace(
        method_family=result["actual_method_family"],
        parameters=params,
        future_wing_readback_contract=contract,
    )

    failures = _readback_contract_failures(selected, "tampered_dynamics_on")

    failure_codes = {failure["code"] for failure in failures}
    assert (
        "tampered_dynamics_on_selected_dynamics_wing_on_mapping_missing"
        in failure_codes
    )
    assert (
        "tampered_dynamics_on_readback_contract_wing_on_mapping_not_required"
        in failure_codes
    )


def test_selected_dynamics_requires_wing_compressor_auto_mapping():
    report = _default_smart_dsp_replay_report(
        generated_at_utc="2026-06-26T16:35:00Z",
    )
    result = {
        row["scenario_id"]: row for row in report["results"]
    }["soundcheck_dynamic_vocal_compression"]
    params = deepcopy(result["selected_parameters"])
    for key in (
        "wing_compressor_auto_osc_value",
        "wing_compressor_auto_value_type",
        "wing_compressor_auto_source",
        "wing_compressor_auto_osc_path_suffix",
        "wing_compressor_auto_original_value",
        "wing_compressor_auto_mapping_policy",
        "wing_compressor_auto_min",
        "wing_compressor_auto_max",
    ):
        params.pop(key, None)
    contract = deepcopy(result["selected_future_wing_readback_contract"])
    contract["dynamics_readback"].pop("wing_compressor_auto_mapping_required", None)
    selected = SimpleNamespace(
        method_family=result["actual_method_family"],
        parameters=params,
        future_wing_readback_contract=contract,
    )

    failures = _readback_contract_failures(selected, "tampered_dynamics_auto")

    failure_codes = {failure["code"] for failure in failures}
    assert (
        "tampered_dynamics_auto_selected_dynamics_wing_auto_mapping_missing"
        in failure_codes
    )
    assert (
        "tampered_dynamics_auto_readback_contract_wing_auto_mapping_not_required"
        in failure_codes
    )


def test_selected_dynamics_requires_wing_compressor_crossover_mapping():
    report = _default_smart_dsp_replay_report(
        generated_at_utc="2026-06-26T18:45:00Z",
    )
    result = {
        row["scenario_id"]: row for row in report["results"]
    }["soundcheck_dynamic_vocal_compression"]
    params = deepcopy(result["selected_parameters"])
    for key in (
        "wing_compressor_crossover_type_osc_value",
        "wing_compressor_crossover_type_value_type",
        "wing_compressor_crossover_depth_osc_value",
        "wing_compressor_crossover_depth_value_type",
        "wing_compressor_crossover_frequency_osc_value",
        "wing_compressor_crossover_frequency_value_type",
        "wing_compressor_crossover_source",
        "wing_compressor_crossover_reference",
    ):
        params.pop(key, None)
    contract = deepcopy(result["selected_future_wing_readback_contract"])
    contract["dynamics_readback"].pop(
        "wing_compressor_crossover_mapping_required",
        None,
    )
    selected = SimpleNamespace(
        method_family=result["actual_method_family"],
        parameters=params,
        future_wing_readback_contract=contract,
    )

    failures = _readback_contract_failures(selected, "tampered_dynamics_crossover")

    failure_codes = {failure["code"] for failure in failures}
    assert (
        "tampered_dynamics_crossover_selected_dynamics_wing_crossover_mapping_missing"
        in failure_codes
    )
    assert (
        "tampered_dynamics_crossover_readback_contract_wing_crossover_mapping_not_required"
        in failure_codes
    )


def test_selected_dynamics_requires_wing_compressor_sidechain_mapping():
    report = _default_smart_dsp_replay_report(
        generated_at_utc="2026-06-26T18:15:00Z",
    )
    result = {
        row["scenario_id"]: row for row in report["results"]
    }["soundcheck_dynamic_vocal_compression"]
    params = deepcopy(result["selected_parameters"])
    for key in (
        "wing_compressor_sidechain_type_osc_value",
        "wing_compressor_sidechain_type_value_type",
        "wing_compressor_sidechain_frequency_osc_value",
        "wing_compressor_sidechain_frequency_value_type",
        "wing_compressor_sidechain_q_osc_value",
        "wing_compressor_sidechain_q_value_type",
        "wing_compressor_sidechain_src_osc_value",
        "wing_compressor_sidechain_src_value_type",
        "wing_compressor_sidechain_tap_osc_value",
        "wing_compressor_sidechain_tap_value_type",
        "wing_compressor_sidechain_source",
        "wing_compressor_sidechain_reference",
    ):
        params.pop(key, None)
    contract = deepcopy(result["selected_future_wing_readback_contract"])
    contract["dynamics_readback"].pop(
        "wing_compressor_sidechain_mapping_required",
        None,
    )
    selected = SimpleNamespace(
        method_family=result["actual_method_family"],
        parameters=params,
        future_wing_readback_contract=contract,
    )

    failures = _readback_contract_failures(selected, "tampered_dynamics_sidechain")

    failure_codes = {failure["code"] for failure in failures}
    assert (
        "tampered_dynamics_sidechain_selected_dynamics_wing_sidechain_mapping_missing"
        in failure_codes
    )
    assert (
        "tampered_dynamics_sidechain_readback_contract_wing_sidechain_mapping_not_required"
        in failure_codes
    )


def test_selected_dynamics_requires_all_wing_values_to_be_actionable():
    report = _default_smart_dsp_replay_report(
        generated_at_utc="2026-06-26T19:55:00Z",
    )
    result = {
        row["scenario_id"]: row for row in report["results"]
    }["soundcheck_dynamic_vocal_compression"]
    contract = deepcopy(result["selected_future_wing_readback_contract"])
    contract["wing_osc_applicability"]["actionable_parameter_keys"].remove(
        "wing_compressor_sidechain_frequency_osc_value"
    )
    selected = SimpleNamespace(
        method_family=result["actual_method_family"],
        parameters=deepcopy(result["selected_parameters"]),
        future_wing_readback_contract=contract,
    )

    failures = _readback_contract_failures(
        selected,
        "tampered_dynamics_actionable_values",
    )

    failure_codes = {failure["code"] for failure in failures}
    assert (
        "tampered_dynamics_actionable_values_selected_dynamics_actionable_keys_missing"
        in failure_codes
    )


def test_selected_dynamics_requires_selected_write_key_contract():
    report = _default_smart_dsp_replay_report(
        generated_at_utc="2026-06-28T08:10:00Z",
    )
    result = {
        row["scenario_id"]: row for row in report["results"]
    }["soundcheck_dynamic_vocal_compression"]
    contract = deepcopy(result["selected_future_wing_readback_contract"])
    contract["dynamics_readback"]["wing_dynamics_forbidden_write_keys"] = [
        "wing_compressor_on_osc_value"
    ]
    selected = SimpleNamespace(
        method_family=result["actual_method_family"],
        parameters=deepcopy(result["selected_parameters"]),
        future_wing_readback_contract=contract,
    )

    failures = _readback_contract_failures(
        selected,
        "tampered_dynamics_write_keys",
    )

    failure_codes = {failure["code"] for failure in failures}
    assert (
        "tampered_dynamics_write_keys_selected_dynamics_wing_dynamics_forbidden_write_keys_mismatch"
        in failure_codes
    )


def test_selected_dynamics_requires_selected_parameter_rollback_operations():
    report = _default_smart_dsp_replay_report(
        generated_at_utc="2026-06-30T08:10:00Z",
    )
    result = {
        row["scenario_id"]: row for row in report["results"]
    }["soundcheck_dynamic_vocal_compression"]
    contract = deepcopy(result["selected_future_wing_readback_contract"])
    contract["dynamics_readback"].pop(
        "wing_dynamics_selected_parameter_rollback_operations"
    )
    selected = SimpleNamespace(
        method_family=result["actual_method_family"],
        parameters=deepcopy(result["selected_parameters"]),
        future_wing_readback_contract=contract,
    )

    failures = _readback_contract_failures(
        selected,
        "tampered_dynamics_rollback_ops",
    )

    failure_codes = {failure["code"] for failure in failures}
    assert (
        "tampered_dynamics_rollback_ops_readback_contract_"
        "wing_dynamics_selected_parameter_rollback_operations_missing"
    ) in failure_codes


def test_selected_dynamics_rejects_generic_threshold_readback_path():
    report = _default_smart_dsp_replay_report(
        generated_at_utc="2026-06-26T01:55:00Z",
    )
    result = {
        row["scenario_id"]: row for row in report["results"]
    }["soundcheck_dynamic_vocal_compression"]
    contract = deepcopy(result["selected_future_wing_readback_contract"])
    dynamics_readback = contract["dynamics_readback"]
    dynamics_readback["read_before_paths"] = [
        "dyn/thr" if path.startswith("dyn/thr") else path
        for path in dynamics_readback["read_before_paths"]
    ]
    dynamics_readback["read_after_paths"] = [
        "dyn/thr" if path.startswith("dyn/thr") else path
        for path in dynamics_readback["read_after_paths"]
    ]
    selected = SimpleNamespace(
        method_family=result["actual_method_family"],
        parameters=deepcopy(result["selected_parameters"]),
        future_wing_readback_contract=contract,
    )

    failures = _readback_contract_failures(
        selected,
        "tampered_dynamics_threshold_readback",
    )

    failure_codes = {failure["code"] for failure in failures}
    assert (
        "tampered_dynamics_threshold_readback_readback_contract_wing_dynamics_generic_threshold_path_present"
        in failure_codes
    )
    assert (
        "tampered_dynamics_threshold_readback_readback_contract_wing_dynamics_threshold_readback_path_missing"
        in failure_codes
    )


def test_selected_dynamics_requires_threshold_osc_value_contract():
    report = _default_smart_dsp_replay_report(
        generated_at_utc="2026-06-26T03:05:00Z",
    )
    result = {
        row["scenario_id"]: row for row in report["results"]
    }["soundcheck_dynamic_vocal_compression"]
    params = deepcopy(result["selected_parameters"])
    contract = deepcopy(result["selected_future_wing_readback_contract"])
    params.pop("wing_compressor_threshold_osc_value")
    contract["dynamics_readback"].pop("wing_compressor_threshold_osc_value")
    contract["dynamics_readback"].pop(
        "wing_compressor_threshold_readback_must_match_osc_value"
    )
    selected = SimpleNamespace(
        method_family=result["actual_method_family"],
        parameters=params,
        future_wing_readback_contract=contract,
    )

    failures = _readback_contract_failures(
        selected,
        "tampered_dynamics_threshold_value",
    )

    failure_codes = {failure["code"] for failure in failures}
    assert (
        "tampered_dynamics_threshold_value_selected_dynamics_wing_threshold_mapping_missing"
        in failure_codes
    )
    assert (
        "tampered_dynamics_threshold_value_selected_dynamics_wing_threshold_value_out_of_range"
        in failure_codes
    )
    assert (
        "tampered_dynamics_threshold_value_readback_contract_wing_threshold_value_readback_missing"
        in failure_codes
    )


def test_selected_gate_requires_wing_model_contract():
    report = _default_smart_dsp_replay_report(
        generated_at_utc="2026-06-25T21:45:00Z",
    )
    result = {
        row["scenario_id"]: row for row in report["results"]
    }["measured_tom_tail_risk_expander"]
    params = deepcopy(result["selected_parameters"])
    params.pop("wing_gate_model_readback_must_equal")
    contract = deepcopy(result["selected_future_wing_readback_contract"])
    contract["gate_expander_readback"].pop("gate_model_readback_must_equal")
    selected = SimpleNamespace(
        method_family=result["actual_method_family"],
        parameters=params,
        future_wing_readback_contract=contract,
    )

    failures = _readback_contract_failures(selected, "tampered_gate_model")

    failure_codes = {failure["code"] for failure in failures}
    assert "tampered_gate_model_selected_gate_wing_model_mismatch" in failure_codes
    assert (
        "tampered_gate_model_readback_contract_wing_gate_model_mismatch"
        in failure_codes
    )


def test_selected_gate_requires_wing_on_mapping():
    report = _default_smart_dsp_replay_report(
        generated_at_utc="2026-06-26T16:55:00Z",
    )
    result = {
        row["scenario_id"]: row for row in report["results"]
    }["clean_tom_bleed_aware_gate"]
    params = deepcopy(result["selected_parameters"])
    for key in (
        "wing_gate_on_osc_value",
        "wing_gate_on_value_type",
        "wing_gate_on_source",
        "wing_gate_on_osc_path_suffix",
        "wing_gate_on_original_value",
        "wing_gate_on_mapping_policy",
        "wing_gate_on_min",
        "wing_gate_on_max",
    ):
        params.pop(key, None)
    contract = deepcopy(result["selected_future_wing_readback_contract"])
    contract["gate_expander_readback"].pop("wing_gate_on_mapping_required", None)
    selected = SimpleNamespace(
        method_family=result["actual_method_family"],
        parameters=params,
        future_wing_readback_contract=contract,
    )

    failures = _readback_contract_failures(selected, "tampered_gate_on")

    failure_codes = {failure["code"] for failure in failures}
    assert "tampered_gate_on_selected_gate_wing_on_mapping_missing" in failure_codes
    assert (
        "tampered_gate_on_readback_contract_wing_gate_on_mapping_not_required"
        in failure_codes
    )


def test_selected_gate_requires_wing_sidechain_mapping():
    report = _default_smart_dsp_replay_report(
        generated_at_utc="2026-06-26T17:40:00Z",
    )
    result = {
        row["scenario_id"]: row for row in report["results"]
    }["clean_tom_bleed_aware_gate"]
    params = deepcopy(result["selected_parameters"])
    for key in (
        "wing_gate_sidechain_type_osc_value",
        "wing_gate_sidechain_type_value_type",
        "wing_gate_sidechain_frequency_osc_value",
        "wing_gate_sidechain_frequency_value_type",
        "wing_gate_sidechain_q_osc_value",
        "wing_gate_sidechain_q_value_type",
        "wing_gate_sidechain_src_osc_value",
        "wing_gate_sidechain_src_value_type",
        "wing_gate_sidechain_tap_osc_value",
        "wing_gate_sidechain_tap_value_type",
        "wing_gate_sidechain_source",
        "wing_gate_sidechain_reference",
    ):
        params.pop(key, None)
    contract = deepcopy(result["selected_future_wing_readback_contract"])
    contract["gate_expander_readback"].pop(
        "wing_gate_sidechain_mapping_required",
        None,
    )
    selected = SimpleNamespace(
        method_family=result["actual_method_family"],
        parameters=params,
        future_wing_readback_contract=contract,
    )

    failures = _readback_contract_failures(selected, "tampered_gate_sidechain")

    failure_codes = {failure["code"] for failure in failures}
    assert (
        "tampered_gate_sidechain_selected_gate_sidechain_mapping_missing"
        in failure_codes
    )
    assert (
        "tampered_gate_sidechain_readback_contract_wing_gate_sidechain_mapping_not_required"
        in failure_codes
    )


def test_selected_gate_requires_all_wing_values_to_be_actionable():
    report = _default_smart_dsp_replay_report(
        generated_at_utc="2026-06-26T19:35:00Z",
    )
    result = {
        row["scenario_id"]: row for row in report["results"]
    }["clean_tom_bleed_aware_gate"]
    contract = deepcopy(result["selected_future_wing_readback_contract"])
    contract["wing_osc_applicability"]["actionable_parameter_keys"].remove(
        "wing_gate_sidechain_frequency_osc_value"
    )
    selected = SimpleNamespace(
        method_family=result["actual_method_family"],
        parameters=deepcopy(result["selected_parameters"]),
        future_wing_readback_contract=contract,
    )

    failures = _readback_contract_failures(
        selected,
        "tampered_gate_actionable_values",
    )

    failure_codes = {failure["code"] for failure in failures}
    assert (
        "tampered_gate_actionable_values_selected_gate_actionable_keys_missing"
        in failure_codes
    )


def test_selected_gate_requires_selected_write_key_contract():
    report = _default_smart_dsp_replay_report(
        generated_at_utc="2026-06-28T08:15:00Z",
    )
    result = {
        row["scenario_id"]: row for row in report["results"]
    }["clean_tom_bleed_aware_gate"]
    contract = deepcopy(result["selected_future_wing_readback_contract"])
    contract["gate_expander_readback"]["wing_gate_selected_write_keys"] = []
    selected = SimpleNamespace(
        method_family=result["actual_method_family"],
        parameters=deepcopy(result["selected_parameters"]),
        future_wing_readback_contract=contract,
    )

    failures = _readback_contract_failures(
        selected,
        "tampered_gate_write_keys",
    )

    failure_codes = {failure["code"] for failure in failures}
    assert (
        "tampered_gate_write_keys_selected_gate_wing_gate_selected_write_keys_mismatch"
        in failure_codes
    )


def test_selected_gate_requires_selected_parameter_rollback_operations():
    report = _default_smart_dsp_replay_report(
        generated_at_utc="2026-06-30T08:15:00Z",
    )
    result = {
        row["scenario_id"]: row for row in report["results"]
    }["clean_tom_bleed_aware_gate"]
    contract = deepcopy(result["selected_future_wing_readback_contract"])
    operations = contract["gate_expander_readback"][
        "wing_gate_selected_parameter_rollback_operations"
    ]
    operations[0]["rollback_path_suffix"] = "gate/not-a-fw305-path"
    selected = SimpleNamespace(
        method_family=result["actual_method_family"],
        parameters=deepcopy(result["selected_parameters"]),
        future_wing_readback_contract=contract,
    )

    failures = _readback_contract_failures(
        selected,
        "tampered_gate_rollback_ops",
    )

    failure_codes = {failure["code"] for failure in failures}
    assert (
        "tampered_gate_rollback_ops_readback_contract_"
        "wing_gate_selected_parameter_rollback_operations_"
        "rollback_path_suffix_mismatch"
    ) in failure_codes


def test_selected_pan_width_requires_all_wing_values_to_be_actionable():
    report = _default_smart_dsp_replay_report(
        generated_at_utc="2026-06-26T20:15:00Z",
    )
    result = {
        row["scenario_id"]: row for row in report["results"]
    }["pan_phase_mono_failure"]
    contract = deepcopy(result["selected_future_wing_readback_contract"])
    contract["wing_osc_applicability"]["actionable_parameter_keys"].remove(
        "wing_width_osc_value"
    )
    selected = SimpleNamespace(
        method_family=result["actual_method_family"],
        parameters=deepcopy(result["selected_parameters"]),
        future_wing_readback_contract=contract,
    )

    failures = _readback_contract_failures(
        selected,
        "tampered_pan_width_actionable_values",
    )

    failure_codes = {failure["code"] for failure in failures}
    assert (
        "tampered_pan_width_actionable_values_selected_pan_width_actionable_keys_missing"
        in failure_codes
    )


def test_selected_gate_hysteresis_advisory_must_be_review_only():
    report = _default_smart_dsp_replay_report(
        generated_at_utc="2026-06-26T03:50:00Z",
    )
    result = {
        row["scenario_id"]: row for row in report["results"]
    }["measured_tom_tail_risk_expander"]
    assert "post_gate_hysteresis_review_db" in result["selected_parameters"]

    contract = deepcopy(result["selected_future_wing_readback_contract"])
    contract["wing_osc_applicability"].pop("review_only_parameter_keys", None)
    gate_readback = contract["gate_expander_readback"]
    gate_readback.pop("wing_gate_hysteresis_review_only_no_osc_path", None)
    gate_readback.pop("wing_gate_hysteresis_review_only_not_apply_ready", None)
    gate_readback.pop("wing_gate_hysteresis_review_tolerance_db", None)
    gate_readback.pop("wing_gate_hysteresis_review_acceptance_min_db", None)
    gate_readback.pop("wing_gate_hysteresis_review_forbidden_path_suffixes", None)
    gate_readback.pop("wing_gate_hysteresis_review_operator_check", None)
    selected = SimpleNamespace(
        method_family=result["actual_method_family"],
        parameters=deepcopy(result["selected_parameters"]),
        future_wing_readback_contract=contract,
    )

    failures = _readback_contract_failures(
        selected,
        "tampered_gate_hysteresis_review_only",
    )

    failure_codes = {failure["code"] for failure in failures}
    assert (
        "tampered_gate_hysteresis_review_only_readback_contract_gate_hysteresis_review_only_key_missing"
        in failure_codes
    )
    assert (
        "tampered_gate_hysteresis_review_only_readback_contract_gate_hysteresis_no_osc_path_missing"
        in failure_codes
    )
    assert (
        "tampered_gate_hysteresis_review_only_readback_contract_gate_hysteresis_apply_ready_not_blocked"
        in failure_codes
    )
    assert (
        "tampered_gate_hysteresis_review_only_readback_contract_gate_hysteresis_tolerance_mismatch"
        in failure_codes
    )
    assert (
        "tampered_gate_hysteresis_review_only_readback_contract_gate_hysteresis_acceptance_min_mismatch"
        in failure_codes
    )
    assert (
        "tampered_gate_hysteresis_review_only_readback_contract_gate_hysteresis_forbidden_paths_mismatch"
        in failure_codes
    )
    assert (
        "tampered_gate_hysteresis_review_only_readback_contract_gate_hysteresis_operator_check_missing"
        in failure_codes
    )


def test_selected_mute_rejects_ambiguous_boolean_payload():
    params = _with_wing_native_parameter_mappings(
        {"proposed_muted": "false-ish"},
        method_family="mute",
        target_id="ch:5",
    )
    contract = _future_readback_contract("mute", "ch:5", params)
    selected = SimpleNamespace(
        method_family="mute",
        parameters=params,
        future_wing_readback_contract=contract,
    )

    failures = _readback_contract_failures(
        selected,
        "tampered_mute_ambiguous_value",
    )

    failure_codes = {failure["code"] for failure in failures}
    assert "wing_mute_osc_value" not in params
    assert contract["wing_osc_applicability"]["native_osc_supported"] is False
    assert (
        contract["wing_osc_applicability"]["native_osc_support_level"]
        == "no_actionable_wing_osc_parameter_value"
    )
    assert (
        "tampered_mute_ambiguous_value_readback_contract_wing_osc_selected_not_native"
        in failure_codes
    )
    assert (
        "tampered_mute_ambiguous_value_selected_no_actionable_wing_osc_parameter_value"
        in failure_codes
    )
    assert (
        "tampered_mute_ambiguous_value_selected_mute_wing_mapping_missing"
        in failure_codes
    )


def test_selected_dynamics_rejects_ambiguous_detector_payload():
    params = _with_wing_native_parameter_mappings(
        {"threshold_dbfs": -24.0, "detector": "smart_auto_blend"},
        method_family="dynamics",
        target_id="ch:5",
    )
    contract = _future_readback_contract("dynamics", "ch:5", params)
    selected = SimpleNamespace(
        method_family="dynamics",
        parameters=params,
        future_wing_readback_contract=contract,
    )

    failures = _readback_contract_failures(
        selected,
        "tampered_dynamics_ambiguous_detector",
    )

    failure_codes = {failure["code"] for failure in failures}
    assert "wing_compressor_detector_osc_value" not in params
    assert params["wing_compressor_detector_parse_blocker"] == (
        WING_COMPRESSOR_DETECTOR_BLOCKER
    )
    assert contract["wing_osc_applicability"]["native_osc_supported"] is False
    assert contract["wing_osc_applicability"]["unsupported_parameter_keys"] == [
        "detector"
    ]
    assert (
        "tampered_dynamics_ambiguous_detector_readback_contract_wing_osc_selected_not_native"
        in failure_codes
    )
    assert (
        "tampered_dynamics_ambiguous_detector_selected_dynamics_wing_detector_mapping_missing"
        in failure_codes
    )
    assert (
        "tampered_dynamics_ambiguous_detector_readback_contract_wing_detector_mapping_not_required"
        in failure_codes
    )


def test_selected_dynamics_requires_gain_reduction_targets_review_only_contract():
    params = _with_wing_native_parameter_mappings(
        {
            "threshold_dbfs": -24.0,
            "ratio": 2.6,
            "target_gain_reduction_db": 3.0,
            "max_gain_reduction_db": 4.2,
        },
        method_family="dynamics",
        target_id="ch:5",
    )
    contract = _future_readback_contract("dynamics", "ch:5", params)
    selected = SimpleNamespace(
        method_family="dynamics",
        parameters=params,
        future_wing_readback_contract=contract,
    )

    assert _readback_contract_failures(
        selected,
        "valid_dynamics_gain_reduction_review_only",
    ) == []
    assert contract["wing_osc_applicability"]["review_only_parameter_keys"] == [
        "max_gain_reduction_db",
        "target_gain_reduction_db",
    ]
    assert contract["wing_osc_applicability"][
        "review_only_parameter_reason_by_key"
    ] == {
        key: WING_DYNAMICS_REVIEW_ONLY_PARAMETER_REASONS[key]
        for key in ("max_gain_reduction_db", "target_gain_reduction_db")
    }
    dynamics_objective = contract["dynamics_readback"]
    assert (
        dynamics_objective["wing_dynamics_gain_reduction_objective_required"]
        is True
    )
    assert dynamics_objective[
        "wing_dynamics_gain_reduction_objective_parameter_keys"
    ] == ["max_gain_reduction_db", "target_gain_reduction_db"]
    assert dynamics_objective["wing_dynamics_gain_reduction_objective_values"] == {
        "max_gain_reduction_db": 4.2,
        "target_gain_reduction_db": 3.0,
    }
    assert dynamics_objective["wing_dynamics_gain_reduction_target_db"] == 3.0
    assert dynamics_objective["wing_dynamics_gain_reduction_max_db"] == 4.2
    assert dynamics_objective[
        "wing_dynamics_gain_reduction_objective_tolerance_db"
    ] == WING_DYNAMICS_GAIN_REDUCTION_OBJECTIVE_TOLERANCE_DB
    assert dynamics_objective["wing_dynamics_gain_reduction_acceptance_min_db"] == 2.0
    assert dynamics_objective["wing_dynamics_gain_reduction_acceptance_max_db"] == 4.0
    assert (
        dynamics_objective["wing_dynamics_gain_reduction_must_not_exceed_db"] == 4.2
    )
    assert (
        dynamics_objective["wing_dynamics_gain_reduction_meter_readback_path_suffix"]
        == "dyn/gr"
    )
    assert dynamics_objective[
        "wing_dynamics_gain_reduction_meter_readback_path_templates"
    ] == ["/ch/{channel}/dyn/gr"]
    assert dynamics_objective[
        "wing_dynamics_gain_reduction_meter_fallback_path_templates"
    ] == ["/ch/{channel}/dyn/$gr"]
    assert (
        dynamics_objective["wing_dynamics_gain_reduction_meter_value_type"]
        == "read_only_meter_db"
    )
    assert (
        dynamics_objective["wing_dynamics_gain_reduction_meter_is_not_apply_value"]
        is True
    )
    assert dynamics_objective[
        "wing_dynamics_gain_reduction_meter_path_source"
    ] == "wing_client_cached_state_and_operator_product_state"
    assert "dyn/gr" in contract["dynamics_readback"]["read_before_paths"]
    assert "dyn/gr" in contract["dynamics_readback"]["read_after_paths"]
    assert (
        dynamics_objective["wing_dynamics_gain_reduction_objective_is_review_only"]
        is True
    )
    assert (
        dynamics_objective["wing_dynamics_gain_reduction_objective_is_not_osc_write"]
        is True
    )
    assert (
        dynamics_objective[
            "wing_dynamics_gain_reduction_objective_no_direct_osc_write_path"
        ]
        is True
    )
    assert (
        dynamics_objective["wing_dynamics_gain_reduction_objective_apply_ready"]
        is False
    )
    assert (
        dynamics_objective[
            "wing_dynamics_gain_reduction_objective_requires_post_apply_meter_readback"
        ]
        is True
    )
    assert (
        dynamics_objective[
            "wing_dynamics_gain_reduction_objective_requires_operator_confirmation"
        ]
        is True
    )
    assert dynamics_objective[
        "wing_dynamics_gain_reduction_objective_live_promotion_status"
    ] == "blocked_until_actual_gain_reduction_meter_readback_and_operator_review"
    assert "actual WING compressor gain-reduction meter" in dynamics_objective[
        "wing_dynamics_gain_reduction_objective_operator_check"
    ]
    assert dynamics_objective[
        "wing_dynamics_gain_reduction_objective_direct_osc_write_keys_excluded"
    ] == ["max_gain_reduction_db", "target_gain_reduction_db"]

    tampered_contract = deepcopy(contract)
    tampered_contract["wing_osc_applicability"].pop("review_only_parameter_keys")
    tampered_contract["dynamics_readback"].pop(
        "wing_dynamics_review_only_no_direct_osc_path"
    )
    tampered_contract["dynamics_readback"].pop(
        "wing_dynamics_review_only_not_apply_ready"
    )
    tampered_contract["dynamics_readback"].pop("wing_dynamics_review_only_values")
    tampered_contract["dynamics_readback"].pop(
        "wing_dynamics_gain_reduction_objective_required"
    )
    tampered_contract["dynamics_readback"].pop(
        "wing_dynamics_gain_reduction_objective_values"
    )
    tampered_contract["dynamics_readback"].pop(
        "wing_dynamics_gain_reduction_objective_apply_ready"
    )
    tampered_selected = SimpleNamespace(
        method_family="dynamics",
        parameters=params,
        future_wing_readback_contract=tampered_contract,
    )

    failures = _readback_contract_failures(
        tampered_selected,
        "tampered_dynamics_gain_reduction_review_only",
    )

    failure_codes = {failure["code"] for failure in failures}
    assert (
        "tampered_dynamics_gain_reduction_review_only_readback_contract_dynamics_review_only_keys_missing"
        in failure_codes
    )
    assert (
        "tampered_dynamics_gain_reduction_review_only_readback_contract_dynamics_review_only_no_osc_path_missing"
        in failure_codes
    )
    assert (
        "tampered_dynamics_gain_reduction_review_only_readback_contract_dynamics_review_only_apply_ready_not_blocked"
        in failure_codes
    )
    assert (
        "tampered_dynamics_gain_reduction_review_only_readback_contract_dynamics_review_only_values_missing"
        in failure_codes
    )
    assert (
        "tampered_dynamics_gain_reduction_review_only_readback_contract_dynamics_gr_objective_missing"
        in failure_codes
    )
    assert (
        "tampered_dynamics_gain_reduction_review_only_readback_contract_dynamics_gr_objective_values_mismatch"
        in failure_codes
    )
    assert (
        "tampered_dynamics_gain_reduction_review_only_readback_contract_dynamics_gr_objective_apply_ready_not_blocked"
        in failure_codes
    )


def test_selected_live_dynamics_requires_conservative_guard_contract():
    params = _with_wing_native_parameter_mappings(
        {
            "threshold_dbfs": -24.0,
            "ratio": 2.5,
            "target_gain_reduction_db": 2.4,
            "makeup_gain_db": 0.0,
            "compression_mix_percent": 78.5,
            "live_dynamics_conservative_proposal": True,
            "live_dynamics_minimal_write_set_required": True,
            "live_dynamics_crossover_write_forbidden": True,
            "live_dynamics_crossover_review_only_values": {},
            "live_dynamics_sidechain_write_forbidden": True,
            "live_dynamics_sidechain_review_only_values": {},
            "live_dynamics_stepwise_threshold_review_required": True,
            "live_dynamics_max_threshold_step_db": (
                LIVE_DYNAMICS_MAX_THRESHOLD_STEP_DB
            ),
            "live_dynamics_max_target_gain_reduction_db": (
                LIVE_DYNAMICS_MAX_TARGET_GAIN_REDUCTION_DB
            ),
            "live_dynamics_no_makeup_gain_required": True,
            "live_dynamics_makeup_gain_db_limit": 0.0,
            "live_dynamics_gain_reduction_meter_readback_after_each_step_required": True,
            "live_dynamics_previous_comp_model_readback_required": True,
            "live_dynamics_global_on_mix_write_forbidden": True,
            "live_dynamics_on_mix_readback_required": True,
            "live_dynamics_compressor_enable_write_forbidden": True,
            "live_dynamics_compressor_mix_write_forbidden": True,
            "live_dynamics_parameter_write_forbidden": True,
            "live_dynamics_parameter_readback_required": True,
            "live_dynamics_parameter_apply_blocked_without_wing_proof": True,
            "live_dynamics_parameter_review_only_reason": (
                "live_dynamics_must_read_existing_compressor_state_and_operator_confirm_before_parameter_apply"
            ),
            "live_dynamics_requires_existing_compressor_enabled_readback": True,
            "live_dynamics_compressor_on_review_only_value": WING_COMPRESSOR_ON_MAX,
            "live_dynamics_compressor_mix_review_only_value": 78.5,
            "live_dynamics_global_on_mix_review_only_reason": (
                "live_dynamics_must_read_existing_compressor_on_and_mix_before_parameter_apply"
            ),
            "live_dynamics_operator_rehearsal_required": True,
            "live_dynamics_live_promotion_status": (
                "blocked_until_real_wing_readback_operator_rehearsal_and_rollback"
            ),
            "live_dynamics_live_apply_blocked_without_wing_proof": True,
        },
        method_family="dynamics",
        target_id="ch:11",
    )
    contract = _future_readback_contract("dynamics", "ch:11", params)
    selected = SimpleNamespace(
        method_family="dynamics",
        parameters=params,
        future_wing_readback_contract=contract,
    )

    assert _readback_contract_failures(
        selected,
        "valid_live_dynamics_guard",
    ) == []

    tampered_contract = deepcopy(contract)
    tampered_contract["dynamics_readback"][
        "live_dynamics_live_apply_blocked_without_wing_proof"
    ] = False
    tampered_selected = SimpleNamespace(
        method_family="dynamics",
        parameters=params,
        future_wing_readback_contract=tampered_contract,
    )

    failures = _readback_contract_failures(
        tampered_selected,
        "tampered_live_dynamics_guard",
    )

    failure_codes = {failure["code"] for failure in failures}
    assert (
        "tampered_live_dynamics_guard_readback_contract_live_dynamics_guard_live_dynamics_live_apply_blocked_without_wing_proof_mismatch"
        in failure_codes
    )

    tampered_params = dict(params)
    tampered_params["wing_compressor_on_osc_value"] = WING_COMPRESSOR_ON_MAX
    tampered_params["wing_compressor_mix_osc_value"] = 78.5
    tampered_selected = SimpleNamespace(
        method_family="dynamics",
        parameters=tampered_params,
        future_wing_readback_contract=contract,
    )

    failures = _readback_contract_failures(
        tampered_selected,
        "tampered_live_dynamics_global_on_mix_write",
    )

    failure_codes = {failure["code"] for failure in failures}
    assert (
        "tampered_live_dynamics_global_on_mix_write_readback_contract_live_dynamics_guard_global_on_mix_write_violation"
        in failure_codes
    )


def test_selected_soundcheck_dynamics_requires_bounded_guard_contract():
    report = _default_smart_dsp_replay_report(
        generated_at_utc="2026-06-27T00:45:00Z",
    )
    result = {
        row["scenario_id"]: row for row in report["results"]
    }["soundcheck_dynamic_vocal_compression"]
    contract = deepcopy(result["selected_future_wing_readback_contract"])
    contract["dynamics_readback"][
        "soundcheck_dynamics_live_apply_blocked_without_wing_proof"
    ] = False
    selected = SimpleNamespace(
        method_family=result["actual_method_family"],
        parameters=deepcopy(result["selected_parameters"]),
        future_wing_readback_contract=contract,
    )

    failures = _readback_contract_failures(
        selected,
        "tampered_soundcheck_dynamics_guard",
    )

    failure_codes = {failure["code"] for failure in failures}
    assert (
        "tampered_soundcheck_dynamics_guard_readback_contract_soundcheck_dynamics_guard_soundcheck_dynamics_live_apply_blocked_without_wing_proof_mismatch"
        in failure_codes
    )


def test_selected_soundcheck_gate_expander_requires_bounded_guard_contract():
    report = _default_smart_dsp_replay_report(
        generated_at_utc="2026-06-27T01:25:00Z",
    )
    result = {
        row["scenario_id"]: row for row in report["results"]
    }["measured_tom_tail_risk_expander"]
    contract = deepcopy(result["selected_future_wing_readback_contract"])
    contract["gate_expander_readback"][
        "soundcheck_gate_expander_live_apply_blocked_without_wing_proof"
    ] = False
    selected = SimpleNamespace(
        method_family=result["actual_method_family"],
        parameters=deepcopy(result["selected_parameters"]),
        future_wing_readback_contract=contract,
    )

    failures = _readback_contract_failures(
        selected,
        "tampered_soundcheck_gate_expander_guard",
    )

    failure_codes = {failure["code"] for failure in failures}
    assert (
        "tampered_soundcheck_gate_expander_guard_readback_contract_soundcheck_gate_expander_guard_soundcheck_gate_expander_live_apply_blocked_without_wing_proof_mismatch"
        in failure_codes
    )


def test_selected_fx_requires_wing_suffix_readback_contract():
    report = _default_smart_dsp_replay_report(
        generated_at_utc="2026-06-25T21:45:00Z",
    )
    result = {
        row["scenario_id"]: row for row in report["results"]
    }["soundcheck_fx_wash_muddy_vocal_cleanup"]
    contract = deepcopy(result["selected_future_wing_readback_contract"])
    contract["fx_readback"]["read_before_paths"] = ["fx/send", "fx/return"]
    contract["fx_readback"]["read_after_paths"] = ["fx/send", "fx/return"]
    contract["fx_readback"].pop("wing_fx_readback_paths_are_wing_fw_3_0_5_suffixes")
    selected = SimpleNamespace(
        method_family=result["actual_method_family"],
        parameters=result["selected_parameters"],
        future_wing_readback_contract=contract,
    )

    failures = _readback_contract_failures(selected, "tampered_fx_suffixes")

    failure_codes = {failure["code"] for failure in failures}
    assert "tampered_fx_suffixes_readback_contract_fx_pseudo_path_present" in (
        failure_codes
    )
    assert "tampered_fx_suffixes_readback_contract_fx_suffix_contract_missing" in (
        failure_codes
    )
    assert "tampered_fx_suffixes_readback_contract_fx_send_read_before_path_missing" in (
        failure_codes
    )


def test_fx_model_specific_candidate_requires_appendix_readback_contract():
    report = _default_smart_dsp_replay_report(
        generated_at_utc="2026-06-26T10:05:00Z",
    )
    result = {
        row["scenario_id"]: row for row in report["results"]
    }["soundcheck_long_fx_tail_decay_filter"]
    candidate_id = "shorten_fx_decay_filter"
    params = deepcopy(result["candidate_parameters"][candidate_id])
    contract = deepcopy(result["candidate_future_wing_readback_contracts"][candidate_id])
    fx_readback = contract["fx_readback"]
    for key in (
        "wing_fx_model_parameter_binding_candidates",
        "wing_fx_model_parameter_verified_suffixes",
        "wing_fx_model_parameter_verified_path_templates",
        "wing_fx_model_parameter_partial_coverage_models",
    ):
        fx_readback.pop(key, None)
    contract["wing_osc_applicability"].pop(
        "fx_model_specific_verified_path_templates",
        None,
    )
    selected = SimpleNamespace(
        method_family="fx",
        parameters=params,
        future_wing_readback_contract=contract,
    )

    failures = _readback_contract_failures(selected, "tampered_fx_appendix")

    failure_codes = {failure["code"] for failure in failures}
    assert (
        "tampered_fx_appendix_selected_readback_contract_fx_model_appendix_fields_missing"
        in failure_codes
    )
    assert (
        "tampered_fx_appendix_selected_readback_contract_fx_model_appendix_value_mismatch"
        in failure_codes
    )
    assert (
        "tampered_fx_appendix_selected_wing_osc_fx_model_appendix_field_mismatch"
        in failure_codes
    )


def test_candidate_readback_contract_guard_blocks_wing_authority_drift():
    report = _default_smart_dsp_replay_report(
        generated_at_utc="2026-06-26T10:25:00Z",
    )
    result = {
        row["scenario_id"]: row for row in report["results"]
    }["soundcheck_muddy_vocal"]
    candidate_id = result["actual_selected_strategy_id"]
    contract = deepcopy(result["candidate_future_wing_readback_contracts"][candidate_id])
    candidate = {
        "strategy_id": candidate_id,
        "method_family": contract["method_family"],
        "parameters": deepcopy(result["candidate_parameters"][candidate_id]),
        "future_wing_readback_contract": contract,
        "blockers": deepcopy(result["candidate_blockers"][candidate_id]),
    }
    assert (
        _candidate_readback_contract_failures(
            candidate=candidate,
            scenario_id="valid_candidate_contract",
            candidate_id=candidate_id,
        )
        == []
    )

    tampered = deepcopy(candidate)
    tampered_contract = deepcopy(contract)
    tampered_contract["live_write"] = True
    tampered_contract["allowed_live_write_scope_before_wing_proof"] = ["eq"]
    tampered_contract["wing_osc_applicability"]["firmware"] = "3.0.4"
    tampered_contract["wing_osc_applicability"]["live_authority_granted"] = True
    tampered["future_wing_readback_contract"] = tampered_contract

    failures = _candidate_readback_contract_failures(
        candidate=tampered,
        scenario_id="tampered_candidate_contract",
        candidate_id=candidate_id,
    )

    failure_codes = {failure["code"] for failure in failures}
    assert (
        "tampered_candidate_contract_candidate_readback_contract_live_write_mismatch"
        in failure_codes
    )
    assert (
        "tampered_candidate_contract_candidate_readback_contract_live_scope_not_empty"
        in failure_codes
    )
    assert (
        "tampered_candidate_contract_candidate_wing_osc_firmware_mismatch"
        in failure_codes
    )
    assert (
        "tampered_candidate_contract_candidate_wing_osc_live_authority_granted_mismatch"
        in failure_codes
    )


def test_non_native_candidate_contract_requires_blocker_and_reason():
    report = _default_smart_dsp_replay_report(
        generated_at_utc="2026-06-26T10:30:00Z",
    )
    result = {
        row["scenario_id"]: row for row in report["results"]
    }["soundcheck_long_fx_tail_decay_filter"]
    candidate_id = "shorten_fx_decay_filter"
    contract = deepcopy(result["candidate_future_wing_readback_contracts"][candidate_id])
    candidate = {
        "strategy_id": candidate_id,
        "method_family": contract["method_family"],
        "parameters": deepcopy(result["candidate_parameters"][candidate_id]),
        "future_wing_readback_contract": contract,
        "blockers": deepcopy(result["candidate_blockers"][candidate_id]),
    }
    assert (
        _candidate_readback_contract_failures(
            candidate=candidate,
            scenario_id="valid_non_native_candidate_contract",
            candidate_id=candidate_id,
        )
        == []
    )

    tampered = deepcopy(candidate)
    tampered["blockers"] = []
    tampered["future_wing_readback_contract"]["wing_osc_applicability"].pop(
        "unsupported_strategy_reason",
        None,
    )

    failures = _candidate_readback_contract_failures(
        candidate=tampered,
        scenario_id="tampered_non_native_candidate_contract",
        candidate_id=candidate_id,
    )

    failure_codes = {failure["code"] for failure in failures}
    assert (
        "tampered_non_native_candidate_contract_candidate_wing_osc_unsupported_reason_missing"
        in failure_codes
    )
    assert (
        "tampered_non_native_candidate_contract_candidate_wing_osc_blocker_missing"
        in failure_codes
    )


def test_non_native_fx_candidate_tradeoff_surfaces_wing_osc_mapping_blocker():
    report = _default_smart_dsp_replay_report(
        generated_at_utc="2026-06-26T10:35:00Z",
    )
    result = {
        row["scenario_id"]: row for row in report["results"]
    }["soundcheck_long_fx_tail_decay_filter"]
    tradeoffs = {row["strategy_id"]: row for row in result["alternative_tradeoffs"]}
    decay_tradeoff = tradeoffs["shorten_fx_decay_filter"]

    _assert_replay_tradeoff_wing_osc_summary(
        decay_tradeoff,
        native_supported=False,
        support_level="partial_model_specific_fx_mapping_required",
    )
    assert decay_tradeoff["wing_osc_unsupported_strategy_reason"] == (
        "fx_strategy_requires_wing_fx_slot_model_and_parameter_mapping_before_apply"
    )
    assert set(decay_tradeoff["wing_osc_unsupported_parameter_keys"]) >= {
        "decay_s",
        "pre_delay_ms",
        "return_hpf_hz",
        "return_lpf_hz",
        "delay_feedback_percent",
    }
    assert decay_tradeoff["wing_osc_candidate_apply_status"] == (
        "not_wing_applicable_until_mapping_or_operator_review"
    )


def test_default_scenarios_cover_required_goal_fixture_families():
    scenario_ids = {scenario.scenario_id for scenario in build_default_smart_dsp_replay_scenarios()}

    assert {
        "soundcheck_muddy_vocal",
        "soundcheck_rumble_vocal_hpf",
        "soundcheck_plosive_vocal_dynamic_hpf",
        "soundcheck_plosive_sibilant_vocal_de_esser_first",
        "soundcheck_plosive_vocal_deq_gate_slot_conflict_hpf",
        "soundcheck_plosive_vocal_eq_model_conflict_hold",
        "live_plosive_vocal_hold_before_hpf",
        "soundcheck_vocal_noise_floor_expander",
        "soundcheck_vocal_phrase_noise_before_de_esser_compressor",
        "soundcheck_quiet_noisy_vocal_gain_stage_review",
        "live_vocal_noise_floor_review_before_expander",
        "soundcheck_line_noise_hum_notch",
        "soundcheck_bass_di_line_noise_before_gain_up",
        "live_line_noise_physical_path_before_notch_or_dynamics",
        "live_vocal_plosive_line_noise_mic_review_before_hum",
        "soundcheck_dynamic_muddy_vocal",
        "live_dynamic_muddy_vocal_static_low_mid_priority",
        "live_muddy_vocal_hot_summed_bus_headroom_pullback",
        "soundcheck_dynamic_muddy_guitar_low_mid_bloom",
        "live_dynamic_muddy_guitar_static_low_mid_priority",
        "soundcheck_fx_wash_muddy_vocal_cleanup",
        "soundcheck_fx_wash_operator_owned_level_holds_review",
        "live_fx_wash_muddy_vocal_native_pullback_before_eq",
        "live_operator_owned_fx_holds_muddy_vocal",
        "soundcheck_dull_vocal_presence_lift",
        "soundcheck_dull_vocal_monitor_feedback_hold",
        "soundcheck_dull_vocal_elevated_feedback_path_hold",
        "soundcheck_dull_vocal_summed_bus_headroom_hold",
        "soundcheck_backing_vocal_role_aware_presence_lift",
        "live_dull_vocal_presence_hold",
        "live_vocal_feedback_gain_pullback",
        "live_feedback_low_headroom_gain_pullback",
        "soundcheck_feedback_blocks_presence_boost",
        "soundcheck_measured_feedback_ring_notch",
        "soundcheck_measured_feedback_low_headroom_pullback_before_notch",
        "soundcheck_harsh_sibilant_vocal_split_chain",
        "live_sibilant_vocal",
        "soundcheck_vocal_level_pressure_before_de_essing",
        "live_harsh_vocal_static_presence_cut",
        "live_operator_owned_eq_holds_harsh_vocal",
        "soundcheck_operator_owned_eq_holds_harsh_vocal",
        "live_feedback_risk_beats_sibilance_correction",
        "soundcheck_harsh_dynamic_vocal_envelope",
        "live_dynamic_harsh_vocal_ride_before_tone_or_compressor",
        "soundcheck_harsh_guitar_dynamic_presence_tame",
        "soundcheck_harsh_overhead_cymbal_dynamic_de_edge",
        "soundcheck_harsh_overhead_top_octave_high_cut",
        "live_harsh_overhead_cymbal_mic_level_review",
        "inferred_vocal_identity_requires_confirmation",
        "low_confidence_source_identity_blocks_eq",
        "unconfirmed_source_identity_blocks_eq_without_inference",
        "low_confidence_audio_evidence_holds_before_eq",
        "stale_live_audio_evidence_holds_before_eq",
        "auto_apply_request_holds_before_dsp",
        "unsupported_mode_holds_before_dsp",
        "routing_scene_snapshot_intent_holds_before_dsp",
        "live_master_output_intent_holds_before_channel_eq",
        "live_eq_intent_holds_before_channel_eq_or_gain_up_guard",
        "live_feedback_risk_beats_eq_intent_hold",
        "live_source_gain_up_intent_holds_before_channel_eq",
        "live_processing_change_intent_holds_before_channel_eq",
        "live_pan_width_intent_holds_before_channel_eq",
        "declared_vocal_audio_identity_conflict_holds",
        "soundcheck_dynamic_vocal_compression",
        "soundcheck_aux_playback_pse_la_dynamics",
        "soundcheck_aux_playback_operator_owned_pse_la_hold",
        "soundcheck_dynamic_vocal_dynamics_model_conflict_ride",
        "soundcheck_dynamic_vocal_active_dynamics_model_readback_ride",
        "soundcheck_master_true_peak_beats_source_vocal_compression",
        "soundcheck_master_true_peak_beats_source_headroom_pullback",
        "live_vocal_dynamics_ride_before_compression",
        "live_operator_owned_dynamics_manual_ride",
        "soundcheck_operator_owned_dynamics_manual_ride",
        "live_lead_guitar_ride_before_compression",
        "soundcheck_dynamic_vocal_level_plane_before_compression",
        "soundcheck_quiet_vocal_level_plane_gain_up",
        "soundcheck_quiet_vocal_operator_owned_gain_holds_review",
        "soundcheck_loud_vocal_operator_owned_gain_down_holds_review",
        "soundcheck_quiet_vocal_feedback_path_holds_gain_up",
        "live_quiet_vocal_source_gain_up_review_hold",
        "soundcheck_quiet_muddy_vocal_level_first",
        "soundcheck_hot_muddy_vocal_level_down_before_eq",
        "soundcheck_quiet_muddy_vocal_hot_bus_rebalance_before_eq",
        "soundcheck_quiet_vocal_hot_bus_rebalance",
        "live_loud_bass_level_plane_gain_down",
        "live_operator_owned_loud_bass_gain_hold",
        "soundcheck_hot_drum_bus_group_trim",
        "soundcheck_dynamic_vocal_bus_group_compression",
        "live_quiet_vocal_bus_group_gain_hold",
        "live_group_feedback_member_review_before_bus_dsp",
        "kick_bass_masking_low_headroom",
        "live_source_true_peak_headroom_pullback_before_eq",
        "soundcheck_general_masking_static_unmasking_eq",
        "soundcheck_vocal_intelligibility_dynamic_unmasking",
        "live_vocal_intelligibility_static_pocket",
        "soundcheck_vocal_masking_over_level_rebalance",
        "live_vocal_masking_over_level_rebalance",
        "soundcheck_kick_bass_dynamic_pocket",
        "soundcheck_kick_bass_hot_bus_rebalance_before_ducking",
        "soundcheck_kick_target_avoids_ducking_kick_from_bass",
        "soundcheck_kick_bass_arrangement_overlap_hold",
        "soundcheck_arrangement_overlap_pair_context_hold",
        "live_kick_bass_conservative_eq_split",
        "soundcheck_snare_transient_tamer",
        "live_snare_low_headroom_transient_hold",
        "noisy_tom_tail_risk",
        "measured_tom_tail_risk_expander",
        "clean_tom_bleed_aware_gate",
        "soundcheck_operator_owned_gate_slot_hold",
        "low_margin_tom_clean_hit_evidence_hold",
        "extreme_noisy_tom_manual_mute_review",
        "soundcheck_extreme_noisy_tom_operator_owned_mute_holds_review",
        "live_noisy_tom_hold_before_gate_or_expander",
        "live_extreme_noisy_tom_hold_before_manual_mute",
        "wet_fx_monitor_risk",
        "soundcheck_long_fx_tail_decay_filter",
        "soundcheck_delay_smear_fx_ducking_cleanup",
        "soundcheck_fx_monitor_risk_pullback_before_time_edits",
        "soundcheck_source_fx_return_headroom_send_only_until_slot_bound",
        "soundcheck_source_fx_return_headroom_bound_slot_uses_send_and_return",
        "soundcheck_aux_playback_fx_send_pullback",
        "soundcheck_fx_return_headroom_before_delay_ducking",
        "live_delay_smear_fx_pullback_before_ducking",
        "pan_phase_mono_failure",
        "soundcheck_matrix_phase_mono_width_reduction",
        "soundcheck_phase_safe_pan_delta",
        "soundcheck_measured_stereo_balance_pan_target",
        "live_measured_stereo_balance_pan_review_hold",
        "live_matrix_pan_review_hold_output_feed_identity",
        "soundcheck_phase_mono_beats_measured_pan_target",
        "live_pan_change_holds_for_operator_review",
        "soundcheck_phase_mono_risk_width_reduction",
        "live_phase_mono_risk_hold",
        "live_matrix_phase_mono_hold_output_feed_identity",
        "soundcheck_phase_mono_risk_beats_harshness_eq",
        "live_phase_mono_risk_beats_harshness_eq",
        "master_true_peak_overload",
        "live_master_true_peak_overload_native_pullback",
        "soundcheck_operator_owned_master_output_gain_holds_true_peak_pullback",
        "live_operator_owned_master_output_gain_holds_true_peak_pullback",
        "master_low_headroom_trim_before_tone_or_compression",
        "master_reference_feed_holds_processing",
        "reference_capture_channel_feed_holds_source_dsp",
        "master_low_end_broad_eq_cleanup",
        "master_presence_broad_eq_cleanup",
        "master_static_loudness_trim_before_bus_compression",
        "master_static_loudness_trim_before_low_end_recheck",
        "master_loud_dynamic_trim_before_bus_compression",
        "master_quiet_loudness_lift_before_hold_or_processing",
        "soundcheck_operator_owned_quiet_master_output_gain_holds_lift",
        "master_quiet_loudness_feedback_hold_before_lift",
        "live_master_presence_holds_before_master_eq",
        "master_dynamic_range_bus_compression",
        "live_master_dynamics_holds_before_bus_compression",
        "stale_context_blocks_strategy",
        "live_authority_context_blocks_strategy",
        "missing_readback_context_blocks_strategy",
        "missing_rollback_context_blocks_strategy",
    } <= scenario_ids


def test_replay_evaluation_reports_selection_mismatch_without_authority():
    scenario = build_default_smart_dsp_replay_scenarios()[0]
    tampered = SmartDSPReplayScenario(
        scenario_id="tampered_muddy_vocal",
        description=scenario.description,
        analysis=scenario.analysis,
        expected_status=scenario.expected_status,
        expected_selected_strategy_id="high_pass_cleanup",
        expected_method_family=scenario.expected_method_family,
        min_candidate_count=scenario.min_candidate_count,
    )

    report = evaluate_smart_dsp_replay_scenarios(
        [tampered],
        generated_at_utc="2026-06-24T20:31:00Z",
    )

    assert report["status"] == "failed"
    assert report["scenario_count"] == 1
    assert report["passed_scenario_count"] == 0
    assert report["failed_scenario_count"] == 1
    failure_codes = {failure["code"] for failure in report["failed_checks"]}
    assert "tampered_muddy_vocal_selected_strategy_mismatch" in failure_codes
    assert report["live_authority_granted"] is False
    assert report["live_apply_allowed"] is False
    assert report["queue_import_allowed"] is False


def test_replay_evaluation_writes_json_and_markdown_artifacts(tmp_path):
    report = _default_smart_dsp_replay_report(
        generated_at_utc="2026-06-24T20:32:00Z",
    )
    paths = write_smart_dsp_replay_evaluation_artifacts(
        report=report,
        output_dir=tmp_path / "smart_dsp_replay",
    )

    summary = json.loads(Path(paths["summary_json"]).read_text(encoding="utf-8"))
    markdown = Path(paths["summary_markdown"]).read_text(encoding="utf-8")

    assert summary == report
    assert summary["status"] == "passed"
    assert "# Smart DSP Replay Evaluation" in markdown
    assert "- status: `passed`" in markdown
    assert "- selected_review_only_wing_osc_parameter_count: `13`" in markdown
    assert "- selected_review_only_wing_osc_by_family: `" in markdown
    assert "'dynamics': 8" in markdown
    assert "'expander': 4" in markdown
    assert "'gate': 1" in markdown
    assert "soundcheck_muddy_vocal" in markdown
    assert "live_authority_granted: `False`" in markdown


def test_replay_evaluation_markdown_lists_failures_for_failed_report():
    scenario = build_default_smart_dsp_replay_scenarios()[0]
    tampered = SmartDSPReplayScenario(
        scenario_id="tampered_score_floor",
        description=scenario.description,
        analysis=scenario.analysis,
        expected_status=scenario.expected_status,
        expected_selected_strategy_id=scenario.expected_selected_strategy_id,
        expected_method_family=scenario.expected_method_family,
        min_candidate_count=scenario.min_candidate_count,
        min_selected_total_score=1.1,
    )
    report = evaluate_smart_dsp_replay_scenarios([tampered])
    markdown = render_smart_dsp_replay_evaluation_markdown(report)

    assert report["status"] == "failed"
    assert "tampered_score_floor_selected_total_score_low" in markdown
    assert "smart_dsp_replay_pass_rate_below_threshold" in markdown


def test_replay_evaluation_cli_writes_default_artifacts(tmp_path):
    exit_code = evaluate_smart_dsp_replay_main(
        [
            "--artifact-root",
            str(tmp_path),
            "--run-id",
            "20260624T203300Z_smart_dsp_replay",
            "--generated-at-utc",
            "2026-06-24T20:33:00Z",
        ]
    )

    summary_path = tmp_path / "20260624T203300Z_smart_dsp_replay" / "summary.json"
    markdown_path = tmp_path / "20260624T203300Z_smart_dsp_replay" / "summary.md"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    markdown = markdown_path.read_text(encoding="utf-8")

    assert exit_code == 0
    assert summary["status"] == "passed"
    assert summary["live_authority_granted"] is False
    assert markdown.startswith("# Smart DSP Replay Evaluation")
