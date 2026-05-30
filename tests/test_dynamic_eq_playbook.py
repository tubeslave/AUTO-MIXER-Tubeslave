from pathlib import Path

from dynamic_eq_playbook import (
    DynamicEQSafetyConfig,
    build_dynamic_eq_playbook,
    build_dynamic_eq_playbook_entry,
    parse_dynamic_eq_prediction,
    validate_dynamic_eq_prediction,
)


def _valid_diff_deq_payload(**overrides):
    bands = []
    for index, freq in enumerate([120, 250, 500, 1200, 3500, 9000], start=1):
        bands.append(
            {
                "band_index": index,
                "center_hz": freq,
                "q": 1.2,
                "peq_gain_db": -1.5 if index in {2, 3} else 1.0,
                "threshold_db": -24.0,
                "ratio": 2.0,
                "attack_ms": 12.0,
                "release_ms": 120.0,
                "dynamic_gain_range_db": 3.0,
                "makeup_gain_db": 0.5,
                "label": f"band_{index}",
            }
        )
    payload = {
        "channel_id": 12,
        "source_model": "Diff-DEQ",
        "model_version": "research_import_v0",
        "instrument_role": "lead_vocal",
        "sample_rate": 24000,
        "window_start_ms": 0,
        "window_duration_ms": 3000,
        "confidence": 0.82,
        "bands": bands,
        "source_metadata": {"paper": "EUSIPCO 2025"},
    }
    payload.update(overrides)
    return payload


def test_parse_dynamic_eq_prediction_accepts_diff_deq_style_payload():
    prediction = parse_dynamic_eq_prediction(_valid_diff_deq_payload())

    assert prediction.channel_id == 12
    assert prediction.source_model == "Diff-DEQ"
    assert prediction.instrument_role == "lead_vocal"
    assert prediction.sample_rate == 24000
    assert len(prediction.bands) == 6
    assert prediction.bands[0].center_hz == 120
    assert prediction.bands[0].ratio == 2.0


def test_valid_dynamic_eq_prediction_becomes_shadow_only_playbook_entry():
    prediction = parse_dynamic_eq_prediction(_valid_diff_deq_payload())
    validation = validate_dynamic_eq_prediction(prediction)
    entry = build_dynamic_eq_playbook_entry(prediction)

    assert validation.passed is True
    assert entry.safety_status == "shadow_only"
    assert entry.dry_run is True
    assert entry.action_type == "dynamic_eq_playbook"
    assert entry.suggested_params["bands"][0]["center_hz"] == 120


def test_low_confidence_prediction_requires_review_not_live_action():
    prediction = parse_dynamic_eq_prediction(_valid_diff_deq_payload(confidence=0.4))
    entry = build_dynamic_eq_playbook_entry(prediction)

    assert entry.safety_status == "review_shadow_only"
    assert any(issue["code"] == "low_model_confidence" for issue in entry.validation["issues"])
    assert entry.dry_run is True


def test_unsafe_dynamic_eq_values_are_blocked():
    payload = _valid_diff_deq_payload()
    payload["bands"][0]["peq_gain_db"] = 12.0
    payload["bands"][1]["q"] = 14.0
    payload["bands"][2]["ratio"] = 30.0
    payload["bands"][3]["makeup_gain_db"] = 9.0
    prediction = parse_dynamic_eq_prediction(payload)

    result = validate_dynamic_eq_prediction(prediction, config=DynamicEQSafetyConfig(max_abs_eq_gain_db=6.0))
    entry = build_dynamic_eq_playbook_entry(prediction)

    assert result.passed is False
    assert entry.safety_status == "blocked_shadow_only"
    assert {
        issue.code for issue in result.issues
    } >= {"eq_gain_limit_exceeded", "eq_q_limit_exceeded", "ratio_out_of_range", "makeup_gain_limit_exceeded"}


def test_dynamic_eq_playbook_summarizes_multiple_predictions():
    valid = _valid_diff_deq_payload(channel_id=1)
    unsafe = _valid_diff_deq_payload(channel_id=2)
    unsafe["bands"][0]["threshold_db"] = 10.0

    playbook = build_dynamic_eq_playbook([valid, unsafe])

    assert playbook["type"] == "dynamic_eq_playbook"
    assert playbook["dry_run"] is True
    assert playbook["live_write_allowed"] is False
    assert playbook["entry_count"] == 2
    assert playbook["blocked_count"] == 1


def test_dynamic_eq_playbook_registry_entry_stays_shadow_only():
    from backend.processing_registry import get_processing_method

    method = get_processing_method("diff_deq_playbook_bridge")

    assert method is not None
    assert method.status == "shadow_only"
    assert method.can_write_live is False
    assert "lab_dsp" in method.pipeline_profiles
    assert "shadow_quality" in method.pipeline_profiles


def test_dynamic_eq_playbook_module_is_transport_free():
    source = Path("backend/dynamic_eq_playbook.py").read_text()

    forbidden_fragments = [
        "WingClient",
        "mixer_client",
        "pythonosc",
        "python-osc",
        "set_eq",
        "set_compressor",
        "set_fader",
        "set_gain",
        ".send(",
    ]
    for fragment in forbidden_fragments:
        assert fragment not in source
