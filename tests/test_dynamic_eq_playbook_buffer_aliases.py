from dynamic_eq_playbook import (
    DynamicEQSafetyConfig,
    build_dynamic_eq_playbook_entry,
    parse_dynamic_eq_prediction,
)


def _payload_with_ramp_alias(alias_name: str, ramp_ms: float):
    bands = []
    for index, freq in enumerate([120, 250, 500, 1200, 3500, 9000], start=1):
        band = {
            "band_index": index,
            "center_hz": freq,
            "q": 1.2,
            "peq_gain_db": -1.0,
            "threshold_db": -24.0,
            "ratio": 2.0,
            "attack_ms": 12.0,
            "release_ms": 120.0,
            "dynamic_gain_range_db": 3.0,
            "makeup_gain_db": 0.5,
            "label": f"band_{index}",
        }
        if index == 1:
            band[alias_name] = ramp_ms
        bands.append(band)
    return {
        "channel_id": 12,
        "source_model": "Diff-DEQ",
        "model_version": "research_import_v0",
        "instrument_role": "lead_vocal",
        "sample_rate": 48000,
        "confidence": 0.82,
        "bands": bands,
    }


def test_dynamic_eq_ramp_alias_is_preserved_in_shadow_playbook_entry():
    prediction = parse_dynamic_eq_prediction(_payload_with_ramp_alias("ramp_ms", 30.0))
    entry = build_dynamic_eq_playbook_entry(
        prediction,
        config=DynamicEQSafetyConfig(buffer_size_samples=256),
    )

    assert entry.safety_status == "shadow_only"
    assert entry.dry_run is True
    assert entry.validation["metadata"]["live_write_allowed"] is False
    assert entry.validation["metadata"]["buffer_ms"] == 5.333
    assert entry.suggested_params["bands"][0]["parameter_ramp_ms"] == 30.0


def test_dynamic_eq_smoothing_alias_below_buffer_guard_is_blocked():
    prediction = parse_dynamic_eq_prediction(_payload_with_ramp_alias("smoothing_ms", 10.0))
    entry = build_dynamic_eq_playbook_entry(
        prediction,
        config=DynamicEQSafetyConfig(buffer_size_samples=256),
    )

    assert entry.safety_status == "blocked_shadow_only"
    assert entry.dry_run is True
    assert entry.validation["metadata"]["live_write_allowed"] is False
    assert {
        issue["code"] for issue in entry.validation["issues"]
    } >= {"parameter_ramp_too_fast_for_buffer"}
