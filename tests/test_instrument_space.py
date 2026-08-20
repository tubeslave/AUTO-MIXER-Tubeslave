from pathlib import Path

from backend.instrument_space import InstrumentSpaceHistory, build_instrument_space


def test_instrument_space_builds_read_only_projection_without_guessing_frequency():
    scene = build_instrument_space(
        [
            {
                "id": 1,
                "name": "KICK",
                "role": "Kick",
                "group": "DRUMS",
                "status": "active",
                "level_db": -18.0,
                "control_state": {"pan": 0.5},
                "ai_recognition": {
                    "label": "Kick",
                    "confidence": 99,
                    "source": "cached_mixer_name",
                },
                "read_coverage": {"status": "partial"},
            }
        ],
        generated_at="2026-08-04T12:00:00Z",
    )

    assert scene["read_only"] is True
    assert scene["live_write"] is False
    assert scene["can_apply_to_console"] is False
    assert scene["physical_localization"] is False
    assert scene["generated_at"] == "2026-08-04T12:00:00Z"
    assert scene["decision_support"]["read_only"] is True
    assert scene["decision_support"]["live_authority_granted"] is False
    assert scene["summary"] == {
        "object_count": 1,
        "plottable_object_count": 1,
        "complete_object_count": 0,
        "missing_pan_count": 0,
        "missing_frequency_count": 1,
        "missing_level_count": 0,
    }
    [kick] = scene["objects"]
    assert kick["coordinates"] == {
        "x_pan": 0.0,
        "y_frequency": None,
        "z_level": 0.7,
    }
    assert kick["evidence"]["frequency"]["available"] is False
    assert kick["evidence"]["frequency"]["value"] is None


def test_instrument_space_uses_measured_centroid_and_clamps_display_coordinates():
    scene = build_instrument_space(
        [
            {
                "id": 2,
                "name": "OH R",
                "level_db": 4.0,
                "spectral": {"centroid_hz": 20_000.0},
                "control_state": {"pan": 100.0},
            }
        ]
    )

    [overhead] = scene["objects"]
    assert overhead["coordinates"] == {
        "x_pan": 1.0,
        "y_frequency": 1.0,
        "z_level": 1.0,
    }
    assert overhead["evidence"]["frequency"]["value"] == 20_000.0
    assert scene["summary"]["complete_object_count"] == 1


def test_instrument_space_preserves_spectral_cache_freshness_provenance():
    scene = build_instrument_space(
        [
            {
                "id": 4,
                "name": "GTR",
                "level_db": -24.0,
                "spectral_centroid_hz": 1875.0,
                "spectral": {
                    "available": True,
                    "centroid_hz": 1875.0,
                    "source": "audio_capture_spectral_cache",
                    "age_sec": 0.312,
                    "freshness_status": "fresh",
                },
                "control_state": {"pan": -25.0},
            }
        ]
    )

    frequency = scene["objects"][0]["evidence"]["frequency"]
    assert frequency["source"] == "audio_capture_spectral_cache"
    assert frequency["age_sec"] == 0.312
    assert frequency["freshness_status"] == "fresh"


def test_instrument_space_history_supplies_runtime_drift_and_resets_on_source_change():
    history = InstrumentSpaceHistory(max_snapshots=3)
    channel = {
        "id": 1,
        "name": "Vocal",
        "role": "Lead Vocal",
        "group": "VOCALS",
        "status": "active",
        "level_db": -30.0,
        "control_state": {"pan": 0.5},
        "spectral": {
            "available": True,
            "centroid_hz": 1_000.0,
            "source": "audio_capture_spectral_cache",
            "age_sec": 0.1,
            "freshness_status": "fresh",
        },
        "ai_recognition": {
            "label": "Lead Vocal",
            "confidence": 99,
            "source": "cached_mixer_name",
        },
    }

    first = build_instrument_space(
        [channel],
        generated_at="2026-08-04T12:00:00Z",
        source_context={"identity": "wing|rack-a"},
        history=history,
    )
    assert first["history"]["status"] == "baseline_required"
    assert first["decision_support"]["drift"]["status"] == "baseline_required"

    moved_channel = dict(channel)
    moved_channel["level_db"] = -10.0
    moved_channel["control_state"] = {"pan": 0.8}
    second = build_instrument_space(
        [moved_channel],
        generated_at="2026-08-04T12:00:02Z",
        source_context={"identity": "wing|rack-a"},
        history=history,
    )
    assert second["history"]["status"] == "tracking"
    assert second["history"]["snapshot_count"] == 2
    assert second["history"]["baseline_snapshot_id"] == first["history"]["current_snapshot_id"]
    assert second["decision_support"]["drift"]["status"] == "review_required"
    assert second["decision_support"]["drift"]["changes"][0]["changed_axes"] == [
        "x_pan",
        "z_level",
    ]

    reset = build_instrument_space(
        [moved_channel],
        generated_at="2026-08-04T12:00:04Z",
        source_context={"identity": "dlive|surface-a"},
        history=history,
    )
    assert reset["history"]["source_changed"] is True
    assert reset["history"]["snapshot_count"] == 1
    assert reset["decision_support"]["drift"]["status"] == "baseline_required"


def test_instrument_space_module_has_no_transport_or_actuation_dependencies():
    source = Path("backend/instrument_space.py").read_text(encoding="utf-8")
    forbidden = (
        "wing_client",
        "dlive_client",
        "mixer_client_base",
        "correction_action_executor",
        "operator_proposal_queue",
        "live_apply(",
        ".send(",
    )
    for token in forbidden:
        assert token not in source
