from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path
from time import perf_counter

from backend.instrument_space import build_instrument_space
from backend.instrument_space_decision_support import (
    annotate_operator_proposal_with_instrument_space,
    build_instrument_space_decision_support,
)
from backend.operator_recommendation_bridge import build_soundcheck_operator_proposals


def _channel(channel_id, name, *, pan, centroid_hz, level_db, confidence=90):
    return {
        "id": channel_id,
        "name": name,
        "role": name,
        "group": "BAND",
        "status": "active",
        "level_db": level_db,
        "control_state": {"pan": pan},
        "spectral_centroid_hz": centroid_hz,
        "spectral": {
            "available": True,
            "centroid_hz": centroid_hz,
            "source": "audio_capture_spectral_cache",
            "freshness_status": "fresh",
            "age_sec": 0.2,
        },
        "ai_recognition": {
            "label": name,
            "confidence": confidence,
            "source": "cached_mixer_name",
        },
        "read_coverage": {"status": "partial"},
    }


def _now():
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


def test_decision_support_detects_competing_pair_and_routes_existing_analyzers():
    scene = build_instrument_space(
        [
            _channel(1, "Lead Vocal", pan=0.5, centroid_hz=1_000, level_db=-18),
            _channel(2, "Guitar", pan=0.52, centroid_hz=1_100, level_db=-20),
            _channel(3, "OH R", pan=0.95, centroid_hz=6_000, level_db=-32),
        ]
    )

    support = scene["decision_support"]
    assert support["schema_version"] == "automixer.instrument_space_decision_support.v1"
    assert support["status"] == "review_required"
    assert support["read_only"] is True
    assert support["live_write"] is False
    assert support["queue_import_allowed"] is False
    assert support["can_apply_to_console"] is False
    assert support["live_authority_granted"] is False

    relation = support["relations"][0]
    assert relation["channel_ids"] == [1, 2]
    assert relation["status"] == "review_required"
    assert relation["score"] > 0.8
    assert relation["recommended_next_analyzers"] == [
        "bus_group_balance_proposal",
        "pan_spatial_proposal",
        "frequency_separation_proposal",
        "cross_adaptive_eq",
    ]
    assert relation["can_apply_to_console"] is False
    assert relation["evidence_provenance"]["channels"][0]["pan_source"] == "mixer_state_cache"
    assert relation["evidence_provenance"]["channels"][0]["frequency_source"] == "audio_capture_spectral_cache"
    assert support["proposal_annotation_index"]["1"][0]["relation_id"] == relation["id"]
    assert support["proposal_annotation_index"]["1"][0]["measurements"] == relation["measurements"]


def test_decision_support_keeps_stale_spectrum_out_of_spectral_relation():
    channels = [
        _channel(1, "Vocal", pan=0.5, centroid_hz=1_000, level_db=-18),
        _channel(2, "Keys", pan=0.5, centroid_hz=1_020, level_db=-19),
    ]
    channels[1]["spectral"]["freshness_status"] = "stale"
    scene = build_instrument_space(channels)

    relation = scene["decision_support"]["relations"][0]
    assert relation["measurements"]["centroid_distance_octaves"] is None
    assert relation["components"]["spectral_proximity"] is None
    assert "fresh_spectral_evidence_missing" in relation["blockers"]
    assert "frequency_separation_proposal" not in relation["recommended_next_analyzers"]
    assert "cross_adaptive_eq" not in relation["recommended_next_analyzers"]


def test_decision_support_reports_drift_only_with_explicit_safe_baseline():
    previous = build_instrument_space(
        [_channel(1, "Guitar", pan=0.3, centroid_hz=1_200, level_db=-28)]
    )
    current = build_instrument_space(
        [_channel(1, "Guitar", pan=0.8, centroid_hz=2_400, level_db=-16)]
    )

    baseline_missing = current["decision_support"]["drift"]
    assert baseline_missing["status"] == "baseline_required"

    drift = build_instrument_space_decision_support(
        current,
        previous_scene=previous,
    )["drift"]
    assert drift["status"] == "review_required"
    assert drift["changes"][0]["channel_id"] == 1
    assert set(drift["changes"][0]["changed_axes"]) == {
        "x_pan",
        "y_frequency",
        "z_level",
    }
    assert drift["live_authority_granted"] is False


def test_proposal_annotation_preserves_values_and_rejects_unsafe_support():
    scene = build_instrument_space(
        [
            _channel(1, "Vocal", pan=0.5, centroid_hz=1_000, level_db=-18),
            _channel(2, "Guitar", pan=0.5, centroid_hz=1_050, level_db=-19),
        ],
        generated_at=_now(),
    )
    proposal = {
        "id": "p1",
        "channel": 1,
        "kind": "fader",
        "requested_change": {
            "value_type": "fader",
            "channel": 1,
            "old_value": -12.0,
            "value": -14.0,
            "rollback_value": -12.0,
        },
    }
    original = deepcopy(proposal)

    annotated = annotate_operator_proposal_with_instrument_space(
        proposal,
        scene["decision_support"],
    )
    assert annotated["requested_change"] == original["requested_change"]
    assert proposal == original
    assert annotated["instrument_space_review"]["status"] == "review_required"
    assert annotated["instrument_space_review"]["proposal_value_changed"] is False
    assert annotated["instrument_space_review"]["transaction_authority_changed"] is False
    assert annotated["instrument_space_review"]["source_age_sec"] is not None
    assert annotated["instrument_space_review"]["relations"][0]["measurements"]["level_delta_db"] == 1.0

    unsafe = deepcopy(scene["decision_support"])
    unsafe["live_write"] = True
    rejected = annotate_operator_proposal_with_instrument_space(proposal, unsafe)
    assert rejected["requested_change"] == original["requested_change"]
    assert rejected["instrument_space_review"]["accepted"] is False
    assert rejected["instrument_space_review"]["status"] == "rejected"


def test_soundcheck_bridge_adds_evidence_without_changing_correction_or_transaction():
    scene = build_instrument_space(
        [
            _channel(1, "Vocal", pan=0.5, centroid_hz=1_000, level_db=-18),
            _channel(2, "Guitar", pan=0.5, centroid_hz=1_050, level_db=-19),
        ],
        generated_at=_now(),
    )
    bundle = {
        "schema_version": "test",
        "instrument_space_decision_support": scene["decision_support"],
        "proposals": [
            {
                "id": "fader-down",
                "action_type": "set_fader_db",
                "target": {"mixer_channel": 1},
                "current_state": {"fader_db": -12.0},
                "requested_state": {"fader_db": -14.0},
                "mode": "approval_required",
            }
        ],
    }

    [proposal] = build_soundcheck_operator_proposals(bundle)
    assert proposal["requested_change"]["value"] == -14.0
    assert proposal["requested_change"]["rollback_value"] == -12.0
    assert proposal["instrument_space_review"]["status"] == "review_required"
    assert proposal["instrument_space_review"]["live_apply_allowed"] is False


def test_proposal_annotation_rejects_stale_decision_evidence_without_mutation():
    scene = build_instrument_space(
        [
            _channel(1, "Vocal", pan=0.5, centroid_hz=1_000, level_db=-18),
            _channel(2, "Guitar", pan=0.5, centroid_hz=1_050, level_db=-19),
        ],
        generated_at="2020-01-01T00:00:00Z",
    )
    proposal = {
        "id": "stale",
        "channel": 1,
        "requested_change": {"value_type": "fader", "value": -14.0},
    }

    annotated = annotate_operator_proposal_with_instrument_space(
        proposal,
        scene["decision_support"],
    )

    assert annotated["requested_change"] == proposal["requested_change"]
    assert annotated["instrument_space_review"]["accepted"] is False
    assert annotated["instrument_space_review"]["status"] == "rejected"
    assert annotated["instrument_space_review"]["reason"] == "instrument_space_decision_support_stale_or_undated"


def test_decision_support_64_channel_runtime_and_output_are_bounded():
    channels = [
        _channel(
            channel_id,
            f"Ch {channel_id}",
            pan=(channel_id % 17) / 16,
            centroid_hz=600 + (channel_id % 12) * 80,
            level_db=-18 - (channel_id % 6),
        )
        for channel_id in range(1, 65)
    ]

    started = perf_counter()
    scene = build_instrument_space(channels, generated_at=_now())
    elapsed = perf_counter() - started

    summary = scene["decision_support"]["summary"]
    assert summary["relation_count"] == 2016
    assert summary["emitted_relation_count"] <= 128
    assert summary["relations_truncated"] is True
    assert elapsed < 0.25


def test_decision_support_module_has_no_transport_queue_or_action_dependencies():
    source = Path("backend/instrument_space_decision_support.py").read_text(encoding="utf-8")
    forbidden = (
        "wing_client",
        "dlive_client",
        "mixer_client_base",
        "operator_proposal_queue",
        "correction_action_executor",
        "TypedCorrectionAction",
        "live_apply(",
        ".send(",
    )
    for token in forbidden:
        assert token not in source
