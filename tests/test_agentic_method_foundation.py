import math

import numpy as np

from backend.agentic_method_foundation import (
    AgenticFoundationConfig,
    ChannelFeatureSnapshot,
    assert_no_live_transport_imports,
    build_agentic_recommendations,
    build_inter_channel_context,
    build_offline_teacher_example,
    build_offline_teacher_examples_from_channel_inventory,
    extract_channel_features,
    safety_review_recommendations,
)
from backend.context_aware_gate import (
    ContextAwareGateAgent,
    assert_no_live_transport_imports as context_gate_has_no_live_transport_imports,
    extract_bleed_telemetry,
    recommendation_to_decision_trace,
)
from backend.operator_product_state import build_channel_intelligence
from backend.processing_proposal_adapters import adapt_agentic_foundation_gain_fader_recommendations


def _tone(freq_hz=440.0, *, amplitude=0.2, seconds=0.25, sample_rate=48000):
    t = np.arange(int(sample_rate * seconds), dtype=float) / sample_rate
    return amplitude * np.sin(2.0 * np.pi * freq_hz * t)


def test_feature_extractor_builds_level_spectral_dynamics_and_bleed_metrics():
    signal = _tone(120.0, amplitude=0.24) + _tone(4200.0, amplitude=0.04)
    peer = signal * 0.8 + _tone(900.0, amplitude=0.01)

    snapshot = extract_channel_features(
        signal,
        channel_id=7,
        role="Snare",
        group="DRUMS",
        peer_samples=[peer],
    )

    assert snapshot.channel_id == 7
    assert snapshot.rms_db is not None and snapshot.rms_db > -30.0
    assert snapshot.peak_db is not None and snapshot.peak_db < 0.0
    assert snapshot.true_peak_db == snapshot.peak_db
    assert snapshot.crest_factor_db is not None and snapshot.crest_factor_db > 0.0
    assert snapshot.headroom_db is not None and snapshot.headroom_db > 0.0
    assert snapshot.signal_activity > 0.9
    assert snapshot.bleed_index is not None and snapshot.bleed_index > 0.85
    assert set(snapshot.band_energy_db) >= {"sub", "low", "low_mid", "mid", "presence", "air"}
    assert snapshot.spectral_centroid_hz is not None and snapshot.spectral_centroid_hz > 0.0
    assert snapshot.dynamic_range_db is not None


def test_clipping_gain_trim_recommendation_has_safe_contract_fields():
    clipped = np.ones(2048, dtype=float)
    snapshot = extract_channel_features(
        clipped,
        channel_id=2,
        role="Vocal",
        group="VOCALS",
        current_gain_db=3.0,
    )

    recommendations = build_agentic_recommendations(snapshot)
    gain_trim = next(item for item in recommendations if item.action_type == "gain_trim")
    payload = gain_trim.to_dict()

    assert payload["agent_name"] == "Auto Gain / Balance Agent"
    assert payload["channel_id"] == 2
    assert payload["suggested_value"] <= 2.5
    assert payload["current_value"] == 3.0
    assert payload["confidence"] >= 0.7
    assert payload["risk_score"] < 0.5
    assert payload["safety_status"] == "dry_run_only"
    assert payload["dry_run"] is True
    assert payload["idempotency_key"].startswith("rec-")
    assert "peak_db" in payload["evidence"]


def test_upward_gain_is_blocked_for_auto_generated_quiet_sources():
    snapshot = ChannelFeatureSnapshot(
        channel_id=4,
        role="Vocal",
        group="VOCALS",
        peak_db=-18.0,
        lufs=-38.0,
        signal_activity=1.0,
        current_gain_db=0.0,
    )

    recommendations = build_agentic_recommendations(snapshot)
    review = safety_review_recommendations(recommendations)
    gain_review = next(item for item in recommendations if item.action_type == "gain_review")

    assert gain_review.safety_status == "blocked"
    assert gain_review.dry_run is True
    assert "auto_generated_upward_gain_blocked" in gain_review.evidence["blocked_reason"]
    assert review["summary"]["blocked_count"] == len(recommendations)


def test_harsh_vocal_eq_recommendation_is_proposal_only():
    harsh_vocal = _tone(4200.0, amplitude=0.25)
    snapshot = extract_channel_features(
        harsh_vocal,
        channel_id=8,
        role="Vocal",
        group="VOCALS",
    )

    recommendations = build_agentic_recommendations(snapshot)
    eq = next(item for item in recommendations if item.action_type == "eq")

    assert eq.safety_status == "proposal_only"
    assert eq.dry_run is True
    assert eq.suggested_params["gain_db"] < 0.0
    assert abs(eq.suggested_params["gain_db"]) <= AgenticFoundationConfig().max_eq_cut_db


def test_inter_channel_context_and_teacher_example_are_json_ready():
    vocal = ChannelFeatureSnapshot(
        channel_id=1,
        role="Vocal",
        group="VOCALS",
        lufs=-26.0,
        signal_activity=1.0,
        band_energy_db={"presence": -32.0},
    )
    guitar = ChannelFeatureSnapshot(
        channel_id=2,
        role="Guitar",
        group="GUITARS",
        lufs=-18.0,
        signal_activity=1.0,
        band_energy_db={"presence": -18.0},
    )

    context = build_inter_channel_context([vocal, guitar])
    recommendations = build_agentic_recommendations(guitar, context)
    example = build_offline_teacher_example(
        run_id="unit-run",
        snapshot=guitar,
        recommendations=recommendations,
        reference_metrics={"mix_lufs": -16.0},
        operator_decision="pending",
    ).to_dict()

    assert context.active_channel_count == 2
    assert context.masking_index["vocal_presence_vs_guitar_keys"] > 0.5
    assert example["example_id"].startswith("teacher-")
    assert example["feature_snapshot"]["channel_id"] == 2
    assert isinstance(example["recommendations"], list)
    assert math.isclose(example["reference_metrics"]["mix_lufs"], -16.0)


def test_inter_channel_context_excludes_vocal_group_from_band_sum():
    vocal_bus = ChannelFeatureSnapshot(
        channel_id=1,
        role="Lead Bus",
        group="VOCALS",
        lufs=-24.0,
        signal_activity=1.0,
        band_energy_db={"presence": -30.0},
    )
    guitar = ChannelFeatureSnapshot(
        channel_id=2,
        role="Guitar",
        group="GUITARS",
        lufs=-24.0,
        signal_activity=1.0,
        band_energy_db={"presence": -24.0},
    )

    context = build_inter_channel_context([vocal_bus, guitar])

    assert context.vocal_to_band_ratio_db == 0.0
    assert context.masking_index["vocal_presence_vs_guitar_keys"] > 0.7


def test_teacher_examples_can_be_built_from_channel_inventory_rows():
    channels = [
        {
            "id": 1,
            "name": "Lead Vocal",
            "role": "Vocal",
            "group": "VOCALS",
            "meter": {"rms_db": -12.0, "peak_db": -0.5, "lufs": -14.0, "dynamic_range_db": 11.5},
            "control_state": {"gain_db": 2.0, "fader_db": -6.0},
        },
        {
            "id": 2,
            "name": "Idle",
            "role": "Channel",
            "group": "CHANNELS",
            "meter": {"rms_db": -100.0, "peak_db": -100.0, "lufs": -100.0},
            "control_state": {},
        },
    ]

    examples = build_offline_teacher_examples_from_channel_inventory(
        channels,
        run_id="unit-teacher-run",
        reference_metrics={"mix_lufs": -16.0},
    )

    assert len(examples) == 1
    assert examples[0]["run_id"] == "unit-teacher-run"
    assert examples[0]["feature_snapshot"]["channel_id"] == 1
    assert examples[0]["teacher_policy"]["offline_only"] is True
    assert examples[0]["teacher_policy"]["live_write"] is False
    assert examples[0]["recommendations"]


class _FakeCapture:
    running = True
    num_channels = 2
    sample_rate = 48000

    def get_status(self):
        return {"running": True, "num_channels": 2, "sample_rate": 48000}

    def get_rms(self, channel):
        return {1: -15.0, 2: -38.0}.get(channel, -100.0)

    def get_peak(self, channel):
        return {1: -0.5, 2: -18.0}.get(channel, -100.0)

    def get_lufs(self, channel):
        return {1: -17.0, 2: -39.0}.get(channel, -100.0)


class _FakeServer:
    mixer_client = None
    connection_mode = "offline"
    audio_capture = _FakeCapture()


def test_channel_intelligence_exposes_agentic_foundation_as_read_only_dto():
    payload = build_channel_intelligence(_FakeServer(), total_channels=2)

    assert payload["type"] == "channel_intelligence"
    assert payload["summary"]["foundation_source"] == "agentic_foundation_v1"
    assert payload["summary"]["live_apply_allowed"] is False
    first = payload["channels"][0]
    assert first["foundation"]["source"] == "agentic_foundation_v1"
    assert first["foundation"]["recommendation_count"] >= 1
    assert first["foundation"]["safety_review"]["live_writes_enabled"] is False
    assert all(item["dry_run"] is True for item in first["foundation"]["recommendations"])


def test_agentic_foundation_has_no_live_transport_imports():
    assert assert_no_live_transport_imports()


def test_context_aware_gate_treats_cymbal_bleed_differently_by_channel_context():
    cymbal_bleed = _tone(10000.0, amplitude=0.22)
    quiet_vocal_body = _tone(240.0, amplitude=0.03)
    vocal_channel = quiet_vocal_body + cymbal_bleed
    overhead_channel = cymbal_bleed

    vocal_telemetry = extract_bleed_telemetry(
        vocal_channel,
        channel_id=12,
        channel_label="Lead Vocal",
        expected_instrument="Lead Vocal",
        peer_samples=[cymbal_bleed],
    )
    overhead_telemetry = extract_bleed_telemetry(
        overhead_channel,
        channel_id=8,
        channel_label="OH L",
        expected_instrument="Overhead",
        peer_samples=[cymbal_bleed],
    )

    agent = ContextAwareGateAgent()
    vocal_recommendation = agent.recommend(vocal_telemetry)
    overhead_recommendation = agent.recommend(overhead_telemetry)

    assert vocal_telemetry.bleed_correlation_score > 0.7
    assert vocal_telemetry.spectral_outlier_score > overhead_telemetry.spectral_outlier_score
    assert vocal_recommendation is not None
    assert vocal_recommendation.action_type == "expander"
    assert vocal_recommendation.suggested_params["mode"] == "expander"
    assert abs(vocal_recommendation.suggested_params["range_db"]) <= 8.0
    assert vocal_recommendation.dry_run is True
    assert "threshold-only gate" in vocal_recommendation.reason
    assert overhead_recommendation is not None
    assert overhead_recommendation.action_type == "expander"
    assert overhead_recommendation.suggested_params["range_db"] == -3.0
    assert "avoids hard gating" in overhead_recommendation.reason


def test_context_aware_gate_recommendation_is_decision_trace_ready_without_live_write():
    snare_bleed = extract_bleed_telemetry(
        _tone(200.0, amplitude=0.18) + _tone(4000.0, amplitude=0.12),
        channel_id=3,
        channel_label="Snare Top",
        expected_instrument="Snare",
        peer_samples=[_tone(200.0, amplitude=0.18)],
    )

    recommendation = ContextAwareGateAgent().recommend(snare_bleed)

    assert recommendation is not None
    assert recommendation.dry_run is True
    assert recommendation.safety_status in {"proposal_only", "review_required_high_risk"}
    trace = recommendation_to_decision_trace(recommendation)
    assert trace["source"] == "ContextAwareGateAgent"
    assert trace["target"] == {"type": "channel", "id": 3}
    assert trace["proof_status"] == "dry_run_only"
    assert trace["blocked_reason"] == "gate_expander_live_write_quarantined"
    assert trace["safety_limits"]["live_write_allowed"] is False
    assert trace["metadata"]["evidence"]["telemetry"]["channel_label"] == "Snare Top"
    assert trace["idempotency_key"].startswith("context-gate-")


def test_agentic_foundation_uses_context_aware_gate_for_gate_recommendations():
    snapshot = ChannelFeatureSnapshot(
        channel_id=12,
        role="Lead Vocal",
        group="VOCALS",
        noise_floor_db=-52.0,
        signal_activity=0.25,
        bleed_index=0.82,
        band_energy_db={
            "sub": -80.0,
            "low": -70.0,
            "low_mid": -52.0,
            "mid": -48.0,
            "presence": -40.0,
            "air": -20.0,
        },
        false_open_risk=0.45,
        false_close_risk=0.38,
        chatter_risk=0.12,
    )

    recommendations = build_agentic_recommendations(snapshot)
    gate = next(item for item in recommendations if item.agent_name == "ContextAwareGateAgent")

    assert gate.action_type == "expander"
    assert gate.safety_status in {"proposal_only", "review_required_high_risk"}
    assert gate.dry_run is True
    assert gate.evidence["live_write"] is False
    assert gate.evidence["telemetry"]["expected_instrument"] == "vocal"


def test_context_aware_gate_has_no_live_transport_imports():
    assert context_gate_has_no_live_transport_imports()


def test_agentic_foundation_adapter_exports_only_bounded_downward_gain_fader_preview():
    recommendations = [
        {
            "channel_id": 1,
            "agent_name": "Auto Gain / Balance Agent",
            "action_type": "fader",
            "current_value": -6.0,
            "suggested_value": -7.0,
            "confidence": 0.9,
            "risk_score": 0.2,
            "reason": "bounded down",
            "safety_status": "dry_run_only",
            "dry_run": True,
            "idempotency_key": "rec-safe-fader",
        },
        {
            "channel_id": 2,
            "agent_name": "Auto Gain / Balance Agent",
            "action_type": "gain_trim",
            "current_value": 1.0,
            "suggested_value": 0.5,
            "confidence": 0.9,
            "risk_score": 0.2,
            "reason": "bounded trim down",
            "safety_status": "dry_run_only",
            "dry_run": True,
            "idempotency_key": "rec-safe-gain",
        },
        {
            "channel_id": 3,
            "agent_name": "Auto Gain / Balance Agent",
            "action_type": "gain_review",
            "current_value": 0.0,
            "suggested_value": 0.5,
            "confidence": 0.9,
            "risk_score": 0.2,
            "reason": "upward change blocked",
            "safety_status": "blocked",
            "dry_run": True,
        },
        {
            "channel_id": 4,
            "agent_name": "AutoEQ Agent",
            "action_type": "eq",
            "current_value": None,
            "suggested_params": {"gain_db": -2.0},
            "confidence": 0.9,
            "risk_score": 0.2,
            "reason": "not live executor",
            "safety_status": "proposal_only",
            "dry_run": True,
        },
    ]

    proposals = adapt_agentic_foundation_gain_fader_recommendations(recommendations)

    assert [proposal["requested_change"]["value_type"] for proposal in proposals] == ["fader", "gain"]
    assert all(proposal["adapter_id"] == "agentic_foundation_gain_fader" for proposal in proposals)
    assert all(proposal["method_id"] == "agentic_method_foundation" for proposal in proposals)
    assert all(proposal["queue_action"] == "preview_only_not_queued" for proposal in proposals)
    assert all(proposal["can_import_to_queue"] is False for proposal in proposals)
    assert all(proposal["requested_change"]["delta_db"] < 0 for proposal in proposals)
    assert all("requested_change" in proposal for proposal in proposals)
