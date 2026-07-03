from __future__ import annotations

import numpy as np
import pytest

from automixer.production_mix_v1.autoeq import (
    AutoEQEngine,
    AutoEQSettings,
    AutoEQTrack,
    DynamicEQDecisionEngine,
    EQDecision,
    EQDecisionEngine,
    EQResultValidator,
    EQSafetyLimiter,
    InstrumentProfileLibrary,
    MaskingAnalyzer,
    SpectralAnalyzer,
)


def test_instrument_profile_library_returns_required_profiles():
    library = InstrumentProfileLibrary()
    required = [
        "kick",
        "snare_top",
        "snare_bottom",
        "tom",
        "floor_tom",
        "overhead_l",
        "overhead_r",
        "overhead_pair",
        "bass",
        "guitar_l",
        "guitar_r",
        "guitar",
        "accordion",
        "keys",
        "playback_l",
        "playback_r",
        "playback",
        "lead_vocal",
        "back_vocal",
        "vocal_group",
        "drums_group",
        "harmonic_group",
        "low_end_group",
        "cymbals_group",
        "master",
    ]

    for name in required:
        profile = library.get(name)
        assert profile.instrument == name
        assert profile.useful_bands
        assert profile.problematic_bands
        assert profile.protected_bands
        assert profile.max_static_gain_db > 0
        assert profile.max_dynamic_cut_db > 0
        assert profile.preferred_q_range[0] < profile.preferred_q_range[1]


def test_spectral_analyzer_returns_stable_metrics():
    sr = 48_000
    audio = _tone(sr, 440.0, 0.08)
    track = AutoEQTrack(1, "Vocal.wav", "lead_vocal", audio, sr, "vocals")

    metrics = SpectralAnalyzer().analyze_track(track)

    assert metrics["band_energy_db"]
    assert 300.0 <= metrics["spectral_centroid_hz"] <= 700.0
    assert 0.0 <= metrics["mud_score"] <= 1.0
    assert "integrated_lufs" in metrics
    assert "mono_compatibility" in metrics


def test_masking_analyzer_detects_vocal_guitar_overlap():
    sr = 48_000
    vocal = AutoEQTrack(1, "Vocal.wav", "lead_vocal", _tone(sr, 3000.0, 0.04), sr, "vocals")
    guitar = AutoEQTrack(2, "Guitar L.wav", "guitar_l", _tone(sr, 3000.0, 0.08), sr, "guitars")
    tracks = [vocal, guitar]
    analyzer = SpectralAnalyzer()
    metrics = analyzer.analyze_tracks(tracks)

    relations = MaskingAnalyzer(InstrumentProfileLibrary()).analyze(tracks, metrics)

    relation = next(item for item in relations if item.masked_source == "Vocal.wav" and item.masker_source == "Guitar L.wav")
    assert relation.suggested_action == "dynamic_bell_cut"
    assert relation.masking_score > 0.5
    assert relation.confidence >= 0.6


def test_decision_engine_creates_explainable_candidate_structure():
    sr = 48_000
    guitar = AutoEQTrack(2, "Guitar L.wav", "guitar_l", _mix(sr, [(260.0, 0.12), (3200.0, 0.05)]), sr, "guitars")
    metrics = SpectralAnalyzer().analyze_tracks([guitar])

    decisions = EQDecisionEngine(InstrumentProfileLibrary()).generate_channel_decisions([guitar], metrics, [], AutoEQSettings.from_mapping({"enabled": True}))

    assert decisions
    payload = decisions[0].to_dict()
    assert payload["channel"] == 2
    assert payload["decision"] in {"static_bell_cut", "notch", "static_bell_boost"}
    assert payload["reason"]
    assert "source_metrics_before" in payload


def test_safety_limiter_clamps_and_rejects_unsafe_decisions():
    settings = AutoEQSettings.from_mapping({"enabled": True, "mode": "live"})
    limiter = EQSafetyLimiter(InstrumentProfileLibrary())
    loud_metrics = {"peak_dbfs": -0.5}
    decisions = [
        EQDecision(1, "Guitar.wav", "guitar", "static_bell_cut", 3000.0, 8.0, -8.0, confidence=0.9, source_metrics_before=loud_metrics, reason="harshness"),
        EQDecision(1, "Guitar.wav", "guitar", "static_bell_boost", 3000.0, 1.0, 2.0, confidence=0.9, source_metrics_before=loud_metrics, reason="presence boost"),
        EQDecision(1, "Guitar.wav", "guitar", "static_bell_cut", 260.0, 1.0, -0.8, confidence=0.2, source_metrics_before=loud_metrics, reason="low confidence"),
    ]

    limited, rejected = limiter.limit(decisions, settings)

    assert limited[0].gain_db == pytest.approx(-1.5)
    assert limited[0].q == pytest.approx(4.0)
    assert limited[0].safety_limited is True
    assert any(item.rejection_reason == "never boost when channel is near peak limit" for item in rejected)
    assert any(item.rejection_reason == "confidence too low" for item in rejected)


def test_result_validator_rolls_back_worsening_decision():
    sr = 48_000
    track = AutoEQTrack(1, "Guitar.wav", "guitar", _tone(sr, 260.0, 0.06), sr, "guitars")
    bad = EQDecision(1, "Guitar.wav", "guitar", "static_bell_boost", 260.0, 1.0, 2.0, confidence=0.9, source_metrics_before={}, reason="mud boost regression")

    _processed, accepted, rejected, summary = EQResultValidator(SpectralAnalyzer()).apply_and_validate(
        [track],
        [bad],
        AutoEQSettings.from_mapping({"enabled": True}),
    )

    assert not accepted
    assert rejected
    assert rejected[0].rollback is True
    assert summary["rejected_count"] == 1


def test_dynamic_eq_engine_chooses_dynamic_for_temporary_masking():
    sr = 48_000
    vocal = AutoEQTrack(1, "Vocal.wav", "lead_vocal", _burst_tone(sr, 3000.0, 0.08), sr, "vocals")
    guitar = AutoEQTrack(2, "Guitar.wav", "guitar", _tone(sr, 3000.0, 0.08), sr, "guitars")
    tracks = [vocal, guitar]
    metrics = SpectralAnalyzer().analyze_tracks(tracks)
    relations = MaskingAnalyzer(InstrumentProfileLibrary()).analyze(tracks, metrics)

    decisions = DynamicEQDecisionEngine().generate(tracks, metrics, relations, AutoEQSettings.from_mapping({"enabled": True}))

    assert any(decision.dynamic and decision.sidechain_source == "Vocal.wav" and decision.channel == 2 for decision in decisions)


def test_autoeq_engine_integration_report_and_limits():
    sr = 48_000
    tracks = [
        AutoEQTrack(1, "Vocal.wav", "lead_vocal", _burst_tone(sr, 3000.0, 0.06), sr, "vocals"),
        AutoEQTrack(2, "Guitar L.wav", "guitar_l", _tone(sr, 3200.0, 0.08), sr, "guitars"),
        AutoEQTrack(3, "KICK.wav", "kick", _tone(sr, 70.0, 0.08), sr, "drums"),
        AutoEQTrack(4, "Bass.wav", "bass", _tone(sr, 75.0, 0.10), sr, "bass"),
        AutoEQTrack(5, "OH L.wav", "overhead_l", _tone(sr, 7000.0, 0.07), sr, "drums"),
        AutoEQTrack(6, "OH R.wav", "overhead_r", _tone(sr, 7000.0, 0.07), sr, "drums"),
    ]

    result = AutoEQEngine(AutoEQSettings.from_mapping({"enabled": True, "mode": "offline", "apply_master_eq": False})).run(
        tracks,
        master_audio=sum((track.audio for track in tracks), np.zeros_like(tracks[0].audio)),
    )
    report = result.report

    assert report["autoeq_settings"]["enabled"] is True
    assert report["autoeq_channel_analysis"]
    assert report["autoeq_decisions"]
    assert report["autoeq_rejected_decisions"] is not None
    assert report["autoeq_validation_summary"]["proposed_count"] >= len(report["autoeq_decisions"])
    assert report["autoeq_master_decisions"] == []
    assert any(item["decision"] == "dynamic_bell_cut" and item["channel"] == 2 for item in report["autoeq_decisions"])
    assert not any(item["channel"] == 1 and "boost" in item["decision"] for item in report["autoeq_decisions"])
    assert any(item["channel"] == 4 and item["decision"] == "dynamic_bell_cut" for item in report["autoeq_decisions"])
    assert all(
        item.get("linked_group") == "overhead_pair"
        for item in report["autoeq_decisions"]
        if item["file"] in {"OH L.wav", "OH R.wav"} and item["dynamic"]
    )
    assert all(abs(float(item["gain_db"])) <= 5.0 for item in report["autoeq_decisions"] if item["target_type"] == "channel")
    assert any(cmd["mode"] == "dry_run" for cmd in report["autoeq_osc_commands"])


def test_autoeq_report_only_does_not_modify_audio():
    sr = 48_000
    audio = _tone(sr, 3200.0, 0.08)
    track = AutoEQTrack(1, "Guitar.wav", "guitar", audio, sr, "guitars")
    settings = AutoEQSettings.from_mapping({"enabled": True, "report_only": True})

    result = AutoEQEngine(settings).run([track], master_audio=audio)

    np.testing.assert_allclose(result.processed_audio_by_channel[1], audio)
    assert result.report["autoeq_settings"]["report_only"] is True
    assert result.report["autoeq_validation_summary"]["report_only"] is True


def _tone(sample_rate: int, freq: float, amp: float) -> np.ndarray:
    t = np.arange(sample_rate, dtype=np.float32) / sample_rate
    mono = amp * np.sin(2.0 * np.pi * freq * t)
    return np.column_stack([mono, mono]).astype(np.float32)


def _burst_tone(sample_rate: int, freq: float, amp: float) -> np.ndarray:
    audio = _tone(sample_rate, freq, amp)
    audio[: sample_rate // 3] = 0.0
    audio[sample_rate * 2 // 3 :] = 0.0
    return audio


def _mix(sample_rate: int, parts: list[tuple[float, float]]) -> np.ndarray:
    t = np.arange(sample_rate, dtype=np.float32) / sample_rate
    mono = np.zeros(sample_rate, dtype=np.float32)
    for freq, amp in parts:
        mono += amp * np.sin(2.0 * np.pi * freq * t)
    return np.column_stack([mono, mono]).astype(np.float32)
