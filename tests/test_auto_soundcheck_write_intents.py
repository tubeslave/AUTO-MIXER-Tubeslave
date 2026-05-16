from types import SimpleNamespace
from unittest.mock import MagicMock

from auto_soundcheck_engine import AutoSoundcheckEngine, ChannelInfo


class WingSoundcheckMixerDouble:
    def __init__(self):
        self.set_observation_mode = MagicMock()
        self.set_gain = MagicMock()
        self.set_fader = MagicMock()
        self.set_hpf = MagicMock()
        self.set_eq_band = MagicMock()
        self.set_compressor = MagicMock()
        self.set_pan = MagicMock()
        self.set_send_level = MagicMock()
        self.reset_channel_processing = MagicMock()
        self.set_mute = MagicMock()


def _build_metrics():
    return SimpleNamespace(
        spectral=SimpleNamespace(
            band_energy={"sub": -80.0, "bass": -32.0, "low_mid": -26.0},
            centroid_hz=2200.0,
            mud_ratio=0.12,
            presence_ratio=0.14,
            brightness=0.11,
            warmth=0.16,
            spectral_tilt_db=0.0,
            flatness=0.22,
            flux=0.18,
        ),
        level=SimpleNamespace(
            rms_db=-24.0,
            lufs_integrated=-24.0,
            true_peak_dbtp=-8.0,
            crest_factor_db=11.0,
        ),
        dynamics=SimpleNamespace(
            dynamic_range_db=10.0,
            attack_time_ms=18.0,
            release_time_ms=140.0,
            transient_density=1.2,
            sustain_level_db=-24.0,
        ),
    )


def test_auto_soundcheck_gain_emits_replay_intent_and_skips_live_wing_write():
    engine = AutoSoundcheckEngine(auto_apply=False)
    engine.mixer_type = "wing"
    engine.mixer_client = WingSoundcheckMixerDouble()
    engine.replay_arrangement_id = "intro_a"
    engine.replay_scene_id = "soundcheck_scene_1"
    engine.replay_scene_label = "preflight"

    info = ChannelInfo(
        channel=1,
        name="Kick In",
        preset="kick",
        peak_db=-8.0,
        rms_db=-20.0,
        fader_db=-12.0,
    )

    engine._apply_input_gain(1, info, "kick")

    engine.mixer_client.set_gain.assert_not_called()

    intents = engine.get_replay_write_intents()
    assert len(intents) == 1
    record = intents[0]
    payload = record["timeline_artifact"]["events"][0]["source_payload"]

    assert record["write_kind"] == "gain"
    assert record["allow_live_write"] is False
    assert record["metadata"]["arrangement_id"] == "intro_a"
    assert record["metadata"]["scene_id"] == "soundcheck_scene_1"
    assert record["metadata"]["scene_label"] == "preflight"
    assert payload["readback_source"] == "analysis_snapshot"
    assert payload["requires_confirmation"] is True
    assert payload["recommended_target_trim_db"] < 0.0
    assert payload["autonomy_mode"] == "semi_autonomous_calibration"
    assert payload["required_approval_level"] == "operator_required"
    assert payload["action_envelope"]["transport_policy"] == "dry_run_only"
    assert payload["action_envelope"]["trust_gates"]["summary"]["effective_autonomy_mode"] == (
        "semi_autonomous_calibration"
    )


def test_wing_headless_defaults_force_no_write_and_enable_observation_mode():
    engine = AutoSoundcheckEngine(mixer_type="wing", auto_apply=True)
    engine.mixer_client = WingSoundcheckMixerDouble()

    engine._enforce_wing_headless_no_write_defaults()

    assert engine.auto_apply is False
    assert engine.auto_apply_forced_off is True
    assert engine.auto_apply_forced_off_reason == "wing_headless_no_write_default"
    engine.mixer_client.set_observation_mode.assert_called_once_with(
        True,
        reason="wing_headless_no_write_default",
    )


def test_wing_headless_non_fader_soundcheck_surfaces_emit_recommendations_only():
    engine = AutoSoundcheckEngine(mixer_type="wing", auto_apply=False)
    engine.mixer_client = WingSoundcheckMixerDouble()

    info = ChannelInfo(
        channel=1,
        name="Lead Vox",
        preset="leadVocal",
        has_signal=True,
        metrics=_build_metrics(),
    )
    engine.channels[1] = info

    engine._apply_hpf(1, "leadVocal")
    engine._apply_eq(1, "leadVocal")
    engine._apply_compressor(1, info, "leadVocal")
    engine._apply_pan(1, info, "leadVocal")
    engine._apply_fx_sends(1, info, "leadVocal")

    engine.mixer_client.set_hpf.assert_not_called()
    engine.mixer_client.set_eq_band.assert_not_called()
    engine.mixer_client.set_compressor.assert_not_called()
    engine.mixer_client.set_pan.assert_not_called()
    engine.mixer_client.set_send_level.assert_not_called()

    bundle = engine.get_soundcheck_recommendation_bundle()
    assert bundle is not None
    assert bundle["dry_run_only"] is True
    assert bundle["live_mixer_writes"] is False

    recommendation_kinds = {item["kind"] for item in bundle["recommendations"]}
    assert {"hpf", "eq", "compressor", "pan", "fx_send"} <= recommendation_kinds


def test_wing_headless_reset_path_skips_live_reset_writes():
    engine = AutoSoundcheckEngine(mixer_type="wing", auto_apply=False)
    engine.mixer_client = WingSoundcheckMixerDouble()
    engine.channels = {
        1: ChannelInfo(channel=1, name="Kick In"),
        2: ChannelInfo(channel=2, name="Snare Top"),
    }

    engine._reset_channels()

    engine.mixer_client.reset_channel_processing.assert_not_called()


def test_wing_run_still_processes_artifacts_after_auto_apply_is_forced_off(monkeypatch):
    engine = AutoSoundcheckEngine(mixer_type="wing", auto_apply=True)
    apply_calls = []

    monkeypatch.setattr(engine, "_connect_mixer", lambda: True)
    monkeypatch.setattr(engine, "_start_audio", lambda: True)
    monkeypatch.setattr(engine, "_scan_channels", lambda: None)
    monkeypatch.setattr(engine, "_read_channel_state", lambda: None)
    monkeypatch.setattr(engine, "_reset_channels", lambda: None)
    monkeypatch.setattr(engine, "_wait_and_analyze", lambda: None)

    def _fake_apply():
        apply_calls.append("apply")
        engine._stop_event.set()

    monkeypatch.setattr(engine, "_apply_corrections", _fake_apply)
    monkeypatch.setattr(engine, "stop", lambda: None)

    result = engine.run()

    assert result is True
    assert apply_calls == ["apply"]
    assert engine.auto_apply is False
    assert engine.auto_apply_forced_off is True
