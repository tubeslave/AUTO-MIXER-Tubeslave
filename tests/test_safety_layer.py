from integrations.mixing_station.models import AutomixCorrection
from integrations.mixing_station.safety import (
    MixingStationSafetyConfig,
    MixingStationSafetyLayer,
)


def correction(parameter, value, previous=None, mode="offline_visualization"):
    return AutomixCorrection(
        console_profile="wing_rack",
        mode=mode,
        channel_index=0,
        parameter=parameter,
        value=value,
        value_unit="db",
        previous_value=previous,
        reason="test",
    )


def test_safety_clamps_fader_step_and_ceiling():
    layer = MixingStationSafetyLayer(
        MixingStationSafetyConfig(dry_run_default=True, max_fader_step_db=1.5)
    )

    result = layer.validate(correction("fader", -5.0, previous=-10.0))
    ceiling = layer.validate(correction("fader", 3.0))

    assert result.allowed is True
    assert result.status == "clamped"
    assert result.correction.value == -8.5
    assert ceiling.correction.value == 0.0


def test_safety_clamps_eq_and_hpf():
    layer = MixingStationSafetyLayer(
        MixingStationSafetyConfig(
            max_eq_gain_step_db=1.5,
            max_eq_gain_absolute_db=12.0,
            max_eq_boost_db=6.0,
            min_hpf_hz=20.0,
            max_hpf_hz=250.0,
        )
    )

    eq = layer.validate(correction("peq.band1.gain", 9.0, previous=0.0))
    hpf = layer.validate(correction("hpf.frequency", 500.0))

    assert eq.correction.value == 1.5
    assert hpf.correction.value == 250.0


def test_safety_clamps_delay_time():
    layer = MixingStationSafetyLayer(
        MixingStationSafetyConfig(min_delay_ms=0.0, max_delay_ms=10.0)
    )

    too_long = layer.validate(correction("delay.time", 25.0))
    negative = layer.validate(correction("delay.time", -2.0))

    assert too_long.correction.value == 10.0
    assert negative.correction.value == 0.0


def test_safety_clamps_compressor_parameters():
    layer = MixingStationSafetyLayer(MixingStationSafetyConfig())

    threshold = layer.validate(correction("compressor.threshold", 6.0))
    ratio = layer.validate(correction("compressor.ratio", 40.0))
    mix = layer.validate(correction("compressor.mix", 140.0))
    makeup = layer.validate(correction("compressor.makeup_gain", 20.0))
    attack = layer.validate(correction("compressor.attack", 0.0))
    sidechain = layer.validate(correction("compressor.filter.frequency", 5.0))
    q = layer.validate(correction("compressor.filter.band1.q", 40.0))

    assert threshold.correction.value == 0.0
    assert ratio.correction.value == 20.0
    assert mix.correction.value == 100.0
    assert makeup.correction.value == 12.0
    assert attack.correction.value == 0.1
    assert sidechain.correction.value == 20.0
    assert q.correction.value == 10.0


def test_safety_blocks_scene_recall_and_phantom_power():
    layer = MixingStationSafetyLayer(MixingStationSafetyConfig())

    scene = layer.validate(correction("scene.recall", 1))
    phantom = layer.validate(correction("phantom_power.enabled", True))

    assert scene.allowed is False
    assert scene.status == "blocked"
    assert phantom.allowed is False


def test_safety_blocks_live_control_by_default():
    layer = MixingStationSafetyLayer(MixingStationSafetyConfig(allow_live_control=False))

    result = layer.validate(correction("fader", -5.0, mode="live_control"))

    assert result.allowed is False
    assert "live_control" in result.message


def test_safety_rate_limits_per_channel():
    now = [100.0]
    layer = MixingStationSafetyLayer(
        MixingStationSafetyConfig(rate_limit_per_channel_hz=5.0),
        time_provider=lambda: now[0],
    )
    first = correction("fader", -9.0, previous=-10.0)
    accepted = layer.validate(first)
    layer.record_sent(accepted.correction)

    blocked = layer.validate(correction("fader", -8.0, previous=-9.0))

    assert blocked.allowed is False
    assert blocked.rate_limited is True


def test_safety_emergency_stop_file_blocks(tmp_path):
    flag = tmp_path / "EMERGENCY_STOP"
    flag.write_text("stop", encoding="utf-8")
    layer = MixingStationSafetyLayer(
        MixingStationSafetyConfig(emergency_stop_file=str(flag))
    )

    result = layer.validate(correction("fader", -5.0))

    assert result.allowed is False
    assert result.emergency_stop is True
