from integrations.mixing_station.mapper import load_mapper
from integrations.mixing_station.models import AutomixCorrection


def correction(parameter, value=-5.0):
    return AutomixCorrection(
        console_profile="dlive",
        mode="offline_visualization",
        channel_index=0,
        parameter=parameter,
        value=value,
        value_unit="db",
        reason="test",
    )


def test_dlive_fader_mute_pan_mapping():
    mapper = load_mapper("config/mixing_station/maps/dlive.yaml")

    assert mapper.map(correction("fader")).command.data_path == "ch.0.mix.lvl"
    assert mapper.map(correction("mute", True)).command.data_path == "ch.0.mix.mute"
    assert mapper.map(correction("pan", 0.0)).command.data_path == "ch.0.mix.pan"


def test_dlive_deep_parameters_require_discovery():
    mapper = load_mapper("config/mixing_station/maps/dlive.yaml")

    hpf = mapper.map(correction("hpf.frequency", 120.0))
    eq = mapper.map(correction("peq.band2.frequency", 350.0))

    assert hpf.success is False
    assert hpf.needs_discovery is True
    assert eq.success is False
    assert eq.needs_discovery is True
