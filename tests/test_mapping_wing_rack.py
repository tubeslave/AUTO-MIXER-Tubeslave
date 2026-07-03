from integrations.mixing_station.mapper import load_mapper
from integrations.mixing_station.models import AutomixCorrection


def correction(parameter, value=-5.0):
    return AutomixCorrection(
        console_profile="wing_rack",
        mode="offline_visualization",
        channel_index=0,
        parameter=parameter,
        value=value,
        value_unit="db",
        reason="test",
    )


def test_wing_fader_mute_pan_mapping():
    mapper = load_mapper("config/mixing_station/maps/wing_rack.yaml")

    assert mapper.map(correction("fader")).command.data_path == "ch.0.mix.lvl"
    mute = mapper.map(correction("mute", True)).command
    pan = mapper.map(correction("pan", 0.25)).command

    assert mute.data_path == "ch.0.mix.on"
    assert mute.value is False
    assert pan.data_path == "ch.0.mix.pan"
    assert pan.value == 25.0


def test_wing_send_and_eq_mapping_uses_discovered_paths():
    mapper = load_mapper("config/mixing_station/maps/wing_rack.yaml")

    send = mapper.map(correction("send.fx13.level"))
    eq = mapper.map(correction("peq.band1.gain"))
    hpf = mapper.map(correction("hpf.frequency", 120.0))
    phase = mapper.map(correction("phase.invert", True)).command
    delay = mapper.map(correction("delay.time", 4.5)).command

    assert send.command.data_path == "ch.0.mix.sends.12.lvl"
    assert eq.command.data_path == "ch.0.peq.bands.0.gain"
    assert hpf.command.data_path == "ch.0.preamp.filter.0.freq"
    assert phase.data_path == "ch.0.preamp.inv"
    assert phase.value is True
    assert delay.data_path == "ch.0.delay.time"
    assert delay.value == 4.5


def test_wing_dynamics_enable_and_gate_mapping():
    mapper = load_mapper("config/mixing_station/maps/wing_rack.yaml")

    comp_on = mapper.map(correction("compressor.enabled", True)).command
    comp_model = mapper.map(correction("compressor.model", 0)).command
    comp_mix = mapper.map(correction("compressor.mix", 75.0)).command
    comp_attack = mapper.map(correction("compressor.attack", 12.0)).command
    comp_filter_q = mapper.map(correction("compressor.filter.band1.q", 2.1)).command
    gate_on = mapper.map(correction("gate.enabled", True)).command

    assert comp_on.data_path == "ch.0.dyn.on"
    assert comp_on.value is True
    assert comp_model.data_path == "ch.0.rawDyn.model"
    assert comp_mix.data_path == "ch.0.dyn.mix"
    assert comp_attack.data_path == "ch.0.dyn.attack"
    assert comp_filter_q.data_path == "ch.0.dyn.filter.filters.bands.0.q"
    assert gate_on.data_path == "ch.0.gate.on"


def test_wing_channel_name_and_main_mapping():
    mapper = load_mapper("config/mixing_station/maps/wing_rack.yaml")

    name = mapper.map(correction("channel.name", "Lead Vocal Long Name")).command
    main = AutomixCorrection(
        console_profile="wing_rack",
        mode="offline_visualization",
        channel_index=0,
        strip_type="main",
        parameter="peq.band1.gain",
        value=-0.8,
        value_unit="db",
        reason="test",
    )

    assert name.data_path == "ch.0.cfg.name"
    assert name.value == "Lead Vocal Long"
    assert mapper.map(main).command.data_path == "ch.72.peq.bands.0.gain"
