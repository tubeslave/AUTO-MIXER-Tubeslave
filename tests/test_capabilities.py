from integrations.mixing_station.capabilities import load_capability_map


def test_wing_capability_marks_core_visualization_writable():
    capabilities = load_capability_map("config/mixing_station/wing_rack_capabilities.yaml")

    fader = capabilities.status_for("fader")
    eq_gain = capabilities.status_for("peq.band1.gain")
    phase = capabilities.status_for("phase.invert")
    delay = capabilities.status_for("delay.time")
    comp_on = capabilities.status_for("compressor.enabled")
    comp_model = capabilities.status_for("compressor.model")
    comp_filter = capabilities.status_for("compressor.filter.band1.q")
    scene = capabilities.status_for("scene.recall")

    assert fader.supported is True
    assert fader.can_write_visualization is True
    assert eq_gain.supported is True
    assert eq_gain.needs_discovery is False
    assert eq_gain.can_write_visualization is True
    assert phase.supported is True
    assert delay.supported is True
    assert comp_on.supported is True
    assert comp_model.supported is True
    assert comp_model.read_only is True
    assert comp_filter.supported is True
    assert scene.supported is False
    assert scene.forbidden_live is True


def test_dlive_capability_keeps_scenes_read_only_and_blocks_actions():
    capabilities = load_capability_map("config/mixing_station/dlive_capabilities.yaml")

    scene = capabilities.status_for("scene.recall")
    soft_key = capabilities.status_for("soft_key.trigger")

    assert scene.supported is False
    assert scene.read_only is True
    assert scene.forbidden_live is True
    assert soft_key.supported is False
    assert soft_key.forbidden_live is True
