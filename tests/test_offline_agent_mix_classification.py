import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest


def load_offline_agent_mix():
    spec = importlib.util.spec_from_file_location(
        "offline_agent_mix_test_module",
        Path(__file__).resolve().parents[1] / "tools" / "offline_agent_mix.py",
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_plain_snare_is_treated_as_combined_top_bottom_source():
    mod = load_offline_agent_mix()
    instrument, pan, hpf, target_rms, eq, comp, phase = mod.classify_track(Path("SNARE.wav"))

    assert instrument == "snare"
    assert pan == 0.0
    assert hpf == 110.0
    assert target_rms == -23.0
    assert eq[0] == (220, -2.0, 1.1)
    assert comp == (-19, 3.0, 8, 130)
    assert phase is False


def test_snare_top_and_bottom_keep_their_specific_profiles():
    mod = load_offline_agent_mix()

    instrument_t, pan_t, hpf_t, _, eq_t, comp_t, phase_t = mod.classify_track(Path("SNARE T.wav"))
    instrument_b, pan_b, hpf_b, _, eq_b, comp_b, phase_b = mod.classify_track(Path("Snare B.wav"))

    assert instrument_t == "snare"
    assert pan_t == -0.02
    assert hpf_t == 90.0
    assert eq_t[0] == (200, 2.0, 1.0)
    assert comp_t == (-20, 4.0, 6, 120)
    assert phase_t is False

    assert instrument_b == "snare"
    assert pan_b == 0.02
    assert hpf_b == 120.0
    assert eq_b[0] == (220, 1.0, 1.0)
    assert comp_b == (-22, 3.5, 8, 110)
    assert phase_b is True


@pytest.mark.parametrize("name, canonical", [
    ("04_overhead_L_066_phrase_rock_complex_fast_sticks", "overhead l"),
    ("05_overhead_R_066_phrase_rock_complex_fast_sticks", "overhead r"),
    ("04-OH-L-take1", "oh l"),
    ("05.OH.R.take1", "oh r"),
    ("Overheads Left", "overhead l"),
    ("Overheads Right", "overhead r"),
    ("OH L", "oh l"), ("OH R", "oh r"),
    ("OHL", "oh l"), ("OHR", "oh r"),
])
def test_explicit_overhead_sides_keep_existing_profiles(name, canonical):
    mod = load_offline_agent_mix()
    assert mod.classify_track(Path(name + ".wav")) == mod.classify_track(
        Path(canonical + ".wav")
    )


@pytest.mark.parametrize("name", ["oh_later", "overhead_rightward", "school", "ohrchestra"])
def test_overhead_side_match_does_not_consume_partial_words(name):
    mod = load_offline_agent_mix()
    assert mod.classify_track(Path(name + ".wav"))[0] == "custom"


def test_explicit_overhead_sides_override_pan_and_channel_order():
    mod = load_offline_agent_mix()
    right = SimpleNamespace(instrument="overhead", name="05_overhead_R_take", pan=-0.1)
    left = SimpleNamespace(instrument="overhead", name="04_overhead_L_take", pan=0.1)
    assert mod._overhead_lr({1: right, 2: left}) == ((2, left), (1, right))


@pytest.mark.parametrize("side, louder_channel", [("L", 0), ("R", 1)])
def test_overhead_filename_pan_reaches_stereo_samples(side, louder_channel):
    mod = load_offline_agent_mix()
    _, pan, *_ = mod.classify_track(Path(f"04_overhead_{side}_take.wav"))
    mono = 0.1 * np.sin(2 * np.pi * 1000 * np.arange(4800) / 48000)
    stereo = mod.pan_mono(mono, pan)
    power = np.mean(stereo ** 2, axis=0)
    assert power[louder_channel] > 10 * power[1 - louder_channel]


def test_unsided_and_conflicting_overhead_labels_do_not_choose_a_side():
    mod = load_offline_agent_mix()
    assert mod.classify_track(Path("Overheads.wav"))[:2] == ("overhead", 0.0)
    assert mod._overhead_side("overhead_L_overhead_R") is None
