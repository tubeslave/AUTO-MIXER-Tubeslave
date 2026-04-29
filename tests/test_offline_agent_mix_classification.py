import importlib.util
import sys
from pathlib import Path


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
