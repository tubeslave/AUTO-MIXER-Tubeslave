"""Live-параметры гейтов барабанов: range −20, sidechain HPF200/LPF4k."""

from auto_gate_caig import (
    DrumKitRules,
    GateSettings,
    live_gate_settings_for_instrument,
)


class TestDrumGateParams:
    def test_drum_range_keeps_natural_tail(self):
        for inst in ("kick", "snare", "tom"):
            gs = live_gate_settings_for_instrument(inst, nearest_source_db=-42.0)
            assert gs.range_db == -20.0, inst

    def test_drum_sidechain_filtering(self):
        gs = live_gate_settings_for_instrument("kick", nearest_source_db=-42.0)
        assert gs.sidechain_hpf_hz == 200.0
        assert gs.sidechain_lpf_hz == 4000.0

    def test_drum_attack_preserves_transient(self):
        gs = live_gate_settings_for_instrument("snare", nearest_source_db=-42.0)
        assert gs.attack_ms < 5.0

    def test_non_drum_unchanged(self):
        gs = live_gate_settings_for_instrument("vocal", target_peak_db=-18.0)
        assert gs.range_db == -80.0
        assert gs.sidechain_hpf_hz is None
        assert gs.sidechain_lpf_hz is None

    def test_default_settings_backward_compatible(self):
        gs = GateSettings()
        assert gs.range_db == -80.0
        assert gs.sidechain_hpf_hz is None


class TestDrumKitRulesMinGain:
    def test_drums_limited_to_minus_20(self):
        rules = DrumKitRules()
        for inst in ("kick", "snare", "tom"):
            assert rules.get_min_gain(inst) == -20.0

    def test_overheads_never_fully_closed(self):
        assert DrumKitRules().get_min_gain("overheads") == -10.0

    def test_unknown_keeps_full_range(self):
        assert DrumKitRules().get_min_gain("vocal") == -80.0
