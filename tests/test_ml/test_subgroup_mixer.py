"""
Tests for backend.ml.subgroup_mixer — hierarchical subgroup/bus mixing,
channel assignment, bus processing, master output, and convenience API.

All tests use numpy-generated audio. scipy is optional.
"""

import numpy as np
import pytest

from backend.ml.subgroup_mixer import (
    BUS_ASSIGNMENTS,
    BUS_PROCESSING,
    MASTER_PROCESSING,
    SubgroupMixer,
    HAS_SCIPY,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def mixer():
    return SubgroupMixer(sr=48000)


@pytest.fixture
def sine_1s():
    """1-second 440 Hz mono sine at 48 kHz."""
    sr = 48000
    t = np.linspace(0, 1.0, sr, endpoint=False, dtype=np.float64)
    return np.sin(2 * np.pi * 440 * t) * 0.5


@pytest.fixture
def multi_channel_audio():
    """Dict of channel audio for a small virtual concert."""
    sr = 48000
    n = sr  # 1 second
    t = np.linspace(0, 1.0, n, endpoint=False, dtype=np.float64)
    rng = np.random.default_rng(42)
    return {
        "ch_kick": np.sin(2 * np.pi * 60 * t) * 0.7,
        "ch_snare": rng.standard_normal(n).astype(np.float64) * 0.4,
        "ch_vocals": np.sin(2 * np.pi * 300 * t) * 0.5,
        "ch_bass": np.sin(2 * np.pi * 80 * t) * 0.6,
        "ch_guitar": np.sin(2 * np.pi * 800 * t) * 0.3,
        "ch_keys": np.sin(2 * np.pi * 1000 * t) * 0.25,
    }


@pytest.fixture
def channel_classifications():
    return {
        "ch_kick": "kick",
        "ch_snare": "snare",
        "ch_vocals": "vocals",
        "ch_bass": "bass_guitar",
        "ch_guitar": "electric_guitar",
        "ch_keys": "keys",
    }


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

class TestBusConstants:

    def test_bus_assignments_has_6_buses(self):
        assert len(BUS_ASSIGNMENTS) == 6

    def test_expected_buses_present(self):
        expected = ["drums_bus", "guitar_bus", "vocal_bus", "keys_bus", "bass_bus", "aux_bus"]
        for bus in expected:
            assert bus in BUS_ASSIGNMENTS, f"{bus} missing"

    def test_drums_bus_instruments(self):
        drums = BUS_ASSIGNMENTS["drums_bus"]
        for inst in ("kick", "snare", "hihat", "toms", "overheads", "percussion"):
            assert inst in drums, f"{inst} missing from drums_bus"

    def test_bus_processing_matches_assignments(self):
        """Every bus in BUS_ASSIGNMENTS should have processing params."""
        for bus in BUS_ASSIGNMENTS:
            assert bus in BUS_PROCESSING, f"{bus} missing from BUS_PROCESSING"

    def test_bus_processing_keys(self):
        """Each bus processing dict should have the expected keys."""
        expected_keys = {
            "hpf_freq", "comp_threshold", "comp_ratio",
            "comp_attack_ms", "comp_release_ms", "bus_gain_db",
        }
        for bus, params in BUS_PROCESSING.items():
            for key in expected_keys:
                assert key in params, f"{bus} missing '{key}'"

    def test_master_processing_keys(self):
        expected = {"comp_threshold", "comp_ratio", "comp_attack_ms",
                    "comp_release_ms", "limiter_ceiling_db"}
        for key in expected:
            assert key in MASTER_PROCESSING, f"MASTER_PROCESSING missing '{key}'"


# ---------------------------------------------------------------------------
# assign_channels
# ---------------------------------------------------------------------------

class TestAssignChannels:

    def test_returns_dict(self, mixer, channel_classifications):
        result = mixer.assign_channels(channel_classifications)
        assert isinstance(result, dict)

    def test_kick_goes_to_drums_bus(self, mixer, channel_classifications):
        result = mixer.assign_channels(channel_classifications)
        assert "ch_kick" in result["drums_bus"]

    def test_vocals_go_to_vocal_bus(self, mixer, channel_classifications):
        result = mixer.assign_channels(channel_classifications)
        assert "ch_vocals" in result["vocal_bus"]

    def test_bass_goes_to_bass_bus(self, mixer, channel_classifications):
        result = mixer.assign_channels(channel_classifications)
        assert "ch_bass" in result["bass_bus"]

    def test_unknown_instrument_to_aux(self, mixer):
        result = mixer.assign_channels({"ch_x": "didgeridoo"})
        assert "ch_x" in result["aux_bus"]

    def test_all_channels_assigned(self, mixer, channel_classifications):
        result = mixer.assign_channels(channel_classifications)
        all_assigned = []
        for ch_list in result.values():
            all_assigned.extend(ch_list)
        for ch in channel_classifications:
            assert ch in all_assigned


# ---------------------------------------------------------------------------
# get_bus_for_channel
# ---------------------------------------------------------------------------

class TestGetBusForChannel:

    def test_after_assignment(self, mixer, channel_classifications):
        mixer.assign_channels(channel_classifications)
        assert mixer.get_bus_for_channel("ch_kick") == "drums_bus"
        assert mixer.get_bus_for_channel("ch_vocals") == "vocal_bus"

    def test_unknown_channel_returns_aux(self, mixer):
        assert mixer.get_bus_for_channel("nonexistent") == "aux_bus"


# ---------------------------------------------------------------------------
# compute_bus_mix
# ---------------------------------------------------------------------------

class TestComputeBusMix:

    def test_returns_array(self, mixer, sine_1s):
        result = mixer.compute_bus_mix("drums_bus", [sine_1s])
        assert isinstance(result, np.ndarray)
        assert len(result) > 0

    def test_empty_channels_returns_empty(self, mixer):
        result = mixer.compute_bus_mix("drums_bus", [])
        assert len(result) == 0

    def test_with_gain(self, mixer, sine_1s):
        result = mixer.compute_bus_mix("drums_bus", [sine_1s], channel_gains_db=[0.0])
        assert len(result) == len(sine_1s)

    def test_negative_gain_reduces_level(self, mixer, sine_1s):
        result_0db = mixer.compute_bus_mix("drums_bus", [sine_1s], channel_gains_db=[0.0])
        result_neg = mixer.compute_bus_mix("drums_bus", [sine_1s], channel_gains_db=[-12.0])
        rms_0 = np.sqrt(np.mean(result_0db ** 2))
        rms_neg = np.sqrt(np.mean(result_neg ** 2))
        assert rms_neg < rms_0

    def test_multiple_channels_summed(self, mixer, sine_1s):
        """Two identical channels should sum to higher level than one."""
        one = mixer.compute_bus_mix("drums_bus", [sine_1s], channel_gains_db=[0.0])
        two = mixer.compute_bus_mix("drums_bus", [sine_1s, sine_1s], channel_gains_db=[0.0, 0.0])
        rms_one = np.sqrt(np.mean(one ** 2))
        rms_two = np.sqrt(np.mean(two ** 2))
        # Two channels should be louder; bus processing (compression) may reduce the ratio
        assert rms_two > rms_one * 1.1


# ---------------------------------------------------------------------------
# compute_master
# ---------------------------------------------------------------------------

class TestComputeMaster:

    def test_returns_array(self, mixer, sine_1s):
        master = mixer.compute_master({"drums_bus": sine_1s})
        assert isinstance(master, np.ndarray)
        assert len(master) > 0

    def test_empty_input(self, mixer):
        master = mixer.compute_master({})
        assert len(master) == 0

    def test_limiter_prevents_clipping(self, mixer):
        """Master limiter should prevent output from exceeding ceiling."""
        loud = np.ones(48000, dtype=np.float64) * 5.0  # way over 0 dBFS
        master = mixer.compute_master({"test": loud})
        ceiling = 10.0 ** (MASTER_PROCESSING["limiter_ceiling_db"] / 20.0)
        assert np.max(np.abs(master)) <= ceiling + 1e-6

    def test_accepts_list_input(self, mixer, sine_1s):
        master = mixer.compute_master([sine_1s, sine_1s * 0.5])
        assert len(master) == len(sine_1s)


# ---------------------------------------------------------------------------
# process_full_mix
# ---------------------------------------------------------------------------

class TestProcessFullMix:

    def test_returns_expected_keys(self, mixer, multi_channel_audio, channel_classifications):
        result = mixer.process_full_mix(multi_channel_audio, channel_classifications)
        assert "master" in result
        assert "buses" in result
        assert "assignments" in result

    def test_master_is_array(self, mixer, multi_channel_audio, channel_classifications):
        result = mixer.process_full_mix(multi_channel_audio, channel_classifications)
        assert isinstance(result["master"], np.ndarray)
        assert len(result["master"]) > 0

    def test_buses_is_dict(self, mixer, multi_channel_audio, channel_classifications):
        result = mixer.process_full_mix(multi_channel_audio, channel_classifications)
        assert isinstance(result["buses"], dict)

    def test_with_channel_gains(self, mixer, multi_channel_audio, channel_classifications):
        gains = {ch: -3.0 for ch in multi_channel_audio}
        result = mixer.process_full_mix(
            multi_channel_audio, channel_classifications, channel_gains_db=gains
        )
        assert len(result["master"]) > 0


# ---------------------------------------------------------------------------
# Convenience: assign_channel, get_group_settings, compute_group_mix
# ---------------------------------------------------------------------------

class TestConvenienceAPI:

    def test_assign_channel_known_group(self, mixer):
        mixer.assign_channel("ch1", "drums")
        assert mixer.get_bus_for_channel("ch1") == "drums_bus"

    def test_assign_channel_unknown_group(self, mixer):
        mixer.assign_channel("ch1", "theremin_section")
        assert mixer.get_bus_for_channel("ch1") == "aux_bus"

    def test_get_group_settings_known(self, mixer):
        settings = mixer.get_group_settings("drums")
        assert settings is not None
        assert "hpf_freq" in settings
        assert "comp_threshold" in settings

    def test_get_group_settings_unknown(self, mixer):
        settings = mixer.get_group_settings("nonexistent_group")
        assert settings is None

    def test_compute_group_mix(self, mixer):
        mixer.assign_channel("ch1", "drums")
        mixer.assign_channel("ch2", "vocals")
        features = {
            "ch1": {"rms_db": -18.0, "instrument_type": "kick"},
            "ch2": {"rms_db": -14.0, "instrument_type": "vocals"},
        }
        result = mixer.compute_group_mix(features)
        assert isinstance(result, dict)
        # Should have at least drums and vocals groups
        assert len(result) >= 2
        for group_name, group_info in result.items():
            assert "channels" in group_info
            assert "avg_rms_db" in group_info
            assert "recommended_gain_db" in group_info

    def test_compute_group_mix_gain_bounds(self, mixer):
        mixer.assign_channel("ch1", "drums")
        features = {"ch1": {"rms_db": -50.0}}
        result = mixer.compute_group_mix(features)
        for group_info in result.values():
            assert -20.0 <= group_info["recommended_gain_db"] <= 12.0


# ---------------------------------------------------------------------------
# get_bus_processing
# ---------------------------------------------------------------------------

class TestGetBusProcessing:

    def test_known_group(self, mixer):
        bp = mixer.get_bus_processing("drums")
        assert bp is not None
        assert "hpf_freq" in bp
        assert "compression" in bp
        assert "group_level_db" in bp
        assert "eq_curve" in bp

    def test_compression_sub_dict(self, mixer):
        bp = mixer.get_bus_processing("vocals")
        comp = bp["compression"]
        for key in ("threshold_db", "ratio", "attack_ms", "release_ms"):
            assert key in comp

    def test_eq_curve_is_list(self, mixer):
        bp = mixer.get_bus_processing("bass")
        assert isinstance(bp["eq_curve"], list)
        for band in bp["eq_curve"]:
            assert "freq" in band
            assert "gain_db" in band

    def test_unknown_returns_none(self, mixer):
        assert mixer.get_bus_processing("xyzzy") is None


# ---------------------------------------------------------------------------
# set_bus_param / set_master_param
# ---------------------------------------------------------------------------

class TestParameterMutation:

    def test_set_bus_param(self, mixer):
        mixer.set_bus_param("drums_bus", "comp_threshold", -20.0)
        assert mixer.bus_processing["drums_bus"]["comp_threshold"] == -20.0

    def test_set_bus_param_unknown_bus(self, mixer):
        """Setting a param on an unknown bus should not crash."""
        mixer.set_bus_param("fake_bus", "comp_threshold", -10.0)
        assert "fake_bus" not in mixer.bus_processing

    def test_set_master_param(self, mixer):
        mixer.set_master_param("comp_ratio", 3.0)
        assert mixer.master_processing["comp_ratio"] == 3.0
