"""
Tests for backend/mixer_state.py — MixerState, ChannelState, dataclasses,
_resolve_param, snapshots, update_from_osc, export/import JSON, diff.

All tests work without hardware or network.
"""

import json
import os
import pytest

try:
    from mixer_state import (
        MixerState,
        ChannelState,
        EQBandState,
        CompressorState,
        GateState,
        FilterState,
        SendState,
        InputState,
        InsertState,
        _resolve_param,
        _parse_eq_band_index,
    )
except ImportError:
    pytest.skip("mixer_state module not importable", allow_module_level=True)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def mixer():
    """A fresh MixerState with 4 channels for faster tests."""
    return MixerState(num_channels=4)


@pytest.fixture
def full_mixer():
    """A fresh MixerState with all 40 channels."""
    return MixerState(num_channels=40)


# ---------------------------------------------------------------------------
# ChannelState dataclass tests
# ---------------------------------------------------------------------------

class TestChannelStateDataclass:

    def test_default_values(self):
        ch = ChannelState(channel_id=1)
        assert ch.channel_id == 1
        assert ch.fader == -144.0
        assert ch.mute is False
        assert ch.pan == 0.0

    def test_to_dict(self):
        ch = ChannelState(channel_id=1, name="Vocals", fader=-5.0)
        d = ch.to_dict()
        assert d["channel_id"] == 1
        assert d["name"] == "Vocals"
        assert d["fader"] == -5.0
        assert "compressor" in d
        assert "gate" in d
        assert "eq_bands" in d

    def test_from_dict_roundtrip(self):
        ch = ChannelState(channel_id=3, name="Kick", fader=-10.0, mute=True)
        d = ch.to_dict()
        restored = ChannelState.from_dict(d)
        assert restored.channel_id == 3
        assert restored.name == "Kick"
        assert restored.fader == -10.0
        assert restored.mute is True

    def test_from_dict_defaults(self):
        ch = ChannelState.from_dict({})
        assert ch.channel_id == 0
        assert ch.fader == -144.0
        assert ch.eq_on is False

    def test_sends_default(self):
        ch = ChannelState(channel_id=1)
        assert len(ch.sends) == 16
        assert len(ch.main_sends) == 4


# ---------------------------------------------------------------------------
# EQ and sub-dataclass tests
# ---------------------------------------------------------------------------

class TestSubDataclasses:

    def test_eq_band_roundtrip(self):
        band = EQBandState(gain=3.5, frequency=800.0, q=2.0, band_type="PEQ")
        d = band.to_dict()
        restored = EQBandState.from_dict(d)
        assert restored.gain == 3.5
        assert restored.frequency == 800.0

    def test_compressor_roundtrip(self):
        comp = CompressorState(on=True, threshold=-30.0, ratio=8.0)
        d = comp.to_dict()
        restored = CompressorState.from_dict(d)
        assert restored.on is True
        assert restored.threshold == -30.0
        assert restored.ratio == 8.0

    def test_gate_roundtrip(self):
        gate = GateState(on=True, threshold=-50.0)
        d = gate.to_dict()
        restored = GateState.from_dict(d)
        assert restored.on is True
        assert restored.threshold == -50.0

    def test_send_roundtrip(self):
        send = SendState(on=True, level=-5.0, pan=50.0, mode="PRE")
        d = send.to_dict()
        restored = SendState.from_dict(d)
        assert restored.on is True
        assert restored.level == -5.0
        assert restored.mode == "PRE"


# ---------------------------------------------------------------------------
# _resolve_param tests
# ---------------------------------------------------------------------------

class TestResolveParam:

    def test_simple_fader(self):
        ch = ChannelState(channel_id=1)
        parent, attr = _resolve_param(ch, "fader")
        assert parent is ch
        assert attr == "fader"

    def test_alias_fdr(self):
        ch = ChannelState(channel_id=1)
        parent, attr = _resolve_param(ch, "fdr")
        assert attr == "fader"

    def test_compressor_threshold(self):
        ch = ChannelState(channel_id=1)
        parent, attr = _resolve_param(ch, "compressor.threshold")
        assert parent is ch.compressor
        assert attr == "threshold"

    def test_gate_attack(self):
        ch = ChannelState(channel_id=1)
        parent, attr = _resolve_param(ch, "gate.attack")
        assert parent is ch.gate
        assert attr == "attack"

    def test_eq_band_gain(self):
        ch = ChannelState(channel_id=1)
        parent, attr = _resolve_param(ch, "eq.2.gain")
        assert parent is ch.eq_bands[2]
        assert attr == "gain"

    def test_eq_on(self):
        ch = ChannelState(channel_id=1)
        parent, attr = _resolve_param(ch, "eq.on")
        assert parent is ch
        assert attr == "eq_on"

    def test_send_level(self):
        ch = ChannelState(channel_id=1)
        parent, attr = _resolve_param(ch, "send.3.level")
        assert parent is ch.sends[3]
        assert attr == "level"

    def test_input_trim(self):
        ch = ChannelState(channel_id=1)
        parent, attr = _resolve_param(ch, "input.trim")
        assert parent is ch.input_state
        assert attr == "trim"

    def test_filter_low_cut_on(self):
        ch = ChannelState(channel_id=1)
        parent, attr = _resolve_param(ch, "filter.low_cut_on")
        assert parent is ch.filter_state
        assert attr == "low_cut_on"

    def test_unknown_param_raises(self):
        ch = ChannelState(channel_id=1)
        with pytest.raises(KeyError):
            _resolve_param(ch, "nonexistent_param")


# ---------------------------------------------------------------------------
# _parse_eq_band_index tests
# ---------------------------------------------------------------------------

class TestParseEQBandIndex:

    def test_low(self):
        assert _parse_eq_band_index("low") == 0
        assert _parse_eq_band_index("l") == 0

    def test_high(self):
        assert _parse_eq_band_index("high") == 5
        assert _parse_eq_band_index("h") == 5

    def test_numeric(self):
        assert _parse_eq_band_index("1") == 1
        assert _parse_eq_band_index("4") == 4

    def test_out_of_range(self):
        with pytest.raises(KeyError):
            _parse_eq_band_index("99")


# ---------------------------------------------------------------------------
# MixerState get/set tests
# ---------------------------------------------------------------------------

class TestMixerStateGetSet:

    def test_get_fader(self, mixer):
        assert mixer.get(1, "fader") == -144.0

    def test_set_fader(self, mixer):
        old = mixer.set(1, "fader", -5.0)
        assert old == -144.0
        assert mixer.get(1, "fader") == -5.0

    def test_set_compressor_threshold(self, mixer):
        mixer.set(1, "compressor.threshold", -30.0)
        assert mixer.get(1, "compressor.threshold") == -30.0

    def test_set_eq_band(self, mixer):
        mixer.set(1, "eq.2.gain", 3.5)
        assert mixer.get(1, "eq.2.gain") == 3.5

    def test_get_nonexistent_channel_raises(self, mixer):
        with pytest.raises(KeyError):
            mixer.get(99, "fader")

    def test_get_all_channels(self, mixer):
        channels = mixer.get_all_channels()
        assert len(channels) == 4

    def test_channel_ids(self, mixer):
        assert mixer.channel_ids == [1, 2, 3, 4]


# ---------------------------------------------------------------------------
# MixerState update_from_osc tests
# ---------------------------------------------------------------------------

class TestMixerStateUpdateFromOSC:

    def test_update_fader(self, mixer):
        result = mixer.update_from_osc("/ch/1/fdr", -5.0)
        assert result is True
        assert mixer.get(1, "fader") == -5.0

    def test_update_mute(self, mixer):
        result = mixer.update_from_osc("/ch/2/mute", True)
        assert result is True
        assert mixer.get(2, "mute") is True

    def test_update_compressor(self, mixer):
        result = mixer.update_from_osc("/ch/1/dyn/thr", -25.0)
        assert result is True
        assert mixer.get(1, "compressor.threshold") == -25.0

    def test_update_eq_band(self, mixer):
        result = mixer.update_from_osc("/ch/1/eq/2g", 4.0)
        assert result is True
        assert mixer.get(1, "eq.2.gain") == 4.0

    def test_update_gate(self, mixer):
        result = mixer.update_from_osc("/ch/1/gate/thr", -50.0)
        assert result is True
        assert mixer.get(1, "gate.threshold") == -50.0

    def test_unknown_osc_address(self, mixer):
        result = mixer.update_from_osc("/unknown/address", 0)
        assert result is False

    def test_invalid_channel(self, mixer):
        result = mixer.update_from_osc("/ch/99/fdr", 0)
        assert result is False


# ---------------------------------------------------------------------------
# MixerState snapshot tests
# ---------------------------------------------------------------------------

class TestMixerStateSnapshots:

    def test_snapshot_save_and_list(self, mixer):
        mixer.snapshot_save("test_snap")
        snaps = mixer.snapshot_list()
        assert len(snaps) == 1
        assert snaps[0]["name"] == "test_snap"

    def test_snapshot_recall(self, mixer):
        mixer.set(1, "fader", -5.0)
        mixer.snapshot_save("before_change")
        mixer.set(1, "fader", 0.0)
        assert mixer.get(1, "fader") == 0.0
        result = mixer.snapshot_recall("before_change")
        assert result is True
        assert mixer.get(1, "fader") == -5.0

    def test_snapshot_recall_nonexistent(self, mixer):
        result = mixer.snapshot_recall("nonexistent")
        assert result is False

    def test_snapshot_delete(self, mixer):
        mixer.snapshot_save("to_delete")
        assert mixer.snapshot_delete("to_delete") is True
        assert mixer.snapshot_delete("to_delete") is False

    def test_snapshot_empty_name_raises(self, mixer):
        with pytest.raises(ValueError):
            mixer.snapshot_save("")


# ---------------------------------------------------------------------------
# MixerState export/import JSON tests
# ---------------------------------------------------------------------------

class TestMixerStateJSON:

    def test_export_json_string(self, mixer):
        mixer.set(1, "fader", -5.0)
        json_str = mixer.export_json()
        data = json.loads(json_str)
        assert "channels" in data
        assert "version" in data

    def test_export_json_file(self, mixer, tmp_dir):
        mixer.set(1, "fader", -5.0)
        path = os.path.join(tmp_dir, "state.json")
        mixer.export_json(path=path)
        assert os.path.isfile(path)

    def test_import_json_string(self, mixer):
        mixer.set(1, "fader", -5.0)
        json_str = mixer.export_json()
        # Create a new mixer and import
        mixer2 = MixerState(num_channels=4)
        mixer2.import_json(json_str=json_str)
        assert mixer2.get(1, "fader") == -5.0

    def test_import_json_file(self, mixer, tmp_dir):
        mixer.set(1, "fader", -5.0)
        path = os.path.join(tmp_dir, "state.json")
        mixer.export_json(path=path)
        mixer2 = MixerState(num_channels=4)
        mixer2.import_json(path=path)
        assert mixer2.get(1, "fader") == -5.0

    def test_import_json_no_data_raises(self, mixer):
        with pytest.raises(ValueError):
            mixer.import_json()


# ---------------------------------------------------------------------------
# MixerState diff tests
# ---------------------------------------------------------------------------

class TestMixerStateDiff:

    def test_diff_identical(self, mixer):
        mixer2 = MixerState(num_channels=4)
        diffs = mixer.diff(mixer2)
        assert len(diffs) == 0

    def test_diff_detects_change(self, mixer):
        mixer2 = MixerState(num_channels=4)
        mixer.set(1, "fader", -5.0)
        diffs = mixer.diff(mixer2)
        assert len(diffs) > 0
        fader_diff = [d for d in diffs if "fader" in d["param"]]
        assert len(fader_diff) > 0

    def test_diff_from_snapshot(self, mixer):
        mixer.snapshot_save("snap1")
        mixer.set(1, "fader", -5.0)
        diffs = mixer.diff_from_snapshot("snap1")
        assert diffs is not None
        assert len(diffs) > 0

    def test_diff_from_nonexistent_snapshot(self, mixer):
        result = mixer.diff_from_snapshot("nonexistent")
        assert result is None


# ---------------------------------------------------------------------------
# MixerState change listeners
# ---------------------------------------------------------------------------

class TestMixerStateListeners:

    def test_add_and_notify_listener(self, mixer):
        changes = []

        def listener(ch_id, param, old, new):
            changes.append((ch_id, param, old, new))

        mixer.add_listener(listener)
        mixer.set(1, "fader", -5.0)
        assert len(changes) == 1
        assert changes[0] == (1, "fader", -144.0, -5.0)

    def test_remove_listener(self, mixer):
        changes = []

        def listener(ch_id, param, old, new):
            changes.append(True)

        mixer.add_listener(listener)
        mixer.remove_listener(listener)
        mixer.set(1, "fader", -5.0)
        assert len(changes) == 0

    def test_no_notification_on_same_value(self, mixer):
        """Setting a param to its current value should not notify."""
        changes = []

        def listener(ch_id, param, old, new):
            changes.append(True)

        mixer.add_listener(listener)
        current = mixer.get(1, "fader")
        mixer.set(1, "fader", current)
        assert len(changes) == 0


# ---------------------------------------------------------------------------
# MixerState _osc_to_param_path tests
# ---------------------------------------------------------------------------

class TestOSCToParamPath:

    def test_fdr(self):
        assert MixerState._osc_to_param_path("fdr") == "fader"

    def test_mute(self):
        assert MixerState._osc_to_param_path("mute") == "mute"

    def test_pan(self):
        assert MixerState._osc_to_param_path("pan") == "pan"

    def test_eq_on(self):
        assert MixerState._osc_to_param_path("eq/on") == "eq.on"

    def test_eq_band_gain(self):
        assert MixerState._osc_to_param_path("eq/2g") == "eq.2.gain"

    def test_eq_band_frequency(self):
        assert MixerState._osc_to_param_path("eq/1f") == "eq.1.frequency"

    def test_dyn_threshold(self):
        assert MixerState._osc_to_param_path("dyn/thr") == "compressor.threshold"

    def test_gate_attack(self):
        assert MixerState._osc_to_param_path("gate/att") == "gate.attack"

    def test_filter_low_cut(self):
        assert MixerState._osc_to_param_path("flt/lc") == "filter.low_cut_on"

    def test_unknown_returns_none(self):
        assert MixerState._osc_to_param_path("totally/unknown") is None
