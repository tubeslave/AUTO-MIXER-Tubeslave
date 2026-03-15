"""
Tests for MixerState management (channel parameters, snapshots, diffing).

Covers:
- get/set channel parameters via dotted path
- Snapshot save, recall, list, delete
- State diff between two MixerState instances
- OSC address parsing and update
"""

import pytest

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
)


class TestSetGetChannelParam:
    """Tests for channel parameter access via dotted path."""

    def test_set_get_channel_param_fader(self):
        """Setting and getting the fader value should round-trip."""
        ms = MixerState(num_channels=4)
        ms.set(1, "fader", -10.0)
        assert ms.get(1, "fader") == -10.0

    def test_set_get_channel_param_mute(self):
        """Setting and getting the mute value should round-trip."""
        ms = MixerState(num_channels=4)
        ms.set(1, "mute", True)
        assert ms.get(1, "mute") is True

    def test_set_get_compressor_threshold(self):
        """Setting compressor.threshold via dotted path should work."""
        ms = MixerState(num_channels=4)
        ms.set(2, "compressor.threshold", -30.0)
        assert ms.get(2, "compressor.threshold") == -30.0

    def test_set_get_eq_band_gain(self):
        """Setting eq.2.gain via dotted path should work."""
        ms = MixerState(num_channels=4)
        ms.set(1, "eq.2.gain", 3.5)
        assert ms.get(1, "eq.2.gain") == 3.5

    def test_set_get_gate_attack(self):
        """Setting gate.attack via dotted path should work."""
        ms = MixerState(num_channels=4)
        ms.set(1, "gate.attack", 5.0)
        assert ms.get(1, "gate.attack") == 5.0

    def test_set_get_send_level(self):
        """Setting send.1.level via dotted path should work."""
        ms = MixerState(num_channels=4)
        ms.set(1, "send.1.level", -6.0)
        assert ms.get(1, "send.1.level") == -6.0

    def test_set_get_input_trim(self):
        """Setting input.trim via dotted path should work."""
        ms = MixerState(num_channels=4)
        ms.set(1, "input.trim", 6.0)
        assert ms.get(1, "input.trim") == 6.0

    def test_set_get_filter_low_cut_freq(self):
        """Setting filter.low_cut_freq via dotted path should work."""
        ms = MixerState(num_channels=4)
        ms.set(1, "filter.low_cut_freq", 120.0)
        assert ms.get(1, "filter.low_cut_freq") == 120.0

    def test_unknown_param_raises_key_error(self):
        """Getting an unknown parameter should raise KeyError."""
        ms = MixerState(num_channels=4)
        with pytest.raises(KeyError):
            ms.get(1, "nonexistent_param")

    def test_unknown_channel_raises_key_error(self):
        """Getting a param for a non-existent channel should raise KeyError."""
        ms = MixerState(num_channels=4)
        with pytest.raises(KeyError):
            ms.get(99, "fader")

    def test_set_returns_old_value(self):
        """set() should return the previous value."""
        ms = MixerState(num_channels=4)
        old = ms.set(1, "fader", -5.0)
        assert old == -144.0  # Default fader value

    def test_channel_name(self):
        """Setting and getting channel name should work."""
        ms = MixerState(num_channels=4)
        ms.set(1, "name", "Vocals")
        assert ms.get(1, "name") == "Vocals"

    def test_pan_parameter(self):
        """Setting and getting pan should work."""
        ms = MixerState(num_channels=4)
        ms.set(1, "pan", 50.0)
        assert ms.get(1, "pan") == 50.0


class TestSnapshotSaveRecall:
    """Tests for snapshot save, recall, list, and delete."""

    def test_snapshot_save_recall(self):
        """Saving and recalling a snapshot should restore state."""
        ms = MixerState(num_channels=4)
        ms.set(1, "fader", -10.0)
        ms.set(2, "fader", -20.0)

        ms.snapshot_save("test_snap")

        # Modify state
        ms.set(1, "fader", 0.0)
        ms.set(2, "fader", 0.0)

        # Recall
        result = ms.snapshot_recall("test_snap")
        assert result is True
        assert ms.get(1, "fader") == -10.0
        assert ms.get(2, "fader") == -20.0

    def test_snapshot_recall_nonexistent(self):
        """Recalling a non-existent snapshot should return False."""
        ms = MixerState(num_channels=4)
        result = ms.snapshot_recall("does_not_exist")
        assert result is False

    def test_snapshot_list(self):
        """snapshot_list should return all saved snapshot names."""
        ms = MixerState(num_channels=4)
        ms.snapshot_save("snap_a")
        ms.snapshot_save("snap_b")

        snapshots = ms.snapshot_list()
        names = [s["name"] for s in snapshots]
        assert "snap_a" in names
        assert "snap_b" in names
        assert len(snapshots) == 2

    def test_snapshot_delete(self):
        """snapshot_delete should remove the named snapshot."""
        ms = MixerState(num_channels=4)
        ms.snapshot_save("to_delete")
        assert ms.snapshot_delete("to_delete") is True
        assert ms.snapshot_delete("to_delete") is False  # Already gone

    def test_snapshot_save_empty_name_raises(self):
        """Saving a snapshot with empty name should raise ValueError."""
        ms = MixerState(num_channels=4)
        with pytest.raises(ValueError):
            ms.snapshot_save("")

    def test_snapshot_overwrites_existing(self):
        """Saving a snapshot with the same name should overwrite it."""
        ms = MixerState(num_channels=4)
        ms.set(1, "fader", -10.0)
        ms.snapshot_save("overwrite_me")

        ms.set(1, "fader", -5.0)
        ms.snapshot_save("overwrite_me")

        # Recall should get the latest
        ms.set(1, "fader", 0.0)
        ms.snapshot_recall("overwrite_me")
        assert ms.get(1, "fader") == -5.0


class TestStateDiff:
    """Tests for state diffing between MixerState instances."""

    def test_state_diff_identical(self):
        """Diffing identical states should produce no differences."""
        ms1 = MixerState(num_channels=4)
        ms2 = MixerState(num_channels=4)
        diffs = ms1.diff(ms2)
        assert len(diffs) == 0

    def test_state_diff_detects_changes(self):
        """Diffing should detect parameter differences."""
        ms1 = MixerState(num_channels=4)
        ms2 = MixerState(num_channels=4)

        ms1.set(1, "fader", -10.0)
        ms2.set(1, "fader", -20.0)

        diffs = ms1.diff(ms2)
        assert len(diffs) > 0

        # Find the fader diff
        fader_diffs = [d for d in diffs if d["channel"] == 1 and "fader" in d["param"]]
        assert len(fader_diffs) > 0
        fader_diff = fader_diffs[0]
        assert fader_diff["current"] == -10.0
        assert fader_diff["other"] == -20.0

    def test_diff_from_snapshot(self):
        """diff_from_snapshot should detect changes since a saved snapshot."""
        ms = MixerState(num_channels=4)
        ms.set(1, "fader", -10.0)
        ms.snapshot_save("baseline")

        ms.set(1, "fader", -5.0)
        diffs = ms.diff_from_snapshot("baseline")

        assert diffs is not None
        assert len(diffs) > 0

    def test_diff_from_snapshot_nonexistent(self):
        """diff_from_snapshot with non-existent snapshot should return None."""
        ms = MixerState(num_channels=4)
        result = ms.diff_from_snapshot("nope")
        assert result is None


class TestOSCUpdate:
    """Tests for update_from_osc address parsing."""

    def test_update_from_osc_fader(self):
        """OSC address /ch/1/fdr should update the fader."""
        ms = MixerState(num_channels=4)
        result = ms.update_from_osc("/ch/1/fdr", -12.0)
        assert result is True
        assert ms.get(1, "fader") == -12.0

    def test_update_from_osc_eq(self):
        """OSC address /ch/2/eq/1g should update EQ band 1 gain."""
        ms = MixerState(num_channels=4)
        result = ms.update_from_osc("/ch/2/eq/1g", 3.0)
        assert result is True
        assert ms.get(2, "eq.1.gain") == 3.0

    def test_update_from_osc_compressor(self):
        """OSC address /ch/1/dyn/thr should update compressor threshold."""
        ms = MixerState(num_channels=4)
        result = ms.update_from_osc("/ch/1/dyn/thr", -25.0)
        assert result is True
        assert ms.get(1, "compressor.threshold") == -25.0

    def test_update_from_osc_invalid(self):
        """Invalid OSC address should return False."""
        ms = MixerState(num_channels=4)
        result = ms.update_from_osc("/invalid/path", 0.0)
        assert result is False

    def test_update_from_osc_gate(self):
        """OSC address /ch/1/gate/thr should update gate threshold."""
        ms = MixerState(num_channels=4)
        result = ms.update_from_osc("/ch/1/gate/thr", -50.0)
        assert result is True
        assert ms.get(1, "gate.threshold") == -50.0


class TestChannelStateDataclass:
    """Tests for the ChannelState dataclass serialization."""

    def test_channel_state_to_dict(self):
        """ChannelState.to_dict() should produce a serializable dict."""
        ch = ChannelState(channel_id=1, name="Kick", fader=-10.0)
        d = ch.to_dict()
        assert isinstance(d, dict)
        assert d["channel_id"] == 1
        assert d["name"] == "Kick"
        assert d["fader"] == -10.0

    def test_channel_state_from_dict(self):
        """ChannelState.from_dict() should reconstruct the object."""
        original = ChannelState(channel_id=3, name="Bass", fader=-6.0, mute=True)
        d = original.to_dict()
        restored = ChannelState.from_dict(d)

        assert restored.channel_id == 3
        assert restored.name == "Bass"
        assert restored.fader == -6.0
        assert restored.mute is True

    def test_eq_band_state_round_trip(self):
        """EQBandState should serialize and deserialize correctly."""
        band = EQBandState(gain=3.0, frequency=2000.0, q=2.5, band_type="PEQ")
        d = band.to_dict()
        restored = EQBandState.from_dict(d)
        assert restored.gain == 3.0
        assert restored.frequency == 2000.0
        assert restored.q == 2.5

    def test_compressor_state_round_trip(self):
        """CompressorState should serialize and deserialize correctly."""
        comp = CompressorState(on=True, threshold=-18.0, ratio=6.0, attack=5.0)
        d = comp.to_dict()
        restored = CompressorState.from_dict(d)
        assert restored.on is True
        assert restored.threshold == -18.0
        assert restored.ratio == 6.0

    def test_mixer_state_channel_ids(self):
        """MixerState should have the expected channel IDs."""
        ms = MixerState(num_channels=8)
        assert ms.channel_ids == list(range(1, 9))

    def test_mixer_state_get_channel(self):
        """get_channel should return a ChannelState instance."""
        ms = MixerState(num_channels=4)
        ch = ms.get_channel(1)
        assert isinstance(ch, ChannelState)
        assert ch.channel_id == 1


class TestChangeListeners:
    """Tests for the change listener mechanism."""

    def test_listener_called_on_set(self):
        """Change listeners should be called when a parameter is set."""
        ms = MixerState(num_channels=4)
        changes = []

        def on_change(ch_id, param, old, new):
            changes.append((ch_id, param, old, new))

        ms.add_listener(on_change)
        ms.set(1, "fader", -10.0)

        assert len(changes) == 1
        assert changes[0] == (1, "fader", -144.0, -10.0)

    def test_remove_listener(self):
        """Removing a listener should stop it from receiving updates."""
        ms = MixerState(num_channels=4)
        changes = []

        def on_change(ch_id, param, old, new):
            changes.append(1)

        ms.add_listener(on_change)
        ms.set(1, "fader", -10.0)
        assert len(changes) == 1

        ms.remove_listener(on_change)
        ms.set(1, "fader", -5.0)
        assert len(changes) == 1  # Not called again
