"""Tests for ml.subgroup_mixer -- hierarchical subgroup mixing and routing."""
import pytest
import numpy as np
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'backend'))


class TestDefaultSubgroups:
    """Tests for the DEFAULT_SUBGROUPS constant."""

    def test_default_subgroups_not_empty(self):
        from ml.subgroup_mixer import DEFAULT_SUBGROUPS
        assert len(DEFAULT_SUBGROUPS) > 0

    def test_expected_groups_present(self):
        from ml.subgroup_mixer import DEFAULT_SUBGROUPS
        expected = ['drums', 'bass', 'guitars', 'keys', 'vocals']
        for group in expected:
            assert group in DEFAULT_SUBGROUPS

    def test_drums_group_instruments(self):
        from ml.subgroup_mixer import DEFAULT_SUBGROUPS
        drums = DEFAULT_SUBGROUPS['drums']
        assert 'kick' in drums.instruments
        assert 'snare' in drums.instruments
        assert 'hi_hat' in drums.instruments

    def test_subgroup_config_fields(self):
        from ml.subgroup_mixer import DEFAULT_SUBGROUPS
        vocals = DEFAULT_SUBGROUPS['vocals']
        assert hasattr(vocals, 'name')
        assert hasattr(vocals, 'instruments')
        assert hasattr(vocals, 'bus_gain_db')
        assert hasattr(vocals, 'bus_pan')
        assert hasattr(vocals, 'bus_comp_threshold')
        assert hasattr(vocals, 'bus_comp_ratio')
        assert hasattr(vocals, 'mute')
        assert hasattr(vocals, 'solo')


class TestSubgroupConfig:
    """Tests for the SubgroupConfig dataclass."""

    def test_creation_defaults(self):
        from ml.subgroup_mixer import SubgroupConfig
        config = SubgroupConfig(name='test', instruments=['a', 'b'])
        assert config.bus_gain_db == 0.0
        assert config.bus_pan == 0.0
        assert config.mute is False
        assert config.solo is False

    def test_creation_custom(self):
        from ml.subgroup_mixer import SubgroupConfig
        config = SubgroupConfig(
            name='custom', instruments=['x'], bus_gain_db=-3.0,
            bus_pan=0.5, bus_comp_threshold=-15.0, bus_comp_ratio=4.0,
        )
        assert config.bus_gain_db == -3.0
        assert config.bus_comp_ratio == 4.0


class TestChannelAssignment:
    """Tests for the ChannelAssignment dataclass."""

    def test_creation(self):
        from ml.subgroup_mixer import ChannelAssignment
        assignment = ChannelAssignment(
            channel_id=1, instrument='kick', subgroup='drums',
            gain_db=-6.0, pan=-0.3,
        )
        assert assignment.channel_id == 1
        assert assignment.instrument == 'kick'
        assert assignment.subgroup == 'drums'
        assert assignment.mute is False


class TestSubgroupMixer:
    """Tests for the SubgroupMixer."""

    def test_instantiation_default(self):
        from ml.subgroup_mixer import SubgroupMixer, DEFAULT_SUBGROUPS
        mixer = SubgroupMixer()
        assert len(mixer.subgroups) == len(DEFAULT_SUBGROUPS)

    def test_instantiation_custom(self):
        from ml.subgroup_mixer import SubgroupMixer, SubgroupConfig
        custom = {
            'group_a': SubgroupConfig('group_a', ['inst_1', 'inst_2']),
        }
        mixer = SubgroupMixer(subgroups=custom)
        assert 'group_a' in mixer.subgroups
        assert len(mixer.subgroups) == 1

    def test_assign_channel(self):
        from ml.subgroup_mixer import SubgroupMixer
        mixer = SubgroupMixer()
        mixer.assign_channel(channel_id=0, instrument='kick', gain_db=-3.0)
        assert 0 in mixer.channel_assignments
        assert mixer.channel_assignments[0].subgroup == 'drums'
        assert mixer.channel_assignments[0].gain_db == -3.0

    def test_assign_unknown_instrument_to_fx_ambient(self):
        from ml.subgroup_mixer import SubgroupMixer
        mixer = SubgroupMixer()
        mixer.assign_channel(channel_id=5, instrument='theremin')
        assert mixer.channel_assignments[5].subgroup == 'fx_ambient'

    def test_remove_channel(self):
        from ml.subgroup_mixer import SubgroupMixer
        mixer = SubgroupMixer()
        mixer.assign_channel(0, 'kick')
        mixer.remove_channel(0)
        assert 0 not in mixer.channel_assignments

    def test_remove_nonexistent_channel(self):
        from ml.subgroup_mixer import SubgroupMixer
        mixer = SubgroupMixer()
        # Should not raise
        mixer.remove_channel(999)

    def test_get_group_channels(self):
        from ml.subgroup_mixer import SubgroupMixer
        mixer = SubgroupMixer()
        mixer.assign_channel(0, 'kick')
        mixer.assign_channel(1, 'snare')
        mixer.assign_channel(2, 'lead_vocal')
        drums = mixer.get_group_channels('drums')
        assert len(drums) == 2
        vocals = mixer.get_group_channels('vocals')
        assert len(vocals) == 1

    def test_get_bus_levels(self):
        from ml.subgroup_mixer import SubgroupMixer
        mixer = SubgroupMixer()
        mixer.assign_channel(0, 'kick')
        mixer.assign_channel(1, 'snare')
        channel_levels = {0: -12.0, 1: -18.0}
        bus_levels = mixer.get_bus_levels(channel_levels)
        assert 'drums' in bus_levels
        assert bus_levels['drums'] > -100.0
        # Groups with no channels should be -100
        assert bus_levels['bass'] == -100.0

    def test_get_mix_bus_level(self):
        from ml.subgroup_mixer import SubgroupMixer
        mixer = SubgroupMixer()
        mixer.assign_channel(0, 'kick')
        channel_levels = {0: -12.0}
        bus_levels = mixer.get_bus_levels(channel_levels)
        mix_level = mixer.get_mix_bus_level(bus_levels)
        assert isinstance(mix_level, float)
        assert mix_level > -100.0

    def test_apply_solo(self):
        from ml.subgroup_mixer import SubgroupMixer
        mixer = SubgroupMixer()
        mixer.apply_solo('drums')
        assert mixer.subgroups['drums'].solo is True
        # Other groups should be muted
        assert mixer.subgroups['vocals'].mute is True
        assert mixer.subgroups['bass'].mute is True

    def test_clear_solo(self):
        from ml.subgroup_mixer import SubgroupMixer
        mixer = SubgroupMixer()
        mixer.apply_solo('drums')
        mixer.clear_solo()
        for config in mixer.subgroups.values():
            assert config.solo is False
            assert config.mute is False

    def test_get_routing_map(self):
        from ml.subgroup_mixer import SubgroupMixer
        mixer = SubgroupMixer()
        mixer.assign_channel(0, 'kick')
        mixer.assign_channel(1, 'lead_vocal')
        routing = mixer.get_routing_map()
        assert 0 in routing['drums']
        assert 1 in routing['vocals']

    def test_to_osc_commands(self):
        from ml.subgroup_mixer import SubgroupMixer
        mixer = SubgroupMixer()
        commands = mixer.to_osc_commands()
        assert len(commands) == len(mixer.subgroups)
        for cmd in commands:
            assert cmd['type'] == 'bus_config'
            assert 'group' in cmd
            assert 'gain_db' in cmd
            assert 'mute' in cmd
