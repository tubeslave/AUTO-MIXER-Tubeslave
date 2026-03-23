"""
Hierarchical subgroup mixer -- groups channels into submixes (drums, guitars, vocals, etc.)
and applies bus-level processing.
"""
import numpy as np
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field


@dataclass
class SubgroupConfig:
    """Configuration for a submix group."""
    name: str
    instruments: List[str]
    bus_gain_db: float = 0.0
    bus_pan: float = 0.0  # -1 to 1
    bus_comp_threshold: float = -20.0
    bus_comp_ratio: float = 2.0
    mute: bool = False
    solo: bool = False


DEFAULT_SUBGROUPS: Dict[str, SubgroupConfig] = {
    'drums': SubgroupConfig(
        'drums',
        ['kick', 'snare', 'hi_hat', 'toms', 'overheads', 'room_mics', 'percussion'],
        bus_comp_threshold=-18.0, bus_comp_ratio=3.0,
    ),
    'bass': SubgroupConfig('bass', ['bass_guitar'], bus_gain_db=-2.0),
    'guitars': SubgroupConfig(
        'guitars', ['electric_guitar', 'acoustic_guitar'],
        bus_comp_threshold=-16.0,
    ),
    'keys': SubgroupConfig('keys', ['keys_piano', 'synth', 'organ']),
    'vocals': SubgroupConfig(
        'vocals', ['lead_vocal', 'backing_vocal', 'choir'],
        bus_comp_threshold=-14.0, bus_comp_ratio=2.5,
    ),
    'brass_winds': SubgroupConfig(
        'brass_winds', ['brass', 'woodwind', 'strings'],
    ),
    'fx_ambient': SubgroupConfig(
        'fx_ambient', ['ambient_mic', 'audience', 'room_mics'],
        bus_gain_db=-6.0,
    ),
    'playback': SubgroupConfig(
        'playback', ['dj_playback', 'click_track'],
    ),
}


@dataclass
class ChannelAssignment:
    """Assignment of a channel to a subgroup."""
    channel_id: int
    instrument: str
    subgroup: str
    gain_db: float = 0.0
    pan: float = 0.0
    mute: bool = False


class SubgroupMixer:
    """Manages hierarchical submix routing and processing."""

    def __init__(self, subgroups: Optional[Dict[str, SubgroupConfig]] = None):
        self.subgroups = subgroups or dict(DEFAULT_SUBGROUPS)
        self.channel_assignments: Dict[int, ChannelAssignment] = {}
        self._instrument_to_group: Dict[str, str] = {}
        self._rebuild_instrument_map()

    def _rebuild_instrument_map(self):
        self._instrument_to_group.clear()
        for group_name, config in self.subgroups.items():
            for inst in config.instruments:
                self._instrument_to_group[inst] = group_name

    def assign_channel(self, channel_id: int, instrument: str,
                       gain_db: float = 0.0, pan: float = 0.0):
        group = self._instrument_to_group.get(instrument, 'fx_ambient')
        self.channel_assignments[channel_id] = ChannelAssignment(
            channel_id=channel_id, instrument=instrument,
            subgroup=group, gain_db=gain_db, pan=pan,
        )

    def remove_channel(self, channel_id: int):
        self.channel_assignments.pop(channel_id, None)

    def get_group_channels(self, group_name: str) -> List[ChannelAssignment]:
        return [a for a in self.channel_assignments.values()
                if a.subgroup == group_name]

    def get_bus_levels(self,
                       channel_levels: Dict[int, float]) -> Dict[str, float]:
        bus_levels = {}
        for group_name, config in self.subgroups.items():
            channels = self.get_group_channels(group_name)
            if not channels:
                bus_levels[group_name] = -100.0
                continue
            energies = []
            for ch in channels:
                if ch.mute or config.mute:
                    continue
                level_db = channel_levels.get(ch.channel_id, -100.0) + ch.gain_db
                energies.append(10 ** (level_db / 10))
            if energies:
                total_energy = sum(energies)
                bus_levels[group_name] = (
                    10 * np.log10(total_energy + 1e-12) + config.bus_gain_db
                )
            else:
                bus_levels[group_name] = -100.0
        return bus_levels

    def get_mix_bus_level(self, bus_levels: Dict[str, float]) -> float:
        energies = []
        for group_name, level in bus_levels.items():
            config = self.subgroups.get(group_name)
            if config and not config.mute:
                energies.append(10 ** (level / 10))
        if energies:
            return 10 * np.log10(sum(energies) + 1e-12)
        return -100.0

    def apply_solo(self, group_name: str, solo: bool = True):
        if group_name in self.subgroups:
            self.subgroups[group_name].solo = solo
            if solo:
                for name, config in self.subgroups.items():
                    if name != group_name:
                        config.mute = True

    def clear_solo(self):
        for config in self.subgroups.values():
            config.solo = False
            config.mute = False

    def get_routing_map(self) -> Dict[str, List[int]]:
        routing = {}
        for group_name in self.subgroups:
            channels = self.get_group_channels(group_name)
            routing[group_name] = [ch.channel_id for ch in channels]
        return routing

    def to_osc_commands(self) -> List[Dict]:
        commands = []
        for group_name, config in self.subgroups.items():
            commands.append({
                'type': 'bus_config',
                'group': group_name,
                'gain_db': config.bus_gain_db,
                'pan': config.bus_pan,
                'mute': config.mute,
                'comp_threshold': config.bus_comp_threshold,
                'comp_ratio': config.bus_comp_ratio,
            })
        return commands
