"""
Channel state management and snapshot handling.

Provides a unified view of the mixer's channel state with
methods for reading, writing, and taking snapshots.
"""

import logging
import time
import copy
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


@dataclass
class ChannelState:
    """Complete state of a single mixer channel."""
    number: int
    name: str = ""
    fader_db: float = -144.0
    mute: bool = False
    solo: bool = False
    pan: float = 0.0
    gain_db: float = 0.0
    phantom: bool = False
    phase_invert: bool = False
    # EQ
    eq_on: bool = False
    eq_bands: Dict[str, Dict[str, float]] = field(default_factory=dict)
    # Compressor
    comp_on: bool = False
    comp_threshold: float = 0.0
    comp_ratio: str = "4.0"
    comp_attack: float = 10.0
    comp_release: float = 100.0
    comp_gain: float = 0.0
    # Gate
    gate_on: bool = False
    gate_threshold: float = -80.0
    # Filters
    lc_on: bool = False
    lc_freq: float = 80.0
    hc_on: bool = False
    hc_freq: float = 20000.0
    # Classification
    instrument_type: str = "unknown"
    color: int = 0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "number": self.number,
            "name": self.name,
            "fader_db": self.fader_db,
            "mute": self.mute,
            "solo": self.solo,
            "pan": self.pan,
            "gain_db": self.gain_db,
            "eq_on": self.eq_on,
            "comp_on": self.comp_on,
            "gate_on": self.gate_on,
            "lc_on": self.lc_on,
            "lc_freq": self.lc_freq,
            "instrument_type": self.instrument_type,
        }


@dataclass
class MixerSnapshot:
    """Full mixer snapshot."""
    name: str
    timestamp: float
    channels: Dict[int, ChannelState]
    main_fader: float = 0.0
    description: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "timestamp": self.timestamp,
            "main_fader": self.main_fader,
            "description": self.description,
            "channels": {ch: state.to_dict() for ch, state in self.channels.items()},
        }


class MixerStateManager:
    """
    Manages the complete mixer state.

    Provides:
    - Channel state read/write
    - Snapshot capture and restore
    - State diff tracking
    - OSC state synchronization
    """

    def __init__(self, num_channels: int = 40):
        self.num_channels = num_channels
        self._channels: Dict[int, ChannelState] = {
            ch: ChannelState(number=ch) for ch in range(1, num_channels + 1)
        }
        self._snapshots: List[MixerSnapshot] = []
        self._main_fader: float = 0.0
        self._last_update = time.time()

    def get_channel(self, ch: int) -> Optional[ChannelState]:
        """Get channel state."""
        return self._channels.get(ch)

    def get_all_channels(self) -> Dict[int, ChannelState]:
        """Get all channel states (deep copy)."""
        return copy.deepcopy(self._channels)

    def update_channel(self, ch: int, **kwargs):
        """Update channel parameters."""
        if ch in self._channels:
            for key, value in kwargs.items():
                if hasattr(self._channels[ch], key):
                    setattr(self._channels[ch], key, value)
            self._last_update = time.time()

    def update_from_osc_state(self, osc_state: Dict[str, Any]):
        """Synchronize from raw OSC state dictionary."""
        for address, value in osc_state.items():
            parts = address.strip("/").split("/")
            if len(parts) < 3 or parts[0] != "ch":
                continue
            try:
                ch = int(parts[1])
            except ValueError:
                continue
            if ch not in self._channels:
                continue

            param = parts[2] if len(parts) == 3 else "/".join(parts[2:])
            channel = self._channels[ch]

            if param == "fdr":
                channel.fader_db = float(value)
            elif param == "mute":
                channel.mute = bool(value)
            elif param == "pan":
                channel.pan = float(value)
            elif param == "name" or param == "$name":
                channel.name = str(value)
            elif param == "eq/on":
                channel.eq_on = bool(value)
            elif param == "dyn/on":
                channel.comp_on = bool(value)
            elif param == "gate/on":
                channel.gate_on = bool(value)

        self._last_update = time.time()

    def take_snapshot(self, name: str, description: str = "") -> MixerSnapshot:
        """Capture current state as a snapshot."""
        snap = MixerSnapshot(
            name=name,
            timestamp=time.time(),
            channels=copy.deepcopy(self._channels),
            main_fader=self._main_fader,
            description=description,
        )
        self._snapshots.append(snap)
        logger.info(f"Snapshot '{name}' captured with {len(self._channels)} channels")
        return snap

    def restore_snapshot(self, snapshot: MixerSnapshot):
        """Restore mixer state from a snapshot."""
        self._channels = copy.deepcopy(snapshot.channels)
        self._main_fader = snapshot.main_fader
        self._last_update = time.time()
        logger.info(f"Restored snapshot '{snapshot.name}'")

    def get_snapshots(self) -> List[Dict]:
        """List all stored snapshots."""
        return [
            {"name": s.name, "timestamp": s.timestamp, "description": s.description}
            for s in self._snapshots
        ]

    def find_snapshot(self, name: str) -> Optional[MixerSnapshot]:
        """Find a snapshot by name."""
        for snap in self._snapshots:
            if snap.name == name:
                return snap
        return None

    def get_active_channels(self, threshold_db: float = -100.0) -> List[int]:
        """Get channels with fader above threshold."""
        return [
            ch for ch, state in self._channels.items()
            if state.fader_db > threshold_db and not state.mute
        ]

    def get_status(self) -> Dict:
        """Get state manager status."""
        active = self.get_active_channels()
        return {
            "num_channels": self.num_channels,
            "active_channels": len(active),
            "snapshots_stored": len(self._snapshots),
            "last_update": self._last_update,
        }
