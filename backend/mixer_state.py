"""
Channel state management and snapshot handling.

Provides a compatibility-rich mixer state API used by both older tests and
newer infrastructure helpers.
"""

from __future__ import annotations

import copy
import json
import logging
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


@dataclass
class EQBandState:
    gain: float = 0.0
    frequency: float = 1000.0
    q: float = 1.0
    band_type: str = "PEQ"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "gain": self.gain,
            "frequency": self.frequency,
            "q": self.q,
            "band_type": self.band_type,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "EQBandState":
        return cls(
            gain=float(data.get("gain", 0.0)),
            frequency=float(data.get("frequency", 1000.0)),
            q=float(data.get("q", 1.0)),
            band_type=str(data.get("band_type", "PEQ")),
        )


@dataclass
class CompressorState:
    on: bool = False
    threshold: float = 0.0
    ratio: float = 4.0
    attack: float = 10.0
    release: float = 100.0
    gain: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "on": self.on,
            "threshold": self.threshold,
            "ratio": self.ratio,
            "attack": self.attack,
            "release": self.release,
            "gain": self.gain,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "CompressorState":
        return cls(
            on=bool(data.get("on", False)),
            threshold=float(data.get("threshold", 0.0)),
            ratio=float(data.get("ratio", 4.0)),
            attack=float(data.get("attack", 10.0)),
            release=float(data.get("release", 100.0)),
            gain=float(data.get("gain", 0.0)),
        )


@dataclass
class GateState:
    on: bool = False
    threshold: float = -80.0
    attack: float = 5.0
    release: float = 100.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "on": self.on,
            "threshold": self.threshold,
            "attack": self.attack,
            "release": self.release,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "GateState":
        return cls(
            on=bool(data.get("on", False)),
            threshold=float(data.get("threshold", -80.0)),
            attack=float(data.get("attack", 5.0)),
            release=float(data.get("release", 100.0)),
        )


@dataclass
class FilterState:
    low_cut_on: bool = False
    low_cut_frequency: float = 80.0
    high_cut_on: bool = False
    high_cut_frequency: float = 20000.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "low_cut_on": self.low_cut_on,
            "low_cut_frequency": self.low_cut_frequency,
            "high_cut_on": self.high_cut_on,
            "high_cut_frequency": self.high_cut_frequency,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "FilterState":
        return cls(
            low_cut_on=bool(data.get("low_cut_on", False)),
            low_cut_frequency=float(data.get("low_cut_frequency", 80.0)),
            high_cut_on=bool(data.get("high_cut_on", False)),
            high_cut_frequency=float(data.get("high_cut_frequency", 20000.0)),
        )


@dataclass
class SendState:
    on: bool = False
    level: float = -144.0
    pan: float = 0.0
    mode: str = "POST"

    def to_dict(self) -> Dict[str, Any]:
        return {"on": self.on, "level": self.level, "pan": self.pan, "mode": self.mode}

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "SendState":
        return cls(
            on=bool(data.get("on", False)),
            level=float(data.get("level", -144.0)),
            pan=float(data.get("pan", 0.0)),
            mode=str(data.get("mode", "POST")),
        )


@dataclass
class InputState:
    trim: float = 0.0
    phantom: bool = False
    phase_invert: bool = False

    def to_dict(self) -> Dict[str, Any]:
        return {
            "trim": self.trim,
            "phantom": self.phantom,
            "phase_invert": self.phase_invert,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "InputState":
        return cls(
            trim=float(data.get("trim", 0.0)),
            phantom=bool(data.get("phantom", False)),
            phase_invert=bool(data.get("phase_invert", False)),
        )


@dataclass
class InsertState:
    on: bool = False
    slot: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {"on": self.on, "slot": self.slot}

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "InsertState":
        return cls(on=bool(data.get("on", False)), slot=str(data.get("slot", "")))


@dataclass
class ChannelState:
    channel_id: int = 0
    name: str = ""
    fader: float = -144.0
    mute: bool = False
    solo: bool = False
    pan: float = 0.0
    color: int = 0
    eq_on: bool = False
    compressor: CompressorState = field(default_factory=CompressorState)
    gate: GateState = field(default_factory=GateState)
    filter_state: FilterState = field(default_factory=FilterState)
    input_state: InputState = field(default_factory=InputState)
    insert: InsertState = field(default_factory=InsertState)
    eq_bands: Dict[int, EQBandState] = field(default_factory=lambda: {i: EQBandState() for i in range(6)})
    sends: Dict[int, SendState] = field(default_factory=lambda: {i: SendState() for i in range(16)})
    main_sends: Dict[int, SendState] = field(default_factory=lambda: {i: SendState() for i in range(4)})

    @property
    def number(self) -> int:
        return self.channel_id

    @property
    def fader_db(self) -> float:
        return self.fader

    @fader_db.setter
    def fader_db(self, value: float):
        self.fader = float(value)

    @property
    def gain_db(self) -> float:
        return self.input_state.trim

    @gain_db.setter
    def gain_db(self, value: float):
        self.input_state.trim = float(value)

    @property
    def comp_on(self) -> bool:
        return self.compressor.on

    @comp_on.setter
    def comp_on(self, value: bool):
        self.compressor.on = bool(value)

    @property
    def gate_on(self) -> bool:
        return self.gate.on

    @gate_on.setter
    def gate_on(self, value: bool):
        self.gate.on = bool(value)

    @property
    def lc_on(self) -> bool:
        return self.filter_state.low_cut_on

    @property
    def lc_freq(self) -> float:
        return self.filter_state.low_cut_frequency

    def to_dict(self) -> Dict[str, Any]:
        return {
            "channel_id": self.channel_id,
            "name": self.name,
            "fader": self.fader,
            "mute": self.mute,
            "solo": self.solo,
            "pan": self.pan,
            "eq_on": self.eq_on,
            "compressor": self.compressor.to_dict(),
            "gate": self.gate.to_dict(),
            "filter": self.filter_state.to_dict(),
            "input": self.input_state.to_dict(),
            "insert": self.insert.to_dict(),
            "eq_bands": {str(i): band.to_dict() for i, band in self.eq_bands.items()},
            "sends": {str(i): send.to_dict() for i, send in self.sends.items()},
            "main_sends": {str(i): send.to_dict() for i, send in self.main_sends.items()},
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ChannelState":
        channel = cls(
            channel_id=int(data.get("channel_id", 0)),
            name=str(data.get("name", "")),
            fader=float(data.get("fader", -144.0)),
            mute=bool(data.get("mute", False)),
            solo=bool(data.get("solo", False)),
            pan=float(data.get("pan", 0.0)),
            eq_on=bool(data.get("eq_on", False)),
            compressor=CompressorState.from_dict(data.get("compressor", {})),
            gate=GateState.from_dict(data.get("gate", {})),
            filter_state=FilterState.from_dict(data.get("filter", {})),
            input_state=InputState.from_dict(data.get("input", {})),
            insert=InsertState.from_dict(data.get("insert", {})),
        )
        for key, value in data.get("eq_bands", {}).items():
            channel.eq_bands[int(key)] = EQBandState.from_dict(value)
        for key, value in data.get("sends", {}).items():
            channel.sends[int(key)] = SendState.from_dict(value)
        for key, value in data.get("main_sends", {}).items():
            channel.main_sends[int(key)] = SendState.from_dict(value)
        return channel


def _parse_eq_band_index(value: str) -> int:
    aliases = {"low": 0, "l": 0, "high": 5, "h": 5}
    if value in aliases:
        return aliases[value]
    if value.isdigit():
        idx = int(value)
        if 0 <= idx <= 5:
            return idx
    raise KeyError(value)


def _resolve_param(channel: ChannelState, path: str) -> Tuple[Any, str]:
    aliases = {"fdr": "fader"}
    path = aliases.get(path, path)
    if path in {"fader", "mute", "solo", "pan", "eq_on"}:
        return channel, path
    if path.startswith("compressor."):
        return channel.compressor, path.split(".", 1)[1]
    if path.startswith("gate."):
        return channel.gate, path.split(".", 1)[1]
    if path.startswith("eq."):
        parts = path.split(".")
        if len(parts) == 2 and parts[1] == "on":
            return channel, "eq_on"
        if len(parts) == 3:
            return channel.eq_bands[_parse_eq_band_index(parts[1])], parts[2]
    if path.startswith("send."):
        _, index, attr = path.split(".", 2)
        return channel.sends[int(index)], attr
    if path.startswith("input."):
        return channel.input_state, path.split(".", 1)[1]
    if path.startswith("filter."):
        return channel.filter_state, path.split(".", 1)[1]
    raise KeyError(path)


@dataclass
class MixerSnapshot:
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
            "channels": {str(ch): state.to_dict() for ch, state in self.channels.items()},
        }


class MixerState:
    VERSION = 1

    def __init__(self, num_channels: int = 40):
        self.num_channels = num_channels
        self._channels: Dict[int, ChannelState] = {
            ch: ChannelState(channel_id=ch) for ch in range(1, num_channels + 1)
        }
        self._snapshots: List[MixerSnapshot] = []
        self._listeners: List[Callable[[int, str, Any, Any], None]] = []

    @property
    def channel_ids(self) -> List[int]:
        return list(self._channels.keys())

    def get_channel(self, ch: int) -> Optional[ChannelState]:
        return self._channels.get(ch)

    def get_all_channels(self) -> Dict[int, ChannelState]:
        return copy.deepcopy(self._channels)

    def add_listener(self, callback: Callable[[int, str, Any, Any], None]):
        self._listeners.append(callback)

    def remove_listener(self, callback: Callable[[int, str, Any, Any], None]):
        self._listeners = [cb for cb in self._listeners if cb != callback]

    def get(self, ch: int, path: str) -> Any:
        channel = self._channels.get(ch)
        if channel is None:
            raise KeyError(ch)
        parent, attr = _resolve_param(channel, path)
        return getattr(parent, attr)

    def set(self, ch: int, path: str, value: Any) -> Any:
        channel = self._channels.get(ch)
        if channel is None:
            raise KeyError(ch)
        parent, attr = _resolve_param(channel, path)
        old = getattr(parent, attr)
        if old == value:
            return old
        setattr(parent, attr, value)
        for callback in list(self._listeners):
            callback(ch, path, old, value)
        return old

    @staticmethod
    def _osc_to_param_path(param: str) -> Optional[str]:
        mapping = {
            "fdr": "fader",
            "mute": "mute",
            "pan": "pan",
            "eq/on": "eq.on",
            "dyn/thr": "compressor.threshold",
            "gate/att": "gate.attack",
            "gate/thr": "gate.threshold",
            "flt/lc": "filter.low_cut_on",
        }
        if param in mapping:
            return mapping[param]
        if param.startswith("eq/") and param.endswith("g"):
            return f"eq.{param.split('/')[1][0]}.gain"
        if param.startswith("eq/") and param.endswith("f"):
            return f"eq.{param.split('/')[1][0]}.frequency"
        return None

    def update_from_osc(self, address: str, value: Any) -> bool:
        parts = address.strip("/").split("/")
        if len(parts) < 3 or parts[0] != "ch":
            return False
        try:
            channel = int(parts[1])
        except ValueError:
            return False
        if channel not in self._channels:
            return False
        param_path = self._osc_to_param_path("/".join(parts[2:]))
        if param_path is None:
            return False
        self.set(channel, param_path, value)
        return True

    def update_from_osc_state(self, osc_state: Dict[str, Any]):
        for address, value in osc_state.items():
            self.update_from_osc(address, value)

    def snapshot_save(self, name: str) -> MixerSnapshot:
        if not name:
            raise ValueError("Snapshot name is required")
        snapshot = MixerSnapshot(
            name=name,
            timestamp=time.time(),
            channels=copy.deepcopy(self._channels),
        )
        self._snapshots = [snap for snap in self._snapshots if snap.name != name]
        self._snapshots.append(snapshot)
        return snapshot

    def snapshot_list(self) -> List[Dict[str, Any]]:
        return [{"name": snap.name, "timestamp": snap.timestamp} for snap in self._snapshots]

    def snapshot_recall(self, name: str) -> bool:
        snapshot = self.find_snapshot(name)
        if snapshot is None:
            return False
        self._channels = copy.deepcopy(snapshot.channels)
        return True

    def snapshot_delete(self, name: str) -> bool:
        before = len(self._snapshots)
        self._snapshots = [snap for snap in self._snapshots if snap.name != name]
        return len(self._snapshots) != before

    def find_snapshot(self, name: str) -> Optional[MixerSnapshot]:
        for snapshot in self._snapshots:
            if snapshot.name == name:
                return snapshot
        return None

    def export_json(self, path: Optional[str] = None) -> str:
        data = {
            "version": self.VERSION,
            "channels": {str(ch): state.to_dict() for ch, state in self._channels.items()},
        }
        payload = json.dumps(data, indent=2)
        if path:
            with open(path, "w", encoding="utf-8") as f:
                f.write(payload)
        return payload

    def import_json(self, path: Optional[str] = None, json_str: Optional[str] = None):
        if not path and not json_str:
            raise ValueError("Either path or json_str is required")
        if path:
            with open(path, encoding="utf-8") as f:
                data = json.load(f)
        else:
            data = json.loads(json_str)
        self._channels = {
            int(ch): ChannelState.from_dict(state)
            for ch, state in data.get("channels", {}).items()
        }

    def diff(self, other: "MixerState") -> List[Dict[str, Any]]:
        diffs: List[Dict[str, Any]] = []
        for ch in sorted(set(self._channels) & set(other._channels)):
            if self.get(ch, "fader") != other.get(ch, "fader"):
                diffs.append({
                    "channel": ch,
                    "param": "fader",
                    "old": other.get(ch, "fader"),
                    "new": self.get(ch, "fader"),
                })
        return diffs

    def diff_from_snapshot(self, name: str) -> Optional[List[Dict[str, Any]]]:
        snapshot = self.find_snapshot(name)
        if snapshot is None:
            return None
        other = MixerState(num_channels=len(snapshot.channels))
        other._channels = copy.deepcopy(snapshot.channels)
        return self.diff(other)

    def take_snapshot(self, name: str, description: str = "") -> MixerSnapshot:
        snapshot = self.snapshot_save(name)
        snapshot.description = description
        return snapshot

    def restore_snapshot(self, snapshot: MixerSnapshot):
        self._channels = copy.deepcopy(snapshot.channels)

    def get_snapshots(self) -> List[Dict[str, Any]]:
        return [
            {"name": snap.name, "timestamp": snap.timestamp, "description": snap.description}
            for snap in self._snapshots
        ]

    def get_active_channels(self, threshold_db: float = -100.0) -> List[int]:
        return [ch for ch, state in self._channels.items() if state.fader > threshold_db and not state.mute]

    def get_status(self) -> Dict[str, Any]:
        return {
            "num_channels": self.num_channels,
            "active_channels": len(self.get_active_channels()),
            "snapshots_stored": len(self._snapshots),
        }

    def get_snapshot(self) -> Dict[int, ChannelState]:
        return self.get_all_channels()

    capture_snapshot = get_snapshot


class MixerStateManager(MixerState):
    """Backwards-compatible alias for the simpler manager API."""

    def update_channel(self, ch: int, **kwargs):
        for key, value in kwargs.items():
            try:
                self.set(ch, key, value)
            except KeyError:
                continue

