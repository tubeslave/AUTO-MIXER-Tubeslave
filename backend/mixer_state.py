"""
Mixer State Management and Snapshot Handling for Behringer Wing

Provides a structured in-memory representation of the entire mixer state,
including per-channel parameters (fader, mute, pan, EQ, compressor, gate,
sends, inserts, etc.).

Key features:
- Typed dataclasses for every channel sub-section
- get/set by channel and parameter path
- In-memory snapshot save/recall with named slots
- State diffing for efficient delta updates
- JSON import/export for persistence
"""

import copy
import json
import logging
import threading
import time
from dataclasses import dataclass, field, asdict
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Dataclasses for channel sub-sections
# ---------------------------------------------------------------------------

@dataclass
class EQBandState:
    """Single parametric EQ band (Wing has Low-shelf, bands 1-4, High-shelf)."""
    gain: float = 0.0         # dB, -15..+15
    frequency: float = 1000.0  # Hz, 20..20000
    q: float = 1.0             # Q factor, 0.44..10
    band_type: str = "PEQ"     # PEQ, SHV (shelving)

    def to_dict(self) -> Dict[str, Any]:
        return {"gain": self.gain, "frequency": self.frequency,
                "q": self.q, "band_type": self.band_type}

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "EQBandState":
        return cls(
            gain=float(d.get("gain", 0.0)),
            frequency=float(d.get("frequency", 1000.0)),
            q=float(d.get("q", 1.0)),
            band_type=str(d.get("band_type", "PEQ")),
        )


@dataclass
class CompressorState:
    """Dynamics / compressor section of a Wing channel."""
    on: bool = False
    model: str = "STD"
    mix: float = 100.0          # % 0..100
    makeup_gain: float = 0.0    # dB -6..+12
    threshold: float = -20.0    # dB -60..0
    ratio: float = 4.0          # 1.1 .. 100
    knee: float = 2.0           # 0..5
    detect: str = "RMS"         # PEAK, RMS
    attack: float = 10.0        # ms 0..120
    hold: float = 0.0           # ms 1..200
    release: float = 100.0      # ms 4..4000
    envelope: str = "LOG"       # LIN, LOG
    auto: bool = False

    # Sidechain
    sc_type: str = "Off"        # Off, LP12, HP12, BP
    sc_frequency: float = 1000.0
    sc_q: float = 1.0
    sc_source: str = "SELF"

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "CompressorState":
        return cls(
            on=bool(d.get("on", False)),
            model=str(d.get("model", "STD")),
            mix=float(d.get("mix", 100.0)),
            makeup_gain=float(d.get("makeup_gain", 0.0)),
            threshold=float(d.get("threshold", -20.0)),
            ratio=float(d.get("ratio", 4.0)),
            knee=float(d.get("knee", 2.0)),
            detect=str(d.get("detect", "RMS")),
            attack=float(d.get("attack", 10.0)),
            hold=float(d.get("hold", 0.0)),
            release=float(d.get("release", 100.0)),
            envelope=str(d.get("envelope", "LOG")),
            auto=bool(d.get("auto", False)),
            sc_type=str(d.get("sc_type", "Off")),
            sc_frequency=float(d.get("sc_frequency", 1000.0)),
            sc_q=float(d.get("sc_q", 1.0)),
            sc_source=str(d.get("sc_source", "SELF")),
        )


@dataclass
class GateState:
    """Noise gate section of a Wing channel."""
    on: bool = False
    model: str = "STD"
    threshold: float = -40.0   # dB -80..0
    gate_range: float = 20.0   # dB 3..60
    attack: float = 1.0        # ms 0..120
    hold: float = 50.0         # ms 0..200
    release: float = 200.0     # ms 4..4000
    accent: float = 0.0        # 0..100
    ratio: str = "GATE"        # 1:1.5, 1:2, 1:3, 1:4, GATE

    # Sidechain
    sc_type: str = "Off"
    sc_frequency: float = 1000.0
    sc_q: float = 1.0
    sc_source: str = "SELF"

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "GateState":
        return cls(
            on=bool(d.get("on", False)),
            model=str(d.get("model", "STD")),
            threshold=float(d.get("threshold", -40.0)),
            gate_range=float(d.get("gate_range", 20.0)),
            attack=float(d.get("attack", 1.0)),
            hold=float(d.get("hold", 50.0)),
            release=float(d.get("release", 200.0)),
            accent=float(d.get("accent", 0.0)),
            ratio=str(d.get("ratio", "GATE")),
            sc_type=str(d.get("sc_type", "Off")),
            sc_frequency=float(d.get("sc_frequency", 1000.0)),
            sc_q=float(d.get("sc_q", 1.0)),
            sc_source=str(d.get("sc_source", "SELF")),
        )


@dataclass
class FilterState:
    """High-pass / low-pass filter section."""
    low_cut_on: bool = False
    low_cut_freq: float = 80.0       # Hz 20..2000
    low_cut_slope: int = 18          # 6, 12, 18, 24 dB/oct
    high_cut_on: bool = False
    high_cut_freq: float = 20000.0   # Hz 50..20000
    high_cut_slope: int = 12         # 6, 12 dB/oct
    tilt_filter_on: bool = False
    tilt_model: str = "TILT"         # TILT, MAX, AP1, AP2
    tilt_level: float = 0.0          # dB -6..+6

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "FilterState":
        return cls(
            low_cut_on=bool(d.get("low_cut_on", False)),
            low_cut_freq=float(d.get("low_cut_freq", 80.0)),
            low_cut_slope=int(d.get("low_cut_slope", 18)),
            high_cut_on=bool(d.get("high_cut_on", False)),
            high_cut_freq=float(d.get("high_cut_freq", 20000.0)),
            high_cut_slope=int(d.get("high_cut_slope", 12)),
            tilt_filter_on=bool(d.get("tilt_filter_on", False)),
            tilt_model=str(d.get("tilt_model", "TILT")),
            tilt_level=float(d.get("tilt_level", 0.0)),
        )


@dataclass
class SendState:
    """A single bus send."""
    on: bool = False
    level: float = -144.0  # dB -144..+10
    pan: float = 0.0       # -100..+100
    mode: str = "POST"     # PRE, POST, GRP

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "SendState":
        return cls(
            on=bool(d.get("on", False)),
            level=float(d.get("level", -144.0)),
            pan=float(d.get("pan", 0.0)),
            mode=str(d.get("mode", "POST")),
        )


@dataclass
class InputState:
    """Channel input configuration."""
    trim: float = 0.0         # dB -18..+18
    phase_invert: bool = False
    balance: float = 0.0      # dB -9..+9
    delay_on: bool = False
    delay_ms: float = 0.0
    source_group: str = ""
    source_index: int = 0

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "InputState":
        return cls(
            trim=float(d.get("trim", 0.0)),
            phase_invert=bool(d.get("phase_invert", False)),
            balance=float(d.get("balance", 0.0)),
            delay_on=bool(d.get("delay_on", False)),
            delay_ms=float(d.get("delay_ms", 0.0)),
            source_group=str(d.get("source_group", "")),
            source_index=int(d.get("source_index", 0)),
        )


@dataclass
class InsertState:
    """Pre- or post-insert slot."""
    on: bool = False
    fx_slot: str = "NONE"  # NONE, FX1..FX16
    mode: str = "FX"       # FX, AUTO_X, AUTO_Y (post-insert only)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "InsertState":
        return cls(
            on=bool(d.get("on", False)),
            fx_slot=str(d.get("fx_slot", "NONE")),
            mode=str(d.get("mode", "FX")),
        )


# ---------------------------------------------------------------------------
# Full channel state
# ---------------------------------------------------------------------------

@dataclass
class ChannelState:
    """Complete state of a single Wing input channel."""

    # Identity
    channel_id: int = 0
    name: str = ""
    icon: str = ""
    color: str = ""

    # Main controls
    fader: float = -144.0     # dB -144..+10
    mute: bool = False
    pan: float = 0.0          # -100..+100
    width: float = 100.0      # % -150..+150
    solo: bool = False
    solo_safe: bool = False

    # Input
    input_state: InputState = field(default_factory=InputState)

    # Filters
    filter_state: FilterState = field(default_factory=FilterState)

    # Pre-send EQ (3-band)
    peq_on: bool = False
    peq_bands: List[EQBandState] = field(
        default_factory=lambda: [EQBandState() for _ in range(3)]
    )

    # Main EQ (6-band: Low-shelf, bands 1-4, High-shelf)
    eq_on: bool = False
    eq_model: str = "STD"
    eq_mix: float = 100.0
    eq_bands: List[EQBandState] = field(
        default_factory=lambda: [EQBandState() for _ in range(6)]
    )

    # Gate
    gate: GateState = field(default_factory=GateState)

    # Compressor / Dynamics
    compressor: CompressorState = field(default_factory=CompressorState)

    # Inserts
    pre_insert: InsertState = field(default_factory=InsertState)
    post_insert: InsertState = field(default_factory=InsertState)

    # Bus sends (Wing supports up to 16 sends)
    sends: Dict[int, SendState] = field(
        default_factory=lambda: {i: SendState() for i in range(1, 17)}
    )

    # Main sends (up to 4 mains)
    main_sends: Dict[int, SendState] = field(
        default_factory=lambda: {i: SendState() for i in range(1, 5)}
    )

    # Metadata
    last_updated: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to a plain dictionary suitable for JSON."""
        return {
            "channel_id": self.channel_id,
            "name": self.name,
            "icon": self.icon,
            "color": self.color,
            "fader": self.fader,
            "mute": self.mute,
            "pan": self.pan,
            "width": self.width,
            "solo": self.solo,
            "solo_safe": self.solo_safe,
            "input_state": self.input_state.to_dict(),
            "filter_state": self.filter_state.to_dict(),
            "peq_on": self.peq_on,
            "peq_bands": [b.to_dict() for b in self.peq_bands],
            "eq_on": self.eq_on,
            "eq_model": self.eq_model,
            "eq_mix": self.eq_mix,
            "eq_bands": [b.to_dict() for b in self.eq_bands],
            "gate": self.gate.to_dict(),
            "compressor": self.compressor.to_dict(),
            "pre_insert": self.pre_insert.to_dict(),
            "post_insert": self.post_insert.to_dict(),
            "sends": {str(k): v.to_dict() for k, v in self.sends.items()},
            "main_sends": {str(k): v.to_dict() for k, v in self.main_sends.items()},
            "last_updated": self.last_updated,
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "ChannelState":
        """Deserialize from a plain dictionary."""
        ch = cls(
            channel_id=int(d.get("channel_id", 0)),
            name=str(d.get("name", "")),
            icon=str(d.get("icon", "")),
            color=str(d.get("color", "")),
            fader=float(d.get("fader", -144.0)),
            mute=bool(d.get("mute", False)),
            pan=float(d.get("pan", 0.0)),
            width=float(d.get("width", 100.0)),
            solo=bool(d.get("solo", False)),
            solo_safe=bool(d.get("solo_safe", False)),
            input_state=InputState.from_dict(d.get("input_state", {})),
            filter_state=FilterState.from_dict(d.get("filter_state", {})),
            peq_on=bool(d.get("peq_on", False)),
            peq_bands=[
                EQBandState.from_dict(b) for b in d.get("peq_bands", [{}, {}, {}])
            ],
            eq_on=bool(d.get("eq_on", False)),
            eq_model=str(d.get("eq_model", "STD")),
            eq_mix=float(d.get("eq_mix", 100.0)),
            eq_bands=[
                EQBandState.from_dict(b)
                for b in d.get("eq_bands", [{}, {}, {}, {}, {}, {}])
            ],
            gate=GateState.from_dict(d.get("gate", {})),
            compressor=CompressorState.from_dict(d.get("compressor", {})),
            pre_insert=InsertState.from_dict(d.get("pre_insert", {})),
            post_insert=InsertState.from_dict(d.get("post_insert", {})),
            last_updated=float(d.get("last_updated", 0.0)),
        )
        # Sends
        for key, val in d.get("sends", {}).items():
            ch.sends[int(key)] = SendState.from_dict(val)
        for key, val in d.get("main_sends", {}).items():
            ch.main_sends[int(key)] = SendState.from_dict(val)
        return ch


# ---------------------------------------------------------------------------
# Parameter path resolver: maps dotted path to ChannelState attributes
# ---------------------------------------------------------------------------

# Map of short param names to (attribute_path, sub_type)
# Used by MixerState.get / .set to translate string paths like
# "eq.1.gain" or "compressor.threshold" into the nested dataclass fields.

_SIMPLE_PARAMS = {
    "fader": "fader",
    "fdr": "fader",
    "mute": "mute",
    "pan": "pan",
    "width": "width",
    "wid": "width",
    "name": "name",
    "solo": "solo",
    "solo_safe": "solo_safe",
}


def _resolve_param(channel: ChannelState, param_path: str) -> Tuple[Any, str]:
    """
    Resolve a dot-separated parameter path into (parent_object, attribute_name).

    Examples:
        'fader'              -> (channel, 'fader')
        'compressor.threshold' -> (channel.compressor, 'threshold')
        'eq.2.gain'          -> (channel.eq_bands[2], 'gain')
        'gate.attack'        -> (channel.gate, 'attack')
        'send.3.level'       -> (channel.sends[3], 'level')
        'input.trim'         -> (channel.input_state, 'trim')
        'filter.low_cut_on'  -> (channel.filter_state, 'low_cut_on')

    Returns:
        (parent_object, attribute_name) so caller can getattr/setattr.

    Raises:
        KeyError: if the path cannot be resolved.
    """
    parts = param_path.split(".")

    # Simple top-level params
    if len(parts) == 1:
        alias = _SIMPLE_PARAMS.get(parts[0])
        if alias:
            return (channel, alias)
        # Check for direct attribute
        if hasattr(channel, parts[0]):
            return (channel, parts[0])
        raise KeyError(f"Unknown parameter: {param_path}")

    section = parts[0]

    if section in ("compressor", "comp", "dyn"):
        attr = parts[1]
        return (channel.compressor, attr)

    if section == "gate":
        attr = parts[1]
        return (channel.gate, attr)

    if section == "eq":
        if len(parts) == 2:
            # e.g. eq.on, eq.model
            attr = parts[1]
            if attr == "on":
                return (channel, "eq_on")
            if attr == "model":
                return (channel, "eq_model")
            if attr == "mix":
                return (channel, "eq_mix")
            raise KeyError(f"Unknown EQ attribute: {attr}")
        if len(parts) == 3:
            # e.g. eq.2.gain -> eq_bands[idx]
            idx = _parse_eq_band_index(parts[1])
            attr = parts[2]
            return (channel.eq_bands[idx], attr)
        raise KeyError(f"Invalid EQ path: {param_path}")

    if section == "peq":
        if len(parts) == 2:
            if parts[1] == "on":
                return (channel, "peq_on")
            raise KeyError(f"Unknown PEQ attribute: {parts[1]}")
        if len(parts) == 3:
            idx = int(parts[1]) - 1  # 1-based to 0-based
            if not (0 <= idx < len(channel.peq_bands)):
                raise KeyError(f"PEQ band index out of range: {parts[1]}")
            return (channel.peq_bands[idx], parts[2])
        raise KeyError(f"Invalid PEQ path: {param_path}")

    if section in ("send", "sends"):
        send_num = int(parts[1])
        if send_num not in channel.sends:
            raise KeyError(f"Send {send_num} not found")
        if len(parts) == 3:
            return (channel.sends[send_num], parts[2])
        raise KeyError(f"Invalid send path: {param_path}")

    if section in ("main", "main_send"):
        main_num = int(parts[1])
        if main_num not in channel.main_sends:
            raise KeyError(f"Main send {main_num} not found")
        if len(parts) == 3:
            return (channel.main_sends[main_num], parts[2])
        raise KeyError(f"Invalid main send path: {param_path}")

    if section in ("input", "in"):
        return (channel.input_state, parts[1])

    if section in ("filter", "flt"):
        return (channel.filter_state, parts[1])

    if section in ("pre_insert", "preins"):
        return (channel.pre_insert, parts[1])

    if section in ("post_insert", "postins"):
        return (channel.post_insert, parts[1])

    raise KeyError(f"Unknown parameter section: {section}")


def _parse_eq_band_index(name: str) -> int:
    """
    Parse an EQ band name into a 0-based index.

    Accepts: 'low'/'l' (0), '1'..'4' (1..4), 'high'/'h' (5),
    or direct 0-based integers.
    """
    lower = name.lower()
    if lower in ("low", "l", "ls"):
        return 0
    if lower in ("high", "h", "hs"):
        return 5
    idx = int(name)
    if 1 <= idx <= 4:
        return idx  # bands 1-4 map to indices 1-4
    if 0 <= idx <= 5:
        return idx
    raise KeyError(f"EQ band index out of range: {name}")


# ---------------------------------------------------------------------------
# MixerState: aggregate state for the entire mixer
# ---------------------------------------------------------------------------

class MixerState:
    """
    In-memory representation of the full Behringer Wing mixer state.

    Holds ChannelState objects for all 40 input channels and provides
    methods for parameter access, snapshot management, and state diffing.
    """

    MAX_CHANNELS = 40

    def __init__(self, num_channels: int = MAX_CHANNELS):
        self._lock = threading.RLock()
        self._channels: Dict[int, ChannelState] = {}
        for ch_id in range(1, num_channels + 1):
            self._channels[ch_id] = ChannelState(channel_id=ch_id)

        # Named snapshots stored in memory
        self._snapshots: Dict[str, Dict[str, Any]] = {}

        # Change listeners: called with (channel_id, param_path, old_value, new_value)
        self._listeners: List[Any] = []
        self._listeners_lock = threading.Lock()

        logger.info("MixerState initialized with %d channels", num_channels)

    # ------------------------------------------------------------------
    # Channel access
    # ------------------------------------------------------------------

    def get_channel(self, channel_id: int) -> ChannelState:
        """Get the ChannelState for a given channel number (1-based)."""
        with self._lock:
            ch = self._channels.get(channel_id)
            if ch is None:
                raise KeyError(f"Channel {channel_id} not found")
            return ch

    def get_all_channels(self) -> Dict[int, ChannelState]:
        """Return a copy of the channel dict."""
        with self._lock:
            return dict(self._channels)

    @property
    def channel_ids(self) -> List[int]:
        """List of all channel IDs."""
        with self._lock:
            return list(self._channels.keys())

    # ------------------------------------------------------------------
    # Generic get/set by parameter path
    # ------------------------------------------------------------------

    def get(self, channel_id: int, param_path: str) -> Any:
        """
        Get a channel parameter by dotted path.

        Args:
            channel_id: Channel number (1-based).
            param_path: Dotted parameter path, e.g. 'fader', 'eq.2.gain',
                        'compressor.threshold', 'gate.attack'.

        Returns:
            The current value of the parameter.

        Raises:
            KeyError: if channel or parameter not found.
        """
        with self._lock:
            ch = self.get_channel(channel_id)
            parent, attr = _resolve_param(ch, param_path)
            return getattr(parent, attr)

    def set(self, channel_id: int, param_path: str, value: Any) -> Any:
        """
        Set a channel parameter by dotted path.

        Args:
            channel_id: Channel number (1-based).
            param_path: Dotted parameter path.
            value: New value.

        Returns:
            The previous value.

        Raises:
            KeyError: if channel or parameter not found.
        """
        with self._lock:
            ch = self.get_channel(channel_id)
            parent, attr = _resolve_param(ch, param_path)
            old_value = getattr(parent, attr)
            setattr(parent, attr, value)
            ch.last_updated = time.time()

        # Notify listeners outside the lock
        self._notify_change(channel_id, param_path, old_value, value)
        return old_value

    # ------------------------------------------------------------------
    # Bulk update from OSC address
    # ------------------------------------------------------------------

    def update_from_osc(self, address: str, value: Any) -> bool:
        """
        Update state from a raw Wing OSC address.

        Parses addresses like '/ch/3/fdr', '/ch/5/eq/2g', '/ch/1/dyn/thr'
        and routes them to the correct ChannelState field.

        Returns True if the address was recognized and applied.
        """
        parts = address.strip("/").split("/")
        if len(parts) < 3 or parts[0] != "ch":
            return False

        try:
            channel_id = int(parts[1])
        except ValueError:
            return False

        if channel_id not in self._channels:
            return False

        osc_param = "/".join(parts[2:])
        param_path = self._osc_to_param_path(osc_param)
        if param_path is None:
            return False

        try:
            self.set(channel_id, param_path, value)
            return True
        except (KeyError, AttributeError) as exc:
            logger.debug("Failed to apply OSC %s: %s", address, exc)
            return False

    @staticmethod
    def _osc_to_param_path(osc_param: str) -> Optional[str]:
        """
        Convert a Wing OSC parameter suffix to a dotted param path.

        Examples:
            'fdr' -> 'fader'
            'mute' -> 'mute'
            'pan' -> 'pan'
            'eq/2g' -> 'eq.2.gain'
            'dyn/thr' -> 'compressor.threshold'
            'gate/att' -> 'gate.attack'
        """
        # Simple top-level
        simple_map = {
            "fdr": "fader", "mute": "mute", "pan": "pan", "wid": "width",
            "name": "name", "solosafe": "solo_safe",
        }
        if osc_param in simple_map:
            return simple_map[osc_param]

        parts = osc_param.split("/")

        # EQ: eq/on, eq/1g, eq/1f, eq/1q, eq/lg, eq/lf, eq/hg etc.
        if parts[0] == "eq" and len(parts) == 2:
            sub = parts[1]
            if sub == "on":
                return "eq.on"
            if sub == "mdl":
                return "eq.model"
            if sub == "mix":
                return "eq.mix"
            # Parse band params like '1g', '2f', 'lg', 'hg' etc.
            band_map = {
                "l": "low", "h": "high",
                "1": "1", "2": "2", "3": "3", "4": "4",
            }
            attr_map = {"g": "gain", "f": "frequency", "q": "q"}
            if len(sub) >= 2:
                band_key = sub[:-1]
                attr_key = sub[-1]
                if band_key in band_map and attr_key in attr_map:
                    return f"eq.{band_key}.{attr_map[attr_key]}"
            # Band type: leq, heq
            if sub in ("leq", "heq"):
                band = "low" if sub == "leq" else "high"
                return f"eq.{band}.band_type"
            return None

        # Pre-EQ: peq/on, peq/1g etc.
        if parts[0] == "peq" and len(parts) == 2:
            sub = parts[1]
            if sub == "on":
                return "peq.on"
            attr_map = {"g": "gain", "f": "frequency", "q": "q"}
            if len(sub) >= 2:
                band_key = sub[:-1]
                attr_key = sub[-1]
                if attr_key in attr_map:
                    return f"peq.{band_key}.{attr_map[attr_key]}"
            return None

        # Compressor / dynamics: dyn/on, dyn/thr, dyn/ratio etc.
        if parts[0] == "dyn" and len(parts) == 2:
            dyn_map = {
                "on": "on", "mdl": "model", "mix": "mix", "gain": "makeup_gain",
                "thr": "threshold", "ratio": "ratio", "knee": "knee",
                "det": "detect", "att": "attack", "hld": "hold",
                "rel": "release", "env": "envelope", "auto": "auto",
            }
            mapped = dyn_map.get(parts[1])
            if mapped:
                return f"compressor.{mapped}"
            return None

        # Dynamics sidechain: dynsc/type, dynsc/f etc.
        if parts[0] == "dynsc" and len(parts) == 2:
            sc_map = {
                "type": "sc_type", "f": "sc_frequency",
                "q": "sc_q", "src": "sc_source",
            }
            mapped = sc_map.get(parts[1])
            if mapped:
                return f"compressor.{mapped}"
            return None

        # Gate: gate/on, gate/thr etc.
        if parts[0] == "gate" and len(parts) == 2:
            gate_map = {
                "on": "on", "mdl": "model", "thr": "threshold",
                "range": "gate_range", "att": "attack", "hld": "hold",
                "rel": "release", "acc": "accent", "ratio": "ratio",
            }
            mapped = gate_map.get(parts[1])
            if mapped:
                return f"gate.{mapped}"
            return None

        # Gate sidechain
        if parts[0] == "gatesc" and len(parts) == 2:
            sc_map = {
                "type": "sc_type", "f": "sc_frequency",
                "q": "sc_q", "src": "sc_source",
            }
            mapped = sc_map.get(parts[1])
            if mapped:
                return f"gate.{mapped}"
            return None

        # Filters: flt/lc, flt/lcf etc.
        if parts[0] == "flt" and len(parts) == 2:
            flt_map = {
                "lc": "low_cut_on", "lcf": "low_cut_freq", "lcs": "low_cut_slope",
                "hc": "high_cut_on", "hcf": "high_cut_freq", "hcs": "high_cut_slope",
                "tf": "tilt_filter_on", "mdl": "tilt_model", "tilt": "tilt_level",
            }
            mapped = flt_map.get(parts[1])
            if mapped:
                return f"filter.{mapped}"
            return None

        # Input: in/set/trim, in/set/inv etc.
        if parts[0] == "in" and len(parts) >= 3 and parts[1] == "set":
            in_map = {
                "trim": "trim", "inv": "phase_invert", "bal": "balance",
                "dlyon": "delay_on", "dly": "delay_ms",
            }
            mapped = in_map.get(parts[2])
            if mapped:
                return f"input.{mapped}"
            return None

        # Input connection
        if parts[0] == "in" and len(parts) >= 3 and parts[1] == "conn":
            conn_map = {"grp": "source_group", "in": "source_index"}
            mapped = conn_map.get(parts[2])
            if mapped:
                return f"input.{mapped}"
            return None

        # Sends: send/N/on, send/N/lvl
        if parts[0] == "send" and len(parts) == 3:
            send_attr_map = {"on": "on", "lvl": "level", "mode": "mode"}
            mapped = send_attr_map.get(parts[2])
            if mapped:
                return f"send.{parts[1]}.{mapped}"
            return None

        # Main sends: main/N/on, main/N/lvl
        if parts[0] == "main" and len(parts) == 3:
            main_attr_map = {"on": "on", "lvl": "level"}
            mapped = main_attr_map.get(parts[2])
            if mapped:
                return f"main.{parts[1]}.{mapped}"
            return None

        # Inserts
        if parts[0] == "preins" and len(parts) == 2:
            ins_map = {"on": "on", "ins": "fx_slot", "mode": "mode"}
            mapped = ins_map.get(parts[1])
            if mapped:
                return f"pre_insert.{mapped}"
            return None

        if parts[0] == "postins" and len(parts) == 2:
            ins_map = {"on": "on", "ins": "fx_slot", "mode": "mode"}
            mapped = ins_map.get(parts[1])
            if mapped:
                return f"post_insert.{mapped}"
            return None

        return None

    # ------------------------------------------------------------------
    # Snapshots
    # ------------------------------------------------------------------

    def snapshot_save(self, name: str) -> None:
        """
        Save the current mixer state as a named snapshot.

        Args:
            name: Snapshot name (must be non-empty).
        """
        if not name:
            raise ValueError("Snapshot name cannot be empty")

        with self._lock:
            data = self._serialize_all()

        self._snapshots[name] = {
            "timestamp": time.time(),
            "state": data,
        }
        logger.info("Snapshot '%s' saved (%d channels)", name, len(data))

    def snapshot_recall(self, name: str) -> bool:
        """
        Recall a previously saved snapshot, overwriting current state.

        Args:
            name: Snapshot name.

        Returns:
            True if recalled, False if snapshot not found.
        """
        snap = self._snapshots.get(name)
        if snap is None:
            logger.warning("Snapshot '%s' not found", name)
            return False

        with self._lock:
            self._deserialize_all(snap["state"])

        logger.info("Snapshot '%s' recalled", name)
        return True

    def snapshot_list(self) -> List[Dict[str, Any]]:
        """List all saved snapshots with names and timestamps."""
        return [
            {"name": name, "timestamp": info["timestamp"],
             "channels": len(info["state"])}
            for name, info in self._snapshots.items()
        ]

    def snapshot_delete(self, name: str) -> bool:
        """Delete a named snapshot. Returns True if it existed."""
        removed = self._snapshots.pop(name, None)
        if removed:
            logger.info("Snapshot '%s' deleted", name)
        return removed is not None

    # ------------------------------------------------------------------
    # Diffing
    # ------------------------------------------------------------------

    def diff(self, other: "MixerState") -> List[Dict[str, Any]]:
        """
        Compare this state with another MixerState, returning a list of
        differences.

        Each difference is a dict:
            {'channel': int, 'param': str, 'current': value, 'other': value}

        Only compares channels present in both states.
        """
        differences: List[Dict[str, Any]] = []

        with self._lock:
            self_data = self._serialize_all()
        with other._lock:
            other_data = other._serialize_all()

        for ch_id_str, self_ch in self_data.items():
            ch_id = int(ch_id_str)
            other_ch = other_data.get(ch_id_str)
            if other_ch is None:
                continue
            self._diff_dicts(
                ch_id, "", self_ch, other_ch, differences
            )

        return differences

    def diff_from_snapshot(self, name: str) -> Optional[List[Dict[str, Any]]]:
        """
        Diff current state against a named snapshot.

        Returns None if snapshot not found.
        """
        snap = self._snapshots.get(name)
        if snap is None:
            return None

        with self._lock:
            current_data = self._serialize_all()

        differences: List[Dict[str, Any]] = []
        for ch_id_str, current_ch in current_data.items():
            snap_ch = snap["state"].get(ch_id_str)
            if snap_ch is None:
                continue
            self._diff_dicts(
                int(ch_id_str), "", current_ch, snap_ch, differences
            )
        return differences

    @staticmethod
    def _diff_dicts(
        channel_id: int,
        prefix: str,
        a: Any,
        b: Any,
        out: List[Dict[str, Any]],
    ) -> None:
        """Recursively compare two nested dicts/values and collect differences."""
        if isinstance(a, dict) and isinstance(b, dict):
            all_keys = set(a.keys()) | set(b.keys())
            for key in sorted(all_keys):
                path = f"{prefix}.{key}" if prefix else key
                MixerState._diff_dicts(
                    channel_id, path, a.get(key), b.get(key), out
                )
        elif isinstance(a, list) and isinstance(b, list):
            for i in range(max(len(a), len(b))):
                path = f"{prefix}[{i}]"
                va = a[i] if i < len(a) else None
                vb = b[i] if i < len(b) else None
                MixerState._diff_dicts(channel_id, path, va, vb, out)
        else:
            if a != b:
                out.append({
                    "channel": channel_id,
                    "param": prefix,
                    "current": a,
                    "other": b,
                })

    # ------------------------------------------------------------------
    # Change listeners
    # ------------------------------------------------------------------

    def add_listener(self, callback: Any) -> None:
        """
        Register a change listener.

        Callback signature: callback(channel_id, param_path, old_value, new_value)
        """
        with self._listeners_lock:
            self._listeners.append(callback)

    def remove_listener(self, callback: Any) -> bool:
        with self._listeners_lock:
            before = len(self._listeners)
            self._listeners = [c for c in self._listeners if c is not callback]
            return len(self._listeners) < before

    def _notify_change(
        self, channel_id: int, param_path: str, old_value: Any, new_value: Any
    ) -> None:
        if old_value == new_value:
            return
        with self._listeners_lock:
            listeners = list(self._listeners)
        for listener in listeners:
            try:
                listener(channel_id, param_path, old_value, new_value)
            except Exception as exc:
                logger.error("Listener error: %s", exc)

    # ------------------------------------------------------------------
    # JSON import/export
    # ------------------------------------------------------------------

    def export_json(self, path: Optional[str] = None, indent: int = 2) -> str:
        """
        Export the full mixer state to JSON.

        Args:
            path: If given, write to this file path.
            indent: JSON indentation (default 2).

        Returns:
            The JSON string.
        """
        with self._lock:
            data = {
                "version": 1,
                "timestamp": time.time(),
                "channels": self._serialize_all(),
            }

        json_str = json.dumps(data, indent=indent, ensure_ascii=False)

        if path:
            with open(path, "w", encoding="utf-8") as f:
                f.write(json_str)
            logger.info("State exported to %s", path)

        return json_str

    def import_json(self, json_str: Optional[str] = None,
                    path: Optional[str] = None) -> None:
        """
        Import mixer state from JSON.

        Args:
            json_str: JSON string. Mutually exclusive with path.
            path: Path to a JSON file.
        """
        if path:
            with open(path, "r", encoding="utf-8") as f:
                json_str = f.read()
        if not json_str:
            raise ValueError("No JSON data provided")

        data = json.loads(json_str)
        channels_data = data.get("channels", data)

        with self._lock:
            self._deserialize_all(channels_data)

        logger.info("State imported (%d channels)", len(channels_data))

    # ------------------------------------------------------------------
    # Internal serialization helpers
    # ------------------------------------------------------------------

    def _serialize_all(self) -> Dict[str, Any]:
        """Serialize all channels to a dict keyed by channel ID string."""
        return {
            str(ch_id): ch.to_dict()
            for ch_id, ch in self._channels.items()
        }

    def _deserialize_all(self, data: Dict[str, Any]) -> None:
        """Restore channels from serialized data."""
        for ch_id_str, ch_data in data.items():
            ch_id = int(ch_id_str)
            if ch_id in self._channels:
                self._channels[ch_id] = ChannelState.from_dict(ch_data)

    def __repr__(self) -> str:
        return (
            f"<MixerState channels={len(self._channels)} "
            f"snapshots={len(self._snapshots)}>"
        )
