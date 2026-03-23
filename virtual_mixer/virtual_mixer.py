"""
Virtual Wing Mixer - OSC Simulator

Simulates Behringer Wing OSC API for testing Auto Mixer.
- 32 input channels
- 16 group busses
- 16 matrix busses
- Master LR
- Full OSC command support
- Web UI for visualization

OSC Commands:
- /ch/{01-32}/mix/fader - Channel fader (0.0-1.0)
- /ch/{01-32}/mix/on - Channel on/off (0/1)
- /ch/{01-32}/mix/pan - Pan (-1.0 to 1.0)
- /ch/{01-32}/gain - Preamp gain (-12 to +60 dB)
- /ch/{01-32}/eq/on - EQ on/off
- /ch/{01-32}/dyn/on - Compressor on/off
- /ch/{01-32}/gate/on - Gate on/off
- /bus/{01-16}/mix/fader - Group fader
- /mtx/{01-16}/mix/fader - Matrix fader
- /main/st/mix/fader - Master fader
"""

import asyncio
import logging
import numpy as np
import json
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field, asdict
from enum import Enum
from pythonosc.udp_client import SimpleUDPClient
from pythonosc.dispatcher import Dispatcher
from pythonosc.osc_server import AsyncIOOSCUDPServer

logger = logging.getLogger(__name__)


class ChannelType(Enum):
    INPUT = "input"
    GROUP = "group"
    MATRIX = "matrix"
    MASTER = "master"


@dataclass
class EQBand:
    """4-band EQ."""
    freq: float = 1000.0
    gain: float = 0.0
    q: float = 1.0
    on: bool = True
    type: str = "bell"  # bell, high_shelf, low_shelf
    
    def to_dict(self):
        return asdict(self)


@dataclass
class Dynamics:
    """Compressor/Limiter."""
    on: bool = False
    threshold_db: float = -20.0
    ratio: float = 3.0
    attack_ms: float = 10.0
    release_ms: float = 100.0
    gain_db: float = 0.0
    
    def to_dict(self):
        return asdict(self)


@dataclass
class Gate:
    """Noise gate."""
    on: bool = False
    threshold_db: float = -60.0
    attack_ms: float = 0.5
    hold_ms: float = 10.0
    release_ms: float = 80.0
    range_db: float = -80.0
    
    def to_dict(self):
        return asdict(self)


@dataclass
class Channel:
    """Mixer channel."""
    id: int
    name: str = ""
    ch_type: ChannelType = ChannelType.INPUT
    
    # Mix
    fader: float = 0.75  # 0.0-1.0 (-∞ to +10dB)
    pan: float = 0.0     # -1.0 to 1.0
    on: bool = True
    mute: bool = False
    solo: bool = False
    
    # Preamp
    gain_db: float = 0.0  # -12 to +60 dB
    phantom: bool = False
    invert: bool = False
    
    # EQ (4 bands)
    eq_on: bool = True
    eq_bands: List[EQBand] = field(default_factory=lambda: [
        EQBand(freq=80, type="low_shelf"),
        EQBand(freq=250, q=1.4),
        EQBand(freq=2500, q=1.4),
        EQBand(freq=8000, type="high_shelf")
    ])
    
    # Dynamics
    dyn: Dynamics = field(default_factory=Dynamics)
    
    # Gate
    gate: Gate = field(default_factory=Gate)
    
    # Audio
    input_signal: float = -100.0  # dB
    output_level: float = -100.0  # dB
    
    def to_dict(self):
        return {
            'id': self.id,
            'name': self.name,
            'type': self.ch_type.value,
            'fader': self.fader,
            'fader_db': self.fader_to_db(self.fader),
            'pan': self.pan,
            'on': self.on,
            'mute': self.mute,
            'solo': self.solo,
            'gain_db': self.gain_db,
            'phantom': self.phantom,
            'invert': self.invert,
            'eq_on': self.eq_on,
            'eq_bands': [b.to_dict() for b in self.eq_bands],
            'dyn': self.dyn.to_dict(),
            'gate': self.gate.to_dict(),
            'input_signal': self.input_signal,
            'output_level': self.output_level
        }
    
    @staticmethod
    def fader_to_db(fader: float) -> float:
        """Convert fader 0-1 to dB."""
        if fader <= 0:
            return -np.inf
        return 20 * np.log10(fader * 3.16)  # 0-1 -> -∞ to +10dB
    
    @staticmethod
    def db_to_fader(db: float) -> float:
        """Convert dB to fader 0-1."""
        if db <= -100:
            return 0.0
        return min(1.0, 10 ** (db / 20) / 3.16)


@dataclass
class MasterBus:
    """Master stereo bus."""
    fader: float = 0.75
    on: bool = True
    pan: float = 0.0
    
    def to_dict(self):
        return {
            'fader': self.fader,
            'fader_db': Channel.fader_to_db(self.fader),
            'on': self.on,
            'pan': self.pan
        }


class VirtualWingMixer:
    """
    Virtual Behringer Wing Mixer.
    
    Simulates complete OSC API for testing.
    """
    
    def __init__(self, osc_port: int = 2222, client_port: int = 3333):
        self.osc_port = osc_port
        self.client_port = client_port
        
        # Channels
        self.input_channels: Dict[int, Channel] = {}
        self.group_busses: Dict[int, Channel] = {}
        self.matrix_busses: Dict[int, Channel] = {}
        self.master = MasterBus()
        
        # Initialize
        self._init_channels()
        
        # OSC
        self.dispatcher = Dispatcher()
        self.server = None
        self.client = None
        
        # Callbacks
        self.on_parameter_change: Optional[callable] = None
        
        logger.info(f"VirtualWingMixer initialized: {len(self.input_channels)} inputs, "
                   f"{len(self.group_busses)} groups, {len(self.matrix_busses)} matrix")
    
    def _init_channels(self):
        """Initialize all channels."""
        # Input channels 1-32
        for i in range(1, 33):
            ch = Channel(id=i, name=f"CH{i:02d}", ch_type=ChannelType.INPUT)
            
            # Default names based on typical setup
            default_names = {
                1: "Kick In", 2: "Kick Out", 3: "Snare Top", 4: "Snare Bot",
                5: "HiHat", 6: "Tom1", 7: "Tom2", 8: "Tom3",
                9: "OH L", 10: "OH R", 11: "Room L", 12: "Room R",
                13: "Bass DI", 14: "Bass Amp", 15: "Guitar L", 16: "Guitar R",
                17: "Keys L", 18: "Keys R", 19: "Lead Vox", 20: "Back Vox 1",
                21: "Back Vox 2", 22: "Back Vox 3", 23: "Talkback", 24: "Playback L",
                25: "Playback R", 26: "Synth 1", 27: "Synth 2", 28: "Trumpet",
                29: "Sax", 30: "Trombone", 31: "Spare 1", 32: "Spare 2"
            }
            if i in default_names:
                ch.name = default_names[i]
            
            self.input_channels[i] = ch
        
        # Group busses 1-16
        for i in range(1, 17):
            ch = Channel(id=i, name=f"GRP{i:02d}", ch_type=ChannelType.GROUP)
            self.group_busses[i] = ch
        
        # Matrix busses 1-16
        for i in range(1, 17):
            ch = Channel(id=i, name=f"MTX{i:02d}", ch_type=ChannelType.MATRIX)
            self.matrix_busses[i] = ch
    
    def setup_osc(self):
        """Setup OSC server and client."""
        # Dispatcher
        self.dispatcher.set_default_handler(self._default_handler)
        
        # Channel handlers (Wing format: /ch/{n}/fdr, /ch/{n}/mix/fader)
        for i in range(1, 33):
            self._setup_channel_handlers(i)
            # Wing native /ch/{n}/fdr (dB) - for WingClient compatibility
            self.dispatcher.map(
                f"/ch/{i}/fdr",
                (lambda c: lambda addr, *args: self._handle_ch_fdr(c, args))(i)
            )
        
        # Group handlers
        for i in range(1, 17):
            self._setup_bus_handlers(i, "bus")
        
        # Matrix handlers
        for i in range(1, 17):
            self._setup_bus_handlers(i, "mtx")
        
        # Master handlers
        self.dispatcher.map("/main/st/mix/fader", self._handle_master_fader)
        self.dispatcher.map("/main/st/mix/on", self._handle_master_on)
        self.dispatcher.map("/main/st/mix/pan", self._handle_master_pan)
        
        # XRemote (keep alive)
        self.dispatcher.map("/xremote", self._handle_xremote)
        
        logger.info("OSC handlers registered")
    
    def _setup_channel_handlers(self, ch_num: int):
        """Setup OSC handlers for input channel."""
        prefix = f"/ch/{ch_num:02d}"
        
        # Mix
        self.dispatcher.map(f"{prefix}/mix/fader", 
                           lambda addr, val, ch=ch_num: self._handle_ch_fader(ch, val))
        self.dispatcher.map(f"{prefix}/mix/on",
                           lambda addr, val, ch=ch_num: self._handle_ch_on(ch, val))
        self.dispatcher.map(f"{prefix}/mix/pan",
                           lambda addr, val, ch=ch_num: self._handle_ch_pan(ch, val))
        self.dispatcher.map(f"{prefix}/mix/mute",
                           lambda addr, val, ch=ch_num: self._handle_ch_mute(ch, val))
        self.dispatcher.map(f"{prefix}/mix/solo",
                           lambda addr, val, ch=ch_num: self._handle_ch_solo(ch, val))
        
        # Preamp
        self.dispatcher.map(f"{prefix}/preamp/gain",
                           lambda addr, val, ch=ch_num: self._handle_ch_gain(ch, val))
        self.dispatcher.map(f"{prefix}/preamp/phantom",
                           lambda addr, val, ch=ch_num: self._handle_ch_phantom(ch, val))
        
        # EQ
        self.dispatcher.map(f"{prefix}/eq/on",
                           lambda addr, val, ch=ch_num: self._handle_ch_eq_on(ch, val))
        
        for band in range(1, 5):
            self.dispatcher.map(f"{prefix}/eq/{band}/freq",
                               lambda addr, val, ch=ch_num, b=band: self._handle_ch_eq_freq(ch, b, val))
            self.dispatcher.map(f"{prefix}/eq/{band}/gain",
                               lambda addr, val, ch=ch_num, b=band: self._handle_ch_eq_gain(ch, b, val))
            self.dispatcher.map(f"{prefix}/eq/{band}/q",
                               lambda addr, val, ch=ch_num, b=band: self._handle_ch_eq_q(ch, b, val))
        
        # Dynamics
        self.dispatcher.map(f"{prefix}/dyn/on",
                           lambda addr, val, ch=ch_num: self._handle_ch_dyn_on(ch, val))
        self.dispatcher.map(f"{prefix}/dyn/thr",
                           lambda addr, val, ch=ch_num: self._handle_ch_dyn_thr(ch, val))
        self.dispatcher.map(f"{prefix}/dyn/ratio",
                           lambda addr, val, ch=ch_num: self._handle_ch_dyn_ratio(ch, val))
        
        # Gate
        self.dispatcher.map(f"{prefix}/gate/on",
                           lambda addr, val, ch=ch_num: self._handle_ch_gate_on(ch, val))
        self.dispatcher.map(f"{prefix}/gate/thr",
                           lambda addr, val, ch=ch_num: self._handle_ch_gate_thr(ch, val))
    
    def _setup_bus_handlers(self, bus_num: int, bus_type: str):
        """Setup OSC handlers for group/matrix bus."""
        prefix = f"/{bus_type}/{bus_num:02d}"
        
        self.dispatcher.map(f"{prefix}/mix/fader",
                           lambda addr, val, b=bus_num, t=bus_type: self._handle_bus_fader(b, t, val))
        self.dispatcher.map(f"{prefix}/mix/on",
                           lambda addr, val, b=bus_num, t=bus_type: self._handle_bus_on(b, t, val))
    
    # Handlers
    def _handle_ch_fdr(self, ch: int, args: tuple):
        """Handle /ch/{n}/fdr - Wing native fader (dB). Get (no args) or set (with value)."""
        if args:
            db_val = float(args[0])
            fader = Channel.db_to_fader(db_val)
            self.input_channels[ch].fader = fader
            self._notify_change(f"ch_{ch}_fader", fader)
            logger.debug(f"CH{ch:02d} fdr: {db_val:.1f} dB")
        else:
            # Get request - return current value in dB
            fader = self.input_channels[ch].fader
            db_val = Channel.fader_to_db(fader)
            return (f"/ch/{ch}/fdr", db_val)

    def _handle_ch_fader(self, ch: int, value: float):
        self.input_channels[ch].fader = float(value)
        self._notify_change(f"ch_{ch}_fader", value)
        logger.debug(f"CH{ch:02d} fader: {value:.2f}")
    
    def _handle_ch_on(self, ch: int, value: int):
        self.input_channels[ch].on = bool(value)
        self._notify_change(f"ch_{ch}_on", value)
    
    def _handle_ch_pan(self, ch: int, value: float):
        self.input_channels[ch].pan = float(value)
        self._notify_change(f"ch_{ch}_pan", value)
    
    def _handle_ch_mute(self, ch: int, value: int):
        self.input_channels[ch].mute = bool(value)
        self._notify_change(f"ch_{ch}_mute", value)
    
    def _handle_ch_solo(self, ch: int, value: int):
        self.input_channels[ch].solo = bool(value)
        self._notify_change(f"ch_{ch}_solo", value)
    
    def _handle_ch_gain(self, ch: int, value: float):
        self.input_channels[ch].gain_db = float(value)
        self._notify_change(f"ch_{ch}_gain", value)
    
    def _handle_ch_phantom(self, ch: int, value: int):
        self.input_channels[ch].phantom = bool(value)
        self._notify_change(f"ch_{ch}_phantom", value)
    
    def _handle_ch_eq_on(self, ch: int, value: int):
        self.input_channels[ch].eq_on = bool(value)
        self._notify_change(f"ch_{ch}_eq_on", value)
    
    def _handle_ch_eq_freq(self, ch: int, band: int, value: float):
        if 1 <= band <= 4:
            self.input_channels[ch].eq_bands[band-1].freq = float(value)
            self._notify_change(f"ch_{ch}_eq_{band}_freq", value)
    
    def _handle_ch_eq_gain(self, ch: int, band: int, value: float):
        if 1 <= band <= 4:
            self.input_channels[ch].eq_bands[band-1].gain = float(value)
            self._notify_change(f"ch_{ch}_eq_{band}_gain", value)
    
    def _handle_ch_eq_q(self, ch: int, band: int, value: float):
        if 1 <= band <= 4:
            self.input_channels[ch].eq_bands[band-1].q = float(value)
            self._notify_change(f"ch_{ch}_eq_{band}_q", value)
    
    def _handle_ch_dyn_on(self, ch: int, value: int):
        self.input_channels[ch].dyn.on = bool(value)
        self._notify_change(f"ch_{ch}_dyn_on", value)
    
    def _handle_ch_dyn_thr(self, ch: int, value: float):
        self.input_channels[ch].dyn.threshold_db = float(value)
        self._notify_change(f"ch_{ch}_dyn_thr", value)
    
    def _handle_ch_dyn_ratio(self, ch: int, value: float):
        self.input_channels[ch].dyn.ratio = float(value)
        self._notify_change(f"ch_{ch}_dyn_ratio", value)
    
    def _handle_ch_gate_on(self, ch: int, value: int):
        self.input_channels[ch].gate.on = bool(value)
        self._notify_change(f"ch_{ch}_gate_on", value)
    
    def _handle_ch_gate_thr(self, ch: int, value: float):
        self.input_channels[ch].gate.threshold_db = float(value)
        self._notify_change(f"ch_{ch}_gate_thr", value)
    
    def _handle_bus_fader(self, bus: int, bus_type: str, value: float):
        if bus_type == "bus":
            self.group_busses[bus].fader = float(value)
        else:
            self.matrix_busses[bus].fader = float(value)
        self._notify_change(f"{bus_type}_{bus}_fader", value)
    
    def _handle_bus_on(self, bus: int, bus_type: str, value: int):
        if bus_type == "bus":
            self.group_busses[bus].on = bool(value)
        else:
            self.matrix_busses[bus].on = bool(value)
        self._notify_change(f"{bus_type}_{bus}_on", value)
    
    def _handle_master_fader(self, addr, value):
        self.master.fader = float(value)
        self._notify_change("master_fader", value)
    
    def _handle_master_on(self, addr, value):
        self.master.on = bool(value)
        self._notify_change("master_on", value)
    
    def _handle_master_pan(self, addr, value):
        self.master.pan = float(value)
        self._notify_change("master_pan", value)
    
    def _handle_xremote(self, addr, *args):
        """Handle xremote (keep alive)."""
        pass
    
    def _default_handler(self, addr, *args):
        """Default handler for unmapped addresses."""
        logger.debug(f"Unhandled OSC: {addr} {args}")
    
    def _notify_change(self, param: str, value: Any):
        """Notify about parameter change."""
        if self.on_parameter_change:
            self.on_parameter_change(param, value)
    
    # Server control
    async def start(self):
        """Start OSC server with WING? handshake support for WingClient compatibility."""
        self.setup_osc()
        loop = asyncio.get_event_loop()

        class WingHandshakeProtocol(asyncio.DatagramProtocol):
            """Handles WING? handshake and OSC."""
            def __init__(self, dispatcher):
                self.dispatcher = dispatcher
                self.transport = None

            def connection_made(self, transport):
                self.transport = transport

            def datagram_received(self, data: bytes, addr):
                if data == b'WING?':
                    self.transport.sendto(b'WING', addr)
                    logger.info(f"WING? handshake from {addr}")
                    return
                try:
                    from pythonosc.osc_message_builder import build_msg
                    resp = self.dispatcher.call_handlers_for_packet(data, addr)
                    for r in resp:
                        if not isinstance(r, tuple):
                            r = [r]
                        msg = build_msg(r[0], r[1:])
                        self.transport.sendto(msg.dgram, addr)
                except Exception as e:
                    logger.warning(f"OSC from {addr}: {e}")

        transport, protocol = await loop.create_datagram_endpoint(
            lambda: WingHandshakeProtocol(self.dispatcher),
            local_addr=("0.0.0.0", self.osc_port),
        )
        self._transport = transport

        logger.info(f"OSC server started on port {self.osc_port} (WING? handshake enabled)")
        return transport
    
    def get_state(self) -> Dict[str, Any]:
        """Get complete mixer state."""
        return {
            'inputs': {i: ch.to_dict() for i, ch in self.input_channels.items()},
            'groups': {i: ch.to_dict() for i, ch in self.group_busses.items()},
            'matrix': {i: ch.to_dict() for i, ch in self.matrix_busses.items()},
            'master': self.master.to_dict()
        }
    
    def set_input_signal(self, ch: int, level_db: float):
        """Set input signal level (for simulation)."""
        if ch in self.input_channels:
            self.input_channels[ch].input_signal = level_db
            # Calculate output with fader
            fader_db = Channel.fader_to_db(self.input_channels[ch].fader)
            self.input_channels[ch].output_level = level_db + fader_db + self.input_channels[ch].gain_db


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    mixer = VirtualWingMixer(osc_port=2222)
    
    loop = asyncio.get_event_loop()
    transport = loop.run_until_complete(mixer.start())
    
    print("Virtual Wing Mixer running on port 2222")
    print("Press Ctrl+C to stop")
    
    try:
        loop.run_forever()
    except KeyboardInterrupt:
        pass
    finally:
        transport.close()
        loop.close()
