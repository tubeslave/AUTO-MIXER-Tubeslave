"""
Auto Soundcheck Engine — headless orchestrator for automatic mixing.

Connects to mixer (dLive or WING), detects audio device (SoundGrid/Dante),
reads channel names, recognizes instruments, waits for audio signals,
and automatically applies gain staging, EQ, compressor, and fader corrections
without user confirmation.

This is the main entry point for fully automatic operation.
"""

import asyncio
import logging
import time
import threading
import json
import numpy as np
from typing import Any, Callable, Dict, List, Optional, Tuple
from dataclasses import dataclass, field
from enum import Enum

from audio_capture import (
    AudioCapture, AudioSourceType, AudioDeviceType,
    detect_audio_device, find_device_by_name, list_audio_devices,
)
from channel_recognizer import (
    recognize_instrument, recognize_instrument_spectral_fallback,
    scan_and_recognize, AVAILABLE_PRESETS,
)
from feedback_detector import FeedbackDetector, FeedbackEvent
from mixer_discovery import (
    discover_mixer_auto, discover_mixers, DiscoveredMixer,
)
from audio_device_scanner import (
    scan_audio_devices, select_best_device, detect_and_report,
    AudioDevice, AudioProtocol,
)
from signal_metrics import (
    SignalAnalyzer, ChannelMetrics, compare_channels,
    InterChannelMetrics, LevelMetrics, DynamicsMetrics, SpectralMetrics,
)
from observation_mixer import ObservationMixerClient

logger = logging.getLogger(__name__)


class EngineState(Enum):
    IDLE = "idle"
    DISCOVERING = "discovering"
    CONNECTING = "connecting"
    SCANNING_AUDIO = "scanning_audio"
    SCANNING_CHANNELS = "scanning_channels"
    READING_STATE = "reading_state"
    RESETTING = "resetting"
    WAITING_FOR_SIGNAL = "waiting_for_signal"
    ANALYZING = "analyzing"
    APPLYING = "applying"
    RUNNING = "running"
    ERROR = "error"
    STOPPED = "stopped"


@dataclass
class ChannelSnapshot:
    """Backup of a channel's settings before reset."""
    channel: int
    fader_db: float = -100.0
    muted: bool = False
    eq_bands: Optional[List[Tuple[float, float, float]]] = None
    hpf_freq: float = 20.0
    hpf_enabled: bool = False
    gain_db: float = 0.0
    had_processing: bool = False
    raw_settings: Optional[Dict] = None


@dataclass
class ChannelInfo:
    """Per-channel information gathered during soundcheck."""
    channel: int
    name: str = ""
    preset: Optional[str] = None
    recognized: bool = False
    has_signal: bool = False
    peak_db: float = -100.0
    rms_db: float = -100.0
    lufs: float = -100.0
    gain_correction_db: float = 0.0
    eq_applied: bool = False
    compressor_applied: bool = False
    fader_db: float = -100.0
    spectral_centroid: float = 0.0
    was_reset: bool = False
    original_snapshot: Optional[ChannelSnapshot] = None
    metrics: Optional[ChannelMetrics] = None
    analyzer: Optional[Any] = None


# Instrument-specific EQ presets: (freq_hz, gain_db, q)
# 4-band PEQ per instrument type
INSTRUMENT_EQ_PRESETS: Dict[str, List[Tuple[float, float, float]]] = {
    "kick": [
        (60.0, 3.0, 1.0),      # Sub thump
        (200.0, -3.0, 2.0),    # Cut mud
        (3500.0, 2.0, 2.0),    # Attack click
        (8000.0, -2.0, 1.5),   # Reduce bleed
    ],
    "snare": [
        (120.0, 1.5, 1.5),     # Body
        (400.0, -2.0, 2.0),    # Cut boxiness
        (2500.0, 2.0, 2.5),    # Crack
        (8000.0, 1.0, 1.5),    # Sizzle
    ],
    "tom": [
        (80.0, 2.0, 1.2),      # Low body
        (300.0, -2.5, 2.0),    # Reduce boxiness
        (3000.0, 1.5, 2.0),    # Attack
        (7000.0, -1.5, 1.5),   # Reduce bleed
    ],
    "hihat": [
        (200.0, -4.0, 1.0),    # HPF slope assist
        (1000.0, -1.0, 1.5),   # Cut honk
        (6000.0, 1.5, 2.0),    # Presence
        (12000.0, 1.0, 1.0),   # Air
    ],
    "ride": [
        (200.0, -3.0, 1.0),    # HPF slope assist
        (800.0, -1.0, 2.0),    # Cut mud
        (3000.0, 1.0, 2.0),    # Definition
        (10000.0, 1.5, 1.5),   # Shimmer
    ],
    "cymbals": [
        (200.0, -3.0, 1.0),    # Cut lows
        (1000.0, -1.5, 2.0),   # Cut mid
        (5000.0, 1.0, 2.0),    # Presence
        (12000.0, 1.0, 1.0),   # Air
    ],
    "overheads": [
        (100.0, -2.0, 1.0),    # Trim lows
        (500.0, -1.0, 1.5),    # Cut mud
        (4000.0, 1.5, 2.0),    # Presence
        (12000.0, 1.0, 1.0),   # Air
    ],
    "room": [
        (80.0, -3.0, 0.8),     # Cut rumble
        (400.0, -1.0, 1.5),    # Cut box
        (2000.0, 1.0, 2.0),    # Clarity
        (10000.0, 0.5, 1.0),   # Air
    ],
    "bass": [
        (60.0, 2.0, 1.2),      # Sub
        (250.0, -2.0, 2.0),    # Cut mud
        (800.0, 1.0, 2.0),     # Growl/definition
        (3000.0, 1.5, 2.5),    # Pick attack
    ],
    "electricGuitar": [
        (100.0, -2.0, 1.0),    # Cut sub lows
        (500.0, -1.5, 2.0),    # Reduce mud
        (3000.0, 2.0, 2.0),    # Presence
        (6000.0, 1.0, 1.5),    # Bite
    ],
    "acousticGuitar": [
        (80.0, -3.0, 1.0),     # Cut rumble
        (250.0, -1.5, 2.0),    # Reduce body boom
        (3000.0, 2.0, 2.0),    # String definition
        (8000.0, 1.0, 1.5),    # Air
    ],
    "accordion": [
        (100.0, -2.0, 1.0),    # Cut sub
        (500.0, -1.0, 2.0),    # Reduce box
        (2500.0, 1.5, 2.0),    # Reed clarity
        (6000.0, 1.0, 1.5),    # Presence
    ],
    "synth": [
        (40.0, -1.0, 1.0),     # Trim sub
        (300.0, -1.0, 2.0),    # Reduce mud
        (2000.0, 1.0, 2.0),    # Clarity
        (8000.0, 1.5, 1.5),    # Brightness
    ],
    "playback": [
        (60.0, 0.0, 1.0),      # Flat low
        (300.0, -1.0, 1.5),    # Slight mud cut
        (3000.0, 0.5, 2.0),    # Slight presence
        (10000.0, 0.5, 1.0),   # Air
    ],
    "leadVocal": [
        (80.0, -4.0, 0.8),     # HPF assist
        (300.0, -2.0, 2.0),    # Reduce mud
        (3000.0, 2.5, 2.5),    # Presence
        (8000.0, 1.5, 1.5),    # Air
    ],
    "backVocal": [
        (100.0, -3.0, 0.8),    # HPF assist
        (400.0, -2.0, 2.0),    # Reduce mud
        (2500.0, 1.5, 2.0),    # Clarity
        (6000.0, 1.0, 1.5),    # Air
    ],
    "custom": [
        (60.0, 0.0, 1.0),
        (300.0, -1.0, 1.5),
        (3000.0, 0.5, 2.0),
        (10000.0, 0.5, 1.0),
    ],
}

# HPF frequencies per instrument type
INSTRUMENT_HPF: Dict[str, float] = {
    "kick": 30.0,
    "snare": 80.0,
    "tom": 60.0,
    "hihat": 300.0,
    "ride": 250.0,
    "cymbals": 300.0,
    "overheads": 120.0,
    "room": 80.0,
    "bass": 30.0,
    "electricGuitar": 80.0,
    "acousticGuitar": 80.0,
    "accordion": 80.0,
    "synth": 30.0,
    "playback": 30.0,
    "leadVocal": 80.0,
    "backVocal": 100.0,
    "custom": 80.0,
}

# Target LUFS per instrument category
INSTRUMENT_TARGET_LUFS: Dict[str, float] = {
    "kick": -25.0,
    "snare": -25.0,
    "tom": -27.0,
    "hihat": -35.0,
    "ride": -35.0,
    "cymbals": -35.0,
    "overheads": -30.0,
    "room": -35.0,
    "bass": -23.0,
    "electricGuitar": -23.0,
    "acousticGuitar": -25.0,
    "accordion": -23.0,
    "synth": -22.0,
    "playback": -23.0,
    "leadVocal": -20.0,
    "backVocal": -23.0,
    "custom": -23.0,
}

# Compressor presets per instrument: (threshold_db, ratio, attack_ms, release_ms)
INSTRUMENT_COMPRESSOR: Dict[str, Tuple[float, float, float, float]] = {
    "kick": (-20.0, 4.0, 10.0, 80.0),
    "snare": (-18.0, 3.5, 5.0, 100.0),
    "tom": (-18.0, 3.5, 8.0, 100.0),
    "hihat": (-25.0, 2.5, 5.0, 60.0),
    "ride": (-25.0, 2.0, 10.0, 100.0),
    "cymbals": (-25.0, 2.0, 10.0, 100.0),
    "overheads": (-20.0, 2.0, 15.0, 150.0),
    "room": (-20.0, 3.0, 10.0, 200.0),
    "bass": (-15.0, 4.0, 10.0, 100.0),
    "electricGuitar": (-18.0, 3.0, 10.0, 100.0),
    "acousticGuitar": (-18.0, 3.0, 15.0, 150.0),
    "accordion": (-18.0, 2.5, 15.0, 150.0),
    "synth": (-18.0, 2.5, 10.0, 100.0),
    "playback": (-20.0, 2.0, 10.0, 100.0),
    "leadVocal": (-15.0, 3.0, 5.0, 80.0),
    "backVocal": (-18.0, 3.0, 5.0, 80.0),
    "custom": (-20.0, 2.5, 10.0, 100.0),
}

# Default pan positions per instrument (-100=L, 0=C, +100=R)
INSTRUMENT_PAN: Dict[str, float] = {
    "kick": 0.0,
    "snare": 0.0,
    "tom": -15.0,       # Slightly left (Tom 1 default; engine adjusts per Tom number)
    "hihat": 30.0,      # Slightly right (audience perspective)
    "ride": -30.0,      # Slightly left
    "cymbals": 0.0,
    "overheads": 0.0,   # Handled as stereo pair: L=-60, R=+60
    "room": 0.0,
    "bass": 0.0,
    "electricGuitar": -25.0,
    "acousticGuitar": 25.0,
    "accordion": 15.0,
    "synth": 0.0,       # Stereo pair: L=-40, R=+40
    "playback": 0.0,
    "leadVocal": 0.0,
    "backVocal": 0.0,   # Spread: BVox1=-30, BVox2=+30
    "custom": 0.0,
}

# Target input gain (preamp trim) in dB per instrument
INSTRUMENT_TARGET_INPUT_DB: Dict[str, float] = {
    "kick": -10.0,
    "snare": -10.0,
    "tom": -10.0,
    "hihat": -15.0,
    "ride": -15.0,
    "cymbals": -15.0,
    "overheads": -15.0,
    "room": -10.0,
    "bass": -10.0,
    "electricGuitar": -10.0,
    "acousticGuitar": -10.0,
    "accordion": -10.0,
    "synth": -15.0,
    "playback": -15.0,
    "leadVocal": -8.0,
    "backVocal": -10.0,
    "custom": -12.0,
}

# FX send presets: reverb bus, delay bus (send_bus_number, level_db)
# Assumes: Bus 1 = Reverb (plate/hall), Bus 2 = Delay
INSTRUMENT_FX_SENDS: Dict[str, List[Tuple[int, float]]] = {
    "kick": [],
    "snare": [(1, -20.0)],            # Light reverb
    "tom": [(1, -22.0)],
    "hihat": [],
    "ride": [],
    "cymbals": [],
    "overheads": [(1, -25.0)],
    "room": [],
    "bass": [],
    "electricGuitar": [(1, -18.0), (2, -25.0)],   # Reverb + delay
    "acousticGuitar": [(1, -15.0)],
    "accordion": [(1, -18.0)],
    "synth": [(1, -20.0), (2, -22.0)],
    "playback": [],
    "leadVocal": [(1, -12.0), (2, -18.0)],         # More reverb + delay
    "backVocal": [(1, -15.0), (2, -20.0)],
    "custom": [],
}

# Channels that typically come in correlated pairs (for phase check)
STEREO_PAIR_PRESETS = ["overheads", "synth", "playback"]

SIGNAL_THRESHOLD_DB = -50.0
SIGNAL_ANALYSIS_SECONDS = 3.0
GAIN_CORRECTION_MAX_DB = 12.0


class AutoSoundcheckEngine:
    """
    Fully automatic soundcheck engine.

    Workflow:
    0. Auto-discover mixer on the network (WING broadcast / dLive TCP scan)
    1. Connect to mixer (dLive / WING)
    2. Detect audio device (SoundGrid / Dante)
    3. Start audio capture
    4. Read channel names from mixer → recognize instruments
    5. Wait for audio signal on each channel
    6. For channels with signal:
       a. Analyze spectrum → spectral instrument classification fallback
       b. Compute gain correction (target LUFS per instrument)
       c. Apply HPF, EQ preset, compressor preset
       d. Set fader to initial position
    7. Enter running mode: continuous feedback detection + level monitoring
    """

    def __init__(
        self,
        mixer_type: Optional[str] = None,
        mixer_ip: Optional[str] = None,
        mixer_port: Optional[int] = None,
        mixer_tls: bool = False,
        midi_base_channel: int = 0,
        audio_device_name: Optional[str] = None,
        num_channels: int = 48,
        sample_rate: int = 48000,
        block_size: int = 1024,
        buffer_seconds: float = 5.0,
        auto_apply: bool = True,
        observe_only: bool = False,
        selected_channels: Optional[List[int]] = None,
        auto_discover: bool = True,
        scan_subnet: bool = True,
        on_state_change: Optional[Callable] = None,
        on_channel_update: Optional[Callable] = None,
        on_observation: Optional[Callable] = None,
    ):
        self.mixer_type = mixer_type
        self.mixer_ip = mixer_ip
        self.mixer_port = mixer_port
        self.mixer_tls = mixer_tls
        self.midi_base_channel = midi_base_channel
        self.audio_device_name = audio_device_name
        self.num_channels = num_channels
        self.sample_rate = sample_rate
        self.block_size = block_size
        self.buffer_seconds = buffer_seconds
        self.auto_apply = auto_apply
        self.observe_only = observe_only
        self.auto_discover = auto_discover
        self.scan_subnet = scan_subnet
        self.selected_channels = sorted({
            int(ch) for ch in (selected_channels or [])
            if int(ch) > 0
        })

        self.mixer_client = None
        self._real_mixer_client = None
        self._observation_mixer_client: Optional[ObservationMixerClient] = None
        self.audio_capture: Optional[AudioCapture] = None
        self.feedback_detector: Optional[FeedbackDetector] = None
        self.discovered_mixer: Optional[DiscoveredMixer] = None
        self.selected_audio_device: Optional[AudioDevice] = None
        self.audio_devices: List[AudioDevice] = []

        self.state = EngineState.IDLE
        self.channels: Dict[int, ChannelInfo] = {}
        self._stop_event = threading.Event()
        self._engine_thread: Optional[threading.Thread] = None
        self._monitor_thread: Optional[threading.Thread] = None

        self.on_state_change = on_state_change
        self.on_channel_update = on_channel_update
        self.on_observation = on_observation

        self._applied_channels: set = set()
        self._analyzers: Dict[int, SignalAnalyzer] = {}

        mode = "auto-discover" if auto_discover and not mixer_ip else "manual"
        target = f"{mixer_type or '?'}@{mixer_ip or 'auto'}:{mixer_port or 'auto'}"
        logger.info(
            f"AutoSoundcheckEngine: {mode}, target={target}, "
            f"channels={self._configured_channel_count()}, sr={sample_rate}, "
            f"observe_only={observe_only}"
        )

    def _configured_channel_count(self) -> int:
        if self.selected_channels:
            return len(self.selected_channels)
        return self.num_channels

    def _iter_channels(self) -> List[int]:
        if self.selected_channels:
            return [ch for ch in self.selected_channels if ch <= self.num_channels]
        return list(range(1, self.num_channels + 1))

    def _emit_observation(self, operation: Dict[str, Any] = None, message: str = "", summary: Dict[str, Any] = None):
        if not self.on_observation:
            return
        payload = {}
        if message:
            payload["message"] = message
        if operation is not None:
            payload["operation"] = operation
            payload.setdefault("message", f"[OBSERVE] {operation.get('message', '')}")
        if summary is not None:
            payload["summary"] = summary
        try:
            self.on_observation(payload)
        except Exception:
            pass

    def _activate_observation_mode(self):
        if not self.observe_only or not self.mixer_client:
            return
        if isinstance(self.mixer_client, ObservationMixerClient):
            return
        self._real_mixer_client = self.mixer_client
        self._observation_mixer_client = ObservationMixerClient(
            self._real_mixer_client,
            on_command=lambda op: self._emit_observation(operation=op),
        )
        self.mixer_client = self._observation_mixer_client
        self._emit_observation(message="[OBSERVE] Engine observation mode active: mixer writes are intercepted.")

    def _set_state(self, new_state: EngineState, message: str = ""):
        old = self.state
        self.state = new_state
        logger.info(f"Engine state: {old.value} -> {new_state.value} {message}")
        if self.on_state_change:
            try:
                self.on_state_change(new_state.value, message)
            except Exception:
                pass

    # ── 0. Auto-discover mixer ───────────────────────────────────

    def _discover_mixer(self) -> bool:
        """Scan the network for available mixers and select the best one."""
        self._set_state(EngineState.DISCOVERING, "Scanning network for mixers...")

        discovered = discover_mixer_auto(
            preferred_type=self.mixer_type,
            preferred_ip=self.mixer_ip,
            scan_subnet=self.scan_subnet,
            timeout=2.0,
        )

        if discovered is None:
            logger.warning("No mixer found on the network")
            return False

        self.discovered_mixer = discovered
        self.mixer_type = discovered.mixer_type
        self.mixer_ip = discovered.ip
        self.mixer_port = discovered.port
        self.mixer_tls = discovered.tls

        logger.info(
            f"Auto-discovered: {discovered.mixer_type.upper()} "
            f"@ {discovered.ip}:{discovered.port} "
            f"('{discovered.name}', {discovered.discovery_method}, "
            f"{discovered.response_time_ms:.0f}ms)"
        )
        return True

    # ── 1. Connect to mixer ──────────────────────────────────────

    def _connect_mixer(self) -> bool:
        """Connect to the mixer console.

        If auto_discover is True and no IP is specified, runs network
        discovery first to find the mixer automatically.
        """
        need_discovery = self.auto_discover and (
            self.mixer_ip is None or self.mixer_type is None
        )

        if need_discovery:
            if not self._discover_mixer():
                # Discovery failed — try known defaults as last resort
                if self.mixer_ip is None:
                    self._set_state(
                        EngineState.ERROR,
                        "No mixer found on the network and no IP specified",
                    )
                    return False
        elif self.auto_discover and self.mixer_ip:
            # Have a preferred IP but still try discovery to confirm type
            self._set_state(EngineState.DISCOVERING, f"Probing {self.mixer_ip}...")
            discovered = discover_mixer_auto(
                preferred_type=self.mixer_type,
                preferred_ip=self.mixer_ip,
                scan_subnet=False,
                timeout=2.0,
            )
            if discovered:
                self.discovered_mixer = discovered
                self.mixer_type = discovered.mixer_type
                self.mixer_ip = discovered.ip
                self.mixer_port = discovered.port
                self.mixer_tls = discovered.tls
                logger.info(f"Confirmed mixer at {discovered.ip}: {discovered.mixer_type.upper()}")
            else:
                logger.info(
                    f"Preferred IP {self.mixer_ip} not responding to probes, "
                    f"trying direct connect..."
                )

        # Ensure we have a type and port
        if self.mixer_type is None:
            self.mixer_type = "dlive"
        if self.mixer_port is None:
            if self.mixer_type == "dlive":
                self.mixer_port = 51328
            else:
                self.mixer_port = 2223

        self._set_state(
            EngineState.CONNECTING,
            f"Connecting to {self.mixer_type.upper()} @ {self.mixer_ip}:{self.mixer_port}...",
        )

        try:
            if self.mixer_type == "dlive":
                from dlive_client import DLiveClient
                self.mixer_client = DLiveClient(
                    ip=self.mixer_ip,
                    port=self.mixer_port,
                    tls=self.mixer_tls,
                    midi_base_channel=self.midi_base_channel,
                )
            elif self.mixer_type == "wing":
                from wing_client import WingClient
                self.mixer_client = WingClient(
                    ip=self.mixer_ip,
                    port=self.mixer_port,
                )
            else:
                logger.error(f"Unsupported mixer type: {self.mixer_type}")
                return False

            success = self.mixer_client.connect(timeout=10.0)
            if success:
                logger.info(f"Connected to {self.mixer_type.upper()} at {self.mixer_ip}:{self.mixer_port}")
                self._activate_observation_mode()
                return True
            else:
                logger.error(f"Failed to connect to {self.mixer_type.upper()}")
                return False

        except Exception as e:
            logger.error(f"Mixer connection error: {e}")
            return False

    # ── 2. Detect & start audio ──────────────────────────────────

    def _start_audio(self) -> bool:
        """Scan for audio devices, select the best one, and start capture."""
        self._set_state(EngineState.SCANNING_AUDIO, "Scanning audio devices...")

        # Step 1: Scan all available audio input devices
        self.audio_devices = scan_audio_devices()

        if not self.audio_devices:
            logger.warning("No audio input devices found via scan")

        # Step 2: Select the best device
        preferred_proto = None
        if self.audio_device_name:
            name_lower = self.audio_device_name.lower()
            if "soundgrid" in name_lower or "waves" in name_lower:
                preferred_proto = AudioProtocol.SOUNDGRID
            elif "dante" in name_lower:
                preferred_proto = AudioProtocol.DANTE

        best = select_best_device(
            self.audio_devices,
            preferred_protocol=preferred_proto,
            preferred_name=self.audio_device_name,
            min_channels=2,
        )

        device_index = None
        source_type = AudioSourceType.SOUNDDEVICE

        if best:
            self.selected_audio_device = best
            device_index = best.index
            num_hw_channels = best.max_input_channels

            # Map protocol to source type
            proto_source_map = {
                AudioProtocol.SOUNDGRID: AudioSourceType.SOUNDGRID,
                AudioProtocol.DANTE: AudioSourceType.DANTE,
            }
            source_type = proto_source_map.get(best.protocol, AudioSourceType.SOUNDDEVICE)

            # Adjust channel count to what the device actually supports
            if num_hw_channels < self.num_channels:
                logger.info(
                    f"Device '{best.name}' has {num_hw_channels}ch, "
                    f"reducing capture from {self.num_channels} to {num_hw_channels}"
                )
                self.num_channels = num_hw_channels
                if self.selected_channels:
                    before = list(self.selected_channels)
                    self.selected_channels = [ch for ch in self.selected_channels if ch <= self.num_channels]
                    removed = sorted(set(before) - set(self.selected_channels))
                    if removed:
                        logger.warning(
                            "Selected channels exceed device channel count and will be skipped: %s",
                            removed,
                        )

            logger.info(
                f"Audio device selected: [{best.index}] '{best.name}' "
                f"({best.max_input_channels}ch, {best.protocol.value}, "
                f"score={best.score})"
            )

            device_sample_rate = getattr(best, "default_samplerate", None)
            if device_sample_rate:
                selected_sample_rate = int(device_sample_rate)
                if selected_sample_rate != self.sample_rate:
                    logger.info(
                        "Using selected audio device sample rate: %sHz "
                        "(configured %sHz)",
                        selected_sample_rate,
                        self.sample_rate,
                    )
                    self.sample_rate = selected_sample_rate
        else:
            logger.warning("No suitable audio device found, using system default")

        # Log all discovered devices for diagnostics
        if self.audio_devices:
            logger.info(f"All audio devices ({len(self.audio_devices)}):")
            for dev in self.audio_devices:
                marker = " <<<" if dev == best else ""
                logger.info(
                    f"  [{dev.index}] {dev.name} | {dev.max_input_channels}ch | "
                    f"{dev.protocol.value} | score={dev.score}{marker}"
                )

        # Step 3: Create AudioCapture and start
        self.audio_capture = AudioCapture(
            num_channels=self.num_channels,
            sample_rate=self.sample_rate,
            buffer_seconds=self.buffer_seconds,
            block_size=self.block_size,
            source_type=source_type,
            device_name=device_index,
        )

        try:
            self.audio_capture.start()
            dev_name = best.name if best else "system default"
            logger.info(
                f"Audio capture started: '{dev_name}', "
                f"{source_type.value}, {self.num_channels}ch"
            )
            return True
        except Exception as e:
            logger.error(f"Failed to start audio capture: {e}")
            return False

    # ── 3. Scan channel names ────────────────────────────────────

    def _scan_channels(self):
        """Read channel names from mixer and recognize instruments."""
        self._set_state(EngineState.SCANNING_CHANNELS, "Reading channel names...")

        channel_names: Dict[int, str] = {}

        for ch in self._iter_channels():
            try:
                name = self.mixer_client.get_channel_name(ch)
                channel_names[ch] = name
            except Exception:
                channel_names[ch] = f"Ch {ch}"

        recognition = scan_and_recognize(channel_names)

        for ch in self._iter_channels():
            info = ChannelInfo(channel=ch)
            info.name = channel_names.get(ch, f"Ch {ch}")

            rec = recognition.get(ch, {})
            info.preset = rec.get("preset")
            info.recognized = rec.get("recognized", False)
            self.channels[ch] = info

        recognized = sum(1 for c in self.channels.values() if c.recognized)
        logger.info(f"Channel scan complete: {recognized}/{len(self.channels)} recognized")

    # ── 3b. Read existing channel settings from mixer ────────────

    def _read_channel_state(self):
        """Read current processing settings for all channels.

        Determines which channels already have non-default settings
        (EQ, HPF, fader, gain) so we can back them up before resetting.
        """
        self._set_state(
            EngineState.READING_STATE,
            "Reading existing channel settings from mixer..."
        )

        channels_with_processing = 0

        for ch in self._iter_channels():
            info = self.channels.get(ch)
            if info is None:
                continue

            try:
                raw_settings = self.mixer_client.get_channel_settings(ch)
            except Exception as e:
                logger.debug(f"Ch {ch}: could not read settings: {e}")
                raw_settings = {}

            fader_db = raw_settings.get("fader_db", -100.0)
            muted = raw_settings.get("muted", False)

            # Detect if channel has non-default processing
            has_processing = False
            if fader_db > -90.0:
                has_processing = True
            if not muted and fader_db > -50.0:
                has_processing = True

            snapshot = ChannelSnapshot(
                channel=ch,
                fader_db=fader_db,
                muted=muted,
                had_processing=has_processing,
                raw_settings=raw_settings,
            )
            info.original_snapshot = snapshot

            if has_processing:
                channels_with_processing += 1
                logger.debug(
                    f"Ch {ch} '{info.name}': existing settings detected "
                    f"(fader={fader_db:.1f}dB, muted={muted})"
                )

        logger.info(
            f"Channel state read: {channels_with_processing}/{len(self.channels)} "
            f"channels have existing settings"
        )

    # ── 3c. Reset channels to neutral before analysis ────────────

    def _reset_channels(self):
        """Reset all channel processing to flat/neutral.

        This ensures that the audio we analyze is the raw unprocessed
        signal from the stage. After analysis, new settings will be
        applied from scratch.
        """
        self._set_state(
            EngineState.RESETTING,
            "Resetting channel processing to neutral..."
        )

        reset_count = 0

        for ch in self._iter_channels():
            info = self.channels.get(ch)
            if info is None:
                continue

            try:
                self.mixer_client.reset_channel_processing(ch)
                info.was_reset = True
                reset_count += 1
            except Exception as e:
                logger.warning(f"Ch {ch}: reset failed: {e}")

            # Small delay between channels to avoid overwhelming the mixer
            if reset_count % 8 == 0:
                time.sleep(0.05)

        # Wait for resets to take effect on mixer DSP
        time.sleep(0.5)

        logger.info(f"Reset complete: {reset_count} channels set to neutral")

    # ── 4. Wait for signal and analyze ───────────────────────────

    def _wait_and_analyze(self):
        """Wait for audio signals, then run full multi-metric analysis.

        Phase 1 (WAITING): detect signal presence on each channel.
        Phase 2 (ANALYZING): feed 3+ seconds of audio through SignalAnalyzer
        to compute momentary/short-term/integrated LUFS, true peak, RMS,
        crest factor, ADSR envelope, transient metrics, and full spectral
        analysis per channel.
        """
        self._set_state(EngineState.WAITING_FOR_SIGNAL, "Waiting for audio signals...")

        warmup = 2.0
        time.sleep(warmup)

        analysis_start = time.time()
        analysis_timeout = 30.0

        while not self._stop_event.is_set():
            elapsed = time.time() - analysis_start
            if elapsed > analysis_timeout:
                logger.info("Analysis timeout reached, proceeding with available data")
                break

            any_new = False
            for ch, info in self.channels.items():
                if info.has_signal:
                    continue

                peak = self.audio_capture.get_peak(ch, self.block_size * 4)

                if peak > SIGNAL_THRESHOLD_DB:
                    info.has_signal = True
                    info.peak_db = peak
                    any_new = True

                    # Create analyzer for this channel
                    analyzer = SignalAnalyzer(ch, self.sample_rate, self.block_size)
                    self._analyzers[ch] = analyzer
                    info.analyzer = analyzer

                    logger.info(f"Ch {ch} '{info.name}': signal detected (peak={peak:.1f}dB)")

                    if not info.recognized:
                        self._classify_by_spectrum(ch, info)

            channels_with_signal = sum(1 for c in self.channels.values() if c.has_signal)
            if channels_with_signal > 0 and not any_new and elapsed > 10.0:
                logger.info(f"No new signals for a while, proceeding with {channels_with_signal} channels")
                break

            time.sleep(0.5)

        # ── Phase 2: Full multi-metric analysis ──────────────────
        self._set_state(EngineState.ANALYZING, "Running full signal analysis...")

        analysis_duration = max(SIGNAL_ANALYSIS_SECONDS, 3.0)
        analysis_blocks = int(analysis_duration * self.sample_rate / self.block_size)
        block_interval = self.block_size / self.sample_rate

        for block_i in range(analysis_blocks):
            if self._stop_event.is_set():
                break

            for ch, info in self.channels.items():
                if not info.has_signal:
                    continue
                analyzer = self._analyzers.get(ch)
                if analyzer is None:
                    continue

                samples = self.audio_capture.get_buffer(ch, self.block_size)
                if len(samples) >= self.block_size:
                    analyzer.process(samples)

            time.sleep(block_interval)

        # ── Collect metrics ──────────────────────────────────────
        for ch, info in self.channels.items():
            if not info.has_signal:
                continue

            analyzer = self._analyzers.get(ch)
            if analyzer:
                m = analyzer.get_metrics()
                info.metrics = m

                info.peak_db = m.level.peak_db
                info.rms_db = m.level.rms_db
                info.lufs = m.level.lufs_integrated if m.level.lufs_integrated > -90 else m.level.lufs_short_term
                info.spectral_centroid = m.spectral.centroid_hz

                logger.info(
                    f"Ch {ch} '{info.name}' metrics: "
                    f"LUFS M={m.level.lufs_momentary:.1f} S={m.level.lufs_short_term:.1f} "
                    f"I={m.level.lufs_integrated:.1f} | "
                    f"TP={m.level.true_peak_dbtp:.1f}dBTP | "
                    f"RMS={m.level.rms_db:.1f} | "
                    f"Crest={m.level.crest_factor_db:.1f} | "
                    f"DR={m.dynamics.dynamic_range_db:.1f}dB | "
                    f"Atk={m.dynamics.attack_time_ms:.0f}ms "
                    f"Dec={m.dynamics.decay_time_ms:.0f}ms "
                    f"Sus={m.dynamics.sustain_level_db:.1f}dB "
                    f"Rel={m.dynamics.release_time_ms:.0f}ms | "
                    f"Trans={m.dynamics.transient_density:.1f}/s "
                    f"Str={m.dynamics.transient_strength_db:.1f}dB | "
                    f"Centroid={m.spectral.centroid_hz:.0f}Hz "
                    f"Bright={m.spectral.brightness:.3f} "
                    f"Mud={m.spectral.mud_ratio:.3f} "
                    f"Pres={m.spectral.presence_ratio:.3f}"
                )

    def _classify_by_spectrum(self, ch: int, info: ChannelInfo):
        """Classify an unrecognized channel by spectral characteristics."""
        spec = self.audio_capture.get_spectrum(ch, 4096)
        freqs = spec["frequencies"]
        mags_db = spec["magnitude_db"]

        if len(freqs) == 0:
            return

        linear = 10 ** (mags_db / 20.0)
        total = np.sum(linear) + 1e-10
        centroid = float(np.sum(freqs * linear) / total)

        low_mask = (freqs >= 100) & (freqs < 300)
        mid_mask = (freqs >= 1000) & (freqs < 4000)
        high_mask = (freqs >= 4000) & (freqs < 10000)

        energy_bands = {
            "low_100_300": float(np.mean(linear[low_mask])) if np.any(low_mask) else 0.0,
            "mid_1k_4k": float(np.mean(linear[mid_mask])) if np.any(mid_mask) else 0.0,
            "high_4k_10k": float(np.mean(linear[high_mask])) if np.any(high_mask) else 0.0,
        }
        max_e = max(energy_bands.values()) + 1e-10
        energy_bands = {k: v / max_e for k, v in energy_bands.items()}

        preset = recognize_instrument_spectral_fallback(
            info.name, centroid, energy_bands
        )
        if preset:
            info.preset = preset
            info.recognized = True
            logger.info(f"Ch {ch} '{info.name}': spectral classification -> {preset} (centroid={centroid:.0f}Hz)")

    # ── 5. Apply corrections ─────────────────────────────────────

    def _apply_corrections(self):
        """Apply full channel strip processing to all active channels.

        Order: input gain → HPF → EQ → compressor → pan → fader → FX sends.
        Then: phase/polarity check across correlated pairs.

        Channels have been reset to neutral before analysis, so all
        settings are applied from scratch to clean channels.
        """
        self._set_state(EngineState.APPLYING, "Applying full processing chain...")

        for ch, info in self.channels.items():
            if not info.has_signal:
                continue
            if ch in self._applied_channels:
                continue

            preset = info.preset or "custom"
            had_proc = ""
            if info.original_snapshot and info.original_snapshot.had_processing:
                had_proc = " (previous settings cleared)"
            logger.info(
                f"Ch {ch} '{info.name}' (preset={preset}): "
                f"applying full processing chain{had_proc}..."
            )

            try:
                # 1. Input gain staging (preamp trim)
                self._apply_input_gain(ch, info, preset)
                # 2. HPF
                self._apply_hpf(ch, preset)
                # 3. Parametric EQ (4-band)
                self._apply_eq(ch, preset)
                # 4. Compressor
                self._apply_compressor(ch, info, preset)
                # 5. Pan
                self._apply_pan(ch, info, preset)
                # 6. Gain correction (output-side, LUFS-based)
                self._apply_gain_correction(ch, info, preset)
                # 7. Fader
                self._apply_fader(ch, info, preset)
                # 8. FX sends (reverb/delay)
                self._apply_fx_sends(ch, info, preset)

                self._applied_channels.add(ch)

                if self.on_channel_update:
                    self.on_channel_update(ch, {
                        "name": info.name,
                        "preset": preset,
                        "peak_db": info.peak_db,
                        "rms_db": info.rms_db,
                        "lufs": info.lufs,
                        "gain_correction_db": info.gain_correction_db,
                        "eq_applied": info.eq_applied,
                        "compressor_applied": info.compressor_applied,
                        "fader_db": info.fader_db,
                    })

            except Exception as e:
                logger.error(f"Ch {ch}: error applying corrections: {e}")

        # 9. Phase / polarity check across correlated channel pairs
        self._detect_and_fix_phase()

    def _apply_hpf(self, ch: int, preset: str):
        """Set HPF cutoff using spectral band energy metrics.

        Metrics used:
        - spectral.band_energy['sub'] (20-60 Hz)
        - spectral.band_energy['bass'] (60-250 Hz)
        - spectral.centroid_hz — low centroid = real low content
        """
        info = self.channels[ch]
        base_freq = INSTRUMENT_HPF.get(preset, 80.0)
        adaptive_freq = base_freq

        m = info.metrics
        if m:
            sub_e = m.spectral.band_energy.get('sub', -100.0)
            bass_e = m.spectral.band_energy.get('bass', -100.0)
            low_mid_e = m.spectral.band_energy.get('low_mid', -100.0)
            centroid = m.spectral.centroid_hz

            # Convert from dB to linear for ratio comparison
            sub_lin = 10 ** (sub_e / 20.0) if sub_e > -90 else 0
            bass_lin = 10 ** (bass_e / 20.0) if bass_e > -90 else 0
            lm_lin = 10 ** (low_mid_e / 20.0) if low_mid_e > -90 else 0
            useful_ref = max(bass_lin, lm_lin, 1e-10)

            sub_ratio = sub_lin / useful_ref

            if sub_ratio < 0.05 and centroid > 500:
                # No useful sub, high centroid → raise HPF
                adaptive_freq = min(base_freq * 2.0, 400.0)
            elif sub_ratio < 0.15 and centroid > 200:
                adaptive_freq = min(base_freq * 1.4, 350.0)
            elif sub_ratio > 0.7 and preset in ("kick", "bass") and centroid < 200:
                # Real sub content on bass instrument → lower HPF
                adaptive_freq = max(base_freq * 0.6, 20.0)
            elif sub_ratio > 0.5 and centroid < 150:
                adaptive_freq = max(base_freq * 0.8, 25.0)

        try:
            if hasattr(self.mixer_client, 'set_hpf'):
                self.mixer_client.set_hpf(ch, adaptive_freq, enabled=True)
                if abs(adaptive_freq - base_freq) > 5:
                    logger.info(f"Ch {ch}: HPF={adaptive_freq:.0f}Hz (preset {base_freq:.0f}Hz)")
                else:
                    logger.debug(f"Ch {ch}: HPF={adaptive_freq:.0f}Hz")
        except Exception as e:
            logger.warning(f"Ch {ch}: HPF failed: {e}")

    def _apply_eq(self, ch: int, preset: str):
        """Apply adaptive EQ using spectral metrics.

        Metrics used:
        - spectral.mud_ratio — excess 200-500Hz energy
        - spectral.presence_ratio — 2-5kHz clarity
        - spectral.brightness — high-frequency content
        - spectral.warmth — 200-800Hz warmth
        - spectral.spectral_tilt_db — overall spectrum slope
        - spectral.band_energy (7 bands) — per-band energy
        - spectral.flatness — tonal vs noise character
        """
        info = self.channels[ch]
        eq_bands = INSTRUMENT_EQ_PRESETS.get(preset)
        if not eq_bands:
            return

        adapted_bands = list(eq_bands)
        m = info.metrics

        if m and m.spectral.centroid_hz > 0:
            sm = m.spectral

            for i, (freq, gain, q) in enumerate(adapted_bands):
                adapted_gain = gain

                # Band 1: Low end (typically < 200 Hz)
                if freq <= 150:
                    if preset in ("kick", "bass"):
                        # Boost sub only if it's actually weak
                        sub_e = sm.band_energy.get('sub', -80)
                        bass_e = sm.band_energy.get('bass', -80)
                        if sub_e < bass_e - 10:
                            adapted_gain = max(gain, gain + 1.5)
                    else:
                        # For non-bass instruments, cut more if muddy
                        if sm.warmth > 0.4:
                            adapted_gain = min(gain, gain - 2.0)

                # Band 2: Low-mid / mud region (200-600 Hz)
                elif 150 < freq <= 600:
                    if sm.mud_ratio > 0.25:
                        mud_excess = (sm.mud_ratio - 0.15) * 20
                        adapted_gain = min(gain, gain - mud_excess * 0.5)
                    elif sm.mud_ratio < 0.05 and sm.warmth < 0.1:
                        adapted_gain = max(gain, gain + 1.0)

                # Band 3: Presence / clarity (1.5-5 kHz)
                elif 1500 <= freq <= 5000:
                    if sm.presence_ratio > 0.3:
                        excess = (sm.presence_ratio - 0.2) * 15
                        adapted_gain = min(gain, gain - excess * 0.4)
                    elif sm.presence_ratio < 0.08:
                        deficit = (0.15 - sm.presence_ratio) * 20
                        adapted_gain = max(gain, gain + deficit * 0.5)
                    if sm.flatness > 0.7 and gain > 0:
                        adapted_gain = gain * 0.5

                # Band 4: Air / brightness (6+ kHz)
                elif freq >= 6000:
                    if sm.brightness > 0.25:
                        bright_excess = (sm.brightness - 0.15) * 15
                        adapted_gain = min(gain, gain - bright_excess * 0.4)
                    elif sm.brightness < 0.03:
                        adapted_gain = max(gain, gain + 1.5)

                # Spectral tilt compensation
                if sm.spectral_tilt_db < -5.0 and freq >= 2000:
                    adapted_gain += min(2.0, abs(sm.spectral_tilt_db + 5) * 0.3)
                elif sm.spectral_tilt_db > 3.0 and freq <= 500:
                    adapted_gain += min(1.5, (sm.spectral_tilt_db - 3) * 0.2)

                adapted_gain = max(-8.0, min(8.0, round(adapted_gain, 1)))
                adapted_bands[i] = (freq, adapted_gain, q)

        try:
            for band_idx, (freq, gain, q) in enumerate(adapted_bands, start=1):
                if band_idx > 4:
                    break
                self.mixer_client.set_eq_band(ch, band_idx, freq, gain, q)

            info.eq_applied = True
            changes = []
            for i, ((_, og, _), (_, ag, _)) in enumerate(zip(eq_bands, adapted_bands)):
                if abs(og - ag) > 0.3:
                    changes.append(f"B{i+1}: {og:+.1f}→{ag:+.1f}dB")
            if changes:
                logger.info(f"Ch {ch}: EQ adapted: {', '.join(changes)}")
            else:
                logger.debug(f"Ch {ch}: EQ preset '{preset}' (no adaptation)")
        except Exception as e:
            logger.warning(f"Ch {ch}: EQ failed: {e}")

    def _apply_gain_correction(self, ch: int, info: ChannelInfo, preset: str):
        """Compute LUFS-based gain correction and fold it into fader offset.

        NOTE: This does NOT call set_gain() — that's used by input gain
        staging (preamp trim). Calling set_gain() twice would overwrite
        the preamp setting. Instead, the LUFS correction is stored in
        info.gain_correction_db and applied via the fader position.

        Metrics used:
        - level.lufs_integrated — gated loudness (most accurate)
        - level.lufs_short_term — fallback
        - level.true_peak_dbtp — headroom check
        - level.loudness_range_lu — wide LRA → gentler correction
        """
        target_lufs = INSTRUMENT_TARGET_LUFS.get(preset, -23.0)

        m = info.metrics
        if m and m.level.lufs_integrated > -90:
            current_lufs = m.level.lufs_integrated
            true_peak = m.level.true_peak_dbtp
            lra = m.level.loudness_range_lu
        elif m and m.level.lufs_short_term > -90:
            current_lufs = m.level.lufs_short_term
            true_peak = m.level.true_peak_dbtp
            lra = 0.0
        else:
            current_lufs = self.audio_capture.get_lufs(ch, 48000)
            true_peak = info.peak_db
            lra = 0.0

        info.lufs = current_lufs

        if current_lufs <= -90.0:
            info.gain_correction_db = 0.0
            return

        diff = target_lufs - current_lufs

        scale = 1.0
        if lra > 15.0:
            scale = 0.7
        elif lra > 10.0:
            scale = 0.85

        correction = diff * scale
        correction = max(-GAIN_CORRECTION_MAX_DB, min(GAIN_CORRECTION_MAX_DB, correction))

        if true_peak + correction > -1.0:
            correction = -1.0 - true_peak
            correction = max(-GAIN_CORRECTION_MAX_DB, correction)

        info.gain_correction_db = correction

        if abs(correction) > 0.5:
            logger.info(
                f"Ch {ch} '{info.name}': LUFS correction {correction:+.1f}dB "
                f"(LUFS_I={current_lufs:.1f}→{target_lufs:.0f}, "
                f"TP={true_peak:.1f}dBTP, LRA={lra:.1f}LU) "
                f"→ applied via fader"
            )

    def _apply_fader(self, ch: int, info: ChannelInfo, preset: str):
        """Compute fader from LUFS balance, gain correction, and crest factor.

        The fader incorporates two components:
        1. Base position from instrument preset
        2. LUFS gain correction (computed by _apply_gain_correction)
        3. Inter-channel balance adjustment based on measured LUFS
        4. Crest factor compensation (peaky signals sound quieter)

        Metrics used:
        - level.lufs_integrated of all channels — relative balance
        - level.crest_factor_db — perceptual loudness compensation
        - gain_correction_db — LUFS-based correction folded into fader
        """
        fader_levels = {
            "kick": -5.0, "snare": -5.0, "tom": -8.0, "hihat": -12.0,
            "ride": -12.0, "cymbals": -12.0, "overheads": -10.0,
            "room": -15.0, "bass": -5.0, "electricGuitar": -8.0,
            "acousticGuitar": -8.0, "accordion": -8.0, "synth": -8.0,
            "playback": -10.0, "leadVocal": -3.0, "backVocal": -8.0,
        }
        base_fader = fader_levels.get(preset, -10.0)

        # Start with base + LUFS gain correction (from _apply_gain_correction)
        fader_db = base_fader + info.gain_correction_db

        # Collect LUFS from all active channels for relative balance
        all_lufs: Dict[int, Tuple[float, str]] = {}
        for c, ci in self.channels.items():
            if ci.has_signal:
                ci_lufs = -100.0
                if ci.metrics and ci.metrics.level.lufs_integrated > -90:
                    ci_lufs = ci.metrics.level.lufs_integrated
                elif ci.lufs > -90:
                    ci_lufs = ci.lufs
                if ci_lufs > -90:
                    all_lufs[c] = (ci_lufs, ci.preset or "custom")

        my_entry = all_lufs.get(ch)
        if my_entry is not None and len(all_lufs) > 1:
            my_lufs = my_entry[0]

            # Find reference: the loudest channel's LUFS and its target
            ref_ch = max(all_lufs, key=lambda c: all_lufs[c][0])
            ref_lufs, ref_preset = all_lufs[ref_ch]
            ref_target = INSTRUMENT_TARGET_LUFS.get(ref_preset, -23.0)

            # How far is this channel from where it should be relative to ref?
            my_target = INSTRUMENT_TARGET_LUFS.get(preset, -23.0)
            expected_diff = my_target - ref_target
            actual_diff = my_lufs - ref_lufs
            deviation = actual_diff - expected_diff

            # Crest factor compensation
            m = info.metrics
            crest_adj = 0.0
            if m and m.level.crest_factor_db > 18.0:
                crest_adj = 1.0
            elif m and m.level.crest_factor_db < 6.0:
                crest_adj = -1.0

            balance_adj = max(-4.0, min(4.0, -deviation * 0.5 + crest_adj))
            fader_db += balance_adj

        fader_db = max(-30.0, min(0.0, fader_db))
        info.fader_db = fader_db

        try:
            self.mixer_client.set_fader(ch, fader_db)
            total_adj = fader_db - base_fader
            if abs(total_adj) > 0.5:
                logger.info(
                    f"Ch {ch} '{info.name}': fader={fader_db:.1f}dB "
                    f"(base={base_fader:.1f}, lufs_corr={info.gain_correction_db:+.1f}, "
                    f"lufs={info.lufs:.1f})"
                )
            else:
                logger.info(f"Ch {ch} '{info.name}': fader={fader_db:.1f}dB")
        except Exception as e:
            logger.warning(f"Ch {ch}: fader set failed: {e}")

    # ── 5d. Input gain staging ───────────────────────────────────

    def _apply_input_gain(self, ch: int, info: ChannelInfo, preset: str):
        """Set preamp gain using true peak and crest factor.

        Metrics used:
        - level.true_peak_dbtp — inter-sample accurate peak level
        - level.rms_db — average signal level
        - level.crest_factor_db — peak/RMS ratio (high = transient, low = sustained)
        - dynamics.dynamic_range_db — signal variability
        """
        m = info.metrics
        if m and m.level.true_peak_dbtp > -90:
            true_peak = m.level.true_peak_dbtp
            rms = m.level.rms_db
            crest = m.level.crest_factor_db
        else:
            true_peak = info.peak_db
            rms = info.rms_db
            crest = true_peak - rms if rms > -90 else 0

        if true_peak <= -90.0:
            return

        target_peak = INSTRUMENT_TARGET_INPUT_DB.get(preset, -12.0)

        # For high-crest signals (drums), target peak can be higher
        # because perceived loudness is lower
        if crest > 18.0:
            target_peak += 3.0
        elif crest < 6.0:
            target_peak -= 2.0

        diff = target_peak - true_peak
        correction = max(-20.0, min(20.0, diff))

        if abs(correction) < 1.0:
            return

        # Safety: true peak after correction must not exceed -1 dBTP
        if true_peak + correction > -1.0:
            correction = -1.0 - true_peak

        # Absolute gain safety clamp (per .cursorrules: gain ≤ +60 dB)
        SAFE_MAX_GAIN_DB = 12.0
        correction = max(-SAFE_MAX_GAIN_DB, min(SAFE_MAX_GAIN_DB, correction))

        try:
            self.mixer_client.set_gain(ch, correction)
            logger.info(
                f"Ch {ch} '{info.name}': input gain {correction:+.1f}dB "
                f"(TP={true_peak:.1f}dBTP, RMS={rms:.1f}, "
                f"crest={crest:.1f}, target={target_peak:.0f}dB)"
            )
        except Exception as e:
            logger.warning(f"Ch {ch}: input gain failed: {e}")

    # ── 5e. Phase / polarity detection ────────────────────────

    def _detect_and_fix_phase(self):
        """Detect phase issues using inter-channel metrics.

        Metrics used (from compare_channels):
        - cross_correlation — GCC-PHAT peak value and sign
        - delay_ms — time offset between channels
        - coherence — frequency-domain coherence (high = correlated)
        - spectral_similarity — cosine similarity of spectra
        - phase_inverted — detected polarity inversion
        - level_difference_db — for weighting decisions
        """
        pairs = self._find_correlated_pairs()
        if not pairs:
            logger.info("Phase check: no correlated pairs found")
            return

        for ch_a, ch_b, pair_type in pairs:
            try:
                buf_a = self.audio_capture.get_buffer(ch_a, self.sample_rate * 2)
                buf_b = self.audio_capture.get_buffer(ch_b, self.sample_rate * 2)

                if len(buf_a) < 4096 or len(buf_b) < 4096:
                    continue

                # Use the full inter-channel comparison
                icm = compare_channels(
                    buf_a, buf_b, self.sample_rate, ch_a, ch_b
                )

                info_a = self.channels.get(ch_a)
                info_b = self.channels.get(ch_b)
                name_a = info_a.name if info_a else f"Ch {ch_a}"
                name_b = info_b.name if info_b else f"Ch {ch_b}"

                logger.info(
                    f"Phase: {name_a} ↔ {name_b}: "
                    f"corr={icm.cross_correlation:.3f} "
                    f"delay={icm.delay_ms:.2f}ms "
                    f"coherence={icm.coherence:.3f} "
                    f"spec_sim={icm.spectral_similarity:.3f} "
                    f"level_diff={icm.level_difference_db:.1f}dB"
                )

                # Only act if signals are actually related (high coherence)
                if icm.coherence < 0.2 and icm.spectral_similarity < 0.5:
                    logger.debug(f"Phase: skipping — low coherence/similarity")
                    continue

                # Phase inversion detected
                if icm.phase_inverted:
                    logger.warning(
                        f"Phase: {name_b} is inverted (corr={icm.cross_correlation:.3f}), "
                        f"flipping polarity"
                    )
                    if hasattr(self.mixer_client, 'set_polarity'):
                        self.mixer_client.set_polarity(ch_b, True)

                # Time alignment
                elif icm.delay_ms > 0.1 and icm.coherence > 0.3:
                    align_ch = ch_b if icm.delay_samples > 0 else ch_a
                    logger.info(
                        f"Phase: applying {icm.delay_ms:.2f}ms delay "
                        f"to Ch {align_ch} (coherence={icm.coherence:.3f})"
                    )
                    if hasattr(self.mixer_client, 'set_delay'):
                        self.mixer_client.set_delay(align_ch, icm.delay_ms, enabled=True)

            except Exception as e:
                logger.warning(f"Phase check error for pair ({ch_a}, {ch_b}): {e}")

    def _find_correlated_pairs(self) -> List[Tuple[int, int, str]]:
        """Find channel pairs that should be phase-correlated.

        Looks for adjacent channels with the same preset type
        (e.g. two 'overheads', two 'synth', etc.).
        """
        pairs = []
        preset_channels: Dict[str, List[int]] = {}

        for ch, info in self.channels.items():
            if info.has_signal and info.preset:
                preset_channels.setdefault(info.preset, []).append(ch)

        for preset, channels in preset_channels.items():
            if len(channels) < 2:
                continue
            channels.sort()
            # Pair adjacent channels
            for i in range(0, len(channels) - 1, 2):
                pairs.append((channels[i], channels[i + 1], preset))

        return pairs

    # ── 5f. Pan positioning ──────────────────────────────────────

    def _apply_pan(self, ch: int, info: ChannelInfo, preset: str):
        """Set pan position based on instrument type."""
        if not hasattr(self.mixer_client, 'set_pan'):
            return

        pan = INSTRUMENT_PAN.get(preset, 0.0)

        # Smart panning for stereo pairs and multiple instruments
        preset_channels = [
            c for c, i in self.channels.items()
            if i.preset == preset and i.has_signal
        ]
        preset_channels.sort()

        if len(preset_channels) >= 2:
            idx = preset_channels.index(ch)
            if preset in STEREO_PAIR_PRESETS:
                # Hard stereo pair: L/R
                pan = -60.0 if idx % 2 == 0 else 60.0
            elif preset == "tom":
                # Spread toms L to R
                spread = 50.0
                n = len(preset_channels)
                if n > 1:
                    pan = -spread + (2.0 * spread * idx / (n - 1))
                else:
                    pan = 0.0
            elif preset == "backVocal":
                spread = 35.0
                n = len(preset_channels)
                if n == 2:
                    pan = -spread if idx == 0 else spread
                else:
                    pan = -spread + (2.0 * spread * idx / max(1, n - 1))

        try:
            self.mixer_client.set_pan(ch, pan)
            logger.info(f"Ch {ch} '{info.name}': pan = {pan:+.0f}")
        except Exception as e:
            logger.warning(f"Ch {ch}: pan set failed: {e}")

    # ── 5g. Compressor ───────────────────────────────────────────

    def _apply_compressor(self, ch: int, info: ChannelInfo, preset: str):
        """Apply compressor adapted to measured signal dynamics.

        Uses full metrics from SignalAnalyzer: crest factor, dynamic range,
        ADSR envelope, transient density/strength, and spectral flux to
        precisely adapt threshold, ratio, attack, and release.
        """
        if not hasattr(self.mixer_client, 'set_compressor'):
            return

        comp_params = INSTRUMENT_COMPRESSOR.get(preset)
        if not comp_params:
            return

        base_threshold, base_ratio, base_attack, base_release = comp_params
        threshold = base_threshold
        ratio = base_ratio
        attack = base_attack
        release = base_release

        m = info.metrics
        if m and m.level.rms_db > -80:
            rms_db = m.level.rms_db
            crest_factor = m.level.crest_factor_db
            dynamic_range = m.dynamics.dynamic_range_db
            measured_attack = m.dynamics.attack_time_ms
            measured_release = m.dynamics.release_time_ms
            transient_density = m.dynamics.transient_density
            spectral_flux = m.spectral.flux

            # Threshold: ~6dB above measured RMS, bounded by preset ±6
            threshold = rms_db + 6.0
            threshold = max(base_threshold - 6.0, min(base_threshold + 6.0, threshold))
            threshold = max(-50.0, min(-5.0, threshold))

            # Ratio: scale by crest factor
            if crest_factor > 20.0:
                ratio = min(base_ratio * 1.3, 10.0)
            elif crest_factor < 8.0:
                ratio = max(base_ratio * 0.7, 1.5)

            # Attack: use measured attack time as guide
            if measured_attack > 0 and measured_attack < 100:
                attack = max(1.0, measured_attack * 0.5)
            elif crest_factor > 15.0:
                attack = max(base_attack, 5.0)
            elif crest_factor < 6.0:
                attack = max(1.0, base_attack * 0.7)

            # If high transient density (drums), preserve transients
            if transient_density > 4.0:
                attack = max(attack, 3.0)

            # Release: use measured release or adapt by dynamic range
            if measured_release > 10 and measured_release < 2000:
                release = max(30.0, measured_release * 0.7)
            elif dynamic_range > 15.0:
                release = min(base_release * 1.3, 500.0)
            elif dynamic_range < 6.0:
                release = max(base_release * 0.7, 30.0)

            # High spectral flux → more dynamic signal → longer release
            if spectral_flux > 0.5:
                release = min(release * 1.2, 600.0)

            logger.debug(
                f"Ch {ch}: comp adaptation: crest={crest_factor:.1f}dB "
                f"DR={dynamic_range:.1f}dB atk_measured={measured_attack:.0f}ms "
                f"trans_density={transient_density:.1f}/s flux={spectral_flux:.3f}"
            )

        try:
            self.mixer_client.set_compressor(
                ch,
                threshold_db=round(threshold, 1),
                ratio=round(ratio, 1),
                attack_ms=round(attack, 1),
                release_ms=round(release, 1),
                makeup_db=0.0,
                enabled=True,
            )
            info.compressor_applied = True

            changes = []
            if abs(threshold - base_threshold) > 1.0:
                changes.append(f"thr: {base_threshold:.0f}→{threshold:.0f}")
            if abs(ratio - base_ratio) > 0.3:
                changes.append(f"ratio: {base_ratio:.1f}→{ratio:.1f}")
            if abs(attack - base_attack) > 2.0:
                changes.append(f"atk: {base_attack:.0f}→{attack:.0f}")
            if abs(release - base_release) > 10.0:
                changes.append(f"rel: {base_release:.0f}→{release:.0f}")

            adapted = f" (adapted: {', '.join(changes)})" if changes else ""
            logger.info(
                f"Ch {ch} '{info.name}': compressor thr={threshold:.0f}dB "
                f"ratio={ratio:.1f}:1 atk={attack:.0f}ms rel={release:.0f}ms{adapted}"
            )
        except Exception as e:
            logger.warning(f"Ch {ch}: compressor failed: {e}")

    # ── 5h. FX Sends (reverb / delay) ────────────────────────────

    def _apply_fx_sends(self, ch: int, info: ChannelInfo, preset: str):
        """Set FX send levels using signal metrics.

        Metrics used:
        - level.lufs_integrated — loudness-based scaling
        - dynamics.dynamic_range_db — dynamic signals need less reverb
        - dynamics.sustain_level_db — sustained signals blend with reverb
        - spectral.brightness — bright signals → less reverb to avoid wash
        - spectral.warmth — warm signals → more reverb (natural fit)
        - dynamics.transient_density — percussive → shorter/less reverb
        """
        if not hasattr(self.mixer_client, 'set_send_level'):
            return

        sends = INSTRUMENT_FX_SENDS.get(preset, [])
        if not sends:
            return

        m = info.metrics
        adapted_sends = []

        for send_bus, base_level in sends:
            level = base_level

            if m:
                # Loudness scaling
                lufs_i = m.level.lufs_integrated
                if lufs_i > -16.0:
                    level -= 4.0
                elif lufs_i > -20.0:
                    level -= 2.0
                elif lufs_i < -35.0:
                    level += 3.0

                # Dynamic range: wide DR → less send (reverb masks dynamics)
                if m.dynamics.dynamic_range_db > 18.0:
                    level -= 2.0

                # Transient density: percussive → less reverb
                if m.dynamics.transient_density > 5.0:
                    level -= 2.0
                elif m.dynamics.transient_density < 1.0 and m.dynamics.sustain_level_db > -30:
                    level += 1.5

                # Brightness: bright → less reverb to keep clarity
                if m.spectral.brightness > 0.25:
                    level -= 2.0
                elif m.spectral.brightness < 0.05:
                    level += 1.0

                # Warmth: warm signals blend well with reverb
                if m.spectral.warmth > 0.3:
                    level += 1.0

            level = max(-40.0, min(-5.0, round(level, 1)))
            adapted_sends.append((send_bus, level))

        for send_bus, level_db in adapted_sends:
            try:
                self.mixer_client.set_send_level(ch, send_bus, level_db)
            except Exception as e:
                logger.warning(f"Ch {ch}: send bus {send_bus} failed: {e}")

        bus_desc = ", ".join(f"bus{b}={l:.0f}dB" for b, l in adapted_sends)
        logger.info(f"Ch {ch} '{info.name}': FX sends: {bus_desc}")

    # ── 6. Continuous monitoring ─────────────────────────────────

    def _monitor_loop(self):
        """Continuous monitoring: feedback detection + new channel detection."""
        self.feedback_detector = FeedbackDetector(
            sample_rate=self.sample_rate,
            fft_size=2048,
        )

        while not self._stop_event.is_set():
            try:
                for ch, info in self.channels.items():
                    if not info.has_signal:
                        peak = self.audio_capture.get_peak(ch, self.block_size * 4)
                        if peak > SIGNAL_THRESHOLD_DB:
                            info.has_signal = True
                            info.peak_db = peak
                            info.rms_db = self.audio_capture.get_rms(ch, self.block_size * 4)
                            info.lufs = self.audio_capture.get_lufs(ch, 0.4)
                            if not info.recognized:
                                self._classify_by_spectrum(ch, info)
                            logger.info(f"Monitor: new signal on Ch {ch} '{info.name}'")
                            self._apply_corrections_single(ch)
                        continue

                    samples = self.audio_capture.get_buffer(ch, 2048)
                    if len(samples) >= 2048:
                        events = self.feedback_detector.process(ch, samples)
                        for evt in events:
                            self._handle_feedback_event(ch, evt)

            except Exception as e:
                logger.debug(f"Monitor loop error: {e}")

            time.sleep(0.1)

    def _apply_corrections_single(self, ch: int):
        """Apply full processing chain for a single newly detected channel.

        Resets the channel, runs a short analysis pass for fresh metrics,
        then applies the full processing chain.
        """
        info = self.channels.get(ch)
        if info is None or ch in self._applied_channels:
            return
        preset = info.preset or "custom"
        try:
            self.mixer_client.reset_channel_processing(ch)
            info.was_reset = True
            time.sleep(0.3)

            # Quick analysis pass (1.5s) for fresh metrics
            analyzer = SignalAnalyzer(ch, self.sample_rate, self.block_size)
            analysis_blocks = int(1.5 * self.sample_rate / self.block_size)
            interval = self.block_size / self.sample_rate
            for _ in range(analysis_blocks):
                samples = self.audio_capture.get_buffer(ch, self.block_size)
                if len(samples) >= self.block_size:
                    analyzer.process(samples)
                time.sleep(interval)
            info.metrics = analyzer.get_metrics()

            self._apply_input_gain(ch, info, preset)
            self._apply_hpf(ch, preset)
            self._apply_eq(ch, preset)
            self._apply_compressor(ch, info, preset)
            self._apply_pan(ch, info, preset)
            self._apply_gain_correction(ch, info, preset)
            self._apply_fader(ch, info, preset)
            self._apply_fx_sends(ch, info, preset)
            self._applied_channels.add(ch)
        except Exception as e:
            logger.error(f"Ch {ch}: single-channel correction error: {e}")

    def _handle_feedback_event(self, ch: int, event: FeedbackEvent):
        """React to a feedback event by applying notch or reducing fader."""
        logger.warning(
            f"FEEDBACK Ch {ch}: {event.action} at {event.frequency_hz:.0f}Hz "
            f"({event.magnitude_db:.1f}dB)"
        )
        if event.action == "notch" and hasattr(self.mixer_client, "set_eq_band"):
            band = 4
            try:
                self.mixer_client.set_eq_band(ch, band, event.frequency_hz, -6.0, 10.0)
            except Exception:
                pass
        elif event.action == "fader_reduce":
            try:
                info = self.channels.get(ch)
                if info is None:
                    return
                current = info.fader_db
                new_fader = max(-30.0, current - 3.0)
                self.mixer_client.set_fader(ch, new_fader)
                info.fader_db = new_fader
                logger.warning(f"FEEDBACK: Ch {ch} fader reduced to {new_fader:.1f}dB")
            except Exception:
                pass

    # ── Main run ─────────────────────────────────────────────────

    def run(self):
        """Run the full auto-soundcheck pipeline (blocking).

        Pipeline:
        1. Discover mixer on network (OSC/MIDI scan)
        2. Connect to mixer
        3. Scan audio devices, select best multichannel input
        4. Start audio capture
        5. Read channel names → recognize instruments
        6. Read existing channel settings (backup)
        7. Reset all channels to neutral (flat EQ, HPF off)
        8. Wait for audio signals (now receiving clean/raw audio)
        9. Analyze signals (LUFS, peak, spectrum)
        10. Apply new corrections (HPF, EQ, gain, fader)
        11. Enter monitoring mode (feedback detection, new channels)
        """
        self._stop_event.clear()

        # Steps 1-2: Discover and connect mixer
        if not self._connect_mixer():
            self._set_state(EngineState.ERROR, "Mixer connection failed")
            return False

        # Steps 3-4: Scan and start audio
        if not self._start_audio():
            self._set_state(EngineState.ERROR, "Audio capture failed")
            return False

        # Step 5: Scan channel names and recognize instruments
        self._scan_channels()

        # Step 6: Read existing settings from mixer
        self._read_channel_state()

        # Step 7: Reset all channels to neutral before analysis
        self._reset_channels()

        # Steps 8-9: Wait for clean audio signals and analyze
        self._wait_and_analyze()

        # Step 10: Apply new corrections from scratch
        if self.auto_apply:
            self._apply_corrections()

        self._set_state(EngineState.RUNNING, "Auto-soundcheck complete, monitoring...")

        self._monitor_thread = threading.Thread(
            target=self._monitor_loop, daemon=True
        )
        self._monitor_thread.start()

        try:
            while not self._stop_event.is_set():
                time.sleep(1.0)
        except KeyboardInterrupt:
            logger.info("Keyboard interrupt, stopping...")

        self.stop()
        return True

    def start_async(self):
        """Start engine in background thread."""
        self._engine_thread = threading.Thread(target=self.run, daemon=True)
        self._engine_thread.start()

    def restore_original_settings(self):
        """Restore channels to their original fader positions (before reset).

        This is a safety fallback: if the auto-soundcheck results are
        unacceptable, calling this method restores the fader levels
        that were captured before the reset step.
        """
        if not self.mixer_client:
            logger.warning("No mixer client — cannot restore settings")
            return

        restored = 0
        for ch, info in self.channels.items():
            snap = info.original_snapshot
            if snap is None or not snap.had_processing:
                continue
            try:
                if snap.fader_db > -90.0:
                    self.mixer_client.set_fader(ch, snap.fader_db)
                if snap.muted:
                    self.mixer_client.set_mute(ch, True)
                restored += 1
                logger.info(f"Ch {ch} '{info.name}': restored fader={snap.fader_db:.1f}dB")
            except Exception as e:
                logger.warning(f"Ch {ch}: restore failed: {e}")

        logger.info(f"Restored original settings for {restored} channel(s)")

    def stop(self):
        """Stop the engine, joining threads before releasing resources."""
        self._stop_event.set()
        self._set_state(EngineState.STOPPED, "Engine stopped")

        if self._observation_mixer_client is not None:
            summary = self._observation_mixer_client.get_summary()
            self._emit_observation(
                message=(
                    f"[OBSERVE] Engine complete. Intercepted "
                    f"{summary['total_operations']} mixer write operation(s)."
                ),
                summary=summary,
            )

        # Join monitor thread first so it stops accessing audio/mixer
        if self._monitor_thread and self._monitor_thread.is_alive():
            self._monitor_thread.join(timeout=3.0)

        if self.audio_capture:
            try:
                self.audio_capture.stop()
            except Exception:
                pass

        if self.mixer_client:
            try:
                self.mixer_client.disconnect()
            except Exception:
                pass
        if self._real_mixer_client is not None:
            self.mixer_client = self._real_mixer_client
        self._observation_mixer_client = None
        self._real_mixer_client = None

    def get_status(self) -> Dict:
        """Get engine status."""
        channels_info = {}
        for ch, info in self.channels.items():
            channels_info[ch] = {
                "name": info.name,
                "preset": info.preset,
                "recognized": info.recognized,
                "has_signal": info.has_signal,
                "peak_db": round(info.peak_db, 1),
                "rms_db": round(info.rms_db, 1),
                "lufs": round(info.lufs, 1),
                "gain_correction_db": round(info.gain_correction_db, 1),
                "eq_applied": info.eq_applied,
                "fader_db": round(info.fader_db, 1),
            }

        discovery_info = None
        if self.discovered_mixer:
            discovery_info = {
                "mixer_type": self.discovered_mixer.mixer_type,
                "ip": self.discovered_mixer.ip,
                "port": self.discovered_mixer.port,
                "name": self.discovered_mixer.name,
                "method": self.discovered_mixer.discovery_method,
                "response_ms": round(self.discovered_mixer.response_time_ms, 0),
            }

        audio_device_info = None
        if self.selected_audio_device:
            audio_device_info = {
                "index": self.selected_audio_device.index,
                "name": self.selected_audio_device.name,
                "channels": self.selected_audio_device.max_input_channels,
                "protocol": self.selected_audio_device.protocol.value,
                "samplerate": self.selected_audio_device.default_samplerate,
                "score": self.selected_audio_device.score,
            }

        audio_devices_list = [
            {
                "index": d.index,
                "name": d.name,
                "channels": d.max_input_channels,
                "protocol": d.protocol.value,
                "is_multichannel": d.is_multichannel,
                "score": d.score,
            }
            for d in self.audio_devices
        ]

        return {
            "state": self.state.value,
            "mixer_type": self.mixer_type,
            "mixer_ip": self.mixer_ip,
            "mixer_connected": self.mixer_client.is_connected if self.mixer_client else False,
            "audio_running": self.audio_capture.running if self.audio_capture else False,
            "audio_device": audio_device_info,
            "audio_devices_available": audio_devices_list,
            "total_channels": len(self.channels) if self.channels else self._configured_channel_count(),
            "selected_channels": self._iter_channels(),
            "observe_only": self.observe_only,
            "applied_channel_ids": sorted(self._applied_channels),
            "channels_with_signal": sum(1 for c in self.channels.values() if c.has_signal),
            "channels_recognized": sum(1 for c in self.channels.values() if c.recognized),
            "discovered": discovery_info,
            "channels": channels_info,
        }


# ── CLI entry point ──────────────────────────────────────────────

def main():
    """CLI entry point for headless auto-soundcheck."""
    import argparse

    parser = argparse.ArgumentParser(
        description="AUTO-MIXER Tubeslave — Automatic Soundcheck Engine"
    )
    parser.add_argument("--mixer", default=None, choices=["dlive", "wing"],
                        help="Mixer type (default: auto-detect)")
    parser.add_argument("--ip", default=None,
                        help="Mixer IP address (default: auto-discover)")
    parser.add_argument("--port", type=int, default=None,
                        help="Mixer port (default: auto)")
    parser.add_argument("--tls", action="store_true",
                        help="Use TLS for dLive connection")
    parser.add_argument("--no-discover", action="store_true",
                        help="Skip auto-discovery, use --ip directly")
    parser.add_argument("--full-scan", action="store_true",
                        help="Full /24 subnet scan (slower but thorough)")
    parser.add_argument("--audio-device", default=None,
                        help="Audio device name pattern (e.g. 'soundgrid', 'dante')")
    parser.add_argument("--channels", type=int, default=48,
                        help="Number of input channels (default: 48)")
    parser.add_argument("--sample-rate", type=int, default=48000,
                        help="Sample rate (default: 48000)")
    parser.add_argument("--no-apply", action="store_true",
                        help="Analyze only, do not apply corrections")
    parser.add_argument("--log-level", default="INFO",
                        choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    parser.add_argument("--scan-only", action="store_true",
                        help="Only scan for mixers, do not connect")

    args = parser.parse_args()

    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    # Scan-only mode
    if args.scan_only:
        from mixer_discovery import main as discovery_main
        discovery_main()
        return

    auto_discover = not args.no_discover

    def on_state(state, msg):
        print(f"\n>>> [{state}] {msg}")

    def on_channel(ch, data):
        print(f"  Ch {ch}: {data['name']} -> {data.get('preset', '?')} | "
              f"peak={data['peak_db']:.1f}dB | lufs={data['lufs']:.1f} | "
              f"gain_corr={data['gain_correction_db']:+.1f}dB | "
              f"fader={data['fader_db']:.1f}dB | EQ={data['eq_applied']}")

    engine = AutoSoundcheckEngine(
        mixer_type=args.mixer,
        mixer_ip=args.ip,
        mixer_port=args.port,
        mixer_tls=args.tls,
        audio_device_name=args.audio_device,
        num_channels=args.channels,
        sample_rate=args.sample_rate,
        auto_apply=not args.no_apply,
        auto_discover=auto_discover,
        scan_subnet=args.full_scan,
        on_state_change=on_state,
        on_channel_update=on_channel,
    )

    print("=" * 60)
    print("  AUTO-MIXER Tubeslave — Automatic Soundcheck")
    if args.ip:
        mixer_desc = f"{(args.mixer or 'auto').upper()} @ {args.ip}:{args.port or 'auto'}"
    else:
        mixer_desc = "Auto-discover on network"
    print(f"  Mixer: {mixer_desc}")
    print(f"  Audio: {args.audio_device or 'auto-detect'}")
    print(f"  Channels: {args.channels}")
    print(f"  Auto-apply: {not args.no_apply}")
    print(f"  Auto-discover: {auto_discover}")
    print("=" * 60)
    print()

    engine.run()


if __name__ == "__main__":
    main()
