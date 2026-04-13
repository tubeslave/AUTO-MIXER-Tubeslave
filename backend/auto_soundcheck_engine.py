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
import os
import sys
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
}

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
        auto_discover: bool = True,
        scan_subnet: bool = True,
        on_state_change: Optional[Callable] = None,
        on_channel_update: Optional[Callable] = None,
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
        self.auto_discover = auto_discover
        self.scan_subnet = scan_subnet

        self.mixer_client = None
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

        self._applied_channels: set = set()

        mode = "auto-discover" if auto_discover and not mixer_ip else "manual"
        target = f"{mixer_type or '?'}@{mixer_ip or 'auto'}:{mixer_port or 'auto'}"
        logger.info(
            f"AutoSoundcheckEngine: {mode}, target={target}, "
            f"channels={num_channels}, sr={sample_rate}"
        )

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
            self.mixer_port = 51328 if self.mixer_type == "dlive" else 2223

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

            logger.info(
                f"Audio device selected: [{best.index}] '{best.name}' "
                f"({best.max_input_channels}ch, {best.protocol.value}, "
                f"score={best.score})"
            )
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

        for ch in range(1, self.num_channels + 1):
            try:
                name = self.mixer_client.get_channel_name(ch)
                channel_names[ch] = name
            except Exception:
                channel_names[ch] = f"Ch {ch}"

        recognition = scan_and_recognize(channel_names)

        for ch in range(1, self.num_channels + 1):
            info = ChannelInfo(channel=ch)
            info.name = channel_names.get(ch, f"Ch {ch}")

            rec = recognition.get(ch, {})
            info.preset = rec.get("preset")
            info.recognized = rec.get("recognized", False)
            self.channels[ch] = info

        recognized = sum(1 for c in self.channels.values() if c.recognized)
        logger.info(f"Channel scan complete: {recognized}/{self.num_channels} recognized")

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

        for ch in range(1, self.num_channels + 1):
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
            f"Channel state read: {channels_with_processing}/{self.num_channels} "
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

        for ch in range(1, self.num_channels + 1):
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
        """Wait for audio signals and classify unrecognized channels by spectrum."""
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

                peak = self.audio_capture.get_peak(ch, self.sample_rate)
                rms = self.audio_capture.get_rms(ch, self.sample_rate)
                lufs = self.audio_capture.get_lufs(ch, 0.4)

                info.peak_db = peak
                info.rms_db = rms
                info.lufs = lufs

                if peak > SIGNAL_THRESHOLD_DB:
                    info.has_signal = True
                    any_new = True
                    logger.info(
                        f"Ch {ch} '{info.name}': signal detected "
                        f"(peak={peak:.1f}dB, rms={rms:.1f}dB, lufs={lufs:.1f})"
                    )

                    if not info.recognized:
                        self._classify_by_spectrum(ch, info)

            channels_with_signal = sum(1 for c in self.channels.values() if c.has_signal)
            if channels_with_signal > 0 and not any_new and elapsed > 10.0:
                logger.info(f"No new signals for a while, proceeding with {channels_with_signal} channels")
                break

            time.sleep(0.5)

        self._set_state(EngineState.ANALYZING, "Analyzing signals...")
        time.sleep(SIGNAL_ANALYSIS_SECONDS)

        for ch, info in self.channels.items():
            if info.has_signal:
                info.peak_db = self.audio_capture.get_peak(ch, self.sample_rate)
                info.rms_db = self.audio_capture.get_rms(ch, self.sample_rate)
                info.lufs = self.audio_capture.get_lufs(ch, 3.0)

                spec = self.audio_capture.get_spectrum(ch, 4096)
                freqs = spec["frequencies"]
                mags = spec["magnitude_db"]
                if len(freqs) > 0 and len(mags) > 0:
                    linear = 10 ** (mags / 20.0)
                    total = np.sum(linear) + 1e-10
                    info.spectral_centroid = float(np.sum(freqs * linear) / total)

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
        """Apply gain, EQ, compressor, and fader to all active channels.

        Channels have been reset to neutral before analysis, so all
        settings are applied from scratch to clean channels.
        """
        self._set_state(EngineState.APPLYING, "Applying corrections to clean channels...")

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
                f"applying corrections{had_proc}..."
            )

            try:
                self._apply_hpf(ch, preset)
                self._apply_eq(ch, preset)
                self._apply_gain_correction(ch, info, preset)
                self._apply_fader(ch, info, preset)
                self._applied_channels.add(ch)

                if self.on_channel_update:
                    self.on_channel_update(ch, {
                        "name": info.name,
                        "preset": preset,
                        "peak_db": info.peak_db,
                        "lufs": info.lufs,
                        "gain_correction_db": info.gain_correction_db,
                        "eq_applied": info.eq_applied,
                        "fader_db": info.fader_db,
                    })

            except Exception as e:
                logger.error(f"Ch {ch}: error applying corrections: {e}")

    def _apply_hpf(self, ch: int, preset: str):
        """Apply HPF based on instrument type."""
        hpf_freq = INSTRUMENT_HPF.get(preset, 80.0)
        try:
            if hasattr(self.mixer_client, 'set_hpf'):
                self.mixer_client.set_hpf(ch, hpf_freq, enabled=True)
                logger.debug(f"Ch {ch}: HPF={hpf_freq:.0f}Hz")
        except Exception as e:
            logger.warning(f"Ch {ch}: HPF failed: {e}")

    def _apply_eq(self, ch: int, preset: str):
        """Apply 4-band parametric EQ based on instrument type."""
        eq_bands = INSTRUMENT_EQ_PRESETS.get(preset)
        if not eq_bands:
            return

        try:
            for band_idx, (freq, gain, q) in enumerate(eq_bands, start=1):
                if band_idx > 4:
                    break
                self.mixer_client.set_eq_band(ch, band_idx, freq, gain, q)

            info = self.channels[ch]
            info.eq_applied = True
            logger.debug(f"Ch {ch}: EQ preset '{preset}' applied (4 bands)")
        except Exception as e:
            logger.warning(f"Ch {ch}: EQ failed: {e}")

    def _apply_gain_correction(self, ch: int, info: ChannelInfo, preset: str):
        """Compute and apply gain correction to reach target LUFS."""
        target_lufs = INSTRUMENT_TARGET_LUFS.get(preset, -23.0)

        if info.lufs <= -90.0:
            info.gain_correction_db = 0.0
            return

        diff = target_lufs - info.lufs
        correction = max(-GAIN_CORRECTION_MAX_DB, min(GAIN_CORRECTION_MAX_DB, diff))

        if info.peak_db + correction > -1.0:
            correction = -1.0 - info.peak_db
            correction = max(-GAIN_CORRECTION_MAX_DB, correction)

        info.gain_correction_db = correction

        if abs(correction) > 0.5:
            try:
                current_gain_db = correction
                self.mixer_client.set_gain(ch, current_gain_db)
                logger.info(f"Ch {ch} '{info.name}': gain correction {correction:+.1f}dB (target={target_lufs:.0f} LUFS)")
            except Exception as e:
                logger.warning(f"Ch {ch}: gain correction failed: {e}")

    def _apply_fader(self, ch: int, info: ChannelInfo, preset: str):
        """Set initial fader position based on instrument type."""
        fader_levels = {
            "kick": -5.0,
            "snare": -5.0,
            "tom": -8.0,
            "hihat": -12.0,
            "ride": -12.0,
            "cymbals": -12.0,
            "overheads": -10.0,
            "room": -15.0,
            "bass": -5.0,
            "electricGuitar": -8.0,
            "acousticGuitar": -8.0,
            "accordion": -8.0,
            "synth": -8.0,
            "playback": -10.0,
            "leadVocal": -3.0,
            "backVocal": -8.0,
        }

        fader_db = fader_levels.get(preset, -10.0)
        info.fader_db = fader_db

        try:
            self.mixer_client.set_fader(ch, fader_db)
            logger.info(f"Ch {ch} '{info.name}': fader set to {fader_db:.1f}dB")
        except Exception as e:
            logger.warning(f"Ch {ch}: fader set failed: {e}")

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
                        peak = self.audio_capture.get_peak(ch, self.sample_rate)
                        if peak > SIGNAL_THRESHOLD_DB:
                            info.has_signal = True
                            info.peak_db = peak
                            info.rms_db = self.audio_capture.get_rms(ch, self.sample_rate)
                            info.lufs = self.audio_capture.get_lufs(ch, 3.0)
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
        """Apply corrections for a single newly detected channel.

        Resets the channel first to ensure we apply to a clean state.
        """
        info = self.channels[ch]
        if ch in self._applied_channels:
            return
        preset = info.preset or "custom"
        try:
            # Reset this channel before applying new settings
            self.mixer_client.reset_channel_processing(ch)
            info.was_reset = True
            time.sleep(0.1)

            self._apply_hpf(ch, preset)
            self._apply_eq(ch, preset)
            self._apply_gain_correction(ch, info, preset)
            self._apply_fader(ch, info, preset)
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
                current = self.channels[ch].fader_db
                new_fader = current - 3.0
                self.mixer_client.set_fader(ch, new_fader)
                self.channels[ch].fader_db = new_fader
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
        """Stop the engine."""
        self._stop_event.set()
        self._set_state(EngineState.STOPPED, "Engine stopped")

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
            "total_channels": self.num_channels,
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
