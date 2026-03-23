"""
Unified AudioCapture service with ring buffers per channel.

Provides subscribe/unsubscribe pattern, get_buffer/get_lufs/get_spectrum methods,
Dante input via sounddevice, and fallback test generators.
"""

import numpy as np
import logging
import threading
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional
from collections import deque
from enum import Enum

logger = logging.getLogger(__name__)

try:
    import sounddevice as sd
    HAS_SOUNDDEVICE = True
except (ImportError, OSError):
    HAS_SOUNDDEVICE = False
    logger.info("sounddevice not available; audio capture will use test generators")


class AudioSourceType(Enum):
    DANTE = "dante"
    SOUNDGRID = "soundgrid"
    SOUNDDEVICE = "sounddevice"
    TEST_SINE = "test_sine"
    TEST_PINK_NOISE = "test_pink_noise"
    TEST_FILE = "test_file"
    SILENCE = "silence"


class AudioDeviceType(Enum):
    DANTE = "dante"
    SOUNDGRID = "soundgrid"
    ASIO = "asio"
    DEFAULT = "default"


# ── Device discovery helpers ──────────────────────────────────

SOUNDGRID_PATTERNS = ("waves", "soundgrid", "sg ")

def list_audio_devices() -> list:
    """Return list of available audio input devices.

    Each entry is a dict with 'index', 'name', 'max_input_channels', 'default_samplerate'.
    """
    if not HAS_SOUNDDEVICE:
        return []
    devices = []
    try:
        for idx, dev in enumerate(sd.query_devices()):
            if dev.get("max_input_channels", 0) > 0:
                devices.append({
                    "index": idx,
                    "name": dev["name"],
                    "max_input_channels": dev["max_input_channels"],
                    "default_samplerate": dev.get("default_samplerate", 48000),
                })
    except Exception as e:
        logger.error(f"Error listing audio devices: {e}")
    return devices


def find_device_by_name(pattern: str) -> Optional[int]:
    """Find audio device index whose name contains *pattern* (case-insensitive)."""
    pattern_lower = pattern.lower()
    for dev in list_audio_devices():
        if pattern_lower in dev["name"].lower():
            logger.info(f"Audio device matched: '{dev['name']}' (index {dev['index']})")
            return dev["index"]
    return None


def detect_audio_device() -> tuple:
    """Auto-detect best audio device.

    Priority: SoundGrid > Dante > system default.
    Returns (device_index_or_name, AudioDeviceType).
    """
    devices = list_audio_devices()
    # 1) SoundGrid
    for dev in devices:
        name_lower = dev["name"].lower()
        if any(p in name_lower for p in SOUNDGRID_PATTERNS):
            logger.info(f"SoundGrid device detected: '{dev['name']}' ({dev['max_input_channels']}ch)")
            return dev["index"], AudioDeviceType.SOUNDGRID
    # 2) Dante
    for dev in devices:
        if "dante" in dev["name"].lower():
            logger.info(f"Dante device detected: '{dev['name']}' ({dev['max_input_channels']}ch)")
            return dev["index"], AudioDeviceType.DANTE
    # 3) Default
    logger.info("No SoundGrid/Dante device found — using system default")
    return None, AudioDeviceType.DEFAULT


@dataclass
class ChannelBuffer:
    """Ring buffer for a single channel's audio data."""
    channel: int
    buffer: np.ndarray
    write_pos: int = 0
    sample_rate: int = 48000
    filled: bool = False

    def write(self, samples: np.ndarray):
        """Write samples to ring buffer."""
        n = len(samples)
        buf_len = len(self.buffer)
        if n >= buf_len:
            self.buffer[:] = samples[-buf_len:]
            self.write_pos = 0
            self.filled = True
        elif self.write_pos + n <= buf_len:
            self.buffer[self.write_pos:self.write_pos + n] = samples
            self.write_pos += n
            if self.write_pos >= buf_len:
                self.write_pos = 0
                self.filled = True
        else:
            first = buf_len - self.write_pos
            self.buffer[self.write_pos:] = samples[:first]
            remainder = n - first
            self.buffer[:remainder] = samples[first:]
            self.write_pos = remainder
            self.filled = True

    def read(self, num_samples: int = 0) -> np.ndarray:
        """Read the most recent samples from the buffer."""
        buf_len = len(self.buffer)
        if num_samples <= 0:
            num_samples = buf_len

        num_samples = min(num_samples, buf_len)

        if not self.filled and self.write_pos < num_samples:
            return self.buffer[:self.write_pos].copy()

        start = (self.write_pos - num_samples) % buf_len
        if start < self.write_pos:
            return self.buffer[start:self.write_pos].copy()
        else:
            return np.concatenate([
                self.buffer[start:],
                self.buffer[:self.write_pos]
            ])

    @property
    def available_samples(self) -> int:
        if self.filled:
            return len(self.buffer)
        return self.write_pos


class AudioCapture:
    """
    Unified audio capture service.

    Manages ring buffers per channel, supports multiple audio sources,
    and provides a subscribe/unsubscribe pattern for consumers.
    """

    def __init__(
        self,
        num_channels: int = 40,
        sample_rate: int = 48000,
        buffer_seconds: float = 5.0,
        block_size: int = 1024,
        source_type: AudioSourceType = AudioSourceType.SILENCE,
        device_name: Optional[str] = None,
    ):
        self.num_channels = num_channels
        self.sample_rate = sample_rate
        self.buffer_seconds = buffer_seconds
        self.block_size = block_size
        self.source_type = source_type
        self.device_name = device_name

        buffer_size = int(sample_rate * buffer_seconds)
        self._buffers: Dict[int, ChannelBuffer] = {
            ch: ChannelBuffer(
                channel=ch,
                buffer=np.zeros(buffer_size, dtype=np.float32),
                sample_rate=sample_rate,
            )
            for ch in range(1, num_channels + 1)
        }

        self._subscribers: Dict[str, Callable] = {}
        self._running = False
        self._capture_thread: Optional[threading.Thread] = None
        self._stream = None
        self._lock = threading.Lock()
        self._test_phase: Dict[int, float] = {ch: 0.0 for ch in range(1, num_channels + 1)}

        logger.info(
            f"AudioCapture initialized: {num_channels}ch, {sample_rate}Hz, "
            f"{buffer_seconds}s buffer, source={source_type.value}"
        )

    def start(self):
        """Start audio capture."""
        if self._running:
            return
        self._running = True

        if self.source_type in (AudioSourceType.DANTE, AudioSourceType.SOUNDDEVICE, AudioSourceType.SOUNDGRID):
            if HAS_SOUNDDEVICE:
                self._start_sounddevice()
            else:
                logger.warning("sounddevice not available, falling back to silence")
                self.source_type = AudioSourceType.SILENCE
                self._start_test_generator()
        else:
            self._start_test_generator()

        logger.info(f"AudioCapture started ({self.source_type.value})")

    def stop(self):
        """Stop audio capture."""
        self._running = False
        if self._stream:
            try:
                self._stream.stop()
                self._stream.close()
            except Exception:
                pass
            self._stream = None
        if self._capture_thread and self._capture_thread.is_alive():
            self._capture_thread.join(timeout=2.0)
        logger.info("AudioCapture stopped")

    def _start_sounddevice(self):
        """Start capture via sounddevice (Dante/system audio)."""
        device = self.device_name
        try:
            self._stream = sd.InputStream(
                device=device,
                channels=min(self.num_channels, 64),
                samplerate=self.sample_rate,
                blocksize=self.block_size,
                dtype="float32",
                callback=self._audio_callback,
            )
            self._stream.start()
        except Exception as e:
            logger.error(f"Failed to start sounddevice stream: {e}")
            self.source_type = AudioSourceType.SILENCE
            self._start_test_generator()

    def _audio_callback(self, indata, frames, time_info, status):
        """sounddevice callback for real audio input."""
        if status:
            logger.debug(f"Audio callback status: {status}")
        num_ch = min(indata.shape[1], self.num_channels)
        for ch_idx in range(num_ch):
            ch = ch_idx + 1
            if ch in self._buffers:
                self._buffers[ch].write(indata[:, ch_idx])
        self._notify_subscribers()

    def _start_test_generator(self):
        """Start test signal generator thread."""
        self._capture_thread = threading.Thread(
            target=self._test_generator_loop, daemon=True
        )
        self._capture_thread.start()

    def _test_generator_loop(self):
        """Generate test audio signals."""
        interval = self.block_size / self.sample_rate
        t = np.arange(self.block_size, dtype=np.float32) / self.sample_rate

        while self._running:
            for ch in range(1, self.num_channels + 1):
                if ch not in self._buffers:
                    continue

                if self.source_type == AudioSourceType.TEST_SINE:
                    freq = 200.0 + ch * 50.0  # Different freq per channel
                    phase = self._test_phase[ch]
                    samples = 0.5 * np.sin(
                        2.0 * np.pi * freq * t + phase
                    ).astype(np.float32)
                    self._test_phase[ch] = phase + 2.0 * np.pi * freq * interval
                elif self.source_type == AudioSourceType.TEST_PINK_NOISE:
                    white = np.random.randn(self.block_size).astype(np.float32)
                    # Simple pink noise approximation (1/f)
                    fft = np.fft.rfft(white)
                    freqs = np.fft.rfftfreq(self.block_size, 1.0 / self.sample_rate)
                    freqs[0] = 1.0  # Avoid division by zero
                    fft /= np.sqrt(freqs)
                    samples = np.fft.irfft(fft, n=self.block_size).astype(np.float32)
                    samples *= 0.3 / (np.max(np.abs(samples)) + 1e-10)
                else:  # SILENCE
                    samples = np.zeros(self.block_size, dtype=np.float32)

                self._buffers[ch].write(samples)

            self._notify_subscribers()
            time.sleep(interval)

    def subscribe(self, name: str, callback: Callable):
        """Subscribe to audio data notifications."""
        with self._lock:
            self._subscribers[name] = callback
        logger.debug(f"Subscriber '{name}' added")

    def unsubscribe(self, name: str):
        """Remove a subscriber."""
        with self._lock:
            self._subscribers.pop(name, None)
        logger.debug(f"Subscriber '{name}' removed")

    def _notify_subscribers(self):
        """Notify all subscribers of new data."""
        with self._lock:
            subs = list(self._subscribers.values())
        for cb in subs:
            try:
                cb()
            except Exception as e:
                logger.debug(f"Subscriber callback error: {e}")

    def get_buffer(self, channel: int, num_samples: int = 0) -> np.ndarray:
        """Get audio samples from a channel's ring buffer."""
        buf = self._buffers.get(channel)
        if buf is None:
            return np.array([], dtype=np.float32)
        return buf.read(num_samples)

    def get_lufs(self, channel: int, window_seconds: float = 0.4) -> float:
        """Compute momentary LUFS for a channel."""
        num_samples = int(self.sample_rate * window_seconds)
        samples = self.get_buffer(channel, num_samples)
        if len(samples) < 1024:
            return -100.0
        # Simplified LUFS (K-weighted RMS approximation)
        rms = np.sqrt(np.mean(samples ** 2) + 1e-12)
        lufs = 20.0 * np.log10(rms + 1e-10) - 0.691
        return float(max(lufs, -100.0))

    def get_spectrum(self, channel: int, fft_size: int = 2048) -> Dict[str, np.ndarray]:
        """Compute magnitude spectrum for a channel."""
        samples = self.get_buffer(channel, fft_size)
        if len(samples) < fft_size:
            samples = np.pad(samples, (0, fft_size - len(samples)))
        window = np.hanning(fft_size)
        spectrum = np.abs(np.fft.rfft(samples * window))
        freqs = np.fft.rfftfreq(fft_size, 1.0 / self.sample_rate)
        magnitude_db = 20.0 * np.log10(spectrum + 1e-10)
        return {
            "frequencies": freqs,
            "magnitude_db": magnitude_db,
        }

    def get_peak(self, channel: int, num_samples: int = 1024) -> float:
        """Get peak level in dB for a channel."""
        samples = self.get_buffer(channel, num_samples)
        if len(samples) == 0:
            return -100.0
        peak = np.max(np.abs(samples))
        return float(20.0 * np.log10(peak + 1e-10))

    def get_rms(self, channel: int, num_samples: int = 1024) -> float:
        """Get RMS level in dB for a channel."""
        samples = self.get_buffer(channel, num_samples)
        if len(samples) == 0:
            return -100.0
        rms = np.sqrt(np.mean(samples ** 2) + 1e-12)
        return float(20.0 * np.log10(rms + 1e-10))

    @property
    def running(self) -> bool:
        return self._running

    def get_status(self) -> Dict:
        """Get capture service status."""
        return {
            "running": self._running,
            "source_type": self.source_type.value,
            "num_channels": self.num_channels,
            "sample_rate": self.sample_rate,
            "buffer_seconds": self.buffer_seconds,
            "subscribers": len(self._subscribers),
        }
