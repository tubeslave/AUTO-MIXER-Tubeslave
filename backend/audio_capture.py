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

LIVE_CAPTURE_MAX_CHANNELS = 64
LIVE_CAPTURE_MIN_SAMPLE_RATE = 8_000
LIVE_CAPTURE_MAX_SAMPLE_RATE = 192_000
LIVE_CAPTURE_MIN_BUFFER_SECONDS = 0.25
LIVE_CAPTURE_MAX_BUFFER_SECONDS = 30.0
LIVE_CAPTURE_MAX_BUFFER_BYTES = 256 * 1024 * 1024


def validate_live_capture_dimensions(
    num_channels: int,
    sample_rate: int,
    buffer_seconds: float,
) -> tuple[int, int, float]:
    """Validate a live allocation before constructing per-channel buffers."""

    if isinstance(num_channels, bool) or not 1 <= int(num_channels) <= LIVE_CAPTURE_MAX_CHANNELS:
        raise ValueError(f"num_channels must be between 1 and {LIVE_CAPTURE_MAX_CHANNELS}")
    if isinstance(sample_rate, bool) or not (
        LIVE_CAPTURE_MIN_SAMPLE_RATE
        <= int(sample_rate)
        <= LIVE_CAPTURE_MAX_SAMPLE_RATE
    ):
        raise ValueError(
            f"sample_rate must be between {LIVE_CAPTURE_MIN_SAMPLE_RATE} and "
            f"{LIVE_CAPTURE_MAX_SAMPLE_RATE}"
        )
    buffer_seconds = float(buffer_seconds)
    if not np.isfinite(buffer_seconds) or not (
        LIVE_CAPTURE_MIN_BUFFER_SECONDS
        <= buffer_seconds
        <= LIVE_CAPTURE_MAX_BUFFER_SECONDS
    ):
        raise ValueError(
            f"buffer_seconds must be between {LIVE_CAPTURE_MIN_BUFFER_SECONDS} "
            f"and {LIVE_CAPTURE_MAX_BUFFER_SECONDS}"
        )
    estimated_bytes = int(num_channels) * int(sample_rate) * buffer_seconds * 4
    if estimated_bytes > LIVE_CAPTURE_MAX_BUFFER_BYTES:
        raise ValueError(
            "requested audio capture buffer exceeds the 256 MiB live safety limit"
        )
    return int(num_channels), int(sample_rate), buffer_seconds

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


@dataclass
class TestGeneratorConfig:
    """Compatibility config for simple offline test generators."""
    __test__ = False
    generator_type: str = "silence"
    amplitude: float = 0.5
    frequency: float = 1000.0


class RingBuffer:
    """Simple mono ring buffer kept for test/integration compatibility."""

    def __init__(self, max_duration_sec: float, sample_rate: int = 48000):
        self.sample_rate = sample_rate
        self.capacity = max(1, int(max_duration_sec * sample_rate))
        self._buffer = np.zeros(self.capacity, dtype=np.float32)
        self._write_pos = 0
        self._size = 0

    def write(self, samples: np.ndarray):
        samples = np.asarray(samples, dtype=np.float32).flatten()
        if len(samples) >= self.capacity:
            self._buffer[:] = samples[-self.capacity:]
            self._write_pos = 0
            self._size = self.capacity
            return

        end = self._write_pos + len(samples)
        if end <= self.capacity:
            self._buffer[self._write_pos:end] = samples
        else:
            first = self.capacity - self._write_pos
            self._buffer[self._write_pos:] = samples[:first]
            self._buffer[:len(samples) - first] = samples[first:]
        self._write_pos = end % self.capacity
        self._size = min(self.capacity, self._size + len(samples))

    def read(self, num_samples: int) -> np.ndarray:
        if num_samples <= 0:
            return np.array([], dtype=np.float32)

        num_samples = min(num_samples, self.capacity)
        available = min(self._size, num_samples)
        result = np.zeros(num_samples, dtype=np.float32)
        if available == 0:
            return result

        start = (self._write_pos - available) % self.capacity
        if start < self._write_pos:
            data = self._buffer[start:self._write_pos]
        else:
            data = np.concatenate((self._buffer[start:], self._buffer[:self._write_pos]))
        result[-available:] = data[-available:]
        return result

    @property
    def available_samples(self) -> int:
        return self._size


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
        channels: Optional[int] = None,
        buffer_duration: Optional[float] = None,
    ):
        if channels is not None:
            num_channels = channels
        if buffer_duration is not None:
            buffer_seconds = buffer_duration

        self.num_channels = num_channels
        self.sample_rate = sample_rate
        self.buffer_seconds = buffer_seconds
        self.block_size = block_size
        self.source_type = source_type
        self.device_name = device_name
        self._resolved_device_name: Optional[str] = None
        self._resolved_device_index: Optional[int] = None

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
        self._global_subscribers: List[Callable] = []
        self._running = False
        self._capture_thread: Optional[threading.Thread] = None
        self._stream = None
        self._lock = threading.Lock()
        # A callback block is one alignment generation. Writers hold this lock
        # across every channel update and the generation counter; multi-channel
        # readers can therefore never combine adjacent callback blocks.
        self._data_lock = threading.Lock()
        self.allow_silence_fallback = True
        self._last_callback_monotonic: Optional[float] = None
        self._captured_sample_count = 0
        self._test_phase: Dict[int, float] = {ch: 0.0 for ch in range(1, num_channels + 1)}
        # Set when a real-input source silently degraded to the silence generator,
        # so consumers can distinguish "capturing silence" from "capturing audio"
        self.silence_fallback = False
        self.silence_fallback_reason: Optional[str] = None

        logger.info(
            f"AudioCapture initialized: {num_channels}ch, {sample_rate}Hz, "
            f"{buffer_seconds}s buffer, source={source_type.value}"
        )

    def start(self):
        """Start audio capture."""
        if self._running:
            return
        self._running = True

        self.silence_fallback = False
        self.silence_fallback_reason = None
        if self.source_type in (AudioSourceType.DANTE, AudioSourceType.SOUNDDEVICE, AudioSourceType.SOUNDGRID):
            if HAS_SOUNDDEVICE:
                self._start_sounddevice()
            else:
                if not self.allow_silence_fallback:
                    self._running = False
                    raise RuntimeError(
                        "sounddevice is unavailable; refusing synthetic silence fallback"
                    )
                logger.warning("sounddevice not available, falling back to silence")
                self.source_type = AudioSourceType.SILENCE
                self.silence_fallback = True
                self.silence_fallback_reason = "sounddevice not available"
                self._start_test_generator()
        else:
            self._start_test_generator()

        logger.info(f"AudioCapture started ({self.source_type.value})")

    def stop(self):
        """Stop audio capture."""
        self._running = False
        errors = []
        if self._stream:
            stream = self._stream
            try:
                stream.stop()
            except Exception as stop_error:
                logger.error("Failed to stop sounddevice stream cleanly: %s", stop_error)
                errors.append(("stream_stop", stop_error))
            try:
                stream.close()
            except Exception as close_error:
                logger.error("Failed to close sounddevice stream cleanly: %s", close_error)
                errors.append(("stream_close", close_error))
            if not errors:
                self._stream = None
        if self._capture_thread and self._capture_thread.is_alive():
            self._capture_thread.join(timeout=2.0)
            if self._capture_thread.is_alive():
                errors.append((
                    "capture_thread",
                    RuntimeError("capture thread did not stop within 2 seconds"),
                ))
            else:
                self._capture_thread = None
        elif self._capture_thread is not None:
            self._capture_thread = None
        if errors:
            details = ", ".join(
                f"{step}:{type(error).__name__}" for step, error in errors
            )
            raise RuntimeError(f"AudioCapture stop incomplete ({details})")
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
            stream_device = getattr(self._stream, "device", device)
            if isinstance(stream_device, (tuple, list)) and stream_device:
                stream_device = stream_device[0]
            if isinstance(stream_device, int) and not isinstance(stream_device, bool):
                self._resolved_device_index = int(stream_device)
            elif isinstance(device, int) and not isinstance(device, bool):
                self._resolved_device_index = int(device)
            try:
                device_info = sd.query_devices(
                    self._resolved_device_index
                    if self._resolved_device_index is not None
                    else device,
                    "input",
                )
            except TypeError:
                device_info = sd.query_devices(
                    self._resolved_device_index
                    if self._resolved_device_index is not None
                    else device
                )
            if isinstance(device_info, dict):
                resolved_name = str(device_info.get("name") or "").strip()
                if resolved_name:
                    self._resolved_device_name = resolved_name
        except Exception as e:
            logger.error(f"Failed to start sounddevice stream: {e}")
            stream = self._stream
            cleanup_errors = []
            if stream is not None:
                try:
                    stream.stop()
                except Exception as stop_error:
                    logger.error(
                        "Failed to stop sounddevice stream after startup error: %s",
                        stop_error,
                    )
                    cleanup_errors.append(("stream_stop", stop_error))
                try:
                    stream.close()
                except Exception as close_error:
                    logger.error(
                        "Failed to close sounddevice stream after startup error: %s",
                        close_error,
                    )
                    cleanup_errors.append(("stream_close", close_error))
                if not cleanup_errors:
                    self._stream = None
            if cleanup_errors:
                self._running = False
                detail = ", ".join(
                    f"{step}:{type(error).__name__}"
                    for step, error in cleanup_errors
                )
                raise RuntimeError(
                    f"real audio input startup cleanup incomplete ({detail})"
                ) from e
            if not self.allow_silence_fallback:
                self._running = False
                raise RuntimeError(
                    "real audio input failed; refusing synthetic silence fallback"
                ) from e
            self.source_type = AudioSourceType.SILENCE
            self.silence_fallback = True
            self.silence_fallback_reason = str(e)
            self._start_test_generator()

    def _audio_callback(self, indata, frames, time_info, status):
        """sounddevice callback for real audio input."""
        if status:
            logger.debug(f"Audio callback status: {status}")
        num_ch = min(indata.shape[1], self.num_channels)
        with self._data_lock:
            for ch_idx in range(num_ch):
                ch = ch_idx + 1
                if ch in self._buffers:
                    self._buffers[ch].write(indata[:, ch_idx])
            self._captured_sample_count += max(0, int(frames))
            self._last_callback_monotonic = time.monotonic()
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
            with self._data_lock:
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
            global_subs = list(self._global_subscribers)
        for cb in subs:
            try:
                cb()
            except Exception as e:
                logger.debug(f"Subscriber callback error: {e}")
        if global_subs:
            for ch in self._buffers:
                data = self.get_buffer(ch, self.block_size)
                for cb in global_subs:
                    try:
                        cb(ch, data)
                    except Exception as e:
                        logger.debug(f"Global subscriber callback error: {e}")

    def get_buffer(self, channel: Optional[int] = None, num_samples: int = 0, channel_id: Optional[int] = None) -> np.ndarray:
        """Get audio samples from a channel's ring buffer."""
        if channel is None:
            channel = channel_id
        if channel is None:
            channel = 1
        if channel == 0:
            channel = 1
        with self._data_lock:
            buf = self._buffers.get(channel)
            if buf is None:
                return np.array([], dtype=np.float32)
            return buf.read(num_samples)

    def get_synchronized_buffers(
        self,
        channels: List[int],
        num_samples: int,
    ) -> Dict[str, Any]:
        """Atomically snapshot aligned channels from one completed generation."""

        requested = [int(channel) for channel in channels]
        if not requested or len(set(requested)) != len(requested):
            raise ValueError("Synchronized audio snapshot requires distinct channels")
        if int(num_samples) <= 0:
            raise ValueError("Synchronized audio snapshot requires positive samples")
        with self._data_lock:
            buffers = {}
            available_samples = {}
            for channel in requested:
                buf = self._buffers.get(channel)
                if buf is None:
                    buffers[channel] = np.array([], dtype=np.float32)
                    available_samples[channel] = 0
                else:
                    buffers[channel] = buf.read(int(num_samples))
                    available_samples[channel] = int(buf.available_samples)
            return {
                "buffers": buffers,
                "available_samples": available_samples,
                "captured_sample_count": int(self._captured_sample_count),
                "last_callback_monotonic": self._last_callback_monotonic,
            }

    def get_available_samples(self, channel: int) -> int:
        """Return buffered samples for one channel without exposing buffers."""

        with self._data_lock:
            buf = self._buffers.get(int(channel))
            return int(buf.available_samples) if buf is not None else 0

    @property
    def captured_sample_count(self) -> int:
        """Monotonic number of real callback frames captured by this instance."""

        with self._data_lock:
            return int(self._captured_sample_count)

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
        with self._data_lock:
            last_callback_monotonic = self._last_callback_monotonic
            captured_sample_count = int(self._captured_sample_count)
        callback_age_sec = (
            max(0.0, time.monotonic() - last_callback_monotonic)
            if last_callback_monotonic is not None
            else None
        )
        return {
            "running": self._running,
            "source_type": self.source_type.value,
            "device_name": self._resolved_device_name
            or (str(self.device_name) if self.device_name is not None else ""),
            "device_index": self._resolved_device_index,
            "num_channels": self.num_channels,
            "sample_rate": self.sample_rate,
            "buffer_seconds": self.buffer_seconds,
            "subscribers": len(self._subscribers),
            "callback_age_sec": round(callback_age_sec, 3) if callback_age_sec is not None else None,
            "captured_sample_count": captured_sample_count,
            "real_input": bool(
                self._running
                and self.source_type
                in {
                    AudioSourceType.SOUNDDEVICE,
                    AudioSourceType.DANTE,
                    AudioSourceType.SOUNDGRID,
                }
                and self._stream is not None
                and callback_age_sec is not None
                and callback_age_sec <= 1.0
            ),
        }

    def subscribe_all(self, callback: Callable):
        """Compatibility helper for consumers that expect per-channel callbacks."""
        with self._lock:
            self._global_subscribers.append(callback)

    def unsubscribe_all(self, callback: Callable):
        """Remove a global callback if present."""
        with self._lock:
            self._global_subscribers = [cb for cb in self._global_subscribers if cb != callback]

    @staticmethod
    def _voss_mccartney(num_samples: int, num_rows: int = 16) -> np.ndarray:
        """Generate pink-ish noise using the Voss-McCartney method."""
        if num_samples <= 0:
            return np.array([], dtype=np.float32)

        rows = np.random.randn(num_rows, num_samples).astype(np.float32)
        output = np.zeros(num_samples, dtype=np.float32)
        running = np.zeros(num_rows, dtype=np.float32)

        for i in range(num_samples):
            changed_bits = (~i) & (i + 1)
            for row in range(num_rows):
                if changed_bits & (1 << row):
                    running[row] = rows[row, i]
            output[i] = running.sum()

        peak = np.max(np.abs(output)) + 1e-10
        return (output / peak).astype(np.float32)
