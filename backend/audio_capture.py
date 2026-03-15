"""
Unified AudioCapture service with ring buffers per channel.

Provides subscribe/unsubscribe pattern for real-time audio data,
Dante input via sounddevice, K-weighted LUFS metering per ITU-R BS.1770,
frequency spectrum analysis, and fallback test signal generators
(sine, pink noise, file playback, silence) for development without hardware.

Thread-safe ring buffer implementation supports concurrent read/write
from audio callback and analysis threads.
"""

import logging
import os
import struct
import threading
import time
import wave
from collections import defaultdict
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Optional dependency: sounddevice (Dante / ASIO / PortAudio input)
# ---------------------------------------------------------------------------
try:
    import sounddevice as sd
    HAS_SOUNDDEVICE = True
except ImportError:
    sd = None  # type: ignore[assignment]
    HAS_SOUNDDEVICE = False

# ---------------------------------------------------------------------------
# Optional dependency: scipy (improved pink noise filter, K-weighting)
# ---------------------------------------------------------------------------
try:
    from scipy import signal as scipy_signal
    HAS_SCIPY = True
except ImportError:
    scipy_signal = None  # type: ignore[assignment]
    HAS_SCIPY = False

# ---------------------------------------------------------------------------
# Optional dependency: soundfile (WAV/FLAC/OGG file playback)
# ---------------------------------------------------------------------------
try:
    import soundfile as sf
    HAS_SOUNDFILE = True
except ImportError:
    sf = None  # type: ignore[assignment]
    HAS_SOUNDFILE = False


# ═══════════════════════════════════════════════════════════════════════════
#  Ring Buffer
# ═══════════════════════════════════════════════════════════════════════════

class RingBuffer:
    """Thread-safe circular buffer for mono audio samples.

    Supports concurrent writes from the audio callback and reads from
    analysis threads.  All public methods acquire an internal lock so the
    caller does not need external synchronisation.

    Parameters
    ----------
    max_duration_sec : float
        Maximum buffered duration in seconds (default 10 s).
    sample_rate : int
        Audio sample rate in Hz (default 48 000).
    """

    def __init__(self, max_duration_sec: float = 10.0,
                 sample_rate: int = 48000) -> None:
        self._sample_rate: int = sample_rate
        self._max_samples: int = int(max_duration_sec * sample_rate)
        self._buffer: np.ndarray = np.zeros(self._max_samples, dtype=np.float32)
        self._write_pos: int = 0
        self._total_written: int = 0
        self._lock = threading.Lock()

    # -- write --------------------------------------------------------------

    def write(self, data: np.ndarray) -> None:
        """Append *data* to the ring buffer, overwriting oldest samples."""
        data = np.asarray(data, dtype=np.float32).ravel()
        n = len(data)
        if n == 0:
            return
        with self._lock:
            if n >= self._max_samples:
                # Data exceeds buffer size — keep the most recent samples.
                self._buffer[:] = data[-self._max_samples:]
                self._write_pos = 0
                self._total_written += n
            elif self._write_pos + n <= self._max_samples:
                # Data fits without wrapping.
                self._buffer[self._write_pos:self._write_pos + n] = data
                self._write_pos += n
                self._total_written += n
            else:
                # Data wraps around the end of the buffer.
                first = self._max_samples - self._write_pos
                self._buffer[self._write_pos:] = data[:first]
                self._buffer[:n - first] = data[first:]
                self._write_pos = n - first
                self._total_written += n

    # -- read ---------------------------------------------------------------

    def read(self, num_samples: int) -> np.ndarray:
        """Return the most recent *num_samples* from the buffer.

        If fewer samples are available, the result is zero-padded on the
        left (older side).
        """
        with self._lock:
            available = min(num_samples, self._max_samples, self._total_written)
            if available == 0:
                return np.zeros(num_samples, dtype=np.float32)

            end = self._write_pos
            start = end - available
            if start >= 0:
                result = self._buffer[start:end].copy()
            else:
                result = np.concatenate([
                    self._buffer[start:],  # wraps to end of array
                    self._buffer[:end],
                ])

            if len(result) < num_samples:
                result = np.pad(result, (num_samples - len(result), 0))
            return result

    # -- read_all -----------------------------------------------------------

    def read_all(self) -> np.ndarray:
        """Return all available samples (up to buffer capacity)."""
        return self.read(self.available_samples)

    # -- clear --------------------------------------------------------------

    def clear(self) -> None:
        """Reset the buffer to silence."""
        with self._lock:
            self._buffer[:] = 0.0
            self._write_pos = 0
            self._total_written = 0

    # -- properties ---------------------------------------------------------

    @property
    def available_samples(self) -> int:
        """Number of valid samples currently stored."""
        return min(self._total_written, self._max_samples)

    @property
    def available_duration(self) -> float:
        """Duration of stored audio in seconds."""
        return self.available_samples / self._sample_rate

    @property
    def capacity_samples(self) -> int:
        """Maximum number of samples the buffer can hold."""
        return self._max_samples

    @property
    def sample_rate(self) -> int:
        return self._sample_rate


# ═══════════════════════════════════════════════════════════════════════════
#  K-Weighting Filter (ITU-R BS.1770-4)
# ═══════════════════════════════════════════════════════════════════════════

class KWeightingFilter:
    """Two-stage K-weighting filter per ITU-R BS.1770-4.

    Stage 1 — high-shelf filter (~+4 dB above 1.5 kHz) to model the
    acoustic effect of the listener's head.
    Stage 2 — high-pass filter (~60 Hz) to attenuate very low frequencies.

    The filter is designed for 48 kHz by default; coefficients are
    recalculated for other sample rates.
    """

    def __init__(self, sample_rate: int = 48000) -> None:
        self.sample_rate = sample_rate
        self._design_filters()
        self._reset_state()

    def _design_filters(self) -> None:
        fs = self.sample_rate

        # Stage 1: high-shelf  (+4 dB above ~1681 Hz)
        f0 = 1681.974450955533
        G = 3.999843853973347
        Q = 0.7071752369554196

        K = np.tan(np.pi * f0 / fs)
        Vh = 10.0 ** (G / 20.0)
        Vb = Vh ** 0.4996667741545416

        a0 = 1.0 + K / Q + K * K
        self._shelf_b = np.array([
            (Vh + Vb * K / Q + K * K) / a0,
            2.0 * (K * K - Vh) / a0,
            (Vh - Vb * K / Q + K * K) / a0,
        ])
        self._shelf_a = np.array([
            1.0,
            2.0 * (K * K - 1.0) / a0,
            (1.0 - K / Q + K * K) / a0,
        ])

        # Stage 2: high-pass  (~60 Hz, Q ≈ 0.5)
        f1 = 38.13547087602444
        Q1 = 0.5003270373238773

        K1 = np.tan(np.pi * f1 / fs)
        a0_hp = 1.0 + K1 / Q1 + K1 * K1
        self._hp_b = np.array([
            1.0 / a0_hp,
            -2.0 / a0_hp,
            1.0 / a0_hp,
        ])
        self._hp_a = np.array([
            1.0,
            2.0 * (K1 * K1 - 1.0) / a0_hp,
            (1.0 - K1 / Q1 + K1 * K1) / a0_hp,
        ])

    def _reset_state(self) -> None:
        """Reset filter delay-line state."""
        self._shelf_zi = np.zeros(2, dtype=np.float64)
        self._hp_zi = np.zeros(2, dtype=np.float64)

    def apply(self, audio: np.ndarray) -> np.ndarray:
        """Apply K-weighting filter to *audio* and return the result.

        Uses scipy.signal.lfilter when available, otherwise falls back
        to a pure-numpy direct-form II implementation.
        """
        x = np.asarray(audio, dtype=np.float64)
        if HAS_SCIPY:
            y, self._shelf_zi = scipy_signal.lfilter(
                self._shelf_b, self._shelf_a, x, zi=self._shelf_zi,
            )
            y, self._hp_zi = scipy_signal.lfilter(
                self._hp_b, self._hp_a, y, zi=self._hp_zi,
            )
        else:
            y = self._lfilter_fallback(self._shelf_b, self._shelf_a, x)
            y = self._lfilter_fallback(self._hp_b, self._hp_a, y)
        return y.astype(np.float32)

    @staticmethod
    def _lfilter_fallback(b: np.ndarray, a: np.ndarray,
                          x: np.ndarray) -> np.ndarray:
        """Minimal direct-form II transposed IIR filter (no scipy)."""
        n = len(x)
        y = np.zeros(n, dtype=np.float64)
        order = max(len(b), len(a)) - 1
        d = np.zeros(order, dtype=np.float64)
        b_pad = np.zeros(order + 1, dtype=np.float64)
        a_pad = np.zeros(order + 1, dtype=np.float64)
        b_pad[:len(b)] = b
        a_pad[:len(a)] = a
        for i in range(n):
            y[i] = b_pad[0] * x[i] + d[0]
            for j in range(order - 1):
                d[j] = b_pad[j + 1] * x[i] - a_pad[j + 1] * y[i] + d[j + 1]
            d[order - 1] = b_pad[order] * x[i] - a_pad[order] * y[i]
        return y


# ═══════════════════════════════════════════════════════════════════════════
#  Capture State
# ═══════════════════════════════════════════════════════════════════════════

class CaptureState(Enum):
    """Current state of the AudioCapture service."""
    STOPPED = "stopped"
    RUNNING = "running"
    TEST_GENERATOR = "test_generator"
    ERROR = "error"


# ═══════════════════════════════════════════════════════════════════════════
#  Test Generator Config
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class TestGeneratorConfig:
    """Configuration for test signal generators.

    Attributes
    ----------
    generator_type : str
        One of ``"sine"``, ``"pink_noise"``, ``"file"``, ``"silence"``.
    frequency : float
        Sine wave frequency in Hz (only used when *generator_type* is
        ``"sine"``).
    amplitude : float
        Peak amplitude in the range [0.0, 1.0].
    file_path : str or None
        Path to a WAV/FLAC/OGG file for ``"file"`` generator type.
        Requires the *soundfile* package.
    loop : bool
        Whether to loop file playback.
    channel_spread : bool
        When True, spread the test signal across multiple channels with
        slight frequency offsets (sine) or independent noise (pink_noise).
    num_channels : int
        Number of channels to fill with the test signal.  Defaults to 2.
    """
    generator_type: str = "silence"
    frequency: float = 440.0
    amplitude: float = 0.5
    file_path: Optional[str] = None
    loop: bool = True
    channel_spread: bool = True
    num_channels: int = 2


# ═══════════════════════════════════════════════════════════════════════════
#  Channel Statistics
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class ChannelStats:
    """Per-channel metering statistics, updated on each analysis pass."""
    channel_id: int = 0
    peak_db: float = -100.0
    rms_db: float = -100.0
    lufs_momentary: float = -100.0
    lufs_short_term: float = -100.0
    is_clipping: bool = False
    last_update: float = 0.0


# ═══════════════════════════════════════════════════════════════════════════
#  AudioCapture
# ═══════════════════════════════════════════════════════════════════════════

class AudioCapture:
    """Unified audio capture service with per-channel ring buffers.

    Supports real-time Dante/ASIO input via *sounddevice* when available,
    and provides fallback test signal generators for development
    environments without audio hardware.

    Features
    --------
    - Per-channel thread-safe ring buffers.
    - Subscribe/unsubscribe callbacks for real-time sample delivery.
    - K-weighted LUFS metering (momentary 400 ms, short-term 3 s)
      per ITU-R BS.1770-4.
    - Frequency spectrum (FFT) per channel.
    - True-peak detection.
    - Test generators: sine wave, pink noise, file playback, silence.

    Parameters
    ----------
    sample_rate : int
        Audio sample rate in Hz (default 48 000).
    channels : int
        Maximum number of input channels (default 64).
    buffer_duration : float
        Ring buffer duration per channel in seconds (default 10.0).
    device : int or None
        Sounddevice device index.  ``None`` uses the default device.
    block_size : int
        Audio callback block size in samples (default 1024).
    """

    # Maximum channels sounddevice can handle in one stream.
    _MAX_SD_CHANNELS: int = 64

    def __init__(
        self,
        sample_rate: int = 48000,
        channels: int = 64,
        buffer_duration: float = 10.0,
        device: Optional[int] = None,
        block_size: int = 1024,
    ) -> None:
        self.sample_rate: int = sample_rate
        self.channels: int = channels
        self.buffer_duration: float = buffer_duration
        self.device: Optional[int] = device
        self.block_size: int = block_size

        # Per-channel ring buffers.
        self._buffers: Dict[int, RingBuffer] = {}
        for ch in range(channels):
            self._buffers[ch] = RingBuffer(buffer_duration, sample_rate)

        # Subscriber callbacks:  channel_id -> [callback(channel_id, samples)]
        self._subscribers: Dict[int, List[Callable[[int, np.ndarray], None]]] = \
            defaultdict(list)

        # Global subscribers: called for every channel.
        self._global_subscribers: List[Callable[[int, np.ndarray], None]] = []

        # State management.
        self._state: CaptureState = CaptureState.STOPPED
        self._stream: Any = None  # sd.InputStream or None.
        self._test_thread: Optional[threading.Thread] = None
        self._test_generator_config: Optional[TestGeneratorConfig] = None

        # Per-channel K-weighting filters for LUFS.
        self._k_filters: Dict[int, KWeightingFilter] = {}

        # Per-channel metering stats.
        self._channel_stats: Dict[int, ChannelStats] = {}
        for ch in range(channels):
            self._channel_stats[ch] = ChannelStats(channel_id=ch)

        # File playback state.
        self._file_data: Optional[np.ndarray] = None
        self._file_sample_rate: int = sample_rate
        self._file_pos: int = 0
        self._file_channels: int = 1

        # Thread safety.
        self._lock = threading.Lock()
        self._callback_errors: int = 0

        # Overflow / underrun counters.
        self._xrun_count: int = 0

        logger.info(
            "AudioCapture initialized: %d channels @ %d Hz, "
            "buffer=%.1f s, block=%d samples",
            channels, sample_rate, buffer_duration, block_size,
        )

    # ═══════════════════════════════════════════════════════════════════
    #  Properties
    # ═══════════════════════════════════════════════════════════════════

    @property
    def state(self) -> CaptureState:
        """Current capture state."""
        return self._state

    @property
    def is_running(self) -> bool:
        """True if capture (live or test generator) is active."""
        return self._state in (CaptureState.RUNNING, CaptureState.TEST_GENERATOR)

    # ═══════════════════════════════════════════════════════════════════
    #  Start / Stop  (Dante / sounddevice)
    # ═══════════════════════════════════════════════════════════════════

    def start(self) -> bool:
        """Start audio capture from Dante / sounddevice.

        Returns True on success, False on failure.  If sounddevice is not
        installed the caller should use :meth:`start_test_generator` instead.
        """
        if self.is_running:
            logger.debug("AudioCapture already running, ignoring start()")
            return True

        if not HAS_SOUNDDEVICE:
            logger.warning(
                "sounddevice not available — install it for Dante capture, "
                "or use start_test_generator() for offline testing"
            )
            return False

        try:
            num_ch = min(self.channels, self._MAX_SD_CHANNELS)
            self._stream = sd.InputStream(
                device=self.device,
                samplerate=self.sample_rate,
                channels=num_ch,
                blocksize=self.block_size,
                dtype="float32",
                callback=self._audio_callback,
                finished_callback=self._stream_finished_callback,
            )
            self._stream.start()
            self._state = CaptureState.RUNNING
            logger.info(
                "Audio capture started via sounddevice (%d channels, device=%s)",
                num_ch, self.device,
            )
            return True
        except Exception as exc:
            logger.error("Failed to start audio capture: %s", exc)
            self._state = CaptureState.ERROR
            return False

    def stop(self) -> None:
        """Stop audio capture (both live and test generator)."""
        prev_state = self._state
        self._state = CaptureState.STOPPED

        # Stop sounddevice stream.
        if self._stream is not None:
            try:
                self._stream.stop()
                self._stream.close()
            except Exception as exc:
                logger.debug("Error stopping sounddevice stream: %s", exc)
            self._stream = None

        # Stop test generator thread.
        if self._test_thread is not None and self._test_thread.is_alive():
            self._test_thread.join(timeout=3.0)
            if self._test_thread.is_alive():
                logger.warning("Test generator thread did not stop within timeout")
        self._test_thread = None

        # Free file playback data.
        self._file_data = None
        self._file_pos = 0

        if prev_state != CaptureState.STOPPED:
            logger.info("Audio capture stopped (was %s)", prev_state.value)

    def _stream_finished_callback(self) -> None:
        """Called by sounddevice when the stream ends unexpectedly."""
        if self._state == CaptureState.RUNNING:
            logger.warning("Audio stream ended unexpectedly")
            self._state = CaptureState.ERROR

    # ═══════════════════════════════════════════════════════════════════
    #  Audio Callback  (sounddevice)
    # ═══════════════════════════════════════════════════════════════════

    def _audio_callback(
        self,
        indata: np.ndarray,
        frames: int,
        time_info: Any,
        status: Any,
    ) -> None:
        """Callback invoked by sounddevice for each audio block.

        This runs in a real-time audio thread — keep work minimal.
        """
        if status:
            self._xrun_count += 1
            if self._xrun_count <= 10 or self._xrun_count % 100 == 0:
                logger.debug(
                    "Audio callback status: %s (xrun #%d)",
                    status, self._xrun_count,
                )

        num_ch = min(
            indata.shape[1] if indata.ndim > 1 else 1,
            self.channels,
        )

        for ch in range(num_ch):
            samples = indata[:, ch] if indata.ndim > 1 else indata.ravel()

            # Write to ring buffer.
            buf = self._buffers.get(ch)
            if buf is not None:
                buf.write(samples)

            # Notify per-channel subscribers.
            self._dispatch_subscribers(ch, samples)

    def _dispatch_subscribers(
        self, channel_id: int, samples: np.ndarray,
    ) -> None:
        """Deliver samples to channel-specific and global subscribers."""
        for callback in self._subscribers.get(channel_id, []):
            try:
                callback(channel_id, samples)
            except Exception as exc:
                self._callback_errors += 1
                if self._callback_errors <= 10:
                    logger.debug(
                        "Subscriber callback error ch%d: %s", channel_id, exc,
                    )

        for callback in self._global_subscribers:
            try:
                callback(channel_id, samples)
            except Exception as exc:
                self._callback_errors += 1
                if self._callback_errors <= 10:
                    logger.debug(
                        "Global subscriber error ch%d: %s", channel_id, exc,
                    )

    # ═══════════════════════════════════════════════════════════════════
    #  Subscribe / Unsubscribe
    # ═══════════════════════════════════════════════════════════════════

    def subscribe(self, channel_id: int,
                  callback: Callable[[int, np.ndarray], None]) -> None:
        """Subscribe *callback* to audio data from *channel_id*.

        The callback signature is ``callback(channel_id: int, samples: ndarray)``.
        Duplicate subscriptions are ignored.
        """
        with self._lock:
            if callback not in self._subscribers[channel_id]:
                self._subscribers[channel_id].append(callback)
                logger.debug("Subscribed to channel %d", channel_id)

    def unsubscribe(self, channel_id: int,
                    callback: Callable[[int, np.ndarray], None]) -> None:
        """Remove *callback* from *channel_id* subscriptions."""
        with self._lock:
            subs = self._subscribers.get(channel_id, [])
            if callback in subs:
                subs.remove(callback)
                logger.debug("Unsubscribed from channel %d", channel_id)

    def subscribe_all(self,
                      callback: Callable[[int, np.ndarray], None]) -> None:
        """Subscribe *callback* to audio from **all** channels."""
        with self._lock:
            if callback not in self._global_subscribers:
                self._global_subscribers.append(callback)

    def unsubscribe_all(self,
                        callback: Callable[[int, np.ndarray], None]) -> None:
        """Remove a global subscriber."""
        with self._lock:
            if callback in self._global_subscribers:
                self._global_subscribers.remove(callback)

    # ═══════════════════════════════════════════════════════════════════
    #  Buffer Access
    # ═══════════════════════════════════════════════════════════════════

    def get_buffer(self, channel_id: int,
                   duration_sec: float = 1.0) -> np.ndarray:
        """Return the most recent *duration_sec* of audio for *channel_id*.

        Returns a zero array if the channel does not exist.
        """
        buf = self._buffers.get(channel_id)
        num_samples = int(duration_sec * self.sample_rate)
        if buf is None:
            return np.zeros(num_samples, dtype=np.float32)
        return buf.read(num_samples)

    def get_buffer_all(self, channel_id: int) -> np.ndarray:
        """Return all available audio for *channel_id*."""
        buf = self._buffers.get(channel_id)
        if buf is None:
            return np.zeros(0, dtype=np.float32)
        return buf.read_all()

    def clear_buffer(self, channel_id: int) -> None:
        """Clear the ring buffer for *channel_id*."""
        buf = self._buffers.get(channel_id)
        if buf is not None:
            buf.clear()

    def clear_all_buffers(self) -> None:
        """Clear all ring buffers."""
        for buf in self._buffers.values():
            buf.clear()

    # ═══════════════════════════════════════════════════════════════════
    #  LUFS Metering  (ITU-R BS.1770-4)
    # ═══════════════════════════════════════════════════════════════════

    def _get_k_filter(self, channel_id: int) -> KWeightingFilter:
        """Get or create a K-weighting filter for *channel_id*."""
        filt = self._k_filters.get(channel_id)
        if filt is None:
            filt = KWeightingFilter(self.sample_rate)
            self._k_filters[channel_id] = filt
        return filt

    def get_lufs(self, channel_id: int,
                 duration_sec: float = 0.4) -> float:
        """Calculate K-weighted LUFS for *channel_id*.

        Parameters
        ----------
        channel_id : int
            Channel to measure.
        duration_sec : float
            Measurement window.  0.4 s = momentary LUFS,
            3.0 s = short-term LUFS per EBU R128.

        Returns
        -------
        float
            LUFS value, or -100.0 for silence / missing channels.
        """
        audio = self.get_buffer(channel_id, duration_sec)
        if np.max(np.abs(audio)) < 1e-10:
            return -100.0

        # Apply K-weighting filter.
        k_filter = self._get_k_filter(channel_id)
        filtered = k_filter.apply(audio)

        # Mean square of K-weighted signal.
        mean_sq = float(np.mean(filtered ** 2))
        if mean_sq < 1e-20:
            return -100.0

        # LUFS = -0.691 + 10 * log10(mean_sq)
        lufs = -0.691 + 10.0 * np.log10(mean_sq + 1e-20)
        return float(lufs)

    def get_lufs_momentary(self, channel_id: int) -> float:
        """Momentary LUFS (400 ms window) per ITU-R BS.1770-4."""
        return self.get_lufs(channel_id, duration_sec=0.4)

    def get_lufs_short_term(self, channel_id: int) -> float:
        """Short-term LUFS (3 s window) per EBU R128."""
        return self.get_lufs(channel_id, duration_sec=3.0)

    def get_true_peak(self, channel_id: int,
                      duration_sec: float = 0.4) -> float:
        """Estimate true peak in dBTP for *channel_id*.

        Uses 4x oversampling when scipy is available for inter-sample
        peak detection per ITU-R BS.1770-4.  Falls back to sample-peak
        otherwise.

        Returns
        -------
        float
            True peak in dBTP, or -100.0 for silence.
        """
        audio = self.get_buffer(channel_id, duration_sec)
        peak = self._compute_true_peak(audio)
        if peak < 1e-10:
            return -100.0
        return float(20.0 * np.log10(peak))

    @staticmethod
    def _compute_true_peak(audio: np.ndarray,
                           oversample: int = 4) -> float:
        """Compute true peak with optional oversampling."""
        if len(audio) == 0:
            return 0.0

        sample_peak = float(np.max(np.abs(audio)))

        if not HAS_SCIPY or len(audio) < 16:
            return sample_peak

        # 4x oversampling via polyphase resampling.
        try:
            upsampled = scipy_signal.resample_poly(
                audio.astype(np.float64), oversample, 1,
            )
            true_peak = float(np.max(np.abs(upsampled)))
            return max(sample_peak, true_peak)
        except Exception:
            return sample_peak

    # ═══════════════════════════════════════════════════════════════════
    #  Spectrum Analysis
    # ═══════════════════════════════════════════════════════════════════

    def get_spectrum(self, channel_id: int,
                     fft_size: int = 4096) -> np.ndarray:
        """Compute the magnitude spectrum for *channel_id*.

        Parameters
        ----------
        channel_id : int
            Channel to analyse.
        fft_size : int
            FFT length (default 4096).

        Returns
        -------
        np.ndarray
            Magnitude spectrum with ``fft_size // 2 + 1`` bins
            (0 Hz through Nyquist).
        """
        duration = fft_size / self.sample_rate
        audio = self.get_buffer(channel_id, duration)

        if len(audio) < fft_size:
            audio = np.pad(audio, (0, fft_size - len(audio)))

        window = np.hanning(fft_size)
        windowed = audio[-fft_size:] * window
        spectrum = np.abs(np.fft.rfft(windowed))
        return spectrum

    def get_spectrum_db(self, channel_id: int,
                        fft_size: int = 4096,
                        ref: float = 1.0) -> np.ndarray:
        """Magnitude spectrum in dB relative to *ref*.

        Silence bins are clipped to -120 dB.
        """
        mag = self.get_spectrum(channel_id, fft_size)
        mag_safe = np.maximum(mag, 1e-12)
        db = 20.0 * np.log10(mag_safe / ref)
        return np.clip(db, -120.0, None)

    def get_frequency_bins(self, fft_size: int = 4096) -> np.ndarray:
        """Return the frequency (Hz) for each FFT bin."""
        return np.fft.rfftfreq(fft_size, d=1.0 / self.sample_rate)

    # ═══════════════════════════════════════════════════════════════════
    #  Simple RMS / Peak helpers
    # ═══════════════════════════════════════════════════════════════════

    def get_peak_db(self, channel_id: int,
                    duration_sec: float = 0.1) -> float:
        """Sample-peak in dB for *channel_id*."""
        audio = self.get_buffer(channel_id, duration_sec)
        peak = float(np.max(np.abs(audio)))
        if peak < 1e-10:
            return -100.0
        return float(20.0 * np.log10(peak))

    def get_rms_db(self, channel_id: int,
                   duration_sec: float = 0.3) -> float:
        """RMS level in dB for *channel_id*."""
        audio = self.get_buffer(channel_id, duration_sec)
        rms = float(np.sqrt(np.mean(audio ** 2)))
        if rms < 1e-10:
            return -100.0
        return float(20.0 * np.log10(rms))

    # ═══════════════════════════════════════════════════════════════════
    #  Channel Statistics  (batch update)
    # ═══════════════════════════════════════════════════════════════════

    def update_channel_stats(self, channel_id: int) -> ChannelStats:
        """Recompute metering statistics for *channel_id*."""
        stats = self._channel_stats.get(channel_id)
        if stats is None:
            stats = ChannelStats(channel_id=channel_id)
            self._channel_stats[channel_id] = stats

        stats.peak_db = self.get_peak_db(channel_id, 0.1)
        stats.rms_db = self.get_rms_db(channel_id, 0.3)
        stats.lufs_momentary = self.get_lufs_momentary(channel_id)
        stats.lufs_short_term = self.get_lufs_short_term(channel_id)
        stats.is_clipping = stats.peak_db >= -0.1
        stats.last_update = time.time()
        return stats

    def get_channel_stats(self, channel_id: int) -> ChannelStats:
        """Return the most recently computed stats for *channel_id*."""
        return self._channel_stats.get(
            channel_id, ChannelStats(channel_id=channel_id),
        )

    def get_all_channel_stats(self) -> Dict[int, ChannelStats]:
        """Return a copy of all channel stats."""
        return dict(self._channel_stats)

    # ═══════════════════════════════════════════════════════════════════
    #  Test Signal Generators
    # ═══════════════════════════════════════════════════════════════════

    def start_test_generator(
        self, config: Optional[TestGeneratorConfig] = None,
    ) -> None:
        """Start a test signal generator (for offline / development use).

        Stops any currently active capture or generator first.
        """
        self.stop()

        self._test_generator_config = config or TestGeneratorConfig()
        cfg = self._test_generator_config

        # Pre-load file data if needed.
        if cfg.generator_type == "file":
            if not self._load_file(cfg.file_path):
                logger.warning(
                    "File load failed, falling back to silence generator"
                )
                cfg.generator_type = "silence"

        self._state = CaptureState.TEST_GENERATOR
        self._test_thread = threading.Thread(
            target=self._test_generator_loop,
            name="AudioCapture-TestGen",
            daemon=True,
        )
        self._test_thread.start()
        logger.info(
            "Test generator started: type=%s, channels=%d",
            cfg.generator_type, cfg.num_channels,
        )

    def _load_file(self, file_path: Optional[str]) -> bool:
        """Load an audio file for the file-playback generator.

        Attempts to use *soundfile* first, then falls back to the
        stdlib *wave* module for basic WAV support.
        """
        if file_path is None:
            logger.error("File playback requested but no file_path provided")
            return False

        if not os.path.isfile(file_path):
            logger.error("Audio file not found: %s", file_path)
            return False

        # Try soundfile (supports WAV, FLAC, OGG, etc.)
        if HAS_SOUNDFILE:
            try:
                data, sr = sf.read(file_path, dtype="float32")
                if data.ndim == 1:
                    data = data[:, np.newaxis]
                self._file_data = data
                self._file_sample_rate = sr
                self._file_channels = data.shape[1]
                self._file_pos = 0
                logger.info(
                    "Loaded audio file via soundfile: %s "
                    "(%d samples, %d ch, %d Hz)",
                    file_path, data.shape[0], self._file_channels, sr,
                )
                return True
            except Exception as exc:
                logger.warning("soundfile failed to load %s: %s", file_path, exc)

        # Fallback: stdlib wave (16-bit PCM WAV only).
        try:
            with wave.open(file_path, "rb") as wf:
                n_ch = wf.getnchannels()
                sw = wf.getsampwidth()
                sr = wf.getframerate()
                n_frames = wf.getnframes()
                raw = wf.readframes(n_frames)

            if sw == 2:
                # 16-bit signed PCM.
                samples = np.frombuffer(raw, dtype=np.int16).astype(np.float32)
                samples /= 32768.0
            elif sw == 3:
                # 24-bit PCM — unpack manually.
                n_samples = len(raw) // 3
                samples = np.zeros(n_samples, dtype=np.float32)
                for i in range(n_samples):
                    b = raw[i * 3:(i + 1) * 3]
                    val = struct.unpack_from("<i", b + b"\x00")[0] >> 8
                    samples[i] = val / 8388608.0
            elif sw == 4:
                samples = np.frombuffer(raw, dtype=np.int32).astype(np.float32)
                samples /= 2147483648.0
            else:
                logger.error("Unsupported sample width: %d bytes", sw)
                return False

            data = samples.reshape(-1, n_ch)
            self._file_data = data
            self._file_sample_rate = sr
            self._file_channels = n_ch
            self._file_pos = 0
            logger.info(
                "Loaded audio file via wave: %s "
                "(%d samples, %d ch, %d Hz)",
                file_path, data.shape[0], n_ch, sr,
            )
            return True
        except Exception as exc:
            logger.error("Failed to load audio file %s: %s", file_path, exc)
            return False

    # -- generator loop -----------------------------------------------------

    def _test_generator_loop(self) -> None:
        """Background thread producing test audio blocks."""
        cfg = self._test_generator_config
        if cfg is None:
            return

        phase: float = 0.0
        block_duration: float = self.block_size / self.sample_rate
        num_ch: int = min(cfg.num_channels, self.channels)

        # Pink-noise filter state (per channel).
        pink_states: Dict[int, np.ndarray] = {}

        while self._state == CaptureState.TEST_GENERATOR:
            t = np.arange(self.block_size, dtype=np.float64) / self.sample_rate

            for ch in range(num_ch):
                samples = self._generate_block(cfg, ch, t, phase, pink_states)

                # Write to ring buffer.
                buf = self._buffers.get(ch)
                if buf is not None:
                    buf.write(samples)

                # Notify subscribers.
                self._dispatch_subscribers(ch, samples)

            phase += block_duration
            # Sleep slightly less than real-time to avoid underruns.
            time.sleep(block_duration * 0.9)

    def _generate_block(
        self,
        cfg: TestGeneratorConfig,
        channel: int,
        t: np.ndarray,
        phase: float,
        pink_states: Dict[int, np.ndarray],
    ) -> np.ndarray:
        """Generate one block of audio for a single channel."""
        gen = cfg.generator_type

        if gen == "sine":
            return self._gen_sine(cfg, channel, t, phase)
        elif gen == "pink_noise":
            return self._gen_pink_noise(cfg, channel, pink_states)
        elif gen == "file":
            return self._gen_file(cfg, channel)
        else:
            # silence or unknown
            return np.zeros(self.block_size, dtype=np.float32)

    # -- sine ---------------------------------------------------------------

    def _gen_sine(
        self,
        cfg: TestGeneratorConfig,
        channel: int,
        t: np.ndarray,
        phase: float,
    ) -> np.ndarray:
        """Generate a sine-wave test block.

        When *channel_spread* is True, each channel gets a slight
        frequency offset for easier identification.
        """
        freq = cfg.frequency
        if cfg.channel_spread:
            # +1% per channel so ch0=440, ch1=444.4, ch2=448.8, etc.
            freq *= (1.0 + channel * 0.01)
        samples = cfg.amplitude * np.sin(
            2.0 * np.pi * freq * (t + phase)
        )
        return samples.astype(np.float32)

    # -- pink noise ---------------------------------------------------------

    def _gen_pink_noise(
        self,
        cfg: TestGeneratorConfig,
        channel: int,
        pink_states: Dict[int, np.ndarray],
    ) -> np.ndarray:
        """Generate pink noise (1/f) via IIR filter or Voss-McCartney.

        Uses Paul Kellet's refined 1/f approximation filter when scipy
        is available.  Each channel gets independent noise.
        """
        white = np.random.randn(self.block_size).astype(np.float64)

        # Paul Kellet's pinking filter coefficients.
        b = np.array([0.049922035, -0.095993537, 0.050612699, -0.004709510])
        a = np.array([1.0, -2.494956002, 2.017265875, -0.522189400])

        if HAS_SCIPY:
            # Maintain filter state between blocks for continuity.
            zi = pink_states.get(channel)
            if zi is None:
                zi = scipy_signal.lfilter_zi(b, a) * white[0]
            pink, zi = scipy_signal.lfilter(b, a, white, zi=zi)
            pink_states[channel] = zi
        else:
            # Stateless fallback — slight discontinuities at block
            # boundaries, but adequate for testing.
            pink = self._voss_mccartney(self.block_size)

        # Normalise to target amplitude.
        peak = np.max(np.abs(pink))
        if peak > 1e-10:
            pink = pink / peak * cfg.amplitude
        return pink.astype(np.float32)

    @staticmethod
    def _voss_mccartney(num_samples: int, num_rows: int = 16) -> np.ndarray:
        """Voss-McCartney algorithm for pink noise generation.

        Produces approximate 1/f noise without requiring scipy.
        """
        out = np.zeros(num_samples, dtype=np.float64)
        rows = np.random.randn(num_rows)
        running_sum = np.sum(rows)

        for i in range(num_samples):
            # Determine which row to update based on trailing zeros.
            idx = 0
            n = i
            while n > 0 and (n & 1) == 0 and idx < num_rows - 1:
                n >>= 1
                idx += 1
            running_sum -= rows[idx]
            rows[idx] = np.random.randn()
            running_sum += rows[idx]
            out[i] = running_sum + np.random.randn()

        return out

    # -- file playback ------------------------------------------------------

    def _gen_file(
        self,
        cfg: TestGeneratorConfig,
        channel: int,
    ) -> np.ndarray:
        """Generate a block from the loaded audio file."""
        if self._file_data is None:
            return np.zeros(self.block_size, dtype=np.float32)

        total_frames = self._file_data.shape[0]
        file_ch = min(channel, self._file_channels - 1)

        samples = np.zeros(self.block_size, dtype=np.float32)
        remaining = self.block_size
        write_pos = 0

        while remaining > 0:
            avail = total_frames - self._file_pos
            to_read = min(remaining, avail)

            if to_read > 0:
                chunk = self._file_data[
                    self._file_pos:self._file_pos + to_read, file_ch
                ]
                samples[write_pos:write_pos + to_read] = chunk
                write_pos += to_read
                remaining -= to_read
                self._file_pos += to_read

            if self._file_pos >= total_frames:
                if cfg.loop:
                    self._file_pos = 0
                else:
                    break  # End of file, rest stays zero-filled.

        # Apply amplitude scaling.
        samples *= cfg.amplitude
        return samples

    # ═══════════════════════════════════════════════════════════════════
    #  Device Discovery
    # ═══════════════════════════════════════════════════════════════════

    def get_available_devices(self) -> List[Dict[str, Any]]:
        """List available audio input devices.

        Returns an empty list if *sounddevice* is not installed.
        """
        if not HAS_SOUNDDEVICE:
            return []
        try:
            devices = sd.query_devices()
            result: List[Dict[str, Any]] = []
            for i, dev in enumerate(devices):
                if dev["max_input_channels"] > 0:
                    result.append({
                        "index": i,
                        "name": dev["name"],
                        "channels": dev["max_input_channels"],
                        "sample_rate": dev["default_samplerate"],
                    })
            return result
        except Exception as exc:
            logger.error("Error querying audio devices: %s", exc)
            return []

    def get_device_info(self) -> Optional[Dict[str, Any]]:
        """Return info about the currently configured device."""
        if not HAS_SOUNDDEVICE or self.device is None:
            return None
        try:
            dev = sd.query_devices(self.device)
            return {
                "index": self.device,
                "name": dev["name"],
                "channels": dev["max_input_channels"],
                "sample_rate": dev["default_samplerate"],
            }
        except Exception:
            return None

    # ═══════════════════════════════════════════════════════════════════
    #  Diagnostics / Status
    # ═══════════════════════════════════════════════════════════════════

    def get_status(self) -> Dict[str, Any]:
        """Return a diagnostic status dictionary."""
        active_subs = sum(
            len(cbs) for cbs in self._subscribers.values()
        ) + len(self._global_subscribers)

        buffers_with_data = sum(
            1 for buf in self._buffers.values()
            if buf.available_samples > 0
        )

        return {
            "state": self._state.value,
            "sample_rate": self.sample_rate,
            "channels": self.channels,
            "block_size": self.block_size,
            "device": self.device,
            "buffer_duration": self.buffer_duration,
            "active_subscribers": active_subs,
            "buffers_with_data": buffers_with_data,
            "xrun_count": self._xrun_count,
            "callback_errors": self._callback_errors,
            "has_sounddevice": HAS_SOUNDDEVICE,
            "has_scipy": HAS_SCIPY,
            "has_soundfile": HAS_SOUNDFILE,
            "test_generator": (
                self._test_generator_config.generator_type
                if self._test_generator_config else None
            ),
        }

    def __repr__(self) -> str:
        return (
            f"<AudioCapture state={self._state.value} "
            f"channels={self.channels} sr={self.sample_rate}>"
        )
