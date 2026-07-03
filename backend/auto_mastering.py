"""
Auto mastering module — applies mastering chain to the mix bus.
Uses matchering library for reference-based mastering when available.
"""
from __future__ import annotations

import numpy as np
import logging
from typing import Dict, Optional, Tuple
from dataclasses import dataclass

logger = logging.getLogger(__name__)

# pyloudnorm — accurate LUFS measurement (ITU-R BS.1770-4)
try:
    import pyloudnorm as pyln
    _PYLOUDNORM_AVAILABLE = True
except ImportError:
    _PYLOUDNORM_AVAILABLE = False


try:
    import matchering  # noqa: F401
    HAS_MATCHERING = True
except ImportError:
    HAS_MATCHERING = False

try:
    from scipy.io import wavfile
    HAS_SCIPY = True
except ImportError:
    wavfile = None
    HAS_SCIPY = False

@dataclass
class MasteringResult:
    """Result of mastering process."""
    audio: np.ndarray
    peak_db: float
    lufs: float
    gain_applied_db: float
    limiter_reduction_db: float
    eq_applied: bool
    success: bool
    error: Optional[str] = None

def measure_lufs(audio: np.ndarray, sample_rate: int) -> float:
    """Measure integrated LUFS using pyloudnorm (ITU-R BS.1770-4).

    Falls back to RMS-based approximation if pyloudnorm is unavailable.

    Args:
        audio: Audio array, shape (samples,) for mono or (samples, channels) for stereo.
        sample_rate: Sample rate in Hz.

    Returns:
        Integrated loudness in LUFS.
    """
    if _PYLOUDNORM_AVAILABLE:
        meter = pyln.Meter(sample_rate)
        # pyloudnorm expects (samples, channels) for stereo
        if audio.ndim == 1:
            # Mono — reshape to (samples, 1) for pyloudnorm
            return meter.integrated_loudness(audio.reshape(-1, 1))
        return meter.integrated_loudness(audio)
    else:
        # Fallback: RMS-based approximation (not accurate LUFS)
        rms = np.sqrt(np.mean(audio ** 2) + 1e-12)
        return float(20 * np.log10(rms))


def measure_true_peak(audio: np.ndarray, sample_rate: int) -> float:
    """Measure true peak with 4x oversampling.

    Args:
        audio: Audio array.
        sample_rate: Sample rate in Hz.

    Returns:
        True peak in dBTP.
    """
    from scipy.signal import resample_poly
    # 4x oversample
    if audio.ndim == 1:
        oversampled = resample_poly(audio, 4, 1)
    else:
        oversampled = np.column_stack(
            [resample_poly(audio[:, ch], 4, 1) for ch in range(audio.shape[1])]
        )
    peak_lin = np.max(np.abs(oversampled))
    if peak_lin > 0:
        return float(20 * np.log10(peak_lin))
    return -100.0


def normalize_lufs(
    audio: np.ndarray,
    sample_rate: int,
    target_lufs: float = -14.0,
    true_peak_limit: float = -1.0,
    max_gain_db: float = 12.0,
) -> Tuple[np.ndarray, float, float]:
    """Normalize audio to target LUFS with true peak limiting.

    Args:
        audio: Input audio array.
        sample_rate: Sample rate in Hz.
        target_lufs: Target integrated LUFS.
        true_peak_limit: Maximum true peak in dBTP.
        max_gain_db: Maximum gain adjustment allowed.

    Returns:
        Tuple of (normalized_audio, actual_lufs, gain_applied_db).
    """
    current_lufs = measure_lufs(audio, sample_rate)

    if current_lufs < -70.0:
        logger.warning("Audio too quiet for LUFS normalization (%.1f LUFS)", current_lufs)
        return audio.copy(), current_lufs, 0.0

    gain_db = target_lufs - current_lufs
    gain_db = max(-max_gain_db, min(max_gain_db, gain_db))

    if _PYLOUDNORM_AVAILABLE:
        # Use pyloudnorm for precise normalization
        if audio.ndim == 1:
            normalized = pyln.normalize.loudness(
                audio.reshape(-1, 1), current_lufs, target_lufs
            ).flatten()
        else:
            normalized = pyln.normalize.loudness(audio, current_lufs, target_lufs)
    else:
        gain_linear = 10 ** (gain_db / 20.0)
        normalized = audio * gain_linear

    # True peak limiting
    tp = measure_true_peak(normalized, sample_rate)
    if tp > true_peak_limit:
        reduction_db = tp - true_peak_limit
        reduction_linear = 10 ** (-reduction_db / 20.0)
        normalized = normalized * reduction_linear
        logger.info(
            "True peak limited: %.1f dBTP -> %.1f dBTP (reduced %.1f dB)",
            tp, true_peak_limit, reduction_db,
        )

    actual_lufs = measure_lufs(normalized, sample_rate)
    return normalized.astype(np.float32), actual_lufs, gain_db


class AutoMaster:
    """Automatic mastering processor."""

    def __init__(self, sample_rate: int = 48000, target_lufs: float = -14.0,
                 true_peak_limit: float = -1.0):
        self.sample_rate = sample_rate
        self.target_lufs = target_lufs
        self.true_peak_limit = true_peak_limit
        self._matchering_available = HAS_MATCHERING
        if self._matchering_available:
            logger.info("Matchering library available for reference-based mastering")
        else:
            logger.info("Matchering not available, using built-in mastering")

    def master(
        self,
        audio: np.ndarray,
        reference: Optional[np.ndarray] = None,
        sample_rate: Optional[int] = None,
    ):
        """Apply mastering chain to audio."""
        if sample_rate is not None:
            self.sample_rate = sample_rate

        if len(audio) == 0:
            return MasteringResult(audio=audio, peak_db=-100, lufs=-100,
                                   gain_applied_db=0, limiter_reduction_db=0,
                                   eq_applied=False, success=False, error="Empty audio")

        audio = self._normalize_audio_shape(audio.astype(np.float32))

        if reference is not None:
            reference = self._normalize_audio_shape(reference.astype(np.float32))
            if self._matchering_available:
                result = self._master_with_reference(audio, reference)
                if result.success and result.audio is not None:
                    return self._match_output_shape(result.audio.astype(np.float32), audio)
            return self._master_fallback(audio, reference, self.sample_rate)

        return self._builtin_master(audio)

    @staticmethod
    def _normalize_audio_shape(audio: np.ndarray) -> np.ndarray:
        """Normalize stereo arrays to (samples, channels) for processing."""
        arr = np.asarray(audio, dtype=np.float32)
        if arr.ndim <= 1:
            return arr
        if arr.ndim == 2 and arr.shape[0] <= 4 and arr.shape[1] > arr.shape[0]:
            return arr.T.astype(np.float32, copy=False)
        return arr.astype(np.float32, copy=False)

    @classmethod
    def _monitor_signal(cls, audio: np.ndarray) -> np.ndarray:
        """Return a mono monitor signal for loudness and sidechain decisions."""
        arr = cls._normalize_audio_shape(audio)
        if arr.ndim == 1:
            return arr.astype(np.float32, copy=False)
        return np.mean(arr, axis=1).astype(np.float32, copy=False)

    @classmethod
    def _match_output_shape(cls, audio: np.ndarray, template: np.ndarray) -> np.ndarray:
        """Keep reference-mastered output aligned with the caller's input layout."""
        arr = cls._normalize_audio_shape(audio)
        ref = cls._normalize_audio_shape(template)

        if ref.ndim == 2 and arr.ndim == 1:
            arr = np.column_stack([arr for _ in range(ref.shape[1])]).astype(np.float32)
        elif ref.ndim == 1 and arr.ndim == 2:
            arr = np.mean(arr, axis=1).astype(np.float32)

        if len(arr) != len(ref) and len(arr) > 0 and len(ref) > 0:
            src_pos = np.linspace(0.0, 1.0, num=len(arr), endpoint=False, dtype=np.float64)
            dst_pos = np.linspace(0.0, 1.0, num=len(ref), endpoint=False, dtype=np.float64)
            if arr.ndim == 1:
                arr = np.interp(dst_pos, src_pos, arr.astype(np.float64)).astype(np.float32)
            else:
                channels = [
                    np.interp(dst_pos, src_pos, arr[:, idx].astype(np.float64)).astype(np.float32)
                    for idx in range(arr.shape[1])
                ]
                arr = np.column_stack(channels).astype(np.float32)

        if ref.ndim == 2 and arr.ndim == 2 and arr.shape[1] != ref.shape[1]:
            if arr.shape[1] > ref.shape[1]:
                arr = arr[:, :ref.shape[1]]
            else:
                while arr.shape[1] < ref.shape[1]:
                    arr = np.column_stack([arr, arr[:, -1]]).astype(np.float32)

        return arr.astype(np.float32, copy=False)

    def _builtin_master(self, audio: np.ndarray) -> MasteringResult:
        """Built-in mastering chain: EQ -> Compression -> Limiting -> Normalization."""
        processed = self._normalize_audio_shape(audio).copy()

        # 1. Gentle high-pass filter at 30Hz
        processed = self._apply_hpf(processed, 30.0)

        # 2. Broadband compression
        processed, comp_reduction = self._apply_compression(
            processed, threshold_db=-18.0, ratio=2.0,
            attack_ms=30.0, release_ms=200.0
        )

        # 3. Loudness normalization
        current_rms = np.sqrt(np.mean(self._monitor_signal(processed) ** 2) + 1e-12)
        current_db = 20 * np.log10(current_rms)
        gain_db = self.target_lufs - current_db
        gain_db = max(-12.0, min(12.0, gain_db))
        gain_linear = 10 ** (gain_db / 20.0)
        processed = processed * gain_linear

        # 4. Brick-wall limiter
        processed, limiter_reduction = self._apply_limiter(processed, self.true_peak_limit)

        # Final measurements
        peak_db = float(20 * np.log10(np.max(np.abs(processed)) + 1e-10))
        rms_db = float(20 * np.log10(np.sqrt(np.mean(processed ** 2)) + 1e-10))

        return MasteringResult(
            audio=processed, peak_db=peak_db, lufs=rms_db,
            gain_applied_db=gain_db, limiter_reduction_db=limiter_reduction,
            eq_applied=True, success=True
        )

    @staticmethod
    def _estimate_lufs(audio: np.ndarray) -> float:
        """RMS-based LUFS approximation used by compatibility tests."""
        if audio.size == 0:
            return -100.0
        if audio.ndim > 1:
            audio = np.mean(audio, axis=-1)
        rms = np.sqrt(np.mean(np.square(audio.astype(np.float64))) + 1e-12)
        return float(20.0 * np.log10(rms + 1e-12))

    def _limit(self, audio: np.ndarray) -> np.ndarray:
        """Compatibility helper returning only limited audio."""
        limited, _ = self._apply_limiter(audio.astype(np.float32), self.true_peak_limit)
        return limited.astype(np.float32)

    def _match_target_loudness(self, audio: np.ndarray, target_lufs: Optional[float] = None) -> np.ndarray:
        """Bring audio toward the configured loudness target while honoring the peak limit."""
        working = self._normalize_audio_shape(audio).astype(np.float32, copy=True)
        desired_lufs = float(self.target_lufs if target_lufs is None else target_lufs)

        current_lufs = self._estimate_lufs(working)
        if np.isfinite(current_lufs):
            gain_db = np.clip(desired_lufs - current_lufs, -18.0, 18.0)
            working *= np.float32(10.0 ** (gain_db / 20.0))

        working = self._limit(working)

        post_limit_lufs = self._estimate_lufs(working)
        if np.isfinite(post_limit_lufs):
            correction_db = np.clip(desired_lufs - post_limit_lufs, -6.0, 6.0)
            if abs(correction_db) >= 0.1:
                working *= np.float32(10.0 ** (correction_db / 20.0))
                working = self._limit(working)

        return working.astype(np.float32, copy=False)

    def _master_fallback(self, audio: np.ndarray, reference: np.ndarray, sample_rate: int) -> np.ndarray:
        """Reference-guided fallback mastering without external dependencies."""
        self.sample_rate = sample_rate
        working = self._normalize_audio_shape(audio).astype(np.float32, copy=True)
        reference = self._normalize_audio_shape(reference).astype(np.float32, copy=False)

        working = self._match_target_loudness(working, self.target_lufs)

        working = self._apply_eq_match(working.astype(np.float64), reference.astype(np.float64), sample_rate).astype(np.float32)
        return self._match_target_loudness(working, self.target_lufs)

    def _apply_eq_match(self, audio: np.ndarray, reference: np.ndarray, sample_rate: int) -> np.ndarray:
        """Very lightweight, broad-band spectral tilt matching.

        The fallback is used when a full reference-mastering backend is not
        available. It must be conservative: keep the caller's full program
        length, avoid narrow FFT-bin matching, and never turn short reference
        excerpts into a hard edit of the mix.
        """
        if not HAS_SCIPY:
            return audio

        original_dtype = np.asarray(audio).dtype
        audio = self._normalize_audio_shape(audio)
        reference = self._normalize_audio_shape(reference)

        min_len = min(len(audio), len(reference))
        if min_len == 0 or len(audio) == 0:
            return audio

        analysis_audio = self._monitor_signal(audio[:min_len]).astype(np.float64, copy=False)
        analysis_ref = self._monitor_signal(reference[:min_len]).astype(np.float64, copy=False)
        if not np.any(np.abs(analysis_audio) > 1e-9) or not np.any(np.abs(analysis_ref) > 1e-9):
            return audio.astype(original_dtype, copy=False)

        window = np.hanning(min_len).astype(np.float64)
        if not np.any(window):
            window = np.ones(min_len, dtype=np.float64)
        freqs = np.fft.rfftfreq(min_len, d=1.0 / float(sample_rate))
        audio_mag = np.abs(np.fft.rfft(analysis_audio * window)) + 1e-10
        ref_mag = np.abs(np.fft.rfft(analysis_ref * window)) + 1e-10

        nyquist = float(sample_rate) * 0.5
        bands = [
            (30.0, 60.0),
            (60.0, 120.0),
            (120.0, 250.0),
            (250.0, 500.0),
            (500.0, 1000.0),
            (1000.0, 2000.0),
            (2000.0, 4000.0),
            (4000.0, 8000.0),
            (8000.0, min(12000.0, nyquist * 0.95)),
            (12000.0, min(18000.0, nyquist * 0.98)),
        ]
        centers: list[float] = []
        gains_db: list[float] = []
        for low_hz, high_hz in bands:
            if high_hz <= low_hz or low_hz >= nyquist:
                continue
            mask = (freqs >= low_hz) & (freqs < high_hz)
            if int(np.count_nonzero(mask)) < 2:
                continue
            audio_db = 20.0 * np.log10(float(np.sqrt(np.mean(audio_mag[mask] ** 2))) + 1e-10)
            ref_db = 20.0 * np.log10(float(np.sqrt(np.mean(ref_mag[mask] ** 2))) + 1e-10)
            gain_db = float(np.clip(ref_db - audio_db, -2.5, 2.5))

            # High-frequency boosts from short previews are a common way to
            # manufacture hiss. Keep air matching mostly subtractive.
            if low_hz >= 8000.0:
                gain_db = min(gain_db, 0.75)
            elif low_hz >= 4000.0:
                gain_db = min(gain_db, 1.25)

            centers.append(float(np.sqrt(low_hz * high_hz)))
            gains_db.append(gain_db)

        if not centers:
            return audio.astype(original_dtype, copy=False)

        interp_freqs = np.fft.rfftfreq(len(audio), d=1.0 / float(sample_rate))
        safe_freqs = np.maximum(interp_freqs, 20.0)
        x = np.log2(np.asarray([20.0, *centers, nyquist], dtype=np.float64))
        y = np.asarray([0.0, *gains_db, 0.0], dtype=np.float64)
        curve_db = np.interp(np.log2(safe_freqs), x, y).astype(np.float64)
        curve_db[interp_freqs < 25.0] = 0.0
        if nyquist > 16000.0:
            curve_db[interp_freqs > 16000.0] = np.minimum(curve_db[interp_freqs > 16000.0], 0.0)
        ratio = (10.0 ** (curve_db / 20.0)).astype(np.float64)

        def match_channel(channel_audio: np.ndarray) -> np.ndarray:
            spectrum = np.fft.rfft(channel_audio.astype(np.float64, copy=False))
            matched = np.fft.irfft(spectrum * ratio, n=len(channel_audio))
            return matched.astype(original_dtype, copy=False)

        if audio.ndim == 1:
            return match_channel(audio)

        matched_channels = [match_channel(audio[:, channel_idx]) for channel_idx in range(audio.shape[1])]
        return np.column_stack(matched_channels).astype(original_dtype, copy=False)

    def _write_wav(self, path: str, audio: np.ndarray, sample_rate: int):
        """Write PCM16 WAV when scipy is available."""
        if not HAS_SCIPY:
            raise RuntimeError("scipy not available")
        scaled = np.clip(audio, -1.0, 1.0)
        wavfile.write(path, sample_rate, (scaled * 32767.0).astype(np.int16))

    def _read_wav(self, path: str) -> np.ndarray:
        """Read WAV as float32 mono/stereo normalized audio."""
        if not HAS_SCIPY:
            raise RuntimeError("scipy not available")
        sample_rate, data = wavfile.read(path)
        self.sample_rate = sample_rate
        if np.issubdtype(data.dtype, np.integer):
            max_val = np.iinfo(data.dtype).max
            audio = data.astype(np.float32) / max_val
        else:
            audio = data.astype(np.float32)
        return audio

    def _master_with_reference(self, audio: np.ndarray, reference: np.ndarray) -> MasteringResult:
        """Master using matchering library with a reference track."""
        try:
            import matchering as mg
            import tempfile
            import os

            try:
                import soundfile as sf
            except ImportError:
                raise RuntimeError("soundfile not available for matchering I/O")

            with tempfile.TemporaryDirectory() as tmpdir:
                target_path = os.path.join(tmpdir, 'target.wav')
                ref_path = os.path.join(tmpdir, 'reference.wav')
                output_path = os.path.join(tmpdir, 'mastered.wav')

                sf.write(target_path, audio, self.sample_rate)
                sf.write(ref_path, reference, self.sample_rate)

                mg.process(
                    target=target_path,
                    reference=ref_path,
                    results=[mg.pcm16(output_path)],
                )

                mastered, _ = sf.read(output_path, dtype='float32')

                peak_db = float(20 * np.log10(np.max(np.abs(mastered)) + 1e-10))
                rms_db = float(20 * np.log10(np.sqrt(np.mean(mastered ** 2)) + 1e-10))
                gain_db = rms_db - 20 * np.log10(np.sqrt(np.mean(audio ** 2)) + 1e-10)

                return MasteringResult(
                    audio=mastered, peak_db=peak_db, lufs=rms_db,
                    gain_applied_db=float(gain_db), limiter_reduction_db=0,
                    eq_applied=True, success=True
                )
        except Exception as e:
            logger.error(f"Matchering error: {e}, falling back to reference-guided fallback")
            return MasteringResult(
                audio=self._normalize_audio_shape(audio).astype(np.float32, copy=False),
                peak_db=float(20 * np.log10(np.max(np.abs(audio)) + 1e-10)) if np.size(audio) else -100.0,
                lufs=self._estimate_lufs(audio),
                gain_applied_db=0.0,
                limiter_reduction_db=0.0,
                eq_applied=False,
                success=False,
                error=str(e),
            )

    def _apply_hpf(self, audio: np.ndarray, cutoff_hz: float) -> np.ndarray:
        """Apply simple high-pass filter."""
        try:
            from scipy.signal import butter, sosfilt
            sos = butter(2, cutoff_hz, btype='high', fs=self.sample_rate, output='sos')
            return sosfilt(sos, self._normalize_audio_shape(audio), axis=0).astype(np.float32)
        except ImportError:
            return audio

    def _apply_compression(self, audio: np.ndarray, threshold_db: float,
                          ratio: float, attack_ms: float, release_ms: float) -> Tuple[np.ndarray, float]:
        """Apply dynamic compression."""
        eps = 1e-10
        attack_coeff = np.exp(-1.0 / (attack_ms * self.sample_rate / 1000))
        release_coeff = np.exp(-1.0 / (release_ms * self.sample_rate / 1000))

        threshold_lin = 10 ** (threshold_db / 20.0)
        output = np.copy(self._normalize_audio_shape(audio))
        monitor = np.abs(self._monitor_signal(output))
        envelope = 0.0
        max_reduction = 0.0

        for i in range(len(output)):
            level = float(monitor[i])
            if level > envelope:
                envelope = attack_coeff * envelope + (1 - attack_coeff) * level
            else:
                envelope = release_coeff * envelope + (1 - release_coeff) * level

            if envelope > threshold_lin:
                gain_reduction = (envelope / threshold_lin) ** (1 - 1/ratio)
                output[i] = output[i] / (gain_reduction + eps)
                reduction_db = 20 * np.log10(gain_reduction + eps)
                max_reduction = max(max_reduction, reduction_db)

        return output, max_reduction

    def _apply_true_peak_limiter(self, audio: np.ndarray, ceiling_dbtp: float) -> Tuple[np.ndarray, float]:
        """Oversampled look-ahead true-peak limiter for mastering.

        Key design (similar to Pro-L, L2, etc.):
        1. Upsample 4x to detect inter-sample peaks (true peaks)
        2. Compute gain reduction curve at oversampled rate
        3. Look-ahead (5ms) to anticipate peaks before they arrive
        4. Smooth gain with instant attack + exponential release
        5. Downsample gain curve back to original rate
        6. Apply to original signal

        This guarantees true peak compliance — inter-sample peaks that
        escape sample-level detection are caught by the 4x oversampled
        detector and pre-emptively reduced via look-ahead.
        """
        from scipy.signal import resample_poly

        oversample_factor = 4
        lookahead_ms = 5.0
        release_ms = 50.0

        os_rate = self.sample_rate * oversample_factor
        lookahead_os = int(os_rate * lookahead_ms / 1000)
        release_coeff = np.exp(-1.0 / (release_ms * os_rate / 1000))
        ceiling_lin = 10 ** (ceiling_dbtp / 20.0)

        # Step 1: Upsample 4x for true peak detection
        if audio.ndim == 2:
            oversampled = np.column_stack(
                [resample_poly(audio[:, ch], oversample_factor, 1)
                 for ch in range(audio.shape[1])]
            )
            os_levels = np.max(np.abs(oversampled), axis=1)
        else:
            oversampled = resample_poly(audio, oversample_factor, 1)
            os_levels = np.abs(oversampled)

        num_os_samples = len(os_levels)

        # Step 2: Required gain reduction at each oversampled position
        gain_required = np.ones(num_os_samples, dtype=np.float64)
        over_mask = os_levels > ceiling_lin
        gain_required[over_mask] = ceiling_lin / (os_levels[over_mask] + 1e-10)

        # Step 3: Look-ahead — minimum gain within upcoming window
        # Use rolling minimum via a deque for O(n) complexity
        from collections import deque
        gain_lookahead = np.ones(num_os_samples, dtype=np.float64)
        dq = deque()  # stores indices of potential minimums

        for i in range(num_os_samples):
            # Remove elements outside the look-ahead window
            while dq and dq[0] < i:
                dq.popleft()

            # Add current + look-ahead position
            j = min(i + lookahead_os, num_os_samples - 1)
            while dq and gain_required[j] <= gain_required[dq[-1]]:
                dq.pop()
            dq.append(j)

            gain_lookahead[i] = gain_required[dq[0]]

        # Step 4: Smooth gain curve (instant attack, exponential release)
        gain_smooth_os = np.ones(num_os_samples, dtype=np.float64)
        current_gain = 1.0
        for i in range(num_os_samples):
            target = gain_lookahead[i]
            if target < current_gain:
                current_gain = target  # Instant attack
            else:
                current_gain = release_coeff * current_gain + (1 - release_coeff) * target
            gain_smooth_os[i] = current_gain

        # Step 5: Downsample gain curve back to original rate
        # Take every Nth sample (phase-aligned with original)
        num_orig = len(audio) if audio.ndim == 1 else audio.shape[0]
        gain_smooth = resample_poly(gain_smooth_os, 1, oversample_factor)[:num_orig]

        # Ensure no gain increase (safety clamp)
        gain_smooth = np.minimum(gain_smooth, 1.0)

        # Step 6: Apply gain curve to original signal
        if audio.ndim == 2:
            output = audio * gain_smooth[:, np.newaxis]
        else:
            output = audio * gain_smooth

        max_reduction_db = float(-20 * np.log10(np.min(gain_smooth) + 1e-10))

        if max_reduction_db > 0.1:
            limited_pct = 100 * np.sum(gain_smooth < 0.999) / len(gain_smooth)
            logger.info(
                "Oversampled look-ahead limiter (4x): ceiling %.1f dBTP, "
                "max reduction %.1f dB, limited %.1f%% of samples",
                ceiling_dbtp, max_reduction_db, limited_pct,
            )

        return output.astype(np.float32), max_reduction_db

    def _apply_limiter(self, audio: np.ndarray, ceiling_db: float) -> Tuple[np.ndarray, float]:
        """Apply brick-wall limiter."""
        audio = self._normalize_audio_shape(audio)
        ceiling_lin = 10 ** (ceiling_db / 20.0)
        peak = np.max(np.abs(audio))
        reduction_db = 0.0

        if peak > ceiling_lin:
            gain = ceiling_lin / peak
            audio = audio * gain
            reduction_db = float(20 * np.log10(gain))

        return audio, abs(reduction_db)
