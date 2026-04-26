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
