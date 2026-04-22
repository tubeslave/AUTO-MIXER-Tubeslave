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

    def _master_fallback(self, audio: np.ndarray, reference: np.ndarray, sample_rate: int) -> np.ndarray:
        """Reference-guided fallback mastering without external dependencies."""
        self.sample_rate = sample_rate
        working = self._normalize_audio_shape(audio).astype(np.float32, copy=True)
        reference = self._normalize_audio_shape(reference).astype(np.float32, copy=False)

        target_lufs = self._estimate_lufs(reference)
        current_lufs = self._estimate_lufs(working)
        gain_db = np.clip(target_lufs - current_lufs, -12.0, 12.0)
        working *= np.float32(10.0 ** (gain_db / 20.0))

        working = self._apply_eq_match(working.astype(np.float64), reference.astype(np.float64), sample_rate).astype(np.float32)
        post_eq_lufs = self._estimate_lufs(working)
        correction_db = np.clip(target_lufs - post_eq_lufs, -18.0, 18.0)
        working *= np.float32(10.0 ** (correction_db / 20.0))
        return self._limit(working)

    def _apply_eq_match(self, audio: np.ndarray, reference: np.ndarray, sample_rate: int) -> np.ndarray:
        """Very lightweight spectral tilt matching."""
        if not HAS_SCIPY:
            return audio

        audio = self._normalize_audio_shape(audio)
        reference = self._normalize_audio_shape(reference)

        min_len = min(len(audio), len(reference))
        if min_len == 0:
            return audio

        audio = audio[:min_len]
        reference = reference[:min_len]

        ref_monitor = self._monitor_signal(reference)
        audio_monitor = self._monitor_signal(audio)
        fft_ref = np.fft.rfft(ref_monitor)
        fft_audio_monitor = np.fft.rfft(audio_monitor)
        mag_audio = np.abs(fft_audio_monitor) + 1e-9
        mag_ref = np.abs(fft_ref) + 1e-9
        ratio = np.clip(mag_ref / mag_audio, 0.5, 2.0)

        if audio.ndim == 1:
            matched = np.fft.irfft(np.fft.rfft(audio) * ratio, n=min_len)
            return matched.astype(audio.dtype, copy=False)

        matched_channels = []
        for channel_idx in range(audio.shape[1]):
            fft_audio = np.fft.rfft(audio[:, channel_idx])
            matched_channel = np.fft.irfft(fft_audio * ratio, n=min_len)
            matched_channels.append(matched_channel.astype(audio.dtype, copy=False))
        return np.column_stack(matched_channels).astype(audio.dtype, copy=False)

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
