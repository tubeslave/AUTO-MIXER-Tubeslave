"""Conservative automatic mastering with standards-aware metering.

The built-in path intentionally stays simple: high-pass, gentle broadband
compression, programme-loudness normalization and reconstructed-peak limiting.
Reference matching remains broad-band and bounded.  No live-console control is
implemented here.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)
EPS = 1e-12

try:
    import matchering  # noqa: F401
    HAS_MATCHERING = True
except ImportError:
    HAS_MATCHERING = False

try:
    from scipy.io import wavfile
    from scipy.signal import butter, resample_poly, sosfilt
    HAS_SCIPY = True
except ImportError:
    wavfile = None
    butter = resample_poly = sosfilt = None
    HAS_SCIPY = False


@dataclass
class MasteringResult:
    audio: np.ndarray
    peak_db: float
    lufs: float
    gain_applied_db: float
    limiter_reduction_db: float
    eq_applied: bool
    success: bool
    error: Optional[str] = None


class AutoMaster:
    def __init__(
        self,
        sample_rate: int = 48000,
        target_lufs: float = -14.0,
        true_peak_limit: float = -1.0,
    ):
        self.sample_rate = int(sample_rate)
        self.target_lufs = float(target_lufs)
        self.true_peak_limit = float(true_peak_limit)
        self._matchering_available = HAS_MATCHERING

    @staticmethod
    def _finite(audio: np.ndarray) -> np.ndarray:
        return np.nan_to_num(np.asarray(audio, dtype=np.float32), nan=0.0, posinf=0.0, neginf=0.0)

    @staticmethod
    def _normalize_audio_shape(audio: np.ndarray) -> np.ndarray:
        arr = AutoMaster._finite(audio)
        if arr.ndim <= 1:
            return arr.reshape(-1) if arr.ndim else arr.reshape(1)
        if arr.ndim == 2 and arr.shape[0] <= 8 and arr.shape[1] > arr.shape[0] * 4:
            return arr.T.astype(np.float32, copy=False)
        if arr.ndim > 2:
            return arr.reshape(arr.shape[0], -1).astype(np.float32, copy=False)
        return arr.astype(np.float32, copy=False)

    @classmethod
    def _monitor_signal(cls, audio: np.ndarray) -> np.ndarray:
        arr = cls._normalize_audio_shape(audio)
        if arr.ndim == 1:
            return arr
        return np.mean(arr, axis=1, dtype=np.float32)

    @classmethod
    def _match_output_shape(cls, audio: np.ndarray, template: np.ndarray) -> np.ndarray:
        arr = cls._normalize_audio_shape(audio)
        ref = cls._normalize_audio_shape(template)
        if ref.ndim == 2 and arr.ndim == 1:
            arr = np.column_stack([arr for _ in range(ref.shape[1])]).astype(np.float32)
        elif ref.ndim == 1 and arr.ndim == 2:
            arr = np.mean(arr, axis=1).astype(np.float32)
        if len(arr) != len(ref) and len(arr) and len(ref):
            src = np.linspace(0.0, 1.0, len(arr), endpoint=False)
            dst = np.linspace(0.0, 1.0, len(ref), endpoint=False)
            if arr.ndim == 1:
                arr = np.interp(dst, src, arr).astype(np.float32)
            else:
                arr = np.column_stack(
                    [np.interp(dst, src, arr[:, idx]) for idx in range(arr.shape[1])]
                ).astype(np.float32)
        if ref.ndim == 2 and arr.ndim == 2 and arr.shape[1] != ref.shape[1]:
            if arr.shape[1] > ref.shape[1]:
                arr = arr[:, : ref.shape[1]]
            elif arr.shape[1] > 0:
                while arr.shape[1] < ref.shape[1]:
                    arr = np.column_stack([arr, arr[:, -1]])
        return arr.astype(np.float32, copy=False)

    @staticmethod
    def _fallback_k_weight(audio: np.ndarray, sample_rate: int) -> np.ndarray:
        """K-weighting fallback using the project's BS.1770 filter implementation."""
        arr = AutoMaster._normalize_audio_shape(audio)
        channels = arr[:, None] if arr.ndim == 1 else arr
        try:
            from lufs_gain_staging import KWeightingFilter

            weighted = np.empty_like(channels, dtype=np.float64)
            for ch in range(channels.shape[1]):
                weighted[:, ch] = KWeightingFilter(sample_rate).process(channels[:, ch])
            return np.nan_to_num(weighted)
        except Exception:
            return channels.astype(np.float64, copy=False)

    @staticmethod
    def _estimate_lufs(audio: np.ndarray, sample_rate: int = 48000) -> float:
        """Estimate integrated programme loudness in LUFS.

        ``pyloudnorm`` is used when available.  The fallback applies the
        project's K-weighting filters and BS.1770 absolute/relative gating.
        """
        arr = AutoMaster._normalize_audio_shape(audio)
        if arr.size == 0 or not np.any(np.abs(arr) > 1e-12):
            return -100.0
        try:
            import pyloudnorm as pyln

            if len(arr) >= int(0.4 * sample_rate):
                value = float(pyln.Meter(sample_rate, block_size=0.4).integrated_loudness(arr))
                if np.isfinite(value):
                    return value
        except Exception:
            pass

        weighted = AutoMaster._fallback_k_weight(arr, sample_rate)
        block = max(1, int(round(0.4 * sample_rate)))
        hop = max(1, block // 4)
        values = []
        if weighted.shape[0] < block:
            padded = np.zeros((block, weighted.shape[1]), dtype=np.float64)
            padded[: weighted.shape[0]] = weighted
            weighted = padded
        starts = list(range(0, weighted.shape[0] - block + 1, hop)) or [0]
        for start in starts:
            chunk = weighted[start : start + block]
            energy = float(np.sum(np.mean(np.square(chunk), axis=0)))
            if energy > EPS:
                values.append(float(-0.691 + 10.0 * np.log10(energy)))
        finite = np.asarray([v for v in values if np.isfinite(v) and v > -70.0], dtype=np.float64)
        if finite.size == 0:
            return -100.0
        energies = np.power(10.0, (finite + 0.691) / 10.0)
        ungated = float(-0.691 + 10.0 * np.log10(max(float(np.mean(energies)), EPS)))
        kept = finite[finite >= ungated - 10.0]
        if kept.size == 0:
            return ungated
        energies = np.power(10.0, (kept + 0.691) / 10.0)
        return float(-0.691 + 10.0 * np.log10(max(float(np.mean(energies)), EPS)))

    @staticmethod
    def _true_peak_db(audio: np.ndarray, oversample: int = 4) -> float:
        arr = AutoMaster._normalize_audio_shape(audio)
        if arr.size == 0:
            return -100.0
        if HAS_SCIPY and resample_poly is not None:
            reconstructed = resample_poly(arr, int(oversample), 1, axis=0)
            peak = float(np.max(np.abs(reconstructed)))
        else:
            peak = float(np.max(np.abs(arr)))
        if peak <= EPS:
            return -100.0
        return float(20.0 * np.log10(peak))

    def master(
        self,
        audio: np.ndarray,
        reference: Optional[np.ndarray] = None,
        sample_rate: Optional[int] = None,
    ):
        if sample_rate is not None:
            self.sample_rate = int(sample_rate)
        if len(audio) == 0:
            return MasteringResult(
                audio=np.asarray(audio), peak_db=-100.0, lufs=-100.0,
                gain_applied_db=0.0, limiter_reduction_db=0.0,
                eq_applied=False, success=False, error="Empty audio",
            )
        source = self._normalize_audio_shape(audio)
        if reference is None:
            return self._builtin_master(source)
        ref = self._normalize_audio_shape(reference)
        if self._matchering_available:
            result = self._master_with_reference(source, ref)
            if result.success and result.audio is not None:
                return self._match_output_shape(result.audio, source)
        return self._master_fallback(source, ref, self.sample_rate)

    def _builtin_master(self, audio: np.ndarray) -> MasteringResult:
        processed = self._apply_hpf(self._normalize_audio_shape(audio), 30.0)
        processed, _ = self._apply_compression(
            processed, threshold_db=-18.0, ratio=2.0, attack_ms=30.0, release_ms=200.0
        )
        before = self._estimate_lufs(processed, self.sample_rate)
        gain_db = float(np.clip(self.target_lufs - before, -12.0, 12.0)) if np.isfinite(before) else 0.0
        processed = processed * np.float32(10.0 ** (gain_db / 20.0))
        processed, limiter_reduction = self._apply_limiter(processed, self.true_peak_limit)
        return MasteringResult(
            audio=processed.astype(np.float32),
            peak_db=round(self._true_peak_db(processed), 4),
            lufs=round(self._estimate_lufs(processed, self.sample_rate), 4),
            gain_applied_db=gain_db,
            limiter_reduction_db=float(limiter_reduction),
            eq_applied=True,
            success=True,
        )

    def _limit(self, audio: np.ndarray) -> np.ndarray:
        limited, _ = self._apply_limiter(self._normalize_audio_shape(audio), self.true_peak_limit)
        return limited.astype(np.float32)

    def _match_target_loudness(self, audio: np.ndarray, target_lufs: Optional[float] = None) -> np.ndarray:
        working = self._normalize_audio_shape(audio).astype(np.float32, copy=True)
        target = self.target_lufs if target_lufs is None else float(target_lufs)
        for max_gain in (18.0, 6.0):
            current = self._estimate_lufs(working, self.sample_rate)
            if not np.isfinite(current) or current <= -99.0:
                break
            delta = float(np.clip(target - current, -max_gain, max_gain))
            if abs(delta) < 0.05:
                break
            working *= np.float32(10.0 ** (delta / 20.0))
            working = self._limit(working)
        return working.astype(np.float32, copy=False)

    def _master_fallback(self, audio: np.ndarray, reference: np.ndarray, sample_rate: int) -> np.ndarray:
        self.sample_rate = int(sample_rate)
        working = self._match_target_loudness(audio, self.target_lufs)
        working = self._apply_eq_match(working.astype(np.float64), reference.astype(np.float64), self.sample_rate)
        return self._match_target_loudness(working, self.target_lufs)

    def _apply_eq_match(self, audio: np.ndarray, reference: np.ndarray, sample_rate: int) -> np.ndarray:
        """Conservative broad-band reference tilt match, bounded to ±2.5 dB."""
        original_dtype = np.asarray(audio).dtype
        source = self._normalize_audio_shape(audio).astype(np.float64)
        ref = self._normalize_audio_shape(reference).astype(np.float64)
        if not HAS_SCIPY or len(source) == 0 or len(ref) == 0:
            return source.astype(original_dtype, copy=False)
        n = min(len(source), len(ref))
        src_mono = self._monitor_signal(source[:n]).astype(np.float64)
        ref_mono = self._monitor_signal(ref[:n]).astype(np.float64)
        if not np.any(np.abs(src_mono) > 1e-9) or not np.any(np.abs(ref_mono) > 1e-9):
            return source.astype(original_dtype, copy=False)
        window = np.hanning(n)
        freqs = np.fft.rfftfreq(n, 1.0 / sample_rate)
        src_mag = np.abs(np.fft.rfft(src_mono * window)) + EPS
        ref_mag = np.abs(np.fft.rfft(ref_mono * window)) + EPS
        nyquist = sample_rate * 0.5
        bands = [
            (30, 60), (60, 120), (120, 250), (250, 500), (500, 1000),
            (1000, 2000), (2000, 4000), (4000, 8000),
            (8000, min(12000, nyquist * 0.95)), (12000, min(18000, nyquist * 0.98)),
        ]
        centers, gains = [], []
        for low, high in bands:
            if high <= low or low >= nyquist:
                continue
            mask = (freqs >= low) & (freqs < high)
            if np.count_nonzero(mask) < 2:
                continue
            src_db = 10.0 * np.log10(float(np.mean(src_mag[mask] ** 2)) + EPS)
            ref_db = 10.0 * np.log10(float(np.mean(ref_mag[mask] ** 2)) + EPS)
            gain = float(np.clip(ref_db - src_db, -2.5, 2.5))
            if low >= 8000:
                gain = min(gain, 0.75)
            elif low >= 4000:
                gain = min(gain, 1.25)
            centers.append(float(np.sqrt(low * high)))
            gains.append(gain)
        if not centers:
            return source.astype(original_dtype, copy=False)
        out_freqs = np.fft.rfftfreq(len(source), 1.0 / sample_rate)
        safe = np.maximum(out_freqs, 20.0)
        x = np.log2(np.asarray([20.0, *centers, nyquist]))
        y = np.asarray([0.0, *gains, 0.0])
        curve = np.interp(np.log2(safe), x, y)
        curve[out_freqs < 25.0] = 0.0
        if nyquist > 16000:
            curve[out_freqs > 16000] = np.minimum(curve[out_freqs > 16000], 0.0)
        ratio = 10.0 ** (curve / 20.0)
        def apply_channel(channel: np.ndarray) -> np.ndarray:
            return np.fft.irfft(np.fft.rfft(channel) * ratio, n=len(channel))
        if source.ndim == 1:
            return apply_channel(source).astype(original_dtype)
        return np.column_stack([apply_channel(source[:, ch]) for ch in range(source.shape[1])]).astype(original_dtype)

    def _master_with_reference(self, audio: np.ndarray, reference: np.ndarray) -> MasteringResult:
        try:
            import matchering as mg
            import os
            import tempfile
            import soundfile as sf

            with tempfile.TemporaryDirectory() as tmpdir:
                target_path = os.path.join(tmpdir, "target.wav")
                ref_path = os.path.join(tmpdir, "reference.wav")
                output_path = os.path.join(tmpdir, "mastered.wav")
                sf.write(target_path, audio, self.sample_rate)
                sf.write(ref_path, reference, self.sample_rate)
                mg.process(target=target_path, reference=ref_path, results=[mg.pcm16(output_path)])
                mastered, _ = sf.read(output_path, dtype="float32")
            mastered = self._normalize_audio_shape(mastered)
            input_lufs = self._estimate_lufs(audio, self.sample_rate)
            output_lufs = self._estimate_lufs(mastered, self.sample_rate)
            return MasteringResult(
                audio=mastered,
                peak_db=self._true_peak_db(mastered),
                lufs=output_lufs,
                gain_applied_db=float(output_lufs - input_lufs) if input_lufs > -99 else 0.0,
                limiter_reduction_db=0.0,
                eq_applied=True,
                success=True,
            )
        except Exception as exc:
            logger.error("Matchering error: %s", exc)
            return MasteringResult(
                audio=self._normalize_audio_shape(audio),
                peak_db=self._true_peak_db(audio),
                lufs=self._estimate_lufs(audio, self.sample_rate),
                gain_applied_db=0.0,
                limiter_reduction_db=0.0,
                eq_applied=False,
                success=False,
                error=str(exc),
            )

    def _apply_hpf(self, audio: np.ndarray, cutoff_hz: float) -> np.ndarray:
        arr = self._normalize_audio_shape(audio)
        if not HAS_SCIPY or butter is None or sosfilt is None:
            return arr
        if cutoff_hz <= 0.0 or cutoff_hz >= self.sample_rate * 0.5:
            return arr
        sos = butter(2, cutoff_hz, btype="high", fs=self.sample_rate, output="sos")
        return sosfilt(sos, arr, axis=0).astype(np.float32)

    def _apply_compression(
        self,
        audio: np.ndarray,
        threshold_db: float,
        ratio: float,
        attack_ms: float,
        release_ms: float,
    ) -> Tuple[np.ndarray, float]:
        arr = self._normalize_audio_shape(audio).astype(np.float64, copy=True)
        monitor = np.max(np.abs(arr), axis=1) if arr.ndim > 1 else np.abs(arr)
        attack = math_exp = np.exp(-1.0 / max(1.0, attack_ms * self.sample_rate / 1000.0))
        release = np.exp(-1.0 / max(1.0, release_ms * self.sample_rate / 1000.0))
        env = 0.0
        max_reduction = 0.0
        gains = np.ones(len(arr), dtype=np.float64)
        for i, level in enumerate(monitor):
            coeff = attack if level > env else release
            env = float(level) + coeff * (env - float(level))
            env_db = 20.0 * np.log10(max(env, EPS))
            over = max(0.0, env_db - threshold_db)
            reduction = over * (1.0 - 1.0 / max(1.0, ratio))
            max_reduction = max(max_reduction, reduction)
            gains[i] = 10.0 ** (-reduction / 20.0)
        if arr.ndim == 1:
            arr *= gains
        else:
            arr *= gains[:, None]
        return arr.astype(np.float32), float(max_reduction)

    def _apply_limiter(self, audio: np.ndarray, ceiling_db: float) -> Tuple[np.ndarray, float]:
        """Uniform reconstructed-peak limiter.

        This is deliberately not a loudness-maximizing lookahead limiter.  It
        applies only the attenuation required to keep the 4× reconstructed peak
        at or below the configured ceiling, preserving mix dynamics.
        """
        arr = self._normalize_audio_shape(audio).astype(np.float32, copy=True)
        ceiling = 10.0 ** (float(ceiling_db) / 20.0)
        reduction = 0.0
        for _ in range(2):
            tp_db = self._true_peak_db(arr)
            peak = 10.0 ** (tp_db / 20.0) if tp_db > -99.0 else 0.0
            if peak <= ceiling * (1.0 + 1e-6) or peak <= EPS:
                break
            gain = ceiling / peak
            arr *= np.float32(gain)
            reduction += -20.0 * np.log10(max(gain, EPS))
        return arr, float(max(0.0, reduction))

    def _write_wav(self, path: str, audio: np.ndarray, sample_rate: int):
        if not HAS_SCIPY or wavfile is None:
            raise RuntimeError("scipy not available")
        arr = np.clip(self._normalize_audio_shape(audio), -1.0, 1.0)
        wavfile.write(path, int(sample_rate), (arr * 32767.0).astype(np.int16))

    def _read_wav(self, path: str) -> np.ndarray:
        if not HAS_SCIPY or wavfile is None:
            raise RuntimeError("scipy not available")
        sample_rate, data = wavfile.read(path)
        self.sample_rate = int(sample_rate)
        if np.issubdtype(data.dtype, np.integer):
            scale = max(abs(np.iinfo(data.dtype).min), np.iinfo(data.dtype).max)
            return (data.astype(np.float32) / float(scale)).astype(np.float32)
        return np.asarray(data, dtype=np.float32)
