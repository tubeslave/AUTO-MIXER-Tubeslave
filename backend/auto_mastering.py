"""
Auto mastering module — applies mastering chain to the mix bus.
Uses matchering library for reference-based mastering when available.
"""
import numpy as np
import logging
from typing import Dict, Optional, Tuple
from dataclasses import dataclass

logger = logging.getLogger(__name__)

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
        self._matchering_available = False
        try:
            import matchering
            self._matchering_available = True
            logger.info("Matchering library available for reference-based mastering")
        except ImportError:
            logger.info("Matchering not available, using built-in mastering")

    def master(self, audio: np.ndarray, reference: Optional[np.ndarray] = None) -> MasteringResult:
        """Apply mastering chain to audio."""
        if len(audio) == 0:
            return MasteringResult(audio=audio, peak_db=-100, lufs=-100,
                                   gain_applied_db=0, limiter_reduction_db=0,
                                   eq_applied=False, success=False, error="Empty audio")

        audio = audio.astype(np.float32)

        if reference is not None and self._matchering_available:
            return self._master_with_reference(audio, reference)

        return self._builtin_master(audio)

    def _builtin_master(self, audio: np.ndarray) -> MasteringResult:
        """Built-in mastering chain: EQ -> Compression -> Limiting -> Normalization."""
        processed = audio.copy()

        # 1. Gentle high-pass filter at 30Hz
        processed = self._apply_hpf(processed, 30.0)

        # 2. Broadband compression
        processed, comp_reduction = self._apply_compression(
            processed, threshold_db=-18.0, ratio=2.0,
            attack_ms=30.0, release_ms=200.0
        )

        # 3. Loudness normalization по интегральному LUFS (BS.1770-4,
        # K-weighting + double gating) — невзвешенный RMS расходится с LUFS
        # на музыкальном материале на несколько dB, таргет промахивался
        current_lufs = self._measure_integrated_lufs(processed)
        gain_db = self.target_lufs - current_lufs
        gain_db = max(-12.0, min(12.0, gain_db))
        gain_linear = 10 ** (gain_db / 20.0)
        processed = processed * gain_linear

        # 4. Brick-wall limiter по TRUE peak (4x oversampling): сэмпловый пик
        # не видит межсэмпловых пиков и пропускал их над потолком -1 dBTP
        processed, limiter_reduction = self._apply_limiter(processed, self.true_peak_limit)

        # Final measurements
        peak_db = self._measure_true_peak(processed)
        final_lufs = self._measure_integrated_lufs(processed)

        return MasteringResult(
            audio=processed, peak_db=peak_db, lufs=final_lufs,
            gain_applied_db=gain_db, limiter_reduction_db=limiter_reduction,
            eq_applied=True, success=True
        )

    def _measure_integrated_lufs(self, audio: np.ndarray) -> float:
        """Integrated LUFS по ITU-R BS.1770-4: K-weighting, блоки 400 мс
        с перекрытием 75%, double gating (-70 LUFS абс. + -10 LU отн.)."""
        from lufs_gain_staging import KWeightingFilter

        k_filter = KWeightingFilter(self.sample_rate)
        weighted = k_filter.process(np.asarray(audio, dtype=np.float64))

        block = int(0.4 * self.sample_rate)
        hop = block // 4  # 100 мс, перекрытие 75%
        if len(weighted) < block:
            mean_sq = float(np.mean(weighted ** 2)) if len(weighted) else 0.0
            return -0.691 + 10 * np.log10(max(mean_sq, 1e-10))

        powers = np.array([
            np.mean(weighted[i:i + block] ** 2)
            for i in range(0, len(weighted) - block + 1, hop)
        ])
        loudness = -0.691 + 10 * np.log10(np.maximum(powers, 1e-10))

        abs_gated = powers[loudness > -70.0]
        if len(abs_gated) == 0:
            return -100.0
        rel_threshold = -0.691 + 10 * np.log10(np.mean(abs_gated)) - 10.0
        gated = powers[(loudness > -70.0) & (loudness > rel_threshold)]
        if len(gated) == 0:
            gated = abs_gated
        return float(-0.691 + 10 * np.log10(np.mean(gated)))

    def _measure_true_peak(self, audio: np.ndarray) -> float:
        """True peak (dBTP) с 4x oversampling (стандарт проекта)."""
        from lufs_gain_staging import TruePeakMeter

        meter = TruePeakMeter(sample_rate=self.sample_rate)
        return float(meter.process(np.asarray(audio, dtype=np.float64)))

    def _master_with_reference(self, audio: np.ndarray, reference: np.ndarray) -> MasteringResult:
        """Master using matchering library with a reference track."""
        try:
            import matchering as mg
            import tempfile
            import os

            try:
                import soundfile as sf
            except ImportError:
                logger.warning("soundfile not available for matchering I/O")
                return self._builtin_master(audio)

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
            logger.error(f"Matchering error: {e}, falling back to built-in")
            return self._builtin_master(audio)

    def _apply_hpf(self, audio: np.ndarray, cutoff_hz: float) -> np.ndarray:
        """Apply simple high-pass filter."""
        try:
            from scipy.signal import butter, sosfilt
            sos = butter(2, cutoff_hz, btype='high', fs=self.sample_rate, output='sos')
            return sosfilt(sos, audio).astype(np.float32)
        except ImportError:
            return audio

    def _apply_compression(self, audio: np.ndarray, threshold_db: float,
                          ratio: float, attack_ms: float, release_ms: float) -> Tuple[np.ndarray, float]:
        """Apply dynamic compression."""
        eps = 1e-10
        attack_coeff = np.exp(-1.0 / (attack_ms * self.sample_rate / 1000))
        release_coeff = np.exp(-1.0 / (release_ms * self.sample_rate / 1000))

        threshold_lin = 10 ** (threshold_db / 20.0)
        output = np.copy(audio)
        envelope = 0.0
        max_reduction = 0.0

        for i in range(len(audio)):
            level = abs(audio[i])
            if level > envelope:
                envelope = attack_coeff * envelope + (1 - attack_coeff) * level
            else:
                envelope = release_coeff * envelope + (1 - release_coeff) * level

            if envelope > threshold_lin:
                gain_reduction = (envelope / threshold_lin) ** (1 - 1/ratio)
                output[i] = audio[i] / (gain_reduction + eps)
                reduction_db = 20 * np.log10(gain_reduction + eps)
                max_reduction = max(max_reduction, reduction_db)

        return output, max_reduction

    def _apply_limiter(self, audio: np.ndarray, ceiling_db: float) -> Tuple[np.ndarray, float]:
        """Apply brick-wall limiter против TRUE peak (4x oversampling)."""
        true_peak_db = self._measure_true_peak(audio)
        reduction_db = 0.0

        if true_peak_db > ceiling_db:
            reduction_db = true_peak_db - ceiling_db
            audio = audio * (10 ** (-reduction_db / 20.0))

        return audio, abs(float(reduction_db))
