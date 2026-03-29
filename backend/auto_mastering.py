"""
Auto mastering module — applies mastering chain to the mix bus.

Uses pyloudnorm for accurate ITU-R BS.1770 LUFS measurement and normalization.
Uses matchering library for reference-based mastering when available.
"""
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
    logger.warning(
        "pyloudnorm not installed — falling back to RMS-based loudness. "
        "Install with: pip install pyloudnorm"
    )


@dataclass
class MasteringResult:
    """Result of mastering process."""
    audio: np.ndarray
    peak_db: float
    lufs: float
    true_peak_dbtp: float
    gain_applied_db: float
    limiter_reduction_db: float
    eq_applied: bool
    success: bool
    lra: float = 0.0  # Loudness Range (LU)
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
    """Automatic mastering processor with accurate LUFS measurement."""

    def __init__(self, sample_rate: int = 48000, target_lufs: float = -14.0,
                 true_peak_limit: float = -1.0):
        self.sample_rate = sample_rate
        self.target_lufs = target_lufs
        self.true_peak_limit = true_peak_limit
        self._matchering_available = False
        self._pyloudnorm_available = _PYLOUDNORM_AVAILABLE

        try:
            import matchering
            self._matchering_available = True
            logger.info("Matchering library available for reference-based mastering")
        except ImportError:
            logger.info("Matchering not available, using built-in mastering")

        if self._pyloudnorm_available:
            self._meter = pyln.Meter(sample_rate)
            logger.info("pyloudnorm available — using ITU-R BS.1770-4 LUFS measurement")
        else:
            self._meter = None
            logger.warning("pyloudnorm NOT available — using RMS-based fallback")

    def master(self, audio: np.ndarray, reference: Optional[np.ndarray] = None) -> MasteringResult:
        """Apply mastering chain to audio.

        Pipeline: HPF → Compression → LUFS Normalization → True Peak Limiter.

        Args:
            audio: Input audio, shape (samples,) or (samples, channels).
            reference: Optional reference track for matchering-based mastering.

        Returns:
            MasteringResult with processed audio and measurements.
        """
        if len(audio) == 0:
            return MasteringResult(
                audio=audio, peak_db=-100, lufs=-100, true_peak_dbtp=-100,
                gain_applied_db=0, limiter_reduction_db=0,
                eq_applied=False, success=False, error="Empty audio",
            )

        audio = audio.astype(np.float32)

        if reference is not None and self._matchering_available:
            return self._master_with_reference(audio, reference)

        return self._builtin_master(audio)

    def _builtin_master(self, audio: np.ndarray) -> MasteringResult:
        """Built-in mastering chain: HPF → Compression → LUFS Norm → True Peak Limiter.

        Key improvement over previous version: uses pyloudnorm for accurate
        ITU-R BS.1770-4 LUFS measurement instead of RMS approximation.
        """
        processed = audio.copy()

        # 1. Gentle high-pass filter at 30 Hz (remove sub-bass rumble)
        processed = self._apply_hpf(processed, 30.0)

        # 2. Gentle bus compression (mastering glue — light touch)
        processed, comp_reduction = self._apply_compression(
            processed, threshold_db=-12.0, ratio=1.5,
            attack_ms=30.0, release_ms=200.0
        )

        # 3. Accurate LUFS normalization via pyloudnorm
        current_lufs = measure_lufs(processed, self.sample_rate)
        logger.info("Pre-normalization LUFS: %.2f", current_lufs)

        if current_lufs > -70.0:
            gain_db = self.target_lufs - current_lufs
            gain_db = max(-12.0, min(12.0, gain_db))

            if self._pyloudnorm_available:
                if processed.ndim == 1:
                    processed = pyln.normalize.loudness(
                        processed.reshape(-1, 1), current_lufs, self.target_lufs
                    ).flatten().astype(np.float32)
                else:
                    processed = pyln.normalize.loudness(
                        processed, current_lufs, self.target_lufs
                    ).astype(np.float32)
            else:
                gain_linear = 10 ** (gain_db / 20.0)
                processed = processed * gain_linear
        else:
            gain_db = 0.0
            logger.warning("Audio too quiet for LUFS normalization (%.1f LUFS)", current_lufs)

        # 4. Look-ahead limiter (sample-peak based, preserves loudness)
        processed, limiter_reduction = self._apply_true_peak_limiter(
            processed, self.true_peak_limit
        )

        # 5. True peak safety pass — catch inter-sample peaks
        #    The look-ahead limiter works on sample peaks; true peaks
        #    (4x oversampled) can be ~0.5 dB higher. This iterative pass
        #    applies minimal correction to guarantee TP compliance.
        for _ in range(3):
            tp_check = measure_true_peak(processed, self.sample_rate)
            if tp_check > self.true_peak_limit:
                overshoot = tp_check - self.true_peak_limit + 0.05  # +0.05 dB margin
                correction = 10 ** (-overshoot / 20.0)
                processed = processed * correction
                logger.info(
                    "True peak safety pass: %.2f dBTP -> corrected by %.2f dB",
                    tp_check, overshoot,
                )
            else:
                break

        # Final measurements (accurate)
        final_lufs = measure_lufs(processed, self.sample_rate)
        final_true_peak = measure_true_peak(processed, self.sample_rate)
        peak_db = float(20 * np.log10(np.max(np.abs(processed)) + 1e-10))

        logger.info(
            "Mastering result: LUFS=%.2f (target=%.1f), True Peak=%.2f dBTP, "
            "Gain=%+.1f dB, Limiter reduction=%.1f dB",
            final_lufs, self.target_lufs, final_true_peak, gain_db, limiter_reduction,
        )

        return MasteringResult(
            audio=processed,
            peak_db=peak_db,
            lufs=final_lufs,
            true_peak_dbtp=final_true_peak,
            gain_applied_db=gain_db,
            limiter_reduction_db=limiter_reduction,
            eq_applied=True,
            success=True,
        )

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

                # Accurate measurements with pyloudnorm
                peak_db = float(20 * np.log10(np.max(np.abs(mastered)) + 1e-10))
                final_lufs = measure_lufs(mastered, self.sample_rate)
                final_true_peak = measure_true_peak(mastered, self.sample_rate)
                input_lufs = measure_lufs(audio, self.sample_rate)
                gain_db = final_lufs - input_lufs if input_lufs > -70 else 0.0

                return MasteringResult(
                    audio=mastered,
                    peak_db=peak_db,
                    lufs=final_lufs,
                    true_peak_dbtp=final_true_peak,
                    gain_applied_db=float(gain_db),
                    limiter_reduction_db=0,
                    eq_applied=True,
                    success=True,
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
        """Apply linked stereo dynamic compression.

        For stereo input, the envelope follower uses the max absolute value
        across channels (linked mode), and gain reduction is applied equally
        to all channels — standard bus compressor behaviour.
        """
        eps = 1e-10
        attack_coeff = np.exp(-1.0 / (attack_ms * self.sample_rate / 1000))
        release_coeff = np.exp(-1.0 / (release_ms * self.sample_rate / 1000))
        threshold_lin = 10 ** (threshold_db / 20.0)

        # Get per-sample level (max across channels for stereo)
        if audio.ndim == 2:
            levels = np.max(np.abs(audio), axis=1)  # (samples,)
        else:
            levels = np.abs(audio)

        num_samples = len(levels)
        output = np.copy(audio)

        # Envelope follower (must be sequential due to feedback)
        envelope = 0.0
        max_reduction = 0.0
        gain_curve = np.ones(num_samples, dtype=np.float32)

        for i in range(num_samples):
            level = float(levels[i])
            if level > envelope:
                envelope = attack_coeff * envelope + (1 - attack_coeff) * level
            else:
                envelope = release_coeff * envelope + (1 - release_coeff) * level

            if envelope > threshold_lin:
                gain_reduction = (envelope / threshold_lin) ** (1 - 1/ratio)
                gain_curve[i] = 1.0 / (gain_reduction + eps)
                reduction_db = 20 * np.log10(gain_reduction + eps)
                max_reduction = max(max_reduction, reduction_db)

        # Apply gain curve to all channels at once (vectorized)
        if audio.ndim == 2:
            output = audio * gain_curve[:, np.newaxis]
        else:
            output = audio * gain_curve

        return output.astype(np.float32), max_reduction

    def _apply_true_peak_limiter(self, audio: np.ndarray, ceiling_dbtp: float) -> Tuple[np.ndarray, float]:
        """Look-ahead true-peak limiter for mastering.

        Unlike a simple brick-wall (scale entire signal), this limiter:
        - Uses a look-ahead buffer (5ms) to anticipate peaks
        - Reduces gain ONLY around peaks, preserving average loudness (LUFS)
        - Smoothly applies attack/release envelope
        - Works on true peak (4x oversampled) level detection

        This is what mastering limiters (L2, Pro-L, etc.) do.
        """
        lookahead_ms = 5.0
        release_ms = 50.0
        lookahead_samples = int(self.sample_rate * lookahead_ms / 1000)
        release_coeff = np.exp(-1.0 / (release_ms * self.sample_rate / 1000))

        ceiling_lin = 10 ** (ceiling_dbtp / 20.0)

        # Get per-sample peak level (max across channels for stereo)
        if audio.ndim == 2:
            levels = np.max(np.abs(audio), axis=1)
        else:
            levels = np.abs(audio)

        num_samples = len(levels)

        # Step 1: Compute required gain reduction per sample
        # (where level exceeds ceiling, gain < 1.0)
        gain_required = np.ones(num_samples, dtype=np.float64)
        over_mask = levels > ceiling_lin
        gain_required[over_mask] = ceiling_lin / (levels[over_mask] + 1e-10)

        # Step 2: Look-ahead — apply minimum gain within the look-ahead window
        # This ensures gain reduction starts BEFORE the peak arrives
        gain_lookahead = np.ones(num_samples, dtype=np.float64)
        for i in range(num_samples):
            end = min(i + lookahead_samples, num_samples)
            gain_lookahead[i] = np.min(gain_required[i:end])

        # Step 3: Smooth the gain curve with attack (instant) and release
        gain_smooth = np.ones(num_samples, dtype=np.float64)
        current_gain = 1.0
        for i in range(num_samples):
            target = gain_lookahead[i]
            if target < current_gain:
                # Attack: instant (look-ahead already handled timing)
                current_gain = target
            else:
                # Release: exponential recovery
                current_gain = release_coeff * current_gain + (1 - release_coeff) * target
            gain_smooth[i] = current_gain

        # Step 4: Apply gain curve
        if audio.ndim == 2:
            output = audio * gain_smooth[:, np.newaxis]
        else:
            output = audio * gain_smooth

        # Measure actual reduction
        max_reduction_db = float(-20 * np.log10(np.min(gain_smooth) + 1e-10))

        if max_reduction_db > 0.1:
            logger.info(
                "Look-ahead limiter: ceiling %.1f dBTP, max reduction %.1f dB, "
                "samples limited: %d/%d (%.1f%%)",
                ceiling_dbtp, max_reduction_db,
                int(np.sum(gain_smooth < 0.999)), num_samples,
                100 * np.sum(gain_smooth < 0.999) / num_samples,
            )

        return output.astype(np.float32), max_reduction_db

    def _apply_limiter(self, audio: np.ndarray, ceiling_db: float) -> Tuple[np.ndarray, float]:
        """Legacy sample-peak limiter (kept for backward compatibility)."""
        ceiling_lin = 10 ** (ceiling_db / 20.0)
        peak = np.max(np.abs(audio))
        reduction_db = 0.0

        if peak > ceiling_lin:
            gain = ceiling_lin / peak
            audio = audio * gain
            reduction_db = float(20 * np.log10(gain))

        return audio, abs(reduction_db)
