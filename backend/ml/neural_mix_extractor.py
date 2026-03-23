"""
Neural Mix Extractor - Extract mixing parameters from audio
============================================================
Analyzes dry stems vs processed mix to reverse-engineer mixing decisions.
Uses numpy/scipy spectral analysis to estimate gains, EQ, compression, and panning.
"""

import logging
import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np

try:
    from scipy.signal import stft, find_peaks, butter, filtfilt
    from scipy.interpolate import interp1d

    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False

logger = logging.getLogger(__name__)


@dataclass
class EQBandExtracted:
    """Extracted EQ band parameters."""

    center_freq: float  # Hz
    gain_db: float  # dB boost/cut
    q: float  # Quality factor
    band_type: str = "peak"  # 'peak', 'low_shelf', 'high_shelf'

    def to_dict(self) -> dict:
        return {
            "center_freq": round(self.center_freq, 1),
            "gain_db": round(self.gain_db, 2),
            "q": round(self.q, 2),
            "band_type": self.band_type,
        }


@dataclass
class CompressionParams:
    """Estimated compression parameters."""

    threshold_db: float  # dB below which compression starts
    ratio: float  # Compression ratio (e.g. 4.0 means 4:1)
    attack_ms: float  # Attack time in milliseconds
    release_ms: float  # Release time in milliseconds
    makeup_gain_db: float  # Estimated makeup gain
    dynamic_range_reduction_db: float  # How much DR was reduced

    def to_dict(self) -> dict:
        return {
            "threshold_db": round(self.threshold_db, 1),
            "ratio": round(self.ratio, 1),
            "attack_ms": round(self.attack_ms, 1),
            "release_ms": round(self.release_ms, 1),
            "makeup_gain_db": round(self.makeup_gain_db, 1),
            "dynamic_range_reduction_db": round(self.dynamic_range_reduction_db, 1),
        }


class NeuralMixExtractor:
    """
    Extract mixing parameters from audio by comparing dry stems to a processed mix.

    Uses spectral analysis to reverse-engineer EQ curves, gain values,
    compression settings, and panning positions from before/after audio.
    """

    def __init__(
        self,
        fft_size: int = 4096,
        hop_size: int = 1024,
        num_eq_bands: int = 8,
        envelope_window_ms: float = 10.0,
    ):
        """
        Args:
            fft_size: FFT window size for spectral analysis.
            hop_size: Hop size for STFT.
            num_eq_bands: Number of EQ bands to extract.
            envelope_window_ms: Envelope follower window in ms for compression analysis.
        """
        self.fft_size = fft_size
        self.hop_size = hop_size
        self.num_eq_bands = num_eq_bands
        self.envelope_window_ms = envelope_window_ms

    def extract_gains(
        self, dry_stems: Dict[str, np.ndarray], processed_mix: np.ndarray
    ) -> Dict[str, float]:
        """
        Estimate per-channel gain values by comparing dry stems to the processed mix.

        Uses RMS energy comparison between each dry stem and its estimated
        contribution in the final mix.

        Args:
            dry_stems: Dict mapping channel name -> mono audio array (float32).
            processed_mix: The processed mono mixdown (float32).

        Returns:
            Dict mapping channel name -> estimated gain in dB.
        """
        gains = {}

        mix_rms = self._compute_rms(processed_mix)
        if mix_rms < 1e-10:
            logger.warning("Processed mix is silent, returning 0 dB for all channels")
            return {name: 0.0 for name in dry_stems}

        # Sum of all dry stems to estimate total dry energy
        total_dry_rms = 0.0
        stem_rms_values = {}
        for name, stem in dry_stems.items():
            rms = self._compute_rms(stem)
            stem_rms_values[name] = rms
            total_dry_rms += rms ** 2

        total_dry_rms = math.sqrt(total_dry_rms) if total_dry_rms > 0 else 1e-10

        # Estimate each channel's gain as the ratio of mix energy to dry energy,
        # weighted by the stem's relative contribution
        for name, stem in dry_stems.items():
            stem_rms = stem_rms_values[name]
            if stem_rms < 1e-10:
                gains[name] = -96.0  # Effectively silent
                continue

            # Cross-correlation based gain estimation
            # Find the scaling factor that best maps dry stem to its contribution in mix
            min_len = min(len(stem), len(processed_mix))
            dry_segment = stem[:min_len]
            mix_segment = processed_mix[:min_len]

            # Least squares: find g that minimizes ||mix - g * dry||^2
            # g = (dry . mix) / (dry . dry)
            dot_product = np.dot(dry_segment.astype(np.float64), mix_segment.astype(np.float64))
            dry_energy = np.dot(dry_segment.astype(np.float64), dry_segment.astype(np.float64))

            if dry_energy < 1e-20:
                gains[name] = -96.0
                continue

            linear_gain = dot_product / dry_energy

            # Clamp to reasonable range
            linear_gain = max(linear_gain, 1e-10)

            gain_db = 20.0 * math.log10(linear_gain)
            # Clamp to -96..+24 dB
            gain_db = max(-96.0, min(24.0, gain_db))
            gains[name] = round(gain_db, 2)

        return gains

    def extract_eq(
        self, dry_stem: np.ndarray, processed_stem: np.ndarray, sr: int
    ) -> List[EQBandExtracted]:
        """
        Estimate EQ settings by comparing spectra of dry vs processed audio.

        Computes the spectral difference (transfer function) and fits parametric
        EQ bands to the resulting curve.

        Args:
            dry_stem: Dry mono audio (float32).
            processed_stem: Processed mono audio (float32).
            sr: Sample rate in Hz.

        Returns:
            List of EQBandExtracted with estimated EQ parameters.
        """
        if not HAS_SCIPY:
            logger.warning("scipy not available, using basic EQ extraction")
            return self._extract_eq_basic(dry_stem, processed_stem, sr)

        min_len = min(len(dry_stem), len(processed_stem))
        dry = dry_stem[:min_len].astype(np.float64)
        proc = processed_stem[:min_len].astype(np.float64)

        # Compute STFT for both signals
        f_dry, t_dry, Zxx_dry = stft(dry, fs=sr, nperseg=self.fft_size, noverlap=self.fft_size - self.hop_size)
        _, _, Zxx_proc = stft(proc, fs=sr, nperseg=self.fft_size, noverlap=self.fft_size - self.hop_size)

        # Average magnitude spectra across time
        mag_dry = np.mean(np.abs(Zxx_dry), axis=1)
        mag_proc = np.mean(np.abs(Zxx_proc), axis=1)

        # Compute transfer function in dB
        eps = 1e-10
        transfer_db = 20.0 * np.log10((mag_proc + eps) / (mag_dry + eps))

        # Smooth the transfer function
        if len(transfer_db) > 15:
            from scipy.signal import savgol_filter
            window_len = min(15, len(transfer_db))
            if window_len % 2 == 0:
                window_len -= 1
            if window_len >= 3:
                transfer_db = savgol_filter(transfer_db, window_len, 3)

        # Find peaks and valleys in the transfer function
        freqs = f_dry

        bands = self._fit_eq_bands(freqs, transfer_db, sr)
        return bands

    def _extract_eq_basic(
        self, dry_stem: np.ndarray, processed_stem: np.ndarray, sr: int
    ) -> List[EQBandExtracted]:
        """Fallback EQ extraction without scipy, using basic numpy FFT."""
        min_len = min(len(dry_stem), len(processed_stem))
        dry = dry_stem[:min_len].astype(np.float64)
        proc = processed_stem[:min_len].astype(np.float64)

        # Compute magnitude spectra using numpy FFT
        n_fft = self.fft_size
        # Process in frames and average
        n_frames = max(1, (min_len - n_fft) // self.hop_size)
        mag_dry = np.zeros(n_fft // 2 + 1)
        mag_proc = np.zeros(n_fft // 2 + 1)

        window = np.hanning(n_fft)
        for i in range(n_frames):
            start = i * self.hop_size
            end = start + n_fft
            if end > min_len:
                break
            frame_dry = dry[start:end] * window
            frame_proc = proc[start:end] * window
            mag_dry += np.abs(np.fft.rfft(frame_dry))
            mag_proc += np.abs(np.fft.rfft(frame_proc))

        if n_frames > 0:
            mag_dry /= n_frames
            mag_proc /= n_frames

        eps = 1e-10
        transfer_db = 20.0 * np.log10((mag_proc + eps) / (mag_dry + eps))
        freqs = np.fft.rfftfreq(n_fft, 1.0 / sr)

        return self._fit_eq_bands(freqs, transfer_db, sr)

    def _fit_eq_bands(
        self, freqs: np.ndarray, transfer_db: np.ndarray, sr: int
    ) -> List[EQBandExtracted]:
        """
        Fit parametric EQ bands to a transfer function curve.

        Divides the spectrum into bands on a logarithmic scale,
        then finds the peak deviation within each band.
        """
        bands = []
        nyquist = sr / 2.0

        # Define band center frequencies on a log scale
        min_freq = 40.0
        max_freq = min(nyquist * 0.95, 20000.0)
        band_centers = np.exp(
            np.linspace(np.log(min_freq), np.log(max_freq), self.num_eq_bands)
        )

        # Bandwidth in octaves for each band
        octave_width = np.log2(max_freq / min_freq) / self.num_eq_bands

        for i, center in enumerate(band_centers):
            # Define band edges
            low_edge = center / (2.0 ** (octave_width / 2))
            high_edge = center * (2.0 ** (octave_width / 2))

            # Find frequency bins in this band
            mask = (freqs >= low_edge) & (freqs <= high_edge)
            if not np.any(mask):
                continue

            band_transfer = transfer_db[mask]
            band_freqs = freqs[mask]

            # Find the frequency with maximum deviation
            max_idx = np.argmax(np.abs(band_transfer))
            peak_freq = band_freqs[max_idx]
            peak_gain = band_transfer[max_idx]

            # Skip bands with negligible gain
            if abs(peak_gain) < 0.5:
                continue

            # Estimate Q from the shape of the deviation
            # Narrower peaks = higher Q
            above_half = np.abs(band_transfer) > (abs(peak_gain) * 0.5)
            bandwidth_bins = np.sum(above_half)
            total_bins = len(band_transfer)
            if bandwidth_bins > 0 and total_bins > 0:
                bandwidth_ratio = bandwidth_bins / total_bins
                # Q is inversely related to bandwidth ratio
                q = max(0.5, min(10.0, 1.0 / (bandwidth_ratio + 0.01)))
            else:
                q = 1.0

            # Determine band type
            if i == 0 and peak_freq < 150:
                band_type = "low_shelf"
            elif i == self.num_eq_bands - 1 and peak_freq > 8000:
                band_type = "high_shelf"
            else:
                band_type = "peak"

            bands.append(
                EQBandExtracted(
                    center_freq=float(peak_freq),
                    gain_db=float(peak_gain),
                    q=float(q),
                    band_type=band_type,
                )
            )

        return bands

    def extract_compression(
        self, dry_stem: np.ndarray, processed_stem: np.ndarray, sr: int
    ) -> CompressionParams:
        """
        Estimate compression parameters by analyzing dynamic range changes.

        Compares amplitude envelopes of dry vs processed audio to determine
        threshold, ratio, attack, release, and makeup gain.

        Args:
            dry_stem: Dry mono audio (float32).
            processed_stem: Processed mono audio (float32).
            sr: Sample rate in Hz.

        Returns:
            CompressionParams with estimated settings.
        """
        min_len = min(len(dry_stem), len(processed_stem))
        dry = dry_stem[:min_len].astype(np.float64)
        proc = processed_stem[:min_len].astype(np.float64)

        # Compute amplitude envelopes
        env_samples = max(1, int(self.envelope_window_ms * sr / 1000.0))
        dry_env = self._compute_envelope(dry, env_samples)
        proc_env = self._compute_envelope(proc, env_samples)

        eps = 1e-10

        # Convert to dB
        dry_db = 20.0 * np.log10(dry_env + eps)
        proc_db = 20.0 * np.log10(proc_env + eps)

        # Dynamic range of each signal
        # Use percentiles to avoid noise floor influence
        dry_dr = float(np.percentile(dry_db, 95) - np.percentile(dry_db, 10))
        proc_dr = float(np.percentile(proc_db, 95) - np.percentile(proc_db, 10))
        dr_reduction = max(0.0, dry_dr - proc_dr)

        # Estimate ratio from DR reduction
        # If dry DR is 30dB and processed is 20dB, ratio ~= 30/20 = 1.5:1
        if proc_dr > 1.0:
            ratio = max(1.0, dry_dr / proc_dr)
        else:
            ratio = 1.0

        ratio = min(20.0, ratio)  # Cap at 20:1

        # Estimate threshold: find the dB level where gain reduction begins
        # Compare the transfer curve (proc_db vs dry_db)
        # Sort by dry level to get a transfer characteristic
        sorted_indices = np.argsort(dry_db)
        dry_sorted = dry_db[sorted_indices]
        proc_sorted = proc_db[sorted_indices]

        # Gain reduction = dry - proc (positive means compression)
        gain_reduction = dry_sorted - proc_sorted

        # Find where gain reduction consistently exceeds 1 dB
        threshold_db = -20.0  # Default
        n_points = len(gain_reduction)
        window = max(1, n_points // 50)
        for i in range(window, n_points - window):
            local_gr = np.mean(gain_reduction[max(0, i - window):i + window])
            if local_gr > 1.0:
                threshold_db = float(dry_sorted[i])
                break

        # Estimate makeup gain (average level difference)
        makeup_gain = float(np.median(proc_db) - np.median(dry_db))
        makeup_gain = max(0.0, makeup_gain)

        # Estimate attack time from how quickly the compressor reacts to transients
        attack_ms, release_ms = self._estimate_attack_release(
            dry_env, proc_env, sr
        )

        return CompressionParams(
            threshold_db=round(threshold_db, 1),
            ratio=round(ratio, 1),
            attack_ms=round(attack_ms, 1),
            release_ms=round(release_ms, 1),
            makeup_gain_db=round(makeup_gain, 1),
            dynamic_range_reduction_db=round(dr_reduction, 1),
        )

    def extract_panning(self, processed_mix_stereo: np.ndarray) -> float:
        """
        Extract pan position from a stereo mix.

        Analyzes the L/R energy balance to estimate pan position.

        Args:
            processed_mix_stereo: Stereo audio, shape (N, 2) or (2, N).

        Returns:
            Pan position from -1.0 (full left) to 1.0 (full right).
        """
        if processed_mix_stereo.ndim == 1:
            logger.warning("Mono signal provided, returning center pan")
            return 0.0

        # Ensure shape is (N, 2)
        if processed_mix_stereo.shape[0] == 2 and processed_mix_stereo.shape[1] != 2:
            audio = processed_mix_stereo.T
        else:
            audio = processed_mix_stereo

        if audio.shape[1] < 2:
            return 0.0

        left = audio[:, 0].astype(np.float64)
        right = audio[:, 1].astype(np.float64)

        # Compute RMS energy for each channel
        left_rms = np.sqrt(np.mean(left ** 2))
        right_rms = np.sqrt(np.mean(right ** 2))

        total = left_rms + right_rms
        if total < 1e-10:
            return 0.0

        # Pan law: constant power panning
        # pan = (R - L) / (R + L) gives -1..1 range
        pan = float((right_rms - left_rms) / total)

        # Refine using time-domain correlation for more accuracy
        # Phase difference can indicate panning
        if len(left) > 0 and len(right) > 0:
            # Cross-correlation at zero lag
            correlation = np.dot(left, right) / (
                np.sqrt(np.dot(left, left) * np.dot(right, right)) + 1e-10
            )

            # High correlation with amplitude difference -> amplitude panning
            # Low correlation -> could be decorrelated (wide stereo)
            if correlation > 0.9:
                # Mostly amplitude panning, trust the RMS ratio
                pass
            elif correlation < 0.3:
                # Decorrelated signals, use mid/side analysis
                mid = (left + right) / 2.0
                side = (left - right) / 2.0
                mid_rms = np.sqrt(np.mean(mid ** 2))
                side_rms = np.sqrt(np.mean(side ** 2))
                if mid_rms > 1e-10:
                    width = side_rms / mid_rms
                    # Reduce confidence in pan estimate for wide signals
                    pan *= max(0.1, 1.0 - width * 0.5)

        return max(-1.0, min(1.0, round(pan, 3)))

    def _compute_rms(self, audio: np.ndarray) -> float:
        """Compute RMS energy of an audio signal."""
        return float(np.sqrt(np.mean(audio.astype(np.float64) ** 2)))

    def _compute_envelope(self, audio: np.ndarray, window_samples: int) -> np.ndarray:
        """
        Compute amplitude envelope using a moving RMS window.

        Args:
            audio: Input audio signal.
            window_samples: Window size in samples.

        Returns:
            Amplitude envelope array.
        """
        abs_audio = np.abs(audio)

        if window_samples <= 1:
            return abs_audio

        # Use cumulative sum for efficient moving average
        cumsum = np.cumsum(abs_audio ** 2)
        cumsum = np.insert(cumsum, 0, 0)

        # Moving RMS
        n = len(abs_audio)
        envelope = np.zeros(n)
        for i in range(n):
            start = max(0, i - window_samples // 2)
            end = min(n, i + window_samples // 2 + 1)
            window_energy = (cumsum[end] - cumsum[start]) / (end - start)
            envelope[i] = np.sqrt(max(0.0, window_energy))

        return envelope

    def _estimate_attack_release(
        self,
        dry_env: np.ndarray,
        proc_env: np.ndarray,
        sr: int,
    ) -> Tuple[float, float]:
        """
        Estimate compressor attack and release times by analyzing envelope differences.

        Looks at how quickly the processed envelope responds to transients in the dry signal.

        Returns:
            Tuple of (attack_ms, release_ms).
        """
        eps = 1e-10

        # Compute gain reduction envelope
        gain_reduction = np.log10(proc_env + eps) - np.log10(dry_env + eps)

        # Find transients in dry signal (large positive derivatives)
        dry_diff = np.diff(dry_env)
        threshold = np.percentile(np.abs(dry_diff), 90)

        if threshold < eps:
            return (10.0, 100.0)  # Default values

        # Find onset indices where the dry signal jumps up
        onsets = np.where(dry_diff > threshold)[0]
        if len(onsets) == 0:
            return (10.0, 100.0)

        # Reduce to unique onsets (at least 50ms apart)
        min_gap = int(0.05 * sr)
        filtered_onsets = [onsets[0]]
        for onset in onsets[1:]:
            if onset - filtered_onsets[-1] >= min_gap:
                filtered_onsets.append(onset)

        attack_samples_list = []
        release_samples_list = []

        for onset in filtered_onsets[:20]:  # Analyze up to 20 transients
            # Look at gain reduction after the onset
            search_len = min(int(0.1 * sr), len(gain_reduction) - onset - 1)
            if search_len < 2:
                continue

            gr_segment = gain_reduction[onset: onset + search_len]

            # Attack: time from onset to maximum gain reduction
            min_idx = np.argmin(gr_segment)
            if min_idx > 0:
                attack_samples_list.append(min_idx)

            # Release: look after the peak gain reduction for recovery
            release_search_start = onset + min_idx
            release_search_len = min(int(0.5 * sr), len(gain_reduction) - release_search_start - 1)
            if release_search_len < 2:
                continue

            gr_release = gain_reduction[release_search_start: release_search_start + release_search_len]
            gr_min = gr_release[0]
            if gr_min >= -eps:
                continue

            # Find where GR recovers to 63% of its peak (time constant)
            recovery_level = gr_min * 0.37  # 1 - 1/e
            recovered_indices = np.where(gr_release > recovery_level)[0]
            if len(recovered_indices) > 0:
                release_samples_list.append(recovered_indices[0])

        # Compute median attack and release times
        if attack_samples_list:
            attack_ms = float(np.median(attack_samples_list)) / sr * 1000.0
        else:
            attack_ms = 10.0

        if release_samples_list:
            release_ms = float(np.median(release_samples_list)) / sr * 1000.0
        else:
            release_ms = 100.0

        # Clamp to reasonable ranges
        attack_ms = max(0.1, min(200.0, attack_ms))
        release_ms = max(5.0, min(2000.0, release_ms))

        return (attack_ms, release_ms)
