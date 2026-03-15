"""
Mix quality metrics for evaluating automatic mixes.

Provides objective measurements of mix quality including:
- Spectral balance scoring
- Stereo correlation analysis
- Loudness range (LRA)
- Crest factor measurement
- Spectral masking detection
- A-weighted level calculation
- Overall quality score (0-100)
"""

import numpy as np
import math
import logging

logger = logging.getLogger(__name__)

try:
    import scipy.signal
    import scipy.fft

    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False


class MixQualityMetric:
    """
    Comprehensive mix quality measurement system.

    Evaluates a stereo mix (and optionally individual channels)
    across multiple objective criteria and produces both individual
    metric scores and a combined quality rating.
    """

    # Target spectral balance curve (dB relative to 1kHz)
    # Based on pink noise reference adjusted for musical content
    _TARGET_BALANCE = {
        31.25: 4.0,
        62.5: 3.0,
        125: 2.0,
        250: 1.0,
        500: 0.5,
        1000: 0.0,
        2000: -0.5,
        4000: -1.5,
        8000: -3.0,
        16000: -6.0,
    }

    def __init__(self, sr=48000):
        """
        Args:
            sr: sample rate for analysis
        """
        self.sr = sr

    def spectral_balance(self, mix_audio, sr=None):
        """
        Score the spectral balance of a mix (0-100).

        Compares the average spectrum against a target spectral
        balance curve. Penalizes excessive bass buildup, harsh
        midrange, or missing high frequency content.

        Args:
            mix_audio: 1D numpy array (mono) or 2D (channels, samples)
            sr: sample rate (uses self.sr if None)

        Returns:
            score: float 0-100 (100 = perfectly balanced)
        """
        sr = sr or self.sr
        audio = np.asarray(mix_audio, dtype=np.float64)

        # Convert to mono if stereo
        if audio.ndim == 2:
            audio = np.mean(audio, axis=0)

        if len(audio) < 1024:
            return 50.0  # insufficient data

        # Compute average magnitude spectrum
        n_fft = 4096
        hop = n_fft // 2
        n_frames = max(1, (len(audio) - n_fft) // hop)
        window = np.hanning(n_fft)

        accum_spectrum = np.zeros(n_fft // 2 + 1)
        for i in range(n_frames):
            start = i * hop
            frame = audio[start: start + n_fft]
            if len(frame) < n_fft:
                frame = np.pad(frame, (0, n_fft - len(frame)))
            windowed = frame * window
            spectrum = np.abs(np.fft.rfft(windowed))
            accum_spectrum += spectrum ** 2

        avg_spectrum = np.sqrt(accum_spectrum / max(n_frames, 1))
        freqs = np.fft.rfftfreq(n_fft, d=1.0 / sr)

        # Compute energy in octave bands
        band_energies = {}
        for center_freq in self._TARGET_BALANCE:
            low = center_freq / np.sqrt(2)
            high = center_freq * np.sqrt(2)
            mask = (freqs >= low) & (freqs < high)
            if np.any(mask):
                energy = np.mean(avg_spectrum[mask] ** 2)
                band_energies[center_freq] = 10.0 * np.log10(energy + 1e-10)

        if not band_energies:
            return 50.0

        # Normalize to 1kHz band
        ref_db = band_energies.get(1000, 0.0)
        normalized = {f: db - ref_db for f, db in band_energies.items()}

        # Compare against target curve
        total_deviation = 0.0
        n_bands = 0
        for freq, target_db in self._TARGET_BALANCE.items():
            if freq in normalized:
                deviation = abs(normalized[freq] - target_db)
                total_deviation += deviation
                n_bands += 1

        if n_bands == 0:
            return 50.0

        avg_deviation = total_deviation / n_bands

        # Score: 0 deviation = 100, 12 dB deviation = 0
        score = max(0.0, min(100.0, 100.0 - (avg_deviation / 12.0) * 100.0))
        return float(score)

    def stereo_correlation(self, left, right):
        """
        Compute stereo correlation coefficient between L and R channels.

        Values:
        +1.0 = mono (identical L/R)
        0.0 = uncorrelated
        -1.0 = out of phase

        Ideal mix typically has correlation between 0.3 and 0.95.

        Args:
            left: 1D numpy array, left channel
            right: 1D numpy array, right channel

        Returns:
            correlation: float -1.0 to 1.0
        """
        left = np.asarray(left, dtype=np.float64)
        right = np.asarray(right, dtype=np.float64)

        # Remove DC offset
        left = left - np.mean(left)
        right = right - np.mean(right)

        # Pearson correlation
        l_norm = np.sqrt(np.sum(left ** 2) + 1e-10)
        r_norm = np.sqrt(np.sum(right ** 2) + 1e-10)
        correlation = np.sum(left * right) / (l_norm * r_norm)

        return float(np.clip(correlation, -1.0, 1.0))

    def loudness_range(self, audio, sr=None):
        """
        Estimate Loudness Range (LRA) in LU (Loudness Units).

        LRA measures the dynamic range of the program material.
        Based on EBU R128 methodology (simplified):
        - Compute short-term loudness in 3-second windows with 1-second hop
        - Gate at absolute threshold -70 LUFS and relative threshold -20 LU
        - LRA = difference between 95th and 10th percentile of gated distribution

        Args:
            audio: 1D or 2D numpy array
            sr: sample rate

        Returns:
            lra: float, loudness range in LU
        """
        sr = sr or self.sr
        audio = np.asarray(audio, dtype=np.float64)

        if audio.ndim == 2:
            audio = np.mean(audio, axis=0)

        # Short-term loudness in 3s windows, 1s hop
        window_samples = int(3.0 * sr)
        hop_samples = int(1.0 * sr)

        if len(audio) < window_samples:
            # Too short, compute single loudness
            rms = np.sqrt(np.mean(audio ** 2) + 1e-10)
            return 0.0

        loudness_values = []
        pos = 0
        while pos + window_samples <= len(audio):
            segment = audio[pos: pos + window_samples]
            # K-weighted RMS approximation (simplified: pre-emphasis)
            # Apply simple high-shelf boost at 1500 Hz
            if HAS_SCIPY:
                b, a = scipy.signal.butter(2, 1500.0 / (sr / 2), btype="high")
                filtered = scipy.signal.lfilter(b, a, segment)
                rms = np.sqrt(np.mean(filtered ** 2) + 1e-10)
            else:
                rms = np.sqrt(np.mean(segment ** 2) + 1e-10)

            lufs = -0.691 + 10.0 * np.log10(rms ** 2 + 1e-10)
            loudness_values.append(lufs)
            pos += hop_samples

        if len(loudness_values) < 2:
            return 0.0

        loudness_values = np.array(loudness_values)

        # Absolute gate at -70 LUFS
        gated = loudness_values[loudness_values > -70.0]
        if len(gated) < 2:
            return 0.0

        # Relative gate: -20 LU below mean of absolutely gated values
        mean_loudness = np.mean(gated)
        relative_threshold = mean_loudness - 20.0
        final_gated = gated[gated > relative_threshold]

        if len(final_gated) < 2:
            return 0.0

        # LRA = 95th - 10th percentile
        p95 = np.percentile(final_gated, 95)
        p10 = np.percentile(final_gated, 10)
        lra = p95 - p10

        return float(max(0.0, lra))

    def crest_factor(self, audio):
        """
        Compute crest factor (peak-to-RMS ratio) in dB.

        Lower crest factor indicates more compressed/limited audio.
        Typical values: 12-20 dB for uncompressed, 6-10 dB for compressed.

        Args:
            audio: 1D or 2D numpy array

        Returns:
            crest_db: float, crest factor in dB
        """
        audio = np.asarray(audio, dtype=np.float64)
        if audio.ndim == 2:
            audio = np.mean(audio, axis=0)

        peak = np.max(np.abs(audio))
        rms = np.sqrt(np.mean(audio ** 2) + 1e-10)

        if rms < 1e-10:
            return 0.0

        crest_db = 20.0 * np.log10(peak / rms + 1e-10)
        return float(crest_db)

    def masking_score(self, channels_audio, sr=None):
        """
        Estimate spectral masking between channels (0-100).

        Higher score = less masking = better separation.
        Measures how much each channel's spectrum overlaps with others
        in the same frequency bands.

        Args:
            channels_audio: list of 1D numpy arrays, one per channel
            sr: sample rate

        Returns:
            score: float 0-100 (100 = no masking, 0 = total masking)
        """
        sr = sr or self.sr

        if len(channels_audio) < 2:
            return 100.0  # single channel, no masking possible

        n_fft = 4096
        n_bands = 20  # analysis bands (roughly 1/3 octave)

        # Compute average spectrum per channel
        channel_spectra = []
        for ch_audio in channels_audio:
            ch = np.asarray(ch_audio, dtype=np.float64)
            if ch.ndim == 2:
                ch = np.mean(ch, axis=0)

            if len(ch) < n_fft:
                ch = np.pad(ch, (0, n_fft - len(ch)))

            window = np.hanning(n_fft)
            n_frames = max(1, (len(ch) - n_fft) // (n_fft // 2))
            accum = np.zeros(n_fft // 2 + 1)
            for i in range(n_frames):
                start = i * (n_fft // 2)
                frame = ch[start: start + n_fft]
                if len(frame) < n_fft:
                    frame = np.pad(frame, (0, n_fft - len(frame)))
                accum += np.abs(np.fft.rfft(frame * window)) ** 2
            avg_mag = np.sqrt(accum / max(n_frames, 1))
            channel_spectra.append(avg_mag)

        # Define frequency bands (log-spaced)
        freqs = np.fft.rfftfreq(n_fft, d=1.0 / sr)
        band_edges = np.logspace(np.log10(20), np.log10(sr / 2), n_bands + 1)

        # Compute energy per band per channel
        band_energy = np.zeros((len(channels_audio), n_bands))
        for ch_idx, spec in enumerate(channel_spectra):
            for b in range(n_bands):
                mask = (freqs >= band_edges[b]) & (freqs < band_edges[b + 1])
                if np.any(mask):
                    band_energy[ch_idx, b] = np.sum(spec[mask] ** 2)

        # Normalize per channel
        for ch_idx in range(len(channels_audio)):
            total = np.sum(band_energy[ch_idx]) + 1e-10
            band_energy[ch_idx] /= total

        # Compute pairwise masking: for each pair of channels,
        # measure overlap in each band
        n_ch = len(channels_audio)
        total_overlap = 0.0
        n_pairs = 0

        for i in range(n_ch):
            for j in range(i + 1, n_ch):
                # Overlap = sum of min(energy_i, energy_j) per band
                overlap = np.sum(np.minimum(band_energy[i], band_energy[j]))
                total_overlap += overlap
                n_pairs += 1

        if n_pairs == 0:
            return 100.0

        avg_overlap = total_overlap / n_pairs

        # Score: 0 overlap = 100, 1.0 overlap = 0
        # Typical overlap is 0.1-0.5 for well-mixed content
        score = max(0.0, min(100.0, (1.0 - avg_overlap * 2.0) * 100.0))
        return float(score)

    def a_weighted_level(self, audio, sr=None):
        """
        Compute A-weighted sound level in dBFS.

        A-weighting approximates human hearing sensitivity
        (rolls off low and very high frequencies).

        Args:
            audio: 1D or 2D numpy array
            sr: sample rate

        Returns:
            level_dbfs: float, A-weighted level in dBFS
        """
        sr = sr or self.sr
        audio = np.asarray(audio, dtype=np.float64)
        if audio.ndim == 2:
            audio = np.mean(audio, axis=0)

        if len(audio) == 0:
            return -100.0

        # Apply A-weighting filter
        audio_weighted = self._a_weight_filter(audio, sr)

        # RMS level
        rms = np.sqrt(np.mean(audio_weighted ** 2) + 1e-10)
        level_dbfs = 20.0 * np.log10(rms + 1e-10)

        return float(level_dbfs)

    def _a_weight_filter(self, audio, sr):
        """
        Apply A-weighting filter to audio signal.

        Implements the IEC 61672 A-weighting curve using cascaded
        second-order sections.
        """
        if HAS_SCIPY:
            # Design A-weighting filter using analog prototype
            # A-weighting corner frequencies
            f1 = 20.598997
            f2 = 107.65265
            f3 = 737.86223
            f4 = 12194.217

            # Analog A-weighting transfer function poles/zeros
            nums = [(2 * np.pi * f4) ** 2]
            dens = [1.0,
                    2 * np.pi * (f1 + f1),
                    (2 * np.pi * f1) ** 2]

            # Simplified: use IIR approximation via bilinear transform
            # Create cascaded high-pass and low-pass stages
            # Stage 1: 2nd order high-pass at ~20.6 Hz
            b1, a1 = scipy.signal.butter(
                2, f1 * 2.0 / sr, btype="high"
            )
            # Stage 2: 2nd order high-pass at ~107.6 Hz
            nyq = sr / 2.0
            f2_norm = min(f2 / nyq, 0.99)
            b2, a2 = scipy.signal.butter(2, f2_norm, btype="high")
            # Stage 3: 2nd order low-pass at ~12194 Hz
            f4_norm = min(f4 / nyq, 0.99)
            b3, a3 = scipy.signal.butter(2, f4_norm, btype="low")

            filtered = scipy.signal.lfilter(b1, a1, audio)
            filtered = scipy.signal.lfilter(b2, a2, filtered)
            filtered = scipy.signal.lfilter(b3, a3, filtered)

            return filtered
        else:
            # Simple frequency-domain A-weighting approximation
            n = len(audio)
            spectrum = np.fft.rfft(audio)
            freqs = np.fft.rfftfreq(n, d=1.0 / sr)
            freqs = np.maximum(freqs, 1e-6)  # avoid division by zero

            # A-weighting formula (IEC 61672)
            f2 = freqs ** 2
            a_weight_num = 12194.0 ** 2 * f2 ** 2
            a_weight_den = (
                (f2 + 20.6 ** 2)
                * np.sqrt((f2 + 107.7 ** 2) * (f2 + 737.9 ** 2))
                * (f2 + 12194.0 ** 2)
            )
            a_weight = a_weight_num / (a_weight_den + 1e-10)

            # Normalize so 1kHz = 0 dB
            f1k = 1000.0
            f1k2 = f1k ** 2
            ref_num = 12194.0 ** 2 * f1k2 ** 2
            ref_den = (
                (f1k2 + 20.6 ** 2)
                * np.sqrt((f1k2 + 107.7 ** 2) * (f1k2 + 737.9 ** 2))
                * (f1k2 + 12194.0 ** 2)
            )
            ref = ref_num / (ref_den + 1e-10)
            a_weight = a_weight / (ref + 1e-10)

            weighted_spectrum = spectrum * a_weight
            return np.fft.irfft(weighted_spectrum, n=n)

    def overall(self, mix_audio, sr=None, channels=None):
        """
        Compute overall mix quality score with all sub-metrics.

        Args:
            mix_audio: 2D numpy array (2, samples) stereo or 1D mono
            sr: sample rate
            channels: optional list of 1D numpy arrays for individual channels

        Returns:
            dict with keys:
                'spectral_balance': 0-100
                'stereo_correlation': -1 to 1
                'loudness_range_lu': LRA in LU
                'crest_factor_db': crest factor in dB
                'masking_score': 0-100 (only if channels provided)
                'a_weighted_level_dbfs': dBFS A-weighted
                'overall_score': 0-100 combined score
        """
        sr = sr or self.sr
        audio = np.asarray(mix_audio, dtype=np.float64)

        results = {}

        # Spectral balance
        results["spectral_balance"] = self.spectral_balance(audio, sr)

        # Stereo correlation
        if audio.ndim == 2 and audio.shape[0] == 2:
            results["stereo_correlation"] = self.stereo_correlation(
                audio[0], audio[1]
            )
        else:
            results["stereo_correlation"] = 1.0  # mono

        # Loudness range
        results["loudness_range_lu"] = self.loudness_range(audio, sr)

        # Crest factor
        results["crest_factor_db"] = self.crest_factor(audio)

        # A-weighted level
        results["a_weighted_level_dbfs"] = self.a_weighted_level(audio, sr)

        # Masking score (only if individual channels provided)
        if channels is not None and len(channels) >= 2:
            results["masking_score"] = self.masking_score(channels, sr)
        else:
            results["masking_score"] = None

        # Compute overall score (weighted combination)
        overall = self._compute_overall_score(results)
        results["overall_score"] = overall

        return results

    def _compute_overall_score(self, metrics):
        """
        Combine individual metrics into a single 0-100 score.

        Weights reflect relative importance in live concert mixing:
        - Spectral balance: 30% (most important for sound quality)
        - Stereo correlation: 15% (should be positive but not mono)
        - Loudness range: 15% (preserve dynamics)
        - Crest factor: 15% (avoid over-compression)
        - Masking: 25% (channel separation)
        """
        score = 0.0
        total_weight = 0.0

        # Spectral balance (already 0-100)
        score += 0.30 * metrics["spectral_balance"]
        total_weight += 0.30

        # Stereo correlation: ideal is 0.3-0.8
        corr = metrics["stereo_correlation"]
        if 0.3 <= corr <= 0.8:
            corr_score = 100.0
        elif corr > 0.8:
            corr_score = 100.0 - (corr - 0.8) * 200.0  # penalty for too mono
        elif corr >= 0.0:
            corr_score = corr / 0.3 * 100.0
        else:
            corr_score = max(0.0, (1.0 + corr) * 50.0)  # negative = phase issues
        corr_score = max(0.0, min(100.0, corr_score))
        score += 0.15 * corr_score
        total_weight += 0.15

        # Loudness range: ideal 7-18 LU for live music
        lra = metrics["loudness_range_lu"]
        if 7.0 <= lra <= 18.0:
            lra_score = 100.0
        elif lra < 7.0:
            lra_score = max(0.0, lra / 7.0 * 100.0)  # too compressed
        else:
            lra_score = max(0.0, 100.0 - (lra - 18.0) * 5.0)  # too dynamic
        score += 0.15 * lra_score
        total_weight += 0.15

        # Crest factor: ideal 10-18 dB for live music
        crest = metrics["crest_factor_db"]
        if 10.0 <= crest <= 18.0:
            crest_score = 100.0
        elif crest < 10.0:
            crest_score = max(0.0, crest / 10.0 * 100.0)
        else:
            crest_score = max(0.0, 100.0 - (crest - 18.0) * 5.0)
        score += 0.15 * crest_score
        total_weight += 0.15

        # Masking score (already 0-100)
        if metrics["masking_score"] is not None:
            score += 0.25 * metrics["masking_score"]
            total_weight += 0.25

        # Normalize
        if total_weight > 0:
            score = score / total_weight

        return float(max(0.0, min(100.0, score)))
