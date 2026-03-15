"""
Reference spectral profiles for each instrument type and genre-level
mixing reference profiles.

Instrument profiles: pre-computed numpy arrays describing target spectral
shapes based on professional mixing literature and common EQ practice for
live concert sound reinforcement.

Genre profiles: typical frequency balance, dynamics, stereo width, and
compression parameters for common musical genres (rock, pop, jazz,
electronic, classical, acoustic, metal).  A ``ReferenceProfileManager``
class provides querying, comparison, interpolation, and closest-genre
matching.
"""

import logging
import math

logger = logging.getLogger(__name__)

try:
    import numpy as np
    HAS_NUMPY = True
except ImportError:
    HAS_NUMPY = False
    np = None  # pragma: no cover

# Common frequency points for all profiles (Hz)
# Covers 5 octaves at 1/3-octave resolution from ~25 Hz to ~20 kHz
FREQ_POINTS = np.array([
    25.0, 31.5, 40.0, 50.0, 63.0, 80.0, 100.0, 125.0, 160.0, 200.0,
    250.0, 315.0, 400.0, 500.0, 630.0, 800.0, 1000.0, 1250.0, 1600.0,
    2000.0, 2500.0, 3150.0, 4000.0, 5000.0, 6300.0, 8000.0, 10000.0,
    12500.0, 16000.0, 20000.0,
])

# Reference spectral profiles (dB relative to primary energy)
# Based on:
# - Bob McCarthy "Sound Systems: Design and Optimization"
# - Dave Rat mixing principles for live sound
# - Mixing engineers' consensus for FOH concert mixing
PROFILES = {
    "kick": (
        FREQ_POINTS.copy(),
        np.array([
            -6.0, -3.0, 0.0, 3.0, 6.0, 4.0, 2.0, 0.0, -3.0, -5.0,
            -8.0, -10.0, -12.0, -14.0, -15.0, -14.0, -13.0, -12.0, -11.0,
            -10.0, -12.0, -14.0, -8.0, -10.0, -14.0, -18.0, -22.0,
            -26.0, -30.0, -36.0,
        ]),
    ),
    "snare": (
        FREQ_POINTS.copy(),
        np.array([
            -30.0, -26.0, -22.0, -18.0, -14.0, -10.0, -6.0, -3.0, -1.0, 0.0,
            2.0, 1.0, 0.0, -1.0, -2.0, -1.0, 0.0, 1.0, 2.0,
            3.0, 2.0, 1.0, 0.0, -1.0, -2.0, -4.0, -6.0,
            -8.0, -12.0, -18.0,
        ]),
    ),
    "hihat": (
        FREQ_POINTS.copy(),
        np.array([
            -48.0, -44.0, -40.0, -38.0, -36.0, -34.0, -30.0, -26.0, -22.0, -18.0,
            -16.0, -14.0, -12.0, -10.0, -8.0, -6.0, -4.0, -3.0, -2.0,
            -1.0, 0.0, 1.0, 2.0, 3.0, 2.0, 1.0, 0.0,
            -1.0, -3.0, -6.0,
        ]),
    ),
    "toms": (
        FREQ_POINTS.copy(),
        np.array([
            -12.0, -8.0, -4.0, -1.0, 2.0, 4.0, 3.0, 1.0, 0.0, -1.0,
            -3.0, -4.0, -6.0, -8.0, -9.0, -8.0, -7.0, -6.0, -5.0,
            -4.0, -5.0, -6.0, -3.0, -4.0, -6.0, -8.0, -12.0,
            -16.0, -20.0, -26.0,
        ]),
    ),
    "overheads": (
        FREQ_POINTS.copy(),
        np.array([
            -24.0, -20.0, -16.0, -14.0, -12.0, -10.0, -8.0, -6.0, -4.0, -3.0,
            -2.0, -1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0,
            1.0, 2.0, 2.0, 3.0, 3.0, 2.0, 1.0, 0.0,
            -1.0, -3.0, -6.0,
        ]),
    ),
    "bass_guitar": (
        FREQ_POINTS.copy(),
        np.array([
            -6.0, -2.0, 2.0, 4.0, 6.0, 4.0, 2.0, 0.0, -1.0, -2.0,
            -4.0, -5.0, -6.0, -8.0, -10.0, -9.0, -8.0, -7.0, -6.0,
            -5.0, -6.0, -8.0, -10.0, -12.0, -14.0, -18.0, -22.0,
            -28.0, -34.0, -40.0,
        ]),
    ),
    "electric_guitar": (
        FREQ_POINTS.copy(),
        np.array([
            -36.0, -32.0, -28.0, -24.0, -20.0, -16.0, -10.0, -6.0, -3.0, -1.0,
            0.0, 1.0, 2.0, 2.0, 1.0, 0.0, 0.0, 1.0, 2.0,
            3.0, 4.0, 3.0, 2.0, 3.0, 2.0, 0.0, -2.0,
            -6.0, -10.0, -16.0,
        ]),
    ),
    "acoustic_guitar": (
        FREQ_POINTS.copy(),
        np.array([
            -36.0, -30.0, -24.0, -20.0, -16.0, -12.0, -6.0, -3.0, -1.0, 0.0,
            1.0, 2.0, 2.0, 1.0, 0.0, 0.0, 0.0, 1.0, 2.0,
            3.0, 3.0, 2.0, 1.0, 2.0, 1.0, 0.0, -2.0,
            -4.0, -8.0, -14.0,
        ]),
    ),
    "keys": (
        FREQ_POINTS.copy(),
        np.array([
            -24.0, -20.0, -16.0, -12.0, -8.0, -5.0, -3.0, -1.0, 0.0, 0.0,
            1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0,
            1.0, 0.0, 0.0, 0.0, 0.0, -1.0, -2.0, -4.0,
            -6.0, -10.0, -16.0,
        ]),
    ),
    "vocals": (
        FREQ_POINTS.copy(),
        np.array([
            -48.0, -42.0, -36.0, -30.0, -24.0, -18.0, -12.0, -8.0, -4.0, -2.0,
            -1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 2.0,
            3.0, 4.0, 4.0, 3.0, 2.0, 1.0, 0.0, -2.0,
            -4.0, -8.0, -14.0,
        ]),
    ),
    "brass": (
        FREQ_POINTS.copy(),
        np.array([
            -42.0, -38.0, -34.0, -30.0, -24.0, -18.0, -12.0, -8.0, -4.0, -2.0,
            0.0, 1.0, 2.0, 2.0, 1.0, 0.0, 0.0, 1.0, 2.0,
            4.0, 4.0, 3.0, 2.0, 1.0, 0.0, -2.0, -4.0,
            -8.0, -12.0, -18.0,
        ]),
    ),
    "strings": (
        FREQ_POINTS.copy(),
        np.array([
            -36.0, -32.0, -28.0, -22.0, -18.0, -14.0, -8.0, -4.0, -2.0, -1.0,
            0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0,
            2.0, 2.0, 1.0, 0.0, 0.0, -1.0, -2.0, -4.0,
            -6.0, -10.0, -16.0,
        ]),
    ),
    "percussion": (
        FREQ_POINTS.copy(),
        np.array([
            -24.0, -20.0, -16.0, -12.0, -8.0, -6.0, -4.0, -2.0, -1.0, 0.0,
            1.0, 2.0, 2.0, 1.0, 0.0, 0.0, 0.0, 1.0, 2.0,
            3.0, 4.0, 4.0, 3.0, 3.0, 2.0, 1.0, 0.0,
            -2.0, -4.0, -8.0,
        ]),
    ),
}


def get_profile(instrument_type):
    """
    Get the reference spectral profile for an instrument type.

    Args:
        instrument_type: string instrument name (e.g. "kick", "vocals")

    Returns:
        (frequencies, magnitudes_db): tuple of numpy arrays
            frequencies: 1D array of Hz values (30 points)
            magnitudes_db: 1D array of dB values (30 points)

    Raises:
        KeyError: if instrument_type not found in PROFILES
    """
    if instrument_type not in PROFILES:
        available = ", ".join(sorted(PROFILES.keys()))
        raise KeyError(
            f"Unknown instrument type '{instrument_type}'. "
            f"Available: {available}"
        )
    freqs, mags = PROFILES[instrument_type]
    return freqs.copy(), mags.copy()


def get_profile_interpolated(instrument_type, target_freqs):
    """
    Get a reference profile interpolated to arbitrary frequency points.

    Args:
        instrument_type: string instrument name
        target_freqs: 1D numpy array of desired frequency points (Hz)

    Returns:
        magnitudes_db: 1D numpy array of interpolated dB values
    """
    freqs, mags = get_profile(instrument_type)
    return np.interp(target_freqs, freqs, mags, left=mags[0], right=mags[-1])


def list_instruments():
    """Return sorted list of available instrument profile names."""
    return sorted(PROFILES.keys())


def compare_profiles(type_a, type_b):
    """
    Compute the spectral difference between two instrument profiles.

    Args:
        type_a: first instrument type
        type_b: second instrument type

    Returns:
        dict with keys:
            'mean_diff_db': average absolute dB difference
            'max_diff_db': maximum absolute dB difference
            'overlap_score': 0-1 spectral overlap (1 = identical)
            'frequencies': frequency points used
            'diff_db': per-frequency dB difference (a - b)
    """
    freqs_a, mags_a = get_profile(type_a)
    freqs_b, mags_b = get_profile(type_b)

    # They share the same frequency points
    diff = mags_a - mags_b
    abs_diff = np.abs(diff)

    # Overlap: convert to linear, compute correlation-like metric
    lin_a = 10.0 ** (mags_a / 20.0)
    lin_b = 10.0 ** (mags_b / 20.0)
    overlap = np.sum(np.minimum(lin_a, lin_b)) / (np.sum(np.maximum(lin_a, lin_b)) + 1e-10)

    return {
        "mean_diff_db": float(np.mean(abs_diff)),
        "max_diff_db": float(np.max(abs_diff)),
        "overlap_score": float(overlap),
        "frequencies": freqs_a.copy(),
        "diff_db": diff.copy(),
    }


# ======================================================================
# Genre-level reference profiles
# ======================================================================

# Each genre profile contains:
#   target_lufs          : overall target loudness (LUFS)
#   target_dynamic_range : expected dynamic range (dB)
#   frequency_balance    : dict of band_name -> relative dB target
#   stereo_width         : 0.0 (mono) to 1.0 (wide)
#   compression_ratio    : typical master-bus compression ratio
#   attack_ms            : master compressor attack time (ms)
#   release_ms           : master compressor release time (ms)
#   eq_character         : short text description of tonal goal

GENRE_PROFILES = {
    "rock": {
        "target_lufs": -14.0,
        "target_dynamic_range": 8.0,
        "frequency_balance": {
            "sub_bass":   -2.0,   # 20-60 Hz
            "bass":        1.0,   # 60-250 Hz
            "low_mid":     0.5,   # 250-500 Hz
            "mid":         0.0,   # 500-2k Hz
            "upper_mid":   1.5,   # 2k-4k Hz
            "presence":    1.0,   # 4k-8k Hz
            "brilliance": -1.0,   # 8k-20k Hz
        },
        "stereo_width": 0.70,
        "compression_ratio": 3.0,
        "attack_ms": 10.0,
        "release_ms": 100.0,
        "eq_character": "punchy, mid-forward, aggressive presence",
    },
    "pop": {
        "target_lufs": -14.0,
        "target_dynamic_range": 7.0,
        "frequency_balance": {
            "sub_bass":    0.0,
            "bass":        1.0,
            "low_mid":    -0.5,
            "mid":         0.0,
            "upper_mid":   2.0,
            "presence":    1.5,
            "brilliance":  0.5,
        },
        "stereo_width": 0.75,
        "compression_ratio": 3.5,
        "attack_ms": 8.0,
        "release_ms": 80.0,
        "eq_character": "bright, vocal-forward, tight low end",
    },
    "jazz": {
        "target_lufs": -18.0,
        "target_dynamic_range": 14.0,
        "frequency_balance": {
            "sub_bass":   -3.0,
            "bass":        0.5,
            "low_mid":     0.0,
            "mid":         0.0,
            "upper_mid":   0.5,
            "presence":    0.0,
            "brilliance": -0.5,
        },
        "stereo_width": 0.65,
        "compression_ratio": 1.5,
        "attack_ms": 20.0,
        "release_ms": 200.0,
        "eq_character": "warm, natural, minimal processing, open dynamics",
    },
    "electronic": {
        "target_lufs": -12.0,
        "target_dynamic_range": 6.0,
        "frequency_balance": {
            "sub_bass":    3.0,
            "bass":        2.0,
            "low_mid":    -1.0,
            "mid":        -0.5,
            "upper_mid":   1.0,
            "presence":    1.5,
            "brilliance":  2.0,
        },
        "stereo_width": 0.85,
        "compression_ratio": 4.0,
        "attack_ms": 5.0,
        "release_ms": 60.0,
        "eq_character": "heavy sub bass, wide stereo, bright top end",
    },
    "classical": {
        "target_lufs": -23.0,
        "target_dynamic_range": 18.0,
        "frequency_balance": {
            "sub_bass":   -4.0,
            "bass":       -1.0,
            "low_mid":     0.0,
            "mid":         0.0,
            "upper_mid":   0.0,
            "presence":   -0.5,
            "brilliance": -1.0,
        },
        "stereo_width": 0.80,
        "compression_ratio": 1.2,
        "attack_ms": 30.0,
        "release_ms": 300.0,
        "eq_character": "flat, transparent, preserve natural dynamics",
    },
    "acoustic": {
        "target_lufs": -16.0,
        "target_dynamic_range": 12.0,
        "frequency_balance": {
            "sub_bass":   -3.0,
            "bass":        0.0,
            "low_mid":     0.5,
            "mid":         0.5,
            "upper_mid":   1.0,
            "presence":    0.5,
            "brilliance":  0.0,
        },
        "stereo_width": 0.60,
        "compression_ratio": 2.0,
        "attack_ms": 15.0,
        "release_ms": 150.0,
        "eq_character": "warm, natural body, gentle presence lift",
    },
    "metal": {
        "target_lufs": -12.0,
        "target_dynamic_range": 6.0,
        "frequency_balance": {
            "sub_bass":    0.0,
            "bass":        2.0,
            "low_mid":    -1.0,
            "mid":         1.0,
            "upper_mid":   2.5,
            "presence":    2.0,
            "brilliance":  0.0,
        },
        "stereo_width": 0.80,
        "compression_ratio": 4.0,
        "attack_ms": 5.0,
        "release_ms": 70.0,
        "eq_character": "tight bass, scooped low-mid, aggressive upper-mid",
    },
}

# Frequency band edges (Hz) for the genre balance dictionaries
BAND_EDGES = {
    "sub_bass":   (20, 60),
    "bass":       (60, 250),
    "low_mid":    (250, 500),
    "mid":        (500, 2000),
    "upper_mid":  (2000, 4000),
    "presence":   (4000, 8000),
    "brilliance": (8000, 20000),
}


def list_genres():
    """Return a sorted list of all available genre names."""
    return sorted(GENRE_PROFILES.keys())


# ------------------------------------------------------------------
# ReferenceProfileManager
# ------------------------------------------------------------------

class ReferenceProfileManager:
    """
    Manages genre reference profiles and provides comparison utilities.

    Usage::

        rpm = ReferenceProfileManager()
        profile = rpm.get_profile("rock")
        genres = rpm.list_genres()
        deviations = rpm.compare_to_reference(measured_features, "rock")
    """

    def __init__(self, custom_profiles=None):
        """
        Args:
            custom_profiles: optional dict of additional genre profiles to
                merge with the built-in ones.  Keys are genre names (lowercase),
                values follow the same schema as ``GENRE_PROFILES`` entries.
        """
        self._profiles = dict(GENRE_PROFILES)
        if custom_profiles:
            for genre, profile in custom_profiles.items():
                self._profiles[genre.lower()] = profile
                logger.info("Added custom profile for genre '%s'", genre)

    # ---- query methods ---------------------------------------------------

    def get_profile(self, genre):
        """
        Return the genre reference profile dict (case-insensitive).

        Returns None if the genre is unknown.
        """
        return self._profiles.get(genre.lower()) if genre else None

    def list_genres(self):
        """Return a sorted list of all registered genre names."""
        return sorted(self._profiles.keys())

    def get_band_edges(self):
        """Return mapping of band names to (low_hz, high_hz) tuples."""
        return dict(BAND_EDGES)

    # ---- comparison methods ----------------------------------------------

    def compare_to_reference(self, features, genre):
        """
        Compare measured mix features against a genre reference and return
        per-parameter deviation scores.

        Each deviation is the absolute difference between the measured value
        and the reference target in the parameter's native unit (dB, LU,
        ratio, etc.).  A value of 0.0 means an exact match.

        Args:
            features: dict with some or all of the following keys:
                ``lufs``              -- measured integrated loudness (LUFS)
                ``dynamic_range``     -- measured dynamic range (dB / LRA)
                ``frequency_balance`` -- per-band relative dB dict
                ``stereo_width``      -- 0.0..1.0
                ``compression_ratio`` -- estimated/applied ratio
            genre: genre name string (case-insensitive).

        Returns:
            dict of deviation scores including an ``"overall"`` key
            (0--100, where 100 = perfect match).
            Empty dict if the genre is unknown.
        """
        profile = self.get_profile(genre)
        if profile is None:
            logger.warning("Unknown genre '%s'; cannot compare.", genre)
            return {}

        deviations = {}
        weighted_penalty = 0.0
        total_weight = 0.0

        # --- LUFS ---
        if "lufs" in features:
            ref = profile["target_lufs"]
            dev = abs(features["lufs"] - ref)
            deviations["lufs"] = dev
            weighted_penalty += 0.25 * min(dev / 12.0, 1.0)
            total_weight += 0.25

        # --- Dynamic range ---
        if "dynamic_range" in features:
            ref = profile["target_dynamic_range"]
            dev = abs(features["dynamic_range"] - ref)
            deviations["dynamic_range"] = dev
            weighted_penalty += 0.20 * min(dev / 12.0, 1.0)
            total_weight += 0.20

        # --- Frequency balance (per-band) ---
        if "frequency_balance" in features:
            ref_balance = profile["frequency_balance"]
            meas_balance = features["frequency_balance"]
            band_devs = {}
            band_penalty_sum = 0.0
            band_count = 0
            for band, ref_db in ref_balance.items():
                if band in meas_balance:
                    dev = abs(meas_balance[band] - ref_db)
                    band_devs[band] = dev
                    band_penalty_sum += min(dev / 6.0, 1.0)
                    band_count += 1
            deviations["frequency_balance"] = band_devs
            if band_count > 0:
                avg_band_penalty = band_penalty_sum / band_count
                weighted_penalty += 0.30 * avg_band_penalty
                total_weight += 0.30

        # --- Stereo width ---
        if "stereo_width" in features:
            ref = profile["stereo_width"]
            dev = abs(features["stereo_width"] - ref)
            deviations["stereo_width"] = dev
            weighted_penalty += 0.10 * min(dev / 0.5, 1.0)
            total_weight += 0.10

        # --- Compression ratio ---
        if "compression_ratio" in features:
            ref = profile["compression_ratio"]
            dev = abs(features["compression_ratio"] - ref)
            deviations["compression_ratio"] = dev
            weighted_penalty += 0.15 * min(dev / 3.0, 1.0)
            total_weight += 0.15

        # --- Overall score ---
        if total_weight > 0.0:
            normalised_penalty = weighted_penalty / total_weight
        else:
            normalised_penalty = 0.0
        deviations["overall"] = max(0.0, min(100.0, (1.0 - normalised_penalty) * 100.0))

        return deviations

    def closest_genre(self, features):
        """
        Find the genre whose reference profile best matches *features*.

        Returns:
            (genre_name, overall_score) for the best match,
            or (None, 0.0) if no genres are registered.
        """
        best_genre = None
        best_score = -1.0

        for genre in self._profiles:
            deviations = self.compare_to_reference(features, genre)
            score = deviations.get("overall", 0.0)
            if score > best_score:
                best_score = score
                best_genre = genre

        return best_genre, best_score

    def interpolate_profiles(self, genre_a, genre_b, t=0.5):
        """
        Linearly interpolate between two genre profiles.

        Args:
            genre_a: first genre name.
            genre_b: second genre name.
            t: interpolation factor 0.0 (pure A) to 1.0 (pure B).

        Returns:
            dict with interpolated profile parameters, or None if either
            genre is unknown.
        """
        pa = self.get_profile(genre_a)
        pb = self.get_profile(genre_b)
        if pa is None or pb is None:
            logger.warning(
                "Cannot interpolate: unknown genre (%s or %s).", genre_a, genre_b
            )
            return None

        t = max(0.0, min(1.0, t))

        interpolated = {
            "target_lufs": pa["target_lufs"] * (1.0 - t) + pb["target_lufs"] * t,
            "target_dynamic_range": (
                pa["target_dynamic_range"] * (1.0 - t)
                + pb["target_dynamic_range"] * t
            ),
            "stereo_width": pa["stereo_width"] * (1.0 - t) + pb["stereo_width"] * t,
            "compression_ratio": (
                pa["compression_ratio"] * (1.0 - t) + pb["compression_ratio"] * t
            ),
            "attack_ms": pa["attack_ms"] * (1.0 - t) + pb["attack_ms"] * t,
            "release_ms": pa["release_ms"] * (1.0 - t) + pb["release_ms"] * t,
            "eq_character": (
                "blend of %s (%.0f%%) and %s (%.0f%%)"
                % (genre_a, (1.0 - t) * 100, genre_b, t * 100)
            ),
        }

        # Interpolate per-band frequency balance
        bands_a = pa["frequency_balance"]
        bands_b = pb["frequency_balance"]
        all_bands = set(bands_a.keys()) | set(bands_b.keys())
        interpolated["frequency_balance"] = {
            band: (
                bands_a.get(band, 0.0) * (1.0 - t) + bands_b.get(band, 0.0) * t
            )
            for band in all_bands
        }

        return interpolated


# ------------------------------------------------------------------
# Standalone demo
# ------------------------------------------------------------------
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    # --- Instrument profiles demo ---
    print("=== Instrument spectral profiles ===")
    print("Available instruments:", list_instruments())

    # --- Genre profiles demo ---
    rpm = ReferenceProfileManager()
    print("\n=== Genre reference profiles ===")
    print("Available genres:", rpm.list_genres())

    for genre in rpm.list_genres():
        p = rpm.get_profile(genre)
        print(
            "  %-12s  LUFS=%+.1f  DR=%.0f dB  width=%.2f  ratio=%.1f:1"
            % (genre, p["target_lufs"], p["target_dynamic_range"],
               p["stereo_width"], p["compression_ratio"])
        )

    # Example comparison
    measured = {
        "lufs": -15.0,
        "dynamic_range": 9.0,
        "frequency_balance": {
            "sub_bass": -1.0, "bass": 1.5, "low_mid": 0.0,
            "mid": 0.5, "upper_mid": 1.0, "presence": 0.5,
            "brilliance": -0.5,
        },
        "stereo_width": 0.72,
        "compression_ratio": 2.8,
    }

    print("\n--- Comparing measured features against all genres ---")
    for genre in rpm.list_genres():
        result = rpm.compare_to_reference(measured, genre)
        print("  %-12s  overall=%.1f/100" % (genre, result["overall"]))

    best, score = rpm.closest_genre(measured)
    print("\nClosest genre: %s (score %.1f/100)" % (best, score))
