"""
Style Transfer - Reference-based mixing style transfer
========================================================
Analyzes a reference mix to extract its "style" (spectral balance, dynamics,
stereo width, per-instrument settings) and generates mixing parameters that
can be applied to a new set of channel audio to achieve a similar sound.
Generates OSC commands for the Behringer Wing Rack mixer.
"""

import json
import logging
import math
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

try:
    from scipy.signal import stft, welch
    from scipy.stats import pearsonr

    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False

logger = logging.getLogger(__name__)

# Instrument type constants
INSTRUMENT_TYPES = [
    "vocals", "kick", "snare", "hihat", "toms", "overheads",
    "bass", "electric_guitar", "acoustic_guitar", "keys",
    "strings", "brass", "woodwinds", "percussion", "other",
]


@dataclass
class InstrumentStyle:
    """Per-instrument style parameters extracted from a reference mix."""

    instrument_type: str
    gain_db: float = 0.0
    eq_low_shelf_db: float = 0.0  # Gain at ~100 Hz
    eq_low_mid_db: float = 0.0  # Gain at ~400 Hz
    eq_mid_db: float = 0.0  # Gain at ~1 kHz
    eq_high_mid_db: float = 0.0  # Gain at ~4 kHz
    eq_high_shelf_db: float = 0.0  # Gain at ~10 kHz
    compression_ratio: float = 1.0
    compression_threshold_db: float = -10.0
    gate_threshold_db: float = -60.0
    pan: float = 0.0  # -1..1
    bus_send_level: float = -96.0  # dB, for reverb/FX sends

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class StyleProfile:
    """
    Complete style profile extracted from a reference mix.
    Captures the overall aesthetic and per-instrument settings.
    """

    name: str = "untitled"
    spectral_balance: Dict[str, float] = field(default_factory=dict)
    # Keys: "sub_bass", "bass", "low_mid", "mid", "high_mid", "presence", "brilliance"
    dynamic_range: float = 0.0  # Overall dynamic range in dB
    stereo_width: float = 0.0  # 0..1 where 0=mono, 1=full stereo
    loudness_lufs: float = -14.0
    crest_factor: float = 10.0  # Peak to RMS ratio in dB
    per_instrument_settings: Dict[str, InstrumentStyle] = field(default_factory=dict)

    def to_dict(self) -> dict:
        result = {
            "name": self.name,
            "spectral_balance": self.spectral_balance,
            "dynamic_range": round(self.dynamic_range, 1),
            "stereo_width": round(self.stereo_width, 3),
            "loudness_lufs": round(self.loudness_lufs, 1),
            "crest_factor": round(self.crest_factor, 1),
            "per_instrument_settings": {
                k: v.to_dict() for k, v in self.per_instrument_settings.items()
            },
        }
        return result


# Spectral band definitions in Hz
SPECTRAL_BANDS = {
    "sub_bass": (20, 60),
    "bass": (60, 250),
    "low_mid": (250, 500),
    "mid": (500, 2000),
    "high_mid": (2000, 6000),
    "presence": (6000, 12000),
    "brilliance": (12000, 20000),
}


class StyleTransfer:
    """
    Reference-based style transfer for live mixing.

    Analyzes a reference mix to extract its style profile, then generates
    mixing parameters for a new set of channels to match that style.
    """

    def __init__(
        self,
        fft_size: int = 4096,
        hop_size: int = 1024,
        wing_channel_prefix: str = "/ch",
    ):
        """
        Args:
            fft_size: FFT size for spectral analysis.
            hop_size: Hop size for STFT.
            wing_channel_prefix: OSC address prefix for Wing mixer channels.
        """
        self.fft_size = fft_size
        self.hop_size = hop_size
        self.wing_channel_prefix = wing_channel_prefix

    def extract_style(
        self, reference_audio: np.ndarray, sr: int, name: str = "reference"
    ) -> StyleProfile:
        """
        Analyze a reference mix and extract its style profile.

        Args:
            reference_audio: Reference audio, mono (N,) or stereo (N,2) or (2,N).
            sr: Sample rate in Hz.
            name: Name for this style profile.

        Returns:
            StyleProfile capturing the reference's characteristics.
        """
        # Handle stereo/mono
        if reference_audio.ndim == 2:
            if reference_audio.shape[0] == 2 and reference_audio.shape[1] != 2:
                stereo = reference_audio.T.astype(np.float64)
            else:
                stereo = reference_audio.astype(np.float64)
            mono = np.mean(stereo, axis=1)
        else:
            mono = reference_audio.astype(np.float64)
            stereo = None

        profile = StyleProfile(name=name)

        # 1. Spectral balance
        profile.spectral_balance = self._analyze_spectral_balance(mono, sr)

        # 2. Dynamic range
        profile.dynamic_range = self._analyze_dynamic_range(mono)

        # 3. Stereo width
        if stereo is not None and stereo.shape[1] >= 2:
            profile.stereo_width = self._analyze_stereo_width(stereo)
        else:
            profile.stereo_width = 0.0

        # 4. Loudness (integrated LUFS approximation)
        profile.loudness_lufs = self._estimate_lufs(mono, sr)

        # 5. Crest factor
        peak = np.max(np.abs(mono))
        rms = np.sqrt(np.mean(mono ** 2))
        if rms > 1e-10:
            profile.crest_factor = 20.0 * math.log10(peak / rms)
        else:
            profile.crest_factor = 0.0

        logger.info(
            f"Extracted style '{name}': DR={profile.dynamic_range:.1f}dB, "
            f"width={profile.stereo_width:.2f}, LUFS={profile.loudness_lufs:.1f}"
        )
        return profile

    def apply_style(
        self,
        style_profile: StyleProfile,
        channel_audios: Dict[str, np.ndarray],
        channel_types: Dict[str, str],
        sr: int = 48000,
    ) -> Dict[str, Dict[str, Any]]:
        """
        Generate mixing parameters for channels to match a style profile.

        Args:
            style_profile: The target style to match.
            channel_audios: Dict of channel_name -> mono audio array.
            channel_types: Dict of channel_name -> instrument type string.
            sr: Sample rate.

        Returns:
            Dict of channel_name -> mixing parameters dict with keys:
                fader_db, eq_bands, compression, gate_threshold, pan, bus_send.
        """
        mixing_params = {}

        # Compute spectral balance of current channels combined
        all_audio = []
        for name, audio in channel_audios.items():
            all_audio.append(audio.astype(np.float64))

        if all_audio:
            max_len = max(len(a) for a in all_audio)
            padded = [np.pad(a, (0, max_len - len(a))) for a in all_audio]
            current_mix = np.sum(padded, axis=0) / len(padded)
            current_balance = self._analyze_spectral_balance(current_mix, sr)
        else:
            current_balance = {band: 0.0 for band in SPECTRAL_BANDS}

        # Compute spectral correction needed
        spectral_correction = {}
        for band in SPECTRAL_BANDS:
            target = style_profile.spectral_balance.get(band, 0.0)
            current = current_balance.get(band, 0.0)
            spectral_correction[band] = target - current

        for ch_name, audio in channel_audios.items():
            inst_type = channel_types.get(ch_name, "other")
            params = self._compute_channel_params(
                audio, inst_type, style_profile, spectral_correction, sr
            )
            mixing_params[ch_name] = params

        return mixing_params

    def save_preset(self, style_profile: StyleProfile, path: str) -> None:
        """
        Save a style profile to a JSON file.

        Args:
            style_profile: Profile to save.
            path: File path to save to.
        """
        data = style_profile.to_dict()
        filepath = Path(path)
        filepath.parent.mkdir(parents=True, exist_ok=True)
        with open(filepath, "w") as f:
            json.dump(data, f, indent=2)
        logger.info(f"Saved style preset to {path}")

    def load_preset(self, path: str) -> StyleProfile:
        """
        Load a style profile from a JSON file.

        Args:
            path: File path to load from.

        Returns:
            Loaded StyleProfile.
        """
        with open(path, "r") as f:
            data = json.load(f)

        profile = StyleProfile(
            name=data.get("name", "loaded"),
            spectral_balance=data.get("spectral_balance", {}),
            dynamic_range=data.get("dynamic_range", 0.0),
            stereo_width=data.get("stereo_width", 0.0),
            loudness_lufs=data.get("loudness_lufs", -14.0),
            crest_factor=data.get("crest_factor", 10.0),
        )

        for inst_name, inst_data in data.get("per_instrument_settings", {}).items():
            profile.per_instrument_settings[inst_name] = InstrumentStyle(
                instrument_type=inst_data.get("instrument_type", inst_name),
                gain_db=inst_data.get("gain_db", 0.0),
                eq_low_shelf_db=inst_data.get("eq_low_shelf_db", 0.0),
                eq_low_mid_db=inst_data.get("eq_low_mid_db", 0.0),
                eq_mid_db=inst_data.get("eq_mid_db", 0.0),
                eq_high_mid_db=inst_data.get("eq_high_mid_db", 0.0),
                eq_high_shelf_db=inst_data.get("eq_high_shelf_db", 0.0),
                compression_ratio=inst_data.get("compression_ratio", 1.0),
                compression_threshold_db=inst_data.get("compression_threshold_db", -10.0),
                gate_threshold_db=inst_data.get("gate_threshold_db", -60.0),
                pan=inst_data.get("pan", 0.0),
                bus_send_level=inst_data.get("bus_send_level", -96.0),
            )

        logger.info(f"Loaded style preset from {path}: '{profile.name}'")
        return profile

    def generate_wing_osc(
        self, mixing_params: Dict[str, Dict[str, Any]], channel_map: Optional[Dict[str, int]] = None
    ) -> List[Tuple[str, Any]]:
        """
        Convert mixing parameters into OSC commands for the Behringer Wing Rack.

        Args:
            mixing_params: Dict of channel_name -> params dict from apply_style().
            channel_map: Optional mapping of channel_name -> Wing channel number (1-based).
                         If not provided, channels are assigned sequentially.

        Returns:
            List of (osc_address, value) tuples to send to the Wing mixer.
        """
        osc_commands = []

        # Build channel map if not provided
        if channel_map is None:
            channel_map = {}
            for i, name in enumerate(mixing_params.keys(), start=1):
                channel_map[name] = i

        for ch_name, params in mixing_params.items():
            ch_num = channel_map.get(ch_name)
            if ch_num is None:
                logger.warning(f"No channel mapping for '{ch_name}', skipping")
                continue

            # Pad channel number to 2 digits for Wing addressing
            ch_str = f"{ch_num:02d}"
            prefix = f"{self.wing_channel_prefix}/{ch_str}"

            # Fader level (Wing uses 0.0-1.0 float, we convert from dB)
            fader_db = params.get("fader_db", 0.0)
            # Wing fader mapping: -144dB = 0.0, 0dB = 0.75, +10dB = 1.0
            fader_float = self._db_to_wing_fader(fader_db)
            osc_commands.append((f"{prefix}/fader", fader_float))

            # Pan (-100..100 on Wing, -1..1 internally)
            pan = params.get("pan", 0.0)
            wing_pan = pan * 100.0
            osc_commands.append((f"{prefix}/pan", wing_pan))

            # EQ bands
            eq_bands = params.get("eq_bands", [])
            for i, band in enumerate(eq_bands[:6]):  # Wing has 6 EQ bands per channel
                band_num = i + 1
                osc_commands.append(
                    (f"{prefix}/eq/band{band_num}/freq", band.get("frequency", 1000.0))
                )
                osc_commands.append(
                    (f"{prefix}/eq/band{band_num}/gain", band.get("gain_db", 0.0))
                )
                osc_commands.append(
                    (f"{prefix}/eq/band{band_num}/q", band.get("q", 1.0))
                )

            # Compressor
            comp = params.get("compression", {})
            if comp:
                osc_commands.append(
                    (f"{prefix}/dyn/comp/thr", comp.get("threshold_db", -10.0))
                )
                osc_commands.append(
                    (f"{prefix}/dyn/comp/ratio", comp.get("ratio", 2.0))
                )
                osc_commands.append(
                    (f"{prefix}/dyn/comp/attack", comp.get("attack_ms", 10.0))
                )
                osc_commands.append(
                    (f"{prefix}/dyn/comp/release", comp.get("release_ms", 100.0))
                )

            # Gate
            gate_threshold = params.get("gate_threshold", -60.0)
            osc_commands.append((f"{prefix}/dyn/gate/thr", gate_threshold))

            # Bus sends
            bus_send = params.get("bus_send", {})
            for bus_name, send_level in bus_send.items():
                # Bus send format: /ch/XX/send/Y/level
                osc_commands.append(
                    (f"{prefix}/send/{bus_name}/level", send_level)
                )

        logger.info(f"Generated {len(osc_commands)} OSC commands for {len(mixing_params)} channels")
        return osc_commands

    # ---- Internal analysis methods ----

    def _analyze_spectral_balance(self, mono: np.ndarray, sr: int) -> Dict[str, float]:
        """Compute spectral energy distribution across frequency bands in dB."""
        n_fft = self.fft_size

        if HAS_SCIPY:
            freqs, psd = welch(mono, fs=sr, nperseg=n_fft, noverlap=n_fft // 2)
        else:
            # Fallback: basic periodogram with numpy
            n_frames = max(1, (len(mono) - n_fft) // self.hop_size)
            window = np.hanning(n_fft)
            psd = np.zeros(n_fft // 2 + 1)
            for i in range(n_frames):
                start = i * self.hop_size
                frame = mono[start: start + n_fft] * window
                spectrum = np.abs(np.fft.rfft(frame)) ** 2
                psd += spectrum
            if n_frames > 0:
                psd /= n_frames
            freqs = np.fft.rfftfreq(n_fft, 1.0 / sr)

        balance = {}
        total_energy = np.sum(psd) + 1e-20

        for band_name, (low, high) in SPECTRAL_BANDS.items():
            mask = (freqs >= low) & (freqs < high)
            band_energy = np.sum(psd[mask])
            # Relative energy in dB
            if band_energy > 1e-20:
                balance[band_name] = float(10.0 * np.log10(band_energy / total_energy))
            else:
                balance[band_name] = -80.0

        return balance

    def _analyze_dynamic_range(self, mono: np.ndarray) -> float:
        """Compute dynamic range as the difference between loud and quiet percentiles in dB."""
        eps = 1e-10
        abs_signal = np.abs(mono)

        # Use short-term RMS windows (50ms at 48kHz ~ 2400 samples)
        window_size = max(1, len(mono) // 200)
        n_windows = max(1, len(mono) // window_size)

        rms_values = []
        for i in range(n_windows):
            start = i * window_size
            end = start + window_size
            segment = mono[start:end]
            rms = np.sqrt(np.mean(segment ** 2))
            if rms > eps:
                rms_values.append(20.0 * math.log10(rms))

        if not rms_values:
            return 0.0

        rms_arr = np.array(rms_values)
        # Dynamic range: difference between 95th and 10th percentile
        dr = float(np.percentile(rms_arr, 95) - np.percentile(rms_arr, 10))
        return max(0.0, dr)

    def _analyze_stereo_width(self, stereo: np.ndarray) -> float:
        """
        Analyze stereo width from 0 (mono) to 1 (full stereo).
        Uses mid/side energy ratio.
        """
        left = stereo[:, 0]
        right = stereo[:, 1]

        mid = (left + right) / 2.0
        side = (left - right) / 2.0

        mid_energy = np.sum(mid ** 2)
        side_energy = np.sum(side ** 2)
        total = mid_energy + side_energy

        if total < 1e-20:
            return 0.0

        # Width = side_energy / total_energy
        width = side_energy / total
        # Scale: 0 = pure mono, 0.5 = equal mid/side, approaching 1 = mostly side
        # Normalize to 0..1 range where typical stereo mix is ~0.3-0.5
        return float(min(1.0, width * 2.0))

    def _estimate_lufs(self, mono: np.ndarray, sr: int) -> float:
        """
        Estimate integrated loudness in LUFS using K-weighting approximation.

        A simplified LUFS measurement based on ITU-R BS.1770.
        """
        eps = 1e-10
        audio = mono.copy()

        # K-weighting filter approximation
        # Stage 1: High-shelf boost at ~1.5 kHz (+4 dB)
        if HAS_SCIPY:
            from scipy.signal import butter, sosfilt

            # High shelf approximation using high-pass + original
            nyquist = sr / 2.0
            cutoff = min(1500.0 / nyquist, 0.99)
            if cutoff > 0.01:
                sos_hp = butter(2, cutoff, btype="high", output="sos")
                high_part = sosfilt(sos_hp, audio)
                # Boost high frequencies by ~4 dB (factor of ~1.585)
                audio = audio + high_part * 0.585

        # Gate and measure
        # Block size: 400ms with 75% overlap
        block_samples = int(0.4 * sr)
        hop = int(0.1 * sr)
        n_blocks = max(1, (len(audio) - block_samples) // hop)

        block_loudness = []
        for i in range(n_blocks):
            start = i * hop
            block = audio[start: start + block_samples]
            mean_sq = np.mean(block ** 2)
            if mean_sq > eps:
                block_loudness.append(-0.691 + 10.0 * math.log10(mean_sq))

        if not block_loudness:
            return -70.0

        block_arr = np.array(block_loudness)

        # Absolute gate at -70 LUFS
        gated = block_arr[block_arr > -70.0]
        if len(gated) == 0:
            return -70.0

        # Relative gate: -10 dB below ungated mean
        ungated_mean = float(np.mean(10.0 ** (gated / 10.0)))
        if ungated_mean > eps:
            relative_threshold = 10.0 * math.log10(ungated_mean) - 10.0
        else:
            return -70.0

        final_gated = gated[gated > relative_threshold]
        if len(final_gated) == 0:
            return -70.0

        integrated = float(np.mean(10.0 ** (final_gated / 10.0)))
        if integrated > eps:
            lufs = -0.691 + 10.0 * math.log10(integrated)
        else:
            lufs = -70.0

        return round(lufs, 1)

    def _compute_channel_params(
        self,
        audio: np.ndarray,
        instrument_type: str,
        style_profile: StyleProfile,
        spectral_correction: Dict[str, float],
        sr: int,
    ) -> Dict[str, Any]:
        """
        Compute mixing parameters for a single channel based on the style profile.
        """
        audio_f64 = audio.astype(np.float64)

        # Check if we have per-instrument settings
        inst_style = style_profile.per_instrument_settings.get(instrument_type)

        # Default parameters
        params: Dict[str, Any] = {
            "fader_db": 0.0,
            "pan": 0.0,
            "eq_bands": [],
            "compression": {},
            "gate_threshold": -60.0,
            "bus_send": {},
        }

        # --- Fader ---
        # Target level based on style profile loudness
        rms = np.sqrt(np.mean(audio_f64 ** 2))
        if rms > 1e-10:
            current_lufs_approx = 20.0 * math.log10(rms) - 0.691
        else:
            current_lufs_approx = -70.0

        # Instrument-specific gain offsets
        instrument_offsets = {
            "vocals": 0.0,
            "kick": -2.0,
            "snare": -1.0,
            "hihat": -6.0,
            "toms": -3.0,
            "overheads": -5.0,
            "bass": -2.0,
            "electric_guitar": -3.0,
            "acoustic_guitar": -4.0,
            "keys": -4.0,
            "strings": -3.0,
            "brass": -2.0,
            "woodwinds": -3.0,
            "percussion": -5.0,
            "other": -4.0,
        }
        offset = instrument_offsets.get(instrument_type, -4.0)

        if inst_style:
            params["fader_db"] = inst_style.gain_db
            params["pan"] = inst_style.pan
            params["gate_threshold"] = inst_style.gate_threshold_db
            params["bus_send"] = {"1": inst_style.bus_send_level}
        else:
            target_lufs = style_profile.loudness_lufs + offset
            params["fader_db"] = round(target_lufs - current_lufs_approx, 1)
            params["fader_db"] = max(-96.0, min(10.0, params["fader_db"]))

        # --- EQ ---
        # Map spectral correction to EQ bands
        eq_freq_map = {
            "sub_bass": {"frequency": 60.0, "type": "low_shelf"},
            "bass": {"frequency": 150.0, "type": "peak"},
            "low_mid": {"frequency": 400.0, "type": "peak"},
            "mid": {"frequency": 1000.0, "type": "peak"},
            "high_mid": {"frequency": 4000.0, "type": "peak"},
            "presence": {"frequency": 8000.0, "type": "peak"},
            "brilliance": {"frequency": 12000.0, "type": "high_shelf"},
        }

        eq_bands = []
        if inst_style:
            eq_values = [
                ("low_shelf", 100.0, inst_style.eq_low_shelf_db),
                ("peak", 400.0, inst_style.eq_low_mid_db),
                ("peak", 1000.0, inst_style.eq_mid_db),
                ("peak", 4000.0, inst_style.eq_high_mid_db),
                ("high_shelf", 10000.0, inst_style.eq_high_shelf_db),
            ]
            for band_type, freq, gain in eq_values:
                if abs(gain) > 0.3:
                    eq_bands.append({
                        "frequency": freq,
                        "gain_db": round(gain, 1),
                        "q": 1.0,
                        "type": band_type,
                    })
        else:
            for band_name, correction in spectral_correction.items():
                if abs(correction) < 0.5:
                    continue
                mapping = eq_freq_map.get(band_name)
                if mapping is None:
                    continue
                # Scale correction for individual channel (divide by approx number of channels)
                scaled = correction * 0.3  # Don't apply full correction to each channel
                scaled = max(-12.0, min(12.0, scaled))
                eq_bands.append({
                    "frequency": mapping["frequency"],
                    "gain_db": round(scaled, 1),
                    "q": 1.5,
                    "type": mapping["type"],
                })

        params["eq_bands"] = eq_bands

        # --- Compression ---
        # Set compression based on instrument type and style DR target
        comp_presets = {
            "vocals": {"ratio": 3.0, "threshold_db": -18.0, "attack_ms": 5.0, "release_ms": 80.0},
            "kick": {"ratio": 4.0, "threshold_db": -15.0, "attack_ms": 2.0, "release_ms": 50.0},
            "snare": {"ratio": 3.5, "threshold_db": -15.0, "attack_ms": 1.0, "release_ms": 60.0},
            "bass": {"ratio": 4.0, "threshold_db": -20.0, "attack_ms": 10.0, "release_ms": 100.0},
            "electric_guitar": {"ratio": 2.5, "threshold_db": -18.0, "attack_ms": 8.0, "release_ms": 80.0},
            "acoustic_guitar": {"ratio": 2.0, "threshold_db": -20.0, "attack_ms": 15.0, "release_ms": 100.0},
        }

        if inst_style:
            params["compression"] = {
                "ratio": inst_style.compression_ratio,
                "threshold_db": inst_style.compression_threshold_db,
                "attack_ms": 10.0,
                "release_ms": 100.0,
            }
        else:
            comp = comp_presets.get(instrument_type, {
                "ratio": 2.0, "threshold_db": -20.0, "attack_ms": 10.0, "release_ms": 100.0
            })
            # Adjust based on target dynamic range
            if style_profile.dynamic_range < 10.0:
                # Heavily compressed style -> increase ratio
                comp["ratio"] = min(comp["ratio"] * 1.5, 20.0)
                comp["threshold_db"] += 5.0
            elif style_profile.dynamic_range > 25.0:
                # Dynamic style -> less compression
                comp["ratio"] = max(1.0, comp["ratio"] * 0.7)
                comp["threshold_db"] -= 5.0

            params["compression"] = comp

        return params

    @staticmethod
    def _db_to_wing_fader(db: float) -> float:
        """
        Convert dB value to Wing fader float (0.0..1.0).
        Wing fader mapping: -144dB=0.0, -60dB=0.25, -30dB=0.5, 0dB=0.75, +10dB=1.0
        Uses a logarithmic curve approximation.
        """
        if db <= -144.0:
            return 0.0
        if db >= 10.0:
            return 1.0

        # Piecewise linear approximation of Wing fader curve
        if db <= -60.0:
            # -144 to -60 maps to 0.0 to 0.25
            return 0.25 * (db + 144.0) / 84.0
        elif db <= -30.0:
            # -60 to -30 maps to 0.25 to 0.5
            return 0.25 + 0.25 * (db + 60.0) / 30.0
        elif db <= 0.0:
            # -30 to 0 maps to 0.5 to 0.75
            return 0.5 + 0.25 * (db + 30.0) / 30.0
        else:
            # 0 to +10 maps to 0.75 to 1.0
            return 0.75 + 0.25 * db / 10.0
