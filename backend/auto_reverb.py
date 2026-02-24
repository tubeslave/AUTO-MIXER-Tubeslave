"""
Auto Reverb Module - Intelligent Reverberation Control.

Based on IMP 7.5 (Intelligent Music Production, De Man, Reiss & Stables):

Best practices from the book:
- Reverb loudness: -14 LU as a suggested middle ground [82].
- It is better to err on the 'dry' side [12, 82, 230].
- Decay inversely correlated with spectral flux [12, 314].
  High spectral flux -> short reverb times.
- Percussive instruments require shorter and denser reverbs than
  sustained sounds [119].
- Low-frequency sounds require less reverb [119].
- Sparse and slow arrangements require more reverb [119].
- Predelay: 30-50 ms (just past the Haas zone) [119].
- Reverb signals best filtered with 200 Hz HPF and 5 kHz LPF [119].
- Reverb brightness should be higher for long reverb times and dull
  sounds [119].

From Benito & Reiss [70]:
  PSL (Probabilistic Soft Logic) rules derived from [82, 105, 119]:
  - Features: instrument type, bright, percussive, voiced descriptors.
  - Predict reverb type, decay, level, predelay with confidence.

References:
  [12] Pestana & Reiss (2014)
  [65] Chourdakis & Reiss - adaptive reverb via ML
  [69] Chourdakis & Reiss - reverb parameter mapping
  [70] Benito & Reiss - PSL-based reverb
  [82] De Man et al. - perceived reverb in multitrack mixes
  [105] Pestana et al. - delay preference linked to song tempo
  [119] Pestana - interviews with expert engineers
"""

import logging
import math
from typing import Any, Dict, List, Optional
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)

# Default reverb level relative to mix [82]:
# -14 LU as a suggested middle ground.
DEFAULT_REVERB_LEVEL_LU = -14.0

# Predelay range: 30-50 ms (Haas zone) [119].
PREDELAY_MIN_MS = 30.0
PREDELAY_MAX_MS = 50.0
PREDELAY_DEFAULT_MS = 40.0

# Reverb filter settings [119].
REVERB_HPF_HZ = 200.0
REVERB_LPF_HZ = 5000.0

# "Err on the dry side" safety margin [12, 82, 230].
DRY_SIDE_MARGIN_LU = 2.0


@dataclass
class InstrumentReverbProfile:
    """Reverb profile for an instrument type based on PSL rules [70]."""
    reverb_type: str      # "plate", "hall", "room", "spring", "chamber"
    decay_time_s: float   # Reverb decay time in seconds
    level_offset_lu: float  # Offset from DEFAULT_REVERB_LEVEL_LU
    predelay_ms: float    # Predelay in ms
    brightness: float     # 0.0 (dark) to 1.0 (bright)
    density: float        # 0.0 (sparse) to 1.0 (dense)


# PSL-derived reverb profiles per instrument type [70, 119].
# Based on best practices: percussive -> short/dense,
# sustained -> longer, low-freq -> less reverb.
INSTRUMENT_REVERB_PROFILES: Dict[str, InstrumentReverbProfile] = {
    # Vocals: moderate reverb, plate or hall, medium decay [119].
    "leadVocal": InstrumentReverbProfile(
        reverb_type="plate", decay_time_s=1.5, level_offset_lu=0.0,
        predelay_ms=40.0, brightness=0.6, density=0.7,
    ),
    "lead_vocal": InstrumentReverbProfile(
        reverb_type="plate", decay_time_s=1.5, level_offset_lu=0.0,
        predelay_ms=40.0, brightness=0.6, density=0.7,
    ),
    "backingVocal": InstrumentReverbProfile(
        reverb_type="plate", decay_time_s=1.8, level_offset_lu=2.0,
        predelay_ms=35.0, brightness=0.5, density=0.7,
    ),
    "backing_vocal": InstrumentReverbProfile(
        reverb_type="plate", decay_time_s=1.8, level_offset_lu=2.0,
        predelay_ms=35.0, brightness=0.5, density=0.7,
    ),

    # Percussive: short, dense reverb [119].
    "kick": InstrumentReverbProfile(
        reverb_type="room", decay_time_s=0.3, level_offset_lu=-4.0,
        predelay_ms=30.0, brightness=0.3, density=0.9,
    ),
    "snare": InstrumentReverbProfile(
        reverb_type="plate", decay_time_s=0.8, level_offset_lu=-2.0,
        predelay_ms=30.0, brightness=0.5, density=0.9,
    ),
    "tom": InstrumentReverbProfile(
        reverb_type="room", decay_time_s=0.6, level_offset_lu=-3.0,
        predelay_ms=30.0, brightness=0.4, density=0.8,
    ),
    "drums": InstrumentReverbProfile(
        reverb_type="room", decay_time_s=0.5, level_offset_lu=-3.0,
        predelay_ms=30.0, brightness=0.4, density=0.9,
    ),
    "hihat": InstrumentReverbProfile(
        reverb_type="room", decay_time_s=0.3, level_offset_lu=-5.0,
        predelay_ms=30.0, brightness=0.6, density=0.8,
    ),
    "ride": InstrumentReverbProfile(
        reverb_type="room", decay_time_s=0.4, level_offset_lu=-4.0,
        predelay_ms=30.0, brightness=0.6, density=0.7,
    ),
    "overhead": InstrumentReverbProfile(
        reverb_type="room", decay_time_s=0.5, level_offset_lu=-4.0,
        predelay_ms=30.0, brightness=0.5, density=0.7,
    ),
    "percussion": InstrumentReverbProfile(
        reverb_type="room", decay_time_s=0.5, level_offset_lu=-3.0,
        predelay_ms=32.0, brightness=0.5, density=0.8,
    ),

    # Low-frequency: less reverb [119].
    "bass": InstrumentReverbProfile(
        reverb_type="room", decay_time_s=0.3, level_offset_lu=-6.0,
        predelay_ms=30.0, brightness=0.2, density=0.5,
    ),

    # Sustained / melodic: longer reverb.
    "electricGuitar": InstrumentReverbProfile(
        reverb_type="plate", decay_time_s=1.2, level_offset_lu=-1.0,
        predelay_ms=35.0, brightness=0.5, density=0.6,
    ),
    "acousticGuitar": InstrumentReverbProfile(
        reverb_type="hall", decay_time_s=1.5, level_offset_lu=0.0,
        predelay_ms=40.0, brightness=0.5, density=0.6,
    ),
    "piano": InstrumentReverbProfile(
        reverb_type="hall", decay_time_s=1.8, level_offset_lu=0.0,
        predelay_ms=45.0, brightness=0.5, density=0.6,
    ),
    "keys": InstrumentReverbProfile(
        reverb_type="plate", decay_time_s=1.2, level_offset_lu=-1.0,
        predelay_ms=38.0, brightness=0.5, density=0.6,
    ),
    "synth": InstrumentReverbProfile(
        reverb_type="plate", decay_time_s=1.0, level_offset_lu=-2.0,
        predelay_ms=35.0, brightness=0.5, density=0.6,
    ),
    "pads": InstrumentReverbProfile(
        reverb_type="hall", decay_time_s=2.0, level_offset_lu=1.0,
        predelay_ms=45.0, brightness=0.4, density=0.5,
    ),
    "strings": InstrumentReverbProfile(
        reverb_type="hall", decay_time_s=2.2, level_offset_lu=1.0,
        predelay_ms=45.0, brightness=0.5, density=0.5,
    ),
    "brass": InstrumentReverbProfile(
        reverb_type="hall", decay_time_s=1.5, level_offset_lu=-1.0,
        predelay_ms=40.0, brightness=0.6, density=0.6,
    ),
    "sax": InstrumentReverbProfile(
        reverb_type="plate", decay_time_s=1.4, level_offset_lu=0.0,
        predelay_ms=40.0, brightness=0.5, density=0.6,
    ),
    "woodwinds": InstrumentReverbProfile(
        reverb_type="hall", decay_time_s=1.6, level_offset_lu=0.0,
        predelay_ms=42.0, brightness=0.5, density=0.6,
    ),

    # Room microphones: already contain natural reverb.
    "room": InstrumentReverbProfile(
        reverb_type="room", decay_time_s=0.2, level_offset_lu=-8.0,
        predelay_ms=30.0, brightness=0.3, density=0.4,
    ),
}

# Default profile for unknown instruments.
DEFAULT_REVERB_PROFILE = InstrumentReverbProfile(
    reverb_type="plate", decay_time_s=1.2, level_offset_lu=-2.0,
    predelay_ms=38.0, brightness=0.5, density=0.6,
)


@dataclass
class ReverbDecision:
    """Reverb settings decision for a single channel."""
    channel_id: int
    instrument: str
    reverb_type: str
    decay_time_s: float
    reverb_level_lu: float
    predelay_ms: float
    brightness: float
    density: float
    hpf_hz: float
    lpf_hz: float
    reason: str


class AutoReverb:
    """
    Intelligent auto-reverb based on IMP 7.5 best practices and PSL rules [70].

    Determines reverb parameters per channel based on instrument type
    and signal features.  Errs on the dry side [12, 82, 230].
    """

    def __init__(
        self,
        base_reverb_level_lu: float = DEFAULT_REVERB_LEVEL_LU,
        dry_side_margin_lu: float = DRY_SIDE_MARGIN_LU,
        hpf_hz: float = REVERB_HPF_HZ,
        lpf_hz: float = REVERB_LPF_HZ,
    ):
        # Apply "err on dry side" margin [12, 82, 230].
        self.base_reverb_level_lu = base_reverb_level_lu - dry_side_margin_lu
        self.hpf_hz = hpf_hz
        self.lpf_hz = lpf_hz
        self.decisions: Dict[int, ReverbDecision] = {}

    def calculate_reverb(
        self,
        channels: List[int],
        instrument_types: Dict[int, str],
        spectral_centroids: Optional[Dict[int, float]] = None,
        spectral_fluxes: Optional[Dict[int, float]] = None,
    ) -> Dict[int, ReverbDecision]:
        """
        Calculate reverb settings for a list of channels.

        Parameters
        ----------
        channels : list of int
            Mixer channel IDs.
        instrument_types : dict
            channel_id -> instrument name string.
        spectral_centroids : dict, optional
            channel_id -> spectral centroid Hz (for brightness adaptation).
        spectral_fluxes : dict, optional
            channel_id -> spectral flux value (for decay adaptation [12, 314]).

        Returns
        -------
        dict of channel_id -> ReverbDecision
        """
        if spectral_centroids is None:
            spectral_centroids = {}
        if spectral_fluxes is None:
            spectral_fluxes = {}

        decisions: Dict[int, ReverbDecision] = {}

        for ch_id in channels:
            instrument = instrument_types.get(ch_id, "unknown")
            profile = INSTRUMENT_REVERB_PROFILES.get(
                instrument, DEFAULT_REVERB_PROFILE,
            )

            decay = profile.decay_time_s
            level = self.base_reverb_level_lu + profile.level_offset_lu
            predelay = profile.predelay_ms
            brightness = profile.brightness
            density = profile.density

            # Adapt decay based on spectral flux [12, 314]:
            # "decay inversely correlated with spectral flux"
            flux = spectral_fluxes.get(ch_id, 0.0)
            if flux > 0:
                # Higher flux -> shorter decay.
                # Apply logarithmic transformation [12, 314].
                flux_factor = 1.0 / (1.0 + math.log1p(flux))
                decay = decay * max(0.3, min(1.0, flux_factor))

            # Adapt brightness based on centroid [119]:
            # "reverb brightness should be higher for dull sounds"
            centroid = spectral_centroids.get(ch_id, 0.0)
            if centroid > 0:
                # Low centroid (dull sound) -> brighter reverb.
                # High centroid (bright sound) -> darker reverb.
                if centroid < 1000.0:
                    brightness = min(1.0, brightness + 0.2)
                elif centroid > 4000.0:
                    brightness = max(0.0, brightness - 0.2)

            # Clamp predelay to Haas zone [119].
            predelay = max(PREDELAY_MIN_MS, min(PREDELAY_MAX_MS, predelay))

            decisions[ch_id] = ReverbDecision(
                channel_id=ch_id,
                instrument=instrument,
                reverb_type=profile.reverb_type,
                decay_time_s=round(decay, 2),
                reverb_level_lu=round(level, 1),
                predelay_ms=round(predelay, 1),
                brightness=round(brightness, 2),
                density=round(density, 2),
                hpf_hz=self.hpf_hz,
                lpf_hz=self.lpf_hz,
                reason=f"PSL rules [70] for {instrument}; decay adapted by flux [12,314]",
            )

        self.decisions = decisions
        return decisions

    def apply_to_mixer(
        self,
        mixer_client: Any,
        decisions: Optional[Dict[int, ReverbDecision]] = None,
    ) -> int:
        """
        Send reverb settings to the mixer via OSC.

        Returns the number of channels updated.

        NOTE: The specific OSC commands depend on the mixer model.
        This implementation targets Behringer Wing's reverb send
        architecture.  If the mixer_client does not support specific
        reverb methods, settings are logged but not applied.
        """
        if decisions is None:
            decisions = self.decisions
        if not mixer_client or not decisions:
            return 0

        applied = 0
        for ch_id, dec in decisions.items():
            try:
                # Set reverb send level if supported.
                if hasattr(mixer_client, "set_reverb_send"):
                    mixer_client.set_reverb_send(ch_id, dec.reverb_level_lu)
                    applied += 1

                # Set reverb type/decay/predelay on the FX bus
                # if supported.
                if hasattr(mixer_client, "set_reverb_params"):
                    mixer_client.set_reverb_params(
                        ch_id,
                        reverb_type=dec.reverb_type,
                        decay_time=dec.decay_time_s,
                        predelay=dec.predelay_ms,
                        brightness=dec.brightness,
                        density=dec.density,
                        hpf=dec.hpf_hz,
                        lpf=dec.lpf_hz,
                    )

                logger.info(
                    f"AutoReverb: Ch{ch_id} ({dec.instrument}) -> "
                    f"type={dec.reverb_type}, decay={dec.decay_time_s:.2f}s, "
                    f"level={dec.reverb_level_lu:.1f} LU, "
                    f"predelay={dec.predelay_ms:.0f}ms"
                )
            except Exception as e:
                logger.error(
                    f"AutoReverb: Failed to apply reverb for Ch{ch_id}: {e}"
                )

        return applied
