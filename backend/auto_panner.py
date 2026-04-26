"""
Auto Panner Module - Intelligent Stereo Panning.

Based on IMP 7.2 (Intelligent Music Production, De Man, Reiss & Stables):

Best practices from the book:
- Sources with large low-frequency energy should be panned center [12, 74, 120].
- Lead vocal should be panned centrally [12].
- Snare drum is typically centered [12].
- Centered sources panned slightly off-center for intelligibility.
- Higher frequency sources panned progressively toward extremes [12, 49, 120],
  though this relationship tapers off after the low mids [74].
- No hard panning (max ~80%) [49].
- Spectral balancing: uniform distribution of content within each
  frequency band [49].
- Source balancing: equal numbering and symmetric positioning of sources
  on either side of the stereo field [49].
- The mix should not be too narrow but still have a strong central
  component [178].

References:
  [12] Pestana & Reiss (2014)
  [36] Perez Gonzalez & Reiss - first autonomous panning system
  [43] Perez Gonzalez & Reiss - extended panning with subjective evaluation
  [49] Mansbridge et al. - spectral/source/spatial balancing
  [58] Pestana & Reiss - per-band spatial positioning
  [63] Matz et al. - dynamic spectral panning
  [74] De Man, Reiss & Stables - analysis of 600 mixes
"""

import logging
import math
from typing import Any, Dict, List, Optional, Tuple
from dataclasses import dataclass

logger = logging.getLogger(__name__)

# Maximum panning (0-100 scale).  No hard panning per [49].
MAX_PAN_PERCENT = 80.0

# Frequency threshold below which sources are centred [12, 74, 120].
LOW_FREQ_CENTER_HZ = 200.0
PAN_TRANSITION_MS = 22.0

LOW_FREQUENCY_BANDS = frozenset({"sub", "bass", "low", "low_bass"})

# Instruments that must always be centred [12].
CENTER_INSTRUMENTS = frozenset({
    "leadVocal", "lead_vocal",
    "kick",
    "snare",
    "bass",
})

# Instruments that should stay near-center (slight offset allowed).
NEAR_CENTER_INSTRUMENTS = frozenset({
    "backingVocal", "backing_vocal", "back_vocal",
    "drums",
})


@dataclass
class PanDecision:
    """Panning decision for a single channel."""
    channel_id: int
    instrument: str
    pan_percent: float  # -100 (hard left) .. +100 (hard right), 0 = center
    reason: str
    dominant_band: Optional[str] = None
    transition_ms: float = PAN_TRANSITION_MS


class AutoPanner:
    """
    Intelligent auto-panner based on IMP 7.2 best practices.

    Assigns stereo positions to channels based on instrument type,
    spectral content, and balancing heuristics.  Integrates with
    Behringer Wing via OSC (through a mixer_client).
    """

    def __init__(
        self,
        max_pan_percent: float = MAX_PAN_PERCENT,
        low_freq_center_hz: float = LOW_FREQ_CENTER_HZ,
        transition_ms: float = PAN_TRANSITION_MS,
    ):
        self.max_pan = min(abs(max_pan_percent), 100.0)
        self.low_freq_center_hz = low_freq_center_hz
        self.transition_ms = max(0.0, float(transition_ms))
        self.decisions: Dict[int, PanDecision] = {}
        self.band_accumulator: Dict[int, Dict[str, int]] = {}

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def calculate_panning(
        self,
        channels: List[int],
        instrument_types: Dict[int, str],
        spectral_centroids: Optional[Dict[int, float]] = None,
        channel_band_energy: Optional[Dict[int, Dict[str, float]]] = None,
    ) -> Dict[int, PanDecision]:
        """
        Calculate panning positions for a list of channels.

        Parameters
        ----------
        channels : list of int
            Mixer channel IDs to pan.
        instrument_types : dict
            Mapping channel_id -> instrument name string.
        spectral_centroids : dict, optional
            Mapping channel_id -> spectral centroid in Hz (for frequency-
            based panning decisions).
        channel_band_energy : dict, optional
            Mapping channel_id -> band name -> energy dB. When provided,
            dominant bands are accumulated over calls and same-band sources are
            distributed symmetrically with lower physical input numbers closer
            to center.

        Returns
        -------
        dict of channel_id -> PanDecision
        """
        if spectral_centroids is None:
            spectral_centroids = {}
        dominant_bands = self.update_band_accumulator(channel_band_energy or {})

        center_ids: List[int] = []
        near_center_ids: List[int] = []
        pan_ids: List[int] = []

        # 1. Classify each channel.
        for ch_id in channels:
            instrument = (instrument_types.get(ch_id) or "unknown").lower()
            norm_instrument = instrument_types.get(ch_id, "unknown")

            # Rule 1: Low-frequency sources -> center [12, 74, 120].
            centroid = spectral_centroids.get(ch_id, 0.0)
            dominant_band = dominant_bands.get(ch_id)
            if dominant_band in LOW_FREQUENCY_BANDS:
                center_ids.append(ch_id)
                continue
            if centroid > 0 and centroid < self.low_freq_center_hz:
                center_ids.append(ch_id)
                continue

            # Rule 2: Named center instruments [12].
            if norm_instrument in CENTER_INSTRUMENTS or instrument in {
                "leadvocal", "lead_vocal", "kick", "snare", "bass",
            }:
                center_ids.append(ch_id)
                continue

            # Rule 3: Near-center instruments.
            if norm_instrument in NEAR_CENTER_INSTRUMENTS or instrument in {
                "backingvocal", "backing_vocal", "back_vocal", "drums",
            }:
                near_center_ids.append(ch_id)
                continue

            # Everything else gets panned.
            pan_ids.append(ch_id)

        decisions: Dict[int, PanDecision] = {}

        # 2. Center instruments -> pan = 0.
        for ch_id in center_ids:
            decisions[ch_id] = PanDecision(
                channel_id=ch_id,
                instrument=instrument_types.get(ch_id, "unknown"),
                pan_percent=0.0,
                reason="center (low freq / lead vocal / kick / snare / bass) [IMP 7.2]",
                dominant_band=dominant_bands.get(ch_id),
                transition_ms=self.transition_ms,
            )

        # 3. Near-center instruments -> slight offset, alternating L/R.
        near_center_offset = min(10.0, self.max_pan * 0.15)
        side = 1.0
        for ch_id in near_center_ids:
            decisions[ch_id] = PanDecision(
                channel_id=ch_id,
                instrument=instrument_types.get(ch_id, "unknown"),
                pan_percent=round(side * near_center_offset, 1),
                reason="near-center [IMP 7.2]",
                dominant_band=dominant_bands.get(ch_id),
                transition_ms=self.transition_ms,
            )
            side *= -1.0

        # 4. Remaining sources: distribute symmetrically [49].
        #    Higher spectral centroid -> wider panning [12, 49, 120],
        #    but relationship tapers off after low mids [74].
        if pan_ids:
            grouped = self._group_pan_candidates(pan_ids, dominant_bands)
            for band_name, group_ids in grouped.items():
                if band_name:
                    group_ids_sorted = sorted(group_ids)
                else:
                    group_ids_sorted = sorted(
                        group_ids,
                        key=lambda cid: spectral_centroids.get(cid, 500.0),
                    )
                positions = self._symmetric_positions(len(group_ids_sorted))

                for i, ch_id in enumerate(group_ids_sorted):
                    centroid = spectral_centroids.get(ch_id, 500.0)
                    width_factor = self._centroid_to_width(centroid)
                    pan_amount = positions[i] * max(0.25, width_factor)
                    pan_amount = round(max(-self.max_pan, min(self.max_pan, pan_amount)), 1)

                    if band_name:
                        reason = (
                            f"spectral-band pan (band={band_name}, "
                            f"centroid={centroid:.0f} Hz, input priority) [Perez/Reiss]"
                        )
                    else:
                        reason = f"spectral pan (centroid={centroid:.0f} Hz) [IMP 7.2]"

                    decisions[ch_id] = PanDecision(
                        channel_id=ch_id,
                        instrument=instrument_types.get(ch_id, "unknown"),
                        pan_percent=pan_amount,
                        reason=reason,
                        dominant_band=dominant_bands.get(ch_id),
                        transition_ms=self.transition_ms,
                    )

        self.decisions = decisions
        return decisions

    def apply_to_mixer(
        self,
        mixer_client: Any,
        decisions: Optional[Dict[int, PanDecision]] = None,
    ) -> int:
        """
        Send pan positions to the mixer via OSC.

        Returns the number of channels updated.
        """
        if decisions is None:
            decisions = self.decisions
        if not mixer_client or not decisions:
            return 0

        if not hasattr(mixer_client, "set_channel_pan"):
            logger.warning("Mixer client has no set_channel_pan method")
            return 0

        applied = 0
        for ch_id, dec in decisions.items():
            try:
                mixer_client.set_channel_pan(ch_id, dec.pan_percent)
                applied += 1
                logger.info(
                    f"AutoPanner: Ch{ch_id} ({dec.instrument}) -> "
                    f"pan={dec.pan_percent:+.1f}% ({dec.reason})"
                )
            except Exception as e:
                logger.error(f"AutoPanner: Failed to set pan for Ch{ch_id}: {e}")

        return applied

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def update_band_accumulator(
        self,
        channel_band_energy: Dict[int, Dict[str, float]],
    ) -> Dict[int, str]:
        """
        Accumulate dominant spectral bands per channel.

        Perez/Reiss-style panning uses repeated short-window observations
        rather than a single instantaneous sample. Callers can feed 100 ms
        band-energy snapshots; the most frequently dominant band wins.
        """
        dominant: Dict[int, str] = {}
        for ch_id, bands in channel_band_energy.items():
            finite_bands = {
                name: float(value)
                for name, value in bands.items()
                if value is not None and math.isfinite(float(value))
            }
            if not finite_bands:
                continue
            current_band = max(finite_bands, key=finite_bands.get)
            accumulator = self.band_accumulator.setdefault(ch_id, {})
            accumulator[current_band] = accumulator.get(current_band, 0) + 1
            dominant[ch_id] = max(accumulator, key=accumulator.get)

        return dominant

    @staticmethod
    def _group_pan_candidates(
        pan_ids: List[int],
        dominant_bands: Dict[int, str],
    ) -> Dict[Optional[str], List[int]]:
        grouped: Dict[Optional[str], List[int]] = {}
        for ch_id in pan_ids:
            band_name = dominant_bands.get(ch_id)
            grouped.setdefault(band_name, []).append(ch_id)
        return grouped

    def _symmetric_positions(self, count: int) -> List[float]:
        """Return positions ordered so earlier physical inputs are nearer center."""
        if count <= 0:
            return []
        if count == 1:
            return [0.0]

        pair_count = math.ceil(count / 2)
        positions: List[float] = []
        for i in range(count):
            rank = (i // 2) + 1
            amount = self.max_pan * rank / pair_count
            sign = -1.0 if i % 2 == 0 else 1.0
            positions.append(sign * amount)
        return positions

    @staticmethod
    def _centroid_to_width(centroid_hz: float) -> float:
        """
        Map spectral centroid to a 0..1 panning width factor.

        Based on IMP 7.2 / [12, 49, 74]:
        - Low freq (<250 Hz): 0 (center).
        - Low-mid (250-500 Hz): narrow (0.1 - 0.3).
        - Mid (500-2000 Hz): moderate (0.3 - 0.6).
        - High-mid (2000-4000 Hz): wide (0.6 - 0.9).
        - Above 4 kHz: tapers, stays at ~0.9 [74].
        """
        if centroid_hz < 250.0:
            return 0.0
        if centroid_hz < 500.0:
            return 0.1 + 0.2 * (centroid_hz - 250.0) / 250.0
        if centroid_hz < 2000.0:
            return 0.3 + 0.3 * (centroid_hz - 500.0) / 1500.0
        if centroid_hz < 4000.0:
            return 0.6 + 0.3 * (centroid_hz - 2000.0) / 2000.0
        # Tapers off above 4 kHz [74].
        return 0.9
