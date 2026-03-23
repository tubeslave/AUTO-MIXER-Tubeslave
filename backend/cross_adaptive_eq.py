"""
Cross-adaptive mirror EQ utilities.

Based on IMP 7.3 (Intelligent Music Production, De Man, Reiss & Stables):

Full mirror equalization strategy:
- "Mirrored EQ involves boosting one track at a particular frequency
  and cutting the other track at the same frequency" [98, 127].
- "preference for boosting the masked track or for full mirror EQ,
  over just cutting the masker" [154].
- "subtractive EQ (cuts) are generally performed with a higher quality
  factor than additive EQ (boosts)" [12].
- Target frequency contour for the complete mix is similar to pink
  noise [274, 275].

References:
  [12] Pestana & Reiss (2014)
  [40] Perez Gonzalez - cross-adaptive multitrack EQ
  [52] Ma et al. - intelligent target EQ
  [61] Hafezi & Reiss - multitrack intelligent EQ
  [98, 127] - mirror equalization technique
  [154] Wakefield & Dewey - preference for boost + mirror over cut-only
  [274, 275] - pink noise target spectrum
"""

from dataclasses import dataclass
from typing import Dict, List


@dataclass
class MirrorEQAdjustment:
    channel_id: int
    frequency_hz: float
    gain_db: float
    q_factor: float


# Q factors per IMP [12]:
# Subtractive EQ (cuts) use higher Q (narrower, surgical).
# Additive EQ (boosts) use lower Q (broader, more musical).
Q_SUBTRACTIVE = 4.0
Q_ADDITIVE = 2.0


class CrossAdaptiveEQ:
    """
    Full mirror-EQ processor for multitrack band-energy dictionaries.

    Per IMP 7.3 [98, 127, 154]:
    - Cut the lower-priority (masker) channel in overlapping bands (Q=4.0).
    - Boost the higher-priority (masked) channel in overlapping bands (Q=2.0).
    - Boost amount is half the cut amount to err on the conservative side.
    """

    BAND_CENTERS = {
        "sub": 40.0,
        "bass": 120.0,
        "low_mid": 375.0,
        "mid": 1250.0,
        "high_mid": 3000.0,
        "high": 6000.0,
        "air": 12000.0,
    }

    def __init__(
        self,
        min_band_level_db=-80.0,
        overlap_tolerance_db=6.0,
        max_cut_db=-6.0,
        max_boost_db=3.0,
        q_subtractive=Q_SUBTRACTIVE,
        q_additive=Q_ADDITIVE,
    ):
        self.min_band_level_db = min_band_level_db
        self.overlap_tolerance_db = overlap_tolerance_db
        self.max_cut_db = max_cut_db
        self.max_boost_db = max_boost_db
        self.q_subtractive = q_subtractive
        self.q_additive = q_additive

    def calculate_corrections(
        self,
        channel_band_energy,
        channel_priorities,
    ):
        """
        Compute full mirror-EQ adjustments for overlapping bands.

        Per IMP 7.3 [98, 127, 154]:
        - For each overlapping band between two channels with different
          priorities, apply a CUT on the masker (lower-priority) and a
          BOOST on the masked (higher-priority) channel.
        - Cuts use Q=4.0 (subtractive, narrower) [12].
        - Boosts use Q=2.0 (additive, broader) [12].

        Lower numeric priority means higher importance.
        """
        channels = list(channel_band_energy.keys())
        out = []

        for i, ch_a in enumerate(channels):
            for ch_b in channels[i + 1:]:
                bands_a = channel_band_energy.get(ch_a, {})
                bands_b = channel_band_energy.get(ch_b, {})
                if not bands_a or not bands_b:
                    continue

                prio_a = channel_priorities.get(ch_a, 3)
                prio_b = channel_priorities.get(ch_b, 3)

                if prio_a == prio_b:
                    continue

                # Identify masked (higher priority) and masker (lower priority).
                if prio_a < prio_b:
                    masked_ch = ch_a
                    masker_ch = ch_b
                    masked_bands = bands_a
                    masker_bands = bands_b
                else:
                    masked_ch = ch_b
                    masker_ch = ch_a
                    masked_bands = bands_b
                    masker_bands = bands_a

                for band, center_hz in self.BAND_CENTERS.items():
                    masked_db = masked_bands.get(band, -100.0)
                    masker_db = masker_bands.get(band, -100.0)
                    if masked_db < self.min_band_level_db or masker_db < self.min_band_level_db:
                        continue

                    # Overlap when both bands are close in magnitude.
                    diff = abs(masked_db - masker_db)
                    if diff > self.overlap_tolerance_db:
                        continue

                    # Stronger overlap -> stronger correction (bounded).
                    overlap_strength = 1.0 - (diff / max(self.overlap_tolerance_db, 1e-6))

                    # CUT the masker channel [98, 127] with Q=4.0 [12].
                    cut_db = -min(abs(self.max_cut_db), 1.0 + 5.0 * overlap_strength)
                    out.append(
                        MirrorEQAdjustment(
                            channel_id=masker_ch,
                            frequency_hz=center_hz,
                            gain_db=cut_db,
                            q_factor=self.q_subtractive,
                        )
                    )

                    # BOOST the masked channel [154] with Q=2.0 [12].
                    # Boost is half the cut amount (conservative) [154].
                    boost_db = min(self.max_boost_db, abs(cut_db) * 0.5)
                    out.append(
                        MirrorEQAdjustment(
                            channel_id=masked_ch,
                            frequency_hz=center_hz,
                            gain_db=boost_db,
                            q_factor=self.q_additive,
                        )
                    )

        return out
