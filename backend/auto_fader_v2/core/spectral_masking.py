"""
Spectral Masking Detection and Correction

Implements simplified spectral masking detection as recommended in the
"От LUFS до OSC-команд" document.

When vocal dominates in critical frequency range (1-3 kHz), reduces that
band on background channels via EQ to "free up" space for vocals.
"""

import logging
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class EQAdjustment:
    """EQ band adjustment"""
    channel_id: int
    frequency_hz: float
    gain_db: float  # Negative for cut
    q: float = 3.0  # Q factor for the EQ band

    @property
    def q_factor(self) -> float:
        """Backward-compatible alias used by controller."""
        return self.q


class SpectralMaskingDetector:
    """
    Detects spectral masking conflicts and generates EQ adjustments.
    
    Algorithm (per document):
    - Compare energy in critical band (1-3 kHz) between vocals and background
    - If vocal dominates (>6 dB), reduce that band on background via EQ
    - Limit maximum cut to avoid unnatural sound (e.g., -6 dB)
    """
    
    def __init__(
        self,
        critical_band_hz: Tuple[float, float] = (1000.0, 3000.0),
        dominance_threshold_db: float = 6.0,
        max_cut_db: float = -6.0,
        q_factor: float = 3.0
    ):
        """
        Initialize spectral masking detector.
        
        Args:
            critical_band_hz: Critical frequency range for vocals (default 1-3 kHz)
            dominance_threshold_db: Vocal must exceed background by this amount to trigger cut
            max_cut_db: Maximum EQ cut in dB (negative value, e.g., -6.0)
            q_factor: Q factor for EQ band
        """
        self.critical_band_hz = critical_band_hz
        self.dominance_threshold_db = dominance_threshold_db
        self.max_cut_db = max_cut_db
        self.q_factor = q_factor
        
        logger.info(f"SpectralMaskingDetector initialized: band={critical_band_hz} Hz, "
                   f"threshold={dominance_threshold_db} dB, max_cut={max_cut_db} dB")
    
    def _get_band_energy(self, band_energy: Dict[str, float], band_range: Tuple[float, float]) -> float:
        """
        Get energy in specified frequency range from band_energy dict.
        
        Args:
            band_energy: Dictionary with keys like 'mid', 'high_mid', etc.
            band_range: Frequency range (low_hz, high_hz)
        
        Returns:
            Combined energy in dB
        """
        low_hz, high_hz = band_range
        
        # Prefer document-aligned coarse bands if present:
        # - lmf: 250-1000 Hz
        # - umf: 1000-4000 Hz
        # Fallback to legacy {mid, high_mid}.
        if low_hz >= 1000 and high_hz <= 4000 and 'umf' in band_energy:
            return band_energy.get('umf', -100.0)
        if low_hz >= 250 and high_hz <= 1000 and 'lmf' in band_energy:
            return band_energy.get('lmf', -100.0)
        
        energy_linear = 0.0
        
        # Check mid band (500-2000 Hz)
        if low_hz < 2000:
            mid_overlap_low = max(500, low_hz)
            mid_overlap_high = min(2000, high_hz)
            if mid_overlap_low < mid_overlap_high:
                mid_db = band_energy.get('mid', -100.0)
                if mid_db > -100:
                    # Convert to linear, scale by overlap
                    overlap_ratio = (mid_overlap_high - mid_overlap_low) / (high_hz - low_hz)
                    energy_linear += (10 ** (mid_db / 10.0)) * overlap_ratio
        
        # Check high_mid band (2000-4000 Hz)
        if high_hz > 2000:
            high_mid_overlap_low = max(2000, low_hz)
            high_mid_overlap_high = min(4000, high_hz)
            if high_mid_overlap_low < high_mid_overlap_high:
                high_mid_db = band_energy.get('high_mid', -100.0)
                if high_mid_db > -100:
                    overlap_ratio = (high_mid_overlap_high - high_mid_overlap_low) / (high_hz - low_hz)
                    energy_linear += (10 ** (high_mid_db / 10.0)) * overlap_ratio
        
        if energy_linear < 1e-10:
            return -100.0
        
        return 10.0 * (energy_linear ** 0.5)  # Approximate combined energy
    
    def detect_conflicts(
        self,
        vocal_channels: List[int],
        background_channels: List[int],
        channel_band_energy: Dict[int, Dict[str, float]]
    ) -> List[EQAdjustment]:
        """
        Detect spectral masking conflicts and generate EQ adjustments.
        
        Args:
            vocal_channels: List of vocal channel IDs
            background_channels: List of background channel IDs
            channel_band_energy: Dictionary of channel_id -> band_energy dict
        
        Returns:
            List of EQ adjustments to apply
        """
        if not vocal_channels or not background_channels:
            return []
        
        # Calculate average vocal energy in critical band
        vocal_energies = []
        for ch_id in vocal_channels:
            if ch_id in channel_band_energy:
                energy = self._get_band_energy(
                    channel_band_energy[ch_id],
                    self.critical_band_hz
                )
                if energy > -100:
                    vocal_energies.append(energy)
        
        if not vocal_energies:
            return []
        
        # Average vocal energy
        avg_vocal_energy = sum(vocal_energies) / len(vocal_energies)
        
        # Check each background channel
        adjustments = []
        center_freq = (self.critical_band_hz[0] + self.critical_band_hz[1]) / 2.0
        
        for bg_ch_id in background_channels:
            if bg_ch_id not in channel_band_energy:
                continue
            
            bg_energy = self._get_band_energy(
                channel_band_energy[bg_ch_id],
                self.critical_band_hz
            )
            
            if bg_energy < -100:
                continue
            
            # Check if vocal dominates
            energy_diff = avg_vocal_energy - bg_energy
            
            if energy_diff > self.dominance_threshold_db:
                # Vocal dominates - calculate cut amount
                # Cut proportional to dominance, but limited to max_cut_db
                excess_db = energy_diff - self.dominance_threshold_db
                cut_amount = min(abs(self.max_cut_db), excess_db * 0.5)  # Scale factor 0.5
                cut_amount = -abs(cut_amount)  # Ensure negative
                
                adjustments.append(EQAdjustment(
                    channel_id=bg_ch_id,
                    frequency_hz=center_freq,
                    gain_db=cut_amount,
                    q=self.q_factor
                ))
                
                logger.debug(f"Spectral masking: Ch{bg_ch_id} cut {cut_amount:.1f} dB at {center_freq:.0f} Hz "
                            f"(vocal {avg_vocal_energy:.1f} dB vs bg {bg_energy:.1f} dB, diff {energy_diff:.1f} dB)")
        
        return adjustments
