"""
Dynamic Mixer - Real-time adaptive mixing with fuzzy logic

Implements PDF Section 8-9: Dynamic mixing mode
Uses "Calibrate → Stabilize → Maintain" approach:
1. CALIBRATION: Collect LUFS statistics for each channel (first N seconds)
2. STABILIZATION: Calculate optimal fader positions and apply them
3. MAINTENANCE: Only small corrections when deviation exceeds dead zone
"""

import logging
import os
import time
from typing import Dict, Optional, List
from enum import Enum
from dataclasses import dataclass, field
import numpy as np
from .fuzzy_controller import FuzzyFaderController

logger = logging.getLogger(__name__)

# Debug log path: use AUTO_MIXER_DEBUG_LOG env or project .cursor/debug.log (no hardcoded user paths)
_dynamic_mixer_dir = os.path.dirname(os.path.abspath(__file__))
DEBUG_LOG_PATH = os.environ.get(
    "AUTO_MIXER_DEBUG_LOG",
    os.path.normpath(os.path.join(_dynamic_mixer_dir, "..", "..", "..", ".cursor", "debug.log")),
)


class MixerPhase(Enum):
    """Mixer operation phases"""
    CALIBRATING = "calibrating"    # Collecting statistics
    STABILIZING = "stabilizing"    # Applying initial balance
    MAINTAINING = "maintaining"    # Small corrections only


@dataclass
class ChannelCalibration:
    """Calibration data for a channel"""
    lufs_samples: List[float] = field(default_factory=list)
    optimal_fader_offset: float = 0.0  # Calculated optimal offset from current position
    is_calibrated: bool = False
    locked_target_lufs: float = -25.0  # Locked target after calibration
    
    def add_sample(self, lufs: float):
        self.lufs_samples.append(lufs)
        # Keep last 100 samples (~10 seconds at 100ms intervals)
        if len(self.lufs_samples) > 100:
            self.lufs_samples.pop(0)
    
    def get_average_lufs(self) -> float:
        if not self.lufs_samples:
            return -100.0
        # Use median to ignore outliers
        return float(np.median(self.lufs_samples))
    
    def get_stable_lufs(self) -> float:
        """Get stable LUFS (average of middle 80% to ignore outliers)"""
        if len(self.lufs_samples) < 10:
            return self.get_average_lufs()
        
        sorted_samples = sorted(self.lufs_samples)
        # Remove top and bottom 10%
        trim = len(sorted_samples) // 10
        trimmed = sorted_samples[trim:-trim] if trim > 0 else sorted_samples
        return float(np.mean(trimmed))
    
    def get_std_dev(self) -> float:
        """Std dev of LUFS samples (for calibration stability check)."""
        if len(self.lufs_samples) < 5:
            return 999.0
        return float(np.std(self.lufs_samples))


class DynamicMixer:
    """
    Dynamic mixing mode - "Calibrate → Stabilize → Maintain"
    
    Phase 1 (CALIBRATING): Collect LUFS samples for each channel
    Phase 2 (STABILIZING): Calculate and apply optimal fader positions
    Phase 3 (MAINTAINING): Only correct when deviation > dead_zone
    """
    
    def __init__(self, target_lufs: float = -18.0):
        self.target_lufs = target_lufs
        self.fuzzy_controller = FuzzyFaderController()
        
        # State
        self.reference_channels: list[int] = []
        self.instrument_types: Dict[int, str] = {}
        self.genre_offsets: Dict[str, float] = {}
        self.ratio = 2.0
        
        # Settings
        self.max_adjustment_db = 6.0
        self.attack_ms = 100.0
        self.release_ms = 1000.0
        
        # === CALIBRATION SETTINGS ===
        self.calibration_duration_sec = 15.0  # Collect data 15-30s for stable baseline (was 5s)
        self.calibration_min_samples = 60     # Minimum samples (e.g. ~6s at 10 Hz)
        self.calibration_std_dev_max = 2.0    # Only complete when std_dev LUFS < this for all channels (stability)
        
        # === STABILIZATION SETTINGS ===
        self.dead_zone_db = 3.0               # Don't correct if deviation < this (3-4 dB for live reduces "breathing")
        self.maintenance_max_adjustment = 1.5 # Max adjustment per step in maintenance (was 3.0; 1.5 dB less audible)
        self.maintenance_ramp_ms = 500.0      # Min ramp time for fader moves (smoothing)
        self.maintenance_smoothing = 0.4      # Correction smoothing in maintenance
        self.stabilization_max_adjustment = 15.0  # No practical limit during stabilization
        
        # === STATE ===
        self.phase = MixerPhase.CALIBRATING
        self.phase_start_time = time.time()
        self.channel_calibration: Dict[int, ChannelCalibration] = {}
        
        logger.info("DynamicMixer initialized with Calibrate→Stabilize→Maintain mode")
    
    def configure(
        self,
        reference_channels: list[int],
        instrument_types: Dict[int, str],
        genre_offsets: Dict[str, float],
        ratio: float = 2.0,
        max_adjustment_db: float = 6.0
    ):
        """Configure dynamic mixer"""
        self.reference_channels = reference_channels
        self.instrument_types = instrument_types
        self.genre_offsets = genre_offsets
        self.ratio = ratio
        self.max_adjustment_db = max_adjustment_db
        
        # Reset calibration
        self.reset()
    
    def reset(self):
        """Reset to calibration phase"""
        self.phase = MixerPhase.CALIBRATING
        self.phase_start_time = time.time()
        self.channel_calibration.clear()
        logger.info("🔄 DynamicMixer reset - starting CALIBRATION phase")
    
    def _get_target_for_channel(self, ch_id: int) -> float:
        """Get target LUFS for a channel based on instrument type"""
        instrument = self.instrument_types.get(ch_id, 'unknown')
        offset = self.genre_offsets.get(instrument, -7.0)
        return self.target_lufs + offset
    
    def _ensure_calibration(self, ch_id: int):
        """Ensure calibration data exists for channel"""
        if ch_id not in self.channel_calibration:
            self.channel_calibration[ch_id] = ChannelCalibration()
    
    def calculate_adjustments(
        self,
        current_levels: Dict[int, float]  # channel_id -> current LUFS
    ) -> Dict[int, float]:
        """
        Calculate fader adjustments based on current phase
        
        CALIBRATING: Collect samples, return small/no adjustments
        STABILIZING: Return calculated optimal adjustments
        MAINTAINING: Return small adjustments only if deviation > dead_zone
        """
        adjustments = {}
        
        if not current_levels:
            logger.warning("No current levels provided")
            return adjustments
        
        elapsed = time.time() - self.phase_start_time
        
        # ========== CALIBRATION PHASE ==========
        if self.phase == MixerPhase.CALIBRATING:
            # Collect samples
            for ch_id, current_lufs in current_levels.items():
                self._ensure_calibration(ch_id)
                self.channel_calibration[ch_id].add_sample(current_lufs)
                adjustments[ch_id] = 0.0  # No adjustments during calibration
            
            # #region agent log
            try:
                import json
                import time as time_module
                with open(DEBUG_LOG_PATH, "a") as f:
                    f.write(json.dumps({
                        "sessionId": "debug-session",
                        "runId": "run1",
                        "hypothesisId": "A,D",
                        "location": "dynamic_mixer.py:CALIBRATING",
                        "message": "Calibration sample collected",
                        "data": {
                            "elapsed": elapsed,
                            "sample_levels": dict(list(current_levels.items())[:5]),
                            "sample_counts": {ch: len(self.channel_calibration[ch].lufs_samples) for ch in list(current_levels.keys())[:5]}
                        },
                        "timestamp": int(time_module.time() * 1000)
                    }) + "\n")
            except: pass
            # #endregion
            
            # Check if calibration is complete: time + stability (std_dev) for all channels
            if elapsed >= self.calibration_duration_sec:
                stable = True
                for ch_id in list(current_levels.keys()):
                    self._ensure_calibration(ch_id)
                    cal = self.channel_calibration[ch_id]
                    if len(cal.lufs_samples) >= self.calibration_min_samples // 2:
                        std = cal.get_std_dev()
                        if std > self.calibration_std_dev_max:
                            stable = False
                            if int(elapsed * 10) % 10 == 0:
                                logger.info(f"📊 CALIBRATING: Ch{ch_id} std_dev={std:.1f} > {self.calibration_std_dev_max}, waiting for stability")
                            break
                if stable or elapsed >= 2.0 * self.calibration_duration_sec:
                    self._complete_calibration()
                elif int(elapsed * 10) % 10 == 0:
                    logger.info(f"📊 CALIBRATING: {elapsed:.1f}s / {self.calibration_duration_sec}s (waiting for stable levels)")
            else:
                if int(elapsed * 10) % 10 == 0:
                    logger.info(f"📊 CALIBRATING: {elapsed:.1f}s / {self.calibration_duration_sec}s ({len(current_levels)} channels)")
            
            return adjustments
        
        # ========== STABILIZATION PHASE ==========
        elif self.phase == MixerPhase.STABILIZING:
            # #region agent log
            stabilization_details = []
            # #endregion
            
            for ch_id, current_lufs in current_levels.items():
                if ch_id in self.reference_channels:
                    adjustments[ch_id] = 0.0
                    continue
                
                self._ensure_calibration(ch_id)
                cal = self.channel_calibration[ch_id]
                
                # Calculate adjustment to reach locked target
                diff = cal.locked_target_lufs - current_lufs
                
                # Full adjustment during stabilization (no practical limit)
                adjustment = max(-self.stabilization_max_adjustment, 
                               min(self.stabilization_max_adjustment, diff))
                
                adjustments[ch_id] = adjustment
                
                # #region agent log
                if ch_id in [13, 14, 15, 1, 2, 11, 17]:
                    stabilization_details.append({
                        "ch": ch_id,
                        "current": current_lufs,
                        "target": cal.locked_target_lufs,
                        "diff": diff,
                        "adj": adjustment
                    })
                # #endregion
            
            # Transition to maintenance after initial adjustments are applied
            # Check if ALL channels are close to target (within 2.0 dB)
            close_count = sum(1 for ch_id in current_levels 
                            if ch_id in self.channel_calibration 
                            and abs(self.channel_calibration[ch_id].locked_target_lufs - current_levels[ch_id]) < 2.0)
            
            # #region agent log
            try:
                import json
                import time as time_module
                with open(DEBUG_LOG_PATH, "a") as f:
                    f.write(json.dumps({
                        "sessionId": "debug-session",
                        "runId": "run1",
                        "hypothesisId": "E",
                        "location": "dynamic_mixer.py:STABILIZING",
                        "message": "Stabilization phase",
                        "data": {
                            "elapsed": elapsed,
                            "close_count": close_count,
                            "total": len(current_levels),
                            "threshold_to_maintain": len(current_levels) * 0.9,
                            "channels": stabilization_details
                        },
                        "timestamp": int(time_module.time() * 1000)
                    }) + "\n")
            except: pass
            # #endregion
            
            # Require 90% of channels within 2dB OR 10 seconds max stabilization time
            if close_count >= len(current_levels) * 0.9 or elapsed > 10.0:
                self.phase = MixerPhase.MAINTAINING
                self.phase_start_time = time.time()
                logger.info(f"✅ STABILIZATION complete - entering MAINTENANCE mode")
                logger.info(f"   {close_count}/{len(current_levels)} channels within 2dB of target")
                logger.info(f"   Dead zone: ±{self.dead_zone_db} dB - faders will now stay stable")
            
            return adjustments
        
        # ========== MAINTENANCE PHASE ==========
        else:  # MixerPhase.MAINTAINING
            # #region agent log
            maintenance_details = []
            # #endregion
            
            for ch_id, current_lufs in current_levels.items():
                if ch_id in self.reference_channels:
                    adjustments[ch_id] = 0.0
                    continue
                
                self._ensure_calibration(ch_id)
                cal = self.channel_calibration[ch_id]
                
                # Update running average (slowly)
                cal.add_sample(current_lufs)
                
                # Calculate deviation from target
                diff = cal.locked_target_lufs - current_lufs
                
                # === DEAD ZONE: No correction if deviation is small ===
                if abs(diff) < self.dead_zone_db:
                    adjustments[ch_id] = 0.0
                    # #region agent log
                    if ch_id in [13, 14, 15, 1, 2, 11, 17]:  # Vocals, kick, snare, accordion, guitar
                        maintenance_details.append({
                            "ch": ch_id,
                            "current": current_lufs,
                            "target": cal.locked_target_lufs,
                            "diff": diff,
                            "in_dead_zone": True,
                            "adj": 0.0
                        })
                    # #endregion
                    continue
                
                # Only correct the amount BEYOND the dead zone
                if diff > 0:
                    effective_diff = diff - self.dead_zone_db
                else:
                    effective_diff = diff + self.dead_zone_db
                
                # Very small, slow adjustment
                adjustment = effective_diff * self.maintenance_smoothing
                
                # Limit maintenance adjustments
                adjustment = max(-self.maintenance_max_adjustment, 
                               min(self.maintenance_max_adjustment, adjustment))
                
                adjustments[ch_id] = adjustment
                
                # #region agent log
                if ch_id in [13, 14, 15, 1, 2, 11, 17]:  # Vocals, kick, snare, accordion, guitar
                    maintenance_details.append({
                        "ch": ch_id,
                        "current": current_lufs,
                        "target": cal.locked_target_lufs,
                        "diff": diff,
                        "effective_diff": effective_diff,
                        "in_dead_zone": False,
                        "adj": adjustment,
                        "limited": abs(effective_diff * self.maintenance_smoothing) > self.maintenance_max_adjustment
                    })
                # #endregion
                
                if abs(adjustment) > 0.1:
                    instrument = self.instrument_types.get(ch_id, 'unknown')
                    logger.debug(f"Ch{ch_id} ({instrument}): deviation={diff:+.1f}dB, adj={adjustment:+.2f}dB")
            
            # #region agent log
            try:
                import json
                import time as time_module
                with open(DEBUG_LOG_PATH, "a") as f:
                    f.write(json.dumps({
                        "sessionId": "debug-session",
                        "runId": "run1",
                        "hypothesisId": "B,C,E",
                        "location": "dynamic_mixer.py:MAINTAINING",
                        "message": "Maintenance phase details",
                        "data": {
                            "dead_zone_db": self.dead_zone_db,
                            "max_adjustment": self.maintenance_max_adjustment,
                            "smoothing": self.maintenance_smoothing,
                            "channels": maintenance_details
                        },
                        "timestamp": int(time_module.time() * 1000)
                    }) + "\n")
            except: pass
            # #endregion
            
            return adjustments
    
    def _complete_calibration(self):
        """Complete calibration and calculate optimal positions"""
        logger.info("📊 CALIBRATION complete - calculating optimal fader positions...")
        
        # #region agent log
        calibration_results = []
        # #endregion
        
        for ch_id, cal in self.channel_calibration.items():
            if len(cal.lufs_samples) < self.calibration_min_samples // 2:
                logger.warning(f"Ch{ch_id}: Not enough samples ({len(cal.lufs_samples)})")
                continue
            
            # Get stable average LUFS
            avg_lufs = cal.get_stable_lufs()
            
            # Get target for this instrument
            target = self._get_target_for_channel(ch_id)
            
            # Lock the target
            cal.locked_target_lufs = target
            cal.is_calibrated = True
            
            # Calculate needed offset
            cal.optimal_fader_offset = target - avg_lufs
            
            instrument = self.instrument_types.get(ch_id, 'unknown')
            logger.info(f"  Ch{ch_id} ({instrument}): avg={avg_lufs:.1f}, target={target:.1f}, offset={cal.optimal_fader_offset:+.1f} dB")
            
            # #region agent log
            if ch_id in [13, 14, 15, 1, 2, 11, 17]:  # Vocals, kick, snare, accordion, guitar
                calibration_results.append({
                    "ch": ch_id,
                    "instrument": instrument,
                    "avg_lufs": avg_lufs,
                    "target": target,
                    "offset_needed": cal.optimal_fader_offset,
                    "sample_count": len(cal.lufs_samples)
                })
            # #endregion
        
        # #region agent log
        try:
            import json
            import time as time_module
            with open(DEBUG_LOG_PATH, "a") as f:
                f.write(json.dumps({
                    "sessionId": "debug-session",
                    "runId": "run1",
                    "hypothesisId": "A,D",
                    "location": "dynamic_mixer.py:_complete_calibration",
                    "message": "Calibration completed",
                    "data": {
                        "channels": calibration_results,
                        "total_calibrated": sum(1 for c in self.channel_calibration.values() if c.is_calibrated)
                    },
                    "timestamp": int(time_module.time() * 1000)
                }) + "\n")
        except: pass
        # #endregion
        
        # Move to stabilization phase
        self.phase = MixerPhase.STABILIZING
        self.phase_start_time = time.time()
        logger.info("🎚️ Entering STABILIZATION phase - applying initial balance...")
    
    def get_phase_info(self) -> Dict:
        """Get current phase information"""
        elapsed = time.time() - self.phase_start_time
        
        return {
            "phase": self.phase.value,
            "elapsed_seconds": elapsed,
            "calibration_duration": self.calibration_duration_sec,
            "dead_zone_db": self.dead_zone_db,
            "channels_calibrated": sum(1 for c in self.channel_calibration.values() if c.is_calibrated)
        }
