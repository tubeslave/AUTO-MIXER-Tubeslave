"""
Auto Fader - 6-Level Hybrid Architecture

Based on Kimi AI debates results:
- Level 0: Hardware Safety
- Level 1: Signal Analyzer (Peak 5ms + RMS 50ms + LUFS 400ms)
- Level 2: Hybrid Metric Fusion (0.45*LUFS + 0.35*RMS + 0.20*Peak)
- Level 3: Scenario Detector (SILENCE → NORMAL → LOUD → PEAK)
- Level 4: Decision Engine (Adaptive rate limiter)
- Level 5: Safety Validator
- Level 6: OSC Output

Key Features:
- Hybrid metric: L_hybrid = 0.45*LUFS + 0.35*RMS + 0.20*Peak
- 100 Hz control loop (10ms)
- Adaptive rate limiting: 0.5-6.0 dB/cycle
- Scenario-based decision making
- Hardware safety with 10dB headroom
"""

import numpy as np
import logging
import time
from typing import Dict, List, Optional, Tuple, Any
from collections import deque
from dataclasses import dataclass, field
from enum import Enum

# Import existing LUFS components
from lufs_gain_staging import LUFSMeter, TruePeakMeter

logger = logging.getLogger(__name__)


class Scenario(Enum):
    """Audio level scenarios for adaptive processing."""
    SILENCE = "silence"      # Below -60 dB
    QUIET = "quiet"          # -60 to -40 dB
    NORMAL = "normal"        # -40 to -20 dB (target zone)
    LOUD = "loud"            # -20 to -10 dB
    PEAK = "peak"            # Above -10 dB (danger zone)
    EMERGENCY = "emergency"  # Near 0 dB (feedback risk)


class FaderMode(Enum):
    """Operating modes for Auto Fader."""
    OFF = "off"
    MANUAL = "manual"
    AUTO_ASSIST = "auto_assist"  # Suggests corrections
    FULL_AUTO = "full_auto"      # Applies automatically


@dataclass
class ChannelFeatures:
    """Extracted features from audio signal."""
    peak_db: float = -100.0
    rms_db: float = -100.0
    lufs_momentary: float = -100.0
    lufs_short_term: float = -100.0
    crest_factor_db: float = 0.0
    hybrid_level_db: float = -100.0
    scenario: Scenario = Scenario.SILENCE
    is_active: bool = False
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'peak_db': self.peak_db,
            'rms_db': self.rms_db,
            'lufs_momentary': self.lufs_momentary,
            'lufs_short_term': self.lufs_short_term,
            'crest_factor_db': self.crest_factor_db,
            'hybrid_level_db': self.hybrid_level_db,
            'scenario': self.scenario.value,
            'is_active': self.is_active
        }


@dataclass
class FaderDecision:
    """Decision for fader adjustment."""
    channel_id: int
    target_gain_db: float
    correction_db: float
    rate_limited_correction: float
    scenario: Scenario
    confidence: float
    safety_override: bool = False
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'channel_id': self.channel_id,
            'target_gain_db': self.target_gain_db,
            'correction_db': self.correction_db,
            'rate_limited_correction': self.rate_limited_correction,
            'scenario': self.scenario.value,
            'confidence': self.confidence,
            'safety_override': self.safety_override
        }


class HybridMetricFusion:
    """
    Level 2: Hybrid Metric Fusion
    
    Combines multiple metrics into single hybrid level:
    L_hybrid = 0.45*LUFS + 0.35*RMS + 0.20*Peak
    
    Weights optimized for:
    - LUFS (45%): Perceptual loudness, slow but accurate
    - RMS (35%): Signal energy, medium speed
    - Peak (20%): Protection against clipping, fastest
    """
    
    # Weights for hybrid metric
    W_LUFS = 0.45
    W_RMS = 0.35
    W_PEAK = 0.20
    
    def __init__(self):
        self.weights = np.array([self.W_LUFS, self.W_RMS, self.W_PEAK])
        logger.info(f"HybridMetricFusion initialized: weights={self.weights}")
    
    def compute_hybrid_level(self, lufs_db: float, rms_db: float, peak_db: float) -> float:
        """
        Compute hybrid level from individual metrics.
        
        Args:
            lufs_db: LUFS level in dB
            rms_db: RMS level in dB
            peak_db: Peak level in dB
            
        Returns:
            Hybrid level in dB
        """
        # Clamp to valid range
        lufs_db = np.clip(lufs_db, -100, 0)
        rms_db = np.clip(rms_db, -100, 0)
        peak_db = np.clip(peak_db, -100, 0)
        
        # Weighted sum (in linear domain for proper mixing)
        lufs_linear = 10 ** (lufs_db / 20)
        rms_linear = 10 ** (rms_db / 20)
        peak_linear = 10 ** (peak_db / 20)
        
        hybrid_linear = (
            self.W_LUFS * lufs_linear +
            self.W_RMS * rms_linear +
            self.W_PEAK * peak_linear
        )
        
        # Back to dB
        hybrid_db = 20 * np.log10(hybrid_linear + 1e-10)
        
        return float(hybrid_db)


class ScenarioDetector:
    """
    Level 3: Scenario Detector
    
    Detects audio scenario based on hybrid level:
    - SILENCE: < -60 dB (no signal)
    - QUIET: -60 to -40 dB (background)
    - NORMAL: -40 to -20 dB (target zone)
    - LOUD: -20 to -10 dB (approaching limit)
    - PEAK: > -10 dB (danger zone)
    - EMERGENCY: > -3 dB (feedback risk)
    """
    
    # Thresholds in dB
    THRESHOLD_SILENCE = -60.0
    THRESHOLD_QUIET = -40.0
    THRESHOLD_NORMAL = -20.0
    THRESHOLD_LOUD = -10.0
    THRESHOLD_PEAK = -6.0
    THRESHOLD_EMERGENCY = -3.0
    
    def detect_scenario(self, hybrid_level_db: float, peak_db: float) -> Scenario:
        """
        Detect scenario from hybrid level and peak.
        
        Args:
            hybrid_level_db: Hybrid level in dB
            peak_db: Peak level in dB
            
        Returns:
            Scenario enum
        """
        # Emergency check first (highest priority)
        if peak_db > self.THRESHOLD_EMERGENCY or hybrid_level_db > self.THRESHOLD_EMERGENCY:
            return Scenario.EMERGENCY
        
        if hybrid_level_db > self.THRESHOLD_PEAK:
            return Scenario.PEAK
        
        if hybrid_level_db > self.THRESHOLD_LOUD:
            return Scenario.LOUD
        
        if hybrid_level_db > self.THRESHOLD_NORMAL:
            return Scenario.NORMAL
        
        if hybrid_level_db > self.THRESHOLD_QUIET:
            return Scenario.QUIET
        
        return Scenario.SILENCE
    
    def get_rate_limit_db(self, scenario: Scenario) -> float:
        """
        Get rate limit for scenario (dB per cycle).
        
        Args:
            scenario: Current scenario
            
        Returns:
            Max correction in dB per cycle
        """
        limits = {
            Scenario.SILENCE: 6.0,      # Fast recovery from silence
            Scenario.QUIET: 3.0,        # Moderate
            Scenario.NORMAL: 1.0,       # Slow, stable
            Scenario.LOUD: 3.0,         # Moderate reduction
            Scenario.PEAK: 6.0,         # Fast reduction
            Scenario.EMERGENCY: 12.0    # Immediate
        }
        return limits.get(scenario, 1.0)
    
    def get_target_lufs(self, scenario: Scenario, base_target: float = -18.0) -> float:
        """
        Get target LUFS for scenario.
        
        Args:
            scenario: Current scenario
            base_target: Base target LUFS
            
        Returns:
            Target LUFS for scenario
        """
        targets = {
            Scenario.SILENCE: base_target - 10,   # Don't boost silence
            Scenario.QUIET: base_target - 5,
            Scenario.NORMAL: base_target,
            Scenario.LOUD: base_target - 3,       # Reduce target when loud
            Scenario.PEAK: base_target - 6,
            Scenario.EMERGENCY: base_target - 10
        }
        return targets.get(scenario, base_target)


class DecisionEngine:
    """
    Level 4: Decision Engine
    
    Calculates corrections with adaptive rate limiting.
    Uses PI-controller with anti-windup.
    """
    
    def __init__(
        self,
        target_lufs: float = -18.0,
        kp: float = 0.5,      # Proportional gain
        ki: float = 0.05,     # Integral gain
        max_integral: float = 10.0  # Anti-windup limit
    ):
        self.target_lufs = target_lufs
        self.kp = kp
        self.ki = ki
        self.max_integral = max_integral
        
        # Integral term per channel
        self.integral: Dict[int, float] = {}
        
        logger.info(f"DecisionEngine initialized: target={target_lufs}dB, kp={kp}, ki={ki}")
    
    def calculate_correction(
        self,
        channel_id: int,
        hybrid_level_db: float,
        scenario: Scenario,
        rate_limit_db: float
    ) -> Tuple[float, float, float]:
        """
        Calculate correction with rate limiting.

        C-06 FIX: Return type annotation was ``Tuple[float, float]`` but the
        method actually returns three values (correction_db, rate_limited,
        confidence).  The mismatch caused type-checker failures and confused
        callers.  Updated annotation to ``Tuple[float, float, float]``.

        Args:
            channel_id: Channel ID
            hybrid_level_db: Current hybrid level
            scenario: Detected scenario
            rate_limit_db: Max correction per cycle

        Returns:
            (correction_db, rate_limited_correction_db, confidence)
        """
        # C-13 FIX: Avoid creating a new ScenarioDetector() on every call.
        # Store a single shared instance on the engine to avoid unnecessary
        # object allocations in the real-time audio control loop.
        if not hasattr(self, '_scenario_detector'):
            self._scenario_detector = ScenarioDetector()
        # Get target for scenario
        target = self._scenario_detector.get_target_lufs(scenario, self.target_lufs)
        
        # Error (positive = need to increase gain)
        error_db = target - hybrid_level_db
        
        # Initialize integral if needed
        if channel_id not in self.integral:
            self.integral[channel_id] = 0.0
        
        # Update integral (with anti-windup)
        self.integral[channel_id] += error_db * self.ki
        self.integral[channel_id] = np.clip(
            self.integral[channel_id],
            -self.max_integral,
            self.max_integral
        )
        
        # PI control
        correction_db = self.kp * error_db + self.integral[channel_id]
        
        # Rate limiting
        rate_limited = np.clip(correction_db, -rate_limit_db, rate_limit_db)
        
        # Confidence based on error magnitude
        confidence = 1.0 - min(abs(error_db) / 20.0, 1.0)
        
        return correction_db, rate_limited, confidence
    
    def reset_integral(self, channel_id: Optional[int] = None):
        """Reset integral term(s)."""
        if channel_id is not None:
            self.integral[channel_id] = 0.0
        else:
            self.integral.clear()


class SafetyValidator:
    """
    Level 5: Safety Validator
    
    Enforces hard limits and emergency overrides.
    """
    
    # Hard limits
    MAX_GAIN_DB = 10.0
    MIN_GAIN_DB = -60.0
    MAX_TRUE_PEAK_DB = -1.0  # dBTP
    SAFETY_HEADROOM_DB = 10.0
    EMERGENCY_THRESHOLD_DB = -3.0
    
    def __init__(self):
        self.emergency_active = False
        self.emergency_channels: set = set()
        logger.info("SafetyValidator initialized")
    
    def validate_and_limit(
        self,
        channel_id: int,
        proposed_gain_db: float,
        peak_db: float,
        scenario: Scenario
    ) -> Tuple[float, bool]:
        """
        Validate and apply safety limits.
        
        Args:
            channel_id: Channel ID
            proposed_gain_db: Proposed fader gain
            peak_db: Current peak level
            scenario: Current scenario
            
        Returns:
            (limited_gain_db, safety_override)
        """
        safety_override = False
        limited_gain = proposed_gain_db
        
        # Emergency: near clipping
        if peak_db > self.EMERGENCY_THRESHOLD_DB or scenario == Scenario.EMERGENCY:
            # Immediate reduction
            limited_gain = min(proposed_gain_db, -10.0)
            safety_override = True
            self.emergency_active = True
            self.emergency_channels.add(channel_id)
            logger.warning(f"EMERGENCY: Channel {channel_id} peak={peak_db:.1f}dB, forcing gain to {limited_gain:.1f}dB")
        
        # Hard limits
        limited_gain = np.clip(limited_gain, self.MIN_GAIN_DB, self.MAX_GAIN_DB)
        
        # Check if emergency can be cleared
        if self.emergency_active and peak_db < self.EMERGENCY_THRESHOLD_DB - 6:
            if channel_id in self.emergency_channels:
                self.emergency_channels.remove(channel_id)
            if not self.emergency_channels:
                self.emergency_active = False
                logger.info("Emergency cleared")
        
        return limited_gain, safety_override
    
    def check_master_safety(self, master_peak_db: float) -> bool:
        """
        Check master level safety.
        
        Args:
            master_peak_db: Master peak level
            
        Returns:
            True if safe, False if emergency
        """
        if master_peak_db > self.MAX_TRUE_PEAK_DB:
            logger.error(f"MASTER OVERLOAD: {master_peak_db:.1f}dB")
            return False
        return True


class ChannelState:
    """State for single channel."""
    
    def __init__(self, channel_id: int, mixer_channel: int, sample_rate: int = 48000):
        self.channel_id = channel_id
        self.mixer_channel = mixer_channel
        self.sample_rate = sample_rate
        
        # Level 1: Signal Analyzers
        self.lufs_meter = LUFSMeter(sample_rate)
        self.true_peak_meter = TruePeakMeter(sample_rate)
        
        # RMS calculation
        self.rms_window_ms = 50
        self.rms_buffer = deque(maxlen=int(sample_rate * self.rms_window_ms / 1000))
        
        # Peak (5ms window)
        self.peak_window_ms = 5
        self.peak_buffer = deque(maxlen=int(sample_rate * self.peak_window_ms / 1000))
        
        # Current state
        self.features = ChannelFeatures()
        self.current_gain_db = 0.0
        self.target_gain_db = 0.0
        self.is_active = False
        
        # History for smoothing
        self.gain_history = deque(maxlen=10)
        
        logger.debug(f"ChannelState initialized: ch{channel_id} (mixer {mixer_channel})")
    
    def process_audio(self, audio: np.ndarray) -> ChannelFeatures:
        """Process audio and extract features."""
        if audio.size == 0:
            return self.features
        
        # Update meters
        lufs = self.lufs_meter.process(audio)
        peak = self.true_peak_meter.process(audio)
        
        # Update RMS buffer
        rms_squared = np.mean(audio ** 2)
        self.rms_buffer.append(rms_squared)
        rms = np.sqrt(np.mean(list(self.rms_buffer)) + 1e-10)
        
        # Update peak buffer
        frame_peak = np.max(np.abs(audio))
        self.peak_buffer.append(frame_peak)
        peak_5ms = np.max(list(self.peak_buffer))
        
        # Convert to dB
        lufs_db = lufs if lufs > -100 else -100
        rms_db = 20 * np.log10(rms + 1e-10)
        peak_db = 20 * np.log10(peak_5ms + 1e-10)
        
        # Compute hybrid level
        hybrid = HybridMetricFusion()
        hybrid_db = hybrid.compute_hybrid_level(lufs_db, rms_db, peak_db)
        
        # Detect scenario
        detector = ScenarioDetector()
        scenario = detector.detect_scenario(hybrid_db, peak_db)
        
        # Update features
        self.features = ChannelFeatures(
            peak_db=peak_db,
            rms_db=rms_db,
            lufs_momentary=lufs_db,
            lufs_short_term=lufs_db,  # Simplified
            crest_factor_db=peak_db - rms_db if rms_db > -100 else 0,
            hybrid_level_db=hybrid_db,
            scenario=scenario,
            is_active=hybrid_db > -60
        )
        
        return self.features
    
    def apply_gain(self, gain_db: float):
        """Apply new gain with smoothing."""
        self.gain_history.append(gain_db)
        # Smooth with median
        if len(self.gain_history) >= 3:
            smoothed = np.median(list(self.gain_history))
        else:
            smoothed = gain_db
        
        self.current_gain_db = smoothed
        return smoothed
    
    def reset(self):
        """Reset channel state."""
        self.lufs_meter.reset()
        self.true_peak_meter.reset()
        self.rms_buffer.clear()
        self.peak_buffer.clear()
        self.gain_history.clear()
        self.features = ChannelFeatures()
        self.current_gain_db = 0.0
        self.target_gain_db = 0.0


class AutoFaderController:
    """
    Main controller for 6-Level Auto Fader system.
    
    Architecture:
    - Level 1: Signal Analyzer (Peak 5ms + RMS 50ms + LUFS 400ms)
    - Level 2: Hybrid Metric Fusion (0.45*LUFS + 0.35*RMS + 0.20*Peak)
    - Level 3: Scenario Detector
    - Level 4: Decision Engine (PI controller with rate limiting)
    - Level 5: Safety Validator (10dB headroom, emergency override)
    - Level 6: OSC Output
    """
    
    def __init__(
        self,
        mixer_client=None,
        sample_rate: int = 48000,
        target_lufs: float = -18.0,
        loop_frequency_hz: float = 100.0
    ):
        self.mixer_client = mixer_client
        self.sample_rate = sample_rate
        self.target_lufs = target_lufs
        self.loop_period = 1.0 / loop_frequency_hz
        
        # Level 2-5 components
        self.hybrid_fusion = HybridMetricFusion()
        self.scenario_detector = ScenarioDetector()
        self.decision_engine = DecisionEngine(target_lufs)
        self.safety_validator = SafetyValidator()
        
        # Channel states
        self.channels: Dict[int, ChannelState] = {}
        
        # Master state
        self.master_peak_db = -100.0
        self.master_hybrid_db = -100.0
        
        # Control
        self.is_running = False
        self.mode = FaderMode.OFF
        
        logger.info(f"AutoFaderController initialized: target={target_lufs}dB, loop={loop_frequency_hz}Hz")
    
    def add_channel(self, channel_id: int, mixer_channel: int):
        """Add channel to controller."""
        self.channels[channel_id] = ChannelState(channel_id, mixer_channel, self.sample_rate)
        logger.info(f"Added channel {channel_id} (mixer {mixer_channel})")
    
    def remove_channel(self, channel_id: int):
        """Remove channel from controller."""
        if channel_id in self.channels:
            del self.channels[channel_id]
            logger.info(f"Removed channel {channel_id}")
    
    def process_cycle(self, audio_data: Dict[int, np.ndarray]) -> Dict[int, FaderDecision]:
        """
        Process one control cycle.
        
        Args:
            audio_data: {channel_id: audio_samples}
            
        Returns:
            {channel_id: FaderDecision}
        """
        decisions = {}
        
        # Level 1: Signal Analysis + Level 2: Hybrid Fusion + Level 3: Scenario Detection
        all_hybrid_levels = []
        all_peaks = []
        
        for channel_id, audio in audio_data.items():
            if channel_id not in self.channels:
                continue
            
            state = self.channels[channel_id]
            features = state.process_audio(audio)
            
            all_hybrid_levels.append(features.hybrid_level_db)
            all_peaks.append(features.peak_db)
        
        # Calculate master level (sum of correlated signals)
        if all_hybrid_levels:
            # Master is roughly the sum of energy
            master_linear = sum(10 ** (x / 20) for x in all_hybrid_levels)
            self.master_hybrid_db = 20 * np.log10(master_linear + 1e-10)
            self.master_peak_db = max(all_peaks)
        
        # Level 5: Master Safety Check
        if not self.safety_validator.check_master_safety(self.master_peak_db):
            logger.warning("Master safety violation - applying emergency reduction")
            # Reduce all channels proportionally
            for channel_id in self.channels:
                state = self.channels[channel_id]
                emergency_gain = state.current_gain_db - 6.0
                state.apply_gain(emergency_gain)
                decisions[channel_id] = FaderDecision(
                    channel_id=channel_id,
                    target_gain_db=emergency_gain,
                    correction_db=-6.0,
                    rate_limited_correction=-6.0,
                    scenario=Scenario.EMERGENCY,
                    confidence=1.0,
                    safety_override=True
                )
            return decisions
        
        # Level 4: Decision Engine + Level 5: Safety Validation
        for channel_id, state in self.channels.items():
            if channel_id not in audio_data:
                continue
            
            features = state.features
            scenario = features.scenario
            
            # Get rate limit for scenario
            rate_limit = self.scenario_detector.get_rate_limit_db(scenario)
            
            # Calculate correction
            correction, rate_limited, confidence = self.decision_engine.calculate_correction(
                channel_id,
                features.hybrid_level_db,
                scenario,
                rate_limit
            )
            
            # Proposed new gain
            proposed_gain = state.current_gain_db + rate_limited
            
            # Level 5: Safety validation
            final_gain, safety_override = self.safety_validator.validate_and_limit(
                channel_id,
                proposed_gain,
                features.peak_db,
                scenario
            )
            
            # Apply gain
            state.apply_gain(final_gain)
            
            # Create decision
            decisions[channel_id] = FaderDecision(
                channel_id=channel_id,
                target_gain_db=final_gain,
                correction_db=correction,
                rate_limited_correction=rate_limited,
                scenario=scenario,
                confidence=confidence,
                safety_override=safety_override
            )
            
            # Send OSC if mixer client available (Level 6)
            if self.mixer_client and self.mode == FaderMode.FULL_AUTO:
                try:
                    self.mixer_client.set_channel_fader(
                        state.mixer_channel,
                        self._db_to_linear(final_gain)
                    )
                except Exception as e:
                    logger.error(f"Failed to send OSC for ch{channel_id}: {e}")
        
        return decisions
    
    def _db_to_linear(self, db: float) -> float:
        """Convert dB to linear gain."""
        return 10 ** (db / 20)
    
    def set_mode(self, mode: FaderMode):
        """Set operating mode."""
        self.mode = mode
        logger.info(f"Mode changed to {mode.value}")
    
    def reset(self):
        """Reset controller state."""
        self.decision_engine.reset_integral()
        for state in self.channels.values():
            state.reset()
        logger.info("AutoFaderController reset")
    
    def get_status(self) -> Dict[str, Any]:
        """Get current status."""
        return {
            'mode': self.mode.value,
            'target_lufs': self.target_lufs,
            'master_hybrid_db': self.master_hybrid_db,
            'master_peak_db': self.master_peak_db,
            'emergency_active': self.safety_validator.emergency_active,
            'channels': {
                ch: {
                    'gain_db': state.current_gain_db,
                    'hybrid_db': state.features.hybrid_level_db,
                    'scenario': state.features.scenario.value
                }
                for ch, state in self.channels.items()
            }
        }


# Test function
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    print("=" * 70)
    print("Auto Fader - 6-Level Hybrid Architecture Test")
    print("=" * 70)
    
    # Test hybrid metric fusion
    print("\n1. Testing Hybrid Metric Fusion:")
    fusion = HybridMetricFusion()
    test_cases = [
        (-20, -22, -15, "Normal level"),
        (-10, -12, -6, "Loud"),
        (-30, -35, -25, "Quiet"),
    ]
    for lufs, rms, peak, desc in test_cases:
        hybrid = fusion.compute_hybrid_level(lufs, rms, peak)
        print(f"  {desc}: LUFS={lufs}, RMS={rms}, Peak={peak} → Hybrid={hybrid:.1f} dB")
    
    # Test scenario detection
    print("\n2. Testing Scenario Detection:")
    detector = ScenarioDetector()
    levels = [-70, -50, -25, -15, -8, -2]
    for level in levels:
        scenario = detector.detect_scenario(level, level + 3)
        rate_limit = detector.get_rate_limit_db(scenario)
        print(f"  Level={level:+.0f} dB → {scenario.value:12s} (rate_limit={rate_limit:.1f} dB/cycle)")
    
    # Test decision engine
    print("\n3. Testing Decision Engine:")
    engine = DecisionEngine(target_lufs=-18.0)
    scenarios = [
        (-25, Scenario.NORMAL),
        (-10, Scenario.LOUD),
        (-50, Scenario.QUIET),
    ]
    for level, scenario in scenarios:
        corr, limited, conf = engine.calculate_correction(1, level, scenario, 3.0)
        print(f"  Level={level:+.0f} dB, {scenario.value:8s} → correction={corr:+.1f} dB, "
              f"limited={limited:+.1f} dB, confidence={conf:.2f}")
    
    print("\n" + "=" * 70)
    print("Test complete!")
    print("=" * 70)
