"""
Vocal Activity Detector for Ducking

Detects when vocal channels are active to trigger ducking of background channels.
Based on threshold detection as recommended in the "От LUFS до OSC-команд" document.
"""

import logging
import time
from typing import Dict, List, Optional
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class VocalActivityState:
    """State for vocal activity detection"""
    is_active: bool = False
    last_active_time: float = 0.0
    attack_smoothing: float = 0.0  # 0.0-1.0 for smooth transitions
    release_smoothing: float = 0.0  # 0.0-1.0 for smooth transitions


class VocalActivityDetector:
    """
    Detects vocal activity for ducking control.
    
    When vocal level exceeds threshold, background channels should be ducked.
    Uses attack/release smoothing for smooth transitions.
    """
    
    def __init__(
        self,
        threshold_lufs: float = -30.0,
        attack_ms: float = 10.0,
        release_ms: float = 100.0,
        hold_ms: float = 50.0
    ):
        """
        Initialize vocal activity detector.
        
        Args:
            threshold_lufs: LUFS threshold for vocal activity (default -30 dB)
            attack_ms: Attack time in milliseconds (how fast to detect activity)
            release_ms: Release time in milliseconds (how fast to release after silence)
            hold_ms: Hold time in milliseconds (minimum active duration)
        """
        self.threshold_lufs = threshold_lufs
        self.attack_ms = attack_ms
        self.release_ms = release_ms
        self.hold_ms = hold_ms
        
        # Per-channel state
        self.channel_states: Dict[int, VocalActivityState] = {}
        
        # Calculate smoothing coefficients
        self.attack_coef = 1.0 - (1.0 / (1.0 + attack_ms / 10.0)) if attack_ms > 0 else 1.0
        self.release_coef = 1.0 - (1.0 / (1.0 + release_ms / 10.0)) if release_ms > 0 else 1.0
        
        logger.info(f"VocalActivityDetector initialized: threshold={threshold_lufs} dB, "
                   f"attack={attack_ms}ms, release={release_ms}ms, hold={hold_ms}ms")
    
    def update(self, vocal_levels: Dict[int, float], dt: Optional[float] = None):
        """
        Update vocal activity detection.
        
        Args:
            vocal_levels: Dictionary of channel_id -> current LUFS level
            dt: Time delta since last update (seconds). If None, will calculate automatically.
        
        Returns:
            Dictionary of channel_id -> is_active (bool)
        """
        current_time = time.time()
        
        if dt is None:
            # Estimate dt from last update time
            if hasattr(self, '_last_update_time'):
                dt = current_time - self._last_update_time
            else:
                dt = 0.1  # Default 100ms
            self._last_update_time = current_time
        
        results = {}
        
        for channel_id, level in vocal_levels.items():
            # Initialize state if needed
            if channel_id not in self.channel_states:
                self.channel_states[channel_id] = VocalActivityState()
            
            state = self.channel_states[channel_id]
            
            # Check if level exceeds threshold
            above_threshold = level >= self.threshold_lufs
            
            # Smooth attack (when becoming active)
            if above_threshold:
                state.attack_smoothing = min(1.0, state.attack_smoothing + self.attack_coef)
            else:
                state.attack_smoothing = max(0.0, state.attack_smoothing - self.release_coef)
            
            # Determine activity based on smoothed value
            was_active = state.is_active
            state.is_active = state.attack_smoothing > 0.5
            
            # Hold time: once active, stay active for minimum duration
            if state.is_active and not was_active:
                # Just became active
                state.last_active_time = current_time
            elif was_active and not state.is_active:
                # Just became inactive - check hold time
                time_since_active = current_time - state.last_active_time
                if time_since_active < (self.hold_ms / 1000.0):
                    # Still in hold period
                    state.is_active = True
            
            results[channel_id] = state.is_active
        
        return results
    
    def is_any_vocal_active(self, vocal_channels: List[int]) -> bool:
        """
        Check if any vocal channel is currently active.
        
        Args:
            vocal_channels: List of vocal channel IDs
        
        Returns:
            True if any vocal channel is active
        """
        for ch_id in vocal_channels:
            if ch_id in self.channel_states:
                if self.channel_states[ch_id].is_active:
                    return True
        return False
    
    def get_activity_level(self, vocal_channels: List[int]) -> float:
        """
        Get overall vocal activity level (0.0-1.0) for smooth ducking.
        
        Args:
            vocal_channels: List of vocal channel IDs
        
        Returns:
            Activity level 0.0 (silent) to 1.0 (fully active)
        """
        if not vocal_channels:
            return 0.0
        
        total_activity = 0.0
        count = 0
        
        for ch_id in vocal_channels:
            if ch_id in self.channel_states:
                state = self.channel_states[ch_id]
                total_activity += state.attack_smoothing
                count += 1
        
        return total_activity / max(count, 1) if count > 0 else 0.0
    
    def reset(self):
        """Reset all channel states"""
        self.channel_states.clear()
        if hasattr(self, '_last_update_time'):
            delattr(self, '_last_update_time')
