"""
PID Controller for Loudness Balance

Implements Proportional-Integral-Derivative controller for maintaining
target LUFS levels as recommended in the "От LUFS до OSC-команд" document.

Error = target_lufs - current_lufs
Output = Kp * error + Ki * integral + Kd * derivative
"""

import logging
import time
from typing import Dict, Optional
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


@dataclass
class PIDState:
    """State for a single channel PID controller"""
    last_error: float = 0.0
    integral: float = 0.0
    last_time: Optional[float] = None
    last_output: float = 0.0


class PIDLoudnessController:
    """
    PID controller for maintaining loudness balance.
    
    According to the document:
    - P (Proportional): Reacts to current error
    - I (Integral): Eliminates steady-state error (with anti-windup)
    - D (Derivative): Reduces oscillations (optional, can start with PI)
    """
    
    def __init__(
        self,
        kp: float = 0.5,
        ki: float = 0.05,
        kd: float = 0.0,
        integral_limit: float = 10.0,
        output_limit: float = 6.0,
        dead_zone: float = 0.5
    ):
        """
        Initialize PID controller.
        
        Args:
            kp: Proportional gain
            ki: Integral gain
            kd: Derivative gain
            integral_limit: Maximum absolute value for integral term (anti-windup)
            output_limit: Maximum absolute output adjustment in dB
            dead_zone: Don't adjust if error is smaller than this (dB)
        """
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.integral_limit = integral_limit
        self.output_limit = output_limit
        self.dead_zone = dead_zone
        
        # Per-channel state
        self.channel_states: Dict[int, PIDState] = {}
        
        logger.info(f"PID Controller initialized: Kp={kp}, Ki={ki}, Kd={kd}, "
                   f"integral_limit={integral_limit}, output_limit={output_limit}, "
                   f"dead_zone={dead_zone}")
    
    def reset_channel(self, channel_id: int):
        """Reset PID state for a channel"""
        if channel_id in self.channel_states:
            del self.channel_states[channel_id]
    
    def reset_all(self):
        """Reset all channel states"""
        self.channel_states.clear()
    
    def calculate_adjustments(
        self,
        current_levels: Dict[int, float],  # channel_id -> current LUFS
        target_levels: Dict[int, float],   # channel_id -> target LUFS
        dt: Optional[float] = None
    ) -> Dict[int, float]:
        """
        Calculate PID adjustments for channels.
        
        Args:
            current_levels: Current LUFS levels per channel
            target_levels: Target LUFS levels per channel
            dt: Time delta since last call (seconds). If None, will calculate automatically.
        
        Returns:
            Dictionary of channel_id -> adjustment_db
        """
        adjustments = {}
        current_time = time.time()
        
        for channel_id, current_lufs in current_levels.items():
            if channel_id not in target_levels:
                continue
            
            target_lufs = target_levels[channel_id]
            
            # Initialize state if needed
            if channel_id not in self.channel_states:
                self.channel_states[channel_id] = PIDState()
            
            state = self.channel_states[channel_id]
            
            # Calculate error
            error = target_lufs - current_lufs
            
            # Dead zone: don't adjust if error is too small
            if abs(error) < self.dead_zone:
                adjustments[channel_id] = 0.0
                # Still update state for derivative calculation
                state.last_error = error
                state.last_time = current_time
                continue
            
            # Calculate time delta
            if state.last_time is None:
                dt_actual = 0.1  # Default 100ms if first call
            else:
                dt_actual = dt if dt is not None else (current_time - state.last_time)
                # Clamp dt to reasonable range (10ms to 1s)
                dt_actual = max(0.01, min(1.0, dt_actual))
            
            # Proportional term
            p_term = self.kp * error
            
            # Integral term with anti-windup
            state.integral += error * dt_actual
            # Clamp integral to prevent windup
            state.integral = max(-self.integral_limit, min(self.integral_limit, state.integral))
            i_term = self.ki * state.integral
            
            # Derivative term (rate of change of error)
            if self.kd > 0 and dt_actual > 0 and state.last_time is not None:
                error_rate = (error - state.last_error) / dt_actual
                d_term = self.kd * error_rate
            else:
                d_term = 0.0
            
            # Calculate output
            output = p_term + i_term + d_term
            
            # Limit output
            output = max(-self.output_limit, min(self.output_limit, output))
            
            adjustments[channel_id] = output
            
            # Update state
            state.last_error = error
            state.last_time = current_time
            state.last_output = output
        
        return adjustments
    
    def get_state(self, channel_id: int) -> Optional[Dict]:
        """Get current PID state for a channel (for debugging)"""
        if channel_id not in self.channel_states:
            return None
        
        state = self.channel_states[channel_id]
        return {
            "last_error": state.last_error,
            "integral": state.integral,
            "last_output": state.last_output,
            "last_time": state.last_time
        }
    
    def update_gains(self, kp: Optional[float] = None, ki: Optional[float] = None, 
                     kd: Optional[float] = None):
        """Update PID gains dynamically"""
        if kp is not None:
            self.kp = kp
        if ki is not None:
            self.ki = ki
        if kd is not None:
            self.kd = kd
        logger.info(f"PID gains updated: Kp={self.kp}, Ki={self.ki}, Kd={self.kd}")
