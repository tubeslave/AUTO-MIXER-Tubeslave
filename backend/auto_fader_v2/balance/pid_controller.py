"""
Cross-adaptive gain-sharing controller for loudness balance.

Implements cross-adaptive average normalization (IMP, Section 4.3):

    c_m[n] = l_avg[n] - l_m[n]

with EMA smoothing (IMP, Eq. 4.2):

    c_m[n+1] = alpha * c'_m[n+1] + (1 - alpha) * c_m[n]

When per-channel targets are provided, the instantaneous estimate is the
difference between the target and the current level for that channel,
as the target already encodes the desired relative balance derived from
best practices (e.g., vocals ~3 LU below total mix loudness, see IMP 7.1.1).
"""

import logging
from typing import Dict, Optional

logger = logging.getLogger(__name__)


class GainSharingController:
    """
    Cross-adaptive gain-sharing with EMA smoothing.

    Based on IMP Section 4.3 and Eq. 4.2:
    - Feature extraction per channel gives l_m[n]
    - Cross-adaptive control: c_m[n] = l_avg[n] - l_m[n]
    - EMA smoothing: c_m[n+1] = alpha * c'_m[n+1] + (1 - alpha) * c_m[n]
    - Adaptive gating: control vector only updated when x_RMS,m > r_RMS
    """

    def __init__(
        self,
        alpha: float = 0.3,
        output_limit: float = 6.0,
        dead_zone: float = 0.5,
        gate_threshold: float = -50.0,
    ):
        self.alpha = max(0.01, min(1.0, alpha))
        self.output_limit = float(output_limit)
        self.dead_zone = float(dead_zone)
        self.gate_threshold = float(gate_threshold)
        self.current_gains: Dict[int, float] = {}

        logger.info(
            "GainSharingController initialized: "
            f"alpha={self.alpha}, output_limit={self.output_limit}, "
            f"dead_zone={self.dead_zone}, gate_threshold={self.gate_threshold}"
        )

    def reset_channel(self, channel_id: int):
        self.current_gains.pop(channel_id, None)

    def reset_all(self):
        self.current_gains.clear()

    def calculate_adjustments(
        self,
        current_levels: Dict[int, float],
        target_levels: Dict[int, float],
        dt: Optional[float] = None,  # kept for compatibility
    ) -> Dict[int, float]:
        del dt
        # Adaptive gating (IMP Eq. 4.2): skip channels below gate threshold.
        active_levels = {
            ch: lvl for ch, lvl in current_levels.items() if lvl > self.gate_threshold
        }
        if not active_levels:
            return {}

        # Cross-adaptive average l_avg[n] (IMP Section 4.3)
        avg_level = sum(active_levels.values()) / max(len(active_levels), 1)
        adjustments: Dict[int, float] = {}

        for channel_id, current_lufs in active_levels.items():
            # Instantaneous control estimate c'_m[n] (IMP Section 4.3):
            #   c_m[n] = l_avg[n] - l_m[n]
            # When per-channel targets are provided, use target as the
            # desired level for that channel (IMP 7.1.1: targets encode
            # instrument-dependent balance such as vocal prominence).
            if channel_id in target_levels:
                instant_gain = target_levels[channel_id] - current_lufs
            else:
                instant_gain = avg_level - current_lufs

            if abs(instant_gain) < self.dead_zone:
                instant_gain = 0.0

            # EMA smoothing (IMP Eq. 4.2):
            #   c_m[n+1] = alpha * c'_m[n+1] + (1 - alpha) * c_m[n]
            prev = self.current_gains.get(channel_id, 0.0)
            smoothed = (self.alpha * instant_gain) + ((1.0 - self.alpha) * prev)
            smoothed = max(-self.output_limit, min(self.output_limit, smoothed))

            self.current_gains[channel_id] = smoothed
            adjustments[channel_id] = smoothed

        return adjustments

    def get_state(self, channel_id: int) -> Optional[Dict]:
        if channel_id not in self.current_gains:
            return None
        return {"last_output": self.current_gains[channel_id]}

    def update_gains(
        self,
        kp: Optional[float] = None,
        ki: Optional[float] = None,
        kd: Optional[float] = None,
        alpha: Optional[float] = None,
    ):
        # Compatibility signature: kp/ki/kd ignored for new controller.
        del kp, ki, kd
        if alpha is not None:
            self.alpha = max(0.01, min(1.0, float(alpha)))
        logger.info(
            f"GainSharingController updated: alpha={self.alpha}, output_limit={self.output_limit}"
        )


# Backward-compatible alias for old imports.
PIDLoudnessController = GainSharingController
