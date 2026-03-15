"""
Gain Agent - ODA agent for automatic gain staging.

Monitors channel levels and adjusts gain/trim to maintain
target LUFS per channel. Thread-safe shared state.
"""

import logging
import threading
from typing import Any, Dict, Optional

from .base_agent import BaseAgent

logger = logging.getLogger(__name__)


class GainAgent(BaseAgent):
    """
    Automatic gain staging agent.

    Observes current LUFS levels, decides on trim adjustments,
    and acts by sending OSC commands to the mixer.
    """

    def __init__(self, mixer_client, target_lufs: float = -18.0,
                 cycle_interval: float = 1.0):
        super().__init__(name="GainAgent", cycle_interval=cycle_interval)
        self._mixer = mixer_client
        self._target_lufs = target_lufs
        # Thread-safe channel state: {ch_id: {"current_lufs": float, "trim_db": float, ...}}
        self._channel_states: Dict[int, Dict[str, Any]] = {}
        self._channel_lock = threading.Lock()

    # ------ Configuration ------

    def set_target_lufs(self, target: float):
        self._target_lufs = target

    def get_channel_states(self) -> Dict[int, Dict[str, Any]]:
        with self._channel_lock:
            return dict(self._channel_states)

    # ------ ODA cycle ------

    async def observe(self) -> Dict[str, Any]:
        """Read current LUFS levels from mixer state."""
        levels = {}
        state = getattr(self._mixer, 'state', {})
        for ch_id in range(1, 49):
            key = f"ch/{ch_id}/level"
            if key in state:
                levels[ch_id] = state[key]
        return {"levels": levels}

    async def decide(self, observation: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate trim adjustments for channels outside target range."""
        levels = observation.get("levels", {})
        adjustments = {}
        tolerance_db = 1.5

        for ch_id, current_lufs in levels.items():
            if current_lufs < -70:
                continue  # channel is silent
            diff = self._target_lufs - current_lufs
            if abs(diff) > tolerance_db:
                # Proportional correction, clamped
                adj = max(-3.0, min(3.0, diff * 0.3))
                adjustments[ch_id] = adj

        if adjustments:
            return {"adjustments": adjustments}
        return {}

    async def act(self, decision: Dict[str, Any]) -> None:
        """Apply trim adjustments via OSC."""
        adjustments = decision.get("adjustments", {})
        for ch_id, adj_db in adjustments.items():
            try:
                current_trim = getattr(self._mixer, 'get_channel_trim',
                                       lambda c: 0.0)(ch_id)
                new_trim = max(-18.0, min(18.0, current_trim + adj_db))
                self._mixer.send(f"/ch/{ch_id}/trim", new_trim)
                with self._channel_lock:
                    self._channel_states[ch_id] = {
                        "trim_db": new_trim,
                        "adjustment": adj_db,
                        "target_lufs": self._target_lufs,
                    }
                logger.debug(f"GainAgent: Ch{ch_id} trim {current_trim:.1f} -> {new_trim:.1f} dB")
            except Exception as e:
                logger.error(f"GainAgent: Ch{ch_id} trim failed: {e}")
