"""
Gain Agent - ODA agent for automatic gain staging.

Monitors channel levels and builds proposal-only gain/trim evidence to maintain
target LUFS per channel. Thread-safe shared state.
"""

import logging
import threading
from typing import Any, Dict, Optional

from .base_agent import BaseAgent

logger = logging.getLogger(__name__)

GAIN_AGENT_WRITE_POLICY = "proposal_only_quarantined"


class GainAgent(BaseAgent):
    """
    Automatic gain staging agent.

    Observes current LUFS levels, decides on trim adjustments,
    and records blocked trim write evidence for operator review.
    """

    def __init__(self, mixer_client, target_lufs: float = -18.0,
                 cycle_interval: float = 1.0):
        super().__init__(name="GainAgent", cycle_interval=cycle_interval)
        self._mixer = mixer_client
        self._target_lufs = target_lufs
        # Thread-safe channel state: {ch_id: {"current_lufs": float, "trim_db": float, ...}}
        self._channel_states: Dict[int, Dict[str, Any]] = {}
        self._channel_lock = threading.Lock()
        self.live_write_allowed = False
        self.write_policy = GAIN_AGENT_WRITE_POLICY
        self.last_write_attempted = False
        self.last_write_blocked = False
        self.last_write_count = 0
        self.last_blocked_calls: list[Dict[str, Any]] = []

    # ------ Configuration ------

    def set_target_lufs(self, target: float):
        self._target_lufs = target

    def get_channel_states(self) -> Dict[int, Dict[str, Any]]:
        with self._channel_lock:
            return dict(self._channel_states)

    def get_write_policy_state(self) -> Dict[str, Any]:
        return {
            "live_write_allowed": False,
            "write_policy": self.write_policy,
            "last_write_attempted": self.last_write_attempted,
            "last_write_blocked": self.last_write_blocked,
            "last_write_count": self.last_write_count,
            "last_blocked_calls": list(self.last_blocked_calls),
        }

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
        """Record trim adjustments as blocked evidence without live writes."""
        adjustments = decision.get("adjustments", {})
        blocked_calls = []
        for ch_id, adj_db in adjustments.items():
            try:
                current_trim = getattr(self._mixer, 'get_channel_trim',
                                       lambda c: 0.0)(ch_id)
                new_trim = max(-18.0, min(18.0, current_trim + adj_db))
                blocked_call = {
                    "cmd": "set_channel_trim",
                    "path": f"/ch/{ch_id}/trim",
                    "channel": ch_id,
                    "current_trim_db": current_trim,
                    "target_trim_db": new_trim,
                    "delta_trim_db": adj_db,
                    "bounded": new_trim != current_trim + adj_db,
                    "target_lufs": self._target_lufs,
                    "semantic": "legacy_gain_agent_trim",
                }
                blocked_calls.append(blocked_call)
                with self._channel_lock:
                    self._channel_states[ch_id] = {
                        "trim_db": new_trim,
                        "adjustment": adj_db,
                        "target_lufs": self._target_lufs,
                        "blocked": True,
                    }
                logger.debug(
                    "GainAgent: blocked Ch%s trim %.1f -> %.1f dB",
                    ch_id,
                    current_trim,
                    new_trim,
                )
            except Exception as e:
                logger.error(f"GainAgent: Ch{ch_id} trim failed: {e}")
        self.last_blocked_calls = blocked_calls
        self.last_write_attempted = bool(blocked_calls)
        self.last_write_blocked = bool(blocked_calls)
        self.last_write_count = 0
