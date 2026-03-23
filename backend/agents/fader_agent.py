"""
Fader Agent - ODA agent for automatic fader balancing.

Monitors channel levels and adjusts faders to maintain
the target mix balance. Thread-safe shared state.
"""

import logging
import threading
from typing import Any, Dict, Optional

from .base_agent import BaseAgent

logger = logging.getLogger(__name__)


class FaderAgent(BaseAgent):
    """
    Automatic fader balancing agent.

    Observes current levels and fader positions, decides on
    fader moves, and acts by sending OSC commands to the mixer.
    """

    def __init__(self, mixer_client, target_lufs: float = -18.0,
                 cycle_interval: float = 0.5):
        super().__init__(name="FaderAgent", cycle_interval=cycle_interval)
        self._mixer = mixer_client
        self._target_lufs = target_lufs
        self._dead_zone_db = 3.0
        self._max_adjustment_db = 1.5
        # Thread-safe channel state
        self._channel_states: Dict[int, Dict[str, Any]] = {}
        self._channel_lock = threading.Lock()
        self._reference_channels: list[int] = []

    # ------ Configuration ------

    def set_target_lufs(self, target: float):
        self._target_lufs = target

    def set_reference_channels(self, channels: list[int]):
        self._reference_channels = list(channels)

    def get_channel_states(self) -> Dict[int, Dict[str, Any]]:
        with self._channel_lock:
            return dict(self._channel_states)

    # ------ ODA cycle ------

    async def observe(self) -> Dict[str, Any]:
        """Read current fader positions and levels."""
        faders = {}
        levels = {}
        state = getattr(self._mixer, 'state', {})
        for ch_id in range(1, 49):
            fdr_key = f"ch/{ch_id}/fdr"
            lvl_key = f"ch/{ch_id}/level"
            if fdr_key in state:
                faders[ch_id] = state[fdr_key]
            if lvl_key in state:
                levels[ch_id] = state[lvl_key]
        return {"faders": faders, "levels": levels}

    async def decide(self, observation: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate fader adjustments for channels outside dead zone."""
        levels = observation.get("levels", {})
        faders = observation.get("faders", {})
        adjustments = {}

        for ch_id, current_lufs in levels.items():
            if ch_id in self._reference_channels:
                continue
            if current_lufs < -70:
                continue  # silent

            diff = self._target_lufs - current_lufs
            if abs(diff) <= self._dead_zone_db:
                continue

            # Correct only beyond dead zone
            if diff > 0:
                effective = diff - self._dead_zone_db
            else:
                effective = diff + self._dead_zone_db

            adj = max(-self._max_adjustment_db,
                      min(self._max_adjustment_db, effective * 0.4))
            adjustments[ch_id] = adj

        if adjustments:
            return {"adjustments": adjustments, "faders": faders}
        return {}

    async def act(self, decision: Dict[str, Any]) -> None:
        """Apply fader adjustments via OSC."""
        adjustments = decision.get("adjustments", {})
        faders = decision.get("faders", {})
        for ch_id, adj_db in adjustments.items():
            try:
                current_fader = faders.get(ch_id, 0.75)
                # Convert dB adjustment to fader scale (approximate: 10dB ≈ 0.1 fader)
                fader_delta = adj_db / 100.0
                new_fader = max(0.0, min(1.0, current_fader + fader_delta))
                self._mixer.send(f"/ch/{ch_id}/fdr", new_fader)
                with self._channel_lock:
                    self._channel_states[ch_id] = {
                        "fader": new_fader,
                        "adjustment_db": adj_db,
                        "target_lufs": self._target_lufs,
                    }
                logger.debug(f"FaderAgent: Ch{ch_id} fader {current_fader:.3f} -> {new_fader:.3f}")
            except Exception as e:
                logger.error(f"FaderAgent: Ch{ch_id} fader failed: {e}")
