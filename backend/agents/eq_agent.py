"""
EQ Agent - ODA agent for automatic equalization.

Monitors spectral balance and applies EQ corrections
based on instrument profiles. Thread-safe shared state.
"""

import logging
import threading
from typing import Any, Dict, Optional

from .base_agent import BaseAgent

logger = logging.getLogger(__name__)


class EQAgent(BaseAgent):
    """
    Automatic EQ adjustment agent.

    Observes spectral features, decides on EQ band adjustments,
    and acts by sending OSC commands to the mixer.
    """

    def __init__(self, mixer_client, cycle_interval: float = 2.0):
        super().__init__(name="EQAgent", cycle_interval=cycle_interval)
        self._mixer = mixer_client
        # Thread-safe channel state: {ch_id: {"bands": {...}, "corrections": {...}}}
        self._channel_states: Dict[int, Dict[str, Any]] = {}
        self._channel_lock = threading.Lock()
        self._instrument_profiles: Dict[int, str] = {}

    # ------ Configuration ------

    def set_instrument_profiles(self, profiles: Dict[int, str]):
        """Set instrument type per channel for EQ target curves."""
        self._instrument_profiles = dict(profiles)

    def get_channel_states(self) -> Dict[int, Dict[str, Any]]:
        with self._channel_lock:
            return dict(self._channel_states)

    # ------ ODA cycle ------

    async def observe(self) -> Dict[str, Any]:
        """Read current EQ state and spectral data from mixer."""
        eq_data = {}
        state = getattr(self._mixer, 'state', {})
        for ch_id in range(1, 49):
            bands = {}
            for band in range(1, 5):
                freq_key = f"ch/{ch_id}/eq/{band}/f"
                gain_key = f"ch/{ch_id}/eq/{band}/g"
                if freq_key in state:
                    bands[band] = {
                        "freq": state.get(freq_key, 1000.0),
                        "gain": state.get(gain_key, 0.0),
                    }
            if bands:
                eq_data[ch_id] = bands
        return {"eq_state": eq_data}

    async def decide(self, observation: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate EQ corrections based on instrument profiles."""
        eq_state = observation.get("eq_state", {})
        corrections = {}

        for ch_id, bands in eq_state.items():
            instrument = self._instrument_profiles.get(ch_id, "unknown")
            if instrument == "unknown":
                continue
            # Placeholder: in a full implementation this would compare
            # current spectral shape against target curves and produce
            # per-band gain deltas.
            ch_corrections = {}
            for band_id, band_data in bands.items():
                current_gain = band_data.get("gain", 0.0)
                # Only correct large deviations (> 3 dB from flat)
                if abs(current_gain) > 3.0:
                    ch_corrections[band_id] = -current_gain * 0.2  # gentle pull toward flat
            if ch_corrections:
                corrections[ch_id] = ch_corrections

        if corrections:
            return {"corrections": corrections}
        return {}

    async def act(self, decision: Dict[str, Any]) -> None:
        """Apply EQ corrections via OSC."""
        corrections = decision.get("corrections", {})
        for ch_id, bands in corrections.items():
            for band_id, adj_db in bands.items():
                try:
                    gain_addr = f"/ch/{ch_id}/eq/{band_id}/g"
                    state = getattr(self._mixer, 'state', {})
                    current = state.get(f"ch/{ch_id}/eq/{band_id}/g", 0.0)
                    new_gain = max(-15.0, min(15.0, current + adj_db))
                    self._mixer.send(gain_addr, new_gain)
                    with self._channel_lock:
                        if ch_id not in self._channel_states:
                            self._channel_states[ch_id] = {"bands": {}, "corrections": {}}
                        self._channel_states[ch_id]["corrections"][band_id] = adj_db
                    logger.debug(f"EQAgent: Ch{ch_id} band{band_id} gain {current:.1f} -> {new_gain:.1f} dB")
                except Exception as e:
                    logger.error(f"EQAgent: Ch{ch_id} band{band_id} failed: {e}")
