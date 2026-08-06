"""
EQ Agent - ODA agent for automatic equalization.

Monitors spectral balance and builds proposal-only EQ correction evidence
based on instrument profiles. Thread-safe shared state.
"""

import logging
import math
import threading
from typing import Any, Dict, Optional

from .base_agent import BaseAgent

logger = logging.getLogger(__name__)

EQ_AGENT_WRITE_POLICY = "proposal_only_quarantined"


class EQAgent(BaseAgent):
    """
    Automatic EQ adjustment agent.

    Observes spectral features, decides on EQ band adjustments,
    and records blocked EQ write evidence for operator review.
    """

    def __init__(self, mixer_client, cycle_interval: float = 2.0):
        super().__init__(name="EQAgent", cycle_interval=cycle_interval)
        self._mixer = mixer_client
        # Thread-safe channel state: {ch_id: {"bands": {...}, "corrections": {...}}}
        self._channel_states: Dict[int, Dict[str, Any]] = {}
        self._channel_lock = threading.Lock()
        self._instrument_profiles: Dict[int, str] = {}
        self.live_write_allowed = False
        self.write_policy = EQ_AGENT_WRITE_POLICY
        self.last_write_attempted = False
        self.last_write_blocked = False
        self.last_write_count = 0
        self.last_blocked_calls: list[Dict[str, Any]] = []

    # ------ Configuration ------

    def set_instrument_profiles(self, profiles: Dict[int, str]):
        """Set instrument type per channel for EQ target curves."""
        self._instrument_profiles = dict(profiles)

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
        """Build bounded proposals only from explicit measured spectral deltas."""
        eq_state = observation.get("eq_state", {})
        spectral_deltas = observation.get("spectral_gain_delta_db")
        provenance = observation.get("spectral_measurement_provenance")
        if not isinstance(spectral_deltas, dict) or not isinstance(provenance, dict):
            return {
                "status": "unavailable",
                "reason": "spectral_target_measurement_unavailable",
                "corrections": {},
            }
        source = str(provenance.get("source", "")).strip().lower()
        if provenance.get("measured") is not True or source in {"mock", "synthetic", "fixture"}:
            return {
                "status": "unavailable",
                "reason": "spectral_measurement_provenance_untrusted",
                "corrections": {},
            }

        corrections = {}

        for ch_id, bands in eq_state.items():
            instrument = self._instrument_profiles.get(ch_id, "unknown")
            if instrument == "unknown":
                continue
            ch_corrections = {}
            channel_deltas = spectral_deltas.get(ch_id, spectral_deltas.get(str(ch_id), {}))
            if not isinstance(channel_deltas, dict):
                continue
            for band_id in bands:
                raw_delta = channel_deltas.get(band_id, channel_deltas.get(str(band_id)))
                if isinstance(raw_delta, bool) or not isinstance(raw_delta, (int, float)):
                    continue
                if not math.isfinite(float(raw_delta)):
                    continue
                bounded_delta = max(-3.0, min(3.0, float(raw_delta)))
                if abs(bounded_delta) >= 0.1:
                    ch_corrections[band_id] = bounded_delta
            if ch_corrections:
                corrections[ch_id] = ch_corrections

        if corrections:
            return {
                "status": "proposal_only",
                "measurement_provenance": dict(provenance),
                "corrections": corrections,
            }
        return {
            "status": "no_change",
            "measurement_provenance": dict(provenance),
            "corrections": {},
        }

    async def act(self, decision: Dict[str, Any]) -> None:
        """Record EQ corrections as blocked evidence without live writes."""
        corrections = decision.get("corrections", {})
        blocked_calls = []
        for ch_id, bands in corrections.items():
            for band_id, adj_db in bands.items():
                try:
                    gain_addr = f"/ch/{ch_id}/eq/{band_id}/g"
                    state = getattr(self._mixer, 'state', {})
                    current = state.get(f"ch/{ch_id}/eq/{band_id}/g", 0.0)
                    new_gain = max(-15.0, min(15.0, current + adj_db))
                    blocked_call = {
                        "cmd": "set_eq_band_gain",
                        "path": gain_addr,
                        "channel": ch_id,
                        "band": band_id,
                        "current_gain_db": current,
                        "target_gain_db": new_gain,
                        "delta_gain_db": adj_db,
                        "bounded": new_gain != current + adj_db,
                        "semantic": "legacy_eq_agent_band_gain",
                    }
                    blocked_calls.append(blocked_call)
                    with self._channel_lock:
                        if ch_id not in self._channel_states:
                            self._channel_states[ch_id] = {"bands": {}, "corrections": {}}
                        self._channel_states[ch_id]["corrections"][band_id] = {
                            "delta_gain_db": adj_db,
                            "target_gain_db": new_gain,
                            "blocked": True,
                        }
                    logger.debug(
                        "EQAgent: blocked Ch%s band%s gain %.1f -> %.1f dB",
                        ch_id,
                        band_id,
                        current,
                        new_gain,
                    )
                except Exception as e:
                    logger.error(f"EQAgent: Ch{ch_id} band{band_id} failed: {e}")
        self.last_blocked_calls = blocked_calls
        self.last_write_attempted = bool(blocked_calls)
        self.last_write_blocked = bool(blocked_calls)
        self.last_write_count = 0
