"""Rule-based song section detector with hysteresis."""

from __future__ import annotations

import time
from typing import Dict, Optional, Tuple

from .models import ActivityState, ChannelRoleInfo, DensityState, SectionState


class SectionDetector:
    """Estimate song section from role activity and arrangement density."""

    def __init__(self, hold_time_sec: float = 3.0):
        self.hold_time_sec = float(hold_time_sec)
        self.section = "unknown"
        self.previous_section = "unknown"
        self.section_since: Optional[float] = None
        self._candidate: Optional[str] = None
        self._candidate_since: Optional[float] = None

    def update(
        self,
        activities: Dict[int, ActivityState],
        roles: Dict[int, ChannelRoleInfo],
        density: DensityState,
        timestamp: Optional[float] = None,
    ) -> SectionState:
        now = time.time() if timestamp is None else float(timestamp)
        candidate, confidence, reason = self._candidate_section(activities, roles, density)

        if self.section == "unknown" and self.section_since is None:
            self._commit(candidate, now)
        elif candidate != self.section:
            if self.hold_time_sec <= 0.0:
                self._commit(candidate, now)
                self._candidate = None
                self._candidate_since = None
            elif candidate != self._candidate:
                self._candidate = candidate
                self._candidate_since = now
            elif (
                self._candidate_since is not None
                and now - self._candidate_since >= self.hold_time_sec
            ):
                self._commit(candidate, now)
                self._candidate = None
                self._candidate_since = None
        else:
            self._candidate = None
            self._candidate_since = None

        time_in_section = 0.0 if self.section_since is None else max(0.0, now - self.section_since)
        return SectionState(
            section=self.section,
            confidence=confidence,
            previous_section=self.previous_section,
            time_in_section_sec=round(time_in_section, 3),
            reason=reason,
        )

    def _commit(self, section: str, timestamp: float) -> None:
        if section != self.section:
            self.previous_section = self.section
            self.section = section
            self.section_since = timestamp

    @staticmethod
    def _candidate_section(
        activities: Dict[int, ActivityState],
        roles: Dict[int, ChannelRoleInfo],
        density: DensityState,
    ) -> Tuple[str, float, str]:
        active_roles = {
            roles[ch].role
            for ch, state in activities.items()
            if state.active and ch in roles
        }
        lead = "lead_vocal" in active_roles
        backing = "backing_vocal" in active_roles
        bass = "bass" in active_roles
        full_drums = (
            "kick" in active_roles
            and ("snare" in active_roles or "overhead" in active_roles)
        )
        solo = bool({"solo_instrument", "lead_guitar"} & active_roles)
        active_count = density.active_channel_count
        adi = density.density_index

        if not lead and solo and (bass or full_drums):
            return "solo", 0.78, "lead vocal inactive and solo source active"
        if lead and backing and full_drums and adi >= 0.62:
            return "chorus", 0.82, "lead vocal, backing vocals, full groove, high ADI"
        if lead and backing and 0.45 <= adi < 0.72:
            return "pre_chorus", 0.68, "lead vocal with backing vocals and rising density"
        if lead:
            return "verse", 0.74, "lead vocal active with moderate arrangement"
        if not lead and active_count <= 2 and adi < 0.30:
            return "intro", 0.55, "low density without lead vocal"
        if not lead and adi < 0.35:
            return "breakdown", 0.58, "reduced density without lead vocal"
        if active_count == 0:
            return "unknown", 0.3, "no active channels"
        return "unknown", 0.45, "ambiguous arrangement"
