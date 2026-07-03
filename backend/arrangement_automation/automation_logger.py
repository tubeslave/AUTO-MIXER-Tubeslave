"""Decision logging for arrangement-aware automation."""

from __future__ import annotations

import logging
import time
from collections import deque
from typing import Any, Deque, Dict, Iterable

from .models import ActivityState, ChannelRoleInfo, MaskingSource, PlannedOffset

logger = logging.getLogger(__name__)


class AutomationDecisionLogger:
    """Store and emit explainable automation decisions."""

    def __init__(self, enabled: bool = True, max_entries: int = 1000):
        self.enabled = bool(enabled)
        self.history: Deque[Dict[str, Any]] = deque(maxlen=max_entries)

    def log_cycle(
        self,
        roles: Dict[int, ChannelRoleInfo],
        activities: Dict[int, ActivityState],
        section: str,
        density_index: float,
        masking_sources: Iterable[MaskingSource],
        planned_offsets: Dict[int, PlannedOffset],
        safe_offsets: Dict[int, float],
        blocked_offsets: Dict[int, Dict[str, Any]],
        applied_offsets: Dict[int, float],
        safety_reasons: Dict[int, list[str]],
    ) -> None:
        if not self.enabled:
            return
        masking_by_channel = {src.channel_id: src for src in masking_sources}
        now = time.time()
        for channel_id, planned in planned_offsets.items():
            role = roles.get(channel_id)
            activity = activities.get(channel_id)
            masking = masking_by_channel.get(channel_id)
            entry = {
                "timestamp": now,
                "channel_id": channel_id,
                "channel_name": role.channel_name if role else planned.channel_name,
                "role": planned.role,
                "section": section,
                "ADI": round(float(density_index), 4),
                "active": bool(activity.active) if activity else False,
                "masking_score": masking.masking_score if masking else 0.0,
                "requested_offset_db": planned.requested_offset_db,
                "final_safe_offset_db": safe_offsets.get(channel_id, 0.0),
                "OSC address": f"/ch/{channel_id}/fdr",
                "applied": channel_id in applied_offsets,
                "blocked": channel_id in blocked_offsets,
                "reason": "; ".join(safety_reasons.get(channel_id, [])) or planned.reason,
            }
            self.history.append(entry)
            if channel_id in applied_offsets or channel_id in blocked_offsets:
                logger.info("Arrangement automation decision: %s", entry)

    def recent(self, limit: int = 50) -> list[Dict[str, Any]]:
        return list(self.history)[-limit:]
