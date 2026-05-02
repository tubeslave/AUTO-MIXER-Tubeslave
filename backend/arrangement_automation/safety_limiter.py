"""Safety limiter for arrangement automation offsets."""

from __future__ import annotations

from typing import Any, Dict, Optional

from .channel_role_classifier import DEFAULT_ROLE_LIMITS
from .models import ChannelRoleInfo, PlannedOffset, SafetyResult


class SafetyLimiter:
    """Clamp, smooth, and block unsafe automation before OSC apply."""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        cfg = config or {}
        self.max_step_db_per_tick = float(cfg.get("max_step_db_per_tick", 0.25))
        self.deadband_db = float(cfg.get("deadband_db", 0.1))
        self.min_confidence_for_live_apply = float(cfg.get("min_confidence_for_live_apply", 0.65))
        self.very_low_confidence = float(cfg.get("very_low_confidence", 0.3))
        self.fader_ceiling_db = float(cfg.get("fader_ceiling_db", 0.0))
        self.emergency_bypass = bool(cfg.get("emergency_bypass", False))
        self.role_limits = dict(DEFAULT_ROLE_LIMITS)
        for role, values in (cfg.get("role_limits") or {}).items():
            merged = dict(self.role_limits.get(role, self.role_limits["unknown"]))
            merged.update(values)
            self.role_limits[role] = merged
        self._previous_offsets: Dict[int, float] = {}

    def limit(
        self,
        planned_offsets: Dict[int, PlannedOffset],
        roles: Dict[int, ChannelRoleInfo],
        base_fader_positions: Optional[Dict[int, float]] = None,
        automation_enabled: bool = True,
        live_apply_enabled: bool = False,
    ) -> SafetyResult:
        safe_offsets: Dict[int, float] = {}
        blocked_offsets: Dict[int, Dict[str, Any]] = {}
        reasons: Dict[int, list[str]] = {}
        confidences = []

        for channel_id, planned in planned_offsets.items():
            channel_reasons: list[str] = []
            previous = self._previous_offsets.get(channel_id, 0.0)
            role = planned.role
            limits = self.role_limits.get(role, self.role_limits["unknown"])
            min_db = float(limits.get("min_db", -2.0))
            max_db = float(limits.get("max_db", 2.0))
            requested = float(planned.requested_offset_db)
            confidence = float(planned.confidence)
            confidences.append(confidence)

            if self.emergency_bypass:
                blocked_offsets[channel_id] = self._blocked(planned, "emergency_bypass")
                reasons[channel_id] = ["emergency_bypass"]
                safe_offsets[channel_id] = previous
                continue
            if not automation_enabled:
                blocked_offsets[channel_id] = self._blocked(planned, "automation_disabled")
                reasons[channel_id] = ["automation_disabled"]
                safe_offsets[channel_id] = previous
                continue
            if live_apply_enabled and confidence < self.very_low_confidence:
                blocked_offsets[channel_id] = self._blocked(planned, "confidence_too_low")
                reasons[channel_id] = ["confidence_too_low"]
                safe_offsets[channel_id] = previous
                continue
            if live_apply_enabled and confidence < self.min_confidence_for_live_apply:
                blocked_offsets[channel_id] = self._blocked(
                    planned,
                    "below_live_confidence_threshold",
                )
                reasons[channel_id] = ["below_live_confidence_threshold"]
                safe_offsets[channel_id] = previous
                continue

            clamped = max(min_db, min(max_db, requested))
            if clamped != requested:
                channel_reasons.append("role_limit_clamped")

            delta = clamped - previous
            step_limit = self.max_step_db_per_tick
            if confidence < self.min_confidence_for_live_apply:
                step_limit *= 0.5
                channel_reasons.append("low_confidence_soft_step")
            if abs(delta) > step_limit:
                clamped = previous + step_limit * (1.0 if delta > 0 else -1.0)
                channel_reasons.append("max_step_limited")

            base = (base_fader_positions or {}).get(channel_id)
            if base is not None and base + clamped > self.fader_ceiling_db:
                clamped = self.fader_ceiling_db - base
                channel_reasons.append("fader_ceiling_limited")

            if abs(clamped - previous) < self.deadband_db:
                safe_offsets[channel_id] = round(previous, 4)
                blocked_offsets[channel_id] = self._blocked(planned, "deadband")
                reasons[channel_id] = channel_reasons + ["deadband"]
                continue

            safe_offsets[channel_id] = round(clamped, 4)
            self._previous_offsets[channel_id] = round(clamped, 4)
            reasons[channel_id] = channel_reasons or ["ok"]

        final_confidence = min(confidences) if confidences else 0.0
        return SafetyResult(
            safe_offsets=safe_offsets,
            blocked_offsets=blocked_offsets,
            reasons=reasons,
            final_confidence=round(final_confidence, 4),
        )

    @staticmethod
    def _blocked(planned: PlannedOffset, reason: str) -> Dict[str, Any]:
        return {
            "channel_id": planned.channel_id,
            "role": planned.role,
            "requested_offset_db": planned.requested_offset_db,
            "confidence": planned.confidence,
            "reason": reason,
        }

    def reset(self) -> None:
        """Clear offset smoothing state."""
        self._previous_offsets.clear()
