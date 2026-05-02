"""Plan musical relative fader offsets before safety limiting."""

from __future__ import annotations

from typing import Dict, Iterable, List

from .models import (
    ActivityState,
    ChannelRoleInfo,
    DensityState,
    MaskingSource,
    PlannedOffset,
    PriorityState,
    SectionState,
)


def _clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, float(value)))


class LevelAutomationPlanner:
    """Calculate requested fader offsets from section, density, masking, and role priority."""

    def plan(
        self,
        roles: Dict[int, ChannelRoleInfo],
        activities: Dict[int, ActivityState],
        density: DensityState,
        section: SectionState,
        masking_sources: Iterable[MaskingSource],
        priority: PriorityState,
    ) -> Dict[int, PlannedOffset]:
        masking_by_channel = {source.channel_id: source for source in masking_sources}
        planned: Dict[int, PlannedOffset] = {}

        for channel_id, role_info in roles.items():
            activity = activities.get(channel_id)
            active = bool(activity and activity.active)
            components = {
                "base_mix_position": self._base_mix_position(role_info.role),
                "section_offset": self._section_offset(
                    role_info.role,
                    section.section,
                    priority,
                    channel_id,
                ),
                "density_offset": self._density_offset(role_info.role, density, priority),
                "masking_offset": self._masking_offset(channel_id, masking_by_channel),
                "role_priority_offset": self._priority_offset(role_info.role, priority, channel_id),
                "safety_penalty": self._safety_penalty(role_info.role, activity),
            }

            if not active and channel_id != priority.primary_channel_id:
                components["section_offset"] = min(0.0, components["section_offset"])
                components["density_offset"] = min(0.0, components["density_offset"])
                components["role_priority_offset"] = min(0.0, components["role_priority_offset"])
                components["masking_offset"] = 0.0

            requested = (
                components["base_mix_position"]
                + components["section_offset"]
                + components["density_offset"]
                + components["masking_offset"]
                + components["role_priority_offset"]
                - components["safety_penalty"]
            )
            confidence = min(
                role_info.confidence,
                density.confidence,
                section.confidence,
                priority.confidence,
                activity.activity_confidence if activity else 0.3,
            )
            reason = self._reason(
                role_info.role,
                section.section,
                density.density_label,
                priority,
                channel_id,
            )
            planned[channel_id] = PlannedOffset(
                channel_id=channel_id,
                channel_name=role_info.channel_name,
                role=role_info.role,
                requested_offset_db=round(requested, 3),
                confidence=round(_clamp(confidence, 0.0, 1.0), 4),
                components={key: round(value, 3) for key, value in components.items()},
                reason=reason,
            )

        return planned

    @staticmethod
    def _base_mix_position(role: str) -> float:
        return {
            "lead_vocal": 0.0,
            "solo_instrument": 0.0,
            "lead_guitar": -0.2,
            "backing_vocal": -0.6,
            "rhythm_guitar": -0.5,
            "acoustic_guitar": -0.4,
            "keys": -0.5,
            "pad": -0.8,
            "overhead": -0.8,
            "fx": -1.0,
            "unknown": 0.0,
        }.get(role, 0.0)

    @staticmethod
    def _section_offset(role: str, section: str, priority: PriorityState, channel_id: int) -> float:
        if section == "verse":
            return {
                "lead_vocal": 0.6,
                "rhythm_guitar": -0.4,
                "keys": -0.4,
                "pad": -0.6,
                "backing_vocal": -1.0,
            }.get(role, 0.0)
        if section == "chorus":
            return {
                "lead_vocal": 1.2,
                "backing_vocal": 0.8,
                "rhythm_guitar": 0.2,
                "lead_guitar": 0.2,
                "keys": 0.2,
                "kick": 0.3,
                "snare": 0.3,
                "bass": 0.2,
            }.get(role, 0.0)
        if section == "solo":
            if channel_id == priority.primary_channel_id:
                return 2.4
            return {
                "lead_vocal": -1.0,
                "backing_vocal": -1.0,
                "rhythm_guitar": -0.6,
                "keys": -0.5,
                "pad": -0.8,
            }.get(role, 0.0)
        if section in {"intro", "breakdown"}:
            return {"pad": -0.3, "fx": -0.3, "overhead": -0.3}.get(role, 0.0)
        return 0.0

    @staticmethod
    def _density_offset(role: str, density: DensityState, priority: PriorityState) -> float:
        if density.density_index < 0.45:
            return 0.0
        if role in priority.protected_roles:
            return 0.25 + density.density_index * 0.35
        if role in priority.roles_to_duck:
            return -density.density_index * 1.4
        if role in {"overhead", "pad", "fx"} and density.density_index >= 0.70:
            return -density.density_index
        return 0.0

    @staticmethod
    def _masking_offset(channel_id: int, masking_by_channel: Dict[int, MaskingSource]) -> float:
        source = masking_by_channel.get(channel_id)
        if not source or source.suggested_action == "ignore":
            return 0.0
        return float(source.suggested_gain_reduction_db)

    @staticmethod
    def _priority_offset(role: str, priority: PriorityState, channel_id: int) -> float:
        if channel_id == priority.primary_channel_id:
            return 0.8
        if role in priority.protected_roles:
            return 0.2
        if role in priority.roles_to_duck:
            return -0.4
        return 0.0

    @staticmethod
    def _safety_penalty(role: str, activity: ActivityState | None) -> float:
        if role == "unknown":
            return 0.4
        if activity and activity.activity_confidence < 0.45:
            return 0.4
        return 0.0

    @staticmethod
    def _reason(
        role: str,
        section: str,
        density_label: str,
        priority: PriorityState,
        channel_id: int,
    ) -> str:
        if channel_id == priority.primary_channel_id:
            return f"{role} is primary source in {section}; density={density_label}"
        if role in priority.roles_to_duck:
            return f"{role} makes room for {priority.primary_source_role}; density={density_label}"
        return f"{role} adjusted for {section}; density={density_label}"
