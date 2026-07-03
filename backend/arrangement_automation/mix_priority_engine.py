"""Musical priority engine for foreground/source protection decisions."""

from __future__ import annotations

from typing import Dict, Optional

from .models import ActivityState, ChannelRoleInfo, PriorityState, SectionState


class MixPriorityEngine:
    """Determine the primary musical source and roles that should make room."""

    def analyze(
        self,
        roles: Dict[int, ChannelRoleInfo],
        activities: Dict[int, ActivityState],
        section: Optional[SectionState] = None,
    ) -> PriorityState:
        primary_channel = self._first_active_role(roles, activities, "lead_vocal")
        if primary_channel is not None:
            return PriorityState(
                primary_source_role="lead_vocal",
                primary_channel_id=primary_channel,
                protected_roles=["lead_vocal", "kick", "bass", "snare"],
                secondary_roles=["backing_vocal", "lead_guitar", "acoustic_guitar"],
                roles_to_duck=["rhythm_guitar", "keys", "pad", "overhead"],
                confidence=0.9,
                reason="lead vocal is active",
            )

        primary_channel = self._first_active_role(roles, activities, "solo_instrument")
        primary_role = "solo_instrument"
        if primary_channel is None:
            primary_channel = self._first_active_role(roles, activities, "lead_guitar")
            primary_role = "lead_guitar"
        if primary_channel is None:
            primary_channel = self._first_active_role(roles, activities, "keys")
            primary_role = "keys"

        if primary_channel is not None:
            return PriorityState(
                primary_source_role=primary_role,
                primary_channel_id=primary_channel,
                protected_roles=[primary_role, "kick", "bass", "snare"],
                secondary_roles=["rhythm_guitar", "keys", "pad"],
                roles_to_duck=["rhythm_guitar", "keys", "pad", "backing_vocal"],
                confidence=0.72,
                reason=f"{primary_role} is active while lead vocal is inactive",
            )

        return PriorityState(
            primary_source_role="unknown",
            primary_channel_id=None,
            protected_roles=["kick", "bass"],
            secondary_roles=[],
            roles_to_duck=[],
            confidence=0.35,
            reason="no clear foreground source",
        )

    @staticmethod
    def _first_active_role(
        roles: Dict[int, ChannelRoleInfo],
        activities: Dict[int, ActivityState],
        role: str,
    ) -> Optional[int]:
        for channel_id, role_info in sorted(roles.items(), key=lambda item: item[1].priority):
            activity = activities.get(channel_id)
            if role_info.role == role and activity and activity.active:
                return channel_id
        return None
