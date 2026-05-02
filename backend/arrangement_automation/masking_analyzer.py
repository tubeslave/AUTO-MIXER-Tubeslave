"""Masking analysis for lead vocal and solo source protection."""

from __future__ import annotations

from typing import Dict, List, Optional

from .models import ActivityState, ChannelRoleInfo, MaskingSource


MASKING_CANDIDATE_ROLES = {"rhythm_guitar", "keys", "pad", "overhead", "snare"}


def _clamp01(value: float) -> float:
    return max(0.0, min(1.0, float(value)))


def _presence_energy(bands: Dict[str, float]) -> float:
    return max(
        float(bands.get("umf", -100.0)),
        float(bands.get("mid", -100.0)),
        float(bands.get("high_mid", -100.0)),
    )


class MaskingAnalyzer:
    """Find channels likely to mask the current foreground source."""

    def __init__(self, min_masking_score: float = 0.35):
        self.min_masking_score = float(min_masking_score)

    def analyze(
        self,
        roles: Dict[int, ChannelRoleInfo],
        activities: Dict[int, ActivityState],
        band_energy: Dict[int, Dict[str, float]],
        primary_channel_id: Optional[int] = None,
    ) -> List[MaskingSource]:
        target = primary_channel_id or self._find_target(roles, activities)
        if target is None or not activities.get(target) or not activities[target].active:
            return []

        target_presence = _presence_energy(band_energy.get(target, {}))
        sources: List[MaskingSource] = []
        for channel_id, role_info in roles.items():
            if channel_id == target:
                continue
            activity = activities.get(channel_id)
            if not activity or not activity.active:
                continue
            if role_info.role not in MASKING_CANDIDATE_ROLES:
                continue

            bands = band_energy.get(channel_id, {})
            presence = _presence_energy(bands)
            low_mid = max(float(bands.get("low_mid", -100.0)), float(bands.get("lmf", -100.0)))
            harshness = max(float(bands.get("high_mid", -100.0)), float(bands.get("hf", -100.0)))
            low_end = max(float(bands.get("sub", -100.0)), float(bands.get("bass", -100.0)))

            presence_score = _clamp01((presence - max(-60.0, target_presence - 10.0)) / 24.0)
            congestion_score = _clamp01((low_mid + 60.0) / 42.0)
            harshness_score = _clamp01((harshness + 58.0) / 40.0)
            low_conflict = (
                _clamp01((low_end + 55.0) / 40.0)
                if role_info.role in {"keys", "pad"}
                else 0.0
            )
            score = _clamp01(
                0.55 * presence_score
                + 0.20 * congestion_score
                + 0.20 * harshness_score
                + 0.05 * low_conflict
            )

            if score < self.min_masking_score:
                action = "ignore"
                reduction = 0.0
            elif score < 0.72:
                action = "level_duck"
                reduction = -round(0.5 + score * 1.5, 2)
            else:
                action = "level_duck"
                reduction = -round(1.0 + score * 2.0, 2)

            sources.append(
                MaskingSource(
                    channel_id=channel_id,
                    role=role_info.role,
                    masking_score=round(score, 4),
                    suggested_action=action,
                    suggested_gain_reduction_db=reduction,
                )
            )

        return sorted(sources, key=lambda item: item.masking_score, reverse=True)

    @staticmethod
    def _find_target(
        roles: Dict[int, ChannelRoleInfo],
        activities: Dict[int, ActivityState],
    ) -> Optional[int]:
        for role in ("lead_vocal", "solo_instrument", "lead_guitar"):
            for channel_id, role_info in roles.items():
                activity = activities.get(channel_id)
                if role_info.role == role and activity and activity.active:
                    return channel_id
        return None
