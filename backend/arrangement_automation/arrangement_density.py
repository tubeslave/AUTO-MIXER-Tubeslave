"""Arrangement Density Index calculation."""

from __future__ import annotations

from typing import Dict, Optional

from .models import ActivityState, ChannelRoleInfo, DensityState


def _clamp01(value: float) -> float:
    return max(0.0, min(1.0, float(value)))


def _db_to_density(value_db: float, low_db: float = -60.0, high_db: float = -12.0) -> float:
    return _clamp01((float(value_db) - low_db) / (high_db - low_db))


class ArrangementDensityAnalyzer:
    """Calculate ADI from activity, broadband level, bands, transients, and masking."""

    def __init__(self, max_channels: int = 40):
        self.max_channels = max(1, int(max_channels))

    def analyze(
        self,
        activities: Dict[int, ActivityState],
        roles: Optional[Dict[int, ChannelRoleInfo]] = None,
        band_energy: Optional[Dict[int, Dict[str, float]]] = None,
        transient_density: Optional[float] = None,
        masking_pressure_on_lead: Optional[float] = None,
    ) -> DensityState:
        active = {ch: st for ch, st in activities.items() if st.active}
        active_count = len(active)
        denominator = max(len(activities), self.max_channels)
        active_channel_ratio = _clamp01(active_count / denominator)

        if active:
            avg_rms = sum(st.rms_db for st in active.values()) / active_count
            broadband = _db_to_density(avg_rms)
        else:
            broadband = 0.0

        bands = band_energy or {}
        midrange = self._band_density(
            active,
            bands,
            ("low_mid", "mid", "high_mid", "lmf", "umf"),
        )
        low_end = self._band_density(active, bands, ("sub", "bass", "lf"))
        transient = (
            self._estimate_transient_density(active, roles)
            if transient_density is None
            else transient_density
        )
        masking = (
            self._estimate_masking_pressure(active, roles, bands)
            if masking_pressure_on_lead is None
            else masking_pressure_on_lead
        )

        components = {
            "active_channel_ratio": active_channel_ratio,
            "broadband_loudness_density": broadband,
            "midrange_congestion": midrange,
            "transient_density": _clamp01(transient),
            "low_end_occupancy": low_end,
            "masking_pressure_on_lead": _clamp01(masking),
        }
        index = (
            0.25 * components["active_channel_ratio"]
            + 0.20 * components["broadband_loudness_density"]
            + 0.20 * components["midrange_congestion"]
            + 0.15 * components["transient_density"]
            + 0.10 * components["low_end_occupancy"]
            + 0.10 * components["masking_pressure_on_lead"]
        )
        confidence = 0.35 + 0.45 * min(1.0, len(activities) / max(1, self.max_channels))
        if band_energy:
            confidence += 0.15
        if roles:
            confidence += 0.05

        return DensityState(
            density_index=round(_clamp01(index), 4),
            density_label=self._label(index),
            active_channel_count=active_count,
            midrange_congestion=round(midrange, 4),
            low_end_occupancy=round(low_end, 4),
            masking_pressure_on_lead=round(_clamp01(masking), 4),
            confidence=round(_clamp01(confidence), 4),
            components={key: round(value, 4) for key, value in components.items()},
        )

    @staticmethod
    def _band_density(
        active: Dict[int, ActivityState],
        band_energy: Dict[int, Dict[str, float]],
        names: tuple[str, ...],
    ) -> float:
        values = []
        for ch in active:
            bands = band_energy.get(ch, {})
            channel_values = [float(bands[name]) for name in names if name in bands]
            if channel_values:
                values.append(_db_to_density(max(channel_values), -65.0, -15.0))
        if not values:
            return 0.0
        return _clamp01(sum(values) / len(values))

    @staticmethod
    def _estimate_transient_density(
        active: Dict[int, ActivityState],
        roles: Optional[Dict[int, ChannelRoleInfo]],
    ) -> float:
        if not active or not roles:
            return 0.0
        transient_roles = {"kick", "snare", "tom", "overhead"}
        transient_count = sum(
            1
            for ch in active
            if roles.get(ch) and roles[ch].role in transient_roles
        )
        return _clamp01(transient_count / max(1, len(active)))

    @staticmethod
    def _estimate_masking_pressure(
        active: Dict[int, ActivityState],
        roles: Optional[Dict[int, ChannelRoleInfo]],
        band_energy: Dict[int, Dict[str, float]],
    ) -> float:
        if not roles or not band_energy:
            return 0.0
        lead_active = any(
            st.active
            and roles.get(ch)
            and roles[ch].role in {"lead_vocal", "solo_instrument", "lead_guitar"}
            for ch, st in active.items()
        )
        if not lead_active:
            return 0.0
        mask_roles = {"rhythm_guitar", "keys", "pad", "overhead", "snare"}
        scores = []
        for ch in active:
            role = roles.get(ch).role if roles.get(ch) else "unknown"
            if role not in mask_roles:
                continue
            bands = band_energy.get(ch, {})
            presence = max(
                float(bands.get("umf", -100.0)),
                float(bands.get("mid", -100.0)),
                float(bands.get("high_mid", -100.0)),
            )
            scores.append(_db_to_density(presence, -60.0, -18.0))
        return _clamp01(sum(scores) / len(scores)) if scores else 0.0

    @staticmethod
    def _label(index: float) -> str:
        if index < 0.25:
            return "sparse"
        if index < 0.45:
            return "light"
        if index < 0.70:
            return "medium"
        if index < 0.90:
            return "dense"
        return "very_dense"
