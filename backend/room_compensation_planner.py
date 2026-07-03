"""Dry-run room compensation planning and operator payload helpers.

This module turns room-measurement outputs into bounded, report-only EQ plans.
It intentionally does not send anything to the mixer.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import TYPE_CHECKING, Any, Iterable, Sequence

import numpy as np

if TYPE_CHECKING:
    from system_measurement import EQCorrection, MeasurementResult


TARGET_MASTER = "master"
TARGET_GROUP = "group"
TARGET_MATRIX = "matrix"
SUPPORTED_TARGETS = (TARGET_MASTER, TARGET_GROUP, TARGET_MATRIX)


@dataclass(slots=True)
class PlannedRoomFilter:
    frequency: float
    gain_db: float
    q: float
    filter_type: str
    limited_by: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["frequency"] = round(float(self.frequency), 2)
        payload["gain_db"] = round(float(self.gain_db), 2)
        payload["q"] = round(float(self.q), 2)
        payload["limited_by"] = list(self.limited_by)
        return payload


@dataclass(slots=True)
class RoomCompensationSummary:
    classification: str
    quality_indicator: str
    quality_score: float
    confidence_score: float
    measured_positions: int
    avg_rt60_sec: float
    rt60_spread_sec: float
    coherence_mean: float
    low_band_offset_db: float
    high_band_offset_db: float
    broad_band_variance_db: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "classification": self.classification,
            "quality_indicator": self.quality_indicator,
            "quality_score": round(float(self.quality_score), 4),
            "confidence_score": round(float(self.confidence_score), 4),
            "measured_positions": int(self.measured_positions),
            "avg_rt60_sec": round(float(self.avg_rt60_sec), 4),
            "rt60_spread_sec": round(float(self.rt60_spread_sec), 4),
            "coherence_mean": round(float(self.coherence_mean), 4),
            "low_band_offset_db": round(float(self.low_band_offset_db), 4),
            "high_band_offset_db": round(float(self.high_band_offset_db), 4),
            "broad_band_variance_db": round(float(self.broad_band_variance_db), 4),
        }


@dataclass(slots=True)
class RoomTargetOption:
    target: str
    suitability: str
    reason: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "target": self.target,
            "suitability": self.suitability,
            "reason": self.reason,
        }


@dataclass(slots=True)
class RoomCompensationPlan:
    dry_run_only: bool
    applied: bool
    status: str
    send_status: str
    blocked: bool
    blocked_reason: str | None
    summary: RoomCompensationSummary
    recommended_target: str
    target_options: list[RoomTargetOption]
    filter_cap: int
    max_cut_db: float
    max_boost_db: float
    filters: list[PlannedRoomFilter]
    safe_override_guidance: list[str] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)

    def message_for_user(self) -> str:
        if self.blocked:
            return (
                "Dry-run room compensation withheld: measurement confidence is too low. "
                "No console write was sent."
            )
        return (
            f"Dry-run room compensation plan ready for {self.recommended_target}: "
            f"{len(self.filters)} bounded filters, confidence "
            f"{self.summary.confidence_score:.2f}. No console write was sent."
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "dry_run_only": self.dry_run_only,
            "applied": self.applied,
            "status": self.status,
            "send_status": self.send_status,
            "blocked": self.blocked,
            "blocked_reason": self.blocked_reason,
            "summary": self.summary.to_dict(),
            "recommended_target": self.recommended_target,
            "target_options": [option.to_dict() for option in self.target_options],
            "filter_cap": int(self.filter_cap),
            "max_cut_db": round(float(self.max_cut_db), 2),
            "max_boost_db": round(float(self.max_boost_db), 2),
            "filters": [planned.to_dict() for planned in self.filters],
            "safe_override_guidance": list(self.safe_override_guidance),
            "notes": list(self.notes),
            "message_for_user": self.message_for_user(),
        }


@dataclass(slots=True)
class RoomCompensationPlannerConfig:
    min_quality_score: float = 0.35
    min_confidence_score: float = 0.45
    min_positions_for_high_confidence: int = 6
    max_filter_gap_ratio: float = 0.12
    master_filter_cap: int = 4
    group_filter_cap: int = 5
    matrix_filter_cap: int = 4
    master_max_cut_db: float = 6.0
    group_max_cut_db: float = 6.0
    matrix_max_cut_db: float = 5.0
    master_max_boost_db: float = 2.5
    group_max_boost_db: float = 3.0
    matrix_max_boost_db: float = 2.0
    master_max_q: float = 2.2
    group_max_q: float = 2.6
    matrix_max_q: float = 2.0
    min_q: float = 0.7


class RoomCompensationPlanner:
    """Plan safe, bounded room compensation without creating a write path."""

    def __init__(self, config: RoomCompensationPlannerConfig | None = None) -> None:
        self.config = config or RoomCompensationPlannerConfig()

    def plan(
        self,
        measurement: MeasurementResult,
        candidate_corrections: Sequence[EQCorrection] | None = None,
    ) -> RoomCompensationPlan:
        summary = self.summarize_measurement(measurement)
        recommended_target = self._recommended_target(summary)
        target_options = self._target_options(summary, recommended_target)
        target_caps = self._target_caps(recommended_target)
        raw_candidates = list(candidate_corrections or self._fallback_candidates(measurement))
        notes: list[str] = []

        blocked = (
            summary.quality_score < self.config.min_quality_score
            or summary.confidence_score < self.config.min_confidence_score
        )
        blocked_reason = None
        if summary.quality_score < self.config.min_quality_score:
            blocked_reason = "measurement_quality_too_low"
            notes.append("Repeat the sweep with a quieter room or stronger measurement signal.")
        elif summary.confidence_score < self.config.min_confidence_score:
            blocked_reason = "measurement_confidence_too_low"
            notes.append("Increase the number of valid positions before trusting a room-wide correction.")

        planned_filters: list[PlannedRoomFilter] = []
        if not blocked:
            planned_filters = self._bounded_filters(
                raw_candidates,
                classification=summary.classification,
                max_filters=target_caps["filter_cap"],
                max_cut_db=target_caps["max_cut_db"],
                max_boost_db=target_caps["max_boost_db"],
                max_q=target_caps["max_q"],
            )
            if not planned_filters:
                blocked = True
                blocked_reason = "no_safe_filters_generated"
                notes.append("No correction survived the room safety limits, so the plan stays report-only.")

        safe_override_guidance = [
            "Dry-run only: this payload must not write to the console.",
            (
                f"Keep overrides within {target_caps['filter_cap']} filters, "
                f"cuts up to {target_caps['max_cut_db']:.1f} dB, boosts up to "
                f"{target_caps['max_boost_db']:.1f} dB."
            ),
            f"Prefer {recommended_target} first; change target only if the measurement scope clearly matches that bus.",
        ]
        if summary.confidence_score < 0.65:
            safe_override_guidance.append(
                "Re-measure at more audience positions before accepting any positive boosts."
            )
        if summary.classification == "reverberant":
            safe_override_guidance.append(
                "Favor cuts over boosts in reverberant rooms; do not chase late reflections with EQ."
            )

        return RoomCompensationPlan(
            dry_run_only=True,
            applied=False,
            status="dry_run",
            send_status="not_sent",
            blocked=blocked,
            blocked_reason=blocked_reason,
            summary=summary,
            recommended_target=recommended_target,
            target_options=target_options,
            filter_cap=target_caps["filter_cap"],
            max_cut_db=target_caps["max_cut_db"],
            max_boost_db=target_caps["max_boost_db"],
            filters=planned_filters,
            safe_override_guidance=safe_override_guidance,
            notes=notes,
        )

    def summarize_measurement(self, measurement: MeasurementResult) -> RoomCompensationSummary:
        response = measurement.magnitude_response.get_smoothed(1.0 / 6.0)
        freqs = np.asarray(response.frequencies, dtype=float)
        magnitude_db = np.asarray(response.magnitude_db, dtype=float)
        coherence = np.asarray(response.coherence, dtype=float)
        rt60_values = np.asarray(getattr(getattr(measurement, "rt60", None), "rt60", []), dtype=float)
        positions = list(getattr(measurement, "positions", []) or [])

        if freqs.size == 0 or magnitude_db.size == 0:
            quality_score = max(0.0, float(getattr(measurement, "overall_quality", 0.0)))
            return RoomCompensationSummary(
                classification="unknown",
                quality_indicator=self._quality_indicator(quality_score),
                quality_score=quality_score,
                confidence_score=min(quality_score, 0.25),
                measured_positions=len(positions),
                avg_rt60_sec=float(np.mean(rt60_values)) if rt60_values.size else 0.0,
                rt60_spread_sec=float(np.std(rt60_values)) if rt60_values.size else 0.0,
                coherence_mean=float(np.mean(coherence)) if coherence.size else 0.0,
                low_band_offset_db=0.0,
                high_band_offset_db=0.0,
                broad_band_variance_db=0.0,
            )

        band_mask = (freqs >= 40.0) & (freqs <= 10_000.0)
        working_freqs = freqs[band_mask] if np.any(band_mask) else freqs
        working_mag = magnitude_db[band_mask] if np.any(band_mask) else magnitude_db
        reference_level = float(np.median(working_mag))
        deviations = working_mag - reference_level

        low_band_offset = self._band_average(working_freqs, deviations, 40.0, 160.0)
        high_band_offset = self._band_average(working_freqs, deviations, 2500.0, 10_000.0)
        broad_band_variance = float(np.std(deviations)) if deviations.size else 0.0
        coherence_mean = float(np.mean(coherence)) if coherence.size else float(
            getattr(measurement, "overall_quality", 0.0)
        )
        avg_rt60 = float(np.mean(rt60_values)) if rt60_values.size else 0.0
        rt60_spread = float(np.std(rt60_values)) if rt60_values.size else 0.0
        positions_factor = min(len(positions) / max(self.config.min_positions_for_high_confidence, 1), 1.0)
        consistency = 1.0 - min(rt60_spread / 1.5, 1.0)
        quality_score = float(np.clip(getattr(measurement, "overall_quality", 0.0), 0.0, 1.0))
        confidence = float(
            np.clip(
                0.4 * quality_score
                + 0.3 * np.clip(coherence_mean, 0.0, 1.0)
                + 0.2 * positions_factor
                + 0.1 * consistency,
                0.0,
                1.0,
            )
        )

        classification = "balanced"
        if avg_rt60 >= 1.8:
            classification = "reverberant"
        elif low_band_offset >= 3.0:
            classification = "boomy"
        elif high_band_offset <= -3.0:
            classification = "dull"
        elif broad_band_variance >= 5.0:
            classification = "uneven"

        return RoomCompensationSummary(
            classification=classification,
            quality_indicator=self._quality_indicator(quality_score),
            quality_score=quality_score,
            confidence_score=confidence,
            measured_positions=len(positions),
            avg_rt60_sec=avg_rt60,
            rt60_spread_sec=rt60_spread,
            coherence_mean=coherence_mean,
            low_band_offset_db=low_band_offset,
            high_band_offset_db=high_band_offset,
            broad_band_variance_db=broad_band_variance,
        )

    @staticmethod
    def _quality_indicator(score: float) -> str:
        if score >= 0.85:
            return "excellent"
        if score >= 0.65:
            return "good"
        if score >= 0.45:
            return "fair"
        return "poor"

    @staticmethod
    def _band_average(freqs: np.ndarray, values: np.ndarray, low_hz: float, high_hz: float) -> float:
        mask = (freqs >= low_hz) & (freqs <= high_hz)
        if not np.any(mask):
            return 0.0
        return float(np.mean(values[mask]))

    def _recommended_target(self, summary: RoomCompensationSummary) -> str:
        if summary.classification == "reverberant":
            return TARGET_MATRIX
        if summary.classification in {"boomy", "uneven"}:
            return TARGET_GROUP
        return TARGET_MASTER

    def _target_options(
        self,
        summary: RoomCompensationSummary,
        recommended_target: str,
    ) -> list[RoomTargetOption]:
        reasons = {
            TARGET_MASTER: "Use for broad tonal correction that should follow the whole PA.",
            TARGET_GROUP: "Use when low-mid buildup or subgroup voicing dominates the room problem.",
            TARGET_MATRIX: "Use for zone-specific or highly reverberant coverage work.",
        }
        options: list[RoomTargetOption] = []
        for target in SUPPORTED_TARGETS:
            suitability = "secondary"
            if target == recommended_target:
                suitability = "primary"
            elif summary.classification == "reverberant" and target == TARGET_MASTER:
                suitability = "avoid"
            elif summary.classification == "balanced" and target == TARGET_MATRIX:
                suitability = "avoid"
            options.append(RoomTargetOption(target=target, suitability=suitability, reason=reasons[target]))
        return options

    def _target_caps(self, target: str) -> dict[str, float | int]:
        if target == TARGET_GROUP:
            return {
                "filter_cap": self.config.group_filter_cap,
                "max_cut_db": self.config.group_max_cut_db,
                "max_boost_db": self.config.group_max_boost_db,
                "max_q": self.config.group_max_q,
            }
        if target == TARGET_MATRIX:
            return {
                "filter_cap": self.config.matrix_filter_cap,
                "max_cut_db": self.config.matrix_max_cut_db,
                "max_boost_db": self.config.matrix_max_boost_db,
                "max_q": self.config.matrix_max_q,
            }
        return {
            "filter_cap": self.config.master_filter_cap,
            "max_cut_db": self.config.master_max_cut_db,
            "max_boost_db": self.config.master_max_boost_db,
            "max_q": self.config.master_max_q,
        }

    def _bounded_filters(
        self,
        candidate_corrections: Sequence[Any],
        *,
        classification: str,
        max_filters: int,
        max_cut_db: float,
        max_boost_db: float,
        max_q: float,
    ) -> list[PlannedRoomFilter]:
        bounded: list[PlannedRoomFilter] = []
        for correction in sorted(candidate_corrections, key=lambda item: abs(float(item.gain_db)), reverse=True):
            frequency = float(correction.frequency)
            gain_db = float(correction.gain_db)
            q = float(getattr(correction, "q", 1.8))
            filter_type = str(getattr(correction, "type", getattr(correction, "filter_type", "peak")))
            limited_by: list[str] = []

            if any(abs(existing.frequency - frequency) <= max(frequency * self.config.max_filter_gap_ratio, 40.0) for existing in bounded):
                continue

            if gain_db < -max_cut_db:
                gain_db = -max_cut_db
                limited_by.append("max_cut_clamped")
            if gain_db > max_boost_db:
                gain_db = max_boost_db
                limited_by.append("max_boost_clamped")
            if classification == "reverberant" and gain_db > 0.0:
                gain_db = min(gain_db, 1.0)
                limited_by.append("reverberant_room_boost_reduced")
            q = max(self.config.min_q, min(max_q, q))
            if q == self.config.min_q or q == max_q:
                limited_by.append("q_clamped")
            if abs(gain_db) < 0.75:
                continue

            bounded.append(
                PlannedRoomFilter(
                    frequency=frequency,
                    gain_db=gain_db,
                    q=q,
                    filter_type=filter_type,
                    limited_by=tuple(limited_by),
                )
            )
            if len(bounded) >= max_filters:
                break

        return bounded

    def _fallback_candidates(self, measurement: MeasurementResult) -> Iterable[Any]:
        response = measurement.magnitude_response.get_smoothed(1.0 / 6.0)
        freqs = np.asarray(response.frequencies, dtype=float)
        magnitude_db = np.asarray(response.magnitude_db, dtype=float)
        if freqs.size == 0 or magnitude_db.size == 0:
            return []

        anchor_freqs = (63.0, 125.0, 250.0, 500.0, 1000.0, 2000.0, 4000.0, 8000.0)
        mask = (freqs >= 40.0) & (freqs <= 10_000.0)
        reference = float(np.median(magnitude_db[mask])) if np.any(mask) else float(np.median(magnitude_db))
        candidates: list[_FallbackCorrection] = []
        for anchor in anchor_freqs:
            idx = int(np.argmin(np.abs(freqs - anchor)))
            deviation = float(magnitude_db[idx] - reference)
            if abs(deviation) < 2.5:
                continue
            candidates.append(
                _FallbackCorrection(
                    frequency=float(freqs[idx]),
                    gain_db=float(np.clip(-deviation, -8.0, 6.0)),
                    q=1.4 if anchor <= 250.0 else 1.8,
                    type="peak",
                )
            )
        return candidates


@dataclass(slots=True)
class _FallbackCorrection:
    frequency: float
    gain_db: float
    q: float
    type: str = "peak"


def build_room_compensation_operator_payload(
    plan: RoomCompensationPlan,
    *,
    event_type: str,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    payload = {
        "type": event_type,
        "accepted": not plan.blocked,
        "blocked": plan.blocked,
        "blocked_reason": plan.blocked_reason,
        "dry_run_only": plan.dry_run_only,
        "applied": plan.applied,
        "non_applied_status": plan.status,
        "send_status": plan.send_status,
        "message_for_user": plan.message_for_user(),
        "recommended_target": plan.recommended_target,
        "target_options": [option.to_dict() for option in plan.target_options],
        "filter_cap": plan.filter_cap,
        "max_cut_db": round(float(plan.max_cut_db), 2),
        "max_boost_db": round(float(plan.max_boost_db), 2),
        "planned_filters": [planned.to_dict() for planned in plan.filters],
        "room_quality_indicator": plan.summary.quality_indicator,
        "room_quality_score": round(float(plan.summary.quality_score), 4),
        "confidence_score": round(float(plan.summary.confidence_score), 4),
        "room_classification": plan.summary.classification,
        "measured_positions": int(plan.summary.measured_positions),
        "avg_rt60_sec": round(float(plan.summary.avg_rt60_sec), 4),
        "safe_override_guidance": list(plan.safe_override_guidance),
        "notes": list(plan.notes),
    }
    if extra:
        payload.update(extra)
    return payload
