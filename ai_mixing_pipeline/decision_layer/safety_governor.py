"""Decision-layer Safety Governor for offline correction candidates."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from ai_mixing_pipeline.audio_utils import measure_audio_file
from ai_mixing_pipeline.numeric_safety import finite_float

from .action_schema import CandidateActionSet


class DecisionSafetyGovernor:
    """Check action-level, render-level, and critic-level safety."""

    def __init__(self, config: dict[str, Any] | None = None):
        self.config = dict(config or {})
        self.safety = dict(self.config.get("safety", self.config) or {})

    def check_actions(self, candidate: CandidateActionSet) -> dict[str, Any]:
        reasons: list[str] = []
        warnings: list[str] = []
        max_gain = self._number(
            self.safety.get("max_gain_change_db_per_step", 1.0),
            "safety_limit:max_gain_change_db_per_step",
            reasons,
        )
        max_eq = self._number(
            self.safety.get("max_eq_change_db_per_step", 1.5),
            "safety_limit:max_eq_change_db_per_step",
            reasons,
        )
        max_master = self._number(
            self.safety.get("max_master_gain_boost_db", 0.5),
            "safety_limit:max_master_gain_boost_db",
            reasons,
        )
        max_send = self._number(
            (self.config.get("action_space", {}).get("fx", {}) or {}).get(
                "max_send_change_db", 1.0
            ),
            "safety_limit:max_send_change_db",
            reasons,
        )
        for action in candidate.actions:
            if action.action_type == "gain":
                gain = self._number(
                    getattr(action, "gain_db", None),
                    f"action:{action.action_type}:gain_db",
                    reasons,
                )
                channel_id = str(getattr(action, "channel_id", ""))
                if gain is None:
                    continue
                if channel_id == "master" and max_master is not None and gain > max_master + 1e-6:
                    reasons.append(f"master_gain_boost_exceeds_limit:{gain:.2f}>{max_master:.2f}")
                if max_gain is not None and abs(gain) > max_gain + 1e-6 and channel_id != "master":
                    reasons.append(f"gain_change_exceeds_limit:{channel_id}:{gain:.2f}")
            elif action.action_type == "eq":
                gain = self._number(
                    getattr(action, "gain_db", None),
                    f"action:{action.action_type}:gain_db",
                    reasons,
                )
                if gain is not None and max_eq is not None and abs(gain) > max_eq + 1e-6:
                    reasons.append(f"eq_change_exceeds_limit:{getattr(action, 'channel_id', '')}:{gain:.2f}")
            elif action.action_type == "compressor":
                ratio = self._number(
                    getattr(action, "ratio", None),
                    f"action:{action.action_type}:ratio",
                    reasons,
                )
                makeup = self._number(
                    getattr(action, "makeup_gain_db", None),
                    f"action:{action.action_type}:makeup_gain_db",
                    reasons,
                )
                if ratio is not None and bool(self.safety.get("forbid_excessive_compression", True)) and ratio > 3.0:
                    reasons.append(f"excessive_compression_ratio:{ratio:.2f}")
                if makeup is not None and max_master is not None and makeup > max_master:
                    reasons.append(f"compressor_makeup_exceeds_limit:{makeup:.2f}")
            elif action.action_type == "fx_send":
                send = self._number(
                    getattr(action, "send_db", None),
                    f"action:{action.action_type}:send_db",
                    reasons,
                )
                if send is not None and max_send is not None and abs(send) > max_send + 1e-6:
                    reasons.append(f"fx_send_change_exceeds_limit:{send:.2f}")
                warnings.append("fx_send is offline/report-only; no OSC/MIDI command is sent.")
        return self._result(candidate.candidate_id, reasons, warnings)

    def check_render(self, candidate: CandidateActionSet, render_path: str | Path) -> dict[str, Any]:
        reasons: list[str] = []
        warnings: list[str] = []
        penalties: list[str] = []
        metrics = measure_audio_file(render_path)
        level = metrics.get("level", {}) or {}
        stereo = metrics.get("stereo", {}) or {}
        max_true_peak = self._number(self.safety.get("max_true_peak_dbfs", -1.0), "safety_limit:max_true_peak_dbfs", reasons)
        min_headroom = self._number(self.safety.get("min_headroom_db", 1.0), "safety_limit:min_headroom_db", reasons)
        soft_margin_db = self._number(self.safety.get("soft_true_peak_margin_db", 2.0), "safety_limit:soft_true_peak_margin_db", reasons)
        soft_headroom_margin_db = self._number(self.safety.get("soft_headroom_margin_db", 2.0), "safety_limit:soft_headroom_margin_db", reasons)
        true_peak = self._number(level.get("true_peak_dbtp", level.get("peak_dbfs")), "render:true_peak_dbtp", reasons)
        headroom = self._number(level.get("headroom_db"), "render:headroom_db", reasons)
        clips = self._number(level.get("clip_count"), "render:clip_count", reasons)
        if clips is not None and bool(self.safety.get("forbid_clipping", True)) and clips > 0:
            reasons.append("clipping_detected")
        if true_peak is not None and max_true_peak is not None and true_peak > max_true_peak + 1e-6:
            reasons.append(f"true_peak_exceeds_limit:{true_peak:.2f}>{max_true_peak:.2f}")
        elif true_peak is not None and max_true_peak is not None and soft_margin_db is not None and true_peak > max_true_peak - soft_margin_db:
            penalties.append(f"true_peak_near_limit:{true_peak:.2f}>{max_true_peak - soft_margin_db:.2f}")
        if headroom is not None and min_headroom is not None and headroom < min_headroom - 1e-6:
            reasons.append(f"insufficient_headroom:{headroom:.2f}<{min_headroom:.2f}")
        elif headroom is not None and min_headroom is not None and soft_headroom_margin_db is not None and headroom < min_headroom + soft_headroom_margin_db:
            penalties.append(f"headroom_near_limit:{headroom:.2f}<{min_headroom + soft_headroom_margin_db:.2f}")
        if bool(self.safety.get("forbid_phase_collapse", True)):
            if bool(stereo.get("phase_cancellation_risk", False)):
                reasons.append("phase_cancellation_risk")
            correlation = self._number(
                stereo.get("inter_channel_correlation"),
                "render:inter_channel_correlation",
                reasons,
            )
            if correlation is not None and correlation < 0.05:
                reasons.append("phase_correlation_too_low")
        result = self._result(candidate.candidate_id, reasons, warnings)
        if penalties:
            result["safety_score"] = max(0.0, float(result["safety_score"]) - 0.05 * len(penalties))
        result["penalties"] = penalties
        result["metrics"] = metrics
        return result

    def check_critics(self, candidate: CandidateActionSet, critic_scores: dict[str, Any]) -> dict[str, Any]:
        reasons: list[str] = []
        warnings: list[str] = []
        if bool(self.safety.get("forbid_vocal_clarity_drop", True)):
            mert_delta = (critic_scores.get("mert", {}) or {}).get("delta", {})
            vocal_delta = mert_delta.get("vocal_clarity")
            if vocal_delta is not None:
                vocal_delta = self._number(vocal_delta, "critic:vocal_clarity", reasons)
            if vocal_delta is not None and vocal_delta < -0.01:
                reasons.append("vocal_clarity_drop")
        if bool(self.safety.get("forbid_bleed_boost", True)):
            identity = (critic_scores.get("panns_or_beats", {}) or {}).get("scores", {})
            bleed_score = identity.get("bleed_score")
            if bleed_score is not None:
                bleed_score = self._number(bleed_score, "critic:bleed_score", reasons)
            if bleed_score is not None and bleed_score < 0.25:
                warnings.append("low_bleed_confidence_candidate_not_boosted")
        return self._result(candidate.candidate_id, reasons, warnings)

    def combine(self, candidate_id: str, *results: dict[str, Any]) -> dict[str, Any]:
        reasons: list[str] = []
        warnings: list[str] = []
        penalties: list[str] = []
        metrics: dict[str, Any] = {}
        for result in results:
            if result.get("passed") is not True or finite_float(result.get("safety_score")) is None:
                reasons.append("invalid_or_failed_safety_result")
            reasons.extend(result.get("reasons", []))
            warnings.extend(result.get("warnings", []))
            penalties.extend(result.get("penalties", []))
            if result.get("metrics"):
                metrics = result["metrics"]
        combined = self._result(candidate_id, reasons, warnings)
        if penalties:
            combined["safety_score"] = max(0.0, float(combined["safety_score"]) - 0.05 * len(penalties))
        combined["penalties"] = penalties
        combined["metrics"] = metrics
        return combined

    @staticmethod
    def _result(candidate_id: str, reasons: list[str], warnings: list[str]) -> dict[str, Any]:
        score = max(0.0, min(1.0, 1.0 - 0.25 * len(reasons) - 0.05 * len(warnings)))
        return {
            "candidate_id": candidate_id,
            "passed": not reasons,
            "safety_score": score,
            "reasons": list(reasons),
            "warnings": list(warnings),
        }

    @staticmethod
    def _number(value: Any, label: str, reasons: list[str]) -> float | None:
        numeric = finite_float(value)
        if numeric is None:
            reasons.append(f"invalid_numeric:{label}")
        return numeric
