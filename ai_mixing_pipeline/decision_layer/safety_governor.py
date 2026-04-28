"""Decision-layer Safety Governor for offline correction candidates."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from ai_mixing_pipeline.audio_utils import measure_audio_file

from .action_schema import CandidateActionSet


class DecisionSafetyGovernor:
    """Check action-level, render-level, and critic-level safety."""

    def __init__(self, config: dict[str, Any] | None = None):
        self.config = dict(config or {})
        self.safety = dict(self.config.get("safety", self.config) or {})

    def check_actions(self, candidate: CandidateActionSet) -> dict[str, Any]:
        reasons: list[str] = []
        warnings: list[str] = []
        penalties: list[dict[str, Any]] = []
        max_gain = float(self.safety.get("max_gain_change_db_per_step", 1.0))
        max_eq = float(self.safety.get("max_eq_change_db_per_step", 1.5))
        max_master = float(self.safety.get("max_master_gain_boost_db", 0.5))
        max_send = float((self.config.get("action_space", {}).get("fx", {}) or {}).get("max_send_change_db", 1.0))
        for action in candidate.actions:
            if action.action_type == "gain":
                gain = float(getattr(action, "gain_db", 0.0))
                channel_id = str(getattr(action, "channel_id", ""))
                if channel_id == "master" and gain > max_master + 1e-6:
                    reasons.append(f"master_gain_boost_exceeds_limit:{gain:.2f}>{max_master:.2f}")
                if abs(gain) > max_gain + 1e-6 and channel_id != "master":
                    reasons.append(f"gain_change_exceeds_limit:{channel_id}:{gain:.2f}")
                limit = max_master if channel_id == "master" and gain > 0.0 else max_gain
                penalties.append(self._bounded_penalty("gain_change_margin", abs(gain), max(1e-6, limit), 0.12))
            elif action.action_type == "eq":
                gain = float(getattr(action, "gain_db", 0.0))
                if abs(gain) > max_eq + 1e-6:
                    reasons.append(f"eq_change_exceeds_limit:{getattr(action, 'channel_id', '')}:{gain:.2f}")
                penalties.append(self._bounded_penalty("eq_change_margin", abs(gain), max(1e-6, max_eq), 0.10))
            elif action.action_type == "compressor":
                ratio = float(getattr(action, "ratio", 1.0))
                makeup = float(getattr(action, "makeup_gain_db", 0.0))
                if bool(self.safety.get("forbid_excessive_compression", True)) and ratio > 3.0:
                    reasons.append(f"excessive_compression_ratio:{ratio:.2f}")
                if makeup > max_master:
                    reasons.append(f"compressor_makeup_exceeds_limit:{makeup:.2f}")
                penalties.append(self._bounded_penalty("compressor_ratio_margin", max(0.0, ratio - 1.0), 2.0, 0.10))
                penalties.append(self._bounded_penalty("compressor_makeup_margin", max(0.0, makeup), max(1e-6, max_master), 0.08))
            elif action.action_type == "fx_send":
                send = float(getattr(action, "send_db", 0.0))
                if abs(send) > max_send + 1e-6:
                    reasons.append(f"fx_send_change_exceeds_limit:{send:.2f}")
                warnings.append("fx_send is offline/report-only; no OSC/MIDI command is sent.")
                penalties.append(self._bounded_penalty("fx_send_margin", abs(send), max(1e-6, max_send), 0.06))
        return self._result(candidate.candidate_id, reasons, warnings, penalties=penalties)

    def check_render(self, candidate: CandidateActionSet, render_path: str | Path) -> dict[str, Any]:
        reasons: list[str] = []
        warnings: list[str] = []
        penalties: list[dict[str, Any]] = []
        metrics = measure_audio_file(render_path)
        level = metrics.get("level", {}) or {}
        stereo = metrics.get("stereo", {}) or {}
        dynamics = metrics.get("dynamics", {}) or {}
        max_true_peak = float(self.safety.get("max_true_peak_dbfs", -1.0))
        min_headroom = float(self.safety.get("min_headroom_db", 1.0))
        true_peak = float(level.get("true_peak_dbtp", level.get("peak_dbfs", -120.0)) or -120.0)
        headroom = float(level.get("headroom_db", 0.0) or 0.0)
        clips = int(level.get("clip_count", 0) or 0)
        lufs = float(level.get("integrated_lufs", -120.0) or -120.0)
        target_lufs = self.safety.get("target_lufs", self.config.get("target_lufs"))
        baseline_metrics = (candidate.metadata.get("baseline_metrics", {}) if hasattr(candidate, "metadata") else {}) or {}
        baseline_lufs = baseline_metrics.get("integrated_lufs")
        if bool(self.safety.get("forbid_clipping", True)) and clips > 0:
            reasons.append("clipping_detected")
            penalties.append({"name": "clip_count", "penalty": min(0.35, clips / 1000.0), "value": clips})
        if true_peak > max_true_peak + 1e-6:
            reasons.append(f"true_peak_exceeds_limit:{true_peak:.2f}>{max_true_peak:.2f}")
        peak_margin = max_true_peak - true_peak
        penalties.append(
            {
                "name": "true_peak_margin",
                "penalty": max(0.0, min(0.20, (1.5 - peak_margin) / 1.5 * 0.20)),
                "value": round(true_peak, 3),
                "limit": max_true_peak,
                "margin_db": round(peak_margin, 3),
            }
        )
        if headroom < min_headroom - 1e-6:
            reasons.append(f"insufficient_headroom:{headroom:.2f}<{min_headroom:.2f}")
        penalties.append(
            {
                "name": "headroom_margin",
                "penalty": max(0.0, min(0.20, (min_headroom + 1.0 - headroom) / max(1.0, min_headroom + 1.0) * 0.20)),
                "value": round(headroom, 3),
                "limit": min_headroom,
            }
        )
        if isinstance(baseline_lufs, (int, float)):
            lufs_jump = abs(lufs - float(baseline_lufs))
        elif isinstance(target_lufs, (int, float)):
            lufs_jump = abs(lufs - float(target_lufs))
        else:
            lufs_jump = 0.0
        if lufs_jump:
            max_lufs_jump = float(self.safety.get("max_lufs_jump_db", 1.5))
            if lufs_jump > max_lufs_jump:
                reasons.append(f"lufs_jump_exceeds_limit:{lufs_jump:.2f}>{max_lufs_jump:.2f}")
            penalties.append(self._bounded_penalty("lufs_jump", lufs_jump, max(1e-6, max_lufs_jump), 0.12))
        if bool(self.safety.get("forbid_phase_collapse", True)):
            if bool(stereo.get("phase_cancellation_risk", False)):
                reasons.append("phase_cancellation_risk")
            if float(stereo.get("inter_channel_correlation", 1.0) or 1.0) < 0.05:
                reasons.append("phase_correlation_too_low")
        correlation = float(stereo.get("inter_channel_correlation", 1.0) or 1.0)
        if correlation < 0.30:
            penalties.append({"name": "phase_correlation_margin", "penalty": min(0.15, (0.30 - correlation) * 0.5), "value": round(correlation, 3)})
        pumping = dynamics.get("compression_pumping_proxy")
        if isinstance(pumping, (int, float)):
            penalties.append(self._bounded_penalty("compression_pumping", float(pumping), 4.0, 0.08))
        result = self._result(candidate.candidate_id, reasons, warnings, penalties=penalties)
        result["metrics"] = metrics
        return result

    def check_critics(self, candidate: CandidateActionSet, critic_scores: dict[str, Any]) -> dict[str, Any]:
        reasons: list[str] = []
        warnings: list[str] = []
        penalties: list[dict[str, Any]] = []
        if bool(self.safety.get("forbid_vocal_clarity_drop", True)):
            mert_delta = (critic_scores.get("mert", {}) or {}).get("delta", {})
            vocal_delta = mert_delta.get("vocal_clarity")
            if isinstance(vocal_delta, (int, float)) and float(vocal_delta) < -0.01:
                reasons.append("vocal_clarity_drop")
                penalties.append({"name": "vocal_clarity_drop", "penalty": min(0.20, abs(float(vocal_delta)) * 2.0), "value": float(vocal_delta)})
        if bool(self.safety.get("forbid_bleed_boost", True)):
            identity = (critic_scores.get("panns_or_beats", {}) or {}).get("scores", {})
            bleed_score = identity.get("bleed_score")
            if isinstance(bleed_score, (int, float)) and float(bleed_score) < 0.25:
                warnings.append("low_bleed_confidence_candidate_not_boosted")
                penalties.append({"name": "bleed_confidence", "penalty": 0.05, "value": float(bleed_score)})
        return self._result(candidate.candidate_id, reasons, warnings, penalties=penalties)

    def combine(self, candidate_id: str, *results: dict[str, Any]) -> dict[str, Any]:
        reasons: list[str] = []
        warnings: list[str] = []
        penalties: list[dict[str, Any]] = []
        metrics: dict[str, Any] = {}
        for result in results:
            reasons.extend(result.get("reasons", []))
            warnings.extend(result.get("warnings", []))
            penalties.extend(result.get("penalties", []))
            if result.get("metrics"):
                metrics = result["metrics"]
        combined = self._result(candidate_id, reasons, warnings, penalties=penalties)
        combined["metrics"] = metrics
        return combined

    @staticmethod
    def _bounded_penalty(name: str, value: float, limit: float, max_penalty: float) -> dict[str, Any]:
        ratio = max(0.0, min(1.5, float(value) / max(1e-9, float(limit))))
        return {
            "name": name,
            "penalty": float(max(0.0, min(max_penalty, ratio * max_penalty))),
            "value": round(float(value), 4),
            "limit": round(float(limit), 4),
        }

    @staticmethod
    def _result(
        candidate_id: str,
        reasons: list[str],
        warnings: list[str],
        *,
        penalties: list[dict[str, Any]] | None = None,
    ) -> dict[str, Any]:
        penalties = list(penalties or [])
        penalty_value = sum(float(item.get("penalty", 0.0) or 0.0) for item in penalties)
        score = max(0.0, min(1.0, 1.0 - penalty_value - 0.25 * len(reasons) - 0.03 * len(warnings)))
        return {
            "candidate_id": candidate_id,
            "passed": not reasons,
            "safety_score": score,
            "reasons": list(reasons),
            "warnings": list(warnings),
            "penalties": penalties,
            "penalty_total": float(penalty_value),
        }
