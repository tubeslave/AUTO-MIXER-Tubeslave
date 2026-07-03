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
            elif action.action_type == "eq":
                gain = float(getattr(action, "gain_db", 0.0))
                if abs(gain) > max_eq + 1e-6:
                    reasons.append(f"eq_change_exceeds_limit:{getattr(action, 'channel_id', '')}:{gain:.2f}")
            elif action.action_type == "compressor":
                ratio = float(getattr(action, "ratio", 1.0))
                makeup = float(getattr(action, "makeup_gain_db", 0.0))
                if bool(self.safety.get("forbid_excessive_compression", True)) and ratio > 3.0:
                    reasons.append(f"excessive_compression_ratio:{ratio:.2f}")
                if makeup > max_master:
                    reasons.append(f"compressor_makeup_exceeds_limit:{makeup:.2f}")
            elif action.action_type == "fx_send":
                send = float(getattr(action, "send_db", 0.0))
                if abs(send) > max_send + 1e-6:
                    reasons.append(f"fx_send_change_exceeds_limit:{send:.2f}")
                warnings.append("fx_send is offline/report-only; no OSC/MIDI command is sent.")
        return self._result(candidate.candidate_id, reasons, warnings)

    def check_render(self, candidate: CandidateActionSet, render_path: str | Path) -> dict[str, Any]:
        reasons: list[str] = []
        warnings: list[str] = []
        metrics = measure_audio_file(render_path)
        level = metrics.get("level", {}) or {}
        stereo = metrics.get("stereo", {}) or {}
        max_true_peak = float(self.safety.get("max_true_peak_dbfs", -1.0))
        min_headroom = float(self.safety.get("min_headroom_db", 1.0))
        true_peak = float(level.get("true_peak_dbtp", level.get("peak_dbfs", -120.0)) or -120.0)
        headroom = float(level.get("headroom_db", 0.0) or 0.0)
        clips = int(level.get("clip_count", 0) or 0)
        if bool(self.safety.get("forbid_clipping", True)) and clips > 0:
            reasons.append("clipping_detected")
        if true_peak > max_true_peak + 1e-6:
            reasons.append(f"true_peak_exceeds_limit:{true_peak:.2f}>{max_true_peak:.2f}")
        if headroom < min_headroom - 1e-6:
            reasons.append(f"insufficient_headroom:{headroom:.2f}<{min_headroom:.2f}")
        if bool(self.safety.get("forbid_phase_collapse", True)):
            if bool(stereo.get("phase_cancellation_risk", False)):
                reasons.append("phase_cancellation_risk")
            if float(stereo.get("inter_channel_correlation", 1.0) or 1.0) < 0.05:
                reasons.append("phase_correlation_too_low")
        result = self._result(candidate.candidate_id, reasons, warnings)
        result["metrics"] = metrics
        return result

    def check_critics(self, candidate: CandidateActionSet, critic_scores: dict[str, Any]) -> dict[str, Any]:
        reasons: list[str] = []
        warnings: list[str] = []
        if bool(self.safety.get("forbid_vocal_clarity_drop", True)):
            mert_delta = (critic_scores.get("mert", {}) or {}).get("delta", {})
            vocal_delta = mert_delta.get("vocal_clarity")
            if isinstance(vocal_delta, (int, float)) and float(vocal_delta) < -0.01:
                reasons.append("vocal_clarity_drop")
        if bool(self.safety.get("forbid_bleed_boost", True)):
            identity = (critic_scores.get("panns_or_beats", {}) or {}).get("scores", {})
            bleed_score = identity.get("bleed_score")
            if isinstance(bleed_score, (int, float)) and float(bleed_score) < 0.25:
                warnings.append("low_bleed_confidence_candidate_not_boosted")
        return self._result(candidate.candidate_id, reasons, warnings)

    def combine(self, candidate_id: str, *results: dict[str, Any]) -> dict[str, Any]:
        reasons: list[str] = []
        warnings: list[str] = []
        metrics: dict[str, Any] = {}
        for result in results:
            reasons.extend(result.get("reasons", []))
            warnings.extend(result.get("warnings", []))
            if result.get("metrics"):
                metrics = result["metrics"]
        combined = self._result(candidate_id, reasons, warnings)
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
