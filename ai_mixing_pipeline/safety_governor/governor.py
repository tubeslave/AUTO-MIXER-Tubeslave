"""Final offline safety layer for candidate actions and rendered audio."""

from __future__ import annotations

from typing import Any

from ai_mixing_pipeline.audio_utils import measure_audio_file
from ai_mixing_pipeline.models import MixCandidate, SafetyResult
from ai_mixing_pipeline.numeric_safety import finite_float


class SafetyGovernor:
    """Reject dangerous offline candidate actions before acceptance."""

    def __init__(self, config: dict[str, Any] | None = None):
        self.config = dict(config or {})
        self.safety = dict((self.config.get("safety", {}) or {}))

    def evaluate(
        self,
        candidate: MixCandidate,
        render_path: str,
        *,
        critic_results: dict[str, dict[str, Any]] | None = None,
        score_improvement: float | None = None,
        enforce_min_improvement: bool = False,
    ) -> SafetyResult:
        """Return safety verdict for candidate actions and render metrics."""

        critic_results = dict(critic_results or {})
        reasons: list[str] = []
        warnings: list[str] = []
        metrics = measure_audio_file(render_path)
        level = metrics.get("level", {})
        stereo = metrics.get("stereo", {})
        spectral = metrics.get("spectral", {})

        max_true_peak = self._number(self.safety.get("max_true_peak_dbfs", -1.0), "safety_limit:max_true_peak_dbfs", reasons)
        min_headroom = self._number(self.safety.get("min_headroom_db", 1.0), "safety_limit:min_headroom_db", reasons)
        true_peak = self._number(level.get("true_peak_dbtp", level.get("peak_dbfs")), "render:true_peak_dbtp", reasons)
        headroom = self._number(level.get("headroom_db"), "render:headroom_db", reasons)
        clip_count = self._number(level.get("clip_count"), "render:clip_count", reasons)

        if clip_count is not None and bool(self.safety.get("forbid_clipping", True)) and clip_count > 0:
            reasons.append("clipping_detected")
        if true_peak is not None and max_true_peak is not None and true_peak > max_true_peak + 1e-6:
            reasons.append(f"true_peak_exceeds_limit:{true_peak:.2f}>{max_true_peak:.2f}")
        if headroom is not None and min_headroom is not None and headroom < min_headroom - 1e-6:
            reasons.append(f"insufficient_headroom:{headroom:.2f}<{min_headroom:.2f}")

        self._check_actions(candidate, reasons, warnings)
        self._check_critic_regressions(critic_results, reasons, warnings)

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

        harsh = self._number(spectral.get("harshness_proxy"), "render:harshness_proxy", reasons)
        mud = self._number(spectral.get("muddiness_proxy"), "render:muddiness_proxy", reasons)
        if harsh is not None and harsh > 0.45:
            reasons.append("major_harshness_increase_or_excess")
        if mud is not None and mud > 0.45:
            reasons.append("low_mid_mud_excess")

        if enforce_min_improvement:
            min_improvement = self._number(self.safety.get("min_score_improvement", 0.03), "safety_limit:min_score_improvement", reasons)
            if candidate.candidate_id != "000_initial_mix":
                improvement = self._number(score_improvement, "score_improvement", reasons)
                if improvement is not None and min_improvement is not None and improvement < min_improvement:
                    reasons.append("score_improvement_below_threshold")

        safety_score = self._score_from_reasons(reasons, warnings)
        return SafetyResult(
            candidate_id=candidate.candidate_id,
            passed=not reasons,
            safety_score=safety_score,
            reasons=reasons,
            warnings=warnings,
            metrics=metrics,
        )

    def _check_actions(
        self,
        candidate: MixCandidate,
        reasons: list[str],
        warnings: list[str],
    ) -> None:
        max_gain = self._number(self.safety.get("max_gain_change_db_per_step", 1.0), "safety_limit:max_gain_change_db_per_step", reasons)
        max_eq = self._number(self.safety.get("max_eq_change_db_per_step", 1.5), "safety_limit:max_eq_change_db_per_step", reasons)
        for action in candidate.actions:
            params = action.parameters
            if action.action_type == "gain_change":
                gain = self._number(params.get("gain_db"), f"action:{action.action_type}:gain_db", reasons)
                if gain is not None and max_gain is not None and abs(gain) > max_gain + 1e-6:
                    reasons.append(f"excessive_gain_change:{action.target}:{gain:.2f}")
            if action.action_type == "eq_correction":
                gain = self._number(params.get("gain_db"), f"action:{action.action_type}:gain_db", reasons)
                if gain is not None and max_eq is not None and abs(gain) > max_eq + 1e-6:
                    reasons.append(f"excessive_eq_change:{action.target}:{gain:.2f}")
            if action.action_type == "compression_correction":
                ratio = self._number(params.get("ratio"), f"action:{action.action_type}:ratio", reasons)
                makeup = self._number(params.get("makeup_db"), f"action:{action.action_type}:makeup_db", reasons)
                if ratio is not None and bool(self.safety.get("forbid_excessive_compression", True)) and ratio > 3.0:
                    reasons.append(f"excessive_compression_ratio:{action.target}:{ratio:.2f}")
                if makeup is not None and max_gain is not None and makeup > max_gain:
                    reasons.append(f"compressor_makeup_exceeds_gain_limit:{action.target}:{makeup:.2f}")
            if action.action_type == "fx_send_change":
                warnings.append("fx_send_change is report-only in offline_test; no OSC/MIDI command is sent.")

    def _check_critic_regressions(
        self,
        critic_results: dict[str, dict[str, Any]],
        reasons: list[str],
        warnings: list[str],
    ) -> None:
        if bool(self.safety.get("forbid_vocal_clarity_drop", True)):
            mert_delta = (critic_results.get("mert", {}) or {}).get("delta", {})
            vocal_delta = mert_delta.get("vocal_clarity")
            if vocal_delta is not None:
                vocal_delta = self._number(vocal_delta, "critic:vocal_clarity", reasons)
            if vocal_delta is not None and vocal_delta < -0.01:
                reasons.append("vocal_clarity_drop")
        identity_scores = (critic_results.get("panns_or_beats", {}) or {}).get("scores", {})
        identity = identity_scores.get("identity_confidence")
        bleed = identity_scores.get("bleed_score")
        if identity is not None:
            identity = self._number(identity, "critic:identity_confidence", reasons)
        if identity is not None and identity < 0.35:
            reasons.append("bad_channel_identity_confidence")
        if bleed is not None:
            bleed = self._number(bleed, "critic:bleed_score", reasons)
        if bleed is not None and bleed < 0.35:
            warnings.append("identity_bleed_detector_reported_low_bleed_score")

    @staticmethod
    def _score_from_reasons(reasons: list[str], warnings: list[str]) -> float:
        score = 1.0 - 0.25 * len(reasons) - 0.05 * len(warnings)
        return float(max(0.0, min(1.0, score)))

    @staticmethod
    def _number(value: Any, label: str, reasons: list[str]) -> float | None:
        numeric = finite_float(value)
        if numeric is None:
            reasons.append(f"invalid_numeric:{label}")
        return numeric
