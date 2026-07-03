"""Final offline safety layer for candidate actions and rendered audio."""

from __future__ import annotations

from typing import Any

from ai_mixing_pipeline.audio_utils import measure_audio_file
from ai_mixing_pipeline.models import MixCandidate, SafetyResult


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

        max_true_peak = float(self.safety.get("max_true_peak_dbfs", -1.0))
        min_headroom = float(self.safety.get("min_headroom_db", 1.0))
        true_peak = float(level.get("true_peak_dbtp", level.get("peak_dbfs", -120.0)) or -120.0)
        headroom = float(level.get("headroom_db", 0.0) or 0.0)
        clip_count = int(level.get("clip_count", 0) or 0)

        if bool(self.safety.get("forbid_clipping", True)) and clip_count > 0:
            reasons.append("clipping_detected")
        if true_peak > max_true_peak + 1e-6:
            reasons.append(f"true_peak_exceeds_limit:{true_peak:.2f}>{max_true_peak:.2f}")
        if headroom < min_headroom - 1e-6:
            reasons.append(f"insufficient_headroom:{headroom:.2f}<{min_headroom:.2f}")

        self._check_actions(candidate, reasons, warnings)
        self._check_critic_regressions(critic_results, reasons, warnings)

        if bool(self.safety.get("forbid_phase_collapse", True)):
            if bool(stereo.get("phase_cancellation_risk", False)):
                reasons.append("phase_cancellation_risk")
            if float(stereo.get("inter_channel_correlation", 1.0) or 1.0) < 0.05:
                reasons.append("phase_correlation_too_low")

        harsh = float(spectral.get("harshness_proxy", 0.0) or 0.0)
        mud = float(spectral.get("muddiness_proxy", 0.0) or 0.0)
        if harsh > 0.45:
            reasons.append("major_harshness_increase_or_excess")
        if mud > 0.45:
            reasons.append("low_mid_mud_excess")

        if enforce_min_improvement:
            min_improvement = float(self.safety.get("min_score_improvement", 0.03))
            if candidate.candidate_id != "000_initial_mix" and float(score_improvement or 0.0) < min_improvement:
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
        max_gain = float(self.safety.get("max_gain_change_db_per_step", 1.0))
        max_eq = float(self.safety.get("max_eq_change_db_per_step", 1.5))
        for action in candidate.actions:
            params = action.parameters
            if action.action_type == "gain_change":
                gain = float(params.get("gain_db", 0.0))
                if abs(gain) > max_gain + 1e-6:
                    reasons.append(f"excessive_gain_change:{action.target}:{gain:.2f}")
            if action.action_type == "eq_correction":
                gain = float(params.get("gain_db", 0.0))
                if abs(gain) > max_eq + 1e-6:
                    reasons.append(f"excessive_eq_change:{action.target}:{gain:.2f}")
            if action.action_type == "compression_correction":
                ratio = float(params.get("ratio", 1.0))
                makeup = float(params.get("makeup_db", 0.0))
                if bool(self.safety.get("forbid_excessive_compression", True)) and ratio > 3.0:
                    reasons.append(f"excessive_compression_ratio:{action.target}:{ratio:.2f}")
                if makeup > max_gain:
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
            if isinstance(vocal_delta, (int, float)) and float(vocal_delta) < -0.01:
                reasons.append("vocal_clarity_drop")
        identity_scores = (critic_results.get("panns_or_beats", {}) or {}).get("scores", {})
        identity = identity_scores.get("identity_confidence")
        bleed = identity_scores.get("bleed_score")
        if isinstance(identity, (int, float)) and float(identity) < 0.35:
            reasons.append("bad_channel_identity_confidence")
        if isinstance(bleed, (int, float)) and float(bleed) < 0.35:
            warnings.append("identity_bleed_detector_reported_low_bleed_score")

    @staticmethod
    def _score_from_reasons(reasons: list[str], warnings: list[str]) -> float:
        score = 1.0 - 0.25 * len(reasons) - 0.05 * len(warnings)
        return float(max(0.0, min(1.0, score)))
