"""Deterministic scoring adapters for production_mix_v1."""

from __future__ import annotations

from dataclasses import replace
from typing import Any, Mapping

import numpy as np

from .analyzers import score_dimensions, weighted_rule_score
from .config import ProductionMixConfig
from .models import ACTION_FX_SEND, ACTION_PARALLEL_COMPRESSION, CandidateEvaluation, MixCandidate, SafetyResult


class ProductionCriticSuite:
    def __init__(self, config: ProductionMixConfig, *, enable_optional_critics: bool = False):
        self.config = config
        self.enable_optional_critics = bool(enable_optional_critics)
        self.warnings = [] if enable_optional_critics else ["optional_critics_disabled"]

    def evaluate(
        self,
        *,
        candidate: MixCandidate,
        analysis: Mapping[str, Any],
        baseline_audio: np.ndarray,
        candidate_audio: np.ndarray,
        sample_rate: int,
        safety: SafetyResult,
        critical_degradations: list[str],
        baseline_analysis: Mapping[str, Any] | None = None,
        render_verification: Mapping[str, Any] | None = None,
        **_: Any,
    ) -> CandidateEvaluation:
        del baseline_audio, candidate_audio, sample_rate, baseline_analysis
        dimensions = score_dimensions(analysis)
        musical = self._musical_score(candidate, dimensions, render_verification or {})
        technical = 0.86 * float(dimensions.get("safety_margin", 0.0)) + 0.14 * (1.0 if safety.passed else 0.0)
        final = max(0.0, min(1.0, 0.82 * musical + 0.18 * technical))
        return CandidateEvaluation(
            candidate=candidate,
            analysis=dict(analysis),
            dimensions=dimensions,
            critic_scores={"rules": {"score": weighted_rule_score(dimensions, self.config), "primary": True}},
            aggregate_score=final,
            musical_score=musical,
            technical_safety_score=technical,
            final_score=final,
            primary_evidence=["rules"],
            warnings=list(self.warnings),
            safety=safety,
            render_verification=dict(render_verification or {}),
            critical_degradations=critical_degradations,
        )

    def with_delta(self, evaluation: CandidateEvaluation, baseline: CandidateEvaluation) -> CandidateEvaluation:
        return replace(
            evaluation,
            score_delta=float(evaluation.aggregate_score - baseline.aggregate_score),
            musical_score_delta=float(evaluation.musical_score - baseline.musical_score),
            technical_score_delta=float(evaluation.technical_safety_score - baseline.technical_safety_score),
            final_score_delta=float(evaluation.final_score - baseline.final_score),
        )

    def _musical_score(self, candidate: MixCandidate, dimensions: Mapping[str, float], verification: Mapping[str, Any]) -> float:
        weights = {k: v for k, v in self.config.weights.items() if k != "safety_margin"}
        total = sum(float(dimensions.get(k, 0.0)) * float(w) for k, w in weights.items())
        weight_sum = sum(float(w) for w in weights.values()) or 1.0
        score = total / weight_sum
        scoring = dict(self.config.reference_recipe.get("scoring", {}) or {})
        musical_actions = [
            a for a in candidate.actions
            if a.channel_id is not None and not a.parameters.get("utility_gain_staging")
        ]
        score -= len(musical_actions) * float(scoring.get("action_count_penalty", 0.0015))
        if any(a.action_type == ACTION_PARALLEL_COMPRESSION and a.parameters.get("role") in {"overheads", "cymbal", "ride", "hihat"} for a in musical_actions):
            score -= float(scoring.get("overhead_dynamics_penalty", 0.020))
        if candidate.metadata.get("candidate_type") == "drum_glue" and any(a.parameters.get("role") == "bass" for a in musical_actions):
            score -= float(scoring.get("bass_in_drum_glue_penalty", 0.015))
        if candidate.metadata.get("candidate_type") in {"reference_taste", "reference_taste_guitar_control"}:
            score += float(scoring.get("reference_taste_bias", 0.003))
        fx_wetness = float(verification.get("fx_wetness_proxy", 0.0))
        fx_threshold = float(scoring.get("fx_wetness_threshold", 0.002))
        spatial_ok = bool(verification.get("spatial_fx_verified", False))
        modulation_ok = bool(verification.get("modulation_fx_verified", False))
        if any(a.action_type == ACTION_FX_SEND for a in musical_actions) and fx_wetness >= fx_threshold and spatial_ok and modulation_ok:
            score += min(0.012, fx_wetness * 0.20)
        guitar = dict(verification.get("guitar_forwardness", {}) or {})
        if guitar:
            if bool(guitar.get("guitars_too_forward")):
                score -= float(scoring.get("guitar_too_forward_penalty", 0.018)) * float(guitar.get("guitar_vocal_masking_index_after", guitar.get("guitar_vocal_masking_index_before", 0.0)))
            score -= float(scoring.get("guitar_vocal_masking_penalty", 0.014)) * max(0.0, float(guitar.get("guitar_vocal_masking_index_after", 0.0)) - 0.62)
            if bool(guitar.get("guitar_masking_improved")) and bool(guitar.get("reference_preserved")):
                before = float(guitar.get("guitar_vocal_masking_index_before", 0.0))
                after = float(guitar.get("guitar_vocal_masking_index_after", 0.0))
                score += min(float(scoring.get("guitar_masking_improvement_reward", 0.026)), max(0.0, before - after) * 0.65)
            if bool(guitar.get("guitar_power_loss_detected")):
                score -= float(scoring.get("guitar_power_loss_penalty", 0.030))
        if verification.get("pre_trim_headroom_rejection") in {"review_required", "hard_reject"}:
            score -= 0.05
        return float(max(0.0, min(1.0, score)))
