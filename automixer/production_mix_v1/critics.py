"""Critic and scoring adapters for `production_mix_v1`."""

from __future__ import annotations

from dataclasses import replace
from typing import Any, Mapping

import numpy as np

from .analyzers import score_dimensions, weighted_rule_score
from .config import ProductionMixConfig
from .models import (
    ACTION_COMPRESSION,
    ACTION_FX_SEND,
    ACTION_GAIN,
    ACTION_PARALLEL_COMPRESSION,
    CandidateEvaluation,
    MixAction,
    MixCandidate,
    SafetyResult,
)


class ProductionCriticSuite:
    """Score candidates with rules plus optional primary/proxy critics."""

    def __init__(
        self,
        config: ProductionMixConfig,
        *,
        enable_optional_critics: bool = True,
    ):
        self.config = config
        self.enable_optional_critics = bool(enable_optional_critics)
        self.warnings: list[str] = []
        self._muq_service: Any | None = None
        self._mert_evaluator: Any | None = None
        self._essentia: Any | None = None
        if self.enable_optional_critics:
            self._load_optional_critics()
        else:
            self.warnings.append("optional_critics_disabled")

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
        musical_analysis: Mapping[str, Any] | None = None,
        render_verification: Mapping[str, Any] | None = None,
    ) -> CandidateEvaluation:
        """Return a candidate evaluation. Score deltas are filled by caller."""
        dimensions = score_dimensions(analysis)
        del musical_analysis
        musical_dimensions = score_dimensions(analysis)
        rule_score = weighted_rule_score(dimensions, self.config)
        musical_score = self._musical_score(
            candidate,
            musical_dimensions,
            baseline_analysis,
            render_verification or {},
        )
        technical_score = self._technical_safety_score(analysis, safety)
        final_score = self._final_score(musical_score, technical_score)
        critic_scores: dict[str, Any] = {
            "rules": {
                "score": rule_score,
                "musical_score": musical_score,
                "technical_safety_score": technical_score,
                "final_score": final_score,
                "primary": True,
                "proxy": False,
                "rationale": (
                    "Deterministic production_mix_v1 rule dimensions with musical/technical split."
                ),
            }
        }
        primary_evidence = ["rules"]
        proxy_evidence: list[str] = []
        warnings = list(self.warnings)

        muq_delta = None
        muq = self._score_muq(baseline_audio, candidate_audio, sample_rate)
        if muq is not None:
            critic_scores["muq_eval"] = muq
            if muq["primary"]:
                primary_evidence.append("muq_eval")
                muq_delta = float(muq.get("delta", 0.0))
            else:
                proxy_evidence.append("muq_eval")
                warnings.append("muq_eval_fallback_excluded_from_final_score")

        mert = self._score_mert(baseline_audio, candidate_audio, sample_rate, candidate)
        if mert is not None:
            critic_scores["mert"] = mert
            if mert["primary"]:
                primary_evidence.append("mert")
            else:
                proxy_evidence.append("mert")
                warnings.append("mert_fallback_excluded_from_final_score")

        essentia = self._score_essentia(candidate_audio, sample_rate)
        if essentia is not None:
            critic_scores["essentia"] = essentia
            if essentia["primary"]:
                primary_evidence.append("essentia")
            else:
                proxy_evidence.append("essentia")

        aggregate = self._aggregate_primary_scores(final_score, critic_scores)
        rationale = [
            f"rules={rule_score:.4f}",
            f"musical_score={musical_score:.4f}",
            f"technical_safety_score={technical_score:.4f}",
            f"final_score={aggregate:.4f}",
            f"aggregate_primary={aggregate:.4f}",
            "proxy/fallback critic scores are logged but excluded from final authority.",
        ]
        if safety.passed:
            rationale.append("Safety Governor precheck passed.")
        else:
            rationale.append("Safety Governor precheck blocked this candidate.")

        return CandidateEvaluation(
            candidate=candidate,
            analysis=dict(analysis),
            dimensions=dimensions,
            critic_scores=critic_scores,
            aggregate_score=aggregate,
            musical_score=musical_score,
            technical_safety_score=technical_score,
            final_score=aggregate,
            score_delta=0.0,
            musical_score_delta=0.0,
            technical_score_delta=0.0,
            final_score_delta=0.0,
            muq_delta=muq_delta,
            primary_evidence=primary_evidence,
            proxy_evidence=proxy_evidence,
            warnings=warnings,
            safety=safety,
            render_verification=dict(render_verification or {}),
            critical_degradations=critical_degradations,
            rationale=rationale,
        )

    def with_delta(
        self,
        evaluation: CandidateEvaluation,
        baseline: CandidateEvaluation,
    ) -> CandidateEvaluation:
        """Return evaluation with aggregate delta relative to baseline."""
        return replace(
            evaluation,
            score_delta=float(evaluation.aggregate_score - baseline.aggregate_score),
            musical_score_delta=float(evaluation.musical_score - baseline.musical_score),
            technical_score_delta=float(
                evaluation.technical_safety_score - baseline.technical_safety_score
            ),
            final_score_delta=float(evaluation.final_score - baseline.final_score),
        )

    def _musical_score(
        self,
        candidate: MixCandidate,
        dimensions: Mapping[str, float],
        baseline_analysis: Mapping[str, Any] | None,
        render_verification: Mapping[str, Any],
    ) -> float:
        del baseline_analysis
        weights = {
            key: value
            for key, value in self.config.weights.items()
            if key != "safety_margin"
        }
        base = _weighted_score(dimensions, weights)
        actions = list(candidate.actions)
        has_space_actions = any(action.action_type == ACTION_FX_SEND for action in actions)
        has_dynamics_actions = any(
            action.action_type in {ACTION_COMPRESSION, ACTION_PARALLEL_COMPRESSION}
            for action in actions
        )
        utility_actions = [action for action in actions if _is_utility_action(action)]
        musical_actions = [action for action in actions if not _is_utility_action(action)]
        if not musical_actions and utility_actions:
            return _clamp01(base)

        scoring = dict(self.config.reference_recipe.get("scoring", {}) or {})
        score = base
        balance = dict(render_verification.get("balance_before_after", {}) or {})
        stem_delta = _metric_delta(balance, "stem_balance_score")
        spectral_delta = _metric_delta(balance, "spectral_balance")
        crest_error_delta = _metric_delta(balance, "crest_target_error_db")
        vocal_delta = _metric_delta(balance, "vocal_to_mix_ratio_db")
        if stem_delta > 0.0:
            score += min(0.040, stem_delta * 0.20)
        if spectral_delta > 0.0:
            score += min(0.030, spectral_delta * 0.20)
        if crest_error_delta < 0.0:
            score += min(0.030, abs(crest_error_delta) / 18.0)
        if vocal_delta < -1.5:
            score -= min(0.080, abs(vocal_delta) / 24.0)
        if has_space_actions and not bool(render_verification.get("candidate_is_not_dry", False)):
            score -= float(scoring.get("dry_mix_penalty", 0.060))
        if has_dynamics_actions and not bool(
            render_verification.get("candidate_is_not_overdynamic", False)
        ):
            score -= float(scoring.get("overdynamic_penalty", 0.060))
        if actions and not bool(render_verification.get("required_actions_rendered", False)):
            score -= 0.12
        score -= float(render_verification.get("dryness_penalty", 0.0)) * float(
            scoring.get("dryness_penalty_weight", 0.080)
        )
        action_overage = max(0, len(musical_actions) - 10)
        score -= action_overage * float(scoring.get("action_count_penalty_per_action_over_10", 0.004))
        large_gain_db = sum(
            max(0.0, abs(float(action.parameters.get("gain_db", 0.0))) - 2.0)
            for action in musical_actions
            if action.action_type == ACTION_GAIN and action.channel_id is not None
        )
        score -= large_gain_db * float(scoring.get("large_gain_penalty_per_db_over_2", 0.010))
        return _clamp01(score)

    def _technical_safety_score(
        self,
        analysis: Mapping[str, Any],
        safety: SafetyResult,
    ) -> float:
        mix = analysis.get("mix", analysis)
        dims = score_dimensions(analysis)
        true_peak = float(mix.get("true_peak_dbfs", -120.0))
        ceiling = float(self.config.safety_limits.get("max_true_peak_dbfs", -1.0))
        phase = max(0.0, min(1.0, float(mix.get("phase_correlation", 1.0))))
        true_peak_score = _clamp01(1.0 - max(0.0, true_peak - ceiling) * 0.35)
        return _clamp01(
            0.42 * float(dims.get("safety_margin", 0.0))
            + 0.28 * phase
            + 0.20 * true_peak_score
            + 0.10 * (1.0 if safety.passed else 0.0)
        )

    @staticmethod
    def _final_score(musical_score: float, technical_score: float) -> float:
        return _clamp01(0.82 * musical_score + 0.18 * technical_score)

    @staticmethod
    def _crest_factor_improved(
        musical_analysis: Mapping[str, Any],
        baseline_analysis: Mapping[str, Any] | None,
    ) -> bool:
        if not baseline_analysis:
            return False
        base = float(
            baseline_analysis.get("mix", baseline_analysis).get("crest_factor_db", 0.0)
        )
        cand = float(
            musical_analysis.get("mix", musical_analysis).get("crest_factor_db", 0.0)
        )
        target = 12.0
        return abs(cand - target) + 0.25 < abs(base - target)

    def _load_optional_critics(self) -> None:
        try:
            try:
                from evaluation.muq_eval_service import MuQEvalService
            except ImportError:
                from backend.evaluation.muq_eval_service import MuQEvalService

            self._muq_service = MuQEvalService(
                {
                    "enabled": True,
                    "fallback_enabled": True,
                    "local_files_only": True,
                    "shadow_mode": True,
                    "log_scores": False,
                    "min_seconds_between_quality_decisions": 0.0,
                }
            )
        except Exception as exc:
            self.warnings.append(f"muq_eval_unavailable:{exc}")
            self._muq_service = None

        try:
            try:
                from perceptual.perceptual_evaluator import PerceptualEvaluator
            except ImportError:
                from backend.perceptual.perceptual_evaluator import PerceptualEvaluator

            self._mert_evaluator = PerceptualEvaluator(
                {
                    "enabled": True,
                    "mode": "offline",
                    "backend": "mert",
                    "local_files_only": True,
                    "fallback_to_lightweight": True,
                    "async_evaluation": False,
                    "log_scores": False,
                }
            )
        except Exception as exc:
            self.warnings.append(f"mert_unavailable:{exc}")
            self._mert_evaluator = None

        try:
            import essentia.standard as essentia_standard

            self._essentia = essentia_standard
        except Exception as exc:
            self.warnings.append(f"essentia_unavailable:{exc}")
            self._essentia = None

    def _score_muq(
        self,
        baseline_audio: np.ndarray,
        candidate_audio: np.ndarray,
        sample_rate: int,
    ) -> dict[str, Any] | None:
        if self._muq_service is None:
            return None
        try:
            before = self._muq_service.evaluate(baseline_audio, sample_rate)
            after = self._muq_service.evaluate(candidate_audio, sample_rate)
            primary = str(after.model_status) == "available"
            return {
                "score": float(after.quality_score),
                "before": before.to_dict(),
                "after": after.to_dict(),
                "delta": float(after.quality_score - before.quality_score),
                "primary": primary,
                "proxy": not primary,
                "model_status": after.model_status,
            }
        except Exception as exc:
            self.warnings.append(f"muq_eval_score_failed:{exc}")
            return None

    def _score_mert(
        self,
        baseline_audio: np.ndarray,
        candidate_audio: np.ndarray,
        sample_rate: int,
        candidate: MixCandidate,
    ) -> dict[str, Any] | None:
        if self._mert_evaluator is None:
            return None
        try:
            result = self._mert_evaluator.score_change(
                baseline_audio,
                candidate_audio,
                sample_rate,
                context={
                    "action": candidate.to_dict(),
                    "osc_sent": False,
                    "sample_rate": sample_rate,
                },
            )
            primary = str(result.backend) == "mert"
            return {
                "score": float(result.score_after),
                "delta": float(result.delta_score),
                "backend": result.backend,
                "primary": primary,
                "proxy": not primary,
                "verdict": result.verdict,
                "confidence": float(result.confidence),
            }
        except Exception as exc:
            self.warnings.append(f"mert_score_failed:{exc}")
            return None

    def _score_essentia(self, candidate_audio: np.ndarray, sample_rate: int) -> dict[str, Any] | None:
        if self._essentia is None:
            return None
        try:
            del sample_rate
            audio = np.mean(np.asarray(candidate_audio, dtype=np.float32), axis=1)
            dynamic_complexity = self._essentia.DynamicComplexity()
            score, _loudness = dynamic_complexity(audio)
            normalized = max(0.0, min(1.0, 1.0 - abs(float(score) - 3.0) / 6.0))
            return {
                "score": normalized,
                "primary": True,
                "proxy": False,
                "rationale": "Essentia dynamic complexity sanity score.",
            }
        except Exception as exc:
            self.warnings.append(f"essentia_score_failed:{exc}")
            return None

    def _aggregate_primary_scores(
        self,
        rule_score: float,
        critic_scores: Mapping[str, Any],
    ) -> float:
        total = rule_score * float(self.config.critic_weights.get("rules", 0.15))
        weight_sum = float(self.config.critic_weights.get("rules", 0.15))
        for name in ("muq_eval", "mert", "essentia"):
            item = critic_scores.get(name)
            if not item or not bool(item.get("primary", False)):
                continue
            weight = float(self.config.critic_weights.get(name, 0.0))
            total += float(item.get("score", 0.0)) * weight
            weight_sum += weight
        if weight_sum <= 0:
            return float(rule_score)
        return float(total / weight_sum)


def _weighted_score(dimensions: Mapping[str, float], weights: Mapping[str, float]) -> float:
    total = 0.0
    weight_sum = 0.0
    for key, weight in weights.items():
        total += float(dimensions.get(key, 0.0)) * float(weight)
        weight_sum += float(weight)
    if weight_sum <= 0:
        return 0.0
    return float(total / weight_sum)


def _is_utility_action(action: MixAction) -> bool:
    return (
        action.stage == "final_safety_trim"
        or (
            action.action_type == ACTION_GAIN
            and action.channel_id is None
            and action.target == "mix_bus"
        )
    )


def _metric_delta(balance: Mapping[str, Any], key: str) -> float:
    try:
        payload = dict(balance.get(key, {}) or {})
        value = payload.get("delta")
        if value is None:
            return 0.0
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def _clamp01(value: float) -> float:
    return float(max(0.0, min(1.0, float(value))))
