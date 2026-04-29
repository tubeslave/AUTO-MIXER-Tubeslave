"""Candidate decision engine for `production_mix_v1`."""

from __future__ import annotations

from dataclasses import replace
from typing import Iterable

from .config import ProductionMixConfig
from .models import (
    ACTION_COMPRESSION,
    ACTION_FX_SEND,
    ACTION_GAIN,
    ACTION_PARALLEL_COMPRESSION,
    CandidateEvaluation,
    DecisionResult,
    MixAction,
)


class ProductionMixDecisionEngine:
    """Choose a winning candidate without sending OSC."""

    def __init__(self, config: ProductionMixConfig):
        self.config = config

    def decide(
        self,
        evaluations: Iterable[CandidateEvaluation],
        *,
        mode: str,
        dry_run: bool = True,
    ) -> DecisionResult:
        items = list(evaluations)
        if not items:
            raise ValueError("at least one candidate evaluation is required")
        baseline = self._baseline(items)
        items = [
            replace(
                item,
                material_improvement=self._material_improvement(item, mode=mode),
            )
            for item in items
        ]
        baseline = self._baseline(items)
        threshold = self.config.min_score_improvement(mode)
        min_muq = self.config.min_muq_delta(mode)
        rejected: list[dict[str, object]] = []
        viable: list[CandidateEvaluation] = []

        for item in items:
            if item.candidate.name == baseline.candidate.name:
                continue
            reasons = self._reject_reasons(
                item,
                threshold=threshold,
                min_muq=min_muq,
                mode=mode,
            )
            if reasons:
                rejected.append(
                    {
                        "candidate": item.candidate.name,
                        "candidate_type": self._candidate_type(item),
                        "reasons": reasons,
                        "score_delta": item.score_delta,
                        "muq_delta": item.muq_delta,
                        "material_improvement": item.material_improvement,
                    }
                )
                continue
            viable.append(item)

        if not viable:
            state = "review_required" if any(
                any(
                    str(reason).startswith("review_required")
                    or "dryness" in str(reason)
                    or "compression_gain_reduction_outliers" in str(reason)
                    or "fixed_reference_group_targets" in str(reason)
                    for reason in item.get("reasons", [])
                )
                for item in rejected
            ) else "no_change"
            return DecisionResult(
                selected=baseline,
                baseline=baseline,
                accepted=False,
                decision_state=state,
                rejected=rejected,
                rationale=[
                    "No candidate cleared score, critic, and safety requirements.",
                    "Selected no-change baseline as rollback-safe output.",
                ],
            )

        selected = max(viable, key=lambda item: (self._musical_score(item), item.final_score))
        offline_render_accepted = mode == "offline"
        console_actions_accepted = bool(not dry_run)
        live_ready = bool(mode == "live" and not dry_run)
        if live_ready:
            decision_state = "live_ready"
        elif console_actions_accepted:
            decision_state = "console_actions_accepted"
        elif offline_render_accepted:
            decision_state = "offline_render_accepted"
        else:
            decision_state = "review_required"
        rationale = [
            (
                f"Selected musical candidate {selected.candidate.name}: "
                f"musical_score_delta={self._musical_delta(selected):.4f}, "
                f"final_score_delta={self._final_delta(selected):.4f}."
            ),
            "Utility safety trim is applied only after musical candidate selection.",
            "Decision Engine did not send OSC; actions must pass Safety Governor and queue.",
        ]
        highest_final = max(viable, key=lambda item: item.final_score)
        if highest_final.candidate.name != selected.candidate.name:
            rationale.append(
                (
                    f"{selected.candidate.name} was selected over higher final_score candidate "
                    f"{highest_final.candidate.name} because musical_score is prioritized before "
                    "technical/final score; this is reported for review."
                )
            )
        if dry_run:
            rationale.append("dry_run=True: console_actions_accepted=false and live_ready=false.")
        return DecisionResult(
            selected=selected,
            baseline=baseline,
            accepted=True,
            decision_state=decision_state,
            offline_render_accepted=offline_render_accepted,
            console_actions_accepted=console_actions_accepted,
            live_ready=live_ready,
            rejected=rejected,
            rationale=rationale,
        )

    def _baseline(self, evaluations: list[CandidateEvaluation]) -> CandidateEvaluation:
        for item in evaluations:
            if item.candidate.name == self.config.baseline_candidate_name:
                return item
        return evaluations[0]

    def _reject_reasons(
        self,
        item: CandidateEvaluation,
        *,
        threshold: float,
        min_muq: float,
        mode: str,
    ) -> list[str]:
        reasons: list[str] = []
        decision_cfg = self.config.decision
        musical_delta = self._musical_delta(item)
        technical_delta = self._technical_delta(item)
        final_delta = self._final_delta(item)
        candidate_type = self._candidate_type(item)
        muq_unavailable = self._muq_unavailable_or_excluded(item)
        if self._is_utility_only(item):
            reasons.append("utility_only_candidate")
        if not self._has_musical_actions(item):
            reasons.append("no_musical_recipe_actions")
        if musical_delta < threshold and not self._allow_small_dynamics(item, mode):
            reasons.append("below_min_score_improvement")
        if not item.material_improvement and not self._fixes_critical_safety_issue(item):
            if candidate_type == "single_fix" and len(self._musical_actions(item)) <= 1:
                reasons.append("review_required_micro_fix_without_critical_safety_fix")
            else:
                reasons.append("review_required_no_material_improvement")
        if mode == "offline" and muq_unavailable:
            min_final, min_musical = self._material_thresholds(item, mode=mode)
            if final_delta < min_final:
                reasons.append("review_required_muq_unavailable_below_material_final_delta")
            if musical_delta < min_musical:
                reasons.append("review_required_muq_unavailable_below_material_musical_delta")
        if (
            self._has_proxy_fallback(item)
            and musical_delta < threshold
            and (technical_delta > threshold or final_delta > threshold)
        ):
            reasons.append("review_required_proxy_fallback_technical_only")
        if (
            item.muq_delta is not None
            and "muq_eval" in item.primary_evidence
            and item.muq_delta < min_muq
        ):
            reasons.append("below_min_muq_delta")
        if bool(decision_cfg.get("reject_proxy_only_winners", True)) and not item.primary_evidence:
            reasons.append("proxy_only_winner")
        if bool(decision_cfg.get("require_safety_pass", True)):
            if item.safety is None or not item.safety.passed:
                reasons.append("safety_governor_rejected")
        if bool(decision_cfg.get("require_no_critical_degradation", True)):
            if item.critical_degradations:
                reasons.extend(f"critical_degradation:{name}" for name in item.critical_degradations)
        if candidate_type != "reference_taste" and self._final_delta(item) < 0.01 and len(item.candidate.actions) > 10:
            reasons.append("review_required_small_final_delta_many_actions")
        if (
            candidate_type != "reference_taste"
            and muq_unavailable
            and (candidate_type == "large_remix" or self._is_large_remix_without_muq(item))
        ):
            reasons.append("review_required_muq_excluded_large_remix")
        if self._uses_fixed_reference_group_targets(item):
            reasons.append("fixed_reference_group_targets_not_allowed_for_new_song")
        reasons.extend(self._render_reject_reasons(item, mode=mode))
        if candidate_type == "reference_taste" and reasons:
            return self._reference_taste_reasons(item, reasons)
        return reasons

    @staticmethod
    def _musical_score(item: CandidateEvaluation) -> float:
        if item.musical_score or item.musical_score_delta:
            return float(item.musical_score)
        return float(item.aggregate_score)

    @staticmethod
    def _musical_delta(item: CandidateEvaluation) -> float:
        if item.musical_score or item.musical_score_delta:
            return float(item.musical_score_delta)
        return float(item.score_delta)

    @staticmethod
    def _technical_delta(item: CandidateEvaluation) -> float:
        if item.technical_safety_score or item.technical_score_delta:
            return float(item.technical_score_delta)
        return 0.0

    @staticmethod
    def _final_delta(item: CandidateEvaluation) -> float:
        if item.final_score or item.final_score_delta:
            return float(item.final_score_delta)
        return float(item.score_delta)

    @staticmethod
    def _has_musical_actions(item: CandidateEvaluation) -> bool:
        return bool(ProductionMixDecisionEngine._musical_actions(item))

    @staticmethod
    def _musical_actions(item: CandidateEvaluation) -> list[MixAction]:
        return [action for action in item.candidate.actions if not _is_utility_action(action)]

    @staticmethod
    def _is_utility_only(item: CandidateEvaluation) -> bool:
        actions = list(item.candidate.actions)
        return bool(actions) and all(_is_utility_action(action) for action in actions)

    def _allow_small_dynamics(self, item: CandidateEvaluation, mode: str) -> bool:
        if self._muq_unavailable_or_excluded(item):
            return False
        if not bool(self.config.mode_config(mode).get("allow_small_improvements", True)):
            return False
        if not any(
            action.action_type in {ACTION_COMPRESSION, ACTION_PARALLEL_COMPRESSION}
            for action in item.candidate.actions
        ):
            return False
        return item.material_improvement and self._musical_delta(item) > -self.config.min_score_improvement(mode)

    def _material_improvement(self, item: CandidateEvaluation, *, mode: str) -> bool:
        if item.candidate.name == self.config.baseline_candidate_name:
            return False
        if self._fixes_critical_safety_issue(item):
            return True
        final_delta = self._final_delta(item)
        musical_delta = self._musical_delta(item)
        min_final, min_musical = self._material_thresholds(item, mode=mode)
        if mode == "offline" and self._muq_unavailable_or_excluded(item):
            return final_delta >= min_final and musical_delta >= min_musical
        return final_delta >= min_final or musical_delta >= min_musical

    def _material_thresholds(self, item: CandidateEvaluation, *, mode: str) -> tuple[float, float]:
        mode_cfg = self.config.mode_config(mode)
        if mode == "offline" and self._muq_unavailable_or_excluded(item):
            return (
                float(mode_cfg.get("min_material_final_delta_when_muq_unavailable", 0.010)),
                float(mode_cfg.get("min_material_musical_delta_when_muq_unavailable", 0.020)),
            )
        threshold = self.config.min_score_improvement(mode)
        return threshold, threshold

    @staticmethod
    def _fixes_critical_safety_issue(item: CandidateEvaluation) -> bool:
        verification = dict(item.render_verification or {})
        if bool(verification.get("critical_safety_fixed", False)):
            return True
        for action in item.candidate.actions:
            if bool(action.parameters.get("critical_safety_fix", False)):
                return True
        return False

    @staticmethod
    def _candidate_type(item: CandidateEvaluation) -> str:
        return str(item.candidate.metadata.get("candidate_type", "large_remix" if len(item.candidate.actions) > 10 else "single_fix"))

    @staticmethod
    def _reference_taste_reasons(item: CandidateEvaluation, reasons: list[str]) -> list[str]:
        accurate: list[str] = []
        if item.final_score_delta < 0.0 or item.musical_score_delta < 0.0:
            accurate.append("farther_from_current_song")
        if any(
            "safety" in reason
            or "critical_degradation" in reason
            or "render" in reason
            or "dry" in reason
            or "metrics_conflict" in reason
            for reason in reasons
        ):
            accurate.append("reference_not_suitable_for_song")
        if any(
            reason in {
                "below_min_score_improvement",
                "review_required_no_material_improvement",
                "review_required_muq_unavailable_below_material_final_delta",
                "review_required_muq_unavailable_below_material_musical_delta",
            }
            for reason in reasons
        ):
            accurate.append("below_min_score_improvement")
        if not accurate:
            accurate.append("reference_not_suitable_for_song")
        return list(dict.fromkeys(accurate))

    @staticmethod
    def _has_proxy_fallback(item: CandidateEvaluation) -> bool:
        joined = " ".join([*item.proxy_evidence, *item.warnings])
        return "fallback" in joined or "proxy" in joined

    @staticmethod
    def _muq_unavailable_or_excluded(item: CandidateEvaluation) -> bool:
        joined = " ".join([*item.proxy_evidence, *item.warnings])
        return (
            "muq_eval_unavailable" in joined
            or "muq_eval_fallback_excluded" in joined
            or "optional_critics_disabled" in joined
        )

    def _is_large_remix_without_muq(self, item: CandidateEvaluation) -> bool:
        guard = dict(self.config.large_correction_guard or {})
        max_actions = int(guard.get("max_actions_without_muq", 10))
        max_gain = float(guard.get("max_gain_without_muq_db", 1.5))
        musical_actions = [action for action in item.candidate.actions if not _is_utility_action(action)]
        if len(musical_actions) > max_actions:
            return True
        for action in musical_actions:
            if action.action_type == ACTION_GAIN and action.channel_id is not None:
                if abs(float(action.parameters.get("gain_db", 0.0))) > max_gain:
                    return True
        return self._uses_fixed_reference_group_targets(item)

    @staticmethod
    def _uses_fixed_reference_group_targets(item: CandidateEvaluation) -> bool:
        return any(
            action.stage == "role_group_balance_from_report"
            or action.parameters.get("reason") == "role_group_balance_from_report"
            for action in item.candidate.actions
        )

    def _render_reject_reasons(self, item: CandidateEvaluation, *, mode: str) -> list[str]:
        reasons: list[str] = []
        verification = dict(item.render_verification or {})
        if mode == "offline":
            if item.candidate.actions and not bool(verification.get("actual_offline_render", False)):
                reasons.append("offline_render_missing")
            if item.candidate.actions and not bool(verification.get("required_actions_rendered", False)):
                reasons.append("required_actions_not_rendered")
            if item.candidate.actions and not bool(verification.get("required_actions_verified", False)):
                reasons.append("required_actions_not_verified")
            if item.candidate.actions and not bool(verification.get("metrics_improved", False)):
                reasons.append("rendered_audio_metrics_not_improved")
        if item.candidate.name == "candidate_075_combined_musical_mix":
            if not bool(verification.get("has_static_gain_balance", False)):
                reasons.append("combined_candidate_missing_fader_balance")
        if any(action.action_type == ACTION_FX_SEND for action in item.candidate.actions):
            if not bool(verification.get("fx_return_present", False)):
                reasons.append("fx_return_not_present_in_offline_render")
            if not bool(verification.get("candidate_is_not_dry", False)):
                reasons.append("dry_mix_penalty_fx_wetness_below_threshold")
        if float(verification.get("dryness_penalty", 0.0)) > 0.0:
            reasons.append("natural_ambience_destroyed_dryness_penalty")
        if any(
            action.action_type in {ACTION_COMPRESSION, ACTION_PARALLEL_COMPRESSION}
            for action in item.candidate.actions
        ):
            if not bool(verification.get("candidate_is_not_overdynamic", False)):
                reasons.append("overdynamic_mix_compression_proxy_below_threshold")
            if verification.get("compression_gr_outliers"):
                reasons.append("compression_gain_reduction_outliers")
        if verification.get("metrics_conflicts"):
            reasons.append("metrics_conflict")
        if not bool(verification.get("vocal_balance_ok", True)):
            reasons.append("vocal_balance_worsened")
        if not bool(verification.get("stem_balance_ok", True)):
            reasons.append("stem_balance_worsened")
        if not bool(verification.get("kick_bass_balance_ok", True)):
            reasons.append("kick_bass_balance_worsened")
        reasons.extend(_action_contract_reasons(item))
        return reasons


def _is_utility_action(action: MixAction) -> bool:
    return (
        action.stage == "final_safety_trim"
        or (
            action.action_type == ACTION_GAIN
            and action.channel_id is None
            and action.target == "mix_bus"
        )
    )


def _action_contract_reasons(item: CandidateEvaluation) -> list[str]:
    reasons: list[str] = []
    for action in item.candidate.actions:
        if action.action_type in {ACTION_COMPRESSION, ACTION_PARALLEL_COMPRESSION}:
            missing = [
                key
                for key in (
                    "threshold_db",
                    "ratio",
                    "attack_ms",
                    "release_ms",
                    "knee_db",
                    "makeup_gain_db",
                    "wet_mix_percent",
                    "target_gain_reduction_db",
                    "reason",
                    "expected_metric_change",
                )
                if key not in action.parameters or action.parameters.get(key) in ("", None)
            ]
            if missing:
                reasons.append("compression_parameters_missing")
        if action.action_type == ACTION_FX_SEND:
            missing = [
                key
                for key in (
                    "fx_slot",
                    "fx_model",
                    "send_level_db",
                    "return_level_db",
                    "send_mode",
                    "route_to_main",
                    "wet_target_percent",
                    "reason",
                )
                if key not in action.parameters or action.parameters.get(key) in ("", None)
            ]
            if missing:
                reasons.append("fx_send_parameters_missing")
            elif not bool(action.parameters.get("route_to_main", False)):
                reasons.append("fx_send_return_not_routed_to_main")
    return sorted(set(reasons))
