"""Decision engine for production_mix_v1."""

from __future__ import annotations

from dataclasses import replace
from typing import Iterable

from .config import ProductionMixConfig
from .models import (
    ACTION_GAIN,
    ACTION_PARALLEL_COMPRESSION,
    CandidateEvaluation,
    DecisionResult,
    MixAction,
)


class ProductionMixDecisionEngine:
    def __init__(self, config: ProductionMixConfig):
        self.config = config

    def decide(self, evaluations: Iterable[CandidateEvaluation], *, mode: str, dry_run: bool = True) -> DecisionResult:
        items = list(evaluations)
        baseline = self._baseline(items)
        reference = self._reference(items)
        items = [self._with_material_flags(item, baseline=baseline, reference=reference, mode=mode) for item in items]
        baseline = self._baseline(items)
        reference = self._reference(items)
        rejected: list[dict[str, object]] = []
        viable: list[CandidateEvaluation] = []
        for item in items:
            if item.candidate.name == baseline.candidate.name:
                continue
            reasons = self._reject_reasons(item, baseline=baseline, reference=reference, mode=mode)
            if reasons:
                rejected.append({"candidate": item.candidate.name, "candidate_type": self._candidate_type(item), "reasons": reasons, "material_improvement": item.material_improvement})
            else:
                viable.append(item)
        if not viable:
            return DecisionResult(baseline, baseline, accepted=False, decision_state="review_required", rejected=rejected, rationale=["No candidate cleared material, reference, and safety requirements.", "Selected no-change baseline."])
        selected = max(viable, key=lambda item: (self._selection_score(item), item.final_score))
        selected_by_tiebreak = False
        tiebreak_reason = ""
        guitar_control = self._best_guitar_control(viable)
        if reference in viable and guitar_control is not None and self._guitar_control_can_beat_reference(guitar_control, reference, mode):
            selected = guitar_control
            selected_by_tiebreak = True
            guitar = dict(guitar_control.render_verification.get("guitar_forwardness", {}) or {})
            tiebreak_reason = (
                "guitar_control_preferred_over_reference "
                f"masking_index={float(guitar.get('guitar_vocal_masking_index_before', 0.0)):.4f}->{float(guitar.get('guitar_vocal_masking_index_after', 0.0)):.4f}"
            )
        if reference in viable and selected.candidate.name != reference.candidate.name:
            final_margin = selected.final_score - reference.final_score
            musical_margin = selected.musical_score - reference.musical_score
            eps_final = float(self.config.mode_config(mode).get("reference_tiebreak_final_epsilon", 0.005))
            eps_musical = float(self.config.mode_config(mode).get("reference_tiebreak_musical_epsilon", 0.005))
            if not self._guitar_control_can_beat_reference(selected, reference, mode) and (final_margin < eps_final or musical_margin < eps_musical):
                selected = reference
                selected_by_tiebreak = True
                tiebreak_reason = f"accepted_reference_preferred_within_epsilon final_margin={final_margin:.4f} musical_margin={musical_margin:.4f}"
        elif reference is not None and selected.candidate.name == reference.candidate.name:
            close_rejections = [
                item for item in items
                if item.candidate.name != reference.candidate.name
                and item.candidate.name != baseline.candidate.name
                and item.final_score > reference.final_score
                and (
                    item.final_score - reference.final_score < float(self.config.mode_config(mode).get("reference_tiebreak_final_epsilon", 0.005))
                    or item.musical_score - reference.musical_score < float(self.config.mode_config(mode).get("reference_tiebreak_musical_epsilon", 0.005))
                )
            ]
            if close_rejections:
                selected_by_tiebreak = True
                best_close = max(close_rejections, key=lambda item: item.final_score)
                tiebreak_reason = (
                    "accepted_reference_preferred_within_epsilon "
                    f"over={best_close.candidate.name} "
                    f"final_margin={best_close.final_score - reference.final_score:.4f} "
                    f"musical_margin={best_close.musical_score - reference.musical_score:.4f}"
                )
        offline = mode == "offline"
        accepted = selected.candidate.name != baseline.candidate.name
        state = "offline_render_accepted" if accepted and offline else "no_change"
        rationale = [
            f"Selected {selected.candidate.name}: musical_delta={selected.musical_score_delta:.4f}, final_delta={selected.final_score_delta:.4f}.",
            "Decision Engine did not send OSC; dry-run queue remains separate.",
        ]
        if selected_by_tiebreak:
            rationale.append(tiebreak_reason)
        return DecisionResult(
            selected,
            baseline,
            accepted=accepted,
            decision_state=state,
            offline_render_accepted=bool(accepted and offline),
            console_actions_accepted=bool(accepted and not dry_run),
            live_ready=bool(accepted and mode == "live" and not dry_run),
            selected_by_tiebreak=selected_by_tiebreak,
            tiebreak_reason=tiebreak_reason,
            rejected=rejected,
            rationale=rationale,
        )

    def _with_material_flags(self, item: CandidateEvaluation, *, baseline: CandidateEvaluation, reference: CandidateEvaluation | None, mode: str) -> CandidateEvaluation:
        muq_unavailable = self._muq_unavailable(item)
        min_final, min_musical = self._material_thresholds(mode, muq_unavailable=muq_unavailable)
        vs_no_change = item.final_score - baseline.final_score >= min_final and item.musical_score - baseline.musical_score >= min_musical
        if self._uses_verified_reference_threshold(item, reference=reference, mode=mode, muq_unavailable=muq_unavailable):
            cfg = self.config.mode_config(mode)
            ref_final = float(cfg.get("min_verified_reference_final_delta_when_muq_unavailable", 0.001))
            ref_musical = float(cfg.get("min_verified_reference_musical_delta_when_muq_unavailable", 0.001))
            vs_no_change = item.final_score - baseline.final_score >= ref_final and item.musical_score - baseline.musical_score >= ref_musical
        candidate_type = self._candidate_type(item)
        if reference is None or item.candidate.name == reference.candidate.name:
            vs_reference = True
        elif candidate_type == "reference_taste_guitar_control":
            eps = float(self.config.mode_config(mode).get("reference_tiebreak_final_epsilon", 0.005))
            guitar = dict(item.render_verification.get("guitar_forwardness", {}) or {})
            vs_reference = bool(
                self._guitar_control_can_beat_reference(item, reference, mode)
                and item.final_score >= reference.final_score - eps
                and item.musical_score >= reference.musical_score - eps
                and bool(guitar.get("reference_preserved"))
            )
        else:
            margin = float(self.config.mode_config(mode).get("win_margin_over_reference_when_muq_unavailable", 0.010 if muq_unavailable else 0.005))
            vs_reference = item.final_score - reference.final_score >= margin and item.musical_score - reference.musical_score >= margin
        return replace(
            item,
            material_improvement=bool(vs_no_change and vs_reference),
            material_improvement_vs_no_change=bool(vs_no_change),
            material_improvement_vs_reference=bool(vs_reference),
        )

    def _reject_reasons(self, item: CandidateEvaluation, *, baseline: CandidateEvaluation, reference: CandidateEvaluation | None, mode: str) -> list[str]:
        del baseline
        reasons: list[str] = []
        candidate_type = self._candidate_type(item)
        if item.safety is None or not item.safety.passed:
            reasons.append("safety_governor_rejected")
        critical_safety_fix = self._critical_safety_fix(item)
        if not item.material_improvement_vs_no_change and not critical_safety_fix:
            reasons.append("review_required_no_material_improvement_vs_no_change")
        if reference is not None and candidate_type not in {"reference_taste", "reference_taste_guitar_control"} and not item.material_improvement_vs_reference:
            reasons.append("review_required_no_material_improvement_vs_reference")
        if candidate_type == "reference_taste_guitar_control" and not item.material_improvement_vs_reference:
            reasons.append("review_required_no_reference_preserved_guitar_masking_improvement")
        if self._muq_unavailable(item) and candidate_type == "large_remix":
            reasons.append("review_required_muq_excluded_large_remix")
        verification = dict(item.render_verification or {})
        if verification.get("pre_trim_headroom_rejection") == "review_required":
            reasons.append("review_required_internal_pre_trim_true_peak")
        if verification.get("pre_trim_headroom_rejection") == "hard_reject":
            reasons.append("internal_pre_trim_true_peak_hard_reject")
        if verification.get("metrics_conflicts"):
            reasons.append("metrics_conflict")
        if verification.get("overhead_parallel_compression_risk"):
            reasons.append("review_required_parallel_compression_on_overheads")
        if candidate_type == "drum_glue" and any(a.parameters.get("role") == "bass" for a in item.candidate.actions):
            reasons.append("drum_glue_contains_bass_requires_rhythm_section_glue")
        if candidate_type == "rhythm_section_glue" and not bool(verification.get("kick_bass_balance_ok", False)):
            reasons.append("rhythm_section_glue_requires_kick_bass_balance_ok")
        if dict(verification.get("guitar_forwardness", {}) or {}).get("guitar_power_loss_detected"):
            reasons.append("guitar_power_loss_detected")
        return list(dict.fromkeys(reasons))

    def _selection_score(self, item: CandidateEvaluation) -> float:
        penalty = 0.0
        penalty += len([a for a in item.candidate.actions if a.channel_id is not None and not a.parameters.get("utility_gain_staging")]) * 0.0005
        if self._candidate_type(item) == "reference_taste":
            penalty -= 0.002
        if self._candidate_type(item) == "reference_taste_guitar_control" and dict(item.render_verification.get("guitar_forwardness", {}) or {}).get("guitar_masking_improved"):
            penalty -= 0.0025
        return float(item.musical_score - penalty)

    @staticmethod
    def _critical_safety_fix(item: CandidateEvaluation) -> bool:
        verification = dict(item.render_verification or {})
        return bool(
            item.candidate.metadata.get("critical_safety_fix")
            and item.safety is not None
            and item.safety.passed
            and verification.get("pre_trim_headroom_rejection") == "none"
        )

    def _best_guitar_control(self, items: list[CandidateEvaluation]) -> CandidateEvaluation | None:
        controls = [item for item in items if self._candidate_type(item) == "reference_taste_guitar_control"]
        return max(controls, key=lambda item: item.final_score, default=None)

    def _guitar_control_can_beat_reference(self, item: CandidateEvaluation, reference: CandidateEvaluation | None, mode: str) -> bool:
        if reference is None or self._candidate_type(item) != "reference_taste_guitar_control":
            return False
        guitar = dict(item.render_verification.get("guitar_forwardness", {}) or {})
        eps = float(self.config.mode_config(mode).get("reference_tiebreak_final_epsilon", 0.005))
        return bool(
            guitar.get("guitar_masking_improved")
            and guitar.get("reference_preserved")
            and not guitar.get("guitar_power_loss_detected")
            and item.final_score >= reference.final_score - eps
            and item.musical_score >= reference.musical_score - eps
        )

    def _material_thresholds(self, mode: str, *, muq_unavailable: bool) -> tuple[float, float]:
        cfg = self.config.mode_config(mode)
        if mode == "offline" and muq_unavailable:
            return (
                float(cfg.get("min_material_final_delta_when_muq_unavailable", 0.010)),
                float(cfg.get("min_material_musical_delta_when_muq_unavailable", 0.020)),
            )
        threshold = self.config.min_score_improvement(mode)
        return threshold, threshold

    def _baseline(self, items: list[CandidateEvaluation]) -> CandidateEvaluation:
        return next((item for item in items if item.candidate.name == self.config.baseline_candidate_name), items[0])

    def _reference(self, items: list[CandidateEvaluation]) -> CandidateEvaluation | None:
        return next((item for item in items if item.candidate.name == self.config.accepted_reference_candidate_name), None)

    @staticmethod
    def _candidate_type(item: CandidateEvaluation) -> str:
        return str(item.candidate.metadata.get("candidate_type", "single_fix"))

    @staticmethod
    def _muq_unavailable(item: CandidateEvaluation) -> bool:
        joined = " ".join([*item.warnings, *item.proxy_evidence])
        return "optional_critics_disabled" in joined or "muq_eval_unavailable" in joined or "muq_eval_fallback_excluded" in joined

    def _uses_verified_reference_threshold(self, item: CandidateEvaluation, *, reference: CandidateEvaluation | None, mode: str, muq_unavailable: bool) -> bool:
        if mode != "offline" or not muq_unavailable or reference is None:
            return False
        if item.candidate.name != reference.candidate.name or self._candidate_type(item) != "reference_taste":
            return False
        if not bool(self.config.mode_config(mode).get("allow_small_improvements", False)):
            return False
        verification = dict(item.render_verification or {})
        return bool(
            verification.get("offline_fx_return_present")
            and verification.get("spatial_fx_verified")
            and verification.get("modulation_fx_verified")
            and verification.get("pre_trim_headroom_rejection") != "hard_reject"
            and not verification.get("metrics_conflicts")
        )


def is_utility_action(action: MixAction) -> bool:
    return action.stage == "final_safety_trim" or (action.action_type == ACTION_GAIN and action.channel_id is None)
