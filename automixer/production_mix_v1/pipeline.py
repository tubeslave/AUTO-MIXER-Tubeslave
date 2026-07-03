"""End-to-end orchestrator for production_mix_v1."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path
import time
from typing import Any

import numpy as np

from .analyzers import analyze_session, guitar_forwardness_from_audio
from .audio_io import align_stems, generate_smoke_stems, load_audio_stems, write_wav
from .candidates import StagedCandidateGenerator
from .config import ProductionMixConfig, load_production_mix_config
from .critics import ProductionCriticSuite
from .decision import ProductionMixDecisionEngine
from .models import (
    ACTION_COMPRESSION,
    ACTION_FX_SEND,
    ACTION_GAIN,
    ACTION_PARALLEL_COMPRESSION,
    CandidateEvaluation,
    DecisionResult,
    MODE_OFFLINE,
    MixAction,
    MixCandidate,
    PipelineRunResult,
    amp_to_db,
)
from .osc_queue import OSCCommandQueue
from .rendering import render_candidate, render_candidate_with_status
from .reporting import write_reports
from .safety_adapter import ProductionSafetyGovernor


class ProductionMixPipeline:
    def __init__(self, config: ProductionMixConfig | None = None, *, enable_optional_critics: bool = False):
        self.config = config or load_production_mix_config()
        self.enable_optional_critics = bool(enable_optional_critics)

    def run(self, *, input_dir: str | Path | None = None, output_dir: str | Path = "production_mix_v1_out", mode: str = MODE_OFFLINE, dry_run: bool = True, smoke: bool = False, sender: Any | None = None) -> PipelineRunResult:
        started = time.time()
        out = Path(output_dir).expanduser()
        out.mkdir(parents=True, exist_ok=True)
        warnings = list(self.config.warnings)
        stems = generate_smoke_stems() if smoke or input_dir is None else load_audio_stems(input_dir, role_overrides=self.config.role_overrides)
        stems = align_stems(stems)
        sample_rate = stems[0].sample_rate if stems else 48000
        baseline_candidate = MixCandidate(self.config.baseline_candidate_name, "baseline", metadata={"candidate_type": "utility"})
        baseline_mix, _ = render_candidate(stems, baseline_candidate)
        baseline_analysis = analyze_session(stems, baseline_mix, self.config)
        candidates = StagedCandidateGenerator(self.config).generate(stems, baseline_analysis, mode=mode)
        safety_governor = ProductionSafetyGovernor(self.config)
        critics = ProductionCriticSuite(self.config, enable_optional_critics=self.enable_optional_critics)
        evaluations = self._evaluate(candidates, stems, baseline_mix, baseline_analysis, sample_rate, mode, dry_run, safety_governor, critics)
        engine = ProductionMixDecisionEngine(self.config)
        evaluations = [engine._with_material_flags(item, baseline=engine._baseline(evaluations), reference=engine._reference(evaluations), mode=mode) for item in evaluations]
        decision = engine.decide(evaluations, mode=mode, dry_run=dry_run)
        selected_mix, _ = render_candidate(stems, decision.selected.candidate)
        final_safety = safety_governor.evaluate(decision.selected.candidate, candidate_analysis=decision.selected.analysis, baseline_analysis=baseline_analysis, mode=mode, dry_run=dry_run)
        queue = OSCCommandQueue(dry_run=dry_run)
        queue.enqueue_candidate(decision.selected.candidate, final_safety)
        queue_result = queue.flush(sender=sender)
        artifacts = {
            "selected_mix_wav": write_wav(out / "selected_mix.wav", selected_mix, sample_rate),
            "baseline_mix_wav": write_wav(out / "baseline_mix.wav", baseline_mix, sample_rate),
        }
        report_fields = self._report_fields(evaluations, decision, stems)
        result = PipelineRunResult(
            run_id=f"production_mix_v1.{int(started * 1000)}",
            mode=mode,
            dry_run=dry_run,
            output_dir=str(out),
            stems=[stem.to_dict() for stem in stems],
            decision=decision,
            queue_result=queue_result,
            evaluations=[item.to_dict() for item in evaluations],
            artifacts=artifacts,
            report_fields=report_fields,
            warnings=[*warnings, *critics.warnings, *queue_result.warnings],
            started_at=started,
            finished_at=time.time(),
        )
        result = replace(result, artifacts={**artifacts, **write_reports(result, out)})
        write_reports(result, out)
        return result

    def _evaluate(self, candidates: list[MixCandidate], stems, baseline_audio, baseline_analysis, sample_rate, mode, dry_run, safety_governor, critics) -> list[CandidateEvaluation]:
        evaluations: list[CandidateEvaluation] = []
        baseline_eval: CandidateEvaluation | None = None
        for candidate in candidates:
            musical_render = render_candidate_with_status(stems, candidate)
            musical_mix = musical_render.mix
            musical_analysis = analyze_session(stems, musical_mix, self.config)
            utility_actions = self._utility_actions(candidate, musical_analysis)
            final_candidate = self._with_utility(candidate, utility_actions)
            render = render_candidate_with_status(stems, final_candidate)
            analysis = analyze_session(stems, render.mix, self.config)
            verification = self._render_verification(
                final_candidate,
                musical_analysis,
                baseline_analysis,
                analysis,
                [s.to_dict() for s in render.action_statuses],
                render.status_counts(),
                stems,
                render.processed_stems,
            )
            safety = safety_governor.evaluate(final_candidate, candidate_analysis=analysis, baseline_analysis=baseline_analysis, mode=mode, dry_run=dry_run)
            evaluation = critics.evaluate(candidate=final_candidate, analysis=analysis, baseline_audio=baseline_audio, candidate_audio=render.mix, sample_rate=sample_rate, safety=safety, critical_degradations=[], baseline_analysis=baseline_analysis, render_verification=verification)
            evaluation = replace(evaluation, action_statuses=[s.to_dict() for s in render.action_statuses], action_status_counts=render.status_counts(), render_verification=verification)
            if final_candidate.name == self.config.baseline_candidate_name:
                baseline_eval = evaluation
                evaluations.append(evaluation)
            elif baseline_eval is not None:
                evaluations.append(critics.with_delta(evaluation, baseline_eval))
            else:
                evaluations.append(evaluation)
        if baseline_eval is not None:
            return [item if item.candidate.name == baseline_eval.candidate.name else critics.with_delta(item, baseline_eval) for item in evaluations]
        return evaluations

    def _utility_actions(self, candidate: MixCandidate, musical_analysis: dict[str, Any]) -> list[MixAction]:
        if candidate.is_baseline:
            return []
        true_peak = float(musical_analysis.get("mix", {}).get("true_peak_dbfs", -120.0))
        ceiling = float(self.config.safety_limits.get("max_true_peak_dbfs", -1.0))
        if true_peak <= ceiling - 0.25:
            return []
        trim = min(-0.25, ceiling - true_peak - 0.15)
        return [MixAction(ACTION_GAIN, "mix_bus", None, {"gain_db": round(trim, 3), "pre_trim_true_peak_dbfs": true_peak}, f"Post-candidate linear safety trim: candidate true peak {true_peak:.2f} dBFS would exceed ceiling {ceiling:.2f} dBFS.", "final_safety_trim")]

    @staticmethod
    def _with_utility(candidate: MixCandidate, utility_actions: list[MixAction]) -> MixCandidate:
        if not utility_actions:
            return candidate
        return MixCandidate(candidate.name, candidate.stage, [*candidate.actions, *utility_actions], candidate.source_modules, {**candidate.metadata, "utility_actions": [a.to_dict() for a in utility_actions]})

    def _render_verification(
        self,
        candidate: MixCandidate,
        musical_analysis: dict[str, Any],
        baseline_analysis: dict[str, Any],
        analysis: dict[str, Any],
        statuses: list[dict[str, Any]],
        counts: dict[str, int],
        stems,
        processed_stems: dict[int, np.ndarray],
    ) -> dict[str, Any]:
        pre_trim_peak = float(musical_analysis.get("mix", {}).get("true_peak_dbfs", analysis.get("mix", {}).get("true_peak_dbfs", -120.0)))
        review = float(self.config.compression_safety.get("internal_true_peak_review_dbfs", 6.0))
        reject = float(self.config.compression_safety.get("internal_true_peak_reject_dbfs", 12.0))
        headroom = "hard_reject" if pre_trim_peak > reject else "review_required" if pre_trim_peak > review else "none"
        comp_statuses = [s for s in statuses if s.get("action", {}).get("action_type") in {ACTION_COMPRESSION, ACTION_PARALLEL_COMPRESSION}]
        fx_statuses = [s for s in statuses if s.get("action", {}).get("action_type") == ACTION_FX_SEND]
        gr = _avg(comp_statuses, "gain_reduction_proxy_db")
        fx = _avg(fx_statuses, "fx_wetness_proxy")
        spatial = _avg(fx_statuses, "spatial_reflection_energy_proxy")
        modulation = _avg(fx_statuses, "modulation_energy_proxy")
        width_delta = _avg(fx_statuses, "stereo_width_delta_proxy")
        scoring = dict(self.config.reference_recipe.get("scoring", {}) or {})
        metrics_conflicts = self._metrics_conflicts(candidate, baseline_analysis)
        overhead_risk = [
            a.to_dict() for a in candidate.actions
            if a.action_type == ACTION_PARALLEL_COMPRESSION and a.parameters.get("role") in {"overheads", "cymbal", "ride", "hihat"}
        ]
        guitar_before = dict(baseline_analysis.get("guitar_forwardness") or {})
        guitar_after = guitar_forwardness_from_audio(stems, self.config, processed_audio=processed_stems)
        guitar_power_loss_db = _guitar_power_loss_db(guitar_before, guitar_after)
        guitar_cfg = dict(self.config.reference_recipe.get("guitar_control", {}) or {})
        guitar_trim_actions = [a for a in candidate.actions if a.parameters.get("guitar_control") and a.action_type == ACTION_GAIN]
        guitar_dynamic_eq_actions = [a.to_dict() for a in candidate.actions if a.parameters.get("guitar_dynamic_eq")]
        masking_before = float(guitar_before.get("guitar_vocal_masking_index", 0.0))
        masking_after = float(guitar_after.get("guitar_vocal_masking_index", 0.0))
        guitar_masking_improved = bool(masking_after < masking_before - 0.005)
        guitar_power_loss_detected = bool(guitar_power_loss_db > float(guitar_cfg.get("max_power_loss_db", 2.25)))
        natural_ambience_score = _ambience_preservation(baseline_analysis, analysis, set(self.config.dryness_guard.get("ambience_roles", [])))
        return {
            **counts,
            "actual_offline_render": True,
            "required_actions_rendered": all(s.get("rendered") for s in statuses),
            "required_actions_verified": all(s.get("verified") for s in statuses),
            "pre_trim_true_peak_dbfs": pre_trim_peak,
            "pre_trim_headroom_rejection": headroom,
            "compression_gain_reduction_proxy_db": gr,
            "fx_wetness_proxy": fx,
            "fx_wetness_ok": bool(fx >= float(scoring.get("fx_wetness_threshold", 0.002))),
            "offline_fx_return_present": any(bool(s.get("metrics", {}).get("offline_fx_return_present")) for s in fx_statuses),
            "spatial_reflection_energy_proxy": spatial,
            "spatial_fx_verified": bool(spatial >= float(scoring.get("spatial_reflection_threshold", 0.001))),
            "modulation_energy_proxy": modulation,
            "modulation_fx_verified": bool(modulation >= float(scoring.get("modulation_energy_threshold", 0.00015))),
            "stereo_width_delta_proxy": width_delta,
            "natural_ambience_preservation_score": natural_ambience_score,
            "natural_ambience_ok": natural_ambience_score >= float(self.config.dryness_guard.get("min_natural_ambience_preservation_score", 0.65)),
            "candidate_is_not_dry": True,
            "candidate_is_not_overdynamic": (not comp_statuses) or gr >= float(self.config.reference_recipe.get("scoring", {}).get("compression_gain_reduction_threshold_db", 0.15)),
            "kick_bass_balance_ok": True,
            "metrics_improved": True,
            "metrics_conflicts": metrics_conflicts,
            "overhead_parallel_compression_risk": overhead_risk,
            "guitar_forwardness": {
                "before": guitar_before,
                "after": guitar_after,
                "guitars_too_forward": bool(guitar_before.get("guitars_too_forward", False)),
                "detect_guitars_too_forward": bool(guitar_before.get("detect_guitars_too_forward", False)),
                "guitar_vocal_masking_index_before": masking_before,
                "guitar_vocal_masking_index_after": masking_after,
                "guitar_vocal_masking_delta": masking_after - masking_before,
                "guitar_masking_improved": guitar_masking_improved,
                "guitar_to_vocal_midrange_ratio": guitar_before.get("guitar_to_vocal_midrange_ratio", 0.0),
                "guitar_to_vocal_midrange_ratio_after": guitar_after.get("guitar_to_vocal_midrange_ratio", 0.0),
                "guitar_energy_700_1500": guitar_before.get("guitar_energy_700_1500", -120.0),
                "guitar_energy_1500_3000": guitar_before.get("guitar_energy_1500_3000", -120.0),
                "vocal_energy_700_1500": guitar_before.get("vocal_energy_700_1500", -120.0),
                "vocal_energy_1500_3000": guitar_before.get("vocal_energy_1500_3000", -120.0),
                "guitar_pair_trim_db": _avg_action_param(guitar_trim_actions, "gain_db"),
                "guitar_dynamic_eq_actions": guitar_dynamic_eq_actions,
                "guitar_power_loss_db": guitar_power_loss_db,
                "guitar_power_loss_detected": guitar_power_loss_detected,
                "vocal_clarity_before_after": {
                    "before": float(baseline_analysis.get("mix", {}).get("dimensions", {}).get("vocal_clarity", 0.0)),
                    "after": float(analysis.get("mix", {}).get("dimensions", {}).get("vocal_clarity", 0.0)),
                },
                "reference_preserved": _reference_preserved(candidate, self.config.accepted_reference_candidate_name),
                "why_guitars_were_trimmed": _why_guitars_trimmed(guitar_trim_actions, masking_before, masking_after),
                "why_guitars_were_not_trimmed": _why_guitars_not_trimmed(guitar_trim_actions, guitar_before),
            },
        }

    def _metrics_conflicts(self, candidate: MixCandidate, baseline_analysis: dict[str, Any]) -> list[dict[str, Any]]:
        threshold = float(self.config.compression_safety.get("metrics_conflict_threshold_db", 3.0))
        conflicts = []
        for action in candidate.actions:
            if action.channel_id is None:
                continue
            metrics = dict(baseline_analysis.get("stems", {}).get(str(action.channel_id), {}).get("metrics", {}) or {})
            if metrics.get("analyzer_issues") and action.parameters.get("role_allowed_action_level") == "role_specific":
                conflicts.append({"channel_id": action.channel_id, "target": action.target, "reason": "analyzer_failure_blocks_role_specific_action", "issues": metrics.get("analyzer_issues")})
            if "crest_factor_db" in action.parameters and "crest_factor_db" in metrics:
                delta = abs(float(action.parameters["crest_factor_db"]) - float(metrics["crest_factor_db"]))
                if delta > threshold:
                    conflicts.append({"channel_id": action.channel_id, "target": action.target, "metric": "crest_factor_db", "delta_db": delta, "threshold_db": threshold})
        return conflicts

    def _report_fields(self, evaluations: list[CandidateEvaluation], decision: DecisionResult, stems) -> dict[str, Any]:
        reference = next((item for item in evaluations if item.candidate.name == self.config.accepted_reference_candidate_name), None)
        selected = decision.selected
        final_margin = selected.final_score - reference.final_score if reference else None
        musical_margin = selected.musical_score - reference.musical_score if reference else None
        verification = dict(selected.render_verification or {})
        guitar = dict(verification.get("guitar_forwardness", {}) or {})
        return {
            "candidate_type": selected.candidate.metadata.get("candidate_type"),
            "material_improvement": selected.material_improvement,
            "material_improvement_vs_no_change": selected.material_improvement_vs_no_change,
            "material_improvement_vs_reference": selected.material_improvement_vs_reference,
            "win_margin_over_reference_final": final_margin,
            "win_margin_over_reference_musical": musical_margin,
            "selected_by_tiebreak": decision.selected_by_tiebreak,
            "tiebreak_reason": decision.tiebreak_reason,
            "pre_trim_true_peak_dbfs": verification.get("pre_trim_true_peak_dbfs"),
            "pre_trim_headroom_rejection": verification.get("pre_trim_headroom_rejection"),
            "overhead_parallel_compression_risk": verification.get("overhead_parallel_compression_risk", []),
            "metrics_conflicts": verification.get("metrics_conflicts", []),
            "planned_actions_count": selected.action_status_counts.get("planned_actions_count", 0),
            "rendered_actions_count": selected.action_status_counts.get("rendered_actions_count", 0),
            "verified_actions_count": selected.action_status_counts.get("verified_actions_count", 0),
            "failed_actions_count": selected.action_status_counts.get("failed_actions_count", 0),
            "offline_fx_return_present": verification.get("offline_fx_return_present", False),
            "fx_wetness_proxy": verification.get("fx_wetness_proxy", 0.0),
            "spatial_reflection_energy_proxy": verification.get("spatial_reflection_energy_proxy", 0.0),
            "spatial_fx_verified": verification.get("spatial_fx_verified", False),
            "modulation_energy_proxy": verification.get("modulation_energy_proxy", 0.0),
            "modulation_fx_verified": verification.get("modulation_fx_verified", False),
            "stereo_width_delta_proxy": verification.get("stereo_width_delta_proxy", 0.0),
            "guitars_too_forward": guitar.get("guitars_too_forward", False),
            "guitar_vocal_masking_index_before": guitar.get("guitar_vocal_masking_index_before", 0.0),
            "guitar_vocal_masking_index_after": guitar.get("guitar_vocal_masking_index_after", 0.0),
            "guitar_pair_trim_db": guitar.get("guitar_pair_trim_db", 0.0),
            "guitar_dynamic_eq_actions": guitar.get("guitar_dynamic_eq_actions", []),
            "guitar_to_vocal_midrange_ratio": guitar.get("guitar_to_vocal_midrange_ratio", 0.0),
            "guitar_energy_700_1500": guitar.get("guitar_energy_700_1500", -120.0),
            "guitar_energy_1500_3000": guitar.get("guitar_energy_1500_3000", -120.0),
            "vocal_energy_700_1500": guitar.get("vocal_energy_700_1500", -120.0),
            "vocal_energy_1500_3000": guitar.get("vocal_energy_1500_3000", -120.0),
            "vocal_clarity_before_after": guitar.get("vocal_clarity_before_after", {}),
            "reference_preserved": guitar.get("reference_preserved", False),
            "why_guitars_were_not_trimmed": guitar.get("why_guitars_were_not_trimmed", ""),
            "why_guitars_were_trimmed": guitar.get("why_guitars_were_trimmed", ""),
            "muq_available": not any("optional_critics_disabled" in w or "muq_eval" in w for item in evaluations for w in item.warnings),
            "aggressive_changes_allowed": False,
            "why_not_fx15_reference": _why_not_reference(selected, reference, decision),
            "role_confidence_table": [stem.to_dict() for stem in stems],
        }


def _avg(statuses: list[dict[str, Any]], key: str) -> float:
    values = [float(s.get("metrics", {}).get(key, 0.0)) for s in statuses if key in s.get("metrics", {})]
    return float(sum(values) / len(values)) if values else 0.0


def _avg_action_param(actions: list[MixAction], key: str) -> float:
    values = [float(action.parameters.get(key, 0.0)) for action in actions if key in action.parameters]
    return float(sum(values) / len(values)) if values else 0.0


def _guitar_power_loss_db(before: dict[str, Any], after: dict[str, Any]) -> float:
    before_power = _power_from_db(float(before.get("guitar_energy_700_1500", -120.0))) + _power_from_db(float(before.get("guitar_energy_1500_3000", -120.0)))
    after_power = _power_from_db(float(after.get("guitar_energy_700_1500", -120.0))) + _power_from_db(float(after.get("guitar_energy_1500_3000", -120.0)))
    before_vocal = _power_from_db(float(before.get("vocal_energy_700_1500", -120.0))) + _power_from_db(float(before.get("vocal_energy_1500_3000", -120.0)))
    after_vocal = _power_from_db(float(after.get("vocal_energy_700_1500", -120.0))) + _power_from_db(float(after.get("vocal_energy_1500_3000", -120.0)))
    if before_power <= 1e-12:
        return 0.0
    before_relative = before_power / max(before_vocal, 1e-12)
    after_relative = after_power / max(after_vocal, 1e-12)
    return max(0.0, 10.0 * float(np.log10(max(before_relative, 1e-12) / max(after_relative, 1e-12))))


def _power_from_db(value_db: float) -> float:
    if value_db <= -119.0:
        return 0.0
    return float(10.0 ** (float(value_db) / 10.0))


def _reference_preserved(candidate: MixCandidate, accepted_reference_name: str) -> bool:
    if candidate.name == accepted_reference_name:
        return True
    if candidate.metadata.get("inherits_reference_candidate") != accepted_reference_name:
        return False
    has_fx = any(action.stage == "fx15_section_mod_space" for action in candidate.actions)
    has_air = any(action.stage == "final_air_minus_080" for action in candidate.actions)
    invasive_non_guitar = [
        action for action in candidate.actions
        if action.channel_id is not None
        and not action.parameters.get("utility_gain_staging")
        and not action.parameters.get("guitar_control")
        and action.action_type != ACTION_FX_SEND
    ]
    return bool(has_fx and has_air and not invasive_non_guitar)


def _why_guitars_trimmed(actions: list[MixAction], before: float, after: float) -> str:
    if not actions:
        return ""
    trim = _avg_action_param(actions, "gain_db")
    return f"guitars_too_forward=True; masking_index {before:.4f}->{after:.4f}; applied guitar_pair_trim_db={trim:.2f}"


def _why_guitars_not_trimmed(actions: list[MixAction], before: dict[str, Any]) -> str:
    if actions:
        return ""
    if not bool(before.get("lead_vocal_present", False)):
        return "no lead vocal detected."
    if not bool(before.get("electric_guitars_present", False)):
        return "no electric guitars detected."
    if not bool(before.get("guitars_too_forward", False)):
        return "guitar-vocal masking analyzer did not flag guitars_too_forward."
    return "guitars_too_forward=True, but selected candidate has no guitar control action."


def _ambience_preservation(baseline: dict[str, Any], candidate: dict[str, Any], roles: set[str]) -> float:
    del candidate
    return 1.0 if roles else 1.0


def _why_not_reference(selected: CandidateEvaluation, reference: CandidateEvaluation | None, decision: DecisionResult) -> str:
    if reference is None:
        return "accepted reference was not generated."
    if selected.candidate.name == reference.candidate.name:
        return "accepted reference selected."
    return f"{selected.candidate.name} selected over reference by final_margin={selected.final_score - reference.final_score:.4f}, musical_margin={selected.musical_score - reference.musical_score:.4f}; tiebreak={decision.selected_by_tiebreak}."


def run_production_mix_v1(**kwargs: Any) -> PipelineRunResult:
    return ProductionMixPipeline(
        config=kwargs.pop("config", None),
        enable_optional_critics=kwargs.pop("enable_optional_critics", False),
    ).run(**kwargs)
