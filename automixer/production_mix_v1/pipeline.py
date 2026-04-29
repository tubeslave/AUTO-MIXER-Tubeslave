"""Orchestrator for the opt-in `production_mix_v1` pipeline."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path
import time
from typing import Any

import numpy as np

from .analyzers import analyze_session
from .audio_io import align_stems, generate_smoke_stems, load_audio_stems, write_wav
from .candidates import StagedCandidateGenerator
from .config import ProductionMixConfig, load_production_mix_config
from .critics import ProductionCriticSuite
from .decision import ProductionMixDecisionEngine
from .models import (
    ACTION_GAIN,
    ACTION_COMPRESSION,
    ACTION_FX_SEND,
    ACTION_PARALLEL_COMPRESSION,
    CandidateEvaluation,
    MODE_OFFLINE,
    DecisionResult,
    MixAction,
    MixCandidate,
    PipelineRunResult,
)
from .osc_queue import OSCCommandQueue
from .rendering import render_candidate, render_candidate_with_status
from .reporting import write_reports
from .safety_adapter import ProductionSafetyGovernor


class ProductionMixPipeline:
    """End-to-end production mix v1 dry-run/offline pipeline."""

    def __init__(
        self,
        config: ProductionMixConfig | None = None,
        *,
        enable_optional_critics: bool = True,
    ):
        self.config = config or load_production_mix_config()
        self.enable_optional_critics = bool(enable_optional_critics)

    def run(
        self,
        *,
        input_dir: str | Path | None = None,
        output_dir: str | Path = "production_mix_v1_out",
        mode: str = MODE_OFFLINE,
        dry_run: bool = True,
        smoke: bool = False,
        sender: Any | None = None,
    ) -> PipelineRunResult:
        """Run a candidate pass and write reports/artifacts."""
        started_at = time.time()
        output_path = Path(output_dir).expanduser()
        output_path.mkdir(parents=True, exist_ok=True)
        warnings = list(self.config.warnings)

        stems = self._load_stems(input_dir=input_dir, smoke=smoke, warnings=warnings)
        if not stems:
            stems = generate_smoke_stems()
            warnings.append("no_input_stems_generated_smoke_audio")
        stems = align_stems(stems)
        sample_rate = int(stems[0].sample_rate) if stems else 48000

        baseline_candidate = StagedCandidateGenerator(self.config).generate(
            stems,
            {"mix": {}, "stems": {}},
            mode=mode,
        )[0]
        baseline_mix, _baseline_processed = render_candidate(stems, baseline_candidate)
        baseline_analysis = analyze_session(stems, baseline_mix, self.config)

        generator = StagedCandidateGenerator(self.config)
        candidates = generator.generate(stems, baseline_analysis, mode=mode)
        safety_governor = ProductionSafetyGovernor(self.config)
        critics = ProductionCriticSuite(
            self.config,
            enable_optional_critics=self.enable_optional_critics,
        )
        evaluations = self._evaluate_candidates(
            stems=stems,
            candidates=candidates,
            baseline_audio=baseline_mix,
            baseline_analysis=baseline_analysis,
            sample_rate=sample_rate,
            mode=mode,
            dry_run=dry_run,
            safety_governor=safety_governor,
            critics=critics,
        )

        decision_engine = ProductionMixDecisionEngine(self.config)
        evaluations = [
            replace(
                evaluation,
                material_improvement=decision_engine._material_improvement(evaluation, mode=mode),
            )
            for evaluation in evaluations
        ]
        decision = decision_engine.decide(
            evaluations,
            mode=mode,
            dry_run=dry_run,
        )
        selected_mix, _selected_processed = render_candidate(stems, decision.selected.candidate)
        final_safety = safety_governor.evaluate(
            decision.selected.candidate,
            candidate_analysis=decision.selected.analysis,
            baseline_analysis=baseline_analysis,
            mode=mode,
            dry_run=dry_run,
        )
        if decision.accepted and not final_safety.passed:
            warnings.append("final_safety_governor_rejected_selected_candidate_using_baseline")
            decision = DecisionResult(
                selected=decision.baseline,
                baseline=decision.baseline,
                accepted=False,
                decision_state="review_required",
                rejected=[
                    *decision.rejected,
                    {
                        "candidate": decision.selected.candidate.name,
                        "reasons": ["final_safety_governor_rejected"],
                    },
                ],
                rationale=[
                    *decision.rationale,
                    "Final Safety Governor rejected selected candidate; reverted to baseline.",
                ],
            )
            selected_mix = baseline_mix
            final_safety = safety_governor.evaluate(
                decision.selected.candidate,
                candidate_analysis=decision.selected.analysis,
                baseline_analysis=baseline_analysis,
                mode=mode,
                dry_run=dry_run,
            )

        queue = OSCCommandQueue(dry_run=dry_run)
        queue.enqueue_candidate(decision.selected.candidate, final_safety)
        queue_result = queue.flush(sender=sender)
        evaluation_warnings = sorted(
            {
                warning
                for evaluation in evaluations
                for warning in evaluation.warnings
            }
        )
        report_fields = self._report_fields(evaluations, decision, stems)

        artifacts = {
            "selected_mix_wav": write_wav(output_path / "selected_mix.wav", selected_mix, sample_rate),
            "baseline_mix_wav": write_wav(output_path / "baseline_mix.wav", baseline_mix, sample_rate),
        }

        result = PipelineRunResult(
            run_id=f"production_mix_v1.{int(started_at * 1000)}",
            mode=mode,
            dry_run=dry_run,
            output_dir=str(output_path),
            stems=[stem.to_dict() for stem in stems],
            decision=decision,
            queue_result=queue_result,
            evaluations=[evaluation.to_dict() for evaluation in evaluations],
            artifacts=artifacts,
            report_fields=report_fields,
            warnings=[*warnings, *evaluation_warnings, *queue_result.warnings],
            started_at=started_at,
            finished_at=time.time(),
        )
        report_artifacts = write_reports(result, output_path)
        result = replace(result, artifacts={**artifacts, **report_artifacts})
        write_reports(result, output_path)
        return result

    def _load_stems(
        self,
        *,
        input_dir: str | Path | None,
        smoke: bool,
        warnings: list[str],
    ):
        if smoke or input_dir is None:
            return generate_smoke_stems()
        try:
            return load_audio_stems(input_dir, role_overrides=self.config.role_overrides)
        except Exception as exc:
            warnings.append(f"input_load_failed:{exc}")
            return []

    def _evaluate_candidates(
        self,
        *,
        stems,
        candidates,
        baseline_audio,
        baseline_analysis,
        sample_rate,
        mode,
        dry_run,
        safety_governor: ProductionSafetyGovernor,
        critics: ProductionCriticSuite,
    ) -> list[CandidateEvaluation]:
        evaluations: list[CandidateEvaluation] = []
        baseline_evaluation: CandidateEvaluation | None = None
        evaluated_names: set[str] = set()

        for candidate in candidates:
            if candidate.name in evaluated_names:
                continue
            evaluated_names.add(candidate.name)
            musical_render = render_candidate_with_status(stems, candidate)
            musical_mix = musical_render.mix
            musical_processed = musical_render.processed_stems
            musical_rendered_stems = [
                replace(stem, audio=musical_processed.get(stem.channel_id, stem.audio))
                for stem in stems
            ]
            musical_analysis = analyze_session(musical_rendered_stems, musical_mix, self.config)
            utility_actions = self._utility_actions_for_candidate(candidate, musical_analysis)
            final_candidate = self._with_utility_actions(candidate, utility_actions)
            render_result = render_candidate_with_status(stems, final_candidate)
            mix = render_result.mix
            processed = render_result.processed_stems
            rendered_stems = [
                replace(stem, audio=processed.get(stem.channel_id, stem.audio))
                for stem in stems
            ]
            analysis = analyze_session(rendered_stems, mix, self.config)
            render_verification = self._render_verification(
                final_candidate=final_candidate,
                stems=stems,
                baseline_analysis=baseline_analysis,
                candidate_analysis=analysis,
                action_statuses=[status.to_dict() for status in render_result.action_statuses],
                status_counts=render_result.status_counts(),
                mode=mode,
                dry_run=dry_run,
            )
            safety = safety_governor.evaluate(
                final_candidate,
                candidate_analysis=analysis,
                baseline_analysis=baseline_analysis,
                mode=mode,
                dry_run=dry_run,
            )
            degradations = safety_governor.critical_degradations_for_candidate(
                final_candidate,
                baseline_analysis,
                analysis,
            )
            evaluation = critics.evaluate(
                candidate=final_candidate,
                analysis=analysis,
                baseline_audio=baseline_audio,
                candidate_audio=mix,
                sample_rate=sample_rate,
                safety=safety,
                critical_degradations=degradations,
                baseline_analysis=baseline_analysis,
                musical_analysis=musical_analysis,
                render_verification=render_verification,
            )
            evaluation = replace(
                evaluation,
                action_statuses=[status.to_dict() for status in render_result.action_statuses],
                action_status_counts=render_result.status_counts(),
                render_verification=render_verification,
            )
            if candidate.name == self.config.baseline_candidate_name:
                baseline_evaluation = evaluation
                evaluations.append(evaluation)
                continue
            if baseline_evaluation is not None:
                evaluation = critics.with_delta(evaluation, baseline_evaluation)
            evaluations.append(evaluation)

        if baseline_evaluation is not None:
            return [
                item if item.candidate.name == baseline_evaluation.candidate.name
                else critics.with_delta(item, baseline_evaluation)
                for item in evaluations
            ]
        return evaluations

    def _utility_actions_for_candidate(
        self,
        candidate: MixCandidate,
        analysis,
    ) -> list[MixAction]:
        if any(action.stage == "final_safety_trim" for action in candidate.actions):
            return []
        mix = analysis.get("mix", analysis)
        true_peak = float(mix.get("true_peak_dbfs", -120.0))
        ceiling = float(self.config.safety_limits.get("max_true_peak_dbfs", -1.0))
        if true_peak <= ceiling - 0.25:
            return []
        trim_db = min(-0.25, ceiling - true_peak - 0.15)
        action = MixAction(
            action_type=ACTION_GAIN,
            target="mix_bus",
            channel_id=None,
            parameters={
                "gain_db": round(trim_db, 3),
                "true_peak_before_dbfs": round(true_peak, 3),
            },
            rationale=(
                f"Post-candidate linear safety trim: candidate true peak {true_peak:.2f} dBFS "
                f"would exceed ceiling {ceiling:.2f} dBFS."
            ),
            stage="final_safety_trim",
            source_modules=["production_mix_v1.safety_trim"],
        )
        return [action]

    @staticmethod
    def _with_utility_actions(
        candidate: MixCandidate,
        utility_actions: list[MixAction],
    ) -> MixCandidate:
        if not utility_actions:
            return candidate
        return MixCandidate(
            name=candidate.name,
            stage=candidate.stage,
            actions=[*candidate.actions, *utility_actions],
            source_modules=[*candidate.source_modules, "production_mix_v1.safety_trim"],
            metadata={
                **candidate.metadata,
                "utility_actions": [action.to_dict() for action in utility_actions],
                "post_candidate_safety_trim_db": utility_actions[0].parameters.get("gain_db"),
            },
        )

    def _render_verification(
        self,
        *,
        final_candidate: MixCandidate,
        stems,
        baseline_analysis: dict[str, Any],
        candidate_analysis: dict[str, Any],
        action_statuses: list[dict[str, Any]],
        status_counts: dict[str, int],
        mode: str,
        dry_run: bool,
    ) -> dict[str, Any]:
        del mode
        action_types = [action.action_type for action in final_candidate.actions]
        has_space_fx = ACTION_FX_SEND in action_types
        has_compression = any(
            action.action_type in {ACTION_COMPRESSION, ACTION_PARALLEL_COMPRESSION}
            for action in final_candidate.actions
        )
        has_gain_balance = any(
            action.action_type == ACTION_GAIN
            and action.channel_id is not None
            and action.stage in {"static_gain_balance", "vocal_balance", "kick_bass_balance", "ayaic_balance"}
            for action in final_candidate.actions
        )
        fx_statuses = [
            item for item in action_statuses if item.get("action", {}).get("action_type") == ACTION_FX_SEND
        ]
        compression_statuses = [
            item
            for item in action_statuses
            if item.get("action", {}).get("action_type") in {ACTION_COMPRESSION, ACTION_PARALLEL_COMPRESSION}
        ]
        fx_wetness = _average_metric(fx_statuses, "fx_wetness_proxy")
        gain_reduction = _average_metric(compression_statuses, "gain_reduction_proxy_db")
        before = self._balance_metrics(baseline_analysis)
        after = self._balance_metrics(candidate_analysis)
        balance_before_after = {
            key: {
                "before": before.get(key),
                "after": after.get(key),
                "delta": (
                    float(after.get(key, 0.0)) - float(before.get(key, 0.0))
                    if before.get(key) is not None and after.get(key) is not None
                    else None
                ),
            }
            for key in sorted(set(before) | set(after))
        }
        fx_threshold = float(
            self.config.reference_recipe.get("scoring", {}).get("fx_wetness_threshold", 0.002)
        )
        gr_threshold = float(
            self.config.reference_recipe.get("scoring", {}).get(
                "compression_gain_reduction_threshold_db",
                0.15,
            )
        )
        dryness_cfg = dict(self.config.dryness_guard or {})
        ambience_roles = set(dryness_cfg.get("ambience_roles", []))
        ambience_before = _role_energy_db(dict(baseline_analysis.get("stems", {}) or {}), ambience_roles)
        ambience_after = _role_energy_db(dict(candidate_analysis.get("stems", {}) or {}), ambience_roles)
        ambience_delta = (
            ambience_after - ambience_before
            if ambience_before is not None and ambience_after is not None
            else 0.0
        )
        ambience_cut_limit = float(dryness_cfg.get("ambience_cut_penalty_db", -4.0))
        natural_ambience_preservation_score = (
            1.0
            if ambience_before is None or ambience_after is None
            else max(0.0, min(1.0, 1.0 + min(0.0, ambience_delta) / 12.0))
        )
        min_ambience_score = float(
            dryness_cfg.get("min_natural_ambience_preservation_score", 0.65)
        )
        dryness_penalty = (
            max(0.0, abs(ambience_delta) - abs(ambience_cut_limit)) / 12.0
            if ambience_delta < ambience_cut_limit
            else 0.0
        )
        natural_ambience_destroyed = bool(ambience_delta < ambience_cut_limit)
        statuses_ok = all(
            item.get("status") in {"rendered", "verified"}
            for item in action_statuses
        )
        verified_required = self._required_actions_verified(action_statuses)
        fx_return_present = all(
            bool(item.get("metrics", {}).get("offline_fx_return_present", False))
            for item in fx_statuses
        ) if fx_statuses else True
        fx_wetness_ok = bool(has_space_fx and fx_wetness >= fx_threshold)
        natural_ambience_ok = bool(
            natural_ambience_preservation_score >= min_ambience_score
            and ambience_delta >= ambience_cut_limit
        )
        candidate_is_not_dry = bool((fx_wetness_ok or natural_ambience_ok) and not natural_ambience_destroyed)
        candidate_is_not_overdynamic = (not has_compression) or gain_reduction >= gr_threshold
        compression_gr_outliers = self._compression_gr_outliers(compression_statuses)
        metrics_conflicts = self._metrics_conflicts(
            final_candidate,
            stems,
            baseline_analysis,
        )
        large_gain_moves = self._large_gain_moves(final_candidate)
        large_cuts_to_room_overheads_cymbals = [
            item
            for item in large_gain_moves
            if item.get("gain_db", 0.0) < float(
                self.config.large_correction_guard.get("ambience_or_playback_cut_review_db", -4.0)
            )
            and item.get("role") in {"drum_room", "overheads", "hihat", "ride", "cymbal", "playback"}
        ]
        vocal_delta = float(
            balance_before_after.get("vocal_to_mix_ratio_db", {}).get("delta") or 0.0
        )
        stem_balance_delta = float(
            balance_before_after.get("stem_balance_score", {}).get("delta") or 0.0
        )
        kick_bass_delta = float(
            balance_before_after.get("kick_bass_balance_score", {}).get("delta") or 0.0
        )
        spectral_delta = float(
            balance_before_after.get("spectral_balance", {}).get("delta") or 0.0
        )
        crest_error_delta = float(
            balance_before_after.get("crest_target_error_db", {}).get("delta") or 0.0
        )
        vocal_balance_ok = vocal_delta >= -1.5
        stem_balance_ok = stem_balance_delta >= -0.08
        kick_bass_balance_ok = kick_bass_delta >= -0.10
        metrics_improved = (
            stem_balance_delta > 0.01
            or spectral_delta > 0.01
            or crest_error_delta < -0.20
            or (
                has_space_fx
                and candidate_is_not_dry
                and (fx_wetness_ok or natural_ambience_ok)
            )
            or (has_compression and candidate_is_not_overdynamic)
            or has_gain_balance
        )
        return {
            **status_counts,
            "dry_run": bool(dry_run),
            "actual_offline_render": True,
            "required_actions_rendered": bool(statuses_ok),
            "required_actions_verified": bool(statuses_ok and verified_required),
            "has_static_gain_balance": has_gain_balance,
            "has_space_fx": has_space_fx,
            "has_compression": has_compression,
            "fx_return_present": fx_return_present,
            "fx_wetness_proxy": fx_wetness,
            "fx_wetness_threshold": fx_threshold,
            "compression_gain_reduction_proxy_db": gain_reduction,
            "compression_gain_reduction_threshold_db": gr_threshold,
            "compression_gr_outliers": compression_gr_outliers,
            "metrics_conflicts": metrics_conflicts,
            "candidate_is_not_dry": candidate_is_not_dry,
            "fx_wetness_ok": fx_wetness_ok,
            "natural_ambience_ok": natural_ambience_ok,
            "natural_ambience_destroyed": natural_ambience_destroyed,
            "candidate_is_not_overdynamic": candidate_is_not_overdynamic,
            "natural_ambience_before_db": ambience_before,
            "natural_ambience_after_db": ambience_after,
            "natural_ambience_delta_db": ambience_delta,
            "natural_ambience_preservation_score": natural_ambience_preservation_score,
            "natural_ambience_preservation_min_score": min_ambience_score,
            "dryness_penalty": dryness_penalty,
            "large_gain_moves": large_gain_moves,
            "large_cuts_to_room_overheads_cymbals": large_cuts_to_room_overheads_cymbals,
            "vocal_balance_ok": vocal_balance_ok,
            "stem_balance_ok": stem_balance_ok,
            "kick_bass_balance_ok": kick_bass_balance_ok,
            "metrics_improved": metrics_improved,
            "balance_before_after": balance_before_after,
            "why_balance_was_changed": self._why_balance_was_changed(final_candidate),
            "why_candidate_is_not_dry": (
                (
                    f"No space-FX action is claimed by this candidate; natural ambience "
                    f"preservation {natural_ambience_preservation_score:.3f} "
                    f"{'>=' if natural_ambience_ok else '<'} {min_ambience_score:.3f}."
                )
                if not has_space_fx
                else (
                    f"FX wetness proxy {fx_wetness:.4f} "
                    f"{'>=' if fx_wetness_ok else '<'} threshold {fx_threshold:.4f}; "
                    f"natural ambience preservation {natural_ambience_preservation_score:.3f} "
                    f"{'>=' if natural_ambience_ok else '<'} {min_ambience_score:.3f}; "
                    f"candidate_is_not_dry={candidate_is_not_dry}."
                )
            ),
            "why_candidate_is_not_overdynamic": (
                "No compression action is claimed by this candidate; overdynamic review is not action-gated."
                if not has_compression
                else f"Measured compression GR proxy {gain_reduction:.3f} dB >= threshold {gr_threshold:.3f} dB."
                if candidate_is_not_overdynamic
                else f"Measured compression GR proxy {gain_reduction:.3f} dB below threshold {gr_threshold:.3f} dB."
            ),
        }

    @staticmethod
    def _balance_metrics(analysis: dict[str, Any]) -> dict[str, float | None]:
        mix = dict(analysis.get("mix", analysis) or {})
        stems = dict(analysis.get("stems", {}) or {})
        mix_rms = float(mix.get("rms_db", -120.0))
        vocal_rms = _role_energy_db(stems, {"lead_vocal", "backing_vocal"})
        kick_rms = _role_energy_db(stems, {"kick"})
        bass_rms = _role_energy_db(stems, {"bass"})
        stem_lufs = [
            float(payload.get("metrics", {}).get("lufs_integrated_approx", -120.0))
            for payload in stems.values()
            if float(payload.get("metrics", {}).get("lufs_integrated_approx", -120.0)) > -90.0
        ]
        stem_balance_score = None
        if len(stem_lufs) >= 2:
            stem_balance_score = max(0.0, min(1.0, 1.0 - float(np.std(stem_lufs)) / 18.0))
        kick_bass_balance = None
        if kick_rms is not None and bass_rms is not None:
            kick_bass_balance = max(0.0, min(1.0, 1.0 - abs(kick_rms - bass_rms) / 12.0))
        return {
            "lufs": float(mix.get("lufs_integrated_approx", mix_rms)),
            "rms_db": mix_rms,
            "peak_db": float(mix.get("peak_db", -120.0)),
            "true_peak_dbfs": float(mix.get("true_peak_dbfs", -120.0)),
            "crest_factor_db": float(mix.get("crest_factor_db", 0.0)),
            "crest_target_error_db": abs(float(mix.get("crest_factor_db", 0.0)) - 12.0),
            "vocal_to_mix_ratio_db": (
                vocal_rms - mix_rms if vocal_rms is not None and mix_rms > -119.0 else None
            ),
            "kick_bass_balance_score": kick_bass_balance,
            "stem_balance_score": stem_balance_score,
            "spectral_balance": float(mix.get("dimensions", {}).get("spectral_balance", 0.0)),
        }

    @staticmethod
    def _why_balance_was_changed(candidate: MixCandidate) -> list[str]:
        reasons = []
        for action in candidate.actions:
            if action.action_type != ACTION_GAIN or action.channel_id is None:
                continue
            if action.stage in {"static_gain_balance", "vocal_balance", "kick_bass_balance", "ayaic_balance"}:
                reasons.append(action.rationale)
        return reasons

    @staticmethod
    def _required_actions_verified(action_statuses: list[dict[str, Any]]) -> bool:
        for item in action_statuses:
            action = item.get("action", {})
            action_type = action.get("action_type")
            if action.get("stage") == "final_safety_trim":
                continue
            if action_type in {ACTION_COMPRESSION, ACTION_PARALLEL_COMPRESSION}:
                if item.get("status") not in {"rendered", "verified"}:
                    return False
                continue
            if not bool(item.get("verified", False)):
                return False
        return True

    def _compression_gr_outliers(self, compression_statuses: list[dict[str, Any]]) -> list[dict[str, Any]]:
        cfg = dict(self.config.compression_safety or {})
        any_reject = float(cfg.get("any_reject_gr_db", 7.0))
        vocal_review = float(cfg.get("vocal_review_gr_db", 5.0))
        rhythm_review = float(cfg.get("rhythm_review_gr_db", 6.0))
        outliers: list[dict[str, Any]] = []
        for item in compression_statuses:
            action = dict(item.get("action", {}) or {})
            params = dict(action.get("parameters", {}) or {})
            metrics = dict(item.get("metrics", {}) or {})
            role = str(params.get("role") or _role_from_target(str(action.get("target", ""))))
            gr = float(metrics.get("gain_reduction_proxy_db", 0.0))
            reason = ""
            if gr > any_reject:
                reason = "compression_gr_reject_outlier"
            elif role in {"lead_vocal", "backing_vocal"} and gr > vocal_review:
                reason = "review_required_vocal_compression_gr"
            elif role in {"kick", "snare", "bass", "toms"} and gr > rhythm_review:
                reason = "review_required_rhythm_compression_gr"
            if reason:
                outliers.append(
                    {
                        "reason": reason,
                        "role": role,
                        "target": action.get("target"),
                        "channel_id": action.get("channel_id"),
                        "gain_reduction_proxy_db": gr,
                    }
                )
        return outliers

    def _metrics_conflicts(
        self,
        candidate: MixCandidate,
        stems,
        baseline_analysis: dict[str, Any],
    ) -> list[dict[str, Any]]:
        threshold = float(
            dict(self.config.compression_safety or {}).get("metrics_conflict_threshold_db", 3.0)
        )
        stems_by_channel = {int(stem.channel_id): stem for stem in stems}
        analysis_stems = dict(baseline_analysis.get("stems", {}) or {})
        conflicts: list[dict[str, Any]] = []
        for action in candidate.actions:
            if action.channel_id is None:
                continue
            channel_id = int(action.channel_id)
            metrics = dict(analysis_stems.get(str(channel_id), {}).get("metrics", {}) or {})
            analysis_crest = metrics.get("crest_factor_db")
            if analysis_crest is None:
                continue
            analysis_crest = float(analysis_crest)
            action_crest = action.parameters.get("crest_factor_db")
            if action_crest is not None and abs(float(action_crest) - analysis_crest) > threshold:
                conflicts.append(
                    {
                        "channel_id": channel_id,
                        "target": action.target,
                        "metric": "crest_factor_db",
                        "analysis_value": analysis_crest,
                        "action_value": float(action_crest),
                        "delta_db": abs(float(action_crest) - analysis_crest),
                        "threshold_db": threshold,
                    }
                )
            stem = stems_by_channel.get(channel_id)
            role_crest = None
            if stem is not None and stem.role_detection is not None:
                role_crest = (
                    dict(stem.role_detection.evidence or {})
                    .get("spectral_features", {})
                    .get("crest_factor_db")
                )
            if role_crest is not None and abs(float(role_crest) - analysis_crest) > threshold:
                conflicts.append(
                    {
                        "channel_id": channel_id,
                        "target": action.target,
                        "metric": "crest_factor_db",
                        "analysis_value": analysis_crest,
                        "role_evidence_value": float(role_crest),
                        "delta_db": abs(float(role_crest) - analysis_crest),
                        "threshold_db": threshold,
                    }
                )
        return conflicts

    @staticmethod
    def _large_gain_moves(candidate: MixCandidate) -> list[dict[str, Any]]:
        result: list[dict[str, Any]] = []
        for action in candidate.actions:
            if action.action_type != ACTION_GAIN or action.channel_id is None:
                continue
            gain_db = float(action.parameters.get("gain_db", 0.0))
            if abs(gain_db) <= 2.0:
                continue
            result.append(
                {
                    "target": action.target,
                    "channel_id": action.channel_id,
                    "role": str(action.parameters.get("role") or _role_from_target(action.target)),
                    "gain_db": gain_db,
                    "stage": action.stage,
                }
            )
        return result

    def _report_fields(
        self,
        evaluations: list[CandidateEvaluation],
        decision: DecisionResult,
        stems,
    ) -> dict[str, Any]:
        selected_actions = list(decision.selected.candidate.actions)
        space_actions = [
            action.to_dict()
            for action in selected_actions
            if action.action_type == "fx_send"
        ]
        dynamics_actions = [
            action.to_dict()
            for action in selected_actions
            if action.action_type in {"compression", "parallel_compression"}
        ]
        utility_actions = [
            action.to_dict()
            for action in selected_actions
            if action.stage == "final_safety_trim"
        ]
        static_gain_actions = [
            action.to_dict()
            for action in selected_actions
            if action.action_type == "gain"
            and action.channel_id is not None
            and action.stage in {"static_gain_balance", "vocal_balance", "kick_bass_balance", "ayaic_balance"}
        ]
        compression_actions = [
            action for action in selected_actions if action.action_type in {"compression", "parallel_compression"}
        ]
        fx_actions = [action for action in selected_actions if action.action_type == "fx_send"]
        verification = dict(decision.selected.render_verification or {})
        status_counts = dict(decision.selected.action_status_counts or {})
        role_confidence_table = [
            {
                "channel_id": stem.channel_id,
                "name": stem.name,
                "role": stem.role,
                "confidence": (
                    stem.role_detection.confidence if stem.role_detection is not None else None
                ),
                "allowed_action_level": (
                    stem.role_detection.allowed_action_level if stem.role_detection is not None else None
                ),
                "is_ambiguous": (
                    stem.role_detection.is_ambiguous if stem.role_detection is not None else None
                ),
                "evidence": stem.role_detection.evidence if stem.role_detection is not None else {},
            }
            for stem in stems
        ]
        ambiguous_roles = [
            item for item in role_confidence_table
            if item.get("is_ambiguous") or item.get("allowed_action_level") != "role_specific"
        ]
        user_overrides_applied = [
            item for item in role_confidence_table
            if item.get("evidence", {}).get("user_override")
        ]
        muq_available = not any(
            "muq_eval_unavailable" in warning
            or "muq_eval_fallback_excluded" in warning
            or "optional_critics_disabled" in warning
            for evaluation in evaluations
            for warning in evaluation.warnings
        )
        aggressive_changes_allowed = bool(muq_available)
        return {
            "inherited_reference_recipe": self.config.reference_recipe.get(
                "inherited_reference_recipe",
                "",
            ),
            "planned_actions_count": status_counts.get("planned_actions_count", len(selected_actions)),
            "rendered_actions_count": status_counts.get("rendered_actions_count", 0),
            "verified_actions_count": status_counts.get("verified_actions_count", 0),
            "failed_actions_count": status_counts.get("failed_actions_count", 0),
            "action_statuses": decision.selected.action_statuses,
            "role_confidence_table": role_confidence_table,
            "ambiguous_roles": ambiguous_roles,
            "user_overrides_applied": user_overrides_applied,
            "blocked_role_specific_actions": [
                {
                    "channel_id": item.get("channel_id"),
                    "name": item.get("name"),
                    "role": item.get("role"),
                    "confidence": item.get("confidence"),
                    "allowed_action_level": item.get("allowed_action_level"),
                }
                for item in ambiguous_roles
            ],
            "static_gain_actions": static_gain_actions,
            "fader_balance_actions": static_gain_actions,
            "space_fx_actions": space_actions,
            "dynamics_actions": dynamics_actions,
            "utility_actions": utility_actions,
            "compression_parameters": [action.parameters for action in compression_actions],
            "expected_gain_reduction": [
                action.parameters.get("target_gain_reduction_db")
                for action in compression_actions
            ],
            "measured_gain_reduction_proxy": verification.get(
                "compression_gain_reduction_proxy_db",
                0.0,
            ),
            "fx_slot": [action.parameters.get("fx_slot") for action in fx_actions],
            "fx_model": [action.parameters.get("fx_model") for action in fx_actions],
            "fx_send_level": [action.parameters.get("send_level_db") for action in fx_actions],
            "fx_return_level": [action.parameters.get("return_level_db") for action in fx_actions],
            "fx_wetness_proxy": verification.get("fx_wetness_proxy", 0.0),
            "large_gain_moves": verification.get("large_gain_moves", []),
            "large_cuts_to_room_overheads_cymbals": verification.get(
                "large_cuts_to_room_overheads_cymbals",
                [],
            ),
            "compression_gr_outliers": verification.get("compression_gr_outliers", []),
            "metrics_conflicts": verification.get("metrics_conflicts", []),
            "dryness_penalty": verification.get("dryness_penalty", 0.0),
            "natural_ambience_preservation_score": verification.get(
                "natural_ambience_preservation_score",
                1.0,
            ),
            "fx_wetness_ok": verification.get("fx_wetness_ok", False),
            "natural_ambience_ok": verification.get("natural_ambience_ok", False),
            "natural_ambience_destroyed": verification.get("natural_ambience_destroyed", False),
            "candidate_is_not_dry": verification.get("candidate_is_not_dry", False),
            "metrics_improved": verification.get("metrics_improved", False),
            "material_improvement": decision.selected.material_improvement,
            "candidate_type": decision.selected.candidate.metadata.get("candidate_type", ""),
            "muq_available": muq_available,
            "aggressive_changes_allowed": aggressive_changes_allowed,
            "reason_if_aggressive_allowed": (
                "primary MuQ-Eval available"
                if aggressive_changes_allowed
                else "MuQ-Eval unavailable/fallback/excluded; large remix candidates require review"
            ),
            "balance_before_after": verification.get("balance_before_after", {}),
            "why_balance_was_changed": verification.get("why_balance_was_changed", []),
            "why_candidate_is_not_dry": verification.get("why_candidate_is_not_dry", ""),
            "why_candidate_is_not_overdynamic": verification.get(
                "why_candidate_is_not_overdynamic",
                "",
            ),
            "musical_score": decision.selected.musical_score,
            "technical_score": decision.selected.technical_safety_score,
            "final_score": decision.selected.final_score,
            "why_not_fx15_reference": self._why_not_fx15_reference(evaluations, decision),
        }

    @staticmethod
    def _why_not_fx15_reference(
        evaluations: list[CandidateEvaluation],
        decision: DecisionResult,
    ) -> str:
        target = "candidate_065_fx15_air_minus_080_reference"
        if decision.selected.candidate.name == target:
            return "candidate_065_fx15_air_minus_080_reference was selected."
        fx15 = next((item for item in evaluations if item.candidate.name == target), None)
        if fx15 is None:
            return "candidate_065_fx15_air_minus_080_reference was not generated."
        rejected = next(
            (
                item
                for item in decision.rejected
                if item.get("candidate") == target
            ),
            None,
        )
        if rejected:
            return "Rejected: " + ", ".join(str(reason) for reason in rejected.get("reasons", []))
        selected = decision.selected
        return (
            f"Selected {selected.candidate.name} because musical_score "
            f"{selected.musical_score:.4f} exceeded FX15 reference "
            f"{fx15.musical_score:.4f}."
        )


def _average_metric(statuses: list[dict[str, Any]], key: str) -> float:
    values = []
    for item in statuses:
        metrics = dict(item.get("metrics", {}) or {})
        if key in metrics:
            try:
                values.append(float(metrics[key]))
            except (TypeError, ValueError):
                continue
    if not values:
        return 0.0
    return float(sum(values) / len(values))


def _role_energy_db(stems: dict[str, Any], roles: set[str]) -> float | None:
    powers = []
    for payload in stems.values():
        if payload.get("role") not in roles:
            continue
        metrics = dict(payload.get("metrics", {}) or {})
        rms_db = float(metrics.get("rms_db", -120.0))
        if rms_db > -119.0:
            powers.append(10.0 ** (rms_db / 10.0))
    if not powers:
        return None
    return float(10.0 * np.log10(sum(powers) / len(powers) + 1e-12))


def _role_from_target(target: str) -> str:
    text = str(target).lower()
    if "playback" in text:
        return "playback"
    if "room" in text:
        return "drum_room"
    if "ride" in text:
        return "ride"
    if "hat" in text:
        return "hihat"
    if "oh" in text or "overhead" in text:
        return "overheads"
    if "kick" in text:
        return "kick"
    if "snare" in text:
        return "snare"
    if "tom" in text:
        return "toms"
    if "bass" in text:
        return "bass"
    if "back" in text or "bgv" in text:
        return "backing_vocal"
    if "vox" in text or "vocal" in text:
        return "lead_vocal"
    return "unknown"


def run_production_mix_v1(**kwargs: Any) -> PipelineRunResult:
    """Convenience wrapper for callers that do not need a pipeline instance."""
    return ProductionMixPipeline(
        config=kwargs.pop("config", None),
        enable_optional_critics=kwargs.pop("enable_optional_critics", True),
    ).run(**kwargs)
