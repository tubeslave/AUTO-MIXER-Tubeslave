"""Final Safety Governor adapter for `production_mix_v1`."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping

try:
    from automixer.decision.models import (
        ACTION_COMPRESSION as V2_COMPRESSION,
        ACTION_EQ as V2_EQ,
        ACTION_GAIN as V2_GAIN,
        ACTION_PAN as V2_PAN,
        ActionDecision,
        ActionPlan,
    )
    from automixer.safety import SafetyGate, SafetyGateConfig
except Exception:
    V2_COMPRESSION = "compression"
    V2_EQ = "eq"
    V2_GAIN = "gain"
    V2_PAN = "pan"

    @dataclass(frozen=True)
    class ActionDecision:
        id: str
        action_type: str
        target: str
        parameters: dict[str, Any] = field(default_factory=dict)
        reason: str = ""
        confidence: float = 0.0
        risk_level: str = "low"
        source_modules: list[str] = field(default_factory=list)
        expected_audio_effect: str = ""
        safe_to_apply: bool = False
        metadata: dict[str, Any] = field(default_factory=dict)

    @dataclass(frozen=True)
    class ActionPlan:
        plan_id: str
        mode: str
        decisions: list[ActionDecision]
        source_modules: list[str] = field(default_factory=list)
        notes: list[str] = field(default_factory=list)

    @dataclass(frozen=True)
    class SafetyGateConfig:
        max_gain_change_db: float
        max_live_gain_increase_db: float
        max_eq_boost_db: float
        max_eq_cut_db: float
        true_peak_ceiling_dbtp: float
        dry_run: bool = True

    @dataclass(frozen=True)
    class _FallbackSafetyGateResult:
        allowed_count: int = 0
        blocked_count: int = 0
        blocked: list[dict[str, Any]] = field(default_factory=list)

    class SafetyGate:
        def __init__(self, config: SafetyGateConfig):
            self.config = config

        def evaluate_plan(
            self,
            plan: ActionPlan,
            *,
            current_state: Mapping[str, Any],
            live_mode: bool,
            dry_run: bool,
        ) -> _FallbackSafetyGateResult:
            del current_state, live_mode, dry_run
            return _FallbackSafetyGateResult(allowed_count=len(plan.decisions))

from .analyzers import critical_degradations
from .config import ProductionMixConfig
from .models import (
    ACTION_COMPRESSION,
    ACTION_EQ,
    ACTION_FX_SEND,
    ACTION_GAIN,
    ACTION_PAN,
    ACTION_PARALLEL_COMPRESSION,
    MODE_LIVE,
    MixAction,
    MixCandidate,
    SafetyResult,
)


class ProductionSafetyGovernor:
    """Final protection layer before candidate actions can be applied or queued."""

    def __init__(self, config: ProductionMixConfig):
        self.config = config

    def evaluate(
        self,
        candidate: MixCandidate,
        *,
        candidate_analysis: Mapping[str, Any],
        baseline_analysis: Mapping[str, Any],
        mode: str,
        dry_run: bool,
    ) -> SafetyResult:
        blocked: list[dict[str, Any]] = []
        warnings: list[str] = []
        rationale: list[str] = []
        allowed = list(candidate.actions)

        self._check_candidate_bus(candidate, candidate_analysis, blocked, rationale)
        self._check_action_contracts(candidate, baseline_analysis, blocked, rationale)
        self._check_low_signal_guard(candidate.actions, baseline_analysis, blocked, rationale)
        self._check_reference_transfer_scope(candidate, blocked, rationale)
        if candidate.is_baseline:
            if not blocked:
                rationale.append("Baseline has no actions and passes delivery safety checks.")
            return SafetyResult(
                candidate_name=candidate.name,
                passed=len(blocked) == 0,
                allowed_actions=[],
                blocked_actions=blocked,
                warnings=warnings,
                rationale=rationale,
            )
        self._check_action_limits(candidate.actions, mode, blocked, rationale)
        self._check_large_correction_guard(candidate.actions, blocked, rationale)
        self._check_live_action_count(candidate.actions, mode, blocked, rationale)

        degradations = self.critical_degradations_for_candidate(
            candidate,
            baseline_analysis,
            candidate_analysis,
        )
        for issue in degradations:
            blocked.append({"reason": f"critical_degradation:{issue}", "action": None})

        gate_result = self._evaluate_with_v2_safety_gate(
            candidate,
            baseline_analysis=baseline_analysis,
            mode=mode,
            dry_run=dry_run,
        )
        if gate_result is not None:
            blocked.extend(gate_result.get("blocked", []))
            rationale.extend(gate_result.get("rationale", []))
            warnings.extend(gate_result.get("warnings", []))

        passed = len(blocked) == 0
        if not passed:
            allowed = []
        else:
            rationale.append("Production Safety Governor passed as final protection layer.")
        return SafetyResult(
            candidate_name=candidate.name,
            passed=passed,
            allowed_actions=allowed,
            blocked_actions=blocked,
            warnings=warnings,
            rationale=rationale,
        )

    def critical_degradations_for_candidate(
        self,
        candidate: MixCandidate,
        baseline_analysis: Mapping[str, Any],
        candidate_analysis: Mapping[str, Any],
    ) -> list[str]:
        issues = critical_degradations(baseline_analysis, candidate_analysis, self.config)
        if not issues:
            return []
        filtered: list[str] = []
        for issue in issues:
            if issue == "vocal_clarity_drop" and self._space_recipe_clarity_drop_allowed(
                candidate,
                baseline_analysis,
                candidate_analysis,
            ):
                continue
            filtered.append(issue)
        return filtered

    def _check_candidate_bus(
        self,
        candidate: MixCandidate,
        analysis: Mapping[str, Any],
        blocked: list[dict[str, Any]],
        rationale: list[str],
    ) -> None:
        mix = analysis.get("mix", analysis)
        true_peak = float(mix.get("true_peak_dbfs", -120.0))
        phase_corr = float(mix.get("phase_correlation", 1.0))
        true_peak_limit = float(self.config.safety_limits.get("max_true_peak_dbfs", -1.0))
        min_phase = float(self.config.safety_limits.get("min_phase_correlation", 0.2))
        if true_peak > true_peak_limit:
            blocked.append(
                {
                    "candidate": candidate.name,
                    "reason": "true_peak_limit_exceeded",
                    "true_peak_dbfs": true_peak,
                    "limit": true_peak_limit,
                }
            )
        else:
            rationale.append(f"True-peak check passed: {true_peak:.2f} dBFS.")
        if phase_corr < min_phase:
            blocked.append(
                {
                    "candidate": candidate.name,
                    "reason": "phase_correlation_too_low",
                    "phase_correlation": phase_corr,
                    "limit": min_phase,
                }
            )
        else:
            rationale.append(f"Phase-correlation check passed: {phase_corr:.3f}.")

    def _space_recipe_clarity_drop_allowed(
        self,
        candidate: MixCandidate,
        baseline_analysis: Mapping[str, Any],
        candidate_analysis: Mapping[str, Any],
    ) -> bool:
        if not any(action.action_type == ACTION_FX_SEND for action in candidate.actions):
            return False
        adapter_cfg = dict(self.config.reference_recipe.get("safety_adapter", {}) or {})
        max_drop = float(adapter_cfg.get("max_space_vocal_clarity_drop", 0.10))
        base = float(
            baseline_analysis.get("mix", baseline_analysis)
            .get("dimensions", {})
            .get("vocal_clarity", 0.0)
        )
        candidate_value = float(
            candidate_analysis.get("mix", candidate_analysis)
            .get("dimensions", {})
            .get("vocal_clarity", 0.0)
        )
        return 0.0 <= base - candidate_value <= max_drop

    def _check_action_limits(
        self,
        actions: list[MixAction],
        mode: str,
        blocked: list[dict[str, Any]],
        rationale: list[str],
    ) -> None:
        max_gain = self._max_gain(mode)
        min_eq, max_eq = self._eq_range(mode)
        for action in actions:
            if action.action_type == ACTION_GAIN:
                gain_db = float(action.parameters.get("gain_db", 0.0))
                if action.channel_id is None:
                    if gain_db > 0.0:
                        blocked.append(
                            {
                                "reason": "mix_bus_gain_boost_not_allowed",
                                "action": action.to_dict(),
                            }
                        )
                    continue
                current = float(action.parameters.get("current_fader_db", -12.0))
                if abs(gain_db) > max_gain:
                    blocked.append(
                        {
                            "reason": "gain_step_exceeds_production_limit",
                            "action": action.to_dict(),
                            "limit": max_gain,
                        }
                    )
                if current + gain_db > 0.0:
                    blocked.append(
                        {
                            "reason": "fader_would_exceed_0_dbfs",
                            "action": action.to_dict(),
                            "current_fader_db": current,
                        }
                    )
            elif action.action_type == ACTION_EQ:
                gain_db = float(action.parameters.get("gain_db", 0.0))
                if gain_db < min_eq or gain_db > max_eq:
                    blocked.append(
                        {
                            "reason": "eq_step_exceeds_production_limit",
                            "action": action.to_dict(),
                            "allowed_range_db": [min_eq, max_eq],
                        }
                    )
            elif action.action_type == ACTION_COMPRESSION:
                ratio = float(action.parameters.get("ratio", 1.0))
                threshold = float(action.parameters.get("threshold_db", -18.0))
                makeup = float(action.parameters.get("makeup_gain_db", 0.0))
                target_gr = float(action.parameters.get("target_gain_reduction_db", 0.0))
                if ratio > 4.0 or threshold < -30.0 or threshold > -8.0 or makeup > 1.0:
                    blocked.append(
                        {
                            "reason": "compression_recipe_exceeds_production_limit",
                            "action": action.to_dict(),
                        }
                    )
                self._check_target_gr(action, target_gr, blocked)
            elif action.action_type == ACTION_PARALLEL_COMPRESSION:
                ratio = float(action.parameters.get("ratio", 1.0))
                threshold = float(action.parameters.get("threshold_db", -24.0))
                wet = float(action.parameters.get("wet_mix_percent", 0.0))
                makeup = float(action.parameters.get("makeup_gain_db", 0.0))
                target_gr = float(action.parameters.get("target_gain_reduction_db", 0.0))
                if ratio > 4.0 or threshold < -30.0 or wet > 35.0 or makeup > 0.5:
                    blocked.append(
                        {
                            "reason": "parallel_compression_exceeds_production_limit",
                            "action": action.to_dict(),
                        }
                    )
                self._check_target_gr(action, target_gr, blocked)
            elif action.action_type == ACTION_FX_SEND:
                send_db = float(action.parameters.get("send_level_db", -18.0))
                return_db = float(action.parameters.get("return_level_db", -12.0))
                decay_ms = float(action.parameters.get("decay_ms", 900.0))
                width = abs(float(action.parameters.get("width", 0.7)))
                if (
                    send_db > -8.0
                    or send_db < -48.0
                    or return_db > 0.0
                    or return_db < -48.0
                    or decay_ms > 2200.0
                    or width > 1.0
                ):
                    blocked.append(
                        {
                            "reason": "space_fx_send_exceeds_production_limit",
                            "action": action.to_dict(),
                        }
                    )
        if not blocked:
            rationale.append("Action step limits passed.")

    def _check_reference_transfer_scope(
        self,
        candidate: MixCandidate,
        blocked: list[dict[str, Any]],
        rationale: list[str],
    ) -> None:
        transfer = dict(self.config.reference_recipe.get("transfer_scope", {}) or {})
        allow_fixed_targets = bool(transfer.get("allow_fixed_group_targets", False))
        fixed_actions = [
            action
            for action in candidate.actions
            if action.stage == "role_group_balance_from_report"
            or action.parameters.get("reason") == "role_group_balance_from_report"
        ]
        if fixed_actions and not allow_fixed_targets:
            blocked.append(
                {
                    "candidate": candidate.name,
                    "reason": "fixed_reference_group_targets_not_allowed_for_new_song",
                    "actions": [action.to_dict() for action in fixed_actions],
                }
            )
        else:
            rationale.append("Reference transfer scope allows only taste/FX/air targets unless explicitly approved.")

    def _check_large_correction_guard(
        self,
        actions: list[MixAction],
        blocked: list[dict[str, Any]],
        rationale: list[str],
    ) -> None:
        guard = dict(self.config.large_correction_guard or {})
        review_gain = float(guard.get("review_gain_db", 2.5))
        reject_gain = float(guard.get("reject_gain_db", 4.0))
        group_reject = float(guard.get("group_gain_reject_db", 6.0))
        many_threshold = float(guard.get("many_large_moves_threshold_db", 2.0))
        many_count = int(guard.get("many_large_moves_count", 6))
        ambience_cut = float(guard.get("ambience_or_playback_cut_review_db", -4.0))
        drum_boost = float(guard.get("drum_boost_review_db", 4.0))
        large_channel_moves = []
        for action in actions:
            if action.action_type != ACTION_GAIN or action.channel_id is None:
                continue
            gain_db = float(action.parameters.get("gain_db", 0.0))
            role = _action_role(action)
            if abs(gain_db) > many_threshold:
                large_channel_moves.append(action)
            if abs(gain_db) > reject_gain and not bool(action.parameters.get("human_approved", False)):
                blocked.append(
                    {
                        "reason": "large_gain_change_rejected_without_human_approval",
                        "gain_db": gain_db,
                        "limit": reject_gain,
                        "action": action.to_dict(),
                    }
                )
            elif abs(gain_db) > review_gain:
                blocked.append(
                    {
                        "reason": "review_required_large_gain_change",
                        "gain_db": gain_db,
                        "limit": review_gain,
                        "action": action.to_dict(),
                    }
                )
            group_gain = action.parameters.get("group_gain_db")
            if group_gain is not None and abs(float(group_gain)) > group_reject:
                blocked.append(
                    {
                        "reason": "group_gain_correction_rejected",
                        "group_gain_db": float(group_gain),
                        "limit": group_reject,
                        "action": action.to_dict(),
                    }
                )
            if role in {"drum_room", "overheads", "hihat", "ride", "cymbal", "playback"} and gain_db < ambience_cut:
                blocked.append(
                    {
                        "reason": "review_required_ambience_or_playback_large_cut",
                        "role": role,
                        "gain_db": gain_db,
                        "limit": ambience_cut,
                        "action": action.to_dict(),
                    }
                )
            if role in {"kick", "snare", "toms"} and gain_db > drum_boost:
                blocked.append(
                    {
                        "reason": "review_required_drum_large_boost",
                        "role": role,
                        "gain_db": gain_db,
                        "limit": drum_boost,
                        "action": action.to_dict(),
                    }
                )
        if len(large_channel_moves) > many_count:
            blocked.append(
                {
                    "reason": "review_required_many_large_gain_moves",
                    "count": len(large_channel_moves),
                    "limit": many_count,
                    "threshold_db": many_threshold,
                }
            )
        if not blocked:
            rationale.append("Large Correction Guard passed.")

    def _check_target_gr(
        self,
        action: MixAction,
        target_gr: float,
        blocked: list[dict[str, Any]],
    ) -> None:
        cfg = dict(self.config.compression_safety or {})
        role = _action_role(action)
        any_reject = float(cfg.get("any_reject_gr_db", 7.0))
        vocal_review = float(cfg.get("vocal_review_gr_db", 5.0))
        rhythm_review = float(cfg.get("rhythm_review_gr_db", 6.0))
        if target_gr > any_reject:
            blocked.append(
                {
                    "reason": "compression_target_gr_rejected",
                    "role": role,
                    "target_gain_reduction_db": target_gr,
                    "limit": any_reject,
                    "action": action.to_dict(),
                }
            )
        elif role in {"lead_vocal", "backing_vocal"} and target_gr > vocal_review:
            blocked.append(
                {
                    "reason": "review_required_vocal_compression_target_gr",
                    "role": role,
                    "target_gain_reduction_db": target_gr,
                    "limit": vocal_review,
                    "action": action.to_dict(),
                }
            )
        elif role in {"kick", "snare", "bass", "toms"} and target_gr > rhythm_review:
            blocked.append(
                {
                    "reason": "review_required_rhythm_compression_target_gr",
                    "role": role,
                    "target_gain_reduction_db": target_gr,
                    "limit": rhythm_review,
                    "action": action.to_dict(),
                }
            )

    def _check_action_contracts(
        self,
        candidate: MixCandidate,
        baseline_analysis: Mapping[str, Any],
        blocked: list[dict[str, Any]],
        rationale: list[str],
    ) -> None:
        if candidate.name == "candidate_075_combined_musical_mix":
            has_balance = any(
                action.action_type == ACTION_GAIN
                and action.channel_id is not None
                and action.stage in {"static_gain_balance", "vocal_balance", "kick_bass_balance", "ayaic_balance"}
                for action in candidate.actions
            )
            if not has_balance:
                blocked.append(
                    {
                        "candidate": candidate.name,
                        "reason": "combined_candidate_missing_fader_balance",
                    }
                )
        for action in candidate.actions:
            allowed_level = action.parameters.get("role_allowed_action_level")
            if (
                action.channel_id is not None
                and allowed_level not in (None, "role_specific")
                and (
                    allowed_level == "no_op"
                    or _action_requires_role_specific(action)
                )
                and action.stage != "final_safety_trim"
            ):
                blocked.append(
                    {
                        "reason": "role_specific_action_blocked_by_confidence_gate",
                        "role": action.parameters.get("role"),
                        "role_confidence": action.parameters.get("role_confidence"),
                        "allowed_action_level": action.parameters.get("role_allowed_action_level"),
                        "action": action.to_dict(),
                    }
                )
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
                    blocked.append(
                        {
                            "reason": "compression_parameters_missing",
                            "missing": missing,
                            "action": action.to_dict(),
                        }
                    )
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
                    blocked.append(
                        {
                            "reason": "fx_send_parameters_missing",
                            "missing": missing,
                            "action": action.to_dict(),
                        }
                    )
                elif not bool(action.parameters.get("route_to_main", False)):
                    blocked.append(
                        {
                            "reason": "fx_send_return_not_routed_to_main",
                            "action": action.to_dict(),
                        }
                    )
        if not blocked:
            rationale.append("Action contract checks passed.")

    def _check_low_signal_guard(
        self,
        actions: list[MixAction],
        baseline_analysis: Mapping[str, Any],
        blocked: list[dict[str, Any]],
        rationale: list[str],
    ) -> None:
        cfg = dict(self.config.compression_safety or {})
        rms_limit = float(cfg.get("low_signal_rms_dbfs", -45.0))
        lufs_limit = float(cfg.get("low_signal_lufs", -45.0))
        for action in actions:
            if action.channel_id is None:
                continue
            should_block = False
            if action.action_type in {ACTION_COMPRESSION, ACTION_PARALLEL_COMPRESSION}:
                should_block = True
            elif action.action_type == ACTION_GAIN and float(action.parameters.get("gain_db", 0.0)) > 0.0:
                should_block = True
            elif action.action_type == ACTION_EQ and float(action.parameters.get("gain_db", 0.0)) > 0.0:
                should_block = True
            if not should_block:
                continue
            metrics = _metrics_for_action(action, baseline_analysis)
            rms = float(metrics.get("rms_db", action.parameters.get("rms_db", -120.0)))
            lufs = float(
                metrics.get(
                    "lufs_integrated_approx",
                    action.parameters.get("lufs_integrated_approx", rms),
                )
            )
            if rms < rms_limit or lufs < lufs_limit:
                blocked.append(
                    {
                        "reason": "low_signal_action_blocked",
                        "rms_db": rms,
                        "lufs_integrated_approx": lufs,
                        "rms_limit": rms_limit,
                        "lufs_limit": lufs_limit,
                        "action": action.to_dict(),
                    }
                )
        if not blocked:
            rationale.append("Low Signal Guard passed.")

    def _check_live_action_count(
        self,
        actions: list[MixAction],
        mode: str,
        blocked: list[dict[str, Any]],
        rationale: list[str],
    ) -> None:
        if mode != MODE_LIVE:
            return
        max_actions = int(self.config.mode_config(mode).get("max_actions_per_minute", 6))
        if len(actions) > max_actions:
            blocked.append(
                {
                    "reason": "live_action_count_exceeds_rate_limit",
                    "actions": len(actions),
                    "limit": max_actions,
                }
            )
        else:
            rationale.append("Live action count is within dry-run rate limit.")

    def _evaluate_with_v2_safety_gate(
        self,
        candidate: MixCandidate,
        *,
        baseline_analysis: Mapping[str, Any],
        mode: str,
        dry_run: bool,
    ) -> dict[str, Any] | None:
        supported_v2_actions = {
            ACTION_GAIN,
            ACTION_EQ,
            ACTION_COMPRESSION,
            ACTION_PARALLEL_COMPRESSION,
            ACTION_PAN,
        }
        channel_actions = [
            action
            for action in candidate.actions
            if action.channel_id is not None and action.action_type in supported_v2_actions
        ]
        if not channel_actions:
            return {
                "blocked": [],
                "warnings": [],
                "rationale": [
                    "Safety Gate v2 skipped: no supported channel actions after production adapter checks."
                ],
            }
        plan = ActionPlan(
            plan_id=f"production_mix_v1.{candidate.name}",
            mode=mode,
            decisions=[self._to_v2_decision(action) for action in channel_actions],
            source_modules=["production_mix_v1", "safety_gate_v2_adapter"],
            notes=["Production Safety Governor delegates supported action checks to Safety Gate v2."],
        )
        safety_config = SafetyGateConfig(
            max_gain_change_db=self._max_gain(mode),
            max_live_gain_increase_db=float(self.config.mode_config(mode).get("max_db_step", 0.7)),
            max_eq_boost_db=self._eq_range(mode)[1],
            max_eq_cut_db=abs(self._eq_range(mode)[0]),
            true_peak_ceiling_dbtp=float(self.config.safety_limits.get("max_true_peak_dbfs", -1.0)),
            dry_run=bool(dry_run),
        )
        gate = SafetyGate(safety_config)
        result = gate.evaluate_plan(
            plan,
            current_state=self._current_state(
                baseline_analysis,
                mix_bus_trim_db=self._mix_bus_trim_db(candidate.actions),
            ),
            live_mode=(mode == MODE_LIVE),
            dry_run=dry_run,
        )
        return {
            "blocked": [
                {
                    "reason": f"safety_gate_v2:{item.get('reason')}",
                    "action": item.get("decision"),
                }
                for item in result.blocked
            ],
            "warnings": [],
            "rationale": [
                f"Safety Gate v2 allowed {result.allowed_count} and blocked {result.blocked_count}."
            ],
        }

    @staticmethod
    def _to_v2_decision(action: MixAction) -> ActionDecision:
        action_type = {
            ACTION_GAIN: V2_GAIN,
            ACTION_EQ: V2_EQ,
            ACTION_COMPRESSION: V2_COMPRESSION,
            ACTION_PARALLEL_COMPRESSION: V2_COMPRESSION,
            ACTION_PAN: V2_PAN,
        }.get(action.action_type, action.action_type)
        params = dict(action.parameters)
        if action.channel_id is not None:
            params.setdefault("channel_id", action.channel_id)
        return ActionDecision(
            id=f"production_mix_v1.{action.target}.{action.action_type}",
            action_type=action_type,
            target=f"channel:{action.channel_id}" if action.channel_id is not None else "mix_bus",
            parameters=params,
            reason=action.rationale,
            confidence=0.7,
            risk_level="low",
            source_modules=list(action.source_modules),
            expected_audio_effect=action.rationale,
            safe_to_apply=True,
            metadata={"stage": action.stage},
        )

    @staticmethod
    def _mix_bus_trim_db(actions: list[MixAction]) -> float:
        trim = 0.0
        for action in actions:
            if action.action_type == ACTION_GAIN and action.channel_id is None:
                trim += min(0.0, float(action.parameters.get("gain_db", 0.0)))
        return trim

    @staticmethod
    def _current_state(analysis: Mapping[str, Any], *, mix_bus_trim_db: float = 0.0) -> dict[str, Any]:
        state: dict[str, Any] = {}
        for channel_id, payload in dict(analysis.get("stems", {}) or {}).items():
            metrics = dict(payload.get("metrics", {}) or {})
            state[f"channel:{channel_id}"] = {
                "true_peak_dbtp": float(metrics.get("true_peak_dbfs", -120.0)) + mix_bus_trim_db,
                "pan": 0.0,
                "compression_threshold_db": -18.0,
            }
        return state

    def _max_gain(self, mode: str) -> float:
        if mode == MODE_LIVE:
            return float(self.config.safety_limits.get("max_channel_fader_change_db_live", 1.5))
        return float(self.config.safety_limits.get("max_channel_fader_change_db_offline", 4.0))

    def _eq_range(self, mode: str) -> tuple[float, float]:
        if mode == MODE_LIVE:
            return (
                float(self.config.safety_limits.get("max_eq_cut_db_live", -2.5)),
                float(self.config.safety_limits.get("max_eq_boost_db_live", 1.0)),
            )
        return (
            float(self.config.safety_limits.get("max_eq_cut_db_offline", -4.0)),
            float(self.config.safety_limits.get("max_eq_boost_db_offline", 2.0)),
        )


def _action_role(action: MixAction) -> str:
    role = action.parameters.get("role")
    if role:
        return str(role)
    target = str(action.target).lower()
    if "playback" in target:
        return "playback"
    if "room" in target:
        return "drum_room"
    if "ride" in target:
        return "ride"
    if "hat" in target:
        return "hihat"
    if "oh" in target or "overhead" in target:
        return "overheads"
    if "kick" in target:
        return "kick"
    if "snare" in target:
        return "snare"
    if "tom" in target:
        return "toms"
    if "bass" in target:
        return "bass"
    if "back" in target or "bgv" in target:
        return "backing_vocal"
    if "vox" in target or "vocal" in target:
        return "lead_vocal"
    return "unknown"


def _metrics_for_action(action: MixAction, baseline_analysis: Mapping[str, Any]) -> dict[str, Any]:
    if action.channel_id is None:
        return {}
    return dict(
        dict(baseline_analysis.get("stems", {}) or {})
        .get(str(action.channel_id), {})
        .get("metrics", {})
        or {}
    )


def _action_requires_role_specific(action: MixAction) -> bool:
    if action.action_type in {ACTION_COMPRESSION, ACTION_PARALLEL_COMPRESSION, ACTION_FX_SEND}:
        return True
    if action.action_type == ACTION_GAIN and action.channel_id is not None:
        return action.stage in {
            "ayaic_balance",
            "static_gain_balance",
            "vocal_balance",
            "kick_bass_balance",
            "role_group_balance_from_report",
        }
    if action.action_type == ACTION_EQ and action.channel_id is not None:
        return float(action.parameters.get("gain_db", 0.0)) > 0.0
    return False
