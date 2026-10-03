"""Safety Gate v2 for ActionPlans.

Safety Gate is the mandatory boundary before any v2 plan reaches an Executor.
It applies live-safe limits, rate limiting, hysteresis, cooldown and dry-run
semantics. It does not send OSC.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import time
from typing import Any, Dict, Mapping, Tuple

from automixer.decision.models import (
    ACTION_COMPRESSION,
    ACTION_EQ,
    ACTION_GAIN,
    ACTION_NO_ACTION,
    ACTION_PAN,
    RISK_CRITICAL,
    RISK_HIGH,
    ActionDecision,
    ActionPlan,
    jsonable,
)


@dataclass(frozen=True)
class SafetyGateConfig:
    """Configurable limits for the v2 Safety Gate."""

    max_gain_change_db: float = 1.0
    max_live_gain_increase_db: float = 0.5
    min_gain_change_db: float = 0.25
    max_eq_boost_db: float = 1.0
    max_eq_cut_db: float = 3.0
    max_eq_q: float = 4.0
    min_eq_frequency_hz: float = 20.0
    max_eq_frequency_hz: float = 20000.0
    max_pan_step: float = 0.15
    max_pan_abs: float = 1.0
    max_compression_ratio: float = 4.0
    max_compression_threshold_change_db: float = 6.0
    min_interval_sec: float = 3.0
    cooldown_sec: float = 2.0
    hysteresis_db: float = 0.25
    live_mode_blocks_abrupt_changes: bool = True
    true_peak_ceiling_dbtp: float = -1.0
    dry_run: bool = False
    emergency_bypass: bool = False

    @classmethod
    def from_mapping(cls, payload: Mapping[str, Any] | None) -> "SafetyGateConfig":
        payload = payload or {}
        return cls(
            max_gain_change_db=float(payload.get("max_gain_change_db", 1.0)),
            max_live_gain_increase_db=float(payload.get("max_live_gain_increase_db", 0.5)),
            min_gain_change_db=float(payload.get("min_gain_change_db", 0.25)),
            max_eq_boost_db=float(payload.get("max_eq_boost_db", 1.0)),
            max_eq_cut_db=float(payload.get("max_eq_cut_db", 3.0)),
            max_eq_q=float(payload.get("max_eq_q", 4.0)),
            min_eq_frequency_hz=float(payload.get("min_eq_frequency_hz", 20.0)),
            max_eq_frequency_hz=float(payload.get("max_eq_frequency_hz", 20000.0)),
            max_pan_step=float(payload.get("max_pan_step", 0.15)),
            max_pan_abs=float(payload.get("max_pan_abs", 1.0)),
            max_compression_ratio=float(payload.get("max_compression_ratio", 4.0)),
            max_compression_threshold_change_db=float(
                payload.get("max_compression_threshold_change_db", 6.0)
            ),
            min_interval_sec=float(payload.get("min_interval_sec", 3.0)),
            cooldown_sec=float(payload.get("cooldown_sec", 2.0)),
            hysteresis_db=float(payload.get("hysteresis_db", 0.25)),
            live_mode_blocks_abrupt_changes=bool(
                payload.get("live_mode_blocks_abrupt_changes", True)
            ),
            true_peak_ceiling_dbtp=float(payload.get("true_peak_ceiling_dbtp", -1.0)),
            dry_run=bool(payload.get("dry_run", False)),
            emergency_bypass=bool(payload.get("emergency_bypass", False)),
        )


@dataclass(frozen=True)
class SafetyGateResult:
    """Safety Gate output for one input plan."""

    original_plan: ActionPlan
    allowed_plan: ActionPlan
    blocked: list[Dict[str, Any]] = field(default_factory=list)
    dry_run: bool = False

    @property
    def allowed_count(self) -> int:
        return len([d for d in self.allowed_plan.decisions if d.action_type != ACTION_NO_ACTION])

    @property
    def blocked_count(self) -> int:
        return len(self.blocked)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "original_plan": self.original_plan.to_dict(),
            "allowed_plan": self.allowed_plan.to_dict(),
            "blocked": jsonable(self.blocked),
            "dry_run": self.dry_run,
            "allowed_count": self.allowed_count,
            "blocked_count": self.blocked_count,
        }


class SafetyGate:
    """Filter an ActionPlan before execution."""

    def __init__(
        self,
        config: SafetyGateConfig | Mapping[str, Any] | None = None,
        *,
        time_provider: Any | None = None,
    ):
        self.config = config if isinstance(config, SafetyGateConfig) else SafetyGateConfig.from_mapping(config)
        self.time_provider = time_provider or time.monotonic
        self._last_allowed_at: Dict[Tuple[str, str], float] = {}

    def evaluate_plan(
        self,
        plan: ActionPlan,
        *,
        current_state: Mapping[str, Any] | None = None,
        live_mode: bool = True,
        dry_run: bool | None = None,
    ) -> SafetyGateResult:
        """Return a gated plan and block reasons without sending anything."""
        now = float(self.time_provider())
        dry = self.config.dry_run if dry_run is None else bool(dry_run)
        current_state = current_state or {}
        allowed: list[ActionDecision] = []
        blocked: list[Dict[str, Any]] = []

        for decision in plan.decisions:
            checked, reason = self._evaluate_decision(
                decision,
                current_state=current_state,
                live_mode=live_mode,
                now=now,
                dry_run=dry,
            )
            if checked is None:
                blocked.append(
                    {
                        "id": decision.id,
                        "target": decision.target,
                        "action_type": decision.action_type,
                        "reason": reason,
                        "decision": decision.to_dict(),
                    }
                )
                continue
            allowed.append(checked)
            if checked.action_type != ACTION_NO_ACTION and not dry:
                self._last_allowed_at[(checked.target, checked.action_type)] = now

        allowed_plan = ActionPlan(
            plan_id=f"{plan.plan_id}.safety_gate",
            created_at=plan.created_at,
            mode=plan.mode,
            decisions=allowed,
            source_modules=[*plan.source_modules, "safety_gate_v2"],
            input_summary=plan.input_summary,
            notes=[
                *plan.notes,
                "Safety Gate v2 evaluated this plan before Executor.",
                "Dry-run mode prevented mixer writes." if dry else "Plan may be executed by Executor.",
            ],
        )
        return SafetyGateResult(
            original_plan=plan,
            allowed_plan=allowed_plan,
            blocked=blocked,
            dry_run=dry,
        )

    def _evaluate_decision(
        self,
        decision: ActionDecision,
        *,
        current_state: Mapping[str, Any],
        live_mode: bool,
        now: float,
        dry_run: bool,
    ) -> tuple[ActionDecision | None, str]:
        if decision.action_type == ACTION_NO_ACTION:
            return self._mark_allowed(decision, "no_action", dry_run), "no action"

        emergency = self.config.emergency_bypass or bool(decision.metadata.get("emergency", False))
        if not emergency:
            if not decision.safe_to_apply:
                return None, "decision_engine_marked_unsafe"
            if decision.risk_level in {RISK_HIGH, RISK_CRITICAL}:
                return None, f"risk_level_{decision.risk_level}"
            limited = self._rate_or_cooldown_limited(decision, now)
            if limited:
                return None, limited

        if decision.action_type == ACTION_GAIN:
            return self._check_gain(decision, current_state, live_mode, dry_run, emergency)
        if decision.action_type == ACTION_EQ:
            return self._check_eq(decision, dry_run, emergency)
        if decision.action_type == ACTION_COMPRESSION:
            return self._check_compression(decision, current_state, dry_run, emergency)
        if decision.action_type == ACTION_PAN:
            return self._check_pan(decision, current_state, live_mode, dry_run, emergency)
        return None, "unsupported_action_type"

    def _check_gain(
        self,
        decision: ActionDecision,
        current_state: Mapping[str, Any],
        live_mode: bool,
        dry_run: bool,
        emergency: bool,
    ) -> tuple[ActionDecision | None, str]:
        gain_db = _number(decision.parameters.get("gain_db"), 0.0)
        if not emergency and abs(gain_db) < max(self.config.hysteresis_db, self.config.min_gain_change_db):
            return None, "inside_gain_hysteresis"
        if not emergency and abs(gain_db) > self.config.max_gain_change_db:
            return None, "gain_step_exceeds_limit"
        if (
            not emergency
            and live_mode
            and self.config.live_mode_blocks_abrupt_changes
            and gain_db > self.config.max_live_gain_increase_db
        ):
            return None, "live_gain_increase_exceeds_limit"
        state = _target_state(current_state, decision.target)
        true_peak = _number(state.get("true_peak_dbtp"), None)
        if not emergency and gain_db > 0.0 and true_peak is not None:
            if true_peak + gain_db > self.config.true_peak_ceiling_dbtp:
                return None, "true_peak_headroom_would_be_violated"
        return self._mark_allowed(decision, "gain_allowed", dry_run), "gain allowed"

    def _check_eq(
        self,
        decision: ActionDecision,
        dry_run: bool,
        emergency: bool,
    ) -> tuple[ActionDecision | None, str]:
        params = dict(decision.parameters)
        gain_db = _number(params.get("gain_db"), 0.0)
        q = _number(params.get("q"), 1.0)
        freq = _number(params.get("frequency_hz"), 1000.0)
        if not emergency and gain_db > self.config.max_eq_boost_db:
            return None, "eq_boost_exceeds_limit"
        if not emergency and gain_db < -self.config.max_eq_cut_db:
            return None, "eq_cut_exceeds_limit"
        if not emergency and abs(gain_db) < self.config.hysteresis_db:
            return None, "inside_eq_hysteresis"
        if q > self.config.max_eq_q:
            return None, "eq_q_exceeds_limit"
        if not (self.config.min_eq_frequency_hz <= freq <= self.config.max_eq_frequency_hz):
            return None, "eq_frequency_out_of_range"
        return self._mark_allowed(decision, "eq_allowed", dry_run), "eq allowed"

    def _check_compression(
        self,
        decision: ActionDecision,
        current_state: Mapping[str, Any],
        dry_run: bool,
        emergency: bool,
    ) -> tuple[ActionDecision | None, str]:
        ratio = _number(decision.parameters.get("ratio"), 1.0)
        if not emergency and ratio > self.config.max_compression_ratio:
            return None, "compression_ratio_exceeds_limit"
        state = _target_state(current_state, decision.target)
        current_threshold = _number(state.get("compression_threshold_db"), None)
        requested = _number(decision.parameters.get("threshold_db"), None)
        if (
            not emergency
            and current_threshold is not None
            and requested is not None
            and abs(requested - current_threshold) > self.config.max_compression_threshold_change_db
        ):
            return None, "compression_threshold_change_exceeds_limit"
        return self._mark_allowed(decision, "compression_allowed", dry_run), "compression allowed"

    def _check_pan(
        self,
        decision: ActionDecision,
        current_state: Mapping[str, Any],
        live_mode: bool,
        dry_run: bool,
        emergency: bool,
    ) -> tuple[ActionDecision | None, str]:
        pan = _number(decision.parameters.get("pan"), 0.0)
        if abs(pan) > self.config.max_pan_abs:
            return None, "pan_out_of_range"
        state = _target_state(current_state, decision.target)
        current_pan = _number(state.get("pan"), None)
        if (
            not emergency
            and live_mode
            and current_pan is not None
            and abs(pan - current_pan) > self.config.max_pan_step
        ):
            return None, "pan_step_exceeds_limit"
        return self._mark_allowed(decision, "pan_allowed", dry_run), "pan allowed"

    def _rate_or_cooldown_limited(self, decision: ActionDecision, now: float) -> str:
        last = self._last_allowed_at.get((decision.target, decision.action_type))
        if last is None:
            return ""
        elapsed = now - last
        if elapsed < self.config.min_interval_sec:
            return "rate_limited"
        if elapsed < self.config.cooldown_sec:
            return "cooldown_active"
        return ""

    def _mark_allowed(
        self,
        decision: ActionDecision,
        reason: str,
        dry_run: bool,
    ) -> ActionDecision:
        metadata = {
            **decision.metadata,
            "safety_gate": {
                "allowed": True,
                "reason": reason,
                "dry_run": bool(dry_run),
                "send_to_executor": not bool(dry_run) and decision.action_type != ACTION_NO_ACTION,
            },
        }
        return decision.with_updates(safe_to_apply=True, metadata=metadata)


def _number(value: Any, default: float | None) -> float | None:
    try:
        if value is None:
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def _target_state(current_state: Mapping[str, Any], target: str) -> Dict[str, Any]:
    if target in current_state and isinstance(current_state[target], Mapping):
        return dict(current_state[target])
    if target.startswith("channel:"):
        channel_id = target.split(":", 1)[1]
        for key in (channel_id, int(channel_id) if channel_id.isdigit() else channel_id):
            item = current_state.get(key)
            if isinstance(item, Mapping):
                return dict(item)
    return {}
