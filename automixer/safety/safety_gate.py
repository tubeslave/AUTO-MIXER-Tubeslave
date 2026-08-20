"""Safety Gate v2 for ActionPlans.

Safety Gate is the mandatory boundary before any v2 plan reaches an Executor.
It applies live-safe limits, rate limiting, hysteresis, cooldown and dry-run
semantics. It does not send OSC.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import math
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
    # Anti-oscillation лимиты (0.0 = выключено, поведение как раньше):
    # max_rate_db_per_sec — скорость изменения в дБ/с (а не дБ/цикл: при
    # смене периода цикла лимит в дБ/цикл молча меняется);
    # correction_budget_db — суммарный |бюджет| коррекций на target в окне
    # budget_window_sec, при исчерпании — стоп до релаксации;
    # min_hold_sec — минимальное время удержания решения: реверс направления
    # (буст после ката и наоборот) блокируется, пока hold не истёк.
    max_rate_db_per_sec: float = 0.0
    correction_budget_db: float = 0.0
    budget_window_sec: float = 60.0
    min_hold_sec: float = 0.0

    def __post_init__(self) -> None:
        numeric_fields = {
            name: value
            for name, value in vars(self).items()
            if name not in {
                "live_mode_blocks_abrupt_changes",
                "dry_run",
                "emergency_bypass",
            }
        }
        for name, value in numeric_fields.items():
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise ValueError(f"{name} must be a finite numeric value")
            if not math.isfinite(float(value)):
                raise ValueError(f"{name} must be a finite numeric value")

        non_negative = (
            "max_gain_change_db",
            "max_live_gain_increase_db",
            "min_gain_change_db",
            "max_eq_boost_db",
            "max_eq_cut_db",
            "max_pan_step",
            "max_compression_threshold_change_db",
            "min_interval_sec",
            "cooldown_sec",
            "hysteresis_db",
            "max_rate_db_per_sec",
            "correction_budget_db",
            "min_hold_sec",
        )
        for name in non_negative:
            if float(getattr(self, name)) < 0.0:
                raise ValueError(f"{name} must be non-negative")
        if self.max_eq_q <= 0.0:
            raise ValueError("max_eq_q must be positive")
        if self.min_eq_frequency_hz <= 0.0:
            raise ValueError("min_eq_frequency_hz must be positive")
        if self.max_eq_frequency_hz <= self.min_eq_frequency_hz:
            raise ValueError("max_eq_frequency_hz must exceed min_eq_frequency_hz")
        if not 0.0 <= self.max_pan_abs <= 1.0:
            raise ValueError("max_pan_abs must be in range 0..1")
        if self.max_compression_ratio < 1.0:
            raise ValueError("max_compression_ratio must be at least 1")
        if self.budget_window_sec <= 0.0:
            raise ValueError("budget_window_sec must be positive")
        if self.true_peak_ceiling_dbtp > 0.0:
            raise ValueError("true_peak_ceiling_dbtp must not exceed 0 dBTP")

    @classmethod
    def from_mapping(cls, payload: Mapping[str, Any] | None) -> "SafetyGateConfig":
        payload = payload or {}
        def exact_bool(name: str, default: bool) -> bool:
            value = payload.get(name, default)
            if type(value) is not bool:
                raise ValueError(f"{name} must be an exact boolean")
            return value
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
            live_mode_blocks_abrupt_changes=exact_bool("live_mode_blocks_abrupt_changes", True),
            true_peak_ceiling_dbtp=float(payload.get("true_peak_ceiling_dbtp", -1.0)),
            dry_run=exact_bool("dry_run", False),
            emergency_bypass=exact_bool("emergency_bypass", False),
            max_rate_db_per_sec=float(payload.get("max_rate_db_per_sec", 0.0)),
            correction_budget_db=float(payload.get("correction_budget_db", 0.0)),
            budget_window_sec=float(payload.get("budget_window_sec", 60.0)),
            min_hold_sec=float(payload.get("min_hold_sec", 0.0)),
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
        # (target, action_type) -> [(t, |change_db|)] для бюджета коррекций
        self._change_history: Dict[Tuple[str, str], list] = {}
        # (target, action_type) -> (t, sign) для min-hold (запрет реверса)
        self._last_change_sign: Dict[Tuple[str, str], Tuple[float, float]] = {}

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
                key = (checked.target, checked.action_type)
                self._last_allowed_at[key] = now
                change_db = _number(checked.parameters.get("gain_db"), None)
                if change_db is not None and change_db != 0.0:
                    self._change_history.setdefault(key, []).append((now, abs(change_db)))
                    self._last_change_sign[key] = (now, 1.0 if change_db > 0 else -1.0)

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

        if _contains_non_finite_number(decision.parameters):
            return None, "non_finite_parameter"

        # Emergency authority is code/config owned. Untrusted decision metadata
        # is descriptive and can never relax safety limits.
        emergency = self.config.emergency_bypass is True
        if not emergency:
            if not decision.safe_to_apply:
                return None, "decision_engine_marked_unsafe"
            if decision.risk_level in {RISK_HIGH, RISK_CRITICAL}:
                return None, f"risk_level_{decision.risk_level}"
            limited = self._rate_or_cooldown_limited(decision, now)
            if limited:
                return None, limited
            limited = self._anti_oscillation_limited(decision, now)
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
        if not emergency and live_mode and gain_db > 0.0:
            if true_peak is None:
                return None, "true_peak_headroom_unknown"
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

    def _anti_oscillation_limited(self, decision: ActionDecision, now: float) -> str:
        """Бюджет коррекций, min-hold (запрет реверса) и лимит дБ/с.

        Применяется к решениям с дельтой в дБ (gain/eq). Все три лимита
        по умолчанию выключены (0.0) и включаются live-конфигом.
        """
        if decision.action_type not in (ACTION_GAIN, ACTION_EQ):
            return ""
        change_db = _number(decision.parameters.get("gain_db"), None)
        if change_db is None or change_db == 0.0:
            return ""
        key = (decision.target, decision.action_type)

        if self.config.max_rate_db_per_sec > 0.0:
            last = self._last_allowed_at.get(key)
            if last is not None:
                dt = now - last
                if dt <= 0.0 or abs(change_db) / dt > self.config.max_rate_db_per_sec:
                    return "rate_db_per_sec_exceeded"

        if self.config.min_hold_sec > 0.0:
            held = self._last_change_sign.get(key)
            if held is not None:
                held_at, held_sign = held
                new_sign = 1.0 if change_db > 0 else -1.0
                if new_sign == -held_sign and (now - held_at) < self.config.min_hold_sec:
                    return "min_hold_reversal_blocked"

        if self.config.correction_budget_db > 0.0:
            history = self._change_history.get(key, [])
            horizon = now - self.config.budget_window_sec
            history = [(t, c) for t, c in history if t >= horizon]
            self._change_history[key] = history
            spent = sum(c for _, c in history)
            if spent + abs(change_db) > self.config.correction_budget_db:
                return "correction_budget_exhausted"

        return ""

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
        numeric = float(value)
        return numeric if math.isfinite(numeric) else default
    except (TypeError, ValueError):
        return default


def _contains_non_finite_number(value: Any) -> bool:
    if isinstance(value, Mapping):
        return any(_contains_non_finite_number(item) for item in value.values())
    if isinstance(value, (list, tuple)):
        return any(_contains_non_finite_number(item) for item in value)
    if isinstance(value, bool) or value is None:
        return False
    if isinstance(value, (int, float)):
        return not math.isfinite(float(value))
    if isinstance(value, str):
        try:
            return not math.isfinite(float(value))
        except ValueError:
            return False
    return False


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
