"""SafetyGate v2: бюджет коррекций, min-hold, лимит дБ/с + level riding."""

import pytest

from automixer.decision.models import ACTION_EQ, ACTION_GAIN, ActionDecision, ActionPlan
from automixer.safety.safety_gate import SafetyGate, SafetyGateConfig
from level_riding import (
    loudness_hysteresis_gate,
    priority_target_lu,
    riding_step_db,
)


class _Clock:
    def __init__(self):
        self.t = 0.0

    def __call__(self):
        return self.t


def _plan(gain_db: float, target="ch3", action=ACTION_GAIN, decision_id="d1"):
    return ActionPlan(
        plan_id="p1", mode="live",
        decisions=[ActionDecision(
            id=decision_id, action_type=action, target=target,
            parameters={"gain_db": gain_db, "frequency_hz": 500.0, "q": 1.0},
            reason="test", confidence=1.0, risk_level="low", safe_to_apply=True,
        )],
        source_modules=["test"],
    )


def _gate(clock, **cfg):
    base = dict(min_interval_sec=0.0, cooldown_sec=0.0, hysteresis_db=0.1,
                min_gain_change_db=0.1, max_gain_change_db=6.0,
                live_mode_blocks_abrupt_changes=False)
    base.update(cfg)
    return SafetyGate(SafetyGateConfig.from_mapping(base), time_provider=clock)


class TestDefaultsUnchanged:
    def test_disabled_limits_do_not_block(self):
        clock = _Clock()
        gate = _gate(clock)
        for i in range(5):
            clock.t = i * 0.01  # очень частые крупные шаги
            res = gate.evaluate_plan(_plan(-3.0), live_mode=False)
            assert res.allowed_count == 1, res.blocked


class TestRatePerSec:
    def test_second_fast_step_blocked(self):
        clock = _Clock()
        gate = _gate(clock, max_rate_db_per_sec=3.0)
        clock.t = 1.0
        assert gate.evaluate_plan(_plan(-3.0), live_mode=False).allowed_count == 1
        clock.t = 1.2  # -3 дБ за 0.2 c = 15 дБ/с > 3 дБ/с
        res = gate.evaluate_plan(_plan(-3.0), live_mode=False)
        assert res.blocked[0]["reason"] == "rate_db_per_sec_exceeded"
        clock.t = 3.0  # -3 дБ за 1.8 c = 1.7 дБ/с — можно
        assert gate.evaluate_plan(_plan(-3.0), live_mode=False).allowed_count == 1


class TestMinHold:
    def test_reversal_blocked_within_hold(self):
        clock = _Clock()
        gate = _gate(clock, min_hold_sec=1.0)
        clock.t = 1.0
        assert gate.evaluate_plan(_plan(-2.0), live_mode=False).allowed_count == 1
        clock.t = 1.3  # реверс (буст после ката) внутри hold
        res = gate.evaluate_plan(_plan(+1.0), live_mode=False)
        assert res.blocked[0]["reason"] == "min_hold_reversal_blocked"
        # то же направление внутри hold — разрешено
        res = gate.evaluate_plan(_plan(-1.0), live_mode=False)
        assert res.allowed_count == 1
        clock.t = 3.0  # hold истёк — реверс разрешён
        assert gate.evaluate_plan(_plan(+1.0), live_mode=False).allowed_count == 1


class TestBudget:
    def test_budget_exhaustion_blocks_until_relaxation(self):
        clock = _Clock()
        gate = _gate(clock, correction_budget_db=5.0, budget_window_sec=10.0)
        clock.t = 1.0
        assert gate.evaluate_plan(_plan(-3.0), live_mode=False).allowed_count == 1
        clock.t = 2.0
        assert gate.evaluate_plan(_plan(-2.0), live_mode=False).allowed_count == 1
        clock.t = 3.0  # бюджет 5 дБ исчерпан
        res = gate.evaluate_plan(_plan(-1.0), live_mode=False)
        assert res.blocked[0]["reason"] == "correction_budget_exhausted"
        clock.t = 12.5  # первые изменения вышли из окна — релаксация
        assert gate.evaluate_plan(_plan(-1.0), live_mode=False).allowed_count == 1

    def test_budget_applies_to_eq_separately_per_action(self):
        clock = _Clock()
        gate = _gate(clock, correction_budget_db=4.0, budget_window_sec=10.0,
                     max_eq_cut_db=6.0, max_eq_boost_db=2.0)
        clock.t = 1.0
        assert gate.evaluate_plan(
            _plan(-3.0, action=ACTION_EQ), live_mode=False).allowed_count == 1
        clock.t = 2.0
        res = gate.evaluate_plan(_plan(-3.0, action=ACTION_EQ), live_mode=False)
        assert res.blocked[0]["reason"] == "correction_budget_exhausted"
        # бюджет EQ не съедает бюджет GAIN того же target
        res = gate.evaluate_plan(_plan(-3.0), live_mode=False)
        assert res.allowed_count == 1


class TestLevelRiding:
    def test_deadband(self):
        assert riding_step_db(-24.5, -25.0, dt_s=1.0) == 0.0

    def test_step_and_rate_limits(self):
        # error +6 LU, шаг ограничен 1.5 дБ
        assert riding_step_db(-31.0, -25.0, dt_s=1.0) == pytest.approx(1.5)
        # маленький dt: скорость 3 дБ/с ограничивает сильнее шага
        assert riding_step_db(-31.0, -25.0, dt_s=0.2) == pytest.approx(0.6)

    def test_priority_offsets(self):
        assert priority_target_lu(-28.0, "lead_vocal") == pytest.approx(-25.0)
        assert priority_target_lu(-28.0, "overheads") == pytest.approx(-31.0)

    def test_loudness_gate_blocks_silence_and_intentional_quiet(self):
        # молчание
        assert not loudness_hysteresis_gate([-60.0] * 8)
        # намеренно тихий куплет: типичный уровень -20, сейчас -30
        series = [-20.0] * 10 + [-30.0]
        assert not loudness_hysteresis_gate(series)
        # рабочий диапазон — райдинг разрешён
        assert loudness_hysteresis_gate([-21.0, -20.0, -19.5, -20.5, -20.0])

    def test_loudness_gate_needs_statistics(self):
        assert not loudness_hysteresis_gate([-20.0, -20.0])
