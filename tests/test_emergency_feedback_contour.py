"""Ф2 (задача #7, DEC-0005): привилегированный аварийный контур заводки.

Инверсия тестов мёртвого пути (Р5): раньше тесты закрепляли «блокировано
всегда»; теперь контракт — «блокировано по умолчанию, отправляется с полным
gated-циклом при явном операторском опт-ине feedback_emergency_live_apply».
"""

from unittest.mock import MagicMock

import pytest

from autofoh_models import RuntimeState
from autofoh_safety import (
    AutoFOHSafetyConfig,
    AutoFOHSafetyController,
    EmergencyChannelFaderMove,
    EmergencyFeedbackNotch,
)


def _mixer(fader_db=-6.0):
    mixer = MagicMock()
    mixer.get_fader.return_value = fader_db
    mixer.set_fader.return_value = True
    mixer.set_eq_band.return_value = True
    mixer.set_eq_on.return_value = True
    return mixer


def _controller(mixer, **cfg):
    return AutoFOHSafetyController(mixer, config=AutoFOHSafetyConfig(**cfg))


class TestDefaultQuarantine:
    """Default OFF: контур остаётся proposal-only (совместимо с карантином)."""

    def test_notch_blocked_by_default(self):
        mixer = _mixer()
        ctl = _controller(mixer)
        d = ctl.execute(
            EmergencyFeedbackNotch(channel_id=3, band=2, freq_hz=3150.0,
                                   q=8.0, gain_db=-6.0, reason="fb"),
            RuntimeState.EMERGENCY_FEEDBACK,
        )
        assert d.blocked is True and d.sent is False
        mixer.set_eq_band.assert_not_called()
        # предложение несёт исполнимые вызовы для оператора
        assert d.payload["blocked_calls"]

    def test_fader_dip_blocked_by_default(self):
        mixer = _mixer()
        ctl = _controller(mixer)
        d = ctl.execute(
            EmergencyChannelFaderMove(channel_id=3, target_db=-9.0, reason="fb"),
            RuntimeState.EMERGENCY_FEEDBACK,
        )
        assert d.blocked is True and d.sent is False
        mixer.set_fader.assert_not_called()


class TestOptInLiveApply:
    """Опт-ин: family emergency_fader ПРОХОДИТ политику EMERGENCY_FEEDBACK
    (в отличие от family fader — P0 ревью) и реально пишет в пульт."""

    def test_notch_sent_with_eq_on(self):
        mixer = _mixer()
        ctl = _controller(mixer, feedback_emergency_live_apply=True)
        d = ctl.execute(
            EmergencyFeedbackNotch(channel_id=3, band=2, freq_hz=3150.0,
                                   q=8.0, gain_db=-6.0, reason="fb"),
            RuntimeState.EMERGENCY_FEEDBACK,
        )
        assert d.sent is True, d.message
        mixer.set_eq_band.assert_called_once_with(
            3, 2, freq=3150.0, gain=-6.0, q=8.0)
        mixer.set_eq_on.assert_called_once_with(3, 1)

    def test_fader_dip_sent_in_emergency_state(self):
        mixer = _mixer(fader_db=-6.0)
        ctl = _controller(mixer, feedback_emergency_live_apply=True)
        d = ctl.execute(
            EmergencyChannelFaderMove(channel_id=3, target_db=-9.0, reason="fb"),
            RuntimeState.EMERGENCY_FEEDBACK,
        )
        assert d.sent is True, d.message
        mixer.set_fader.assert_called_once_with(3, -9.0)

    def test_plain_fader_family_still_forbidden_in_emergency(self):
        """Обычный ChannelFaderMove в EMERGENCY_FEEDBACK остаётся запрещён —
        привилегия только у emergency_fader."""
        from autofoh_safety import ChannelFaderMove
        mixer = _mixer()
        ctl = _controller(mixer, feedback_emergency_live_apply=True)
        d = ctl.execute(
            ChannelFaderMove(channel_id=3, target_db=-9.0, reason="fb"),
            RuntimeState.EMERGENCY_FEEDBACK,
        )
        assert d.allowed is False
        mixer.set_fader.assert_not_called()


class TestEmergencyBudgets:
    def test_step_clamped_to_3db(self):
        mixer = _mixer(fader_db=-6.0)
        ctl = _controller(mixer, feedback_emergency_live_apply=True)
        d = ctl.execute(
            EmergencyChannelFaderMove(channel_id=3, target_db=-20.0, reason="fb"),
            RuntimeState.EMERGENCY_FEEDBACK,
        )
        assert d.sent is True
        mixer.set_fader.assert_called_once_with(3, -9.0)  # −6 − 3

    def test_cumulative_budget_9db_from_baseline(self):
        mixer = _mixer(fader_db=-6.0)
        ctl = _controller(mixer, feedback_emergency_live_apply=True)
        # три dip'а по −3: −9, −12, −15 = baseline−9 → пол
        for expected in (-9.0, -12.0, -15.0):
            ctl.execute(
                EmergencyChannelFaderMove(channel_id=3, target_db=-40.0, reason="fb"),
                RuntimeState.EMERGENCY_FEEDBACK,
            )
            mixer.get_fader.return_value = expected
        # четвёртый: бюджет исчерпан — цель клампится в пол −15
        d = ctl.execute(
            EmergencyChannelFaderMove(channel_id=3, target_db=-40.0, reason="fb"),
            RuntimeState.EMERGENCY_FEEDBACK,
        )
        assert mixer.set_fader.call_args[0] == (3, -15.0)

    def test_recovery_capped_at_baseline(self):
        mixer = _mixer(fader_db=-6.0)
        ctl = _controller(mixer, feedback_emergency_live_apply=True)
        ctl.execute(
            EmergencyChannelFaderMove(channel_id=3, target_db=-9.0, reason="fb"),
            RuntimeState.EMERGENCY_FEEDBACK,
        )
        mixer.get_fader.return_value = -9.0
        # восстановление: просим 0 дБ — не выше baseline −6 и шага +3
        ctl.execute(
            EmergencyChannelFaderMove(channel_id=3, target_db=0.0, reason="recover"),
            RuntimeState.EMERGENCY_FEEDBACK,
        )
        assert mixer.set_fader.call_args[0] == (3, -6.0)

    def test_unknown_current_refused(self):
        mixer = _mixer()
        mixer.get_fader.return_value = None
        ctl = _controller(mixer, feedback_emergency_live_apply=True)
        d = ctl.execute(
            EmergencyChannelFaderMove(channel_id=3, target_db=-9.0, reason="fb"),
            RuntimeState.EMERGENCY_FEEDBACK,
        )
        assert d.blocked is True and d.sent is False
        mixer.set_fader.assert_not_called()
