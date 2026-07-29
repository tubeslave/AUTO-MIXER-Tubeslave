"""Ф5 (Р10): blessed mix снимок + план REVERT bounded-рампой."""

from unittest.mock import MagicMock

import pytest

from blessed_mix_revert import (
    REVERT_FADER_STEP_DB,
    build_revert_plan,
    capture_blessed_snapshot,
)


def _mixer(fader=-6.0, mute=0, trim=2.0, dca=-3.0, bus=-2.0):
    m = MagicMock()
    m.get_fader.return_value = fader
    m.get_channel_mute.return_value = mute
    m.get_channel_gain.return_value = trim
    m.get_dca_fader.return_value = dca
    m.get_bus_fader.return_value = bus
    return m


class TestSnapshot:
    def test_captures_known_values(self):
        snap = capture_blessed_snapshot(_mixer(), [1, 2], dca_ids=[1], bus_ids=[3])
        assert snap.channels[1] == {"fader_db": -6.0, "muted": 0, "trim_db": 2.0}
        assert snap.dca[1] == -3.0
        assert snap.buses[3] == -2.0

    def test_unknown_channel_excluded_not_defaulted(self):
        """Никаких подстановок −144: неизвестный канал не входит в снимок."""
        m = _mixer()
        m.get_fader.return_value = None
        m.get_channel_mute.return_value = None
        m.get_channel_gain.return_value = None
        snap = capture_blessed_snapshot(m, [1])
        assert 1 not in snap.channels


class TestRevertPlan:
    def test_ramp_steps_bounded(self):
        """Возврат −18 → −6: рампа шагами ≤ 1.5 дБ, финал — точная цель."""
        snap = capture_blessed_snapshot(_mixer(fader=-6.0), [1])
        mixer_now = _mixer(fader=-18.0)
        plan = build_revert_plan(snap, mixer_now)
        fader_steps = [p for p in plan if p["action"] == "channel_fader"]
        assert fader_steps, "рампа не построена"
        prev = -18.0
        for p in fader_steps:
            assert abs(p["target_db"] - prev) <= REVERT_FADER_STEP_DB + 1e-9
            prev = p["target_db"]
        assert fader_steps[-1]["target_db"] == pytest.approx(-6.0)
        assert len(fader_steps) == 8  # 12 дБ / 1.5

    def test_idempotent_when_already_at_blessed(self):
        snap = capture_blessed_snapshot(_mixer(fader=-6.0), [1])
        plan = build_revert_plan(snap, _mixer(fader=-6.0))
        assert [p for p in plan if p["action"] == "channel_fader"] == []
        assert [p for p in plan if p["action"] == "channel_trim"] == []

    def test_mute_and_trim_restored(self):
        snap = capture_blessed_snapshot(_mixer(mute=0, trim=2.0), [1])
        mixer_now = _mixer(mute=1, trim=-4.0)
        plan = build_revert_plan(snap, mixer_now)
        assert {"action": "channel_mute", "id": 1, "muted": 0} in plan
        trims = [p for p in plan if p["action"] == "channel_trim"]
        assert trims and trims[0]["target_db"] == pytest.approx(2.0)

    def test_unknown_current_skipped_loudly(self):
        """current неизвестен → skip-пометка, не слепая рампа."""
        snap = capture_blessed_snapshot(_mixer(fader=-6.0), [1])
        mixer_now = _mixer()
        mixer_now.get_fader.return_value = None
        plan = build_revert_plan(snap, mixer_now)
        skips = [p for p in plan if p["action"] == "skip"]
        assert skips and skips[0]["reason"] == "current_unknown"

    def test_dca_and_bus_ramped(self):
        snap = capture_blessed_snapshot(
            _mixer(dca=-3.0, bus=-2.0), [], dca_ids=[1], bus_ids=[5])
        mixer_now = _mixer(dca=-9.0, bus=-8.0)
        plan = build_revert_plan(snap, mixer_now)
        assert [p for p in plan if p["action"] == "dca_fader"][-1][
            "target_db"] == pytest.approx(-3.0)
        assert [p for p in plan if p["action"] == "bus_fader"][-1][
            "target_db"] == pytest.approx(-2.0)
