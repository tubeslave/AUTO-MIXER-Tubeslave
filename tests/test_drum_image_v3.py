"""Drum image v3 — операторская схема панорамы и фазы ударных.

Синтетика: снейр-клик-трек с известными лагами/уровнями в OH-паре.
"""
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "backend"))

from heuristics.drum_image_v3 import (  # noqa: E402
    equal_power_gains,
    image_position_pan,
    oh_pans_for_center,
    plan_drum_image,
)

SR = 48000


def _click_train(n_hits=8, gap_s=0.4, lag_ms=0.0, gain_db=0.0,
                 total_s=4.0, seed=1):
    """Затухающий шумовой всплеск (снейроподобный) каждые gap_s."""
    rng = np.random.default_rng(seed)
    x = np.zeros(int(total_s * SR))
    burst_len = int(0.05 * SR)
    burst = rng.standard_normal(burst_len) * np.exp(
        -np.arange(burst_len) / (0.008 * SR))
    lag = int(round(lag_ms / 1000.0 * SR))
    g = 10.0 ** (gain_db / 20.0)
    for k in range(n_hits):
        t = int((0.3 + k * gap_s) * SR) + lag
        if 0 <= t < len(x) - burst_len:
            x[t:t + burst_len] += burst * g
    return x


def test_oh_pans_center_balanced_pair_stays_hard():
    pan_l, pan_r = oh_pans_for_center(0.0)
    assert pan_l == pytest.approx(-100.0, abs=1.0)
    assert pan_r == pytest.approx(100.0, abs=1.0)


def test_oh_pans_louder_left_moves_left_oh_inward():
    pan_l, pan_r = oh_pans_for_center(6.0)   # снейр громче в OH L
    assert pan_r == pytest.approx(100.0)
    assert -100.0 < pan_l < 0.0
    # и симметрично для правого
    pl2, pr2 = oh_pans_for_center(-6.0)
    assert pl2 == pytest.approx(-100.0)
    assert pr2 == pytest.approx(-pan_l, abs=1.0)


def test_oh_pans_actually_center_the_snare():
    """После панорам суммарный снейр в шинах L и R равен."""
    for d in (2.0, 6.0, 11.0):
        pan_l, pan_r = oh_pans_for_center(d)
        al, ar = 10.0 ** (d / 20.0), 1.0   # снейр в OH L громче на d дБ
        gll, glr = equal_power_gains(pan_l)
        grl, grr = equal_power_gains(pan_r)
        bus_l = al * gll + ar * grl
        bus_r = al * glr + ar * grr
        assert bus_l == pytest.approx(bus_r, rel=0.02)


def test_image_position_pan_centered_source_is_center():
    pan = image_position_pan(-30.0, -30.0, -100.0, 100.0)
    assert pan == pytest.approx(0.0, abs=2.0)


def test_image_position_pan_right_heavy_source_goes_right():
    pan = image_position_pan(-40.0, -28.0, -100.0, 100.0)
    assert pan > 30.0


def test_plan_aligns_oh_pair_and_close_mics_into_oh_image():
    """OH R позже OH L на 1.0 мс -> задержка на OH L, опора OH R.
    OH слышит снейр на 3 мс позже близкого -> Snare T задержан на ~3 мс.
    Том слышит снейр раньше OH на 2 мс -> том задержан на ~2 мс."""
    snare = _click_train()
    oh_l = _click_train(lag_ms=3.0, gain_db=-12.0)
    oh_r = _click_train(lag_ms=4.0, gain_db=-12.0)
    tom = _click_train(lag_ms=2.0, gain_db=-20.0)   # снейр-bleed в томе
    kick_quiet = np.random.default_rng(9).standard_normal(
        int(4.0 * SR)) * 1e-4   # «кик» молчит: ровный шумовой пол
    buffers = {1: kick_quiet,
               2: snare, 3: snare * 0.5, 4: tom, 6: oh_l, 7: oh_r}
    plan = plan_drum_image(buffers, SR, oh_pair=(6, 7), snare_t=2,
                           snare_b=3, kick=1, toms=[4])
    # OH L раньше -> задержка на нём, опора OH R
    assert plan.oh_ref == 7
    assert plan.delays_ms.get(6) == pytest.approx(1.0, abs=0.3)
    # снейр догоняет опорный OH (4 мс полёта)
    assert plan.delays_ms.get(2) == pytest.approx(4.0, abs=0.5)
    # том догоняет OH по снейр-bleed (4 - 2 = 2 мс)
    assert plan.delays_ms.get(4) == pytest.approx(2.0, abs=0.5)
    # kick/snare в центре
    assert plan.pans[1] == 0.0
    assert plan.pans[2] == 0.0
    assert plan.pans[3] == 0.0
    # OH-пара с равным снейром остаётся жёсткой
    assert plan.pans[6] == pytest.approx(-100.0, abs=3.0)
    assert plan.pans[7] == pytest.approx(100.0, abs=3.0)


def test_plan_rejects_implausible_oh_internal_delay():
    """Внутренняя задержка OH > 2 мс — пара стоит криво, не трогаем."""
    snare = _click_train()
    oh_l = _click_train(lag_ms=3.0, gain_db=-12.0)
    oh_r = _click_train(lag_ms=8.0, gain_db=-12.0)   # 5 мс между OH
    buffers = {2: snare, 6: oh_l, 7: oh_r}
    plan = plan_drum_image(buffers, SR, oh_pair=(6, 7), snare_t=2)
    assert 6 not in plan.delays_ms and 7 not in plan.delays_ms
    assert any("криво" in n for n in plan.notes)


def test_plan_without_snare_does_nothing():
    plan = plan_drum_image({}, SR, oh_pair=(6, 7), snare_t=None)
    assert not plan.delays_ms and not plan.polarity_flip


def test_implausible_snare_tb_lag_skips_polarity_too():
    """Правило CLAUDE.md («9.90 мс» live 2026-07-06): лаг T↔B за пределами
    физики пары — периодическая неоднозначность GCC-PHAT; знак r на таком
    лаге мусор. Ревью C3: флип полярности уезжал ДО этой проверки —
    теперь пара пропускается ЦЕЛИКОМ, включая polarity."""
    snare_t = _click_train()
    oh_l = _click_train(lag_ms=3.0, gain_db=-12.0)
    oh_r = _click_train(lag_ms=3.5, gain_db=-12.0)
    # нижний снейр: ИНВЕРТИРОВАН и «замерен» с неправдоподобным лагом 9.9 мс
    snare_b = -_click_train(lag_ms=9.9, gain_db=-3.0)
    buffers = {2: snare_t, 3: snare_b, 6: oh_l, 7: oh_r}
    plan = plan_drum_image(buffers, SR, oh_pair=(6, 7), snare_t=2, snare_b=3)
    assert 3 not in plan.polarity_flip, (
        "полярность применена по знаку r на неоднозначном лаге"
    )
    assert 3 not in plan.delays_ms
    assert any("пропущена целиком" in n for n in plan.notes)


def test_plausible_inverted_snare_b_still_flipped():
    """Контроль: инверсия при правдоподобном лаге 0.5 мс детектируется."""
    snare_t = _click_train()
    oh_l = _click_train(lag_ms=3.0, gain_db=-12.0)
    oh_r = _click_train(lag_ms=3.5, gain_db=-12.0)
    snare_b = -_click_train(lag_ms=0.5, gain_db=-3.0)
    buffers = {2: snare_t, 3: snare_b, 6: oh_l, 7: oh_r}
    plan = plan_drum_image(buffers, SR, oh_pair=(6, 7), snare_t=2, snare_b=3)
    assert plan.polarity_flip.get(3) is True
