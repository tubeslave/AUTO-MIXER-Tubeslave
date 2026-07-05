"""Phase v3: кластеры, белые списки, полярность, задержки, live-детекция."""

import numpy as np
import pytest

from heuristics.phase_v3 import (
    DEFAULT_CLUSTERS,
    AlignmentCluster,
    LiveDriftMonitor,
    PhaseSnapshot,
    align_cluster,
    can_apply_live,
    detect_onsets,
    estimate_pair_delay,
    geometry_plausible,
    isolated_hits,
    pair_alignment_allowed,
    polarity_decision,
    quantize_delay_ms,
    sum_energy_gain_db,
)

SR = 48000
RNG = np.random.default_rng(31)


def _hits_signal(n_sec=6.0, period_s=0.5, f0=180.0, decay=40.0, amp=0.6):
    """Периодические затухающие удары (снейроподобные, ШИРОКОПОЛОСНЫЕ).

    GCC-PHAT деградирует на узкополосных сигналах (Clifford & Reiss) —
    реальный снейр широкополосен, синтетика должна быть тоже.
    """
    n = int(n_sec * SR)
    x = np.zeros(n)
    period = int(period_s * SR)
    t = np.arange(period) / SR
    rng = np.random.default_rng(7)
    body = np.sin(2 * np.pi * f0 * t)
    snap = rng.normal(0, 1, period)          # широкополосный «треск»
    hit = amp * np.exp(-t * decay) * (0.5 * body + 0.5 * snap)
    for start in range(0, n - period, period):
        x[start:start + period] += hit
    return x


def _delayed(x, delay_ms, polarity=1.0, gain=0.7):
    shift = int(round(delay_ms / 1000.0 * SR))
    return polarity * gain * np.roll(x, shift)


class TestWhitelist:
    def test_vocals_never_aligned(self):
        ok, reason = pair_alignment_allowed("lead_vocal", "kick")
        assert not ok and "never_align" in reason

    def test_stereo_pair_interior_forbidden(self):
        ok, reason = pair_alignment_allowed("overheads", "overheads")
        assert not ok and "stereo" in reason

    def test_room_forbidden(self):
        ok, _ = pair_alignment_allowed("room", "kick")
        assert not ok

    def test_drums_pair_allowed(self):
        ok, _ = pair_alignment_allowed("snare", "overheads")
        assert ok

    def test_room_cluster_align_disabled(self):
        room = next(c for c in DEFAULT_CLUSTERS if c.name == "room_mics")
        assert not room.align
        result = align_cluster({"rl": 0.0, "rr": 1.0},
                               {"rl": "room", "rr": "room"}, room)
        assert result["delays_ms"] == {}


class TestIsolatedHits:
    def test_guard_removes_simultaneous_hits(self):
        snare = _hits_signal(period_s=0.5)
        kick = np.roll(_hits_signal(period_s=1.0, f0=60.0), int(0.005 * SR))
        hits = isolated_hits(snare, kick, SR)
        all_hits = detect_onsets(snare, SR)
        assert 0 < len(hits) < len(all_hits)  # совпавшие с киком отброшены

    def test_no_other_hits_keeps_all(self):
        snare = _hits_signal()
        silent = np.zeros_like(snare)
        assert len(isolated_hits(snare, silent, SR)) == len(detect_onsets(snare, SR))


class TestDelayEstimation:
    def test_recovers_known_delay(self):
        close = _hits_signal()
        far = _delayed(close, 3.0) + 0.02 * RNG.normal(0, 1, len(close))
        hits = detect_onsets(close, SR)
        est = estimate_pair_delay(close, far, SR, hits)
        assert est.accepted, est.reject_reason
        assert abs(abs(est.delay_ms) - 3.0) < 0.5

    def test_insufficient_hits_rejected(self):
        close = _hits_signal(n_sec=1.5)   # ~3 удара
        far = _delayed(close, 3.0)
        hits = detect_onsets(close, SR)
        est = estimate_pair_delay(close, far, SR, hits)
        assert not est.accepted
        assert est.reject_reason in ("insufficient_hits", "no_valid_hits")

    def test_mad_rule_project(self):
        """MAD ≤ 0.5 мс — иначе нужен соло-проход (правило CLAUDE.md)."""
        close = _hits_signal()
        hits = detect_onsets(close, SR)
        # шумный «дальний»: половина ударов с другой задержкой
        far = _delayed(close, 3.0)
        far[len(far) // 2:] = _delayed(close, 8.0)[len(far) // 2:]
        est = estimate_pair_delay(close, far, SR, hits)
        if not est.accepted:
            assert est.reject_reason in (
                "mad_above_0.5ms_needs_solo_pass", "iqr_unstable",
                "insufficient_hits")


class TestPolarity:
    def test_inverted_pair_flip(self):
        x = _hits_signal()
        y = _delayed(x, 0.0, polarity=-1.0)
        d = polarity_decision(x, y, SR)
        assert d["flip"] is True
        assert d["confidence"] >= 0.4

    def test_in_polarity_no_flip(self):
        x = _hits_signal()
        d = polarity_decision(x, _delayed(x, 0.0), SR)
        assert d["flip"] is False

    def test_uncorrelated_manual_flag(self):
        x = RNG.normal(0, 0.3, 2 * SR)
        y = RNG.normal(0, 0.3, 2 * SR)
        d = polarity_decision(x, y, SR)
        assert d["flip"] is None and d["flag"] == "manual"


class TestClusterSolver:
    ROLES = {1: "kick", 2: "snare", 6: "overheads"}
    DRUMS = next(c for c in DEFAULT_CLUSTERS if c.name == "drums_close_oh")

    def test_farthest_reference_positive_delays(self):
        # OH прибывает позже всех (дальше): kick 0, snare 0.5, OH 3.5 мс
        arrivals = {1: 0.0, 2: 0.5, 6: 3.5}
        result = align_cluster(arrivals, self.ROLES, self.DRUMS)
        assert result["reference"] == 6
        assert result["delays_ms"][1] == pytest.approx(3.5)
        assert result["delays_ms"][2] == pytest.approx(3.0)
        assert result["delays_ms"][6] == pytest.approx(0.0)
        assert all(d >= 0 for d in result["delays_ms"].values())

    def test_closest_policy(self):
        cluster = AlignmentCluster(name="d", member_roles=("kick",),
                                   reference_policy="closest", cap_ms=10.0)
        arrivals = {1: 0.0, 2: 0.5, 6: 3.5}
        result = align_cluster(arrivals, self.ROLES, cluster)
        assert result["reference"] == 1
        # «ускорить» дальних нельзя — задержки 0 (физика)
        assert all(d == 0.0 for d in result["delays_ms"].values())

    def test_cap_excludes_room_scale_delays(self):
        arrivals = {1: 0.0, 6: 23.0}   # room-масштаб — вне кэпа 10 мс
        result = align_cluster(arrivals, {1: "kick", 6: "overheads"}, self.DRUMS)
        assert any("exceeds_cap" in reason for _, reason in result["excluded"])

    def test_vocal_zero_added_delay(self):
        arrivals = {1: 0.0, 16: -2.0}   # вокал «ближе» — получил бы 2 мс
        roles = {1: "kick", 16: "lead_vocal"}
        result = align_cluster(arrivals, roles, self.DRUMS)
        assert 16 not in result["delays_ms"]

    def test_quantized_to_console_step(self):
        assert quantize_delay_ms(3.14) == pytest.approx(3.1)
        arrivals = {1: 0.0, 6: 3.14}
        result = align_cluster(arrivals, {1: "kick", 6: "overheads"}, self.DRUMS)
        assert result["delays_ms"][1] == pytest.approx(3.1)

    def test_geometry(self):
        assert geometry_plausible(3.0)        # ~1 м
        assert not geometry_plausible(30.0)   # ~10 м — room, не кластер


class TestSumVerification:
    def test_alignment_improves_low_band_sum(self):
        t = np.arange(4 * SR) / SR
        a = np.sin(2 * np.pi * 80.0 * t)
        b = _delayed(a, 6.25 / 2)  # ~половина периода 80 Гц → частичное гашение
        gain = sum_energy_gain_db(a, b, delay_ms=-6.25 / 2, sample_rate=SR)
        assert gain > 1.0          # выравнивание вернуло НЧ

    def test_degradation_detected(self):
        t = np.arange(4 * SR) / SR
        a = np.sin(2 * np.pi * 80.0 * t)
        b = 0.7 * a                # уже выровнено
        gain = sum_energy_gain_db(a, b, delay_ms=6.25, sample_rate=SR)
        assert gain < 0.0          # «коррекция» ухудшила сумму → откат


class TestLiveMonitor:
    def _snap(self):
        return PhaseSnapshot(delays_ms={2: 3.0}, polarity_flips={3: True},
                             polarity_r_signs={3: -0.8})

    def test_drift_needs_two_consecutive_windows(self):
        mon = LiveDriftMonitor(self._snap())
        assert mon.check_drift({2: 5.0}) is None       # первое окно
        alert = mon.check_drift({2: 5.2})              # второе подряд
        assert alert is not None and alert["name"] == "DELAY_DRIFT"

    def test_drift_resets_on_stable_window(self):
        mon = LiveDriftMonitor(self._snap())
        mon.check_drift({2: 5.0})
        mon.check_drift({2: 3.1})                      # стабильно
        assert mon.check_drift({2: 5.0}) is None       # счётчик сброшен

    def test_polarity_flip_alert(self):
        mon = LiveDriftMonitor(self._snap())
        alert = mon.check_polarity_flip({3: +0.7})     # знак сменился
        assert alert is not None and alert["name"] == "POLARITY_FLIP"

    def test_apply_only_in_silence_and_armed(self):
        assert not can_apply_live(silence_detected=False, operator_armed=True)
        assert not can_apply_live(silence_detected=True, operator_armed=False)
        assert can_apply_live(silence_detected=True, operator_armed=True)
