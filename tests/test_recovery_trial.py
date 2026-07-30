"""Тесты стенда испытания восстановлением (Ф8, DEC-0008 §6)."""

import numpy as np
import pytest

from recovery_trial import (
    DEFAULT_CORPUS,
    Defect,
    TrialReport,
    apply_defect,
    run_trial,
    spectral_distance_db,
)

SR = 48000


def pink_like(seconds=2.0, seed=7):
    """Розовоподобный шум: белый через каскад полюсов (Voss-приближение)."""
    rng = np.random.default_rng(seed)
    white = rng.standard_normal(int(seconds * SR))
    from scipy.signal import lfilter
    b = [0.049922035, -0.095993537, 0.050612699, -0.004408786]
    a = [1, -2.494956002, 2.017265875, -0.522189400]
    x = lfilter(b, a, white)
    return 0.3 * x / np.abs(x).max()


class TestDefects:
    def test_defect_changes_spectrum(self):
        x = pink_like()
        broken = apply_defect(x, SR, Defect("mud", 300.0, 4.0, 0.9))
        assert spectral_distance_db(broken, x, SR) > 0.3

    def test_inverse_restores(self):
        x = pink_like()
        d = Defect("mud", 300.0, 4.0, 0.9)
        restored = apply_defect(apply_defect(x, SR, d), SR, d.inverse())
        assert spectral_distance_db(restored, x, SR) < 0.05

    def test_distance_zero_on_identical(self):
        x = pink_like()
        assert spectral_distance_db(x, x, SR) == pytest.approx(0.0, abs=1e-9)


class TestTrial:
    def test_oracle_repair_scores_near_one(self):
        """Оракул знает дефект — стенд обязан дать ratio ≈ 1."""
        x = pink_like()
        corpus = (Defect("mud", 300.0, 4.0, 0.9),)

        def oracle(broken, sr):
            return apply_defect(broken, sr, corpus[0].inverse())

        report = run_trial(x, SR, oracle, corpus)
        assert report.mean_recovery_ratio > 0.95
        assert report.gate_passed()

    def test_no_repair_scores_zero(self):
        x = pink_like()
        report = run_trial(x, SR, lambda b, sr: b, DEFAULT_CORPUS[:3])
        assert report.mean_recovery_ratio == pytest.approx(0.0, abs=0.02)
        assert not report.gate_passed()

    def test_partial_repair_scores_partial(self):
        x = pink_like()
        d = Defect("mud", 300.0, 4.0, 0.9)

        def half_repair(broken, sr):
            return apply_defect(broken, sr, Defect("half", 300.0, -2.0, 0.9))

        report = run_trial(x, SR, half_repair, (d,))
        assert 0.3 < report.mean_recovery_ratio < 0.75

    def test_harmful_repair_scores_negative(self):
        x = pink_like()
        d = Defect("mud", 300.0, 4.0, 0.9)

        def harmful(broken, sr):
            return apply_defect(broken, sr, Defect("worse", 300.0, 3.0, 0.9))

        report = run_trial(x, SR, harmful, (d,))
        assert report.mean_recovery_ratio < 0.0

    def test_report_dict_schema(self):
        x = pink_like()
        report = run_trial(x, SR, lambda b, sr: b, DEFAULT_CORPUS[:2])
        payload = report.to_dict()
        assert payload["schema"] == "automixer.recovery_trial.v1"
        assert len(payload["results"]) == 2
        assert set(payload["results"][0]) == {
            "defect", "broken_distance_db", "repaired_distance_db", "recovery_ratio",
        }
