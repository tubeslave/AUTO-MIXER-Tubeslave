import numpy as np

from feedback_detector import FeedbackDetector, FeedbackDetectorConfig


SR = 48000
FFT = 2048


def _tone(freq: float, amplitude: float = 0.5) -> np.ndarray:
    t = np.arange(FFT, dtype=np.float64) / SR
    return (amplitude * np.sin(2.0 * np.pi * freq * t)).astype(np.float32)


def _detector(persistence: int = 6) -> FeedbackDetector:
    cfg = FeedbackDetectorConfig(
        sample_rate=SR,
        fft_size=FFT,
        persistence_frames=persistence,
        min_confidence=0.5,
        peak_height_db=-20.0,
        peak_prominence_db=10.0,
        peak_distance_bins=4,
    )
    return FeedbackDetector(config=cfg)


def test_stable_narrowband_feedback_is_eventually_detected():
    detector = _detector(persistence=6)
    block = _tone(1992.1875, 0.6)  # exact FFT bin
    events = []
    for _ in range(14):
        events.extend(detector.process(1, block))
    assert events, "persistent stable feedback must not require monotonically rising level"
    assert abs(events[0].frequency_hz - 1992.1875) <= detector.freq_resolution
    assert events[0].action in {"notch", "fader_reduce"}


def test_rising_narrowband_feedback_is_detected():
    detector = _detector(persistence=5)
    events = []
    for amplitude in np.linspace(0.08, 0.8, 12):
        events.extend(detector.process(2, _tone(3000.0, float(amplitude))))
    assert events
    assert abs(events[0].frequency_hz - 3000.0) <= detector.freq_resolution * 1.5


def test_broadband_noise_does_not_false_trigger():
    detector = _detector(persistence=6)
    rng = np.random.default_rng(20260907)
    events = []
    for _ in range(30):
        block = (rng.standard_normal(FFT) * 0.12).astype(np.float32)
        events.extend(detector.process(3, block))
    assert events == []


def test_invalid_samples_are_sanitized_and_do_not_trigger():
    detector = _detector(persistence=4)
    block = np.zeros(FFT, dtype=np.float32)
    block[100] = np.nan
    block[200] = np.inf
    block[300] = -np.inf
    events = []
    for _ in range(10):
        events.extend(detector.process(4, block))
    assert events == []
    diagnostics = detector.get_diagnostics(4)
    for peak in diagnostics.get("peak_details", []):
        assert np.isfinite(peak["frequency"])
        assert np.isfinite(peak["magnitude_db"])


def test_disable_prevents_detection():
    detector = _detector(persistence=3)
    detector.disable()
    events = []
    block = _tone(1000.0, 0.9)
    for _ in range(12):
        events.extend(detector.process(5, block))
    assert events == []
