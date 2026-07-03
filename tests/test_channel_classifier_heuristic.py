"""Тесты эвристического классификатора каналов на синтетических сигналах.

Проверяем не точное имя класса (эвристика грубее ML), а что признаки
корректно разделяют ГРУППЫ: перкуссия vs тон, низ vs верх, шум vs тон.
"""
import numpy as np
import pytest

from channel_classifier_heuristic import (
    extract_features,
    classify,
    classify_audio,
)

SR = 48000


def _tone(freq, dur=1.0, amp=0.3, sr=SR):
    t = np.arange(int(dur * sr)) / sr
    # тон с гармониками (реалистичнее чистого синуса)
    sig = amp * (np.sin(2*np.pi*freq*t)
                 + 0.4*np.sin(2*np.pi*2*freq*t)
                 + 0.2*np.sin(2*np.pi*3*freq*t))
    return sig.astype(np.float32)


def _noise_band(lo, hi, dur=1.0, amp=0.3, sr=SR):
    n = int(dur * sr)
    white = np.random.default_rng(0).standard_normal(n)
    spec = np.fft.rfft(white)
    freqs = np.fft.rfftfreq(n, 1/sr)
    spec[(freqs < lo) | (freqs > hi)] = 0
    return (amp * np.fft.irfft(spec, n)).astype(np.float32)


def _transient_low(dur=1.0, sr=SR):
    """kick-подобный: НЧ-транзиент с быстрым спадом."""
    t = np.arange(int(dur*sr))/sr
    env = np.exp(-t*25)
    sig = 0.6 * env * np.sin(2*np.pi*60*t)
    return sig.astype(np.float32)


# ── Признаки разделяют группы ──────────────────────────────────

def test_tone_is_tonal_noise_is_flat():
    tone_f = extract_features(_tone(440), SR)
    noise_f = extract_features(_noise_band(200, 8000), SR)
    assert tone_f.harmonicity > noise_f.harmonicity
    assert tone_f.flatness < noise_f.flatness


def test_centroid_tracks_frequency_content():
    low = extract_features(_tone(80), SR)
    high = extract_features(_noise_band(6000, 16000), SR)
    assert low.centroid_hz < 500
    assert high.centroid_hz > 4000


def test_transient_has_high_crest():
    kick = extract_features(_transient_low(), SR)
    steady = extract_features(_tone(440), SR)
    assert kick.crest_db > steady.crest_db


# ── Классификация групп ────────────────────────────────────────

def test_low_transient_classified_as_low_percussion():
    r = classify_audio(_transient_low(), SR)
    assert r.instrument in ('kick', 'tom', 'bass'), r.to_dict()


def test_high_noise_classified_as_cymbals_or_hihat():
    r = classify_audio(_noise_band(6000, 18000), SR)
    assert r.instrument in ('hihat', 'cymbals', 'overheads'), r.to_dict()


def test_low_tone_classified_as_bass():
    r = classify_audio(_tone(70), SR)
    assert r.instrument in ('bass', 'kick'), r.to_dict()


def test_mid_tone_classified_as_tonal_instrument():
    # тон 500 Гц с гармониками → тональный средне-частотный (vocal/guitar/keys)
    r = classify_audio(_tone(500), SR)
    assert r.instrument in ('leadVocal', 'guitar', 'keys'), r.to_dict()


def test_result_serializable_and_confidence_bounded():
    r = classify_audio(_tone(440), SR)
    d = r.to_dict()
    assert 0.0 <= r.confidence <= 1.0
    assert 'instrument' in d and 'centroid_hz' in d
    assert set(r.scores.keys())  # непустые скоры


def test_silence_does_not_crash():
    r = classify_audio(np.zeros(SR, dtype=np.float32), SR)
    assert r.instrument in r.scores
