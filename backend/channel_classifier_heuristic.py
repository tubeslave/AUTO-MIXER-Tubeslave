"""
Эвристический классификатор каналов на акустических признаках (MIR).

Работает БЕЗ ML-весов и без librosa — только numpy/scipy. Предназначен
как надёжный fallback, когда PyTorch-классификатор (ml/channel_classifier.py,
25 классов) недоступен из-за отсутствия весов/torch.

Различает основные группы источников по различимым без обучения признакам:
спектральный центроид, распределение энергии по полосам, crest factor,
zero-crossing rate, спектральная плоскостность (tonality), rolloff,
гармоничность (наличие чёткой f0).

Возвращает тип инструмента, совместимый с
BalanceProfile.instrument_target_lufs (kick/snare/tom/hihat/cymbals/
bass/leadVocal/guitar/keys/overheads), и confidence 0..1.

Точность заведомо ниже обученной CNN, но НАМНОГО выше центроид-only
заглушки: она использует 7 признаков и разделяет перкуссию/тон/шум.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Tuple

import numpy as np

# Границы полос (Гц) — согласованы с band_energy в SpectralAnalyzer
BANDS = {
    'sub': (20, 60),
    'bass': (60, 250),
    'low_mid': (250, 500),
    'mid': (500, 2000),
    'high_mid': (2000, 4000),
    'high': (4000, 8000),
    'air': (8000, 20000),
}

# Классы, которые эвристика реально различает (совместимы с профилем баланса)
CLASSES = [
    'kick', 'bass', 'snare', 'tom', 'hihat', 'cymbals',
    'overheads', 'leadVocal', 'guitar', 'keys',
]


@dataclass
class ChannelFeatures:
    """Акустические признаки канала."""
    centroid_hz: float
    rolloff_hz: float
    crest_db: float
    zcr: float
    flatness: float          # 0 = чистый тон, 1 = белый шум
    harmonicity: float       # 0..1, наличие чёткой основной частоты
    band_ratios: Dict[str, float] = field(default_factory=dict)
    rms_db: float = -60.0


@dataclass
class ClassificationResult:
    instrument: str
    confidence: float
    features: ChannelFeatures
    scores: Dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            'instrument': self.instrument,
            'confidence': round(self.confidence, 3),
            'centroid_hz': round(self.features.centroid_hz, 0),
            'crest_db': round(self.features.crest_db, 1),
            'flatness': round(self.features.flatness, 3),
            'harmonicity': round(self.features.harmonicity, 3),
        }


def extract_features(audio: np.ndarray, sample_rate: int = 48000) -> ChannelFeatures:
    """Извлечь акустические признаки из моно-сигнала."""
    audio = np.asarray(audio, dtype=np.float64).flatten()
    n = len(audio)
    if n < 2048:
        audio = np.pad(audio, (0, 2048 - n))
        n = len(audio)

    # RMS / crest
    rms = float(np.sqrt(np.mean(audio ** 2)) + 1e-12)
    peak = float(np.max(np.abs(audio)) + 1e-12)
    crest_db = 20.0 * np.log10(peak / rms)
    rms_db = 20.0 * np.log10(rms)

    # Zero-crossing rate (шумность/высокочастотность)
    zcr = float(np.mean(np.abs(np.diff(np.sign(audio))) > 0))

    # Спектр (усреднённый по окнам для устойчивости)
    win = 4096
    hop = 2048
    mags = []
    window = np.hanning(win)
    for i in range(0, max(1, n - win), hop):
        seg = audio[i:i + win]
        if len(seg) < win:
            seg = np.pad(seg, (0, win - len(seg)))
        mags.append(np.abs(np.fft.rfft(seg * window)))
    spec = np.mean(mags, axis=0) if mags else np.abs(np.fft.rfft(audio[:win] * window))
    freqs = np.fft.rfftfreq(win, 1.0 / sample_rate)
    power = spec ** 2
    total_power = float(np.sum(power)) + 1e-12

    # Спектральный центроид
    centroid = float(np.sum(freqs * power) / total_power)

    # Спектральный rolloff (85% энергии)
    cumulative = np.cumsum(power)
    rolloff_idx = int(np.searchsorted(cumulative, 0.85 * total_power))
    rolloff = float(freqs[min(rolloff_idx, len(freqs) - 1)])

    # Спектральная плоскостность (geometric/arithmetic mean) — тональность
    pos = spec[spec > 1e-10]
    if len(pos) > 0:
        flatness = float(np.exp(np.mean(np.log(pos))) / (np.mean(pos) + 1e-12))
    else:
        flatness = 1.0
    flatness = min(1.0, max(0.0, flatness))

    # Энергия по полосам (доля от общей)
    band_ratios = {}
    for name, (lo, hi) in BANDS.items():
        mask = (freqs >= lo) & (freqs < hi)
        band_ratios[name] = float(np.sum(power[mask]) / total_power)

    # Гармоничность: отношение энергии на пиках-гармониках к общей.
    # Ищем основную частоту по автокорреляции спектра тона.
    harmonicity = _estimate_harmonicity(audio, sample_rate)

    return ChannelFeatures(
        centroid_hz=centroid,
        rolloff_hz=rolloff,
        crest_db=crest_db,
        zcr=zcr,
        flatness=flatness,
        harmonicity=harmonicity,
        band_ratios=band_ratios,
        rms_db=rms_db,
    )


def _estimate_harmonicity(audio: np.ndarray, sample_rate: int) -> float:
    """Оценка тональности через нормированную автокорреляцию.

    Тональный сигнал (вокал, бас, гитара) имеет выраженный пик автокорреляции
    на периоде основной частоты; перкуссия/шум — нет.
    """
    seg = audio[:min(len(audio), sample_rate // 4)]  # до 250 мс
    seg = seg - np.mean(seg)
    if np.sum(seg ** 2) < 1e-9:
        return 0.0
    corr = np.correlate(seg, seg, mode='full')
    corr = corr[len(corr) // 2:]
    corr0 = corr[0] + 1e-12
    # Диапазон f0: 50-1000 Гц → лаги
    lag_min = sample_rate // 1000
    lag_max = min(len(corr) - 1, sample_rate // 50)
    if lag_max <= lag_min:
        return 0.0
    peak = float(np.max(corr[lag_min:lag_max]) / corr0)
    return min(1.0, max(0.0, peak))


def classify(features: ChannelFeatures) -> ClassificationResult:
    """Rule-based классификация по признакам. Возвращает лучший класс + confidence."""
    f = features
    br = f.band_ratios
    low_energy = br.get('sub', 0) + br.get('bass', 0)
    high_energy = br.get('high', 0) + br.get('air', 0)
    mid_energy = br.get('low_mid', 0) + br.get('mid', 0)

    scores: Dict[str, float] = {c: 0.0 for c in CLASSES}

    # --- Перкуссия vs тон: crest factor + flatness + harmonicity ---
    percussive = f.crest_db > 12.0
    tonal = f.harmonicity > 0.35 and f.flatness < 0.3
    noisy = f.flatness > 0.35

    # kick: низ, транзиент, не тональный
    scores['kick'] = (
        low_energy * 2.0
        + (1.0 if f.centroid_hz < 200 else 0.0)
        + (0.8 if percussive else 0.0)
        + br.get('sub', 0) * 2.0
        - high_energy
    )
    # bass: низ, тональный (гармоники), устойчивый
    scores['bass'] = (
        low_energy * 2.0
        + (1.2 if tonal else 0.0)
        + (0.6 if f.centroid_hz < 400 else 0.0)
        - (0.8 if percussive else 0.0)
        - high_energy
    )
    # snare: середина, шумовой хвост, транзиент
    scores['snare'] = (
        mid_energy * 1.5
        + (1.0 if percussive else 0.0)
        + (0.8 if 400 < f.centroid_hz < 2500 else 0.0)
        + (0.5 if noisy else 0.0)
        - low_energy
    )
    # tom: низ-середина, транзиент, немного тональный
    scores['tom'] = (
        br.get('low_mid', 0) * 2.0 + br.get('bass', 0)
        + (0.7 if percussive else 0.0)
        + (0.6 if 150 < f.centroid_hz < 800 else 0.0)
    )
    # hihat: верх, шумный, высокий ZCR
    scores['hihat'] = (
        high_energy * 1.5
        + (1.5 if f.centroid_hz > 5000 else 0.0)
        + f.zcr * 2.0
        + (0.8 if noisy else 0.0)
        - low_energy * 2.0
    )
    # cymbals/overheads: очень широкий верх, шумный, sustained
    scores['cymbals'] = (
        high_energy * 1.3 + br.get('high_mid', 0)
        + (1.2 if f.centroid_hz > 4000 else 0.0)
        + (0.8 if noisy else 0.0)
        - low_energy * 1.5
    )
    scores['overheads'] = (
        high_energy + mid_energy * 0.5
        + (0.8 if f.centroid_hz > 3000 else 0.0)
        + (0.5 if noisy else 0.0)
        - low_energy
    )
    # leadVocal: середина, форманты, тональный, умеренный crest
    scores['leadVocal'] = (
        mid_energy * 1.5
        + (1.5 if tonal else 0.0)
        + (1.0 if 500 < f.centroid_hz < 2500 else 0.0)
        + (0.5 if 6 < f.crest_db < 16 else 0.0)
        + f.harmonicity * 1.5
        - (0.5 if percussive else 0.0)
    )
    # guitar: широкая середина, тональный
    scores['guitar'] = (
        mid_energy * 1.3 + br.get('high_mid', 0) * 0.8
        + (1.0 if tonal else 0.0)
        + (0.8 if 800 < f.centroid_hz < 3500 else 0.0)
        + f.harmonicity
    )
    # keys/piano: широкий спектр, тональный, сустейн
    scores['keys'] = (
        mid_energy + br.get('high_mid', 0) * 0.6 + br.get('bass', 0) * 0.5
        + (0.9 if tonal else 0.0)
        + f.harmonicity * 0.8
    )

    best = max(scores, key=scores.get)
    # confidence: разрыв между лучшим и вторым, нормированный
    sorted_scores = sorted(scores.values(), reverse=True)
    top, second = sorted_scores[0], sorted_scores[1] if len(sorted_scores) > 1 else 0.0
    gap = top - second
    confidence = float(min(1.0, max(0.0, gap / (abs(top) + 1e-6))))

    return ClassificationResult(
        instrument=best,
        confidence=confidence,
        features=features,
        scores={k: round(v, 3) for k, v in scores.items()},
    )


def classify_audio(audio: np.ndarray, sample_rate: int = 48000) -> ClassificationResult:
    """Удобная обёртка: признаки + классификация из сырого аудио."""
    return classify(extract_features(audio, sample_rate))
