"""Санитарный слой входа: клики, клиппинг, дропауты, тишина (Ф8, DEC-0008).

Первая стадия конвейера по DEC-0008: не судить о балансе по битым данным.
Мотив — инцидент 2026-07-29: синхронные полномасштабные глитчи во всех 16
каналах мультитрека SONG REPA утопили рендер до −45 LUFS через
пик-нормализацию (CLD-0018 п.1).

Отличия дефектов по физике:
- клик: изолированный выброс короче ~1 мс, окружение тише на ≥20 дБ,
  часто синхронен по каналам (цифровой глитч интерфейса);
- клиппинг: плато подряд идущих сэмплов у потолка (настоящий перегруз);
- дропаут: пробег нулей внутри звучащего сигнала;
- тишина: канал без полезной энергии целиком.

Только numpy; входы — mono (n,) или multi (n, ch) float массивы.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Sequence

import numpy as np

CLICK_THRESHOLD = 0.95          # |x| выше — кандидат в клик/клиппинг
CLICK_MAX_MS = 1.0              # клик короче этого; длиннее — не клик
CLICK_MERGE_MS = 10.0           # события ближе — одно событие
CLICK_ISOLATION_DB = 12.0       # окружение (медиана) тише пика минимум на столько
# 12 дБ: медиана |sin| = 0.707·amp, поэтому стационарный синус 0.3 против
# клика 1.0 даёт 13.4 дБ — ловится; громкий контекст 0.9 даёт 3.9 дБ —
# музыкальный пик не ловится (запас 8 дБ). Реальная музыка имеет больший
# crest, отрыв клика ещё выше; синхронность по каналам — отдельный сигнал.
CLICK_CONTEXT_MS = 25.0         # окно оценки окружения вокруг события
CLIP_CEILING = 0.999
CLIP_MIN_RUN = 4                # ≥ стольких сэмплов подряд у потолка
DROPOUT_MIN_MS = 2.0
DROPOUT_FLOOR = 1e-9            # «ноль» для дропаута
DROPOUT_CONTEXT_RMS_DB = -60.0  # вокруг дропаута должен быть сигнал
SILENCE_RMS_DB = -70.0
REPAIR_PAD_MS = 3.0
SYNC_MIN_CHANNELS = 3           # клик в стольких каналах одновременно = глитч записи


@dataclass
class ClickEvent:
    start: int
    end: int                    # включительно, сэмплы
    peak_abs: float

    def duration_ms(self, sr: int) -> float:
        return (self.end - self.start + 1) * 1000.0 / sr


@dataclass
class ChannelHealth:
    clicks: List[ClickEvent] = field(default_factory=list)
    clipping_runs: int = 0
    clipped_samples: int = 0
    dropout_runs: int = 0
    silent: bool = False
    rms_db: float = -np.inf

    @property
    def healthy(self) -> bool:
        return not self.clicks and self.clipping_runs == 0 and self.dropout_runs == 0

    @property
    def reasons(self) -> List[str]:
        out = []
        if self.clicks:
            out.append(f"clicks:{len(self.clicks)}")
        if self.clipping_runs:
            out.append(f"clipping_runs:{self.clipping_runs}")
        if self.dropout_runs:
            out.append(f"dropout_runs:{self.dropout_runs}")
        if self.silent:
            out.append("silent")
        return out


def _to_mono_abs(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    if x.ndim > 1:
        return np.abs(x).max(axis=1)
    return np.abs(x)


def detect_clicks(
    x: np.ndarray,
    sr: int,
    threshold: float = CLICK_THRESHOLD,
    max_click_ms: float = CLICK_MAX_MS,
    isolation_db: float = CLICK_ISOLATION_DB,
) -> List[ClickEvent]:
    """Изолированные полномасштабные выбросы (не музыкальные пики).

    Критерии: |x| > threshold, длительность события ≤ max_click_ms, и
    медиана окружения (±CLICK_CONTEXT_MS без самого события) тише пика
    события минимум на isolation_db.
    """
    mono = _to_mono_abs(x)
    hot = np.flatnonzero(mono > threshold)
    if hot.size == 0:
        return []
    merge_gap = max(1, int(CLICK_MERGE_MS * sr / 1000.0))
    events: List[List[int]] = []
    for s in hot:
        if events and s - events[-1][1] <= merge_gap:
            events[-1][1] = int(s)
        else:
            events.append([int(s), int(s)])

    max_len = max(1, int(max_click_ms * sr / 1000.0))
    ctx = max(1, int(CLICK_CONTEXT_MS * sr / 1000.0))
    pad = max(1, int(REPAIR_PAD_MS * sr / 1000.0))
    out: List[ClickEvent] = []
    for a, b in events:
        if (b - a + 1) > max_len:
            continue  # длинное плато — это клиппинг, не клик
        peak = float(mono[a:b + 1].max())
        left = mono[max(0, a - ctx):max(0, a - pad)]
        right = mono[b + pad:b + ctx]
        context = np.concatenate([left, right]) if left.size + right.size else np.array([0.0])
        ctx_med = float(np.median(context))
        if ctx_med <= 0 or 20.0 * np.log10(peak / max(ctx_med, 1e-12)) >= isolation_db:
            out.append(ClickEvent(start=a, end=b, peak_abs=peak))
    return out


def detect_clipping(
    x: np.ndarray,
    ceiling: float = CLIP_CEILING,
    min_run: int = CLIP_MIN_RUN,
) -> tuple[int, int]:
    """Плато у потолка: (число пробегов, всего клиппированных сэмплов)."""
    mono = _to_mono_abs(x)
    at_ceiling = mono >= ceiling
    if not at_ceiling.any():
        return 0, 0
    runs = 0
    clipped = 0
    edges = np.diff(at_ceiling.astype(np.int8))
    starts = np.flatnonzero(edges == 1) + 1
    ends = np.flatnonzero(edges == -1) + 1
    if at_ceiling[0]:
        starts = np.concatenate([[0], starts])
    if at_ceiling[-1]:
        ends = np.concatenate([ends, [len(mono)]])
    for a, b in zip(starts, ends):
        if b - a >= min_run:
            runs += 1
            clipped += int(b - a)
    return runs, clipped


def detect_dropouts(
    x: np.ndarray,
    sr: int,
    min_ms: float = DROPOUT_MIN_MS,
    context_rms_db: float = DROPOUT_CONTEXT_RMS_DB,
) -> int:
    """Пробеги нулей внутри звучащего сигнала (обрывы потока)."""
    mono = _to_mono_abs(x)
    near_zero = mono <= DROPOUT_FLOOR
    if not near_zero.any():
        return 0
    min_run = max(1, int(min_ms * sr / 1000.0))
    ctx = max(min_run, int(50.0 * sr / 1000.0))
    edges = np.diff(near_zero.astype(np.int8))
    starts = np.flatnonzero(edges == 1) + 1
    ends = np.flatnonzero(edges == -1) + 1
    if near_zero[0]:
        starts = np.concatenate([[0], starts])
    if near_zero[-1]:
        ends = np.concatenate([ends, [len(mono)]])
    runs = 0
    for a, b in zip(starts, ends):
        if b - a < min_run:
            continue
        # дропаут только если вокруг был сигнал (иначе это пауза/тишина)
        left = mono[max(0, a - ctx):a]
        right = mono[b:b + ctx]
        around = np.concatenate([left, right])
        if around.size == 0:
            continue
        rms = float(np.sqrt(np.mean(around ** 2)))
        if 20.0 * np.log10(max(rms, 1e-12)) > context_rms_db:
            runs += 1
    return runs


def channel_rms_db(x: np.ndarray) -> float:
    x = np.asarray(x, dtype=np.float64)
    if x.ndim > 1:
        x = x.mean(axis=1)
    rms = float(np.sqrt(np.mean(x ** 2))) if x.size else 0.0
    return 20.0 * np.log10(max(rms, 1e-12))


def analyze_channel(x: np.ndarray, sr: int) -> ChannelHealth:
    health = ChannelHealth()
    health.rms_db = channel_rms_db(x)
    health.silent = health.rms_db < SILENCE_RMS_DB
    if not health.silent:
        health.clicks = detect_clicks(x, sr)
        health.clipping_runs, health.clipped_samples = detect_clipping(x)
        health.dropout_runs = detect_dropouts(x, sr)
    return health


def synchronized_click_groups(
    healths: Sequence[ChannelHealth],
    sr: int,
    min_channels: int = SYNC_MIN_CHANNELS,
    window_ms: float = CLICK_MERGE_MS,
) -> List[List[int]]:
    """Группы кликов, совпадающих по времени в ≥ min_channels каналах.

    Возвращает список групп; группа = отсортированные стартовые сэмплы.
    Синхронный клик = цифровой глитч записи, максимальная уверенность.
    """
    window = max(1, int(window_ms * sr / 1000.0))
    starts: List[int] = []
    for h in healths:
        starts.extend(int(c.start) for c in h.clicks)
    if not starts:
        return []
    starts.sort()
    groups: List[List[int]] = [[starts[0]]]
    for s in starts[1:]:
        if s - groups[-1][-1] <= window:
            groups[-1].append(s)
        else:
            groups.append([s])
    return [g for g in groups if len(g) >= min_channels]


def repair_clicks(
    x: np.ndarray,
    events: Sequence[ClickEvent],
    sr: int,
    pad_ms: float = REPAIR_PAD_MS,
) -> np.ndarray:
    """Линейная интерполяция ±pad вокруг каждого события. Возвращает копию."""
    y = np.array(x, dtype=np.float64, copy=True)
    pad = max(1, int(pad_ms * sr / 1000.0))
    n = y.shape[0]
    for ev in events:
        i0 = max(0, ev.start - pad)
        i1 = min(n, ev.end + 1 + pad)
        if i1 - i0 < 2:
            continue
        if y.ndim > 1:
            for c in range(y.shape[1]):
                lo = y[i0 - 1, c] if i0 > 0 else 0.0
                hi = y[i1, c] if i1 < n else 0.0
                y[i0:i1, c] = np.linspace(lo, hi, i1 - i0)
        else:
            lo = y[i0 - 1] if i0 > 0 else 0.0
            hi = y[i1] if i1 < n else 0.0
            y[i0:i1] = np.linspace(lo, hi, i1 - i0)
    return y


def robust_peak(x: np.ndarray, sr: int) -> float:
    """Пик для нормализации, исключая клики.

    Именно отсутствие этой функции превратило два щелчка в −45 LUFS:
    нормализация обязана игнорировать не-музыкальные выбросы.
    """
    events = detect_clicks(x, sr)
    if not events:
        return float(_to_mono_abs(x).max()) if np.asarray(x).size else 0.0
    repaired = repair_clicks(x, events, sr)
    return float(_to_mono_abs(repaired).max())
