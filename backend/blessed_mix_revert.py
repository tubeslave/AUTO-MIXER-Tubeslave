"""Blessed mix: параметрический снимок и план REVERT (Р10, совет 2026-07-29).

Оператор в конце саундчека «благословляет» микс — система снимает
ПАРАМЕТРИЧЕСКИЙ снимок (фейдеры/мьюты/тримы/DCA) как точку возврата.
REVERT в концерте — возврат к blessed-состоянию bounded-РАМПОЙ через
gated executor (НЕ мьют мастера и НЕ snapshot GO: GO перезаписывает всё
и официального адреса SAVE у WING нет — резолюция Р10).

Модуль — чистые вычисления (снимок + план шагов); физическое исполнение —
существующий supervised-путь. Никаких прямых записей в пульт отсюда.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

# Рампа возврата: шаг ≤ 1.5 дБ (неслышимый в музыке), интервал ≥ 0.15 с
# (больше интервала OSC-троттлера ~10 Гц)
REVERT_FADER_STEP_DB = 1.5
REVERT_STEP_INTERVAL_S = 0.15


@dataclass
class BlessedMixSnapshot:
    captured_at: float
    channels: Dict[int, Dict[str, Any]] = field(default_factory=dict)
    dca: Dict[int, float] = field(default_factory=dict)
    buses: Dict[int, float] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "captured_at": self.captured_at,
            "channels": {str(k): dict(v) for k, v in self.channels.items()},
            "dca": {str(k): v for k, v in self.dca.items()},
            "buses": {str(k): v for k, v in self.buses.items()},
        }


def capture_blessed_snapshot(
    mixer_client: Any,
    channels: List[int],
    dca_ids: Optional[List[int]] = None,
    bus_ids: Optional[List[int]] = None,
) -> BlessedMixSnapshot:
    """Снять параметрическую точку возврата с фактического состояния пульта.

    Fader/mute стримятся /xremote (пассивный кэш честен); trim читается
    активным GET (не стрится — правило проекта). Канал без известного
    значения в снимок НЕ входит (никаких подстановок −144: правило
    «current неизвестен → не писать» действует и на снимок).
    """
    snap = BlessedMixSnapshot(captured_at=time.time())
    for ch in channels:
        entry: Dict[str, Any] = {}
        fader = _safe(lambda: mixer_client.get_fader(ch))
        if fader is not None:
            entry["fader_db"] = float(fader)
        mute = _safe(lambda: mixer_client.get_channel_mute(ch))
        if mute is not None:
            entry["muted"] = int(mute)
        trim = _safe(lambda: mixer_client.get_channel_gain(ch))
        if trim is not None:
            entry["trim_db"] = float(trim)
        if entry:
            snap.channels[ch] = entry
        else:
            logger.warning(
                f"Blessed snapshot: ch {ch} без единого известного "
                "значения — канал не вошёл в точку возврата"
            )
    for dca in dca_ids or []:
        value = _safe(lambda: mixer_client.get_dca_fader(dca))
        if value is not None:
            snap.dca[dca] = float(value)
    for bus in bus_ids or []:
        value = _safe(lambda: mixer_client.get_bus_fader(bus))
        if value is not None:
            snap.buses[bus] = float(value)
    logger.info(
        f"Blessed mix snapshot: {len(snap.channels)} каналов, "
        f"{len(snap.dca)} DCA, {len(snap.buses)} шин"
    )
    return snap


def build_revert_plan(
    snapshot: BlessedMixSnapshot,
    mixer_client: Any,
    *,
    fader_step_db: float = REVERT_FADER_STEP_DB,
) -> List[Dict[str, Any]]:
    """План возврата к blessed-состоянию: список шагов для gated executor.

    Каждый шаг — АБСОЛЮТНАЯ цель (идемпотентность: повторный вызов после
    частичного исполнения строит план от фактического состояния). Фейдеры —
    рампой ≤ fader_step_db за шаг; mute/trim — одним шагом (mute мгновенен
    по природе; трим в REVERT допустим — это операторская команда, не
    автокоррекция). Канал с неизвестным current пропускается с пометкой.
    """
    plan: List[Dict[str, Any]] = []
    step = abs(float(fader_step_db)) or REVERT_FADER_STEP_DB

    def _ramp(kind: str, ident: int, target: float, current: Optional[float]):
        if current is None:
            plan.append({
                "action": "skip", "kind": kind, "id": ident,
                "reason": "current_unknown",
            })
            return
        current_f = float(current)
        target_f = float(target)
        while abs(target_f - current_f) > 1e-6:
            delta = max(-step, min(step, target_f - current_f))
            current_f = round(current_f + delta, 4)
            plan.append({
                "action": f"{kind}_fader", "id": ident,
                "target_db": current_f,
                "interval_s": REVERT_STEP_INTERVAL_S,
            })

    for ch, entry in snapshot.channels.items():
        if "fader_db" in entry:
            _ramp("channel", ch, entry["fader_db"],
                  _safe(lambda: mixer_client.get_fader(ch)))
        if "muted" in entry:
            current_mute = _safe(lambda: mixer_client.get_channel_mute(ch))
            if current_mute is None or int(current_mute) != int(entry["muted"]):
                plan.append({
                    "action": "channel_mute", "id": ch,
                    "muted": int(entry["muted"]),
                })
        if "trim_db" in entry:
            current_trim = _safe(lambda: mixer_client.get_channel_gain(ch))
            if current_trim is None or \
                    abs(float(current_trim) - entry["trim_db"]) > 0.05:
                plan.append({
                    "action": "channel_trim", "id": ch,
                    "target_db": float(entry["trim_db"]),
                })
    for dca, target in snapshot.dca.items():
        _ramp("dca", dca, target, _safe(lambda: mixer_client.get_dca_fader(dca)))
    for bus, target in snapshot.buses.items():
        _ramp("bus", bus, target, _safe(lambda: mixer_client.get_bus_fader(bus)))
    return plan


def _safe(callback):
    try:
        return callback()
    except Exception:
        return None
