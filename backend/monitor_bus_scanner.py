"""Классификатор мониторных шин по именам (Ф10 этап 1, DEC-0010).

Offline-логика: имя шины -> тип мониторинга + уверенность. Съём имён по
OSC и маршрутизация голоса — отдельные этапы за операторским
подтверждением; без подтверждённого маппинга голос никуда не шлётся.

Паттерны из директивы оператора: «IEM» (in-ear), «MON»/«MONITOR»,
имя музыканта; плюс WEDGE и русские варианты.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Dict, List, Mapping, Optional, Sequence

IEM_PATTERNS = (
    r"\biem\b", r"\bin[- ]?ears?\b", r"\bears?\b", r"\bиэм\b", r"\bуши\b",
)
WEDGE_PATTERNS = (r"\bwedge\b", r"\bклин\b", r"\bпрострел\b")
MONITOR_PATTERNS = (r"\bmon(?:itor)?s?\b", r"\bмонитор\w*\b", r"\bмон\b")

CONFIDENCE = {
    "iem": 0.9,
    "wedge": 0.8,
    "monitor": 0.7,
    "musician_only": 0.6,
    "type_plus_musician": 0.95,
}


@dataclass
class BusClassification:
    bus: int
    name: str
    kind: str = "unknown"            # iem | wedge | monitor | unknown
    musician: Optional[str] = None   # совпавшее имя музыканта
    confidence: float = 0.0
    requires_confirmation: bool = True  # всегда: голос без подтверждения не шлётся

    @property
    def is_monitoring(self) -> bool:
        return self.kind != "unknown" or self.musician is not None


def _match_any(patterns: Sequence[str], text: str) -> bool:
    return any(re.search(p, text, flags=re.IGNORECASE) for p in patterns)


def classify_bus_name(
    bus: int,
    name: str,
    musicians: Sequence[str] = (),
) -> BusClassification:
    """Одно имя шины -> классификация. musicians — состав (имена/алиасы)."""
    result = BusClassification(bus=int(bus), name=str(name or ""))
    text = result.name.strip()
    if not text:
        return result

    if _match_any(IEM_PATTERNS, text):
        result.kind = "iem"
    elif _match_any(WEDGE_PATTERNS, text):
        result.kind = "wedge"
    elif _match_any(MONITOR_PATTERNS, text):
        result.kind = "monitor"

    lowered = text.lower()
    for musician in musicians:
        m = str(musician or "").strip()
        if len(m) >= 3 and m.lower() in lowered:
            result.musician = m
            break

    if result.kind != "unknown" and result.musician:
        result.confidence = CONFIDENCE["type_plus_musician"]
    elif result.kind != "unknown":
        result.confidence = CONFIDENCE[result.kind]
    elif result.musician:
        result.confidence = CONFIDENCE["musician_only"]
    return result


def scan_buses(
    bus_names: Mapping[int, str],
    musicians: Sequence[str] = (),
) -> List[BusClassification]:
    """Все шины -> кандидаты мониторинга, отсортированы по уверенности.

    Возвращает ТОЛЬКО кандидатов (kind/musician найдены); полный отчёт —
    забота вызывающего. Каждый кандидат requires_confirmation=True:
    подтверждение маппинга — за оператором во frontend (DEC-0010 §3).
    """
    results = [
        classify_bus_name(bus, name, musicians)
        for bus, name in sorted(bus_names.items())
    ]
    candidates = [r for r in results if r.is_monitoring]
    candidates.sort(key=lambda r: (-r.confidence, r.bus))
    return candidates


def proposal_payload(candidates: Sequence[BusClassification]) -> Dict[str, object]:
    """JSON-совместимый payload для frontend-подтверждения маппинга."""
    return {
        "type": "monitor_bus_mapping_proposal",
        "requires_confirmation": True,
        "auto_routing": False,  # этап 3, класс A — только после разблокировки
        "candidates": [
            {
                "bus": c.bus,
                "name": c.name,
                "kind": c.kind,
                "musician": c.musician,
                "confidence": round(c.confidence, 2),
            }
            for c in candidates
        ],
    }
