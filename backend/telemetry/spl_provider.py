"""SPL telemetry providers.

Providers are read-only data sources. They do not communicate with the mixer.
"""

from __future__ import annotations

import csv
import json
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Dict, Iterator, Optional

try:
    from .schemas import SPLTelemetry, timestamp_ms
except ImportError:  # pragma: no cover - direct PYTHONPATH=backend compatibility
    from telemetry.schemas import SPLTelemetry, timestamp_ms


class BaseSPLProvider(ABC):
    @abstractmethod
    def read_once(self, *, session_id: str) -> Optional[SPLTelemetry]:
        """Return one SPL row or None when unavailable."""


class MockSPLProvider(BaseSPLProvider):
    def __init__(self, *, laeq_1s: float = 85.0, lcpeak: float = 102.0, lafmax: float = 90.0):
        self.laeq_1s = laeq_1s
        self.lcpeak = lcpeak
        self.lafmax = lafmax

    def read_once(self, *, session_id: str) -> SPLTelemetry:
        return SPLTelemetry(
            timestamp_ms=timestamp_ms(),
            session_id=session_id,
            laeq_1s=self.laeq_1s,
            lcpeak=self.lcpeak,
            lafmax=self.lafmax,
            source="mock",
            calibration_notes="mock provider; not calibrated",
        )


class FileSPLProvider(BaseSPLProvider):
    """Read SPL rows from CSV or JSONL.

    CSV headers may include: timestamp_ms, laeq_1s, lcpeak, lafmax, source,
    calibration_notes. JSONL rows use the same keys.
    """

    def __init__(self, file_path: str | Path):
        self.file_path = Path(file_path)
        self._rows = self._iter_rows()

    def read_once(self, *, session_id: str) -> Optional[SPLTelemetry]:
        try:
            row = next(self._rows)
        except StopIteration:
            return None
        return SPLTelemetry(
            timestamp_ms=int(row.get("timestamp_ms") or timestamp_ms()),
            session_id=session_id,
            laeq_1s=row.get("laeq_1s"),
            lcpeak=row.get("lcpeak"),
            lafmax=row.get("lafmax"),
            source=str(row.get("source") or f"file:{self.file_path.name}"),
            calibration_notes=str(row.get("calibration_notes") or ""),
        )

    def _iter_rows(self) -> Iterator[Dict[str, Any]]:
        if not self.file_path.exists():
            return iter(())
        if self.file_path.suffix.lower() == ".jsonl":
            return self._iter_jsonl()
        return self._iter_csv()

    def _iter_jsonl(self) -> Iterator[Dict[str, Any]]:
        with self.file_path.open(encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                value = json.loads(line)
                if isinstance(value, dict):
                    yield value

    def _iter_csv(self) -> Iterator[Dict[str, Any]]:
        with self.file_path.open(newline="", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            for row in reader:
                yield dict(row)
