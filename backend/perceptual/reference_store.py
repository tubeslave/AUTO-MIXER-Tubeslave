"""Small reference embedding store for perceptual reward experiments."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Iterable, Optional

import numpy as np


class ReferenceStore:
    """In-memory reference embeddings keyed by channel/instrument/context."""

    def __init__(self, path: Optional[str | Path] = None):
        self.path = Path(path) if path else None
        self._references: Dict[str, np.ndarray] = {}
        if self.path and self.path.exists():
            self.load(self.path)

    @staticmethod
    def make_key(
        channel_name: Optional[str] = None,
        instrument_type: Optional[str] = None,
        context_id: Optional[str] = None,
    ) -> str:
        parts = [
            str(context_id or "default").strip().lower(),
            str(instrument_type or "unknown").strip().lower(),
            str(channel_name or "mix").strip().lower(),
        ]
        return "::".join(parts)

    def put(self, key: str, embedding: Iterable[float] | np.ndarray) -> None:
        self._references[str(key)] = np.asarray(embedding, dtype=np.float32).reshape(-1)

    def get(self, key: str) -> Optional[np.ndarray]:
        value = self._references.get(str(key))
        if value is None:
            return None
        return value.copy()

    def get_for_context(
        self,
        channel_name: Optional[str] = None,
        instrument_type: Optional[str] = None,
        context_id: Optional[str] = None,
    ) -> Optional[np.ndarray]:
        return self.get(self.make_key(channel_name, instrument_type, context_id))

    def to_dict(self) -> Dict[str, list]:
        return {key: value.astype(float).tolist() for key, value in self._references.items()}

    def load(self, path: str | Path) -> None:
        path = Path(path)
        with path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle) or {}
        self._references = {
            str(key): np.asarray(value, dtype=np.float32).reshape(-1)
            for key, value in dict(payload).items()
        }
        self.path = path

    def save(self, path: Optional[str | Path] = None) -> Path:
        target = Path(path) if path else self.path
        if target is None:
            raise ValueError("ReferenceStore.save requires a path")
        target.parent.mkdir(parents=True, exist_ok=True)
        with target.open("w", encoding="utf-8") as handle:
            json.dump(self.to_dict(), handle, ensure_ascii=True, indent=2)
        self.path = target
        return target
