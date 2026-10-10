"""File-backed mixing knowledge for v2 decisions."""

from __future__ import annotations

from collections.abc import Mapping
import json
from pathlib import Path
import re
from typing import Any, Dict, Iterable


DEFAULT_KNOWLEDGE_PATH = Path(__file__).with_name("mixing_knowledge_base.json")

ROLE_ALIASES = {
    "backvocal": "backing_vocal",
    "back_vocal": "backing_vocal",
    "bgv": "backing_vocal",
    "electricguitar": "electric_guitar",
    "guitar": "electric_guitar",
    "leadguitar": "electric_guitar",
    "rhythmguitar": "electric_guitar",
    "hihat": "overheads",
    "hi_hat": "overheads",
    "ride": "overheads",
    "cymbals": "overheads",
    "oh": "overheads",
    "overhead": "overheads",
    "tom": "toms",
    "rack_tom": "toms",
    "floor_tom": "toms",
    "lead": "lead_vocal",
    "leadvocal": "lead_vocal",
    "vocal": "lead_vocal",
    "synth": "keys",
    "piano": "keys",
    "organ": "keys",
    "pad": "keys",
    "mix": "master_bus",
    "mix_bus": "master_bus",
    "master": "master_bus",
    "main": "master_bus",
}


def normalize_role(role: str | None) -> str:
    """Normalize legacy and camelCase source labels to knowledge categories."""
    text = str(role or "unknown").strip()
    if not text:
        return "unknown"
    text = re.sub(r"([a-z0-9])([A-Z])", r"\1_\2", text)
    text = re.sub(r"[^a-zA-Z0-9]+", "_", text).strip("_").lower()
    return ROLE_ALIASES.get(text, text)


class MixingKnowledgeBase:
    """Load and query the v2 mixing knowledge file.

    The data is a decision reference, not a preset bank. Decision Engine still
    needs analyzer evidence and critic feedback before it proposes an action.
    """

    def __init__(self, payload: Mapping[str, Any] | None = None):
        self.payload = dict(payload or {})
        self.categories: Dict[str, Dict[str, Any]] = {
            normalize_role(key): dict(value or {})
            for key, value in dict(self.payload.get("categories", {})).items()
        }

    @classmethod
    def load(cls, path: str | Path | None = None) -> "MixingKnowledgeBase":
        resolved = Path(path).expanduser() if path else DEFAULT_KNOWLEDGE_PATH
        payload = json.loads(resolved.read_text(encoding="utf-8"))
        return cls(payload)

    @classmethod
    def from_mapping(cls, payload: Mapping[str, Any]) -> "MixingKnowledgeBase":
        if "categories" in payload:
            return cls(payload)
        return cls({"categories": dict(payload)})

    def category_for(self, role: str | None) -> Dict[str, Any]:
        normalized = normalize_role(role)
        return dict(self.categories.get(normalized, self.categories.get("master_bus", {})))

    def allowed_actions_for(self, role: str | None) -> set[str]:
        entry = self.category_for(role)
        return {str(action) for action in entry.get("allowed_actions", [])}

    def risky_actions_for(self, role: str | None) -> set[str]:
        entry = self.category_for(role)
        return {str(action) for action in entry.get("risky_actions", [])}

    def live_safe_limits_for(self, role: str | None) -> Dict[str, float]:
        limits = self.category_for(role).get("live_safe_limits", {})
        return {str(key): float(value) for key, value in dict(limits).items()}

    def categories_for_audit(self) -> Iterable[str]:
        return sorted(self.categories)
