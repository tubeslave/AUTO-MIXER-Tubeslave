"""Virtual mixer interface for offline sandbox rendering."""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

from .action_schema import CandidateActionSet


class VirtualMixer(ABC):
    """Headless offline mixer contract. It never sends OSC/MIDI."""

    @abstractmethod
    def load_project(self, multitrack_dir: str | Path, channel_map: dict[str, Any] | None = None) -> dict[str, Any]:
        ...

    @abstractmethod
    def render(self, actions: CandidateActionSet, output_path: str | Path) -> dict[str, Any]:
        ...

    @abstractmethod
    def export_state(self) -> dict[str, Any]:
        ...

    @abstractmethod
    def import_state(self, state: dict[str, Any]) -> None:
        ...
