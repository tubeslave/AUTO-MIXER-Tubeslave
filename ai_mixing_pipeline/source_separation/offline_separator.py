"""Demucs/Open-Unmix offline source separation adapter."""

from __future__ import annotations

from pathlib import Path
from typing import Any


class OfflineSourceSeparator:
    """Offline-only source separation facade.

    The adapter intentionally does not run in live threads.  It reports
    unavailability unless an operator has installed a supported separator.
    """

    name = "demucs_or_openunmix"
    role = "offline_source_separator"

    def __init__(self, config: dict[str, Any] | None = None):
        self.config = dict(config or {})
        self.available_backend = self._detect_backend()

    def separate_reference(
        self,
        reference_path: str | Path | None,
        output_dir: str | Path,
    ) -> dict[str, Any]:
        if not bool(self.config.get("enabled", True)):
            return {
                "name": self.name,
                "role": self.role,
                "available": False,
                "participated": True,
                "warnings": ["Source separation disabled in config."],
                "stems": {},
            }
        if reference_path is None:
            return {
                "name": self.name,
                "role": self.role,
                "available": False,
                "participated": True,
                "warnings": ["No reference mix supplied; Demucs/Open-Unmix stage recorded a skipped fallback."],
                "stems": {},
            }
        if not self.available_backend:
            return {
                "name": self.name,
                "role": self.role,
                "available": False,
                "participated": True,
                "warnings": ["Demucs/Open-Unmix unavailable; reference separation skipped."],
                "stems": {},
                "reference_path": str(reference_path),
                "output_dir": str(output_dir),
            }
        return {
            "name": self.name,
            "role": self.role,
            "available": False,
            "participated": True,
            "backend": self.available_backend,
            "warnings": [
                "Source separator detected but automatic invocation is disabled in this safe adapter."
            ],
            "stems": {},
            "reference_path": str(reference_path),
            "output_dir": str(output_dir),
        }

    @staticmethod
    def _detect_backend() -> str:
        for module_name, label in (("demucs", "demucs"), ("openunmix", "openunmix")):
            try:
                __import__(module_name)
                return label
            except Exception:
                continue
        return ""
