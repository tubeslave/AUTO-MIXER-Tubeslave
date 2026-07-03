"""Capability map loading and parameter checks for Mixing Station profiles."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import re
from typing import Any, Dict, Iterable, Optional

from .config import load_yaml_file, normalize_console_profile, resolve_repo_path


@dataclass(frozen=True)
class CapabilityStatus:
    """Capability lookup result for one correction parameter."""

    parameter: str
    supported: bool
    visualization: bool = True
    read_only: bool = False
    live_control: bool = False
    forbidden_live: bool = False
    needs_discovery: bool = False
    reason: str = ""

    @property
    def can_write_visualization(self) -> bool:
        return self.supported and self.visualization and not self.read_only

    @property
    def can_live_control(self) -> bool:
        return self.can_write_visualization and self.live_control and not self.forbidden_live


class CapabilityMap:
    """Profile-specific capability metadata."""

    def __init__(self, profile: str, parameters: Dict[str, Dict[str, Any]]):
        self.profile = normalize_console_profile(profile)
        self.parameters = parameters

    @classmethod
    def from_file(cls, path: str | Path) -> "CapabilityMap":
        data = load_yaml_file(path)
        profile = normalize_console_profile(str(data.get("profile", "wing_rack")))
        parameters = dict(data.get("parameters") or {})
        return cls(profile=profile, parameters=parameters)

    def status_for(self, parameter: str) -> CapabilityStatus:
        """Return the most specific matching status for a parameter."""
        entry = self._lookup(parameter)
        if entry is None:
            return CapabilityStatus(
                parameter=parameter,
                supported=False,
                visualization=False,
                reason="parameter not present in capability map",
            )
        supported = bool(entry.get("supported", False))
        reason = str(entry.get("reason", ""))
        return CapabilityStatus(
            parameter=parameter,
            supported=supported,
            visualization=bool(entry.get("visualization", supported)),
            read_only=bool(entry.get("read_only", False)),
            live_control=bool(entry.get("live_control", False)),
            forbidden_live=bool(entry.get("forbidden_live", False)),
            needs_discovery=bool(entry.get("needs_discovery", False)),
            reason=reason,
        )

    def supported_parameters(self) -> Iterable[str]:
        """Return parameter names that are not explicitly unsupported."""
        for name, entry in self.parameters.items():
            if bool(entry.get("supported", False)):
                yield name

    def _lookup(self, parameter: str) -> Optional[Dict[str, Any]]:
        if parameter in self.parameters:
            return self.parameters[parameter]
        for pattern, entry in self.parameters.items():
            if "{" in pattern and _template_matches(pattern, parameter):
                return entry
            if "*" in pattern and re.fullmatch(pattern.replace(".", r"\.").replace("*", ".*"), parameter):
                return entry
        return None


def load_capability_map(path: str | Path) -> CapabilityMap:
    """Load capability map from a YAML file."""
    return CapabilityMap.from_file(resolve_repo_path(path))


def _template_matches(template: str, parameter: str) -> bool:
    escaped = re.escape(template)
    escaped = escaped.replace(r"\{send_index\}", r"(?P<send_index>\d+)")
    escaped = escaped.replace(r"\{band\}", r"(?P<band>\d+)")
    return re.fullmatch(escaped, parameter) is not None
