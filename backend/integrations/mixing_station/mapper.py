"""Map normalized AutomixCorrection objects to Mixing Station dataPaths."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import re
from typing import Any, Dict, Optional

from .config import load_yaml_file, normalize_console_profile, resolve_repo_path
from .models import AutomixCorrection, MixingStationCommand


@dataclass(frozen=True)
class MappingResult:
    """Result of mapping an AutomixCorrection to a command."""

    command: Optional[MixingStationCommand]
    success: bool
    error: Optional[str] = None
    needs_discovery: bool = False


class MixingStationMapper:
    """Profile-aware dataPath mapper."""

    def __init__(self, profile: str, entries: Dict[str, Dict[str, Any]], default_transport: str):
        self.profile = normalize_console_profile(profile)
        self.entries = entries
        self.default_transport = default_transport

    @classmethod
    def from_file(cls, path: str | Path, *, default_transport: str = "websocket"):
        data = load_yaml_file(path)
        profile = normalize_console_profile(str(data.get("profile", "wing_rack")))
        entries = dict(data.get("parameters") or {})
        return cls(profile=profile, entries=entries, default_transport=default_transport)

    def map(self, correction: AutomixCorrection) -> MappingResult:
        """Map a correction to a concrete Mixing Station dataPath."""
        entry, captures = self._lookup(correction.parameter, correction.strip_type)
        if entry is None:
            return MappingResult(
                command=None,
                success=False,
                error=(
                    f"No Mixing Station mapping for {correction.console_profile} "
                    f"{correction.strip_type}.{correction.parameter}"
                ),
            )
        if bool(entry.get("needs_discovery", False)):
            return MappingResult(
                command=None,
                success=False,
                needs_discovery=True,
                error=(
                    f"Mapping for {correction.parameter} requires discovery in "
                    "Mixing Station API Explorer"
                ),
            )

        template = str(entry.get("data_path", "")).strip()
        if not template or "TODO" in template:
            return MappingResult(
                command=None,
                success=False,
                needs_discovery=True,
                error=f"Mapping for {correction.parameter} has no verified dataPath",
            )

        channel_index = int(correction.channel_index)
        format_vars = {
            "channel_index": channel_index,
            "channel_number": channel_index + 1,
            "channel_absolute_index": _absolute_channel_index(correction.strip_type, channel_index),
            "strip_type": correction.strip_type,
            **captures,
        }
        if "band" in captures:
            format_vars["band_index"] = max(0, int(captures["band"]) - 1)
        if "send_index" in captures:
            format_vars["send_index_zero"] = max(0, int(captures["send_index"]) - 1)
            format_vars["send_index0"] = format_vars["send_index_zero"]
            format_vars["send_index_minus_1"] = format_vars["send_index_zero"]
        try:
            data_path = template.format(**format_vars)
        except KeyError as exc:
            return MappingResult(
                command=None,
                success=False,
                error=f"Mapping template missing variable: {exc}",
            )

        value = _coerce_value(correction.value, entry)
        command = MixingStationCommand(
            transport=str(entry.get("transport", self.default_transport)),
            data_path=data_path,
            value=value,
            value_format=str(entry.get("format", "plain")),
            method=str(entry.get("method", "SET")),
            parameter=correction.parameter,
        )
        return MappingResult(command=command, success=True)

    def _lookup(self, parameter: str, strip_type: str) -> tuple[Optional[Dict[str, Any]], Dict[str, Any]]:
        keyed_name = f"{strip_type}.{parameter}"
        for name in (keyed_name, parameter):
            if name in self.entries:
                return self.entries[name], {}

        for template, entry in self.entries.items():
            if not template.startswith(f"{strip_type}."):
                continue
            stripped = template.split(".", 1)[1]
            captures = _match_template(stripped, parameter)
            if captures is not None:
                return entry, captures

        for template, entry in self.entries.items():
            if template.startswith(f"{strip_type}."):
                continue
            captures = _match_template(template, parameter)
            if captures is not None:
                return entry, captures
        return None, {}


def load_mapper(path: str | Path, *, default_transport: str = "websocket") -> MixingStationMapper:
    """Load mapper config from YAML."""
    return MixingStationMapper.from_file(resolve_repo_path(path), default_transport=default_transport)


def _match_template(template: str, parameter: str) -> Optional[Dict[str, Any]]:
    if "{" not in template:
        return None
    escaped = re.escape(template)
    escaped = escaped.replace(r"\{send_index\}", r"(?P<send_index>\d+)")
    escaped = escaped.replace(r"\{band\}", r"(?P<band>\d+)")
    match = re.fullmatch(escaped, parameter)
    if not match:
        return None
    return {key: int(value) for key, value in match.groupdict().items()}


def _absolute_channel_index(strip_type: str, channel_index: int) -> int:
    offsets = {
        "input": 0,
        "aux": 40,
        "bus": 48,
        "matrix": 64,
        "main": 72,
        "dca": 76,
    }
    return int(offsets.get(strip_type, 0)) + int(channel_index)


def _coerce_value(value: Any, entry: Dict[str, Any]) -> Any:
    if bool(entry.get("invert_bool", False)):
        value = not bool(value)

    enum_map = entry.get("enum_map") or {}
    if isinstance(enum_map, dict) and isinstance(value, str):
        mapped = enum_map.get(value) or enum_map.get(value.lower())
        if mapped is not None:
            value = mapped

    value_type = str(entry.get("value_type", "")).lower()
    if value_type == "bool":
        coerced = bool(value)
    elif value_type == "int":
        coerced = int(value)
    elif value_type == "float":
        coerced = float(value)
    elif value_type == "string":
        coerced = str(value)
    else:
        coerced = value

    if isinstance(coerced, (int, float)) and not isinstance(coerced, bool):
        coerced = float(coerced) * float(entry.get("value_multiplier", 1.0))
        if value_type == "int":
            coerced = int(round(coerced))

    max_length = entry.get("max_length")
    if isinstance(coerced, str) and max_length:
        coerced = coerced.encode("ascii", errors="ignore").decode("ascii")
        coerced = coerced[: int(max_length)].strip()

    return coerced
