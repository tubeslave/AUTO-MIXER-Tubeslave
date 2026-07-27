"""Safe show-profile loading and validation."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Tuple

import yaml

from .io import InspectorInputError, MAX_INPUT_FILE_BYTES, require_mapping, require_text


@dataclass(frozen=True)
class RequiredDeviceProfile:
    stable_id: str
    kind: str
    name: str


@dataclass(frozen=True)
class RoleProfile:
    name: str
    expected_source: str
    expected_destination: str
    expected_console_channel: str
    forbidden_destinations: Tuple[str, ...]


@dataclass(frozen=True)
class ShowProfile:
    name: str
    sample_rate: int
    required_devices: Tuple[RequiredDeviceProfile, ...]
    roles: Tuple[RoleProfile, ...]
    expected_dante_interface: str
    expected_control_interface: str
    require_wired_dante: bool


def load_show_profile(path: Path) -> ShowProfile:
    candidate = Path(path)
    if candidate.is_symlink():
        raise InspectorInputError("show_profile_symlink_rejected")
    try:
        stat = candidate.stat()
    except FileNotFoundError as exc:
        raise InspectorInputError("show_profile_missing") from exc
    if stat.st_size > MAX_INPUT_FILE_BYTES:
        raise InspectorInputError("show_profile_too_large")
    try:
        raw = yaml.safe_load(candidate.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, yaml.YAMLError) as exc:
        raise InspectorInputError("show_profile_invalid_yaml") from exc
    if not isinstance(raw, dict):
        raise InspectorInputError("show_profile_top_level_object_required")

    profile = require_mapping(raw, "profile", "show_profile")
    name = require_text(profile, "name", "show_profile.profile")
    sample_rate = _positive_int(profile.get("sample_rate"), "profile.sample_rate")

    devices_section = require_mapping(raw, "devices", "show_profile")
    required_rows = devices_section.get("required")
    if not isinstance(required_rows, list) or not required_rows:
        raise InspectorInputError("show_profile.devices.required:list_required")
    required_devices: list[RequiredDeviceProfile] = []
    seen_device_ids: set[str] = set()
    for index, row in enumerate(required_rows):
        if not isinstance(row, dict):
            raise InspectorInputError(f"devices.required[{index}]:object_required")
        stable_id = require_text(row, "id", f"devices.required[{index}]")
        if stable_id in seen_device_ids:
            raise InspectorInputError(f"devices.required:duplicate_id:{stable_id}")
        seen_device_ids.add(stable_id)
        required_devices.append(
            RequiredDeviceProfile(
                stable_id=stable_id,
                kind=require_text(row, "type", f"devices.required[{index}]"),
                name=require_text(row, "name", f"devices.required[{index}]"),
            )
        )

    roles_raw = require_mapping(raw, "roles", "show_profile")
    roles: list[RoleProfile] = []
    for role_name in sorted(roles_raw):
        row = roles_raw[role_name]
        if not isinstance(row, dict):
            raise InspectorInputError(f"roles.{role_name}:object_required")
        forbidden = row.get("forbidden_destinations", [])
        if not isinstance(forbidden, list) or any(
            not isinstance(item, str) for item in forbidden
        ):
            raise InspectorInputError(
                f"roles.{role_name}.forbidden_destinations:string_list_required"
            )
        roles.append(
            RoleProfile(
                name=str(role_name),
                expected_source=require_text(
                    row, "expected_source", f"roles.{role_name}"
                ),
                expected_destination=require_text(
                    row, "expected_destination", f"roles.{role_name}"
                ),
                expected_console_channel=require_text(
                    row, "expected_console_channel", f"roles.{role_name}"
                ),
                forbidden_destinations=tuple(sorted(set(forbidden))),
            )
        )

    network = require_mapping(raw, "network", "show_profile")
    dante = require_mapping(network, "dante", "show_profile.network")
    control = require_mapping(network, "control", "show_profile.network")
    return ShowProfile(
        name=name,
        sample_rate=sample_rate,
        required_devices=tuple(required_devices),
        roles=tuple(roles),
        expected_dante_interface=require_text(
            dante, "expected_interface", "network.dante"
        ),
        expected_control_interface=require_text(
            control, "expected_interface", "network.control"
        ),
        require_wired_dante=bool(dante.get("require_wired", True)),
    )


def _positive_int(value: Any, field_name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise InspectorInputError(f"{field_name}:positive_integer_required")
    return value

