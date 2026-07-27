"""Bounded, versioned input loading for offline Inspector replay."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping


MAX_INPUT_FILE_BYTES = 1_048_576
MAX_BUNDLE_BYTES = 4 * MAX_INPUT_FILE_BYTES
EXPECTED_FILES = ("macos.json", "wing.json", "dante.json", "baseline.json")
EXPECTED_SCHEMAS = {
    "macos.json": "automixer.inspector.macos.v1",
    "wing.json": "automixer.inspector.wing.v1",
    "dante.json": "automixer.inspector.dante-import.v1",
    "baseline.json": "automixer.inspector.baseline.v1",
}


class InspectorInputError(ValueError):
    """Invalid, ambiguous, or oversized replay input."""


@dataclass(frozen=True)
class SnapshotBundle:
    root: Path
    macos: Mapping[str, Any]
    wing: Mapping[str, Any]
    dante: Mapping[str, Any]
    baseline: Mapping[str, Any]

    @property
    def captured_at(self) -> dict[str, str]:
        return {
            "macos": str(self.macos.get("captured_at", "")),
            "wing": str(self.wing.get("captured_at", "")),
            "dante": str(self.dante.get("captured_at", "")),
            "baseline": str(self.baseline.get("created_at", "")),
        }


def load_snapshot_bundle(root: Path) -> SnapshotBundle:
    bundle_root = _resolve_bundle_root(root)
    loaded: dict[str, Mapping[str, Any]] = {}
    total_size = 0
    for filename in EXPECTED_FILES:
        path = bundle_root / filename
        payload, file_size = _load_json_file(path)
        total_size += file_size
        if total_size > MAX_BUNDLE_BYTES:
            raise InspectorInputError("snapshot_bundle_too_large")
        expected_schema = EXPECTED_SCHEMAS[filename]
        if payload.get("schema_version") != expected_schema:
            raise InspectorInputError(
                f"{filename}:schema_version_must_be:{expected_schema}"
            )
        loaded[filename] = payload
    return SnapshotBundle(
        root=bundle_root,
        macos=loaded["macos.json"],
        wing=loaded["wing.json"],
        dante=loaded["dante.json"],
        baseline=loaded["baseline.json"],
    )


def _resolve_bundle_root(root: Path) -> Path:
    candidate = Path(root)
    if candidate.is_symlink():
        raise InspectorInputError("snapshot_bundle_symlink_rejected")
    resolved = candidate.resolve(strict=True)
    if not resolved.is_dir():
        raise InspectorInputError("snapshot_bundle_directory_required")
    return resolved


def _load_json_file(path: Path) -> tuple[Mapping[str, Any], int]:
    if path.is_symlink():
        raise InspectorInputError(f"{path.name}:symlink_rejected")
    try:
        stat = path.stat()
    except FileNotFoundError as exc:
        raise InspectorInputError(f"{path.name}:required_file_missing") from exc
    if not path.is_file():
        raise InspectorInputError(f"{path.name}:regular_file_required")
    if stat.st_size > MAX_INPUT_FILE_BYTES:
        raise InspectorInputError(f"{path.name}:file_too_large")
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise InspectorInputError(f"{path.name}:invalid_json") from exc
    if not isinstance(payload, dict):
        raise InspectorInputError(f"{path.name}:top_level_object_required")
    return payload, stat.st_size


def require_list(payload: Mapping[str, Any], key: str, source: str) -> list[Any]:
    value = payload.get(key)
    if not isinstance(value, list):
        raise InspectorInputError(f"{source}:{key}:list_required")
    return value


def require_mapping(
    payload: Mapping[str, Any], key: str, source: str
) -> Mapping[str, Any]:
    value = payload.get(key)
    if not isinstance(value, dict):
        raise InspectorInputError(f"{source}:{key}:object_required")
    return value


def require_text(payload: Mapping[str, Any], key: str, source: str) -> str:
    value = payload.get(key)
    if not isinstance(value, str) or not value.strip():
        raise InspectorInputError(f"{source}:{key}:non_empty_string_required")
    return value.strip()


def validate_unique_ids(rows: list[Any], source: str) -> None:
    seen: set[str] = set()
    for index, row in enumerate(rows):
        if not isinstance(row, dict):
            raise InspectorInputError(f"{source}[{index}]:object_required")
        stable_id = require_text(row, "id", f"{source}[{index}]")
        if stable_id in seen:
            raise InspectorInputError(f"{source}:duplicate_id:{stable_id}")
        seen.add(stable_id)

