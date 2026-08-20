"""Read-only intake for dLive-state physical readback artifacts.

This layer inventories operator-captured JSON artifacts and validates them
against the dLive-state physical readback contract. A valid artifact is only a
manual-review candidate; this module never closes fixture gaps, grants MIDI/TCP
authority, writes dLive state, applies dLive changes, or grants live authority.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, Tuple

try:
    from dlive_state_physical_readback_contract import (
        DLIVE_STATE_PHYSICAL_READBACK_CONTRACT_METHOD_ID,
        DLIVE_STATE_PHYSICAL_READBACK_SCHEMA_ID,
        DLIVE_STATE_REMEDIATION_ACTION_ID,
        DLIVE_STATE_SURFACE_ID,
        validate_dlive_state_physical_readback_artifact,
    )
except ImportError:  # pragma: no cover - package import compatibility
    from backend.dlive_state_physical_readback_contract import (
        DLIVE_STATE_PHYSICAL_READBACK_CONTRACT_METHOD_ID,
        DLIVE_STATE_PHYSICAL_READBACK_SCHEMA_ID,
        DLIVE_STATE_REMEDIATION_ACTION_ID,
        DLIVE_STATE_SURFACE_ID,
        validate_dlive_state_physical_readback_artifact,
    )


DLIVE_STATE_PHYSICAL_READBACK_ARTIFACT_INTAKE_METHOD_ID = (
    "dlive_state_physical_readback_artifact_intake"
)
DEFAULT_DLIVE_STATE_PHYSICAL_READBACK_ARTIFACT_ROOTS = (
    "artifacts/read_surface_physical_readback/dlive_state",
    "artifacts/dlive_state_physical_readback",
)
DLIVE_STATE_INTAKE_STATUSES = {
    "operator_artifact_missing",
    "valid_operator_artifact_available",
    "invalid_operator_artifact_present",
}
DLIVE_STATE_MAX_ARTIFACT_ROOTS = 16
DLIVE_STATE_MAX_ARTIFACT_COUNT = 128
DLIVE_STATE_MAX_ARTIFACT_BYTES = 1 * 1024 * 1024
DLIVE_STATE_MAX_TOTAL_ARTIFACT_BYTES = 8 * 1024 * 1024
DLIVE_STATE_MAX_JSON_DEPTH = 32
DLIVE_STATE_MAX_JSON_NODES = 50_000
DLIVE_STATE_MAX_TOTAL_JSON_NODES = 200_000
DLIVE_STATE_MAX_SCALAR_CHARACTERS = 1 * 1024 * 1024
DLIVE_STATE_MAX_TOTAL_SCALAR_CHARACTERS = 8 * 1024 * 1024


class DLiveStateArtifactResourceLimitError(ValueError):
    """Raised when untrusted artifact intake exceeds a hard resource budget."""


@dataclass
class _ArtifactResourceBudget:
    artifact_count: int = 0
    total_file_bytes: int = 0
    total_json_nodes: int = 0
    total_scalar_characters: int = 0

    def reserve_artifact(self) -> None:
        if self.artifact_count >= DLIVE_STATE_MAX_ARTIFACT_COUNT:
            raise DLiveStateArtifactResourceLimitError(
                f"artifact_count_exceeds_{DLIVE_STATE_MAX_ARTIFACT_COUNT}"
            )
        self.artifact_count += 1

    def account_file_bytes(self, size: int) -> None:
        if size > DLIVE_STATE_MAX_ARTIFACT_BYTES:
            raise DLiveStateArtifactResourceLimitError(
                f"artifact_bytes_exceed_{DLIVE_STATE_MAX_ARTIFACT_BYTES}"
            )
        if self.total_file_bytes + size > DLIVE_STATE_MAX_TOTAL_ARTIFACT_BYTES:
            raise DLiveStateArtifactResourceLimitError(
                f"total_artifact_bytes_exceed_{DLIVE_STATE_MAX_TOTAL_ARTIFACT_BYTES}"
            )
        self.total_file_bytes += size

    def account_json_shape(self, payload: Any) -> None:
        remaining_nodes = DLIVE_STATE_MAX_TOTAL_JSON_NODES - self.total_json_nodes
        remaining_characters = (
            DLIVE_STATE_MAX_TOTAL_SCALAR_CHARACTERS
            - self.total_scalar_characters
        )
        nodes, scalar_characters = _json_shape_cost(
            payload,
            node_limit=min(DLIVE_STATE_MAX_JSON_NODES, remaining_nodes),
            scalar_character_limit=min(
                DLIVE_STATE_MAX_SCALAR_CHARACTERS,
                remaining_characters,
            ),
        )
        self.total_json_nodes += nodes
        self.total_scalar_characters += scalar_characters


@dataclass(frozen=True)
class DLiveStatePhysicalReadbackArtifactRecord:
    source_id: str
    status: str
    accepted_as_physical_fixture_candidate: bool
    channel_count: int
    captured_at: str | None
    console_model: str | None
    source: str | None
    violations: Tuple[str, ...]

    def __post_init__(self) -> None:
        if self.status not in {
            "valid_physical_readback_artifact",
            "invalid_physical_readback_artifact",
            "artifact_load_error",
        }:
            raise ValueError(f"Unknown dLive-state artifact status: {self.status}")

    def to_dict(self) -> Dict[str, Any]:
        payload = asdict(self)
        payload["violations"] = list(self.violations)
        return payload


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


def _mapping(value: Any) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


def _string(value: Any) -> str | None:
    return value if isinstance(value, str) and value else None


def _safe_int(value: Any) -> int:
    return value if isinstance(value, int) and value >= 0 else 0


def _string_tuple(value: Any) -> Tuple[str, ...]:
    if not isinstance(value, (list, tuple)):
        return ()
    return tuple(str(item) for item in value if isinstance(item, str) and item)


def _json_shape_cost(
    payload: Any,
    *,
    node_limit: int,
    scalar_character_limit: int,
) -> Tuple[int, int]:
    """Measure JSON-like input iteratively before recursive contract validation."""

    nodes = 0
    scalar_characters = 0
    stack: list[tuple[Any, int]] = [(payload, 1)]
    while stack:
        value, depth = stack.pop()
        if depth > DLIVE_STATE_MAX_JSON_DEPTH:
            raise DLiveStateArtifactResourceLimitError(
                f"json_depth_exceeds_{DLIVE_STATE_MAX_JSON_DEPTH}"
            )
        nodes += 1
        if nodes > node_limit:
            raise DLiveStateArtifactResourceLimitError(
                f"json_nodes_exceed_{node_limit}"
            )
        if isinstance(value, Mapping):
            if nodes + len(stack) + len(value) > node_limit:
                raise DLiveStateArtifactResourceLimitError(
                    f"json_nodes_exceed_{node_limit}"
                )
            for key, nested in value.items():
                if not isinstance(key, str):
                    raise ValueError("artifact JSON object keys must be strings")
                scalar_characters += len(key)
                stack.append((nested, depth + 1))
        elif isinstance(value, (list, tuple)):
            if nodes + len(stack) + len(value) > node_limit:
                raise DLiveStateArtifactResourceLimitError(
                    f"json_nodes_exceed_{node_limit}"
                )
            stack.extend((nested, depth + 1) for nested in value)
        elif isinstance(value, str):
            scalar_characters += len(value)
        elif value is not None and type(value) not in {bool, int, float}:
            raise ValueError("artifact contains a non-JSON value")
        if scalar_characters > scalar_character_limit:
            raise DLiveStateArtifactResourceLimitError(
                "scalar_characters_exceed_"
                f"{scalar_character_limit}"
            )
    return nodes, scalar_characters


def _validate_encoded_json_depth(encoded: bytes) -> None:
    """Reject excessive object/list nesting before the recursive JSON decoder."""

    depth = 0
    in_string = False
    escaped = False
    for byte in encoded:
        if in_string:
            if escaped:
                escaped = False
            elif byte == 0x5C:  # backslash
                escaped = True
            elif byte == 0x22:  # quote
                in_string = False
            continue
        if byte == 0x22:
            in_string = True
        elif byte in {0x5B, 0x7B}:  # [ or {
            depth += 1
            if depth > DLIVE_STATE_MAX_JSON_DEPTH:
                raise DLiveStateArtifactResourceLimitError(
                    f"json_depth_exceeds_{DLIVE_STATE_MAX_JSON_DEPTH}"
                )
        elif byte in {0x5D, 0x7D}:  # ] or }
            depth = max(0, depth - 1)


def _artifact_roots(artifact_roots: Iterable[str | Path] | None) -> Tuple[Path, ...]:
    roots = (
        artifact_roots
        if artifact_roots is not None
        else DEFAULT_DLIVE_STATE_PHYSICAL_READBACK_ARTIFACT_ROOTS
    )
    bounded: list[Path] = []
    for root in roots:
        if len(bounded) >= DLIVE_STATE_MAX_ARTIFACT_ROOTS:
            raise DLiveStateArtifactResourceLimitError(
                f"artifact_roots_exceed_{DLIVE_STATE_MAX_ARTIFACT_ROOTS}"
            )
        bounded.append(Path(root))
    return tuple(bounded)


def _discover_artifact_paths(artifact_roots: Iterable[str | Path] | None) -> Tuple[Path, ...]:
    paths: list[Path] = []
    for root in _artifact_roots(artifact_roots):
        if root.is_file() and root.suffix.lower() == ".json":
            paths.append(root)
            if len(paths) > DLIVE_STATE_MAX_ARTIFACT_COUNT:
                raise DLiveStateArtifactResourceLimitError(
                    f"artifact_count_exceeds_{DLIVE_STATE_MAX_ARTIFACT_COUNT}"
                )
        if not root.is_dir():
            continue
        for path in root.rglob("*.json"):
            if not path.is_file():
                continue
            paths.append(path)
            if len(paths) > DLIVE_STATE_MAX_ARTIFACT_COUNT:
                raise DLiveStateArtifactResourceLimitError(
                    f"artifact_count_exceeds_{DLIVE_STATE_MAX_ARTIFACT_COUNT}"
                )
    return tuple(sorted(dict.fromkeys(paths), key=lambda path: str(path)))


def _load_json_artifact(
    path: Path,
    budget: _ArtifactResourceBudget,
) -> Mapping[str, Any]:
    remaining_total_bytes = (
        DLIVE_STATE_MAX_TOTAL_ARTIFACT_BYTES - budget.total_file_bytes
    )
    read_limit = min(DLIVE_STATE_MAX_ARTIFACT_BYTES, remaining_total_bytes)
    with path.open("rb") as handle:
        encoded = handle.read(read_limit + 1)
    budget.account_file_bytes(len(encoded))
    _validate_encoded_json_depth(encoded)
    payload = json.loads(encoded.decode("utf-8"))
    if not isinstance(payload, Mapping):
        raise ValueError("artifact JSON root must be an object")
    budget.account_json_shape(payload)
    return payload


def _record_from_validation(
    source_id: str,
    validation: Mapping[str, Any],
) -> DLiveStatePhysicalReadbackArtifactRecord:
    evidence = _mapping(validation.get("evidence"))
    return DLiveStatePhysicalReadbackArtifactRecord(
        source_id=source_id,
        status=str(validation.get("status") or "invalid_physical_readback_artifact"),
        accepted_as_physical_fixture_candidate=(
            validation.get("accepted_as_physical_fixture_candidate") is True
        ),
        channel_count=_safe_int(validation.get("channel_count")),
        captured_at=_string(evidence.get("captured_at")),
        console_model=_string(evidence.get("console_model")),
        source=_string(evidence.get("source")),
        violations=_string_tuple(validation.get("violations")),
    )


def _record_from_load_error(
    source_id: str,
    exc: Exception,
) -> DLiveStatePhysicalReadbackArtifactRecord:
    return DLiveStatePhysicalReadbackArtifactRecord(
        source_id=source_id,
        status="artifact_load_error",
        accepted_as_physical_fixture_candidate=False,
        channel_count=0,
        captured_at=None,
        console_model=None,
        source=None,
        violations=(f"artifact_load_error:{type(exc).__name__}",),
    )


def _artifact_records(
    *,
    artifact_paths: Iterable[str | Path] | None,
    artifact_payloads: Iterable[Mapping[str, Any]] | None,
    artifact_roots: Iterable[str | Path] | None,
) -> Tuple[DLiveStatePhysicalReadbackArtifactRecord, ...]:
    records: list[DLiveStatePhysicalReadbackArtifactRecord] = []
    budget = _ArtifactResourceBudget()
    try:
        if artifact_paths is not None:
            paths_list: list[Path] = []
            for path in artifact_paths:
                if len(paths_list) >= DLIVE_STATE_MAX_ARTIFACT_COUNT:
                    raise DLiveStateArtifactResourceLimitError(
                        f"artifact_count_exceeds_{DLIVE_STATE_MAX_ARTIFACT_COUNT}"
                    )
                paths_list.append(Path(path))
            paths = tuple(paths_list)
        else:
            paths = _discover_artifact_paths(artifact_roots)
    except Exception as exc:
        return (_record_from_load_error("artifact_discovery", exc),)

    for path in paths:
        try:
            budget.reserve_artifact()
            validation = validate_dlive_state_physical_readback_artifact(
                _load_json_artifact(path, budget)
            )
            records.append(_record_from_validation(str(path), validation))
        except Exception as exc:  # pragma: no cover - defensive artifact parser
            records.append(_record_from_load_error(str(path), exc))

    if artifact_payloads is not None:
        for index, payload in enumerate(artifact_payloads):
            source_id = f"inline_payload_{index}"
            if budget.artifact_count >= DLIVE_STATE_MAX_ARTIFACT_COUNT:
                limit_record = _record_from_load_error(
                    source_id,
                    DLiveStateArtifactResourceLimitError(
                        f"artifact_count_exceeds_{DLIVE_STATE_MAX_ARTIFACT_COUNT}"
                    ),
                )
                if records:
                    records[-1] = limit_record
                else:
                    records.append(limit_record)
                break
            try:
                budget.reserve_artifact()
                budget.account_json_shape(payload)
                validation = validate_dlive_state_physical_readback_artifact(payload)
                records.append(_record_from_validation(source_id, validation))
            except Exception as exc:
                records.append(_record_from_load_error(source_id, exc))

    return tuple(records)


def build_dlive_state_physical_readback_artifact_intake(
    *,
    artifact_paths: Iterable[str | Path] | None = None,
    artifact_payloads: Iterable[Mapping[str, Any]] | None = None,
    artifact_roots: Iterable[str | Path] | None = None,
) -> Dict[str, Any]:
    """Build a read-only intake snapshot for dLive-state readbacks."""

    try:
        normalized_roots = _artifact_roots(artifact_roots)
        records = _artifact_records(
            artifact_paths=artifact_paths,
            artifact_payloads=artifact_payloads,
            artifact_roots=normalized_roots,
        )
    except Exception as exc:
        normalized_roots = ()
        records = (_record_from_load_error("artifact_roots", exc),)
    valid_records = tuple(row for row in records if row.accepted_as_physical_fixture_candidate)
    invalid_records = tuple(row for row in records if not row.accepted_as_physical_fixture_candidate)
    if invalid_records:
        status = "invalid_operator_artifact_present"
        blocked_by = ("operator_must_replace_invalid_dlive_state_physical_readback_artifact",)
    elif valid_records:
        status = "valid_operator_artifact_available"
        blocked_by = (
            "operator_must_review_valid_dlive_state_candidate_before_fixture_closure",
        )
    else:
        status = "operator_artifact_missing"
        blocked_by = ("operator_must_capture_dlive_state_physical_readback_artifact",)

    all_violations = tuple(
        dict.fromkeys(violation for row in records for violation in row.violations)
    )
    latest_valid = valid_records[-1] if valid_records else None
    valid_channel_count = sum(row.channel_count for row in valid_records)
    return {
        "type": "dlive_state_physical_readback_artifact_intake",
        "generated_at": _utc_now(),
        "method_id": DLIVE_STATE_PHYSICAL_READBACK_ARTIFACT_INTAKE_METHOD_ID,
        "required_contract_method_id": DLIVE_STATE_PHYSICAL_READBACK_CONTRACT_METHOD_ID,
        "schema_id": DLIVE_STATE_PHYSICAL_READBACK_SCHEMA_ID,
        "status": status,
        "surface_id": DLIVE_STATE_SURFACE_ID,
        "remediation_action_id": DLIVE_STATE_REMEDIATION_ACTION_ID,
        "safety": {
            "read_only": True,
            "live_write": False,
            "mutates_console_state": False,
            "queries_console": False,
            "executes_tests": False,
            "operator_required": True,
            "execution_allowed_by_intake": False,
            "fixture_closure_allowed_by_intake": False,
            "live_widening_allowed": False,
            "full_autonomous_ready": False,
            "midi_send_allowed": False,
            "tcp_send_allowed": False,
            "dlive_write_allowed": False,
            "dlive_apply_allowed": False,
        },
        "summary": {
            "artifact_search_roots": [str(root) for root in normalized_roots],
            "artifact_count": len(records),
            "valid_candidate_count": len(valid_records),
            "invalid_candidate_count": len(invalid_records),
            "valid_channel_count": valid_channel_count,
            "accepted_as_physical_fixture_candidate": bool(valid_records)
            and not invalid_records,
            "operator_review_required": True,
            "physical_fixture_gap_remains": True,
            "fixture_closure_allowed_by_intake": False,
            "midi_send_allowed": False,
            "tcp_send_allowed": False,
            "dlive_write_allowed": False,
            "dlive_apply_allowed": False,
            "live_authority_granted": False,
            "full_autonomous_ready": False,
            "promotion_families_requiring_surface": [
                "fader",
                "gain",
                "mute",
                "bus_group_matrix",
                "routing",
                "scene_preset",
                "mix_balance",
                "performance_adaptation",
            ],
            "latest_valid_candidate_source_id": latest_valid.source_id if latest_valid else None,
            "latest_valid_candidate_captured_at": latest_valid.captured_at if latest_valid else None,
            "next_global_step": (
                "Operator reviews the valid dLive-state physical readback "
                "candidate before any fixture gap can be closed."
                if valid_records
                else "Operator captures dLive-state physical readback evidence "
                "matching the schema contract."
            ),
            "blocked_by": list(blocked_by),
            "violations": list(all_violations),
        },
        "artifacts": [record.to_dict() for record in records],
    }
