"""Contract for routing and send physical readback artifacts.

This module is transport-free. It defines the JSON-like artifact shape a future
operator-run physical readback must produce before the ``routing_and_sends``
read surface can be treated as physical fixture evidence for routing, sends,
FX-send/insert, bus/matrix balance, and scene/preset promotion decisions.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import math
from typing import Any, Dict, Mapping, Tuple


ROUTING_SENDS_PHYSICAL_READBACK_CONTRACT_METHOD_ID = (
    "routing_sends_physical_readback_contract"
)
ROUTING_SENDS_PHYSICAL_READBACK_SCHEMA_ID = "routing_sends_physical_readback.v1"
ROUTING_SENDS_PHYSICAL_READBACK_ARTIFACT_TYPE = "routing_sends_physical_readback"
ROUTING_SENDS_SURFACE_ID = "routing_and_sends"
ROUTING_SENDS_REMEDIATION_ACTION_ID = "remediate_routing_and_sends"

ROUTING_SENDS_REQUIRED_TOP_LEVEL_FIELDS = (
    "type",
    "schema_id",
    "surface_id",
    "remediation_action_id",
    "captured_at",
    "source",
    "console_model",
    "static_dante_route_count",
    "channel_bus_send_count",
    "channel_main_send_count",
    "channel_matrix_send_count",
    "output_route_count",
    "static_dante_routes",
    "channel_bus_sends",
    "channel_main_sends",
    "channel_matrix_sends",
    "output_routes",
    "safety",
)
ROUTING_SENDS_REQUIRED_STATIC_ROUTE_FIELDS = (
    "start",
    "end",
    "role",
    "tap_point",
    "label_short",
    "label_full",
    "required",
    "wing_routing_hint",
    "used_by",
    "readback_status",
    "readback_source",
    "captured_at",
    "coverage",
)
ROUTING_SENDS_REQUIRED_BUS_SEND_FIELDS = (
    "channel",
    "send",
    "target_bus",
    "readback_status",
    "readback_source",
    "captured_at",
    "on",
    "level_db",
    "pre_on",
    "mode",
    "plink",
    "pan",
    "active",
    "destructive_change_marker",
    "coverage",
)
ROUTING_SENDS_REQUIRED_MAIN_SEND_FIELDS = (
    "channel",
    "main",
    "readback_status",
    "readback_source",
    "captured_at",
    "on",
    "level_db",
    "pre",
    "active",
    "destructive_change_marker",
    "coverage",
)
ROUTING_SENDS_REQUIRED_MATRIX_SEND_FIELDS = (
    "channel",
    "send",
    "target_matrix",
    "readback_status",
    "readback_source",
    "captured_at",
    "on",
    "level_db",
    "active",
    "destructive_change_marker",
    "coverage",
)
ROUTING_SENDS_REQUIRED_OUTPUT_ROUTE_FIELDS = (
    "output_group",
    "output_number",
    "readback_status",
    "readback_source",
    "captured_at",
    "source_group",
    "source_channel",
    "destructive_change_marker",
    "coverage",
)
ROUTING_SENDS_REQUIRED_STATIC_ROUTE_COVERAGE_FLAGS = (
    "identity",
    "static_config_match",
    "physical_route_mapping",
    "scoped_diff_marker",
)
ROUTING_SENDS_REQUIRED_SEND_COVERAGE_FLAGS = (
    "assignment",
    "level",
    "mode",
    "scoped_diff_marker",
    "cache_matches_physical_readback",
)
ROUTING_SENDS_REQUIRED_OUTPUT_ROUTE_COVERAGE_FLAGS = (
    "assignment",
    "source",
    "scoped_diff_marker",
    "cache_matches_physical_readback",
)
ROUTING_SENDS_PHYSICAL_READBACK_SOURCES = (
    "physical_readback",
    "wing_physical_readback",
    "dlive_physical_readback",
)
ROUTING_SENDS_VERIFIED_PHYSICAL_STATUSES = (
    "verified_from_physical_readback",
)
ROUTING_SENDS_REQUIRED_FALSE_SAFETY_FLAGS = (
    "live_write",
    "mutates_console_state",
    "live_widening_allowed",
    "full_autonomous_ready",
)
ROUTING_SENDS_FORBIDDEN_MUTATION_EVIDENCE_FIELDS = (
    "write_events",
    "manual_write_events",
    "apply_result",
    "rollback_result",
    "proposal_apply_result",
    "routing_write_events",
    "route_write_events",
    "send_write_events",
    "bus_send_write_events",
    "main_send_write_events",
    "matrix_send_write_events",
    "output_route_write_events",
    "dante_patch_write_events",
    "scene_recall_events",
)
MINIMUM_STATIC_DANTE_ROUTE_COUNT = 1
MINIMUM_CHANNEL_BUS_SEND_COUNT = 1
MINIMUM_CHANNEL_MAIN_SEND_COUNT = 1
MINIMUM_CHANNEL_MATRIX_SEND_COUNT = 1
MINIMUM_OUTPUT_ROUTE_COUNT = 1


@dataclass(frozen=True)
class RoutingSendsPhysicalReadbackValidation:
    method_id: str
    schema_id: str
    surface_id: str
    remediation_action_id: str
    status: str
    accepted_as_physical_fixture_candidate: bool
    live_authority_granted: bool
    full_autonomous_ready: bool
    static_dante_route_count: int
    channel_bus_send_count: int
    channel_main_send_count: int
    channel_matrix_send_count: int
    output_route_count: int
    violations: Tuple[str, ...]
    evidence: Dict[str, Any]

    def __post_init__(self) -> None:
        if self.live_authority_granted:
            raise ValueError("Routing/sends validation must not grant live authority")
        if self.full_autonomous_ready:
            raise ValueError(
                "Routing/sends validation must not claim full autonomous readiness"
            )

    def to_dict(self) -> Dict[str, Any]:
        payload = asdict(self)
        payload["violations"] = list(self.violations)
        return payload


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


def _mapping(value: Any) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


def _bool_true(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if value is None:
        return False
    return str(value).strip().lower() in {"1", "true", "yes", "y"}


def _string(value: Any) -> str:
    return value if isinstance(value, str) else ""


def _finite_number(value: Any) -> bool:
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(float(value))
    )


def _bool_or_0_1(value: Any) -> bool:
    return isinstance(value, bool) or (
        isinstance(value, int) and not isinstance(value, bool) and value in (0, 1)
    )


def _append_required_field_violations(payload: Mapping[str, Any], violations: list[str]) -> None:
    for key in ROUTING_SENDS_REQUIRED_TOP_LEVEL_FIELDS:
        if key not in payload:
            violations.append(f"missing_top_level_field:{key}")


def _append_identity_violations(payload: Mapping[str, Any], violations: list[str]) -> None:
    if payload.get("type") != ROUTING_SENDS_PHYSICAL_READBACK_ARTIFACT_TYPE:
        violations.append("artifact_type_mismatch")
    if payload.get("schema_id") != ROUTING_SENDS_PHYSICAL_READBACK_SCHEMA_ID:
        violations.append("schema_id_mismatch")
    if payload.get("surface_id") != ROUTING_SENDS_SURFACE_ID:
        violations.append("surface_id_mismatch")
    if payload.get("remediation_action_id") != ROUTING_SENDS_REMEDIATION_ACTION_ID:
        violations.append("remediation_action_id_mismatch")
    if payload.get("source") not in ROUTING_SENDS_PHYSICAL_READBACK_SOURCES:
        violations.append("source_not_physical_readback")
    if not _string(payload.get("captured_at")):
        violations.append("captured_at_missing")
    if not _string(payload.get("console_model")):
        violations.append("console_model_missing")


def _mutation_field_paths(value: Any, *, prefix: str = "") -> Tuple[str, ...]:
    if isinstance(value, Mapping):
        paths: list[str] = []
        for key, nested in value.items():
            key_text = str(key)
            path = f"{prefix}.{key_text}" if prefix else key_text
            if key_text in ROUTING_SENDS_FORBIDDEN_MUTATION_EVIDENCE_FIELDS:
                paths.append(path)
            paths.extend(_mutation_field_paths(nested, prefix=path))
        return tuple(paths)
    if isinstance(value, list):
        paths = []
        for index, nested in enumerate(value):
            path = f"{prefix}[{index}]" if prefix else f"[{index}]"
            paths.extend(_mutation_field_paths(nested, prefix=path))
        return tuple(paths)
    return ()


def _append_safety_violations(payload: Mapping[str, Any], violations: list[str]) -> None:
    safety = _mapping(payload.get("safety"))
    if safety.get("read_only") is not True:
        violations.append("safety_read_only_not_true")
    for key in ROUTING_SENDS_REQUIRED_FALSE_SAFETY_FLAGS:
        if safety.get(key) is not False:
            violations.append(f"safety_{key}_not_false")
        if _bool_true(payload.get(key)):
            violations.append(f"top_level_{key}_not_false")
    for path in _mutation_field_paths(payload):
        violations.append(f"mutation_evidence_field_present:{path}")


def _append_nested_field_violations(
    *,
    row_label: str,
    mapping: Mapping[str, Any],
    required_fields: Tuple[str, ...],
    label: str,
    violations: list[str],
) -> None:
    for key in required_fields:
        if key not in mapping:
            violations.append(f"{row_label}_{label}_missing_field:{key}")


def _append_coverage_violations(
    *,
    row_label: str,
    coverage: Mapping[str, Any],
    required_flags: Tuple[str, ...],
    violations: list[str],
) -> None:
    _append_nested_field_violations(
        row_label=row_label,
        mapping=coverage,
        required_fields=required_flags,
        label="coverage",
        violations=violations,
    )
    for key in required_flags:
        if coverage.get(key) is not True:
            violations.append(f"{row_label}_coverage_{key}_not_true")


def _append_physical_row_identity_violations(
    *,
    row_label: str,
    row: Mapping[str, Any],
    violations: list[str],
) -> None:
    if row.get("readback_status") not in ROUTING_SENDS_VERIFIED_PHYSICAL_STATUSES:
        violations.append(f"{row_label}_readback_status_not_physical_verified")
    if row.get("readback_source") not in ROUTING_SENDS_PHYSICAL_READBACK_SOURCES:
        violations.append(f"{row_label}_readback_source_not_physical_readback")
    if not _string(row.get("captured_at")):
        violations.append(f"{row_label}_captured_at_missing")


def _append_channel_id_violations(
    *,
    row_label: str,
    row: Mapping[str, Any],
    key: str = "channel",
    violations: list[str],
) -> None:
    value = row.get(key)
    if not isinstance(value, int) or value < 1:
        violations.append(f"{row_label}_{key}_invalid")


def _append_send_common_violations(
    *,
    row_label: str,
    row: Mapping[str, Any],
    coverage_flags: Tuple[str, ...],
    violations: list[str],
) -> None:
    _append_physical_row_identity_violations(row_label=row_label, row=row, violations=violations)
    _append_channel_id_violations(row_label=row_label, row=row, violations=violations)
    if "on" in row and not _bool_or_0_1(row.get("on")):
        violations.append(f"{row_label}_on_not_bool_or_0_1")
    if "level_db" in row and not _finite_number(row.get("level_db")):
        violations.append(f"{row_label}_level_db_not_number")
    if "active" in row and not isinstance(row.get("active"), bool):
        violations.append(f"{row_label}_active_not_bool")
    if not _string(row.get("destructive_change_marker")):
        violations.append(f"{row_label}_destructive_change_marker_missing")
    _append_coverage_violations(
        row_label=row_label,
        coverage=_mapping(row.get("coverage")),
        required_flags=coverage_flags,
        violations=violations,
    )


def _append_static_route_violations(payload: Mapping[str, Any], violations: list[str]) -> int:
    rows = payload.get("static_dante_routes")
    if not isinstance(rows, list) or len(rows) < MINIMUM_STATIC_DANTE_ROUTE_COUNT:
        violations.append("static_dante_routes_missing")
        return 0
    if (
        not isinstance(payload.get("static_dante_route_count"), int)
        or payload.get("static_dante_route_count") != len(rows)
    ):
        violations.append("static_dante_route_count_mismatch")
    seen_spans: set[Tuple[int, int, str]] = set()
    for index, row_value in enumerate(rows):
        row = _mapping(row_value)
        row_label = f"static_route_{index}"
        if not row:
            violations.append(f"{row_label}_not_object")
            continue
        for key in ROUTING_SENDS_REQUIRED_STATIC_ROUTE_FIELDS:
            if key not in row:
                violations.append(f"{row_label}_missing_field:{key}")
        start = row.get("start")
        end = row.get("end")
        role = row.get("role")
        if not isinstance(start, int) or start < 1:
            violations.append(f"{row_label}_start_invalid")
        if not isinstance(end, int) or end < 1:
            violations.append(f"{row_label}_end_invalid")
        if isinstance(start, int) and isinstance(end, int) and start > end:
            violations.append(f"{row_label}_range_invalid")
        if not _string(role):
            violations.append(f"{row_label}_role_missing")
        elif isinstance(start, int) and isinstance(end, int):
            span = (start, end, role)
            if span in seen_spans:
                violations.append(f"{row_label}_span_duplicate")
            else:
                seen_spans.add(span)
        for key in ("tap_point", "label_short", "label_full", "wing_routing_hint"):
            if key in row and not _string(row.get(key)):
                violations.append(f"{row_label}_{key}_missing")
        if "required" in row and not isinstance(row.get("required"), bool):
            violations.append(f"{row_label}_required_not_bool")
        if "used_by" in row and not isinstance(row.get("used_by"), list):
            violations.append(f"{row_label}_used_by_not_list")
        _append_physical_row_identity_violations(
            row_label=row_label,
            row=row,
            violations=violations,
        )
        _append_coverage_violations(
            row_label=row_label,
            coverage=_mapping(row.get("coverage")),
            required_flags=ROUTING_SENDS_REQUIRED_STATIC_ROUTE_COVERAGE_FLAGS,
            violations=violations,
        )
    return len(rows)


def _append_bus_send_violations(payload: Mapping[str, Any], violations: list[str]) -> int:
    rows = payload.get("channel_bus_sends")
    if not isinstance(rows, list) or len(rows) < MINIMUM_CHANNEL_BUS_SEND_COUNT:
        violations.append("channel_bus_sends_missing")
        return 0
    if (
        not isinstance(payload.get("channel_bus_send_count"), int)
        or payload.get("channel_bus_send_count") != len(rows)
    ):
        violations.append("channel_bus_send_count_mismatch")
    seen_keys: set[Tuple[int, int]] = set()
    for index, row_value in enumerate(rows):
        row = _mapping(row_value)
        row_label = f"bus_send_{index}"
        if not row:
            violations.append(f"{row_label}_not_object")
            continue
        for key in ROUTING_SENDS_REQUIRED_BUS_SEND_FIELDS:
            if key not in row:
                violations.append(f"{row_label}_missing_field:{key}")
        _append_send_common_violations(
            row_label=row_label,
            row=row,
            coverage_flags=ROUTING_SENDS_REQUIRED_SEND_COVERAGE_FLAGS,
            violations=violations,
        )
        send = row.get("send")
        target_bus = row.get("target_bus")
        if not isinstance(send, int) or send < 1:
            violations.append(f"{row_label}_send_invalid")
        if not isinstance(target_bus, int) or target_bus < 1:
            violations.append(f"{row_label}_target_bus_invalid")
        if isinstance(send, int) and isinstance(target_bus, int) and send != target_bus:
            violations.append(f"{row_label}_send_target_bus_mismatch")
        channel = row.get("channel")
        if isinstance(channel, int) and isinstance(send, int):
            key = (channel, send)
            if key in seen_keys:
                violations.append(f"{row_label}_channel_send_duplicate")
            else:
                seen_keys.add(key)
        if "pre_on" in row and not _bool_or_0_1(row.get("pre_on")):
            violations.append(f"{row_label}_pre_on_not_bool_or_0_1")
        if "plink" in row and not _bool_or_0_1(row.get("plink")):
            violations.append(f"{row_label}_plink_not_bool_or_0_1")
        if "pan" in row and not _finite_number(row.get("pan")):
            violations.append(f"{row_label}_pan_not_number")
        if "mode" in row and not _string(row.get("mode")):
            violations.append(f"{row_label}_mode_missing")
    return len(rows)


def _append_main_send_violations(payload: Mapping[str, Any], violations: list[str]) -> int:
    rows = payload.get("channel_main_sends")
    if not isinstance(rows, list) or len(rows) < MINIMUM_CHANNEL_MAIN_SEND_COUNT:
        violations.append("channel_main_sends_missing")
        return 0
    if (
        not isinstance(payload.get("channel_main_send_count"), int)
        or payload.get("channel_main_send_count") != len(rows)
    ):
        violations.append("channel_main_send_count_mismatch")
    seen_keys: set[Tuple[int, int]] = set()
    for index, row_value in enumerate(rows):
        row = _mapping(row_value)
        row_label = f"main_send_{index}"
        if not row:
            violations.append(f"{row_label}_not_object")
            continue
        for key in ROUTING_SENDS_REQUIRED_MAIN_SEND_FIELDS:
            if key not in row:
                violations.append(f"{row_label}_missing_field:{key}")
        _append_send_common_violations(
            row_label=row_label,
            row=row,
            coverage_flags=ROUTING_SENDS_REQUIRED_SEND_COVERAGE_FLAGS,
            violations=violations,
        )
        main = row.get("main")
        if not isinstance(main, int) or main < 1:
            violations.append(f"{row_label}_main_invalid")
        channel = row.get("channel")
        if isinstance(channel, int) and isinstance(main, int):
            key = (channel, main)
            if key in seen_keys:
                violations.append(f"{row_label}_channel_main_duplicate")
            else:
                seen_keys.add(key)
        if "pre" in row and not _bool_or_0_1(row.get("pre")):
            violations.append(f"{row_label}_pre_not_bool_or_0_1")
    return len(rows)


def _append_matrix_send_violations(payload: Mapping[str, Any], violations: list[str]) -> int:
    rows = payload.get("channel_matrix_sends")
    if not isinstance(rows, list) or len(rows) < MINIMUM_CHANNEL_MATRIX_SEND_COUNT:
        violations.append("channel_matrix_sends_missing")
        return 0
    if (
        not isinstance(payload.get("channel_matrix_send_count"), int)
        or payload.get("channel_matrix_send_count") != len(rows)
    ):
        violations.append("channel_matrix_send_count_mismatch")
    seen_keys: set[Tuple[int, int]] = set()
    for index, row_value in enumerate(rows):
        row = _mapping(row_value)
        row_label = f"matrix_send_{index}"
        if not row:
            violations.append(f"{row_label}_not_object")
            continue
        for key in ROUTING_SENDS_REQUIRED_MATRIX_SEND_FIELDS:
            if key not in row:
                violations.append(f"{row_label}_missing_field:{key}")
        _append_send_common_violations(
            row_label=row_label,
            row=row,
            coverage_flags=ROUTING_SENDS_REQUIRED_SEND_COVERAGE_FLAGS,
            violations=violations,
        )
        target_matrix = row.get("target_matrix")
        send = row.get("send")
        if not isinstance(target_matrix, int) or target_matrix < 1:
            violations.append(f"{row_label}_target_matrix_invalid")
        expected_send = f"MX{target_matrix}" if isinstance(target_matrix, int) else ""
        if send != expected_send:
            violations.append(f"{row_label}_send_target_matrix_mismatch")
        channel = row.get("channel")
        if isinstance(channel, int) and isinstance(target_matrix, int):
            key = (channel, target_matrix)
            if key in seen_keys:
                violations.append(f"{row_label}_channel_matrix_duplicate")
            else:
                seen_keys.add(key)
    return len(rows)


def _append_output_route_violations(payload: Mapping[str, Any], violations: list[str]) -> int:
    rows = payload.get("output_routes")
    if not isinstance(rows, list) or len(rows) < MINIMUM_OUTPUT_ROUTE_COUNT:
        violations.append("output_routes_missing")
        return 0
    if (
        not isinstance(payload.get("output_route_count"), int)
        or payload.get("output_route_count") != len(rows)
    ):
        violations.append("output_route_count_mismatch")
    seen_keys: set[Tuple[str, int]] = set()
    for index, row_value in enumerate(rows):
        row = _mapping(row_value)
        row_label = f"output_route_{index}"
        if not row:
            violations.append(f"{row_label}_not_object")
            continue
        for key in ROUTING_SENDS_REQUIRED_OUTPUT_ROUTE_FIELDS:
            if key not in row:
                violations.append(f"{row_label}_missing_field:{key}")
        _append_physical_row_identity_violations(
            row_label=row_label,
            row=row,
            violations=violations,
        )
        output_group = row.get("output_group")
        output_number = row.get("output_number")
        if not _string(output_group):
            violations.append(f"{row_label}_output_group_missing")
        if not isinstance(output_number, int) or output_number < 1:
            violations.append(f"{row_label}_output_number_invalid")
        if isinstance(output_group, str) and isinstance(output_number, int):
            key = (output_group, output_number)
            if key in seen_keys:
                violations.append(f"{row_label}_output_duplicate")
            else:
                seen_keys.add(key)
        if not _string(row.get("source_group")):
            violations.append(f"{row_label}_source_group_missing")
        if "source_channel" in row and not isinstance(row.get("source_channel"), int):
            violations.append(f"{row_label}_source_channel_not_int")
        if not _string(row.get("destructive_change_marker")):
            violations.append(f"{row_label}_destructive_change_marker_missing")
        _append_coverage_violations(
            row_label=row_label,
            coverage=_mapping(row.get("coverage")),
            required_flags=ROUTING_SENDS_REQUIRED_OUTPUT_ROUTE_COVERAGE_FLAGS,
            violations=violations,
        )
    return len(rows)


def build_routing_sends_physical_readback_contract() -> Dict[str, Any]:
    """Return the non-executing artifact contract for routing/send proof."""

    return {
        "type": "routing_sends_physical_readback_contract",
        "generated_at": _utc_now(),
        "method_id": ROUTING_SENDS_PHYSICAL_READBACK_CONTRACT_METHOD_ID,
        "schema_id": ROUTING_SENDS_PHYSICAL_READBACK_SCHEMA_ID,
        "status": "contract_ready_physical_fixture_missing",
        "surface_id": ROUTING_SENDS_SURFACE_ID,
        "remediation_action_id": ROUTING_SENDS_REMEDIATION_ACTION_ID,
        "safety": {
            "read_only": True,
            "live_write": False,
            "mutates_console_state": False,
            "queries_console": False,
            "executes_tests": False,
            "live_widening_allowed": False,
            "full_autonomous_ready": False,
        },
        "artifact_contract": {
            "type": ROUTING_SENDS_PHYSICAL_READBACK_ARTIFACT_TYPE,
            "schema_id": ROUTING_SENDS_PHYSICAL_READBACK_SCHEMA_ID,
            "required_top_level_fields": list(ROUTING_SENDS_REQUIRED_TOP_LEVEL_FIELDS),
            "required_static_route_fields": list(
                ROUTING_SENDS_REQUIRED_STATIC_ROUTE_FIELDS
            ),
            "required_bus_send_fields": list(ROUTING_SENDS_REQUIRED_BUS_SEND_FIELDS),
            "required_main_send_fields": list(ROUTING_SENDS_REQUIRED_MAIN_SEND_FIELDS),
            "required_matrix_send_fields": list(
                ROUTING_SENDS_REQUIRED_MATRIX_SEND_FIELDS
            ),
            "required_output_route_fields": list(
                ROUTING_SENDS_REQUIRED_OUTPUT_ROUTE_FIELDS
            ),
            "required_static_route_coverage_flags": list(
                ROUTING_SENDS_REQUIRED_STATIC_ROUTE_COVERAGE_FLAGS
            ),
            "required_send_coverage_flags": list(
                ROUTING_SENDS_REQUIRED_SEND_COVERAGE_FLAGS
            ),
            "required_output_route_coverage_flags": list(
                ROUTING_SENDS_REQUIRED_OUTPUT_ROUTE_COVERAGE_FLAGS
            ),
            "required_false_safety_flags": list(
                ROUTING_SENDS_REQUIRED_FALSE_SAFETY_FLAGS
            ),
            "allowed_sources": list(ROUTING_SENDS_PHYSICAL_READBACK_SOURCES),
            "accepted_row_statuses": list(ROUTING_SENDS_VERIFIED_PHYSICAL_STATUSES),
            "forbidden_mutation_evidence_fields": list(
                ROUTING_SENDS_FORBIDDEN_MUTATION_EVIDENCE_FIELDS
            ),
            "minimum_static_dante_route_count": MINIMUM_STATIC_DANTE_ROUTE_COUNT,
            "minimum_channel_bus_send_count": MINIMUM_CHANNEL_BUS_SEND_COUNT,
            "minimum_channel_main_send_count": MINIMUM_CHANNEL_MAIN_SEND_COUNT,
            "minimum_channel_matrix_send_count": MINIMUM_CHANNEL_MATRIX_SEND_COUNT,
            "minimum_output_route_count": MINIMUM_OUTPUT_ROUTE_COUNT,
        },
        "summary": {
            "physical_readback_surface_id": ROUTING_SENDS_SURFACE_ID,
            "physical_readback_action_id": ROUTING_SENDS_REMEDIATION_ACTION_ID,
            "artifact_required": True,
            "physical_fixture_gap_remains": True,
            "accepted_as_physical_fixture_candidate": False,
            "live_authority_granted": False,
            "promotion_families_requiring_surface": [
                "routing",
                "fx_send_insert",
                "bus_group_matrix",
                "scene_preset",
                "mix_balance",
                "performance_adaptation",
            ],
            "claim": (
                "contract only; a separate operator-run routing/send physical "
                "readback artifact is still required"
            ),
        },
    }


def validate_routing_sends_physical_readback_artifact(
    payload: Mapping[str, Any],
) -> Dict[str, Any]:
    """Validate a future routing/sends artifact without granting authority."""

    violations: list[str] = []
    artifact = _mapping(payload)
    _append_required_field_violations(artifact, violations)
    _append_identity_violations(artifact, violations)
    _append_safety_violations(artifact, violations)
    static_route_count = _append_static_route_violations(artifact, violations)
    bus_send_count = _append_bus_send_violations(artifact, violations)
    main_send_count = _append_main_send_violations(artifact, violations)
    matrix_send_count = _append_matrix_send_violations(artifact, violations)
    output_route_count = _append_output_route_violations(artifact, violations)

    status = (
        "valid_physical_readback_artifact"
        if not violations
        else "invalid_physical_readback_artifact"
    )
    validation = RoutingSendsPhysicalReadbackValidation(
        method_id=ROUTING_SENDS_PHYSICAL_READBACK_CONTRACT_METHOD_ID,
        schema_id=ROUTING_SENDS_PHYSICAL_READBACK_SCHEMA_ID,
        surface_id=str(artifact.get("surface_id") or ""),
        remediation_action_id=str(artifact.get("remediation_action_id") or ""),
        status=status,
        accepted_as_physical_fixture_candidate=not violations,
        live_authority_granted=False,
        full_autonomous_ready=False,
        static_dante_route_count=static_route_count,
        channel_bus_send_count=bus_send_count,
        channel_main_send_count=main_send_count,
        channel_matrix_send_count=matrix_send_count,
        output_route_count=output_route_count,
        violations=tuple(sorted(set(violations))),
        evidence={
            "source": artifact.get("source"),
            "captured_at": artifact.get("captured_at"),
            "console_model": artifact.get("console_model"),
            "static_dante_route_count": artifact.get("static_dante_route_count"),
            "channel_bus_send_count": artifact.get("channel_bus_send_count"),
            "channel_main_send_count": artifact.get("channel_main_send_count"),
            "channel_matrix_send_count": artifact.get("channel_matrix_send_count"),
            "output_route_count": artifact.get("output_route_count"),
            "validated_at": _utc_now(),
        },
    )
    return validation.to_dict()
