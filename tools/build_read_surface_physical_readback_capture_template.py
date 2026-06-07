#!/usr/bin/env python3
"""Build a fail-closed template for operator physical-readback capture artifacts."""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence


REPO_ROOT = Path(__file__).resolve().parents[1]
BACKEND_ROOT = REPO_ROOT / "backend"
for path in (str(REPO_ROOT), str(BACKEND_ROOT)):
    if path not in sys.path:
        sys.path.insert(0, path)

from tools.build_read_surface_physical_readback_capture_packet import (  # noqa: E402
    build_capture_packet,
)
from backend.bus_group_main_dca_physical_readback_contract import (  # noqa: E402
    build_bus_group_main_dca_physical_readback_contract,
)
from backend.channel_basic_controls_physical_readback_contract import (  # noqa: E402
    build_channel_basic_controls_physical_readback_contract,
)
from backend.channel_processing_physical_readback_contract import (  # noqa: E402
    build_channel_processing_physical_readback_contract,
)
from backend.dlive_state_physical_readback_contract import (  # noqa: E402
    build_dlive_state_physical_readback_contract,
)
from backend.fx_slots_inserts_physical_readback_contract import (  # noqa: E402
    build_fx_slots_inserts_physical_readback_contract,
)
from backend.matrix_state_physical_readback_contract import (  # noqa: E402
    build_matrix_state_physical_readback_contract,
)
from backend.raw_mixer_state_physical_readback_contract import (  # noqa: E402
    build_raw_mixer_state_physical_readback_contract,
)
from backend.routing_sends_physical_readback_contract import (  # noqa: E402
    build_routing_sends_physical_readback_contract,
)
from backend.snapshots_scenes_physical_readback_contract import (  # noqa: E402
    build_snapshots_scenes_physical_readback_contract,
)
from backend.user_presets_physical_readback_contract import (  # noqa: E402
    build_user_presets_physical_readback_contract,
)


DEFAULT_OUTPUT = (
    "artifacts/read_surface_physical_readback_capture/"
    "pending_capture_artifact_template.json"
)
TEMPLATE_TYPE = "read_surface_physical_readback_artifact_capture_template"
READY_STATUS = "operator_capture_template_ready"
COMPLETE_STATUS = "read_surface_physical_readback_complete"
TARGET_MISSING_STATUS = "operator_capture_target_missing"
PENDING_ARTIFACT_SOURCE_ID = "<path-to-operator-captured-json>"
PENDING_CAPTURED_AT = "<UTC ISO timestamp, e.g. 2026-06-07T12:00:00Z>"
PENDING_CAPTURED_BY = "<operator-name>"
PENDING_SOURCE = "<physical_readback|wing_physical_readback|dlive_physical_readback>"
PENDING_CONSOLE_MODEL = "<console model and firmware>"
PENDING_CAPTURE_SCOPE = "<manual capture scope and show-state notes>"
PENDING_PHYSICAL_READBACK_PAYLOAD = (
    "<replace with JSON payload that matches required_contract_method_id>"
)
PAYLOAD_TEMPLATE_STATUS = "operator_must_fill_from_physical_readback"
FOCUSED_VALIDATION_COMMAND = [
    "PYTHONPATH=backend",
    "python3",
    "-m",
    "pytest",
    "-q",
    "tests/test_read_surface_physical_readback_capture_template.py",
    "tests/test_no_wing_release_check.py",
]
CONTRACT_BUILDERS = {
    "raw_mixer_state_physical_readback_contract": build_raw_mixer_state_physical_readback_contract,
    "channel_basic_controls_physical_readback_contract": build_channel_basic_controls_physical_readback_contract,
    "channel_processing_physical_readback_contract": build_channel_processing_physical_readback_contract,
    "bus_group_main_dca_physical_readback_contract": build_bus_group_main_dca_physical_readback_contract,
    "matrix_state_physical_readback_contract": build_matrix_state_physical_readback_contract,
    "fx_slots_inserts_physical_readback_contract": build_fx_slots_inserts_physical_readback_contract,
    "routing_sends_physical_readback_contract": build_routing_sends_physical_readback_contract,
    "snapshots_scenes_physical_readback_contract": build_snapshots_scenes_physical_readback_contract,
    "user_presets_physical_readback_contract": build_user_presets_physical_readback_contract,
    "dlive_state_physical_readback_contract": build_dlive_state_physical_readback_contract,
}
COLLECTION_FIELD_TO_REQUIRED_FIELDS = {
    "readback_rows": "required_row_fields",
    "channels": "required_channel_fields",
    "buses": "required_bus_fields",
    "mains": "required_main_fields",
    "dcas": "required_dca_fields",
    "matrices": "required_matrix_fields",
    "fx_slots": "required_fx_slot_fields",
    "inserts": "required_insert_fields",
    "static_dante_routes": "required_static_route_fields",
    "channel_bus_sends": "required_bus_send_fields",
    "channel_main_sends": "required_main_send_fields",
    "channel_matrix_sends": "required_matrix_send_fields",
    "output_routes": "required_output_route_fields",
    "snapshots": "required_snapshot_fields",
    "presets": "required_preset_fields",
}
OBJECT_FIELD_TO_REQUIRED_FIELDS = {
    "active_scene": "required_active_scene_fields",
    "midi_state": "required_midi_state_fields",
    "parity": "required_parity_fields",
    "identity": "required_identity_fields",
    "controls": "required_control_fields",
    "hpf": "required_hpf_fields",
    "eq": "required_eq_fields",
    "compressor": "required_compressor_fields",
    "gate": "required_gate_fields",
    "dynamics": "required_dynamics_fields",
    "insert": "required_insert_fields",
    "metering": "required_metering_fields",
}


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


def _mapping(value: Any) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


def _string_list(value: Any) -> list[str]:
    return [str(item) for item in value if item] if isinstance(value, list) else []


def _slug(value: str) -> str:
    chars = [char.lower() if char.isalnum() else "-" for char in value]
    slug = "-".join(part for part in "".join(chars).split("-") if part)
    return slug or "pending"


def _contract_for_method(method_id: str) -> Mapping[str, Any]:
    builder = CONTRACT_BUILDERS.get(method_id)
    if builder is None:
        return {}
    contract = builder()
    return contract if isinstance(contract, Mapping) else {}


def _artifact_contract_for_method(method_id: str) -> Mapping[str, Any]:
    contract = _contract_for_method(method_id)
    artifact_contract = contract.get("artifact_contract")
    return artifact_contract if isinstance(artifact_contract, Mapping) else {}


def _required_field_sets(artifact_contract: Mapping[str, Any]) -> dict[str, list[str]]:
    return {
        str(key): _string_list(value)
        for key, value in artifact_contract.items()
        if isinstance(key, str)
        and key.startswith("required_")
        and key != "required_top_level_fields"
    }


def _placeholder_for_field(field_name: str) -> str:
    return f"<operator-captured {field_name}>"


def _coverage_template(artifact_contract: Mapping[str, Any]) -> dict[str, str]:
    for key in (
        "required_coverage_flags",
        "required_bus_coverage_flags",
        "required_main_coverage_flags",
        "required_dca_coverage_flags",
        "required_fx_slot_coverage_flags",
        "required_insert_coverage_flags",
        "required_static_route_coverage_flags",
        "required_send_coverage_flags",
        "required_output_route_coverage_flags",
    ):
        flags = _string_list(artifact_contract.get(key))
        if flags:
            return {
                flag: "<true only after physical readback confirms this coverage>"
                for flag in flags
            }
    return {"cache_matches_physical_readback": "<true only after physical readback>"}


def _object_template(
    fields: list[str],
    artifact_contract: Mapping[str, Any],
) -> dict[str, Any]:
    payload: dict[str, Any] = {}
    for field in fields:
        nested_key = OBJECT_FIELD_TO_REQUIRED_FIELDS.get(field)
        if nested_key:
            payload[field] = _object_template(
                _string_list(artifact_contract.get(nested_key)),
                artifact_contract,
            )
        elif field == "coverage":
            payload[field] = _coverage_template(artifact_contract)
        elif field == "source":
            payload[field] = PENDING_SOURCE
        elif field == "captured_at":
            payload[field] = PENDING_CAPTURED_AT
        elif field == "status":
            payload[field] = "<verified_from_physical_readback>"
        elif field.endswith("_count"):
            payload[field] = f"<integer {field}>"
        else:
            payload[field] = _placeholder_for_field(field)
    return payload


def _top_level_value(
    field: str,
    *,
    target: Mapping[str, Any],
    artifact_contract: Mapping[str, Any],
) -> Any:
    if field == "type":
        return str(artifact_contract.get("type") or "")
    if field == "schema_id":
        return str(artifact_contract.get("schema_id") or "")
    if field == "surface_id":
        return str(target.get("surface_id") or "")
    if field == "remediation_action_id":
        return str(target.get("remediation_action_id") or "")
    if field == "captured_at":
        return PENDING_CAPTURED_AT
    if field == "source":
        return PENDING_SOURCE
    if field == "console_model":
        return PENDING_CONSOLE_MODEL
    if field == "safety":
        safety = {"read_only": True}
        for flag in _string_list(artifact_contract.get("required_false_safety_flags")):
            safety[flag] = False
        safety.setdefault("live_write", False)
        safety.setdefault("mutates_console_state", False)
        safety.setdefault("live_widening_allowed", False)
        safety.setdefault("full_autonomous_ready", False)
        safety["live_authority_granted"] = False
        safety["accepted_as_physical_fixture_candidate"] = False
        return safety
    collection_key = COLLECTION_FIELD_TO_REQUIRED_FIELDS.get(field)
    if collection_key:
        fields = _string_list(artifact_contract.get(collection_key))
        return [_object_template(fields, artifact_contract)]
    object_key = OBJECT_FIELD_TO_REQUIRED_FIELDS.get(field)
    if object_key:
        return _object_template(
            _string_list(artifact_contract.get(object_key)),
            artifact_contract,
        )
    if field.endswith("_count"):
        return f"<integer {field}>"
    return _placeholder_for_field(field)


def _physical_readback_payload_template(target: Mapping[str, Any]) -> dict[str, Any]:
    contract_method_id = str(target.get("contract_method_id") or "")
    artifact_contract = _artifact_contract_for_method(contract_method_id)
    required_top_level_fields = _string_list(
        artifact_contract.get("required_top_level_fields")
    )
    payload = {
        field: _top_level_value(
            field,
            target=target,
            artifact_contract=artifact_contract,
        )
        for field in required_top_level_fields
    }
    return {
        "template_status": PAYLOAD_TEMPLATE_STATUS,
        "required_contract_method_id": contract_method_id,
        "artifact_type": str(artifact_contract.get("type") or ""),
        "schema_id": str(artifact_contract.get("schema_id") or ""),
        "required_top_level_fields": required_top_level_fields,
        "required_field_sets": _required_field_sets(artifact_contract),
        "payload": payload,
    }


def _targets_from_packet(
    packet: Mapping[str, Any],
    *,
    surface_id: str | None = None,
    include_all_targets: bool = False,
) -> list[Mapping[str, Any]]:
    targets = [
        _mapping(target)
        for target in packet.get("remaining_capture_targets", [])
        if isinstance(target, Mapping)
    ]
    if surface_id:
        return [target for target in targets if target.get("surface_id") == surface_id]
    if include_all_targets:
        return targets
    next_surface_id = packet.get("next_physical_readback_surface_id")
    if isinstance(next_surface_id, str) and next_surface_id:
        return [target for target in targets if target.get("surface_id") == next_surface_id]
    return targets[:1]


def _template_record(target: Mapping[str, Any]) -> dict[str, Any]:
    surface_id = str(target.get("surface_id") or "")
    return {
        "template_id": f"pending-{_slug(surface_id)}-physical-readback",
        "surface_id": surface_id,
        "remediation_action_id": str(target.get("remediation_action_id") or ""),
        "target_console_surface": str(target.get("target_console_surface") or ""),
        "read_mode": str(target.get("read_mode") or ""),
        "requires_connection": target.get("requires_connection") is True,
        "required_contract_method_id": str(target.get("contract_method_id") or ""),
        "required_artifact_contracts": _string_list(
            target.get("required_artifact_contracts")
        ),
        "expected_artifacts": _string_list(target.get("expected_artifacts")),
        "blockers_to_clear": _string_list(target.get("blockers_to_clear")),
        "artifact_source_id": PENDING_ARTIFACT_SOURCE_ID,
        "captured_at": PENDING_CAPTURED_AT,
        "captured_by": PENDING_CAPTURED_BY,
        "source": PENDING_SOURCE,
        "console_model": PENDING_CONSOLE_MODEL,
        "capture_scope": PENDING_CAPTURE_SCOPE,
        "physical_readback_payload": PENDING_PHYSICAL_READBACK_PAYLOAD,
        "physical_readback_payload_template": _physical_readback_payload_template(
            target
        ),
        "safety": {
            "read_only": True,
            "live_write": False,
            "mutates_console_state": False,
            "queries_console": False,
            "executes_tests": False,
            "operator_required": True,
            "execution_allowed_by_template": False,
            "accepted_as_physical_fixture_candidate": False,
            "fixture_closure_allowed_by_template": False,
            "live_widening_allowed": False,
            "live_authority_granted": False,
            "full_autonomous_ready": False,
        },
    }


def build_capture_template(
    *,
    capture_packet: Mapping[str, Any] | None = None,
    surface_id: str | None = None,
    include_all_targets: bool = False,
    generated_at: str | None = None,
) -> dict[str, Any]:
    packet = capture_packet if capture_packet is not None else build_capture_packet()
    targets = _targets_from_packet(
        packet,
        surface_id=surface_id,
        include_all_targets=include_all_targets,
    )
    records = [_template_record(target) for target in targets]
    selected_surface_id = (
        surface_id
        or packet.get("next_physical_readback_surface_id")
        if records and not include_all_targets
        else surface_id
    )
    blocked_by = _string_list(packet.get("blocked_by"))
    if records:
        blocked_by = list(
            dict.fromkeys(
                blocked_by
                + [
                    "operator_must_replace_capture_template_placeholders",
                    "operator_must_validate_filled_artifact_with_required_contract",
                    "operator_must_review_artifact_before_fixture_acceptance",
                ]
            )
        )
    elif surface_id:
        blocked_by = [
            "requested_surface_id_has_no_remaining_capture_target",
        ]
    status = READY_STATUS if records else COMPLETE_STATUS
    if not records and surface_id:
        status = TARGET_MISSING_STATUS
    return {
        "type": TEMPLATE_TYPE,
        "generated_at": generated_at or utc_now(),
        "status": status,
        "source_packet_type": packet.get("type"),
        "source_packet_status": packet.get("status"),
        "source_method_id": packet.get("source_method_id"),
        "selected_surface_id": selected_surface_id,
        "include_all_targets": include_all_targets,
        "read_only": True,
        "live_write": False,
        "mutates_console_state": False,
        "queries_console": False,
        "executes_tests": False,
        "operator_required": True,
        "execution_allowed_by_template": False,
        "accepted_as_physical_fixture_candidate": False,
        "fixture_closure_allowed_by_template": False,
        "live_widening_allowed": False,
        "live_authority_granted": False,
        "full_autonomous_ready": False,
        "all_remaining_target_count": int(
            packet.get("remaining_capture_target_count") or 0
        ),
        "capture_template_count": len(records),
        "capture_templates": records,
        "required_operator_edits": [
            "artifact_source_id",
            "captured_at",
            "captured_by",
            "source",
            "console_model",
            "capture_scope",
            "physical_readback_payload",
            "physical_readback_payload_template.payload",
        ],
        "blocked_by": blocked_by,
        "operator_sequence": [
            "Confirm the selected surface and console/show state before capture.",
            "Replace every placeholder with real operator-captured physical readback data.",
            "Reject the capture path if it would send a mixer-control message or mutate console state.",
            "Keep authority, execution, fixture-closure, and autonomous-readiness flags false.",
            "Validate the filled artifact with the listed required contract before review queue intake.",
            "Use the review queue and acceptance ledger before any separate fixture-closure decision.",
        ],
        "commands": {
            "focused_validation": list(FOCUSED_VALIDATION_COMMAND),
        },
    }


def _bool(value: Any) -> str:
    return str(bool(value)).lower()


def _command_text(command: Sequence[str]) -> str:
    return " ".join(command)


def render_markdown(template: Mapping[str, Any]) -> str:
    commands = _mapping(template.get("commands"))
    lines = [
        "# Read Surface Physical Readback Capture Template",
        "",
        f"Generated: `{template['generated_at']}`",
        f"Status: `{template['status']}`",
        "",
        "## Safety Boundary",
        "",
        f"- read_only: `{_bool(template['read_only'])}`",
        f"- live_write: `{_bool(template['live_write'])}`",
        f"- mutates_console_state: `{_bool(template['mutates_console_state'])}`",
        f"- queries_console: `{_bool(template['queries_console'])}`",
        f"- executes_tests: `{_bool(template['executes_tests'])}`",
        f"- operator_required: `{_bool(template['operator_required'])}`",
        "- execution_allowed_by_template: "
        f"`{_bool(template['execution_allowed_by_template'])}`",
        "- fixture_closure_allowed_by_template: "
        f"`{_bool(template['fixture_closure_allowed_by_template'])}`",
        f"- live_authority_granted: `{_bool(template['live_authority_granted'])}`",
        f"- full_autonomous_ready: `{_bool(template['full_autonomous_ready'])}`",
        "",
        "## Capture Summary",
        "",
        f"- source_packet_status: `{template['source_packet_status']}`",
        f"- selected_surface_id: `{template['selected_surface_id']}`",
        f"- all_remaining_target_count: `{template['all_remaining_target_count']}`",
        f"- capture_template_count: `{template['capture_template_count']}`",
        f"- blocked_by: `{json.dumps(template['blocked_by'])}`",
        "",
        "## Capture Templates",
        "",
    ]
    records = template.get("capture_templates")
    if isinstance(records, list) and records:
        lines.extend(["```json", json.dumps(records, ensure_ascii=False, indent=2), "```"])
    else:
        lines.append("- none")
    lines.extend(["", "## Operator Sequence", ""])
    for index, item in enumerate(template.get("operator_sequence") or [], start=1):
        lines.append(f"{index}. {item}")
    focused_validation = commands.get("focused_validation")
    if isinstance(focused_validation, list):
        lines.extend(
            [
                "",
                "## Focused Validation",
                "",
                "```bash",
                _command_text([str(item) for item in focused_validation]),
                "```",
            ]
        )
    lines.append("")
    return "\n".join(lines)


def write_output(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Build an offline template for operator read-surface physical "
            "readback capture artifacts."
        )
    )
    parser.add_argument("--format", choices=("json", "markdown"), default="json")
    parser.add_argument("--surface-id")
    parser.add_argument("--include-all-targets", action="store_true")
    parser.add_argument("--output", default=DEFAULT_OUTPUT)
    parser.add_argument("--quiet", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    template = build_capture_template(
        surface_id=args.surface_id,
        include_all_targets=args.include_all_targets,
    )
    text = (
        json.dumps(template, ensure_ascii=False, indent=2, sort_keys=True)
        if args.format == "json"
        else render_markdown(template)
    )
    write_output(Path(args.output), text)
    if not args.quiet:
        print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
