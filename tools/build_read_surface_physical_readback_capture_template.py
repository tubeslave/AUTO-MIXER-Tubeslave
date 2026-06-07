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
FOCUSED_VALIDATION_COMMAND = [
    "PYTHONPATH=backend",
    "python3",
    "-m",
    "pytest",
    "-q",
    "tests/test_read_surface_physical_readback_capture_template.py",
    "tests/test_no_wing_release_check.py",
]


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
