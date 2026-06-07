#!/usr/bin/env python3
"""Build an offline acceptance-record template for physical readback candidates."""

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

from backend.read_surface_physical_readback_artifact_review_queue import (  # noqa: E402
    build_read_surface_physical_readback_artifact_review_queue,
)


DEFAULT_OUTPUT = (
    "artifacts/read_surface_physical_readback_fixture_acceptance/"
    "pending_acceptance_record_template.json"
)
PENDING_DECISION_VALUE = "<accepted|rejected>"
PENDING_REVIEWED_BY_VALUE = "<operator-name>"
PENDING_REVIEWED_AT_VALUE = "<UTC ISO timestamp, e.g. 2026-06-07T12:00:00Z>"
FOCUSED_VALIDATION_COMMAND = [
    "PYTHONPATH=backend",
    "python3",
    "-m",
    "pytest",
    "-q",
    "tests/test_read_surface_physical_readback_fixture_acceptance_template.py",
    "tests/test_read_surface_physical_readback_fixture_acceptance_ledger.py",
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


def _candidate_items(
    queue: Mapping[str, Any],
    *,
    surface_id: str | None = None,
) -> list[Mapping[str, Any]]:
    items = queue.get("queue_items") if isinstance(queue.get("queue_items"), list) else []
    candidates = [
        _mapping(item)
        for item in items
        if isinstance(item, Mapping)
        and item.get("accepted_as_physical_fixture_candidate") is True
    ]
    if surface_id:
        candidates = [item for item in candidates if item.get("surface_id") == surface_id]
    return candidates


def _first_artifact_source_id(item: Mapping[str, Any]) -> str:
    artifact_ids = _string_list(item.get("artifact_source_ids"))
    return artifact_ids[0] if artifact_ids else ""


def _candidate_trace(item: Mapping[str, Any], queue: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "source_queue_method_id": str(queue.get("method_id") or ""),
        "source_queue_status": str(queue.get("status") or ""),
        "candidate_status": str(item.get("status") or ""),
        "intake_status": str(item.get("intake_status") or ""),
        "artifact_source_ids": _string_list(item.get("artifact_source_ids")),
        "valid_candidate_count": int(item.get("valid_candidate_count") or 0),
        "invalid_candidate_count": int(item.get("invalid_candidate_count") or 0),
        "target_consistency": item.get("target_consistency") is True,
        "candidate_source_id_missing": item.get("candidate_source_id_missing") is True,
        "accepted_as_physical_fixture_candidate": (
            item.get("accepted_as_physical_fixture_candidate") is True
        ),
        "fixture_closure_allowed_by_queue": False,
        "live_authority_granted": False,
        "full_autonomous_ready": False,
    }


def _template_record(item: Mapping[str, Any], queue: Mapping[str, Any]) -> dict[str, Any]:
    surface_id = str(item.get("surface_id") or "")
    artifact_source_id = _first_artifact_source_id(item)
    return {
        "review_id": f"pending-{_slug(surface_id)}-{_slug(artifact_source_id)}",
        "surface_id": surface_id,
        "artifact_source_id": artifact_source_id,
        "required_contract_method_id": str(
            item.get("required_contract_method_id") or ""
        ),
        "candidate_trace": _candidate_trace(item, queue),
        "decision": PENDING_DECISION_VALUE,
        "reviewed_by": PENDING_REVIEWED_BY_VALUE,
        "reviewed_at": PENDING_REVIEWED_AT_VALUE,
        "review_notes": "<manual review notes>",
        "live_write": False,
        "mutates_console_state": False,
        "queries_console": False,
        "executes_tests": False,
        "execution_allowed_by_acceptance": False,
        "fixture_closure_allowed_by_acceptance": False,
        "fixture_closure_allowed": False,
        "live_widening_allowed": False,
        "live_authority_granted": False,
        "full_autonomous_ready": False,
    }


def build_acceptance_template(
    *,
    review_queue: Mapping[str, Any] | None = None,
    surface_id: str | None = None,
    generated_at: str | None = None,
) -> dict[str, Any]:
    queue = (
        review_queue
        if review_queue is not None
        else build_read_surface_physical_readback_artifact_review_queue()
    )
    summary = _mapping(queue.get("summary"))
    candidates = _candidate_items(queue, surface_id=surface_id)
    template_records = [
        _template_record(item, queue)
        for item in candidates
        if _first_artifact_source_id(item)
    ]
    blocked_by = (
        ["operator_must_review_candidate_artifact_before_acceptance_record"]
        if template_records
        else _string_list(summary.get("blocked_by"))
        or ["operator_must_capture_valid_physical_readback_candidate_first"]
    )
    return {
        "type": "read_surface_physical_readback_fixture_acceptance_template",
        "generated_at": generated_at or utc_now(),
        "status": (
            "operator_acceptance_template_ready"
            if template_records
            else "operator_candidate_missing"
        ),
        "source_queue_status": queue.get("status"),
        "source_queue_method_id": queue.get("method_id"),
        "selected_surface_id": surface_id,
        "read_only": True,
        "live_write": False,
        "mutates_console_state": False,
        "queries_console": False,
        "executes_tests": False,
        "operator_required": True,
        "execution_allowed_by_template": False,
        "fixture_closure_allowed_by_template": False,
        "live_widening_allowed": False,
        "live_authority_granted": False,
        "full_autonomous_ready": False,
        "review_candidate_count": int(summary.get("review_candidate_count") or 0),
        "template_record_count": len(template_records),
        "template_records": template_records,
        "required_operator_edits": [
            "decision",
            "reviewed_by",
            "reviewed_at",
            "review_notes",
        ],
        "blocked_by": blocked_by,
        "operator_sequence": [
            "Confirm the artifact is a valid review candidate from the review queue.",
            "Confirm candidate_trace still matches the review queue candidate.",
            "Set decision to accepted or rejected after manual review.",
            "Set reviewed_by and reviewed_at using a UTC ISO timestamp.",
            "Keep all authority and fixture-closure flags false.",
            "Run the fixture acceptance ledger before any separate closure decision.",
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
        "# Read Surface Physical Readback Fixture Acceptance Template",
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
        f"- live_widening_allowed: `{_bool(template['live_widening_allowed'])}`",
        f"- live_authority_granted: `{_bool(template['live_authority_granted'])}`",
        f"- full_autonomous_ready: `{_bool(template['full_autonomous_ready'])}`",
        "",
        "## Candidate Summary",
        "",
        f"- source_queue_status: `{template['source_queue_status']}`",
        f"- selected_surface_id: `{template['selected_surface_id']}`",
        f"- review_candidate_count: `{template['review_candidate_count']}`",
        f"- template_record_count: `{template['template_record_count']}`",
        f"- blocked_by: `{json.dumps(template['blocked_by'])}`",
        "",
        "## Template Records",
        "",
    ]
    records = template.get("template_records")
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
            "Build an offline read-surface physical readback fixture acceptance "
            "template."
        )
    )
    parser.add_argument("--format", choices=("json", "markdown"), default="json")
    parser.add_argument("--surface-id")
    parser.add_argument("--output", default=DEFAULT_OUTPUT)
    parser.add_argument("--quiet", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    template = build_acceptance_template(surface_id=args.surface_id)
    text = (
        render_markdown(template)
        if args.format == "markdown"
        else json.dumps(template, ensure_ascii=False, indent=2, sort_keys=True)
    )
    write_output(Path(args.output), text)
    if not args.quiet:
        print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
