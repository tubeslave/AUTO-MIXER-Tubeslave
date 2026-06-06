#!/usr/bin/env python3
"""Review WING read-only preflight artifacts without contacting hardware."""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence


REPO_ROOT = Path(__file__).resolve().parents[1]
BACKEND_ROOT = REPO_ROOT / "backend"
for path in (str(REPO_ROOT), str(BACKEND_ROOT)):
    if path not in sys.path:
        sys.path.insert(0, path)

from backend.wing_readonly_preflight_evidence import (  # noqa: E402
    DEFAULT_WING_READONLY_PREFLIGHT_ARTIFACT_ROOTS,
    build_wing_readonly_preflight_evidence,
)


DEFAULT_OUTPUT = REPO_ROOT / "artifacts" / "wing_readonly_preflight_review" / "latest_review.md"


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


def _string_list(value: Any) -> list[str]:
    return [str(item) for item in value if item] if isinstance(value, list) else []


def build_review_report(
    *,
    artifact_roots: Iterable[str | Path] | None = None,
    artifact_paths: Iterable[str | Path] | None = None,
) -> dict[str, Any]:
    evidence = build_wing_readonly_preflight_evidence(
        artifact_roots=artifact_roots,
        artifact_paths=artifact_paths,
    )
    summary = evidence.get("summary") if isinstance(evidence.get("summary"), Mapping) else {}
    safety = evidence.get("safety") if isinstance(evidence.get("safety"), Mapping) else {}
    artifacts = evidence.get("artifacts") if isinstance(evidence.get("artifacts"), list) else []
    valid_candidate_count = int(summary.get("valid_candidate_count") or 0)
    invalid_candidate_count = int(summary.get("invalid_candidate_count") or 0)
    candidate_available = (
        evidence.get("status") == "readonly_preflight_candidate_available"
        and valid_candidate_count > 0
        and invalid_candidate_count == 0
        and summary.get("readiness_matrix_update_allowed") is True
        and summary.get("readiness_closure_allowed_by_evidence") is not True
        and summary.get("live_authority_granted") is not True
        and summary.get("full_autonomous_ready") is not True
    )
    return {
        "type": "wing_readonly_preflight_artifact_review",
        "generated_at": utc_now(),
        "status": "candidate_ready_for_manual_review" if candidate_available else "blocked",
        "readiness_update_candidate_available": candidate_available,
        "read_only": safety.get("read_only") is True,
        "live_write": safety.get("live_write") is True,
        "queries_console": safety.get("queries_console") is True,
        "execution_allowed_by_review": False,
        "live_authority_granted": False,
        "full_autonomous_ready": False,
        "evidence_status": evidence.get("status"),
        "artifact_count": int(summary.get("artifact_count") or 0),
        "valid_candidate_count": valid_candidate_count,
        "invalid_candidate_count": invalid_candidate_count,
        "latest_valid_candidate_run_id": summary.get("latest_valid_candidate_run_id"),
        "blocked_by": _string_list(summary.get("blocked_by")),
        "violations": _string_list(summary.get("violations")),
        "next_required_action": (
            "Operator manually reviews summary.json and events.jsonl before readiness update."
            if candidate_available
            else str(summary.get("next_global_step") or "Capture a clean WING read-only preflight artifact.")
        ),
        "evidence": evidence,
        "artifact_sources": [
            str(item.get("source_id"))
            for item in artifacts
            if isinstance(item, Mapping) and item.get("source_id")
        ],
    }


def exit_code_for_report(report: Mapping[str, Any]) -> int:
    if report.get("readiness_update_candidate_available") is True:
        return 0
    if int(report.get("artifact_count") or 0) <= 0:
        return 2
    return 3


def render_markdown(report: Mapping[str, Any]) -> str:
    lines = [
        "# WING Read-Only Preflight Artifact Review",
        "",
        f"Generated: `{report['generated_at']}`",
        f"Status: `{report['status']}`",
        "",
        "## Summary",
        "",
        f"- evidence_status: `{report['evidence_status']}`",
        f"- artifact_count: `{report['artifact_count']}`",
        f"- valid_candidate_count: `{report['valid_candidate_count']}`",
        f"- invalid_candidate_count: `{report['invalid_candidate_count']}`",
        f"- readiness_update_candidate_available: `{str(report['readiness_update_candidate_available']).lower()}`",
        f"- latest_valid_candidate_run_id: `{report['latest_valid_candidate_run_id']}`",
        "",
        "## Safety",
        "",
        f"- read_only: `{str(report['read_only']).lower()}`",
        f"- live_write: `{str(report['live_write']).lower()}`",
        f"- queries_console: `{str(report['queries_console']).lower()}`",
        f"- execution_allowed_by_review: `{str(report['execution_allowed_by_review']).lower()}`",
        f"- live_authority_granted: `{str(report['live_authority_granted']).lower()}`",
        f"- full_autonomous_ready: `{str(report['full_autonomous_ready']).lower()}`",
        "",
        "## Blockers",
        "",
    ]
    blocked_by = report.get("blocked_by") if isinstance(report.get("blocked_by"), list) else []
    lines.extend(f"- `{item}`" for item in blocked_by)
    if not blocked_by:
        lines.append("- none")
    lines.extend(["", "## Violations", ""])
    violations = report.get("violations") if isinstance(report.get("violations"), list) else []
    lines.extend(f"- `{item}`" for item in violations)
    if not violations:
        lines.append("- none")
    lines.extend(["", "## Artifact Sources", ""])
    sources = report.get("artifact_sources") if isinstance(report.get("artifact_sources"), list) else []
    lines.extend(f"- `{item}`" for item in sources)
    if not sources:
        lines.append("- none")
    lines.extend(
        [
            "",
            "## Next Required Action",
            "",
            str(report.get("next_required_action") or ""),
            "",
            "## Boundary",
            "",
            "This review is offline and read-only. It does not contact WING,",
            "does not grant live authority, and does not close readiness rows",
            "without operator review of real artifacts.",
            "",
        ]
    )
    return "\n".join(lines)


def write_output(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Review WING read-only preflight artifacts.")
    parser.add_argument(
        "--artifact-root",
        action="append",
        default=[],
        help="Artifact root to scan for summary.json files. May be passed multiple times.",
    )
    parser.add_argument(
        "--artifact-path",
        action="append",
        default=[],
        help="Exact summary.json path to review. May be passed multiple times.",
    )
    parser.add_argument("--format", choices=("markdown", "json"), default="markdown")
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT))
    parser.add_argument("--quiet", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    roots = tuple(args.artifact_root) or DEFAULT_WING_READONLY_PREFLIGHT_ARTIFACT_ROOTS
    paths = tuple(args.artifact_path) or None
    report = build_review_report(artifact_roots=roots, artifact_paths=paths)
    text = (
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True)
        if args.format == "json"
        else render_markdown(report)
    )
    write_output(Path(args.output), text)
    if not args.quiet:
        print(text)
    return exit_code_for_report(report)


if __name__ == "__main__":
    sys.exit(main())
