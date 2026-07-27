"""Deterministic JSON and concise operator text reports."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .model import InspectorReport


def canonical_json(report: InspectorReport, *, pretty: bool = True) -> str:
    kwargs: dict[str, Any] = {
        "ensure_ascii": False,
        "sort_keys": True,
    }
    if pretty:
        kwargs["indent"] = 2
    else:
        kwargs["separators"] = (",", ":")
    return json.dumps(report.to_dict(), **kwargs) + "\n"


def write_json_report(
    report: InspectorReport,
    output_path: Path,
) -> None:
    path = Path(output_path)
    if path.exists() and path.is_symlink():
        raise ValueError("inspector_report_symlink_rejected")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(canonical_json(report), encoding="utf-8")


def render_text_report(report: InspectorReport) -> str:
    readiness = report.readiness
    score = str(readiness.score) if readiness.score is not None else "UNKNOWN"
    lines = [
        "SYSTEM INSPECTOR — READ ONLY / SNAPSHOT REPLAY",
        f"Profile: {report.profile_name}",
        f"SYSTEM READINESS SCORE: {score}",
        f"Status: {readiness.status.value}",
        f"Coverage: {readiness.observed_coverage:.1%}",
        f"Confidence: {readiness.confidence:.2f}",
        f"Findings: {len(report.findings)}",
    ]
    for finding in report.findings:
        lines.append(
            f"- [{finding.severity.value}] {finding.id}: {finding.title}"
        )
        lines.append(f"  {finding.summary}")
    if readiness.unknown_checks:
        lines.append("Unknown checks:")
        lines.extend(f"- {item}" for item in readiness.unknown_checks)
    return "\n".join(lines)

