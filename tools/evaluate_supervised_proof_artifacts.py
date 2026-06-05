#!/usr/bin/env python3
"""Build an offline report from supervised fader/gain proof summaries.

The tool reads local JSON files only. It is intended for pre-promotion review
after dry-run or physical proof runners have written artifacts.
"""

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

from backend.supervised_proof_artifact_evaluator import (  # noqa: E402
    EVALUATION_STATUSES,
    evaluate_supervised_proof_artifact_dict,
)
from backend.autonomy_evidence_overlay import build_autonomy_evidence_overlay  # noqa: E402
from backend.autonomy_mode_transition_gate import build_autonomy_mode_transition_gate  # noqa: E402
from backend.autonomy_promotion_readiness import build_autonomy_promotion_readiness_snapshot  # noqa: E402
from backend.supervised_promotion_evidence import build_supervised_promotion_evidence_snapshot  # noqa: E402
from backend.supervised_proof_remediation_plan import build_supervised_proof_remediation_plan  # noqa: E402
from backend.supervised_wing_proof_plan import build_supervised_wing_proof_plan  # noqa: E402


DEFAULT_ARTIFACT_ROOTS = (REPO_ROOT / "artifacts" / "wing_telemetry",)
DEFAULT_OUTPUT_ROOT = REPO_ROOT / "artifacts" / "supervised_proof_artifact_report"


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


def compact_stamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def discover_summary_paths(artifact_roots: Iterable[Path]) -> list[Path]:
    paths: set[Path] = set()
    for root in artifact_roots:
        path = Path(root)
        if path.is_file():
            if path.name == "summary.json":
                paths.add(path)
            continue
        if path.is_dir():
            paths.update(path.rglob("summary.json"))
    return sorted(paths)


def load_json_object(path: Path) -> tuple[dict[str, Any] | None, str | None]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        return None, f"json_load_error:{type(exc).__name__}"
    if not isinstance(payload, dict):
        return None, "json_payload_not_object"
    return payload, None


def is_supervised_proof_like_payload(payload: Mapping[str, Any]) -> bool:
    if "proof_mode" in payload and "value_type" in payload:
        return True
    if "apply_result" in payload and "rollback_result" in payload:
        return True
    if payload.get("write_policy") and payload.get("channel") and "apply_result" in payload:
        return True
    if payload.get("run_id") and "proof" in str(payload.get("run_id")) and "apply_result" in payload:
        return True
    return False


def _row_with_path(path: Path, payload: Mapping[str, Any], *, require_physical: bool) -> dict[str, Any]:
    result = evaluate_supervised_proof_artifact_dict(payload, require_physical=require_physical).to_dict()
    result["artifact_path"] = str(path)
    return result


def _mode_gate_rows(report: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    gate = report.get("autonomy_mode_transition_gate")
    if not isinstance(gate, Mapping):
        return []
    rows = gate.get("rows")
    if not isinstance(rows, list):
        return []
    return [row for row in rows if isinstance(row, Mapping)]


def _mode_gate_row(report: Mapping[str, Any], mode: str) -> Mapping[str, Any]:
    for row in _mode_gate_rows(report):
        if row.get("mode") == mode:
            return row
    return {}


def _string_list(value: Any) -> list[str]:
    if not isinstance(value, list):
        return []
    return [str(item) for item in value if item]


def _gap_bucket_counts(value: Any) -> dict[str, int]:
    if not isinstance(value, Mapping):
        return {}
    return {
        str(bucket): max(0, int(count or 0))
        for bucket, count in value.items()
        if int(count or 0) > 0
    }


def _gap_targets_by_bucket(value: Any) -> dict[str, list[str]]:
    if not isinstance(value, Mapping):
        return {}
    normalized: dict[str, list[str]] = {}
    for bucket, target_ids in value.items():
        targets = _string_list(target_ids)
        if targets:
            normalized[str(bucket)] = targets
    return normalized


def _string_list_map(value: Any) -> dict[str, list[str]]:
    if not isinstance(value, Mapping):
        return {}
    normalized: dict[str, list[str]] = {}
    for key, raw_items in value.items():
        items = _string_list(raw_items)
        if items:
            normalized[str(key)] = items
    return normalized


def _promotion_gap_summary(report: Mapping[str, Any]) -> Mapping[str, Any]:
    readiness = report.get("autonomy_promotion_readiness")
    if not isinstance(readiness, Mapping):
        readiness = build_autonomy_promotion_readiness_snapshot()
    summary = readiness.get("summary") if isinstance(readiness.get("summary"), Mapping) else {}
    gap_summary = summary.get("promotion_gap_summary")
    return gap_summary if isinstance(gap_summary, Mapping) else {}


def build_autonomy_gap_rollup(report: Mapping[str, Any]) -> dict[str, Any]:
    auto_row = _mode_gate_row(report, "auto")
    supervised_row = _mode_gate_row(report, "supervised")
    gap_summary = _promotion_gap_summary(report)
    gap_counts = _gap_bucket_counts(auto_row.get("readiness_gap_bucket_counts"))
    gap_targets = _gap_targets_by_bucket(auto_row.get("readiness_gap_target_ids_by_bucket"))
    evidence_by_family = _string_list_map(gap_summary.get("evidence_requirement_ids_by_action_family"))
    next_by_family = _string_list_map(gap_summary.get("next_required_evidence_by_action_family"))
    supervised_live_write_kinds = _string_list(supervised_row.get("current_live_write_kinds"))

    return {
        "read_only": True,
        "live_write": False,
        "live_widening_allowed": False,
        "full_autonomous_ready": False,
        "source": "autonomy_mode_transition_gate.rows[mode=auto]",
        "full_autonomous_blocked": True,
        "full_auto_gap_bucket_counts": gap_counts,
        "full_auto_gap_target_ids_by_bucket": gap_targets,
        "evidence_requirement_ids_by_action_family": evidence_by_family,
        "next_required_evidence_by_action_family": next_by_family,
        "supervised_live_apply_allowed": bool(supervised_row.get("live_apply_allowed")),
        "supervised_current_live_write_kinds": supervised_live_write_kinds,
        "next_required_evidence": [
            "Reduce every Full Autonomous readiness gap bucket to zero before reviewing auto mode.",
            "Keep live apply limited to current supervised write kinds until physical readback and rollback evidence exists.",
        ],
        "claim": (
            "Full Autonomous remains blocked; this rollup is report-only and does not widen "
            "the current supervised fader/gain live-write boundary."
        ),
    }


def build_supervised_proof_artifact_report(
    summary_paths: Iterable[Path],
    *,
    require_physical: bool = False,
) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    invalid: list[dict[str, str]] = []
    ignored: list[str] = []
    source_paths = [Path(path) for path in summary_paths]

    for path in source_paths:
        payload, error = load_json_object(path)
        if error is not None:
            invalid.append({"artifact_path": str(path), "reason": error})
            continue
        assert payload is not None
        if not is_supervised_proof_like_payload(payload):
            ignored.append(str(path))
            continue
        rows.append(_row_with_path(path, payload, require_physical=require_physical))

    status_counts: dict[str, int] = {status: 0 for status in sorted(EVALUATION_STATUSES)}
    proof_mode_counts: dict[str, int] = {}
    value_type_counts: dict[str, int] = {}
    for row in rows:
        status_counts[str(row["status"])] += 1
        proof_mode = str(row["proof_mode"])
        value_type = str(row["value_type"])
        proof_mode_counts[proof_mode] = proof_mode_counts.get(proof_mode, 0) + 1
        value_type_counts[value_type] = value_type_counts.get(value_type, 0) + 1

    promotion_ids = [str(row["proof_id"]) for row in rows if row["promotion_allowed"]]
    blocked_ids = [str(row["proof_id"]) for row in rows if row["status"] == "blocked"]
    dry_run_ids = [str(row["proof_id"]) for row in rows if row["status"] == "dry_run_ready"]
    if invalid:
        report_status = "blocked"
    elif not rows:
        report_status = "no_artifacts"
    else:
        report_status = "ok"

    return {
        "type": "supervised_proof_artifact_report",
        "generated_at": utc_now(),
        "status": report_status,
        "safety": {
            "read_only": True,
            "live_write": False,
            "live_widening_allowed": False,
            "full_autonomous_ready": False,
            "supervised_only": True,
        },
        "summary": {
            "source_summary_count": len(source_paths),
            "artifact_count": len(rows),
            "ignored_summary_count": len(ignored),
            "invalid_summary_count": len(invalid),
            "status_counts": status_counts,
            "proof_mode_counts": proof_mode_counts,
            "value_type_counts": value_type_counts,
            "promotion_allowed_proof_ids": promotion_ids,
            "dry_run_ready_proof_ids": dry_run_ids,
            "blocked_proof_ids": blocked_ids,
            "autonomous_ready_proof_ids": [],
            "require_physical": require_physical,
            "next_required_evidence": [
                "Run physical supervised fader/gain proof before changing safety coverage rows."
                if not promotion_ids
                else "Review physical proof rows manually before changing safety coverage rows."
            ],
        },
        "rows": rows,
        "invalid_summaries": invalid,
        "ignored_summaries": ignored,
    }


def write_report(report: Mapping[str, Any], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, indent=2, ensure_ascii=False, sort_keys=True) + "\n", encoding="utf-8")


def exit_code_for_report(
    report: Mapping[str, Any],
    *,
    min_promotion_candidates: int = 0,
    fail_on_blocked: bool = False,
) -> int:
    summary = report.get("summary") if isinstance(report.get("summary"), Mapping) else {}
    invalid_count = int(summary.get("invalid_summary_count", 0))
    promotion_count = len(summary.get("promotion_allowed_proof_ids", []))
    blocked_count = len(summary.get("blocked_proof_ids", []))
    if invalid_count:
        return 3
    if promotion_count < min_promotion_candidates:
        return 2
    if fail_on_blocked and blocked_count:
        return 4
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Evaluate local supervised proof summary artifacts without connecting to mixer hardware."
    )
    parser.add_argument(
        "--artifact-root",
        action="append",
        default=None,
        help="Directory or summary.json file to inspect. May be repeated.",
    )
    parser.add_argument(
        "--require-physical",
        action="store_true",
        help="Require physical pre-read, apply readback, and rollback readback sources for promotion.",
    )
    parser.add_argument(
        "--min-promotion-candidates",
        type=int,
        default=0,
        help="Return non-zero unless at least this many physical promotion candidates are present.",
    )
    parser.add_argument(
        "--fail-on-blocked",
        action="store_true",
        help="Return non-zero if any proof-like artifact is blocked.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Report JSON path. Defaults to artifacts/supervised_proof_artifact_report/<stamp>/report.json.",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Do not print the report JSON to stdout.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    roots = [Path(value) for value in (args.artifact_root or DEFAULT_ARTIFACT_ROOTS)]
    summary_paths = discover_summary_paths(roots)
    report = build_supervised_proof_artifact_report(summary_paths, require_physical=args.require_physical)
    report["proof_remediation_plan"] = build_supervised_proof_remediation_plan(report)
    report["promotion_evidence"] = build_supervised_promotion_evidence_snapshot(report)
    report["autonomy_evidence_overlay"] = build_autonomy_evidence_overlay(report)
    report["autonomy_mode_transition_gate"] = build_autonomy_mode_transition_gate(report)
    report["summary"]["autonomy_gap_rollup"] = build_autonomy_gap_rollup(report)
    report["next_wing_proof_plan"] = build_supervised_wing_proof_plan(report["promotion_evidence"])
    output_path = args.output or DEFAULT_OUTPUT_ROOT / f"{compact_stamp()}_supervised_proof_artifact_report" / "report.json"
    report["report_path"] = str(output_path)
    write_report(report, output_path)
    if not args.quiet:
        print(json.dumps(report, indent=2, ensure_ascii=False, sort_keys=True))
    return exit_code_for_report(
        report,
        min_promotion_candidates=max(0, args.min_promotion_candidates),
        fail_on_blocked=bool(args.fail_on_blocked),
    )


if __name__ == "__main__":
    raise SystemExit(main())
