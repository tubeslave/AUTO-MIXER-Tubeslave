"""Build a combined WING replay and telemetry readiness bundle."""

from __future__ import annotations

import argparse
import json
from collections.abc import Mapping
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from wing_telemetry_compare import compare_sessions, summarize_session


READINESS_BUNDLE_SCHEMA_VERSION = "wing_replay_readiness_bundle/v1"


def _load_json(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError(f"Expected JSON object at {path}")
    return payload


def _load_optional_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return _load_json(path)


def _normalized_timestamp(value: str | None = None) -> str:
    if value:
        return str(value)
    return datetime.now(timezone.utc).isoformat()


def _pick_truthy(*values: Any) -> Any:
    for value in values:
        if value not in (None, "", [], {}):
            return value
    return None


def _build_shadow_status(shadow_report_dir: str | Path) -> dict[str, Any]:
    root = Path(shadow_report_dir)
    summary = _load_json(root / "shadow_mode_summary.json")
    comparison = _load_json(root / "shadow_mode_comparison.json")
    readiness = _load_json(root / "promotion_readiness_report.json")
    rollback = _load_optional_json(root / "rollback_simulation_report.json")
    trajectory = _load_optional_json(root / "confidence_trajectory.json")
    aggregation = dict(readiness.get("aggregation") or {})
    blocked_gates = [
        str(gate.get("key"))
        for gate in readiness.get("promotion_readiness_gates", [])
        if not gate.get("ok", False)
    ]
    return {
        "report_dir": str(root),
        "generated_at": summary.get("generated_at"),
        "manifest_id": summary.get("manifest_id"),
        "selected_proposal_id": summary.get("selected_proposal_id"),
        "ready": bool(readiness.get("ready", False)),
        "promotion_readiness_score": readiness.get("promotion_readiness_score"),
        "validation_highest_severity": readiness.get("validation_highest_severity"),
        "dry_run_only": bool(readiness.get("dry_run_only", comparison.get("evidence", {}).get("dry_run_only", True))),
        "live_mixer_writes": bool(
            readiness.get("live_mixer_writes", comparison.get("evidence", {}).get("live_mixer_writes", False))
        ),
        "disagreement_count": len(comparison.get("disagreements") or []),
        "disagreement_rate": aggregation.get(
            "disagreement_rate",
            (comparison.get("disagreement_summary") or {}).get("disagreement_rate", 0.0),
        ),
        "audit_coverage": aggregation.get("audit_coverage", {}),
        "rollback_frequency": aggregation.get("rollback_frequency", {}),
        "confidence_trend": aggregation.get("confidence_trend", trajectory.get("trend_summary", {})),
        "blocked_reasons": list(rollback.get("blocked_reasons", [])),
        "blocking_gates": blocked_gates,
    }


def _build_supervised_status(supervised_summary_path: str | Path | None) -> dict[str, Any]:
    if supervised_summary_path is None:
        return {"available": False}
    summary = _load_json(supervised_summary_path)
    telemetry = dict(summary.get("telemetry_capture_stop") or {})
    results = dict(summary.get("results") or {})
    blocked_reasons = sorted(
        {
            str(result.get("reason"))
            for result in results.values()
            if isinstance(result, Mapping) and result.get("reason")
        }
    )
    return {
        "available": True,
        "summary_path": str(supervised_summary_path),
        "issue": summary.get("issue"),
        "runner": summary.get("runner"),
        "session_label": summary.get("session_label"),
        "safe_channel": summary.get("safe_channel"),
        "connection_status": dict(summary.get("connection_status") or {}),
        "telemetry_capture_stop": telemetry,
        "telemetry_artifacts_present": bool(telemetry.get("events_path") and telemetry.get("metadata_path")),
        "manual_write_session": dict(summary.get("manual_write_session") or {}),
        "result_reasons": blocked_reasons,
        "emergency_stop_seen": bool((results.get("emergency_stop") or {}).get("type") == "emergency_stop_applied"),
    }


def _build_telemetry_status(
    *,
    live_session_path: str | Path,
    replay_session_path: str | Path | None = None,
) -> dict[str, Any]:
    live_summary = summarize_session(live_session_path)
    replay_summary = summarize_session(replay_session_path) if replay_session_path is not None else None
    comparison = compare_sessions(live_session_path, replay_session_path) if replay_session_path is not None else None

    observation_paths = [live_summary]
    if replay_summary is not None:
        observation_paths.append(replay_summary)

    dangerous_write_sessions = [
        {
            "label": summary.get("session_label") or summary.get("session_id") or "unknown",
            "session_dir": summary.get("session_dir"),
            "write_sent_count": summary.get("write_sent_count", 0),
        }
        for summary in observation_paths
        if summary.get("metadata", {}).get("observation_only") is True and summary.get("write_sent_count", 0) > 0
    ]

    blocked_write_total = sum(int(summary.get("blocked_write_count", 0)) for summary in observation_paths)
    return {
        "live": live_summary,
        "replay": replay_summary,
        "comparison": comparison,
        "dangerous_write_sessions": dangerous_write_sessions,
        "dangerous_write_detected": bool(dangerous_write_sessions),
        "blocked_write_total": blocked_write_total,
        "fallback_only_detected": any(summary.get("fallback_only_observation", False) for summary in observation_paths),
    }


def build_wing_replay_readiness_bundle(
    *,
    output_dir: str | Path,
    shadow_report_dir: str | Path,
    live_telemetry_session: str | Path,
    replay_telemetry_session: str | Path | None = None,
    supervised_summary_path: str | Path | None = None,
    generated_at: str | None = None,
) -> dict[str, Path]:
    root = Path(output_dir)
    root.mkdir(parents=True, exist_ok=True)

    shadow = _build_shadow_status(shadow_report_dir)
    telemetry = _build_telemetry_status(
        live_session_path=live_telemetry_session,
        replay_session_path=replay_telemetry_session,
    )
    supervised = _build_supervised_status(supervised_summary_path)

    blockers: list[str] = []
    if shadow.get("live_mixer_writes"):
        blockers.append("shadow_report_live_mixer_writes")
    if not shadow.get("dry_run_only", True):
        blockers.append("shadow_report_not_dry_run_only")
    if shadow.get("validation_highest_severity") == "error":
        blockers.append("shadow_report_validation_error")
    if telemetry.get("dangerous_write_detected"):
        blockers.append("observation_only_telemetry_contains_write_sent")
    if not supervised.get("available"):
        blockers.append("missing_supervised_summary")
    elif not supervised.get("telemetry_artifacts_present"):
        blockers.append("supervised_summary_missing_telemetry_artifacts")
    if shadow.get("blocking_gates"):
        blockers.extend(f"promotion_gate:{gate}" for gate in shadow["blocking_gates"])

    ready = (
        bool(shadow.get("ready"))
        and not blockers
    )

    summary = {
        "schema_version": READINESS_BUNDLE_SCHEMA_VERSION,
        "generated_at": _normalized_timestamp(generated_at),
        "ready": ready,
        "blockers": blockers,
        "sources": {
            "shadow_report_dir": str(shadow_report_dir),
            "live_telemetry_session": str(live_telemetry_session),
            "replay_telemetry_session": str(replay_telemetry_session) if replay_telemetry_session is not None else None,
            "supervised_summary_path": str(supervised_summary_path) if supervised_summary_path is not None else None,
        },
        "shadow": shadow,
        "telemetry": telemetry,
        "supervised": supervised,
        "evidence": {
            "selected_proposal_id": shadow.get("selected_proposal_id"),
            "promotion_readiness_score": shadow.get("promotion_readiness_score"),
            "dangerous_write_detected": telemetry.get("dangerous_write_detected"),
            "dangerous_write_sessions": telemetry.get("dangerous_write_sessions"),
            "telemetry_blocked_write_total": telemetry.get("blocked_write_total"),
            "supervised_session_label": supervised.get("session_label"),
            "supervised_result_reasons": supervised.get("result_reasons"),
            "telemetry_events_path": _pick_truthy(
                (supervised.get("telemetry_capture_stop") or {}).get("events_path"),
                (supervised.get("connection_status") or {}).get("telemetry_capture", {}).get("events_path"),
            ),
        },
    }

    summary_path = root / "wing_replay_readiness_bundle.json"
    markdown_path = root / "wing_replay_readiness_bundle.md"
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    markdown_path.write_text(_build_markdown(summary), encoding="utf-8")
    return {
        "json": summary_path,
        "markdown": markdown_path,
    }


def _build_markdown(summary: dict[str, Any]) -> str:
    shadow = dict(summary.get("shadow") or {})
    telemetry = dict(summary.get("telemetry") or {})
    supervised = dict(summary.get("supervised") or {})
    lines = [
        "# WING Replay Readiness Bundle",
        "",
        f"- generated_at: {summary.get('generated_at')}",
        f"- ready: {summary.get('ready')}",
        f"- selected_proposal_id: {shadow.get('selected_proposal_id')}",
        f"- promotion_readiness_score: {shadow.get('promotion_readiness_score')}",
        f"- dangerous_write_detected: {telemetry.get('dangerous_write_detected')}",
        f"- telemetry_blocked_write_total: {telemetry.get('blocked_write_total')}",
        f"- supervised_session_label: {supervised.get('session_label')}",
        "",
        "## Blockers",
    ]
    blockers = list(summary.get("blockers") or [])
    if blockers:
        lines.extend(f"- {item}" for item in blockers)
    else:
        lines.append("- none")
    lines.extend(
        [
            "",
            "## Sources",
            f"- shadow_report_dir: {summary.get('sources', {}).get('shadow_report_dir')}",
            f"- live_telemetry_session: {summary.get('sources', {}).get('live_telemetry_session')}",
            f"- replay_telemetry_session: {summary.get('sources', {}).get('replay_telemetry_session')}",
            f"- supervised_summary_path: {summary.get('sources', {}).get('supervised_summary_path')}",
        ]
    )
    return "\n".join(lines) + "\n"


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build a combined WING replay readiness bundle.")
    parser.add_argument("--output-dir", required=True, help="Destination directory for bundle outputs")
    parser.add_argument("--shadow-report-dir", required=True, help="Directory containing shadow_mode_*.json reports")
    parser.add_argument("--live-telemetry-session", required=True, help="Live telemetry session dir or events.jsonl")
    parser.add_argument("--replay-telemetry-session", help="Replay telemetry session dir or events.jsonl")
    parser.add_argument("--supervised-summary-path", help="Optional supervised live summary JSON")
    parser.add_argument("--generated-at", help="Override generated_at timestamp")
    return parser


def _main(argv: list[str] | None = None) -> int:
    args = _build_arg_parser().parse_args(argv)
    result = build_wing_replay_readiness_bundle(
        output_dir=args.output_dir,
        shadow_report_dir=args.shadow_report_dir,
        live_telemetry_session=args.live_telemetry_session,
        replay_telemetry_session=args.replay_telemetry_session,
        supervised_summary_path=args.supervised_summary_path,
        generated_at=args.generated_at,
    )
    print(
        json.dumps(
            {key: str(value) for key, value in result.items()},
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(_main())
