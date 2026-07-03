"""Report writers for shadow-mode replay safety comparisons."""

from __future__ import annotations

from datetime import datetime, timezone
from hashlib import sha256
from pathlib import Path
import json
from typing import Any


REPLAY_SHADOW_REPORT_SCHEMA_VERSION = "replay_shadow_report/v1"


def write_shadow_mode_reports(
    reports_dir: str | Path,
    *,
    manifest: dict[str, Any],
    executor: Any | None = None,
    validation_report: dict[str, Any] | None = None,
    operator_family: str = "operator_assistance",
) -> dict[str, str]:
    """Write operator-vs-AI comparison and rollback simulation reports."""

    reports_root = Path(reports_dir).expanduser()
    reports_root.mkdir(parents=True, exist_ok=True)

    graph_checkpoint = dict((manifest.get("graph_checkpoint") or {}).copy())
    ranking = dict((graph_checkpoint.get("ranking") or {}).copy())
    ranked_proposals = list(ranking.get("ranked_proposals") or ())
    ranked_by_id = {str(item.get("proposal_id", "")): dict(item) for item in ranked_proposals}
    selected_proposal_id = str(ranking.get("selected_proposal_id") or "")

    proposals = [dict(item) for item in (graph_checkpoint.get("proposals") or ())]

    operator_rows: list[dict[str, Any]] = []
    ai_rows: list[dict[str, Any]] = []
    all_rows: list[dict[str, Any]] = []
    for proposal in proposals:
        row = {
            "proposal_id": str(proposal.get("proposal_id", "")),
            "family": str(proposal.get("family", "")),
            "source_system": str(proposal.get("source_system", "")),
            "action_type": str(proposal.get("action_type", "")),
            "replay_correlation_id": str(proposal.get("replay_correlation_id", "")),
            "confidence": float(proposal.get("confidence", 0.0)),
            "dry_run_only": bool(proposal.get("dry_run_only", True)),
            "auto_apply_blocked": bool(proposal.get("auto_apply_blocked", True)),
            "target": dict(proposal.get("target") or {}),
            "requested_state": dict(proposal.get("requested_state") or {}),
            "metadata": dict(proposal.get("metadata") or {}),
        }
        ranked = ranked_by_id.get(row["proposal_id"], {})
        row["rank"] = int(ranked.get("rank", 0))
        row["selected"] = row["proposal_id"] == selected_proposal_id
        row["confidence_weighted_score"] = float(ranked.get("confidence_weighted_score", 0.0))
        all_rows.append(row)
        if row["family"] == operator_family:
            operator_rows.append(row)
        else:
            ai_rows.append(row)

    all_rows.sort(key=lambda item: (item["rank"] or 10_000, item["proposal_id"]))

    disagreements: list[dict[str, Any]] = []
    top_operator = _first_by_rank(operator_rows)
    top_ai = _first_by_rank(ai_rows)
    if top_operator and top_ai:
        if top_operator["selected"]:
            disagreements.append(
                {
                    "code": "operator_selected",
                    "severity": "warning",
                    "message": "Operator recommendation selected over AI candidates.",
                    "operator_proposal_id": top_operator["proposal_id"],
                    "ai_proposal_id": top_ai["proposal_id"],
                    "rank_gap": int(top_operator["rank"] - top_ai["rank"]) if int(top_operator["rank"]) and int(top_ai["rank"]) else None,
                    "confidence_gap": round(float(top_operator["confidence"]) - float(top_ai["confidence"]), 6),
                }
            )
        if int(top_operator["rank"]) <= 3 and int(top_ai["rank"]) > 3:
            disagreements.append(
                {
                    "code": "operator_top3_only",
                    "severity": "info",
                    "message": "Top operator suggestion is in first 3 while top AI is outside first 3.",
                    "operator_proposal_id": top_operator["proposal_id"],
                    "ai_proposal_id": top_ai["proposal_id"],
                    "rank_gap": int(top_operator["rank"] - top_ai["rank"]) if int(top_operator["rank"]) and int(top_ai["rank"]) else None,
                }
            )

    comparison_payload = {
        "schema_version": REPLAY_SHADOW_REPORT_SCHEMA_VERSION,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "manifest_id": str(manifest.get("manifest_id", "")),
        "selected_proposal_id": selected_proposal_id,
        "selected_family": _family_for_proposal_id(all_rows, selected_proposal_id),
        "operator_count": len(operator_rows),
        "ai_count": len(ai_rows),
        "proposals": all_rows,
        "operator_rows": operator_rows,
        "ai_rows": ai_rows,
        "disagreements": disagreements,
    }
    comparison_path = _write_json(reports_root / "shadow_mode_comparison.json", comparison_payload)

    confidence_trajectory_payload = {
        "schema_version": REPLAY_SHADOW_REPORT_SCHEMA_VERSION,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "manifest_id": str(manifest.get("manifest_id", "")),
        "selected_proposal_id": selected_proposal_id,
        "trajectory": [
            {
                "rank": int(item.get("rank", 0)),
                "proposal_id": str(item.get("proposal_id", "")),
                "family": str(item.get("family", "")),
                "confidence": float(item.get("confidence", 0.0)),
                "confidence_weighted_score": float(item.get("confidence_weighted_score", 0.0)),
            }
            for item in all_rows
            if int(item.get("rank", 0)) > 0
        ],
    }
    confidence_payload = _annotate_confidence_trajectory(confidence_trajectory_payload)
    confidence_path = _write_json(
        reports_root / "confidence_trajectory.json",
        confidence_payload,
    )

    rollback_payload = _build_rollback_payload(executor, manifest)
    rollback_path = _write_json(
        reports_root / "rollback_simulation_report.json",
        rollback_payload,
    )

    readiness_payload = _build_promotion_readiness(manifest, executor, validation_report, disagreements)
    readiness_path = _write_json(
        reports_root / "promotion_readiness_report.json",
        readiness_payload,
    )

    summary_payload = {
        "schema_version": REPLAY_SHADOW_REPORT_SCHEMA_VERSION,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "report_files": {
            "comparison": str(comparison_path),
            "confidence_trajectory": str(confidence_path),
            "rollback_simulation": str(rollback_path),
            "promotion_readiness": str(readiness_path),
        },
        "validation_summary": dict((validation_report or {}).get("summary") or {}),
        "readiness": readiness_payload.get("ready", False),
        "disagreement_count": len(disagreements),
    }
    _write_markdown(reports_root / "shadow_mode_summary.md", summary_payload)

    return {
        "shadow_mode_comparison": str(comparison_path),
        "confidence_trajectory": str(confidence_path),
        "rollback_simulation": str(rollback_path),
        "promotion_readiness": str(readiness_path),
        "shadow_mode_summary": str(reports_root / "shadow_mode_summary.md"),
    }


def _annotate_confidence_trajectory(payload: dict[str, Any]) -> dict[str, Any]:
    trajectory = list(payload.get("trajectory") or [])
    annotated: list[dict[str, Any]] = []
    previous: float | None = None
    for item in trajectory:
        current = float(item.get("confidence", 0.0))
        annotation = {
            **item,
            "delta_from_previous": round(current - previous, 6) if previous is not None else 0.0,
            "cumulative_delta": 0.0,
        }
        if previous is None:
            annotation["cumulative_delta"] = 0.0
        else:
            annotation["cumulative_delta"] = round(current - trajectory[0].get("confidence", 0.0), 6)
        annotated.append(annotation)
        previous = current
    payload["trajectory"] = annotated
    return payload


def _build_rollback_payload(executor: Any | None, manifest: dict[str, Any]) -> dict[str, Any]:
    if executor is None:
        return {
            "schema_version": REPLAY_SHADOW_REPORT_SCHEMA_VERSION,
            "manifest_id": str(manifest.get("manifest_id", "")),
            "error": "executor_missing",
            "possible_rollbacks": [],
        }
    executor_payload = _coerce_executor_dict(executor)
    dry_run_only = bool(executor_payload.get("dry_run_only", True))
    snapshots = list(executor_payload.get("state_snapshots") or ())
    possible_rollbacks: list[dict[str, Any]] = []
    for index in range(0, len(snapshots) + 1):
        state = snapshots[index - 1] if index > 0 else executor_payload.get("restored_state") or {}
        possible_rollbacks.append(
            {
                "rollback_to": int(index),
                "applied_proposals": int(index),
                "state_signature": _stable_signature(state),
                "state": state,
            }
        )
    return {
        "schema_version": REPLAY_SHADOW_REPORT_SCHEMA_VERSION,
        "manifest_id": str(manifest.get("manifest_id", "")),
        "trace_signature": str(executor_payload.get("trace_signature", "")),
        "dry_run_only": dry_run_only,
        "possible_rollbacks": possible_rollbacks,
    }


def _build_promotion_readiness(
    manifest: dict[str, Any],
    executor: Any | None,
    validation_report: dict[str, Any] | None,
    disagreements: list[dict[str, Any]],
) -> dict[str, Any]:
    summary = dict((validation_report or {}).get("summary") or {})
    valid = bool((validation_report or {}).get("valid", True))
    errors = int((validation_report or {}).get("summary", {}).get("error_count", 0) if validation_report else 0)
    dry_run_only = bool(manifest.get("dry_run_only", True))
    live_writes = bool(manifest.get("live_mixer_writes", False))
    executor_payload = _coerce_executor_dict(executor)
    executor_dry_run_only = bool(executor_payload.get("dry_run_only", True))
    blocking = []
    score = 0
    if dry_run_only and executor_dry_run_only:
        score += 40
    else:
        blocking.append("replay_must_stay_dry_run")
    if not live_writes:
        score += 25
    else:
        blocking.append("live_writes_disabled_required")
    if valid and errors == 0:
        score += 20
    else:
        blocking.append("validation_errors_or_invalid")
    if not disagreements:
        score += 15
    else:
        blocking.append("operator_ai_disagreement_detected")
    return {
        "schema_version": REPLAY_SHADOW_REPORT_SCHEMA_VERSION,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "manifest_id": str(manifest.get("manifest_id", "")),
        "ready": score >= 80,
        "promotion_readiness_score": min(100, score),
        "blocking_reasons": blocking,
        "policy_checks": {
            "manifest_dry_run_only": dry_run_only,
            "executor_dry_run_only": executor_dry_run_only,
            "manifest_live_writes": bool(manifest.get("live_mixer_writes", False)),
            "validation_valid": valid,
            "error_count": errors,
            "disagreement_count": len(disagreements),
            "validation_summary": summary,
        },
    }


def _coerce_executor_dict(executor: Any) -> dict[str, Any]:
    if isinstance(executor, dict):
        return {
            "dry_run_only": bool(executor.get("dry_run_only", True)),
            "trace_signature": str(executor.get("trace_signature", "")),
            "state_snapshots": list(executor.get("state_snapshots") or ()),
            "restored_state": dict(executor.get("restored_state") or {}),
        }
    return {
        "dry_run_only": bool(getattr(executor, "dry_run_only", True)),
        "trace_signature": str(getattr(executor, "trace_signature", "")),
        "state_snapshots": [dict(item) for item in getattr(executor, "state_snapshots", []) or ()],
        "restored_state": dict(getattr(executor, "restored_state") or {}),
    }


def _family_for_proposal_id(rows: list[dict[str, Any]], proposal_id: str) -> str:
    for row in rows:
        if str(row.get("proposal_id")) == str(proposal_id):
            return str(row.get("family", ""))
    return ""


def _first_by_rank(rows: list[dict[str, Any]]) -> dict[str, Any] | None:
    ranked = [row for row in rows if int(row.get("rank", 0)) > 0]
    if not ranked:
        return None
    ranked.sort(key=lambda item: int(item.get("rank", 0)))
    return ranked[0]


def _stable_signature(payload: Any) -> str:
    return sha256(
        json.dumps(payload, sort_keys=True, ensure_ascii=True).encode("utf-8")
    ).hexdigest()


def _write_json(path: Path, payload: Any) -> Path:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=True), encoding="utf-8")
    return path


def _write_markdown(path: Path, payload: dict[str, Any]) -> None:
    lines = [
        "# Shadow-Mode Simulation Summary",
        "",
        f"- Schema: `{payload.get('schema_version')}`",
        f"- Created: `{payload.get('created_at')}`",
        f"- Validation summary available: `{bool(payload.get('validation_summary'))}`",
        f"- Readiness: `{payload.get('readiness', False)}`",
        f"- Disagreements: `{payload.get('disagreement_count', 0)}`",
        "",
        "## Report Files",
        f"- Comparison: `{payload.get('report_files', {}).get('comparison', '')}`",
        f"- Confidence trajectory: `{payload.get('report_files', {}).get('confidence_trajectory', '')}`",
        f"- Rollback simulation: `{payload.get('report_files', {}).get('rollback_simulation', '')}`",
        f"- Promotion readiness: `{payload.get('report_files', {}).get('promotion_readiness', '')}`",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
