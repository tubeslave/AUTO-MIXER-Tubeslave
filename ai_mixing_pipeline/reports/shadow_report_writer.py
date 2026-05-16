"""Report writers for shadow-mode replay safety comparisons."""

from __future__ import annotations

from datetime import datetime, timezone
from hashlib import sha256
from pathlib import Path
import json
from typing import Any


REPLAY_SHADOW_REPORT_SCHEMA_VERSION = "replay_shadow_report/v1"


def _stable_signature(payload: Any) -> str:
    text = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    return sha256(text.encode("utf-8")).hexdigest()


def _coerce_executor_dict(payload: Any) -> dict[str, Any]:
    if payload is None:
        return {"events": [], "selected_proposal_id": "", "initial_state": {}, "final_state": {}}
    if hasattr(payload, "to_dict"):
        return payload.to_dict()
    return dict(payload)


def _family_for_proposal_id(proposals: list[dict[str, Any]], proposal_id: str) -> str | None:
    for proposal in proposals:
        if proposal.get("proposal_id") == proposal_id:
            return str(proposal.get("family"))
    return None


def _first_by_rank(rows: list[dict[str, Any]]) -> dict[str, Any] | None:
    return min(rows, key=lambda row: (row.get("rank", 999999), row.get("proposal_id", ""))) if rows else None


def _annotate_confidence_trajectory(rows: list[dict[str, Any]]) -> dict[str, Any]:
    trajectory = [
        {
            "rank": row["rank"],
            "proposal_id": row["proposal_id"],
            "source_proposal_id": row["source_proposal_id"],
            "family": row["family"],
            "confidence": row["confidence"],
            "selected": row["selected"],
        }
        for row in sorted(rows, key=lambda row: (row["rank"], row["proposal_id"]))
    ]
    payload = {
        "schema_version": REPLAY_SHADOW_REPORT_SCHEMA_VERSION,
        "trajectory": trajectory,
        "trajectory_signature": _stable_signature(trajectory),
    }
    return payload


def _build_rollback_payload(manifest: dict[str, Any], executor: dict[str, Any]) -> dict[str, Any]:
    proposals = {proposal["proposal_id"]: proposal for proposal in manifest["graph_checkpoint"].get("proposals", [])}
    initial_state = dict(executor.get("initial_state") or {})
    events = list(executor.get("events") or [])
    possible_rollbacks = [
        {
            "step_index": 0,
            "proposal_id": None,
            "source_proposal_id": None,
            "family": None,
            "rollback_target": initial_state,
            "restored_state_signature": _stable_signature(initial_state),
            "blocked_reason": None,
        }
    ]
    proposed_action_timeline: list[dict[str, Any]] = []
    blocked_reasons: list[str] = []
    for index, event in enumerate(events, start=1):
        proposal = proposals.get(event.get("proposal_id"), {})
        rollback_target = dict(event.get("rollback_target") or event.get("previous_state") or initial_state)
        correlation_id = proposal.get("replay_correlation_id") or event.get("replay_correlation_id") or event.get("proposal_id")
        possible_rollbacks.append(
            {
                "step_index": index,
                "proposal_id": correlation_id,
                "source_proposal_id": event.get("proposal_id"),
                "family": event.get("family"),
                "rollback_target": rollback_target,
                "restored_state_signature": _stable_signature(rollback_target),
                "blocked_reason": event.get("blocked_reason"),
            }
        )
        proposed_action_timeline.append(
            {
                "step_index": index,
                "proposal_id": correlation_id,
                "source_proposal_id": event.get("proposal_id"),
                "family": event.get("family"),
                "status": event.get("status"),
                "selected": event.get("proposal_id") == executor.get("selected_proposal_id"),
                "blocked_reason": event.get("blocked_reason"),
                "requested_state": proposal.get("requested_state") or event.get("payload", {}).get("requested_state"),
                "target": proposal.get("target") or event.get("payload", {}).get("target"),
            }
        )
        if event.get("blocked_reason"):
            blocked_reasons.append(str(event["blocked_reason"]))

    return {
        "schema_version": REPLAY_SHADOW_REPORT_SCHEMA_VERSION,
        "manifest_id": manifest["manifest_id"],
        "selected_proposal_id": executor.get("selected_proposal_id"),
        "proposed_action_timeline": proposed_action_timeline,
        "simulated_rollback_target": initial_state,
        "restored_state_signature": _stable_signature(initial_state),
        "possible_rollbacks": possible_rollbacks,
        "blocked_reasons": sorted(set(blocked_reasons)),
        "replay_correlation_ids": manifest.get("replay_correlation_ids", []),
    }


def _build_promotion_readiness(
    manifest: dict[str, Any],
    comparison: dict[str, Any],
    validation_report: dict[str, Any],
    rollback: dict[str, Any],
) -> dict[str, Any]:
    findings = validation_report.get("findings", [])
    error_count = sum(1 for item in findings if item.get("severity") == "error")
    warning_count = sum(1 for item in findings if item.get("severity") == "warning")
    disagreement_penalty = 15 if comparison.get("disagreements") else 0
    blocked_penalty = 10 if rollback.get("blocked_reasons") else 0
    score = max(0, 100 - error_count * 25 - warning_count * 10 - disagreement_penalty - blocked_penalty)
    ready = (
        error_count == 0
        and bool(manifest.get("dry_run_only", True))
        and not bool(manifest.get("live_mixer_writes", False))
        and score >= 75
    )
    return {
        "schema_version": REPLAY_SHADOW_REPORT_SCHEMA_VERSION,
        "ready": ready,
        "promotion_readiness_score": score,
        "dry_run_only": bool(manifest.get("dry_run_only", True)),
        "live_mixer_writes": bool(manifest.get("live_mixer_writes", False)),
        "validation_highest_severity": validation_report.get("highest_severity"),
        "disagreement_count": len(comparison.get("disagreements", [])),
        "blocked_reason_count": len(rollback.get("blocked_reasons", [])),
    }


def _write_json(path: Path, payload: dict[str, Any]) -> str:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    return str(path)


def _write_markdown(path: Path, payload: dict[str, Any]) -> str:
    lines = [
        "# Shadow Mode Summary",
        "",
        f"- generated_at: {payload['generated_at']}",
        f"- manifest_id: {payload['manifest_id']}",
        f"- selected_proposal_id: {payload['selected_proposal_id']}",
        f"- disagreement_count: {payload['disagreement_count']}",
        f"- readiness_score: {payload['promotion_readiness_score']}",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return str(path)


def write_shadow_mode_reports(
    output_dir: str | Path,
    manifest: dict[str, Any],
    *,
    executor: Any = None,
    validation_report: dict[str, Any] | None = None,
    operator_family: str = "operator_assistance",
) -> dict[str, str]:
    root = Path(output_dir)
    root.mkdir(parents=True, exist_ok=True)
    executor_dict = _coerce_executor_dict(executor)
    validation = dict(validation_report or {})
    graph = dict(manifest.get("graph_checkpoint") or {})
    proposals = list(graph.get("proposals") or [])
    ranking_trace = list((graph.get("ranking") or {}).get("trace") or [])
    rank_map = {item["proposal_id"]: int(item["rank"]) for item in ranking_trace}
    selected_id = str(executor_dict.get("selected_proposal_id") or graph.get("selected_proposal_id") or "")

    rows: list[dict[str, Any]] = []
    for proposal in proposals:
        correlation_id = proposal.get("replay_correlation_id") or proposal["proposal_id"]
        rows.append(
            {
                "proposal_id": correlation_id,
                "source_proposal_id": proposal["proposal_id"],
                "family": proposal["family"],
                "confidence": float(proposal.get("confidence", 0.0)),
                "rank": rank_map.get(proposal["proposal_id"], 999999),
                "selected": proposal["proposal_id"] == selected_id,
                "replay_signature": proposal.get("replay_signature"),
            }
        )

    operator_rows = sorted(
        [row for row in rows if _family_for_proposal_id(proposals, row["source_proposal_id"]) == operator_family],
        key=lambda row: (row["rank"], row["proposal_id"]),
    )
    ai_rows = sorted(
        [row for row in rows if _family_for_proposal_id(proposals, row["source_proposal_id"]) != operator_family],
        key=lambda row: (row["rank"], row["proposal_id"]),
    )
    selected_family = _family_for_proposal_id(proposals, selected_id)
    disagreements: list[dict[str, Any]] = []
    if selected_family == operator_family:
        top_ai = _first_by_rank(ai_rows)
        selected_operator = next((row for row in operator_rows if row["selected"]), _first_by_rank(operator_rows))
        disagreements.append(
            {
                "code": "operator_selected",
                "proposal_id": (selected_operator or {}).get("proposal_id"),
                "selected_family": operator_family,
                "top_ai_proposal_id": (top_ai or {}).get("proposal_id"),
            }
        )

    comparison = {
        "schema_version": REPLAY_SHADOW_REPORT_SCHEMA_VERSION,
        "operator_count": len(operator_rows),
        "ai_count": len(ai_rows),
        "operator_rows": operator_rows,
        "ai_rows": ai_rows,
        "selected_proposal_id": next(
            (row["proposal_id"] for row in rows if row["selected"]),
            selected_id,
        ),
        "disagreements": disagreements,
    }
    trajectory = _annotate_confidence_trajectory(rows)
    rollback = _build_rollback_payload(manifest, executor_dict)
    readiness = _build_promotion_readiness(manifest, comparison, validation, rollback)
    summary = {
        "schema_version": REPLAY_SHADOW_REPORT_SCHEMA_VERSION,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "manifest_id": manifest["manifest_id"],
        "selected_proposal_id": comparison["selected_proposal_id"],
        "disagreement_count": len(disagreements),
        "promotion_readiness_score": readiness["promotion_readiness_score"],
        "files": {
            "comparison": "shadow_mode_comparison.json",
            "rollback": "rollback_simulation_report.json",
            "readiness": "promotion_readiness_report.json",
            "confidence_trajectory": "confidence_trajectory.json",
        },
        "summary_signature": _stable_signature(
            {
                "comparison": comparison,
                "rollback": rollback,
                "readiness": readiness,
                "trajectory": trajectory,
            }
        ),
    }

    result = {
        "shadow_mode_comparison": _write_json(root / "shadow_mode_comparison.json", comparison),
        "rollback_simulation": _write_json(root / "rollback_simulation_report.json", rollback),
        "promotion_readiness": _write_json(root / "promotion_readiness_report.json", readiness),
        "confidence_trajectory": _write_json(root / "confidence_trajectory.json", trajectory),
        "shadow_mode_summary": _write_json(root / "shadow_mode_summary.json", summary),
    }
    _write_markdown(root / "shadow_mode_summary.md", summary)
    return result
