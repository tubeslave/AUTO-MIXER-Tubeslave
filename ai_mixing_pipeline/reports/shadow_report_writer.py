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


def _selected_row(rows: list[dict[str, Any]]) -> dict[str, Any] | None:
    return next((row for row in rows if row.get("selected")), None)


def _rounded_metric(value: float | None) -> float | None:
    if value is None:
        return None
    return round(float(value), 4)


def _normalized_timestamp(generated_at: str | datetime | None) -> str:
    if generated_at is None:
        return datetime.now(timezone.utc).isoformat()
    if isinstance(generated_at, datetime):
        moment = generated_at if generated_at.tzinfo else generated_at.replace(tzinfo=timezone.utc)
        return moment.astimezone(timezone.utc).isoformat()
    return str(generated_at)


def _build_comparison_row(
    proposal: dict[str, Any],
    *,
    correlation_id: str,
    rank: int,
    selected: bool,
) -> dict[str, Any]:
    return {
        "proposal_id": correlation_id,
        "source_proposal_id": proposal["proposal_id"],
        "family": proposal["family"],
        "source_system": proposal.get("source_system"),
        "action_type": proposal.get("action_type"),
        "target": dict(proposal.get("target") or {}),
        "current_state": dict(proposal.get("current_state") or {}),
        "requested_state": dict(proposal.get("requested_state") or {}),
        "source_payload": dict(proposal.get("source_payload") or {}),
        "confidence": float(proposal.get("confidence", 0.0)),
        "rank": rank,
        "selected": selected,
        "replay_signature": proposal.get("replay_signature"),
        "replay_correlation_id": proposal.get("replay_correlation_id") or correlation_id,
        "auto_apply_blocked": bool(proposal.get("auto_apply_blocked", False)),
        "transport_policy": str(proposal.get("transport_policy") or "dry_run_only"),
        "live_mixer_writes": bool(proposal.get("live_mixer_writes", False)),
        "safety": dict(proposal.get("safety") or {}),
    }


def _build_disagreements(
    *,
    selected_family: str | None,
    selected_row: dict[str, Any] | None,
    top_operator: dict[str, Any] | None,
    top_ai: dict[str, Any] | None,
    operator_family: str,
) -> list[dict[str, Any]]:
    disagreements: list[dict[str, Any]] = []
    if top_operator and top_ai:
        if selected_family == operator_family:
            disagreements.append(
                {
                    "code": "operator_selected",
                    "taxonomy": "operator_override",
                    "selected_proposal_id": (selected_row or {}).get("proposal_id"),
                    "selected_family": selected_family,
                    "operator_candidate_proposal_id": top_operator["proposal_id"],
                    "ai_candidate_proposal_id": top_ai["proposal_id"],
                    "reason": "operator recommendation outranked the top non-operator proposal in dry-run replay.",
                }
            )
    elif top_operator and selected_family == operator_family:
        disagreements.append(
            {
                "code": "operator_only",
                "taxonomy": "operator_only",
                "selected_proposal_id": (selected_row or top_operator).get("proposal_id"),
                "selected_family": selected_family,
                "operator_candidate_proposal_id": top_operator["proposal_id"],
                "ai_candidate_proposal_id": None,
                "reason": "Only operator-side recommendations were captured for this dry-run replay.",
            }
        )
    elif top_ai:
        disagreements.append(
            {
                "code": "ai_only",
                "taxonomy": "ai_only",
                "selected_proposal_id": (selected_row or top_ai).get("proposal_id"),
                "selected_family": selected_family,
                "operator_candidate_proposal_id": None,
                "ai_candidate_proposal_id": top_ai["proposal_id"],
                "reason": "Only AI-side recommendations were captured for this dry-run replay.",
            }
        )
    return disagreements


def _build_disagreement_summary(
    *,
    rows: list[dict[str, Any]],
    disagreements: list[dict[str, Any]],
    top_operator: dict[str, Any] | None,
    top_ai: dict[str, Any] | None,
) -> dict[str, Any]:
    by_code: dict[str, int] = {}
    by_taxonomy: dict[str, int] = {}
    for item in disagreements:
        code = str(item.get("code") or "unknown")
        taxonomy = str(item.get("taxonomy") or "unknown")
        by_code[code] = by_code.get(code, 0) + 1
        by_taxonomy[taxonomy] = by_taxonomy.get(taxonomy, 0) + 1
    comparison_count = 1 if rows else 0
    disagreement_count = len(disagreements)
    return {
        "comparison_count": comparison_count,
        "disagreement_count": disagreement_count,
        "disagreement_rate": _rounded_metric(
            disagreement_count / comparison_count if comparison_count else 0.0
        ),
        "classes": {
            "by_code": dict(sorted(by_code.items())),
            "by_taxonomy": dict(sorted(by_taxonomy.items())),
        },
        "operator_candidate_present": bool(top_operator),
        "ai_candidate_present": bool(top_ai),
    }


def _build_confidence_trend(
    *,
    rows: list[dict[str, Any]],
    selected_row: dict[str, Any] | None,
    top_operator: dict[str, Any] | None,
    top_ai: dict[str, Any] | None,
) -> dict[str, Any]:
    ordered = sorted(rows, key=lambda row: (row["rank"], row["proposal_id"]))
    top_row = ordered[0] if ordered else None
    bottom_row = ordered[-1] if ordered else None
    top_operator_confidence = (
        float(top_operator["confidence"]) if top_operator and top_operator.get("confidence") is not None else None
    )
    top_ai_confidence = float(top_ai["confidence"]) if top_ai and top_ai.get("confidence") is not None else None
    selected_confidence = (
        float(selected_row["confidence"]) if selected_row and selected_row.get("confidence") is not None else None
    )
    if top_operator_confidence is None or top_ai_confidence is None:
        direction = "insufficient_data"
    elif abs(top_operator_confidence - top_ai_confidence) < 1e-9:
        direction = "flat"
    elif top_operator_confidence > top_ai_confidence:
        direction = "operator_above_ai"
    else:
        direction = "ai_above_operator"
    return {
        "direction": direction,
        "selected_confidence": _rounded_metric(selected_confidence),
        "top_confidence": _rounded_metric(float(top_row["confidence"])) if top_row else None,
        "bottom_confidence": _rounded_metric(float(bottom_row["confidence"])) if bottom_row else None,
        "confidence_spread": (
            _rounded_metric(float(top_row["confidence"]) - float(bottom_row["confidence"]))
            if top_row and bottom_row
            else None
        ),
        "top_operator_confidence": _rounded_metric(top_operator_confidence),
        "top_ai_confidence": _rounded_metric(top_ai_confidence),
        "selected_vs_top_operator_delta": (
            _rounded_metric(selected_confidence - top_operator_confidence)
            if selected_confidence is not None and top_operator_confidence is not None
            else None
        ),
        "selected_vs_top_ai_delta": (
            _rounded_metric(selected_confidence - top_ai_confidence)
            if selected_confidence is not None and top_ai_confidence is not None
            else None
        ),
        "operator_vs_ai_delta": (
            _rounded_metric(top_operator_confidence - top_ai_confidence)
            if top_operator_confidence is not None and top_ai_confidence is not None
            else None
        ),
    }


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
    graph = dict(manifest.get("graph_checkpoint") or {})
    proposals = {proposal["proposal_id"]: proposal for proposal in graph.get("proposals", [])}
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
        "trim_snapshot_id": (manifest.get("trim_snapshot") or {}).get("snapshot_id"),
        "graph_checkpoint_id": graph.get("checkpoint_id"),
        "selected_proposal_id": executor.get("selected_proposal_id"),
        "proposed_action_timeline": proposed_action_timeline,
        "simulated_rollback_target": initial_state,
        "restored_state_signature": _stable_signature(initial_state),
        "possible_rollbacks": possible_rollbacks,
        "blocked_reasons": sorted(set(blocked_reasons)),
        "replay_correlation_ids": manifest.get("replay_correlation_ids", []),
    }


def _build_rollback_frequency(rollback: dict[str, Any]) -> dict[str, Any]:
    timeline = list(rollback.get("proposed_action_timeline") or [])
    blocked_event_count = sum(1 for item in timeline if item.get("blocked_reason"))
    event_count = len(timeline)
    return {
        "event_count": event_count,
        "blocked_event_count": blocked_event_count,
        "blocked_event_rate": _rounded_metric(blocked_event_count / event_count if event_count else 0.0),
        "rollback_point_count": len(rollback.get("possible_rollbacks") or []),
        "blocked_reason_count": len(rollback.get("blocked_reasons") or []),
    }


def _build_audit_coverage(
    manifest: dict[str, Any],
    comparison: dict[str, Any],
    validation_report: dict[str, Any],
) -> dict[str, Any]:
    graph = dict(manifest.get("graph_checkpoint") or {})
    timeline = dict(graph.get("timeline_artifact") or {})
    evidence = dict(comparison.get("evidence") or {})
    checks = [
        {
            "key": "manifest_id",
            "ok": bool(manifest.get("manifest_id")),
            "detail": "Replay manifest id is present.",
        },
        {
            "key": "trim_snapshot_id",
            "ok": bool((manifest.get("trim_snapshot") or {}).get("snapshot_id")),
            "detail": "Trim snapshot id is present.",
        },
        {
            "key": "graph_checkpoint_id",
            "ok": bool(graph.get("checkpoint_id")),
            "detail": "Graph checkpoint id is present.",
        },
        {
            "key": "selected_proposal_id",
            "ok": bool(comparison.get("selected_proposal_id")),
            "detail": "Selected proposal id is present.",
        },
        {
            "key": "manifest_signature",
            "ok": bool(manifest.get("manifest_signature")),
            "detail": "Manifest signature is present.",
        },
        {
            "key": "timeline_signature",
            "ok": bool(timeline.get("timeline_signature")),
            "detail": "Timeline signature is present.",
        },
        {
            "key": "replay_correlation_ids",
            "ok": bool(manifest.get("replay_correlation_ids")),
            "detail": "Replay correlation ids are present for traceability.",
        },
        {
            "key": "validation_findings",
            "ok": bool(validation_report.get("findings")),
            "detail": "Validation findings are present.",
        },
        {
            "key": "dry_run_evidence",
            "ok": bool(evidence.get("dry_run_only", True)),
            "detail": "Dry-run evidence remains asserted in the comparison report.",
        },
        {
            "key": "no_live_mixer_writes",
            "ok": not bool(evidence.get("live_mixer_writes", False)),
            "detail": "Evidence shows live mixer writes stayed disabled.",
        },
    ]
    passed_check_count = sum(1 for check in checks if check["ok"])
    required_check_count = len(checks)
    return {
        "complete": passed_check_count == required_check_count,
        "passed_check_count": passed_check_count,
        "required_check_count": required_check_count,
        "coverage_rate": _rounded_metric(
            passed_check_count / required_check_count if required_check_count else 0.0
        ),
        "checks": checks,
    }


def _build_promotion_readiness_gates(
    *,
    manifest: dict[str, Any],
    score: int,
    error_count: int,
    disagreement_summary: dict[str, Any],
    rollback_frequency: dict[str, Any],
    audit_coverage: dict[str, Any],
) -> list[dict[str, Any]]:
    return [
        {
            "key": "dry_run_only",
            "ok": bool(manifest.get("dry_run_only", True)),
            "blocking": True,
            "detail": "Replay manifest remains dry-run only.",
        },
        {
            "key": "no_live_mixer_writes",
            "ok": not bool(manifest.get("live_mixer_writes", False)),
            "blocking": True,
            "detail": "Replay evidence shows no live mixer writes.",
        },
        {
            "key": "validation_errors_clear",
            "ok": error_count == 0,
            "blocking": True,
            "detail": "Validation report contains no error-severity findings.",
        },
        {
            "key": "audit_coverage_complete",
            "ok": bool(audit_coverage.get("complete")),
            "blocking": True,
            "detail": "Required audit evidence is present and signed.",
        },
        {
            "key": "readiness_score_threshold",
            "ok": score >= 75,
            "blocking": True,
            "detail": "Promotion readiness score meets the minimum threshold of 75.",
        },
        {
            "key": "disagreement_rate_clear",
            "ok": float(disagreement_summary.get("disagreement_rate", 0.0) or 0.0) == 0.0,
            "blocking": False,
            "detail": "Operator-vs-AI disagreement rate is zero for this shadow replay.",
        },
        {
            "key": "rollback_review_clear",
            "ok": int(rollback_frequency.get("blocked_event_count", 0) or 0) == 0,
            "blocking": False,
            "detail": "Replay timeline contains no blocked events that would require rollback review.",
        },
    ]


def _build_promotion_readiness(
    manifest: dict[str, Any],
    comparison: dict[str, Any],
    validation_report: dict[str, Any],
    rollback: dict[str, Any],
    confidence_trend: dict[str, Any],
) -> dict[str, Any]:
    graph = dict(manifest.get("graph_checkpoint") or {})
    timeline = dict(graph.get("timeline_artifact") or {})
    findings = validation_report.get("findings", [])
    error_count = sum(1 for item in findings if item.get("severity") == "error")
    warning_count = sum(1 for item in findings if item.get("severity") == "warning")
    info_count = sum(1 for item in findings if item.get("severity") == "info")
    disagreement_summary = dict(comparison.get("disagreement_summary") or {})
    rollback_frequency = _build_rollback_frequency(rollback)
    audit_coverage = _build_audit_coverage(manifest, comparison, validation_report)
    disagreement_penalty = 15 if comparison.get("disagreements") else 0
    blocked_penalty = 10 if rollback.get("blocked_reasons") else 0
    score = max(0, 100 - error_count * 25 - warning_count * 10 - disagreement_penalty - blocked_penalty)
    gates = _build_promotion_readiness_gates(
        manifest=manifest,
        score=score,
        error_count=error_count,
        disagreement_summary=disagreement_summary,
        rollback_frequency=rollback_frequency,
        audit_coverage=audit_coverage,
    )
    ready = all(gate["ok"] for gate in gates if gate["blocking"])
    return {
        "schema_version": REPLAY_SHADOW_REPORT_SCHEMA_VERSION,
        "ready": ready,
        "promotion_readiness_score": score,
        "dry_run_only": bool(manifest.get("dry_run_only", True)),
        "live_mixer_writes": bool(manifest.get("live_mixer_writes", False)),
        "validation_highest_severity": validation_report.get("highest_severity"),
        "validation_counts": {
            "error": error_count,
            "warning": warning_count,
            "info": info_count,
        },
        "disagreement_count": len(comparison.get("disagreements", [])),
        "blocked_reason_count": len(rollback.get("blocked_reasons", [])),
        "aggregation": {
            "disagreement_rate": disagreement_summary.get("disagreement_rate", 0.0),
            "disagreement_classes": dict((disagreement_summary.get("classes") or {})),
            "confidence_trend": confidence_trend,
            "rollback_frequency": rollback_frequency,
            "audit_coverage": audit_coverage,
        },
        "promotion_readiness_gates": gates,
        "gate_summary": {
            "blocking_gate_count": sum(1 for gate in gates if gate["blocking"]),
            "blocking_gate_passed_count": sum(1 for gate in gates if gate["blocking"] and gate["ok"]),
            "advisory_gate_count": sum(1 for gate in gates if not gate["blocking"]),
            "advisory_gate_passed_count": sum(1 for gate in gates if not gate["blocking"] and gate["ok"]),
        },
        "evidence": {
            "trim_snapshot_id": (manifest.get("trim_snapshot") or {}).get("snapshot_id"),
            "graph_checkpoint_id": graph.get("checkpoint_id"),
            "timeline_id": timeline.get("timeline_id"),
            "timeline_signature": timeline.get("timeline_signature"),
            "manifest_signature": manifest.get("manifest_signature"),
            "selected_proposal_id": comparison.get("selected_proposal_id"),
            "disagreement_codes": [item.get("code") for item in comparison.get("disagreements", [])],
            "rollback_blocked_reasons": list(rollback.get("blocked_reasons", [])),
            "validation_findings": findings,
            "replay_correlation_ids": list(manifest.get("replay_correlation_ids", [])),
        },
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
    generated_at: str | datetime | None = None,
) -> dict[str, str]:
    root = Path(output_dir)
    root.mkdir(parents=True, exist_ok=True)
    executor_dict = _coerce_executor_dict(executor)
    validation = dict(validation_report or {})
    graph = dict(manifest.get("graph_checkpoint") or {})
    timeline = dict(graph.get("timeline_artifact") or {})
    proposals = list(graph.get("proposals") or [])
    ranking_trace = list((graph.get("ranking") or {}).get("trace") or [])
    rank_map = {item["proposal_id"]: int(item["rank"]) for item in ranking_trace}
    selected_id = str(executor_dict.get("selected_proposal_id") or graph.get("selected_proposal_id") or "")
    generated_at_value = _normalized_timestamp(generated_at)

    rows: list[dict[str, Any]] = []
    for proposal in proposals:
        correlation_id = proposal.get("replay_correlation_id") or proposal["proposal_id"]
        rows.append(
            _build_comparison_row(
                proposal,
                correlation_id=correlation_id,
                rank=rank_map.get(proposal["proposal_id"], 999999),
                selected=proposal["proposal_id"] == selected_id,
            )
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
    top_operator = _first_by_rank(operator_rows)
    top_ai = _first_by_rank(ai_rows)
    selected_row = _selected_row(rows)
    disagreements = _build_disagreements(
        selected_family=selected_family,
        selected_row=selected_row,
        top_operator=top_operator,
        top_ai=top_ai,
        operator_family=operator_family,
    )
    disagreement_summary = _build_disagreement_summary(
        rows=rows,
        disagreements=disagreements,
        top_operator=top_operator,
        top_ai=top_ai,
    )

    comparison = {
        "schema_version": REPLAY_SHADOW_REPORT_SCHEMA_VERSION,
        "manifest_id": manifest["manifest_id"],
        "trim_snapshot_id": (manifest.get("trim_snapshot") or {}).get("snapshot_id"),
        "graph_checkpoint_id": graph.get("checkpoint_id"),
        "selected_family": selected_family,
        "operator_count": len(operator_rows),
        "ai_count": len(ai_rows),
        "operator_rows": operator_rows,
        "ai_rows": ai_rows,
        "selected_proposal_id": next(
            (row["proposal_id"] for row in rows if row["selected"]),
            selected_id,
        ),
        "ai_would_do_this": top_ai,
        "operator_action_capture": _selected_row(operator_rows) or top_operator,
        "disagreements": disagreements,
        "disagreement_summary": disagreement_summary,
        "evidence": {
            "ranking_signature": (graph.get("ranking") or {}).get("signature"),
            "timeline_id": timeline.get("timeline_id"),
            "timeline_signature": timeline.get("timeline_signature"),
            "manifest_signature": manifest.get("manifest_signature"),
            "replay_correlation_ids": list(manifest.get("replay_correlation_ids", [])),
            "dry_run_only": bool(manifest.get("dry_run_only", True)),
            "live_mixer_writes": bool(manifest.get("live_mixer_writes", False)),
        },
    }
    trajectory = _annotate_confidence_trajectory(rows)
    trajectory["manifest_id"] = manifest["manifest_id"]
    trajectory["selected_proposal_id"] = comparison["selected_proposal_id"]
    trajectory["selected_family"] = selected_family
    trajectory["top_operator_proposal_id"] = (top_operator or {}).get("proposal_id")
    trajectory["top_ai_proposal_id"] = (top_ai or {}).get("proposal_id")
    trajectory["trend_summary"] = _build_confidence_trend(
        rows=rows,
        selected_row=selected_row,
        top_operator=top_operator,
        top_ai=top_ai,
    )
    rollback = _build_rollback_payload(manifest, executor_dict)
    readiness = _build_promotion_readiness(
        manifest,
        comparison,
        validation,
        rollback,
        trajectory["trend_summary"],
    )
    summary = {
        "schema_version": REPLAY_SHADOW_REPORT_SCHEMA_VERSION,
        "generated_at": generated_at_value,
        "manifest_id": manifest["manifest_id"],
        "selected_proposal_id": comparison["selected_proposal_id"],
        "disagreement_count": len(disagreements),
        "disagreement_rate": disagreement_summary["disagreement_rate"],
        "promotion_readiness_score": readiness["promotion_readiness_score"],
        "audit_coverage_rate": ((readiness.get("aggregation") or {}).get("audit_coverage") or {}).get("coverage_rate"),
        "gate_failures": [
            gate["key"]
            for gate in readiness.get("promotion_readiness_gates", [])
            if not gate.get("ok")
        ],
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
