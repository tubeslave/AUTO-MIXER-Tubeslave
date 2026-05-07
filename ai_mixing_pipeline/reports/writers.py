"""Output writers for the offline AI mixing chain."""

from __future__ import annotations

from datetime import datetime, timezone
import csv
import json
from pathlib import Path
from typing import Any

from ai_mixing_pipeline.models import DecisionResult, MixCandidate, SafetyResult, jsonable


def write_json(path: str | Path, payload: Any) -> Path:
    target = Path(path).expanduser()
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(jsonable(payload), indent=2, ensure_ascii=True), encoding="utf-8")
    return target


def write_decision_log(
    path: str | Path,
    *,
    mode: str,
    candidates: list[MixCandidate],
    evaluations: dict[str, dict[str, dict[str, Any]]],
    decision: DecisionResult,
    safety_results: dict[str, SafetyResult],
    render_paths: dict[str, str],
) -> Path:
    target = Path(path).expanduser()
    target.parent.mkdir(parents=True, exist_ok=True)
    now = datetime.now(timezone.utc).isoformat()
    with target.open("w", encoding="utf-8") as handle:
        for candidate in candidates:
            safety = safety_results.get(candidate.candidate_id)
            accepted = candidate.candidate_id == decision.selected_candidate_id
            row = {
                "timestamp": now,
                "mode": mode,
                "candidate_id": candidate.candidate_id,
                "actions": [action.to_dict() for action in candidate.actions],
                "critic_scores": evaluations.get(candidate.candidate_id, {}),
                "final_score": decision.final_scores.get(candidate.candidate_id, 0.0),
                "safety_passed": bool(safety.passed if safety else False),
                "accepted": accepted,
                "rejection_reason": None if accepted else _rejection_reason(safety),
                "render_path": render_paths.get(candidate.candidate_id, ""),
                "explanation": decision.explanations.get(candidate.candidate_id, ""),
            }
            handle.write(json.dumps(jsonable(row), ensure_ascii=True) + "\n")
    return target


def write_critic_scores_csv(
    path: str | Path,
    evaluations: dict[str, dict[str, dict[str, Any]]],
    decision: DecisionResult,
) -> Path:
    target = Path(path).expanduser()
    target.parent.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []
    for candidate_id, by_critic in evaluations.items():
        for critic_name, result in by_critic.items():
            rows.append(
                {
                    "candidate_id": candidate_id,
                    "critic_name": critic_name,
                    "role": result.get("role", ""),
                    "overall_score": _score(result, "overall"),
                    "overall_delta": _delta(result, "overall"),
                    "confidence": result.get("confidence", 0.0),
                    "model_available": result.get("model_available", False),
                    "final_score": decision.final_scores.get(candidate_id, 0.0),
                    "warnings": " | ".join(result.get("warnings", [])),
                }
            )
    fieldnames = [
        "candidate_id",
        "critic_name",
        "role",
        "overall_score",
        "overall_delta",
        "confidence",
        "model_available",
        "final_score",
        "warnings",
    ]
    with target.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    return target


def write_accepted_rejected_actions(
    reports_dir: str | Path,
    candidates: list[MixCandidate],
    decision: DecisionResult,
    safety_results: dict[str, SafetyResult],
) -> tuple[Path, Path]:
    reports = Path(reports_dir).expanduser()
    accepted: list[dict[str, Any]] = []
    rejected: list[dict[str, Any]] = []
    for candidate in candidates:
        safety = safety_results.get(candidate.candidate_id)
        payload = {
            "candidate_id": candidate.candidate_id,
            "label": candidate.label,
            "actions": [action.to_dict() for action in candidate.actions],
            "safety": safety.to_dict() if safety else None,
            "final_score": decision.final_scores.get(candidate.candidate_id, 0.0),
        }
        if candidate.candidate_id == decision.selected_candidate_id and decision.no_change_selected:
            continue
        if candidate.candidate_id == decision.selected_candidate_id:
            accepted.extend(payload["actions"])
        else:
            rejected.append(payload)
    accepted_path = write_json(reports / "accepted_actions.json", accepted)
    rejected_path = write_json(reports / "rejected_actions.json", rejected)
    return accepted_path, rejected_path


def write_summary_report(
    path: str | Path,
    *,
    mode: str,
    candidates: list[MixCandidate],
    evaluations: dict[str, dict[str, dict[str, Any]]],
    decision: DecisionResult,
    safety_results: dict[str, SafetyResult],
    render_paths: dict[str, str],
    module_status: dict[str, Any],
    source_separation: dict[str, Any] | None = None,
) -> Path:
    target = Path(path).expanduser()
    target.parent.mkdir(parents=True, exist_ok=True)
    selected = decision.selected_candidate_id
    lines = [
        "# AI Mixing Offline Test Summary",
        "",
        f"- Mode: `{mode}`",
        f"- Selected candidate: `{selected}`",
        f"- No change selected: `{decision.no_change_selected}`",
        f"- Best render: `{render_paths.get(selected, '')}`",
        "",
        "## Module Availability",
        "",
    ]
    for name, status in sorted(module_status.items()):
        available = status.get("available", False) if isinstance(status, dict) else bool(status)
        warning = "; ".join(status.get("warnings", [])) if isinstance(status, dict) else ""
        lines.append(f"- `{name}`: {'available' if available else 'fallback/unavailable'} {warning}".rstrip())
    if source_separation:
        lines.extend(["", "## Source Separation", "", f"- Status: `{source_separation.get('available', False)}`"])
        for warning in source_separation.get("warnings", []):
            lines.append(f"- {warning}")
    lines.extend(["", "## Candidates", ""])
    for candidate in candidates:
        safety = safety_results.get(candidate.candidate_id)
        lines.append(f"### {candidate.candidate_id}")
        lines.append("")
        lines.append(f"- Label: `{candidate.label}`")
        lines.append(f"- Render: `{render_paths.get(candidate.candidate_id, '')}`")
        lines.append(f"- Final score: `{decision.final_scores.get(candidate.candidate_id, 0.0):.6f}`")
        lines.append(f"- Safety: `{'passed' if safety and safety.passed else 'blocked'}`")
        if safety:
            level = (safety.metrics.get("level", {}) or {}) if isinstance(safety.metrics, dict) else {}
            lines.append(
                f"- Safety level: true_peak=`{_as_float(level.get('true_peak_dbtp'), -120.0):.3f} dBTP`, "
                f"headroom=`{_as_float(level.get('headroom_db'), 0.0):.3f} dB`, "
                f"clips=`{int(level.get('clip_count', 0) or 0)}`"
            )
        if safety and safety.reasons:
            lines.append(f"- Rejection reasons: `{', '.join(safety.reasons)}`")
        if candidate.actions:
            lines.append("- Actions:")
            for action in candidate.actions:
                lines.append(f"  - `{action.action_type}` on `{action.target}`: {action.reason}")
        else:
            lines.append("- Actions: none")
        lines.append("")
    lines.extend(["## Critic Scores", ""])
    for candidate_id, by_critic in evaluations.items():
        lines.append(f"### {candidate_id}")
        lines.append("")
        for critic_name, result in by_critic.items():
            lines.append(
                f"- `{critic_name}` delta={_delta(result, 'overall'):.4f} "
                f"score={_score(result, 'overall'):.4f} confidence={float(result.get('confidence', 0.0)):.2f}"
            )
        lines.append("")
    lines.extend(
        [
            "## Decision",
            "",
            decision.explanations.get(selected, ""),
            "",
            "## Safety Governor",
            "",
            "The selected candidate is accepted only after action bounds, clipping, headroom, phase, vocal clarity, compression, and identity/bleed checks pass. MuQ-Eval and other critics never write mixer parameters directly.",
        ]
    )
    target.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return target


def _score(result: dict[str, Any], key: str) -> float:
    value = (result.get("scores", {}) or {}).get(key)
    return float(value) if isinstance(value, (int, float)) else 0.0


def _delta(result: dict[str, Any], key: str) -> float:
    value = (result.get("delta", {}) or {}).get(key)
    return float(value) if isinstance(value, (int, float)) else 0.0


def _as_float(value: Any, default: float) -> float:
    return float(value) if isinstance(value, (int, float)) else float(default)


def _rejection_reason(safety: SafetyResult | None) -> str:
    if safety and safety.reasons:
        return ";".join(safety.reasons)
    if safety and safety.passed:
        return "not_selected"
    return "not_selected_or_missing_safety"
