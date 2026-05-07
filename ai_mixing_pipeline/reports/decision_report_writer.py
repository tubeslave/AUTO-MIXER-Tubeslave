"""Report writer for the decision/correction layer."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Any
import csv
import json


def write_decision_layer_reports(
    reports_dir: str | Path,
    *,
    run_id: str,
    candidates: list[Any],
    render_results: dict[str, dict[str, Any]],
    critic_scores: dict[str, dict[str, dict[str, Any]]],
    safety_results: dict[str, dict[str, Any]],
    decision: dict[str, Any],
    optimizer_status: dict[str, Any],
    mixer_status: dict[str, Any],
    dependency_status: dict[str, Any],
    module_status: dict[str, Any],
    rejected_pre_render: list[dict[str, Any]] | None = None,
) -> dict[str, str]:
    reports = Path(reports_dir).expanduser()
    reports.mkdir(parents=True, exist_ok=True)
    rejected_pre_render = list(rejected_pre_render or [])
    paths = {
        "decision_log": reports / "decision_log.jsonl",
        "critic_scores": reports / "critic_scores.csv",
        "accepted_actions": reports / "accepted_actions.json",
        "rejected_actions": reports / "rejected_actions.json",
        "optimizer_history": reports / "optimizer_history.json",
        "summary_report": reports / "summary_report.md",
    }
    _write_decision_log(paths["decision_log"], run_id, candidates, render_results, critic_scores, safety_results, decision)
    _write_critic_scores(paths["critic_scores"], critic_scores, decision)
    _write_actions(paths["accepted_actions"], paths["rejected_actions"], candidates, safety_results, decision, rejected_pre_render)
    _write_json(paths["optimizer_history"], optimizer_status)
    _write_summary(
        paths["summary_report"],
        run_id,
        candidates,
        render_results,
        critic_scores,
        safety_results,
        decision,
        optimizer_status,
        mixer_status,
        dependency_status,
        module_status,
        rejected_pre_render,
    )
    return {key: str(value) for key, value in paths.items()}


def _write_decision_log(
    path: Path,
    run_id: str,
    candidates: list[Any],
    render_results: dict[str, dict[str, Any]],
    critic_scores: dict[str, dict[str, dict[str, Any]]],
    safety_results: dict[str, dict[str, Any]],
    decision: dict[str, Any],
) -> None:
    now = datetime.now(timezone.utc).isoformat()
    selected = decision.get("selected_candidate_id")
    with path.open("w", encoding="utf-8") as handle:
        for candidate in candidates:
            safety = safety_results.get(candidate.candidate_id, {})
            accepted = candidate.candidate_id == selected
            row = {
                "timestamp": now,
                "run_id": run_id,
                "mode": "offline_test",
                "candidate_id": candidate.candidate_id,
                "actions": [action.to_dict() for action in candidate.actions],
                "critic_scores": critic_scores.get(candidate.candidate_id, {}),
                "final_score": decision.get("final_scores", {}).get(candidate.candidate_id, 0.0),
                "safety_passed": bool(safety.get("passed", False)),
                "accepted": accepted,
                "rejection_reason": None if accepted else _rejection_reason(safety),
                "render_path": render_results.get(candidate.candidate_id, {}).get("path", ""),
                "explanation": decision.get("reason", ""),
            }
            handle.write(json.dumps(row, ensure_ascii=True, default=str) + "\n")


def _write_critic_scores(path: Path, critic_scores: dict[str, dict[str, dict[str, Any]]], decision: dict[str, Any]) -> None:
    rows: list[dict[str, Any]] = []
    for candidate_id, by_critic in critic_scores.items():
        for name, result in by_critic.items():
            rows.append(
                {
                    "candidate_id": candidate_id,
                    "critic_name": name,
                    "role": result.get("role", ""),
                    "overall_score": (result.get("scores", {}) or {}).get("overall", ""),
                    "overall_delta": (result.get("delta", {}) or {}).get("overall", ""),
                    "confidence": result.get("confidence", 0.0),
                    "model_available": result.get("model_available", False),
                    "final_score": decision.get("final_scores", {}).get(candidate_id, 0.0),
                    "warnings": " | ".join(result.get("warnings", [])),
                }
            )
    with path.open("w", encoding="utf-8", newline="") as handle:
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
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _write_actions(
    accepted_path: Path,
    rejected_path: Path,
    candidates: list[Any],
    safety_results: dict[str, dict[str, Any]],
    decision: dict[str, Any],
    rejected_pre_render: list[dict[str, Any]],
) -> None:
    accepted: list[dict[str, Any]] = []
    rejected = list(rejected_pre_render)
    selected = decision.get("selected_candidate_id")
    for candidate in candidates:
        payload = {
            "candidate_id": candidate.candidate_id,
            "description": candidate.description,
            "actions": [action.to_dict() for action in candidate.actions],
            "safety": safety_results.get(candidate.candidate_id, {}),
            "final_score": decision.get("final_scores", {}).get(candidate.candidate_id, 0.0),
        }
        if candidate.candidate_id == selected and candidate.actions and not candidate.is_no_change:
            accepted.extend(payload["actions"])
        elif candidate.candidate_id != selected:
            rejected.append(payload)
    _write_json(accepted_path, accepted)
    _write_json(rejected_path, rejected)


def _write_summary(
    path: Path,
    run_id: str,
    candidates: list[Any],
    render_results: dict[str, dict[str, Any]],
    critic_scores: dict[str, dict[str, dict[str, Any]]],
    safety_results: dict[str, dict[str, Any]],
    decision: dict[str, Any],
    optimizer_status: dict[str, Any],
    mixer_status: dict[str, Any],
    dependency_status: dict[str, Any],
    module_status: dict[str, Any],
    rejected_pre_render: list[dict[str, Any]],
) -> None:
    selected = decision.get("selected_candidate_id", "")
    lines = [
        "# AI Decision Correction Layer Summary",
        "",
        f"- Run ID: `{run_id}`",
        f"- Selected candidate: `{selected}`",
        f"- Decision: `{decision.get('decision', '')}`",
        f"- Reason: {decision.get('reason', '')}",
        f"- Best mix: `{render_results.get('best_mix', {}).get('path', '')}`",
        "",
        "## Dependencies",
        "",
    ]
    for name, status in sorted(dependency_status.items()):
        lines.append(f"- `{name}`: {'available' if status else 'missing'}")
    lines.extend(["", "## Virtual Mixer", "", f"- Actual: `{mixer_status.get('actual_virtual_mixer', mixer_status.get('virtual_mixer', ''))}`"])
    for warning in mixer_status.get("warnings", []):
        lines.append(f"- Warning: {warning}")
    lines.extend(["", "## Optimizer", "", f"- Name: `{optimizer_status.get('name', '')}`", f"- Available: `{optimizer_status.get('available', False)}`"])
    for warning in optimizer_status.get("warnings", []):
        lines.append(f"- Warning: {warning}")
    lines.extend(["", "## Modules", ""])
    for name, status in sorted(module_status.items()):
        lines.append(f"- `{name}`: {'available' if status.get('available') else 'fallback/unavailable'}")
    lines.extend(["", "## Candidate Counts", ""])
    lines.append(f"- Created: `{len(candidates)}`")
    lines.append(f"- Rejected before render: `{len(rejected_pre_render)}`")
    lines.append(f"- Rendered: `{len(render_results) - (1 if 'best_mix' in render_results else 0)}`")
    lines.extend(["", "## Candidates", ""])
    for candidate in candidates:
        safety = safety_results.get(candidate.candidate_id, {})
        lines.append(f"### {candidate.candidate_id}")
        lines.append(f"- Description: {candidate.description}")
        lines.append(f"- Render: `{render_results.get(candidate.candidate_id, {}).get('path', '')}`")
        lines.append(f"- Final score: `{decision.get('final_scores', {}).get(candidate.candidate_id, 0.0):.6f}`")
        lines.append(f"- Safety: `{'passed' if safety.get('passed') else 'blocked'}`")
        if safety.get("reasons"):
            lines.append(f"- Rejection reasons: `{', '.join(safety.get('reasons', []))}`")
        lines.append(f"- Actions: `{len(candidate.actions)}`")
        lines.append("")
    lines.extend(["## Critic Scores", ""])
    for candidate_id, by_critic in critic_scores.items():
        lines.append(f"### {candidate_id}")
        for name, result in by_critic.items():
            delta = (result.get("delta", {}) or {}).get("overall", 0.0)
            lines.append(f"- `{name}` delta=`{float(delta) if isinstance(delta, (int, float)) else 0.0:.4f}` confidence=`{result.get('confidence', 0.0)}`")
        lines.append("")
    lines.extend(
        [
            "## Safety Governor",
            "",
            "The Safety Governor is the final protection layer. It may reject the highest-scoring candidate for action bounds, clipping, headroom, phase, vocal clarity, excessive compression, or bleed risk.",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=True, default=str), encoding="utf-8")


def _rejection_reason(safety: dict[str, Any]) -> str:
    if safety.get("reasons"):
        return ";".join(safety["reasons"])
    if safety.get("passed"):
        return "not_selected"
    return "not_rendered_or_missing_safety"
