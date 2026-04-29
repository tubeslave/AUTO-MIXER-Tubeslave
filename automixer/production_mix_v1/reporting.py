"""Report writers for `production_mix_v1`."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .models import PipelineRunResult, jsonable


def write_reports(result: PipelineRunResult, output_dir: str | Path) -> dict[str, str]:
    """Write JSON and Markdown reports for a pipeline run."""
    root = Path(output_dir).expanduser()
    root.mkdir(parents=True, exist_ok=True)
    json_path = root / "production_mix_v1_report.json"
    md_path = root / "production_mix_v1_report.md"
    payload = result.to_dict()
    json_path.write_text(json.dumps(jsonable(payload), indent=2, ensure_ascii=True), encoding="utf-8")
    md_path.write_text(render_markdown_report(payload), encoding="utf-8")
    return {"json": str(json_path), "markdown": str(md_path)}


def render_markdown_report(payload: dict[str, Any]) -> str:
    decision = payload.get("decision", {})
    selected = decision.get("selected", {}).get("candidate", {})
    queue = payload.get("queue_result", {})
    fields = payload.get("report_fields", {}) or {}
    lines = [
        "# production_mix_v1 Report",
        "",
        f"- run_id: `{payload.get('run_id')}`",
        f"- mode: `{payload.get('mode')}`",
        f"- dry_run: `{payload.get('dry_run')}`",
        f"- selected: `{selected.get('name')}`",
        f"- accepted: `{decision.get('accepted')}`",
        f"- decision_state: `{decision.get('decision_state')}`",
        f"- offline_render_accepted: `{decision.get('offline_render_accepted')}`",
        f"- console_actions_accepted: `{decision.get('console_actions_accepted')}`",
        f"- live_ready: `{decision.get('live_ready')}`",
        f"- inherited_reference_recipe: `{fields.get('inherited_reference_recipe', '')}`",
        f"- musical_score: {float(fields.get('musical_score', 0.0)):.4f}",
        f"- technical_score: {float(fields.get('technical_score', 0.0)):.4f}",
        f"- final_score: {float(fields.get('final_score', 0.0)):.4f}",
        f"- queued dry-run commands: {len(queue.get('would_send', []))}",
        f"- sent commands: {len(queue.get('sent', []))}",
        "",
        "## Rationale",
        "",
    ]
    for item in decision.get("rationale", []):
        lines.append(f"- {item}")
    lines.extend(["", "## Selected Candidate", ""])
    lines.append(f"- stage: `{selected.get('stage')}`")
    lines.append(f"- actions: {len(selected.get('actions', []))}")
    for action in selected.get("actions", []):
        lines.append(
            f"- `{action.get('action_type')}` ch={action.get('channel_id')} "
            f"{action.get('rationale', '')}"
        )
    lines.extend(["", "## Musical Recipe Fields", ""])
    lines.append(f"- inherited_reference_recipe: `{fields.get('inherited_reference_recipe', '')}`")
    lines.append(f"- candidate_type: `{fields.get('candidate_type', '')}`")
    lines.append(f"- material_improvement: {fields.get('material_improvement')}")
    lines.append(f"- planned_actions_count: {fields.get('planned_actions_count', 0)}")
    lines.append(f"- rendered_actions_count: {fields.get('rendered_actions_count', 0)}")
    lines.append(f"- verified_actions_count: {fields.get('verified_actions_count', 0)}")
    lines.append(f"- failed_actions_count: {fields.get('failed_actions_count', 0)}")
    lines.append(f"- muq_available: {fields.get('muq_available')}")
    lines.append(f"- aggressive_changes_allowed: {fields.get('aggressive_changes_allowed')}")
    lines.append(f"- reason_if_aggressive_allowed: {fields.get('reason_if_aggressive_allowed', '')}")
    lines.append(f"- static_gain_actions: {len(fields.get('static_gain_actions', []))}")
    lines.append(f"- fader_balance_actions: {len(fields.get('fader_balance_actions', []))}")
    lines.append(f"- space_fx_actions: {len(fields.get('space_fx_actions', []))}")
    lines.append(f"- dynamics_actions: {len(fields.get('dynamics_actions', []))}")
    lines.append(f"- utility_actions: {len(fields.get('utility_actions', []))}")
    lines.append(f"- measured_gain_reduction_proxy: {fields.get('measured_gain_reduction_proxy', 0.0)}")
    lines.append(f"- fx_slot: {fields.get('fx_slot', [])}")
    lines.append(f"- fx_model: {fields.get('fx_model', [])}")
    lines.append(f"- fx_send_level: {fields.get('fx_send_level', [])}")
    lines.append(f"- fx_return_level: {fields.get('fx_return_level', [])}")
    lines.append(f"- fx_wetness_proxy: {fields.get('fx_wetness_proxy', 0.0)}")
    lines.append(f"- fx_wetness_ok: {fields.get('fx_wetness_ok')}")
    lines.append(f"- natural_ambience_ok: {fields.get('natural_ambience_ok')}")
    lines.append(f"- natural_ambience_destroyed: {fields.get('natural_ambience_destroyed')}")
    lines.append(f"- candidate_is_not_dry: {fields.get('candidate_is_not_dry')}")
    lines.append(f"- large_gain_moves: {fields.get('large_gain_moves', [])}")
    lines.append(
        f"- large_cuts_to_room_overheads_cymbals: {fields.get('large_cuts_to_room_overheads_cymbals', [])}"
    )
    lines.append(f"- compression_gr_outliers: {fields.get('compression_gr_outliers', [])}")
    lines.append(f"- metrics_conflicts: {fields.get('metrics_conflicts', [])}")
    lines.append(f"- dryness_penalty: {fields.get('dryness_penalty', 0.0)}")
    lines.append(
        f"- natural_ambience_preservation_score: {fields.get('natural_ambience_preservation_score', 1.0)}"
    )
    lines.append(f"- why_balance_was_changed: {fields.get('why_balance_was_changed', [])}")
    lines.append(f"- why_candidate_is_not_dry: {fields.get('why_candidate_is_not_dry', '')}")
    lines.append(
        f"- why_candidate_is_not_overdynamic: {fields.get('why_candidate_is_not_overdynamic', '')}"
    )
    lines.append(f"- why_not_fx15_reference: {fields.get('why_not_fx15_reference', '')}")
    lines.extend(["", "## Render Verification", ""])
    selected_eval = decision.get("selected", {})
    verification = selected_eval.get("render_verification", {}) or {}
    if verification:
        for key in (
            "actual_offline_render",
            "required_actions_rendered",
            "required_actions_verified",
            "candidate_is_not_dry",
            "candidate_is_not_overdynamic",
            "natural_ambience_preservation_score",
            "dryness_penalty",
            "vocal_balance_ok",
            "stem_balance_ok",
            "kick_bass_balance_ok",
            "metrics_improved",
            "material_improvement",
            "fx_wetness_ok",
            "natural_ambience_ok",
            "natural_ambience_destroyed",
            "metrics_conflicts",
        ):
            value = verification.get(key)
            if key == "material_improvement":
                value = selected_eval.get("material_improvement")
            lines.append(f"- {key}: {value}")
    statuses = selected_eval.get("action_statuses", []) or []
    if statuses:
        lines.append("")
        lines.append("### Action Status")
        for item in statuses:
            action = item.get("action", {})
            lines.append(
                f"- `{action.get('action_type')}` ch={action.get('channel_id')} "
                f"stage={action.get('stage')} status={item.get('status')} "
                f"verified={item.get('verified')}"
            )
    role_table = fields.get("role_confidence_table", []) or []
    if role_table:
        lines.extend(["", "## Role Confidence", ""])
        for item in role_table:
            lines.append(
                f"- ch={item.get('channel_id')} `{item.get('name')}` role={item.get('role')} "
                f"confidence={item.get('confidence')} allowed={item.get('allowed_action_level')} "
                f"ambiguous={item.get('is_ambiguous')}"
            )
        lines.append(f"- ambiguous_roles: {fields.get('ambiguous_roles', [])}")
        lines.append(f"- user_overrides_applied: {fields.get('user_overrides_applied', [])}")
        lines.append(
            f"- blocked_role_specific_actions: {fields.get('blocked_role_specific_actions', [])}"
        )
    lines.extend(["", "## Rejections", ""])
    rejected = decision.get("rejected", [])
    if not rejected:
        lines.append("- none")
    for item in rejected:
        lines.append(
            f"- `{item.get('candidate')}`: {', '.join(item.get('reasons', []))}"
        )
    lines.extend(["", "## Candidate Scores", ""])
    evaluations = payload.get("evaluations", [])
    if not evaluations:
        lines.append("- none")
    for item in evaluations:
        candidate = item.get("candidate", {})
        safety = item.get("safety", {}) or {}
        lines.append(
            f"- `{candidate.get('name')}` type={candidate.get('metadata', {}).get('candidate_type', '')} "
            f"musical={item.get('musical_score'):.4f} "
            f"musical_delta={item.get('musical_score_delta'):.4f} "
            f"technical={item.get('technical_safety_score'):.4f} "
            f"final={item.get('final_score'):.4f} "
            f"final_delta={item.get('final_score_delta'):.4f} "
            f"material={item.get('material_improvement')} "
            f"safety={safety.get('passed')}"
        )
    lines.extend(["", "## Warnings", ""])
    warnings = payload.get("warnings", [])
    if not warnings:
        lines.append("- none")
    for warning in warnings:
        lines.append(f"- {warning}")
    return "\n".join(lines) + "\n"
