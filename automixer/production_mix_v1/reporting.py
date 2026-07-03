"""Report writers for production_mix_v1."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .models import PipelineRunResult, jsonable


def write_reports(result: PipelineRunResult, output_dir: str | Path) -> dict[str, str]:
    root = Path(output_dir)
    root.mkdir(parents=True, exist_ok=True)
    payload = result.to_dict()
    json_path = root / "production_mix_v1_report.json"
    md_path = root / "production_mix_v1_report.md"
    json_path.write_text(json.dumps(jsonable(payload), indent=2, ensure_ascii=True), encoding="utf-8")
    md_path.write_text(render_markdown_report(payload), encoding="utf-8")
    return {"json": str(json_path), "markdown": str(md_path)}


def render_markdown_report(payload: dict[str, Any]) -> str:
    decision = payload.get("decision", {})
    selected_eval = decision.get("selected", {})
    selected = selected_eval.get("candidate", {})
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
        "",
        "## Rationale",
        "",
    ]
    lines.extend(f"- {item}" for item in decision.get("rationale", []))
    lines.extend(["", "## Selected Candidate", ""])
    lines.append(f"- candidate_type: `{fields.get('candidate_type')}`")
    lines.append(f"- actions: {len(selected.get('actions', []))}")
    for action in selected.get("actions", []):
        lines.append(f"- `{action.get('action_type')}` ch={action.get('channel_id')} {action.get('rationale', '')}")
    lines.extend(["", "## Decision Margins", ""])
    for key in (
        "material_improvement",
        "material_improvement_vs_no_change",
        "material_improvement_vs_reference",
        "win_margin_over_reference_final",
        "win_margin_over_reference_musical",
        "selected_by_tiebreak",
        "tiebreak_reason",
        "pre_trim_true_peak_dbfs",
        "pre_trim_headroom_rejection",
        "overhead_parallel_compression_risk",
        "metrics_conflicts",
        "offline_fx_return_present",
        "fx_wetness_proxy",
        "spatial_reflection_energy_proxy",
        "spatial_fx_verified",
        "modulation_energy_proxy",
        "modulation_fx_verified",
        "stereo_width_delta_proxy",
        "guitars_too_forward",
        "guitar_vocal_masking_index_before",
        "guitar_vocal_masking_index_after",
        "guitar_pair_trim_db",
        "guitar_dynamic_eq_actions",
        "guitar_to_vocal_midrange_ratio",
        "guitar_energy_700_1500",
        "guitar_energy_1500_3000",
        "vocal_energy_700_1500",
        "vocal_energy_1500_3000",
        "vocal_clarity_before_after",
        "reference_preserved",
        "why_guitars_were_trimmed",
        "why_guitars_were_not_trimmed",
        "muq_available",
        "aggressive_changes_allowed",
        "why_not_fx15_reference",
    ):
        lines.append(f"- {key}: {fields.get(key)}")
    lines.extend(["", "## Render Verification", ""])
    verification = selected_eval.get("render_verification", {}) or {}
    for key, value in verification.items():
        if key in {"pre_trim_true_peak_dbfs", "pre_trim_headroom_rejection", "metrics_conflicts", "overhead_parallel_compression_risk", "kick_bass_balance_ok", "candidate_is_not_overdynamic", "guitar_forwardness"}:
            lines.append(f"- {key}: {value}")
    lines.extend(["", "## Rejections", ""])
    for item in decision.get("rejected", []) or []:
        lines.append(f"- `{item.get('candidate')}`: {', '.join(item.get('reasons', []))}")
    lines.extend(["", "## Candidate Scores", ""])
    for item in payload.get("evaluations", []) or []:
        candidate = item.get("candidate", {})
        lines.append(
            f"- `{candidate.get('name')}` type={candidate.get('metadata', {}).get('candidate_type')} "
            f"musical={float(item.get('musical_score', 0.0)):.4f} "
            f"final={float(item.get('final_score', 0.0)):.4f} "
            f"material_ref={item.get('material_improvement_vs_reference')}"
        )
    lines.extend(["", "## Warnings", ""])
    warnings = payload.get("warnings", [])
    lines.extend([f"- {warning}" for warning in warnings] or ["- none"])
    return "\n".join(lines) + "\n"
