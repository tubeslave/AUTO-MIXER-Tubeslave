"""Offline decision loop and conservative apply workflow."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable, List, Optional

from mix_agent.actions import render_conservative_mix
from mix_agent.analysis import analyze_loaded_context, load_audio_context
from mix_agent.models import MixAction, SuggestionPlan
from mix_agent.reporting import write_json_report, write_markdown_report

from .evaluator import compare_before_after
from .planner import build_suggestion_plan


def analyze(
    *,
    stems: str | None = None,
    mix: str | None = None,
    reference: str | None = None,
    genre: str = "neutral",
    target_platform: str = "streaming",
) -> SuggestionPlan:
    """Load audio, analyze it and build a suggestion plan."""
    loaded = load_audio_context(
        stems_path=stems,
        mix_path=mix,
        reference_path=reference,
        genre=genre,
        target_platform=target_platform,
    )
    analysis = analyze_loaded_context(loaded)
    return build_suggestion_plan(analysis)


def suggest(
    *,
    stems: str | None = None,
    mix: str | None = None,
    reference: str | None = None,
    genre: str = "neutral",
    target_platform: str = "streaming",
    out: str | None = None,
) -> SuggestionPlan:
    """Analyze and optionally write a JSON/Markdown report."""
    plan = analyze(
        stems=stems,
        mix=mix,
        reference=reference,
        genre=genre,
        target_platform=target_platform,
    )
    if out:
        if str(out).lower().endswith(".md"):
            write_markdown_report(plan, out)
        else:
            write_json_report(plan, out)
    return plan


def _load_actions_from_suggestions(path: str | Path) -> List[MixAction]:
    payload = json.loads(Path(path).expanduser().read_text(encoding="utf-8"))
    action_payloads = payload.get("actions", payload if isinstance(payload, list) else [])
    return [
        MixAction(
            id=str(item.get("id", f"action_{idx}")),
            action_type=str(item.get("action_type", "")),
            target=str(item.get("target", "")),
            parameters=dict(item.get("parameters", {})),
            safe_range=dict(item.get("safe_range", {})),
            mode=str(item.get("mode", "recommend")),
            reason=str(item.get("reason", "")),
            expected_improvement=str(item.get("expected_improvement", "")),
            risk=str(item.get("risk", "")),
            confidence=float(item.get("confidence", 0.5)),
            reversible=bool(item.get("reversible", True)),
            backend_mapping=dict(item.get("backend_mapping", {})),
            status=str(item.get("status", "proposed")),
        )
        for idx, item in enumerate(action_payloads)
        if isinstance(item, dict)
    ]


def apply_conservative(
    *,
    stems: str,
    out: str,
    suggestions: str | None = None,
    reference: str | None = None,
    genre: str = "neutral",
    target_platform: str = "streaming",
    actions: Optional[Iterable[MixAction]] = None,
    report: str | None = None,
) -> SuggestionPlan:
    """Apply only conservative reversible offline operations and re-analyze."""
    loaded = load_audio_context(
        stems_path=stems,
        mix_path=None,
        reference_path=reference,
        genre=genre,
        target_platform=target_platform,
    )
    before_audio = loaded.mix_audio.copy()
    plan = build_suggestion_plan(analyze_loaded_context(loaded))
    selected_actions = list(actions or [])
    if suggestions:
        selected_actions = _load_actions_from_suggestions(suggestions)
    if not selected_actions:
        selected_actions = [
            action
            for action in plan.actions
            if action.action_type in {"gain_adjustment", "high_pass_filter", "parametric_eq", "pan_adjustment"}
        ]

    roles = {name: info.role for name, info in loaded.stem_info.items()}
    render_report = render_conservative_mix(
        loaded.stems,
        roles,
        loaded.mix_sample_rate,
        selected_actions,
        out,
    )
    after_loaded = load_audio_context(
        stems_path=None,
        mix_path=out,
        reference_path=reference,
        genre=genre,
        target_platform=target_platform,
    )
    after_audio = after_loaded.mix_audio
    plan.applied_actions = selected_actions
    plan.audit_trail.append({"render": render_report})
    plan.audit_trail.append(
        {
            "before_after": compare_before_after(
                before_audio,
                after_audio,
                loaded.mix_sample_rate,
            )
        }
    )
    if report:
        if str(report).lower().endswith(".md"):
            write_markdown_report(plan, report)
        else:
            write_json_report(plan, report)
    return plan
