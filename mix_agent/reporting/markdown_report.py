"""Markdown report generation for mix engineers."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

from mix_agent.models import RuleIssue, SuggestionPlan


def _line_items(items: Iterable[str]) -> str:
    values = [str(item) for item in items if str(item)]
    if not values:
        return "- none"
    return "\n".join(f"- {item}" for item in values)


def _metrics_table(plan: SuggestionPlan) -> str:
    mix = plan.analysis.mix
    level = mix.level
    stereo = mix.stereo
    rows = [
        ("Integrated LUFS", level.get("integrated_lufs")),
        ("True peak", level.get("true_peak_dbtp")),
        ("Peak", level.get("peak_dbfs")),
        ("RMS", level.get("rms_dbfs")),
        ("LRA", level.get("loudness_range_lu")),
        ("Crest factor", level.get("crest_factor_db")),
        ("PLR", level.get("plr_db")),
        ("Headroom", level.get("headroom_db")),
        ("Stereo width", stereo.get("stereo_width")),
        ("Correlation", stereo.get("inter_channel_correlation")),
        ("Mono fold-down loss", stereo.get("mono_fold_down_loss_db")),
    ]
    body = "\n".join(f"| {name} | {value} |" for name, value in rows)
    return "| Metric | Value |\n| --- | --- |\n" + body


def _render_issue(issue: RuleIssue) -> str:
    actions = []
    for action in issue.actions:
        actions.append(
            f"{action.action_type} on `{action.target}` with {action.parameters} "
            f"(confidence {action.confidence:.2f})"
        )
    return "\n".join(
        [
            f"### {issue.severity.upper()}: {issue.name}",
            "",
            f"**Issue:** {issue.explanation}",
            "",
            "**Evidence:**",
            _line_items(issue.evidence),
            "",
            f"**Suggested action:** {issue.suggested_action}",
            "",
            f"**Expected improvement:** {issue.expected_improvement}",
            "",
            f"**Risk:** {issue.risk}",
            "",
            f"**Confidence:** {issue.confidence:.2f}",
            "",
            "**Do not apply constraints:**",
            _line_items(issue.do_not_apply_constraints),
            "",
            "**Machine-readable actions:**",
            _line_items(actions),
        ]
    )


def render_markdown_report(plan: SuggestionPlan) -> str:
    """Render a Markdown report with metrics, issues, actions and limitations."""
    ctx = plan.analysis.context
    dash = plan.dashboard
    limitations = list(dict.fromkeys(plan.analysis.limitations))
    ref = plan.analysis.reference
    sections = [
        "# Mix Agent Report",
        "",
        "## Summary",
        "",
        f"- Mode: {ctx.mode}",
        f"- Genre/profile: {ctx.genre}",
        f"- Target platform: {ctx.target_platform}",
        f"- Sample rate: {ctx.sample_rate}",
        f"- Issues detected: {len(plan.issues)}",
        f"- Proposed actions: {len(plan.actions)}",
        f"- Overall recommendation confidence: {dash.overall_recommendation_confidence:.2f}",
        "",
        "This report is an engineering decision aid. It does not replace loudness-matched listening, manual intent checks or a subjective final approval.",
        "",
        "## Quality Dashboard",
        "",
        "| Component | Score |",
        "| --- | --- |",
        f"| Technical health | {dash.technical_health_score:.2f} |",
        f"| Gain staging | {dash.gain_staging_score:.2f} |",
        f"| Balance | {dash.balance_score:.2f} |",
        f"| Tonal balance | {dash.tonal_balance_score:.2f} |",
        f"| Masking | {dash.masking_score:.2f} |",
        f"| Dynamics | {dash.dynamics_score:.2f} |",
        f"| Stereo/mono | {dash.stereo_mono_score:.2f} |",
        f"| Space clarity | {dash.space_clarity_score:.2f} |",
        f"| Reference match | {dash.reference_match_score:.2f} |",
        f"| Translation | {dash.translation_score:.2f} |",
        f"| Artifact risk | {dash.artifact_risk_score:.2f} |",
        "",
        "## Metrics",
        "",
        _metrics_table(plan),
        "",
        "## Detected Issues",
        "",
    ]
    if plan.issues:
        sections.extend(_render_issue(issue) + "\n" for issue in plan.issues)
    else:
        sections.append("No rule issues were triggered. Manual listening checks are still required.")

    sections.extend(
        [
            "",
            "## Reference Comparison",
            "",
            f"- Enabled: {ref.enabled}",
            f"- Loudness matched for analysis: {ref.loudness_matched}",
            f"- Spectral distance: {ref.spectral_distance}",
            f"- Band deltas: {ref.band_differences_db}",
            "",
            "## Next Manual Checks",
            "",
            "- Loudness-match A/B before judging any change.",
            "- Check vocal/lead readability in full mix, not solo.",
            "- Check mono fold-down and small-speaker translation.",
            "- Confirm low-end decisions on a system that reveals sub and one that does not.",
            "- Treat reference matching as tolerance guidance, not imitation.",
            "",
            "## Limitations",
            "",
            _line_items(limitations or ["No additional limitations reported."]),
        ]
    )
    return "\n".join(sections).rstrip() + "\n"


def write_markdown_report(plan: SuggestionPlan, path: str | Path) -> Path:
    output = Path(path).expanduser()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(render_markdown_report(plan), encoding="utf-8")
    return output
