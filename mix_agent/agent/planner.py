"""Suggestion planner for mix analysis snapshots."""

from __future__ import annotations

from mix_agent.models import MixAnalysis, SuggestionPlan
from mix_agent.rules import MixRuleEngine, build_quality_dashboard


def build_suggestion_plan(analysis: MixAnalysis) -> SuggestionPlan:
    """Run rules, rank issues and collect actions."""
    engine = MixRuleEngine.for_genre(analysis.context.genre)
    issues = engine.evaluate(analysis)
    actions = []
    seen = set()
    for issue in issues:
        for action in issue.actions:
            if action.id in seen:
                continue
            seen.add(action.id)
            actions.append(action)
    dashboard = build_quality_dashboard(analysis, issues)
    return SuggestionPlan(
        analysis=analysis,
        issues=issues,
        actions=actions,
        dashboard=dashboard,
    )
