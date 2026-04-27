"""Research-grade automatic mixing facade.

The package exposes a conservative, explainable layer for offline stem mixing
and for backend/live-console recommendation validation.  It intentionally
reuses the project's existing AutoFOH safety layer for real mixer writes.
"""

from .models import AnalysisContext, MixAction, MixAnalysis, QualityDashboard, RuleIssue

__all__ = [
    "AnalysisContext",
    "MixAction",
    "MixAnalysis",
    "QualityDashboard",
    "RuleIssue",
]
