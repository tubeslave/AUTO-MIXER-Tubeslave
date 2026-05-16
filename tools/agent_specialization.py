from __future__ import annotations

from dataclasses import dataclass
from typing import Any


READY_AGENT_STATUSES = {"idle", "available", "ready"}
BUSY_AGENT_STATUSES = {
    "active",
    "busy",
    "executing",
    "in_progress",
    "pending",
    "processing",
    "queued",
    "running",
    "started",
    "working",
}
ERROR_AGENT_STATUSES = {"crashed", "degraded", "error", "errored", "failed", "unhealthy"}
EXECUTION_ADAPTER_TYPE = "codex_local"
DEFAULT_SPECIALIZATION_KEY = "architecture_owner"


@dataclass(frozen=True)
class AgentSpecialization:
    key: str
    label: str
    fixed_agent_id: str
    name_fragments: tuple[str, ...]
    issue_terms: tuple[str, ...]
    boundary: str
    execution_role: bool = True
    adapter_type: str = EXECUTION_ADAPTER_TYPE
    fallback_priority: int = 100


@dataclass(frozen=True)
class IssueRoute:
    specialization: AgentSpecialization
    score: int
    matched_terms: tuple[str, ...]
    text: str


@dataclass(frozen=True)
class SpecialistSelection:
    specialization: AgentSpecialization
    agent: dict[str, Any] | None
    route: IssueRoute
    health: str
    matched_by: str
    reason: str

    @property
    def agent_id(self) -> str | None:
        return get_agent_id(self.agent or {})


SPECIALIZATIONS: tuple[AgentSpecialization, ...] = (
    AgentSpecialization(
        key="console_safety",
        label="OSC Console Safety Engineer",
        fixed_agent_id="c50c0ba1-9c94-4591-a9a4-37bcc7157cf6",
        name_fragments=("osc console safety engineer", "console safety"),
        issue_terms=(
            "wing",
            "wing rack",
            "osc",
            "console",
            "telemetry",
            "supervised write",
            "write gate",
            "dry-run",
            "dry run",
            "panic stop",
            "cooldown",
            "rate limit",
            "live safety",
            "rollback",
        ),
        boundary="Owns console-control safety, dry-run discipline, write gating, cooldowns, and panic stop paths.",
        fallback_priority=0,
    ),
    AgentSpecialization(
        key="qa_replay",
        label="Automixer QA Engineer",
        fixed_agent_id="dfdeb920-7662-48bf-9868-8283dad59af1",
        name_fragments=("automixer qa engineer", "qa and replay engineer", "qa engineer"),
        issue_terms=(
            "qa",
            "test",
            "tests",
            "pytest",
            "replay",
            "regression",
            "validate",
            "validation",
            "shadow mode",
            "prove",
            "verification",
        ),
        boundary="Owns replay validation, regression coverage, dry-run proofs, and safety verification evidence.",
        fallback_priority=10,
    ),
    AgentSpecialization(
        key="dsp_analyzer",
        label="DSP Analyzer Engineer",
        fixed_agent_id="b532752b-c709-465a-8a29-8e63d6d8573d",
        name_fragments=("dsp analyzer engineer",),
        issue_terms=(
            "dsp",
            "analyzer",
            "analysis",
            "lufs",
            "rms",
            "spectral",
            "masking",
            "transient",
            "eq",
            "gain staging",
            "band-limited",
        ),
        boundary="Owns analyzer math, spectral metrics, LUFS/RMS behavior, and instrument-aware DSP measurements.",
        fallback_priority=20,
    ),
    AgentSpecialization(
        key="ux_control_center",
        label="Control Center UX Engineer",
        fixed_agent_id="dcd63dfc-da3c-415a-83d8-a0b164c387fe",
        name_fragments=("control center ux engineer", "ui/ux engineer", "ux engineer"),
        issue_terms=(
            "ui",
            "ux",
            "dashboard",
            "frontend",
            "human supervision",
            "visibility",
            "indicator",
            "history",
            "tab",
            "cockpit",
        ),
        boundary="Owns operator-facing dashboards, safety visibility, correction history, and supervision UX.",
        fallback_priority=30,
    ),
    AgentSpecialization(
        key="research",
        label="Live Sound Research Engineer",
        fixed_agent_id="0e0e6523-8acd-4193-bc95-a509845218fe",
        name_fragments=("live sound research engineer", "researcher"),
        issue_terms=(
            "research",
            "paper",
            "literature",
            "benchmark",
            "method",
            "evaluation",
            "study",
            "survey",
        ),
        boundary="Owns research synthesis, external method review, and safe conversion of findings into engineering tasks.",
        fallback_priority=40,
    ),
    AgentSpecialization(
        key="architecture_owner",
        label="Automixer Chief Architect",
        fixed_agent_id="63750d40-cec7-43a2-a008-5b320d85aa75",
        name_fragments=("automixer chief architect", "architecture owner", "cto / architecture owner"),
        issue_terms=(
            "architecture",
            "orchestration",
            "routing",
            "specialization",
            "specialisation",
            "agent model",
            "execution model",
            "task tree",
            "refactor",
            "boundaries",
            "coordinator",
            "technical map",
            "plan",
        ),
        boundary="Owns architecture, execution boundaries, and safe refactors when a task is not a narrow domain-specialist fit.",
        fallback_priority=90,
    ),
    AgentSpecialization(
        key="director",
        label="Automixer Director",
        fixed_agent_id="6bd352cb-96ba-48c7-a173-3ddd2086c007",
        name_fragments=("automixer director", "task coordinator"),
        issue_terms=("delegate", "coordination", "triage"),
        boundary="Coordination-only role; should stay out of first-line execution routing except explicit management work.",
        execution_role=False,
        fallback_priority=999,
    ),
    AgentSpecialization(
        key="company_cto",
        label="CTO",
        fixed_agent_id="083e62de-fae5-4149-b7fd-3f988959fba8",
        name_fragments=("chief technology officer", "agent cto", " cto "),
        issue_terms=("staffing", "organization", "roadmap"),
        boundary="Management/oversight role; not part of the default execution rotation for project-domain work.",
        execution_role=False,
        fallback_priority=1000,
    ),
)

SPECIALIZATION_BY_KEY = {item.key: item for item in SPECIALIZATIONS}


def normalize_text(value: Any) -> str:
    return " ".join(str(value or "").strip().lower().split())


def clean_text(value: Any) -> str:
    return str(value or "").strip()


def get_agent_id(agent: dict[str, Any]) -> str | None:
    for key in ("id", "agentId", "agent_id", "uuid"):
        value = clean_text(agent.get(key))
        if value:
            return value
    return None


def get_agent_name(agent: dict[str, Any]) -> str:
    return clean_text(
        agent.get("name") or agent.get("agentName") or agent.get("title") or agent.get("role")
    ) or "unnamed agent"


def get_agent_status(agent: dict[str, Any]) -> str:
    return normalize_text(agent.get("status") or agent.get("state") or "unknown")


def get_agent_adapter(agent: dict[str, Any]) -> str:
    return normalize_text(agent.get("adapterType") or agent.get("adapter_type") or "")


def agent_health(agent: dict[str, Any]) -> str:
    status = get_agent_status(agent)
    adapter = get_agent_adapter(agent)
    if adapter and adapter != EXECUTION_ADAPTER_TYPE:
        return "unsupported_adapter"
    if status in READY_AGENT_STATUSES:
        return "ready"
    if status in BUSY_AGENT_STATUSES:
        return "busy"
    if status in ERROR_AGENT_STATUSES:
        return "error"
    return "unavailable"


def issue_text(issue: dict[str, Any]) -> str:
    parts = [
        clean_text(issue.get("identifier") or issue.get("key") or issue.get("issueKey")),
        clean_text(issue.get("title") or issue.get("name")),
        clean_text(issue.get("description") or issue.get("body") or issue.get("summary")),
    ]
    return normalize_text(" ".join(part for part in parts if part))


def route_issue(issue: dict[str, Any]) -> IssueRoute:
    text = issue_text(issue)
    ranked: list[IssueRoute] = []
    for specialization in SPECIALIZATIONS:
        if not specialization.execution_role:
            continue
        matched_terms = tuple(term for term in specialization.issue_terms if term in text)
        score = len(matched_terms)
        ranked.append(
            IssueRoute(
                specialization=specialization,
                score=score,
                matched_terms=matched_terms,
                text=text,
            )
        )

    ranked.sort(
        key=lambda route: (
            0 if route.score > 0 else 1,
            -route.score,
            route.specialization.fallback_priority,
            route.specialization.label.lower(),
        )
    )
    return ranked[0]


def resolve_specialist_agent(
    specialization: AgentSpecialization,
    agents: list[dict[str, Any]],
) -> tuple[dict[str, Any] | None, str]:
    for agent in agents:
        if get_agent_id(agent) == specialization.fixed_agent_id:
            return agent, "fixed_agent_id"

    fragments = specialization.name_fragments
    if not fragments:
        return None, "missing"

    matches = [
        agent
        for agent in agents
        if any(fragment in normalize_text(get_agent_name(agent)) for fragment in fragments)
    ]
    if len(matches) == 1:
        return matches[0], "name_fragment"
    return None, "missing"


def specialization_for_agent(agent: dict[str, Any]) -> AgentSpecialization | None:
    agent_id = get_agent_id(agent)
    agent_name = normalize_text(get_agent_name(agent))
    for specialization in SPECIALIZATIONS:
        if agent_id and agent_id == specialization.fixed_agent_id:
            return specialization
        if any(fragment in agent_name for fragment in specialization.name_fragments):
            return specialization
    return None


def select_specialist_for_issue(
    issue: dict[str, Any],
    agents: list[dict[str, Any]],
    *,
    allowlist_agent_ids: set[str] | None = None,
    require_ready: bool = True,
) -> SpecialistSelection:
    route = route_issue(issue)
    specialization = route.specialization
    agent, matched_by = resolve_specialist_agent(specialization, agents)
    health = agent_health(agent) if agent else "missing"
    allowed = True
    if agent and allowlist_agent_ids is not None:
        allowed = (get_agent_id(agent) or "") in allowlist_agent_ids
    if require_ready and health != "ready":
        agent = None
    if not allowed:
        agent = None
        health = "not_allowlisted"

    matched_terms = ", ".join(route.matched_terms) if route.matched_terms else "fallback"
    reason = (
        f"{specialization.label} selected via {matched_terms}; "
        f"boundary: {specialization.boundary}"
    )
    return SpecialistSelection(
        specialization=specialization,
        agent=agent,
        route=route,
        health=health,
        matched_by=matched_by,
        reason=reason,
    )


def specialization_health_summary(agents: list[dict[str, Any]]) -> dict[str, Any]:
    roles: list[dict[str, Any]] = []
    counts = {"ready": 0, "busy": 0, "error": 0, "unavailable": 0}

    for specialization in SPECIALIZATIONS:
        agent, matched_by = resolve_specialist_agent(specialization, agents)
        health = agent_health(agent) if agent else "unavailable"
        if health not in counts:
            counts["unavailable"] += 1
        else:
            counts[health] += 1
        roles.append(
            {
                "key": specialization.key,
                "label": specialization.label,
                "fixed_agent_id": specialization.fixed_agent_id,
                "execution_role": specialization.execution_role,
                "boundary": specialization.boundary,
                "agent_id": get_agent_id(agent or {}),
                "agent_name": get_agent_name(agent or {}) if agent else None,
                "agent_status": get_agent_status(agent or {}) if agent else None,
                "health": health,
                "matched_by": matched_by,
            }
        )

    return {
        "available": True,
        "counts": counts,
        "roles": roles,
    }
