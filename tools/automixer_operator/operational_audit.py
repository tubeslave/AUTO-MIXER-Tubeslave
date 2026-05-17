from __future__ import annotations

import json
import re
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from tools.gpt_control_bridge import server as bridge
from tools.automixer_operator import recovery_governor


OPEN_ISSUE_STATUSES = {"backlog", "todo", "in_progress", "in_review", "blocked"}
STALE_ISSUE_STATUSES = {"blocked", "in_progress", "in_review"}
UNHEALTHY_AGENT_STATUSES = {"crashed", "degraded", "error", "errored", "failed", "unhealthy"}
ACTIVE_RUN_STATUSES = {"active", "busy", "in_progress", "pending", "processing", "queued", "running", "started", "working"}
RECOVERY_TITLE_RE = re.compile(r"\brecover\s+stalled\s+issue\s+([A-Z]+-\d+)\b", re.IGNORECASE)

DEFAULT_STALE_HOURS = 24.0
DEFAULT_ORPHAN_HOURS = 24.0
DEFAULT_HEARTBEAT_STALE_HOURS = 24.0


@dataclass(frozen=True)
class AuditThresholds:
    stale_hours: float = DEFAULT_STALE_HOURS
    orphan_hours: float = DEFAULT_ORPHAN_HOURS
    heartbeat_stale_hours: float = DEFAULT_HEARTBEAT_STALE_HOURS

    def as_dict(self) -> dict[str, float]:
        return {
            "stale_hours": self.stale_hours,
            "orphan_hours": self.orphan_hours,
            "heartbeat_stale_hours": self.heartbeat_stale_hours,
        }


@dataclass
class OperationalAuditResult:
    generated_at: str
    company_id: str
    issue_total: int
    agent_total: int
    active_run_total: int
    issue_status_counts: dict[str, int]
    agent_status_counts: dict[str, int]
    thresholds: AuditThresholds
    stale_issues: list[dict[str, Any]]
    orphaned_issues: list[dict[str, Any]]
    blocked_chains: list[dict[str, Any]]
    blocked_cycles: list[dict[str, Any]]
    unhealthy_agents: list[dict[str, Any]]
    duplicate_recovery_issues: list[dict[str, Any]]
    recovery_governor: dict[str, Any]
    run_source: str
    read_only: bool = True
    report_paths: dict[str, str] = field(default_factory=dict)

    def summary_counts(self) -> dict[str, int]:
        return {
            "stale_issues": len(self.stale_issues),
            "orphaned_issues": len(self.orphaned_issues),
            "blocked_chains": len(self.blocked_chains),
            "blocked_cycles": len(self.blocked_cycles),
            "unhealthy_agents": len(self.unhealthy_agents),
            "duplicate_recovery_issues": len(self.duplicate_recovery_issues),
            "recovery_governor_allow": int(self.recovery_governor.get("summary_counts", {}).get("allow", 0)),
            "recovery_governor_suppress": int(self.recovery_governor.get("summary_counts", {}).get("suppress", 0)),
            "recovery_governor_manual_review": int(self.recovery_governor.get("summary_counts", {}).get("manual_review", 0)),
        }

    def as_dict(self) -> dict[str, Any]:
        return {
            "generated_at": self.generated_at,
            "company_id": self.company_id,
            "read_only": self.read_only,
            "issue_total": self.issue_total,
            "agent_total": self.agent_total,
            "active_run_total": self.active_run_total,
            "run_source": self.run_source,
            "issue_status_counts": self.issue_status_counts,
            "agent_status_counts": self.agent_status_counts,
            "thresholds": self.thresholds.as_dict(),
            "summary_counts": self.summary_counts(),
            "stale_issues": self.stale_issues,
            "orphaned_issues": self.orphaned_issues,
            "blocked_chains": self.blocked_chains,
            "blocked_cycles": self.blocked_cycles,
            "unhealthy_agents": self.unhealthy_agents,
            "duplicate_recovery_issues": self.duplicate_recovery_issues,
            "recovery_governor": self.recovery_governor,
            "priority_actions": build_priority_actions(self),
            "report_paths": self.report_paths,
        }


def create_operational_audit(
    *,
    project_root: Path,
    api_url: str,
    company_id: str,
    timeout_seconds: float = 3.0,
    thresholds: AuditThresholds | None = None,
    now: datetime | None = None,
    client: Any | None = None,
) -> OperationalAuditResult:
    thresholds = thresholds or AuditThresholds()
    now = ensure_aware(now or datetime.now(timezone.utc))
    paperclip = client or bridge.PaperclipClient(api_url, timeout_seconds)

    issues = load_required_items(
        paperclip,
        f"/api/companies/{bridge.quote(company_id)}/issues",
        params={"status": "backlog,todo,in_progress,in_review,blocked", "limit": 500},
        candidate_keys=["issues", "data", "items", "results"],
    )
    agents = load_required_items(
        paperclip,
        f"/api/companies/{bridge.quote(company_id)}/agents",
        candidate_keys=["agents", "data", "items", "results"],
    )
    runs, run_source = load_active_runs(paperclip, company_id)

    agent_by_id = {agent_id(agent): agent for agent in agents if agent_id(agent)}
    issue_by_id = {issue_id(issue): issue for issue in issues if issue_id(issue)}

    stale_issues = detect_stale_issues(issues, now, thresholds)
    orphaned_issues = detect_orphaned_issues(issues, agent_by_id, now, thresholds)
    blocked_chains, blocked_cycles = detect_blocked_dependency_chains(issues, issue_by_id)
    unhealthy_agents = detect_unhealthy_agents(agents, now, thresholds)
    duplicate_recovery_issues = detect_duplicate_recovery_issues(issues)
    governed_recovery = recovery_governor.evaluate_recovery_governor(
        issues,
        stale_issues=stale_issues,
        orphaned_issues=orphaned_issues,
        now=now,
    )

    result = OperationalAuditResult(
        generated_at=now.astimezone(timezone.utc).isoformat().replace("+00:00", "Z"),
        company_id=company_id,
        issue_total=len(issues),
        agent_total=len(agents),
        active_run_total=len(runs),
        issue_status_counts=count_by_status(issues),
        agent_status_counts=count_by_status(agents),
        thresholds=thresholds,
        stale_issues=stale_issues,
        orphaned_issues=orphaned_issues,
        blocked_chains=blocked_chains,
        blocked_cycles=blocked_cycles,
        unhealthy_agents=unhealthy_agents,
        duplicate_recovery_issues=duplicate_recovery_issues,
        recovery_governor=governed_recovery,
        run_source=run_source,
    )
    result.report_paths = write_report_files(project_root, result, now)
    return result


def load_required_items(
    client: Any,
    path: str,
    *,
    params: dict[str, Any] | None = None,
    candidate_keys: list[str],
) -> list[dict[str, Any]]:
    payload, error = bridge.safe_get(client, path, params=params)
    if error:
        raise RuntimeError(f"{path}: {error}")
    return bridge.extract_items(payload, candidate_keys)


def load_active_runs(client: Any, company_id: str) -> tuple[list[dict[str, Any]], str]:
    primary_path = f"/api/companies/{bridge.quote(company_id)}/live-runs"
    payload, error = bridge.safe_get(client, primary_path, params={"limit": 200, "minCount": 0})
    if error is None:
        return filter_active_runs(
            bridge.extract_items(payload, ["liveRuns", "heartbeatRuns", "runs", "data", "items", "results"])
        ), "live-runs"

    fallback_path = f"/api/companies/{bridge.quote(company_id)}/heartbeat-runs"
    payload, fallback_error = bridge.safe_get(client, fallback_path, params={"limit": 200})
    if fallback_error:
        raise RuntimeError(f"{primary_path}: {error}; {fallback_path}: {fallback_error}")
    return filter_active_runs(
        bridge.extract_items(payload, ["heartbeatRuns", "liveRuns", "runs", "data", "items", "results"])
    ), "heartbeat-runs-fallback"


def filter_active_runs(runs: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [run for run in runs if normalize_status(run.get("status") or run.get("state")) in ACTIVE_RUN_STATUSES]


def detect_stale_issues(
    issues: list[dict[str, Any]],
    now: datetime,
    thresholds: AuditThresholds,
) -> list[dict[str, Any]]:
    findings: list[dict[str, Any]] = []
    for issue in issues:
        status = normalize_status(issue.get("status") or issue.get("state"))
        if status not in STALE_ISSUE_STATUSES:
            continue
        if issue_has_active_run(issue):
            continue
        age_hours = issue_age_hours(issue, now)
        if age_hours is None or age_hours < thresholds.stale_hours:
            continue
        reason = f"no recorded activity for {format_hours(age_hours)}"
        if status == "blocked" and not blocker_ids(issue):
            reason += "; blocked without explicit blockers"
        findings.append(issue_summary(issue, now, reason=reason, age_hours=age_hours))
    return sorted(findings, key=lambda item: item["age_hours"], reverse=True)


def detect_orphaned_issues(
    issues: list[dict[str, Any]],
    agent_by_id: dict[str, dict[str, Any]],
    now: datetime,
    thresholds: AuditThresholds,
) -> list[dict[str, Any]]:
    findings: list[dict[str, Any]] = []
    seen_keys: set[tuple[str, str]] = set()
    for issue in issues:
        status = normalize_status(issue.get("status") or issue.get("state"))
        if status not in OPEN_ISSUE_STATUSES:
            continue
        age_hours = issue_age_hours(issue, now)
        assignee = clean_missing(issue.get("assigneeAgentId"))
        if status in {"backlog", "todo"} and not assignee and age_hours is not None and age_hours >= thresholds.orphan_hours:
            reason = f"unassigned {status} issue for {format_hours(age_hours)}"
            findings.append(issue_summary(issue, now, reason=reason, age_hours=age_hours))
            seen_keys.add((issue_id(issue) or "", reason))
            continue
        if not assignee:
            continue
        agent = agent_by_id.get(assignee)
        if agent is None:
            reason = "assigned agent record is missing"
        else:
            agent_status = normalize_status(agent.get("status") or agent.get("state"))
            if agent_status in UNHEALTHY_AGENT_STATUSES:
                reason = f"assigned agent is unhealthy ({agent_status})"
            else:
                continue
        key = (issue_id(issue) or "", reason)
        if key in seen_keys:
            continue
        findings.append(issue_summary(issue, now, reason=reason, age_hours=age_hours))
        seen_keys.add(key)
    return sorted(findings, key=lambda item: (status_priority(item["priority"]), -item["age_hours"]))


def detect_blocked_dependency_chains(
    issues: list[dict[str, Any]],
    issue_by_id: dict[str, dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    chains: list[dict[str, Any]] = []
    cycles: list[dict[str, Any]] = []
    seen_chains: set[tuple[str, ...]] = set()
    seen_cycles: set[tuple[str, ...]] = set()
    for issue in issues:
        if normalize_status(issue.get("status") or issue.get("state")) != "blocked":
            continue
        path, cycle = follow_blocker_chain(issue, issue_by_id)
        if cycle:
            cycle_key = canonical_cycle_key([node["id"] for node in cycle if node.get("id")])
            if cycle_key not in seen_cycles:
                seen_cycles.add(cycle_key)
                cycles.append(
                    {
                        "path": cycle,
                        "path_identifiers": [node["identifier"] for node in cycle],
                        "reason": "dependency cycle",
                    }
                )
        elif len(path) >= 2:
            chain_key = tuple(node["id"] for node in path)
            if chain_key not in seen_chains:
                seen_chains.add(chain_key)
                terminal = path[-1]
                chains.append(
                    {
                        "path": path,
                        "path_identifiers": [node["identifier"] for node in path],
                        "depth": len(path) - 1,
                        "terminal_status": terminal["status"],
                        "terminal_reason": terminal.get("chain_reason") or "blocked dependency chain",
                    }
                )
    chains.sort(key=lambda item: (-item["depth"], " -> ".join(item["path_identifiers"])))
    cycles.sort(key=lambda item: " -> ".join(item["path_identifiers"]))
    return chains, cycles


def follow_blocker_chain(
    start_issue: dict[str, Any],
    issue_by_id: dict[str, dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]] | None]:
    path: list[dict[str, Any]] = []
    seen_indexes: dict[str, int] = {}
    current = start_issue
    while current:
        current_id = issue_id(current)
        if not current_id:
            break
        node = chain_node(current)
        path.append(node)
        seen_indexes[current_id] = len(path) - 1
        next_blocker = preferred_blocker(current, issue_by_id)
        if not next_blocker:
            break
        next_id = clean_missing(next_blocker.get("id"))
        if not next_id:
            break
        if next_id in seen_indexes:
            cycle_start = seen_indexes[next_id]
            cycle_nodes = path[cycle_start:] + [chain_node(issue_by_id.get(next_id, next_blocker))]
            return path, cycle_nodes
        blocker_issue = issue_by_id.get(next_id)
        if blocker_issue is None:
            path.append(chain_node(next_blocker, chain_reason="blocker record not loaded"))
            break
        current = blocker_issue
        if normalize_status(current.get("status") or current.get("state")) not in OPEN_ISSUE_STATUSES:
            path.append(chain_node(current, chain_reason="terminal blocker"))
            break
    return path, None


def preferred_blocker(
    issue: dict[str, Any],
    issue_by_id: dict[str, dict[str, Any]],
) -> dict[str, Any] | None:
    blockers = blocker_refs(issue)
    if not blockers:
        return None
    decorated: list[tuple[tuple[int, int, str], dict[str, Any]]] = []
    for blocker in blockers:
        blocker_full = issue_by_id.get(clean_missing(blocker.get("id")) or "") or blocker
        status = normalize_status(blocker_full.get("status") or blocker_full.get("state"))
        rank = {
            "blocked": 0,
            "in_progress": 1,
            "in_review": 2,
            "todo": 3,
            "backlog": 4,
        }.get(status, 5)
        decorated.append(((rank, -issue_number(blocker_full), issue_identifier(blocker_full) or ""), blocker_full))
    decorated.sort(key=lambda item: item[0])
    return decorated[0][1]


def detect_unhealthy_agents(
    agents: list[dict[str, Any]],
    now: datetime,
    thresholds: AuditThresholds,
) -> list[dict[str, Any]]:
    findings: list[dict[str, Any]] = []
    for agent in agents:
        status = normalize_status(agent.get("status") or agent.get("state"))
        heartbeat_at = parse_timestamp(agent.get("lastHeartbeatAt"))
        heartbeat_age = hours_since(heartbeat_at, now)
        reason = None
        if status in UNHEALTHY_AGENT_STATUSES:
            reason = f"agent status is {status}"
        elif heartbeat_age is not None and heartbeat_age >= thresholds.heartbeat_stale_hours:
            reason = f"last heartbeat is {format_hours(heartbeat_age)} old"
        if reason is None:
            continue
        findings.append(
            {
                "id": agent_id(agent),
                "name": clean_missing(agent.get("name")) or clean_missing(agent.get("title")) or "unnamed agent",
                "status": status or "unknown",
                "adapter_type": clean_missing(agent.get("adapterType")) or "unknown",
                "last_heartbeat_at": heartbeat_at.isoformat().replace("+00:00", "Z") if heartbeat_at else None,
                "heartbeat_age_hours": round(heartbeat_age, 1) if heartbeat_age is not None else None,
                "reason": reason,
            }
        )
    return sorted(findings, key=lambda item: (status_priority(item["status"]), -(item["heartbeat_age_hours"] or 0.0)))


def detect_duplicate_recovery_issues(issues: list[dict[str, Any]]) -> list[dict[str, Any]]:
    findings: list[dict[str, Any]] = []
    recovery_groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    fingerprint_groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for issue in issues:
        status = normalize_status(issue.get("status") or issue.get("state"))
        if status not in OPEN_ISSUE_STATUSES:
            continue
        match = RECOVERY_TITLE_RE.search(clean_missing(issue.get("title")) or "")
        if match:
            recovery_groups[match.group(1).upper()].append(issue)
        action = issue.get("activeRecoveryAction")
        if isinstance(action, dict):
            fingerprint = clean_missing(action.get("fingerprint"))
            action_status = normalize_status(action.get("status") or action.get("state"))
            if fingerprint and action_status in {"active", "open", "running"}:
                fingerprint_groups[fingerprint].append(issue)

    for target, members in recovery_groups.items():
        if len(members) < 2:
            continue
        findings.append(
            {
                "kind": "recovery_issue_title",
                "target_identifier": target,
                "issue_identifiers": [issue_identifier(issue) or issue_id(issue) or "unknown" for issue in members],
                "count": len(members),
                "reason": f"multiple open recovery issues target {target}",
            }
        )

    for fingerprint, members in fingerprint_groups.items():
        if len(members) < 2:
            continue
        findings.append(
            {
                "kind": "active_recovery_fingerprint",
                "fingerprint": fingerprint,
                "issue_identifiers": [issue_identifier(issue) or issue_id(issue) or "unknown" for issue in members],
                "count": len(members),
                "reason": "multiple active recovery actions share the same fingerprint",
            }
        )
    return sorted(findings, key=lambda item: (-item["count"], item["reason"]))


def write_report_files(project_root: Path, result: OperationalAuditResult, now: datetime) -> dict[str, str]:
    reports_dir = project_root / ".paperclip" / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)
    stamp = ensure_aware(now).astimezone(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    timestamped_md = reports_dir / f"operational_audit_{stamp}.md"
    timestamped_json = reports_dir / f"operational_audit_{stamp}.json"
    latest_md = reports_dir / "operational_audit_latest.md"
    latest_json = reports_dir / "operational_audit_latest.json"
    paths = {
        "markdown": str(timestamped_md),
        "json": str(timestamped_json),
        "latest_markdown": str(latest_md),
        "latest_json": str(latest_json),
    }
    result.report_paths = paths
    markdown = render_markdown(result)
    payload = json.dumps(result.as_dict(), ensure_ascii=False, indent=2, sort_keys=True)
    timestamped_md.write_text(markdown, encoding="utf-8")
    timestamped_json.write_text(payload, encoding="utf-8")
    latest_md.write_text(markdown, encoding="utf-8")
    latest_json.write_text(payload, encoding="utf-8")
    return paths


def render_markdown(result: OperationalAuditResult) -> str:
    lines = [
        "# Daily Operational Audit",
        "",
        "Read-only report. No WING/OSC writes, Paperclip dispatches, or issue mutations were performed while generating this audit.",
        "",
        "## Summary",
        f"- generated_at: {result.generated_at}",
        f"- company_id: {result.company_id}",
        f"- issue_total: {result.issue_total}",
        f"- agent_total: {result.agent_total}",
        f"- active_run_total: {result.active_run_total}",
        f"- run_source: {result.run_source}",
        f"- issue_status_counts: {format_counts(result.issue_status_counts)}",
        f"- agent_status_counts: {format_counts(result.agent_status_counts)}",
        f"- stale_threshold_hours: {result.thresholds.stale_hours:g}",
        f"- orphan_threshold_hours: {result.thresholds.orphan_hours:g}",
        f"- heartbeat_stale_threshold_hours: {result.thresholds.heartbeat_stale_hours:g}",
        f"- stale_issues: {len(result.stale_issues)}",
        f"- orphaned_issues: {len(result.orphaned_issues)}",
        f"- blocked_chains: {len(result.blocked_chains)}",
        f"- blocked_cycles: {len(result.blocked_cycles)}",
        f"- unhealthy_agents: {len(result.unhealthy_agents)}",
        f"- duplicate_recovery_issues: {len(result.duplicate_recovery_issues)}",
        f"- recovery_governor_allow: {result.recovery_governor.get('summary_counts', {}).get('allow', 0)}",
        f"- recovery_governor_suppress: {result.recovery_governor.get('summary_counts', {}).get('suppress', 0)}",
        f"- recovery_governor_manual_review: {result.recovery_governor.get('summary_counts', {}).get('manual_review', 0)}",
        "",
    ]
    lines.extend(render_priority_actions_section("Priority Actions", build_priority_actions(result), max_items=8))
    lines.extend(
        [
        "## Report Files",
        f"- markdown: {result.report_paths.get('markdown') or 'missing'}",
        f"- json: {result.report_paths.get('json') or 'missing'}",
        f"- latest_markdown: {result.report_paths.get('latest_markdown') or 'missing'}",
        f"- latest_json: {result.report_paths.get('latest_json') or 'missing'}",
        "",
        ]
    )
    lines.extend(render_issue_section("Stale Issues", result.stale_issues, max_items=15))
    lines.extend(render_issue_section("Orphaned Issues", result.orphaned_issues, max_items=15))
    lines.extend(render_chain_section("Blocked Dependency Chains", result.blocked_chains, max_items=10))
    lines.extend(render_chain_section("Blocked Dependency Cycles", result.blocked_cycles, max_items=10))
    lines.extend(render_agent_section("Unhealthy Agents", result.unhealthy_agents, max_items=15))
    lines.extend(render_duplicate_section("Duplicate Recovery Issues", result.duplicate_recovery_issues, max_items=15))
    lines.extend(render_recovery_governor_section("Recovery Governor", result.recovery_governor, max_items=15))
    return "\n".join(lines).rstrip() + "\n"


def build_priority_actions(result: OperationalAuditResult) -> list[dict[str, Any]]:
    actions: list[dict[str, Any]] = []
    blocked_count = int(result.issue_status_counts.get("blocked", 0))
    todo_count = int(result.issue_status_counts.get("todo", 0))
    in_progress_count = int(result.issue_status_counts.get("in_progress", 0))
    recovery_counts = result.recovery_governor.get("summary_counts", {}) if isinstance(result.recovery_governor, dict) else {}

    if result.unhealthy_agents:
        first = result.unhealthy_agents[0]
        actions.append(
            priority_action(
                "repair_unhealthy_agents",
                "critical",
                (
                    f"{len(result.unhealthy_agents)} agent(s) are unhealthy; "
                    "blocked or assigned work may not progress until they are reviewed."
                ),
                first.get("name"),
                first.get("reason"),
                "python3 tools/automixer_operator/operator.py agent-health",
            )
        )

    if todo_count == 0 and in_progress_count == 0 and blocked_count > 0:
        actions.append(
            priority_action(
                "triage_blocked_queue",
                "critical",
                f"{blocked_count} issue(s) are blocked while no todo or in-progress work is available.",
                "blocked_queue",
                "automation has no safe preview target",
                "review blockers before creating or dispatching new work",
            )
        )

    if result.orphaned_issues:
        first = result.orphaned_issues[0]
        actions.append(
            priority_action(
                "review_orphaned_assignments",
                "high",
                f"{len(result.orphaned_issues)} open issue(s) are unassigned, assigned to a missing agent, or assigned to an unhealthy agent.",
                first.get("identifier"),
                first.get("reason"),
                "manual Paperclip review",
            )
        )

    if result.blocked_cycles:
        first = result.blocked_cycles[0]
        actions.append(
            priority_action(
                "break_blocked_dependency_cycles",
                "high",
                f"{len(result.blocked_cycles)} blocked dependency cycle(s) were found.",
                " -> ".join(first.get("path_identifiers") or []),
                first.get("reason"),
                "manual dependency review",
            )
        )

    if result.blocked_chains:
        first = result.blocked_chains[0]
        actions.append(
            priority_action(
                "resolve_deep_blocked_chains",
                "medium",
                f"{len(result.blocked_chains)} blocked dependency chain(s) need review.",
                " -> ".join(first.get("path_identifiers") or []),
                first.get("terminal_reason") or first.get("reason"),
                "manual dependency review",
            )
        )

    if result.duplicate_recovery_issues:
        first = result.duplicate_recovery_issues[0]
        actions.append(
            priority_action(
                "dedupe_recovery_issues",
                "medium",
                f"{len(result.duplicate_recovery_issues)} duplicate recovery issue group(s) were found.",
                first.get("target_identifier") or ", ".join(first.get("issue_identifiers") or []),
                first.get("reason"),
                "manual recovery review",
            )
        )

    manual_review = int(recovery_counts.get("manual_review", 0) or 0)
    suppressed = int(recovery_counts.get("suppress", 0) or 0)
    if manual_review or suppressed:
        actions.append(
            priority_action(
                "review_recovery_governor_blocks",
                "medium",
                f"Recovery governor has {manual_review} manual-review and {suppressed} suppressed target(s).",
                "recovery_governor",
                "recursive or duplicate recovery paths need operator review",
                "operator_read_only_review",
            )
        )

    if not actions:
        actions.append(
            priority_action(
                "continue_safe_dry_run_flow",
                "low",
                "No audit blockers were found; continue normal status -> preview -> approval-template flow.",
                "operator_cycle",
                "audit clear",
                "python3 tools/automixer_operator/operator.py cycle",
            )
        )

    return actions


def priority_action(
    key: str,
    severity: str,
    reason: str,
    target: Any,
    evidence: Any,
    next_step: str,
) -> dict[str, Any]:
    return {
        "key": key,
        "severity": severity,
        "reason": reason,
        "target": target or "none",
        "evidence": evidence or "none",
        "next_step": next_step,
        "write_permitted": False,
    }


def render_priority_actions_section(
    title: str,
    actions: list[dict[str, Any]],
    *,
    max_items: int,
) -> list[str]:
    lines = [f"## {title}"]
    if not actions:
        return lines + ["- none", ""]
    for item in actions[:max_items]:
        lines.append(f"- {item.get('severity')}: {item.get('key')}")
        lines.append(f"  reason: {item.get('reason')}")
        lines.append(f"  target: {item.get('target')}")
        lines.append(f"  evidence: {item.get('evidence')}")
        lines.append(f"  next_step: {item.get('next_step')}")
        lines.append(f"  write_permitted: {item.get('write_permitted')}")
    lines.append("")
    return lines


def render_issue_section(title: str, items: list[dict[str, Any]], *, max_items: int) -> list[str]:
    lines = [f"## {title}"]
    if not items:
        return lines + ["- none", ""]
    for item in items[:max_items]:
        identifier = item.get("identifier") or item.get("id") or "unknown"
        lines.append(
            "- "
            + f"{identifier} [{item.get('status')}] "
            + f"{item.get('title') or 'untitled'} "
            + f"(priority={item.get('priority') or 'unknown'}, age_hours={item.get('age_hours')}, "
            + f"assignee={item.get('assignee_agent_id') or 'none'})"
        )
        lines.append(f"  reason: {item.get('reason')}")
    lines.append("")
    return lines


def render_chain_section(title: str, items: list[dict[str, Any]], *, max_items: int) -> list[str]:
    lines = [f"## {title}"]
    if not items:
        return lines + ["- none", ""]
    for item in items[:max_items]:
        lines.append(f"- {' -> '.join(item.get('path_identifiers') or [])}")
        reason = item.get("terminal_reason") or item.get("reason") or "dependency chain"
        lines.append(f"  reason: {reason}")
    lines.append("")
    return lines


def render_agent_section(title: str, items: list[dict[str, Any]], *, max_items: int) -> list[str]:
    lines = [f"## {title}"]
    if not items:
        return lines + ["- none", ""]
    for item in items[:max_items]:
        lines.append(
            "- "
            + f"{item.get('name')} [{item.get('status')}] "
            + f"(adapter={item.get('adapter_type')}, heartbeat_age_hours={item.get('heartbeat_age_hours')})"
        )
        lines.append(f"  reason: {item.get('reason')}")
    lines.append("")
    return lines


def render_duplicate_section(title: str, items: list[dict[str, Any]], *, max_items: int) -> list[str]:
    lines = [f"## {title}"]
    if not items:
        return lines + ["- none", ""]
    for item in items[:max_items]:
        identifiers = ", ".join(item.get("issue_identifiers") or [])
        lines.append(f"- {item.get('kind')}: {identifiers}")
        lines.append(f"  reason: {item.get('reason')}")
    lines.append("")
    return lines


def render_recovery_governor_section(title: str, report: dict[str, Any], *, max_items: int) -> list[str]:
    lines = [f"## {title}"]
    summary = report.get("summary_counts", {}) if isinstance(report, dict) else {}
    thresholds = report.get("thresholds", {}) if isinstance(report, dict) else {}
    targets = report.get("targets", []) if isinstance(report, dict) else []
    lines.append(
        "- "
        + f"allow={summary.get('allow', 0)}, suppress={summary.get('suppress', 0)}, "
        + f"manual_review={summary.get('manual_review', 0)}, "
        + f"cooldown_hours={thresholds.get('cooldown_hours', 'unknown')}, "
        + f"max_open_attempts_per_target={thresholds.get('max_open_attempts_per_target', 'unknown')}"
    )
    if not targets:
        return lines + ["- none", ""]
    for item in targets[:max_items]:
        chain = " -> ".join(item.get("recursive_chain") or [])
        lines.append(
            "- "
            + f"{item.get('target_identifier')} "
            + f"(decision={item.get('decision')}, open_recovery_issues={item.get('open_recovery_issue_count')}, "
            + f"cooldown_remaining_hours={item.get('cooldown_remaining_hours')})"
        )
        if chain:
            lines.append(f"  chain: {chain}")
        for reason in item.get("reasons") or []:
            lines.append(f"  reason: {reason}")
    lines.append("")
    return lines


def issue_summary(
    issue: dict[str, Any],
    now: datetime,
    *,
    reason: str,
    age_hours: float | None = None,
) -> dict[str, Any]:
    computed_age = age_hours if age_hours is not None else issue_age_hours(issue, now)
    return {
        "id": issue_id(issue),
        "identifier": issue_identifier(issue) or issue_id(issue) or "unknown",
        "title": clean_missing(issue.get("title")) or "untitled",
        "status": normalize_status(issue.get("status") or issue.get("state")) or "unknown",
        "priority": normalize_status(issue.get("priority")) or "unknown",
        "assignee_agent_id": clean_missing(issue.get("assigneeAgentId")),
        "last_activity_at": normalize_timestamp(issue_activity_time(issue)),
        "age_hours": round(computed_age, 1) if computed_age is not None else None,
        "reason": reason,
    }


def chain_node(issue: dict[str, Any], chain_reason: str | None = None) -> dict[str, Any]:
    return {
        "id": issue_id(issue),
        "identifier": issue_identifier(issue) or issue_id(issue) or "unknown",
        "title": clean_missing(issue.get("title")) or "untitled",
        "status": normalize_status(issue.get("status") or issue.get("state")) or "unknown",
        "chain_reason": chain_reason,
    }


def blocker_refs(issue: dict[str, Any]) -> list[dict[str, Any]]:
    raw = issue.get("blockedBy")
    if not isinstance(raw, list):
        return []
    return [item for item in raw if isinstance(item, dict) and clean_missing(item.get("id"))]


def blocker_ids(issue: dict[str, Any]) -> list[str]:
    return [clean_missing(item.get("id")) or "" for item in blocker_refs(issue)]


def issue_activity_time(issue: dict[str, Any]) -> datetime | None:
    for key in ("lastActivityAt", "updatedAt", "startedAt", "createdAt"):
        value = parse_timestamp(issue.get(key))
        if value is not None:
            return value
    return None


def issue_age_hours(issue: dict[str, Any], now: datetime) -> float | None:
    return hours_since(issue_activity_time(issue), now)


def parse_timestamp(value: Any) -> datetime | None:
    if not value:
        return None
    text = str(value).strip()
    if not text:
        return None
    try:
        parsed = datetime.fromisoformat(text.replace("Z", "+00:00"))
    except ValueError:
        return None
    return ensure_aware(parsed)


def normalize_timestamp(value: datetime | None) -> str | None:
    if value is None:
        return None
    return ensure_aware(value).astimezone(timezone.utc).isoformat().replace("+00:00", "Z")


def hours_since(value: datetime | None, now: datetime) -> float | None:
    if value is None:
        return None
    delta = ensure_aware(now) - ensure_aware(value)
    return max(delta.total_seconds() / 3600.0, 0.0)


def ensure_aware(value: datetime) -> datetime:
    return value if value.tzinfo is not None else value.replace(tzinfo=timezone.utc)


def count_by_status(items: list[dict[str, Any]]) -> dict[str, int]:
    counts: dict[str, int] = defaultdict(int)
    for item in items:
        counts[normalize_status(item.get("status") or item.get("state")) or "unknown"] += 1
    return dict(sorted(counts.items()))


def format_counts(counts: dict[str, int]) -> str:
    return ", ".join(f"{key}={value}" for key, value in sorted(counts.items())) or "none"


def issue_id(issue: dict[str, Any]) -> str | None:
    return clean_missing(issue.get("id")) or clean_missing(issue.get("issueId")) or clean_missing(issue.get("uuid"))


def issue_identifier(issue: dict[str, Any]) -> str | None:
    identifier = clean_missing(issue.get("identifier")) or clean_missing(issue.get("key"))
    if identifier:
        return identifier
    number = issue_number(issue)
    if number <= 0:
        return None
    prefix = clean_missing(issue.get("issuePrefix")) or clean_missing(issue.get("prefix")) or "TUB"
    return f"{prefix}-{number}"


def issue_number(issue: dict[str, Any]) -> int:
    for key in ("issueNumber", "issue_number", "number"):
        value = parse_intish(issue.get(key))
        if value > 0:
            return value
    text = clean_missing(issue.get("identifier")) or clean_missing(issue.get("title")) or ""
    match = re.search(r"\b[A-Z]+-(\d+)\b", text)
    return parse_intish(match.group(1)) if match else 0


def agent_id(agent: dict[str, Any]) -> str | None:
    return clean_missing(agent.get("id")) or clean_missing(agent.get("agentId")) or clean_missing(agent.get("uuid"))


def issue_has_active_run(issue: dict[str, Any]) -> bool:
    for key in ("activeRunId", "active_run_id", "currentRunId", "current_run_id"):
        if clean_missing(issue.get(key)):
            return True
    for key in ("activeRun", "active_run", "currentRun", "current_run"):
        value = issue.get(key)
        if isinstance(value, dict):
            status = normalize_status(value.get("status") or value.get("state"))
            return status not in bridge.TERMINAL_RUN_STATUSES
        if isinstance(value, list):
            return any(
                isinstance(item, dict)
                and normalize_status(item.get("status") or item.get("state")) not in bridge.TERMINAL_RUN_STATUSES
                for item in value
            )
    return False


def status_priority(value: str | None) -> int:
    return {
        "critical": 0,
        "error": 0,
        "failed": 0,
        "high": 1,
        "blocked": 1,
        "in_progress": 2,
        "in_review": 2,
        "todo": 3,
        "medium": 4,
        "low": 5,
    }.get(normalize_status(value), 6)


def normalize_status(value: Any) -> str:
    return str(value or "").strip().lower().replace("-", "_").replace(" ", "_")


def parse_intish(value: Any) -> int:
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        return 0
    return parsed if parsed > 0 else 0


def format_hours(value: float) -> str:
    return f"{round(value, 1)}h"


def clean_missing(value: Any) -> str | None:
    text = str(value or "").strip()
    return None if text.lower() in {"", "none", "null", "missing", "unavailable"} else text


def canonical_cycle_key(ids: list[str]) -> tuple[str, ...]:
    if not ids:
        return ()
    base = ids[:-1] if len(ids) > 1 and ids[0] == ids[-1] else ids[:]
    if not base:
        return ()
    rotations = [tuple(base[index:] + base[:index]) for index in range(len(base))]
    reversed_base = list(reversed(base))
    rotations.extend(
        tuple(reversed_base[index:] + reversed_base[:index]) for index in range(len(reversed_base))
    )
    return min(rotations)
