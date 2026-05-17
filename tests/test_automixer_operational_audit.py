from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

from tools.automixer_operator import operational_audit


class FakeClient:
    def __init__(self, routes):
        self.routes = routes
        self.calls: list[tuple[str, dict | None]] = []

    def get(self, path, params=None):
        self.calls.append((path, params))
        if path not in self.routes:
            raise OSError(f"unexpected route: {path}")
        return self.routes[path]


def test_create_operational_audit_detects_required_conditions(tmp_path: Path):
    now = datetime(2026, 5, 17, 9, 0, tzinfo=timezone.utc)
    client = FakeClient(
        {
            "/api/companies/company-1/issues": [
                issue(
                    "stale-1",
                    "TUB-400",
                    "Stale in progress",
                    "in_progress",
                    last_activity="2026-05-15T00:00:00Z",
                    assignee="agent-healthy",
                    priority="high",
                ),
                issue(
                    "orphan-1",
                    "TUB-401",
                    "Unassigned todo",
                    "todo",
                    last_activity="2026-05-15T06:00:00Z",
                    priority="high",
                ),
                issue(
                    "chain-root",
                    "TUB-500",
                    "Chain root",
                    "blocked",
                    last_activity="2026-05-16T00:00:00Z",
                    blockers=[blocker("chain-mid", "TUB-501", "blocked")],
                ),
                issue(
                    "chain-mid",
                    "TUB-501",
                    "Chain middle",
                    "blocked",
                    last_activity="2026-05-16T00:00:00Z",
                    blockers=[blocker("chain-end", "TUB-502", "blocked")],
                ),
                issue(
                    "chain-end",
                    "TUB-502",
                    "Chain end",
                    "blocked",
                    last_activity="2026-05-15T18:00:00Z",
                ),
                issue(
                    "cycle-a",
                    "TUB-510",
                    "Cycle A",
                    "blocked",
                    last_activity="2026-05-16T00:00:00Z",
                    blockers=[blocker("cycle-b", "TUB-511", "blocked")],
                ),
                issue(
                    "cycle-b",
                    "TUB-511",
                    "Cycle B",
                    "blocked",
                    last_activity="2026-05-16T00:00:00Z",
                    blockers=[blocker("cycle-a", "TUB-510", "blocked")],
                ),
                issue(
                    "recovery-a",
                    "TUB-600",
                    "Recover stalled issue TUB-700",
                    "todo",
                    last_activity="2026-05-16T12:00:00Z",
                ),
                issue(
                    "recovery-b",
                    "TUB-601",
                    "Recover stalled issue TUB-700",
                    "todo",
                    last_activity="2026-05-16T12:30:00Z",
                ),
                issue(
                    "assigned-error",
                    "TUB-402",
                    "Assigned to unhealthy agent",
                    "todo",
                    last_activity="2026-05-16T00:00:00Z",
                    assignee="agent-error",
                    priority="critical",
                ),
            ],
            "/api/companies/company-1/agents": [
                agent("agent-healthy", "Healthy Agent", "idle", "2026-05-17T08:00:00Z"),
                agent("agent-error", "Errored Agent", "error", "2026-05-17T08:30:00Z"),
                agent("agent-stale", "Stale Agent", "idle", "2026-05-15T00:00:00Z"),
            ],
            "/api/companies/company-1/live-runs": [
                {"id": "run-1", "status": "running", "issueId": "chain-root"},
                {"id": "run-2", "status": "done", "issueId": "done-issue"},
            ],
        }
    )

    result = operational_audit.create_operational_audit(
        project_root=tmp_path,
        api_url="http://paperclip.test",
        company_id="company-1",
        now=now,
        client=client,
    )

    assert result.read_only is True
    assert result.summary_counts()["stale_issues"] >= 1
    assert result.summary_counts()["orphaned_issues"] >= 2
    assert result.summary_counts()["blocked_chains"] >= 1
    assert result.summary_counts()["blocked_cycles"] == 1
    assert result.summary_counts()["unhealthy_agents"] == 2
    assert result.summary_counts()["duplicate_recovery_issues"] == 1
    assert result.summary_counts()["recovery_governor_allow"] >= 1
    assert "TUB-500" in result.blocked_chains[0]["path_identifiers"]
    assert result.duplicate_recovery_issues[0]["target_identifier"] == "TUB-700"
    assert "summary_counts" in result.recovery_governor
    assert "targets" in result.recovery_governor
    priority_actions = operational_audit.build_priority_actions(result)
    priority_keys = [item["key"] for item in priority_actions]
    assert priority_keys[:3] == [
        "repair_unhealthy_agents",
        "review_orphaned_assignments",
        "break_blocked_dependency_cycles",
    ]
    assert all(item["write_permitted"] is False for item in priority_actions)
    assert result.as_dict()["priority_actions"] == priority_actions
    assert Path(result.report_paths["markdown"]).exists()
    assert Path(result.report_paths["json"]).exists()
    assert Path(result.report_paths["latest_markdown"]).exists()
    markdown = Path(result.report_paths["markdown"]).read_text(encoding="utf-8")
    assert "Daily Operational Audit" in markdown
    assert "## Priority Actions" in markdown
    assert "repair_unhealthy_agents" in markdown
    assert client.calls == [
        (
            "/api/companies/company-1/issues",
            {"status": "backlog,todo,in_progress,in_review,blocked", "limit": 500},
        ),
        ("/api/companies/company-1/agents", None),
        ("/api/companies/company-1/live-runs", {"limit": 200, "minCount": 0}),
    ]


def issue(
    issue_id: str,
    identifier: str,
    title: str,
    status: str,
    *,
    last_activity: str,
    assignee: str | None = None,
    blockers: list[dict] | None = None,
    priority: str = "medium",
):
    return {
        "id": issue_id,
        "identifier": identifier,
        "title": title,
        "status": status,
        "priority": priority,
        "assigneeAgentId": assignee,
        "blockedBy": blockers or [],
        "lastActivityAt": last_activity,
        "updatedAt": last_activity,
        "createdAt": last_activity,
        "activeRun": None,
    }


def blocker(blocker_id: str, identifier: str, status: str):
    return {
        "id": blocker_id,
        "identifier": identifier,
        "status": status,
        "title": identifier,
    }


def agent(agent_id: str, name: str, status: str, last_heartbeat: str):
    return {
        "id": agent_id,
        "name": name,
        "status": status,
        "adapterType": "codex_local",
        "lastHeartbeatAt": last_heartbeat,
    }
