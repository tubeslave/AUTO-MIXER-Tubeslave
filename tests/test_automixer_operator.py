from __future__ import annotations

import subprocess
from pathlib import Path

from tools.automixer_operator import operator as automixer_operator


def make_config(tmp_path: Path) -> automixer_operator.OperatorConfig:
    return automixer_operator.build_config(
        {
            "AUTOMIXER_PROJECT_ROOT": str(tmp_path),
            "PAPERCLIP_API_URL": "http://127.0.0.1:9",
            "GPT_CONTROL_BRIDGE_CONTEXT_URL": "http://127.0.0.1:9/v1/context/automixer",
            "AUTOMIXER_OPERATOR_BRIDGE_TIMEOUT_SECONDS": "0.01",
            "GPT_CONTROL_BRIDGE_PAPERCLIP_TIMEOUT_SECONDS": "0.01",
        }
    )


def write_dry_run_report(tmp_path: Path) -> Path:
    reports_dir = tmp_path / ".paperclip" / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)
    path = reports_dir / "watchdog_dispatch_report.md"
    path.write_text(
        "\n".join(
            [
                "# Watchdog Dispatch Report",
                "",
                "## Summary",
                "- generated_at: 2026-05-15T00:00:00+00:00",
                "- mode: dry_run",
                "- issue: issue-1",
                "- selected_agent: agent-1",
                "- selected_agent_reason: explicit AUTOMIXER_WATCHDOG_DISPATCH_AGENT_ID",
                "- dispatch_allowed: False",
                "- dispatch_executed: False",
                "",
                "## Safety Checks",
                "- idempotency_ok: task:issue-1:agent-1:patch_issue_assignment (passed=True)",
                "- live_osc_blocked: True (passed=True)",
                "- wing_runtime_write_blocked: True (passed=True)",
                "",
                "## Decision",
                "- result: dry_run_preview",
                "- reason: dry-run enabled; PATCH not sent",
                "",
                "## Intended Action",
                "- method: PATCH",
                "- endpoint: /api/issues/issue-1",
                '- body: `{"assigneeAgentId": "agent-1", "status": "todo"}`',
                "",
            ]
        ),
        encoding="utf-8",
    )
    return path


def test_status_works_without_paperclip(tmp_path, capsys):
    exit_code = automixer_operator.command_status(make_config(tmp_path))

    output = capsys.readouterr().out
    assert exit_code == 0
    assert "Automixer Operator Status" in output
    assert "- api_ok: False" in output
    assert "Recommended next action:" in output


def test_preview_runs_watchdog_dispatch_preview_without_real_dispatch(tmp_path, monkeypatch, capsys):
    config = make_config(tmp_path)
    calls = []

    def fake_run(cmd, cwd, env, text, capture_output, check):
        calls.append(
            {
                "cmd": cmd,
                "cwd": cwd,
                "env": env,
                "text": text,
                "capture_output": capture_output,
                "check": check,
            }
        )
        write_dry_run_report(tmp_path)
        return subprocess.CompletedProcess(
            cmd,
            0,
            stdout="dry-run enabled; PATCH not sent\nreport: .paperclip/reports/watchdog_dispatch_report.md\n",
            stderr="",
        )

    monkeypatch.setattr(automixer_operator.subprocess, "run", fake_run)

    exit_code = automixer_operator.command_preview(config)

    output = capsys.readouterr().out
    assert exit_code == 0
    assert len(calls) == 1
    assert calls[0]["cmd"][-1] == "dispatch-preview"
    assert calls[0]["cmd"][-1] != "dispatch"
    assert calls[0]["env"]["AUTOMIXER_WATCHDOG_DISPATCH_DRY_RUN"] == "true"
    assert calls[0]["env"]["AUTOMIXER_WATCHDOG_DISPATCH_FORBID_LIVE_OSC"] == "true"
    assert calls[0]["env"]["AUTOMIXER_WATCHDOG_DISPATCH_FORBID_WING_RUNTIME_WRITE"] == "true"
    assert "approval_command:" in output
    assert "automixer_paperclip_watchdog.py dispatch" in output
    assert "AUTOMIXER_WATCHDOG_DISPATCH_DRY_RUN=false" in output


def test_cycle_runs_full_safe_dry_run_flow(tmp_path, monkeypatch, capsys):
    config = make_config(tmp_path)
    calls = []

    def fake_run(cmd, cwd, env, text, capture_output, check):
        calls.append({"cmd": cmd, "env": env})
        write_dry_run_report(tmp_path)
        return subprocess.CompletedProcess(
            cmd,
            0,
            stdout="dry-run enabled; PATCH not sent\nreport: .paperclip/reports/watchdog_dispatch_report.md\n",
            stderr="",
        )

    monkeypatch.setattr(automixer_operator.subprocess, "run", fake_run)

    exit_code = automixer_operator.command_cycle(config)

    output = capsys.readouterr().out
    assert exit_code == 0
    assert len(calls) == 1
    assert calls[0]["cmd"][-1] == "dispatch-preview"
    assert calls[0]["env"]["AUTOMIXER_WATCHDOG_DISPATCH_DRY_RUN"] == "true"
    assert "Automixer Operator Safe Cycle" in output
    assert "[1/5] status/context" in output
    assert "[2/5] watchdog dry-run preview" in output
    assert "[5/5] approval command" in output
    assert "AUTOMIXER_WATCHDOG_DISPATCH_DRY_RUN=false" in output


def test_doctor_reports_healthy_read_only_loop(monkeypatch, tmp_path, capsys):
    def fake_health(config):
        return {"ok": True, "read_only": True}, None

    def fake_context(config):
        return {
            "_operator_context_source": "test",
            "paperclip_api_ok": True,
            "recommended_next_action": {
                "action": "review_latest_report",
                "reason": "test",
                "requires_approval": False,
                "write_permitted": False,
            },
            "safety_flags": {
                "dispatch_allowed": False,
                "read_only": True,
                "wing_osc_allowed": False,
                "write_routes_enabled": False,
            },
        }

    monkeypatch.setattr(automixer_operator, "fetch_bridge_health", fake_health)
    monkeypatch.setattr(automixer_operator, "load_context", fake_context)

    exit_code = automixer_operator.command_doctor(make_config(tmp_path))

    output = capsys.readouterr().out
    assert exit_code == 0
    assert "Automixer Operator Doctor" in output
    assert "- bridge_read_only: True" in output
    assert "- write_routes_disabled: True" in output


def test_bridge_start_is_noop_when_health_is_ready(monkeypatch, tmp_path, capsys):
    calls = []

    def fake_health(config):
        return {"ok": True, "read_only": True}, None

    def fail_run(*args, **kwargs):
        calls.append((args, kwargs))
        raise AssertionError("bridge-start must not call tmux when health is ready")

    monkeypatch.setattr(automixer_operator, "fetch_bridge_health", fake_health)
    monkeypatch.setattr(automixer_operator.subprocess, "run", fail_run)

    exit_code = automixer_operator.command_bridge_start(make_config(tmp_path))

    output = capsys.readouterr().out
    assert exit_code == 0
    assert "already_running" in output
    assert calls == []


def test_bridge_start_refuses_non_loopback(tmp_path, capsys):
    config = automixer_operator.build_config(
        {
            "AUTOMIXER_PROJECT_ROOT": str(tmp_path),
            "GPT_CONTROL_BRIDGE_CONTEXT_URL": "http://0.0.0.0:8788/v1/context/automixer",
        }
    )

    exit_code = automixer_operator.command_bridge_start(config)

    output = capsys.readouterr().out
    assert exit_code == 1
    assert "refused" in output
    assert "loopback" in output


def test_next_recommends_monitor_when_active_runs_exist():
    action, reason, command = automixer_operator.next_operator_step(
        {
            "paperclip_api_ok": True,
            "active_runs_summary": {"available": True, "total": 1},
        },
        None,
    )

    assert action == "monitor"
    assert "Active run" in reason
    assert command.endswith("operator.py monitor")


def test_next_recommends_cycle_without_ready_report():
    action, reason, command = automixer_operator.next_operator_step(
        {
            "paperclip_api_ok": True,
            "active_runs_summary": {"available": True, "total": 0},
        },
        None,
    )

    assert action == "run_safe_cycle"
    assert "dry-run cycle" in reason
    assert command.endswith("operator.py cycle")


def test_next_recommends_cycle_when_blocked_report_but_target_ready(tmp_path):
    report_path = write_dry_run_report(tmp_path)
    report_path.write_text(
        report_path.read_text(encoding="utf-8").replace(
            "- result: dry_run_preview",
            "- result: blocked",
        ),
        encoding="utf-8",
    )
    report = automixer_operator.latest_dry_run_report(tmp_path)
    target = automixer_operator.TargetSelection(
        issue={"id": "issue-1", "identifier": "TUB-291", "title": "Ready", "status": "todo"},
        agent={"id": "agent-1", "name": "OSC Console Safety Engineer", "status": "idle", "adapterType": "codex_local"},
        reason="ready",
        errors=[],
    )

    action, reason, command = automixer_operator.next_operator_step(
        {
            "paperclip_api_ok": True,
            "active_runs_summary": {"available": True, "total": 0},
        },
        report,
        target,
    )

    assert action == "run_safe_cycle"
    assert "blocked/stale" in reason
    assert command.endswith("operator.py cycle")


def test_work_queue_uses_issue_fallback_counts():
    lines = automixer_operator.render_queue_counts(
        {"available": True, "summary": {"tasks": {"blocked": 2}}},
        {"todo": 3, "in_progress": 1},
        None,
    )

    assert "- blocked: 2" in lines
    assert "- todo: 3" in lines
    assert "- in_progress: 1" in lines


def test_recommended_action_triages_blocked_only_queue():
    action = automixer_operator.operator_recommended_action(
        {
            "paperclip_api_ok": True,
            "active_runs_summary": {"available": True, "total": 0},
            "dashboard_summary": {"available": True, "summary": {"tasks": {"blocked": 7}}},
            "recommended_next_action": {
                "action": "review_latest_report",
                "reason": "generic bridge fallback",
                "requires_approval": False,
                "write_permitted": False,
            },
        },
        {"blocked": 7, "todo": 0, "in_progress": 0},
    )

    assert action is not None
    assert action["action"] == "triage_blocked_queue_or_create_safe_local_task"
    assert "7 blocked issues" in action["reason"]
    assert action["write_permitted"] is False


def test_recommended_action_keeps_base_when_todo_exists():
    action = automixer_operator.operator_recommended_action(
        {
            "paperclip_api_ok": True,
            "active_runs_summary": {"available": True, "total": 0},
            "dashboard_summary": {"available": True, "summary": {"tasks": {"blocked": 7}}},
            "recommended_next_action": {
                "action": "review_latest_report",
                "reason": "generic bridge fallback",
                "requires_approval": False,
                "write_permitted": False,
            },
        },
        {"blocked": 7, "todo": 1, "in_progress": 0},
    )

    assert action is not None
    assert action["action"] == "review_latest_report"


def test_approve_template_only_prints_command(tmp_path, monkeypatch, capsys):
    write_dry_run_report(tmp_path)

    def fail_if_called(*args, **kwargs):
        raise AssertionError("approve-template must not execute subprocesses")

    monkeypatch.setattr(automixer_operator.subprocess, "run", fail_if_called)

    exit_code = automixer_operator.command_approve_template(make_config(tmp_path))

    output = capsys.readouterr().out
    assert exit_code == 0
    assert "approval_command:" in output
    assert "AUTOMIXER_WATCHDOG_DISPATCH_ISSUE_ID=issue-1" in output
    assert "AUTOMIXER_WATCHDOG_DISPATCH_AGENT_ID=agent-1" in output
    assert "AUTOMIXER_WATCHDOG_DISPATCH_IDEMPOTENCY_KEY=task:issue-1:agent-1:patch_issue_assignment" in output


def test_monitor_is_read_only(tmp_path, monkeypatch, capsys):
    write_dry_run_report(tmp_path)
    calls = []

    class FakeClient:
        def __init__(self, base_url, timeout_seconds):
            self.base_url = base_url
            self.timeout_seconds = timeout_seconds

        def get(self, path, params=None):
            calls.append((path, params))
            routes = {
                "/api/health": {"status": "ok"},
                "/api/companies/company-1/dashboard": {"tasks": {"todo": 1, "blocked": 0}},
                "/api/companies/company-1/agents": [{"id": "agent-1", "status": "idle"}],
                "/api/companies/company-1/live-runs": [],
                "/api/issues/issue-1": {"id": "issue-1", "status": "done", "title": "Finished"},
                "/api/issues/issue-1/comments": [{"id": "comment-1", "body": "final report"}],
                "/api/issues/issue-1/live-runs": [{"id": "run-1", "status": "done", "issueId": "issue-1"}],
            }
            if path not in routes:
                raise OSError(f"unexpected route: {path}")
            return routes[path]

    env = dict(config_env(tmp_path))
    env["PAPERCLIP_COMPANY_ID"] = "company-1"
    monkeypatch.setattr(automixer_operator.bridge, "PaperclipClient", FakeClient)

    exit_code = automixer_operator.command_monitor(automixer_operator.build_config(env))

    output = capsys.readouterr().out
    assert exit_code == 0
    assert "Automixer Operator Monitor" in output
    assert "issue terminal status: done" in output
    assert calls
    assert all(path.startswith("/api/") for path, _params in calls)


def test_blocked_command_lists_blocked_issues_read_only(tmp_path, monkeypatch, capsys):
    calls = []

    class FakeClient:
        def __init__(self, base_url, timeout_seconds):
            self.base_url = base_url
            self.timeout_seconds = timeout_seconds

        def get(self, path, params=None):
            calls.append((path, params))
            if path == "/api/companies/company-1/issues":
                return [
                    {
                        "id": "issue-2",
                        "identifier": "TUB-2",
                        "title": "Normal blocked issue",
                        "status": "blocked",
                        "blockerAttention": {"state": "none", "unresolvedBlockerCount": 0},
                    },
                    {
                        "id": "issue-1",
                        "identifier": "TUB-1",
                        "title": "Needs attention",
                        "status": "blocked",
                        "assigneeAgentId": "agent-1",
                        "blockerAttention": {
                            "state": "needs_attention",
                            "unresolvedBlockerCount": 2,
                            "sampleBlockerIdentifier": "TUB-0",
                        },
                    },
                ]
            raise OSError(f"unexpected route: {path}")

    def fake_context(config):
        return {
            "_operator_context_source": "test",
            "paperclip_api_ok": True,
            "paperclip_api_url": "http://paperclip.test",
            "company_id": "company-1",
        }

    monkeypatch.setattr(automixer_operator, "load_context", fake_context)
    monkeypatch.setattr(automixer_operator.bridge, "PaperclipClient", FakeClient)

    exit_code = automixer_operator.command_blocked(make_config(tmp_path))

    output = capsys.readouterr().out
    assert exit_code == 0
    assert "Automixer Blocked Queue" in output
    assert "write actions: disabled" in output
    assert "TUB-1" in output
    assert "sample_blocker: TUB-0" in output
    assert calls == [
        ("/api/companies/company-1/issues", {"status": "blocked", "limit": 20}),
    ]


def test_operational_audit_command_prints_report_paths(tmp_path, monkeypatch, capsys):
    captured = {}

    def fake_load_context(config):
        return {
            "paperclip_api_ok": True,
            "paperclip_api_url": "http://paperclip.test",
            "company_id": "company-1",
        }

    def fake_create_operational_audit(**kwargs):
        captured.update(kwargs)
        return automixer_operator.operational_audit.OperationalAuditResult(
            generated_at="2026-05-17T09:00:00Z",
            company_id="company-1",
            issue_total=10,
            agent_total=3,
            active_run_total=0,
            issue_status_counts={"todo": 10},
            agent_status_counts={"idle": 3},
            thresholds=automixer_operator.operational_audit.AuditThresholds(
                stale_hours=48.0,
                orphan_hours=36.0,
                heartbeat_stale_hours=12.0,
            ),
            stale_issues=[],
            orphaned_issues=[],
            blocked_chains=[],
            blocked_cycles=[],
            unhealthy_agents=[],
            duplicate_recovery_issues=[],
            recovery_governor={"summary_counts": {}, "targets": [], "thresholds": {}},
            run_source="live-runs",
            report_paths={
                "markdown": str(tmp_path / ".paperclip" / "reports" / "operational_audit_test.md"),
                "json": str(tmp_path / ".paperclip" / "reports" / "operational_audit_test.json"),
                "latest_markdown": str(tmp_path / ".paperclip" / "reports" / "operational_audit_latest.md"),
                "latest_json": str(tmp_path / ".paperclip" / "reports" / "operational_audit_latest.json"),
            },
        )

    monkeypatch.setattr(automixer_operator, "load_context", fake_load_context)
    monkeypatch.setattr(
        automixer_operator.operational_audit,
        "create_operational_audit",
        fake_create_operational_audit,
    )

    exit_code = automixer_operator.command_operational_audit(
        make_config(tmp_path),
        ["--stale-hours", "48", "--orphan-hours", "36", "--heartbeat-stale-hours", "12"],
    )

    output = capsys.readouterr().out
    assert exit_code == 0
    assert "Automixer Operational Audit" in output
    assert "report_markdown:" in output
    assert captured["company_id"] == "company-1"
    assert captured["thresholds"].stale_hours == 48.0
    assert captured["thresholds"].orphan_hours == 36.0
    assert captured["thresholds"].heartbeat_stale_hours == 12.0


def test_safety_flags_always_block_wing_osc():
    env = automixer_operator.safety_preview_env()

    assert env["AUTOMIXER_WATCHDOG_DISPATCH_FORBID_LIVE_OSC"] == "true"
    assert env["AUTOMIXER_WATCHDOG_DISPATCH_FORBID_WING_RUNTIME_WRITE"] == "true"
    assert env["AUTOMIXER_WATCHDOG_DISPATCH_FORBID_DIRECT_WAKEUP_WITHOUT_ASSIGNMENT"] == "true"
    assert automixer_operator.READ_ONLY_HTTP_METHODS == ("GET",)


def test_preview_context_defaults_fill_company_issue_and_agent():
    env = automixer_operator.safety_preview_env()

    automixer_operator.apply_preview_context_defaults(
        env,
        {
            "company_id": "company-1",
            "watchdog_state_summary": {
                "dispatch": {
                    "last_issue_id": "issue-1",
                    "last_selected_agent_id": "agent-1",
                    "last_task_key": "task-1",
                }
            },
        },
    )

    assert env["PAPERCLIP_COMPANY_ID"] == "company-1"
    assert env["AUTOMIXER_WATCHDOG_DISPATCH_ISSUE_ID"] == "issue-1"
    assert env["AUTOMIXER_WATCHDOG_DISPATCH_AGENT_ID"] == "agent-1"
    assert env["AUTOMIXER_WATCHDOG_DISPATCH_ALLOWLIST_AGENT_IDS"] == "agent-1"
    assert env["AUTOMIXER_WATCHDOG_DISPATCH_TASK_KEY"] == "task-1"


def test_preview_context_defaults_prefer_safe_target(monkeypatch, tmp_path):
    env = automixer_operator.safety_preview_env()
    target = automixer_operator.TargetSelection(
        issue={
            "id": "issue-target",
            "identifier": "TUB-291",
            "title": "Safe issue",
            "status": "todo",
        },
        agent={
            "id": "agent-target",
            "name": "OSC Console Safety Engineer",
            "status": "idle",
            "adapterType": "codex_local",
        },
        reason="test",
        errors=[],
    )

    automixer_operator.apply_preview_context_defaults(
        env,
        {
            "company_id": "company-1",
            "watchdog_state_summary": {
                "dispatch": {
                    "last_issue_id": "old-issue",
                    "last_selected_agent_id": "old-agent",
                    "last_task_key": "old-task",
                }
            },
        },
        config=make_config(tmp_path),
        target=target,
    )

    assert env["PAPERCLIP_COMPANY_ID"] == "company-1"
    assert env["AUTOMIXER_WATCHDOG_DISPATCH_ISSUE_ID"] == "issue-target"
    assert env["AUTOMIXER_WATCHDOG_DISPATCH_AGENT_ID"] == "agent-target"
    assert env["AUTOMIXER_WATCHDOG_DISPATCH_ALLOWLIST_AGENT_IDS"] == "agent-target"
    assert env["AUTOMIXER_WATCHDOG_DISPATCH_TASK_KEY"] == "TUB-291"


def test_target_selection_picks_unassigned_todo_and_preferred_codex_agent(monkeypatch, tmp_path):
    calls = []

    class FakeClient:
        def __init__(self, base_url, timeout_seconds):
            self.base_url = base_url
            self.timeout_seconds = timeout_seconds

        def get(self, path, params=None):
            calls.append((path, params))
            if path == "/api/companies/company-1/issues":
                return [
                    {
                        "id": "assigned",
                        "identifier": "TUB-292",
                        "title": "Assigned",
                        "status": "todo",
                        "assigneeAgentId": "agent-2",
                    },
                    {
                        "id": "issue-safe",
                        "identifier": "TUB-291",
                        "title": "Safe target",
                        "description": "Review WING telemetry and dry-run console safety before supervised write.",
                        "status": "todo",
                        "assigneeAgentId": None,
                        "activeRun": None,
                    },
                ]
            if path == "/api/companies/company-1/agents":
                return [
                    {
                        "id": "agent-plain",
                        "name": "Plain Codex",
                        "status": "idle",
                        "adapterType": "codex_local",
                    },
                    {
                        "id": "agent-safety",
                        "name": "OSC Console Safety Engineer",
                        "status": "idle",
                        "adapterType": "codex_local",
                    },
                ]
            raise OSError(f"unexpected route: {path}")

    monkeypatch.setattr(automixer_operator.bridge, "PaperclipClient", FakeClient)

    target = automixer_operator.select_preview_target(
        automixer_operator.build_config(config_env(tmp_path)),
        {
            "paperclip_api_ok": True,
            "paperclip_api_url": "http://paperclip.test",
            "company_id": "company-1",
            "active_runs_summary": {"available": True, "total": 0},
        },
    )

    assert target.ready is True
    assert automixer_operator.get_issue_id(target.issue or {}) == "issue-safe"
    assert automixer_operator.get_agent_id(target.agent or {}) == "agent-safety"
    assert target.specialization_key == "console_safety"
    assert calls == [
        (
            "/api/companies/company-1/issues",
            {"status": automixer_operator.PREVIEW_ISSUE_STATUSES, "limit": 500},
        ),
        ("/api/companies/company-1/agents", None),
    ]


def test_target_selection_routes_architecture_issue_to_architect(monkeypatch, tmp_path):
    class FakeClient:
        def __init__(self, base_url, timeout_seconds):
            self.base_url = base_url
            self.timeout_seconds = timeout_seconds

        def get(self, path, params=None):
            if path == "/api/companies/company-1/issues":
                return [
                    {
                        "id": "issue-165",
                        "identifier": "TUB-165",
                        "title": "Implement fixed agent specialization model",
                        "description": "Define orchestration boundaries and reduce routing drift.",
                        "status": "todo",
                    }
                ]
            if path == "/api/companies/company-1/agents":
                return [
                    {
                        "id": "63750d40-cec7-43a2-a008-5b320d85aa75",
                        "name": "Automixer Chief Architect",
                        "status": "idle",
                        "adapterType": "codex_local",
                    },
                    {
                        "id": "c50c0ba1-9c94-4591-a9a4-37bcc7157cf6",
                        "name": "OSC Console Safety Engineer",
                        "status": "idle",
                        "adapterType": "codex_local",
                    },
                ]
            raise OSError(f"unexpected route: {path}")

    monkeypatch.setattr(automixer_operator.bridge, "PaperclipClient", FakeClient)

    target = automixer_operator.select_preview_target(
        automixer_operator.build_config(config_env(tmp_path)),
        {
            "paperclip_api_ok": True,
            "paperclip_api_url": "http://paperclip.test",
            "company_id": "company-1",
            "active_runs_summary": {"available": True, "total": 0},
        },
    )

    assert target.ready is True
    assert automixer_operator.get_issue_id(target.issue or {}) == "issue-165"
    assert automixer_operator.get_agent_id(target.agent or {}) == "63750d40-cec7-43a2-a008-5b320d85aa75"
    assert target.specialization_key == "architecture_owner"


def test_target_selection_blocks_when_active_runs_exist(tmp_path):
    target = automixer_operator.select_preview_target(
        automixer_operator.build_config(config_env(tmp_path)),
        {
            "paperclip_api_ok": True,
            "paperclip_api_url": "http://paperclip.test",
            "company_id": "company-1",
            "active_runs_summary": {"available": True, "total": 1},
        },
    )

    assert target.ready is False
    assert "Active runs" in target.reason


def test_preview_context_defaults_use_latest_nonempty_attempt(tmp_path):
    state_path = tmp_path / ".paperclip" / "watchdog_state.json"
    state_path.parent.mkdir()
    state_path.write_text(
        """
        {
          "dispatch": {
            "attempts": [
              {
                "issue_id": "issue-with-agent",
                "selected_agent_id": "agent-1",
                "task_key": "task-with-agent"
              },
              {
                "issue_id": "issue-without-agent",
                "selected_agent_id": null,
                "task_key": "task-without-agent"
              }
            ]
          }
        }
        """,
        encoding="utf-8",
    )
    env = automixer_operator.safety_preview_env()

    automixer_operator.apply_preview_context_defaults(
        env,
        {
            "company_id": "company-1",
            "watchdog_state_summary": {
                "path": str(state_path),
                "dispatch": {
                    "last_issue_id": "issue-without-agent",
                    "last_selected_agent_id": None,
                    "last_task_key": "task-without-agent",
                },
            },
        },
    )

    assert env["AUTOMIXER_WATCHDOG_DISPATCH_ISSUE_ID"] == "issue-with-agent"
    assert env["AUTOMIXER_WATCHDOG_DISPATCH_AGENT_ID"] == "agent-1"
    assert env["AUTOMIXER_WATCHDOG_DISPATCH_TASK_KEY"] == "task-with-agent"


def test_empty_idempotency_check_does_not_become_key(tmp_path):
    report_path = write_dry_run_report(tmp_path)
    text = report_path.read_text(encoding="utf-8").replace(
        "task:issue-1:agent-1:patch_issue_assignment (passed=True)",
        "(passed=True)",
    )
    report_path.write_text(text, encoding="utf-8")

    report = automixer_operator.parse_dispatch_report(report_path)

    assert report.idempotency_key is None
    assert automixer_operator.approval_command_ready(report) is False


def test_blocked_report_does_not_print_approval_command(tmp_path, capsys):
    report_path = write_dry_run_report(tmp_path)
    report_path.write_text(
        report_path.read_text(encoding="utf-8").replace(
            "- result: dry_run_preview",
            "- result: blocked",
        ),
        encoding="utf-8",
    )

    exit_code = automixer_operator.command_approve_template(make_config(tmp_path))

    output = capsys.readouterr().out
    assert exit_code == 1
    assert "approval_command: unavailable" in output
    assert "not dry_run_preview" in output
    assert "python3 tools/paperclip_watchdog/automixer_paperclip_watchdog.py dispatch" not in output


def test_no_write_routes_or_actions_exist():
    parser = automixer_operator.build_parser()
    command_action = next(action for action in parser._actions if action.dest == "command")

    assert set(command_action.choices) == {
        "status",
        "bridge-status",
        "bridge-start",
        "doctor",
        "next",
        "target",
        "preview",
        "cycle",
        "approve-template",
        "monitor",
        "blocked",
        "operational-audit",
        "vector-index",
        "vector-search",
        "vector-add",
        "vector-stats",
        "vector-paperclip-sync",
    }
    assert "dispatch" not in command_action.choices
    assert "approve" not in command_action.choices
    assert "wakeup" not in command_action.choices
    assert automixer_operator.EXECUTABLE_WRITE_COMMANDS == ()


def test_vector_indexer_command_wraps_local_chromadb_tool(tmp_path, monkeypatch, capsys):
    script = tmp_path / "tools" / "project_vector_index" / "automixer_chromadb_index.py"
    script.parent.mkdir(parents=True)
    script.write_text("#!/usr/bin/env python3\n", encoding="utf-8")
    calls = []

    def fake_run(cmd, cwd, text, capture_output, check):
        calls.append(
            {
                "cmd": cmd,
                "cwd": cwd,
                "text": text,
                "capture_output": capture_output,
                "check": check,
            }
        )
        return subprocess.CompletedProcess(cmd, 0, stdout='{"record_count": 1}\n', stderr="")

    monkeypatch.setattr(automixer_operator.subprocess, "run", fake_run)

    exit_code = automixer_operator.command_vector_indexer(
        make_config(tmp_path),
        "vector-index",
        ["--dry-run"],
    )

    output = capsys.readouterr().out
    assert exit_code == 0
    assert calls[0]["cmd"][1] == str(script)
    assert calls[0]["cmd"][2:5] == ["index", "--project-root", str(tmp_path)]
    assert "--dry-run" in calls[0]["cmd"]
    assert '"record_count": 1' in output


def test_missing_reports_and_state_are_handled_gracefully(tmp_path, capsys):
    approve_exit = automixer_operator.command_approve_template(make_config(tmp_path))
    monitor_exit = automixer_operator.command_monitor(make_config(tmp_path))

    output = capsys.readouterr().out
    assert approve_exit == 1
    assert monitor_exit == 0
    assert "no dry-run watchdog dispatch report found" in output
    assert "issue_id: unavailable" in output


def config_env(tmp_path: Path) -> dict[str, str]:
    return {
        "AUTOMIXER_PROJECT_ROOT": str(tmp_path),
        "PAPERCLIP_API_URL": "http://paperclip.test",
        "GPT_CONTROL_BRIDGE_CONTEXT_URL": "http://127.0.0.1:9/v1/context/automixer",
        "AUTOMIXER_OPERATOR_BRIDGE_TIMEOUT_SECONDS": "0.01",
        "GPT_CONTROL_BRIDGE_PAPERCLIP_TIMEOUT_SECONDS": "0.01",
    }
