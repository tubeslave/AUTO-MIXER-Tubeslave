from __future__ import annotations

from datetime import datetime, timezone

import pytest

from tools.paperclip_watchdog import automixer_paperclip_watchdog as watchdog


class FakePaperclipAPI:
    def __init__(self, status: watchdog.PaperclipStatus | None = None) -> None:
        self.status = status or watchdog.PaperclipStatus(api_ok=True)
        self.created_payloads: list[dict] = []
        self.issue: dict = {"id": "issue-dispatch-1", "status": "todo"}
        self.active_runs: list[dict] = []
        self.patch_calls: list[dict] = []
        self.wakeup_calls: list[dict] = []
        self.live_write_calls: list[dict] = []
        self.raise_on_create: Exception | None = None

    def get_status(self) -> watchdog.PaperclipStatus:
        return self.status

    def create_issue(self, payload: dict) -> dict:
        if self.raise_on_create:
            raise self.raise_on_create
        self.created_payloads.append(payload)
        return {"id": f"issue-{len(self.created_payloads)}"}

    def get_issue(self, issue_id: str) -> dict:
        issue = dict(self.issue)
        issue.setdefault("id", issue_id)
        return issue

    def get_issue_active_runs(self, issue_id: str) -> list[dict]:
        _ = issue_id
        return list(self.active_runs)

    def patch_issue_assignment(self, issue_id: str, agent_id: str, status: str) -> dict:
        call = {
            "method": "PATCH",
            "endpoint": f"/api/issues/{issue_id}",
            "body": {"assigneeAgentId": agent_id, "status": status},
        }
        self.patch_calls.append(call)
        return {"id": issue_id, **call["body"]}


def make_settings(
    tmp_path,
    *,
    dry_run=False,
    max_tasks_per_day=8,
    token="secret-token",
    api_url="https://paperclip.example",
):
    return watchdog.Settings(
        api_url=api_url,
        company_id="company-1",
        token=token,
        interval_seconds=180,
        dry_run=dry_run,
        max_tasks_per_day=max_tasks_per_day,
        cooldown_seconds=0,
        log_path=tmp_path / "watchdog.log",
        state_path=tmp_path / "watchdog_state.json",
        pause_path=tmp_path / "watchdog_paused",
    )


def make_dispatch_settings(
    tmp_path,
    *,
    enabled=True,
    dry_run=False,
    allowlist_agent_ids=None,
    agent_id="agent-1",
    issue_id="issue-dispatch-1",
    state=None,
):
    settings = make_settings(tmp_path, dry_run=True)
    settings.reports_dir = tmp_path / "reports"
    settings.dispatch_policy = watchdog.DispatchPolicy(
        enabled=enabled,
        dry_run=dry_run,
        allowlist_agent_ids=list(allowlist_agent_ids if allowlist_agent_ids is not None else ["agent-1"]),
    )
    settings.dispatch_agent_id = agent_id
    settings.dispatch_issue_id = issue_id
    settings.dispatch_task_key = "wing_telemetry_stability"
    if state is not None:
        watchdog.save_state(settings.state_path, state)
    return settings


def test_dry_run_does_not_create_real_tasks(tmp_path):
    settings = make_settings(tmp_path, dry_run=True)
    api = FakePaperclipAPI()

    decision = watchdog.run_once(settings, api=api, now=fixed_now())

    assert decision.action == "dry_run"
    assert api.created_payloads == []
    assert not settings.state_path.exists()


def test_pause_blocks_creation(tmp_path):
    settings = make_settings(tmp_path)
    settings.pause_path.write_text("paused\n", encoding="utf-8")
    api = FakePaperclipAPI()

    decision = watchdog.run_once(settings, api=api, now=fixed_now())

    assert decision.reason == "paused"
    assert api.created_payloads == []


def test_busy_agents_block_creation(tmp_path):
    settings = make_settings(tmp_path)
    api = FakePaperclipAPI(
        watchdog.PaperclipStatus(api_ok=True, running_agents=1, in_progress_tasks=0)
    )

    decision = watchdog.run_once(settings, api=api, now=fixed_now())

    assert decision.message == "agents busy"
    assert api.created_payloads == []


def test_max_tasks_per_day_blocks_creation(tmp_path):
    settings = make_settings(tmp_path, max_tasks_per_day=1)
    today = watchdog.day_key(fixed_now())
    watchdog.save_state(
        settings.state_path,
        {
            "version": 1,
            "days": {
                today: {
                    "created_count": 1,
                    "task_ids": ["wing_telemetry_stability"],
                }
            },
        },
    )
    api = FakePaperclipAPI()

    decision = watchdog.run_once(settings, api=api, now=fixed_now())

    assert decision.reason == "max_tasks_per_day"
    assert api.created_payloads == []


def test_task_rotation_does_not_repeat_immediately(tmp_path):
    settings = make_settings(tmp_path)
    api = FakePaperclipAPI()

    first = watchdog.run_once(settings, api=api, now=fixed_now())
    second = watchdog.run_once(settings, api=api, now=fixed_now())

    assert first.action == "created"
    assert second.action == "created"
    assert len(api.created_payloads) == 2
    assert api.created_payloads[0]["title"] != api.created_payloads[1]["title"]


def test_token_is_not_written_to_log(tmp_path):
    secret = "super-secret-paperclip-token"
    settings = make_settings(tmp_path, token=secret)
    api = FakePaperclipAPI()
    api.raise_on_create = RuntimeError(f"upstream echoed {secret}")

    decision = watchdog.run_once(settings, api=api, now=fixed_now())

    log_text = settings.log_path.read_text(encoding="utf-8")
    assert decision.reason == "create_failed"
    assert secret not in log_text
    assert "<redacted>" in log_text


def test_localhost_paperclip_does_not_require_token(tmp_path):
    settings = make_settings(
        tmp_path,
        api_url="http://127.0.0.1:3100",
        token=None,
    )

    assert watchdog.missing_config(settings) == []
    api = watchdog.build_api(settings)

    assert isinstance(api, watchdog.PaperclipAPI)
    assert api.client.token is None
    assert "Authorization" not in api.client._headers(has_body=False)


def test_external_paperclip_still_requires_token(tmp_path):
    settings = make_settings(tmp_path, token=None)

    assert watchdog.missing_config(settings) == ["PAPERCLIP_TOKEN"]
    with pytest.raises(ValueError, match="PAPERCLIP_TOKEN is required"):
        watchdog.build_api(settings)


def test_dispatch_default_config_does_not_patch(tmp_path):
    settings = make_settings(tmp_path, dry_run=True)
    settings.reports_dir = tmp_path / "reports"
    settings.dispatch_issue_id = "issue-dispatch-1"
    settings.dispatch_agent_id = "agent-1"
    api = FakePaperclipAPI()

    decision = watchdog.run_dispatch(settings, api=api, now=fixed_now())

    assert decision.result == "blocked"
    assert decision.dispatch_executed is False
    assert api.patch_calls == []
    assert decision.checks["dispatch_enabled"]["passed"] is False


def test_dispatch_dry_run_does_not_patch(tmp_path):
    settings = make_dispatch_settings(tmp_path, dry_run=True)
    api = FakePaperclipAPI()

    decision = watchdog.run_dispatch(settings, api=api, now=fixed_now())

    assert decision.result == "dry_run_preview"
    assert decision.dispatch_executed is False
    assert api.patch_calls == []
    assert settings.reports_dir.joinpath("watchdog_dispatch_report.md").exists()


def test_dispatch_enabled_false_blocks(tmp_path):
    settings = make_dispatch_settings(tmp_path, enabled=False)
    api = FakePaperclipAPI()

    decision = watchdog.run_dispatch(settings, api=api, now=fixed_now())

    assert decision.result == "blocked"
    assert api.patch_calls == []
    assert decision.checks["dispatch_enabled"]["passed"] is False


def test_dispatch_empty_allowlist_blocks(tmp_path):
    settings = make_dispatch_settings(tmp_path, allowlist_agent_ids=[])
    api = FakePaperclipAPI()

    decision = watchdog.run_dispatch(settings, api=api, now=fixed_now())

    assert decision.result == "blocked"
    assert api.patch_calls == []
    assert decision.checks["agent_allowlisted"]["passed"] is False


def test_dispatch_running_agents_block(tmp_path):
    settings = make_dispatch_settings(tmp_path)
    api = FakePaperclipAPI(watchdog.PaperclipStatus(api_ok=True, running_agents=1))

    decision = watchdog.run_dispatch(settings, api=api, now=fixed_now())

    assert decision.result == "blocked"
    assert api.patch_calls == []
    assert decision.checks["running_agents"]["passed"] is False


def test_dispatch_in_progress_tasks_block(tmp_path):
    settings = make_dispatch_settings(tmp_path)
    api = FakePaperclipAPI(watchdog.PaperclipStatus(api_ok=True, in_progress_tasks=1))

    decision = watchdog.run_dispatch(settings, api=api, now=fixed_now())

    assert decision.result == "blocked"
    assert api.patch_calls == []
    assert decision.checks["in_progress_tasks"]["passed"] is False


def test_dispatch_active_runs_block(tmp_path):
    settings = make_dispatch_settings(tmp_path)
    api = FakePaperclipAPI()
    api.active_runs = [{"id": "run-1", "status": "running"}]

    decision = watchdog.run_dispatch(settings, api=api, now=fixed_now())

    assert decision.result == "blocked"
    assert api.patch_calls == []
    assert decision.checks["active_runs"]["passed"] is False


def test_dispatch_issue_status_not_todo_blocks(tmp_path):
    settings = make_dispatch_settings(tmp_path)
    api = FakePaperclipAPI()
    api.issue = {"id": "issue-dispatch-1", "status": "backlog"}

    decision = watchdog.run_dispatch(settings, api=api, now=fixed_now())

    assert decision.result == "blocked"
    assert api.patch_calls == []
    assert decision.checks["issue_status"]["passed"] is False


def test_dispatch_agent_outside_allowlist_blocks(tmp_path):
    settings = make_dispatch_settings(
        tmp_path,
        allowlist_agent_ids=["agent-2"],
        agent_id="agent-1",
    )
    api = FakePaperclipAPI()

    decision = watchdog.run_dispatch(settings, api=api, now=fixed_now())

    assert decision.result == "blocked"
    assert api.patch_calls == []
    assert decision.checks["agent_allowlisted"]["passed"] is False


def test_dispatch_idempotency_prevents_repeat(tmp_path):
    settings = make_dispatch_settings(tmp_path)
    api = FakePaperclipAPI()

    first = watchdog.run_dispatch(settings, api=api, now=fixed_now())
    second = watchdog.run_dispatch(settings, api=api, now=fixed_now())

    assert first.result == "dispatched"
    assert second.result == "blocked"
    assert len(api.patch_calls) == 1
    assert second.checks["idempotency_ok"]["passed"] is False


def test_dispatch_success_uses_patch_issue_assignment_only(tmp_path):
    settings = make_dispatch_settings(tmp_path)
    api = FakePaperclipAPI()

    decision = watchdog.run_dispatch(settings, api=api, now=fixed_now())

    assert decision.result == "dispatched"
    assert decision.dispatch_executed is True
    assert api.patch_calls == [
        {
            "method": "PATCH",
            "endpoint": "/api/issues/issue-dispatch-1",
            "body": {"assigneeAgentId": "agent-1", "status": "todo"},
        }
    ]
    assert api.wakeup_calls == []
    assert api.live_write_calls == []


def test_dispatch_force_preview_never_patches_even_when_real_configured(tmp_path):
    settings = make_dispatch_settings(tmp_path, dry_run=False)
    api = FakePaperclipAPI()

    decision = watchdog.run_dispatch(settings, api=api, now=fixed_now(), force_dry_run=True)

    assert decision.result == "dry_run_preview"
    assert api.patch_calls == []


def fixed_now():
    return datetime(2026, 5, 14, 12, 0, 0, tzinfo=timezone.utc)
