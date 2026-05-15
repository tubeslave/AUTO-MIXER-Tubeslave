#!/usr/bin/env python3
"""Conservative local Paperclip watchdog for the Automixer project."""

from __future__ import annotations

import argparse
import json
import logging
import socket
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass, field, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[2]
WATCHDOG_DIR = Path(__file__).resolve().parent

DEFAULT_API_URL = "http://127.0.0.1:3100"
DEFAULT_INTERVAL_SECONDS = 180
DEFAULT_MAX_TASKS_PER_DAY = 8
DEFAULT_COOLDOWN_SECONDS = 1800
DEFAULT_LOG_PATH = "logs/automixer_paperclip_watchdog.log"
DEFAULT_STATE_PATH = ".paperclip/watchdog_state.json"
DEFAULT_PAUSE_PATH = ".paperclip/watchdog_paused"
DEFAULT_REPORTS_DIR = ".paperclip/reports"

ACTIVE_STATUSES = {
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
TERMINAL_STATUSES = {"cancelled", "canceled", "done", "failed", "finished", "succeeded"}
ERROR_STATUSES = {"crashed", "degraded", "error", "errored", "failed", "unhealthy"}
OPEN_BACKLOG_STATUSES = {"backlog", "open", "todo"}


@dataclass(frozen=True)
class WatchdogTask:
    key: str
    title: str
    description: str
    priority: str = "high"

    def to_payload(self, company_id: str) -> dict[str, Any]:
        return {
            "title": self.title,
            "description": self.description,
            "priority": self.priority,
            "companyId": company_id,
        }


SAFE_AUTOMIXER_TASKS = [
    WatchdogTask(
        key="wing_telemetry_stability",
        title="Automixer: проверка стабильности телеметрии WING RACK после supervised writes",
        description=(
            "Проверить последние артефакты supervised WING writes и telemetry capture.\n\n"
            "Ограничения:\n"
            "- не запускать реальные команды в WING RACK;\n"
            "- использовать только существующие логи, runner summary и telemetry artifacts;\n"
            "- итогом должен быть короткий отчет: что стабильно, что требует ручной проверки."
        ),
    ),
    WatchdogTask(
        key="telemetry_dataset_gaps",
        title="Automixer: анализ telemetry dataset на пропуски событий и задержки",
        description=(
            "Проанализировать локальные telemetry datasets на пропущенные события, задержки "
            "между write intent/readback и подозрительные паузы.\n\n"
            "Ограничения:\n"
            "- не подключаться к WING;\n"
            "- не менять production-код;\n"
            "- оформить вывод списком конкретных файлов, временных окон и рисков."
        ),
    ),
    WatchdogTask(
        key="write_gate_vs_runner_summary",
        title="Automixer: сверка write_gate_event с runner summary",
        description=(
            "Сверить журнал write_gate_event против runner summary для последних supervised "
            "soundcheck/write сценариев.\n\n"
            "Ограничения:\n"
            "- только чтение существующих отчетов и логов;\n"
            "- зафиксировать несовпадения по времени, каналу, target value и readback."
        ),
    ),
    WatchdogTask(
        key="legacy_direct_wing_writes_audit",
        title="Automixer: аудит legacy scripts на прямые WING writes",
        description=(
            "Проверить legacy scripts и tooling на обход supervised gate и прямые live-write "
            "вызовы к WING.\n\n"
            "Ограничения:\n"
            "- не запускать live runtime;\n"
            "- не делать device scan/audio capture;\n"
            "- итогом должен быть список файлов, методов и точек риска."
        ),
        priority="critical",
    ),
    WatchdogTask(
        key="autosoundcheck_supervised_gate_audit",
        title="Automixer: доказать, что AutoSoundcheck не пишет в WING без supervised gate",
        description=(
            "Проверить AutoSoundcheck write path и доказать, что запись в WING невозможна без "
            "supervised gate/explicit approval.\n\n"
            "Ограничения:\n"
            "- не вызывать реальный WING;\n"
            "- использовать static analysis, tests или dry-run harness;\n"
            "- если есть gap, описать минимальный безопасный тестовый сценарий."
        ),
        priority="critical",
    ),
    WatchdogTask(
        key="supervised_soundcheck_dry_run",
        title="Automixer: подготовка dry-run сценария supervised soundcheck",
        description=(
            "Подготовить безопасный dry-run сценарий supervised soundcheck для следующей "
            "ручной проверки.\n\n"
            "Ограничения:\n"
            "- сценарий должен быть dry-run по умолчанию;\n"
            "- явно перечислить prerequisites, expected logs, abort conditions и rollback plan;\n"
            "- реальные WING commands не выполнять."
        ),
    ),
    WatchdogTask(
        key="rollback_logic_audit",
        title="Automixer: анализ rollback_last/rollback_all логики",
        description=(
            "Проверить rollback_last и rollback_all: какие write intents они покрывают, "
            "что происходит при частичном failure и как подтверждается readback.\n\n"
            "Ограничения:\n"
            "- не подключаться к WING;\n"
            "- не менять production-код без отдельного approval;\n"
            "- итогом должен быть отчет о live-write safety рисках."
        ),
        priority="critical",
    ),
    WatchdogTask(
        key="next_wing_test_readiness",
        title="Automixer: отчет о готовности к следующему WING тесту",
        description=(
            "Подготовить краткий readiness report для следующего WING теста: что уже доказано "
            "dry-run, что подтверждено supervised run, и что еще не проверено на реальном пульте.\n\n"
            "Ограничения:\n"
            "- не называть систему fully live-ready без фактического подтверждения;\n"
            "- разделить dry-run, guarded/supervised readiness и fully proven live readiness."
        ),
    ),
]


@dataclass
class DispatchPolicy:
    enabled: bool = False
    dry_run: bool = True
    allowlist_agent_ids: list[str] = field(default_factory=list)
    deny_if_running_agents_gt: int = 0
    deny_if_in_progress_tasks_gt: int = 0
    require_issue_status: str = "todo"
    require_no_active_runs: bool = True
    require_idempotency_key: bool = True
    dispatch_method: str = "patch_issue_assignment"
    forbid_direct_wakeup_without_assignment: bool = True
    forbid_live_osc: bool = True
    forbid_wing_runtime_write: bool = True


@dataclass
class Settings:
    api_url: str
    company_id: str
    token: str | None
    interval_seconds: int
    dry_run: bool
    max_tasks_per_day: int
    cooldown_seconds: int
    log_path: Path
    state_path: Path
    pause_path: Path
    reports_dir: Path = field(default_factory=lambda: repo_path(DEFAULT_REPORTS_DIR))
    dispatch_policy: DispatchPolicy = field(default_factory=DispatchPolicy)
    dispatch_issue_id: str | None = None
    dispatch_agent_id: str | None = None
    dispatch_task_key: str | None = None
    dispatch_reason: str = "manual watchdog dispatch safety check"
    dispatch_idempotency_key: str | None = None
    mock: bool = False
    mock_status_json: str | None = None
    timeout_seconds: float = 20.0
    retries: int = 2

    @classmethod
    def from_env(cls, environ: dict[str, str] | None = None) -> "Settings":
        env = environ if environ is not None else dict_env()
        load_env_file(env)
        return cls(
            api_url=env.get("PAPERCLIP_API_URL", DEFAULT_API_URL).rstrip("/"),
            company_id=env.get("PAPERCLIP_COMPANY_ID", "").strip(),
            token=env.get("PAPERCLIP_TOKEN") or None,
            interval_seconds=parse_int(
                env.get("AUTOMIXER_WATCHDOG_INTERVAL_SECONDS"),
                DEFAULT_INTERVAL_SECONDS,
            ),
            dry_run=parse_bool(env.get("AUTOMIXER_WATCHDOG_DRY_RUN"), default=True),
            max_tasks_per_day=parse_int(
                env.get("AUTOMIXER_WATCHDOG_MAX_TASKS_PER_DAY"),
                DEFAULT_MAX_TASKS_PER_DAY,
            ),
            cooldown_seconds=parse_int(
                env.get("AUTOMIXER_WATCHDOG_COOLDOWN_SECONDS"),
                DEFAULT_COOLDOWN_SECONDS,
            ),
            log_path=repo_path(env.get("AUTOMIXER_WATCHDOG_LOG", DEFAULT_LOG_PATH)),
            state_path=repo_path(env.get("AUTOMIXER_WATCHDOG_STATE", DEFAULT_STATE_PATH)),
            pause_path=repo_path(env.get("AUTOMIXER_WATCHDOG_PAUSE", DEFAULT_PAUSE_PATH)),
            reports_dir=repo_path(env.get("AUTOMIXER_WATCHDOG_REPORTS_DIR", DEFAULT_REPORTS_DIR)),
            dispatch_policy=DispatchPolicy(
                enabled=parse_bool(env.get("AUTOMIXER_WATCHDOG_DISPATCH_ENABLED"), default=False),
                dry_run=parse_bool(env.get("AUTOMIXER_WATCHDOG_DISPATCH_DRY_RUN"), default=True),
                allowlist_agent_ids=parse_csv(
                    env.get("AUTOMIXER_WATCHDOG_DISPATCH_ALLOWLIST_AGENT_IDS")
                ),
                deny_if_running_agents_gt=parse_int(
                    env.get("AUTOMIXER_WATCHDOG_DISPATCH_DENY_IF_RUNNING_AGENTS_GT"),
                    0,
                ),
                deny_if_in_progress_tasks_gt=parse_int(
                    env.get("AUTOMIXER_WATCHDOG_DISPATCH_DENY_IF_IN_PROGRESS_TASKS_GT"),
                    0,
                ),
                require_issue_status=(
                    env.get("AUTOMIXER_WATCHDOG_DISPATCH_REQUIRE_ISSUE_STATUS", "todo")
                    .strip()
                    .lower()
                    or "todo"
                ),
                require_no_active_runs=parse_bool(
                    env.get("AUTOMIXER_WATCHDOG_DISPATCH_REQUIRE_NO_ACTIVE_RUNS"),
                    default=True,
                ),
                require_idempotency_key=parse_bool(
                    env.get("AUTOMIXER_WATCHDOG_DISPATCH_REQUIRE_IDEMPOTENCY_KEY"),
                    default=True,
                ),
                dispatch_method=(
                    env.get(
                        "AUTOMIXER_WATCHDOG_DISPATCH_METHOD",
                        "patch_issue_assignment",
                    ).strip()
                    or "patch_issue_assignment"
                ),
                forbid_direct_wakeup_without_assignment=parse_bool(
                    env.get("AUTOMIXER_WATCHDOG_DISPATCH_FORBID_DIRECT_WAKEUP_WITHOUT_ASSIGNMENT"),
                    default=True,
                ),
                forbid_live_osc=parse_bool(
                    env.get("AUTOMIXER_WATCHDOG_DISPATCH_FORBID_LIVE_OSC"),
                    default=True,
                ),
                forbid_wing_runtime_write=parse_bool(
                    env.get("AUTOMIXER_WATCHDOG_DISPATCH_FORBID_WING_RUNTIME_WRITE"),
                    default=True,
                ),
            ),
            dispatch_issue_id=env.get("AUTOMIXER_WATCHDOG_DISPATCH_ISSUE_ID") or None,
            dispatch_agent_id=env.get("AUTOMIXER_WATCHDOG_DISPATCH_AGENT_ID") or None,
            dispatch_task_key=env.get("AUTOMIXER_WATCHDOG_DISPATCH_TASK_KEY") or None,
            dispatch_reason=(
                env.get("AUTOMIXER_WATCHDOG_DISPATCH_REASON")
                or "manual watchdog dispatch safety check"
            ),
            dispatch_idempotency_key=(
                env.get("AUTOMIXER_WATCHDOG_DISPATCH_IDEMPOTENCY_KEY") or None
            ),
            mock=parse_bool(env.get("AUTOMIXER_WATCHDOG_MOCK"), default=False),
            mock_status_json=env.get("AUTOMIXER_WATCHDOG_MOCK_STATUS_JSON") or None,
        )


@dataclass
class PaperclipStatus:
    api_ok: bool
    running_agents: int = 0
    in_progress_tasks: int = 0
    error_agents: int = 0
    blocked_tasks: int = 0
    open_backlog_tasks: int = 0
    warnings: list[str] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)

    def as_dict(self) -> dict[str, Any]:
        return {
            "api_ok": self.api_ok,
            "running_agents": self.running_agents,
            "in_progress_tasks": self.in_progress_tasks,
            "error_agents": self.error_agents,
            "blocked_tasks": self.blocked_tasks,
            "open_backlog_tasks": self.open_backlog_tasks,
            "warnings": self.warnings,
            "errors": self.errors,
        }


@dataclass
class Decision:
    action: str
    reason: str
    message: str
    task: WatchdogTask | None = None
    status: PaperclipStatus | None = None


@dataclass
class DispatchDecision:
    mode: str
    task_key: str
    issue_id: str | None
    selected_agent_id: str | None
    selected_agent_reason: str
    dispatch_allowed: bool
    dispatch_executed: bool
    result: str
    reason: str
    checks: dict[str, dict[str, Any]]
    intended_method: str
    intended_endpoint: str
    intended_body: dict[str, Any]
    actual_response: Any = None
    actual_error: str | None = None
    status: PaperclipStatus | None = None
    issue: dict[str, Any] = field(default_factory=dict)
    active_runs: list[dict[str, Any]] = field(default_factory=list)
    idempotency_key: str | None = None
    report_path: Path | None = None
    approval_required: bool = False
    approval_command: str | None = None
    approval_note: str | None = None

    @property
    def message(self) -> str:
        if self.dispatch_executed:
            return f"dispatch executed: {self.issue_id} -> {self.selected_agent_id}"
        return f"dispatch {self.result}: {self.reason}"


class RedactingFormatter(logging.Formatter):
    def __init__(self, fmt: str, secrets: list[str | None]):
        super().__init__(fmt)
        self._secrets = [secret for secret in secrets if secret]

    def format(self, record: logging.LogRecord) -> str:
        return sanitize_text(super().format(record), self._secrets)


class PaperclipHTTPError(RuntimeError):
    pass


class PaperclipClient:
    def __init__(
        self,
        base_url: str,
        token: str | None,
        timeout_seconds: float = 20.0,
        retries: int = 2,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.token = token
        self.timeout_seconds = timeout_seconds
        self.retries = retries

    def request_json(
        self,
        method: str,
        path: str,
        *,
        params: dict[str, Any] | None = None,
        data: dict[str, Any] | None = None,
    ) -> Any:
        url = self._url(path, params)
        body = json.dumps(data).encode("utf-8") if data is not None else None
        headers = self._headers(has_body=body is not None)

        last_error: Exception | None = None
        for attempt in range(self.retries + 1):
            request = urllib.request.Request(url, data=body, headers=headers, method=method)
            try:
                with urllib.request.urlopen(request, timeout=self.timeout_seconds) as response:
                    text = response.read().decode("utf-8")
                    if not text:
                        return {}
                    try:
                        return json.loads(text)
                    except json.JSONDecodeError:
                        return {"raw": text}
            except urllib.error.HTTPError as error:
                payload = error.read().decode("utf-8", errors="replace")
                if self._should_retry(error.code) and attempt < self.retries:
                    self._sleep_before_retry(attempt)
                    continue
                raise PaperclipHTTPError(
                    f"{method} {path} returned HTTP {error.code}: {payload[:500]}"
                ) from error
            except (urllib.error.URLError, TimeoutError, socket.timeout) as error:
                last_error = error
                if attempt < self.retries:
                    self._sleep_before_retry(attempt)
                    continue
                break

        raise PaperclipHTTPError(f"{method} {path} failed after retries: {last_error}")

    def _url(self, path: str, params: dict[str, Any] | None) -> str:
        query = urllib.parse.urlencode(params or {}, doseq=True)
        separator = "&" if "?" in path else "?"
        suffix = f"{separator}{query}" if query else ""
        return f"{self.base_url}{path}{suffix}"

    def _headers(self, *, has_body: bool) -> dict[str, str]:
        headers = {"Accept": "application/json"}
        if self.token:
            headers["Authorization"] = f"Bearer {self.token}"
        if has_body:
            headers["Content-Type"] = "application/json"
        return headers

    @staticmethod
    def _should_retry(status_code: int) -> bool:
        return status_code in {408, 429} or status_code >= 500

    @staticmethod
    def _sleep_before_retry(attempt: int) -> None:
        time.sleep(0.5 * (attempt + 1))


class PaperclipAPI:
    def __init__(self, client: PaperclipClient, company_id: str) -> None:
        self.client = client
        self.company_id = company_id

    def get_status(self) -> PaperclipStatus:
        errors: list[str] = []
        warnings: list[str] = []

        health = self._required_get("/api/health", errors)
        agents_payload = self._required_get(
            f"/api/companies/{urllib.parse.quote(self.company_id)}/agents",
            errors,
        )
        issues_payload = self._required_get(
            f"/api/companies/{urllib.parse.quote(self.company_id)}/issues",
            errors,
            params={
                "status": "backlog,todo,in_progress,in_review,blocked",
                "limit": 500,
            },
        )
        runs_payload = self._required_get(
            f"/api/companies/{urllib.parse.quote(self.company_id)}/heartbeat-runs",
            errors,
            params={"limit": 200},
        )
        dashboard_payload = self._optional_get(
            f"/api/companies/{urllib.parse.quote(self.company_id)}/dashboard",
            warnings,
        )

        if errors:
            return PaperclipStatus(api_ok=False, warnings=warnings, errors=errors)

        agents = extract_list(agents_payload, ["agents", "data", "items", "results"])
        issues = extract_list(issues_payload, ["issues", "data", "items", "results"])
        runs = extract_list(
            runs_payload,
            ["heartbeatRuns", "heartbeat_runs", "runs", "data", "items", "results"],
        )

        status_counts: dict[str, int] = {}
        for issue in issues:
            status = normalized_status(get_status_value(issue))
            status_counts[status] = status_counts.get(status, 0) + 1

        agent_running_count = sum(1 for agent in agents if is_agent_running(agent))
        active_run_count = sum(1 for run in runs if is_run_active(run))
        error_agent_count = sum(1 for agent in agents if is_agent_error(agent))
        in_progress_count = status_counts.get("in_progress", 0)
        blocked_count = status_counts.get("blocked", 0)
        open_backlog_count = sum(
            count for status, count in status_counts.items() if status in OPEN_BACKLOG_STATUSES
        )

        running_from_dashboard = deep_find_number(
            dashboard_payload,
            ["runningAgents", "running_agents", "activeRuns", "active_runs"],
        )
        in_progress_from_dashboard = deep_find_number(
            dashboard_payload,
            ["inProgressTasks", "in_progress_tasks", "inProgressIssues", "in_progress_issues"],
        )
        error_from_dashboard = deep_find_number(
            dashboard_payload,
            ["errorAgents", "error_agents", "failedAgents", "failed_agents"],
        )

        _ = health
        return PaperclipStatus(
            api_ok=True,
            running_agents=max(agent_running_count, active_run_count, running_from_dashboard or 0),
            in_progress_tasks=max(in_progress_count, in_progress_from_dashboard or 0),
            error_agents=max(error_agent_count, error_from_dashboard or 0),
            blocked_tasks=blocked_count,
            open_backlog_tasks=open_backlog_count,
            warnings=warnings,
        )

    def create_issue(self, payload: dict[str, Any]) -> Any:
        return self.client.request_json(
            "POST",
            f"/api/companies/{urllib.parse.quote(self.company_id)}/issues",
            data=payload,
        )

    def get_issue(self, issue_id: str) -> dict[str, Any]:
        payload = self.client.request_json(
            "GET",
            f"/api/issues/{urllib.parse.quote(issue_id)}",
        )
        return extract_object(payload, ["issue", "data", "item", "result"])

    def get_issue_active_runs(self, issue_id: str) -> list[dict[str, Any]]:
        issue_path = f"/api/issues/{urllib.parse.quote(issue_id)}/live-runs"
        try:
            payload = self.client.request_json("GET", issue_path)
            return extract_list(
                payload,
                ["liveRuns", "live_runs", "heartbeatRuns", "heartbeat_runs", "runs", "data"],
            )
        except Exception as primary_exc:
            company_path = f"/api/companies/{urllib.parse.quote(self.company_id)}/heartbeat-runs"
            try:
                payload = self.client.request_json(
                    "GET",
                    company_path,
                    params={"issueId": issue_id, "limit": 50},
                )
                runs = extract_list(
                    payload,
                    ["heartbeatRuns", "heartbeat_runs", "runs", "data", "items", "results"],
                )
                return [run for run in runs if issue_matches_run(run, issue_id)]
            except Exception as fallback_exc:
                raise PaperclipHTTPError(
                    f"active runs lookup failed: {primary_exc}; fallback failed: {fallback_exc}"
                ) from fallback_exc

    def patch_issue_assignment(self, issue_id: str, agent_id: str, status: str) -> Any:
        return self.client.request_json(
            "PATCH",
            f"/api/issues/{urllib.parse.quote(issue_id)}",
            data={"assigneeAgentId": agent_id, "status": status},
        )

    def _required_get(
        self,
        path: str,
        errors: list[str],
        *,
        params: dict[str, Any] | None = None,
    ) -> Any:
        try:
            return self.client.request_json("GET", path, params=params)
        except Exception as exc:
            errors.append(f"{path}: {exc}")
            return {}

    def _optional_get(
        self,
        path: str,
        warnings: list[str],
        *,
        params: dict[str, Any] | None = None,
    ) -> Any:
        try:
            return self.client.request_json("GET", path, params=params)
        except Exception as exc:
            warnings.append(f"{path}: {exc}")
            return {}


class MockPaperclipAPI:
    def __init__(self, status: PaperclipStatus | None = None) -> None:
        self.status = status or PaperclipStatus(api_ok=True)
        self.created_payloads: list[dict[str, Any]] = []
        self.issue: dict[str, Any] = {"id": "mock-issue", "status": "todo"}
        self.active_runs: list[dict[str, Any]] = []
        self.patch_calls: list[dict[str, Any]] = []

    @classmethod
    def from_env(cls, environ: dict[str, str] | None = None) -> "MockPaperclipAPI":
        env = environ if environ is not None else dict_env()
        return cls.from_status_json(env.get("AUTOMIXER_WATCHDOG_MOCK_STATUS_JSON"))

    @classmethod
    def from_status_json(cls, raw_status: str | None) -> "MockPaperclipAPI":
        if not raw_status:
            return cls()
        try:
            payload = json.loads(raw_status)
        except json.JSONDecodeError:
            return cls(PaperclipStatus(api_ok=False, errors=["invalid mock status JSON"]))
        return cls(
            PaperclipStatus(
                api_ok=bool(payload.get("api_ok", True)),
                running_agents=int(payload.get("running_agents", 0)),
                in_progress_tasks=int(payload.get("in_progress_tasks", 0)),
                error_agents=int(payload.get("error_agents", 0)),
                blocked_tasks=int(payload.get("blocked_tasks", 0)),
                open_backlog_tasks=int(payload.get("open_backlog_tasks", 0)),
                warnings=list(payload.get("warnings", [])),
                errors=list(payload.get("errors", [])),
            )
        )

    def get_status(self) -> PaperclipStatus:
        return self.status

    def create_issue(self, payload: dict[str, Any]) -> dict[str, Any]:
        self.created_payloads.append(payload)
        return {"id": f"mock-watchdog-{len(self.created_payloads)}", "mock": True}

    def get_issue(self, issue_id: str) -> dict[str, Any]:
        issue = dict(self.issue)
        issue.setdefault("id", issue_id)
        return issue

    def get_issue_active_runs(self, issue_id: str) -> list[dict[str, Any]]:
        _ = issue_id
        return list(self.active_runs)

    def patch_issue_assignment(self, issue_id: str, agent_id: str, status: str) -> dict[str, Any]:
        payload = {"assigneeAgentId": agent_id, "status": status}
        self.patch_calls.append({"issue_id": issue_id, "body": payload})
        return {"id": issue_id, **payload}


def dict_env() -> dict[str, str]:
    import os

    return dict(os.environ)


def load_env_file(environ: dict[str, str]) -> None:
    env_file = environ.get("AUTOMIXER_WATCHDOG_ENV_FILE")
    path = Path(env_file).expanduser() if env_file else WATCHDOG_DIR / ".env"
    if not path.exists():
        return

    for line in path.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#") or "=" not in stripped:
            continue
        key, value = stripped.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key and key not in environ:
            environ[key] = value


def parse_bool(value: str | None, *, default: bool = False) -> bool:
    if value is None or value == "":
        return default
    return value.strip().lower() in {"1", "true", "yes", "y", "on"}


def parse_int(value: str | None, default: int) -> int:
    if value is None or value == "":
        return default
    try:
        parsed = int(value)
    except ValueError:
        return default
    return parsed if parsed >= 0 else default


def parse_csv(value: str | None) -> list[str]:
    if not value:
        return []
    return [item.strip() for item in value.split(",") if item.strip()]


def repo_path(value: str | Path) -> Path:
    path = Path(value).expanduser()
    return path if path.is_absolute() else REPO_ROOT / path


def sanitize_text(text: str, secrets: list[str | None]) -> str:
    sanitized = text
    for secret in secrets:
        if secret:
            sanitized = sanitized.replace(secret, "<redacted>")
    return sanitized


def configure_logger(settings: Settings) -> logging.Logger:
    settings.log_path.parent.mkdir(parents=True, exist_ok=True)
    logger_name = f"automixer_paperclip_watchdog.{settings.log_path.resolve()}"
    logger = logging.getLogger(logger_name)
    logger.handlers.clear()
    logger.setLevel(logging.INFO)
    logger.propagate = False

    handler = logging.FileHandler(settings.log_path, encoding="utf-8")
    handler.setFormatter(
        RedactingFormatter("%(asctime)s %(levelname)s %(message)s", [settings.token])
    )
    logger.addHandler(handler)
    return logger


def build_api(settings: Settings) -> PaperclipAPI | MockPaperclipAPI:
    if settings.mock:
        return MockPaperclipAPI.from_status_json(settings.mock_status_json)
    if not settings.token and not is_local_paperclip_api(settings.api_url):
        raise ValueError("PAPERCLIP_TOKEN is required")
    client = PaperclipClient(
        settings.api_url,
        settings.token,
        timeout_seconds=settings.timeout_seconds,
        retries=settings.retries,
    )
    return PaperclipAPI(client, settings.company_id)


def run_once(
    settings: Settings,
    *,
    api: Any | None = None,
    now: datetime | None = None,
    logger: logging.Logger | None = None,
) -> Decision:
    now = now or datetime.now().astimezone()
    logger = logger or configure_logger(settings)
    secrets = [settings.token]

    if settings.pause_path.exists():
        message = "watchdog paused"
        logger.info(message)
        return Decision(action="skipped", reason="paused", message=message)

    missing = missing_config(settings)
    if missing:
        message = f"missing required config: {', '.join(missing)}"
        logger.error(message)
        return Decision(action="error", reason="missing_config", message=message)

    api = api or build_api(settings)
    try:
        status = api.get_status()
    except Exception as exc:
        message = f"status check failed: {sanitize_text(str(exc), secrets)}"
        logger.error(message)
        return Decision(action="error", reason="status_error", message=message)

    logger.info(
        "status api_ok=%s running_agents=%s in_progress_tasks=%s error_agents=%s "
        "blocked_tasks=%s open_backlog_tasks=%s",
        status.api_ok,
        status.running_agents,
        status.in_progress_tasks,
        status.error_agents,
        status.blocked_tasks,
        status.open_backlog_tasks,
    )

    if not status.api_ok:
        message = f"paperclip status unavailable: {sanitize_text('; '.join(status.errors), secrets)}"
        logger.warning(message)
        return Decision(action="skipped", reason="status_unavailable", message=message, status=status)

    if status.running_agents > 0 or status.in_progress_tasks > 0:
        message = "agents busy"
        logger.info(
            "%s: running_agents=%s in_progress_tasks=%s",
            message,
            status.running_agents,
            status.in_progress_tasks,
        )
        return Decision(action="skipped", reason="busy", message=message, status=status)

    if status.error_agents > 0:
        message = "error agents present"
        logger.warning("%s: error_agents=%s", message, status.error_agents)
        return Decision(action="skipped", reason="error_agents", message=message, status=status)

    state = load_state(settings.state_path)
    day = day_key(now)
    day_state = get_day_state(state, day)
    created_today = int(day_state.get("created_count", 0))
    if created_today >= settings.max_tasks_per_day:
        message = "max_tasks_per_day reached"
        logger.info("%s: created_today=%s", message, created_today)
        return Decision(action="skipped", reason="max_tasks_per_day", message=message, status=status)

    cooldown_remaining = get_cooldown_remaining(state, now, settings.cooldown_seconds)
    if cooldown_remaining > 0:
        message = f"cooldown active: {cooldown_remaining}s remaining"
        logger.info(message)
        return Decision(action="skipped", reason="cooldown", message=message, status=status)

    task = select_next_task(state, day)
    if task is None:
        message = "all safe tasks already used today"
        logger.info(message)
        return Decision(action="skipped", reason="all_tasks_used_today", message=message, status=status)

    payload = task.to_payload(settings.company_id or "mock-company")
    if settings.dry_run:
        message = f"dry-run: would create task '{task.title}'"
        logger.info("%s priority=%s", message, task.priority)
        return Decision(action="dry_run", reason="dry_run", message=message, task=task, status=status)

    try:
        response = api.create_issue(payload)
    except Exception as exc:
        message = f"create issue failed: {sanitize_text(str(exc), secrets)}"
        logger.error(message)
        return Decision(action="error", reason="create_failed", message=message, task=task, status=status)

    record_created_task(state, day, task, response, now)
    save_state(settings.state_path, state)
    action = "mock_created" if settings.mock else "created"
    message = f"{action}: {task.title}"
    logger.info("%s response_id=%s", message, extract_response_id(response))
    return Decision(action=action, reason=action, message=message, task=task, status=status)


def missing_config(settings: Settings) -> list[str]:
    if settings.mock:
        return []
    missing: list[str] = []
    if not settings.token and not is_local_paperclip_api(settings.api_url):
        missing.append("PAPERCLIP_TOKEN")
    if not settings.company_id:
        missing.append("PAPERCLIP_COMPANY_ID")
    if not settings.api_url:
        missing.append("PAPERCLIP_API_URL")
    return missing


def run_dispatch(
    settings: Settings,
    *,
    api: Any | None = None,
    now: datetime | None = None,
    logger: logging.Logger | None = None,
    force_dry_run: bool = False,
) -> DispatchDecision:
    now = now or datetime.now().astimezone()
    logger = logger or configure_logger(settings)
    secrets = [settings.token]
    state = load_state(settings.state_path)
    policy = settings.dispatch_policy
    mode = "dry_run" if force_dry_run or policy.dry_run else "real"
    issue_id = settings.dispatch_issue_id or state.get("last_issue_id")
    task_key = settings.dispatch_task_key or state.get("last_task_key") or "manual_dispatch"
    selected_agent_id, selected_agent_reason = select_dispatch_agent(settings)
    idempotency_key = (
        settings.dispatch_idempotency_key
        or build_dispatch_idempotency_key(task_key, issue_id, selected_agent_id, policy.dispatch_method)
    )
    intended_endpoint = f"/api/issues/{issue_id}" if issue_id else "/api/issues/<issue-id>"
    intended_body = {"assigneeAgentId": selected_agent_id, "status": policy.require_issue_status}
    checks: dict[str, dict[str, Any]] = {}

    add_dispatch_check(checks, "not_paused", not settings.pause_path.exists(), settings.pause_path.exists())
    missing = missing_config(settings)
    add_dispatch_check(checks, "config_present", not missing, ",".join(missing) if missing else "ok")
    add_dispatch_check(checks, "dispatch_enabled", policy.enabled, policy.enabled)
    add_dispatch_check(checks, "dispatch_dry_run_off", mode == "real", mode)
    add_dispatch_check(
        checks,
        "dispatch_method_patch_issue_assignment",
        policy.dispatch_method == "patch_issue_assignment",
        policy.dispatch_method,
    )
    add_dispatch_check(
        checks,
        "direct_wakeup_forbidden",
        policy.forbid_direct_wakeup_without_assignment,
        policy.forbid_direct_wakeup_without_assignment,
    )
    add_dispatch_check(checks, "live_osc_blocked", policy.forbid_live_osc, policy.forbid_live_osc)
    add_dispatch_check(
        checks,
        "wing_runtime_write_blocked",
        policy.forbid_wing_runtime_write,
        policy.forbid_wing_runtime_write,
    )
    add_dispatch_check(checks, "issue_id_present", bool(issue_id), issue_id or "")
    add_dispatch_check(checks, "agent_id_selected", bool(selected_agent_id), selected_agent_id or "")
    add_dispatch_check(
        checks,
        "agent_allowlisted",
        bool(selected_agent_id) and selected_agent_id in policy.allowlist_agent_ids,
        selected_agent_id or "",
    )
    add_dispatch_check(
        checks,
        "idempotency_key_present",
        bool(idempotency_key) or not policy.require_idempotency_key,
        idempotency_key or "",
    )
    add_dispatch_check(
        checks,
        "idempotency_ok",
        not idempotency_key_used(state, idempotency_key),
        idempotency_key or "",
    )

    status: PaperclipStatus | None = None
    issue: dict[str, Any] = {}
    active_runs: list[dict[str, Any]] = []
    actual_response: Any = None
    actual_error: str | None = None

    if not missing:
        try:
            api = api or build_api(settings)
            status = api.get_status()
        except Exception as exc:
            actual_error = f"status check failed: {sanitize_text(str(exc), secrets)}"
            status = PaperclipStatus(api_ok=False, errors=[actual_error])
        add_dispatch_check(checks, "paperclip_api_ok", bool(status and status.api_ok), status.as_dict())

        if status and status.api_ok:
            add_dispatch_check(
                checks,
                "running_agents",
                status.running_agents <= policy.deny_if_running_agents_gt,
                status.running_agents,
            )
            add_dispatch_check(
                checks,
                "in_progress_tasks",
                status.in_progress_tasks <= policy.deny_if_in_progress_tasks_gt,
                status.in_progress_tasks,
            )
            if issue_id:
                try:
                    issue = api.get_issue(issue_id)
                except Exception as exc:
                    actual_error = f"issue lookup failed: {sanitize_text(str(exc), secrets)}"
                issue_status = normalized_status(get_status_value(issue))
                add_dispatch_check(
                    checks,
                    "issue_status",
                    issue_status == policy.require_issue_status,
                    issue_status or "",
                )
                existing_assignee = get_assignee_agent_id(issue)
                add_dispatch_check(
                    checks,
                    "issue_not_already_assigned",
                    not existing_assignee,
                    existing_assignee or "",
                )
                try:
                    runs = api.get_issue_active_runs(issue_id)
                    active_runs = [run for run in runs if is_run_active(run)]
                    active_runs_ok = not active_runs or not policy.require_no_active_runs
                    add_dispatch_check(checks, "active_runs", active_runs_ok, len(active_runs))
                except Exception as exc:
                    actual_error = f"active runs lookup failed: {sanitize_text(str(exc), secrets)}"
                    add_dispatch_check(checks, "active_runs", False, actual_error)
            else:
                add_dispatch_check(checks, "issue_status", False, "")
                add_dispatch_check(checks, "issue_not_already_assigned", False, "")
                add_dispatch_check(checks, "active_runs", False, "missing issue id")
        else:
            add_dispatch_check(checks, "running_agents", False, "unknown")
            add_dispatch_check(checks, "in_progress_tasks", False, "unknown")
            add_dispatch_check(checks, "issue_status", False, "unknown")
            add_dispatch_check(checks, "issue_not_already_assigned", False, "unknown")
            add_dispatch_check(checks, "active_runs", False, "unknown")
    else:
        add_dispatch_check(checks, "paperclip_api_ok", False, "missing config")
        add_dispatch_check(checks, "running_agents", False, "unknown")
        add_dispatch_check(checks, "in_progress_tasks", False, "unknown")
        add_dispatch_check(checks, "issue_status", False, "unknown")
        add_dispatch_check(checks, "issue_not_already_assigned", False, "unknown")
        add_dispatch_check(checks, "active_runs", False, "unknown")

    dispatch_allowed = all(check["passed"] for check in checks.values())
    dispatch_executed = False
    result = "blocked"
    reason = first_failed_check_reason(checks)

    if dispatch_allowed and issue_id and selected_agent_id:
        try:
            actual_response = api.patch_issue_assignment(
                issue_id,
                selected_agent_id,
                policy.require_issue_status,
            )
            dispatch_executed = True
            result = "dispatched"
            reason = "PATCH issue assignment executed"
        except Exception as exc:
            actual_error = f"PATCH issue assignment failed: {sanitize_text(str(exc), secrets)}"
            result = "error"
            reason = actual_error
    elif checks.get("dispatch_dry_run_off", {}).get("passed") is False and all(
        check["passed"]
        for name, check in checks.items()
        if name != "dispatch_dry_run_off"
    ):
        result = "dry_run_preview"
        reason = "dry-run enabled; PATCH not sent"

    decision = DispatchDecision(
        mode=mode,
        task_key=str(task_key),
        issue_id=str(issue_id) if issue_id else None,
        selected_agent_id=selected_agent_id,
        selected_agent_reason=selected_agent_reason,
        dispatch_allowed=dispatch_allowed,
        dispatch_executed=dispatch_executed,
        result=result,
        reason=reason,
        checks=checks,
        intended_method="PATCH",
        intended_endpoint=intended_endpoint,
        intended_body=intended_body,
        actual_response=actual_response,
        actual_error=actual_error,
        status=status,
        issue=issue,
        active_runs=active_runs,
        idempotency_key=idempotency_key,
    )
    record_dispatch_attempt(state, decision, now)
    save_state(settings.state_path, state)
    decision.report_path = write_dispatch_report(settings, decision, now)
    logger.info(
        "dispatch result=%s mode=%s issue_id=%s selected_agent_id=%s allowed=%s executed=%s reason=%s",
        decision.result,
        decision.mode,
        decision.issue_id,
        decision.selected_agent_id,
        decision.dispatch_allowed,
        decision.dispatch_executed,
        decision.reason,
    )
    return decision


def run_orchestration_preview(
    settings: Settings,
    *,
    api: Any | None = None,
    now: datetime | None = None,
    logger: logging.Logger | None = None,
) -> DispatchDecision:
    now = now or datetime.now().astimezone()
    selected_agent_id, selected_agent_reason = select_dispatch_agent(settings)
    preview_settings = build_orchestration_preview_settings(settings, selected_agent_id)
    decision = run_dispatch(
        preview_settings,
        api=api,
        now=now,
        logger=logger,
        force_dry_run=True,
    )
    decision.mode = "semi_auto_preview"
    decision.selected_agent_reason = selected_agent_reason
    decision.approval_required = True
    decision.approval_command = build_dispatch_approval_command(decision)
    if decision.result == "dry_run_preview":
        decision.approval_note = (
            "Preview passed. Real dispatch still requires explicit operator approval."
        )
    else:
        decision.approval_note = (
            "Preview did not pass. Do not approve real dispatch until blocked checks are fixed."
        )
    decision.report_path = write_dispatch_report(settings, decision, now)
    return decision


def build_orchestration_preview_settings(
    settings: Settings,
    selected_agent_id: str | None,
) -> Settings:
    preview_policy = replace(
        settings.dispatch_policy,
        enabled=True,
        dry_run=True,
    )
    return replace(
        settings,
        dispatch_policy=preview_policy,
        dispatch_agent_id=selected_agent_id,
    )


def select_dispatch_agent(settings: Settings) -> tuple[str | None, str]:
    if settings.dispatch_agent_id:
        return settings.dispatch_agent_id, "explicit AUTOMIXER_WATCHDOG_DISPATCH_AGENT_ID"
    allowlist = settings.dispatch_policy.allowlist_agent_ids
    if allowlist:
        return allowlist[0], "first agent in explicit allowlist"
    return None, "no allowlisted agent configured"


def build_dispatch_approval_command(decision: DispatchDecision) -> str | None:
    if not decision.issue_id or not decision.selected_agent_id or not decision.idempotency_key:
        return None
    env_parts = {
        "AUTOMIXER_WATCHDOG_DISPATCH_ENABLED": "true",
        "AUTOMIXER_WATCHDOG_DISPATCH_DRY_RUN": "false",
        "AUTOMIXER_WATCHDOG_DISPATCH_ISSUE_ID": decision.issue_id,
        "AUTOMIXER_WATCHDOG_DISPATCH_AGENT_ID": decision.selected_agent_id,
        "AUTOMIXER_WATCHDOG_DISPATCH_IDEMPOTENCY_KEY": decision.idempotency_key,
    }
    prefix = " ".join(f"{key}={shell_quote(value)}" for key, value in env_parts.items())
    return f"{prefix} python3 tools/paperclip_watchdog/automixer_paperclip_watchdog.py dispatch"


def build_dispatch_idempotency_key(
    task_key: str | None,
    issue_id: str | None,
    agent_id: str | None,
    dispatch_method: str,
) -> str | None:
    if not issue_id or not agent_id:
        return None
    return f"{task_key or 'manual_dispatch'}:{issue_id}:{agent_id}:{dispatch_method}"


def add_dispatch_check(
    checks: dict[str, dict[str, Any]],
    name: str,
    passed: bool,
    value: Any,
    detail: str = "",
) -> None:
    checks[name] = {"passed": bool(passed), "value": value, "detail": detail}


def first_failed_check_reason(checks: dict[str, dict[str, Any]]) -> str:
    for name, check in checks.items():
        if not check["passed"]:
            return f"{name} failed: {check['value']}"
    return "all checks passed"


def idempotency_key_used(state: dict[str, Any], idempotency_key: str | None) -> bool:
    if not idempotency_key:
        return False
    dispatch_state = state.get("dispatch")
    if not isinstance(dispatch_state, dict):
        return False
    used = dispatch_state.get("used_idempotency_keys")
    return isinstance(used, dict) and idempotency_key in used


def record_dispatch_attempt(
    state: dict[str, Any],
    decision: DispatchDecision,
    now: datetime,
) -> None:
    dispatch_state = state.setdefault("dispatch", {})
    if not isinstance(dispatch_state, dict):
        dispatch_state = {}
        state["dispatch"] = dispatch_state
    timestamp = now.astimezone(timezone.utc).isoformat()
    entry = {
        "attempted_at": timestamp,
        "task_key": decision.task_key,
        "issue_id": decision.issue_id,
        "last_dispatch_result": decision.result,
        "idempotency_key": decision.idempotency_key,
        "selected_agent_id": decision.selected_agent_id,
        "mode": decision.mode,
        "reason": decision.reason,
        "blocked_reason": None if decision.dispatch_executed else decision.reason,
        "dispatch_allowed": decision.dispatch_allowed,
        "dispatch_executed": decision.dispatch_executed,
    }
    attempts = dispatch_state.setdefault("attempts", [])
    if not isinstance(attempts, list):
        attempts = []
        dispatch_state["attempts"] = attempts
    attempts.append(entry)
    del attempts[:-50]

    dispatch_state["last_dispatch_attempt_time"] = timestamp
    dispatch_state["last_dispatch_result"] = decision.result
    dispatch_state["last_issue_id"] = decision.issue_id
    dispatch_state["last_task_key"] = decision.task_key
    dispatch_state["last_idempotency_key"] = decision.idempotency_key
    dispatch_state["last_selected_agent_id"] = decision.selected_agent_id
    dispatch_state["last_mode"] = decision.mode
    dispatch_state["last_reason"] = decision.reason
    dispatch_state["last_blocked_reason"] = None if decision.dispatch_executed else decision.reason

    if decision.dispatch_executed and decision.idempotency_key:
        used = dispatch_state.setdefault("used_idempotency_keys", {})
        if isinstance(used, dict):
            used[decision.idempotency_key] = {
                "used_at": timestamp,
                "issue_id": decision.issue_id,
                "selected_agent_id": decision.selected_agent_id,
                "result": decision.result,
            }


def write_dispatch_report(settings: Settings, decision: DispatchDecision, now: datetime) -> Path:
    settings.reports_dir.mkdir(parents=True, exist_ok=True)
    path = settings.reports_dir / "watchdog_dispatch_report.md"
    path.write_text(build_dispatch_report(decision, now), encoding="utf-8")
    return path


def build_dispatch_report(decision: DispatchDecision, now: datetime) -> str:
    status = decision.status or PaperclipStatus(api_ok=False)
    issue_status = normalized_status(get_status_value(decision.issue))
    checks = decision.checks
    lines = [
        "# Watchdog Dispatch Report",
        "",
        "## Summary",
        f"- generated_at: {now.astimezone(timezone.utc).isoformat()}",
        f"- mode: {decision.mode}",
        f"- issue: {decision.issue_id or 'none'}",
        f"- selected_agent: {decision.selected_agent_id or 'none'}",
        f"- selected_agent_reason: {decision.selected_agent_reason}",
        f"- dispatch_allowed: {decision.dispatch_allowed}",
        f"- dispatch_executed: {decision.dispatch_executed}",
        f"- approval_required: {decision.approval_required}",
        "",
        "## Safety Checks",
        f"- running_agents: {format_check(checks, 'running_agents')}",
        f"- in_progress_tasks: {format_check(checks, 'in_progress_tasks')}",
        f"- active_runs: {format_check(checks, 'active_runs')}",
        f"- issue_status: {issue_status or format_check(checks, 'issue_status')}",
        f"- agent_allowlisted: {format_check(checks, 'agent_allowlisted')}",
        f"- idempotency_ok: {format_check(checks, 'idempotency_ok')}",
        f"- live_osc_blocked: {format_check(checks, 'live_osc_blocked')}",
        f"- wing_runtime_write_blocked: {format_check(checks, 'wing_runtime_write_blocked')}",
        f"- dispatch_enabled: {format_check(checks, 'dispatch_enabled')}",
        f"- dry_run_off: {format_check(checks, 'dispatch_dry_run_off')}",
        f"- direct_wakeup_forbidden: {format_check(checks, 'direct_wakeup_forbidden')}",
        "",
        "## Decision",
        f"- result: {decision.result}",
        f"- reason: {decision.reason}",
        "",
        "## Intended Action",
        f"- method: {decision.intended_method}",
        f"- endpoint: {decision.intended_endpoint}",
        f"- body: `{json.dumps(decision.intended_body, ensure_ascii=False, sort_keys=True)}`",
        "",
        "## Actual Action",
        f"- executed: {decision.dispatch_executed}",
        f"- response/error: {format_action_response(decision.actual_response, decision.actual_error)}",
        "",
        "## Next Steps",
    ]
    if decision.dispatch_executed:
        lines.append("- Review the assigned Paperclip issue and wait for the assigned agent run/report.")
    elif decision.approval_required:
        lines.append(f"- {decision.approval_note or 'Explicit approval is required before dispatch.'}")
        if decision.approval_command:
            lines.append(f"- approval_dispatch_command: `{decision.approval_command}`")
        else:
            lines.append("- approval_dispatch_command: unavailable until issue and agent are selected.")
    elif decision.result == "dry_run_preview":
        lines.append("- Dry-run preview passed; explicit opt-in is still required before real dispatch.")
    else:
        lines.append("- Fix the blocked check before trying another manual dispatch.")
    lines.append("")
    _ = status
    return "\n".join(lines)


def format_check(checks: dict[str, dict[str, Any]], name: str) -> str:
    check = checks.get(name)
    if not check:
        return "missing"
    return f"{check['value']} (passed={check['passed']})"


def format_action_response(response: Any, error: str | None) -> str:
    if error:
        return error
    if response is None:
        return "none"
    return json.dumps(response, ensure_ascii=False, sort_keys=True)


def shell_quote(value: str) -> str:
    if value and all(ch.isalnum() or ch in "._/:@%+=,-" for ch in value):
        return value
    return "'" + value.replace("'", "'\"'\"'") + "'"


def is_local_paperclip_api(api_url: str) -> bool:
    try:
        parsed = urllib.parse.urlparse(api_url)
    except ValueError:
        return False
    if parsed.scheme != "http":
        return False
    try:
        port = parsed.port
    except ValueError:
        return False
    if port != 3100:
        return False
    return (parsed.hostname or "").lower() in {"127.0.0.1", "localhost", "::1"}


def load_state(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {"version": 1, "days": {}}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {"version": 1, "days": {}}
    if not isinstance(payload, dict):
        return {"version": 1, "days": {}}
    payload.setdefault("version", 1)
    payload.setdefault("days", {})
    return payload


def save_state(path: Path, state: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    tmp_path.write_text(
        json.dumps(state, ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    tmp_path.replace(path)


def day_key(now: datetime) -> str:
    if now.tzinfo is None:
        now = now.replace(tzinfo=timezone.utc)
    return now.astimezone().date().isoformat()


def get_day_state(state: dict[str, Any], day: str) -> dict[str, Any]:
    days = state.setdefault("days", {})
    day_state = days.setdefault(day, {"created_count": 0, "task_ids": []})
    day_state.setdefault("created_count", 0)
    day_state.setdefault("task_ids", [])
    return day_state


def get_cooldown_remaining(state: dict[str, Any], now: datetime, cooldown_seconds: int) -> int:
    if cooldown_seconds <= 0:
        return 0
    raw = state.get("last_created_at")
    if not raw:
        return 0
    try:
        last_created = datetime.fromisoformat(raw)
    except ValueError:
        return 0
    if last_created.tzinfo is None:
        last_created = last_created.replace(tzinfo=timezone.utc)
    elapsed = int((now.astimezone(timezone.utc) - last_created.astimezone(timezone.utc)).total_seconds())
    return max(0, cooldown_seconds - elapsed)


def select_next_task(state: dict[str, Any], day: str) -> WatchdogTask | None:
    day_state = get_day_state(state, day)
    used_today = set(day_state.get("task_ids", []))
    last_key = state.get("last_task_key")
    keys = [task.key for task in SAFE_AUTOMIXER_TASKS]
    start_index = keys.index(last_key) + 1 if last_key in keys else 0

    for offset in range(len(SAFE_AUTOMIXER_TASKS)):
        task = SAFE_AUTOMIXER_TASKS[(start_index + offset) % len(SAFE_AUTOMIXER_TASKS)]
        if task.key not in used_today:
            return task
    return None


def record_created_task(
    state: dict[str, Any],
    day: str,
    task: WatchdogTask,
    response: Any,
    now: datetime,
) -> None:
    day_state = get_day_state(state, day)
    task_ids = list(day_state.get("task_ids", []))
    if task.key not in task_ids:
        task_ids.append(task.key)
    day_state["task_ids"] = task_ids
    day_state["created_count"] = int(day_state.get("created_count", 0)) + 1
    state["last_task_key"] = task.key
    state["last_created_at"] = now.astimezone(timezone.utc).isoformat()
    response_id = extract_response_id(response)
    if response_id:
        state["last_issue_id"] = response_id


def extract_response_id(response: Any) -> str | None:
    if isinstance(response, dict):
        for key in ("id", "issueId", "identifier"):
            value = response.get(key)
            if value:
                return str(value)
        data = response.get("data")
        if isinstance(data, dict):
            return extract_response_id(data)
    return None


def extract_object(payload: Any, candidate_keys: list[str]) -> dict[str, Any]:
    if not isinstance(payload, dict):
        return {}

    for key in candidate_keys:
        value = payload.get(key)
        if isinstance(value, dict):
            return value

    if any(key in payload for key in ("id", "issueId", "status", "state", "title")):
        return payload
    return {}


def extract_list(payload: Any, candidate_keys: list[str]) -> list[dict[str, Any]]:
    if isinstance(payload, list):
        return [item for item in payload if isinstance(item, dict)]
    if not isinstance(payload, dict):
        return []

    for key in candidate_keys:
        if key not in payload:
            continue
        value = payload[key]
        if isinstance(value, list):
            return [item for item in value if isinstance(item, dict)]
        if isinstance(value, dict):
            nested = extract_list(value, candidate_keys)
            if nested:
                return nested

    for value in payload.values():
        if isinstance(value, list) and all(isinstance(item, dict) for item in value):
            return value
    return []


def get_status_value(payload: dict[str, Any]) -> Any:
    for key in ("status", "state", "runStatus", "runtimeStatus"):
        if key in payload:
            return payload[key]
    for key in ("executionState", "runtimeState", "health"):
        value = payload.get(key)
        if isinstance(value, dict):
            nested = get_status_value(value)
            if nested:
                return nested
        elif value:
            return value
    return None


def normalized_status(value: Any) -> str:
    if isinstance(value, dict):
        value = get_status_value(value)
    if value is None:
        return ""
    return str(value).strip().lower().replace("-", "_").replace(" ", "_")


def is_agent_running(agent: dict[str, Any]) -> bool:
    for key in ("isRunning", "running", "busy"):
        if agent.get(key) is True:
            return True
    return normalized_status(get_status_value(agent)) in ACTIVE_STATUSES


def is_agent_error(agent: dict[str, Any]) -> bool:
    if agent.get("lastError") or agent.get("error"):
        return True
    return normalized_status(get_status_value(agent)) in ERROR_STATUSES


def is_run_active(run: dict[str, Any]) -> bool:
    status = normalized_status(get_status_value(run))
    if status in TERMINAL_STATUSES:
        return False
    if status in ACTIVE_STATUSES:
        return True
    return bool(run.get("startedAt") and not (run.get("completedAt") or run.get("finishedAt")))


def issue_matches_run(run: dict[str, Any], issue_id: str) -> bool:
    for key in ("issueId", "issue_id", "taskId", "task_id"):
        if str(run.get(key) or "") == issue_id:
            return True
    issue = run.get("issue")
    if isinstance(issue, dict):
        return str(issue.get("id") or issue.get("issueId") or "") == issue_id
    return False


def get_assignee_agent_id(issue: dict[str, Any]) -> str | None:
    for key in ("assigneeAgentId", "assignedAgentId", "agentId", "assignee_agent_id"):
        value = issue.get(key)
        if value:
            return str(value)
    assignee = issue.get("assignee")
    if isinstance(assignee, dict):
        value = assignee.get("agentId") or assignee.get("id")
        if value:
            return str(value)
    return None


def deep_find_number(payload: Any, candidate_keys: list[str]) -> int | None:
    normalized_keys = {normalize_key(key) for key in candidate_keys}
    if isinstance(payload, dict):
        for key, value in payload.items():
            if normalize_key(key) in normalized_keys and isinstance(value, (int, float)):
                return int(value)
            nested = deep_find_number(value, candidate_keys)
            if nested is not None:
                return nested
    elif isinstance(payload, list):
        for item in payload:
            nested = deep_find_number(item, candidate_keys)
            if nested is not None:
                return nested
    return None


def normalize_key(value: str) -> str:
    return value.replace("_", "").replace("-", "").lower()


def command_status(settings: Settings, logger: logging.Logger) -> int:
    if settings.pause_path.exists():
        print("watchdog paused")
    missing = missing_config(settings)
    if missing:
        message = f"missing required config: {', '.join(missing)}"
        logger.error(message)
        print(message)
        return 2
    api = build_api(settings)
    status = api.get_status()
    print(json.dumps(status.as_dict(), ensure_ascii=False, indent=2, sort_keys=True))
    return 0 if status.api_ok else 1


def command_dispatch_preview(settings: Settings, logger: logging.Logger) -> int:
    decision = run_dispatch(settings, logger=logger, force_dry_run=True)
    print(decision.message)
    if decision.report_path:
        print(f"report: {decision.report_path}")
    return 1 if decision.result == "error" else 0


def command_dispatch(settings: Settings, logger: logging.Logger) -> int:
    decision = run_dispatch(settings, logger=logger)
    print(decision.message)
    if decision.report_path:
        print(f"report: {decision.report_path}")
    return 1 if decision.result == "error" else 0


def command_orchestrate(settings: Settings, logger: logging.Logger) -> int:
    decision = run_orchestration_preview(settings, logger=logger)
    print(decision.message)
    if decision.report_path:
        print(f"report: {decision.report_path}")
    if decision.approval_command:
        print(f"approval dispatch command: {decision.approval_command}")
    else:
        print("approval dispatch command: unavailable")
    return 1 if decision.result == "error" else 0


def command_pause(settings: Settings, logger: logging.Logger) -> int:
    settings.pause_path.parent.mkdir(parents=True, exist_ok=True)
    settings.pause_path.write_text(
        f"paused_at={datetime.now().astimezone().isoformat()}\n",
        encoding="utf-8",
    )
    logger.info("pause file created: %s", settings.pause_path)
    print(f"paused: {settings.pause_path}")
    return 0


def command_resume(settings: Settings, logger: logging.Logger) -> int:
    if settings.pause_path.exists():
        settings.pause_path.unlink()
        logger.info("pause file removed: %s", settings.pause_path)
    print("resumed")
    return 0


def command_loop(settings: Settings, logger: logging.Logger) -> int:
    logger.info(
        "watchdog loop started interval_seconds=%s dry_run=%s mock=%s",
        settings.interval_seconds,
        settings.dry_run,
        settings.mock,
    )
    print(f"watchdog loop started; interval={settings.interval_seconds}s")
    try:
        while True:
            logger.info("heartbeat")
            decision = run_once(settings, logger=logger)
            print(decision.message)
            time.sleep(settings.interval_seconds)
    except KeyboardInterrupt:
        logger.info("watchdog loop stopped by KeyboardInterrupt")
        print("watchdog loop stopped")
        return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Automixer Paperclip watchdog")
    parser.add_argument(
        "command",
        choices=(
            "once",
            "loop",
            "status",
            "orchestrate",
            "dispatch-preview",
            "dispatch",
            "pause",
            "resume",
        ),
        help="command to run",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    settings = Settings.from_env()
    logger = configure_logger(settings)

    if args.command == "once":
        decision = run_once(settings, logger=logger)
        print(decision.message)
        return 0 if decision.action in {"created", "dry_run", "mock_created", "skipped"} else 1
    if args.command == "loop":
        return command_loop(settings, logger)
    if args.command == "status":
        return command_status(settings, logger)
    if args.command == "orchestrate":
        return command_orchestrate(settings, logger)
    if args.command == "dispatch-preview":
        return command_dispatch_preview(settings, logger)
    if args.command == "dispatch":
        return command_dispatch(settings, logger)
    if args.command == "pause":
        return command_pause(settings, logger)
    if args.command == "resume":
        return command_resume(settings, logger)
    parser.error(f"unknown command: {args.command}")
    return 2


if __name__ == "__main__":
    sys.exit(main())
