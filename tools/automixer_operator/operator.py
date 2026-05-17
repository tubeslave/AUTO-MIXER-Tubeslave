#!/usr/bin/env python3
"""One-command safe operator workflow for Automixer/Paperclip."""

from __future__ import annotations

import sys

if sys.path and sys.path[0].replace("\\", "/").endswith("/tools/automixer_operator"):
    sys.path.pop(0)

import argparse
import json
import os
import re
import shlex
import subprocess
import time
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from tools.gpt_control_bridge import server as bridge  # noqa: E402
from tools import agent_specialization  # noqa: E402
from tools.automixer_operator import operational_audit  # noqa: E402


DEFAULT_BRIDGE_CONTEXT_URL = "http://127.0.0.1:8788/v1/context/automixer"
DEFAULT_BRIDGE_TIMEOUT_SECONDS = 1.0
DEFAULT_OPENCLAW_LOCAL_HEALTH_URL = "http://127.0.0.1:18679/"
BRIDGE_TMUX_SESSION = "automixer_gpt_control_bridge"
WATCHDOG_SCRIPT = Path("tools/paperclip_watchdog/automixer_paperclip_watchdog.py")
VECTOR_INDEXER_SCRIPT = Path("tools/project_vector_index/automixer_chromadb_index.py")
REPORT_FIELD_RE = re.compile(r"^- ([A-Za-z0-9_]+):\s*(.*)$")
TERMINAL_ISSUE_STATUSES = {"done", "completed", "closed", "resolved", "cancelled", "canceled"}
PREVIEW_ISSUE_STATUSES = "backlog,todo,in_progress,in_review,blocked"
SAFE_AGENT_ADAPTER_TYPE = "codex_local"
SAFE_AGENT_STATUSES = {"idle", "available", "ready"}
READ_ONLY_HTTP_METHODS = ("GET",)
EXECUTABLE_WRITE_COMMANDS: tuple[str, ...] = ()
LOCAL_PROBE_ALLOWED_HOSTS = {"127.0.0.1", "localhost", "::1"}
VECTOR_OPERATOR_COMMANDS = {
    "vector-index": "index",
    "vector-search": "search",
    "vector-add": "add",
    "vector-stats": "stats",
    "vector-paperclip-sync": "paperclip-sync",
}


@dataclass(frozen=True)
class OperatorConfig:
    project_root: Path
    bridge_context_url: str
    bridge_timeout_seconds: float
    environ: dict[str, str]


@dataclass(frozen=True)
class DispatchReport:
    path: Path
    mode: str | None = None
    issue_id: str | None = None
    selected_agent_id: str | None = None
    idempotency_key: str | None = None
    dispatch_executed: bool | None = None
    result: str | None = None
    reason: str | None = None
    intended_endpoint: str | None = None
    intended_body: dict[str, Any] | None = None

    @property
    def is_dry_run(self) -> bool:
        return self.mode == "dry_run" and self.dispatch_executed is not True


@dataclass(frozen=True)
class MonitorDetails:
    issue: dict[str, Any] | None
    issue_error: str | None
    runs: list[dict[str, Any]]
    runs_error: str | None
    comments: list[dict[str, Any]]
    comments_error: str | None


@dataclass(frozen=True)
class TargetSelection:
    issue: dict[str, Any] | None
    agent: dict[str, Any] | None
    reason: str
    errors: list[str]
    specialization_key: str | None = None
    specialization_label: str | None = None

    @property
    def ready(self) -> bool:
        return bool(self.issue and self.agent)


def build_config(environ: dict[str, str] | None = None) -> OperatorConfig:
    env = dict(environ if environ is not None else os.environ)
    project_root = Path(env.get("AUTOMIXER_PROJECT_ROOT", str(PROJECT_ROOT))).expanduser().resolve()
    return OperatorConfig(
        project_root=project_root,
        bridge_context_url=env.get("GPT_CONTROL_BRIDGE_CONTEXT_URL", DEFAULT_BRIDGE_CONTEXT_URL),
        bridge_timeout_seconds=parse_float(
            env.get("AUTOMIXER_OPERATOR_BRIDGE_TIMEOUT_SECONDS"),
            DEFAULT_BRIDGE_TIMEOUT_SECONDS,
        ),
        environ=env,
    )


def parse_float(value: str | None, default: float) -> float:
    if not value:
        return default
    try:
        parsed = float(value)
    except ValueError:
        return default
    return parsed if parsed > 0 else default


def load_context(config: OperatorConfig) -> dict[str, Any]:
    payload, error = fetch_bridge_context_http(config)
    if payload is not None:
        payload["_operator_context_source"] = "gpt_control_bridge_http"
        return payload

    direct = bridge.build_context(
        bridge.BridgeConfig(
            paperclip_api_url=config.environ.get(
                "PAPERCLIP_API_URL",
                bridge.DEFAULT_PAPERCLIP_API_URL,
            ).rstrip("/"),
            project_root=config.project_root,
            paperclip_company_id=config.environ.get("PAPERCLIP_COMPANY_ID") or None,
            paperclip_timeout_seconds=parse_float(
                config.environ.get("GPT_CONTROL_BRIDGE_PAPERCLIP_TIMEOUT_SECONDS"),
                3.0,
            ),
        )
    )
    direct["_operator_context_source"] = "direct_bridge_module"
    direct["_operator_bridge_http_error"] = error
    return direct


def fetch_bridge_context_http(config: OperatorConfig) -> tuple[dict[str, Any] | None, str | None]:
    try:
        request = urllib.request.Request(config.bridge_context_url, method="GET")
        with urllib.request.urlopen(request, timeout=config.bridge_timeout_seconds) as response:
            raw = response.read().decode("utf-8", errors="replace")
        payload = json.loads(raw)
    except (urllib.error.URLError, TimeoutError, OSError, json.JSONDecodeError, ValueError) as exc:
        return None, f"{type(exc).__name__}: {exc}"
    if isinstance(payload, dict):
        return payload, None
    return None, "bridge context response is not an object"


def fetch_bridge_health(config: OperatorConfig) -> tuple[dict[str, Any] | None, str | None]:
    try:
        request = urllib.request.Request(bridge_health_url(config.bridge_context_url), method="GET")
        with urllib.request.urlopen(request, timeout=config.bridge_timeout_seconds) as response:
            raw = response.read().decode("utf-8", errors="replace")
        payload = json.loads(raw)
    except (urllib.error.URLError, TimeoutError, OSError, json.JSONDecodeError, ValueError) as exc:
        return None, f"{type(exc).__name__}: {exc}"
    if isinstance(payload, dict):
        return payload, None
    return None, "bridge health response is not an object"


def probe_openclaw_local_gateway(config: OperatorConfig) -> dict[str, Any]:
    return probe_local_http_url(
        config.environ.get(
            "AUTOMIXER_OPERATOR_OPENCLAW_LOCAL_HEALTH_URL",
            DEFAULT_OPENCLAW_LOCAL_HEALTH_URL,
        ),
        parse_float(
            config.environ.get("AUTOMIXER_OPERATOR_LOCAL_PROBE_TIMEOUT_SECONDS"),
            1.0,
        ),
    )


def probe_local_http_url(url: str, timeout_seconds: float) -> dict[str, Any]:
    display_url = local_probe_display_url(url)
    parsed = urllib.parse.urlparse(url)
    host = parsed.hostname or ""
    if parsed.scheme != "http" or host not in LOCAL_PROBE_ALLOWED_HOSTS:
        return {
            "url": display_url,
            "allowed_local_only": False,
            "reachable": False,
            "ok": False,
            "status_code": None,
            "openclaw_ui_detected": False,
            "error": "blocked: local probes only allow http loopback URLs",
        }

    try:
        request = urllib.request.Request(url, method="GET")
        with urllib.request.urlopen(request, timeout=timeout_seconds) as response:
            status_code = int(getattr(response, "status", response.getcode()))
            raw = response.read(1024).decode("utf-8", errors="replace")
        return {
            "url": display_url,
            "allowed_local_only": True,
            "reachable": True,
            "ok": 200 <= status_code < 400,
            "status_code": status_code,
            "openclaw_ui_detected": "openclaw" in raw.lower(),
            "error": None,
        }
    except urllib.error.HTTPError as exc:
        raw = exc.read(1024).decode("utf-8", errors="replace")
        return {
            "url": display_url,
            "allowed_local_only": True,
            "reachable": True,
            "ok": 200 <= int(exc.code) < 400,
            "status_code": int(exc.code),
            "openclaw_ui_detected": "openclaw" in raw.lower(),
            "error": f"HTTPError: {exc.code}",
        }
    except (urllib.error.URLError, TimeoutError, OSError, ValueError) as exc:
        return {
            "url": display_url,
            "allowed_local_only": True,
            "reachable": False,
            "ok": False,
            "status_code": None,
            "openclaw_ui_detected": False,
            "error": f"{type(exc).__name__}: {exc}",
        }


def local_probe_display_url(url: str) -> str:
    parsed = urllib.parse.urlparse(url)
    host = parsed.hostname or ""
    if not host:
        return "invalid-url"
    if ":" in host and not host.startswith("["):
        netloc = f"[{host}]"
    else:
        netloc = host
    if parsed.port:
        netloc = f"{netloc}:{parsed.port}"
    return urllib.parse.urlunparse(
        (
            parsed.scheme or "http",
            netloc,
            parsed.path or "/",
            "",
            "",
            "",
        )
    )


def bridge_health_url(context_url: str) -> str:
    parsed = urllib.parse.urlparse(context_url)
    return urllib.parse.urlunparse(
        (
            parsed.scheme or "http",
            parsed.netloc or "127.0.0.1:8788",
            "/health",
            "",
            "",
            "",
        )
    )


def bridge_host_port(context_url: str) -> tuple[str, int]:
    parsed = urllib.parse.urlparse(context_url)
    host = parsed.hostname or "127.0.0.1"
    port = parsed.port or 8788
    return host, port


def command_status(config: OperatorConfig) -> int:
    context = load_context(config)
    print(render_status(config, context))
    return 0


def command_bridge_status(config: OperatorConfig) -> int:
    health, health_error = fetch_bridge_health(config)
    lines = [
        "== Automixer Bridge Status ==",
        f"health_url: {bridge_health_url(config.bridge_context_url)}",
        f"reachable: {health_error is None}",
        f"error: {health_error or 'none'}",
    ]
    if isinstance(health, dict):
        lines.extend(
            [
                f"ok: {health.get('ok')}",
                f"read_only: {health.get('read_only')}",
                f"service: {health.get('service')}",
            ]
        )
    print("\n".join(lines))
    return 0 if health_error is None and isinstance(health, dict) and health.get("read_only") is True else 1


def command_bridge_start(config: OperatorConfig) -> int:
    host, port = bridge_host_port(config.bridge_context_url)
    if host not in {"127.0.0.1", "localhost", "::1"}:
        print("bridge_start: refused")
        print(f"reason: bridge host must be loopback, got {host}")
        return 1

    health, health_error = fetch_bridge_health(config)
    if health_error is None and isinstance(health, dict) and health.get("read_only") is True:
        print("bridge_start: already_running")
        print(f"health_url: {bridge_health_url(config.bridge_context_url)}")
        return 0

    session = tmux_session_exists(BRIDGE_TMUX_SESSION)
    if session:
        print("bridge_start: blocked")
        print(f"reason: tmux session already exists but health failed: {health_error}")
        print(f"session: {BRIDGE_TMUX_SESSION}")
        return 1

    command = (
        f"cd {shlex.quote(str(config.project_root))} && "
        f"{shlex.quote(sys.executable)} tools/gpt_control_bridge/server.py "
        f"--host {shlex.quote(host)} --port {port}"
    )
    result = subprocess.run(
        ["tmux", "new-session", "-d", "-s", BRIDGE_TMUX_SESSION, command],
        cwd=str(config.project_root),
        text=True,
        capture_output=True,
        check=False,
    )
    if result.returncode != 0:
        print("bridge_start: failed")
        print((result.stderr or result.stdout or "tmux failed").strip())
        return result.returncode

    for _ in range(20):
        health, health_error = fetch_bridge_health(config)
        if health_error is None and isinstance(health, dict) and health.get("read_only") is True:
            print("bridge_start: started")
            print(f"session: {BRIDGE_TMUX_SESSION}")
            print(f"health_url: {bridge_health_url(config.bridge_context_url)}")
            return 0
        time.sleep(0.1)

    print("bridge_start: failed")
    print(f"reason: health did not become ready: {health_error}")
    return 1


def tmux_session_exists(name: str) -> bool:
    result = subprocess.run(
        ["tmux", "has-session", "-t", name],
        text=True,
        capture_output=True,
        check=False,
    )
    return result.returncode == 0


def command_doctor(config: OperatorConfig) -> int:
    health, health_error = fetch_bridge_health(config)
    context = load_context(config)
    safety = context.get("safety_flags") if isinstance(context.get("safety_flags"), dict) else {}
    checks = {
        "bridge_http_ok": health_error is None,
        "bridge_read_only": isinstance(health, dict) and health.get("read_only") is True,
        "paperclip_api_ok": context.get("paperclip_api_ok") is True,
        "write_routes_disabled": safety.get("write_routes_enabled") is False,
        "dispatch_blocked": safety.get("dispatch_allowed") is False,
        "wing_osc_blocked": safety.get("wing_osc_allowed") is False,
    }
    lines = [
        "== Automixer Operator Doctor ==",
        f"bridge_health_url: {bridge_health_url(config.bridge_context_url)}",
        f"bridge_health_error: {health_error or 'none'}",
        f"context_source: {context.get('_operator_context_source', 'unknown')}",
        "",
        "Checks:",
    ]
    for name, passed in checks.items():
        lines.append(f"- {name}: {passed}")
    lines.extend(["", render_recommended_action(context)])
    print("\n".join(lines))
    return 0 if all(checks.values()) else 1


def command_next(config: OperatorConfig) -> int:
    context = load_context(config)
    report = latest_dry_run_report(config.project_root)
    target = select_preview_target(config, context)
    direct_counts, _direct_counts_error = load_issue_status_counts(config, context)
    action, reason, command = next_operator_step(
        context,
        report,
        target,
        direct_counts,
        latest_operational_audit(config.project_root),
    )
    lines = [
        "== Automixer Operator Next ==",
        f"action: {action}",
        f"reason: {reason}",
        f"command: {command}",
        "",
        render_target_selection(target),
    ]
    print("\n".join(lines))
    return 0


def next_operator_step(
    context: dict[str, Any],
    report: DispatchReport | None,
    target: TargetSelection | None = None,
    direct_counts: dict[str, int] | None = None,
    audit: dict[str, Any] | None = None,
) -> tuple[str, str, str]:
    if context.get("paperclip_api_ok") is not True:
        return (
            "check_paperclip",
            "Paperclip is unavailable; do not bypass Paperclip.",
            "curl -s http://127.0.0.1:3100/api/health",
        )

    active = context.get("active_runs_summary")
    if isinstance(active, dict) and int(active.get("total") or 0) > 0:
        return (
            "monitor",
            "Active run detected; observe instead of dispatching new work.",
            "python3 tools/automixer_operator/operator.py monitor",
        )

    if report and approval_command_allowed(report):
        return (
            "review_approval_template",
            "Latest dry-run preview is ready; user must explicitly approve before real dispatch.",
            "python3 tools/automixer_operator/operator.py approve-template",
        )

    if target is not None and not target.ready:
        blocked = queue_count(context, "blocked", direct_counts)
        todo = queue_count(context, "todo", direct_counts)
        in_progress = queue_count(context, "in_progress", direct_counts)
        if todo == 0 and in_progress == 0 and (blocked or 0) > 0:
            priority = first_priority_action(audit)
            if priority:
                priority_reason = str(priority.get("reason") or "review audit").rstrip(".")
                return (
                    str(priority.get("key") or "review_operational_audit"),
                    (
                        f"{priority.get('severity') or 'unknown'} priority: {priority_reason}. "
                        f"Target: {priority.get('target') or 'none'}. "
                        f"Evidence: {priority.get('evidence') or 'none'}."
                    ),
                    priority_command(priority),
                )
            return (
                "blocked",
                (
                    f"Paperclip has {blocked} blocked issues and no safe todo target; "
                    "inspect blockers before creating or dispatching new work."
                ),
                "python3 tools/automixer_operator/operator.py blocked",
            )
        return (
            "create_or_select_new_todo_issue",
            target.reason,
            "Create/select an unassigned Paperclip todo issue, then run: python3 tools/automixer_operator/operator.py cycle",
        )

    if target is not None and target.ready:
        issue_label = format_issue_label(target.issue)
        agent_label = format_agent_label(target.agent)
        if report and report.result and report.result != "dry_run_preview":
            return (
                "run_safe_cycle",
                f"Latest dry-run report is blocked/stale, but a safe target is ready: {issue_label} -> {agent_label}.",
                "python3 tools/automixer_operator/operator.py cycle",
            )
        return (
            "run_safe_cycle",
            f"Safe dry-run target is ready: {issue_label} -> {agent_label}.",
            "python3 tools/automixer_operator/operator.py cycle",
        )

    if report and report.result and report.result != "dry_run_preview":
        return (
            "create_or_select_new_todo_issue",
            f"Latest dry-run report is blocked: {report.reason or report.result}.",
            "Create/select a Paperclip todo issue, then run: python3 tools/automixer_operator/operator.py cycle",
        )

    return (
        "run_safe_cycle",
        "No active run and no ready approval template; run a full dry-run cycle.",
        "python3 tools/automixer_operator/operator.py cycle",
    )


def first_priority_action(audit: dict[str, Any] | None) -> dict[str, Any] | None:
    if not isinstance(audit, dict):
        return None
    actions = audit.get("priority_actions")
    if not isinstance(actions, list):
        return None
    return next((item for item in actions if isinstance(item, dict)), None)


def priority_command(action: dict[str, Any]) -> str:
    key = str(action.get("key") or "")
    if key == "triage_blocked_queue":
        return "python3 tools/automixer_operator/operator.py blocked"
    if key == "repair_unhealthy_agents":
        return "python3 tools/automixer_operator/operator.py agent-health"
    return "python3 tools/automixer_operator/operator.py operational-audit"


def command_target(config: OperatorConfig) -> int:
    context = load_context(config)
    target = select_preview_target(config, context)
    print("== Automixer Operator Target ==")
    print(render_target_selection(target))
    return 0


def command_preview(config: OperatorConfig) -> int:
    print("== Automixer Operator Preview ==")
    print("mode: dry-run")
    print("write actions: disabled")
    context = load_context(config)
    target = select_preview_target(config, context)
    print("")
    print(render_target_selection(target))
    result = run_watchdog_preview(config, context=context, target=target)
    if result.stdout.strip():
        print("")
        print("watchdog_stdout:")
        print(indent(result.stdout.rstrip()))
    if result.stderr.strip():
        print("")
        print("watchdog_stderr:")
        print(indent(result.stderr.rstrip()))

    context = load_context(config)
    print("")
    print(render_recommended_action(context))

    report = latest_dry_run_report(config.project_root)
    if report is None:
        print("")
        print("approval_command: unavailable")
        print("reason: no dry-run watchdog dispatch report found")
    else:
        print("")
        print(render_report_summary(report))
        print("")
        print(render_approval_section(report))
    return result.returncode


def command_cycle(config: OperatorConfig) -> int:
    print("== Automixer Operator Safe Cycle ==")
    print("write actions: disabled")
    print("real dispatch: not available in this command")
    print("")
    print("[1/5] status/context")
    before_context = load_context(config)
    print(indent(render_status(config, before_context)))

    print("")
    print("[2/5] watchdog dry-run preview")
    target = select_preview_target(config, before_context)
    print(indent(render_target_selection(target)))
    result = run_watchdog_preview(config, context=before_context, target=target)
    if result.stdout.strip():
        print("watchdog_stdout:")
        print(indent(result.stdout.rstrip()))
    if result.stderr.strip():
        print("watchdog_stderr:")
        print(indent(result.stderr.rstrip()))

    print("")
    print("[3/5] refreshed recommendation")
    after_context = load_context(config)
    print(indent(render_recommended_action(after_context)))

    print("")
    print("[4/5] latest dry-run report")
    report = latest_dry_run_report(config.project_root)
    if report is None:
        print(indent("unavailable: no dry-run watchdog dispatch report found"))
        print("")
        print("[5/5] approval command")
        print(indent("unavailable"))
        return result.returncode if result.returncode else 1

    print(indent(render_report_summary(report)))
    print("")
    print("[5/5] approval command")
    print(indent(render_approval_section(report)))
    return result.returncode


def run_watchdog_preview(
    config: OperatorConfig,
    *,
    context: dict[str, Any] | None = None,
    target: TargetSelection | None = None,
) -> subprocess.CompletedProcess[str]:
    env = dict(config.environ)
    env.update(safety_preview_env())
    apply_preview_context_defaults(env, context or {}, config=config, target=target)
    return subprocess.run(
        [sys.executable, str(WATCHDOG_SCRIPT), "dispatch-preview"],
        cwd=str(config.project_root),
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )


def apply_preview_context_defaults(
    env: dict[str, str],
    context: dict[str, Any],
    *,
    config: OperatorConfig | None = None,
    target: TargetSelection | None = None,
) -> None:
    company_id = clean_missing(str(context.get("company_id") or ""))
    if company_id:
        env.setdefault("PAPERCLIP_COMPANY_ID", company_id)

    target = target if target is not None else (
        select_preview_target(config, context) if config is not None else None
    )
    defaults = dispatch_defaults_from_target(target) if target and target.ready else {}
    if not defaults:
        defaults = dispatch_defaults_from_context(context)
    issue_id = clean_missing(str(defaults.get("issue_id") or ""))
    if issue_id:
        env.setdefault("AUTOMIXER_WATCHDOG_DISPATCH_ISSUE_ID", issue_id)

    agent_id = clean_missing(str(defaults.get("agent_id") or ""))
    if agent_id:
        env.setdefault("AUTOMIXER_WATCHDOG_DISPATCH_AGENT_ID", agent_id)
        env.setdefault("AUTOMIXER_WATCHDOG_DISPATCH_ALLOWLIST_AGENT_IDS", agent_id)

    task_key = clean_missing(str(defaults.get("task_key") or ""))
    if task_key:
        env.setdefault("AUTOMIXER_WATCHDOG_DISPATCH_TASK_KEY", task_key)


def dispatch_defaults_from_target(target: TargetSelection | None) -> dict[str, str | None]:
    if not target or not target.ready:
        return {}
    issue_id = get_issue_id(target.issue or {})
    agent_id = get_agent_id(target.agent or {})
    return {
        "issue_id": issue_id,
        "agent_id": agent_id,
        "task_key": target_task_key(target.issue or {}),
    }


def select_preview_target(config: OperatorConfig, context: dict[str, Any]) -> TargetSelection:
    if context.get("paperclip_api_ok") is not True:
        return TargetSelection(None, None, "Paperclip is unavailable; target selection is blocked.", [])

    active = context.get("active_runs_summary")
    if isinstance(active, dict):
        try:
            active_total = int(active.get("total") or 0)
        except (TypeError, ValueError):
            active_total = 0
        if active.get("available") and active_total > 0:
            return TargetSelection(None, None, "Active runs exist; monitor before selecting new work.", [])
        if not active.get("available"):
            return TargetSelection(None, None, "Active runs are unavailable; target selection stays conservative.", [])

    company_id = clean_missing(str(context.get("company_id") or ""))
    if not company_id:
        return TargetSelection(None, None, "Paperclip company_id is unavailable.", [])

    client = bridge.PaperclipClient(
        str(context.get("paperclip_api_url") or bridge.DEFAULT_PAPERCLIP_API_URL),
        parse_float(
            config.environ.get("GPT_CONTROL_BRIDGE_PAPERCLIP_TIMEOUT_SECONDS"),
            3.0,
        ),
    )
    issues_payload, issues_error = bridge.safe_get(
        client,
        f"/api/companies/{bridge.quote(company_id)}/issues",
        params={"status": PREVIEW_ISSUE_STATUSES, "limit": 500},
    )
    agents_payload, agents_error = bridge.safe_get(
        client,
        f"/api/companies/{bridge.quote(company_id)}/agents",
    )
    errors = [error for error in (issues_error, agents_error) if error]
    issues = bridge.extract_items(issues_payload, ["issues", "data", "items", "results"])
    agents = bridge.extract_items(agents_payload, ["agents", "data", "items", "results"])
    safe_issues = sorted([issue for issue in issues if is_safe_issue_candidate(issue)], key=issue_preference_key)
    safe_agents = sorted(
        [agent for agent in agents if is_safe_agent_candidate(agent)],
        key=agent_preference_key,
    )

    if not safe_issues or not safe_agents:
        missing = []
        if not safe_issues:
            missing.append("unassigned todo issue")
        if not safe_agents:
            missing.append("idle codex_local agent")
        reason = "No safe preview target found: missing " + " and ".join(missing) + "."
        if errors:
            reason = f"{reason} Read errors: {', '.join(errors)}"
        return TargetSelection(None, None, reason, errors)

    best_issue: dict[str, Any] | None = None
    best_selection: agent_specialization.SpecialistSelection | None = None
    missing_roles: list[str] = []
    for issue in safe_issues:
        selection = agent_specialization.select_specialist_for_issue(issue, safe_agents)
        if selection.agent:
            best_issue = issue
            best_selection = selection
            break
        missing_roles.append(selection.specialization.label)

    if best_issue is None or best_selection is None:
        reason = (
            "No safe preview target found: matching specialist is not ready. "
            f"Missing roles: {', '.join(sorted(set(missing_roles))) or 'unknown'}."
        )
        if errors:
            reason = f"{reason} Read errors: {', '.join(errors)}"
        return TargetSelection(None, None, reason, errors)

    return TargetSelection(
        best_issue,
        best_selection.agent,
        (
            f"Selected {format_issue_label(best_issue)} for {format_agent_label(best_selection.agent)} "
            f"via {best_selection.specialization.label}."
        ),
        errors,
        specialization_key=best_selection.specialization.key,
        specialization_label=best_selection.specialization.label,
    )


def is_safe_issue_candidate(issue: dict[str, Any]) -> bool:
    if not get_issue_id(issue):
        return False
    if normalize_status(issue.get("status") or issue.get("state")) != "todo":
        return False
    if get_issue_assignee(issue):
        return False
    return not issue_has_active_run(issue)


def is_safe_agent_candidate(agent: dict[str, Any]) -> bool:
    if not get_agent_id(agent):
        return False
    if normalize_status(agent.get("adapterType") or agent.get("adapter_type")) != SAFE_AGENT_ADAPTER_TYPE:
        return False
    return normalize_status(agent.get("status") or agent.get("state")) in SAFE_AGENT_STATUSES


def issue_preference_key(issue: dict[str, Any]) -> tuple[int, int, str, str]:
    priority_rank = {
        "critical": 0,
        "urgent": 0,
        "high": 1,
        "medium": 2,
        "normal": 3,
        "low": 4,
    }.get(normalize_status(issue.get("priority")), 3)
    return (priority_rank, -issue_number(issue), issue_title(issue).lower(), get_issue_id(issue) or "")


def agent_preference_key(agent: dict[str, Any]) -> tuple[int, int, str, str]:
    agent_id = get_agent_id(agent) or ""
    name = agent_name(agent).lower()
    specialization = agent_specialization.specialization_for_agent(agent)
    preferred = bool(specialization and specialization.execution_role)
    status = normalize_status(agent.get("status") or agent.get("state"))
    return (0 if preferred else 1, 0 if status == "idle" else 1, name, agent_id)


def get_issue_id(issue: dict[str, Any]) -> str | None:
    for key in ("id", "issueId", "issue_id", "uuid"):
        value = clean_missing(str(issue.get(key) or ""))
        if value:
            return value
    return None


def issue_title(issue: dict[str, Any]) -> str:
    return clean_missing(str(issue.get("title") or issue.get("name") or "")) or "untitled"


def issue_identifier(issue: dict[str, Any]) -> str | None:
    for key in ("identifier", "key", "issueKey", "issue_key"):
        value = clean_missing(str(issue.get(key) or ""))
        if value:
            return value
    number = issue_number(issue)
    prefix = clean_missing(str(issue.get("issuePrefix") or issue.get("prefix") or "TUB"))
    return f"{prefix}-{number}" if number > 0 and prefix else None


def issue_number(issue: dict[str, Any]) -> int:
    for key in ("issueNumber", "issue_number", "number"):
        value = parse_intish(issue.get(key))
        if value > 0:
            return value
    for key in ("identifier", "key", "issueKey", "issue_key", "title"):
        match = re.search(r"\b[A-Z]+-(\d+)\b", str(issue.get(key) or ""))
        if match:
            return parse_intish(match.group(1))
    return 0


def parse_intish(value: Any) -> int:
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        return 0
    return parsed if parsed > 0 else 0


def get_issue_assignee(issue: dict[str, Any]) -> str | None:
    for key in ("assigneeAgentId", "assignedAgentId", "agentId", "assignee_agent_id", "assigneeId"):
        value = clean_missing(str(issue.get(key) or ""))
        if value:
            return value
    for key in ("assigneeAgent", "assignee", "agent"):
        value = issue.get(key)
        if isinstance(value, dict):
            nested = get_agent_id(value) or clean_missing(str(value.get("id") or value.get("name") or ""))
            if nested:
                return nested
    return None


def issue_has_active_run(issue: dict[str, Any]) -> bool:
    for key in ("activeRunId", "active_run_id", "currentRunId", "current_run_id"):
        if clean_missing(str(issue.get(key) or "")):
            return True
    for key in ("activeRun", "active_run", "currentRun", "current_run"):
        value = issue.get(key)
        if isinstance(value, dict):
            status = normalize_status(value.get("status") or value.get("state"))
            return status not in bridge.TERMINAL_RUN_STATUSES
        if isinstance(value, list):
            return any(isinstance(item, dict) and normalize_status(item.get("status") or item.get("state")) not in bridge.TERMINAL_RUN_STATUSES for item in value)
        if clean_missing(str(value or "")):
            return True
    return False


def get_agent_id(agent: dict[str, Any]) -> str | None:
    for key in ("id", "agentId", "agent_id", "uuid"):
        value = clean_missing(str(agent.get(key) or ""))
        if value:
            return value
    return None


def agent_name(agent: dict[str, Any]) -> str:
    return clean_missing(str(agent.get("name") or agent.get("agentName") or agent.get("title") or agent.get("role") or "")) or "unnamed agent"


def target_task_key(issue: dict[str, Any]) -> str | None:
    return issue_identifier(issue) or get_issue_id(issue)


def format_issue_label(issue: dict[str, Any] | None) -> str:
    if not issue:
        return "none"
    identifier = issue_identifier(issue)
    issue_id = get_issue_id(issue) or "missing-id"
    title = issue_title(issue)
    return f"{identifier or issue_id} ({issue_id}): {title}"


def format_agent_label(agent: dict[str, Any] | None) -> str:
    if not agent:
        return "none"
    return f"{agent_name(agent)} ({get_agent_id(agent) or 'missing-id'})"


def render_target_selection(target: TargetSelection) -> str:
    lines = [
        "Preview target:",
        f"- ready: {target.ready}",
        f"- reason: {target.reason}",
    ]
    if target.issue:
        lines.append(f"- issue: {format_issue_label(target.issue)}")
        lines.append(f"- issue_status: {normalize_status(target.issue.get('status') or target.issue.get('state')) or 'unknown'}")
    if target.agent:
        lines.append(f"- agent: {format_agent_label(target.agent)}")
        lines.append(f"- agent_status: {normalize_status(target.agent.get('status') or target.agent.get('state')) or 'unknown'}")
        lines.append(f"- agent_adapter: {target.agent.get('adapterType') or target.agent.get('adapter_type') or 'unknown'}")
    if target.specialization_label:
        lines.append(f"- specialization: {target.specialization_label}")
    if target.errors:
        lines.append(f"- read_errors: {', '.join(target.errors)}")
    return "\n".join(lines)


def dispatch_defaults_from_context(context: dict[str, Any]) -> dict[str, str | None]:
    state_summary = context.get("watchdog_state_summary")
    dispatch = state_summary.get("dispatch") if isinstance(state_summary, dict) else {}
    if not isinstance(dispatch, dict):
        dispatch = {}
    defaults = {
        "issue_id": clean_missing(str(dispatch.get("last_issue_id") or "")),
        "agent_id": clean_missing(str(dispatch.get("last_selected_agent_id") or "")),
        "task_key": clean_missing(str(dispatch.get("last_task_key") or "")),
    }

    state_path = state_summary.get("path") if isinstance(state_summary, dict) else None
    state = read_watchdog_state(Path(str(state_path or "")))
    full_dispatch = state.get("dispatch") if isinstance(state.get("dispatch"), dict) else {}
    attempts = full_dispatch.get("attempts") if isinstance(full_dispatch.get("attempts"), list) else []
    for attempt in reversed(attempts):
        if not isinstance(attempt, dict):
            continue
        issue_id = clean_missing(str(attempt.get("issue_id") or ""))
        agent_id = clean_missing(str(attempt.get("selected_agent_id") or ""))
        if not issue_id or not agent_id:
            continue
        return {
            "issue_id": issue_id,
            "agent_id": agent_id,
            "task_key": clean_missing(str(attempt.get("task_key") or "")) or defaults["task_key"],
        }

    return defaults


def safety_preview_env() -> dict[str, str]:
    return {
        "AUTOMIXER_WATCHDOG_DRY_RUN": "true",
        "AUTOMIXER_WATCHDOG_DISPATCH_ENABLED": "true",
        "AUTOMIXER_WATCHDOG_DISPATCH_DRY_RUN": "true",
        "AUTOMIXER_WATCHDOG_DISPATCH_METHOD": "patch_issue_assignment",
        "AUTOMIXER_WATCHDOG_DISPATCH_FORBID_DIRECT_WAKEUP_WITHOUT_ASSIGNMENT": "true",
        "AUTOMIXER_WATCHDOG_DISPATCH_FORBID_LIVE_OSC": "true",
        "AUTOMIXER_WATCHDOG_DISPATCH_FORBID_WING_RUNTIME_WRITE": "true",
    }


def command_approve_template(config: OperatorConfig) -> int:
    report = latest_dry_run_report(config.project_root)
    if report is None:
        print("approval_command: unavailable")
        print("reason: no dry-run watchdog dispatch report found")
        return 1
    if not approval_command_allowed(report):
        print(render_report_summary(report))
        print(render_approval_section(report))
        return 1
    print("approval_command:")
    print(indent(build_approval_command(report)))
    return 0


def command_monitor(config: OperatorConfig) -> int:
    context = load_context(config)
    report = latest_dispatch_report(config.project_root)
    issue_id = resolve_issue_id(context, report)
    details = load_monitor_details(config, context, issue_id)
    print(render_monitor(context, report, issue_id, details))
    return 0


def command_blocked(config: OperatorConfig) -> int:
    context = load_context(config)
    issues, error = load_blocked_issues(config, context)
    print(render_blocked_queue(context, issues, error))
    return 0 if error is None else 1


def command_agent_health(config: OperatorConfig) -> int:
    context = load_context(config)
    details = load_agent_health_details(config, context)
    print(render_agent_health(context, details))
    return 0 if details.get("error") is None else 1


def load_agent_health_details(config: OperatorConfig, context: dict[str, Any]) -> dict[str, Any]:
    if context.get("paperclip_api_ok") is not True:
        return {"error": "Paperclip API is unavailable"}
    company_id = clean_missing(str(context.get("company_id") or ""))
    if not company_id:
        return {"error": "company_id is unavailable"}

    client = bridge.PaperclipClient(
        str(context.get("paperclip_api_url") or bridge.DEFAULT_PAPERCLIP_API_URL),
        parse_float(
            config.environ.get("GPT_CONTROL_BRIDGE_PAPERCLIP_TIMEOUT_SECONDS"),
            3.0,
        ),
    )
    agents_payload, agents_error = bridge.safe_get(
        client,
        f"/api/companies/{bridge.quote(company_id)}/agents",
    )
    issues_payload, issues_error = bridge.safe_get(
        client,
        f"/api/companies/{bridge.quote(company_id)}/issues",
        params={"status": PREVIEW_ISSUE_STATUSES, "limit": 500},
    )
    if agents_error or issues_error:
        return {"error": "; ".join(error for error in (agents_error, issues_error) if error)}

    agents = bridge.extract_items(agents_payload, ["agents", "data", "items", "results"])
    issues = bridge.extract_items(issues_payload, ["issues", "data", "items", "results"])
    audit = latest_operational_audit(config.project_root) or {}
    unhealthy_audit_items = audit.get("unhealthy_agents") if isinstance(audit.get("unhealthy_agents"), list) else []
    unhealthy_ids = {
        clean_missing(str(item.get("id") or ""))
        for item in unhealthy_audit_items
        if isinstance(item, dict)
    }
    unhealthy_names = {
        clean_missing(str(item.get("name") or ""))
        for item in unhealthy_audit_items
        if isinstance(item, dict)
    }
    unhealthy_ids.discard(None)
    unhealthy_names.discard(None)

    selected_agents = [
        agent
        for agent in agents
        if is_agent_unhealthy_by_audit_or_status(agent, unhealthy_ids, unhealthy_names)
    ]
    selected_agents = sorted(selected_agents, key=lambda agent: (agent_name(agent).lower(), get_agent_id(agent) or ""))
    details = []
    for agent in selected_agents:
        agent_id = get_agent_id(agent)
        assigned_issues = [
            issue
            for issue in issues
            if get_issue_assignee(issue) == agent_id
        ]
        detail_payload, detail_error = ({}, None)
        if agent_id:
            detail_payload, detail_error = bridge.safe_get(client, f"/api/agents/{bridge.quote(agent_id)}")
        details.append(
            {
                "agent": bridge.summarize_record(agent, ["id", "name", "title", "role", "status", "adapterType", "lastHeartbeatAt", "updatedAt"]),
                "audit": find_unhealthy_audit_item(agent, unhealthy_audit_items),
                "assigned_issue_count": len(assigned_issues),
                "assigned_issues": [bridge.summarize_record(issue, ["id", "identifier", "title", "status", "priority", "lastActivityAt", "updatedAt"]) for issue in assigned_issues[:10]],
                "detail_error": detail_error,
                "safe_adapter_config": safe_agent_adapter_config(detail_payload if isinstance(detail_payload, dict) else {}),
                "local_gateway_probe": probe_openclaw_local_gateway(config)
                if str(agent.get("adapterType") or agent.get("adapter_type") or "") == "openclaw_gateway"
                else None,
            }
        )
    return {
        "error": None,
        "audit_generated_at": audit.get("generated_at") if isinstance(audit, dict) else None,
        "unhealthy_agent_count": len(details),
        "agents": details,
    }


def is_agent_unhealthy_by_audit_or_status(
    agent: dict[str, Any],
    unhealthy_ids: set[str | None],
    unhealthy_names: set[str | None],
) -> bool:
    agent_id = get_agent_id(agent)
    name = agent_name(agent)
    if agent_id and agent_id in unhealthy_ids:
        return True
    if name and name in unhealthy_names:
        return True
    return bool(agent_attention_items([agent]))


def find_unhealthy_audit_item(
    agent: dict[str, Any],
    unhealthy_audit_items: list[Any],
) -> dict[str, Any]:
    agent_id = get_agent_id(agent)
    name = agent_name(agent)
    for item in unhealthy_audit_items:
        if not isinstance(item, dict):
            continue
        if agent_id and clean_missing(str(item.get("id") or "")) == agent_id:
            return item
        if name and clean_missing(str(item.get("name") or "")) == name:
            return item
    return {}


def safe_agent_adapter_config(agent_detail: dict[str, Any]) -> dict[str, Any]:
    config = agent_detail.get("adapterConfig") if isinstance(agent_detail.get("adapterConfig"), dict) else {}
    headers = config.get("headers") if isinstance(config.get("headers"), dict) else {}
    token = clean_missing(str(headers.get("x-openclaw-token") or headers.get("x-openclaw-auth") or ""))
    device_key = clean_missing(str(config.get("devicePrivateKeyPem") or ""))
    return {
        "url": clean_missing(str(config.get("url") or "")) or "missing",
        "sessionKeyStrategy": clean_missing(str(config.get("sessionKeyStrategy") or "")) or "unknown",
        "disableDeviceAuth": bool(config.get("disableDeviceAuth", False)),
        "token_configured": bool(token),
        "token_length": len(token or ""),
        "device_key_configured": bool(device_key),
    }


def render_agent_health(context: dict[str, Any], details: dict[str, Any]) -> str:
    lines = [
        "== Automixer Agent Health ==",
        f"context_source: {context.get('_operator_context_source', 'unknown')}",
        "write actions: disabled",
        f"paperclip_api_ok: {context.get('paperclip_api_ok')}",
    ]
    if details.get("error"):
        lines.append(f"error: {details['error']}")
        return "\n".join(lines)
    lines.append(f"audit_generated_at: {details.get('audit_generated_at') or 'unknown'}")
    lines.append(f"unhealthy_agents: {details.get('unhealthy_agent_count', 0)}")
    agents = details.get("agents") if isinstance(details.get("agents"), list) else []
    if not agents:
        lines.append("items: none")
        return "\n".join(lines)
    for item in agents:
        agent = item.get("agent") if isinstance(item.get("agent"), dict) else {}
        audit = item.get("audit") if isinstance(item.get("audit"), dict) else {}
        config = item.get("safe_adapter_config") if isinstance(item.get("safe_adapter_config"), dict) else {}
        lines.extend(
            [
                "",
                f"- agent: {agent.get('name') or agent.get('title') or 'unnamed'} ({agent.get('id') or 'missing-id'})",
                f"  status: {agent.get('status') or 'unknown'}",
                f"  adapter: {agent.get('adapterType') or 'unknown'}",
                f"  reason: {audit.get('reason') or 'status needs review'}",
                f"  heartbeat_age_hours: {audit.get('heartbeat_age_hours') if audit.get('heartbeat_age_hours') is not None else 'unknown'}",
                f"  assigned_open_issues: {item.get('assigned_issue_count', 0)}",
            ]
        )
        if item.get("detail_error"):
            lines.append(f"  detail_error: {item['detail_error']}")
        if config:
            lines.extend(
                [
                    f"  adapter_url: {config.get('url')}",
                    f"  session_key_strategy: {config.get('sessionKeyStrategy')}",
                    f"  token_configured: {config.get('token_configured')}",
                    f"  token_length: {config.get('token_length')}",
                    f"  device_key_configured: {config.get('device_key_configured')}",
                    f"  disable_device_auth: {config.get('disableDeviceAuth')}",
                ]
            )
        probe = item.get("local_gateway_probe") if isinstance(item.get("local_gateway_probe"), dict) else {}
        if probe:
            lines.extend(
                [
                    "  external_gateway_probe: disabled",
                    f"  local_gateway_probe_url: {probe.get('url')}",
                    f"  local_gateway_probe_allowed: {probe.get('allowed_local_only')}",
                    f"  local_gateway_reachable: {probe.get('reachable')}",
                    f"  local_gateway_ok: {probe.get('ok')}",
                    f"  local_gateway_status_code: {probe.get('status_code')}",
                    f"  local_gateway_openclaw_detected: {probe.get('openclaw_ui_detected')}",
                ]
            )
            if probe.get("error"):
                lines.append(f"  local_gateway_error: {probe.get('error')}")
        assigned = item.get("assigned_issues") if isinstance(item.get("assigned_issues"), list) else []
        for issue in assigned[:5]:
            lines.append(
                "  assigned_issue: "
                + f"{issue.get('identifier') or issue.get('id')}, "
                + f"status={issue.get('status') or 'unknown'}, "
                + f"title={issue.get('title') or 'untitled'}"
            )
    return "\n".join(lines)


def load_blocked_issues(
    config: OperatorConfig,
    context: dict[str, Any],
    *,
    limit: int = 20,
) -> tuple[list[dict[str, Any]], str | None]:
    if context.get("paperclip_api_ok") is not True:
        return [], "Paperclip API is unavailable"
    company_id = clean_missing(str(context.get("company_id") or ""))
    if not company_id:
        return [], "company_id is unavailable"
    client = bridge.PaperclipClient(
        str(context.get("paperclip_api_url") or bridge.DEFAULT_PAPERCLIP_API_URL),
        parse_float(
            config.environ.get("GPT_CONTROL_BRIDGE_PAPERCLIP_TIMEOUT_SECONDS"),
            3.0,
        ),
    )
    payload, error = bridge.safe_get(
        client,
        f"/api/companies/{bridge.quote(company_id)}/issues",
        params={"status": "blocked", "limit": limit},
    )
    if error:
        return [], error
    issues = bridge.extract_items(payload, ["issues", "data", "items", "results"])
    return sorted(issues, key=blocked_issue_sort_key), None


def command_operational_audit(config: OperatorConfig, extra_args: list[str]) -> int:
    parser = argparse.ArgumentParser(
        prog="operator.py operational-audit",
        description="Generate a read-only daily operational audit report",
    )
    parser.add_argument("--stale-hours", type=float, default=operational_audit.DEFAULT_STALE_HOURS)
    parser.add_argument("--orphan-hours", type=float, default=operational_audit.DEFAULT_ORPHAN_HOURS)
    parser.add_argument(
        "--heartbeat-stale-hours",
        type=float,
        default=operational_audit.DEFAULT_HEARTBEAT_STALE_HOURS,
    )
    args = parser.parse_args(extra_args)

    context = load_context(config)
    company_id = clean_missing(str(context.get("company_id") or "")) or clean_missing(
        config.environ.get("PAPERCLIP_COMPANY_ID")
    )
    if context.get("paperclip_api_ok") is not True:
        print("== Automixer Operational Audit ==")
        print("error: Paperclip API is unavailable")
        return 1
    if not company_id:
        print("== Automixer Operational Audit ==")
        print("error: company_id is unavailable")
        return 1

    result = operational_audit.create_operational_audit(
        project_root=config.project_root,
        api_url=str(context.get("paperclip_api_url") or bridge.DEFAULT_PAPERCLIP_API_URL),
        company_id=company_id,
        timeout_seconds=parse_float(
            config.environ.get("GPT_CONTROL_BRIDGE_PAPERCLIP_TIMEOUT_SECONDS"),
            3.0,
        ),
        thresholds=operational_audit.AuditThresholds(
            stale_hours=max(args.stale_hours, 0.0),
            orphan_hours=max(args.orphan_hours, 0.0),
            heartbeat_stale_hours=max(args.heartbeat_stale_hours, 0.0),
        ),
    )
    print("== Automixer Operational Audit ==")
    print(f"report_markdown: {result.report_paths.get('markdown') or 'missing'}")
    print(f"report_json: {result.report_paths.get('json') or 'missing'}")
    print("")
    print(operational_audit.render_markdown(result), end="")
    return 0


def command_vector_indexer(config: OperatorConfig, operator_command: str, extra_args: list[str]) -> int:
    vector_command = VECTOR_OPERATOR_COMMANDS[operator_command]
    script = config.project_root / VECTOR_INDEXER_SCRIPT
    if not script.exists():
        print(f"vector indexer unavailable: {script}")
        return 1
    cmd = [
        sys.executable,
        str(script),
        vector_command,
        "--project-root",
        str(config.project_root),
        *extra_args,
    ]
    result = subprocess.run(
        cmd,
        cwd=config.project_root,
        text=True,
        capture_output=True,
        check=False,
    )
    if result.stdout:
        print(result.stdout, end="")
    if result.stderr:
        print(result.stderr, end="", file=sys.stderr)
    return result.returncode


def render_status(config: OperatorConfig, context: dict[str, Any]) -> str:
    lines = [
        "== Automixer Operator Status ==",
        f"project_root: {config.project_root}",
        f"context_source: {context.get('_operator_context_source', 'unknown')}",
    ]
    if context.get("_operator_bridge_http_error"):
        lines.append(f"bridge_http_error: {context['_operator_bridge_http_error']}")
    lines.extend(
        [
            "",
            "Paperclip:",
            f"- api_url: {context.get('paperclip_api_url')}",
            f"- api_ok: {context.get('paperclip_api_ok')}",
            f"- company_id: {context.get('company_id') or 'unavailable'}",
        ]
    )
    errors = context.get("paperclip_errors") or []
    if errors:
        lines.append(f"- errors: {', '.join(str(error) for error in errors)}")

    lines.extend(render_active_runs(context.get("active_runs_summary")))
    lines.extend(render_agents(context.get("agents_summary")))
    direct_counts, direct_counts_error = load_issue_status_counts(config, context)
    lines.extend(render_queue_counts(context.get("dashboard_summary"), direct_counts, direct_counts_error))
    lines.extend(render_latest_operational_audit(config.project_root))
    lines.extend(render_state(context.get("watchdog_state_summary")))
    lines.extend(render_latest_report(context))
    lines.extend(["", render_recommended_action(context, direct_counts)])
    return "\n".join(lines)


def render_active_runs(summary: Any) -> list[str]:
    lines = ["", "Active runs:"]
    if not isinstance(summary, dict) or not summary.get("available"):
        return lines + [f"- unavailable: {summary.get('reason') if isinstance(summary, dict) else 'missing'}"]
    lines.append(f"- total: {summary.get('total', 0)}")
    by_status = summary.get("by_status") or {}
    if by_status:
        lines.append(f"- by_status: {format_counts(by_status)}")
    items = summary.get("items") if isinstance(summary.get("items"), list) else []
    if not items:
        lines.append("- items: none")
        return lines
    for item in items[:10]:
        lines.append(f"- {format_record(item)}")
    return lines


def render_agents(summary: Any) -> list[str]:
    lines = ["", "Agents:"]
    if not isinstance(summary, dict) or not summary.get("available"):
        return lines + [f"- unavailable: {summary.get('reason') if isinstance(summary, dict) else 'missing'}"]
    lines.append(f"- total: {summary.get('total', 0)}")
    by_status = summary.get("by_status") or {}
    if by_status:
        lines.append(f"- by_status: {format_counts(by_status)}")
    items = summary.get("items") if isinstance(summary.get("items"), list) else []
    attention_items = agent_attention_items(items)
    if attention_items:
        lines.append(f"- attention: {len(attention_items)} agent(s) need review")
        for item in attention_items[:5]:
            status = normalize_status(item.get("status") or item.get("state")) or "unknown"
            adapter = item.get("adapterType") or item.get("adapter_type") or "unknown"
            lines.append(f"- attention_agent: {format_agent_label(item)}, status={status}, adapter={adapter}")
    for item in items[:10]:
        lines.append(f"- {format_record(item)}")
    if not items:
        lines.append("- items: none")
    specialization_summary = summary.get("specialization_health")
    if isinstance(specialization_summary, dict) and specialization_summary.get("available"):
        counts = specialization_summary.get("counts") or {}
        lines.append(f"- specialization_health: ready={counts.get('ready', 0)} busy={counts.get('busy', 0)} error={counts.get('error', 0)} unavailable={counts.get('unavailable', 0)}")
        roles = specialization_summary.get("roles") if isinstance(specialization_summary.get("roles"), list) else []
        for role in roles:
            label = clean_missing(str(role.get("label") or ""))
            health = clean_missing(str(role.get("health") or "unknown"))
            agent_name_value = clean_missing(str(role.get("agent_name") or ""))
            if label and role.get("execution_role"):
                lines.append(f"- {label}: {health} ({agent_name_value or 'unresolved'})")
    return lines


def agent_attention_items(items: list[Any]) -> list[dict[str, Any]]:
    attention_statuses = {"error", "failed", "offline", "unavailable"}
    return [
        item
        for item in items
        if isinstance(item, dict)
        and normalize_status(item.get("status") or item.get("state")) in attention_statuses
    ]


def load_issue_status_counts(
    config: OperatorConfig,
    context: dict[str, Any],
) -> tuple[dict[str, int], str | None]:
    if context.get("paperclip_api_ok") is not True:
        return {}, "Paperclip unavailable"
    company_id = clean_missing(str(context.get("company_id") or ""))
    if not company_id:
        return {}, "company_id unavailable"
    client = bridge.PaperclipClient(
        str(context.get("paperclip_api_url") or bridge.DEFAULT_PAPERCLIP_API_URL),
        parse_float(
            config.environ.get("GPT_CONTROL_BRIDGE_PAPERCLIP_TIMEOUT_SECONDS"),
            3.0,
        ),
    )
    payload, error = bridge.safe_get(
        client,
        f"/api/companies/{bridge.quote(company_id)}/issues",
        params={"status": PREVIEW_ISSUE_STATUSES, "limit": 500},
    )
    if error:
        return {}, error
    counts: dict[str, int] = {}
    for issue in bridge.extract_items(payload, ["issues", "data", "items", "results"]):
        status = normalize_status(issue.get("status") or issue.get("state"))
        if status:
            counts[status] = counts.get(status, 0) + 1
    return counts, None


def render_queue_counts(
    summary: Any,
    direct_counts: dict[str, int] | None = None,
    direct_counts_error: str | None = None,
) -> list[str]:
    lines = ["", "Work queue:"]
    if not isinstance(summary, dict) or not summary.get("available"):
        if direct_counts:
            for status in ("blocked", "todo", "in_progress"):
                lines.append(f"- {status}: {direct_counts.get(status, 0)}")
            return lines
        reason = summary.get("reason") if isinstance(summary, dict) else "missing"
        if direct_counts_error:
            reason = f"{reason}; issues fallback: {direct_counts_error}"
        return lines + [f"- unavailable: {reason}"]
    counts = find_status_counts(summary.get("summary"))
    for status in ("blocked", "todo", "in_progress"):
        value: int | str = counts.get(status, "unknown")
        if value == "unknown" and direct_counts is not None:
            value = direct_counts.get(status, 0)
        lines.append(f"- {status}: {value}")
    if direct_counts_error:
        lines.append(f"- issues_fallback_warning: {direct_counts_error}")
    return lines


def latest_operational_audit(project_root: Path) -> dict[str, Any] | None:
    path = project_root / ".paperclip" / "reports" / "operational_audit_latest.json"
    if not path.exists():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    return payload if isinstance(payload, dict) else None


def render_latest_operational_audit(project_root: Path) -> list[str]:
    lines = ["", "Latest operational audit:"]
    audit = latest_operational_audit(project_root)
    path = project_root / ".paperclip" / "reports" / "operational_audit_latest.json"
    if not audit:
        return lines + ["- none"]
    lines.append(f"- path: {path}")
    lines.append(f"- generated_at: {audit.get('generated_at') or 'unknown'}")
    issue_counts = audit.get("issue_status_counts") if isinstance(audit.get("issue_status_counts"), dict) else {}
    agent_counts = audit.get("agent_status_counts") if isinstance(audit.get("agent_status_counts"), dict) else {}
    if issue_counts:
        lines.append(f"- issue_status_counts: {format_counts(issue_counts)}")
    if agent_counts:
        lines.append(f"- agent_status_counts: {format_counts(agent_counts)}")
    priority_actions = audit.get("priority_actions") if isinstance(audit.get("priority_actions"), list) else []
    if not priority_actions:
        lines.append("- priority_actions: none")
        return lines
    lines.append(f"- priority_actions: {len(priority_actions)}")
    for item in priority_actions[:5]:
        if not isinstance(item, dict):
            continue
        lines.append(
            "- priority_action: "
            + f"{item.get('severity') or 'unknown'}:{item.get('key') or 'unknown'}, "
            + f"target={item.get('target') or 'none'}, "
            + f"write_permitted={item.get('write_permitted')}"
        )
    return lines


def queue_count(
    context: dict[str, Any],
    status: str,
    direct_counts: dict[str, int] | None = None,
) -> int | None:
    summary = context.get("dashboard_summary")
    if isinstance(summary, dict) and summary.get("available"):
        counts = find_status_counts(summary.get("summary"))
        value = counts.get(status)
        if isinstance(value, int):
            return value
    if direct_counts is not None:
        return direct_counts.get(status, 0)
    return None


def render_state(summary: Any) -> list[str]:
    lines = ["", "Watchdog state:"]
    if not isinstance(summary, dict) or not summary.get("exists"):
        path = summary.get("path") if isinstance(summary, dict) else ".paperclip/watchdog_state.json"
        return lines + [f"- missing: {path}"]
    dispatch = summary.get("dispatch") if isinstance(summary.get("dispatch"), dict) else {}
    full_state = read_watchdog_state(Path(str(summary.get("path") or "")))
    full_dispatch = full_state.get("dispatch") if isinstance(full_state.get("dispatch"), dict) else {}
    lines.extend(
        [
            f"- path: {summary.get('path')}",
            f"- valid_json: {summary.get('valid_json')}",
            f"- last_issue_id: {dispatch.get('last_issue_id') or summary.get('last_issue_id') or 'none'}",
            f"- last_selected_agent_id: {dispatch.get('last_selected_agent_id') or 'none'}",
            f"- last_idempotency_key: {full_dispatch.get('last_idempotency_key') or 'none'}",
            f"- attempts_count: {dispatch.get('attempts_count', 0)}",
            f"- used_idempotency_keys_count: {dispatch.get('used_idempotency_keys_count', 0)}",
        ]
    )
    return lines


def render_latest_report(context: dict[str, Any]) -> list[str]:
    reporting_path = context.get("latest_reporting_summary_path")
    continuity = context.get("reporting_continuity")
    continuity_digest = None
    continuity_is_repeat = False
    if isinstance(continuity, dict):
        continuity_digest = continuity.get("latest_digest")
        continuity_is_repeat = bool(continuity.get("is_repeat"))
    lines = ["", "Latest reporting summary:"]
    if reporting_path:
        lines.append(f"- path: {reporting_path}")
        lines.append(f"- category: {context.get('latest_reporting_summary', {}).get('category') if isinstance(context.get('latest_reporting_summary'), dict) else 'unknown'}")
        heading = clean_missing(str(context.get("latest_reporting_summary_heading") or ""))
        if heading:
            lines.append(f"- heading: {heading}")
        if continuity_digest:
            lines.append(f"- digest: {continuity_digest}")
        lines.append(f"- is_repeat: {continuity_is_repeat}")
    else:
        lines.append("- none")

    path = context.get("latest_watchdog_report_path")
    lines.extend(["", "Latest watchdog report:"])
    if not path:
        return lines + ["- none"]
    lines.append(f"- path: {path}")
    first_heading = clean_missing(str(context.get("latest_watchdog_report_heading") or ""))
    if not first_heading:
        excerpt = str(context.get("latest_watchdog_report_excerpt") or "").strip()
        first_heading = next((line.strip() for line in excerpt.splitlines() if line.strip()), "")
    if first_heading:
        lines.append(f"- heading: {first_heading}")
    return lines


def operator_recommended_action(
    context: dict[str, Any],
    direct_counts: dict[str, int] | None = None,
) -> dict[str, Any] | None:
    action = context.get("recommended_next_action")
    base_action = action if isinstance(action, dict) else None
    if context.get("paperclip_api_ok") is True:
        active = context.get("active_runs_summary")
        active_total = 0
        if isinstance(active, dict) and active.get("available"):
            try:
                active_total = int(active.get("total") or 0)
            except (TypeError, ValueError):
                active_total = 0
        todo = queue_count(context, "todo", direct_counts)
        in_progress = queue_count(context, "in_progress", direct_counts)
        blocked = queue_count(context, "blocked", direct_counts)
        if active_total == 0 and todo == 0 and in_progress == 0 and (blocked or 0) > 0:
            return {
                "action": "triage_blocked_queue_or_create_safe_local_task",
                "reason": (
                    f"Paperclip has {blocked} blocked issues and no todo/in_progress work; "
                    "triage blockers or create a narrow safe local task before previewing dispatch."
                ),
                "requires_approval": False,
                "write_permitted": False,
            }
    return base_action


def render_recommended_action(
    context: dict[str, Any],
    direct_counts: dict[str, int] | None = None,
) -> str:
    action = operator_recommended_action(context, direct_counts)
    if not isinstance(action, dict):
        return "Recommended next action:\n- action: unavailable"
    return "\n".join(
        [
            "Recommended next action:",
            f"- action: {action.get('action')}",
            f"- reason: {action.get('reason')}",
            f"- requires_approval: {action.get('requires_approval')}",
            f"- write_permitted: {action.get('write_permitted')}",
        ]
    )


def render_report_summary(report: DispatchReport) -> str:
    return "\n".join(
        [
            "Dry-run report:",
            f"- path: {report.path}",
            f"- mode: {report.mode or 'unknown'}",
            f"- issue_id: {report.issue_id or 'missing'}",
            f"- selected_agent_id: {report.selected_agent_id or 'missing'}",
            f"- idempotency_key: {report.idempotency_key or 'missing'}",
            f"- approval_command_ready: {approval_command_ready(report)}",
            f"- result: {report.result or 'unknown'}",
            f"- reason: {report.reason or 'unknown'}",
        ]
    )


def render_monitor(
    context: dict[str, Any],
    report: DispatchReport | None,
    issue_id: str | None,
    details: MonitorDetails,
) -> str:
    lines = [
        "== Automixer Operator Monitor ==",
        f"context_source: {context.get('_operator_context_source', 'unknown')}",
        f"issue_id: {issue_id or 'unavailable'}",
    ]
    lines.extend(render_active_runs(context.get("active_runs_summary")))
    if report:
        lines.extend(["", "Latest dispatch report:", f"- path: {report.path}", f"- result: {report.result or 'unknown'}"])
    else:
        lines.extend(["", "Latest dispatch report:", "- none"])

    lines.extend(["", "Issue:"])
    if details.issue_error:
        lines.append(f"- unavailable: {details.issue_error}")
    elif details.issue:
        lines.append(f"- {format_record(details.issue)}")
    else:
        lines.append("- unavailable")

    lines.extend(["", "Runs:"])
    if details.runs_error:
        lines.append(f"- unavailable: {details.runs_error}")
    elif details.runs:
        for run in details.runs[:10]:
            lines.append(f"- {format_record(run)}")
    else:
        lines.append("- none")

    lines.extend(["", "Comments/report:"])
    if details.comments_error:
        lines.append(f"- comments_unavailable: {details.comments_error}")
    elif details.comments:
        for comment in details.comments[:5]:
            lines.append(f"- {format_record(comment)}")
    else:
        lines.append("- comments: none")

    final = final_result(details)
    lines.extend(["", "Final result:", f"- {final}"])
    return "\n".join(lines)


def render_blocked_queue(
    context: dict[str, Any],
    issues: list[dict[str, Any]],
    error: str | None,
) -> str:
    lines = [
        "== Automixer Blocked Queue ==",
        f"context_source: {context.get('_operator_context_source', 'unknown')}",
        "write actions: disabled",
        f"paperclip_api_ok: {context.get('paperclip_api_ok')}",
    ]
    if error:
        lines.append(f"error: {error}")
        return "\n".join(lines)
    lines.append(f"shown: {len(issues)}")
    if not issues:
        lines.append("items: none")
        return "\n".join(lines)
    lines.extend(["", "Blocked issues:"])
    for issue in issues:
        lines.extend(render_blocked_issue(issue))
    return "\n".join(lines)


def render_blocked_issue(issue: dict[str, Any]) -> list[str]:
    attention = issue.get("blockerAttention") if isinstance(issue.get("blockerAttention"), dict) else {}
    blocked_by = issue.get("blockedBy") if isinstance(issue.get("blockedBy"), list) else []
    unresolved = attention.get("unresolvedBlockerCount")
    sample = (
        attention.get("sampleBlockerIdentifier")
        or attention.get("sampleStalledBlockerIdentifier")
        or "none"
    )
    return [
        f"- {format_issue_label(issue)}",
        f"  status: {normalize_status(issue.get('status') or issue.get('state')) or 'unknown'}",
        f"  assignee: {get_issue_assignee(issue) or 'none'}",
        f"  blocker_attention: {attention.get('state') or 'unknown'}",
        f"  unresolved_blockers: {unresolved if unresolved is not None else len(blocked_by)}",
        f"  sample_blocker: {sample}",
    ]


def blocked_issue_sort_key(issue: dict[str, Any]) -> tuple[int, int, str]:
    attention = issue.get("blockerAttention") if isinstance(issue.get("blockerAttention"), dict) else {}
    needs_attention = 0 if attention.get("state") == "needs_attention" else 1
    try:
        priority = int(issue.get("priorityRank") or issue.get("priority") or 99)
    except (TypeError, ValueError):
        priority = 99
    identifier = issue_identifier(issue) or get_issue_id(issue) or ""
    return needs_attention, priority, identifier


def final_result(details: MonitorDetails) -> str:
    if details.issue:
        status = normalize_status(details.issue.get("status") or details.issue.get("state"))
        if status in TERMINAL_ISSUE_STATUSES:
            return f"issue terminal status: {status}"
    for run in details.runs:
        status = normalize_status(run.get("status") or run.get("state"))
        if status in bridge.TERMINAL_RUN_STATUSES:
            return f"run {run.get('id', 'unknown')} terminal status: {status}"
    return "not finished"


def load_monitor_details(
    config: OperatorConfig,
    context: dict[str, Any],
    issue_id: str | None,
) -> MonitorDetails:
    if not issue_id:
        return MonitorDetails(None, "no issue id", [], "no issue id", [], "no issue id")
    client = bridge.PaperclipClient(
        str(context.get("paperclip_api_url") or bridge.DEFAULT_PAPERCLIP_API_URL),
        parse_float(
            config.environ.get("GPT_CONTROL_BRIDGE_PAPERCLIP_TIMEOUT_SECONDS"),
            3.0,
        ),
    )
    issue, issue_error = bridge.safe_get(client, f"/api/issues/{bridge.quote(issue_id)}")
    comments_payload, comments_error = bridge.safe_get(
        client,
        f"/api/issues/{bridge.quote(issue_id)}/comments",
    )
    runs_payload, runs_error = bridge.safe_get(
        client,
        f"/api/issues/{bridge.quote(issue_id)}/live-runs",
    )
    if runs_error and context.get("company_id"):
        runs_payload, runs_error = bridge.safe_get(
            client,
            f"/api/companies/{bridge.quote(str(context['company_id']))}/heartbeat-runs",
            params={"issueId": issue_id, "limit": 20},
        )

    comments = bridge.extract_items(comments_payload, ["comments", "data", "items", "results"])
    runs = bridge.extract_items(runs_payload, ["liveRuns", "heartbeatRuns", "runs", "data", "items", "results"])
    return MonitorDetails(
        issue=issue if isinstance(issue, dict) else None,
        issue_error=issue_error,
        runs=[bridge.summarize_record(run, bridge.run_keys()) for run in runs],
        runs_error=runs_error,
        comments=[summarize_comment(comment) for comment in comments],
        comments_error=comments_error,
    )


def summarize_comment(comment: dict[str, Any]) -> dict[str, Any]:
    keys = ["id", "body", "text", "content", "createdAt", "updatedAt", "authorName", "agentName"]
    summary = bridge.summarize_record(comment, keys)
    for key in ("body", "text", "content"):
        if key in summary and isinstance(summary[key], str) and len(summary[key]) > 180:
            summary[key] = summary[key][:177] + "..."
    return summary


def resolve_issue_id(context: dict[str, Any], report: DispatchReport | None) -> str | None:
    if report and report.issue_id:
        return report.issue_id
    summary = context.get("watchdog_state_summary")
    state_path = summary.get("path") if isinstance(summary, dict) else ""
    state = read_watchdog_state(Path(str(state_path)))
    dispatch = state.get("dispatch") if isinstance(state.get("dispatch"), dict) else {}
    return dispatch.get("last_issue_id") or state.get("last_issue_id")


def latest_dry_run_report(project_root: Path) -> DispatchReport | None:
    for report in iter_dispatch_reports(project_root):
        if report.is_dry_run:
            state = read_watchdog_state(project_root / ".paperclip" / "watchdog_state.json")
            return fill_report_from_state(report, state)
    return None


def latest_dispatch_report(project_root: Path) -> DispatchReport | None:
    reports = iter_dispatch_reports(project_root)
    return reports[0] if reports else None


def iter_dispatch_reports(project_root: Path) -> list[DispatchReport]:
    reports_dir = project_root / ".paperclip" / "reports"
    if not reports_dir.exists():
        return []
    candidates = sorted(reports_dir.glob("*watchdog*.md"), key=lambda path: (path.stat().st_mtime_ns, path.name), reverse=True)
    return [parse_dispatch_report(path) for path in candidates]


def parse_dispatch_report(path: Path) -> DispatchReport:
    fields: dict[str, str] = {}
    current_section = ""
    intended_body: dict[str, Any] | None = None
    try:
        lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    except OSError:
        lines = []
    for line in lines:
        if line.startswith("## "):
            current_section = line[3:].strip().lower()
            continue
        match = REPORT_FIELD_RE.match(line)
        if not match:
            continue
        key, value = match.group(1), match.group(2).strip()
        if current_section == "intended action" and key == "body":
            intended_body = parse_inline_json(value)
            continue
        if current_section == "intended action" and key == "endpoint":
            fields["intended_endpoint"] = clean_field(value)
            continue
        fields[key] = clean_field(value)

    issue_id = clean_missing(fields.get("issue"))
    endpoint = fields.get("intended_endpoint")
    if not issue_id and endpoint:
        issue_id = endpoint.rstrip("/").split("/")[-1]
    selected_agent_id = clean_missing(fields.get("selected_agent"))
    if not selected_agent_id and intended_body:
        selected_agent_id = clean_missing(str(intended_body.get("assigneeAgentId") or ""))
    idempotency_key = parse_idempotency_key(fields.get("idempotency_ok"))
    return DispatchReport(
        path=path,
        mode=clean_missing(fields.get("mode")),
        issue_id=issue_id,
        selected_agent_id=selected_agent_id,
        idempotency_key=idempotency_key,
        dispatch_executed=parse_bool_text(fields.get("dispatch_executed")),
        result=clean_missing(fields.get("result")),
        reason=clean_missing(fields.get("reason")),
        intended_endpoint=endpoint,
        intended_body=intended_body,
    )


def parse_inline_json(value: str) -> dict[str, Any] | None:
    text = clean_field(value)
    if text.startswith("`") and text.endswith("`"):
        text = text[1:-1]
    try:
        payload = json.loads(text)
    except json.JSONDecodeError:
        return None
    return payload if isinstance(payload, dict) else None


def parse_idempotency_key(value: str | None) -> str | None:
    text = clean_missing(clean_field(value or ""))
    if not text:
        return None
    if text.startswith("(passed="):
        return None
    return text.split(" (passed=", 1)[0].strip() or None


def fill_report_from_state(report: DispatchReport, state: dict[str, Any]) -> DispatchReport:
    dispatch = state.get("dispatch") if isinstance(state.get("dispatch"), dict) else {}
    attempts = dispatch.get("attempts") if isinstance(dispatch.get("attempts"), list) else []
    match = None
    for attempt in reversed(attempts):
        if not isinstance(attempt, dict):
            continue
        if report.issue_id and attempt.get("issue_id") != report.issue_id:
            continue
        if report.selected_agent_id and attempt.get("selected_agent_id") != report.selected_agent_id:
            continue
        if report.mode and attempt.get("mode") != report.mode:
            continue
        match = attempt
        break
    idempotency_key = report.idempotency_key or (match or {}).get("idempotency_key") or dispatch.get("last_idempotency_key")
    return DispatchReport(
        path=report.path,
        mode=report.mode,
        issue_id=report.issue_id or (match or {}).get("issue_id") or dispatch.get("last_issue_id"),
        selected_agent_id=report.selected_agent_id or (match or {}).get("selected_agent_id") or dispatch.get("last_selected_agent_id"),
        idempotency_key=idempotency_key,
        dispatch_executed=report.dispatch_executed,
        result=report.result,
        reason=report.reason,
        intended_endpoint=report.intended_endpoint,
        intended_body=report.intended_body,
    )


def read_watchdog_state(path: Path) -> dict[str, Any]:
    if not path or not str(path) or not path.exists():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return payload if isinstance(payload, dict) else {}


def build_approval_command(report: DispatchReport) -> str:
    issue_id = require_value(report.issue_id, "issue-id")
    agent_id = require_value(report.selected_agent_id, "agent-id")
    idempotency_key = require_value(report.idempotency_key, "idempotency-key")
    assignments = [
        ("AUTOMIXER_WATCHDOG_DISPATCH_ENABLED", "true"),
        ("AUTOMIXER_WATCHDOG_DISPATCH_DRY_RUN", "false"),
        ("AUTOMIXER_WATCHDOG_DISPATCH_ISSUE_ID", issue_id),
        ("AUTOMIXER_WATCHDOG_DISPATCH_AGENT_ID", agent_id),
        ("AUTOMIXER_WATCHDOG_DISPATCH_ALLOWLIST_AGENT_IDS", agent_id),
        ("AUTOMIXER_WATCHDOG_DISPATCH_IDEMPOTENCY_KEY", idempotency_key),
        ("AUTOMIXER_WATCHDOG_DISPATCH_METHOD", "patch_issue_assignment"),
        ("AUTOMIXER_WATCHDOG_DISPATCH_FORBID_DIRECT_WAKEUP_WITHOUT_ASSIGNMENT", "true"),
        ("AUTOMIXER_WATCHDOG_DISPATCH_FORBID_LIVE_OSC", "true"),
        ("AUTOMIXER_WATCHDOG_DISPATCH_FORBID_WING_RUNTIME_WRITE", "true"),
    ]
    parts = [f"{key}={shlex.quote(value)}" for key, value in assignments]
    parts.append("python3 tools/paperclip_watchdog/automixer_paperclip_watchdog.py dispatch")
    return " \\\n".join(parts)


def render_approval_section(report: DispatchReport) -> str:
    if not approval_command_allowed(report):
        return "\n".join(
            [
                "approval_command: unavailable",
                f"reason: {approval_block_reason(report)}",
            ]
        )
    return "\n".join(
        [
            "approval_command:",
            indent(build_approval_command(report)),
        ]
    )


def approval_command_allowed(report: DispatchReport) -> bool:
    return approval_command_ready(report) and report.result == "dry_run_preview"


def approval_block_reason(report: DispatchReport) -> str:
    missing = []
    if not report.issue_id:
        missing.append("issue_id")
    if not report.selected_agent_id:
        missing.append("selected_agent_id")
    if not report.idempotency_key:
        missing.append("idempotency_key")
    if missing:
        return "missing " + ", ".join(missing)
    if report.result != "dry_run_preview":
        return f"latest dry-run report result is {report.result or 'unknown'}, not dry_run_preview"
    return "approval command is not safe for this report"


def approval_command_ready(report: DispatchReport) -> bool:
    return bool(report.issue_id and report.selected_agent_id and report.idempotency_key)


def require_value(value: str | None, placeholder: str) -> str:
    return value if value else f"<{placeholder}>"


def find_status_counts(value: Any) -> dict[str, int]:
    counts: dict[str, int] = {}
    collect_status_counts(value, counts)
    return counts


def collect_status_counts(value: Any, counts: dict[str, int]) -> None:
    if isinstance(value, dict):
        for key, nested in value.items():
            normalized = normalize_status(key)
            if normalized in {"blocked", "todo", "in_progress"} and isinstance(nested, (int, float)):
                counts[normalized] = int(nested)
            collect_status_counts(nested, counts)
    elif isinstance(value, list):
        for item in value:
            collect_status_counts(item, counts)


def format_counts(counts: dict[str, Any]) -> str:
    return ", ".join(f"{key}={value}" for key, value in sorted(counts.items()))


def format_record(record: Any) -> str:
    if not isinstance(record, dict):
        return str(record)
    return ", ".join(f"{key}={value}" for key, value in record.items() if value not in (None, ""))


def clean_field(value: str | None) -> str:
    return str(value or "").strip().strip("`").strip()


def clean_missing(value: str | None) -> str | None:
    text = clean_field(value)
    return None if text.lower() in {"", "none", "missing", "null", "unavailable"} else text


def parse_bool_text(value: str | None) -> bool | None:
    text = clean_field(value).lower()
    if text in {"true", "1", "yes"}:
        return True
    if text in {"false", "0", "no"}:
        return False
    return None


def normalize_status(value: Any) -> str:
    return str(value or "").strip().lower().replace("-", "_").replace(" ", "_")


def indent(text: str, prefix: str = "  ") -> str:
    return "\n".join(prefix + line if line else prefix.rstrip() for line in text.splitlines())


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Safe Automixer operator workflow")
    parser.add_argument(
        "command",
        choices=(
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
            "agent-health",
            "operational-audit",
            "vector-index",
            "vector-search",
            "vector-add",
            "vector-stats",
            "vector-paperclip-sync",
        ),
        help="operator command to run",
    )
    parser.add_argument(
        "args",
        nargs=argparse.REMAINDER,
        help="extra arguments passed to vector-* commands",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    config = build_config()
    if args.command == "status":
        return command_status(config)
    if args.command == "bridge-status":
        return command_bridge_status(config)
    if args.command == "bridge-start":
        return command_bridge_start(config)
    if args.command == "doctor":
        return command_doctor(config)
    if args.command == "next":
        return command_next(config)
    if args.command == "target":
        return command_target(config)
    if args.command == "preview":
        return command_preview(config)
    if args.command == "cycle":
        return command_cycle(config)
    if args.command == "approve-template":
        return command_approve_template(config)
    if args.command == "monitor":
        return command_monitor(config)
    if args.command == "blocked":
        return command_blocked(config)
    if args.command == "agent-health":
        return command_agent_health(config)
    if args.command == "operational-audit":
        return command_operational_audit(config, args.args)
    if args.command in VECTOR_OPERATOR_COMMANDS:
        return command_vector_indexer(config, args.command, args.args)
    parser.error(f"unknown command: {args.command}")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
