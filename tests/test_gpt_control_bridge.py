from __future__ import annotations

import http.client
import json
import os
import threading
from http.server import ThreadingHTTPServer
from pathlib import Path

from tools.gpt_control_bridge import server as bridge


class FakePaperclipClient:
    def __init__(self, routes: dict[str, object] | None = None, *, fail: bool = False) -> None:
        self.routes = routes or {}
        self.fail = fail
        self.calls: list[tuple[str, dict | None]] = []

    def get(self, path: str, params: dict | None = None) -> object:
        self.calls.append((path, params))
        if self.fail:
            raise OSError("paperclip unavailable")
        if path not in self.routes:
            raise OSError(f"missing fake route: {path}")
        return self.routes[path]


def make_config(tmp_path: Path, *, company_id: str | None = "company-1") -> bridge.BridgeConfig:
    return bridge.BridgeConfig(
        paperclip_api_url="http://127.0.0.1:3100",
        project_root=tmp_path,
        paperclip_company_id=company_id,
    )


def make_routes(*, active_runs: list[dict] | None = None) -> dict[str, object]:
    return {
        "/api/health": {"status": "ok"},
        "/api/companies/company-1/dashboard": {
            "agents": {"running": 0, "idle": 2},
            "tasks": {"todo": 1, "in_progress": 0},
        },
        "/api/companies/company-1/agents": [
            {"id": "agent-1", "name": "Director", "status": "idle", "adapterType": "codex_local"},
            {"id": "agent-2", "name": "Reviewer", "status": "idle", "adapterType": "codex_local"},
        ],
        "/api/companies/company-1/live-runs": active_runs or [],
    }


def test_health_returns_read_only_true() -> None:
    payload = bridge.health_payload()

    assert payload["ok"] is True
    assert payload["read_only"] is True
    assert payload["service"] == bridge.SERVICE_NAME


def test_context_works_when_paperclip_unavailable(tmp_path: Path) -> None:
    context = bridge.build_context(make_config(tmp_path), client=FakePaperclipClient(fail=True))

    assert context["paperclip_api_ok"] is False
    assert context["dashboard_summary"]["available"] is False
    assert context["agents_summary"]["available"] is False
    assert context["active_runs_summary"]["available"] is False
    assert context["recommended_next_action"]["action"] == "check_paperclip_availability"
    assert context["safety_flags"]["write_routes_enabled"] is False


def test_missing_reports_and_state_are_handled(tmp_path: Path) -> None:
    context = bridge.build_context(
        make_config(tmp_path),
        client=FakePaperclipClient(make_routes()),
    )

    assert context["paperclip_api_ok"] is True
    assert context["latest_watchdog_report_path"] is None
    assert context["latest_watchdog_report_excerpt"] == ""
    assert context["watchdog_state_summary"]["exists"] is False
    assert context["watchdog_state_summary"]["valid_json"] is False


def test_latest_report_excerpt_loaded(tmp_path: Path) -> None:
    reports_dir = tmp_path / ".paperclip" / "reports"
    reports_dir.mkdir(parents=True)
    older = reports_dir / "older_watchdog.md"
    newer = reports_dir / "newer_watchdog.md"
    older.write_text("# Older\nold report\n", encoding="utf-8")
    newer.write_text("# Newer\nlatest watchdog report body\n", encoding="utf-8")
    os.utime(older, (1000, 1000))
    os.utime(newer, (2000, 2000))

    context = bridge.build_context(
        make_config(tmp_path),
        client=FakePaperclipClient(make_routes()),
    )

    assert context["latest_watchdog_report_path"] == str(newer)
    assert "# Newer" in context["latest_watchdog_report_excerpt"]
    assert "latest watchdog report body" in context["latest_watchdog_report_excerpt"]


def test_watchdog_state_summary_loaded(tmp_path: Path) -> None:
    state_dir = tmp_path / ".paperclip"
    state_dir.mkdir()
    (state_dir / "watchdog_state.json").write_text(
        json.dumps(
            {
                "version": 1,
                "last_issue_id": "issue-1",
                "last_task_key": "task-1",
                "dispatch": {
                    "attempts": [{"result": "blocked"}],
                    "last_dispatch_result": "blocked",
                    "used_idempotency_keys": {"key-1": {"result": "dispatched"}},
                },
            }
        ),
        encoding="utf-8",
    )

    context = bridge.build_context(
        make_config(tmp_path),
        client=FakePaperclipClient(make_routes()),
    )

    state = context["watchdog_state_summary"]
    assert state["exists"] is True
    assert state["valid_json"] is True
    assert state["last_issue_id"] == "issue-1"
    assert state["dispatch"]["attempts_count"] == 1
    assert state["dispatch"]["used_idempotency_keys_count"] == 1


def test_safety_flags_always_block_writes(tmp_path: Path) -> None:
    context = bridge.build_context(
        make_config(tmp_path),
        client=FakePaperclipClient(make_routes()),
    )

    assert context["safety_flags"] == {
        "dispatch_allowed": False,
        "read_only": True,
        "wing_osc_allowed": False,
        "write_routes_enabled": False,
    }


def test_no_write_routes_exist(tmp_path: Path) -> None:
    client = FakePaperclipClient(make_routes())
    with running_server(make_config(tmp_path), client) as address:
        for method in ("POST", "PATCH", "DELETE", "PUT"):
            status, payload = request_json(method, address, "/v1/context/automixer")
            assert status == 405
            assert payload["error"] == "write_methods_disabled"
            assert payload["safety_flags"]["write_routes_enabled"] is False


def test_non_loopback_bind_host_is_rejected() -> None:
    try:
        bridge.validate_bind_host("0.0.0.0")
    except ValueError as exc:
        assert "loopback" in str(exc)
    else:
        raise AssertionError("non-loopback host was accepted")


def test_recommended_next_action_changes_when_active_runs_exist_vs_none(tmp_path: Path) -> None:
    no_runs = bridge.build_context(
        make_config(tmp_path),
        client=FakePaperclipClient(make_routes(active_runs=[])),
    )
    with_runs = bridge.build_context(
        make_config(tmp_path),
        client=FakePaperclipClient(
            make_routes(
                active_runs=[
                    {
                        "id": "run-1",
                        "status": "running",
                        "agentId": "agent-1",
                        "agentName": "Director",
                        "issueId": "issue-1",
                    }
                ]
            )
        ),
    )

    assert no_runs["recommended_next_action"]["action"] == "review_status_without_watchdog_report"
    assert with_runs["recommended_next_action"]["action"] == "observe_active_runs"
    assert with_runs["active_runs_summary"]["total"] == 1


def test_health_endpoint_over_http(tmp_path: Path) -> None:
    with running_server(make_config(tmp_path), FakePaperclipClient(make_routes())) as address:
        status, payload = request_json("GET", address, "/health")

    assert status == 200
    assert payload["ok"] is True
    assert payload["read_only"] is True


class running_server:
    def __init__(self, config: bridge.BridgeConfig, client: FakePaperclipClient) -> None:
        self.config = config
        self.client = client
        self.httpd: ThreadingHTTPServer | None = None
        self.thread: threading.Thread | None = None

    def __enter__(self) -> tuple[str, int]:
        handler = bridge.make_handler(self.config, client_factory=lambda _config: self.client)
        self.httpd = ThreadingHTTPServer(("127.0.0.1", 0), handler)
        self.thread = threading.Thread(target=self.httpd.serve_forever, daemon=True)
        self.thread.start()
        return self.httpd.server_address

    def __exit__(self, exc_type, exc, tb) -> None:
        assert self.httpd is not None
        self.httpd.shutdown()
        self.httpd.server_close()
        if self.thread is not None:
            self.thread.join(timeout=2)


def request_json(method: str, address: tuple[str, int], path: str) -> tuple[int, dict]:
    host, port = address
    conn = http.client.HTTPConnection(host, port, timeout=3)
    try:
        conn.request(method, path)
        response = conn.getresponse()
        payload = json.loads(response.read().decode("utf-8"))
        return response.status, payload
    finally:
        conn.close()
