#!/usr/bin/env python3
"""Read-only local GPT control bridge for Automixer/Paperclip status."""

from __future__ import annotations

import argparse
import json
import os
import sys
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Callable


SERVICE_NAME = "automixer-gpt-control-bridge"
DEFAULT_BIND_HOST = "127.0.0.1"
DEFAULT_PORT = 8788
DEFAULT_PAPERCLIP_API_URL = "http://127.0.0.1:3100"
DEFAULT_PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(DEFAULT_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(DEFAULT_PROJECT_ROOT))

from tools.project_vector_index import automixer_chromadb_index as vector_index  # noqa: E402
from tools import agent_specialization  # noqa: E402

ACTIVE_RUN_STATUSES = {"queued", "running", "active", "busy", "in_progress", "started", "working"}
TERMINAL_RUN_STATUSES = {"done", "failed", "succeeded", "cancelled", "canceled", "finished"}
REPORT_EXCERPT_CHARS = 2000
DEFAULT_VECTOR_LIMIT = 5
MAX_VECTOR_LIMIT = 10
RECENT_REPORT_LIMIT = 5


@dataclass(frozen=True)
class BridgeConfig:
    paperclip_api_url: str
    project_root: Path
    paperclip_company_id: str | None = None
    paperclip_timeout_seconds: float = 3.0
    vector_persist_dir: str = vector_index.DEFAULT_PERSIST_DIR
    vector_collection: str = vector_index.DEFAULT_COLLECTION


class PaperclipClient:
    def __init__(self, base_url: str, timeout_seconds: float = 3.0) -> None:
        self.base_url = base_url.rstrip("/")
        self.timeout_seconds = timeout_seconds

    def get(self, path: str, params: dict[str, Any] | None = None) -> Any:
        query = ""
        if params:
            query = "?" + urllib.parse.urlencode(
                {key: value for key, value in params.items() if value is not None}
            )
        url = f"{self.base_url}{path}{query}"
        request = urllib.request.Request(url, method="GET")
        with urllib.request.urlopen(request, timeout=self.timeout_seconds) as response:
            raw = response.read().decode("utf-8", errors="replace")
        if not raw:
            return None
        return json.loads(raw)


def build_config_from_env(environ: dict[str, str] | None = None) -> BridgeConfig:
    env = environ if environ is not None else os.environ
    return BridgeConfig(
        paperclip_api_url=env.get("PAPERCLIP_API_URL", DEFAULT_PAPERCLIP_API_URL).rstrip("/"),
        project_root=Path(env.get("AUTOMIXER_PROJECT_ROOT", str(DEFAULT_PROJECT_ROOT))).expanduser().resolve(),
        paperclip_company_id=env.get("PAPERCLIP_COMPANY_ID") or None,
        paperclip_timeout_seconds=parse_float(env.get("GPT_CONTROL_BRIDGE_PAPERCLIP_TIMEOUT_SECONDS"), 3.0),
        vector_persist_dir=env.get("AUTOMIXER_VECTOR_PERSIST_DIR", vector_index.DEFAULT_PERSIST_DIR),
        vector_collection=env.get("AUTOMIXER_VECTOR_COLLECTION", vector_index.DEFAULT_COLLECTION),
    )


def parse_float(value: str | None, default: float) -> float:
    if not value:
        return default
    try:
        parsed = float(value)
    except ValueError:
        return default
    return parsed if parsed > 0 else default


def parse_positive_int(value: str | None, default: int) -> int:
    if not value:
        return default
    try:
        parsed = int(value)
    except ValueError:
        return default
    return parsed if parsed > 0 else default


def first_param(params: dict[str, list[str]], key: str) -> str | None:
    values = params.get(key)
    if not values:
        return None
    return values[0]


def health_payload() -> dict[str, Any]:
    return {
        "knowledge_search_enabled": True,
        "ok": True,
        "read_only": True,
        "service": SERVICE_NAME,
    }


def safety_flags() -> dict[str, bool]:
    return {
        "dispatch_allowed": False,
        "read_only": True,
        "wing_osc_allowed": False,
        "write_routes_enabled": False,
    }


def build_context(
    config: BridgeConfig,
    *,
    client: Any | None = None,
) -> dict[str, Any]:
    paperclip = client or PaperclipClient(config.paperclip_api_url, config.paperclip_timeout_seconds)
    errors: list[str] = []
    health, health_error = safe_get(paperclip, "/api/health")
    paperclip_api_ok = health_error is None and not is_unhealthy_payload(health)
    if health_error:
        errors.append(f"/api/health: {health_error}")

    company_id = None
    company_id_source = "unavailable"
    dashboard_summary = unavailable_summary("paperclip_unavailable")
    agents_summary = unavailable_summary("paperclip_unavailable")
    active_runs_summary = unavailable_summary("paperclip_unavailable")

    if paperclip_api_ok:
        company_id, company_id_source = resolve_company_id(config, paperclip, errors)
        if company_id:
            dashboard_summary = get_dashboard_summary(paperclip, company_id)
            agents_summary = get_agents_summary(paperclip, company_id)
            active_runs_summary = get_active_runs_summary(paperclip, company_id)
        else:
            missing = unavailable_summary("company_id_unavailable")
            dashboard_summary = missing
            agents_summary = missing
            active_runs_summary = missing

    watchdog_report = latest_watchdog_report(config.project_root)
    reporting_summary = latest_reporting_summary(config.project_root)
    reports = recent_reports(config.project_root)
    continuity = reporting_continuity(reports)
    state_summary = watchdog_state_summary(config.project_root)

    context = {
        "active_runs_summary": active_runs_summary,
        "agents_summary": agents_summary,
        "company_id": company_id,
        "company_id_source": company_id_source,
        "dashboard_summary": dashboard_summary,
        "latest_reporting_summary": reporting_summary,
        "latest_reporting_summary_excerpt": reporting_summary["excerpt"],
        "latest_reporting_summary_heading": reporting_summary["heading"],
        "latest_reporting_summary_path": reporting_summary["path"],
        "latest_watchdog_report_excerpt": watchdog_report["excerpt"],
        "latest_watchdog_report_heading": watchdog_report["heading"],
        "latest_watchdog_report_path": watchdog_report["path"],
        "paperclip_api_ok": paperclip_api_ok,
        "paperclip_api_url": config.paperclip_api_url,
        "paperclip_errors": errors,
        "recent_reports": reports,
        "reporting_continuity": continuity,
        "recommended_next_action": recommended_next_action(
            paperclip_api_ok=paperclip_api_ok,
            company_id=company_id,
            active_runs_summary=active_runs_summary,
            latest_reporting_summary_path=reporting_summary["path"],
        ),
        "safety_flags": safety_flags(),
        "vector_index_summary": vector_index_summary(config),
        "watchdog_state_summary": state_summary,
    }
    return context


def vector_index_summary(config: BridgeConfig) -> dict[str, Any]:
    persist_dir = vector_index.parse_path(config.vector_persist_dir, config.project_root)
    manifest_path = config.project_root / vector_index.DEFAULT_MANIFEST
    summary: dict[str, Any] = {
        "available": False,
        "collection": config.vector_collection,
        "persist_dir": str(persist_dir),
        "read_only_endpoint": "/v1/knowledge/automixer/search?q=...",
    }
    if not persist_dir.exists():
        summary["reason"] = "not_indexed"
        return summary
    try:
        collection = vector_index.chroma_collection(persist_dir, config.vector_collection, reset=False)
        summary["record_count"] = collection.count()
        summary["available"] = True
    except Exception as exc:
        summary["reason"] = f"{type(exc).__name__}: {exc}"
        return summary
    if manifest_path.exists():
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            summary["manifest"] = {
                "indexed_at": manifest.get("indexed_at"),
                "source_count": manifest.get("source_count"),
                "record_count": manifest.get("record_count"),
                "include_paperclip_reports": manifest.get("include_paperclip_reports"),
            }
        except (OSError, json.JSONDecodeError):
            summary["manifest_error"] = "unreadable"
    return summary


def search_vector_index(config: BridgeConfig, query: str, limit: int = DEFAULT_VECTOR_LIMIT) -> dict[str, Any]:
    query = query.strip()
    if not query:
        return {"available": False, "error": "query_required", "read_only": True}
    limit = max(1, min(MAX_VECTOR_LIMIT, limit))
    persist_dir = vector_index.parse_path(config.vector_persist_dir, config.project_root)
    if not persist_dir.exists():
        return {
            "available": False,
            "error": "vector_index_not_found",
            "persist_dir": str(persist_dir),
            "read_only": True,
        }
    try:
        collection = vector_index.chroma_collection(persist_dir, config.vector_collection, reset=False)
        hits = vector_index.search_collection(collection, query, limit=limit)
    except Exception as exc:
        return {
            "available": False,
            "error": f"{type(exc).__name__}: {exc}",
            "persist_dir": str(persist_dir),
            "read_only": True,
        }
    return {
        "available": True,
        "collection": config.vector_collection,
        "query": query,
        "read_only": True,
        "results": hits,
        "total": len(hits),
    }


def safe_get(client: Any, path: str, params: dict[str, Any] | None = None) -> tuple[Any, str | None]:
    try:
        return client.get(path, params=params), None
    except (urllib.error.URLError, TimeoutError, OSError, json.JSONDecodeError, ValueError) as exc:
        return None, f"{type(exc).__name__}: {exc}"
    except Exception as exc:  # defensive: bridge status must remain available
        return None, f"{type(exc).__name__}: {exc}"


def is_unhealthy_payload(payload: Any) -> bool:
    if not isinstance(payload, dict):
        return False
    status = str(payload.get("status") or payload.get("state") or "").strip().lower()
    return status in {"unhealthy", "error", "failed"}


def resolve_company_id(
    config: BridgeConfig,
    client: Any,
    errors: list[str],
) -> tuple[str | None, str]:
    if config.paperclip_company_id:
        return config.paperclip_company_id, "env:PAPERCLIP_COMPANY_ID"

    payload, error = safe_get(client, "/api/companies")
    if error:
        errors.append(f"/api/companies: {error}")
        return None, "missing"

    companies = extract_items(payload, ["companies", "data", "items", "results"])
    if not companies:
        return None, "missing"
    companies = sorted(companies, key=company_preference_key)
    company_id = companies[0].get("id") or companies[0].get("companyId")
    return (str(company_id), "api:/api/companies") if company_id else (None, "missing")


def company_preference_key(company: dict[str, Any]) -> tuple[int, int, str, str]:
    name = str(company.get("name") or "").strip().lower()
    issue_prefix = str(company.get("issuePrefix") or company.get("issue_prefix") or "").strip().lower()
    issue_counter = parse_nonnegative_int(company.get("issueCounter") or company.get("issue_counter"))
    if issue_prefix == "tub" or "tubeslave automixer" in name:
        priority = 0
    elif issue_counter > 0:
        priority = 1
    else:
        priority = 2
    return (priority, -issue_counter, name, str(company.get("id") or company.get("companyId") or ""))


def parse_nonnegative_int(value: Any) -> int:
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        return 0
    return parsed if parsed >= 0 else 0


def get_dashboard_summary(client: Any, company_id: str) -> dict[str, Any]:
    payload, error = safe_get(client, f"/api/companies/{quote(company_id)}/dashboard")
    if error:
        return unavailable_summary(error)
    return {
        "available": True,
        "summary": compact_value(payload),
    }


def get_agents_summary(client: Any, company_id: str) -> dict[str, Any]:
    payload, error = safe_get(client, f"/api/companies/{quote(company_id)}/agents")
    if error:
        return unavailable_summary(error)
    agents = extract_items(payload, ["agents", "data", "items", "results"])
    return {
        "available": True,
        "by_status": count_by_status(agents),
        "items": [summarize_record(agent, agent_keys()) for agent in agents[:25]],
        "specialization_health": agent_specialization.specialization_health_summary(agents),
        "total": len(agents),
    }


def get_active_runs_summary(client: Any, company_id: str) -> dict[str, Any]:
    payload, error = safe_get(
        client,
        f"/api/companies/{quote(company_id)}/live-runs",
        params={"limit": 50, "minCount": 0},
    )
    source = "live-runs"
    if error:
        payload, fallback_error = safe_get(
            client,
            f"/api/companies/{quote(company_id)}/heartbeat-runs",
            params={"limit": 50},
        )
        if fallback_error:
            return unavailable_summary(f"{error}; fallback: {fallback_error}")
        source = "heartbeat-runs-fallback"

    runs = extract_items(payload, ["liveRuns", "heartbeatRuns", "runs", "data", "items", "results"])
    active_runs = [run for run in runs if is_active_run(run)]
    return {
        "available": True,
        "by_status": count_by_status(active_runs),
        "items": [summarize_record(run, run_keys()) for run in active_runs[:25]],
        "source": source,
        "total": len(active_runs),
    }


def latest_watchdog_report(project_root: Path) -> dict[str, Any]:
    reports_dir = project_root / ".paperclip" / "reports"
    if not reports_dir.exists():
        return empty_report_summary()

    candidates = sorted(reports_dir.glob("*watchdog*.md"))
    if not candidates:
        return empty_report_summary()

    return report_summary_for_path(max(candidates, key=report_sort_key))


def latest_reporting_summary(project_root: Path) -> dict[str, Any]:
    reports_dir = project_root / ".paperclip" / "reports"
    if not reports_dir.exists():
        return empty_report_summary()

    candidates = sorted(reports_dir.glob("*.md"))
    if not candidates:
        return empty_report_summary()
    return report_summary_for_path(max(candidates, key=report_sort_key))


def recent_reports(project_root: Path, limit: int = RECENT_REPORT_LIMIT) -> list[dict[str, Any]]:
    reports_dir = project_root / ".paperclip" / "reports"
    if not reports_dir.exists():
        return []
    candidates = sorted(reports_dir.glob("*.md"), key=report_sort_key, reverse=True)
    return [report_summary_for_path(path, include_excerpt=False) for path in candidates[:limit]]


def empty_report_summary() -> dict[str, Any]:
    return {
        "category": None,
        "exists": False,
        "excerpt": "",
        "heading": None,
        "modified_at": None,
        "name": None,
        "path": None,
        "truncated": False,
    }


def report_summary_for_path(path: Path, *, include_excerpt: bool = True) -> dict[str, Any]:
    text = path.read_text(encoding="utf-8", errors="replace").replace("\r\n", "\n")
    excerpt = text[:REPORT_EXCERPT_CHARS] if include_excerpt else ""
    report_id = report_digest(path, text)
    return {
        "category": report_category(path),
        "exists": True,
        "digest": report_id,
        "excerpt": excerpt,
        "heading": first_heading(text),
        "modified_at": isoformat_utc(path.stat().st_mtime),
        "name": path.name,
        "path": str(path),
        "truncated": len(text) > REPORT_EXCERPT_CHARS if include_excerpt else False,
    }


def reporting_continuity(reports: list[dict[str, Any]]) -> dict[str, Any]:
    if not reports:
        return {"continuity": "absent", "latest_digest": None, "previous_digest": None, "is_repeat": False}
    latest_digest = str(reports[0].get("digest") or "")
    previous_digest = str(reports[1].get("digest")) if len(reports) > 1 else None
    return {
        "continuity": "stable",
        "latest_digest": latest_digest,
        "is_repeat": latest_digest and latest_digest == (previous_digest or ""),
        "previous_digest": previous_digest,
    }


def report_digest(path: Path, text: str) -> str:
    import hashlib

    body = text.encode("utf-8", errors="replace")
    return hashlib.sha256(body).hexdigest()[:16]


def report_category(path: Path) -> str:
    name = path.name.lower()
    if name == "operational_audit_latest.md" or name.startswith("operational_audit_"):
        return "operational_audit"
    if name == "watchdog_dispatch_report.md":
        return "watchdog_dispatch"
    if "watchdog" in name:
        return "watchdog"
    if name.startswith("tub"):
        return "issue_report"
    return "report"


def first_heading(text: str) -> str | None:
    for line in text.splitlines():
        heading = line.strip()
        if heading:
            return heading
    return None


def report_sort_key(path: Path) -> tuple[int, str]:
    return (path.stat().st_mtime_ns, path.name)


def isoformat_utc(timestamp: float) -> str:
    from datetime import datetime, timezone

    return datetime.fromtimestamp(timestamp, tz=timezone.utc).isoformat()


def watchdog_state_summary(project_root: Path) -> dict[str, Any]:
    state_path = project_root / ".paperclip" / "watchdog_state.json"
    if not state_path.exists():
        return {
            "exists": False,
            "path": str(state_path),
            "valid_json": False,
        }
    try:
        state = json.loads(state_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        return {
            "error": f"{type(exc).__name__}: {exc}",
            "exists": True,
            "path": str(state_path),
            "valid_json": False,
        }

    if not isinstance(state, dict):
        return {
            "error": "state root is not an object",
            "exists": True,
            "path": str(state_path),
            "valid_json": False,
        }

    dispatch = state.get("dispatch") if isinstance(state.get("dispatch"), dict) else {}
    attempts = dispatch.get("attempts") if isinstance(dispatch.get("attempts"), list) else []
    used_keys = dispatch.get("used_idempotency_keys")
    return {
        "dispatch": {
            "attempts_count": len(attempts),
            "last_dispatch_attempt_time": dispatch.get("last_dispatch_attempt_time"),
            "last_dispatch_result": dispatch.get("last_dispatch_result"),
            "last_issue_id": dispatch.get("last_issue_id"),
            "last_mode": dispatch.get("last_mode"),
            "last_reason": dispatch.get("last_reason"),
            "last_selected_agent_id": dispatch.get("last_selected_agent_id"),
            "last_task_key": dispatch.get("last_task_key"),
            "used_idempotency_keys_count": len(used_keys) if isinstance(used_keys, dict) else 0,
        },
        "exists": True,
        "last_created_at": state.get("last_created_at"),
        "last_issue_id": state.get("last_issue_id"),
        "last_task_key": state.get("last_task_key"),
        "path": str(state_path),
        "valid_json": True,
        "version": state.get("version"),
    }


def recommended_next_action(
    *,
    paperclip_api_ok: bool,
    company_id: str | None,
    active_runs_summary: dict[str, Any],
    latest_reporting_summary_path: str | None,
) -> dict[str, Any]:
    if not paperclip_api_ok:
        return {
            "action": "check_paperclip_availability",
            "reason": "Paperclip API is unavailable or unhealthy; keep bridge read-only and inspect local Paperclip.",
            "requires_approval": False,
            "write_permitted": False,
        }
    if not company_id:
        return {
            "action": "configure_company_id",
            "reason": "Paperclip is reachable, but no company id could be resolved for dashboard/runs.",
            "requires_approval": False,
            "write_permitted": False,
        }
    active_count = int(active_runs_summary.get("total") or 0)
    if active_count > 0:
        return {
            "action": "observe_active_runs",
            "reason": f"{active_count} active run(s) detected; do not dispatch new work.",
            "requires_approval": False,
            "write_permitted": False,
        }
    if not latest_reporting_summary_path:
        return {
            "action": "review_status_without_local_report",
            "reason": "No local reporting summary exists yet; use read-only status before proposing approvals.",
            "requires_approval": False,
            "write_permitted": False,
        }
    return {
        "action": "review_latest_report_and_prepare_manual_approval_if_needed",
        "reason": "No active runs detected; review the latest report before proposing any approval action.",
        "requires_approval": False,
        "write_permitted": False,
    }


def unavailable_summary(reason: str) -> dict[str, Any]:
    return {
        "available": False,
        "reason": reason,
    }


def extract_items(payload: Any, candidate_keys: list[str]) -> list[dict[str, Any]]:
    if isinstance(payload, list):
        return [item for item in payload if isinstance(item, dict)]
    if not isinstance(payload, dict):
        return []
    for key in candidate_keys:
        value = payload.get(key)
        if isinstance(value, list):
            return [item for item in value if isinstance(item, dict)]
        if isinstance(value, dict):
            nested = extract_items(value, candidate_keys)
            if nested:
                return nested
    for value in payload.values():
        if isinstance(value, list) and all(isinstance(item, dict) for item in value):
            return value
    return []


def count_by_status(items: list[dict[str, Any]]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for item in items:
        status = normalized_status(item.get("status") or item.get("state") or "unknown")
        counts[status] = counts.get(status, 0) + 1
    return dict(sorted(counts.items()))


def summarize_record(record: dict[str, Any], keys: list[str]) -> dict[str, Any]:
    return {key: record.get(key) for key in keys if key in record}


def agent_keys() -> list[str]:
    return [
        "id",
        "name",
        "agentName",
        "role",
        "title",
        "status",
        "adapterType",
        "lastHeartbeatAt",
        "updatedAt",
    ]


def run_keys() -> list[str]:
    return [
        "id",
        "status",
        "agentId",
        "agentName",
        "adapterType",
        "issueId",
        "createdAt",
        "startedAt",
        "lastOutputAt",
        "livenessState",
        "livenessReason",
        "nextAction",
    ]


def is_active_run(run: dict[str, Any]) -> bool:
    status = normalized_status(run.get("status") or run.get("state") or "")
    if status in TERMINAL_RUN_STATUSES:
        return False
    if status in ACTIVE_RUN_STATUSES:
        return True
    return bool(run.get("startedAt") and not (run.get("finishedAt") or run.get("completedAt")))


def normalized_status(value: Any) -> str:
    return str(value or "").strip().lower().replace("-", "_").replace(" ", "_")


def compact_value(value: Any, *, depth: int = 0) -> Any:
    if depth >= 4:
        return summarize_leaf(value)
    if isinstance(value, dict):
        return {str(key): compact_value(value[key], depth=depth + 1) for key in sorted(value)}
    if isinstance(value, list):
        return [compact_value(item, depth=depth + 1) for item in value[:50]]
    return value


def summarize_leaf(value: Any) -> Any:
    if isinstance(value, dict):
        return {"type": "object", "keys": sorted(str(key) for key in value.keys())[:25]}
    if isinstance(value, list):
        return {"type": "array", "length": len(value)}
    return value


def quote(value: str) -> str:
    return urllib.parse.quote(value, safe="")


def make_handler(
    config: BridgeConfig,
    client_factory: Callable[[BridgeConfig], Any] | None = None,
) -> type[BaseHTTPRequestHandler]:
    factory = client_factory or (lambda cfg: PaperclipClient(cfg.paperclip_api_url, cfg.paperclip_timeout_seconds))

    class GptControlBridgeHandler(BaseHTTPRequestHandler):
        server_version = f"{SERVICE_NAME}/1.0"

        def do_GET(self) -> None:
            parsed = urllib.parse.urlparse(self.path)
            if parsed.path == "/health":
                self.send_json(200, health_payload())
                return
            if parsed.path == "/v1/context/automixer":
                self.send_json(200, build_context(config, client=factory(config)))
                return
            if parsed.path == "/v1/knowledge/automixer/stats":
                self.send_json(200, vector_index_summary(config))
                return
            if parsed.path == "/v1/knowledge/automixer/search":
                params = urllib.parse.parse_qs(parsed.query)
                query = first_param(params, "q") or first_param(params, "query") or ""
                limit = parse_positive_int(first_param(params, "limit"), DEFAULT_VECTOR_LIMIT)
                payload = search_vector_index(config, query, limit=limit)
                self.send_json(200 if payload.get("available") else 400, payload)
                return
            self.send_json(404, {"error": "not_found", "read_only": True, "service": SERVICE_NAME})

        def do_POST(self) -> None:
            self.method_not_allowed()

        def do_PATCH(self) -> None:
            self.method_not_allowed()

        def do_PUT(self) -> None:
            self.method_not_allowed()

        def do_DELETE(self) -> None:
            self.method_not_allowed()

        def method_not_allowed(self) -> None:
            self.send_json(
                405,
                {
                    "error": "write_methods_disabled",
                    "read_only": True,
                    "safety_flags": safety_flags(),
                    "service": SERVICE_NAME,
                },
            )

        def send_json(self, status: int, payload: dict[str, Any]) -> None:
            body = json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Cache-Control", "no-store")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, format: str, *args: Any) -> None:
            return

    return GptControlBridgeHandler


def validate_bind_host(host: str) -> str:
    normalized = host.strip().lower()
    if normalized not in {"127.0.0.1", "localhost", "::1"}:
        raise ValueError("GPT control bridge must bind to loopback only")
    return host


def run_server(host: str, port: int, config: BridgeConfig) -> None:
    host = validate_bind_host(host)
    server = ThreadingHTTPServer((host, port), make_handler(config))
    print(f"{SERVICE_NAME} listening on http://{host}:{port}")
    print("read_only=true write_routes_enabled=false")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("stopped")
    finally:
        server.server_close()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Read-only Automixer GPT control bridge")
    parser.add_argument("--host", default=DEFAULT_BIND_HOST, help="loopback bind host")
    parser.add_argument("--port", type=int, default=DEFAULT_PORT, help="local port")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        run_server(args.host, args.port, build_config_from_env())
    except ValueError as exc:
        parser.error(str(exc))
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
