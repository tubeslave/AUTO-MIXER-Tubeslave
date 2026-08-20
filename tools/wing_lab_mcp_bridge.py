#!/usr/bin/env python3
"""Fixed-command JSON-lines bridge for the local WING lab MCP server."""

from __future__ import annotations

import atexit
import json
from pathlib import Path
import signal
import sys
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
BACKEND_ROOT = PROJECT_ROOT / "backend"
for import_root in (str(PROJECT_ROOT), str(BACKEND_ROOT)):
    if import_root not in sys.path:
        sys.path.insert(0, import_root)

from pipeline_entrypoint_catalog import load_pipeline_entrypoint_catalog  # noqa: E402
from wing_client import WingClient  # noqa: E402
from wing_lab_autowrite import WingLabAutowriteSession  # noqa: E402


def _transport_factory(host: str, port: int) -> WingClient:
    return WingClient(ip=host, port=port)


SESSION = WingLabAutowriteSession(_transport_factory)


def _cleanup() -> None:
    try:
        if SESSION.status()["armed"]:
            SESSION.emergency_stop(reason="mcp_bridge_exit")
    except Exception:
        pass


atexit.register(_cleanup)


def _signal_handler(_signum: int, _frame: Any) -> None:
    _cleanup()
    raise SystemExit(128 + int(_signum))


for _signal in (signal.SIGINT, signal.SIGTERM):
    signal.signal(_signal, _signal_handler)


def dispatch(method: str, params: dict[str, Any]) -> dict[str, Any]:
    if method == "pipeline_catalog":
        return load_pipeline_entrypoint_catalog(PROJECT_ROOT)
    if method == "status":
        return SESSION.status()
    if method == "arm":
        return SESSION.arm(
            host=params.get("host"),
            port=params.get("port", 2223),
            console_model=params.get("console_model", ""),
            duration_sec=params.get("duration_sec", 900),
            max_corrections=params.get("max_corrections", 128),
        )
    if method == "apply_decisions":
        return SESSION.apply_decisions(params.get("decisions"))
    if method == "trace":
        return SESSION.trace()
    if method == "stop":
        return SESSION.stop(restore=True)
    if method == "emergency_stop":
        return SESSION.emergency_stop(
            reason=str(params.get("reason") or "operator_emergency_stop")
        )
    raise ValueError("wing_lab_bridge_method_not_allowed")


def main() -> int:
    for raw_line in sys.stdin:
        request: Any = None
        try:
            request = json.loads(raw_line)
            if not isinstance(request, dict) or not isinstance(
                request.get("params", {}), dict
            ):
                raise ValueError("wing_lab_bridge_request_invalid")
            result = dispatch(
                str(request.get("method") or ""), request.get("params", {})
            )
            response = {"id": request.get("id"), "result": result}
        except Exception as exc:
            response = {
                "id": request.get("id") if isinstance(request, dict) else None,
                "error": {"type": type(exc).__name__, "message": str(exc)},
            }
        sys.stdout.write(
            json.dumps(response, ensure_ascii=False, sort_keys=True) + "\n"
        )
        sys.stdout.flush()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
