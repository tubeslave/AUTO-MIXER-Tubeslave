"""REST transport helpers for Mixing Station Desktop."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional

import requests

from .models import MixingStationCommand


UNAVAILABLE_MESSAGE = (
    "Mixing Station REST API unavailable. Enable it in Global App Settings."
)


@dataclass(frozen=True)
class TransportResult:
    """Transport send/read result."""

    success: bool
    data: Any = None
    error: Optional[str] = None
    status_code: Optional[int] = None


class MixingStationRestClient:
    """Small REST client with conservative endpoint assumptions."""

    def __init__(
        self,
        host: str = "127.0.0.1",
        port: int = 8080,
        *,
        timeout: float = 2.0,
        command_endpoint: str = "",
        session: Optional[requests.Session] = None,
        app_state_paths: Optional[List[str]] = None,
        mixer_state_paths: Optional[List[str]] = None,
        discovery_paths: Optional[List[str]] = None,
    ):
        self.host = host
        self.port = int(port)
        self.timeout = float(timeout)
        self.command_endpoint = command_endpoint
        self.session = session or requests.Session()
        self.app_state_paths = app_state_paths or [
            "/app/state",
            "/api/app/state",
            "/api/v1/app/state",
        ]
        self.mixer_state_paths = mixer_state_paths or [
            "/mixer/state",
            "/api/mixer/state",
            "/api/v1/mixer/state",
        ]
        self.discovery_paths = discovery_paths or [
            "/console/data/paths",
            "/api/explorer",
            "/api/paths",
            "/api/v1/paths",
            "/openapi.json",
        ]

    @property
    def base_url(self) -> str:
        return f"http://{self.host}:{self.port}"

    def health_check(self) -> TransportResult:
        """Return success if any known read endpoint responds."""
        result = self.get_app_state()
        if result.success:
            return result
        return TransportResult(success=False, error=UNAVAILABLE_MESSAGE)

    def get_app_state(self) -> TransportResult:
        return self._first_success(self.app_state_paths)

    def get_mixer_state(self) -> TransportResult:
        return self._first_success(self.mixer_state_paths)

    def discover_available_paths(self) -> TransportResult:
        """Fetch likely discovery endpoints and extract dataPath-like strings."""
        payloads: List[Any] = []
        for path in [*self.discovery_paths, *self.app_state_paths, *self.mixer_state_paths]:
            result = self._request("GET", path)
            if result.success:
                payloads.append(result.data)
        if not payloads:
            return TransportResult(success=False, error=UNAVAILABLE_MESSAGE)
        paths = sorted(_extract_data_paths(payloads))
        return TransportResult(success=True, data={"paths": paths, "raw_count": len(payloads)})

    def send_command(self, command: MixingStationCommand) -> TransportResult:
        """Send a mapped command through a configured REST write endpoint."""
        if not self.command_endpoint:
            return TransportResult(
                success=False,
                error=(
                    "Mixing Station REST write endpoint is not configured. "
                    "Use the local API Explorer to discover it, then set "
                    "rest_command_endpoint in config/mixing_station.yaml."
                ),
            )
        if "{data_path}" in self.command_endpoint or "{value_format}" in self.command_endpoint:
            path = self.command_endpoint.format(
                data_path=command.data_path,
                value_format="norm" if command.value_format == "normalized" else "val",
            )
            return self._request("POST", path, json={"value": command.value})

        payload = {
            "path": command.data_path,
            "method": command.method,
            "body": {
                "value": command.value,
                "format": command.value_format,
            },
        }
        return self._request("POST", self.command_endpoint, json=payload)

    def _first_success(self, paths: List[str]) -> TransportResult:
        last_error = UNAVAILABLE_MESSAGE
        for path in paths:
            result = self._request("GET", path)
            if result.success:
                return result
            last_error = result.error or last_error
        return TransportResult(success=False, error=last_error)

    def _request(self, method: str, path: str, **kwargs) -> TransportResult:
        url = f"{self.base_url}{_normalize_path(path)}"
        try:
            response = self.session.request(method, url, timeout=self.timeout, **kwargs)
            status_code = int(response.status_code)
            if status_code >= 400:
                return TransportResult(
                    success=False,
                    error=f"HTTP {status_code} from Mixing Station REST API",
                    status_code=status_code,
                )
            if response.content:
                try:
                    data = response.json()
                except ValueError:
                    data = response.text
            else:
                data = None
            return TransportResult(success=True, data=data, status_code=status_code)
        except requests.RequestException as exc:
            return TransportResult(success=False, error=f"{UNAVAILABLE_MESSAGE} {exc}")


def _normalize_path(path: str) -> str:
    return path if path.startswith("/") else f"/{path}"


def _extract_data_paths(payloads: Iterable[Any]) -> set[str]:
    found: set[str] = set()

    def visit_mixing_station_tree(node: Any, prefix: str = "") -> None:
        if not isinstance(node, dict):
            return
        for item in node.get("val") or []:
            if isinstance(item, str) and item:
                found.add(f"{prefix}{item}".strip("."))
        for key, child in (node.get("child") or {}).items():
            visit_mixing_station_tree(child, f"{prefix}{key}.")

    def visit(value: Any) -> None:
        if isinstance(value, dict):
            for key, item in value.items():
                if key in {"dataPath", "data_path", "path", "id"} and isinstance(item, str):
                    if item and not item.startswith(("http://", "https://")):
                        found.add(item.strip("/"))
                visit(item)
        elif isinstance(value, list):
            for item in value:
                visit(item)

    payload_list = list(payloads)
    for payload in payload_list:
        visit_mixing_station_tree(payload)
    visit(payload_list)
    return found
