"""High-level safe adapter for Mixing Station Desktop."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

from .capabilities import CapabilityMap, CapabilityStatus, load_capability_map
from .config import MixingStationConfig, normalize_console_profile, resolve_repo_path
from .logger import MixingStationJSONLLogger
from .mapper import MappingResult, MixingStationMapper, load_mapper
from .models import AutomixCorrection, CorrectionResult, MixingStationCommand
from .osc_client import MixingStationOSCClient
from .rest_client import MixingStationRestClient, TransportResult
from .safety import (
    MixingStationSafetyConfig,
    MixingStationSafetyLayer,
    SafetyValidation,
)
from .websocket_client import MixingStationWebSocketClient


class MixingStationAdapter:
    """Validate, map, log, and optionally send Automix corrections."""

    def __init__(
        self,
        config: Optional[MixingStationConfig] = None,
        *,
        capability_map: Optional[CapabilityMap] = None,
        mapper: Optional[MixingStationMapper] = None,
        safety_layer: Optional[MixingStationSafetyLayer] = None,
        rest_client: Optional[MixingStationRestClient] = None,
        websocket_client: Optional[MixingStationWebSocketClient] = None,
        osc_client: Optional[MixingStationOSCClient] = None,
        logger: Optional[MixingStationJSONLLogger] = None,
    ):
        self.config = config or MixingStationConfig()
        self.config.console_profile = normalize_console_profile(self.config.console_profile)
        self.capability_map = capability_map or load_capability_map(self.config.capabilities_file)
        self.mapper = mapper or load_mapper(
            self.config.mapping_file,
            default_transport=self.config.transport,
        )
        safety_config = MixingStationSafetyConfig.from_file(
            self.config.safety_file,
            emergency_stop_file=self.config.emergency_stop_file,
        )
        self.safety_layer = safety_layer or MixingStationSafetyLayer(safety_config)
        self.rest_client = rest_client or MixingStationRestClient(
            host=self.config.host,
            port=self.config.rest_port,
            command_endpoint=self.config.rest_command_endpoint,
            app_state_paths=self.config.rest_app_state_paths,
            mixer_state_paths=self.config.rest_mixer_state_paths,
            discovery_paths=self.config.rest_discovery_paths,
        )
        self.websocket_client = websocket_client or MixingStationWebSocketClient(
            host=self.config.host,
            port=self.config.rest_port,
            path=self.config.websocket_path,
            command_path_template=self.config.websocket_command_path_template,
        )
        self.osc_client = osc_client or MixingStationOSCClient(
            host=self.config.host,
            port=self.config.osc_port,
        )
        self.logger = logger or MixingStationJSONLLogger(self.config.log_jsonl)
        self.connected = False

    @classmethod
    def from_config_file(
        cls,
        path: str | Path = "config/mixing_station.yaml",
    ) -> "MixingStationAdapter":
        return cls(MixingStationConfig.from_file(path))

    def connect(self) -> bool:
        """Connect or mark dry-run mode ready."""
        if self.config.dry_run:
            self.connected = True
            return True
        health = self.health_check()
        self.connected = bool(health.get("success", False))
        return self.connected

    def disconnect(self) -> None:
        self.connected = False

    def health_check(self) -> Dict[str, Any]:
        """Check REST API availability."""
        result = self.rest_client.health_check()
        return {
            "success": result.success,
            "error": result.error,
            "data": result.data,
            "host": self.config.host,
            "rest_port": self.config.rest_port,
        }

    def get_app_state(self) -> Dict[str, Any]:
        result = self.rest_client.get_app_state()
        return _transport_to_dict(result)

    def get_mixer_state(self) -> Dict[str, Any]:
        result = self.rest_client.get_mixer_state()
        return _transport_to_dict(result)

    def discover_available_paths(self, out: str | Path | None = None) -> Dict[str, Any]:
        result = self.rest_client.discover_available_paths()
        payload = _transport_to_dict(result)
        if out and result.success:
            import json

            path = resolve_repo_path(out)
            path.parent.mkdir(parents=True, exist_ok=True)
            with path.open("w", encoding="utf-8") as handle:
                json.dump(payload.get("data", {}), handle, indent=2, ensure_ascii=True)
        return payload

    def send_batch(self, corrections: Iterable[AutomixCorrection]) -> List[CorrectionResult]:
        return [self.send_correction(correction) for correction in corrections]

    def send_correction(self, correction: AutomixCorrection) -> CorrectionResult:
        """Validate, map, log, and optionally send a single correction."""
        correction = AutomixCorrection.from_dict(correction.to_dict())
        correction.console_profile = normalize_console_profile(correction.console_profile)
        correction.mode = correction.mode or self.config.mode
        correction.dry_run = bool(self.config.dry_run)

        requested_value = correction.value
        validation = self.validate_correction(correction)
        correction = validation.correction

        if not validation.allowed:
            result = CorrectionResult(
                correction=correction,
                success=False,
                dry_run=correction.dry_run,
                requested_value=requested_value,
                sent_value=None,
                safety_status=validation.status,
                error=validation.message,
                message=validation.message,
                blocked=True,
            )
            self.log_correction_result(correction, result)
            return result

        capability = self._capability_status(correction)
        if not _capability_allows(correction, capability, self.config.live_control):
            message = _capability_message(correction, capability)
            result = CorrectionResult(
                correction=correction,
                success=False,
                dry_run=correction.dry_run,
                requested_value=requested_value,
                sent_value=None,
                safety_status="blocked",
                error=message,
                message=message,
                blocked=True,
            )
            self.log_correction_result(correction, result)
            return result

        mapping = self.map_correction_to_mixing_station_path(correction)
        if not mapping.success or mapping.command is None:
            message = mapping.error or "mapping failed"
            result = CorrectionResult(
                correction=correction,
                success=False,
                dry_run=correction.dry_run,
                requested_value=requested_value,
                sent_value=None,
                safety_status="blocked",
                error=message,
                message=message,
                blocked=True,
            )
            self.log_correction_result(correction, result)
            return result

        command = mapping.command
        if correction.dry_run:
            result = self.dry_run_correction(correction, command, requested_value)
            self.safety_layer.record_sent(correction)
            self.log_correction_result(correction, result)
            return result

        transport_result = self._send_command(command)
        result = CorrectionResult(
            correction=correction,
            success=transport_result.success,
            dry_run=False,
            sent=transport_result.success,
            transport=command.transport,
            data_path=command.data_path,
            sent_value=command.value if transport_result.success else None,
            requested_value=requested_value,
            safety_status=correction.safety_status,
            error=transport_result.error,
            message="sent" if transport_result.success else (transport_result.error or "send failed"),
            blocked=False,
            command=command,
        )
        if transport_result.success:
            self.safety_layer.record_sent(correction)
        self.log_correction_result(correction, result)
        return result

    def dry_run_correction(
        self,
        correction: AutomixCorrection,
        command: Optional[MixingStationCommand] = None,
        requested_value: Any = None,
    ) -> CorrectionResult:
        """Return a successful dry-run result without network I/O."""
        if command is None:
            mapping = self.map_correction_to_mixing_station_path(correction)
            command = mapping.command if mapping.success else None
        return CorrectionResult(
            correction=correction,
            success=True,
            dry_run=True,
            sent=False,
            transport=command.transport if command else self.config.transport,
            data_path=command.data_path if command else None,
            sent_value=command.value if command else correction.value,
            requested_value=correction.value if requested_value is None else requested_value,
            safety_status=correction.safety_status,
            message="dry-run",
            command=command,
        )

    def validate_correction(self, correction: AutomixCorrection) -> SafetyValidation:
        return self.safety_layer.validate(
            correction,
            live_control_enabled=bool(self.config.live_control),
        )

    def map_correction_to_mixing_station_path(
        self,
        correction: AutomixCorrection,
    ) -> MappingResult:
        return self.mapper.map(correction)

    def log_correction_result(
        self,
        correction: AutomixCorrection,
        result: CorrectionResult,
    ) -> None:
        self.logger.log_result(result)

    def subscribe_to_parameter(self, path: str) -> Dict[str, Any]:
        return _transport_to_dict(self.websocket_client.subscribe_to_parameter(path))

    def subscribe_to_metering(self, path: str) -> Dict[str, Any]:
        return _transport_to_dict(self.websocket_client.subscribe_to_metering(path))

    def set_emergency_stop(self, active: bool = True) -> None:
        self.safety_layer.set_emergency_stop(active)

    def _send_command(self, command: MixingStationCommand) -> TransportResult:
        transport = command.transport or self.config.transport
        if transport == "rest":
            return self.rest_client.send_command(command)
        if transport == "websocket":
            return self.websocket_client.send_command(command)
        if transport == "osc":
            return self.osc_client.send_command(command)
        return TransportResult(success=False, error=f"Unsupported transport: {transport}")

    def _capability_status(self, correction: AutomixCorrection) -> CapabilityStatus:
        return self.capability_map.status_for(correction.parameter)


def _transport_to_dict(result: TransportResult) -> Dict[str, Any]:
    return {
        "success": result.success,
        "data": result.data,
        "error": result.error,
        "status_code": result.status_code,
    }


def _capability_allows(
    correction: AutomixCorrection,
    capability: CapabilityStatus,
    live_control: bool,
) -> bool:
    if not capability.supported or capability.needs_discovery:
        return False
    if live_control or correction.mode == "live_control":
        return capability.can_live_control
    return capability.can_write_visualization


def _capability_message(correction: AutomixCorrection, capability: CapabilityStatus) -> str:
    if capability.needs_discovery:
        return f"{correction.parameter} needs Mixing Station API Explorer discovery"
    if not capability.supported:
        return f"{correction.parameter} is unsupported for {correction.console_profile}"
    if capability.read_only:
        return f"{correction.parameter} is read-only for {correction.console_profile}"
    if capability.forbidden_live:
        return f"{correction.parameter} is forbidden for live-control"
    return capability.reason or f"{correction.parameter} is not writable"
