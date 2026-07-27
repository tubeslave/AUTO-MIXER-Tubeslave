"""Immutable normalized model used by System Inspector replay."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any, Mapping, Optional, Tuple


class CapabilityState(str, Enum):
    SUPPORTED = "SUPPORTED"
    PARTIALLY_SUPPORTED = "PARTIALLY_SUPPORTED"
    IMPORT_ONLY = "IMPORT_ONLY"
    MANUAL_CONFIRMATION = "MANUAL_CONFIRMATION"
    UNAVAILABLE = "UNAVAILABLE"
    EXPERIMENTAL = "EXPERIMENTAL"


class Availability(str, Enum):
    ONLINE = "ONLINE"
    OFFLINE = "OFFLINE"
    UNKNOWN = "UNKNOWN"


class Severity(str, Enum):
    CRITICAL = "CRITICAL"
    HIGH = "HIGH"
    MEDIUM = "MEDIUM"
    LOW = "LOW"
    INFO = "INFO"
    UNKNOWN = "UNKNOWN"


class CheckStatus(str, Enum):
    OK = "OK"
    MISMATCH = "MISMATCH"
    UNKNOWN = "UNKNOWN"


@dataclass(frozen=True)
class EntityMetadata:
    stable_id: str
    display_name: str
    vendor: str = ""
    model: str = ""
    address: str = ""
    protocol: str = ""
    availability: Availability = Availability.UNKNOWN
    last_seen: str = ""
    source: str = ""
    confidence: float = 0.0
    capabilities: Tuple[str, ...] = ()
    raw_reference: str = ""


@dataclass(frozen=True)
class Device:
    metadata: EntityMetadata
    kind: str
    firmware: str = ""
    sample_rate: Optional[int] = None


@dataclass(frozen=True)
class NetworkInterface:
    metadata: EntityMetadata
    bsd_name: str
    friendly_name: str
    mac_address: str = ""
    ipv4: Tuple[str, ...] = ()
    ipv6: Tuple[str, ...] = ()
    subnets: Tuple[str, ...] = ()
    link_status: str = "UNKNOWN"
    speed_mbps: Optional[int] = None
    service_order: Optional[int] = None
    default_route: bool = False
    roles: Tuple[str, ...] = ()


@dataclass(frozen=True)
class AudioEndpoint:
    metadata: EntityMetadata
    device_id: str
    direction: str
    channel_name: str
    sample_rate: Optional[int] = None


@dataclass(frozen=True)
class TransmitChannel:
    metadata: EntityMetadata
    device_id: str
    channel_name: str
    sample_rate: Optional[int] = None


@dataclass(frozen=True)
class ReceiveChannel:
    metadata: EntityMetadata
    device_id: str
    channel_name: str
    sample_rate: Optional[int] = None


@dataclass(frozen=True)
class Subscription:
    metadata: EntityMetadata
    role: str
    source_endpoint_id: str
    destination_endpoint_id: str
    status: str


@dataclass(frozen=True)
class ConsoleInput:
    metadata: EntityMetadata
    console_device_id: str
    console_channel: str
    source_endpoint_id: str
    main_assignments: Tuple[str, ...] = ()
    buses: Tuple[str, ...] = ()
    stereo_linked_to: str = ""


@dataclass(frozen=True)
class ConsoleChannel:
    metadata: EntityMetadata
    console_device_id: str
    channel: str
    input_id: str = ""


@dataclass(frozen=True)
class Bus:
    metadata: EntityMetadata
    console_device_id: str
    bus: str


@dataclass(frozen=True)
class MainOutput:
    metadata: EntityMetadata
    console_device_id: str
    output: str


@dataclass(frozen=True)
class ClockDomain:
    metadata: EntityMetadata
    sample_rate: Optional[int]
    master_device_id: str = ""
    sync_status: str = "UNKNOWN"


@dataclass(frozen=True)
class AudioSignal:
    metadata: EntityMetadata
    endpoint_id: str
    signal_present: Optional[bool] = None
    rms_dbfs: Optional[float] = None
    peak_dbfs: Optional[float] = None


@dataclass(frozen=True)
class SoftwareProcess:
    metadata: EntityMetadata
    computer_id: str
    running: Optional[bool] = None


@dataclass(frozen=True)
class Computer:
    metadata: EntityMetadata
    operating_system: str
    version: str = ""


@dataclass(frozen=True)
class CableOrLogicalLink:
    metadata: EntityMetadata
    source_id: str
    destination_id: str
    link_type: str
    status: str = "UNKNOWN"


@dataclass(frozen=True)
class ExpectedRoute:
    role: str
    source_endpoint_id: str
    destination_endpoint_id: str
    console_channel: str
    forbidden_destinations: Tuple[str, ...] = ()


@dataclass(frozen=True)
class ObservedRoute:
    role: str
    source_endpoint_id: str
    destination_endpoint_id: str
    console_channel: str
    main_assignments: Tuple[str, ...]
    subscription_status: str
    confidence: float
    evidence_references: Tuple[str, ...]


@dataclass(frozen=True)
class RouteComparison:
    role: str
    status: CheckStatus
    expected: Optional[ExpectedRoute]
    observed: Optional[ObservedRoute]
    differences: Tuple[str, ...] = ()
    missing_evidence: Tuple[str, ...] = ()


@dataclass(frozen=True)
class Evidence:
    source: str
    path: str
    value: Any
    limitation: str = ""


@dataclass(frozen=True)
class Finding:
    id: str
    severity: Severity
    category: str
    title: str
    summary: str
    evidence: Tuple[Evidence, ...]
    confidence: float
    impact: str
    recommended_action: str
    automatic_fix_available: bool = False
    requires_manual_confirmation: bool = True
    status: CheckStatus = CheckStatus.MISMATCH
    missing_evidence: Tuple[str, ...] = ()


@dataclass(frozen=True)
class SystemGraph:
    devices: Tuple[Device, ...] = ()
    network_interfaces: Tuple[NetworkInterface, ...] = ()
    audio_endpoints: Tuple[AudioEndpoint, ...] = ()
    transmit_channels: Tuple[TransmitChannel, ...] = ()
    receive_channels: Tuple[ReceiveChannel, ...] = ()
    subscriptions: Tuple[Subscription, ...] = ()
    console_inputs: Tuple[ConsoleInput, ...] = ()
    console_channels: Tuple[ConsoleChannel, ...] = ()
    buses: Tuple[Bus, ...] = ()
    main_outputs: Tuple[MainOutput, ...] = ()
    clock_domains: Tuple[ClockDomain, ...] = ()
    audio_signals: Tuple[AudioSignal, ...] = ()
    software_processes: Tuple[SoftwareProcess, ...] = ()
    computers: Tuple[Computer, ...] = ()
    links: Tuple[CableOrLogicalLink, ...] = ()
    expected_routes: Tuple[ExpectedRoute, ...] = ()
    observed_routes: Tuple[ObservedRoute, ...] = ()
    route_comparisons: Tuple[RouteComparison, ...] = ()


@dataclass(frozen=True)
class CategoryAssessment:
    name: str
    score: Optional[int]
    status: CheckStatus
    confidence: float
    evidence_count: int
    unknown_checks: Tuple[str, ...] = ()


@dataclass(frozen=True)
class ReadinessAssessment:
    score: Optional[int]
    status: CheckStatus
    confidence: float
    observed_coverage: float
    categories: Tuple[CategoryAssessment, ...]
    unknown_checks: Tuple[str, ...]
    drivers: Tuple[str, ...]
    timestamp: str
    data_freshness: Mapping[str, str] = field(default_factory=dict)


@dataclass(frozen=True)
class InspectorReport:
    schema_version: str
    profile_name: str
    baseline_id: str
    read_only: bool
    graph: SystemGraph
    findings: Tuple[Finding, ...]
    readiness: ReadinessAssessment

    def to_dict(self) -> dict[str, Any]:
        return _serialize(asdict(self))


def _serialize(value: Any) -> Any:
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, dict):
        return {str(key): _serialize(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_serialize(item) for item in value]
    return value
