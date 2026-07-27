"""Normalize imported snapshots into the Inspector system graph."""

from __future__ import annotations

from typing import Any, Mapping, Optional

from .io import (
    InspectorInputError,
    SnapshotBundle,
    require_list,
    require_mapping,
    require_text,
    validate_unique_ids,
)
from .model import (
    AudioEndpoint,
    Availability,
    CheckStatus,
    ConsoleInput,
    Device,
    EntityMetadata,
    ExpectedRoute,
    NetworkInterface,
    ObservedRoute,
    RouteComparison,
    Subscription,
    SystemGraph,
)
from .profile import ShowProfile


def build_system_graph(profile: ShowProfile, bundle: SnapshotBundle) -> SystemGraph:
    devices = _devices(bundle)
    interfaces = _interfaces(bundle.macos)
    endpoints = _endpoints(bundle.dante)
    subscriptions = _subscriptions(bundle.dante)
    console_inputs = _console_inputs(bundle.wing)
    expected_routes = _expected_routes(profile, bundle.baseline)
    observed_routes = _observed_routes(subscriptions, console_inputs)
    comparisons = _compare_routes(expected_routes, observed_routes)
    return SystemGraph(
        devices=tuple(sorted(devices, key=lambda item: item.metadata.stable_id)),
        network_interfaces=tuple(
            sorted(interfaces, key=lambda item: item.metadata.stable_id)
        ),
        audio_endpoints=tuple(
            sorted(endpoints, key=lambda item: item.metadata.stable_id)
        ),
        subscriptions=tuple(
            sorted(subscriptions, key=lambda item: item.metadata.stable_id)
        ),
        console_inputs=tuple(
            sorted(console_inputs, key=lambda item: item.metadata.stable_id)
        ),
        expected_routes=tuple(sorted(expected_routes, key=lambda item: item.role)),
        observed_routes=tuple(sorted(observed_routes, key=lambda item: item.role)),
        route_comparisons=tuple(sorted(comparisons, key=lambda item: item.role)),
    )


def _devices(bundle: SnapshotBundle) -> list[Device]:
    rows: list[tuple[Mapping[str, Any], str, str]] = []
    computer = require_mapping(bundle.macos, "computer", "macos")
    rows.append((computer, "computer", "macos_snapshot.computer"))
    wing_device = require_mapping(bundle.wing, "device", "wing")
    rows.append((wing_device, "console", "wing_snapshot.device"))
    dante_devices = require_list(bundle.dante, "devices", "dante")
    validate_unique_ids(dante_devices, "dante.devices")
    for index, row in enumerate(dante_devices):
        if not isinstance(row, dict):
            raise InspectorInputError(f"dante.devices[{index}]:object_required")
        rows.append((row, str(row.get("kind", "dante_device")), f"dante_snapshot.devices[{index}]"))

    devices: list[Device] = []
    seen: set[str] = set()
    for row, kind, reference in rows:
        stable_id = require_text(row, "id", reference)
        if stable_id in seen:
            raise InspectorInputError(f"system.devices:duplicate_id:{stable_id}")
        seen.add(stable_id)
        devices.append(
            Device(
                metadata=_metadata(row, reference),
                kind=kind,
                firmware=str(row.get("firmware", "")),
                sample_rate=_optional_int(row.get("sample_rate")),
            )
        )
    return devices


def _interfaces(snapshot: Mapping[str, Any]) -> list[NetworkInterface]:
    rows = require_list(snapshot, "network_interfaces", "macos")
    validate_unique_ids(rows, "macos.network_interfaces")
    interfaces: list[NetworkInterface] = []
    for index, row in enumerate(rows):
        reference = f"macos_snapshot.network_interfaces[{index}]"
        interfaces.append(
            NetworkInterface(
                metadata=_metadata(row, reference),
                bsd_name=require_text(row, "bsd_name", reference),
                friendly_name=require_text(row, "friendly_name", reference),
                mac_address=str(row.get("mac_address", "")),
                ipv4=_string_tuple(row.get("ipv4", []), f"{reference}.ipv4"),
                ipv6=_string_tuple(row.get("ipv6", []), f"{reference}.ipv6"),
                subnets=_string_tuple(row.get("subnets", []), f"{reference}.subnets"),
                link_status=str(row.get("link_status", "UNKNOWN")).upper(),
                speed_mbps=_optional_int(row.get("speed_mbps")),
                service_order=_optional_int(row.get("service_order")),
                default_route=bool(row.get("default_route", False)),
                roles=_string_tuple(row.get("roles", []), f"{reference}.roles"),
            )
        )
    return interfaces


def _endpoints(snapshot: Mapping[str, Any]) -> list[AudioEndpoint]:
    rows = require_list(snapshot, "endpoints", "dante")
    validate_unique_ids(rows, "dante.endpoints")
    endpoints: list[AudioEndpoint] = []
    for index, row in enumerate(rows):
        reference = f"dante_snapshot.endpoints[{index}]"
        endpoints.append(
            AudioEndpoint(
                metadata=_metadata(row, reference),
                device_id=require_text(row, "device_id", reference),
                direction=require_text(row, "direction", reference).upper(),
                channel_name=require_text(row, "channel_name", reference),
                sample_rate=_optional_int(row.get("sample_rate")),
            )
        )
    return endpoints


def _subscriptions(snapshot: Mapping[str, Any]) -> list[Subscription]:
    rows = require_list(snapshot, "subscriptions", "dante")
    validate_unique_ids(rows, "dante.subscriptions")
    subscriptions: list[Subscription] = []
    seen_roles: set[str] = set()
    for index, row in enumerate(rows):
        reference = f"dante_snapshot.subscriptions[{index}]"
        role = require_text(row, "role", reference)
        if role in seen_roles:
            raise InspectorInputError(f"dante.subscriptions:duplicate_role:{role}")
        seen_roles.add(role)
        subscriptions.append(
            Subscription(
                metadata=_metadata(
                    row,
                    f"dante_snapshot.subscriptions.{role}",
                    default_protocol="Dante",
                ),
                role=role,
                source_endpoint_id=require_text(row, "source_endpoint_id", reference),
                destination_endpoint_id=require_text(
                    row, "destination_endpoint_id", reference
                ),
                status=str(row.get("status", "UNKNOWN")).upper(),
            )
        )
    return subscriptions


def _console_inputs(snapshot: Mapping[str, Any]) -> list[ConsoleInput]:
    rows = require_list(snapshot, "console_inputs", "wing")
    validate_unique_ids(rows, "wing.console_inputs")
    inputs: list[ConsoleInput] = []
    for index, row in enumerate(rows):
        reference = f"wing_snapshot.console_inputs[{index}]"
        inputs.append(
            ConsoleInput(
                metadata=_metadata(row, reference, default_protocol="OSC snapshot"),
                console_device_id=require_text(row, "console_device_id", reference),
                console_channel=require_text(row, "console_channel", reference),
                source_endpoint_id=require_text(row, "source_endpoint_id", reference),
                main_assignments=_string_tuple(
                    row.get("main_assignments", []),
                    f"{reference}.main_assignments",
                ),
                buses=_string_tuple(row.get("buses", []), f"{reference}.buses"),
                stereo_linked_to=str(row.get("stereo_linked_to", "")),
            )
        )
    return inputs


def _expected_routes(
    profile: ShowProfile, baseline: Mapping[str, Any]
) -> list[ExpectedRoute]:
    baseline_id = require_text(baseline, "baseline_id", "baseline")
    del baseline_id
    rows = require_list(baseline, "expected_routes", "baseline")
    roles = {role.name: role for role in profile.roles}
    expected: list[ExpectedRoute] = []
    seen: set[str] = set()
    for index, row in enumerate(rows):
        if not isinstance(row, dict):
            raise InspectorInputError(f"baseline.expected_routes[{index}]:object_required")
        role_name = require_text(row, "role", f"baseline.expected_routes[{index}]")
        if role_name in seen:
            raise InspectorInputError(f"baseline.expected_routes:duplicate_role:{role_name}")
        seen.add(role_name)
        role_profile = roles.get(role_name)
        if role_profile is None:
            raise InspectorInputError(
                f"baseline.expected_routes:role_not_in_profile:{role_name}"
            )
        source = require_text(row, "source_endpoint_id", f"baseline.{role_name}")
        destination = require_text(
            row, "destination_endpoint_id", f"baseline.{role_name}"
        )
        channel = require_text(row, "console_channel", f"baseline.{role_name}")
        if source != role_profile.expected_source:
            raise InspectorInputError(f"baseline.{role_name}:source_profile_mismatch")
        if destination != role_profile.expected_destination:
            raise InspectorInputError(
                f"baseline.{role_name}:destination_profile_mismatch"
            )
        if channel != role_profile.expected_console_channel:
            raise InspectorInputError(f"baseline.{role_name}:channel_profile_mismatch")
        expected.append(
            ExpectedRoute(
                role=role_name,
                source_endpoint_id=source,
                destination_endpoint_id=destination,
                console_channel=channel,
                forbidden_destinations=role_profile.forbidden_destinations,
            )
        )
    missing = sorted(set(roles) - seen)
    if missing:
        raise InspectorInputError(
            "baseline.expected_routes:missing_profile_roles:" + ",".join(missing)
        )
    return expected


def _observed_routes(
    subscriptions: list[Subscription],
    console_inputs: list[ConsoleInput],
) -> list[ObservedRoute]:
    console_by_source = {
        item.source_endpoint_id: item for item in console_inputs
    }
    routes: list[ObservedRoute] = []
    for subscription in subscriptions:
        console = console_by_source.get(subscription.destination_endpoint_id)
        references = [subscription.metadata.raw_reference]
        if console is not None:
            references.append(console.metadata.raw_reference)
        routes.append(
            ObservedRoute(
                role=subscription.role,
                source_endpoint_id=subscription.source_endpoint_id,
                destination_endpoint_id=subscription.destination_endpoint_id,
                console_channel=console.console_channel if console else "",
                main_assignments=console.main_assignments if console else (),
                subscription_status=subscription.status,
                confidence=min(
                    subscription.metadata.confidence,
                    console.metadata.confidence if console else 0.0,
                ),
                evidence_references=tuple(references),
            )
        )
    return routes


def _compare_routes(
    expected_routes: list[ExpectedRoute],
    observed_routes: list[ObservedRoute],
) -> list[RouteComparison]:
    observed_by_role = {route.role: route for route in observed_routes}
    comparisons: list[RouteComparison] = []
    for expected in expected_routes:
        observed = observed_by_role.get(expected.role)
        if observed is None:
            comparisons.append(
                RouteComparison(
                    role=expected.role,
                    status=CheckStatus.UNKNOWN,
                    expected=expected,
                    observed=None,
                    missing_evidence=(
                        f"dante_snapshot.subscriptions.{expected.role}",
                    ),
                )
            )
            continue
        differences: list[str] = []
        if observed.source_endpoint_id != expected.source_endpoint_id:
            differences.append("source_endpoint")
        if observed.destination_endpoint_id != expected.destination_endpoint_id:
            differences.append("destination_endpoint")
        if not observed.console_channel:
            differences.append("wing_patch_missing")
        elif observed.console_channel != expected.console_channel:
            differences.append("console_channel")
        if observed.subscription_status != "PRESENT":
            differences.append("subscription_status")
        comparisons.append(
            RouteComparison(
                role=expected.role,
                status=CheckStatus.MISMATCH if differences else CheckStatus.OK,
                expected=expected,
                observed=observed,
                differences=tuple(differences),
            )
        )
    return comparisons


def _metadata(
    row: Mapping[str, Any],
    raw_reference: str,
    *,
    default_protocol: str = "",
) -> EntityMetadata:
    confidence = row.get("confidence", 0.0)
    if isinstance(confidence, bool) or not isinstance(confidence, (int, float)):
        raise InspectorInputError(f"{raw_reference}.confidence:number_required")
    if not 0.0 <= float(confidence) <= 1.0:
        raise InspectorInputError(f"{raw_reference}.confidence:out_of_range")
    availability_raw = str(row.get("availability", "UNKNOWN")).upper()
    try:
        availability = Availability(availability_raw)
    except ValueError as exc:
        raise InspectorInputError(
            f"{raw_reference}.availability:invalid:{availability_raw}"
        ) from exc
    return EntityMetadata(
        stable_id=require_text(row, "id", raw_reference),
        display_name=str(row.get("display_name") or row.get("name") or row["id"]),
        vendor=str(row.get("vendor", "")),
        model=str(row.get("model", "")),
        address=str(row.get("address", "")),
        protocol=str(row.get("protocol", default_protocol)),
        availability=availability,
        last_seen=str(row.get("last_seen", "")),
        source=str(row.get("source", "")),
        confidence=float(confidence),
        capabilities=_string_tuple(
            row.get("capabilities", []), f"{raw_reference}.capabilities"
        ),
        raw_reference=raw_reference,
    )


def _string_tuple(value: Any, source: str) -> tuple[str, ...]:
    if not isinstance(value, list) or any(not isinstance(item, str) for item in value):
        raise InspectorInputError(f"{source}:string_list_required")
    return tuple(str(item) for item in value)


def _optional_int(value: Any) -> Optional[int]:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, int):
        raise InspectorInputError("integer_or_null_required")
    return value

