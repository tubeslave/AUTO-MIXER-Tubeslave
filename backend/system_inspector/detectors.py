"""Evidence-backed Phase 0 Inspector detectors."""

from __future__ import annotations

from typing import Iterable, Mapping, Optional

from .io import SnapshotBundle
from .model import (
    CheckStatus,
    Evidence,
    Finding,
    RouteComparison,
    Severity,
    SystemGraph,
)
from .profile import ShowProfile


def run_detectors(
    profile: ShowProfile,
    bundle: SnapshotBundle,
    graph: SystemGraph,
) -> tuple[Finding, ...]:
    findings: list[Finding] = []
    findings.extend(_required_device_findings(profile, graph))
    findings.extend(_network_interface_findings(profile, graph))
    findings.extend(_sample_rate_findings(profile, bundle))
    findings.extend(_route_findings(graph))
    findings.extend(_playback_duplicate_findings(graph))
    findings.extend(_forbidden_main_findings(graph))
    return tuple(sorted(findings, key=lambda item: (item.id, item.severity.value)))


def _required_device_findings(
    profile: ShowProfile, graph: SystemGraph
) -> Iterable[Finding]:
    observed = {device.metadata.stable_id: device for device in graph.devices}
    for required in profile.required_devices:
        device = observed.get(required.stable_id)
        if device is None:
            yield Finding(
                id=f"discovery.required-device-missing.{required.stable_id}",
                severity=Severity.CRITICAL,
                category="discovery",
                title="Обязательное устройство отсутствует",
                summary=f"Нет evidence для устройства {required.name}.",
                evidence=(),
                confidence=1.0,
                impact="Критический участок системы не может быть подтверждён.",
                recommended_action="Проверить питание, сеть и импортировать свежий snapshot.",
                status=CheckStatus.UNKNOWN,
                missing_evidence=(f"devices.{required.stable_id}",),
            )
        elif device.metadata.availability.value != "ONLINE":
            yield Finding(
                id=f"connectivity.required-device-offline.{required.stable_id}",
                severity=Severity.CRITICAL,
                category="connectivity",
                title="Обязательное устройство не online",
                summary=(
                    f"{required.name}: availability="
                    f"{device.metadata.availability.value}."
                ),
                evidence=(
                    Evidence(
                        source=device.metadata.source or "snapshot",
                        path=f"{device.metadata.raw_reference}.availability",
                        value=device.metadata.availability.value,
                    ),
                ),
                confidence=device.metadata.confidence,
                impact="Ожидаемый end-to-end путь недоступен или не подтверждён.",
                recommended_action="Проверить устройство вручную и обновить snapshot.",
            )


def _network_interface_findings(
    profile: ShowProfile, graph: SystemGraph
) -> Iterable[Finding]:
    by_bsd = {item.bsd_name: item for item in graph.network_interfaces}
    expected = (
        ("dante", profile.expected_dante_interface),
        ("control", profile.expected_control_interface),
    )
    for role, bsd_name in expected:
        interface = by_bsd.get(bsd_name)
        if interface is None:
            yield Finding(
                id=f"network.interface-changed.{role}",
                severity=Severity.HIGH,
                category="network",
                title="Ожидаемый сетевой интерфейс отсутствует",
                summary=f"Для роли {role} ожидался {bsd_name}.",
                evidence=(
                    Evidence(
                        source="show_profile",
                        path=f"network.{role}.expected_interface",
                        value=bsd_name,
                    ),
                ),
                confidence=1.0,
                impact="Маршрут управления или аудиосети мог смениться.",
                recommended_action="Проверить подключение адаптера и service order.",
                status=CheckStatus.UNKNOWN,
                missing_evidence=(f"macos.network_interfaces.{bsd_name}",),
            )
            continue
        if role not in interface.roles:
            yield Finding(
                id=f"network.interface-role-drift.{role}",
                severity=Severity.HIGH,
                category="network",
                title="Сетевой интерфейс используется не для ожидаемой роли",
                summary=f"{bsd_name} не помечен ролью {role} в snapshot.",
                evidence=(
                    Evidence(
                        source="macos_snapshot",
                        path=f"{interface.metadata.raw_reference}.roles",
                        value=list(interface.roles),
                    ),
                ),
                confidence=interface.metadata.confidence,
                impact="Критический трафик может уйти через другой интерфейс.",
                recommended_action="Проверить route/service order вручную.",
            )
        if (
            role == "dante"
            and profile.require_wired_dante
            and interface.metadata.protocol.lower() == "wifi"
        ):
            yield Finding(
                id="network.wifi-used-for-dante",
                severity=Severity.CRITICAL,
                category="network",
                title="Dante назначен на Wi-Fi",
                summary=f"Критическая Dante-роль назначена на {bsd_name} (Wi-Fi).",
                evidence=(
                    Evidence(
                        source="macos_snapshot",
                        path=f"{interface.metadata.raw_reference}.protocol",
                        value=interface.metadata.protocol,
                    ),
                ),
                confidence=interface.metadata.confidence,
                impact="Беспроводная сеть не соответствует show profile.",
                recommended_action="Выбрать проводной адаптер вручную.",
            )


def _sample_rate_findings(
    profile: ShowProfile, bundle: SnapshotBundle
) -> Iterable[Finding]:
    observed: list[tuple[str, str, Optional[int], float]] = []
    wing_device = bundle.wing.get("device", {})
    if isinstance(wing_device, dict):
        observed.append(
            (
                "wing_snapshot",
                "device.sample_rate",
                _int_or_none(wing_device.get("sample_rate")),
                _confidence(wing_device),
            )
        )
    for index, row in enumerate(bundle.macos.get("coreaudio_devices", [])):
        if isinstance(row, dict) and row.get("selected_for_playback") is True:
            observed.append(
                (
                    "macos_snapshot",
                    f"coreaudio_devices[{index}].sample_rate",
                    _int_or_none(row.get("sample_rate")),
                    _confidence(row),
                )
            )
    clock = bundle.dante.get("clock", {})
    if isinstance(clock, dict):
        observed.append(
            (
                "dante_snapshot",
                "clock.sample_rate",
                _int_or_none(clock.get("sample_rate")),
                _confidence(clock),
            )
        )
    missing = [path for _, path, value, _ in observed if value is None]
    if not observed or missing:
        yield Finding(
            id="clocking.sample-rate-unknown",
            severity=Severity.UNKNOWN,
            category="clocking",
            title="Sample rate подтверждён не полностью",
            summary="Не все требуемые источники содержат sample rate.",
            evidence=tuple(
                Evidence(source=source, path=path, value=value)
                for source, path, value, _ in observed
            ),
            confidence=0.0,
            impact="Clock compatibility нельзя считать подтверждённой.",
            recommended_action="Получить свежие read-only/import snapshots.",
            status=CheckStatus.UNKNOWN,
            missing_evidence=tuple(missing or ("sample_rate_sources",)),
        )
        return
    mismatches = [
        item for item in observed if item[2] != profile.sample_rate
    ]
    if mismatches:
        yield Finding(
            id="clocking.sample-rate-mismatch",
            severity=Severity.CRITICAL,
            category="clocking",
            title="Sample rate не совпадает с show profile",
            summary=(
                f"Ожидается {profile.sample_rate} Hz; обнаружено несовпадение."
            ),
            evidence=tuple(
                Evidence(source=source, path=path, value=value)
                for source, path, value, _ in mismatches
            ),
            confidence=min(item[3] for item in mismatches),
            impact="Возможны отсутствие звука, артефакты или невозможность subscription.",
            recommended_action="Согласовать sample rate вручную; Inspector ничего не меняет.",
        )


def _route_findings(graph: SystemGraph) -> Iterable[Finding]:
    for comparison in graph.route_comparisons:
        if comparison.status is CheckStatus.OK:
            continue
        if comparison.status is CheckStatus.UNKNOWN:
            yield _missing_route_finding(comparison)
            continue
        expected = comparison.expected
        observed = comparison.observed
        if expected is None or observed is None:
            continue
        yield Finding(
            id=f"routing.drift.{comparison.role}",
            severity=Severity.CRITICAL
            if comparison.role.startswith("playback")
            else Severity.HIGH,
            category="routing",
            title="Маршрут отличается от baseline",
            summary=(
                f"{comparison.role}: expected "
                f"{expected.destination_endpoint_id} → {expected.console_channel}; "
                f"observed {observed.destination_endpoint_id} → "
                f"{observed.console_channel or 'UNKNOWN'}."
            ),
            evidence=(
                Evidence(
                    source="baseline",
                    path=f"expected_routes.{comparison.role}",
                    value={
                        "destination_endpoint_id": expected.destination_endpoint_id,
                        "console_channel": expected.console_channel,
                    },
                ),
                Evidence(
                    source="dante_snapshot",
                    path=f"subscriptions.{comparison.role}.destination_endpoint_id",
                    value=observed.destination_endpoint_id,
                    limitation=(
                        "Imported subscription evidence; it does not prove audio signal."
                    ),
                ),
                Evidence(
                    source="wing_snapshot",
                    path=f"console_inputs.{observed.destination_endpoint_id}.console_channel",
                    value=observed.console_channel or None,
                ),
            ),
            confidence=observed.confidence,
            impact="End-to-end signal path does not match the known-good baseline.",
            recommended_action="Review Dante subscription and WING patch manually.",
        )


def _missing_route_finding(comparison: RouteComparison) -> Finding:
    return Finding(
        id=f"routing.required-route-missing.{comparison.role}",
        severity=Severity.CRITICAL,
        category="routing",
        title="Обязательный playback route не подтверждён",
        summary=f"Нет достаточного evidence для роли {comparison.role}.",
        evidence=(),
        confidence=1.0,
        impact="Соответствующий playback channel может отсутствовать на концерте.",
        recommended_action="Проверить экспорт Dante и WING patch вручную.",
        status=CheckStatus.UNKNOWN,
        missing_evidence=comparison.missing_evidence,
    )


def _playback_duplicate_findings(graph: SystemGraph) -> Iterable[Finding]:
    routes = {route.role: route for route in graph.observed_routes}
    left = routes.get("playback_l")
    right = routes.get("playback_r")
    if left is None or right is None:
        return
    same_endpoint = (
        left.destination_endpoint_id
        and left.destination_endpoint_id == right.destination_endpoint_id
    )
    same_channel = (
        left.console_channel
        and left.console_channel == right.console_channel
    )
    if not (same_endpoint or same_channel):
        return
    yield Finding(
        id="routing.playback-right-duplicate",
        severity=Severity.CRITICAL,
        category="routing",
        title="Playback L и Playback R направлены в один вход",
        summary=(
            "Оба канала стереопары приходят на "
            f"{left.destination_endpoint_id}"
            + (f" / {left.console_channel}." if left.console_channel else ".")
        ),
        evidence=(
            Evidence(
                source="dante_snapshot",
                path="subscriptions.playback_l.destination_endpoint_id",
                value=left.destination_endpoint_id,
                limitation="Imported topology does not prove audio signal.",
            ),
            Evidence(
                source="dante_snapshot",
                path="subscriptions.playback_r.destination_endpoint_id",
                value=right.destination_endpoint_id,
                limitation="Imported topology does not prove audio signal.",
            ),
        ),
        confidence=min(left.confidence, right.confidence),
        impact="Потеря стереоканала или непреднамеренное суммирование каналов.",
        recommended_action="Назначить Playback R на отдельный вход WING вручную.",
    )


def _forbidden_main_findings(graph: SystemGraph) -> Iterable[Finding]:
    comparisons = {item.role: item for item in graph.route_comparisons}
    for role in ("click", "cues"):
        comparison = comparisons.get(role)
        if (
            comparison is None
            or comparison.expected is None
            or comparison.observed is None
        ):
            continue
        forbidden = set(comparison.expected.forbidden_destinations)
        present = sorted(forbidden.intersection(comparison.observed.main_assignments))
        if not present:
            continue
        yield Finding(
            id=f"playback.{role}-routed-to-main",
            severity=Severity.CRITICAL,
            category="playback_safety",
            title=f"{role.capitalize()} направлен в Main",
            summary=f"Запрещённые назначения: {', '.join(present)}.",
            evidence=(
                Evidence(
                    source="wing_snapshot",
                    path=f"console_inputs.{role}.main_assignments",
                    value=list(comparison.observed.main_assignments),
                ),
            ),
            confidence=comparison.observed.confidence,
            impact="Служебный сигнал может попасть в PA.",
            recommended_action="Удалить Main assignment вручную после проверки.",
        )


def _int_or_none(value: object) -> Optional[int]:
    if isinstance(value, bool) or not isinstance(value, int):
        return None
    return value


def _confidence(row: Mapping[str, object]) -> float:
    value = row.get("confidence", 0.0)
    return float(value) if isinstance(value, (int, float)) and not isinstance(value, bool) else 0.0

