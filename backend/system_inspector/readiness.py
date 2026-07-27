"""Transparent readiness scoring with explicit coverage and unknowns."""

from __future__ import annotations

from typing import Iterable

from .io import SnapshotBundle
from .model import (
    CategoryAssessment,
    CheckStatus,
    Finding,
    ReadinessAssessment,
    Severity,
    SystemGraph,
)


CATEGORY_NAMES = (
    "Connectivity",
    "Device Discovery",
    "Dante Health",
    "Clocking",
    "Routing",
    "WING Patch",
    "Signal Presence",
    "Gain and Headroom",
    "Stereo Integrity",
    "Playback Safety",
    "Network Stability",
    "Redundancy",
    "Configuration Drift",
    "Overall Show Risk",
)

MINIMUM_SCORE_COVERAGE = 0.50
PENALTIES = {
    Severity.CRITICAL: 35,
    Severity.HIGH: 20,
    Severity.MEDIUM: 10,
    Severity.LOW: 5,
    Severity.INFO: 0,
    Severity.UNKNOWN: 0,
}


def build_readiness(
    graph: SystemGraph,
    findings: tuple[Finding, ...],
    bundle: SnapshotBundle,
) -> ReadinessAssessment:
    del graph
    category_findings = _group_findings(findings)
    known = {
        "Connectivity": True,
        "Device Discovery": True,
        "Dante Health": False,
        "Clocking": not _has_unknown(findings, "clocking"),
        "Routing": True,
        "WING Patch": True,
        "Signal Presence": False,
        "Gain and Headroom": False,
        "Stereo Integrity": True,
        "Playback Safety": True,
        "Network Stability": _network_health_known(bundle),
        "Redundancy": False,
        "Configuration Drift": True,
        "Overall Show Risk": True,
    }
    category_map = {
        "Connectivity": "connectivity",
        "Device Discovery": "discovery",
        "Clocking": "clocking",
        "Routing": "routing",
        "WING Patch": "routing",
        "Stereo Integrity": "routing",
        "Playback Safety": "playback_safety",
        "Network Stability": "network",
        "Configuration Drift": "routing",
        "Overall Show Risk": "all",
    }
    categories: list[CategoryAssessment] = []
    all_confidences = [
        finding.confidence
        for finding in findings
        if finding.status is not CheckStatus.UNKNOWN
    ]
    for name in CATEGORY_NAMES:
        if not known[name]:
            categories.append(
                CategoryAssessment(
                    name=name,
                    score=None,
                    status=CheckStatus.UNKNOWN,
                    confidence=0.0,
                    evidence_count=0,
                    unknown_checks=(f"{name}:not_observed",),
                )
            )
            continue
        key = category_map.get(name, "")
        relevant = findings if key == "all" else category_findings.get(key, ())
        observed_findings = tuple(
            item for item in relevant if item.status is not CheckStatus.UNKNOWN
        )
        unknown_findings = tuple(
            item for item in relevant if item.status is CheckStatus.UNKNOWN
        )
        score = _score(observed_findings)
        status = (
            CheckStatus.MISMATCH
            if any(item.severity in {Severity.CRITICAL, Severity.HIGH, Severity.MEDIUM, Severity.LOW} for item in observed_findings)
            else CheckStatus.UNKNOWN
            if unknown_findings
            else CheckStatus.OK
        )
        confidence = (
            min(item.confidence for item in observed_findings)
            if observed_findings
            else min(all_confidences, default=1.0)
        )
        categories.append(
            CategoryAssessment(
                name=name,
                score=score,
                status=status,
                confidence=confidence,
                evidence_count=sum(len(item.evidence) for item in relevant),
                unknown_checks=tuple(item.id for item in unknown_findings),
            )
        )

    known_categories = tuple(item for item in categories if item.score is not None)
    coverage = len(known_categories) / len(categories)
    known_score = (
        sum(item.score or 0 for item in known_categories) / len(known_categories)
        if known_categories
        else 0.0
    )
    overall_score = (
        round(known_score * coverage)
        if coverage >= MINIMUM_SCORE_COVERAGE
        else None
    )
    unknown_checks = tuple(
        unknown
        for category in categories
        for unknown in category.unknown_checks
    )
    status = (
        CheckStatus.MISMATCH
        if any(
            finding.severity in {Severity.CRITICAL, Severity.HIGH}
            and finding.status is not CheckStatus.UNKNOWN
            for finding in findings
        )
        else CheckStatus.UNKNOWN
        if overall_score is None or unknown_checks
        else CheckStatus.OK
    )
    confidence = (
        min(
            (item.confidence for item in known_categories),
            default=0.0,
        )
        * coverage
    )
    drivers = tuple(
        finding.title
        for finding in findings
        if finding.severity in {Severity.CRITICAL, Severity.HIGH}
    )[:8]
    timestamp = max(
        (value for value in bundle.captured_at.values() if value),
        default="",
    )
    return ReadinessAssessment(
        score=overall_score,
        status=status,
        confidence=round(confidence, 3),
        observed_coverage=round(coverage, 3),
        categories=tuple(categories),
        unknown_checks=unknown_checks,
        drivers=drivers,
        timestamp=timestamp,
        data_freshness=bundle.captured_at,
    )


def _group_findings(
    findings: Iterable[Finding],
) -> dict[str, tuple[Finding, ...]]:
    grouped: dict[str, list[Finding]] = {}
    for finding in findings:
        grouped.setdefault(finding.category, []).append(finding)
    return {key: tuple(value) for key, value in grouped.items()}


def _has_unknown(findings: Iterable[Finding], category: str) -> bool:
    return any(
        item.category == category and item.status is CheckStatus.UNKNOWN
        for item in findings
    )


def _score(findings: Iterable[Finding]) -> int:
    return max(0, 100 - sum(PENALTIES[item.severity] for item in findings))


def _network_health_known(bundle: SnapshotBundle) -> bool:
    value = bundle.macos.get("network_health")
    return isinstance(value, dict) and value.get("status") in {"STABLE", "DEGRADED"}
