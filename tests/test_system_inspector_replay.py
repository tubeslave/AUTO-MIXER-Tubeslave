from __future__ import annotations

import hashlib
import json
import shutil
import asyncio
from pathlib import Path

from backend.system_inspector.engine import inspect_snapshot_bundle
from backend.system_inspector.model import CheckStatus
from backend.system_inspector.reports import canonical_json
from backend.system_inspector.providers import DanteSnapshotFileProvider


FIXTURES = Path(__file__).parent / "fixtures" / "system_inspector"
HEALTHY = FIXTURES / "healthy_system"
DUPLICATE = FIXTURES / "playback_right_duplicate"
PROFILE = HEALTHY / "profile.yaml"


def test_healthy_snapshot_replay_is_deterministic_and_has_no_findings():
    first = inspect_snapshot_bundle(PROFILE, HEALTHY)
    second = inspect_snapshot_bundle(PROFILE, HEALTHY)

    first_bytes = canonical_json(first).encode("utf-8")
    second_bytes = canonical_json(second).encode("utf-8")

    assert first_bytes == second_bytes
    assert hashlib.sha256(first_bytes).hexdigest() == hashlib.sha256(
        second_bytes
    ).hexdigest()
    assert first.findings == ()
    assert first.readiness.score == 71
    assert first.readiness.status is CheckStatus.UNKNOWN
    assert first.readiness.observed_coverage == 0.714
    assert first.readiness.unknown_checks
    assert first.read_only is True


def test_duplicate_playback_right_emits_evidence_backed_finding():
    report = inspect_snapshot_bundle(PROFILE, DUPLICATE)
    findings = {finding.id: finding for finding in report.findings}

    finding = findings["routing.playback-right-duplicate"]
    assert finding.severity.value == "CRITICAL"
    assert finding.confidence == 0.9
    assert [evidence.path for evidence in finding.evidence] == [
        "subscriptions.playback_l.destination_endpoint_id",
        "subscriptions.playback_r.destination_endpoint_id",
    ]
    assert [evidence.value for evidence in finding.evidence] == [
        "wing-rx-17",
        "wing-rx-17",
    ]
    assert finding.automatic_fix_available is False
    assert finding.requires_manual_confirmation is True
    assert report.readiness.score is not None
    assert report.readiness.score < 100


def test_missing_subscription_is_unknown_not_ok(tmp_path: Path):
    replay = tmp_path / "partial"
    shutil.copytree(HEALTHY, replay)
    dante_path = replay / "dante.json"
    dante = json.loads(dante_path.read_text(encoding="utf-8"))
    dante["subscriptions"] = [
        row for row in dante["subscriptions"] if row["role"] != "playback_r"
    ]
    dante_path.write_text(json.dumps(dante), encoding="utf-8")

    report = inspect_snapshot_bundle(replay / "profile.yaml", replay)
    findings = {finding.id: finding for finding in report.findings}
    missing = findings["routing.required-route-missing.playback_r"]

    assert missing.status is CheckStatus.UNKNOWN
    assert missing.missing_evidence == (
        "dante_snapshot.subscriptions.playback_r",
    )
    comparison = {
        item.role: item for item in report.graph.route_comparisons
    }["playback_r"]
    assert comparison.status is CheckStatus.UNKNOWN
    assert comparison.observed is None


def test_replay_does_not_modify_profile_or_snapshot_bytes():
    paths = [PROFILE, *(HEALTHY / name for name in (
        "macos.json",
        "wing.json",
        "dante.json",
        "baseline.json",
    ))]
    before = {path: path.read_bytes() for path in paths}

    inspect_snapshot_bundle(PROFILE, HEALTHY)

    assert {path: path.read_bytes() for path in paths} == before


def test_snapshot_dante_provider_preserves_import_limitations():
    report = inspect_snapshot_bundle(PROFILE, HEALTHY)
    raw = json.loads((HEALTHY / "dante.json").read_text(encoding="utf-8"))
    provider = DanteSnapshotFileProvider(report.graph, raw)

    devices = asyncio.run(provider.get_devices())
    subscriptions = asyncio.run(provider.get_subscriptions())
    clock = asyncio.run(provider.get_clock_status())
    health = asyncio.run(provider.get_health())

    assert {device.metadata.stable_id for device in devices} == {
        "dvs-playback",
        "wing-dante-card",
    }
    assert len(subscriptions) == 5
    assert clock.status == "IMPORTED"
    assert "not a live" in clock.limitation
    assert health.status == "UNKNOWN"
    assert "No live Dante" in health.limitation
