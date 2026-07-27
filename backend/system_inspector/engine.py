"""System Inspector orchestration for offline snapshot replay."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from .detectors import run_detectors
from .io import SnapshotBundle, load_snapshot_bundle, require_text
from .model import InspectorReport
from .normalize import build_system_graph
from .profile import ShowProfile, load_show_profile
from .readiness import build_readiness
from .safety import INSPECTOR_READ_ONLY, assert_read_only_mode


REPORT_SCHEMA_VERSION = "automixer.inspector.report.v1"


@dataclass(frozen=True)
class InspectorEngine:
    profile: ShowProfile
    bundle: SnapshotBundle

    @classmethod
    def from_paths(cls, profile_path: Path, snapshot_root: Path) -> "InspectorEngine":
        assert_read_only_mode()
        return cls(
            profile=load_show_profile(profile_path),
            bundle=load_snapshot_bundle(snapshot_root),
        )

    def inspect(self) -> InspectorReport:
        assert_read_only_mode()
        graph = build_system_graph(self.profile, self.bundle)
        findings = run_detectors(self.profile, self.bundle, graph)
        readiness = build_readiness(graph, findings, self.bundle)
        return InspectorReport(
            schema_version=REPORT_SCHEMA_VERSION,
            profile_name=self.profile.name,
            baseline_id=require_text(
                self.bundle.baseline, "baseline_id", "baseline"
            ),
            read_only=INSPECTOR_READ_ONLY,
            graph=graph,
            findings=findings,
            readiness=readiness,
        )


def inspect_snapshot_bundle(
    profile_path: Path,
    snapshot_root: Path,
) -> InspectorReport:
    return InspectorEngine.from_paths(profile_path, snapshot_root).inspect()

