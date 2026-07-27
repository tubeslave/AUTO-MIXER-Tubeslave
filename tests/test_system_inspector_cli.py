from __future__ import annotations

import json
from pathlib import Path

from backend.system_inspector.cli import main


FIXTURES = Path(__file__).parent / "fixtures" / "system_inspector"
HEALTHY = FIXTURES / "healthy_system"
DUPLICATE = FIXTURES / "playback_right_duplicate"
PROFILE = HEALTHY / "profile.yaml"


def test_cli_prints_json_for_healthy_replay(capsys):
    result = main(
        [
            "inspector",
            "scan",
            "--profile",
            str(PROFILE),
            "--snapshot",
            str(HEALTHY),
            "--format",
            "json",
        ]
    )

    assert result == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["schema_version"] == "automixer.inspector.report.v1"
    assert payload["read_only"] is True
    assert payload["findings"] == []


def test_cli_returns_nonzero_and_writes_fault_report(tmp_path: Path, capsys):
    output = tmp_path / "report.json"
    result = main(
        [
            "inspector",
            "replay-snapshot",
            "--profile",
            str(PROFILE),
            "--snapshot",
            str(DUPLICATE),
            "--format",
            "json",
            "--output",
            str(output),
        ]
    )

    assert result == 2
    stdout_payload = json.loads(capsys.readouterr().out)
    file_payload = json.loads(output.read_text(encoding="utf-8"))
    assert stdout_payload == file_payload
    assert "routing.playback-right-duplicate" in {
        finding["id"] for finding in file_payload["findings"]
    }

