import json
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
TOOL = ROOT / "tools" / "dlive_no_transport_preflight.py"


def test_preflight_selects_mixrack_plain_port_without_authority():
    result = subprocess.run(
        [
            sys.executable,
            str(TOOL),
            "--host",
            "192.168.3.70",
            "--endpoint",
            "mixrack",
        ],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
    )
    assert result.returncode == 0
    report = json.loads(result.stdout)
    assert report["port"] == 51325
    assert report["console_contacted"] is False
    assert report["midi_sent"] is False
    assert report["live_authority_granted"] is False


def test_preflight_rejects_surface_port_for_mixrack():
    result = subprocess.run(
        [
            sys.executable,
            str(TOOL),
            "--host",
            "192.168.3.70",
            "--endpoint",
            "mixrack",
            "--port",
            "51328",
        ],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
    )
    assert result.returncode == 2
    assert json.loads(result.stdout)["reason"].endswith("expected_51325")


def test_preflight_tool_has_no_transport_calls():
    source = TOOL.read_text(encoding="utf-8")
    assert "socket." not in source
    assert ".connect(" not in source
    assert "_send_raw" not in source
