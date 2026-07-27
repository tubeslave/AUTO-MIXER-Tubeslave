from __future__ import annotations

import ast
import json
from datetime import datetime, timezone
from pathlib import Path

import pytest

from backend.system_inspector.io import InspectorInputError, load_snapshot_bundle
from backend.system_inspector.profile import load_show_profile
from backend.system_inspector.safety import (
    InspectorReadOnlyConfigurationError,
    InspectorWriteProhibited,
    assert_read_only_mode,
    prohibit_write,
)


REPO_ROOT = Path(__file__).resolve().parents[1]
PACKAGE_ROOT = REPO_ROOT / "backend" / "system_inspector"
FIXTURES = Path(__file__).parent / "fixtures" / "system_inspector"


def test_inspector_v1_cannot_disable_read_only():
    with pytest.raises(
        InspectorReadOnlyConfigurationError,
        match="inspector_v1_read_only_cannot_be_disabled",
    ):
        assert_read_only_mode({"INSPECTOR_READ_ONLY": "false"})

    assert_read_only_mode({})
    assert_read_only_mode({"INSPECTOR_READ_ONLY": "true"})


def test_write_attempt_is_blocked_and_redacted_in_security_log(tmp_path: Path):
    audit_path = tmp_path / "security" / "inspector.jsonl"

    with pytest.raises(
        InspectorWriteProhibited,
        match="inspector_write_prohibited:set_channel_gain_secret_value",
    ):
        prohibit_write(
            "set channel/gain secret=value",
            audit_path=audit_path,
            now=lambda: datetime(2026, 7, 27, 8, 0, tzinfo=timezone.utc),
        )

    event = json.loads(audit_path.read_text(encoding="utf-8"))
    assert event == {
        "action": "set_channel_gain_secret_value",
        "blocked": True,
        "detail": "No target, value, address, credential, or payload was logged.",
        "event": "inspector_write_blocked",
        "timestamp": "2026-07-27T08:00:00Z",
    }
    assert "payload" not in event
    assert "target" not in event


def test_inspector_package_has_no_live_or_write_dependency():
    forbidden_import_roots = {
        "audio_capture",
        "coreaudio",
        "osc",
        "pythonosc",
        "socket",
        "sounddevice",
        "subprocess",
        "wing_client",
    }
    violations: list[str] = []
    for path in sorted(PACKAGE_ROOT.glob("*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    if alias.name.split(".")[0] in forbidden_import_roots:
                        violations.append(f"{path.name}:import:{alias.name}")
            elif isinstance(node, ast.ImportFrom) and node.module:
                if node.module.split(".")[0] in forbidden_import_roots:
                    violations.append(f"{path.name}:from:{node.module}")
            elif isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
                if node.func.attr == "send" or node.func.attr.startswith("set_"):
                    violations.append(f"{path.name}:call:{node.func.attr}")
        source = path.read_text(encoding="utf-8")
        assert "WingClient" not in source
        assert "EnhancedOSCClient" not in source

    assert violations == []


def test_unsafe_yaml_constructor_fails_closed(tmp_path: Path):
    profile = tmp_path / "unsafe.yaml"
    profile.write_text(
        "!!python/object/apply:os.system ['echo must-not-run']\n",
        encoding="utf-8",
    )

    with pytest.raises(InspectorInputError, match="show_profile_invalid_yaml"):
        load_show_profile(profile)


def test_duplicate_snapshot_ids_fail_closed(tmp_path: Path):
    replay = tmp_path / "duplicate-id"
    source = FIXTURES / "healthy_system"
    import shutil

    shutil.copytree(source, replay)
    dante_path = replay / "dante.json"
    dante = json.loads(dante_path.read_text(encoding="utf-8"))
    dante["devices"].append(dict(dante["devices"][0]))
    dante_path.write_text(json.dumps(dante), encoding="utf-8")

    bundle = load_snapshot_bundle(replay)
    from backend.system_inspector.normalize import build_system_graph

    with pytest.raises(InspectorInputError, match="dante.devices:duplicate_id"):
        build_system_graph(load_show_profile(replay / "profile.yaml"), bundle)

