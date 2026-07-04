import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "tools"))

import live_wing_soundcheck_v2_stage2 as stage2
from backend.write_surface_inventory import get_write_surface


def test_live_wing_soundcheck_v2_stage2_requires_ack_before_config(monkeypatch, capsys) -> None:
    def fail_if_called(*_args, **_kwargs):
        raise AssertionError("config must not be read without explicit live-write ack")

    monkeypatch.setattr(stage2.yaml, "safe_load", fail_if_called)
    monkeypatch.setattr(sys, "argv", ["live_wing_soundcheck_v2_stage2.py"])

    assert stage2.main() == 2

    captured = capsys.readouterr()
    assert "--confirm-live-wing-write" in captured.err
    assert stage2.LIVE_WING_CONFIRMATION in captured.err


def test_live_wing_soundcheck_v2_stage2_rejects_wrong_ack_before_config(monkeypatch) -> None:
    def fail_if_called(*_args, **_kwargs):
        raise AssertionError("config must not be read with wrong live-write ack")

    monkeypatch.setattr(stage2.yaml, "safe_load", fail_if_called)
    monkeypatch.setattr(
        sys,
        "argv",
        ["live_wing_soundcheck_v2_stage2.py", "--confirm-live-wing-write", "yes"],
    )

    assert stage2.main() == 2


def test_live_wing_soundcheck_v2_stage2_is_blocked_in_write_surface_inventory() -> None:
    surface = get_write_surface("broad_live_wing_soundcheck_v2_stage2_tool")

    assert surface is not None
    assert surface.status == "blocked"
    assert surface.allowed_live is False
    assert "tools/live_wing_soundcheck_v2_stage2.py" in surface.paths
    assert "phase" in surface.write_kinds
    assert "delay" in surface.write_kinds
    assert "not_mvp_release_authority" in surface.notes
