import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "tools"))

import live_wing_correction_test as live_test
from backend.write_surface_inventory import get_write_surface


def test_live_wing_correction_test_requires_explicit_ack_before_config(monkeypatch, capsys) -> None:
    def fail_if_called() -> dict:
        raise AssertionError("load_mixer_config must not run without explicit live-write ack")

    monkeypatch.setattr(live_test, "load_mixer_config", fail_if_called)
    monkeypatch.setattr(sys, "argv", ["live_wing_correction_test.py"])

    assert live_test.main() == 2

    captured = capsys.readouterr()
    assert "writes to live WING" in captured.err or "живой WING" in captured.err
    assert live_test.LIVE_WING_CONFIRMATION in captured.err


def test_live_wing_correction_test_rejects_wrong_ack(monkeypatch) -> None:
    def fail_if_called() -> dict:
        raise AssertionError("load_mixer_config must not run with wrong live-write ack")

    monkeypatch.setattr(live_test, "load_mixer_config", fail_if_called)
    monkeypatch.setattr(
        sys,
        "argv",
        ["live_wing_correction_test.py", "--confirm-live-wing-write", "yes"],
    )

    assert live_test.main() == 2


def test_live_wing_correction_test_is_blocked_in_write_surface_inventory() -> None:
    surface = get_write_surface("broad_live_wing_correction_test_tool")

    assert surface is not None
    assert surface.status == "blocked"
    assert surface.allowed_live is False
    assert "tools/live_wing_correction_test.py" in surface.paths
    assert "not_mvp_release_authority" in surface.notes
