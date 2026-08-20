import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "tools"))

import live_wing_soundcheck_v2 as soundcheck_v2
from backend.write_surface_inventory import get_write_surface


def test_live_wing_soundcheck_v2_is_quarantined_before_config(monkeypatch) -> None:
    def fail_if_called(*_args, **_kwargs):
        raise AssertionError("config must not be read without explicit live-write ack")

    monkeypatch.setattr(soundcheck_v2.yaml, "safe_load", fail_if_called)
    monkeypatch.setattr(sys, "argv", ["live_wing_soundcheck_v2.py"])

    with pytest.raises(RuntimeError, match="legacy_live_mutation_surface_quarantined"):
        soundcheck_v2.main()


def test_live_wing_soundcheck_v2_rejects_wrong_ack_before_config(monkeypatch) -> None:
    def fail_if_called(*_args, **_kwargs):
        raise AssertionError("config must not be read with wrong live-write ack")

    monkeypatch.setattr(soundcheck_v2.yaml, "safe_load", fail_if_called)
    monkeypatch.setattr(
        sys,
        "argv",
        ["live_wing_soundcheck_v2.py", "--confirm-live-wing-write", "yes"],
    )

    with pytest.raises(RuntimeError, match="legacy_live_mutation_surface_quarantined"):
        soundcheck_v2.main()


def test_live_wing_soundcheck_v2_is_blocked_in_write_surface_inventory() -> None:
    surface = get_write_surface("broad_live_wing_soundcheck_v2_tool")

    assert surface is not None
    assert surface.status == "blocked"
    assert surface.allowed_live is False
    assert "tools/live_wing_soundcheck_v2.py" in surface.paths
    assert "eq" in surface.write_kinds
    assert "gate" in surface.write_kinds
    assert "not_mvp_release_authority" in surface.notes
