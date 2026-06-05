from __future__ import annotations

import os
import re
from pathlib import Path

import pytest

from manual_recovery_guard import (
    ACK_ENV_VAR,
    ACK_VALUE,
    has_manual_recovery_ack,
    require_manual_recovery_ack,
)
from backend.write_surface_inventory import iter_write_surfaces


REPO_ROOT = Path(__file__).resolve().parents[1]
SCAN_ROOTS = ("backend", "frontend/src", "tools")
SCAN_SUFFIXES = {".py", ".js", ".ts", ".tsx", ".sh"}

EXCLUDED_PREFIXES = (
    "backend/lab_only/legacy_backups/",
    "backend/ai/knowledge/",
    "frontend/src/services/__tests__/",
    "tools/codex_autoloop/",
    "tools/chatgpt_pulse_review/",
)

PRUNED_DIRS = {
    ".git",
    ".pytest_cache",
    "__pycache__",
    "venv",
    "node_modules",
}

EXCLUDED_FILES = {
    "backend/write_surface_inventory.py",
}

MANUAL_RECOVERY_SCRIPTS = (
    "backend/reset_all_channels.py",
    "backend/reset_modules_trim_faders.py",
    "backend/restore_channels.py",
    "backend/disable_modules_set_faders.py",
)

RISKY_WRITE_TOKEN_RE = re.compile(
    r"\b(?:"
    r"set_fader|set_main_fader|set_bus_fader|set_dca_fader|set_dca_level|"
    r"set_group_fader|set_matrix_fader|set_gain|set_trim|set_input_trim|"
    r"set_channel_trim|set_preamp_trim|set_preamp_gain|set_eq|set_eq_band|"
    r"set_bus_eq_on|set_bus_eq_band|set_main_eq_on|set_main_eq_band|"
    r"set_matrix_eq_on|set_matrix_eq_band|set_compressor|set_compressor_gain|"
    r"set_bus_compressor|set_bus_compressor_gain|"
    r"set_mute|set_channel_mute|set_bus_mute|set_dca_mute|set_group_mute|"
    r"set_main_mute|set_matrix_mute|"
    r"set_gate\w*|set_hpf|set_send_level|set_pan|set_channel_pan|set_delay|set_polarity|"
    r"set_bus_pan|set_main_pan|set_fx_model|set_fx_on|set_fx_mix|"
    r"set_fx_parameter|set_fx_return_level|set_insert|set_channel_main_send|"
    r"set_reverb_send|"
    r"route_output|load_snap|save_snap|reset_channel_(?:eq|hpf|dynamics|processing)"
    r")\s*\("
    r"|['\"](?:"
    r"set_fader|set_main_fader|set_bus_fader|set_dca_fader|set_dca_level|"
    r"set_group_fader|set_matrix_fader|set_gain|set_trim|set_input_trim|"
    r"set_channel_trim|set_preamp_trim|set_preamp_gain|set_eq|set_eq_band|"
    r"set_bus_eq_on|set_bus_eq_band|set_main_eq_on|set_main_eq_band|"
    r"set_matrix_eq_on|set_matrix_eq_band|set_compressor|set_compressor_gain|"
    r"set_bus_compressor|set_bus_compressor_gain|"
    r"set_mute|set_channel_mute|set_bus_mute|set_dca_mute|set_group_mute|"
    r"set_main_mute|set_matrix_mute|"
    r"set_gate|set_hpf|set_send_level|set_pan|set_channel_pan|"
    r"set_delay|set_polarity|set_bus_pan|set_main_pan|set_fx_model|set_fx_on|"
    r"set_fx_mix|set_fx_parameter|set_fx_return_level|set_insert|"
    r"set_channel_main_send|set_reverb_send|route_output|load_snap|save_snap|reset_eq|"
    r"reset_channel_eq|reset_channel_hpf|reset_channel_dynamics|reset_channel_processing|"
    r"mute_channel|unmute_channel"
    r")['\"]"
    r"|\bclient\.send\(\s*(?:addr|address)\s*,\s*value\s*\)"
)


def _inventory_paths() -> tuple[str, ...]:
    return tuple(path for surface in iter_write_surfaces() for path in surface.paths)


def _path_is_inventory_covered(relative_path: str, inventory_paths: tuple[str, ...]) -> bool:
    for inventory_path in inventory_paths:
        if inventory_path.endswith("/"):
            if relative_path.startswith(inventory_path):
                return True
        elif relative_path == inventory_path:
            return True
    return False


def _is_excluded(relative_path: str) -> bool:
    return relative_path in EXCLUDED_FILES or any(
        relative_path.startswith(prefix) for prefix in EXCLUDED_PREFIXES
    )


def _files_with_risky_write_tokens() -> list[str]:
    files: list[str] = []
    for root_name in SCAN_ROOTS:
        root = REPO_ROOT / root_name
        if not root.exists():
            continue
        for dirpath, dirnames, filenames in os.walk(root):
            dirnames[:] = [dirname for dirname in dirnames if dirname not in PRUNED_DIRS]
            base_path = Path(dirpath)
            for filename in filenames:
                path = base_path / filename
                if not path.is_file() or path.suffix not in SCAN_SUFFIXES:
                    continue
                relative_path = path.relative_to(REPO_ROOT).as_posix()
                if _is_excluded(relative_path):
                    continue
                source = path.read_text(encoding="utf-8", errors="ignore")
                if RISKY_WRITE_TOKEN_RE.search(source):
                    files.append(relative_path)
    return sorted(files)


def test_write_surface_inventory_covers_files_with_direct_write_tokens():
    """New direct-write token files must be represented in the safety inventory."""

    inventory_paths = _inventory_paths()
    risky_files = _files_with_risky_write_tokens()

    assert risky_files
    uncovered = [
        path for path in risky_files if not _path_is_inventory_covered(path, inventory_paths)
    ]

    assert uncovered == []


def test_static_scan_treats_string_mute_actions_as_write_tokens():
    """String action names can become write intents and must stay inventoried."""

    assert RISKY_WRITE_TOKEN_RE.search("'mute_channel'")
    assert RISKY_WRITE_TOKEN_RE.search('"unmute_channel"')
    assert RISKY_WRITE_TOKEN_RE.search("action='mute_channel'")
    assert RISKY_WRITE_TOKEN_RE.search("'set_bus_mute'")
    assert RISKY_WRITE_TOKEN_RE.search("'set_dca_mute'")
    assert RISKY_WRITE_TOKEN_RE.search('"set_group_mute"')
    assert RISKY_WRITE_TOKEN_RE.search('"set_main_mute"')
    assert RISKY_WRITE_TOKEN_RE.search("'set_matrix_mute'")


def test_static_scan_treats_group_dca_matrix_balance_aliases_as_write_tokens():
    """Group, DCA, and matrix balance wrappers are live-write adjacent."""

    assert RISKY_WRITE_TOKEN_RE.search("client.set_bus_fader(bus, value)")
    assert RISKY_WRITE_TOKEN_RE.search("client.set_dca_fader(dca, value)")
    assert RISKY_WRITE_TOKEN_RE.search("client.set_dca_level(dca, value)")
    assert RISKY_WRITE_TOKEN_RE.search("client.set_group_fader(group, value)")
    assert RISKY_WRITE_TOKEN_RE.search("client.set_matrix_fader(matrix, value)")
    assert RISKY_WRITE_TOKEN_RE.search("'set_group_fader'")
    assert RISKY_WRITE_TOKEN_RE.search('"set_matrix_fader"')


def test_static_scan_treats_preamp_trim_aliases_as_write_tokens():
    """Preamp/input-trim aliases can mutate headroom and must stay inventoried."""

    assert RISKY_WRITE_TOKEN_RE.search("client.set_trim(channel, value)")
    assert RISKY_WRITE_TOKEN_RE.search("client.set_input_trim(channel, value)")
    assert RISKY_WRITE_TOKEN_RE.search("client.set_channel_trim(channel, value)")
    assert RISKY_WRITE_TOKEN_RE.search("client.set_preamp_trim(channel, value)")
    assert RISKY_WRITE_TOKEN_RE.search("client.set_preamp_gain(channel, value)")
    assert RISKY_WRITE_TOKEN_RE.search("'set_trim'")
    assert RISKY_WRITE_TOKEN_RE.search('"set_input_trim"')
    assert RISKY_WRITE_TOKEN_RE.search("'set_channel_trim'")
    assert RISKY_WRITE_TOKEN_RE.search('"set_preamp_trim"')
    assert RISKY_WRITE_TOKEN_RE.search("'set_preamp_gain'")


def test_static_scan_treats_bus_main_matrix_processing_aliases_as_write_tokens():
    """Bus/main/matrix EQ, dynamics, pan, and send aliases must stay inventoried."""

    assert RISKY_WRITE_TOKEN_RE.search("client.set_bus_eq_on(bus, on)")
    assert RISKY_WRITE_TOKEN_RE.search("client.set_bus_eq_band(bus, band)")
    assert RISKY_WRITE_TOKEN_RE.search("client.set_main_eq_on(main, on)")
    assert RISKY_WRITE_TOKEN_RE.search("client.set_main_eq_band(main, band)")
    assert RISKY_WRITE_TOKEN_RE.search("client.set_matrix_eq_on(matrix, on)")
    assert RISKY_WRITE_TOKEN_RE.search("client.set_matrix_eq_band(matrix, band)")
    assert RISKY_WRITE_TOKEN_RE.search("client.set_bus_compressor(bus, threshold)")
    assert RISKY_WRITE_TOKEN_RE.search("client.set_bus_compressor_gain(bus, gain)")
    assert RISKY_WRITE_TOKEN_RE.search("client.set_bus_pan(bus, pan)")
    assert RISKY_WRITE_TOKEN_RE.search("client.set_main_pan(main, pan)")
    assert RISKY_WRITE_TOKEN_RE.search("client.set_channel_pan(channel, pan)")
    assert RISKY_WRITE_TOKEN_RE.search("'set_matrix_eq_band'")
    assert RISKY_WRITE_TOKEN_RE.search('"set_bus_compressor_gain"')


def test_static_scan_treats_fx_return_and_send_aliases_as_write_tokens():
    """FX-return level and send wrappers can mutate WING mix state."""

    assert RISKY_WRITE_TOKEN_RE.search("client.set_fx_return_level(bus, value)")
    assert RISKY_WRITE_TOKEN_RE.search("client.set_channel_main_send(channel, main)")
    assert RISKY_WRITE_TOKEN_RE.search("client.set_reverb_send(channel, level)")
    assert RISKY_WRITE_TOKEN_RE.search("'set_fx_return_level'")
    assert RISKY_WRITE_TOKEN_RE.search('"set_channel_main_send"')


def test_static_scan_treats_raw_client_send_value_probes_as_write_tokens():
    """Raw OSC probes with a value argument can mutate the console."""

    assert RISKY_WRITE_TOKEN_RE.search("client.send(addr, value)")
    assert RISKY_WRITE_TOKEN_RE.search("client.send(address, value)")


def test_static_scan_treats_fx_and_insert_actions_as_write_tokens():
    """FX and insert action names can rewrite live routing/color state."""

    assert RISKY_WRITE_TOKEN_RE.search("client.set_fx_model(slot, model)")
    assert RISKY_WRITE_TOKEN_RE.search("client.set_fx_on(slot, 1)")
    assert RISKY_WRITE_TOKEN_RE.search("client.set_fx_mix(slot, 100.0)")
    assert RISKY_WRITE_TOKEN_RE.search("client.set_fx_parameter(slot, parameter, value)")
    assert RISKY_WRITE_TOKEN_RE.search("client.set_insert(target_type, target, position)")
    assert RISKY_WRITE_TOKEN_RE.search("'set_fx_parameter'")
    assert RISKY_WRITE_TOKEN_RE.search('"set_insert"')


def test_static_scan_treats_channel_processing_resets_as_write_tokens():
    """Channel reset helpers can erase console state and must stay inventoried."""

    assert RISKY_WRITE_TOKEN_RE.search("client.reset_channel_eq(channel)")
    assert RISKY_WRITE_TOKEN_RE.search("client.reset_channel_hpf(channel)")
    assert RISKY_WRITE_TOKEN_RE.search("client.reset_channel_dynamics(channel)")
    assert RISKY_WRITE_TOKEN_RE.search("client.reset_channel_processing(channel)")
    assert RISKY_WRITE_TOKEN_RE.search("'reset_channel_processing'")


def test_blocked_manual_recovery_scripts_do_not_expose_auto_confirm_cli_flags():
    """Bulk recovery scripts must not offer a non-interactive live-mutation bypass."""

    forbidden_patterns = (
        re.compile(r"\bauto_confirm\b"),
        re.compile(r"['\"]--yes['\"]"),
        re.compile(r"['\"]-y['\"]"),
    )

    offenders = {}
    for relative_path in MANUAL_RECOVERY_SCRIPTS:
        source = (REPO_ROOT / relative_path).read_text(encoding="utf-8", errors="ignore")
        matches = [
            pattern.pattern for pattern in forbidden_patterns if pattern.search(source)
        ]
        if matches:
            offenders[relative_path] = matches

    assert offenders == {}


def test_blocked_manual_recovery_scripts_require_ack_before_connecting():
    """Blocked recovery scripts must fail closed before constructing a WING client."""

    offenders = {}
    for relative_path in MANUAL_RECOVERY_SCRIPTS:
        source = (REPO_ROOT / relative_path).read_text(encoding="utf-8", errors="ignore")
        guard_index = source.find("require_manual_recovery_ack")
        connect_index = source.find("WingClient(")
        if guard_index < 0 or connect_index < 0 or guard_index > connect_index:
            offenders[relative_path] = {
                "guard_index": guard_index,
                "connect_index": connect_index,
            }

    assert offenders == {}


def test_manual_recovery_ack_guard_blocks_by_default(monkeypatch, capsys):
    """The acknowledgement gate must not pass from an ordinary environment."""

    monkeypatch.delenv(ACK_ENV_VAR, raising=False)

    assert not has_manual_recovery_ack()
    with pytest.raises(SystemExit) as excinfo:
        require_manual_recovery_ack("bulk_reset_test.py")

    assert excinfo.value.code == 2
    captured = capsys.readouterr()
    assert ACK_ENV_VAR in captured.err
    assert ACK_VALUE in captured.err


def test_manual_recovery_ack_guard_accepts_exact_operator_ack(monkeypatch):
    """Only the exact manual-recovery acknowledgement string opens the script gate."""

    monkeypatch.setenv(ACK_ENV_VAR, ACK_VALUE)

    require_manual_recovery_ack("bulk_reset_test.py")
