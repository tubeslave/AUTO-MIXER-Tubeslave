import ast
from pathlib import Path

import pytest

from backend import list_snapshots, load_snap, route_channel_inputs, route_custom_outputs
from backend import restore_channels, scan_and_load_snap


class FailIfTransportUsed:
    def __getattr__(self, name):
        raise AssertionError(f"quarantined surface touched transport: {name}")


def test_legacy_snapshot_and_restore_surfaces_stop_before_transport_or_file_reads(tmp_path):
    client = FailIfTransportUsed()
    missing = tmp_path / "attacker" / "snapshot.json"

    assert load_snap.load_snap(client, "scene") is False
    assert scan_and_load_snap.scan_snapshots(client) == {}
    assert scan_and_load_snap.find_snap_by_name(client, "scene") == (None, None)
    assert scan_and_load_snap.load_snap_by_index(client, 1) is False
    assert list_snapshots.get_snapshots_list(client) == {}
    assert restore_channels.restore_from_backup_using_client(client, str(missing)) is False
    with pytest.raises(PermissionError, match="quarantined"):
        restore_channels.restore_channel(client, {"channel": 1})
    assert not missing.exists()


def test_legacy_routing_surfaces_return_blocked_proposals_without_writes():
    client = FailIfTransportUsed()

    inputs = route_channel_inputs.route_channel_inputs(client, num_channels=40)
    outputs = route_custom_outputs.route_custom_outputs(
        client,
        "A",
        [(1, "MOD", 1), (2, "MOD", 2)],
    )

    assert inputs["status"] == "blocked"
    assert inputs["attempted_writes"] == 0
    assert inputs["live_authority_granted"] is False
    assert outputs["status"] == "blocked"
    assert outputs["attempted_writes"] == 0
    assert outputs["live_authority_granted"] is False


def test_legacy_live_tool_entrypoints_raise_before_transport_creation():
    repo = Path(__file__).resolve().parents[1]
    expected = {
        "tools/live_wing_soundcheck_v2.py": {"main"},
        "tools/live_wing_soundcheck_v2_stage2.py": {"main"},
        "tools/live_wing_soundcheck_v2_stage3.py": {"main"},
        "tools/live_wing_soundcheck_daemon.py": {"main"},
        "tools/stem_bus_routing_setup.py": {"main"},
        "tools/master_pink_balance.py": {"main"},
        "tools/mixing_rule_trial.py": {"run_trial", "rollback_trial"},
        "tools/dca_groups_setup.py": {"main"},
        "tools/fx_rack_setup.py": {"connect", "main"},
        "tools/wing_method_proof_runner.py": {"main"},
    }
    for relative, function_names in expected.items():
        tree = ast.parse((repo / relative).read_text(encoding="utf-8"))
        functions = {
            node.name: node
            for node in tree.body
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        }
        for name in function_names:
            assert isinstance(functions[name].body[0], ast.Raise), f"{relative}:{name}"


def test_legacy_mcp_surfaces_are_minimal_fail_closed_stubs():
    repo = Path(__file__).resolve().parents[1]
    files = [
        repo / "tools/mcp_director/server.js",
        repo / "tools/mcp_director/install_mcp_director.sh",
        repo / "tools/mcp_remote_gateway/server.js",
        repo / "tools/mcp_remote_gateway/start.sh",
    ]
    combined = "\n".join(path.read_text(encoding="utf-8") for path in files)
    assert "quarantined" in combined
    for forbidden in ("git checkout", "TELEGRAM_BOT_TOKEN", "createMcpExpressApp", "fs.writeFile"):
        assert forbidden not in combined
