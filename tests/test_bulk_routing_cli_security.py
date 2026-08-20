from pathlib import Path

import route_channels as bulk
import route_dante_outputs as dante


class _FailIfUsedClient:
    def __getattr__(self, name):
        raise AssertionError(f"transport client must remain unused: {name}")


def test_generic_bulk_routing_is_proposal_only():
    result = bulk.route_channels(_FailIfUsedClient(), "MOD", 2, 3, "CRD", 5)

    assert result["status"] == "blocked"
    assert result["attempted_writes"] == 0
    assert result["live_authority_granted"] is False
    assert result["proposals"] == [
        {"output_group": "MOD", "output_number": 2, "source_group": "CRD", "source_channel": 5},
        {"output_group": "MOD", "output_number": 3, "source_group": "CRD", "source_channel": 6},
        {"output_group": "MOD", "output_number": 4, "source_group": "CRD", "source_channel": 7},
    ]


def test_dante_bulk_routing_and_debug_reads_are_proposal_only():
    result = dante.route_channels_to_dante(
        _FailIfUsedClient(),
        start_channel=3,
        num_channels=2,
        dante_start_output=7,
        card_name="WLIVE",
    )

    assert result["status"] == "blocked"
    assert result["attempted_writes"] == 0
    assert result["read_candidates"] == [
        "/ch/3/out/conn/grp",
        "/ch/3/out/conn/out",
        "/ch/3/out/grp",
        "/ch/3/out/out",
    ]
    assert [(row["output_number"], row["source_channel"]) for row in result["proposals"]] == [
        (7, 3),
        (8, 4),
    ]


def test_bulk_routing_clis_fail_closed_without_transport(monkeypatch):
    monkeypatch.setattr(
        bulk.sys,
        "argv",
        ["route_channels.py", "10.0.0.8", "MOD", "1", "2", "CRD", "1"],
    )
    assert bulk.main() == 2

    monkeypatch.setattr(
        dante.sys,
        "argv",
        ["route_dante_outputs.py", "10.0.0.8", "1", "2", "1", "WLIVE"],
    )
    assert dante.main() == 2


def test_bulk_routing_modules_contain_no_transport_or_mutation_sink():
    for module in (bulk, dante):
        source = Path(module.__file__).read_text(encoding="utf-8")
        assert "WingClient" not in source
        assert ".connect(" not in source
        assert ".send(" not in source
        assert ".route_multiple_outputs(" not in source
        assert ".disconnect(" not in source
