from pathlib import Path

import auto_test_routing as routing


class _FailIfUsedClient:
    def __getattr__(self, name):
        raise AssertionError(f"transport client must remain unused: {name}")


def test_auto_test_routing_is_proposal_only_and_never_uses_transport():
    result = routing.auto_test_routing(
        _FailIfUsedClient(),
        channel=1,
        dante_output=2,
    )

    assert result["status"] == "blocked"
    assert result["reason"] == "routing_live_apply_requires_supervised_authority"
    assert result["live_authority_granted"] is False
    assert result["live_write"] is False
    assert result["attempted_writes"] == 0
    assert len(result["proposals"]) == 6
    assert sum(len(item["addresses"]) for item in result["proposals"]) == 12
    assert result["proposals"][0]["addresses"] == [
        ("/routing/outputs/ch/1/srcgrp", "WLIVE PLAY"),
        ("/routing/outputs/ch/1/outgrp", 2),
    ]


def test_cli_fails_closed_without_wing_transport(monkeypatch):
    monkeypatch.setattr(routing.sys, "argv", ["auto_test_routing.py", "10.0.0.8"])

    assert routing.main() == 2


def test_module_contains_no_transport_or_mutation_sink():
    source = Path(routing.__file__).read_text(encoding="utf-8")

    assert "WingClient" not in source
    assert ".connect(" not in source
    assert ".send(" not in source
    assert ".disconnect(" not in source
