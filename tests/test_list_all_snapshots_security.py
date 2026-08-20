import pytest

from backend import list_all_snapshots as snapshot_listing


class _SnapshotClient:
    def __init__(self, names, *, interrupt_at=None):
        self.names = dict(names)
        self.interrupt_at = interrupt_at
        self.calls = []
        self.state = {
            "/$ctl/lib/$active": "ACTIVE_SCENE",
            "/$ctl/lib/$actidx": 7,
            "/$ctl/lib/$actionidx": 7,
        }

    def send(self, address, value=None):
        self.calls.append((address, value))
        if address == "/$ctl/lib/$actionidx" and value is not None:
            self.state[address] = value
        elif address == "/$ctl/lib/$name":
            cursor = self.state["/$ctl/lib/$actionidx"]
            if cursor == self.interrupt_at:
                raise KeyboardInterrupt
            if cursor in self.names:
                self.state[address] = self.names[cursor]
        return True


def test_list_all_snapshots_queries_names_without_loading_scenes(monkeypatch):
    monkeypatch.setattr(snapshot_listing.time, "sleep", lambda _seconds: None)
    client = _SnapshotClient({1: "ONE", 2: "TWO", 3: "TWO"})

    result = snapshot_listing.list_all_snapshots(client, max_index=4)

    assert result == {1: "ONE", 2: "TWO", 3: "TWO"}
    assert client.state["/$ctl/lib/$active"] == "ACTIVE_SCENE"
    assert client.state["/$ctl/lib/$actidx"] == 7
    assert client.state["/$ctl/lib/$actionidx"] == 7
    assert all(address != "/$ctl/lib/$action" for address, _value in client.calls)
    assert all(value not in {"GO", "LOAD"} for _address, value in client.calls)


def test_list_all_snapshots_restores_cursor_when_interrupted(monkeypatch):
    monkeypatch.setattr(snapshot_listing.time, "sleep", lambda _seconds: None)
    client = _SnapshotClient({1: "ONE"}, interrupt_at=2)

    with pytest.raises(KeyboardInterrupt):
        snapshot_listing.list_all_snapshots(client, max_index=3)

    assert client.state["/$ctl/lib/$actionidx"] == 7
    assert all(address != "/$ctl/lib/$action" for address, _value in client.calls)


@pytest.mark.parametrize("max_index", [True, 0, 501, "10"])
def test_list_all_snapshots_rejects_unbounded_or_coerced_scan_ranges(max_index):
    client = _SnapshotClient({})

    with pytest.raises(ValueError, match="max_index must be an integer"):
        snapshot_listing.list_all_snapshots(client, max_index=max_index)

    assert client.calls == []
