"""FX and insert helpers for backend/wing_client.py."""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "backend"))

from wing_client import WingClient


class RecordingWingClient(WingClient):
    def __init__(self):
        super().__init__("127.0.0.1", 2223)
        self.is_connected = True
        self.sent = []

    def send(self, address: str, *values):
        self.sent.append((address, values))
        if values:
            self.state[address] = values[0] if len(values) == 1 else values
        return True


def test_set_fx_mix_uses_official_fxmix_path():
    client = RecordingWingClient()

    client.set_fx_mix("FX7", 82.5)

    assert client.sent == [("/fx/7/fxmix", (82.5,))]
    assert client.state["/fx/7/fxmix"] == 82.5


def test_get_insert_returns_linked_fx_info():
    client = RecordingWingClient()
    client.state.update(
        {
            "/main/1/postins/on": 1,
            "/main/1/postins/ins": "FX3",
            "/main/1/postins/$stat": "OK",
            "/fx/3/mdl": "PLATE",
            "/fx/3/on": 1,
            "/fx/3/fxmix": 100.0,
            "/fx/3/1": 2.5,
        }
    )

    result = client.get_insert("main", 1, "post")

    assert result["slot"] == "FX3"
    assert result["status"] == "OK"
    assert result["fx_module"]["model"] == "PLATE"
    assert result["fx_module"]["mix"] == 100.0
    assert result["fx_module"]["parameters"][1] == 2.5


def test_set_insert_writes_main_post_insert_paths():
    client = RecordingWingClient()

    client.set_insert("main", 1, "post", slot="FX5", on=1)

    assert client.sent == [
        ("/main/1/postins/on", (1,)),
        ("/main/1/postins/ins", ("FX5",)),
    ]
