"""FX and insert helpers for backend/wing_client.py."""

import os
import sys
from pathlib import Path

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


def test_fx_on_fails_closed_because_wing_has_no_slot_on_path():
    client = RecordingWingClient()

    assert client.set_fx_on("FX7", 1) is False

    assert client.sent == []
    assert "/fx/7/on" not in client.state


def test_query_fx_slot_uses_only_official_fw31_paths():
    client = RecordingWingClient()

    client.query_fx_slot("FX7")

    assert client.sent == [
        ("/fx/7/mdl", ()),
        ("/fx/7/fxmix", ()),
        ("/fx/7/$esrc", ()),
        ("/fx/7/$emode", ()),
        ("/fx/7/$a_chn", ()),
        ("/fx/7/$a_pos", ()),
    ]
    assert all(address != "/fx/7/on" for address, _ in client.sent)


def test_wing_client_does_not_write_unsupported_fx_on_path_static():
    source = (Path(__file__).resolve().parents[1] / "backend" / "wing_client.py").read_text(
        encoding="utf-8"
    )

    assert 'send(f"/fx/{fx_num}/on", on)' not in source
    assert 'send(f"/fx/{fx_num}/on")' not in source


def test_reverb_params_do_not_write_unsupported_fx_on_path():
    client = RecordingWingClient()

    assert client.set_reverb_params(
        1,
        "plate",
        decay_time=2.5,
        predelay=20.0,
        brightness=60.0,
        density=80.0,
        hpf=160.0,
        lpf=9000.0,
        fx_slot="FX3",
    ) is True

    sent_addresses = [address for address, _ in client.sent]
    assert "/fx/3/on" not in sent_addresses
    assert sent_addresses == [
        "/fx/3/mdl",
        "/fx/3/fxmix",
        "/fx/3/1",
        "/fx/3/2",
        "/fx/3/3",
        "/fx/3/4",
        "/fx/3/5",
        "/fx/3/6",
    ]


def test_get_insert_returns_linked_fx_info():
    client = RecordingWingClient()
    client.state.update(
        {
            "/main/1/postins/on": 1,
            "/main/1/postins/ins": "FX3",
            "/main/1/postins/$stat": "OK",
            "/fx/3/mdl": "PLATE",
            "/fx/3/fxmix": 100.0,
            "/fx/3/1": 2.5,
        }
    )

    result = client.get_insert("main", 1, "post")

    assert result["slot"] == "FX3"
    assert result["status"] == "OK"
    assert result["fx_module"]["model"] == "PLATE"
    assert result["fx_module"]["on"] == 1
    assert result["fx_module"]["active"] is True
    assert result["fx_module"]["mix"] == 100.0
    assert result["fx_module"]["parameters"][1] == 2.5


def test_set_insert_writes_main_post_insert_paths():
    client = RecordingWingClient()

    client.set_insert("main", 1, "post", slot="FX5", on=1)

    assert client.sent == [
        ("/main/1/postins/on", (1,)),
        ("/main/1/postins/ins", ("FX5",)),
    ]
