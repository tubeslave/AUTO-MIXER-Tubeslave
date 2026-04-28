import requests

from integrations.mixing_station.rest_client import (
    UNAVAILABLE_MESSAGE,
    MixingStationRestClient,
)


class FakeResponse:
    def __init__(self, status_code=200, data=None):
        self.status_code = status_code
        self._data = data if data is not None else {"ok": True}
        self.content = b"{}"
        self.text = "{}"

    def json(self):
        return self._data


class FakeSession:
    def __init__(self, response=None, exc=None):
        self.response = response or FakeResponse()
        self.exc = exc
        self.calls = []

    def request(self, method, url, timeout=None, **kwargs):
        self.calls.append((method, url, timeout, kwargs))
        if self.exc:
            raise self.exc
        return self.response


def test_rest_client_reports_unavailable_when_connection_fails():
    session = FakeSession(exc=requests.ConnectionError("refused"))
    client = MixingStationRestClient(session=session)

    result = client.health_check()

    assert result.success is False
    assert UNAVAILABLE_MESSAGE in result.error


def test_rest_client_reads_app_state_from_first_working_endpoint():
    session = FakeSession(response=FakeResponse(data={"app": "state"}))
    client = MixingStationRestClient(session=session, app_state_paths=["/app/state"])

    result = client.get_app_state()

    assert result.success is True
    assert result.data == {"app": "state"}
    assert session.calls[0][1] == "http://127.0.0.1:8080/app/state"


def test_rest_client_blocks_write_without_configured_endpoint():
    client = MixingStationRestClient()

    result = client.send_command(
        type("Command", (), {
            "data_path": "ch.0.mix.lvl",
            "method": "SET",
            "value": -5.0,
            "value_format": "plain",
        })()
    )

    assert result.success is False
    assert "write endpoint is not configured" in result.error
