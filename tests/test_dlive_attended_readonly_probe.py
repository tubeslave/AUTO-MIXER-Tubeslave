import importlib.util
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
TOOL = ROOT / "tools" / "dlive_attended_readonly_probe.py"
SPEC = importlib.util.spec_from_file_location("dlive_attended_readonly_probe", TOOL)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


class _FakeReadonlyClient:
    def __init__(self, **kwargs):
        self.port = kwargs.get("port") or 51328
        self.is_connected = False
        self._attended_write_authority = False
        self.queries = []

    def connect(self, timeout):
        self.is_connected = True
        return True

    def query_parameter(self, address, timeout_sec):
        self.queries.append(address)
        value = "Vocal" if address.endswith("/name") else (-6.0 if address.endswith("/fdr") else False)
        return {
            "status": "received",
            "address": address,
            "value": value,
            "confirmed_round_trips": 2,
        }

    def disconnect(self):
        self.is_connected = False


def _run(**overrides):
    values = {
        "host": "192.168.3.70",
        "endpoint": "surface",
        "port": 51328,
        "tls": False,
        "tls_ca_file": None,
        "tls_server_hostname": None,
        "midi_base_channel": 0,
        "user_profile": 0,
        "password": "",
        "input_channel": 5,
        "operator_attested_model": "S7000",
        "operator_attested_firmware": "2.0",
        "operator_attested_show": "TEST",
        "expected_name": "Vocal",
        "timeout_sec": 0.1,
        "expected_interface": "en7",
        "client_factory": _FakeReadonlyClient,
        "route_interface_resolver": lambda host: "en7",
    }
    values.update(overrides)
    return MODULE.run_probe(**values)


def test_probe_records_three_double_readbacks_and_zero_authority():
    artifact = _run()
    assert artifact["status"] == "readback_observed"
    assert artifact["midi_get_request_count"] == 6
    assert artifact["mutation_write_attempt_count"] == 0
    assert artifact["live_write"] is False
    assert artifact["live_authority_granted"] is False
    assert artifact["user_profile"] == 0
    assert artifact["teardown"]["client_connected_after_teardown"] is False


def test_expected_name_mismatch_fails_closed():
    artifact = _run(expected_name="Kick")
    assert artifact["status"] == "blocked"
    assert artifact["reason"] == "operator_expected_channel_name_mismatch"
    assert artifact["live_authority_granted"] is False


def test_wrong_route_interface_blocks_before_client_creation():
    def forbidden_client(**kwargs):
        raise AssertionError("TCP client must not be created on the wrong route")

    artifact = _run(
        client_factory=forbidden_client,
        route_interface_resolver=lambda host: "utun4",
    )
    assert artifact["status"] == "blocked"
    assert artifact["reason"] == "dlive_route_interface_mismatch"
    assert artifact["observed_route_interface"] == "utun4"
    assert artifact["console_contacted"] is False


def test_runner_source_has_no_mutation_calls():
    source = TOOL.read_text(encoding="utf-8")
    for forbidden in (
        ".set_fader(",
        ".set_mute(",
        ".set_gain(",
        ".set_phantom(",
        ".recall_scene(",
        "attended_write_authority=True",
    ):
        assert forbidden not in source
