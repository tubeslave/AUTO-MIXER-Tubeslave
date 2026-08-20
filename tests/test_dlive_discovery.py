import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "backend"))

from mixer_discovery import DLIVE_PORTS, _probe_dlive_tcp


def test_all_official_surface_and_mixrack_ports_are_scanned():
    assert DLIVE_PORTS == {
        51325: ("mixrack", False),
        51327: ("mixrack", True),
        51328: ("surface", False),
        51329: ("surface", True),
    }


def test_non_dlive_port_is_rejected_without_socket(monkeypatch):
    monkeypatch.setattr(
        "mixer_discovery.socket.socket",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("socket must not be created")
        ),
    )
    assert _probe_dlive_tcp("192.168.3.70", 12345) is None


class _OpenSocket:
    def settimeout(self, timeout):
        self.timeout = timeout

    def connect_ex(self, address):
        self.address = address
        return 0

    def close(self):
        pass


def test_open_mixrack_tls_port_remains_unverified_candidate(monkeypatch):
    monkeypatch.setattr("mixer_discovery.socket.socket", lambda *args, **kwargs: _OpenSocket())
    candidate = _probe_dlive_tcp("192.168.3.70", 51327)
    assert candidate.endpoint == "mixrack"
    assert candidate.tls is True
    assert candidate.identity_verified is False
