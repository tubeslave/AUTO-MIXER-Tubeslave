"""Zero-contact contracts for operator-owned dLive packet evidence."""

import json
from pathlib import Path
import socket
import subprocess
import sys

import pytest

from dlive_midi_capture_evidence import review_dlive_fader_midi_capture


ROOT = Path(__file__).resolve().parents[1]
TOOL = ROOT / "tools" / "dlive_review_midi_capture.py"


def capture(**overrides):
    value = {
        "schema": "automixer.dlive_midi_capture.v1",
        "capture_id": "capture-c3500-1",
        "captured_at_utc": "2026-08-04T20:00:00Z",
        "capture_tool": "operator-owned-packet-capture",
        "source_client": "allen_heath_dlive_director",
        "operator_owned": True,
        "acquired_by_automixer": False,
        "interface_opened_by_automixer": False,
        "direction": "client_to_selected_endpoint",
        "observation_point": "selected_endpoint_ingress",
        "endpoint": {
            "host": "192.168.3.71",
            "endpoint": "surface",
            "port": 51328,
            "tls": False,
            "midi_base_channel": 11,
            "user_profile": 0,
        },
        "bytes_hex": "BB 63 00 62 17 06 65",
    }
    value.update(overrides)
    return value


def review(value):
    return review_dlive_fader_midi_capture(
        value,
        host="192.168.3.71",
        endpoint="surface",
        port=51328,
        tls=False,
        midi_base_channel=11,
        user_profile=0,
        input_channel=1,
        target_db=-3.0,
    )


def test_exact_compact_nrpn_ingress_capture_is_accepted_without_authority():
    report = review(capture())

    assert report["status"] == "accepted"
    assert report["expected_packet_hex"] == "bb630062170665"
    assert report["captured_packet_hex"] == report["expected_packet_hex"]
    assert report["exact_wire_match"] is True
    assert report["wire_delivery_observed"] is True
    assert report["console_acceptance_proven"] is False
    assert report["physical_fader_movement_proven"] is False
    assert report["console_contacted_by_automixer"] is False
    assert report["midi_sent_by_automixer"] is False
    assert report["live_authority_granted"] is False


def test_review_never_constructs_a_socket(monkeypatch):
    monkeypatch.setattr(
        socket,
        "socket",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("offline review must not construct a socket")
        ),
    )

    assert review(capture())["status"] == "accepted"


@pytest.mark.parametrize(
    "changes,reason",
    [
        ({"bytes_hex": "BB 63 00 62 17 06 64"}, "capture_packet_does_not_match"),
        ({"operator_owned": False}, "capture_must_be_operator_owned"),
        ({"acquired_by_automixer": True}, "capture_must_not_be_acquired"),
        ({"interface_opened_by_automixer": True}, "capture_interface_must_not"),
        ({"source_client": "unknown"}, "capture_source_client_not_known_good"),
        ({"direction": "console_to_client"}, "capture_direction_invalid"),
        ({"observation_point": "client_egress"}, "capture_not_observed"),
        ({"password": "forbidden"}, "capture_contains_sensitive_fields"),
    ],
)
def test_untrusted_or_non_ingress_capture_is_rejected(changes, reason):
    with pytest.raises(ValueError, match=reason):
        review(capture(**changes))


def test_selected_endpoint_binding_is_exact():
    value = capture()
    value["endpoint"] = dict(value["endpoint"], host="192.168.3.70")

    with pytest.raises(ValueError, match="capture_endpoint_binding_mismatch:host"):
        review(value)


def test_cli_reads_supplied_file_and_emits_accepted_report(tmp_path):
    capture_path = tmp_path / "capture.json"
    capture_path.write_text(json.dumps(capture()), encoding="utf-8")

    result = subprocess.run(
        [
            sys.executable,
            str(TOOL),
            "--capture",
            str(capture_path),
            "--host",
            "192.168.3.71",
            "--endpoint",
            "surface",
            "--port",
            "51328",
            "--midi-base-channel",
            "11",
            "--user-profile",
            "0",
            "--input-channel",
            "1",
            "--target-db",
            "-3.0",
        ],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 0
    report = json.loads(result.stdout)
    assert report["status"] == "accepted"
    assert report["console_contacted_by_automixer"] is False
    assert report["mutation_write_attempt_count"] == 0


def test_cli_and_review_module_have_no_transport_calls():
    sources = [
        TOOL.read_text(encoding="utf-8"),
        (ROOT / "backend" / "dlive_midi_capture_evidence.py").read_text(
            encoding="utf-8"
        ),
    ]
    for source in sources:
        assert "socket.socket" not in source
        assert ".connect(" not in source
        assert "_send_raw" not in source
