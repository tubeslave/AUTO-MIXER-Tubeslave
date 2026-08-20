"""CLI contracts for the attended dLive restore-only proof."""

import json

from tools import dlive_attended_fader_restore_proof as runner


def _args(tmp_path, confirmation):
    return [
        "--operator-attended",
        "--execute-restore-only-proof",
        "--pa-outputs-contained",
        "--input-confirmed-muted",
        "--receive-midi-confirmed-enabled",
        "--readonly-artifact",
        str(tmp_path / "probe.json"),
        "--readonly-artifact-sha256",
        "d4f5dd8ff84d0be7aaf0c5db2352450b59edd8f1170557fa8385288adfc44c40",
        "--wire-evidence",
        str(tmp_path / "wire-review.json"),
        "--wire-evidence-sha256",
        "e4f5dd8ff84d0be7aaf0c5db2352450b59edd8f1170557fa8385288adfc44c40",
        "--security-state-attested",
        "--security-mode",
        "no_security",
        "--confirmation",
        confirmation,
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
        "--expected-interface",
        "en7",
        "--input-channel",
        "1",
        "--expected-name",
        "KICK",
        "--model",
        "CDM32+C3500",
        "--firmware",
        "2.0",
        "--show",
        "lv 1 master",
        "--target-db",
        "-3.0",
        "--max-cut-db",
        "3.0",
    ]


def test_generic_confirmation_blocks_before_proof_runner(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(
        runner,
        "run_dlive_fader_restore_proof",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("proof runner must not be called")
        ),
    )

    result = runner.main(
        _args(tmp_path, "APPLY_AND_ROLLBACK_DLIVE_FADER_PROOF")
    )
    payload = json.loads(capsys.readouterr().out)

    assert result == 2
    assert payload["status"] == "blocked"
    assert payload["reason"] == (
        "artifact_target_bound_operator_confirmation_required"
    )
    assert payload["required_confirmation"].startswith(
        "APPLY_AND_ROLLBACK_DLIVE_FADER_PROOF:sha256:"
    )
    assert payload["required_approval_id"]
    assert payload["hardware_contacted"] is False
    assert payload["live_write"] is False
