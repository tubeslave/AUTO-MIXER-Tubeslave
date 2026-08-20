#!/usr/bin/env python3
"""Validate one exact dLive fader proof plan without route lookup or TCP."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import uuid

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "backend"))

from dlive_supervised_fader_proof import (  # noqa: E402
    DLiveFaderProofApproval,
    validate_dlive_fader_proof_plan,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--readonly-artifact", type=Path, required=True)
    parser.add_argument("--readonly-artifact-sha256", required=True)
    parser.add_argument("--wire-evidence", type=Path, required=True)
    parser.add_argument("--wire-evidence-sha256", required=True)
    parser.add_argument("--approval-id", default=None)
    parser.add_argument("--host", required=True)
    parser.add_argument("--endpoint", choices=("surface", "mixrack"), required=True)
    parser.add_argument("--port", type=int)
    parser.add_argument("--tls", action="store_true")
    parser.add_argument("--midi-base-channel", type=int, default=11)
    parser.add_argument("--user-profile", type=int, default=0)
    parser.add_argument("--expected-interface", required=True)
    parser.add_argument("--input-channel", type=int, required=True)
    parser.add_argument("--expected-name", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--firmware", required=True)
    parser.add_argument("--show", required=True)
    parser.add_argument("--target-db", type=float, required=True)
    parser.add_argument("--max-cut-db", type=float, default=3.0)
    parser.add_argument("--security-state-attested", action="store_true", required=True)
    parser.add_argument(
        "--security-mode",
        choices=("no_security", "tls_authenticated"),
        required=True,
    )
    parser.add_argument("--security-changed-for-setup", action="store_true")
    parser.add_argument(
        "--reboot-after-security-change-confirmed",
        action="store_true",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    approval = DLiveFaderProofApproval(
        approval_id=args.approval_id or str(uuid.uuid4()),
        issued_at_epoch=0.0,
        expires_at_epoch=1.0,
        host=args.host,
        endpoint=args.endpoint,
        port=args.port,
        tls=args.tls,
        midi_base_channel=args.midi_base_channel,
        user_profile=args.user_profile,
        expected_interface=args.expected_interface,
        input_channel=args.input_channel,
        expected_name=args.expected_name,
        operator_attested_model=args.model,
        operator_attested_firmware=args.firmware,
        operator_attested_show=args.show,
        target_db=args.target_db,
        max_cut_db=args.max_cut_db,
        pa_outputs_contained=False,
        require_channel_muted=False,
        receive_midi_confirmed_enabled=False,
        readonly_artifact_path=str(args.readonly_artifact),
        readonly_artifact_sha256=args.readonly_artifact_sha256,
        confirmation="",
        security_state_attested=args.security_state_attested,
        security_mode_attested=args.security_mode,
        security_changed_for_setup=args.security_changed_for_setup,
        reboot_after_security_change_confirmed=(
            args.reboot_after_security_change_confirmed
        ),
        wire_evidence_path=str(args.wire_evidence),
        wire_evidence_sha256=args.wire_evidence_sha256,
    )
    try:
        receipt = validate_dlive_fader_proof_plan(approval)
    except (OSError, RuntimeError, ValueError) as exc:
        receipt = {
            "schema": "automixer.dlive_fader_proof_plan_validation.v1",
            "status": "blocked",
            "reason": str(exc) or type(exc).__name__,
            "operator_confirmation_recorded": False,
            "physical_conditions_attested": False,
            "route_checked": False,
            "console_contacted": False,
            "hardware_contacted": False,
            "midi_sent": False,
            "mutation_write_attempt_count": 0,
            "live_write": False,
            "execution_allowed": False,
            "live_authority_granted": False,
        }
    print(json.dumps(receipt, ensure_ascii=False, sort_keys=True))
    return 0 if receipt["status"] == "ready_for_operator_confirmation" else 2


if __name__ == "__main__":
    raise SystemExit(main())
