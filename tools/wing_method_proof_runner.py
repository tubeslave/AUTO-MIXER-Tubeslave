#!/usr/bin/env python3
"""Proof runner for show_assist_auto_v1 AutoFOH live actions.

Default mode is local fake-mixer proof only. Real WING writes require all of:
--execute, --confirm-real-write, and --wing-ip. This file exists so a show-day
operator can run the same gated executor with artifacts before promoting any
method beyond proposal/shadow mode.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional


REPO_ROOT = Path(__file__).resolve().parents[1]
BACKEND_PATH = REPO_ROOT / "backend"
if str(BACKEND_PATH) not in sys.path:
    sys.path.insert(0, str(BACKEND_PATH))

from autofoh_models import RuntimeState
from autofoh_safety import (
    AutoFOHSafetyConfig,
    AutoFOHSafetyController,
    ChannelEQMove,
    ChannelFaderMove,
    ChannelGainMove,
    EmergencyFeedbackNotch,
)
from correction_action_executor import (
    CorrectionActionExecutorConfig,
    GatedCorrectionActionExecutor,
)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


def compact_stamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


class ProofMixer:
    is_connected = True

    def __init__(
        self,
        *,
        channel: int,
        initial_fader_db: float,
        initial_gain_db: float,
        initial_eq_gain_db: float,
    ) -> None:
        self.channel = channel
        self.fader = {channel: float(initial_fader_db)}
        self.gain = {channel: float(initial_gain_db)}
        self.eq_gain = {(channel, "2g"): float(initial_eq_gain_db)}
        self.state = {
            f"/ch/{channel}/fdr": float(initial_fader_db),
            f"/ch/{channel}/in/set/trim": float(initial_gain_db),
            f"/ch/{channel}/eq/2/g": float(initial_eq_gain_db),
        }
        self.calls: list[Dict[str, Any]] = []

    def connect(self, timeout: float = 10.0) -> bool:
        return True

    def disconnect(self) -> None:
        self.is_connected = False

    def get_fader(self, channel: int) -> Optional[float]:
        return self.fader.get(channel)

    def set_fader(self, channel: int, value_db: float) -> bool:
        value = float(value_db)
        self.fader[channel] = value
        self.state[f"/ch/{channel}/fdr"] = value
        self.calls.append({"method": "set_fader", "channel": channel, "value": value})
        return True

    def get_channel_gain(self, channel: int) -> Optional[float]:
        return self.gain.get(channel)

    def set_gain(self, channel: int, value_db: float) -> bool:
        value = max(-18.0, min(18.0, float(value_db)))
        self.gain[channel] = value
        self.state[f"/ch/{channel}/in/set/trim"] = value
        self.calls.append({"method": "set_gain", "channel": channel, "value": value})
        return True

    def get_eq_band_gain(self, channel: int, band: Any) -> Optional[float]:
        return self.eq_gain.get((channel, str(band)), 0.0)

    def set_eq_band(self, channel: int, band: int, freq: float, gain: float, q: float) -> bool:
        value = float(gain)
        self.eq_gain[(channel, f"{band}g")] = value
        self.state[f"/ch/{channel}/eq/{band}/g"] = value
        self.calls.append(
            {
                "method": "set_eq_band",
                "channel": channel,
                "band": band,
                "freq": float(freq),
                "gain": value,
                "q": float(q),
            }
        )
        return True


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run gated AutoFOH method proof.")
    parser.add_argument(
        "--action",
        choices=("channel_fader", "channel_gain", "feedback_notch", "channel_eq"),
        default="channel_fader",
    )
    parser.add_argument("--channel", type=int, default=1)
    parser.add_argument("--target-db", type=float, default=-9.5)
    parser.add_argument("--initial-fader-db", type=float, default=-10.0)
    parser.add_argument("--initial-gain-db", type=float, default=0.0)
    parser.add_argument("--initial-eq-gain-db", type=float, default=0.0)
    parser.add_argument("--band", type=int, default=2)
    parser.add_argument("--freq-hz", type=float, default=350.0)
    parser.add_argument("--q", type=float, default=1.2)
    parser.add_argument("--execute", action="store_true", help="Send the method action instead of dry-run.")
    parser.add_argument("--rollback-check", action="store_true", help="Apply the generated rollback action after proof.")
    parser.add_argument("--proof-run", action="store_true", help="Bypass promotion only for this proof run.")
    parser.add_argument("--confirm-real-write", action="store_true", help="Required for physical WING writes.")
    parser.add_argument("--wing-ip", default="")
    parser.add_argument("--wing-port", type=int, default=2223)
    parser.add_argument(
        "--artifact-root",
        default="artifacts/wing_method_proof",
        help="Directory for proof artifacts.",
    )
    return parser


def validate_args(args: argparse.Namespace) -> Optional[str]:
    if args.channel < 1 or args.channel > 48:
        return "--channel must be in 1..48"
    if args.confirm_real_write and not args.execute:
        return "--confirm-real-write requires --execute"
    if args.confirm_real_write and not args.wing_ip:
        return "--confirm-real-write requires --wing-ip"
    if args.rollback_check and not args.execute:
        return "--rollback-check requires --execute"
    if args.action in {"channel_fader", "channel_gain"} and abs(args.target_db) > 144:
        return "--target-db is outside sane proof bounds"
    return None


def build_action(args: argparse.Namespace):
    if args.action == "channel_fader":
        return ChannelFaderMove(
            channel_id=args.channel,
            target_db=args.target_db,
            reason="wing_method_proof_runner",
        )
    if args.action == "channel_gain":
        return ChannelGainMove(
            channel_id=args.channel,
            target_db=args.target_db,
            reason="wing_method_proof_runner",
        )
    if args.action == "feedback_notch":
        return EmergencyFeedbackNotch(
            channel_id=args.channel,
            band=args.band,
            freq_hz=args.freq_hz,
            gain_db=min(args.target_db, -1.0),
            q=max(args.q, 8.0),
            reason="wing_method_proof_runner",
        )
    return ChannelEQMove(
        channel_id=args.channel,
        band=args.band,
        freq_hz=args.freq_hz,
        gain_db=args.target_db,
        q=args.q,
        reason="wing_method_proof_runner",
    )


def build_mixer(args: argparse.Namespace):
    if not args.confirm_real_write:
        return ProofMixer(
            channel=args.channel,
            initial_fader_db=args.initial_fader_db,
            initial_gain_db=args.initial_gain_db,
            initial_eq_gain_db=args.initial_eq_gain_db,
        ), "fake_mixer"

    from wing_client import WingClient

    mixer = WingClient(ip=args.wing_ip, port=args.wing_port)
    if not mixer.connect(timeout=10.0):
        raise RuntimeError(f"WING connection failed: {args.wing_ip}:{args.wing_port}")
    return mixer, "physical_wing"


def build_executor(mixer: Any, artifact_path: Path) -> GatedCorrectionActionExecutor:
    promoted = {
        "ChannelGainMove": True,
        "ChannelFaderMove": True,
        "EmergencyFeedbackNotch": True,
        "ChannelEQMove": False,
    }
    config = CorrectionActionExecutorConfig.from_config(
        {
            "profile_id": "show_assist_auto_v1",
            "artifact_path": str(artifact_path),
            "promoted_action_types": promoted,
            "demote_on_first_failure": True,
        }
    )
    safety = AutoFOHSafetyController(
        mixer,
        config=AutoFOHSafetyConfig(
            channel_fader_max_step_db=1.0,
            channel_fader_min_interval_sec=0.0,
            gain_min_interval_sec=0.0,
            broad_eq_min_interval_sec=0.0,
        ),
    )
    return GatedCorrectionActionExecutor(safety, config=config)


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    run_id = f"{compact_stamp()}_{args.action}_ch{args.channel}"
    run_dir = Path(args.artifact_root) / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    summary_path = run_dir / "summary.json"
    event_path = run_dir / "gated_events.jsonl"
    summary: Dict[str, Any] = {
        "run_id": run_id,
        "started_at": utc_now(),
        "action": args.action,
        "channel": args.channel,
        "physical_wing": bool(args.confirm_real_write),
        "execute": bool(args.execute),
        "proof_run": bool(args.proof_run),
        "rollback_check": bool(args.rollback_check),
        "artifact_dir": str(run_dir),
        "event_path": str(event_path),
    }

    validation_error = validate_args(args)
    if validation_error:
        summary.update({"status": "blocked", "reason": validation_error})
        summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        print(json.dumps(summary, indent=2, sort_keys=True))
        return 2

    mixer = None
    try:
        mixer, mixer_mode = build_mixer(args)
        executor = build_executor(mixer, event_path)
        action = build_action(args)
        result = executor.execute(
            action,
            RuntimeState.SOURCE_LEARNING,
            dry_run=not args.execute,
            proof_run=bool(args.proof_run),
            idempotency_key=run_id,
        )
        rollback_result = None
        if args.rollback_check and result.rollback_action is not None and result.sent:
            rollback_result = executor.execute(
                result.rollback_action,
                RuntimeState.ROLLBACK,
                proof_run=True,
                idempotency_key=f"{run_id}:rollback",
            )

        summary.update(
            {
                "status": "passed" if (result.sent or not args.execute) else "failed",
                "mixer_mode": mixer_mode,
                "result": result.to_dict(),
                "rollback_result": rollback_result.to_dict() if rollback_result else None,
                "mixer_calls": getattr(mixer, "calls", []),
                "finished_at": utc_now(),
            }
        )
        summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        print(json.dumps(summary, indent=2, sort_keys=True))
        return 0 if summary["status"] == "passed" else 1
    except Exception as exc:
        summary.update({"status": "error", "error": str(exc), "finished_at": utc_now()})
        summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        print(json.dumps(summary, indent=2, sort_keys=True))
        return 1
    finally:
        if mixer is not None and hasattr(mixer, "disconnect"):
            try:
                mixer.disconnect()
            except Exception:
                pass


if __name__ == "__main__":
    raise SystemExit(main())
