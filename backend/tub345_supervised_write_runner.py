#!/usr/bin/env python3
"""
Operator-run harness for the WING multi-step supervised write test (TUB-350).

This script drives the existing backend WebSocket API through the approved
supervised-write sequence and persists a local transcript for the issue report.
It intentionally refuses to send live writes unless --confirm-real-write is set.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
import time
from pathlib import Path
from typing import Any, Iterable

import websockets


DEFAULT_WS_URL = "ws://localhost:8765"
DEFAULT_REASON = "TUB-350 supervised rack test"
DEFAULT_SESSION_PREFIX = "tub350_supervised_live"
COOLDOWN_WAIT_BUFFER_SEC = 0.25
IGNORED_MESSAGE_TYPES = {"mixer_update", "state_update"}


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run a multi-step supervised WING fader/gain runner over the backend WebSocket API."
    )
    parser.add_argument("--ws-url", default=DEFAULT_WS_URL, help="Backend WebSocket URL")
    parser.add_argument("--mixer-ip", required=True, help="WING Rack IP address")
    parser.add_argument("--channel", type=int, required=True, help="Safe test channel number")
    parser.add_argument(
        "--fader-target-db",
        type=float,
        required=True,
        help="Absolute target fader value in dB for the first supervised fader write",
    )
    parser.add_argument(
        "--gain-target-db",
        type=float,
        required=True,
        help="Absolute target trim value in dB for the supervised gain write",
    )
    parser.add_argument(
        "--cooldown-fader-target-db",
        type=float,
        help="Optional second fader target used for the cooldown check. Defaults to fader-target-db - 0.5 dB.",
    )
    parser.add_argument(
        "--session-label",
        help="Telemetry session label. Defaults to tub346_supervised_live_<timestamp>.",
    )
    parser.add_argument(
        "--connect-timeout",
        type=float,
        default=15.0,
        help="Seconds to wait for the initial connect_wing response",
    )
    parser.add_argument(
        "--step-timeout",
        type=float,
        default=10.0,
        help="Seconds to wait for each non-connect response",
    )
    parser.add_argument(
        "--artifacts-dir",
        default="artifacts/tub345_supervised_live",
        help="Directory where the local transcript/report JSON will be written",
    )
    parser.add_argument(
        "--confirm-real-write",
        action="store_true",
        help="Required flag. Without it, the script exits before sending any live write-capable request.",
    )
    return parser


def should_ignore_message(message_type: str | None) -> bool:
    return message_type in IGNORED_MESSAGE_TYPES


def make_session_label(raw_label: str | None) -> str:
    if raw_label:
        return raw_label
    return f"{DEFAULT_SESSION_PREFIX}_{time.strftime('%Y%m%d_%H%M%S')}"


def make_artifact_path(base_dir: str | Path, session_label: str) -> Path:
    safe_label = "".join(ch if ch.isalnum() or ch in {"-", "_"} else "_" for ch in session_label)
    return Path(base_dir) / f"{safe_label}_summary.json"


def require_real_write_confirmation(args: argparse.Namespace) -> None:
    if args.confirm_real_write:
        return
    print(
        "Refusing to continue without --confirm-real-write.\n"
        "This sequence includes real supervised WING writes and an emergency stop.",
        file=sys.stderr,
    )
    raise SystemExit(2)


def summarize_manual_result(payload: dict[str, Any]) -> dict[str, Any]:
    request = payload.get("request", {})
    return {
        "message_type": payload.get("type"),
        "status": payload.get("status", request.get("status")),
        "reason": payload.get("reason", request.get("reason")),
        "channel": request.get("channel"),
        "value_type": payload.get("value_type", request.get("value_type")),
        "effective_value": request.get("effective_value"),
        "readback_value": request.get("readback_value"),
        "old_value": request.get("old_value"),
        "approval_id": request.get("approval_id"),
        "rollback": request.get("rollback"),
        "write_gate_state": request.get("write_gate_state"),
    }


def summarize_rollback_result(payload: dict[str, Any]) -> dict[str, Any]:
    rolled_back_steps = payload.get("rolled_back_steps") or []
    first_step = rolled_back_steps[0] if rolled_back_steps else {}
    original_step = first_step.get("original_step", {})
    rollback_result = first_step.get("rollback_result", {})
    return {
        "message_type": payload.get("type"),
        "status": payload.get("status"),
        "reason": payload.get("reason"),
        "mode": payload.get("mode"),
        "rolled_back_count": payload.get("rolled_back_count", 0),
        "remaining_step_count": payload.get("remaining_step_count", 0),
        "channel": rollback_result.get("channel", original_step.get("channel")),
        "value_type": rollback_result.get("value_type", original_step.get("value_type")),
        "effective_value": rollback_result.get("effective_value"),
        "readback_value": rollback_result.get("readback_value"),
        "restore_target_value": original_step.get("old_value"),
    }


def summarize_sequence_step(payload: dict[str, Any], *, label: str) -> dict[str, Any]:
    request = payload.get("request", {})
    return {
        "label": label,
        "message_type": payload.get("type"),
        "status": payload.get("status", request.get("status")),
        "reason": payload.get("reason", request.get("reason")),
        "value_type": payload.get("value_type", request.get("value_type")),
        "channel": request.get("channel"),
        "approval_id": request.get("approval_id"),
        "requested_value": request.get("requested_value"),
        "effective_value": request.get("effective_value"),
        "old_value": request.get("old_value"),
        "readback_value": request.get("readback_value"),
        "rollback": request.get("rollback"),
    }


def build_run_summary(
    *,
    args: argparse.Namespace,
    session_label: str,
    connect_payload: dict[str, Any],
    supervision_before: dict[str, Any],
    supervision_after_arm: dict[str, Any],
    session_before_start: dict[str, Any],
    session_after_start: dict[str, Any],
    steps: list[dict[str, Any]],
    negative_control: dict[str, Any],
    unsupported_control: dict[str, Any],
    unsupported_compressor: dict[str, Any] | None = None,
    positive_fader_1: dict[str, Any],
    positive_gain_1: dict[str, Any],
    positive_fader_2: dict[str, Any],
    cooldown_check: dict[str, Any],
    post_cooldown_check: dict[str, Any],
    rollback_last: dict[str, Any],
    rollback_all: dict[str, Any],
    session_after_rollback_last: dict[str, Any],
    session_after_rollback_all: dict[str, Any],
    emergency_stop: dict[str, Any],
    blocked_after_stop: dict[str, Any],
    blocked_after_emergency_mid: dict[str, Any] | None = None,
    telemetry_stop: dict[str, Any],
    transcript_path: Path,
) -> dict[str, Any]:
    return {
        "issue": "TUB-350",
        "runner": "TUB-350",
        "session_label": session_label,
        "safe_channel": args.channel,
        "requested_targets": {
            "fader_target_db": args.fader_target_db,
            "gain_target_db": args.gain_target_db,
            "cooldown_fader_target_db": args.cooldown_fader_target_db,
        },
        "connection_status": {
            "connected": connect_payload.get("connected"),
            "mode": connect_payload.get("mode"),
            "ip": connect_payload.get("ip"),
            "observation_only": connect_payload.get("observation_only"),
            "telemetry_capture": connect_payload.get("telemetry_capture"),
        },
        "manual_write_supervision": {
            "before_arm": supervision_before,
            "after_arm": supervision_after_arm,
        },
        "manual_write_session": {
            "before_start": session_before_start,
            "after_start": session_after_start,
            "after_rollback_last": session_after_rollback_last,
            "after_rollback_all": session_after_rollback_all,
        },
        "sequence_steps": [
            {
                **summarize_sequence_step(step.get("response", {}), label=step.get("step_label", "step")),
                "request": step.get("request", {}),
            }
            for step in steps
        ],
        "results": {
            "negative_control": summarize_manual_result(negative_control),
            "unsupported_control": summarize_manual_result(unsupported_control),
            "unsupported_compressor": summarize_manual_result(unsupported_compressor or {}),
            "positive_fader_1": summarize_manual_result(positive_fader_1),
            "positive_gain_1": summarize_manual_result(positive_gain_1),
            "positive_fader_2": summarize_manual_result(positive_fader_2),
            "cooldown_check": summarize_manual_result(cooldown_check),
            "post_cooldown_check": summarize_manual_result(post_cooldown_check),
            "rollback_last": summarize_rollback_result(rollback_last),
            "rollback_all": summarize_rollback_result(rollback_all),
            "emergency_stop": emergency_stop,
            "blocked_after_stop": summarize_manual_result(blocked_after_stop),
            "blocked_after_emergency_mid": summarize_manual_result(blocked_after_emergency_mid or {}),
        },
        "telemetry_capture_stop": telemetry_stop.get("telemetry_capture", {}),
        "local_transcript_path": str(transcript_path),
    }


async def recv_until(
    websocket: Any,
    *,
    expected_types: Iterable[str],
    timeout_sec: float,
    transcript: list[dict[str, Any]],
) -> dict[str, Any]:
    expected = set(expected_types)
    deadline = time.monotonic() + timeout_sec
    while True:
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise TimeoutError(f"Timed out waiting for messages: {sorted(expected)}")
        raw = await asyncio.wait_for(websocket.recv(), timeout=remaining)
        payload = json.loads(raw)
        transcript.append({"direction": "recv", "payload": payload})
        msg_type = payload.get("type")
        if should_ignore_message(msg_type):
            continue
        if msg_type in expected:
            return payload


async def send_and_expect(
    websocket: Any,
    request: dict[str, Any],
    *,
    expected_types: Iterable[str],
    timeout_sec: float,
    transcript: list[dict[str, Any]],
) -> dict[str, Any]:
    transcript.append({"direction": "send", "payload": request})
    await websocket.send(json.dumps(request))
    return await recv_until(
        websocket,
        expected_types=expected_types,
        timeout_sec=timeout_sec,
        transcript=transcript,
    )


async def wait_for_cooldown(
    *,
    supervision_payload: dict[str, Any],
    transcript: list[dict[str, Any]],
    label: str,
) -> None:
    cooldown_sec = max(0.0, float(supervision_payload.get("cooldown_sec", 0.0) or 0.0))
    if cooldown_sec <= 0:
        return
    sleep_sec = cooldown_sec + COOLDOWN_WAIT_BUFFER_SEC
    transcript.append(
        {
            "direction": "local",
            "payload": {
                "type": "cooldown_wait",
                "label": label,
                "sleep_sec": sleep_sec,
                "configured_cooldown_sec": cooldown_sec,
            },
        }
    )
    await asyncio.sleep(sleep_sec)


def append_step(
    transcript: list[dict[str, Any]],
    *,
    label: str,
    request: dict[str, Any],
    response: dict[str, Any],
) -> dict[str, Any]:
    step_record = {
        "step_label": label,
        "request": request,
        "response": response,
    }
    transcript.append({"direction": "step", "payload": step_record})
    return step_record


def extract_request(payload: dict[str, Any]) -> dict[str, Any]:
    request = payload.get("request")
    if not isinstance(request, dict):
        raise RuntimeError(f"Expected payload.request to be an object, got: {payload}")
    return request


def numeric_values_match(left: Any, right: Any, *, tolerance: float = 0.01) -> bool:
    try:
        return abs(float(left) - float(right)) <= tolerance
    except (TypeError, ValueError):
        return left == right


def assert_condition(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


async def run_sequence(args: argparse.Namespace) -> tuple[dict[str, Any], list[dict[str, Any]], Path]:
    require_real_write_confirmation(args)
    session_label = make_session_label(args.session_label)
    artifact_path = make_artifact_path(args.artifacts_dir, session_label)
    transcript: list[dict[str, Any]] = []
    sequence_steps: list[dict[str, Any]] = []
    cooldown_target = (
        args.cooldown_fader_target_db
        if args.cooldown_fader_target_db is not None
        else args.fader_target_db - 0.5
    )
    secondary_fader_target = args.fader_target_db - 0.25

    async with websockets.connect(args.ws_url) as websocket:
        connect_payload = await send_and_expect(
            websocket,
            {
                "type": "connect_wing",
                "ip": args.mixer_ip,
                "send_port": 2223,
                "receive_port": 2223,
                "observation_only": False,
                "record_telemetry": True,
                "telemetry_session_label": session_label,
            },
            expected_types={"connection_status"},
            timeout_sec=args.connect_timeout,
            transcript=transcript,
        )
        assert_condition(connect_payload.get("connected") is True, f"connect_wing failed: {connect_payload}")
        assert_condition(connect_payload.get("observation_only") is False, "connect_wing must stay write-enabled for this supervised test")

        supervision_before = await send_and_expect(
            websocket,
            {"type": "get_manual_write_supervision"},
            expected_types={"manual_write_supervision"},
            timeout_sec=args.step_timeout,
            transcript=transcript,
        )
        assert_condition(supervision_before.get("armed") is False, "Expected manual writes to start disarmed")

        session_before_start = await send_and_expect(
            websocket,
            {"type": "get_manual_write_session"},
            expected_types={"manual_write_session"},
            timeout_sec=args.step_timeout,
            transcript=transcript,
        )
        assert_condition(session_before_start.get("active") is False, "Expected manual write session to start inactive")

        supervision_after_arm = await send_and_expect(
            websocket,
            {
                "type": "set_manual_write_supervision",
                "armed": True,
                "reason": DEFAULT_REASON,
            },
            expected_types={"manual_write_supervision"},
            timeout_sec=args.step_timeout,
            transcript=transcript,
        )
        assert_condition(supervision_after_arm.get("armed") is True, "Manual supervision did not arm")

        session_after_start = await send_and_expect(
            websocket,
            {
                "type": "start_manual_write_session",
                "session_label": session_label,
                "reason": DEFAULT_REASON,
            },
            expected_types={"manual_write_session"},
            timeout_sec=args.step_timeout,
            transcript=transcript,
        )
        assert_condition(session_after_start.get("status") == "started", f"Session did not start: {session_after_start}")
        assert_condition(session_after_start.get("active") is True, "Manual write session did not become active")

        negative_control = await send_and_expect(
            websocket,
            {
                "type": "set_fader",
                "channel": args.channel,
                "value": args.fader_target_db,
                "approved": False,
                "reason": DEFAULT_REASON,
            },
            expected_types={"manual_write_blocked"},
            timeout_sec=args.step_timeout,
            transcript=transcript,
        )
        assert_condition(negative_control.get("reason") == "approval_required", f"Unexpected negative-control result: {negative_control}")
        sequence_steps.append(append_step(
            transcript,
            label="negative_control",
            request={"type": "set_fader", "approved": False, "reason": DEFAULT_REASON, "channel": args.channel, "value": args.fader_target_db},
            response=negative_control,
        ))

        unsupported_control = await send_and_expect(
            websocket,
            {
                "type": "set_eq",
                "channel": args.channel,
                "band": 1,
                "freq": 1000,
                "gain": 1.0,
                "q": 1.0,
            },
            expected_types={"manual_write_blocked"},
            timeout_sec=args.step_timeout,
            transcript=transcript,
        )
        assert_condition(
            unsupported_control.get("reason") == "unsupported_value_type",
            f"Unexpected unsupported-value result: {unsupported_control}",
        )
        sequence_steps.append(append_step(
            transcript,
            label="unsupported_control",
            request={"type": "set_eq", "channel": args.channel, "band": 1, "freq": 1000, "gain": 1.0, "q": 1.0},
            response=unsupported_control,
        ))

        unsupported_compressor = await send_and_expect(
            websocket,
            {
                "type": "set_compressor",
                "channel": args.channel,
                "params": {"threshold": -18.0, "ratio": 2.5, "attack": 5.0, "release": 120.0},
            },
            expected_types={"manual_write_blocked"},
            timeout_sec=args.step_timeout,
            transcript=transcript,
        )
        assert_condition(
            unsupported_compressor.get("reason") == "unsupported_value_type",
            f"Unexpected compressor unsupported-value result: {unsupported_compressor}",
        )
        sequence_steps.append(append_step(
            transcript,
            label="unsupported_compressor",
            request={"type": "set_compressor", "channel": args.channel, "params": {"threshold": -18.0, "ratio": 2.5, "attack": 5.0, "release": 120.0}},
            response=unsupported_compressor,
        ))

        positive_fader_1 = await send_and_expect(
            websocket,
            {
                "type": "set_fader",
                "channel": args.channel,
                "value": args.fader_target_db,
                "approved": True,
                "approval_id": "TUB-346-fader-1",
                "reason": DEFAULT_REASON,
            },
            expected_types={"manual_write_applied", "manual_write_error", "manual_write_blocked"},
            timeout_sec=args.step_timeout,
            transcript=transcript,
        )
        assert_condition(positive_fader_1.get("type") == "manual_write_applied", f"Fader write did not apply: {positive_fader_1}")
        positive_fader_1_request = extract_request(positive_fader_1)
        fader_1_old_value = positive_fader_1_request.get("old_value")
        assert_condition(fader_1_old_value is not None, f"Fader write did not report old_value: {positive_fader_1}")
        assert_condition(
            not numeric_values_match(fader_1_old_value, positive_fader_1_request.get("readback_value")),
            (
                "Fader target matched the existing console value, so the test cannot prove a real change. "
                "Choose a different safe fader target and rerun."
            ),
        )
        sequence_steps.append(append_step(
            transcript,
            label="positive_fader_1",
            request={"type": "set_fader", "approved": True, "reason": DEFAULT_REASON, "approval_id": "TUB-346-fader-1", "channel": args.channel, "value": args.fader_target_db},
            response=positive_fader_1,
        ))

        positive_gain_1 = await send_and_expect(
            websocket,
            {
                "type": "set_gain",
                "channel": args.channel,
                "value": args.gain_target_db,
                "approved": True,
                "approval_id": "TUB-346-gain-1",
                "reason": DEFAULT_REASON,
            },
            expected_types={"manual_write_applied", "manual_write_error", "manual_write_blocked"},
            timeout_sec=args.step_timeout,
            transcript=transcript,
        )
        assert_condition(positive_gain_1.get("type") == "manual_write_applied", f"Gain write did not apply: {positive_gain_1}")
        positive_gain_1_request = extract_request(positive_gain_1)
        gain_old_value = positive_gain_1_request.get("old_value")
        assert_condition(gain_old_value is not None, f"Gain write did not report old_value: {positive_gain_1}")
        assert_condition(
            not numeric_values_match(gain_old_value, positive_gain_1_request.get("readback_value")),
            (
                "Gain target matched the existing console value, so the test cannot prove a real change. "
                "Choose a different safe gain target and rerun."
            ),
        )
        sequence_steps.append(append_step(
            transcript,
            label="positive_gain_1",
            request={"type": "set_gain", "approved": True, "reason": DEFAULT_REASON, "approval_id": "TUB-346-gain-1", "channel": args.channel, "value": args.gain_target_db},
            response=positive_gain_1,
        ))

        positive_fader_2 = await send_and_expect(
            websocket,
            {
                "type": "set_fader",
                "channel": args.channel,
                "value": secondary_fader_target,
                "approved": True,
                "approval_id": "TUB-346-fader-2",
                "reason": DEFAULT_REASON,
            },
            expected_types={"manual_write_applied", "manual_write_error", "manual_write_blocked"},
            timeout_sec=args.step_timeout,
            transcript=transcript,
        )
        assert_condition(positive_fader_2.get("type") == "manual_write_applied", f"Fader secondary write did not apply: {positive_fader_2}")
        positive_fader_2_request = extract_request(positive_fader_2)
        fader_2_old_value = positive_fader_2_request.get("old_value")
        assert_condition(fader_2_old_value is not None, f"Secondary fader write did not report old_value: {positive_fader_2}")
        assert_condition(
            not numeric_values_match(fader_2_old_value, positive_fader_2_request.get("readback_value")),
            (
                "Secondary fader target matched the existing value, so the test cannot prove a real change. "
                "Choose a different safe fader target and rerun."
            ),
        )
        sequence_steps.append(append_step(
            transcript,
            label="positive_fader_2",
            request={"type": "set_fader", "approved": True, "reason": DEFAULT_REASON, "approval_id": "TUB-346-fader-2", "channel": args.channel, "value": secondary_fader_target},
            response=positive_fader_2,
        ))

        emergency_mid = await send_and_expect(
            websocket,
            {"type": "emergency_stop"},
            expected_types={"emergency_stop_applied"},
            timeout_sec=args.step_timeout,
            transcript=transcript,
        )
        sequence_steps.append(append_step(
            transcript,
            label="emergency_mid_sequence",
            request={"type": "emergency_stop"},
            response=emergency_mid,
        ))

        blocked_after_emergency_mid = await send_and_expect(
            websocket,
            {
                "type": "set_fader",
                "channel": args.channel,
                "value": cooldown_target,
                "approved": True,
                "approval_id": "TUB-346-fader-after-mid-stop",
                "reason": DEFAULT_REASON,
            },
            expected_types={"manual_write_blocked", "manual_write_applied", "manual_write_error"},
            timeout_sec=args.step_timeout,
            transcript=transcript,
        )
        assert_condition(
            blocked_after_emergency_mid.get("reason") == "manual_path_not_armed",
            f"Write after mid-sequence emergency stop was not blocked: {blocked_after_emergency_mid}",
        )
        sequence_steps.append(append_step(
            transcript,
            label="blocked_after_emergency_mid",
            request={"type": "set_fader", "channel": args.channel, "value": cooldown_target, "approved": True, "approval_id": "TUB-346-fader-after-mid-stop", "reason": DEFAULT_REASON},
            response=blocked_after_emergency_mid,
        ))

        reamed_mid = await send_and_expect(
            websocket,
            {
                "type": "set_manual_write_supervision",
                "armed": True,
                "reason": "rearm_after_mid_emergency",
            },
            expected_types={"manual_write_supervision"},
            timeout_sec=args.step_timeout,
            transcript=transcript,
        )
        assert_condition(reamed_mid.get("armed") is True, "Manual write supervision did not re-arm after emergency")
        sequence_steps.append(append_step(
            transcript,
            label="rearmed_after_emergency_mid",
            request={"type": "set_manual_write_supervision", "armed": True, "reason": "rearm_after_mid_emergency"},
            response=reamed_mid,
        ))

        cooldown_check = await send_and_expect(
            websocket,
            {
                "type": "set_fader",
                "channel": args.channel,
                "value": cooldown_target,
                "approved": True,
                "approval_id": "TUB-346-fader-cooldown-check",
                "reason": DEFAULT_REASON,
            },
            expected_types={"manual_write_blocked", "manual_write_applied", "manual_write_error"},
            timeout_sec=args.step_timeout,
            transcript=transcript,
        )
        assert_condition(cooldown_check.get("reason") == "cooldown", f"Cooldown check did not block as expected: {cooldown_check}")
        sequence_steps.append(append_step(
            transcript,
            label="cooldown_check",
            request={"type": "set_fader", "approved": True, "reason": DEFAULT_REASON, "approval_id": "TUB-346-fader-cooldown-check", "channel": args.channel, "value": cooldown_target},
            response=cooldown_check,
        ))

        await wait_for_cooldown(
            supervision_payload=supervision_after_arm,
            transcript=transcript,
            label="post_cooldown_recovery",
        )

        post_cooldown_check = await send_and_expect(
            websocket,
            {
                "type": "set_fader",
                "channel": args.channel,
                "value": cooldown_target,
                "approved": True,
                "approval_id": "TUB-350-fader-post-cooldown",
                "reason": DEFAULT_REASON,
            },
            expected_types={"manual_write_applied", "manual_write_error", "manual_write_blocked"},
            timeout_sec=args.step_timeout,
            transcript=transcript,
        )
        assert_condition(post_cooldown_check.get("type") == "manual_write_applied", f"Post-cooldown write did not apply: {post_cooldown_check}")
        sequence_steps.append(append_step(
            transcript,
            label="post_cooldown_check",
            request={"type": "set_fader", "approved": True, "reason": DEFAULT_REASON, "approval_id": "TUB-350-fader-post-cooldown", "channel": args.channel, "value": cooldown_target},
            response=post_cooldown_check,
        ))

        rollback_last = await send_and_expect(
            websocket,
            {
                "type": "rollback_last",
                "approved": True,
                "approval_id": "TUB-350-rollback-last",
                "reason": f"{DEFAULT_REASON} rollback last",
            },
            expected_types={
                "manual_write_rollback_applied",
                "manual_write_rollback_error",
                "manual_write_rollback_blocked",
            },
            timeout_sec=args.step_timeout,
            transcript=transcript,
        )
        assert_condition(rollback_last.get("type") == "manual_write_rollback_applied", f"Last-step rollback did not apply: {rollback_last}")
        assert_condition(rollback_last.get("rolled_back_count") == 1, f"Expected exactly one rollback step: {rollback_last}")
        sequence_steps.append(append_step(
            transcript,
            label="rollback_last",
            request={"type": "rollback_last", "approved": True, "reason": f"{DEFAULT_REASON} rollback last", "approval_id": "TUB-350-rollback-last"},
            response=rollback_last,
        ))

        session_after_rollback_last = await send_and_expect(
            websocket,
            {"type": "get_manual_write_session"},
            expected_types={"manual_write_session"},
            timeout_sec=args.step_timeout,
            transcript=transcript,
        )
        assert_condition(session_after_rollback_last.get("step_count") == 3, f"Expected three pending steps after rollback_last: {session_after_rollback_last}")

        rollback_all = await send_and_expect(
            websocket,
            {
                "type": "rollback_all",
                "approved": True,
                "approval_id": "TUB-350-rollback-all",
                "reason": f"{DEFAULT_REASON} rollback all",
            },
            expected_types={
                "manual_write_rollback_applied",
                "manual_write_rollback_error",
                "manual_write_rollback_blocked",
            },
            timeout_sec=args.step_timeout,
            transcript=transcript,
        )
        assert_condition(
            rollback_all.get("type") == "manual_write_rollback_applied",
            f"Rollback-all did not apply: {rollback_all}",
        )
        assert_condition(rollback_all.get("rolled_back_count") == 3, f"Expected three rollback-all steps: {rollback_all}")
        sequence_steps.append(append_step(
            transcript,
            label="rollback_all",
            request={"type": "rollback_all", "approved": True, "reason": f"{DEFAULT_REASON} rollback all", "approval_id": "TUB-350-rollback-all"},
            response=rollback_all,
        ))

        session_after_rollback_all = await send_and_expect(
            websocket,
            {"type": "get_manual_write_session"},
            expected_types={"manual_write_session"},
            timeout_sec=args.step_timeout,
            transcript=transcript,
        )
        assert_condition(session_after_rollback_all.get("active") is False, f"Session should be inactive after rollback_all: {session_after_rollback_all}")
        assert_condition(session_after_rollback_all.get("step_count") == 0, f"Session should be empty after rollback_all: {session_after_rollback_all}")

        emergency_stop = await send_and_expect(
            websocket,
            {"type": "emergency_stop"},
            expected_types={"emergency_stop_applied"},
            timeout_sec=args.step_timeout,
            transcript=transcript,
        )
        sequence_steps.append(append_step(
            transcript,
            label="emergency_stop_final",
            request={"type": "emergency_stop"},
            response=emergency_stop,
        ))

        blocked_after_stop = await send_and_expect(
            websocket,
            {
                "type": "set_fader",
                "channel": args.channel,
                "value": args.fader_target_db,
                "approved": True,
                "approval_id": "TUB-350-after-final-stop",
                "reason": DEFAULT_REASON,
            },
            expected_types={"manual_write_blocked", "manual_write_applied", "manual_write_error"},
            timeout_sec=args.step_timeout,
            transcript=transcript,
        )
        assert_condition(
            blocked_after_stop.get("reason") == "manual_path_not_armed",
            f"Write after final emergency stop was not blocked: {blocked_after_stop}",
        )
        sequence_steps.append(append_step(
            transcript,
            label="blocked_after_stop",
            request={"type": "set_fader", "channel": args.channel, "value": args.fader_target_db, "approved": True, "approval_id": "TUB-350-after-final-stop", "reason": DEFAULT_REASON},
            response=blocked_after_stop,
        ))

        telemetry_stop = await send_and_expect(
            websocket,
            {"type": "stop_wing_telemetry_capture", "reason": "tub345_sequence_complete"},
            expected_types={"wing_telemetry_capture_status"},
            timeout_sec=args.step_timeout,
            transcript=transcript,
        )

    summary = build_run_summary(
        args=args,
        session_label=session_label,
        connect_payload=connect_payload,
        supervision_before=supervision_before,
        supervision_after_arm=supervision_after_arm,
        session_before_start=session_before_start,
        session_after_start=session_after_start,
        steps=sequence_steps,
        negative_control=negative_control,
        unsupported_control=unsupported_control,
        unsupported_compressor=unsupported_compressor,
        positive_fader_1=positive_fader_1,
        positive_gain_1=positive_gain_1,
        positive_fader_2=positive_fader_2,
        cooldown_check=cooldown_check,
        post_cooldown_check=post_cooldown_check,
        rollback_last=rollback_last,
        rollback_all=rollback_all,
        session_after_rollback_last=session_after_rollback_last,
        session_after_rollback_all=session_after_rollback_all,
        emergency_stop=emergency_stop,
        blocked_after_stop=blocked_after_stop,
        blocked_after_emergency_mid=blocked_after_emergency_mid,
        telemetry_stop=telemetry_stop,
        transcript_path=artifact_path,
    )
    return summary, transcript, artifact_path


def write_summary(path: Path, summary: dict[str, Any], transcript: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "summary": summary,
        "transcript": transcript,
    }
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


async def async_main(args: argparse.Namespace) -> int:
    summary, transcript, artifact_path = await run_sequence(args)
    write_summary(artifact_path, summary, transcript)
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return 0


def main() -> int:
    parser = build_arg_parser()
    args = parser.parse_args()
    try:
        return asyncio.run(async_main(args))
    except KeyboardInterrupt:
        print("Interrupted by operator", file=sys.stderr)
        return 130
    except Exception as exc:
        print(f"TUB-350 supervised write run failed: {exc} (issue=TUB-350)", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
