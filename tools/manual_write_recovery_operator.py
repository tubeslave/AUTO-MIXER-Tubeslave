#!/usr/bin/env python3
"""Operator recovery tool for manual-write supervision: WAL reconcile + emergency-stop reset.

Speaks the AUTO-MIXER backend WebSocket control plane (default ws://127.0.0.1:8765):

  status     read-only supervision snapshot (get_manual_write_supervision).
             Safe: never sends control commands, works without a token.
  reconcile  close unresolved WAL transactions from fresh physical console
             readback (reconcile_console_write_recovery). Requires a live WING
             observation connection on the backend and an operator approval id.
  reset      clear the manual-write emergency stop latch
             (reset_manual_write_emergency_stop). Blocked until WAL is clean.

reconcile/reset are authenticated control commands: the backend must be
started with AUTOMIXER_WS_AUTH_TOKEN (43-char base64url). Without a token the
backend answers control_authentication_required and stays observation-only.

Exit codes: 0 = success / nothing to do, 1 = transport or usage error,
2 = backend blocked or refused (see printed reason + hint).
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import re
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import websockets
from websockets.exceptions import WebSocketException

DEFAULT_WS_URL = "ws://127.0.0.1:8765"
DEFAULT_TIMEOUT_SEC = 30.0
DEFAULT_RESET_REASON = "operator_requested_emergency_reset"
IGNORED_BROADCAST_TYPES = frozenset({"mixer_update", "state_update"})
TERMINAL_ERROR_TYPES = frozenset({"error", "control_authentication_required"})
APPROVAL_ID_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:-]{0,127}$")
MAX_IGNORED_MESSAGES = 1000

BLOCKED_REASON_HINTS = {
    "operator_approval_required": "передайте --approval-id (непустая аттестация оператора); approved=true выставляется скриптом",
    "authenticated_control_session_required": "backend запущен без AUTOMIXER_WS_AUTH_TOKEN: перезапустите backend с токеном и передайте --token",
    "unresolved_transaction_not_found": "transaction_id не из wal_unresolved_transaction_ids — выполните `status` и возьмите id оттуда",
    "recovery_preimage_or_target_missing": "в WAL-записи нет preimage/target — нужен ручной разбор WAL-файла",
    "physical_readback_transport_unavailable": "пульт не подключён: выполните observation-connect к WING (UI или runner) и повторите",
    "physical_recovery_read_failed": "физический readback не удался: проверьте OSC-связь с WING (L2-сеть, порт 2223)",
    "physical_recovery_read_unverified": "readback не подтверждён (нужны 2 подтверждённых round-trip): проверьте связь и повторите",
    "recovery_state_mixed_or_changed": "значение на пульте не совпадает ни с target, ни с preimage: решите ВРУЧНУЮ, какое состояние верное, выставьте его на пульте и повторите",
    "restore_only_proof_target_cannot_be_reconciled_committed": "restore-only proof нельзя закрыть как committed: ожидается откат к preimage",
    "recovery_terminal_record_failed": "не удалось дописать терминальную запись в WAL — проверьте доступность файла WAL",
    "console_write_recovery_pending": "сначала выполните reconcile всех незакрытых транзакций",
    "emergency_stop_teardown_in_progress": "teardown ещё идёт — подождите несколько секунд и повторите",
    "emergency_stop_teardown_incomplete": "teardown завершился с ошибками — смотрите teardown_errors/latch_errors в ответе",
}


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def build_authenticate(token: str, client_id: str) -> dict[str, Any]:
    return {"type": "authenticate", "token": token, "client_id": client_id}


def build_supervision_request() -> dict[str, Any]:
    return {"type": "get_manual_write_supervision"}


def build_reconcile_request(transaction_id: str, approval_id: str) -> dict[str, Any]:
    return {
        "type": "reconcile_console_write_recovery",
        "transaction_id": transaction_id,
        "approved": True,
        "approval_id": approval_id,
    }


def build_reset_request(approval_id: str, reason: str) -> dict[str, Any]:
    return {
        "type": "reset_manual_write_emergency_stop",
        "approved": True,
        "approval_id": approval_id,
        "reason": reason,
    }


async def recv_until(websocket, expected_types, timeout_sec):
    """Wait for a message of an expected type, skipping interleaved broadcasts.

    The backend has no request/response correlation id; clients correlate by
    response `type`. Terminal error envelopes are returned as responses too.
    """
    ignored = 0
    while True:
        raw = await asyncio.wait_for(websocket.recv(), timeout=timeout_sec)
        message = json.loads(raw)
        msg_type = message.get("type")
        if msg_type in expected_types or msg_type in TERMINAL_ERROR_TYPES:
            return message
        ignored += 1
        if ignored > MAX_IGNORED_MESSAGES:
            raise TimeoutError(
                f"no expected response {sorted(expected_types)} after {ignored} interleaved messages"
            )


async def send_and_expect(websocket, request, expected_types, timeout_sec, transcript):
    transcript.append({"ts": _utc_now_iso(), "direction": "out", "message": request})
    await websocket.send(json.dumps(request))
    response = await recv_until(websocket, expected_types, timeout_sec)
    transcript.append({"ts": _utc_now_iso(), "direction": "in", "message": response})
    return response


def _hint_for(reason: str) -> str:
    return BLOCKED_REASON_HINTS.get(
        reason,
        "неизвестная причина — сверьтесь с backend/handlers/automation_handlers.py и Docs/CURRENT_STATE.md",
    )


def _print_supervision(status: dict[str, Any]) -> None:
    print("=== manual_write_supervision_status ===")
    for key in (
        "emergency_stopped",
        "reset_required",
        "wal_recovery_required",
        "wal_unresolved_transaction_ids",
        "approved_live_write_kinds",
        "operator_mode",
        "requires_operator_mode",
        "connection_mode",
        "wing_observation_only",
        "write_ahead_log",
    ):
        if key in status:
            print(f"  {key}: {status[key]}")


def _print_readback(rows) -> None:
    for row in rows or []:
        print(
            "    readback {address}: actual={actual} target_match={tm} "
            "preimage_match={pm} round_trips={rt}".format(
                address=row.get("address"),
                actual=row.get("actual"),
                tm=row.get("target_match"),
                pm=row.get("preimage_match"),
                rt=row.get("confirmed_round_trips"),
            )
        )


async def _open_session(args, transcript):
    websocket = await websockets.connect(
        args.ws_url, open_timeout=10, close_timeout=5, max_size=1 << 20
    )
    token = args.token or os.environ.get("AUTOMIXER_WS_AUTH_TOKEN", "")
    if token:
        auth = await send_and_expect(
            websocket,
            build_authenticate(token, args.client_id),
            {"authentication_status"},
            args.timeout,
            transcript,
        )
        if auth.get("type") != "authentication_status" or not auth.get("authenticated"):
            await websocket.close()
            raise RuntimeError(f"authentication failed: {auth}")
    return websocket


async def _fetch_supervision(websocket, args, transcript):
    response = await send_and_expect(
        websocket,
        build_supervision_request(),
        {"manual_write_supervision_status"},
        args.timeout,
        transcript,
    )
    if response.get("type") != "manual_write_supervision_status":
        raise RuntimeError(f"unexpected supervision response: {response}")
    return response


def _validate_approval_id(args) -> bool:
    if APPROVAL_ID_PATTERN.match(args.approval_id or ""):
        return True
    print(
        "ERROR: --approval-id обязателен: латиница/цифры и . _ : -, до 128 символов "
        "(например operator-ivan-20260728)",
        file=sys.stderr,
    )
    return False


async def cmd_status(args, transcript) -> int:
    websocket = await _open_session(args, transcript)
    try:
        status = await _fetch_supervision(websocket, args, transcript)
    finally:
        await websocket.close()
    _print_supervision(status)
    unresolved = status.get("wal_unresolved_transaction_ids") or []
    if status.get("emergency_stopped") or unresolved:
        print(
            "\nСОСТОЯНИЕ: записи заблокированы — выполните reconcile "
            "(если есть незакрытые id), затем reset."
        )
    else:
        print("\nСОСТОЯНИЕ: emergency-stop не активен, WAL чист — рельс записей №1 снят.")
    return 0


async def cmd_reconcile(args, transcript) -> int:
    if not _validate_approval_id(args):
        return 1
    websocket = await _open_session(args, transcript)
    try:
        status = await _fetch_supervision(websocket, args, transcript)
        _print_supervision(status)
        unresolved = list(status.get("wal_unresolved_transaction_ids") or [])
        targets = list(args.transaction_id) if args.transaction_id else unresolved
        unknown = [t for t in targets if t not in unresolved]
        if unknown:
            print(f"ВНИМАНИЕ: id не числятся незакрытыми: {unknown}")
        if not targets:
            print("\nНезакрытых WAL-транзакций нет — reconcile не требуется.")
            return 0
        rc = 0
        for tid in targets:
            print(f"\n>>> reconcile {tid}")
            response = await send_and_expect(
                websocket,
                build_reconcile_request(tid, args.approval_id),
                {"console_write_recovery_result"},
                args.timeout,
                transcript,
            )
            if response.get("type") != "console_write_recovery_result":
                reason = str(response.get("reason") or response.get("error"))
                print(f"ОТКАЗ КОНТРОЛЯ: {reason}\n  подсказка: {_hint_for(reason)}")
                return 2
            if response.get("status") == "reconciled" and response.get("success") is True:
                print(
                    f"  OK: reason={response.get('reason')}  "
                    f"emergency_reset_still_required={response.get('emergency_reset_still_required')}"
                )
                _print_readback(response.get("physical_readback"))
                continue
            reason = str(response.get("reason"))
            print(f"  BLOCKED: {reason}\n  подсказка: {_hint_for(reason)}")
            _print_readback(response.get("physical_readback"))
            rc = 2
            break
        final = await _fetch_supervision(websocket, args, transcript)
        print("\n=== после reconcile ===")
        _print_supervision(final)
        return rc
    finally:
        await websocket.close()


async def cmd_reset(args, transcript) -> int:
    if not _validate_approval_id(args):
        return 1
    websocket = await _open_session(args, transcript)
    try:
        status = await _fetch_supervision(websocket, args, transcript)
        _print_supervision(status)
        response = await send_and_expect(
            websocket,
            build_reset_request(args.approval_id, args.reason),
            {"manual_write_emergency_reset_result"},
            args.timeout,
            transcript,
        )
        if response.get("type") != "manual_write_emergency_reset_result":
            reason = str(response.get("reason") or response.get("error"))
            print(f"ОТКАЗ КОНТРОЛЯ: {reason}\n  подсказка: {_hint_for(reason)}")
            return 2
        state = response.get("status")
        rc = 0
        if state == "not_required":
            print("\nemergency-stop не активен — reset не требуется.")
        elif state == "reset" and response.get("success") is True:
            print(f"\nRESET OK: reason={response.get('reason')} approval_id={response.get('approval_id')}")
            if response.get("automation_remains_frozen"):
                print(
                    "  внимание: автоматика остаётся замороженной "
                    "(automation_frozen_after_emergency_stop) — это отдельный следующий шаг."
                )
        else:
            reason = str(response.get("reason"))
            print(f"\nBLOCKED: {reason}\n  подсказка: {_hint_for(reason)}")
            rc = 2
        final = await _fetch_supervision(websocket, args, transcript)
        print("\n=== после reset ===")
        _print_supervision(final)
        return rc
    finally:
        await websocket.close()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="manual_write_recovery_operator",
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--ws-url",
        default=os.environ.get("AUTOMIXER_WS_URL", DEFAULT_WS_URL),
        help=f"WS-адрес backend (default {DEFAULT_WS_URL}; env AUTOMIXER_WS_URL)",
    )
    parser.add_argument(
        "--token",
        default="",
        help="43-символьный base64url токен управления (или env AUTOMIXER_WS_AUTH_TOKEN)",
    )
    parser.add_argument(
        "--client-id",
        default=os.environ.get("AUTOMIXER_WS_CLIENT_ID", "operator-recovery"),
        help="client_id для authenticate (default operator-recovery)",
    )
    parser.add_argument("--timeout", type=float, default=DEFAULT_TIMEOUT_SEC)
    parser.add_argument("--report", default="", help="путь для JSON-транскрипта сеанса")
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser(
        "status",
        help="read-only снимок supervision (безопасно: без контрольных команд, токен не нужен)",
    )
    rec = sub.add_parser(
        "reconcile",
        help="закрыть незакрытые WAL-транзакции по физическому readback (нужен живой пульт)",
    )
    rec.add_argument(
        "--transaction-id",
        action="append",
        default=[],
        help="конкретный transaction_id (повторяемый флаг); без флага — все незакрытые",
    )
    rec.add_argument("--approval-id", required=True, help="аттестация оператора, пишется в WAL")
    rst = sub.add_parser(
        "reset",
        help="снять защёлку manual-write emergency-stop (только после reconcile)",
    )
    rst.add_argument("--approval-id", required=True, help="аттестация оператора, пишется в WAL")
    rst.add_argument("--reason", default=DEFAULT_RESET_REASON)
    return parser


async def run_command(args) -> int:
    transcript: list = []
    commands = {"status": cmd_status, "reconcile": cmd_reconcile, "reset": cmd_reset}
    try:
        return await commands[args.command](args, transcript)
    except (RuntimeError, TimeoutError, OSError, WebSocketException) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    finally:
        if args.report:
            Path(args.report).write_text(
                json.dumps(
                    {
                        "command": args.command,
                        "generated_at": _utc_now_iso(),
                        "transcript": transcript,
                    },
                    ensure_ascii=False,
                    indent=2,
                )
            )
            print(f"транскрипт сохранён: {args.report}")


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    return asyncio.run(run_command(args))


if __name__ == "__main__":
    raise SystemExit(main())
