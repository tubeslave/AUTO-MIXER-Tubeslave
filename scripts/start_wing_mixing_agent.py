#!/usr/bin/env python3
"""
Connect to AutoMixer WebSocket, attach WING, scan channel names, start MixingAgent.

Usage (from repo root):
  PYTHONPATH=backend python scripts/start_wing_mixing_agent.py
  PYTHONPATH=backend python scripts/start_wing_mixing_agent.py --ip 192.168.1.50 --mode auto

Requires: backend server running (python backend/server.py).
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys


def _repo_root() -> str:
    return os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))


def _load_config() -> dict:
    path = os.path.join(_repo_root(), "config", "default_config.json")
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _find_dante_input_device() -> int | None:
    try:
        import sounddevice as sd
    except ImportError:
        return None
    try:
        for idx, dev in enumerate(sd.query_devices()):
            if dev.get("max_input_channels", 0) <= 0:
                continue
            name = dev.get("name", "").lower()
            if "dante" in name or "dvs" in name or "virtual soundcard" in name:
                return idx
    except OSError:
        return None
    return None


async def main_async(args: argparse.Namespace) -> int:
    try:
        import websockets
    except ImportError:
        print("Install websockets: pip install websockets", file=sys.stderr)
        return 1

    cfg = _load_config()
    srv = cfg.get("server", {})
    host = srv.get("websocket_host", "localhost")
    port = int(srv.get("websocket_port", 8765))
    uri = f"ws://{host}:{port}"

    wing = cfg.get("wing", {})
    ip = args.ip or wing.get("default_ip") or cfg.get("mixer", {}).get("ip")
    send_port = int(args.send_port or wing.get("send_port", 2223))
    recv_port = int(args.receive_port or wing.get("receive_port", 2223))

    device_id = args.device_id
    if device_id is None and args.auto_device:
        device_id = _find_dante_input_device()
        if device_id is not None:
            print(f"Using audio input device index {device_id} (Dante-like)")

    channels = list(range(1, int(args.max_channel) + 1))
    mapping = {str(c): c for c in channels}

    print(f"Connecting WebSocket {uri} …")
    try:
        ws = await asyncio.wait_for(websockets.connect(uri), timeout=10.0)
    except Exception as e:
        print(f"WebSocket failed: {e}", file=sys.stderr)
        print("Start server: cd backend && python server.py", file=sys.stderr)
        return 1

    async def send(msg: dict) -> None:
        await ws.send(json.dumps(msg))

    await send(
        {
            "type": "connect_wing",
            "ip": ip,
            "send_port": send_port,
            "receive_port": recv_port,
        }
    )

    connected = False
    scan_done = False
    agent_started = False

    try:
        for _ in range(200):
            raw = await asyncio.wait_for(ws.recv(), timeout=30.0)
            data = json.loads(raw)
            t = data.get("type")
            if t == "connection_status":
                connected = bool(data.get("connected"))
                if not connected and data.get("error"):
                    print(f"WING connection failed: {data.get('error')}", file=sys.stderr)
                    return 2
                if connected:
                    print(f"WING connected ({ip})")
                    await send({"type": "scan_channel_names", "channels": channels})
            elif t == "channel_scan_result":
                if data.get("error"):
                    print(f"Scan error: {data.get('error')}", file=sys.stderr)
                    return 3
                scan_done = True
                results = data.get("results") or {}
                print("Channel scan (name → preset):")
                for ch in sorted(results.keys(), key=int):
                    r = results[ch]
                    print(
                        f"  ch{ch}: {r.get('name')} → "
                        f"{r.get('preset') or '?'} "
                        f"({'ok' if r.get('recognized') else 'unrecognized'})"
                    )
                payload = {
                    "type": "start_mixing_agent",
                    "mode": args.mode,
                    "kb_first": True,
                    "channels": channels,
                    "channel_mapping": mapping,
                }
                if device_id is not None:
                    payload["device_id"] = device_id
                    payload["num_audio_channels"] = min(64, int(args.max_channel) + 4)
                await send(payload)
            elif t == "mixing_agent_started":
                agent_started = True
                print("MixingAgent started:", json.dumps(data.get("status"), indent=2))
                break
            elif t == "error":
                print(f"Server error: {data.get('error')}", file=sys.stderr)
                return 4

        if not agent_started:
            print("Timeout waiting for mixing_agent_started", file=sys.stderr)
            return 5

        print(
            f"\nAgent mode={args.mode} kb_first=True. "
            "SUGGEST: approve via UI/WS mixing_agent_approve; "
            "AUTO: corrections apply within safety.enable_limits."
        )
        return 0
    finally:
        await ws.close()


def main() -> None:
    p = argparse.ArgumentParser(description="Start WING + MixingAgent via WebSocket")
    p.add_argument("--ip", default=None, help="WING IP (default: config wing.default_ip)")
    p.add_argument("--send-port", default=None, type=int)
    p.add_argument("--receive-port", default=None, type=int)
    p.add_argument("--max-channel", type=int, default=40)
    p.add_argument(
        "--mode",
        choices=("suggest", "auto", "manual"),
        default="auto",
        help="Agent mode (default: auto applies actions on console)",
    )
    p.add_argument(
        "--device-id",
        default=None,
        help="sounddevice input index for Dante/multichannel (optional)",
    )
    p.add_argument(
        "--no-audio",
        action="store_true",
        help="Do not open audio capture (OSC-only levels for agent)",
    )
    p.add_argument(
        "--no-auto-device",
        action="store_true",
        help="Disable auto Dante device detection",
    )
    args = p.parse_args()
    if args.no_audio:
        args.device_id = None
        auto_dev = False
    else:
        auto_dev = not args.no_auto_device
    ns = argparse.Namespace(**{**vars(args), "auto_device": auto_dev})
    sys.exit(asyncio.run(main_async(ns)))


if __name__ == "__main__":
    main()
