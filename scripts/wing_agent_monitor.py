#!/usr/bin/env python3
"""
Persistent WebSocket client: connect WING, scan channels, run MixingAgent AUTO,
show every action applied to the mixer in real time.

Usage (from repo root):
  PYTHONPATH=backend python scripts/wing_agent_monitor.py
  PYTHONPATH=backend python scripts/wing_agent_monitor.py --ip 192.168.1.102
  PYTHONPATH=backend python scripts/wing_agent_monitor.py --mode suggest   # display only

Press Ctrl+C to stop.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
import time
from datetime import datetime


def _root() -> str:
    return os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))


def _load_config() -> dict:
    with open(os.path.join(_root(), "config", "default_config.json"), encoding="utf-8") as f:
        return json.load(f)


def ts() -> str:
    return datetime.now().strftime("%H:%M:%S")


INSTRUMENT_LABELS = {
    "kick": "🥁 Kick",
    "snare": "🥁 Snare",
    "tom": "🥁 Tom",
    "hihat": "🥁 Hi-Hat",
    "ride": "🥁 Ride",
    "cymbals": "🥁 Cymbals",
    "overheads": "🎙 Overhead",
    "room": "🎙 Room",
    "bass": "🎸 Bass",
    "bass_guitar": "🎸 Bass",
    "electricGuitar": "🎸 Gtr",
    "electric_guitar": "🎸 Gtr",
    "acousticGuitar": "🎸 Ac. Gtr",
    "accordion": "🪗 Accordion",
    "synth": "🎹 Synth",
    "playback": "▶ Playback",
    "leadVocal": "🎤 Lead Vox",
    "lead_vocal": "🎤 Lead Vox",
    "backVocal": "🎤 Back Vox",
    "backing_vocal": "🎤 Back Vox",
}

ACTION_ICONS = {
    "reduce_gain": "↓Gain",
    "adjust_gain": "~Gain",
    "apply_hpf": "HPF",
    "adjust_compressor": "COMP",
    "mute_channel": "MUTE",
    "unmute_channel": "UNMUTE",
    "llm_recommendation": "LLM",
}


async def run(args: argparse.Namespace) -> None:
    try:
        import websockets
    except ImportError:
        print("pip install websockets", file=sys.stderr)
        return

    cfg = _load_config()
    srv = cfg.get("server", {})
    ws_host = srv.get("websocket_host", "localhost")
    ws_port = int(srv.get("websocket_port", 8765))
    uri = f"ws://{ws_host}:{ws_port}"

    wing = cfg.get("wing", {})
    ip = args.ip or wing.get("default_ip") or cfg.get("mixer", {}).get("ip")
    send_port = int(args.send_port or wing.get("send_port", 2223))
    recv_port = int(args.recv_port or wing.get("receive_port", 2223))

    channels = list(range(1, int(args.max_channel) + 1))
    channel_map: dict[int, str] = {}
    audio_device_id = int(args.device_id) if args.device_id is not None else 1

    print(f"[{ts()}] Connecting WebSocket {uri} …")
    try:
        ws = await asyncio.wait_for(websockets.connect(uri), timeout=10.0)
    except Exception as e:
        print(f"WebSocket failed: {e}\nStart server: cd backend && python server.py")
        return

    print(f"[{ts()}] WebSocket OK. Connecting WING {ip}:{send_port} …")

    async def send(msg: dict) -> None:
        await ws.send(json.dumps(msg))

    async def recv_with_timeout(t: float = 30.0) -> dict | None:
        try:
            raw = await asyncio.wait_for(ws.recv(), timeout=t)
            return json.loads(raw)
        except asyncio.TimeoutError:
            return None
        except Exception:
            return None

    # ── STEP 1: connect WING ──────────────────────────────────
    await send({"type": "connect_wing", "ip": ip, "send_port": send_port, "receive_port": recv_port})
    while True:
        d = await recv_with_timeout(30)
        if d is None:
            print("Timeout waiting for connection_status")
            await ws.close(); return
        if d.get("type") == "connection_status":
            if d.get("connected"):
                print(f"[{ts()}] ✅ WING connected: {ip}")
                break
            else:
                print(f"[{ts()}] ❌ WING failed: {d.get('error')}")
                await ws.close(); return

    # ── STEP 2: scan all channel names ───────────────────────
    print(f"[{ts()}] Scanning channel names …")
    await send({"type": "scan_mixer_channel_names"})
    raw_names: dict[int, str] = {}
    while True:
        d = await recv_with_timeout(10)
        if d is None:
            print("Timeout on scan — continuing with empty names")
            break
        t = d.get("type")
        if t == "mixer_channel_names":
            raw_names = {int(k): v for k, v in d.get("channel_names", {}).items()}
            print(f"[{ts()}] Got {len(raw_names)} channel names from mixer")
            break
        if t not in ("mixer_update",):
            pass

    # Recognise instruments
    await send({"type": "scan_channel_names", "channels": channels})
    while True:
        d = await recv_with_timeout(8)
        if d is None:
            break
        if d.get("type") == "channel_scan_result":
            results = d.get("results") or {}
            print(f"\n{'─'*50}")
            print(f"  WING RACK — HULIGANS — канальная карта")
            print(f"{'─'*50}")
            for ch in sorted(results.keys(), key=int):
                r = results[ch]
                preset = r.get("preset") or "?"
                name = r.get("name") or f"Ch{ch}"
                label = INSTRUMENT_LABELS.get(preset, preset)
                star = "✓" if r.get("recognized") else "·"
                channel_map[int(ch)] = preset
                print(f"  Ch{int(ch):>2}: {name:<12} {star} {label}")
            print(f"{'─'*50}\n")
            break

    # ── STEP 3: start MixingAgent ────────────────────────────
    print(f"[{ts()}] Starting MixingAgent mode={args.mode} kb_first=True device={audio_device_id}")
    payload = {
        "type": "start_mixing_agent",
        "mode": args.mode,
        "kb_first": True,
        "channels": channels,
        "channel_mapping": {str(c): c for c in channels},
        "device_id": audio_device_id,
        "num_audio_channels": 32,
        "sample_rate": 48000,
    }
    await send(payload)
    while True:
        d = await recv_with_timeout(15)
        if d is None:
            print("Timeout waiting for mixing_agent_started")
            break
        if d.get("type") == "mixing_agent_started":
            st = d.get("status") or {}
            print(f"[{ts()}] ✅ MixingAgent running — mode={st.get('mode')} kb_first={st.get('kb_first')}")
            print(f"       KB entries tracked across {st.get('channels_tracked',0)} channels\n")
            break

    # ── STEP 4: monitor loop ─────────────────────────────────
    print(f"[{ts()}] Monitoring (Ctrl+C to stop) …\n")
    status_tick = time.time()
    applied_total = 0

    async def poll_status():
        await send({"type": "get_mixing_agent_status"})

    try:
        while True:
            now = time.time()
            if now - status_tick > 5.0:
                await poll_status()
                status_tick = now

            d = await recv_with_timeout(2.0)
            if d is None:
                continue

            t_type = d.get("type", "")

            if t_type == "mixing_agent_status":
                st = d.get("status") or {}
                pending_n = st.get("pending_actions", 0)
                applied_n = st.get("applied_actions", 0)
                cycles = st.get("cycle_count", 0)
                errs = st.get("recent_errors") or []
                ch_tracked = st.get("channels_tracked", 0)
                if applied_n > applied_total:
                    applied_total = applied_n
                print(
                    f"[{ts()}] Status: cycles={cycles} applied={applied_n} "
                    f"pending={pending_n} ch={ch_tracked}"
                    + (f" ERRORS: {errs}" if errs else "")
                )
                # show pending suggestions in SUGGEST mode
                pending = d.get("pending_actions") or []
                for p in pending:
                    ch = p.get("channel")
                    inst = channel_map.get(ch, "?")
                    label = INSTRUMENT_LABELS.get(inst, inst)
                    act = ACTION_ICONS.get(p.get("type", ""), p.get("type", ""))
                    params = p.get("parameters") or {}
                    reason = p.get("reason", "")[:80]
                    conf = p.get("confidence", 0)
                    print(f"  → {act} ch{ch} {label:<14} conf={conf:.2f} | {reason}")

            elif t_type == "mixing_agent_suggestions":
                pending = d.get("pending") or []
                for p in pending:
                    ch = p.get("channel")
                    inst = channel_map.get(ch, "?")
                    label = INSTRUMENT_LABELS.get(inst, inst)
                    act = ACTION_ICONS.get(p.get("type", ""), p.get("type", ""))
                    params = p.get("parameters") or {}
                    reason = p.get("reason", "")[:90]
                    conf = p.get("confidence", 0)
                    print(
                        f"[{ts()}] 💡 {act:<8} ch{ch:>2} {label:<14} "
                        f"conf={conf:.2f} | {reason}"
                    )
                    if args.mode == "suggest":
                        print(f"          approve?  (auto-approving …)")
                        await send({"type": "mixing_agent_approve", "approve_all": True})

            elif t_type == "mixer_update":
                addr = d.get("address", "")
                if "/fdr" in addr or "/mute" in addr or "/dyn" in addr or "/eq" in addr:
                    vals = d.get("values", [])
                    m = __import__("re").search(r"/ch/(\d+)/", addr)
                    ch_n = int(m.group(1)) if m else 0
                    inst = channel_map.get(ch_n, "")
                    label = INSTRUMENT_LABELS.get(inst, "")
                    print(
                        f"[{ts()}] OSC  ch{ch_n:>2} {addr:<30} = {vals}  {label}"
                    )

            elif t_type == "error":
                print(f"[{ts()}] ⚠ Server error: {d.get('error')}")

    except KeyboardInterrupt:
        print(f"\n[{ts()}] Stopping …")
        await send({"type": "stop_mixing_agent"})
        await asyncio.sleep(0.5)
        await ws.close()
        print(f"[{ts()}] Done. Total applied actions: {applied_total}")


def main() -> None:
    p = argparse.ArgumentParser(description="WING Rack — MixingAgent live monitor")
    p.add_argument("--ip", default=None, help="WING IP")
    p.add_argument("--send-port", dest="send_port", type=int, default=None)
    p.add_argument("--recv-port", dest="recv_port", type=int, default=None)
    p.add_argument("--max-channel", dest="max_channel", type=int, default=40)
    p.add_argument("--mode", choices=("auto", "suggest", "manual"), default="auto")
    p.add_argument("--device-id", dest="device_id", default=None, help="Dante DVS device index (default 1)")
    args = p.parse_args()
    if args.device_id is None:
        args.device_id = 1
    asyncio.run(run(args))


if __name__ == "__main__":
    main()
