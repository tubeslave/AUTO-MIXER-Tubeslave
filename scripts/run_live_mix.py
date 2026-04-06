#!/usr/bin/env python3
"""Live mixing session runner.

Connects to WING, scans channels, starts analysis-based MixingAgent,
and prints every correction applied in real time.
"""
import asyncio
import json
import sys
import time
import websockets

WING_IP = "192.168.1.102"
WING_PORT = 2223
DANTE_DEVICE = 1          # sounddevice index for Dante Virtual Soundcard (sd.query_devices() index)
DANTE_CHANNELS = 32       # number of input channels
SERVER_WS = "ws://localhost:8765"
SESSION_SEC = 600          # run for 10 minutes (0 = forever)


async def recv_type(ws, wanted: set, drain_sec: float = 30.0, verbose: bool = False):
    """Read messages until one matching `wanted` types is found."""
    deadline = time.time() + drain_sec
    while time.time() < deadline:
        try:
            raw = await asyncio.wait_for(ws.recv(), timeout=2.0)
            msg = json.loads(raw)
            t = msg.get("type", "")
            if verbose and t not in ("mixer_update",):
                print(f"  [recv] {t}: {str(msg)[:80]}")
            if t in wanted:
                return msg
        except asyncio.TimeoutError:
            pass
    return None


async def run():
    print(f"Connecting to AutoMixerServer at {SERVER_WS} ...")
    try:
        ws = await asyncio.wait_for(
            websockets.connect(SERVER_WS, ping_interval=20, ping_timeout=40),
            timeout=10,
        )
    except Exception as e:
        print(f"ERROR: Cannot connect to server: {e}")
        print("Start the server first:  PYTHONPATH=backend python backend/server.py")
        sys.exit(1)
    print("  ✓ Server connected")

    async with ws:
        # ── 1. Connect to Wing ────────────────────────────────────────────
        print(f"\nConnecting to WING at {WING_IP}:{WING_PORT} ...")
        await ws.send(json.dumps({
            "type": "connect_wing",
            "ip": WING_IP,
            "send_port": WING_PORT,
        }))
        msg = await recv_type(ws, {"connection_status", "error"}, drain_sec=15)
        if not msg:
            print("  ERROR: No connection response from server")
            return
        if not msg.get("connected"):
            print(f"  ERROR: Wing connection failed: {msg.get('error','')}")
            return
        print(f"  ✓ Wing connected ({msg.get('ip','')})")

        # Drain the initial OSC burst (xremote subscription populates state)
        print("  Waiting for Wing state to populate (3s)...")
        await asyncio.sleep(3)

        # ── 2. Scan channel names ─────────────────────────────────────────
        print("\nScanning channel names...")
        await ws.send(json.dumps({"type": "scan_mixer_channel_names"}))
        channels = {}
        deadline = time.time() + 20
        while time.time() < deadline:
            try:
                raw = await asyncio.wait_for(ws.recv(), 2.0)
                msg = json.loads(raw)
                # Server may send "channels" or "channel_names" key
                if "channels" in msg:
                    channels = msg["channels"]
                    break
                elif "channel_names" in msg:
                    channels = msg["channel_names"]
                    break
            except asyncio.TimeoutError:
                pass

        named = {k: v for k, v in channels.items()
                 if v and str(v).strip() and str(v).lower() not in ("unknown", "", "ch")}
        print(f"  {len(channels)} total channels, {len(named)} named:")
        for ch, nm in sorted(named.items(), key=lambda x: int(x[0])):
            print(f"    ch{ch:>2}: {nm}")

        # ── 2b. Run instrument recognition (populates server instrument map) ──
        print("\nRunning instrument recognition...")
        ch_list = [int(k) for k in channels.keys()] if channels else list(range(1, 41))
        await ws.send(json.dumps({
            "type": "scan_channel_names",
            "channels": ch_list,
        }))
        recognized = {}
        deadline = time.time() + 20
        while time.time() < deadline:
            try:
                raw = await asyncio.wait_for(ws.recv(), 2.0)
                msg = json.loads(raw)
                if msg.get("type") == "channel_scan_result":
                    results = msg.get("results", {})
                    for ch, r in results.items():
                        inst = r.get("preset") or r.get("instrument") or ""
                        if inst:
                            recognized[int(ch)] = inst
                    print(f"  Recognized {len(recognized)} instruments:")
                    for ch, inst in sorted(recognized.items())[:16]:
                        print(f"    ch{ch:>2}: {inst}")
                    break
            except asyncio.TimeoutError:
                pass

        # ── 3. Start MixingAgent ──────────────────────────────────────────
        print(f"\nStarting MixingAgent (mode=auto, Dante device={DANTE_DEVICE})...")
        await ws.send(json.dumps({
            "type": "start_mixing_agent",
            "mode": "auto",
            "kb_first": True,
            "device_id": DANTE_DEVICE,
            "num_audio_channels": DANTE_CHANNELS,
            "sample_rate": 48000,
        }))

        # Wait for agent start, draining OSC updates
        deadline = time.time() + 15
        agent_started = False
        while time.time() < deadline:
            try:
                raw = await asyncio.wait_for(ws.recv(), 2.0)
                msg = json.loads(raw)
                t = msg.get("type", "")
                if t == "mixing_agent_started":
                    st = msg.get("status", {})
                    print(f"  ✓ Agent started  mode={st.get('mode','?')} "
                          f"kb_first={st.get('kb_first','?')}")
                    agent_started = True
                    break
                elif t == "error":
                    print(f"  ERROR starting agent: {msg.get('error','')}")
                    return
            except asyncio.TimeoutError:
                pass

        if not agent_started:
            print("  WARNING: No agent-started confirmation (may still be running)")

        # ── 4. Monitor live corrections ───────────────────────────────────
        print(f"\n{'─'*65}")
        print("  LIVE MIXING — analysis-based corrections")
        print(f"{'─'*65}")
        print("  EQ  = equalizer band set from spectrum analysis")
        print("  DYN = compressor threshold adjustment from crest-factor")
        print("  PHZ = phase/delay correction (snare bleed)")
        print("  FDR = fader level balance (gradual LUFS tracking)")
        print(f"{'─'*65}\n")

        t0 = time.time()
        last_status = t0
        counts = {"eq": 0, "dyn": 0, "phz": 0, "fdr": 0, "total": 0}
        session_limit = SESSION_SEC if SESSION_SEC > 0 else float("inf")

        while time.time() - t0 < session_limit:
            now = time.time()

            # Request status every 30s
            if now - last_status >= 30:
                await ws.send(json.dumps({"type": "get_mixing_agent_status"}))
                last_status = now

            try:
                raw = await asyncio.wait_for(ws.recv(), 2.0)
            except asyncio.TimeoutError:
                continue

            try:
                msg = json.loads(raw)
            except Exception:
                continue

            t = msg.get("type", "")

            if t == "mixer_update":
                addr = msg.get("address", "")
                val = msg.get("values", msg.get("args", ""))
                counts["total"] += 1

                if "/eq/" in addr:
                    counts["eq"] += 1
                    ch = addr.split("/")[2] if len(addr.split("/")) > 2 else "?"
                    # Show only significant EQ changes, not every poll
                    if counts["eq"] <= 5 or counts["eq"] % 20 == 0:
                        print(f"  EQ  ch{ch}  {addr.split('/eq/')[-1]:20s} = {val}")

                elif "/dyn/" in addr or "/comp/" in addr:
                    counts["dyn"] += 1
                    ch = addr.split("/")[2] if len(addr.split("/")) > 2 else "?"
                    print(f"  DYN ch{ch}  {addr.split('/')[-1]:20s} = {val}")

                elif "/dly" in addr or "/inv" in addr:
                    counts["phz"] += 1
                    ch = addr.split("/")[2] if len(addr.split("/")) > 2 else "?"
                    print(f"  PHZ ch{ch}  {addr.split('/')[-1]:20s} = {val}")

                elif "/fdr" in addr:
                    counts["fdr"] += 1
                    ch = addr.split("/")[2] if len(addr.split("/")) > 2 else "?"
                    if counts["fdr"] % 8 == 0:  # Print every 8th fader move
                        print(f"  FDR ch{ch}  fader                = {val}")

            elif t in ("mixing_agent_status",):
                st = msg.get("status") or {}
                applied = st.get("applied_actions", 0) if isinstance(st, dict) else 0
                cycle = st.get("cycle_count", 0) if isinstance(st, dict) else 0
                channels_tracked = st.get("channels_tracked", 0) if isinstance(st, dict) else 0
                elapsed = now - t0
                print(f"\n  ── {elapsed:.0f}s: cycle={cycle} applied={applied} "
                      f"tracked_ch={channels_tracked} "
                      f"| EQ={counts['eq']} DYN={counts['dyn']} "
                      f"PHZ={counts['phz']} FDR={counts['fdr']} "
                      f"OSC_total={counts['total']}\n")

        # ── 5. Final report ───────────────────────────────────────────────
        elapsed = time.time() - t0
        print(f"\n{'─'*65}")
        print(f"  Session complete: {elapsed:.0f}s")
        print(f"  EQ corrections:       {counts['eq']}")
        print(f"  Dynamics corrections: {counts['dyn']}")
        print(f"  Phase corrections:    {counts['phz']}")
        print(f"  Fader adjustments:    {counts['fdr']}")
        print(f"  Total OSC commands:   {counts['total']}")
        print(f"{'─'*65}")

        await ws.send(json.dumps({"type": "stop_mixing_agent"}))


if __name__ == "__main__":
    asyncio.run(run())
