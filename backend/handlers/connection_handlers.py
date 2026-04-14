"""Connection (Wing / dLive / Mixing Station) message handlers."""

import asyncio
import logging
from mixer_discovery import discover_mixers, discover_mixer_auto

logger = logging.getLogger(__name__)


def register_handlers(server):
    async def handle_connect_wing(websocket, data):
        await server.connect_wing(data.get("ip"), data.get("send_port"), data.get("receive_port"))

    async def handle_connect_dlive(websocket, data):
        await server.connect_dlive(
            ip=data.get("ip", "192.168.3.70"),
            port=data.get("port", 51328),
            tls=data.get("tls", False),
            midi_base_channel=data.get("midi_base_channel", 0),
        )

    async def handle_connect_mixing_station(websocket, data):
        await server.connect_mixing_station(
            data.get("host", "127.0.0.1"),
            data.get("osc_port", 8000),
            data.get("rest_port", 8080)
        )

    async def handle_discover_mixing_station(websocket, data):
        await server.discover_mixing_station()

    async def handle_disconnect(websocket, data):
        await server.disconnect_mixer()

    async def handle_scan_mixers(websocket, data):
        """Scan the network for all available mixers (WING + dLive)."""
        full_scan = data.get("full_scan", False)
        subnet = data.get("subnet", None)

        loop = asyncio.get_event_loop()
        results = await loop.run_in_executor(
            None,
            lambda: discover_mixers(
                scan_subnet=full_scan, subnet=subnet, timeout=2.0
            ),
        )

        mixer_list = []
        for m in results:
            mixer_list.append({
                "mixer_type": m.mixer_type,
                "ip": m.ip,
                "port": m.port,
                "name": m.name,
                "model": m.model,
                "tls": m.tls,
                "method": m.discovery_method,
                "response_ms": round(m.response_time_ms, 0),
            })

        await server.send_to_client(websocket, {
            "type": "mixer_scan_results",
            "mixers": mixer_list,
            "count": len(mixer_list),
        })
        logger.info(f"Mixer scan complete: found {len(mixer_list)} mixer(s)")

    async def handle_auto_connect(websocket, data):
        """Auto-discover and connect to the best mixer on the network."""
        preferred_type = data.get("preferred_type", None)
        preferred_ip = data.get("preferred_ip", None)
        full_scan = data.get("full_scan", False)

        await server.send_to_client(websocket, {
            "type": "auto_connect_status",
            "status": "scanning",
            "message": "Scanning network for mixers...",
        })

        loop = asyncio.get_event_loop()
        discovered = await loop.run_in_executor(
            None,
            lambda: discover_mixer_auto(
                preferred_type=preferred_type,
                preferred_ip=preferred_ip,
                scan_subnet=full_scan,
                timeout=2.0,
            ),
        )

        if discovered is None:
            await server.send_to_client(websocket, {
                "type": "auto_connect_status",
                "status": "not_found",
                "message": "No mixer found on the network",
            })
            return

        await server.send_to_client(websocket, {
            "type": "auto_connect_status",
            "status": "found",
            "mixer_type": discovered.mixer_type,
            "ip": discovered.ip,
            "port": discovered.port,
            "name": discovered.name,
            "message": f"Found {discovered.mixer_type.upper()} @ {discovered.ip}:{discovered.port}",
        })

        if discovered.mixer_type == "dlive":
            await server.connect_dlive(
                ip=discovered.ip,
                port=discovered.port,
                tls=discovered.tls,
            )
        elif discovered.mixer_type == "wing":
            await server.connect_wing(
                ip=discovered.ip,
                send_port=discovered.port,
                receive_port=discovered.port,
            )

    return {
        "connect_wing": handle_connect_wing,
        "connect_dlive": handle_connect_dlive,
        "connect_mixing_station": handle_connect_mixing_station,
        "discover_mixing_station": handle_discover_mixing_station,
        "disconnect_wing": handle_disconnect,
        "disconnect_mixer": handle_disconnect,
        "scan_mixers": handle_scan_mixers,
        "auto_connect": handle_auto_connect,
    }
