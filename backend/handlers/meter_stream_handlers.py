"""Read-only realtime meter stream (dashboard telemetry) handlers.

Streams `meter_frame` messages (~10 Hz) to subscribed websocket clients while
audio capture is running. Telemetry only: no console writes, no authority.
"""

import asyncio
import logging

from operator_product_state import build_meter_frame

logger = logging.getLogger(__name__)

METER_STREAM_INTERVAL_SEC = 0.1
METER_STREAM_IDLE_INTERVAL_SEC = 1.0


def register_handlers(server):
    if not hasattr(server, "_meter_stream_subscribers"):
        server._meter_stream_subscribers = set()
    if not hasattr(server, "_meter_stream_task"):
        server._meter_stream_task = None

    async def _stream_loop():
        try:
            while server._meter_stream_subscribers:
                frame = build_meter_frame(server)
                stale = []
                for websocket in list(server._meter_stream_subscribers):
                    try:
                        await server.send_to_client(websocket, frame)
                    except Exception:
                        stale.append(websocket)
                for websocket in stale:
                    server._meter_stream_subscribers.discard(websocket)
                interval = (
                    METER_STREAM_INTERVAL_SEC
                    if frame.get("available")
                    else METER_STREAM_IDLE_INTERVAL_SEC
                )
                await asyncio.sleep(interval)
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            logger.error("Meter stream loop stopped: %s", exc, exc_info=True)
        finally:
            server._meter_stream_task = None

    def _ensure_stream_task():
        task = server._meter_stream_task
        if task is None or task.done():
            server._meter_stream_task = asyncio.create_task(_stream_loop())

    async def handle_subscribe_meters(websocket, data):
        server._meter_stream_subscribers.add(websocket)
        _ensure_stream_task()
        await server.send_to_client(websocket, {
            "type": "meter_stream_status",
            "subscribed": True,
            "interval_sec": METER_STREAM_INTERVAL_SEC,
            "read_only": True,
            "live_write": False,
        })

    async def handle_unsubscribe_meters(websocket, data):
        server._meter_stream_subscribers.discard(websocket)
        await server.send_to_client(websocket, {
            "type": "meter_stream_status",
            "subscribed": False,
            "read_only": True,
            "live_write": False,
        })

    return {
        "subscribe_meters": handle_subscribe_meters,
        "unsubscribe_meters": handle_unsubscribe_meters,
    }
