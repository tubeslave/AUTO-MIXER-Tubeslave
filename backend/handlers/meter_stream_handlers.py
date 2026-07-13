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
                try:
                    frame = build_meter_frame(server)
                    previous_readiness = dict(
                        getattr(server, "last_soundcheck_readiness", {}) or {}
                    )
                    refresh_readiness = getattr(
                        server,
                        "refresh_soundcheck_readiness_truth",
                        None,
                    )
                    readiness = (
                        refresh_readiness()
                        if callable(refresh_readiness)
                        else None
                    )
                    if (
                        isinstance(readiness, dict)
                        and (
                            readiness.get("mix_readiness")
                            != previous_readiness.get("mix_readiness")
                            or readiness.get("global_blockers")
                            != previous_readiness.get("global_blockers")
                        )
                        and callable(getattr(server, "broadcast", None))
                    ):
                        await server.broadcast({
                            "type": "auto_soundcheck_status",
                            "status": "readiness_refreshed",
                            "running": bool(
                                getattr(server, "auto_soundcheck_running", False)
                            ),
                            "is_running": bool(
                                getattr(server, "auto_soundcheck_running", False)
                            ),
                            "message": "Soundcheck readiness evidence changed.",
                            "soundcheck_readiness": readiness,
                        })
                except asyncio.CancelledError:
                    raise
                except Exception as exc:
                    logger.error(
                        "Meter/readiness watchdog iteration failed: %s",
                        exc,
                        exc_info=True,
                    )
                    frame = {
                        "type": "meter_frame",
                        "available": False,
                        "read_only": True,
                        "live_write": False,
                        "signal_present": False,
                        "active_channel_count": 0,
                        "channels": [],
                        "capture": {"running": False, "fresh": False},
                        "master_bus": {"available": False, "source_channels": []},
                        "analyzer": {
                            "available": False,
                            "spectrum_bins": [],
                            "source_channels": [],
                            "read_only": True,
                        },
                        "error": "meter_readiness_watchdog_error",
                    }
                    previous = dict(
                        getattr(server, "last_soundcheck_readiness", {}) or {}
                    )
                    if previous:
                        blockers = list(previous.get("global_blockers") or [])
                        if "soundcheck_readiness_watchdog_error" not in blockers:
                            blockers.append("soundcheck_readiness_watchdog_error")
                        previous.update({
                            "status": "soundcheck_incomplete",
                            "mix_readiness": "soundcheck_incomplete",
                            "safety_state": "manual_review_required",
                            "readiness_evidence_complete": False,
                            "global_blockers": blockers,
                        })
                        server.last_soundcheck_readiness = previous
                        broadcast = getattr(server, "broadcast", None)
                        if callable(broadcast):
                            try:
                                await broadcast({
                                    "type": "auto_soundcheck_status",
                                    "status": "readiness_invalidated",
                                    "running": bool(
                                        getattr(server, "auto_soundcheck_running", False)
                                    ),
                                    "is_running": bool(
                                        getattr(server, "auto_soundcheck_running", False)
                                    ),
                                    "message": (
                                        "Soundcheck readiness watchdog failed; "
                                        "readiness is fail-closed until a fresh check."
                                    ),
                                    "soundcheck_readiness": previous,
                                })
                            except Exception:
                                logger.error(
                                    "Failed to broadcast watchdog invalidation",
                                    exc_info=True,
                                )
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
