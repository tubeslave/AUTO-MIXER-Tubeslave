"""
Fader service — логика start_realtime_fader, start_auto_balance и связанных операций.

Вынесено из server.py для разделения ответственности.
"""

import asyncio
import logging
from typing import Dict, List, Optional, Any

from auto_fader import AutoFaderController

logger = logging.getLogger(__name__)


class FaderService:
    """Сервис управления Auto Fader (Real-Time Fader, Auto Balance)."""

    def __init__(self, server: Any):
        self._server = server

    @property
    def mixer_client(self):
        return self._server.mixer_client

    @property
    def config(self):
        return self._server.config

    @property
    def bleed_service(self):
        return self._server.bleed_service

    @property
    def auto_fader_controller(self) -> Optional[AutoFaderController]:
        return self._server.auto_fader_controller

    @auto_fader_controller.setter
    def auto_fader_controller(self, value):
        self._server.auto_fader_controller = value

    async def _broadcast(self, msg: dict):
        await self._server.broadcast(msg)

    async def _send_to_client(self, websocket, msg: dict):
        await self._server.send_to_client(websocket, msg)

    async def start_realtime_fader(
        self,
        websocket,
        device_id: str = None,
        channels: List[int] = None,
        channel_settings: Dict = None,
        channel_mapping: Dict = None,
        settings: Dict = None,
    ):
        """Start Real-Time Fader mode."""
        channels = channels or []
        channel_settings = channel_settings or {}
        channel_mapping = channel_mapping or {}
        settings = settings or {}
        if not isinstance(settings, dict):
            settings = {}

        logger.info(f"Starting Real-Time Fader: device={device_id}, channels={channels}")

        if not self.mixer_client or not self.mixer_client.is_connected:
            await self._send_to_client(websocket, {
                "type": "auto_fader_status",
                "status_type": "error",
                "active": False,
                "error": "Mixer not connected"
            })
            return

        if device_id is None or not channels:
            await self._send_to_client(websocket, {
                "type": "auto_fader_status",
                "status_type": "error",
                "active": False,
                "error": "Missing required parameters: device_id or channels"
            })
            return

        try:
            if self.auto_fader_controller:
                self.auto_fader_controller.stop()

            self.auto_fader_controller = AutoFaderController(
                mixer_client=self.mixer_client,
                sample_rate=48000,
                config=self.config,
                bleed_service=self.bleed_service
            )

            if settings:
                self.auto_fader_controller.update_settings(
                    fader_range_db=settings.get('faderRangeDb'),
                    avg_window_sec=settings.get('avgWindowSec'),
                    sensitivity=settings.get('sensitivity'),
                    attack_ms=settings.get('attackMs'),
                    release_ms=settings.get('releaseMs'),
                    gate_threshold=settings.get('gateThreshold', -50.0)
                )

            loop = asyncio.get_running_loop()

            def on_status_update(status_data):
                try:
                    status_data_copy = dict(status_data)
                    status_type = status_data_copy.pop('type', 'status_update')
                    loop.call_soon_threadsafe(
                        lambda: asyncio.create_task(self._broadcast({
                            "type": "auto_fader_status",
                            "status_type": status_type,
                            **status_data_copy
                        }))
                    )
                except Exception as e:
                    logger.error(f"Error broadcasting auto fader status: {e}")

            def on_levels_updated(levels_data):
                try:
                    loop.call_soon_threadsafe(
                        lambda: asyncio.create_task(self._broadcast({
                            "type": "auto_fader_status",
                            "status_type": "levels_update",
                            "channels": levels_data
                        }))
                    )
                except Exception as e:
                    logger.error(f"Error broadcasting auto fader levels: {e}")

            self.auto_fader_controller.on_status_update = on_status_update
            self.auto_fader_controller.on_levels_updated = on_levels_updated

            int_channel_settings = {}
            instrument_types_for_bleed = {}
            for ch_str, ch_data in channel_settings.items():
                try:
                    ch_id = int(ch_str)
                    instrument_type = ch_data.get('instrumentType', 'custom')
                    int_channel_settings[ch_id] = {'instrument_type': instrument_type}
                    instrument_types_for_bleed[ch_id] = (
                        'tom' if instrument_type == 'toms' else instrument_type
                    )
                except (ValueError, TypeError):
                    pass

            if self.bleed_service and instrument_types_for_bleed:
                self.bleed_service.configure(instrument_types_for_bleed)

            int_channel_mapping = {}
            for ch_str, mixer_ch in channel_mapping.items():
                try:
                    int_channel_mapping[int(ch_str)] = int(mixer_ch)
                except (ValueError, TypeError):
                    pass

            success = self.auto_fader_controller.start(
                device_id=int(device_id),
                channels=[int(ch) for ch in channels],
                channel_settings=int_channel_settings,
                channel_mapping=int_channel_mapping,
                on_status_callback=on_status_update
            )

            if not success:
                await self._send_to_client(websocket, {
                    "type": "auto_fader_status",
                    "status_type": "error",
                    "active": False,
                    "error": "Failed to start audio capture"
                })
                return

            success = self.auto_fader_controller.start_realtime_fader()

            if success:
                await self._broadcast({
                    "type": "auto_fader_status",
                    "status_type": "realtime_fader_started",
                    "active": True,
                    "realtime_enabled": True,
                    "mode": "realtime"
                })
            else:
                await self._send_to_client(websocket, {
                    "type": "auto_fader_status",
                    "status_type": "error",
                    "error": "Failed to start real-time fader"
                })

        except Exception as e:
            logger.error(f"Error starting Real-Time Fader: {e}", exc_info=True)
            await self._send_to_client(websocket, {
                "type": "auto_fader_status",
                "status_type": "error",
                "active": False,
                "error": str(e)
            })

    async def stop_realtime_fader(self, websocket):
        """Stop Real-Time Fader mode."""
        logger.info("Stopping Real-Time Fader")
        try:
            if self.auto_fader_controller:
                self.auto_fader_controller.stop_realtime_fader()
            await self._broadcast({
                "type": "auto_fader_status",
                "status_type": "realtime_fader_stopped",
                "active": False,
                "realtime_enabled": False
            })
        except Exception as e:
            logger.error(f"Error stopping Real-Time Fader: {e}", exc_info=True)
            await self._send_to_client(websocket, {
                "type": "auto_fader_status",
                "status_type": "error",
                "error": str(e)
            })

    async def start_auto_balance(
        self,
        websocket,
        device_id: str = None,
        channels: List[int] = None,
        channel_settings: Dict = None,
        channel_mapping: Dict = None,
        duration: float = 15,
        bleed_threshold: float = -50,
    ):
        """Start Auto Balance collection (LEARN phase)."""
        channels = channels or []
        channel_settings = channel_settings or {}
        channel_mapping = channel_mapping or {}

        logger.info(
            f"Starting Auto Balance LEARN: device={device_id}, channels={channels}, "
            f"duration={duration}s, bleed_threshold={bleed_threshold} LUFS"
        )

        if not self.mixer_client or not self.mixer_client.is_connected:
            await self._send_to_client(websocket, {
                "type": "auto_fader_status",
                "status_type": "error",
                "active": False,
                "error": "Mixer not connected"
            })
            return

        if device_id is None or not channels:
            await self._send_to_client(websocket, {
                "type": "auto_fader_status",
                "status_type": "error",
                "active": False,
                "error": "Missing required parameters: device_id or channels"
            })
            return

        try:
            if self.auto_fader_controller:
                self.auto_fader_controller.stop()

            self.auto_fader_controller = AutoFaderController(
                mixer_client=self.mixer_client,
                sample_rate=48000,
                config=self.config,
                bleed_service=self.bleed_service
            )

            loop = asyncio.get_running_loop()

            def on_status_update(status_data):
                try:
                    status_data_copy = dict(status_data)
                    status_type = status_data_copy.pop('type', 'status_update')
                    loop.call_soon_threadsafe(
                        lambda: asyncio.create_task(self._broadcast({
                            "type": "auto_fader_status",
                            "status_type": status_type,
                            **status_data_copy
                        }))
                    )
                except Exception as e:
                    logger.error(f"Error broadcasting auto balance status: {e}")

            def on_levels_updated(levels_data):
                try:
                    loop.call_soon_threadsafe(
                        lambda: asyncio.create_task(self._broadcast({
                            "type": "auto_fader_status",
                            "status_type": "levels_update",
                            "channels": levels_data
                        }))
                    )
                except Exception as e:
                    logger.error(f"Error broadcasting auto balance levels: {e}")

            self.auto_fader_controller.on_status_update = on_status_update
            self.auto_fader_controller.on_levels_updated = on_levels_updated

            int_channel_settings = {}
            instrument_types_for_bleed = {}
            for ch_str, ch_data in channel_settings.items():
                try:
                    ch_id = int(ch_str)
                    instrument_type = ch_data.get('instrumentType') or ch_data.get('preset', 'custom')
                    int_channel_settings[ch_id] = {'instrument_type': instrument_type}
                    instrument_types_for_bleed[ch_id] = (
                        'tom' if instrument_type == 'toms' else instrument_type
                    )
                except (ValueError, TypeError):
                    pass

            if self.bleed_service and instrument_types_for_bleed:
                self.bleed_service.configure(instrument_types_for_bleed)

            int_channel_mapping = {}
            for ch_str, mixer_ch in channel_mapping.items():
                try:
                    int_channel_mapping[int(ch_str)] = int(mixer_ch)
                except (ValueError, TypeError):
                    pass

            success = self.auto_fader_controller.start(
                device_id=int(device_id),
                channels=[int(ch) for ch in channels],
                channel_settings=int_channel_settings,
                channel_mapping=int_channel_mapping,
                on_status_callback=on_status_update
            )

            if not success:
                await self._send_to_client(websocket, {
                    "type": "auto_fader_status",
                    "status_type": "error",
                    "active": False,
                    "error": "Failed to start audio capture"
                })
                return

            success = self.auto_fader_controller.start_auto_balance(
                duration=duration, bleed_threshold=bleed_threshold
            )

            if success:
                await self._broadcast({
                    "type": "auto_fader_status",
                    "status_type": "auto_balance_started",
                    "active": True,
                    "collecting": True,
                    "duration": duration,
                    "mode": "static"
                })
            else:
                await self._send_to_client(websocket, {
                    "type": "auto_fader_status",
                    "status_type": "error",
                    "error": "Failed to start auto balance collection"
                })

        except Exception as e:
            logger.error(f"Error starting Auto Balance: {e}", exc_info=True)
            await self._send_to_client(websocket, {
                "type": "auto_fader_status",
                "status_type": "error",
                "active": False,
                "error": str(e)
            })

    async def apply_auto_balance(self, websocket):
        """Apply Auto Balance results to mixer."""
        logger.info("Applying Auto Balance")
        try:
            if not self.auto_fader_controller:
                await self._send_to_client(websocket, {
                    "type": "auto_fader_status",
                    "status_type": "error",
                    "error": "Auto Fader controller not initialized"
                })
                return

            success = self.auto_fader_controller.apply_auto_balance()
            if success:
                status = self.auto_fader_controller.get_status()
                auto_balance_result = status.get('auto_balance_result') or {}
                await self._broadcast({
                    "type": "auto_fader_status",
                    "status_type": "auto_balance_applied",
                    "applied_count": len(auto_balance_result),
                    "total_count": len(auto_balance_result),
                    "pass_number": status.get('auto_balance_pass', 1)
                })
            else:
                await self._send_to_client(websocket, {
                    "type": "auto_fader_status",
                    "status_type": "error",
                    "error": "Failed to apply auto balance"
                })
        except Exception as e:
            logger.error(f"Error applying Auto Balance: {e}", exc_info=True)
            await self._send_to_client(websocket, {
                "type": "auto_fader_status",
                "status_type": "error",
                "error": str(e)
            })

    async def cancel_auto_balance(self, websocket):
        """Cancel Auto Balance collection."""
        logger.info("Cancelling Auto Balance")
        try:
            if self.auto_fader_controller:
                self.auto_fader_controller.cancel_auto_balance()
            await self._broadcast({
                "type": "auto_fader_status",
                "status_type": "auto_balance_cancelled"
            })
        except Exception as e:
            logger.error(f"Error cancelling Auto Balance: {e}", exc_info=True)
            await self._send_to_client(websocket, {
                "type": "auto_fader_status",
                "status_type": "error",
                "error": str(e)
            })

    async def lock_auto_balance_channel(self, websocket, channel):
        """Lock a channel so it won't be changed on subsequent Auto Balance passes."""
        try:
            if not self.auto_fader_controller:
                await self._send_to_client(websocket, {
                    "type": "auto_fader_status",
                    "status_type": "error",
                    "error": "Auto Fader controller not initialized"
                })
                return

            ch = int(channel) if channel is not None else None
            if ch is not None:
                success = self.auto_fader_controller.lock_channel(ch)
                if success:
                    await self._broadcast({
                        "type": "auto_fader_status",
                        "status_type": "channel_lock_changed",
                        "channel": ch,
                        "locked": True
                    })
                else:
                    await self._send_to_client(websocket, {
                        "type": "auto_fader_status",
                        "status_type": "error",
                        "error": f"Channel {ch} not found"
                    })
        except Exception as e:
            logger.error(f"Error locking channel: {e}", exc_info=True)

    async def unlock_auto_balance_channel(self, websocket, channel):
        """Unlock a channel so it can be changed on subsequent Auto Balance passes."""
        try:
            if not self.auto_fader_controller:
                await self._send_to_client(websocket, {
                    "type": "auto_fader_status",
                    "status_type": "error",
                    "error": "Auto Fader controller not initialized"
                })
                return

            ch = int(channel) if channel is not None else None
            if ch is not None:
                success = self.auto_fader_controller.unlock_channel(ch)
                if success:
                    await self._broadcast({
                        "type": "auto_fader_status",
                        "status_type": "channel_lock_changed",
                        "channel": ch,
                        "locked": False
                    })
                else:
                    await self._send_to_client(websocket, {
                        "type": "auto_fader_status",
                        "status_type": "error",
                        "error": f"Channel {ch} not found"
                    })
        except Exception as e:
            logger.error(f"Error unlocking channel: {e}", exc_info=True)

    async def set_auto_fader_profile(self, websocket, profile: str):
        """Set Auto Fader genre profile."""
        logger.info(f"Setting Auto Fader profile: {profile}")
        try:
            if self.auto_fader_controller:
                self.auto_fader_controller.set_profile(profile)
            await self._send_to_client(websocket, {
                "type": "auto_fader_status",
                "message": f"Profile set to: {profile}"
            })
        except Exception as e:
            logger.error(f"Error setting Auto Fader profile: {e}", exc_info=True)
            await self._send_to_client(websocket, {
                "type": "auto_fader_status",
                "status_type": "error",
                "error": str(e)
            })

    async def update_auto_fader_settings(self, websocket, settings: Dict):
        """Update Auto Fader settings."""
        logger.info(f"Updating Auto Fader settings: {settings}")
        try:
            if self.auto_fader_controller:
                self.auto_fader_controller.update_settings(
                    target_lufs=settings.get('targetLufs'),
                    max_adjustment_db=settings.get('maxAdjustmentDb'),
                    ratio=settings.get('ratio'),
                    attack_ms=settings.get('attackMs'),
                    release_ms=settings.get('releaseMs'),
                    hold_ms=settings.get('holdMs'),
                )
            await self._send_to_client(websocket, {
                "type": "auto_fader_status",
                "message": "Settings updated"
            })
        except Exception as e:
            logger.error(f"Error updating Auto Fader settings: {e}", exc_info=True)
            await self._send_to_client(websocket, {
                "type": "auto_fader_status",
                "status_type": "error",
                "error": str(e)
            })
