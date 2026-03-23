"""
Feedback service — логика start_feedback_detection, stop_feedback_detection.

Вынесено из server.py для разделения ответственности.
"""

import asyncio
import logging
from typing import Dict, List, Optional, Any

from feedback_detector import FeedbackDetector
from audio_capture import AudioCapture, AudioSourceType

logger = logging.getLogger(__name__)


class FeedbackService:
    """Сервис детекции обратной связи (feedback)."""

    def __init__(self, server: Any):
        self._server = server

    @property
    def mixer_client(self):
        return self._server.mixer_client

    @property
    def config(self):
        return self._server.config

    def _get_feedback_detector(self):
        return self._server.feedback_detector

    def _set_feedback_detector(self, value):
        self._server.feedback_detector = value

    def _get_audio_capture(self):
        return self._server._feedback_audio_capture

    def _set_audio_capture(self, value):
        self._server._feedback_audio_capture = value

    def _get_channel_mapping(self):
        return self._server._feedback_channel_mapping

    def _set_channel_mapping(self, value):
        self._server._feedback_channel_mapping = value

    async def _send_to_client(self, websocket, msg: dict):
        await self._server.send_to_client(websocket, msg)

    async def start_feedback_detection(
        self,
        websocket,
        device_id: str = None,
        channels: List[int] = None,
        channel_mapping: Dict[int, int] = None,
    ):
        """Start feedback detection (requires mixer, audio device, channels)."""
        if not self.mixer_client or not self.mixer_client.is_connected:
            await self._send_to_client(websocket, {
                "type": "feedback_detector_status",
                "active": False,
                "error": "Mixer not connected",
            })
            return

        ac = self._get_audio_capture()
        if self._get_feedback_detector() and ac and ac.running:
            await self._send_to_client(websocket, {
                "type": "feedback_detector_status",
                "active": True,
                "message": "Feedback detection already running",
            })
            return

        channels = channels or [1]
        channel_mapping = channel_mapping or {int(c): int(c) for c in channels}
        cfg = self.config.get("safety", {}).get("feedback_detection", {})
        if not cfg.get("enabled", False):
            logger.info("Feedback detection disabled in config (safety.feedback_detection.enabled)")
        threshold_db = cfg.get("threshold_db", -20.0)
        fader_reduction_db = cfg.get("fader_reduction_db", -6.0)

        try:
            self._set_feedback_detector(FeedbackDetector(
                sample_rate=48000,
                fft_size=2048,
                threshold_db=threshold_db,
                fader_reduction_db=fader_reduction_db,
            ))
            self._set_channel_mapping(channel_mapping)

            device_idx = int(device_id) if device_id else None
            source = AudioSourceType.SOUNDDEVICE if device_id else AudioSourceType.SILENCE
            self._set_audio_capture(AudioCapture(
                num_channels=max(channels) if channels else 40,
                sample_rate=48000,
                buffer_seconds=2.0,
                block_size=2048,
                source_type=source,
                device_name=device_idx,
            ))
            loop = asyncio.get_running_loop()

            def _feedback_poll():
                if not self._get_feedback_detector() or not self.mixer_client:
                    return
                for audio_ch, mixer_ch in self._get_channel_mapping().items():
                    samples = self._get_audio_capture().get_buffer(audio_ch, 2048)
                    if len(samples) < 2048:
                        continue
                    events = self._get_feedback_detector().process(audio_ch, samples)
                    for ev in events:
                        if ev.action == "fader_reduce":
                            try:
                                current = self.mixer_client.get_channel_fader(mixer_ch)
                                current_db = float(current) if current is not None else 0.0
                                reduction = self._get_feedback_detector().get_fader_reduction(audio_ch)
                                new_db = current_db + reduction
                                self.mixer_client.set_channel_fader(mixer_ch, new_db)
                                loop.call_soon_threadsafe(
                                    lambda e=ev: asyncio.create_task(self._server.broadcast({
                                        "type": "feedback_detected",
                                        "channel": mixer_ch,
                                        "frequency_hz": e.frequency_hz,
                                        "magnitude_db": e.magnitude_db,
                                        "action": e.action,
                                    }))
                                )
                            except Exception as e:
                                logger.error(f"Feedback fader reduce failed: {e}")

            self._get_audio_capture().subscribe("feedback_detector", _feedback_poll)
            self._get_audio_capture().start()

            await self._send_to_client(websocket, {
                "type": "feedback_detector_status",
                "active": True,
                "message": "Feedback detection started",
                "channels": list(channel_mapping.keys()),
            })
        except Exception as e:
            logger.error(f"Failed to start feedback detection: {e}", exc_info=True)
            await self._send_to_client(websocket, {
                "type": "feedback_detector_status",
                "active": False,
                "error": str(e),
            })

    async def stop_feedback_detection(self, websocket):
        """Stop feedback detection."""
        ac = self._get_audio_capture()
        if ac:
            try:
                ac.unsubscribe("feedback_detector")
                ac.stop()
            except Exception as e:
                logger.error(f"Error stopping feedback detection: {e}")
            self._set_audio_capture(None)
        self._set_feedback_detector(None)
        self._set_channel_mapping({})
        await self._send_to_client(websocket, {
            "type": "feedback_detector_status",
            "active": False,
            "message": "Feedback detection stopped",
        })
