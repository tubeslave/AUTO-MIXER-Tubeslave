"""
Gain staging service — логика start_realtime_correction, stop_realtime_correction.

Вынесено из server.py для разделения ответственности.
"""

import asyncio
import logging
import threading
from typing import Dict, List, Optional, Any

import numpy as np

from band_analyzer import samples_to_band_metrics
from lufs_gain_staging import LUFSGainStagingController, SafeGainCalibrator

logger = logging.getLogger(__name__)

BLEED_BYPASS_PRESETS = {
    "playback",
    "playback_l",
    "playback_r",
    "electric_guitar",
    "electricguitar",
    "guitar",
    "overhead",
    "overheads",
    "hi_hat",
    "hihat",
    "ride",
    "bass",
    "accordion",
    "tom",
    "tom_mid",
    "tom_lo",
    "tom_hi",
    "tom_floor",
}


def _normalize_bleed_instrument(raw: str) -> str:
    if not raw:
        return "unknown"
    val = str(raw).strip().lower().replace("-", "_").replace(" ", "_")
    aliases = {
        "toms": "tom",
        "tom_hi": "tom",
        "tom_mid": "tom",
        "tom_lo": "tom",
        "hihat": "hihat",
        "overheads": "overhead",
        "lead_vocal": "vocal",
        "leadvocal": "vocal",
        "back_vocal": "vocal",
        "backvocal": "vocal",
        "bgv": "vocal",
        "electricguitar": "guitar",
        "acousticguitar": "guitar",
        "keys": "unknown",
        "keyboard": "unknown",
    }
    return aliases.get(val, val)


class GainStagingService:
    """Сервис управления Gain Staging (Safe Gain Calibration)."""

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
    def gain_staging(self) -> Optional[LUFSGainStagingController]:
        return self._server.gain_staging

    @gain_staging.setter
    def gain_staging(self, value):
        self._server.gain_staging = value

    @property
    def safe_gain_calibrator(self) -> Optional[SafeGainCalibrator]:
        return self._server.safe_gain_calibrator

    @safe_gain_calibrator.setter
    def safe_gain_calibrator(self, value):
        self._server.safe_gain_calibrator = value

    async def _broadcast(self, msg: dict):
        await self._server.broadcast(msg)

    async def _stop_safe_gain_runtime(self, reset_calibrator: bool = True):
        """Stop safe-gain runtime resources (audio thread + analyzer)."""
        if hasattr(self._server, '_safe_gain_audio_thread_stop'):
            self._server._safe_gain_audio_thread_stop.set()

        safe_thread = getattr(self._server, '_safe_gain_audio_thread', None)
        if (
            safe_thread
            and safe_thread.is_alive()
            and threading.current_thread() is not safe_thread
        ):
            safe_thread.join(timeout=1.0)

        if self.gain_staging:
            if self.gain_staging.realtime_correction_enabled:
                self.gain_staging.stop_realtime_correction()
            if self.gain_staging.is_active:
                self.gain_staging.stop()

        if reset_calibrator and self.safe_gain_calibrator:
            self.safe_gain_calibrator.reset()
            self.safe_gain_calibrator = None

    async def start_realtime_correction(
        self,
        device_id: str = None,
        channels: list = None,
        channel_settings: dict = None,
        channel_mapping: dict = None,
        mode: str = "lufs",
        learning_duration_sec: float = None,
    ):
        """Start Safe Gain Staging calibration (анализ → одноразовое применение)."""
        logger.info("Starting Safe Gain Staging calibration (new method)...")

        if not self.mixer_client or not self.mixer_client.is_connected:
            await self._broadcast({
                "type": "gain_staging_status",
                "active": False,
                "realtime_enabled": False,
                "error": "Mixer not connected"
            })
            return

        if not device_id or not channels or not channel_settings:
            await self._broadcast({
                "type": "gain_staging_status",
                "active": False,
                "realtime_enabled": False,
                "error": "Missing required parameters: device_id, channels, or channel_settings"
            })
            logger.warning("Missing required parameters for Safe Gain calibration")
            return

        if self.safe_gain_calibrator:
            logger.info("Stopping existing SafeGainCalibrator before starting new one...")
            if self.safe_gain_calibrator.state.value != 'idle':
                self.safe_gain_calibrator.reset()
            self.safe_gain_calibrator = None

        try:
            if learning_duration_sec is not None:
                if 'automation' not in self.config:
                    self.config['automation'] = {}
                if 'safe_gain_calibration' not in self.config['automation']:
                    self.config['automation']['safe_gain_calibration'] = {}
                self.config['automation']['safe_gain_calibration']['learning_duration_sec'] = (
                    float(learning_duration_sec)
                )
                logger.info(f"Applying learning_duration_sec={learning_duration_sec} seconds")

            import pyaudio
            pa = pyaudio.PyAudio()
            device_info = pa.get_device_info_by_index(int(device_id))
            sample_rate = int(device_info.get('defaultSampleRate', 48000))
            pa.terminate()

            if self.mixer_client:
                logger.info("Auto-resetting TRIM to 0 dB before Safe Gain...")
                for ch_str in channels:
                    ch = int(ch_str) if isinstance(ch_str, str) else ch_str
                    mixer_ch = (
                        channel_mapping.get(ch, ch)
                        if channel_mapping else ch
                    )
                    try:
                        self.mixer_client.set_channel_gain(mixer_ch, 0.0)
                    except Exception as e:
                        logger.debug("TRIM reset ch%s failed: %s", mixer_ch, e)
                import time as _time
                _time.sleep(0.1)

            logger.info("Initializing SafeGainCalibrator...")
            cal_config = self.config.get('automation', {}).get('safe_gain_calibration', {})
            logger.info(
                f"Config learning_duration_sec: {cal_config.get('learning_duration_sec', 'NOT SET')}"
            )

            self.safe_gain_calibrator = SafeGainCalibrator(
                mixer_client=self.mixer_client,
                sample_rate=sample_rate,
                config=self.config,
                bleed_service=self.bleed_service,
            )

            if learning_duration_sec is not None:
                self.safe_gain_calibrator.learning_duration = float(learning_duration_sec)

            loop = asyncio.get_running_loop()

            def on_progress_update(status: dict):
                message = {
                    "type": "gain_staging_status",
                    "status_type": "safe_gain_progress",
                    **status
                }
                loop.call_soon_threadsafe(
                    lambda msg=message: asyncio.create_task(self._broadcast(msg))
                )

            def on_suggestions_ready(suggestions: dict):
                message = {
                    "type": "gain_staging_status",
                    "status_type": "safe_gain_ready",
                    "suggestions": suggestions,
                    "message": "Analysis complete. Applying corrections..."
                }
                loop.call_soon_threadsafe(
                    lambda msg=message: asyncio.create_task(self._broadcast(msg))
                )
                if self.safe_gain_calibrator:
                    applied = self.safe_gain_calibrator.apply_corrections()
                    logger.info(
                        "Safe Gain corrections auto-finished: applied=%s. Stopping analysis runtime.",
                        applied,
                    )
                    loop.call_soon_threadsafe(
                        lambda: asyncio.create_task(self._broadcast({
                            "type": "gain_staging_status",
                            "status_type": "safe_gain_applied" if applied else "safe_gain_apply_skipped",
                            "message": (
                                "Corrections applied successfully"
                                if applied
                                else "Analysis complete. No corrections applied"
                            ),
                        }))
                    )
                    loop.call_soon_threadsafe(
                        lambda: asyncio.create_task(self._stop_safe_gain_runtime(reset_calibrator=True))
                    )
                    loop.call_soon_threadsafe(
                        lambda: asyncio.create_task(self._broadcast({
                            "type": "gain_staging_status",
                            "active": False,
                            "realtime_enabled": False,
                            "status_type": "safe_gain_completed",
                            "completed": True,
                            "button_color": "green",
                            "message": "Gain analysis complete: all channels ready",
                        }))
                    )

            self.safe_gain_calibrator.on_progress_update = on_progress_update
            self.safe_gain_calibrator.on_suggestions_ready = on_suggestions_ready

            for audio_ch_str in channels:
                audio_ch = int(audio_ch_str) if isinstance(audio_ch_str, str) else audio_ch_str
                mixer_ch = (
                    channel_mapping.get(audio_ch, audio_ch)
                    if channel_mapping else audio_ch
                )
                self.safe_gain_calibrator.add_channel(
                    audio_channel=audio_ch, mixer_channel=mixer_ch
                )

            if not self.gain_staging:
                logger.info("Initializing audio analyzer for Safe Gain calibration...")
                self.gain_staging = LUFSGainStagingController(
                    mixer_client=self.mixer_client,
                    config=self.config,
                    bleed_service=self.bleed_service
                )
            else:
                self.gain_staging.mixer_client = self.mixer_client

            converted_settings = {}
            instrument_types_for_bleed = {}
            bleed_bypass_channels = set()
            for ch_str, settings in channel_settings.items():
                ch = int(ch_str) if isinstance(ch_str, str) else ch_str
                converted_settings[ch] = settings
                instrument_type = (
                    settings.get('instrumentType') or
                    settings.get('preset') or
                    'custom'
                )
                instrument_types_for_bleed[ch] = _normalize_bleed_instrument(instrument_type)
                normalized_preset = str(instrument_type).strip().lower().replace("-", "_").replace(" ", "_")
                if normalized_preset in BLEED_BYPASS_PRESETS:
                    bleed_bypass_channels.add(ch)

            if self.bleed_service and instrument_types_for_bleed:
                self.bleed_service.configure(instrument_types_for_bleed)

            converted_mapping = {}
            if channel_mapping:
                for audio_ch_str, mixer_ch in channel_mapping.items():
                    audio_ch = int(audio_ch_str) if isinstance(audio_ch_str, str) else audio_ch_str
                    converted_mapping[audio_ch] = mixer_ch
            else:
                for audio_ch in channels:
                    audio_ch_int = int(audio_ch) if isinstance(audio_ch, str) else audio_ch
                    converted_mapping[audio_ch_int] = audio_ch_int

            self.gain_staging.channel_settings = converted_settings
            self.gain_staging.channel_mapping = converted_mapping
            self.safe_gain_calibrator.channel_settings = converted_settings

            if self.gain_staging.is_active:
                logger.info("Stopping existing audio analyzer...")
                self.gain_staging.stop()

            logger.info("Starting audio analyzer for Safe Gain calibration...")
            self._server._safe_gain_audio_thread_stop = threading.Event()

            def safe_gain_audio_processor():
                import time
                while not self._server._safe_gain_audio_thread_stop.is_set():
                    if not self.safe_gain_calibrator or self.safe_gain_calibrator.state.value != 'learning':
                        time.sleep(0.1)
                        continue
                    if not self.gain_staging or not hasattr(self.gain_staging, '_audio_buffers'):
                        time.sleep(0.1)
                        continue
                    samples_by_channel = {}
                    for audio_ch in channels:
                        audio_ch_int = int(audio_ch) if isinstance(audio_ch, str) else audio_ch
                        if audio_ch_int not in self.gain_staging._audio_buffers:
                            continue
                        buffer = self.gain_staging._audio_buffers[audio_ch_int]
                        if not buffer or len(buffer) == 0:
                            continue
                        chunks = list(buffer)
                        if not chunks:
                            continue
                        try:
                            samples = np.concatenate(chunks)
                            if len(samples) > 0:
                                samples_by_channel[audio_ch_int] = samples
                        except Exception as e:
                            logger.debug(f"Error collecting audio for SafeGain ch{audio_ch_int}: {e}")

                    if not samples_by_channel:
                        time.sleep(0.05)
                        continue

                    centroids = {}
                    levels = {}
                    band_metrics = {}
                    sample_rate = getattr(
                        self.safe_gain_calibrator, 'sample_rate', 48000
                    )
                    for ch_id, samples in samples_by_channel.items():
                        rms = float(np.sqrt(np.mean(samples ** 2) + 1e-10))
                        levels[ch_id] = float(20 * np.log10(rms) if rms > 0 else -100.0)
                        centroids[ch_id] = self.safe_gain_calibrator._spectral_centroid_hz(samples)
                        band_metrics[ch_id] = samples_to_band_metrics(samples, sample_rate)

                    if self.bleed_service and self.bleed_service.enabled:
                        try:
                            self.bleed_service.update(levels, centroids, band_metrics)
                        except Exception as e:
                            logger.debug(f"Bleed update failed in SafeGain: {e}")

                    for ch_id, samples in samples_by_channel.items():
                        try:
                            bleed_ratio = None
                            bleed_confidence = None
                            bleed_method = None
                            if self.bleed_service and self.bleed_service.enabled:
                                if ch_id not in bleed_bypass_channels:
                                    bleed_info = self.bleed_service.get_bleed_info(ch_id)
                                    if bleed_info:
                                        bleed_ratio = float(bleed_info.bleed_ratio)
                                        bleed_confidence = float(getattr(bleed_info, "confidence", 0.0))
                                        bleed_method = str(getattr(bleed_info, "method_used", "service"))
                            self.safe_gain_calibrator.process_audio(
                                ch_id,
                                samples,
                                bleed_ratio_override=bleed_ratio,
                                bleed_confidence_override=bleed_confidence,
                                bleed_method_override=bleed_method,
                                spectral_centroid_override=centroids.get(ch_id),
                                all_channel_levels_db=levels,
                            )
                        except Exception as e:
                            logger.debug(f"Error processing audio for SafeGain ch{ch_id}: {e}")
                    time.sleep(0.05)

            self._server._safe_gain_audio_thread = threading.Thread(
                target=safe_gain_audio_processor,
                daemon=True
            )

            success = self.gain_staging.start(
                device_id=device_id,
                channels=[int(ch) for ch in channels],
                channel_settings=converted_settings,
                channel_mapping=converted_mapping,
                on_status_callback=None
            )

            if not success:
                await self._broadcast({
                    "type": "gain_staging_status",
                    "active": False,
                    "realtime_enabled": False,
                    "error": "Failed to start audio analyzer"
                })
                return

            self._server._safe_gain_audio_thread.start()
            logger.info("Audio processing thread started for Safe Gain calibration")

            analysis_started = self.safe_gain_calibrator.start_analysis()

            if not analysis_started:
                await self._broadcast({
                    "type": "gain_staging_status",
                    "active": False,
                    "realtime_enabled": False,
                    "error": "Failed to start Safe Gain analysis"
                })
                return

            await self._broadcast({
                "type": "gain_staging_status",
                "active": True,
                "realtime_enabled": True,
                "status_type": "safe_gain_started",
                "message": "Safe Gain analysis started (auto-stop when channels are ready)"
            })
            logger.info("Safe Gain calibration started successfully")

        except Exception as e:
            logger.error(f"Error starting real-time correction: {e}", exc_info=True)
            await self._broadcast({
                "type": "gain_staging_status",
                "active": self.gain_staging.is_active if self.gain_staging else False,
                "realtime_enabled": False,
                "error": f"Error: {str(e)}"
            })

    async def stop_realtime_correction(self):
        """Stop Safe Gain calibration or real-time TRIM correction."""
        logger.info("Stopping gain staging...")

        if self.safe_gain_calibrator or self.gain_staging:
            logger.info("Stopping Safe Gain runtime resources...")
            await self._stop_safe_gain_runtime(reset_calibrator=True)

        await self._broadcast({
            "type": "gain_staging_status",
            "active": False,
            "realtime_enabled": False,
            "message": "Gain staging stopped"
        })
        logger.info("Real-time correction stopped")
