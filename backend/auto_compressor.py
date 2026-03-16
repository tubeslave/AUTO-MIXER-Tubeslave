"""
Auto Compressor Controller.

Two modes:
- Soundcheck: record post-fader audio per channel (5-10 s), analyze, classify,
  get preset, adapt params, apply to mixer with smooth transition.
- Live: continuous post-fader monitoring, detect problems (over/under compression,
  pumping, lost transients), auto-correct with limits.

Signal source: post-fader per channel (routing must be configured on mixer).
"""

import asyncio
import logging
import threading
import time
import numpy as np
from collections import deque
from typing import Dict, List, Optional, Callable, Any

from channel_recognizer import recognize_instrument
from compressor_presets import get_preset, get_available_tasks
from compressor_adaptation import adapt_params, THR_MIN, THR_MAX, THR_SAFE_MAX, ratio_float_to_wing
from compressor_profiles import (
    get_profile_for_context, ProfileSelector, ProfileLibrary,
    Genre, Style, TaskType, ProfileContext
)
from signal_analysis import SignalFeatureExtractor, ChannelSignalFeatures

logger = logging.getLogger(__name__)

# Smooth transition: duration and update rate
TRANSITION_DURATION_SEC = 0.8
TRANSITION_UPDATE_HZ = 40
# Live monitoring
LIVE_POLL_INTERVAL_SEC = 0.8
GR_EXCESS_DB = 8.0  # Lowered threshold for over-compression detection
GR_LOW_DB = 1.0
GR_SEVERE_DB = 15.0  # Severe over-compression threshold
AUTO_CORRECTION_COOLDOWN_SEC = 3.0  # Reduced cooldown for faster response
MAX_CORRECTIONS_IN_ROW = 5  # Increased to allow more corrections


def _ease_out(t: float) -> float:
    """Ease-out curve for smooth parameter transition."""
    return 1.0 - (1.0 - t) ** 2


class AutoCompressorController:
    def __init__(
        self,
        mixer_client: Any,
        sample_rate: int = 48000,
        soundcheck_duration_per_channel: float = 7.0,
        target_gr_db: float = 6.0,
        bleed_service=None,
        audio_capture=None,
    ):
        self.mixer_client = mixer_client
        self.sample_rate = sample_rate
        self.soundcheck_duration_per_channel = soundcheck_duration_per_channel
        self.target_gr_db = target_gr_db
        self.bleed_service = bleed_service
        self.chunk_size = 1024
        self._audio_capture = audio_capture

        self._pa = None
        self._stream = None
        self.is_active = False
        self._stop_event = threading.Event()
        self._num_channels = 0
        self.device_index = None
        self.channel_mapping: Dict[int, int] = {}  # audio_ch -> mixer_ch
        self.channel_names: Dict[int, str] = {}
        self.channels: List[int] = []
        self._audio_buffers: Dict[int, deque] = {}
        self._extractors: Dict[int, SignalFeatureExtractor] = {}
        self._lock = threading.Lock()

        self.on_status: Optional[Callable[[Dict], None]] = None
        self._status_ws_callback: Optional[Callable] = None

        # Soundcheck state
        self.soundcheck_running = False
        self.soundcheck_current_channel_index = 0
        self.soundcheck_recording_start_time: Optional[float] = None
        self.soundcheck_recorded_samples: Dict[int, np.ndarray] = {}
        self._soundcheck_task: Optional[asyncio.Task] = None

        # Live state
        self.live_running = False
        self.live_auto_correct_enabled = True
        self._live_task: Optional[asyncio.Task] = None
        self._last_correction_time: Dict[int, float] = {}
        self._corrections_in_row: Dict[int, int] = {}
        self.current_params: Dict[int, Dict[str, Any]] = {}
        self.channel_features: Dict[int, ChannelSignalFeatures] = {}

    def stop(self):
        """Stop all activity and release audio."""
        self._stop_event.set()
        self.soundcheck_running = False
        self.live_running = False
        if self._audio_capture is not None:
            try:
                self._audio_capture.unsubscribe('auto_compressor')
            except Exception:
                pass
        self._stop_stream()
        self.is_active = False
        if self._pa:
            try:
                self._pa.terminate()
            except Exception:
                pass
            self._pa = None
        logger.info("Auto Compressor controller stopped")

    def _stop_stream(self):
        with self._lock:
            if self._stream:
                try:
                    self._stream.stop_stream()
                    self._stream.close()
                except Exception as e:
                    logger.debug(f"Stream close: {e}")
                self._stream = None

    def _audio_callback(self, in_data, frame_count, time_info, status):
        import pyaudio
        if not self.is_active:
            return (None, pyaudio.paComplete)
        if status:
            logger.warning(f"Auto Compressor callback status: {status}")
        try:
            audio_data = np.frombuffer(in_data, dtype=np.float32)
            if self._num_channels > 1:
                audio_data = audio_data.reshape(-1, self._num_channels)
            else:
                audio_data = audio_data.reshape(-1, 1)
            for audio_ch in list(self._audio_buffers.keys()):
                if audio_ch <= self._num_channels:
                    channel_data = audio_data[:, audio_ch - 1]
                    self._audio_buffers[audio_ch].append(channel_data.copy())
        except Exception as e:
            logger.warning(f"Auto Compressor callback error: {e}", exc_info=True)
        return (None, pyaudio.paContinue)

    def _audio_capture_poll(self):
        """Poll AudioCapture buffers and fill local _audio_buffers."""
        if not self._audio_capture:
            return
        for audio_ch in list(self._audio_buffers.keys()):
            data = self._audio_capture.get_buffer(audio_ch, self.chunk_size)
            if data is not None and len(data) > 0:
                self._audio_buffers[audio_ch].append(data.copy())

    def start(
        self,
        device_id: int,
        channels: List[int],
        channel_mapping: Dict[int, int],
        channel_names: Optional[Dict[int, str]] = None,
    ) -> bool:
        """Start audio capture for soundcheck/live. Post-fader signal per channel."""
        if self.is_active:
            logger.warning("Auto Compressor already active")
            return False

        self.device_index = device_id
        self.channels = list(channels)
        self.channel_mapping = dict(channel_mapping)
        self.channel_names = dict(channel_names or {})
        self._stop_event.clear()

        # Use unified AudioCapture if available
        if self._audio_capture is not None:
            try:
                self.sample_rate = self._audio_capture.sample_rate
                self._num_channels = max(channels) if channels else 2
                self._audio_buffers.clear()
                self._extractors.clear()
                for audio_ch in channels:
                    self._audio_buffers[audio_ch] = deque(maxlen=max(500, int(self.sample_rate * 12 / self.chunk_size)))
                    self._extractors[audio_ch] = SignalFeatureExtractor(audio_ch, self.sample_rate, self.chunk_size)
                self._audio_capture.subscribe('auto_compressor', self._audio_capture_poll)
                self.is_active = True
                time.sleep(1.0)
                logger.info(f"Auto Compressor started via AudioCapture: {len(channels)} channels")
                return True
            except Exception as e:
                logger.warning(f"AudioCapture integration failed, falling back to PyAudio: {e}")
                self._audio_capture = None

        # Fallback: direct PyAudio stream
        try:
            import pyaudio
            self._pa = pyaudio.PyAudio()
            device_info = self._pa.get_device_info_by_index(int(device_id))
            max_ch = int(device_info.get("maxInputChannels", 2))
            rate = int(device_info.get("defaultSampleRate", 48000))
            self.sample_rate = rate
            self._num_channels = min(max(channels) if channels else 2, max_ch)

            self._audio_buffers.clear()
            self._extractors.clear()
            for audio_ch in channels:
                self._audio_buffers[audio_ch] = deque(maxlen=max(500, int(rate * 12 / self.chunk_size)))
                self._extractors[audio_ch] = SignalFeatureExtractor(audio_ch, rate, self.chunk_size)

            self._stream = self._pa.open(
                format=pyaudio.paFloat32,
                channels=self._num_channels,
                rate=self.sample_rate,
                input=True,
                input_device_index=int(device_id),
                frames_per_buffer=self.chunk_size,
                stream_callback=self._audio_callback,
            )
            self._stream.start_stream()
            self.is_active = True
            # Give the device time to deliver first chunks so buffers are not empty at soundcheck start
            time.sleep(1.0)
            logger.info(f"Auto Compressor started: {len(channels)} channels, stream channels={self._num_channels}, post-fader")
            return True
        except Exception as e:
            logger.error(f"Auto Compressor start failed: {e}", exc_info=True)
            self.stop()
            return False

    def _send_status(self, msg: Dict):
        if self.on_status:
            try:
                self.on_status(msg)
            except Exception as e:
                logger.debug(f"Auto Compressor status callback: {e}")

    def _apply_compressor_smooth(
        self,
        mixer_ch: int,
        target: Dict[str, Any],
        duration_sec: float = TRANSITION_DURATION_SEC,
    ):
        """Apply compressor params with smooth interpolation (blocking)."""
        if not self.mixer_client or not self.mixer_client.is_connected:
            return
        # Check if compressor is enabled, enable if not
        try:
            compressor_on = self.mixer_client.get_compressor_on(mixer_ch)
            if compressor_on is None or compressor_on == 0:
                logger.info(f"Enabling compressor on channel {mixer_ch} before applying parameters")
                self.mixer_client.set_compressor_on(mixer_ch, 1)
                time.sleep(0.15)  # Increased delay for mixer to process enable command
                logger.debug(f"Compressor enabled on ch{mixer_ch}, waiting before applying params")
        except Exception as e:
            logger.warning(f"Could not check/enable compressor ch{mixer_ch}: {e}, enabling anyway")
            try:
                self.mixer_client.set_compressor_on(mixer_ch, 1)
                time.sleep(0.15)  # Increased delay
            except Exception as e2:
                logger.error(f"Failed to enable compressor ch{mixer_ch}: {e2}")
                return
        
        current = self.current_params.get(mixer_ch, {})
        thr0 = current.get("threshold", -15)
        ratio_wing0 = current.get("ratio_wing", "3.0")
        att0 = current.get("attack_ms", 15)
        rel0 = current.get("release_ms", 150)
        knee0 = current.get("knee", 2)
        gain0 = current.get("gain", 0)
        thr1 = target.get("threshold", thr0)
        ratio_wing1 = target.get("ratio_wing", ratio_wing0)
        att1 = target.get("attack_ms", att0)
        rel1 = target.get("release_ms", rel0)
        knee1 = target.get("knee", knee0)
        gain1 = target.get("gain", gain0)
        logger.info(f"Applying compressor to ch{mixer_ch}: thr={thr1:.1f}dB, ratio={ratio_wing1}, attack={att1:.1f}ms, release={rel1:.1f}ms, knee={knee1:.1f}, makeup_gain={gain1:.1f}dB")
        n = max(1, int(duration_sec * TRANSITION_UPDATE_HZ))
        dt = 1.0 / TRANSITION_UPDATE_HZ
        logger.debug(f"Compressor transition: {n} steps, {dt*1000:.1f}ms per step")
        for i in range(n + 1):
            if not self.is_active:
                logger.warning(f"Compressor application interrupted (is_active=False) for ch{mixer_ch}")
                break
            t = _ease_out(i / n)
            thr = thr0 + (thr1 - thr0) * t
            att = att0 + (att1 - att0) * t
            rel = rel0 + (rel1 - rel0) * t
            knee = knee0 + (knee1 - knee0) * t
            gain = gain0 + (gain1 - gain0) * t
            try:
                self.mixer_client.set_compressor(
                    mixer_ch,
                    threshold=float(thr),
                    ratio=str(ratio_wing1),
                    attack=float(att),
                    release=float(rel),
                    knee=int(round(knee)),
                    gain=float(gain)
                )
            except Exception as e:
                logger.error(f"Set compressor ch{mixer_ch} step {i}/{n}: {e}", exc_info=True)
            time.sleep(dt)
        # Ensure compressor is enabled at the end (redundant but safe)
        try:
            self.mixer_client.set_compressor_on(mixer_ch, 1)
            logger.debug(f"Compressor ch{mixer_ch} enabled confirmed")
        except Exception as e:
            logger.warning(f"Failed to ensure compressor enabled ch{mixer_ch}: {e}")
        self.current_params[mixer_ch] = {
            "threshold": thr1, "ratio_wing": ratio_wing1, "attack_ms": att1,
            "release_ms": rel1, "knee": knee1, "gain": gain1,
        }
        logger.info(f"Compressor params applied to ch{mixer_ch}: thr={thr1:.1f}dB, ratio={ratio_wing1}, attack={att1:.1f}ms, release={rel1:.1f}ms, makeup_gain={gain1:.1f}dB")

    async def start_soundcheck(
        self,
        channels_order: Optional[List[int]] = None,
        genre: str = "unknown",
        style: str = "live",
        genre_factor: float = 1.0,  # Deprecated, use genre instead
        mix_density_factor: float = 1.0,
        bpm: Optional[float] = None,
    ):
        """Run soundcheck: for each channel record 5-10 s, analyze, adapt, apply."""
        if not self.is_active:
            self._send_status({"soundcheck": True, "error": "Controller not active"})
            return
        order = channels_order or self.channels
        self.soundcheck_running = True
        self.soundcheck_recorded_samples.clear()
        # Diagnostic: ensure we have buffers for all channels (keys match order)
        buf_keys = list(self._audio_buffers.keys())[:5]
        logger.info(f"Soundcheck: audio buffers count={len(self._audio_buffers)}, stream channels={self._num_channels}, first keys={buf_keys}, order first 5={order[:5] if order else []}")
        self._send_status({"soundcheck": True, "message": "Soundcheck started", "total_channels": len(order)})
        for idx, audio_ch in enumerate(order):
            if not self.soundcheck_running:
                break
            mixer_ch = self.channel_mapping.get(audio_ch, audio_ch)
            name = self.channel_names.get(mixer_ch, "") or self.channel_names.get(audio_ch, "") or f"Ch {mixer_ch}"
            self._send_status({
                "soundcheck": True,
                "current_channel": mixer_ch,
                "current_channel_audio": audio_ch,
                "channel_name": name,
                "progress": idx,
                "total_channels": len(order),
                "message": f"Recording {name} (post-fader)...",
            })
            await asyncio.sleep(0.5)
            
            # Temporarily disable compressor to analyze uncompressed signal
            compressor_was_on = None
            if self.mixer_client and self.mixer_client.is_connected:
                try:
                    compressor_was_on = self.mixer_client.get_compressor_on(mixer_ch)
                    if compressor_was_on == 1:
                        logger.info(f"Soundcheck: Temporarily disabling compressor on {name} (ch{mixer_ch}) for clean signal analysis")
                        self.mixer_client.set_compressor_on(mixer_ch, 0)
                        await asyncio.sleep(0.5)  # Wait for compressor to fully disengage and clear any compression artifacts
                except Exception as e:
                    logger.warning(f"Soundcheck: Could not disable compressor on ch{mixer_ch} for analysis: {e}")
            
            # Clear audio buffer to avoid analyzing old compressed data
            buf = self._audio_buffers.get(audio_ch)
            if buf:
                buf.clear()
                # Wait briefly for first chunk so we don't record silence if callback is slightly delayed
                await asyncio.sleep(0.2)
                logger.debug(f"Soundcheck: Cleared audio buffer for {name} before recording uncompressed signal")
            if not buf:
                logger.warning(f"Soundcheck: No audio buffer for audio_ch={audio_ch} (type={type(audio_ch).__name__}), mixer_ch={mixer_ch}, name={name}")
                self._send_status({"soundcheck": True, "message": f"No audio buffer for {name}, skipping", "current_channel": mixer_ch})
                # Re-enable compressor if it was on
                if compressor_was_on == 1 and self.mixer_client:
                    try:
                        self.mixer_client.set_compressor_on(mixer_ch, 1)
                    except:
                        pass
                continue
            logger.info(f"Soundcheck: Recording {name} (audio_ch={audio_ch}, mixer_ch={mixer_ch}) for {self.soundcheck_duration_per_channel}s (compressor disabled for analysis)...")
            samples_list = []
            start = time.time()
            while time.time() - start < self.soundcheck_duration_per_channel and self.soundcheck_running:
                await asyncio.sleep(0.1)
                while buf:
                    try:
                        block = buf.popleft()
                        samples_list.append(block)
                    except IndexError:
                        break
            total_samples = sum(len(s) for s in samples_list)
            logger.info(f"Soundcheck: Collected {len(samples_list)} blocks for {name}, total samples: {total_samples}")
            if not samples_list:
                logger.warning(f"Soundcheck: No audio samples collected for {name}, skipping")
                self._send_status({"soundcheck": True, "message": f"No audio for {name}, skipping", "current_channel": mixer_ch})
                continue
            # Check signal level in first few blocks
            if samples_list:
                first_block = samples_list[0]
                first_peak = np.max(np.abs(first_block)) if len(first_block) > 0 else 0.0
                first_peak_db = 20 * np.log10(first_peak + 1e-10)
                logger.info(f"Soundcheck: First block peak for {name}: {first_peak:.6f} linear, {first_peak_db:.1f}dB")
            logger.info(f"Soundcheck: Analyzing {len(samples_list)} blocks for {name}...")
            extractor = self._extractors.get(audio_ch)
            if not extractor:
                extractor = SignalFeatureExtractor(audio_ch, self.sample_rate, self.chunk_size)
                self._extractors[audio_ch] = extractor
            extractor.reset()
            # Process each block separately (LUFS/TruePeak meters need incremental processing)
            features = None
            for block in samples_list:
                if len(block) > 0:
                    features = extractor.process(block)
            # Use final features from last processed block
            if features is None or features.peak_db < -100:
                logger.warning(f"Soundcheck: Invalid features for {name}, using fallback")
                # Fallback: process concatenated samples
                samples = np.concatenate(samples_list)
                extractor.reset()
                features = extractor.process(samples)
            # Validate features
            if features.peak_db < -80 or features.lufs_momentary < -80:
                logger.warning(f"Soundcheck: {name} has very low signal (peak={features.peak_db:.1f}dB, lufs={features.lufs_momentary:.1f}), may be silent or no post-fader routing")
                self._send_status({
                    "soundcheck": True,
                    "current_channel": mixer_ch,
                    "message": f"Warning: {name} signal very low (peak={features.peak_db:.1f}dB). Check post-fader routing.",
                })
            self.channel_features[mixer_ch] = features
            instrument = recognize_instrument(name) or "custom"
            logger.info(f"Soundcheck: {name} -> instrument={instrument}, features: peak={features.peak_db:.1f}dB, true_peak={features.true_peak_db:.1f}dB, lufs={features.lufs_momentary:.1f}, rms={features.rms_db:.1f}dB, crest={features.crest_factor_db:.1f}dB")
            
            # Use new profile system with context-aware selection
            try:
                # Detect task type from context and features
                profile_selector = ProfileSelector()
                task_type = profile_selector.detect_task_from_context(
                    instrument, features,
                    Genre(genre.lower()) if genre.lower() in [g.value for g in Genre] else Genre.UNKNOWN,
                    Style(style.lower()) if style.lower() in [s.value for s in Style] else Style.LIVE
                )
                
                # Get profile with context
                profile_dict = get_profile_for_context(
                    instrument=instrument,
                    task=task_type.value,
                    genre=genre,
                    style=style,
                    bpm=bpm,
                    mix_density=mix_density_factor,
                    features=features
                )
                
                # Apply final adaptation based on signal features
                params = adapt_params(
                    features, profile_dict,
                    target_gr_db=self.target_gr_db,
                )
                
                logger.info(f"Soundcheck: {name} -> profile: instrument={instrument}, task={task_type.value}, genre={genre}, style={style}, bpm={bpm}")
            except Exception as e:
                logger.warning(f"Error using new profile system for {name}, falling back to legacy: {e}")
                # Fallback to legacy system
                task = "base"
                preset = get_preset(instrument, task)
                params = adapt_params(
                    features, preset,
                    target_gr_db=self.target_gr_db,
                )
            logger.info(f"Soundcheck: {name} -> adapted params: thr={params['threshold']:.1f}dB, ratio={params['ratio_wing']}, attack={params['attack_ms']:.1f}ms, release={params['release_ms']:.1f}ms, makeup_gain={params.get('gain', 0):.1f}dB")
            self._send_status({
                "soundcheck": True,
                "current_channel": mixer_ch,
                "message": f"Applying to {name}...",
                "params": params,
                "instrument": instrument,
            })
            loop = asyncio.get_event_loop()
            logger.info(f"Soundcheck: Applying compressor params to {name} (mixer_ch={mixer_ch})...")
            # Capture params in closure to avoid issues with lambda
            params_copy = dict(params)
            mixer_ch_copy = mixer_ch
            
            # Re-enable compressor before applying new parameters (if it was disabled)
            if compressor_was_on == 1 and self.mixer_client:
                try:
                    logger.debug(f"Soundcheck: Re-enabling compressor on ch{mixer_ch} before applying new params")
                    self.mixer_client.set_compressor_on(mixer_ch, 1)
                    await asyncio.sleep(0.1)
                except Exception as e:
                    logger.warning(f"Soundcheck: Could not re-enable compressor on ch{mixer_ch}: {e}")
            
            try:
                await loop.run_in_executor(None, lambda: self._apply_compressor_smooth(mixer_ch_copy, params_copy))
                logger.info(f"Soundcheck: Compressor params applied to {name}")
            except Exception as e:
                logger.error(f"Soundcheck: Error applying compressor to {name}: {e}", exc_info=True)
                # Ensure compressor is enabled even if application failed
                if compressor_was_on == 1 and self.mixer_client:
                    try:
                        self.mixer_client.set_compressor_on(mixer_ch, 1)
                    except:
                        pass
                self._send_status({
                    "soundcheck": True,
                    "current_channel": mixer_ch,
                    "message": f"Error applying compressor to {name}: {str(e)}",
                    "error": str(e),
                })
            self._send_status({
                "soundcheck": True,
                "channel_done": mixer_ch,
                "channel_name": name,
                "progress": idx + 1,
                "total_channels": len(order),
                "message": f"Done: {name}",
            })
            await asyncio.sleep(1.5)
        self.soundcheck_running = False
        self._send_status({"soundcheck": True, "complete": True, "message": "Soundcheck complete"})

    def stop_soundcheck(self):
        self.soundcheck_running = False

    def _estimate_gr(self, mixer_ch: int, features: ChannelSignalFeatures) -> float:
        """
        Get gain reduction: use real GR from mixer meter if available (visible on compressor),
        otherwise estimate from level, threshold and ratio.
        """
        # Prefer real GR from mixer when available (what you see on compressor meters)
        if self.mixer_client and self.mixer_client.is_connected:
            try:
                real_gr = self.mixer_client.get_compressor_gr(mixer_ch)
                if real_gr is not None:
                    gr_val = float(real_gr)
                    if gr_val > GR_EXCESS_DB:
                        logger.info(f"ch{mixer_ch} high GR from mixer: {gr_val:.1f}dB -> over_compression")
                    return max(0.0, gr_val)
            except Exception as e:
                logger.debug(f"Could not get GR from mixer ch{mixer_ch}: {e}")

        # Fallback: estimate from current params and signal level
        params = self.current_params.get(mixer_ch, {})
        thr = params.get("threshold", -15)
        r = params.get("ratio", 3.0)
        try:
            ratio_wing = params.get("ratio_wing", "3.0")
            r = float(ratio_wing)
        except (TypeError, ValueError):
            r = 3.0

        detector_type = str(
            params.get("detector", params.get("level_detector", "rms"))
        ).lower()
        if detector_type == "peak":
            raw_level = max(features.true_peak_db, features.peak_db)
        elif detector_type == "lufs":
            raw_level = features.lufs_momentary
        else:
            detector_type = "rms"
            raw_level = features.rms_db
        
        # Compensate for bleed: use compensated level if bleed detected
        if self.bleed_service and self.bleed_service.enabled:
            # Find audio_ch from mixer_ch (reverse lookup)
            audio_ch = None
            for a_ch, m_ch in self.channel_mapping.items():
                if m_ch == mixer_ch:
                    audio_ch = a_ch
                    break
            if audio_ch is not None:
                bleed_info = self.bleed_service.get_bleed_info(audio_ch)
                if bleed_info and bleed_info.bleed_ratio > 0.3:
                    # Compensate level: reduce level estimate when bleed is high
                    compensated_level = self.bleed_service.get_compensated_level(audio_ch, raw_level)
                    raw_level = compensated_level
                    logger.debug(
                        f"Ch{mixer_ch}: Compensated {detector_type} level for GR estimate: "
                        f"{raw_level:.1f} -> {compensated_level:.1f} dB (bleed={bleed_info.bleed_ratio:.2f})"
                    )
        
        # Cap level to avoid unrealistically high GR from meter errors or clipping (e.g. 0 dB)
        level = min(raw_level, -3.0) if raw_level > -3.0 else raw_level
        if level < thr:
            return 0.0
        over = level - thr
        gr = over * (1.0 - 1.0 / max(r, 1.01))
        if gr > GR_EXCESS_DB:
            logger.info(
                f"ch{mixer_ch} high GR estimate: {gr:.1f}dB "
                f"(detector={detector_type}, level={level:.1f}dB, thr={thr:.1f}dB, ratio={r:.1f}) -> over_compression"
            )
        return gr

    async def _live_loop(self):
        """Monitor levels, detect issues, optionally auto-correct."""
        while self.live_running and self.is_active:
            await asyncio.sleep(LIVE_POLL_INTERVAL_SEC)
            if not self.live_running:
                break
            for audio_ch in list(self.channels):
                mixer_ch = self.channel_mapping.get(audio_ch, audio_ch)
                buf = self._audio_buffers.get(audio_ch)
                ext = self._extractors.get(audio_ch)
                if not buf or not ext:
                    continue
                samples_list = []
                while buf:
                    try:
                        samples_list.append(buf.popleft())
                    except IndexError:
                        break
                if not samples_list:
                    continue
                samples = np.concatenate(samples_list)
                features = ext.process(samples)
                self.channel_features[mixer_ch] = features
                gr = self._estimate_gr(mixer_ch, features)
                status = "normal"
                if gr > GR_EXCESS_DB:
                    status = "over_compression"
                    if self.live_auto_correct_enabled:
                        self._maybe_correct(mixer_ch, "over", features)
                elif gr < GR_LOW_DB and features.lufs_momentary > -40:
                    status = "under_compression"
                    if self.live_auto_correct_enabled:
                        self._maybe_correct(mixer_ch, "under", features)
                self._send_status({
                    "live": True,
                    "channel": mixer_ch,
                    "gr_estimate": round(gr, 1),
                    "lufs": round(features.lufs_momentary, 1),
                    "status": status,
                })

    def _maybe_correct(self, mixer_ch: int, issue: str, features: ChannelSignalFeatures):
        now = time.time()
        last = self._last_correction_time.get(mixer_ch, 0)
        if now - last < AUTO_CORRECTION_COOLDOWN_SEC:
            return
        n = self._corrections_in_row.get(mixer_ch, 0)
        if n >= MAX_CORRECTIONS_IN_ROW:
            self._send_status({"live": True, "notification": "operator_attention", "channel": mixer_ch})
            return
        
        # Check for bleed: skip correction if high bleed detected (problem may be bleed, not compression)
        if self.bleed_service and self.bleed_service.enabled:
            # Find audio_ch from mixer_ch
            audio_ch = None
            for a_ch, m_ch in self.channel_mapping.items():
                if m_ch == mixer_ch:
                    audio_ch = a_ch
                    break
            if audio_ch is not None:
                bleed_info = self.bleed_service.get_bleed_info(audio_ch)
                if bleed_info and bleed_info.bleed_ratio > 0.5:
                    logger.debug(f"Ch{mixer_ch}: Skipping correction due to high bleed (ratio={bleed_info.bleed_ratio:.2f})")
                    return
        
        params = dict(self.current_params.get(mixer_ch, {}))
        
        # If no parameters exist for this channel, skip correction
        # (should run soundcheck first to set initial parameters)
        if not params:
            logger.debug(f"No parameters found for ch{mixer_ch}, skipping correction. Run soundcheck first.")
            return
        
        if issue == "over":
            # Get current GR estimate to determine correction severity
            gr_estimate = self._estimate_gr(mixer_ch, features)
            
            thr = params.get("threshold", -15)
            ratio_wing = params.get("ratio_wing", "3.0")
            try:
                r = float(ratio_wing)
            except (TypeError, ValueError):
                r = 3.0
            
            # Aggressive correction for severe over-compression
            if gr_estimate > GR_SEVERE_DB:
                # Severe: raise threshold significantly and reduce ratio aggressively
                threshold_increase = min(10.0, (gr_estimate - GR_EXCESS_DB) * 0.8)
                params["threshold"] = min(THR_SAFE_MAX, thr + threshold_increase)
                if r > 2.0:
                    params["ratio_wing"] = "2.0"
                logger.warning(f"Severe over-compression detected on ch{mixer_ch}: GR={gr_estimate:.1f}dB, applying aggressive correction")
            elif gr_estimate > GR_EXCESS_DB:
                # Moderate: raise threshold moderately and reduce ratio
                threshold_increase = min(6.0, (gr_estimate - GR_EXCESS_DB) * 0.6)
                params["threshold"] = min(THR_SAFE_MAX, thr + threshold_increase)
                if r > 2.5:
                    params["ratio_wing"] = "2.0" if r > 4.0 else "2.5"
                elif r > 2.0:
                    params["ratio_wing"] = "2.0"
                logger.info(f"Over-compression detected on ch{mixer_ch}: GR={gr_estimate:.1f}dB, applying correction")
            
            # Also check for signs of pumping (rapid level changes)
            if features.envelope_variance > 8.0:
                # High variance suggests pumping - raise threshold more
                params["threshold"] = min(THR_SAFE_MAX, params.get("threshold", thr) + 3.0)
                logger.info(f"Pumping detected on ch{mixer_ch}, raising threshold further")
            
            # Check for transient loss (low crest factor despite compression)
            if features.crest_factor_db < 6.0 and gr_estimate > 6.0:
                # Transients are being squashed - raise threshold and reduce ratio
                params["threshold"] = min(THR_SAFE_MAX, params.get("threshold", thr) + 4.0)
                if r > 2.0:
                    params["ratio_wing"] = "2.0"
                logger.info(f"Transient loss detected on ch{mixer_ch}, reducing compression")
        else:
            thr = params.get("threshold", -15)
            params["threshold"] = max(THR_MIN, thr - 2.0)
        
        # Recalculate makeup gain after parameter correction (IMP [53]: EBU-loudness based)
        from compressor_adaptation import adapt_makeup
        thr_new = params.get("threshold", -15)
        ratio_wing_new = params.get("ratio_wing", "3.0")
        try:
            ratio_float = float(ratio_wing_new)
        except (TypeError, ValueError):
            ratio_float = 3.0
        lufs = features.rms_db if features.rms_db > -70 else features.lufs_momentary
        makeup_gain = adapt_makeup(thr_new, ratio_float, lufs)
        params["gain"] = makeup_gain
        logger.info(f"Live correction ch{mixer_ch} ({issue}): thr={thr_new:.1f}dB, ratio={ratio_wing_new}, recalculated makeup_gain={makeup_gain:.1f}dB")
        
        self._last_correction_time[mixer_ch] = now
        self._corrections_in_row[mixer_ch] = n + 1
        self._apply_compressor_smooth(mixer_ch, params, duration_sec=0.5)
        self._send_status({"live": True, "auto_corrected": mixer_ch, "issue": issue})

    async def start_live(self, auto_correct: bool = True):
        if not self.is_active:
            logger.warning("Cannot start live mode: Auto Compressor not active")
            return
        self.live_auto_correct_enabled = auto_correct
        self.live_running = True
        self._last_correction_time.clear()
        self._corrections_in_row.clear()
        
        # Enable compressors on all active channels for live mode
        # Force enable all compressors regardless of current state to ensure they are on
        if self.mixer_client and self.mixer_client.is_connected:
            logger.info(f"Enabling compressors for live mode on {len(self.channels)} channels")
            enabled_count = 0
            for audio_ch in list(self.channels):
                mixer_ch = self.channel_mapping.get(audio_ch, audio_ch)
                try:
                    # Check current state for logging
                    compressor_on = self.mixer_client.get_compressor_on(mixer_ch)
                    logger.info(f"Channel {mixer_ch}: current compressor_on={compressor_on} (type: {type(compressor_on).__name__})")
                    
                    # Force enable compressor regardless of current state
                    # (state may be out of sync with mixer)
                    logger.info(f"Force enabling compressor on channel {mixer_ch} for live mode")
                    self.mixer_client.set_compressor_on(mixer_ch, 1)
                    enabled_count += 1
                    await asyncio.sleep(0.05)  # Small delay between commands
                except Exception as e:
                    logger.warning(f"Could not enable compressor ch{mixer_ch} at live start: {e}", exc_info=True)
            logger.info(f"Force enabled {enabled_count} compressors for live mode")
        else:
            logger.warning(f"Cannot enable compressors: mixer_client={self.mixer_client is not None}, connected={self.mixer_client.is_connected if self.mixer_client else False}")
        
        self._live_task = asyncio.create_task(self._live_loop())
        logger.info(f"Live mode started with auto_correct={auto_correct}")
        self._send_status({"live": True, "started": True})

    def stop_live(self):
        self.live_running = False
        if self._live_task:
            self._live_task.cancel()
            self._live_task = None
        self._send_status({"live": True, "stopped": True})

    def get_status(self) -> Dict[str, Any]:
        return {
            "active": self.is_active,
            "soundcheck_running": self.soundcheck_running,
            "live_running": self.live_running,
            "channels": list(self.channels),
            "channel_mapping": dict(self.channel_mapping),
            "current_params": dict(self.current_params),
        }
