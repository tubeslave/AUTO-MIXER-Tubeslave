"""
Auto Fader Controller V2 - Main controller integrating all components

Orchestrates the complete auto fader system:
- C++ DSP core via bridge
- Acoustic analysis
- Static/dynamic mixing
- Genre profiles
- ML data collection
"""

import logging
import asyncio
import os
import time
from typing import Dict, Optional, Callable, Any
from enum import Enum

# Debug log path: use AUTO_MIXER_DEBUG_LOG env or project .cursor/debug.log (no hardcoded user paths)
_controller_dir = os.path.dirname(os.path.abspath(__file__))
DEBUG_LOG_PATH = os.environ.get(
    "AUTO_MIXER_DEBUG_LOG",
    os.path.normpath(os.path.join(_controller_dir, "..", "..", ".cursor", "debug.log")),
)

from .bridge.cpp_bridge import CppBridge, ChannelMetrics
from .bridge.metrics_receiver import MetricsReceiver
from .core.acoustic_analyzer import AcousticAnalyzer, AcousticFeatures
from .core.channel_classifier import ChannelClassifier
from .core.activity_detector import ActivityDetector, ACTIVITY_THRESHOLD_LUFS
from .core.bleed_detector import BleedDetector, BleedInfo
from .core.bleed_compensator import get_compensated_level
from .core.vocal_activity_detector import VocalActivityDetector
from .core.spectral_masking import SpectralMaskingDetector, EQAdjustment
from .core.integrated_lufs import RollingIntegratedLufs
from .balance.static_balancer import StaticBalancer
from .balance.dynamic_mixer import DynamicMixer
from .balance.fuzzy_controller import FuzzyFaderController
from .balance.pid_controller import PIDLoudnessController
from .balance.hierarchical_mixer import HierarchicalMixer
from .profiles.genre_profiles import GenreProfile, GenreType, GENRE_PROFILES
from .ml.data_collector import MLDataCollector

logger = logging.getLogger(__name__)


class OperationMode(Enum):
    """Operation modes"""
    STOPPED = "stopped"
    STATIC = "static"          # Static balance mode
    REALTIME = "realtime"      # Real-time dynamic mixing


class AutoFaderControllerV2:
    """
    Main Auto Fader V2 Controller
    
    Integrates all components into a complete automatic mixing system.
    """
    
    def __init__(self, mixer_client, config: Dict):
        self.mixer_client = mixer_client  # OSC mixer client
        self.config = config
        self._warned_missing_eq_support = False
        self._warned_missing_pan_support = False
        
        # Core components
        self.cpp_bridge = CppBridge()
        self.metrics_receiver: Optional[MetricsReceiver] = None
        self.acoustic_analyzer = AcousticAnalyzer()
        self.channel_classifier = ChannelClassifier()
        self.activity_detector = ActivityDetector()
        bleed_cfg = config.get('bleed_protection') or config.get('automation', {}).get('bleed_protection', {})
        self.bleed_detector = BleedDetector(bleed_cfg)
        self.bleed_protection_enabled = bleed_cfg.get('enabled', True)
        self.bleed_compensation_mode = bleed_cfg.get('compensation_mode', 'band_level')
        self.bleed_compensation_factor_db = bleed_cfg.get('compensation_factor_db', 6.0)
        self.latest_bleed_info: Dict[int, Dict] = {}
        
        auto_fader_cfg = config.get('automation', {}).get('auto_fader', {})
        self.controller_mode = auto_fader_cfg.get("controller_mode", auto_fader_cfg.get("controller_type", "dynamic"))
        self.lufs_window_sec = float(auto_fader_cfg.get("lufs_window_sec", 3.0))
        self.integrated_lufs = RollingIntegratedLufs(window_seconds=self.lufs_window_sec)
        self.latest_integrated_levels: Dict[int, float] = {}
        self.channel_priorities: Dict[int, int] = {
            int(ch): int(priority) for ch, priority in auto_fader_cfg.get("channel_priorities", {}).items()
        }
        
        # Ducking configuration
        ducking_enabled = auto_fader_cfg.get('ducking_enabled', False)
        ducking_amount_db = auto_fader_cfg.get('ducking_amount_db', -4.0)
        vocal_channels = auto_fader_cfg.get('vocal_channels', [])
        ducking_attack_ms = auto_fader_cfg.get('ducking_attack_ms', 10.0)
        ducking_release_ms = auto_fader_cfg.get('ducking_release_ms', 100.0)
        
        self.ducking_enabled = ducking_enabled
        self.ducking_amount_db = ducking_amount_db
        self.vocal_channels = vocal_channels
        self.vocal_activity_detector = VocalActivityDetector(
            threshold_lufs=-30.0,  # Default threshold for vocal activity
            attack_ms=ducking_attack_ms,
            release_ms=ducking_release_ms
        ) if ducking_enabled else None
        
        # Spectral masking configuration
        spectral_masking_enabled = auto_fader_cfg.get('spectral_masking_enabled', False)
        self.spectral_masking_enabled = spectral_masking_enabled
        self.spectral_masking_detector = SpectralMaskingDetector(
            critical_band_hz=(1000.0, 3000.0),
            dominance_threshold_db=6.0,
            max_cut_db=-6.0,
            q_factor=3.0
        ) if spectral_masking_enabled else None

        # Balance components
        self.static_balancer = StaticBalancer()
        self.dynamic_mixer = DynamicMixer()
        self.fuzzy_controller = FuzzyFaderController()
        
        # PID controller (per document recommendation)
        controller_type = auto_fader_cfg.get('controller_type', 'dynamic')  # legacy: 'pid' or 'dynamic'
        pid_kp = auto_fader_cfg.get('pid_kp', 0.5)
        pid_ki = auto_fader_cfg.get('pid_ki', 0.05)
        pid_kd = auto_fader_cfg.get('pid_kd', 0.0)
        pid_output_limit = auto_fader_cfg.get('pid_output_limit', 6.0)
        pid_dead_zone = auto_fader_cfg.get('pid_dead_zone', 0.5)
        
        self.pid_controller = PIDLoudnessController(
            kp=pid_kp,
            ki=pid_ki,
            kd=pid_kd,
            output_limit=pid_output_limit,
            dead_zone=pid_dead_zone
        )
        self.controller_type = controller_type
        if self.controller_mode not in {"mvp", "dynamic", "pid"}:
            self.controller_mode = controller_type
        if self.controller_mode not in {"mvp", "dynamic", "pid"}:
            self.controller_mode = "dynamic"
        
        # Hierarchical mixing configuration
        self.hierarchical_mixer = HierarchicalMixer(
            enabled=bool(auto_fader_cfg.get("hierarchical_mix_enabled", False)),
            overload_threshold_lufs=float(auto_fader_cfg.get("mix_overload_threshold_lufs", -18.0)),
            max_step_cut_db=float(auto_fader_cfg.get("hierarchical_max_step_cut_db", 3.0)),
            cut_scale=float(auto_fader_cfg.get("hierarchical_cut_scale", 0.6)),
        )
        
        # Spatial separation (L/R correlation and pan), only if stereo data exists.
        self.pan_separation_enabled = bool(auto_fader_cfg.get("pan_separation_enabled", False))
        self.pan_max_offset = float(auto_fader_cfg.get("pan_max_offset", 20.0))
        
        # Adaptive noise gate
        self.adaptive_noise_gate_enabled = bool(auto_fader_cfg.get("adaptive_noise_gate_enabled", False))
        self.noise_gate_hold_sec = float(auto_fader_cfg.get("noise_gate_hold_sec", 0.8))
        self.noise_gate_threshold_lufs = float(auto_fader_cfg.get("noise_gate_threshold_lufs", ACTIVITY_THRESHOLD_LUFS))
        self._inactive_since: Dict[int, float] = {}
        self._gate_closed_channels: Dict[int, bool] = {}
        
        # ML component
        self.ml_collector = MLDataCollector()
        
        # State
        self.mode = OperationMode.STOPPED
        self.selected_channels: list[int] = []
        self.reference_channels: list[int] = []
        self.instrument_types: Dict[int, str] = {}
        self.current_genre: GenreType = GenreType.CUSTOM
        self.genre_profile: GenreProfile = GENRE_PROFILES[GenreType.CUSTOM]
        
        # Callbacks
        self.status_callback: Optional[Callable] = None
        
        # Latest state
        self.latest_metrics: Dict[int, ChannelMetrics] = {}
        self.latest_features: Dict[int, AcousticFeatures] = {}
        self.latest_adjustments: Dict[int, float] = {}
        
        # Warm-up period after fader reset (to avoid using old POST-FADER values)
        self.realtime_start_time: Optional[float] = None
        self.warmup_period_seconds = 2.0  # Wait 2 seconds before applying adjustments
        
        # Freeze / manual override: global and per-channel
        self.automation_frozen = False  # When True, no fader commands are sent
        self.channel_freeze_until: Dict[int, float] = {}  # channel_id -> unix time until auto-control resumes
        self.freeze_cooldown_seconds = 10.0  # Per-channel cooldown after manual fader move
        self._last_sent_fader: Dict[int, float] = {}  # channel_id -> last value we set
        self._last_send_time: Dict[int, float] = {}   # channel_id -> time we last sent
        self._cpp_fallback_recommended = False  # True when C++ core failed to start; use v1 realtime fader as fallback
    
    def set_automation_frozen(self, frozen: bool):
        """Freeze or unfreeze all automation (no fader commands when frozen)."""
        self.automation_frozen = frozen
        logger.info(f"Automation frozen: {frozen}")
    
    def set_channel_frozen(self, channel_id: int, seconds: float):
        """Exclude channel from auto-control for the given seconds."""
        import time
        self.channel_freeze_until[channel_id] = time.time() + max(0.0, seconds)
        logger.info(f"Channel {channel_id} frozen for {seconds:.1f}s")
    
    def get_freeze_status(self) -> Dict[str, Any]:
        """Return current freeze state for UI."""
        import time
        now = time.time()
        frozen_channels = [ch for ch, until in self.channel_freeze_until.items() if until > now]
        return {
            "automation_frozen": self.automation_frozen,
            "frozen_channels": frozen_channels,
            "freeze_cooldown_seconds": self.freeze_cooldown_seconds,
        }
    
    def set_status_callback(self, callback: Callable[[Dict], None]):
        """Set callback for status updates"""
        self.status_callback = callback
    
    def _send_status(self, status_type: str, data: Dict = None):
        """Send status update"""
        if self.status_callback:
            message = {
                "status_type": status_type,
                "mode": self.mode.value,
                **(data or {})
            }
            try:
                self.status_callback(message)
            except Exception as e:
                logger.error(f"Error in status callback: {e}")

    def _target_lufs_for_channel(self, channel_id: int) -> float:
        """Resolve target LUFS for channel from genre profile."""
        instrument = self.instrument_types.get(channel_id, 'custom')
        offset = self.genre_profile.instrument_offsets.get(instrument, -7.0)
        return self.genre_profile.target_lufs + offset

    def _has_stereo_metrics(self, metric: ChannelMetrics) -> bool:
        """True when metrics contain left/right data required for correlation."""
        return hasattr(metric, "left_rms") and hasattr(metric, "right_rms")

    def _apply_adaptive_noise_gate(self, channel_id: int, metric: ChannelMetrics, now_ts: float) -> None:
        """Open/close gate based on sustained inactivity."""
        if not self.adaptive_noise_gate_enabled or not self.mixer_client:
            return

        level = metric.lufs_momentary
        currently_inactive = level < self.noise_gate_threshold_lufs

        if currently_inactive:
            self._inactive_since.setdefault(channel_id, now_ts)
            inactive_for = now_ts - self._inactive_since[channel_id]
            if inactive_for >= self.noise_gate_hold_sec and not self._gate_closed_channels.get(channel_id, False):
                if hasattr(self.mixer_client, "set_gate_on"):
                    self.mixer_client.set_gate_on(channel_id, 1)
                    self._gate_closed_channels[channel_id] = True
                elif hasattr(self.mixer_client, "set_channel_fader"):
                    # Fallback for setups without gate OSC control.
                    self.mixer_client.set_channel_fader(channel_id, -120.0)
                    self._gate_closed_channels[channel_id] = True
        else:
            self._inactive_since.pop(channel_id, None)
            if self._gate_closed_channels.get(channel_id, False):
                if hasattr(self.mixer_client, "set_gate_on"):
                    self.mixer_client.set_gate_on(channel_id, 0)
                self._gate_closed_channels[channel_id] = False

    def _apply_pan_separation(
        self,
        vocal_channels: list[int],
        background_channels: list[int],
        all_metrics: Dict[int, ChannelMetrics],
    ) -> None:
        """
        Apply simple pan separation when stereo metrics are available.

        Current C++ metrics usually do not expose L/R values, so this path
        stays disabled unless stereo data is present.
        """
        if not self.pan_separation_enabled or not self.mixer_client:
            return
        if not hasattr(self.mixer_client, "set_channel_pan"):
            if not self._warned_missing_pan_support:
                logger.warning("Pan separation enabled, but mixer client has no set_channel_pan")
                self._warned_missing_pan_support = True
            return

        sample_metric = next(iter(all_metrics.values()), None)
        if sample_metric is None or not self._has_stereo_metrics(sample_metric):
            if not self._warned_missing_pan_support:
                logger.info("Pan separation skipped: stereo L/R metrics are unavailable in C++ bridge")
                self._warned_missing_pan_support = True
            return

        if not vocal_channels or not background_channels:
            return

        # Keep lead vocal near center and gently push backgrounds away.
        lead_vocal = vocal_channels[0]
        try:
            self.mixer_client.set_channel_pan(lead_vocal, 0.0)
        except Exception as exc:
            logger.debug(f"Failed to center lead vocal pan for ch {lead_vocal}: {exc}")

        side = -1.0
        for ch_id in background_channels:
            try:
                self.mixer_client.set_channel_pan(ch_id, side * self.pan_max_offset)
            except Exception as exc:
                logger.debug(f"Failed to set pan for ch {ch_id}: {exc}")
            side *= -1.0
    
    def _validate_real_audio_input(self) -> bool:
        """
        Validate that we have real audio input (not test mode)
        
        Test mode characteristics:
        - All channels have similar LUFS values (within 1-2 dB)
        - LUFS values are very low (-70 to -13 dB constant)
        - Sequential pattern (-13, -14, -15, ..., -28)
        
        Real audio characteristics:
        - Wide range of LUFS values (varies by channel/instrument)
        - Some channels active, some silent
        - Natural variation
        """
        metrics = self.cpp_bridge.get_all_metrics()
        
        if not metrics or len(metrics) < 2:
            logger.warning("Not enough metrics for validation")
            return False
        
        # Get LUFS values
        lufs_values = [m.lufs_momentary for m in metrics.values() if m.lufs_momentary > -100]
        
        if len(lufs_values) < 2:
            logger.warning("All channels silent or invalid")
            return False
        
        # Check 1: LUFS range (real audio has wide range)
        lufs_min = min(lufs_values)
        lufs_max = max(lufs_values)
        lufs_range = lufs_max - lufs_min
        
        # Check 2: Average LUFS (test mode is very low, around -13 to -28)
        lufs_avg = sum(lufs_values) / len(lufs_values)
        
        # Check 3: Variation (real audio has more variation)
        import statistics
        lufs_std = statistics.stdev(lufs_values) if len(lufs_values) > 1 else 0
        
        logger.info(f"Audio validation: range={lufs_range:.1f}dB, avg={lufs_avg:.1f}, std={lufs_std:.1f}")
        logger.info(f"  LUFS values sample: {lufs_values[:10]}")
        
        # Real audio typically has:
        # - Range > 5 dB (test mode has ~15 dB but is sequential pattern)
        # - Average > -40 LUFS (test mode is -13 to -28)
        # - Std dev > 3 dB (test mode has ~5 dB in sequential pattern)
        
        # Real audio should have some active channels (> -60 LUFS)
        active_channels = sum(1 for lufs in lufs_values if lufs > -60)
        logger.info(f"  Active channels (>-60 LUFS): {active_channels}/{len(lufs_values)}")
        
        # For now, allow if there are active channels (even if pattern looks like test mode)
        # This allows testing with Dante even if initial pattern is sequential
        # Real audio will show variation over time
        if active_channels < 1:
            logger.warning("❌ No active channels detected (all < -60 LUFS)")
            return False
        
        # Check if this looks like test mode but allow anyway for Dante testing
        if lufs_range > 10 and lufs_range < 20 and lufs_avg < -15 and lufs_std > 4:
            logger.warning("⚠️ Sequential pattern detected (possible test mode)")
            logger.info("  Allowing for Dante testing - real audio will show variation over time")
        
        logger.info(f"✅ Audio input validated: {active_channels} active channels")
        return True
    
    def _reset_faders_to_zero(self):
        """Reset all selected channel faders to 0 dB before starting real-time mode"""
        logger.info("🔄 _reset_faders_to_zero() called")
        
        if not self.mixer_client:
            logger.warning("Cannot reset faders - mixer client not available")
            return
        
        if not self.selected_channels:
            logger.warning("No channels selected for reset")
            return
        
        # #region agent log
        try:
            import json
            import time
            with open(DEBUG_LOG_PATH, "a") as f:
                f.write(json.dumps({
                    "sessionId": "debug-session",
                    "runId": "run1",
                    "hypothesisId": "D",
                    "location": "controller.py:_reset_faders_to_zero",
                    "message": "Reset started",
                    "data": {
                        "selected_channels": self.selected_channels[:10],
                        "channels_count": len(self.selected_channels)
                    },
                    "timestamp": int(time.time() * 1000)
                }) + "\n")
        except Exception as e:
            logger.error(f"Failed to write debug log: {e}")
        # #endregion
        
        logger.info(f"🔄 Resetting {len(self.selected_channels)} faders to 0 dB...")
        
        # CRITICAL: Mark start of fader reset to ignore stale OSC updates
        import time
        if hasattr(self.mixer_client, '_fader_reset_time'):
            self.mixer_client._fader_reset_time = time.time()
            logger.info(f"Started fader reset protection period: {self.mixer_client._fader_reset_ignore_duration}s")
        
        reset_count = 0
        initial_values = {}
        for ch_id in self.selected_channels:
            try:
                # #region agent log
                try:
                    import json
                    import time
                    initial_val = self.mixer_client.get_channel_fader(ch_id)
                    initial_values[ch_id] = initial_val
                    with open(DEBUG_LOG_PATH, "a") as f:
                        f.write(json.dumps({
                            "sessionId": "debug-session",
                            "runId": "run1",
                            "hypothesisId": "D",
                            "location": "controller.py:_reset_faders_to_zero",
                            "message": "Before reset",
                            "data": {
                                "channel": ch_id,
                                "initial_value": initial_val
                            },
                            "timestamp": int(time.time() * 1000)
                        }) + "\n")
                except: pass
                # #endregion
                
                # Set fader to 0 dB
                self.mixer_client.set_channel_fader(ch_id, 0.0)
                # CRITICAL: Force update local cache to 0.0 immediately
                # This prevents reading old values after reset
                if hasattr(self.mixer_client, 'state'):
                    fader_address = f"/ch/{ch_id}/fdr"
                    self.mixer_client.state[fader_address] = 0.0
                reset_count += 1
            except Exception as e:
                logger.error(f"Failed to reset channel {ch_id}: {e}")
        
        logger.info(f"✅ Reset {reset_count}/{len(self.selected_channels)} faders to 0 dB")
        # Give mixer time to apply changes and verify
        import time
        time.sleep(0.8)  # Increased wait time
        
        # Verify reset by reading back values
        verified_count = 0
        verification_results = {}
        for ch_id in self.selected_channels:
            try:
                current = self.mixer_client.get_channel_fader(ch_id)
                verification_results[ch_id] = current
                if current is not None and abs(current) < 0.1:  # Within 0.1 dB of 0
                    verified_count += 1
                else:
                    logger.warning(f"Channel {ch_id} not reset properly: {current} dB (expected 0.0)")
            except Exception as e:
                logger.warning(f"Could not verify channel {ch_id}: {e}")
        
        # #region agent log
        try:
            import json
            import time
            with open(DEBUG_LOG_PATH, "a") as f:
                f.write(json.dumps({
                    "sessionId": "debug-session",
                    "runId": "run1",
                    "hypothesisId": "D",
                    "location": "controller.py:_reset_faders_to_zero",
                    "message": "Reset verification",
                    "data": {
                        "initial_values": initial_values,
                        "verification_results": verification_results,
                        "verified_count": verified_count,
                        "total_count": len(self.selected_channels)
                    },
                    "timestamp": int(time.time() * 1000)
                }) + "\n")
        except: pass
        # #endregion
        
        logger.info(f"Verified {verified_count}/{len(self.selected_channels)} faders at 0 dB")
    
    def configure(
        self,
        selected_channels: list[int],
        reference_channels: list[int],
        instrument_types: Dict[int, str],
        genre: str = "custom"
    ):
        """Configure auto fader"""
        self.selected_channels = selected_channels
        self.reference_channels = reference_channels
        self.instrument_types = instrument_types

        # Fill missing priorities from instrument tags.
        for ch_id in selected_channels:
            if ch_id in self.channel_priorities:
                continue
            instrument = (instrument_types.get(ch_id) or "").lower()
            if "vocal" in instrument:
                self.channel_priorities[ch_id] = 1
            elif "kick" in instrument or "snare" in instrument or "bass" in instrument:
                self.channel_priorities[ch_id] = 2
            else:
                self.channel_priorities[ch_id] = 3
        
        # Configure bleed detector with instrument types
        self.bleed_detector.configure(instrument_types)
        
        # Set genre profile
        try:
            self.current_genre = GenreType(genre)
            self.genre_profile = GENRE_PROFILES[self.current_genre]
            logger.info(f"Using genre profile: {self.genre_profile.name}")
        except ValueError:
            logger.warning(f"Unknown genre '{genre}', using custom")
            self.current_genre = GenreType.CUSTOM
            self.genre_profile = GENRE_PROFILES[GenreType.CUSTOM]
    
    async def start_realtime(self, settings: Dict = None):
        """Start real-time dynamic mixing mode"""
        logger.info("Starting real-time fader mode...")
        
        # Configure OSC throttle if mixer client supports it
        if self.mixer_client and hasattr(self.mixer_client, 'set_osc_throttle'):
            auto_fader_cfg = self.config.get('automation', {}).get('auto_fader', {})
            osc_throttle_hz = auto_fader_cfg.get('osc_throttle_hz', 10.0)
            self.mixer_client.set_osc_throttle(enabled=True, hz=osc_throttle_hz)
            logger.info(f"OSC throttle configured: {osc_throttle_hz} Hz")
        
        # Start C++ core (or connect to existing). On failure, caller can fall back to Python (v1) realtime fader.
        if not self.cpp_bridge.is_running():
            if not self.cpp_bridge.start_cpp_core():
                logger.error("Failed to start C++ core - use Python fallback (start_realtime_fader v1)")
                self._cpp_fallback_recommended = True
                self._send_status("error", {"error": "Failed to start C++ core", "fallback": "use_v1_realtime_fader"})
                return False
        self._cpp_fallback_recommended = False
        
        # Start metrics receiver
        self.metrics_receiver = MetricsReceiver(self.cpp_bridge)
        self.metrics_receiver.add_callback(self._on_metrics_update)
        self.metrics_receiver.start()
        
        # CRITICAL: Wait for real audio input before starting
        logger.info("Waiting for real audio input validation...")
        import asyncio
        await asyncio.sleep(1.0)  # Wait 1 second for metrics
        
        # Validate that we have real audio input (not test mode)
        if not self._validate_real_audio_input():
            logger.error("❌ No real audio input detected - only test/silence")
            self._send_status("error", {
                "error": "No real audio input detected",
                "message": "Please connect Dante audio input or ensure microphones are active"
            })
            # Stop metrics receiver
            if self.metrics_receiver:
                self.metrics_receiver.stop()
                self.metrics_receiver = None
            # Stop C++ core
            self.cpp_bridge.stop_cpp_core()
            return False
        
        logger.info("✅ Real audio input validated")
        
        # CRITICAL: Reset faders to 0 dB before starting real-time mode
        # This prevents using old fader positions from previous sessions
        logger.info("🔄 About to reset faders to zero...")
        self._reset_faders_to_zero()
        logger.info("✅ Fader reset completed")
        
        self.integrated_lufs.reset()
        self.latest_integrated_levels.clear()
        self.pid_controller.reset_all()
        self._inactive_since.clear()
        self._gate_closed_channels.clear()

        # Configure dynamic mixer only for dynamic mode.
        if self.controller_mode == "dynamic":
            self.dynamic_mixer.configure(
                self.reference_channels,
                self.instrument_types,
                self.genre_profile.instrument_offsets,
                ratio=self.genre_profile.ratio,
                max_adjustment_db=settings.get('maxAdjustmentDb', 6.0) if settings else 6.0
            )
            self.dynamic_mixer.reset()
        
        self.mode = OperationMode.REALTIME
        self.realtime_start_time = time.time()  # Track start time for warm-up period
        self._send_status("realtime_fader_started")
        
        logger.info(f"Real-time mode started with controller_mode={self.controller_mode}.")
        if self.controller_mode == "dynamic":
            logger.info(f"  Phase 1: CALIBRATION ({self.dynamic_mixer.calibration_duration_sec}s) - collecting LUFS data")
            logger.info(f"  Phase 2: STABILIZATION - applying optimal fader positions")
            logger.info(f"  Phase 3: MAINTENANCE - small corrections only (dead zone: ±{self.dynamic_mixer.dead_zone_db} dB)")
        
        return True
    
    def stop_realtime(self):
        """Stop real-time mode"""
        logger.info("Stopping real-time fader...")
        
        # Stop metrics receiver
        if self.metrics_receiver:
            self.metrics_receiver.stop()
            self.metrics_receiver = None
        
        # CRITICAL: Stop C++ core process to prevent metrics generation
        if self.cpp_bridge.is_running():
            self.cpp_bridge.stop_cpp_core()
            logger.info("C++ core stopped")
        
        # Reset bleed detector state
        self.bleed_detector.reset()
        self.integrated_lufs.reset()
        self.latest_integrated_levels.clear()
        self._inactive_since.clear()
        self._gate_closed_channels.clear()
        
        self.mode = OperationMode.STOPPED
        self.realtime_start_time = None  # Reset warm-up timer
        self._send_status("realtime_fader_stopped")
    
    async def start_auto_balance(self, duration: float = 15.0):
        """Start auto balance collection"""
        logger.info(f"Starting auto balance collection ({duration}s)...")
        
        # Start C++ core if not running
        if not self.cpp_bridge.is_running():
            if not self.cpp_bridge.start_cpp_core():
                logger.error("Failed to start C++ core")
                return False
        
        # Start metrics receiver
        if not self.metrics_receiver:
            self.metrics_receiver = MetricsReceiver(self.cpp_bridge)
            self.metrics_receiver.add_callback(self._on_metrics_update_auto_balance)
            self.metrics_receiver.start()
        
        # Start collection
        self.static_balancer.start_collection(duration)
        self.mode = OperationMode.STATIC
        self._send_status("auto_balance_started", {"duration": duration})
        
        # Wait for collection to complete
        while not self.static_balancer.is_collection_complete():
            await asyncio.sleep(0.5)
            progress = (asyncio.get_event_loop().time() - self.static_balancer.collection_start_time) / duration
            self._send_status("levels_update", {
                "channels": {ch: {"progress": progress} for ch in self.selected_channels}
            })
        
        # Calculate statistics and balance
        self.static_balancer.calculate_statistics()
        result = self.static_balancer.calculate_balance(
            self.reference_channels,
            self.instrument_types,
            self.genre_profile.instrument_offsets
        )
        
        self._send_status("auto_balance_ready", {"result": result})
        
        return True
    
    def apply_auto_balance(self):
        """Apply calculated auto balance"""
        if not self.static_balancer.balance_result:
            logger.warning("No balance result to apply")
            return False
        
        applied = 0
        for ch_id, adjustment in self.static_balancer.balance_result.items():
            if abs(adjustment) > 0.1:  # Only apply significant adjustments
                # Send OSC command to mixer
                self._send_fader_command(ch_id, adjustment)
                applied += 1
        
        self._send_status("auto_balance_applied", {
            "applied_count": applied,
            "total_count": len(self.static_balancer.balance_result)
        })
        
        return True
    
    def cancel_auto_balance(self):
        """Cancel auto balance"""
        self.static_balancer.is_collecting = False
        self.mode = OperationMode.STOPPED
        self._send_status("auto_balance_cancelled")
    
    def _on_metrics_update(self, all_metrics: Dict[int, ChannelMetrics]):
        """Handle metrics update in real-time mode"""
        import json
        import time
        
        self.latest_metrics = all_metrics
        
        # CRITICAL: Only process metrics if in REALTIME mode
        if self.mode != OperationMode.REALTIME:
            logger.info(f"⚠️ Ignoring metrics update - not in REALTIME mode (current: {self.mode.value})")
            return
        
        # Warm-up period: Don't apply adjustments immediately after start
        # This allows POST-FADER LUFS to stabilize after fader reset
        import time
        if self.realtime_start_time is not None:
            elapsed = time.time() - self.realtime_start_time
            if elapsed < self.warmup_period_seconds:
                logger.debug(f"Warm-up period: {elapsed:.1f}s/{self.warmup_period_seconds}s - collecting metrics only")
                # Still update latest_metrics for status display, but don't apply adjustments
                return
        
        # #region agent log
        try:
            with open(DEBUG_LOG_PATH, "a") as f:
                f.write(json.dumps({
                    "sessionId": "debug-session",
                    "runId": "run1",
                    "hypothesisId": "B",
                    "location": "controller.py:226",
                    "message": "_on_metrics_update entry",
                    "data": {
                        "metrics_count": len(all_metrics),
                        "channel_ids": list(all_metrics.keys())[:10],
                        "selected_channels": self.selected_channels[:10],
                        "has_mixer_client": self.mixer_client is not None
                    },
                    "timestamp": int(time.time() * 1000)
                }) + "\n")
        except: pass
        # #endregion
        
        logger.info(f"_on_metrics_update called with {len(all_metrics)} channels")
        
        # Log channel IDs for debugging
        if all_metrics:
            sample_ids = list(all_metrics.keys())[:5]
            logger.info(f"Sample channel IDs in metrics: {sample_ids}, selected channels: {self.selected_channels[:5]}")
        
        # Check if we have selected channels
        if not self.selected_channels:
            logger.warning("No channels selected for auto fader")
            return
        
        # Check if mixer client is available
        if not self.mixer_client:
            logger.warning("Mixer client not available")
            return
        
        logger.info(f"Processing metrics for {len(self.selected_channels)} selected channels")
        
        # #region agent log
        try:
            matching_channels = [ch_id for ch_id in all_metrics.keys() if ch_id in self.selected_channels]
            with open(DEBUG_LOG_PATH, "a") as f:
                f.write(json.dumps({
                    "sessionId": "debug-session",
                    "runId": "run1",
                    "hypothesisId": "B",
                    "location": "controller.py:250",
                    "message": "Channel matching check",
                    "data": {
                        "matching_channels_count": len(matching_channels),
                        "matching_channels": matching_channels[:10],
                        "all_metric_ids": list(all_metrics.keys())[:10],
                        "selected_channels_sample": self.selected_channels[:10]
                    },
                    "timestamp": int(time.time() * 1000)
                }) + "\n")
        except: pass
        # #endregion
        
        # Analyze metrics and collect band energy for spectral masking
        channel_band_energy = {}
        for ch_id, metrics in all_metrics.items():
            if ch_id not in self.selected_channels:
                continue
            
            # Acoustic analysis
            features = self.acoustic_analyzer.analyze(metrics)
            self.latest_features[ch_id] = features
            
            # Collect band energy for spectral masking
            if self.spectral_masking_enabled and features.band_energy:
                channel_band_energy[ch_id] = features.band_energy
        
        # Calculate adjustments - only for selected channels
        # CRITICAL: Compensate for POST-FADER tap by subtracting current fader position
        # This prevents feedback loop where lowering fader makes LUFS drop, causing more lowering
        
        # Activity threshold - channels below this are considered "silent" and ignored
        # Using unified threshold from activity_detector module
        
        current_levels = {}
        inactive_channels = []
        
        now_ts = time.time()
        for ch, m in all_metrics.items():
            if ch not in self.selected_channels:
                continue
            
            self._apply_adaptive_noise_gate(ch, m, now_ts)

            # CRITICAL: Check if channel is active (has real signal)
            # Use momentary LUFS for activity detection
            if m.lufs_momentary < ACTIVITY_THRESHOLD_LUFS:
                inactive_channels.append(ch)
                continue  # Skip inactive channels - no adjustments needed
            
            # Get current fader position
            current_fader_db = self.mixer_client.get_channel_fader(ch) if self.mixer_client else 0.0
            if current_fader_db is None:
                current_fader_db = 0.0
            
            # #region agent log
            try:
                import json
                import time
                with open(DEBUG_LOG_PATH, "a") as f:
                    f.write(json.dumps({
                        "sessionId": "debug-session",
                        "runId": "run1",
                        "hypothesisId": "E",
                        "location": "controller.py:_on_metrics_update",
                        "message": "Reading fader for compensation",
                        "data": {
                            "channel": ch,
                            "current_fader_db": current_fader_db,
                            "measured_lufs_momentary": m.lufs_momentary,
                            "measured_lufs_short": m.lufs_short_term,
                            "measured_rms": m.rms_level
                        },
                        "timestamp": int(time.time() * 1000)
                    }) + "\n")
            except: pass
            # #endregion
            
            # ADAPTIVE METRIC SELECTION: Choose optimal LUFS metric based on instrument type
            instrument_type = self.instrument_types.get(ch, 'unknown')
            
            # Impulsive instruments (drums) - use RMS for fast transient response
            IMPULSIVE = {'kick', 'snare', 'toms', 'drums'}
            # Semi-impulsive (cymbals) - use short-term for pattern averaging
            SEMI_IMPULSIVE = {'hihat', 'ride', 'overhead', 'room', 'percussion'}
            # Sustained (vocals, bass, etc) - use momentary LUFS
            
            if instrument_type in IMPULSIVE:
                # RMS level for drums - captures transient energy
                measured_lufs = m.rms_level
                metric_used = "RMS"
            elif instrument_type in SEMI_IMPULSIVE:
                # Short-term LUFS for cymbals - averages playing pattern
                measured_lufs = m.lufs_short_term
                metric_used = "Short"
            else:
                # Momentary LUFS for sustained instruments (default)
                measured_lufs = m.lufs_momentary
                metric_used = "Mom"
            
            integrated_lufs = self.integrated_lufs.update(ch, measured_lufs, now_ts)
            self.latest_integrated_levels[ch] = integrated_lufs
            lufs_for_control = integrated_lufs if self.controller_mode == "mvp" else measured_lufs

            # Compensate POST-FADER measurement to get virtual PRE-FADER LUFS
            # Model: POST_FADER = PRE_FADER + FADER_GAIN
            # Therefore: PRE_FADER = POST_FADER - FADER_GAIN
            # If fader = -6 dB and POST = -25 LUFS, then PRE = -25 - (-6) = -19 LUFS
            compensated_lufs = lufs_for_control - current_fader_db
            current_levels[ch] = compensated_lufs
            
            # #region agent log
            try:
                import json
                import time
                with open(DEBUG_LOG_PATH, "a") as f:
                    f.write(json.dumps({
                        "sessionId": "debug-session",
                        "runId": "run1",
                        "hypothesisId": "E",
                        "location": "controller.py:_on_metrics_update",
                        "message": "Compensation calculated",
                        "data": {
                            "channel": ch,
                            "instrument_type": instrument_type,
                            "measured_lufs": measured_lufs,
                            "integrated_lufs_3s": integrated_lufs,
                            "current_fader_db": current_fader_db,
                            "compensated_lufs": compensated_lufs
                        },
                        "timestamp": int(time.time() * 1000)
                    }) + "\n")
            except: pass
            # #endregion
            
            logger.info(
                f"Ch{ch} ({instrument_type}): {metric_used}={measured_lufs:.2f}, "
                f"integrated={integrated_lufs:.2f}, fader={current_fader_db:.2f}, comp={compensated_lufs:.2f}"
            )
        
        # Log inactive channels
        if inactive_channels:
            logger.info(f"⏸️ Skipping {len(inactive_channels)} inactive channels (below {ACTIVITY_THRESHOLD_LUFS} dB): {inactive_channels[:10]}{'...' if len(inactive_channels) > 10 else ''}")
        
        # ====== BLEED DETECTION + COMPENSATION ======
        # Channel stays in processing; we use compensated level (own signal) for balance
        all_channel_centroids = {}
        for ch, m in all_metrics.items():
            if ch in current_levels:
                all_channel_centroids[ch] = m.spectral_centroid

        bleed_channels = []
        self.latest_bleed_info = {}
        for ch in list(current_levels.keys()):
            m = all_metrics[ch]
            raw_lufs = current_levels[ch]

            bleed_info = self.bleed_detector.detect_bleed(
                channel_id=ch,
                current_lufs=raw_lufs,
                spectral_centroid=m.spectral_centroid,
                all_channel_levels=current_levels,
                all_channel_centroids=all_channel_centroids,
                all_channel_metrics=all_metrics,
            )

            if self.bleed_protection_enabled and bleed_info.bleed_ratio > 0 and bleed_info.bleed_source_channel:
                source_m = all_metrics.get(bleed_info.bleed_source_channel)
                compensated = get_compensated_level(
                    raw_lufs=raw_lufs,
                    bleed_info=bleed_info,
                    channel_metrics=m,
                    source_metrics=source_m,
                    compensation_factor_db=self.bleed_compensation_factor_db,
                    compensation_mode=self.bleed_compensation_mode,
                )
                current_levels[ch] = compensated
                bleed_channels.append((ch, bleed_info.bleed_source_channel, bleed_info.bleed_ratio))
            else:
                compensated = raw_lufs

            # Keep latest bleed info for status/UI (channel never excluded)
            self.latest_bleed_info[ch] = {
                'raw_lufs': raw_lufs,
                'compensated_lufs': current_levels[ch],
                'bleed_ratio': bleed_info.bleed_ratio,
                'source': bleed_info.bleed_source_channel,
            }

        if bleed_channels:
            logger.info(f"Bleed compensated in {len(bleed_channels)} channels (no exclusion):")
            for ch, source, ratio in bleed_channels[:5]:
                instrument = self.instrument_types.get(ch, 'unknown')
                source_instrument = self.instrument_types.get(source, 'unknown') if source else 'unknown'
                logger.info(f"   Ch{ch} ({instrument}) <- bleed from Ch{source} ({source_instrument}), ratio={ratio:.1%}")
        
        # #region agent log
        try:
            import json
            import time
            with open(DEBUG_LOG_PATH, "a") as f:
                f.write(json.dumps({
                    "sessionId": "debug-session",
                    "runId": "run1",
                    "hypothesisId": "B",
                    "location": "controller.py:240",
                    "message": "Filtering metrics for selected channels",
                    "data": {
                        "current_levels_count": len(current_levels),
                        "current_levels": dict(list(current_levels.items())[:5]),
                        "all_metrics_channels": list(all_metrics.keys())[:10],
                        "selected_channels_sample": self.selected_channels[:10],
                        "inactive_channels": inactive_channels[:10],
                        "bleed_channels": [(ch, src) for ch, src, _ in bleed_channels[:5]]
                    },
                    "timestamp": int(time.time() * 1000)
                }) + "\n")
        except: pass
        # #endregion
        
        if not current_levels:
            logger.debug("No metrics for selected channels")
            return
        
        # Detect manual fader override: if fader position differs from what we last set (and we didn't send recently), freeze channel
        import time as time_module
        now = time_module.time()
        for ch in list(self.channel_freeze_until.keys()):
            if self.channel_freeze_until[ch] <= now:
                del self.channel_freeze_until[ch]
        if self.mixer_client and not self.automation_frozen:
            for ch in list(current_levels.keys()):
                current_fader = self.mixer_client.get_channel_fader(ch)
                if current_fader is None:
                    continue
                last_sent = self._last_sent_fader.get(ch)
                last_time = self._last_send_time.get(ch, 0)
                if last_sent is not None and (now - last_time) > 2.0:
                    if abs(current_fader - last_sent) > 0.5:
                        self.channel_freeze_until[ch] = now + self.freeze_cooldown_seconds
                        logger.info(f"Channel {ch}: manual override detected (fader {current_fader:.1f} vs sent {last_sent:.1f}), freeze {self.freeze_cooldown_seconds}s")
        
        # Calculate adjustments using selected controller
        if self.controller_mode in {'pid', 'mvp'}:
            # PID controller: need target levels for each channel
            target_levels = {}
            for ch_id in current_levels.keys():
                if ch_id in self.reference_channels:
                    # Reference channels: keep current level (no adjustment)
                    target_levels[ch_id] = current_levels[ch_id]
                else:
                    target_levels[ch_id] = self._target_lufs_for_channel(ch_id)
            
            # Calculate time delta (use update interval from metrics receiver)
            dt = getattr(self.metrics_receiver, 'update_interval', 0.1) if self.metrics_receiver else 0.1
            
            adjustments = self.pid_controller.calculate_adjustments(
                current_levels=current_levels,
                target_levels=target_levels,
                dt=dt
            )
            logger.info(f"{self.controller_mode.upper()} PID controller: calculated {len(adjustments)} adjustments")
        else:
            # Dynamic mixer (default)
            adjustments = self.dynamic_mixer.calculate_adjustments(current_levels)
            logger.info(f"Dynamic mixer: calculated {len(adjustments)} adjustments")
        
        logger.info(f"Calculated {len(adjustments)} adjustments for channels: {list(adjustments.keys())[:10]}")
        
        # Apply ducking if enabled
        if self.ducking_enabled and self.vocal_activity_detector and self.vocal_channels:
            # Get vocal levels for activity detection
            vocal_levels = {}
            for ch_id in self.vocal_channels:
                if ch_id in current_levels:
                    vocal_levels[ch_id] = current_levels[ch_id]
            
            if vocal_levels:
                # Update vocal activity detector
                dt = getattr(self.metrics_receiver, 'update_interval', 0.1) if self.metrics_receiver else 0.1
                vocal_activity = self.vocal_activity_detector.update(vocal_levels, dt)
                
                # Check if any vocal is active
                is_vocal_active = self.vocal_activity_detector.is_any_vocal_active(self.vocal_channels)
                activity_level = self.vocal_activity_detector.get_activity_level(self.vocal_channels)
                
                if is_vocal_active:
                    # Apply ducking to non-vocal channels
                    duck_targets = [ch_id for ch_id in adjustments.keys() 
                                  if ch_id not in self.vocal_channels and ch_id not in self.reference_channels]
                    
                    # Calculate ducking amount (scaled by activity level for smooth transition)
                    duck_amount = self.ducking_amount_db * activity_level
                    
                    for ch_id in duck_targets:
                        # Add ducking to existing adjustment
                        adjustments[ch_id] = adjustments.get(ch_id, 0.0) + duck_amount
                        logger.debug(f"Ducking Ch{ch_id}: {duck_amount:.2f} dB (vocal active, level={activity_level:.2f})")

        # Priority-based overload protection
        hierarchical_cuts = self.hierarchical_mixer.get_cuts(
            current_levels=current_levels,
            selected_channels=self.selected_channels,
            reference_channels=self.reference_channels,
            channel_priorities=self.channel_priorities,
        )
        for ch_id, cut_db in hierarchical_cuts.items():
            adjustments[ch_id] = adjustments.get(ch_id, 0.0) + cut_db
        if hierarchical_cuts:
            logger.info(f"Hierarchical mixer applied cuts to {len(hierarchical_cuts)} channels")
        
        # Apply spectral masking if enabled
        if self.spectral_masking_enabled and self.spectral_masking_detector and channel_band_energy:
            # Identify vocal and background channels
            vocal_chs = self.vocal_channels if self.vocal_channels else []
            background_chs = [ch_id for ch_id in adjustments.keys() 
                            if ch_id not in vocal_chs and ch_id not in self.reference_channels]
            
            if vocal_chs and background_chs:
                # Detect conflicts and get EQ adjustments
                eq_adjustments = self.spectral_masking_detector.detect_conflicts(
                    vocal_channels=vocal_chs,
                    background_channels=background_chs,
                    channel_band_energy=channel_band_energy
                )
                
                # Apply EQ adjustments via mixer client
                for eq_adj in eq_adjustments:
                    if self.mixer_client and hasattr(self.mixer_client, 'set_eq_band'):
                        # Find closest EQ band to target frequency
                        # Wing has 4 parametric bands (1-4) plus low/high shelf
                        # For 1-3 kHz range, use band 2 or 3 (typically 1-2 kHz and 2-4 kHz)
                        # Use band 2 for 1-2 kHz, band 3 for 2-4 kHz
                        if 1000 <= eq_adj.frequency_hz < 2000:
                            band_num = 2
                        elif 2000 <= eq_adj.frequency_hz <= 4000:
                            band_num = 3
                        else:
                            band_num = 2  # Default to band 2
                        
                        try:
                            # Set frequency, gain, and Q
                            self.mixer_client.set_eq_band(
                                channel=eq_adj.channel_id,
                                band=band_num,
                                freq=eq_adj.frequency_hz,
                                gain=eq_adj.gain_db,
                                q=eq_adj.q_factor
                            )
                            logger.info(f"Spectral masking: Ch{eq_adj.channel_id} EQ band {band_num} "
                                      f"cut {eq_adj.gain_db:.1f} dB at {eq_adj.frequency_hz:.0f} Hz")
                        except Exception as e:
                            logger.error(f"Failed to apply EQ adjustment for Ch{eq_adj.channel_id}: {e}")
                    elif self.spectral_masking_enabled and not self._warned_missing_eq_support:
                        logger.warning("Spectral masking enabled but mixer client has no set_eq_band support")
                        self._warned_missing_eq_support = True

                # Optional pan separation if stereo metrics are available.
                self._apply_pan_separation(vocal_chs, background_chs, all_metrics)

        # Pan separation can also run independently from spectral masking.
        if self.pan_separation_enabled and self.vocal_channels:
            pan_background = [
                ch_id for ch_id in adjustments.keys()
                if ch_id not in self.vocal_channels and ch_id not in self.reference_channels
            ]
            self._apply_pan_separation(self.vocal_channels, pan_background, all_metrics)
        
        # #region agent log
        try:
            import json
            import time
            with open(DEBUG_LOG_PATH, "a") as f:
                f.write(json.dumps({
                    "sessionId": "debug-session",
                    "runId": "run1",
                    "hypothesisId": "B",
                    "location": "controller.py:245",
                    "message": "Adjustments calculated",
                    "data": {
                        "adjustments_count": len(adjustments),
                        "adjustments": dict(list(adjustments.items())[:5])
                    },
                    "timestamp": int(time.time() * 1000)
                }) + "\n")
        except: pass
        # #endregion
        
        # Apply adjustments (skip frozen channels)
        adjustment_threshold = 0.1  # Minimum adjustment to apply
        applied_count = 0
        for ch_id, adjustment in adjustments.items():
            if ch_id not in self.selected_channels:
                continue
            if self.automation_frozen:
                continue
            if ch_id in self.channel_freeze_until and time_module.time() < self.channel_freeze_until[ch_id]:
                logger.debug(f"Channel {ch_id}: skipped (frozen)")
                continue
            if abs(adjustment) > adjustment_threshold:
                self._send_fader_command(ch_id, adjustment)
                self.latest_adjustments[ch_id] = adjustment
                applied_count += 1
            else:
                logger.debug(f"Channel {ch_id}: adjustment {adjustment:+.2f} dB below threshold {adjustment_threshold}")
        
        # #region agent log
        try:
            import json
            import time
            with open(DEBUG_LOG_PATH, "a") as f:
                f.write(json.dumps({
                    "sessionId": "debug-session",
                    "runId": "run1",
                    "hypothesisId": "B",
                    "location": "controller.py:260",
                    "message": "Fader commands applied",
                    "data": {
                        "applied_count": applied_count,
                        "total_adjustments": len(adjustments),
                        "threshold": adjustment_threshold
                    },
                    "timestamp": int(time.time() * 1000)
                }) + "\n")
        except: pass
        # #endregion
        
        if applied_count > 0:
            logger.info(f"Applied {applied_count} fader adjustments out of {len(adjustments)} calculated")
        elif adjustments:
            logger.debug(f"Calculated {len(adjustments)} adjustments but all below threshold {adjustment_threshold} dB")
        
        # Send status update (include bleed info for UI: raw/compensated/bleed_ratio/source)
        bleed_info = getattr(self, 'latest_bleed_info', {})
        channels_payload = {}
        for ch, m in all_metrics.items():
            ch_data = {
                "lufs": m.lufs_momentary,
                "integrated_lufs_3s": self.latest_integrated_levels.get(ch, m.lufs_momentary),
                "is_active": m.is_active,
                "correction": adjustments.get(ch, 0.0),
            }
            bi = bleed_info.get(ch, {})
            if bi.get('bleed_ratio', 0) > 0:
                ch_data["raw_lufs"] = bi.get("raw_lufs")
                ch_data["compensated_lufs"] = bi.get("compensated_lufs")
                ch_data["bleed_ratio"] = bi.get("bleed_ratio")
                ch_data["bleed_source"] = bi.get("source")
            channels_payload[ch] = ch_data
        self._send_status("levels_update", {
            "channels": channels_payload,
            "bleed_protection_enabled": self.bleed_protection_enabled,
            "controller_mode": self.controller_mode,
            "hierarchical_mix_enabled": self.hierarchical_mixer.enabled,
        })
    
    def _on_metrics_update_auto_balance(self, all_metrics: Dict[int, ChannelMetrics]):
        """Handle metrics update during auto balance collection"""
        for ch_id, metrics in all_metrics.items():
            if ch_id in self.selected_channels:
                self.static_balancer.collect_sample(
                    ch_id,
                    metrics.lufs_momentary,
                    metrics.spectral_centroid,
                    metrics.is_active
                )
    
    def _send_fader_command(self, channel_id: int, adjustment_db: float):
        """Send fader command to mixer via OSC"""
        try:
            if self.automation_frozen:
                logger.debug("Fader command skipped: automation frozen")
                return
            if not self.mixer_client:
                logger.warning("Mixer client not available for fader command")
                return
            
            # Get current fader value
            current_db = self.mixer_client.get_channel_fader(channel_id)
            if current_db is None:
                current_db = 0.0  # Default to 0dB if unknown
                logger.debug(f"Channel {channel_id}: current fader value unknown, using 0.0 dB")
            
            # #region agent log
            try:
                import json
                import time
                with open(DEBUG_LOG_PATH, "a") as f:
                    f.write(json.dumps({
                        "sessionId": "debug-session",
                        "runId": "run1",
                        "hypothesisId": "F",
                        "location": "controller.py:_send_fader_command",
                        "message": "Applying adjustment",
                        "data": {
                            "channel_id": channel_id,
                            "current_db": current_db,
                            "adjustment_db": adjustment_db
                        },
                        "timestamp": int(time.time() * 1000)
                    }) + "\n")
            except: pass
            # #endregion
            
            # Apply adjustment (relative change with smoothing)
            # Use smoothing factor to prevent oscillation and achieve gradual convergence
            smoothing_factor = 0.30  # Apply 30% of adjustment per update (faster response)
            new_db = current_db + (adjustment_db * smoothing_factor)
            
            # PROTECTION: Limit maximum fader position based on adjustment direction
            # If we're boosting a lot, the channel is probably weak - don't go to maximum
            MAX_FADER_FOR_WEAK_SIGNAL = 6.0  # dB - maximum fader for weak signals
            if adjustment_db > 1.5 and new_db > MAX_FADER_FOR_WEAK_SIGNAL:
                new_db = MAX_FADER_FOR_WEAK_SIGNAL
                logger.debug(f"Channel {channel_id}: Limiting fader to {MAX_FADER_FOR_WEAK_SIGNAL} dB (weak signal protection)")
            
            new_db = max(-144.0, min(10.0, new_db))  # Wing fader range: -144..10 dB
            
            # #region agent log
            try:
                import json
                import time
                with open(DEBUG_LOG_PATH, "a") as f:
                    f.write(json.dumps({
                        "sessionId": "debug-session",
                        "runId": "run1",
                        "hypothesisId": "F",
                        "location": "controller.py:_send_fader_command",
                        "message": "Calculated new fader value",
                        "data": {
                            "channel_id": channel_id,
                            "current_db": current_db,
                            "adjustment_db": adjustment_db,
                            "smoothing_factor": smoothing_factor,
                            "new_db": new_db
                        },
                        "timestamp": int(time.time() * 1000)
                    }) + "\n")
            except: pass
            # #endregion
            
            # Send OSC command
            self.mixer_client.set_channel_fader(channel_id, new_db)
            import time
            self._last_sent_fader[channel_id] = new_db
            self._last_send_time[channel_id] = time.time()
            logger.info(f"Channel {channel_id}: {current_db:.2f} dB -> {new_db:.2f} dB (adjustment: {adjustment_db:+.2f} dB)")
        except Exception as e:
            logger.error(f"Error sending fader command for channel {channel_id}: {e}", exc_info=True)
    
    def stop(self):
        """Stop all operations"""
        self.stop_realtime()
        
        if self.metrics_receiver:
            self.metrics_receiver.stop()
        
        self.cpp_bridge.stop_cpp_core()
        
        self.mode = OperationMode.STOPPED
    
    def get_status(self) -> Dict:
        """Get current status"""
        return {
            "mode": self.mode.value,
            "controller_mode": self.controller_mode,
            "cpp_core_running": self.cpp_bridge.is_running(),
            "selected_channels": self.selected_channels,
            "reference_channels": self.reference_channels,
            "genre": self.current_genre.value,
            "metrics_count": len(self.latest_metrics),
            "bleed_protection_enabled": self.bleed_protection_enabled,
            "bleed_info": self.latest_bleed_info,
            "hierarchical_mix_enabled": self.hierarchical_mixer.enabled,
            "adaptive_noise_gate_enabled": self.adaptive_noise_gate_enabled,
        }
