"""Controller for arrangement-aware level automation."""

from __future__ import annotations

import logging
import threading
import time
from typing import Any, Callable, Dict, Iterable, Optional

try:
    from live_apply import LiveApplyRequest, LiveApplyService
except ImportError:
    from backend.live_apply import LiveApplyRequest, LiveApplyService

from .activity_detector import ActivityDetector
from .arrangement_density import ArrangementDensityAnalyzer
from .automation_logger import AutomationDecisionLogger
from .channel_role_classifier import ChannelRoleClassifier
from .level_automation_planner import LevelAutomationPlanner
from .masking_analyzer import MaskingAnalyzer
from .mix_priority_engine import MixPriorityEngine
from .safety_limiter import SafetyLimiter
from .section_detector import SectionDetector

logger = logging.getLogger(__name__)


class ArrangementAutomationController:
    """Orchestrates arrangement-aware analysis, planning, safety, and optional OSC apply."""

    def __init__(
        self,
        mixer_client=None,
        config: Optional[Dict[str, Any]] = None,
        audio_capture=None,
        status_callback: Optional[Callable[[Dict[str, Any]], None]] = None,
        live_apply_service: Optional[LiveApplyService] = None,
    ):
        self.mixer_client = mixer_client
        self.config = config or {}
        self.audio_capture = audio_capture
        self.status_callback = status_callback
        self.live_apply_service = live_apply_service or LiveApplyService()

        self.cfg = self._get_cfg(self.config)
        role_limits = self.cfg.get("role_limits", {})
        self.role_classifier = ChannelRoleClassifier(role_limits=role_limits)
        self.activity_detector = ActivityDetector(
            min_active_db=float(self.cfg.get("activity_threshold_db", -50.0)),
            noise_margin_db=float(self.cfg.get("noise_margin_db", 10.0)),
            attack_hold_sec=float(self.cfg.get("activity_attack_hold_sec", 0.3)),
            release_hold_sec=float(self.cfg.get("activity_release_hold_sec", 1.0)),
        )
        self.density_analyzer = ArrangementDensityAnalyzer(
            max_channels=int(self.cfg.get("max_channels", 40))
        )
        self.section_detector = SectionDetector(
            hold_time_sec=float(self.cfg.get("section_hold_time_sec", 3.0))
        )
        safety_cfg = dict(self.cfg)
        safety_cfg["role_limits"] = role_limits
        self.safety_limiter = SafetyLimiter(safety_cfg)
        self.masking_analyzer = MaskingAnalyzer()
        self.priority_engine = MixPriorityEngine()
        self.planner = LevelAutomationPlanner()
        self.decision_logger = AutomationDecisionLogger(
            enabled=bool(self.cfg.get("log_decisions", True))
        )
        self.update_interval_sec = float(self.cfg.get("update_interval_ms", 250)) / 1000.0

        self.enabled = bool(self.cfg.get("enabled", True))
        self.live_apply_enabled = bool(self.cfg.get("live_apply_enabled", False))
        self.analysis_only_mode = bool(self.cfg.get("analysis_only_mode", True))
        self.confirm_live_apply = bool(self.cfg.get("confirm_live_apply", False))
        self.fader_ceiling_db = float(self.cfg.get("fader_ceiling_db", 0.0))
        self.channels: list[int] = []
        self.channel_mapping: Dict[int, int] = {}
        self.roles = {}
        self.base_fader_positions: Dict[int, float] = {}
        self.last_state = self._empty_state()
        self.started_at: Optional[float] = None
        self._stop_event = threading.Event()
        self._thread: Optional[threading.Thread] = None

    def start(
        self,
        channels: Iterable[int],
        channel_mapping: Optional[Dict[int, int]] = None,
        channel_names: Optional[Dict[int, str]] = None,
        settings: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Start analysis for selected channels without taking over existing automation."""
        if settings:
            self.live_apply_enabled = bool(
                settings.get("live_apply_enabled", self.live_apply_enabled)
            )
            self.analysis_only_mode = bool(
                settings.get("analysis_only_mode", self.analysis_only_mode)
            )
            self.enabled = bool(settings.get("enabled", self.enabled))
            self.confirm_live_apply = self._resolve_confirm_live_apply(
                enabled=self.live_apply_enabled,
                analysis_only_mode=self.analysis_only_mode,
                explicit=settings.get("confirm_live_apply"),
            )

        self.channels = [int(ch) for ch in channels]
        raw_mapping = channel_mapping or {}
        self.channel_mapping = (
            {int(k): int(v) for k, v in raw_mapping.items()}
            if raw_mapping
            else {ch: ch for ch in self.channels}
        )
        explicit_confirm = None
        if settings and "confirm_live_apply" in settings:
            explicit_confirm = settings.get("confirm_live_apply")
        elif "confirm_live_apply" in self.cfg:
            explicit_confirm = self.cfg.get("confirm_live_apply")
        self.confirm_live_apply = self._resolve_confirm_live_apply(
            enabled=self.live_apply_enabled,
            analysis_only_mode=self.analysis_only_mode,
            explicit=explicit_confirm,
        )
        names = channel_names or self._read_channel_names(self.channel_mapping.values())
        self.roles = {
            mixer_ch: self.role_classifier.classify(
                mixer_ch,
                names.get(mixer_ch, f"Ch {mixer_ch}"),
            )
            for mixer_ch in self.channel_mapping.values()
        }
        self.base_fader_positions = self._capture_base_faders(self.channel_mapping.values())
        self.started_at = time.time()
        self.last_state = self._empty_state()
        self.last_state.update({
            "enabled": self.enabled,
            "live_apply_enabled": self.live_apply_enabled,
            "analysis_only_mode": self.analysis_only_mode,
            "confirm_live_apply": self.confirm_live_apply,
            "roles": {ch: role.to_dict() for ch, role in self.roles.items()},
            "base_fader_positions": dict(self.base_fader_positions),
            "message": "arrangement automation started",
        })
        if bool(self.cfg.get("run_background_loop", True)):
            self.start_background_loop()
        return self.last_state

    def start_background_loop(self) -> None:
        """Start a lightweight analysis loop that consumes existing AudioCapture buffers."""
        if self._thread and self._thread.is_alive():
            return
        self._stop_event.clear()
        self._thread = threading.Thread(target=self._run_loop, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        """Stop analysis/apply and reset smoothing state."""
        self._stop_event.set()
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=1.0)
        self._thread = None
        self.channels = []
        self.roles = {}
        self.safety_limiter.reset()
        self.started_at = None
        self.last_state = self._empty_state()

    def _run_loop(self) -> None:
        logger.info("Arrangement automation loop started")
        while not self._stop_event.is_set():
            try:
                if self.channels:
                    self.process_once()
            except Exception as exc:
                logger.error("Arrangement automation loop error: %s", exc, exc_info=True)
            self._stop_event.wait(self.update_interval_sec)
        logger.info("Arrangement automation loop stopped")

    def process_once(
        self,
        metrics_by_channel: Optional[Dict[int, Dict[str, Any]]] = None,
        band_energy_by_channel: Optional[Dict[int, Dict[str, float]]] = None,
        timestamp: Optional[float] = None,
    ) -> Dict[str, Any]:
        """Run one analysis/planning cycle and optionally apply safe OSC fader offsets."""
        now = time.time() if timestamp is None else float(timestamp)
        metrics = metrics_by_channel or self._read_audio_metrics()
        band_energy = band_energy_by_channel or self._read_band_energy()

        activities = {}
        for audio_ch, mixer_ch in self.channel_mapping.items():
            data = metrics.get(mixer_ch) or metrics.get(audio_ch) or {}
            activities[mixer_ch] = self.activity_detector.update(
                channel_id=mixer_ch,
                rms_db=float(data.get("rms_db", data.get("lufs", -100.0))),
                peak_db=float(data.get("peak_db", -100.0)),
                short_term_loudness_db=float(
                    data.get("lufs", data.get("rms_db", -100.0))
                ),
                spectral_energy_db=float(
                    data.get("spectral_energy_db", data.get("rms_db", -100.0))
                ),
                noise_floor_db=float(data.get("noise_floor_db", -80.0)),
                timestamp=now,
            )

        density = self.density_analyzer.analyze(activities, self.roles, band_energy)
        section = self.section_detector.update(activities, self.roles, density, timestamp=now)
        priority = self.priority_engine.analyze(self.roles, activities, section)
        masking = self.masking_analyzer.analyze(
            self.roles,
            activities,
            band_energy,
            primary_channel_id=priority.primary_channel_id,
        )
        planned = self.planner.plan(self.roles, activities, density, section, masking, priority)
        safety = self.safety_limiter.limit(
            planned,
            self.roles,
            base_fader_positions=self.base_fader_positions,
            automation_enabled=self.enabled,
            live_apply_enabled=self.live_apply_enabled and not self.analysis_only_mode,
        )
        applied = self._apply_safe_offsets(safety.safe_offsets, safety.blocked_offsets)

        self.decision_logger.log_cycle(
            roles=self.roles,
            activities=activities,
            section=section.section,
            density_index=density.density_index,
            masking_sources=masking,
            planned_offsets=planned,
            safe_offsets=safety.safe_offsets,
            blocked_offsets=safety.blocked_offsets,
            applied_offsets=applied,
            safety_reasons=safety.reasons,
        )

        state = {
            "enabled": self.enabled,
            "live_apply_enabled": self.live_apply_enabled,
            "analysis_only_mode": self.analysis_only_mode,
            "current_section": section.section,
            "section_confidence": section.confidence,
            "arrangement_density_index": density.density_index,
            "density_label": density.density_label,
            "primary_source": priority.to_dict(),
            "active_channels_count": density.active_channel_count,
            "masking_sources": [item.to_dict() for item in masking],
            "planned_offsets": {ch: offset.to_dict() for ch, offset in planned.items()},
            "safe_offsets": dict(safety.safe_offsets),
            "applied_offsets": dict(applied),
            "blocked_offsets": dict(safety.blocked_offsets),
            "last_decision_reasons": dict(safety.reasons),
            "roles": {ch: role.to_dict() for ch, role in self.roles.items()},
            "activities": {ch: activity.to_dict() for ch, activity in activities.items()},
            "density": density.to_dict(),
            "section": section.to_dict(),
            "final_confidence": safety.final_confidence,
            "recent_decisions": self.decision_logger.recent(20),
        }
        self.last_state = state
        if self.status_callback:
            self.status_callback(state)
        return state

    def get_status(self) -> Dict[str, Any]:
        return dict(self.last_state)

    def set_live_apply(
        self,
        enabled: bool,
        analysis_only_mode: Optional[bool] = None,
        confirm_live_apply: Optional[bool] = None,
    ) -> Dict[str, Any]:
        was_applying = self.live_apply_enabled and not self.analysis_only_mode
        next_analysis_only = (
            self.analysis_only_mode
            if analysis_only_mode is None
            else bool(analysis_only_mode)
        )
        will_apply = bool(enabled) and not next_analysis_only
        if will_apply and not was_applying:
            self.safety_limiter.reset()
        self.live_apply_enabled = bool(enabled)
        if analysis_only_mode is not None:
            self.analysis_only_mode = bool(analysis_only_mode)
        self.confirm_live_apply = self._resolve_confirm_live_apply(
            enabled=self.live_apply_enabled,
            analysis_only_mode=self.analysis_only_mode,
            explicit=confirm_live_apply,
        )
        self.last_state["live_apply_enabled"] = self.live_apply_enabled
        self.last_state["analysis_only_mode"] = self.analysis_only_mode
        self.last_state["confirm_live_apply"] = self.confirm_live_apply
        return self.get_status()

    def _apply_safe_offsets(
        self,
        safe_offsets: Dict[int, float],
        blocked_offsets: Dict[int, Dict[str, Any]],
    ) -> Dict[int, float]:
        if self.analysis_only_mode or not self.live_apply_enabled or not self.enabled:
            return {}
        if not self.mixer_client or not getattr(self.mixer_client, "is_connected", False):
            return {}

        applied = {}
        for channel_id, offset_db in safe_offsets.items():
            if channel_id in blocked_offsets:
                continue
            base = self.base_fader_positions.get(channel_id, 0.0)
            # The ceiling is a maximum allowed target level (e.g., -4.6 dB).
            # Offset application can only raise values toward the ceiling, never
            # below it, so we must take the maximum of proposed vs ceiling.
            target = max(self.fader_ceiling_db, base + float(offset_db))
            try:
                result = self.live_apply_service.apply(
                    LiveApplyRequest(
                        source="arrangement_automation",
                        operation="set_fader",
                        channel=channel_id,
                        value=target,
                        dry_run=False,
                        confirm_live_apply=bool(self.confirm_live_apply),
                        current_value=base,
                        max_value=self.fader_ceiling_db,
                        metadata={
                            "offset_db": float(offset_db),
                            "base_fader_db": float(base),
                        },
                    ),
                    self.mixer_client,
                )
                if result.accepted and not result.dry_run:
                    confirmed_value = (
                        float(result.confirmed_value)
                        if result.confirmed_value is not None
                        else float(result.desired_value if result.desired_value is not None else target)
                    )
                    applied[channel_id] = round(confirmed_value, 4)
                else:
                    logger.info(
                        "Arrangement automation live apply blocked ch%s: %s",
                        channel_id,
                        result.blocked_reason or result.failure_reason or "not_accepted",
                    )
            except Exception as exc:
                logger.error("Arrangement automation OSC apply failed ch%s: %s", channel_id, exc)
        return applied

    def _read_audio_metrics(self) -> Dict[int, Dict[str, float]]:
        if not self.audio_capture or not getattr(self.audio_capture, "running", False):
            return {}
        metrics = {}
        for audio_ch, mixer_ch in self.channel_mapping.items():
            metrics[mixer_ch] = {
                "rms_db": self.audio_capture.get_rms(audio_ch),
                "peak_db": self.audio_capture.get_peak(audio_ch),
                "lufs": self.audio_capture.get_lufs(audio_ch),
            }
        return metrics

    def _read_band_energy(self) -> Dict[int, Dict[str, float]]:
        if not self.audio_capture or not getattr(self.audio_capture, "running", False):
            return {}
        bands = {}
        for audio_ch, mixer_ch in self.channel_mapping.items():
            spectrum = self.audio_capture.get_spectrum(audio_ch)
            freqs = spectrum.get("frequencies")
            magnitudes = spectrum.get("magnitude_db")
            if freqs is None or magnitudes is None:
                continue
            bands[mixer_ch] = self._coarse_bands(freqs, magnitudes)
        return bands

    @staticmethod
    def _resolve_confirm_live_apply(
        *,
        enabled: bool,
        analysis_only_mode: bool,
        explicit: Any,
    ) -> bool:
        if explicit is not None:
            return bool(explicit)
        return bool(enabled and not analysis_only_mode)

    @staticmethod
    def _coarse_bands(freqs, magnitudes) -> Dict[str, float]:
        def band(low: float, high: float) -> float:
            values = [
                float(mag)
                for freq, mag in zip(freqs, magnitudes)
                if low <= float(freq) < high
            ]
            return max(values) if values else -100.0

        return {
            "sub": band(20.0, 80.0),
            "bass": band(80.0, 250.0),
            "low_mid": band(250.0, 600.0),
            "mid": band(600.0, 1500.0),
            "high_mid": band(1500.0, 4000.0),
            "high": band(4000.0, 8000.0),
            "air": band(8000.0, 16000.0),
            "lf": band(20.0, 250.0),
            "lmf": band(250.0, 1000.0),
            "umf": band(1000.0, 4000.0),
            "hf": band(4000.0, 16000.0),
        }

    def _read_channel_names(self, mixer_channels: Iterable[int]) -> Dict[int, str]:
        names = {}
        channel_list = list(mixer_channels)
        if self.mixer_client and hasattr(self.mixer_client, "get_all_channel_names"):
            try:
                names.update(
                    self.mixer_client.get_all_channel_names(max(channel_list or [40]))
                )
            except Exception as exc:
                logger.debug("Could not read channel names from mixer: %s", exc)
        for channel_id in channel_list:
            if (
                channel_id not in names
                and self.mixer_client
                and hasattr(self.mixer_client, "get_channel_name")
            ):
                try:
                    names[channel_id] = self.mixer_client.get_channel_name(channel_id)
                except Exception:
                    names[channel_id] = f"Ch {channel_id}"
            names.setdefault(channel_id, f"Ch {channel_id}")
        return names

    def _capture_base_faders(self, mixer_channels: Iterable[int]) -> Dict[int, float]:
        positions = {}
        for channel_id in mixer_channels:
            value = None
            if self.mixer_client and hasattr(self.mixer_client, "get_channel_fader"):
                try:
                    value = self.mixer_client.get_channel_fader(channel_id)
                except Exception:
                    value = None
            positions[channel_id] = float(value) if value is not None else 0.0
        return positions

    @staticmethod
    def _get_cfg(config: Dict[str, Any]) -> Dict[str, Any]:
        automation = config.get("automation", {}) if config else {}
        return dict(
            config.get("arrangement_automation")
            or automation.get("arrangement_automation")
            or {}
        )

    @staticmethod
    def _empty_state() -> Dict[str, Any]:
        return {
            "enabled": False,
            "live_apply_enabled": False,
            "analysis_only_mode": True,
            "confirm_live_apply": False,
            "current_section": "unknown",
            "section_confidence": 0.0,
            "arrangement_density_index": 0.0,
            "density_label": "sparse",
            "primary_source": None,
            "active_channels_count": 0,
            "masking_sources": [],
            "planned_offsets": {},
            "applied_offsets": {},
            "blocked_offsets": {},
            "last_decision_reasons": {},
            "roles": {},
        }
