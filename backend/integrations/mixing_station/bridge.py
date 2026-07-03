"""Bridge existing AutoFOH SafetyDecision objects to AutomixCorrection."""

from __future__ import annotations

from typing import Any, Dict, Iterable, List, Optional

from .config import normalize_console_profile
from .models import AutomixCorrection


def corrections_from_safety_decision(
    decision: Any,
    *,
    console_profile: str,
    mode: str = "offline_visualization",
    channel_names: Optional[Dict[int, str]] = None,
    source_metrics: Optional[Dict[str, Any]] = None,
    dry_run: bool = True,
) -> List[AutomixCorrection]:
    """Translate an AutoFOH SafetyDecision into one or more corrections."""
    action = getattr(decision, "action", None)
    if action is None:
        return []

    profile = normalize_console_profile(console_profile)
    status = _safety_status(decision)
    metrics = dict(source_metrics or {})
    channel_names = channel_names or {}
    reason = str(getattr(action, "reason", ""))
    confidence = float(metrics.pop("confidence", 1.0) if "confidence" in metrics else 1.0)

    def correction(
        *,
        channel_id: int,
        parameter: str,
        value: Any,
        unit: str,
        strip_type: str = "input",
        previous_value: Any = None,
        extra_metrics: Optional[Dict[str, Any]] = None,
    ) -> AutomixCorrection:
        combined_metrics = dict(metrics)
        if extra_metrics:
            combined_metrics.update(extra_metrics)
        return AutomixCorrection(
            console_profile=profile,
            mode=mode,
            channel_index=max(0, int(channel_id) - 1),
            channel_name=channel_names.get(int(channel_id)),
            strip_type=strip_type,
            parameter=parameter,
            value=value,
            value_unit=unit,
            previous_value=previous_value,
            reason=reason,
            confidence=confidence,
            source_metrics=combined_metrics,
            safety_status=status,
            dry_run=dry_run,
        )

    name = action.__class__.__name__
    if name == "ChannelFaderMove":
        return [
            correction(
                channel_id=action.channel_id,
                parameter="fader",
                value=action.target_db,
                unit="db",
                extra_metrics={"delta_db": getattr(action, "delta_db", 0.0)},
            )
        ]
    if name == "ChannelGainMove":
        return [
            correction(
                channel_id=action.channel_id,
                parameter="gain",
                value=action.target_db,
                unit="db",
            )
        ]
    if name == "PanAdjust":
        return [
            correction(
                channel_id=action.channel_id,
                parameter="pan",
                value=action.pan,
                unit="normalized",
            )
        ]
    if name == "HighPassAdjust":
        return [
            correction(
                channel_id=action.channel_id,
                parameter="hpf.enabled",
                value=bool(action.enabled),
                unit="bool",
            ),
            correction(
                channel_id=action.channel_id,
                parameter="hpf.frequency",
                value=action.freq_hz,
                unit="hz",
            ),
        ]
    if name in {"ChannelEQMove", "EmergencyFeedbackNotch"}:
        band = int(action.band)
        return [
            correction(
                channel_id=action.channel_id,
                parameter=f"peq.band{band}.frequency",
                value=action.freq_hz,
                unit="hz",
            ),
            correction(
                channel_id=action.channel_id,
                parameter=f"peq.band{band}.gain",
                value=action.gain_db,
                unit="db",
            ),
            correction(
                channel_id=action.channel_id,
                parameter=f"peq.band{band}.q",
                value=action.q,
                unit="q",
            ),
        ]
    if name == "SendLevelAdjust":
        return [
            correction(
                channel_id=action.channel_id,
                parameter=f"send.aux{int(action.send_bus)}.level",
                value=action.level_db,
                unit="db",
            )
        ]
    if name == "CompressorAdjust":
        return [
            correction(
                channel_id=action.channel_id,
                parameter="compressor.threshold",
                value=action.threshold_db,
                unit="db",
            ),
            correction(
                channel_id=action.channel_id,
                parameter="compressor.ratio",
                value=action.ratio,
                unit="normalized",
            ),
        ]
    if name == "CompressorMakeupAdjust":
        return [
            correction(
                channel_id=action.channel_id,
                parameter="compressor.makeup_gain",
                value=action.makeup_db,
                unit="db",
            )
        ]
    if name == "GateAdjust":
        return [
            correction(
                channel_id=action.channel_id,
                parameter="gate.threshold",
                value=action.threshold_db,
                unit="db",
            )
        ]
    if name == "BusFaderMove":
        return [
            correction(
                channel_id=action.bus_id,
                strip_type="bus",
                parameter="fader",
                value=action.target_db,
                unit="db",
            )
        ]
    if name == "DCAFaderMove":
        return [
            correction(
                channel_id=action.dca_id,
                strip_type="dca",
                parameter="fader",
                value=action.target_db,
                unit="db",
            )
        ]
    if name == "MasterFaderMove":
        return [
            correction(
                channel_id=action.main_id,
                strip_type="main",
                parameter="fader",
                value=action.target_db,
                unit="db",
            )
        ]
    return []


def _safety_status(decision: Any) -> str:
    if not bool(getattr(decision, "allowed", False)):
        return "blocked"
    if bool(getattr(decision, "bounded", False)):
        return "clamped"
    return "allowed"
