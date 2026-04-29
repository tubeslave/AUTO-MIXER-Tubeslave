"""Bridge between mix-agent recommendations and the live backend safety layer."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, Optional

from mix_agent.models import BackendBridgeResult, BackendChannelSnapshot, MixAction

REPO_ROOT = Path(__file__).resolve().parents[1]
BACKEND_DIR = REPO_ROOT / "backend"
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))


def _import_autofoh() -> Dict[str, Any]:
    from autofoh_models import RuntimeState
    from autofoh_safety import (
        ChannelEQMove,
        ChannelFaderMove,
        HighPassAdjust,
        PanAdjust,
    )

    return {
        "RuntimeState": RuntimeState,
        "ChannelEQMove": ChannelEQMove,
        "ChannelFaderMove": ChannelFaderMove,
        "HighPassAdjust": HighPassAdjust,
        "PanAdjust": PanAdjust,
    }


def _channel_lookup(
    action: MixAction,
    channel_map: Mapping[str, int] | None,
    snapshots: Mapping[int, BackendChannelSnapshot] | None,
) -> Optional[int]:
    target = action.target.lower()
    channel_map = channel_map or {}
    if target in channel_map:
        return int(channel_map[target])
    if snapshots:
        for channel_id, snapshot in snapshots.items():
            labels = {
                str(snapshot.channel_id).lower(),
                snapshot.name.lower(),
                snapshot.role.lower(),
            }
            if target in labels:
                return int(channel_id)
    return None


class MixAgentBackendBridge:
    """Translate recommendations into bounded live-console actions.

    The bridge never writes directly to a mixer.  If ``apply=True`` it delegates
    to an existing ``AutoFOHSafetyController`` instance, preserving runtime
    policy, fader ceilings, rate limits and mixer-specific translators.
    """

    def __init__(
        self,
        safety_controller: Any | None = None,
        channel_map: Mapping[str, int] | None = None,
        snapshots: Mapping[int, BackendChannelSnapshot] | None = None,
    ):
        self.safety_controller = safety_controller
        self.channel_map = dict(channel_map or {})
        self.snapshots = dict(snapshots or {})
        self._autofoh = _import_autofoh()

    def translate_action(self, action: MixAction) -> tuple[Any | None, Dict[str, Any] | None]:
        """Translate one MixAction into an AutoFOH typed action when safe."""
        channel_id = _channel_lookup(action, self.channel_map, self.snapshots)
        if action.action_type in {"dynamic_eq_placeholder", "sidechain_suggestion", "stereo_width_adjustment", "mid_side_adjustment"}:
            return None, {
                "action": action.to_dict(),
                "reason": "advisory_placeholder_requires_operator_or_plugin_chain",
            }
        if action.target == "mix" and action.action_type == "gain_adjustment":
            return None, {
                "action": action.to_dict(),
                "reason": "mix_bus_gain_changes_require_explicit_operator_target_mapping",
            }
        if channel_id is None:
            return None, {
                "action": action.to_dict(),
                "reason": "no_backend_channel_mapping_for_target",
            }

        reason = action.reason or action.expected_improvement or action.id
        if action.action_type == "gain_adjustment":
            current = None
            snapshot = self.snapshots.get(channel_id)
            if snapshot is not None:
                current = snapshot.current_fader_db
            if current is None and self.safety_controller is not None:
                current = getattr(self.safety_controller.mixer_client, "get_fader")(channel_id)
            current = float(current if current is not None else -12.0)
            gain_db = min(0.0, float(action.parameters.get("gain_db", 0.0)))
            return self._autofoh["ChannelFaderMove"](
                channel_id=channel_id,
                target_db=current + gain_db,
                delta_db=gain_db,
                is_lead="vocal" in action.target.lower(),
                reason=reason,
            ), None

        if action.action_type == "high_pass_filter":
            return self._autofoh["HighPassAdjust"](
                channel_id=channel_id,
                freq_hz=float(action.parameters.get("frequency_hz", 80.0)),
                enabled=True,
                reason=reason,
            ), None

        if action.action_type == "parametric_eq":
            return self._autofoh["ChannelEQMove"](
                channel_id=channel_id,
                band=int(action.parameters.get("band", 2)),
                freq_hz=float(action.parameters.get("frequency_hz", 1000.0)),
                gain_db=float(action.parameters.get("gain_db", 0.0)),
                q=float(action.parameters.get("q", 1.0)),
                reason=reason,
            ), None

        if action.action_type == "pan_adjustment":
            pan = float(action.parameters.get("pan", 0.0))
            # AutoFOH safety uses WING-style -100..100 pan units.
            if -1.0 <= pan <= 1.0:
                pan *= 100.0
            return self._autofoh["PanAdjust"](
                channel_id=channel_id,
                pan=pan,
                reason=reason,
            ), None

        return None, {
            "action": action.to_dict(),
            "reason": "unsupported_backend_action_type",
        }

    def validate_or_apply(
        self,
        actions: Iterable[MixAction],
        runtime_state: Any | str = "SOURCE_LEARNING",
        apply: bool = False,
    ) -> BackendBridgeResult:
        """Translate and optionally apply actions through AutoFOHSafetyController."""
        runtime_cls = self._autofoh["RuntimeState"]
        if isinstance(runtime_state, str):
            runtime_state = runtime_cls[runtime_state] if runtime_state in runtime_cls.__members__ else runtime_cls.SOURCE_LEARNING
        result = BackendBridgeResult(proposed=list(actions), mode="apply" if apply else "suggest")
        for action in result.proposed:
            typed, blocked = self.translate_action(action)
            if blocked is not None:
                result.blocked.append(blocked)
                continue
            result.translated.append(
                {
                    "mix_action_id": action.id,
                    "action_type": typed.action_type,
                    "payload": dict(typed.__dict__),
                    "runtime_state": runtime_state.value,
                }
            )
            if apply:
                if self.safety_controller is None:
                    result.blocked.append(
                        {
                            "action": action.to_dict(),
                            "reason": "apply_requested_without_safety_controller",
                        }
                    )
                    continue
                decision = self.safety_controller.execute(typed, runtime_state)
                result.decisions.append(
                    {
                        "mix_action_id": action.id,
                        "allowed": bool(decision.allowed),
                        "sent": bool(decision.sent),
                        "bounded": bool(decision.bounded),
                        "rate_limited": bool(decision.rate_limited),
                        "message": decision.message,
                        "payload": decision.payload,
                    }
                )
        return result
