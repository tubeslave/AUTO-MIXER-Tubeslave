"""Operator mode policy for the product-facing automixer runtime.

This module gives the backend and UI one small contract for analysis,
suggestions, supervised writes, and locked autonomous control.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List


MODE_ANALYZE = "analyze"
MODE_ASSIST = "assist"
MODE_SUPERVISED = "supervised"
MODE_AUTO = "auto"

DEFAULT_OPERATOR_MODE = MODE_ASSIST
VALID_OPERATOR_MODES = (MODE_ANALYZE, MODE_ASSIST, MODE_SUPERVISED, MODE_AUTO)


@dataclass(frozen=True)
class OperatorModeDefinition:
    key: str
    label: str
    summary: str
    intent: str


MODE_DEFINITIONS: Dict[str, OperatorModeDefinition] = {
    MODE_ANALYZE: OperatorModeDefinition(
        key=MODE_ANALYZE,
        label="Analyze",
        summary="Measure and explain the mix without creating applyable actions.",
        intent="listen_measure_report",
    ),
    MODE_ASSIST: OperatorModeDefinition(
        key=MODE_ASSIST,
        label="Assist",
        summary="Analyze the mix and prepare suggestions for the engineer.",
        intent="listen_measure_suggest",
    ),
    MODE_SUPERVISED: OperatorModeDefinition(
        key=MODE_SUPERVISED,
        label="Supervised",
        summary="Prepare changes and apply only with explicit approval, readback, and rollback.",
        intent="approve_then_apply",
    ),
    MODE_AUTO: OperatorModeDefinition(
        key=MODE_AUTO,
        label="Auto",
        summary="Autonomous console control. Locked on WING deployment until proven safe.",
        intent="autonomous_control",
    ),
}


def normalize_operator_mode(value: Any) -> str:
    mode = str(value or "").strip().lower()
    if mode in VALID_OPERATOR_MODES:
        return mode
    return DEFAULT_OPERATOR_MODE


def get_operator_mode_options() -> List[Dict[str, str]]:
    return [
        {
            "key": mode,
            "label": MODE_DEFINITIONS[mode].label,
            "summary": MODE_DEFINITIONS[mode].summary,
            "intent": MODE_DEFINITIONS[mode].intent,
        }
        for mode in VALID_OPERATOR_MODES
    ]


def build_operator_mode_status(
    *,
    mode: str,
    connection_mode: str | None = None,
    wing_boundary_active: bool = False,
    auto_mode_unlocked: bool = False,
) -> Dict[str, Any]:
    normalized = normalize_operator_mode(mode)
    definition = MODE_DEFINITIONS[normalized]
    wing_deployment = connection_mode == "wing" and bool(wing_boundary_active)

    blocked_reasons: List[str] = []
    warnings: List[str] = []
    can_create_proposals = normalized in {MODE_ASSIST, MODE_SUPERVISED, MODE_AUTO}
    requires_approval = normalized == MODE_SUPERVISED
    can_apply_to_console = normalized == MODE_SUPERVISED
    can_autonomous_apply = False
    live_write_policy = "no_console_writes"
    apply_path = "none"

    if normalized == MODE_ANALYZE:
        live_write_policy = "analysis_only"
    elif normalized == MODE_ASSIST:
        live_write_policy = "suggestions_only"
        apply_path = "proposal_queue"
    elif normalized == MODE_SUPERVISED:
        live_write_policy = "supervised_approval_required"
        apply_path = "supervised_manual_gate"
    elif normalized == MODE_AUTO:
        apply_path = "autonomous_controller"
        if wing_deployment:
            live_write_policy = "auto_locked_on_wing_deployment"
            blocked_reasons.append("auto_mode_not_approved_for_wing_deployment")
        elif auto_mode_unlocked:
            live_write_policy = "autonomous_apply_unlocked"
            can_apply_to_console = True
            can_autonomous_apply = True
            requires_approval = False
            warnings.append("auto_mode_unlocked")
        else:
            live_write_policy = "auto_locked"
            blocked_reasons.append("auto_mode_requires_explicit_runtime_unlock")

    if normalized in {MODE_ANALYZE, MODE_ASSIST}:
        can_apply_to_console = False
        can_autonomous_apply = False
        requires_approval = False

    return {
        "mode": normalized,
        "label": definition.label,
        "summary": definition.summary,
        "intent": definition.intent,
        "connection_mode": connection_mode,
        "wing_deployment": wing_deployment,
        "live_write_policy": live_write_policy,
        "apply_path": apply_path,
        "capabilities": {
            "can_analyze": True,
            "can_create_proposals": can_create_proposals,
            "can_apply_to_console": can_apply_to_console,
            "requires_approval": requires_approval,
            "can_autonomous_apply": can_autonomous_apply,
            "allowed_live_write_kinds": ["fader", "gain"] if can_apply_to_console else [],
        },
        "blocked_reasons": blocked_reasons,
        "warnings": warnings,
        "available_modes": get_operator_mode_options(),
    }
