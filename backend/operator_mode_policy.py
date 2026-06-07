"""Operator mode policy for the product-facing automixer runtime.

This module gives the backend and UI one small contract for analysis,
suggestions, supervised writes, and locked autonomous control.
"""

from __future__ import annotations

from dataclasses import dataclass
import re
from typing import Any, Dict, List, Tuple


MODE_ANALYZE = "analyze"
MODE_ASSIST = "assist"
MODE_SUPERVISED = "supervised"
MODE_AUTO = "auto"

DEFAULT_OPERATOR_MODE = MODE_ASSIST
VALID_OPERATOR_MODES = (MODE_ANALYZE, MODE_ASSIST, MODE_SUPERVISED, MODE_AUTO)
CURRENT_PROVEN_LIVE_WRITE_KINDS = ("fader", "gain")
FULL_AUTONOMOUS_TARGET_SURFACES = (
    "preamp",
    "trim",
    "fader",
    "mute",
    "pan",
    "eq",
    "compressor",
    "gate",
    "routing",
    "dca",
    "groups",
    "matrices",
    "fx",
    "scenes",
    "presets",
)

OPERATOR_MODE_ALIASES = {
    MODE_ANALYZE: MODE_ANALYZE,
    "analysis": MODE_ANALYZE,
    "analysis_mode": MODE_ANALYZE,
    "analysis_mode_mode": MODE_ANALYZE,
    MODE_ASSIST: MODE_ASSIST,
    "assisted": MODE_ASSIST,
    "assisted_mixing": MODE_ASSIST,
    "assisted_mixing_mode": MODE_ASSIST,
    MODE_SUPERVISED: MODE_SUPERVISED,
    "supervised_automix": MODE_SUPERVISED,
    "supervised_automix_mode": MODE_SUPERVISED,
    "supervised_auto_mix": MODE_SUPERVISED,
    "supervised_auto_mix_mode": MODE_SUPERVISED,
    "supervised_auto_mode": MODE_SUPERVISED,
    MODE_AUTO: MODE_AUTO,
    "autonomous": MODE_AUTO,
    "full_autonomous": MODE_AUTO,
    "full_autonomous_mix": MODE_AUTO,
    "full_autonomous_mode": MODE_AUTO,
    "full_autonomous_mixing": MODE_AUTO,
    "full_autonomous_mixing_mode": MODE_AUTO,
}


@dataclass(frozen=True)
class OperatorModeDefinition:
    key: str
    label: str
    summary: str
    intent: str
    project_mode_number: int
    project_mode_name: str
    autonomy_tier: int
    human_gate: str
    proposal_policy: str
    target_live_write_surfaces: Tuple[str, ...] = ()


MODE_DEFINITIONS: Dict[str, OperatorModeDefinition] = {
    MODE_ANALYZE: OperatorModeDefinition(
        key=MODE_ANALYZE,
        label="Analyze",
        summary="Measure and explain the mix without creating applyable actions.",
        intent="listen_measure_report",
        project_mode_number=1,
        project_mode_name="Analysis Mode",
        autonomy_tier=1,
        human_gate="operator_reviews_recommendations_outside_apply_path",
        proposal_policy="analysis_only_no_proposals",
    ),
    MODE_ASSIST: OperatorModeDefinition(
        key=MODE_ASSIST,
        label="Assist",
        summary="Analyze the mix and prepare suggestions for the engineer.",
        intent="listen_measure_suggest",
        project_mode_number=2,
        project_mode_name="Assisted Mixing Mode",
        autonomy_tier=2,
        human_gate="operator_decides_whether_to_apply",
        proposal_policy="proposal_queue_only",
    ),
    MODE_SUPERVISED: OperatorModeDefinition(
        key=MODE_SUPERVISED,
        label="Supervised",
        summary="Prepare changes and apply only with explicit approval, readback, and rollback.",
        intent="approve_then_apply",
        project_mode_number=3,
        project_mode_name="Supervised AutoMix Mode",
        autonomy_tier=3,
        human_gate="explicit_operator_approval_required",
        proposal_policy="proposal_then_supervised_apply",
        target_live_write_surfaces=FULL_AUTONOMOUS_TARGET_SURFACES,
    ),
    MODE_AUTO: OperatorModeDefinition(
        key=MODE_AUTO,
        label="Auto",
        summary="Autonomous console control. Locked on WING deployment until proven safe.",
        intent="autonomous_control",
        project_mode_number=4,
        project_mode_name="Full Autonomous Mixing Mode",
        autonomy_tier=4,
        human_gate="safety_configuration_only",
        proposal_policy="autonomous_apply_when_unlocked",
        target_live_write_surfaces=FULL_AUTONOMOUS_TARGET_SURFACES,
    ),
}


def normalize_operator_mode(value: Any) -> str:
    normalized = str(value or "").strip().lower()
    mode = re.sub(r"[-\s]+", "_", normalized)
    mode = re.sub(r"_+", "_", mode).strip("_")
    return OPERATOR_MODE_ALIASES.get(mode, DEFAULT_OPERATOR_MODE)


def get_operator_mode_options() -> List[Dict[str, str]]:
    return [
        {
            "key": mode,
            "label": MODE_DEFINITIONS[mode].label,
            "summary": MODE_DEFINITIONS[mode].summary,
            "intent": MODE_DEFINITIONS[mode].intent,
            "project_mode_name": MODE_DEFINITIONS[mode].project_mode_name,
            "project_mode_number": str(MODE_DEFINITIONS[mode].project_mode_number),
            "autonomy_tier": str(MODE_DEFINITIONS[mode].autonomy_tier),
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
    current_live_write_kinds: List[str] = []

    if normalized == MODE_ANALYZE:
        live_write_policy = "analysis_only"
    elif normalized == MODE_ASSIST:
        live_write_policy = "suggestions_only"
        apply_path = "proposal_queue"
    elif normalized == MODE_SUPERVISED:
        live_write_policy = "supervised_approval_required"
        apply_path = "supervised_manual_gate"
        current_live_write_kinds = list(CURRENT_PROVEN_LIVE_WRITE_KINDS)
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
            warnings.append("autonomous_scope_limited_to_current_proven_write_kinds")
            current_live_write_kinds = list(CURRENT_PROVEN_LIVE_WRITE_KINDS)
        else:
            live_write_policy = "auto_locked"
            blocked_reasons.append("auto_mode_requires_explicit_runtime_unlock")

    if normalized in {MODE_ANALYZE, MODE_ASSIST}:
        can_apply_to_console = False
        can_autonomous_apply = False
        requires_approval = False
        current_live_write_kinds = []

    return {
        "mode": normalized,
        "label": definition.label,
        "summary": definition.summary,
        "intent": definition.intent,
        "autonomy_tier": definition.autonomy_tier,
        "project_mode": {
            "number": definition.project_mode_number,
            "name": definition.project_mode_name,
            "autonomy_tier": definition.autonomy_tier,
            "human_gate": definition.human_gate,
            "proposal_policy": definition.proposal_policy,
            "target_live_write_surfaces": list(definition.target_live_write_surfaces),
            "current_live_write_kinds": current_live_write_kinds,
        },
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
            "allowed_live_write_kinds": current_live_write_kinds if can_apply_to_console else [],
        },
        "blocked_reasons": blocked_reasons,
        "warnings": warnings,
        "available_modes": get_operator_mode_options(),
    }
