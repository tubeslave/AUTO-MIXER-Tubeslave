"""Read-only WING correction apply readiness matrix.

This module is deliberately declarative. It explains which correction families
are currently live-apply capable on the WING deployment boundary and what proof
is still required for the remaining families. It does not import mixer
transports and must not send OSC.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from typing import Any, Dict, Iterable, Tuple

try:
    from processing_registry import LIVE_ALLOWED_WRITE_KINDS, get_processing_method
except ImportError:  # pragma: no cover - package import compatibility
    from backend.processing_registry import LIVE_ALLOWED_WRITE_KINDS, get_processing_method


PROMOTION_STAGES = (
    "offline_method",
    "dry_run_recommendation",
    "proposal_only",
    "readback_contract",
    "supervised_candidate",
    "supervised_live_allowed",
)


@dataclass(frozen=True)
class CorrectionApplyFamily:
    family_id: str
    label: str
    objective_family: str
    method_ids: Tuple[str, ...]
    write_kinds: Tuple[str, ...]
    current_stage: str
    can_apply_live: bool
    live_status: str
    proposal_status: str
    transport_status: str
    readback_contract: str
    rollback_contract: str
    hardware_proof: str
    blocked_by: Tuple[str, ...]
    next_required_action: str
    promotion_order: int
    policy_allows_supervised_apply: bool = False
    notes: Tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if self.current_stage not in PROMOTION_STAGES:
            raise ValueError(f"Unknown promotion stage: {self.current_stage}")
        if self.policy_allows_supervised_apply:
            missing = [
                kind for kind in self.write_kinds if kind not in LIVE_ALLOWED_WRITE_KINDS
            ]
            if missing:
                raise ValueError(
                    f"{self.family_id} declares supervised policy outside WING boundary: {missing}"
                )
        if self.can_apply_live:
            if not self.policy_allows_supervised_apply:
                raise ValueError(
                    f"{self.family_id} can apply live without supervised policy allowance"
                )
            if self.current_stage != "supervised_live_allowed":
                raise ValueError(
                    f"{self.family_id} can apply live but is not supervised_live_allowed"
                )
            if self.blocked_by:
                raise ValueError(f"{self.family_id} can apply live but still has blockers")
        if not self.method_ids:
            raise ValueError(f"{self.family_id} must list at least one method id")
        if not self.write_kinds:
            raise ValueError(f"{self.family_id} must list at least one write kind")

    def to_dict(self, *, include_method_status: bool = True) -> Dict[str, Any]:
        payload = asdict(self)
        for key in (
            "method_ids",
            "write_kinds",
            "blocked_by",
            "notes",
        ):
            payload[key] = list(payload[key])
        payload["wing_live_allowed_write_kinds"] = list(LIVE_ALLOWED_WRITE_KINDS)
        payload["not_in_wing_live_boundary"] = [
            kind for kind in self.write_kinds if kind not in LIVE_ALLOWED_WRITE_KINDS
        ]
        if include_method_status:
            payload["methods"] = [
                _method_status(method_id) for method_id in self.method_ids
            ]
        return payload


CORRECTION_APPLY_FAMILIES: Tuple[CorrectionApplyFamily, ...] = (
    CorrectionApplyFamily(
        family_id="fader_balance",
        label="Channel fader and balance",
        objective_family="gain_fader",
        method_ids=("auto_fader_v1", "live_shared_mix", "auto_soundcheck_engine"),
        write_kinds=("fader",),
        current_stage="supervised_candidate",
        can_apply_live=False,
        live_status="policy_allows_supervised_operator_proposal_but_release_readiness_blocked",
        proposal_status="applyable_when_queue_change_is_bounded_and_approved",
        transport_status="wing_transport_method_exists",
        readback_contract="physical_readback_required_before_promotion_evidence",
        rollback_contract="rollback_last_manual_write_required",
        hardware_proof="missing_complete_fader_apply_readback_rollback_emergency_stop_proof",
        blocked_by=(
            "physical_rollback_proof_required",
            "emergency_stop_or_blocked_apply_proof_required",
            "release_readiness_not_closed",
        ),
        next_required_action=(
            "Rerun the narrow fader proof until apply, physical readback, rollback, "
            "and emergency-stop/block evidence all pass; do not reuse this proof "
            "for other families."
        ),
        promotion_order=0,
        policy_allows_supervised_apply=True,
        notes=(
            "policy_allows_supervised_operator_proposal",
            "only_channel_fader_scope",
            "no_bus_dca_master_fader_scope",
        ),
    ),
    CorrectionApplyFamily(
        family_id="gain_trim",
        label="Input trim and gain staging",
        objective_family="gain_fader",
        method_ids=("lufs_gain_staging", "auto_soundcheck_engine"),
        write_kinds=("gain",),
        current_stage="supervised_candidate",
        can_apply_live=False,
        live_status="policy_allows_supervised_operator_proposal_but_release_readiness_blocked",
        proposal_status="applyable_when_queue_change_is_bounded_and_approved",
        transport_status="wing_transport_method_exists",
        readback_contract="physical_trim_readback_required_before_promotion_evidence",
        rollback_contract="rollback_last_manual_write_required",
        hardware_proof="missing_gain_apply_readback_rollback_emergency_stop_proof",
        blocked_by=(
            "physical_gain_proof_required",
            "physical_rollback_proof_required",
            "emergency_stop_or_blocked_apply_proof_required",
            "release_readiness_not_closed",
        ),
        next_required_action=(
            "Run a separate negative or headroom-safe gain proof with physical "
            "readback, rollback, and emergency-stop/block evidence before claiming "
            "gain readiness beyond policy allowance."
        ),
        promotion_order=1,
        policy_allows_supervised_apply=True,
        notes=(
            "policy_allows_supervised_operator_proposal",
            "input_trim_only",
            "no_gain_up_without_headroom_and_operator_authoring",
        ),
    ),
    CorrectionApplyFamily(
        family_id="pan_spatial",
        label="Pan and stereo placement",
        objective_family="pan",
        method_ids=("auto_panner", "pan_spatial_proposal"),
        write_kinds=("pan",),
        current_stage="proposal_only",
        can_apply_live=False,
        live_status="blocked_on_wing_deployment_boundary",
        proposal_status="operator_review_only",
        transport_status="wing_transport_method_exists",
        readback_contract="current_pan_physical_readback_required",
        rollback_contract="previous_pan_value_rollback_required",
        hardware_proof="missing_pan_apply_readback_rollback_proof",
        blocked_by=(
            "not_in_wing_deployment_live_write_kinds",
            "separate_hardware_proof_required",
            "mono_phase_safety_required",
        ),
        next_required_action=(
            "Promote first among non-fader/gain families only after a tiny "
            "operator-approved pan proof with physical readback, rollback, and "
            "mono/phase safety evidence."
        ),
        promotion_order=2,
        notes=("lowest_risk_non_level_candidate", "still_not_live_applyable"),
    ),
    CorrectionApplyFamily(
        family_id="mute_management",
        label="Mute and unmute management",
        objective_family="mute",
        method_ids=("mute_management_proposal",),
        write_kinds=("mute",),
        current_stage="proposal_only",
        can_apply_live=False,
        live_status="blocked_on_wing_deployment_boundary",
        proposal_status="operator_review_only",
        transport_status="wing_transport_method_exists",
        readback_contract="fresh_mute_state_readback_required",
        rollback_contract="previous_mute_state_rollback_required",
        hardware_proof="missing_mute_apply_readback_rollback_proof",
        blocked_by=(
            "not_in_wing_deployment_live_write_kinds",
            "explicit_mute_operator_ux_required",
            "separate_hardware_proof_required",
        ),
        next_required_action=(
            "Keep mute changes review-only until there is a mute-specific UX, "
            "intent confirmation, physical readback, rollback, and emergency-state proof."
        ),
        promotion_order=3,
        notes=("can_silence_sources", "not_a_generic_correction_executor"),
    ),
    CorrectionApplyFamily(
        family_id="eq_filtering",
        label="EQ, HPF, and filtering",
        objective_family="eq",
        method_ids=("eq_v3", "auto_eq", "corrective_eq_proposal", "cross_adaptive_eq"),
        write_kinds=("eq", "hpf"),
        current_stage="proposal_only",
        can_apply_live=False,
        live_status="blocked_on_wing_deployment_boundary",
        proposal_status="operator_review_only",
        transport_status="partial_wing_transport_methods_exist",
        readback_contract="band_identity_frequency_gain_q_readback_required",
        rollback_contract="per_band_previous_value_rollback_required",
        hardware_proof="missing_eq_band_apply_readback_rollback_feedback_proof",
        blocked_by=(
            "not_in_wing_deployment_live_write_kinds",
            "eq_band_identity_required",
            "feedback_and_headroom_safety_required",
            "separate_hardware_proof_required",
        ),
        next_required_action=(
            "Do not apply EQ live until one bounded cut-only band proof verifies "
            "address mapping, readback, rollback, protected-band policy, and feedback safety."
        ),
        promotion_order=4,
        notes=("cut_only_first_candidate", "boosts_require_extra_headroom_guard"),
    ),
    CorrectionApplyFamily(
        family_id="dynamics_compressor",
        label="Compressor, dynamics, and de-esser",
        objective_family="dynamics",
        method_ids=(
            "auto_compressor",
            "compressor_dynamics_proposal",
            "deesser_vocal_sibilance",
        ),
        write_kinds=("compressor", "deesser"),
        current_stage="proposal_only",
        can_apply_live=False,
        live_status="blocked_on_wing_deployment_boundary",
        proposal_status="operator_review_only",
        transport_status="partial_wing_transport_methods_exist",
        readback_contract="dyn_on_threshold_ratio_attack_release_makeup_readback_required",
        rollback_contract="previous_dynamics_state_rollback_required",
        hardware_proof="missing_dynamics_apply_readback_rollback_headroom_proof",
        blocked_by=(
            "not_in_wing_deployment_live_write_kinds",
            "makeup_gain_headroom_guard_required",
            "gain_reduction_evidence_required",
            "separate_hardware_proof_required",
        ),
        next_required_action=(
            "Keep compressor/de-esser review-only until a threshold-only or "
            "makeup-neutral proof verifies dynamics readback, GR/headroom safety, and rollback."
        ),
        promotion_order=5,
        notes=("makeup_gain_must_be_bounded", "time_parameters_need_mapping_proof"),
    ),
    CorrectionApplyFamily(
        family_id="gate_expander",
        label="Gate and expander",
        objective_family="gate",
        method_ids=("auto_gate_caig", "context_aware_gate_agent"),
        write_kinds=("gate", "expander"),
        current_stage="proposal_only",
        can_apply_live=False,
        live_status="blocked_on_wing_deployment_boundary",
        proposal_status="operator_review_only",
        transport_status="partial_wing_transport_methods_exist",
        readback_contract="gate_threshold_range_attack_hold_release_readback_required",
        rollback_contract="previous_gate_state_rollback_required",
        hardware_proof="missing_gate_apply_readback_rollback_chatter_proof",
        blocked_by=(
            "not_in_wing_deployment_live_write_kinds",
            "source_identity_trust_required",
            "chatter_safety_required",
            "separate_hardware_proof_required",
        ),
        next_required_action=(
            "Use offline/live telemetry to prove false-open/false-close behavior; "
            "only then run a bounded threshold proof with rollback."
        ),
        promotion_order=6,
        notes=("bleed_context_required", "chatter_guard_required"),
    ),
    CorrectionApplyFamily(
        family_id="phase_delay",
        label="Phase, polarity, and delay",
        objective_family="phase",
        method_ids=("phase_alignment", "phase_aware_eq_guard"),
        write_kinds=("phase", "polarity", "delay"),
        current_stage="proposal_only",
        can_apply_live=False,
        live_status="blocked_on_wing_deployment_boundary",
        proposal_status="operator_review_only",
        transport_status="wing_transport_methods_exist_for_delay_polarity",
        readback_contract="fresh_phase_delay_polarity_readback_required",
        rollback_contract="previous_phase_delay_polarity_rollback_required",
        hardware_proof="missing_phase_delay_apply_readback_rollback_mono_proof",
        blocked_by=(
            "not_in_wing_deployment_live_write_kinds",
            "stage_position_or_measurement_context_required",
            "mono_sum_safety_required",
            "separate_hardware_proof_required",
        ),
        next_required_action=(
            "Keep phase/delay proposals review-only until measurement context, "
            "mono-sum safety, physical readback, and rollback are proven."
        ),
        promotion_order=7,
        notes=("can_change_comb_filtering", "proof_after_pan_and_eq_not_before"),
    ),
    CorrectionApplyFamily(
        family_id="fx_sends",
        label="FX, inserts, sends, and returns",
        objective_family="fx",
        method_ids=("auto_fx", "fx_send_proposal"),
        write_kinds=("fx", "insert", "send"),
        current_stage="proposal_only",
        can_apply_live=False,
        live_status="blocked_on_wing_deployment_boundary",
        proposal_status="operator_review_only",
        transport_status="wing_transport_methods_exist_but_schema_varies_by_fx_model",
        readback_contract="send_return_fx_model_parameter_readback_required",
        rollback_contract="previous_send_return_fx_parameter_rollback_required",
        hardware_proof="missing_fx_apply_readback_rollback_monitor_feedback_proof",
        blocked_by=(
            "not_in_wing_deployment_live_write_kinds",
            "fx_parameter_schema_required",
            "monitor_feedback_safety_required",
            "separate_hardware_proof_required",
        ),
        next_required_action=(
            "Keep FX/send changes proposal-only until send/return readback, monitor safety, "
            "FX parameter schema, feedback checks, and rollback are proven."
        ),
        promotion_order=8,
        notes=("model_specific_parameters", "monitor_sends_are_high_risk"),
    ),
    CorrectionApplyFamily(
        family_id="bus_group_master",
        label="Bus, group, DCA, matrix, and master output",
        objective_family="bus_group_master",
        method_ids=("bus_group_balance_proposal", "master_bus_optimization_proposal"),
        write_kinds=("bus_fader", "dca_fader", "matrix_fader", "master_fader"),
        current_stage="proposal_only",
        can_apply_live=False,
        live_status="blocked_on_wing_deployment_boundary",
        proposal_status="operator_review_only",
        transport_status="wing_transport_methods_exist_for_some_surfaces",
        readback_contract="assignment_fader_main_bus_readback_required",
        rollback_contract="previous_bus_group_master_state_rollback_required",
        hardware_proof="missing_output_apply_readback_rollback_pa_safety_proof",
        blocked_by=(
            "not_in_wing_deployment_live_write_kinds",
            "pa_feed_safety_required",
            "assignment_freshness_required",
            "separate_hardware_proof_required",
        ),
        next_required_action=(
            "Do not promote broad output surfaces until channel-level families are proven; "
            "require PA-feed exclusion, true-peak safety, readback, rollback, and operator UX."
        ),
        promotion_order=9,
        notes=("high_blast_radius", "last_level_family_to_promote"),
    ),
    CorrectionApplyFamily(
        family_id="routing_scene_preset",
        label="Routing, scenes, snapshots, and presets",
        objective_family="routing_scene_preset",
        method_ids=("routing_management_proposal", "scene_preset_management_proposal"),
        write_kinds=("routing", "scene", "snapshot", "preset"),
        current_stage="proposal_only",
        can_apply_live=False,
        live_status="blocked_on_wing_deployment_boundary",
        proposal_status="operator_review_only",
        transport_status="wing_transport_methods_exist_but_are_quarantined",
        readback_contract="full_route_or_scene_diff_readback_required",
        rollback_contract="scoped_previous_state_or_scene_restore_required",
        hardware_proof="missing_scoped_route_scene_apply_readback_rollback_proof",
        blocked_by=(
            "not_in_wing_deployment_live_write_kinds",
            "broad_console_state_risk",
            "scoped_diff_required",
            "separate_hardware_proof_required",
        ),
        next_required_action=(
            "Keep routing/scene/preset behavior manual-review only until there is a scoped "
            "diff, full readback, rollback/restore plan, and explicit operator runbook."
        ),
        promotion_order=10,
        notes=("highest_blast_radius", "not_part_of_audio_correction_mvp_apply"),
    ),
)


def _method_status(method_id: str) -> Dict[str, Any]:
    method = get_processing_method(method_id)
    if method is None:
        return {
            "method_id": method_id,
            "exists": False,
            "status": "missing",
            "can_analyze": False,
            "can_propose": False,
            "can_write_live": False,
        }
    return {
        "method_id": method.id,
        "exists": True,
        "label": method.label,
        "domain": method.domain,
        "status": method.status,
        "can_analyze": bool(method.can_analyze),
        "can_propose": bool(method.can_propose),
        "can_write_live": bool(method.can_write_live),
        "live_write_kinds": list(method.live_write_kinds),
        "blocked_reason": method.blocked_reason,
        "promotion_next": method.promotion_next,
        "notes": list(method.notes),
    }


def iter_correction_apply_families() -> Iterable[CorrectionApplyFamily]:
    return iter(sorted(CORRECTION_APPLY_FAMILIES, key=lambda item: item.promotion_order))


def get_correction_apply_family(family_id: str) -> CorrectionApplyFamily | None:
    return next(
        (family for family in CORRECTION_APPLY_FAMILIES if family.family_id == family_id),
        None,
    )


def build_wing_correction_apply_readiness_snapshot(
    *,
    include_method_status: bool = True,
) -> Dict[str, Any]:
    families = [
        family.to_dict(include_method_status=include_method_status)
        for family in iter_correction_apply_families()
    ]
    policy_allowed = [
        family for family in families if family["policy_allows_supervised_apply"]
    ]
    live_allowed = [family for family in families if family["can_apply_live"]]
    blocked = [family for family in families if not family["can_apply_live"]]
    next_blocked = next(iter(blocked), None)
    return {
        "type": "wing_correction_apply_readiness",
        "schema": "automixer.wing_correction_apply_readiness.v1",
        "generated_at": datetime.now(timezone.utc)
        .isoformat(timespec="seconds")
        .replace("+00:00", "Z"),
        "wing_deployment_boundary": {
            "live_allowed_write_kinds": list(LIVE_ALLOWED_WRITE_KINDS),
            "direct_live_widening_allowed": False,
            "required_before_any_new_family": [
                "separate_ADR",
                "bounded_operator_proposal_contract",
                "physical_readback_mapping",
                "rollback_proof",
                "emergency_stop_or_blocked_apply_proof",
                "explicit_operator_approval",
            ],
        },
        "summary": {
            "family_count": len(families),
            "supervised_policy_family_count": len(policy_allowed),
            "supervised_policy_family_ids": [
                family["family_id"] for family in policy_allowed
            ],
            "live_allowed_family_count": len(live_allowed),
            "blocked_family_count": len(blocked),
            "live_allowed_family_ids": [family["family_id"] for family in live_allowed],
            "blocked_family_ids": [family["family_id"] for family in blocked],
            "next_blocked_family_id": (
                next_blocked["family_id"] if isinstance(next_blocked, dict) else None
            ),
            "next_blocked_family_action": (
                next_blocked["next_required_action"]
                if isinstance(next_blocked, dict)
                else None
            ),
        },
        "families": families,
    }


def render_wing_correction_apply_readiness_markdown(snapshot: Dict[str, Any]) -> str:
    boundary = snapshot["wing_deployment_boundary"]
    summary = snapshot["summary"]
    live_kinds = ", ".join(f"`{kind}`" for kind in boundary["live_allowed_write_kinds"])
    lines = [
        "# WING Correction Apply Readiness",
        "",
        f"Generated: `{snapshot['generated_at']}`",
        "",
        "## Current Boundary",
        "",
        f"- Current supervised policy write kinds: {live_kinds or '`none`'}",
        "- Direct live widening allowed: `false`",
        "- Release-ready live families require complete apply, physical readback, rollback, and emergency-stop/block evidence.",
        "- New correction families require separate ADR, bounded proposal contract, physical readback, rollback, emergency-stop/block proof, and explicit operator approval.",
        "",
        "## Summary",
        "",
        f"- Family count: `{summary['family_count']}`",
        f"- Policy-allowed supervised families: `{summary['supervised_policy_family_count']}`",
        f"- Release-ready live families: `{summary['live_allowed_family_count']}`",
        f"- Blocked families: `{summary['blocked_family_count']}`",
        f"- Next blocked family: `{summary['next_blocked_family_id']}`",
        "",
        "## Families",
        "",
        "| Family | Stage | Policy supervised | Release live ready | Write kinds | Blockers | Next action |",
        "| --- | --- | --- | --- | --- | --- | --- |",
    ]
    for family in snapshot["families"]:
        kinds = ", ".join(f"`{kind}`" for kind in family["write_kinds"])
        blockers = ", ".join(f"`{blocker}`" for blocker in family["blocked_by"]) or "`none`"
        lines.append(
            "| {label} | `{stage}` | `{policy}` | `{live}` | {kinds} | {blockers} | {next_action} |".format(
                label=family["label"],
                stage=family["current_stage"],
                policy=str(family["policy_allows_supervised_apply"]).lower(),
                live=str(family["can_apply_live"]).lower(),
                kinds=kinds,
                blockers=blockers,
                next_action=family["next_required_action"],
            )
        )
    lines.extend(
        [
            "",
            "## Safety Rule",
            "",
            "Do not enable all WING correction methods by adding them to one global live-write list. Promote exactly one family at a time after its own hardware proof.",
            "",
        ]
    )
    return "\n".join(lines)
