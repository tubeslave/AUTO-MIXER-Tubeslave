"""Read-only registry for processing methods and pipeline profiles.

The registry is intentionally declarative. It must not import mixer clients or
DSP modules because Product Layer snapshots need to be safe in offline and live
WING deployments.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, Iterable, List, Optional, Tuple

try:
    from bus_group_main_dca_physical_readback_contract import (
        build_bus_group_main_dca_physical_readback_contract,
    )
    from channel_basic_controls_physical_readback_contract import (
        build_channel_basic_controls_physical_readback_contract,
    )
    from channel_processing_physical_readback_contract import (
        build_channel_processing_physical_readback_contract,
    )
    from dlive_state_physical_readback_contract import (
        build_dlive_state_physical_readback_contract,
    )
    from fx_slots_inserts_physical_readback_contract import (
        build_fx_slots_inserts_physical_readback_contract,
    )
    from matrix_state_physical_readback_contract import (
        build_matrix_state_physical_readback_contract,
    )
    from autonomy_mode_transition_gate import build_autonomy_mode_transition_gate
    from autonomy_promotion_readiness import build_autonomy_promotion_readiness_snapshot
    from project_goal_capability_matrix import build_project_goal_capability_matrix_snapshot
    from read_surface_fixture_evidence import build_read_surface_fixture_evidence_snapshot
    from read_surface_inventory import build_read_surface_inventory_snapshot
    from read_surface_physical_readback_capture_plan import (
        build_read_surface_physical_readback_capture_plan,
    )
    from read_surface_remediation_plan import build_read_surface_remediation_plan
    from raw_mixer_state_physical_readback_contract import (
        build_raw_mixer_state_physical_readback_contract,
    )
    from routing_sends_physical_readback_contract import (
        build_routing_sends_physical_readback_contract,
    )
    from snapshots_scenes_physical_readback_contract import (
        build_snapshots_scenes_physical_readback_contract,
    )
    from user_presets_physical_readback_contract import (
        build_user_presets_physical_readback_contract,
    )
    from safety_coverage_matrix import build_safety_coverage_matrix_snapshot
    from write_surface_inventory import build_write_surface_inventory_snapshot
except ImportError:  # pragma: no cover - package import compatibility
    from backend.bus_group_main_dca_physical_readback_contract import (
        build_bus_group_main_dca_physical_readback_contract,
    )
    from backend.channel_basic_controls_physical_readback_contract import (
        build_channel_basic_controls_physical_readback_contract,
    )
    from backend.channel_processing_physical_readback_contract import (
        build_channel_processing_physical_readback_contract,
    )
    from backend.dlive_state_physical_readback_contract import (
        build_dlive_state_physical_readback_contract,
    )
    from backend.fx_slots_inserts_physical_readback_contract import (
        build_fx_slots_inserts_physical_readback_contract,
    )
    from backend.matrix_state_physical_readback_contract import (
        build_matrix_state_physical_readback_contract,
    )
    from backend.autonomy_mode_transition_gate import build_autonomy_mode_transition_gate
    from backend.autonomy_promotion_readiness import build_autonomy_promotion_readiness_snapshot
    from backend.project_goal_capability_matrix import build_project_goal_capability_matrix_snapshot
    from backend.read_surface_fixture_evidence import build_read_surface_fixture_evidence_snapshot
    from backend.read_surface_inventory import build_read_surface_inventory_snapshot
    from backend.read_surface_physical_readback_capture_plan import (
        build_read_surface_physical_readback_capture_plan,
    )
    from backend.read_surface_remediation_plan import build_read_surface_remediation_plan
    from backend.raw_mixer_state_physical_readback_contract import (
        build_raw_mixer_state_physical_readback_contract,
    )
    from backend.routing_sends_physical_readback_contract import (
        build_routing_sends_physical_readback_contract,
    )
    from backend.snapshots_scenes_physical_readback_contract import (
        build_snapshots_scenes_physical_readback_contract,
    )
    from backend.user_presets_physical_readback_contract import (
        build_user_presets_physical_readback_contract,
    )
    from backend.safety_coverage_matrix import build_safety_coverage_matrix_snapshot
    from backend.write_surface_inventory import build_write_surface_inventory_snapshot


PROCESSING_STATUSES = {
    "canonical",
    "proposal_only",
    "offline_only",
    "shadow_only",
    "lab",
    "placeholder",
    "quarantined",
    "deprecated",
}

PROCESSING_DOMAINS = {
    "gain_fader",
    "source_identity",
    "eq",
    "dynamics",
    "gate",
    "pan",
    "phase",
    "fx",
    "master_bus",
    "bus_group",
    "mute",
    "routing",
    "scene_preset",
    "mix_variant",
    "adaptation",
    "arrangement",
    "decision_evidence",
    "memory_learning",
    "evaluation",
    "offline_render",
}

LIVE_ALLOWED_WRITE_KINDS = ("fader", "gain")


@dataclass(frozen=True)
class ProcessingMethod:
    id: str
    label: str
    domain: str
    status: str
    summary: str
    module: str
    can_analyze: bool = False
    can_propose: bool = False
    can_render_offline: bool = False
    can_write_live: bool = False
    requires_supervision: bool = True
    live_write_kinds: Tuple[str, ...] = ()
    pipeline_profiles: Tuple[str, ...] = ()
    blocked_reason: Optional[str] = None
    promotion_next: Optional[str] = None
    notes: Tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if self.status not in PROCESSING_STATUSES:
            raise ValueError(f"Unknown processing status: {self.status}")
        if self.domain not in PROCESSING_DOMAINS:
            raise ValueError(f"Unknown processing domain: {self.domain}")
        if self.can_write_live and not self.requires_supervision:
            raise ValueError(f"{self.id} can write live without supervision")
        if self.can_write_live and any(kind not in LIVE_ALLOWED_WRITE_KINDS for kind in self.live_write_kinds):
            raise ValueError(f"{self.id} declares unsupported live write kinds: {self.live_write_kinds}")
        if self.status in {"placeholder", "quarantined", "deprecated"} and self.can_write_live:
            raise ValueError(f"{self.id} is {self.status} but can_write_live=True")

    def to_dict(self) -> Dict[str, Any]:
        payload = asdict(self)
        payload["live_write_kinds"] = list(self.live_write_kinds)
        payload["pipeline_profiles"] = list(self.pipeline_profiles)
        payload["notes"] = list(self.notes)
        return payload


@dataclass(frozen=True)
class PipelineProfile:
    id: str
    label: str
    status: str
    summary: str
    method_ids: Tuple[str, ...]
    proposal_method_ids: Tuple[str, ...] = ()
    executor_method_ids: Tuple[str, ...] = ()
    blocked_method_ids: Tuple[str, ...] = ()
    can_write_live: bool = False
    requires_supervision: bool = True
    live_write_kinds: Tuple[str, ...] = ()
    mode: str = "read_only"
    notes: Tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if self.status not in PROCESSING_STATUSES:
            raise ValueError(f"Unknown pipeline status: {self.status}")
        if self.can_write_live and not self.requires_supervision:
            raise ValueError(f"{self.id} can write live without supervision")
        if self.can_write_live and any(kind not in LIVE_ALLOWED_WRITE_KINDS for kind in self.live_write_kinds):
            raise ValueError(f"{self.id} declares unsupported live write kinds: {self.live_write_kinds}")

    def to_dict(self) -> Dict[str, Any]:
        payload = asdict(self)
        payload["method_ids"] = list(self.method_ids)
        payload["proposal_method_ids"] = list(self.proposal_method_ids)
        payload["executor_method_ids"] = list(self.executor_method_ids)
        payload["blocked_method_ids"] = list(self.blocked_method_ids)
        payload["live_write_kinds"] = list(self.live_write_kinds)
        payload["notes"] = list(self.notes)
        return payload


PROCESSING_METHODS: Tuple[ProcessingMethod, ...] = (
    ProcessingMethod(
        id="product_operator_state",
        label="Product / Operator State",
        domain="evaluation",
        status="canonical",
        summary="Read-only state, readiness blockers, proposal queue and operator mode policy.",
        module="backend/operator_product_state.py",
        can_analyze=True,
        can_propose=True,
        pipeline_profiles=("main_live_assist",),
        promotion_next="keep_as_canonical_read_model",
    ),
    ProcessingMethod(
        id="auto_soundcheck_engine",
        label="AutoSoundcheck Engine",
        domain="gain_fader",
        status="canonical",
        summary="Primary live analysis/orchestration hook; live writes must be translated to proposals or supervised writes.",
        module="backend/auto_soundcheck_engine.py",
        can_analyze=True,
        can_propose=True,
        pipeline_profiles=("main_live_assist", "lab_dsp"),
        notes=("guarded_use_only", "do_not_bypass_product_layer"),
    ),
    ProcessingMethod(
        id="source_identity_confidence_proposal",
        label="Source Identity Confidence Proposal Layer",
        domain="source_identity",
        status="proposal_only",
        summary=(
            "Transport-free source identity confidence review layer that turns "
            "low-confidence, unknown, or ambiguous classifier results into "
            "operator-facing review records and downstream rejection reasons."
        ),
        module="backend/source_identity_confidence_proposal.py",
        can_analyze=True,
        can_propose=True,
        pipeline_profiles=("main_live_assist", "shadow_quality", "lab_dsp"),
        blocked_reason=(
            "source_identity_review_needs_operator_acceptance_and_replay_before_"
            "downstream_processing_can_trust_low_confidence_classifications"
        ),
        promotion_next=(
            "route low-confidence or ambiguous source identities to operator "
            "review, then replay accepted/rejected labels before changing "
            "downstream proposal priority"
        ),
        notes=(
            "transport_free",
            "proposal_only",
            "classifier_confidence_gate",
            "downstream_rejection_reasons",
            "no_live_writes",
        ),
    ),
    ProcessingMethod(
        id="source_identity_replay_calibration",
        label="Source Identity Replay Calibration Layer",
        domain="source_identity",
        status="proposal_only",
        summary=(
            "Transport-free replay calibration layer that summarizes accepted, "
            "corrected, and rejected source identity outcomes before downstream "
            "proposal families trust classifier labels."
        ),
        module="backend/source_identity_replay_calibration.py",
        can_analyze=True,
        can_propose=True,
        pipeline_profiles=("main_live_assist", "shadow_quality", "lab_dsp"),
        blocked_reason=(
            "source_identity_calibration_needs_operator_review_before_thresholds_"
            "or_downstream_priority_can_change"
        ),
        promotion_next=(
            "review replay-calibrated source identity trust states, then wire "
            "accepted calibration records into downstream proposal priority without "
            "creating live writes"
        ),
        notes=(
            "transport_free",
            "proposal_only",
            "accepted_rejected_replay_outcomes",
            "no_model_mutation",
            "no_memory_mutation",
            "no_live_writes",
        ),
    ),
    ProcessingMethod(
        id="supervised_manual_fader_gain",
        label="Supervised Manual Fader/Gain Gate",
        domain="gain_fader",
        status="canonical",
        summary="Only accepted WING live executor for MVP: approved channel fader/gain with cooldown, readback and rollback evidence.",
        module="backend/server.py + backend/handlers/mixer_handlers.py",
        can_write_live=True,
        requires_supervision=True,
        live_write_kinds=LIVE_ALLOWED_WRITE_KINDS,
        pipeline_profiles=("main_live_assist",),
        promotion_next="physical_console_proof_for_any_extra_write_kind",
    ),
    ProcessingMethod(
        id="supervised_proof_artifact_evaluator",
        label="Supervised Proof Artifact Evaluator",
        domain="evaluation",
        status="shadow_only",
        summary="Transport-free evaluator for dry-run and physical supervised fader/gain proof artifacts; distinguishes dry-run readiness from physical promotion candidates.",
        module="backend/supervised_proof_artifact_evaluator.py",
        can_analyze=True,
        pipeline_profiles=("shadow_quality", "lab_dsp"),
        blocked_reason="proof_evaluator_only_no_live_transport",
        promotion_next="feed physical fader/gain proof summaries through the evaluator before changing safety coverage rows",
        notes=("transport_free", "proof_evaluator", "no_live_writes", "not_autonomous_authority"),
    ),
    ProcessingMethod(
        id="supervised_promotion_evidence_bridge",
        label="Supervised Promotion Evidence Bridge",
        domain="evaluation",
        status="shadow_only",
        summary="Read-only bridge from proof artifact reports to manual safety-row review candidates; never updates live policy automatically.",
        module="backend/supervised_promotion_evidence.py",
        can_analyze=True,
        pipeline_profiles=("shadow_quality", "lab_dsp"),
        blocked_reason="evidence_bridge_only_manual_safety_review_required",
        promotion_next="use physical fader/gain proof reports to decide whether a manual safety coverage review is justified",
        notes=("transport_free", "proof_report_bridge", "no_live_writes", "no_automatic_safety_update"),
    ),
    ProcessingMethod(
        id="supervised_wing_proof_plan",
        label="Supervised WING Proof Plan",
        domain="evaluation",
        status="shadow_only",
        summary="Read-only operator checklist for the next bounded physical WING fader/gain proof; emits commands but never executes them.",
        module="backend/supervised_wing_proof_plan.py",
        can_analyze=True,
        pipeline_profiles=("shadow_quality", "lab_dsp"),
        blocked_reason="operator_must_execute_physical_proof_manually",
        promotion_next="run the generated physical proof plan only with explicit operator approval and then re-evaluate artifacts",
        notes=("transport_free", "proof_plan_only", "no_live_writes", "execution_allowed_by_plan_false"),
    ),
    ProcessingMethod(
        id="supervised_proof_remediation_plan",
        label="Supervised Proof Remediation Plan",
        domain="evaluation",
        status="shadow_only",
        summary="Read-only remediation checklist for blocked or legacy supervised proof artifacts; emits rerun/fix actions but never executes them.",
        module="backend/supervised_proof_remediation_plan.py",
        can_analyze=True,
        pipeline_profiles=("shadow_quality", "lab_dsp"),
        blocked_reason="proof_remediation_only_operator_must_execute_manually",
        promotion_next="rerun blocked legacy artifacts with current proof schema before reviewing promotion evidence",
        notes=("transport_free", "proof_remediation_only", "no_live_writes", "execution_allowed_by_plan_false"),
    ),
    ProcessingMethod(
        id="autonomy_evidence_overlay",
        label="Autonomy Evidence Overlay",
        domain="evaluation",
        status="shadow_only",
        summary="Read-only overlay that maps current supervised proof reports onto fader/gain autonomy promotion targets.",
        module="backend/autonomy_evidence_overlay.py",
        can_analyze=True,
        pipeline_profiles=("shadow_quality", "lab_dsp"),
        blocked_reason="overlay_only_manual_safety_review_required",
        promotion_next="use overlay rows to connect proof artifacts and remediation actions to promotion targets before manual safety review",
        notes=("transport_free", "proof_overlay_only", "no_live_writes", "no_automatic_safety_update"),
    ),
    ProcessingMethod(
        id="autonomy_mode_transition_gate",
        label="Autonomy Mode Transition Gate",
        domain="evaluation",
        status="shadow_only",
        summary="Read-only four-mode gate that reports available, limited, and blocked operating modes from current proof evidence.",
        module="backend/autonomy_mode_transition_gate.py",
        can_analyze=True,
        pipeline_profiles=("shadow_quality", "lab_dsp"),
        blocked_reason="mode_gate_only_no_runtime_mode_change",
        promotion_next="use gate rows to explain why Supervised remains narrow and Full Autonomous remains blocked",
        notes=("transport_free", "mode_gate_only", "no_live_writes", "no_automatic_mode_change"),
    ),
    ProcessingMethod(
        id="read_surface_remediation_plan",
        label="Read Surface Remediation Plan",
        domain="evaluation",
        status="shadow_only",
        summary="Read-only plan for closing partial and missing console read-surface coverage before autonomous promotion.",
        module="backend/read_surface_remediation_plan.py",
        can_analyze=True,
        pipeline_profiles=("shadow_quality", "lab_dsp"),
        blocked_reason="read_remediation_only_no_console_queries",
        promotion_next="complete read DTOs, coverage flags, and fixture tests before broad autonomous console control",
        notes=("transport_free", "read_remediation_only", "no_live_writes", "mutates_console_state_false"),
    ),
    ProcessingMethod(
        id="read_surface_physical_readback_capture_plan",
        label="Read Surface Physical Readback Capture Plan",
        domain="evaluation",
        status="shadow_only",
        summary=(
            "Read-only operator checklist for capturing the next physical readback "
            "artifact required by console read-surface remediation."
        ),
        module="backend/read_surface_physical_readback_capture_plan.py",
        can_analyze=True,
        pipeline_profiles=("shadow_quality", "lab_dsp"),
        blocked_reason="operator_capture_plan_only_no_console_queries_or_execution",
        promotion_next=(
            "operator captures the named physical readback artifact outside this "
            "plan, then release checks validate the artifact before closing the gap"
        ),
        notes=(
            "transport_free",
            "readback_capture_plan_only",
            "no_live_writes",
            "does_not_query_console",
            "execution_allowed_by_plan_false",
        ),
    ),
    ProcessingMethod(
        id="read_surface_fixture_evidence",
        label="Read Surface Fixture Evidence",
        domain="evaluation",
        status="shadow_only",
        summary=(
            "Read-only evidence map for read-surface DTO fixture coverage and "
            "remaining physical readback gaps."
        ),
        module="backend/read_surface_fixture_evidence.py",
        can_analyze=True,
        pipeline_profiles=("shadow_quality", "lab_dsp"),
        blocked_reason="fixture_evidence_only_no_console_queries_no_test_execution",
        promotion_next=(
            "add physical readback fixtures and dLive/WING parity mappings before "
            "claiming complete console read awareness"
        ),
        notes=(
            "transport_free",
            "fixture_evidence_only",
            "no_live_writes",
            "does_not_execute_tests",
        ),
    ),
    ProcessingMethod(
        id="raw_mixer_state_physical_readback_contract",
        label="Raw Mixer State Physical Readback Contract",
        domain="evaluation",
        status="shadow_only",
        summary=(
            "Transport-free schema contract for the raw mixer-state physical "
            "readback artifact targeted by read-surface remediation."
        ),
        module="backend/raw_mixer_state_physical_readback_contract.py",
        can_analyze=True,
        pipeline_profiles=("shadow_quality", "lab_dsp"),
        blocked_reason="contract_only_physical_readback_artifact_missing",
        promotion_next=(
            "capture an operator-run physical readback artifact matching this "
            "contract before closing the raw mixer-state fixture gap"
        ),
        notes=(
            "transport_free",
            "readback_contract_only",
            "no_live_writes",
            "does_not_query_console",
        ),
    ),
    ProcessingMethod(
        id="channel_basic_controls_physical_readback_contract",
        label="Channel Basic Controls Physical Readback Contract",
        domain="evaluation",
        status="shadow_only",
        summary=(
            "Transport-free schema contract for channel identity and basic-control "
            "physical readback artifacts needed by fader, gain, mute, and pan proof."
        ),
        module="backend/channel_basic_controls_physical_readback_contract.py",
        can_analyze=True,
        pipeline_profiles=("shadow_quality", "lab_dsp"),
        blocked_reason="contract_only_channel_physical_readback_artifact_missing",
        promotion_next=(
            "capture an operator-run channel physical readback artifact matching "
            "this contract before closing the channel identity/basic controls gap"
        ),
        notes=(
            "transport_free",
            "readback_contract_only",
            "no_live_writes",
            "does_not_query_console",
        ),
    ),
    ProcessingMethod(
        id="channel_processing_physical_readback_contract",
        label="Channel Processing Physical Readback Contract",
        domain="evaluation",
        status="shadow_only",
        summary=(
            "Transport-free schema contract for channel HPF, EQ, compressor, "
            "and gate physical readback artifacts needed by processing proof."
        ),
        module="backend/channel_processing_physical_readback_contract.py",
        can_analyze=True,
        pipeline_profiles=("shadow_quality", "lab_dsp"),
        blocked_reason="contract_only_channel_processing_physical_readback_artifact_missing",
        promotion_next=(
            "capture an operator-run channel processing physical readback "
            "artifact before closing the channel EQ/dynamics/gate gap"
        ),
        notes=(
            "transport_free",
            "readback_contract_only",
            "no_live_writes",
            "does_not_query_console",
        ),
    ),
    ProcessingMethod(
        id="bus_group_main_dca_physical_readback_contract",
        label="Bus / Main / DCA Physical Readback Contract",
        domain="evaluation",
        status="shadow_only",
        summary=(
            "Transport-free schema contract for bus, main, and DCA physical "
            "readback artifacts needed by bus/group balance and main-bus proof."
        ),
        module="backend/bus_group_main_dca_physical_readback_contract.py",
        can_analyze=True,
        pipeline_profiles=("shadow_quality", "lab_dsp"),
        blocked_reason="contract_only_bus_group_main_dca_physical_readback_artifact_missing",
        promotion_next=(
            "capture an operator-run bus/group/main/DCA physical readback "
            "artifact before closing the bus/group/main/DCA gap"
        ),
        notes=(
            "transport_free",
            "readback_contract_only",
            "no_live_writes",
            "does_not_query_console",
        ),
    ),
    ProcessingMethod(
        id="matrix_state_physical_readback_contract",
        label="Matrix State Physical Readback Contract",
        domain="evaluation",
        status="shadow_only",
        summary=(
            "Transport-free schema contract for matrix controls, EQ, dynamics, "
            "inserts, and metering physical readback artifacts."
        ),
        module="backend/matrix_state_physical_readback_contract.py",
        can_analyze=True,
        pipeline_profiles=("shadow_quality", "lab_dsp"),
        blocked_reason="contract_only_matrix_state_physical_readback_artifact_missing",
        promotion_next=(
            "capture an operator-run matrix physical readback artifact before "
            "closing the matrix-state physical fixture gap"
        ),
        notes=(
            "transport_free",
            "readback_contract_only",
            "no_live_writes",
            "does_not_query_console",
        ),
    ),
    ProcessingMethod(
        id="fx_slots_inserts_physical_readback_contract",
        label="FX Slots / Inserts Physical Readback Contract",
        domain="evaluation",
        status="shadow_only",
        summary=(
            "Transport-free schema contract for FX slot, parameter, and insert "
            "assignment physical readback artifacts."
        ),
        module="backend/fx_slots_inserts_physical_readback_contract.py",
        can_analyze=True,
        pipeline_profiles=("shadow_quality", "lab_dsp"),
        blocked_reason="contract_only_fx_slots_inserts_physical_readback_artifact_missing",
        promotion_next=(
            "capture an operator-run FX slots/inserts physical readback artifact "
            "before closing the FX slots and insert assignment fixture gap"
        ),
        notes=(
            "transport_free",
            "readback_contract_only",
            "no_live_writes",
            "does_not_query_console",
        ),
    ),
    ProcessingMethod(
        id="routing_sends_physical_readback_contract",
        label="Routing / Sends Physical Readback Contract",
        domain="evaluation",
        status="shadow_only",
        summary=(
            "Transport-free schema contract for static routing, bus/main/matrix "
            "send, and output-route physical readback artifacts."
        ),
        module="backend/routing_sends_physical_readback_contract.py",
        can_analyze=True,
        pipeline_profiles=("shadow_quality", "lab_dsp"),
        blocked_reason="contract_only_routing_sends_physical_readback_artifact_missing",
        promotion_next=(
            "capture an operator-run routing/sends physical readback artifact "
            "before closing the routing and send fixture gap"
        ),
        notes=(
            "transport_free",
            "readback_contract_only",
            "no_live_writes",
            "does_not_query_console",
        ),
    ),
    ProcessingMethod(
        id="snapshots_scenes_physical_readback_contract",
        label="Snapshots / Scenes Physical Readback Contract",
        domain="evaluation",
        status="shadow_only",
        summary=(
            "Transport-free schema contract for snapshot catalog, active scene, "
            "scene diff, and rollback physical readback artifacts."
        ),
        module="backend/snapshots_scenes_physical_readback_contract.py",
        can_analyze=True,
        pipeline_profiles=("shadow_quality", "lab_dsp"),
        blocked_reason="contract_only_snapshots_scenes_physical_readback_artifact_missing",
        promotion_next=(
            "capture an operator-run snapshots/scenes physical readback artifact "
            "before closing scene content, diff, and rollback fixture gaps"
        ),
        notes=(
            "transport_free",
            "readback_contract_only",
            "no_live_writes",
            "does_not_query_console",
            "does_not_load_or_save_scenes",
        ),
    ),
    ProcessingMethod(
        id="user_presets_physical_readback_contract",
        label="User Presets Physical Readback Contract",
        domain="evaluation",
        status="shadow_only",
        summary=(
            "Transport-free schema contract for user-preset catalog, scope, "
            "target, preset content, scoped diff, and rollback physical readback artifacts."
        ),
        module="backend/user_presets_physical_readback_contract.py",
        can_analyze=True,
        pipeline_profiles=("shadow_quality", "lab_dsp"),
        blocked_reason="contract_only_user_presets_physical_readback_artifact_missing",
        promotion_next=(
            "capture an operator-run user-preset physical readback artifact "
            "before closing preset content, scoped diff, and rollback fixture gaps"
        ),
        notes=(
            "transport_free",
            "readback_contract_only",
            "no_live_writes",
            "does_not_query_console",
            "does_not_recall_or_store_presets",
        ),
    ),
    ProcessingMethod(
        id="dlive_state_physical_readback_contract",
        label="dLive State Physical Readback Contract",
        domain="evaluation",
        status="shadow_only",
        summary=(
            "Transport-free schema contract for dLive cached state, MIDI/TCP "
            "state counts, and WING parity physical readback artifacts."
        ),
        module="backend/dlive_state_physical_readback_contract.py",
        can_analyze=True,
        pipeline_profiles=("shadow_quality", "lab_dsp"),
        blocked_reason="contract_only_dlive_state_physical_readback_artifact_missing",
        promotion_next=(
            "capture an operator-run dLive state readback and WING parity "
            "artifact before closing the dLive parity fixture gap"
        ),
        notes=(
            "transport_free",
            "readback_contract_only",
            "no_live_writes",
            "does_not_query_console",
            "midi_tcp_writes_blocked",
        ),
    ),
    ProcessingMethod(
        id="auto_fader_v1",
        label="Auto Fader v1",
        domain="gain_fader",
        status="proposal_only",
        summary="Fader/balance automation candidate; may inform proposals, not direct live execution.",
        module="backend/auto_fader.py",
        can_analyze=True,
        can_propose=True,
        pipeline_profiles=("main_live_assist", "lab_dsp"),
        promotion_next="fake_mixer_proposal_adapter_tests",
    ),
    ProcessingMethod(
        id="lufs_gain_staging",
        label="Input Trim Autogain",
        domain="gain_fader",
        status="proposal_only",
        summary="Main input-trim autogain: headroom-first proposals/offline renders with local preamp character and distortion safeguards.",
        module="backend/lufs_gain_staging.py + tools/offline_input_trim_autogain_mix.py",
        can_analyze=True,
        can_propose=True,
        can_render_offline=True,
        pipeline_profiles=("main_live_assist", "offline_mix"),
    ),
    ProcessingMethod(
        id="live_shared_mix",
        label="Live Shared Mix Planner",
        domain="gain_fader",
        status="proposal_only",
        summary="Grouped balance planner adapted from offline shared-mix rules; no direct routing/name writes.",
        module="backend/live_shared_mix.py",
        can_analyze=True,
        can_propose=True,
        pipeline_profiles=("main_live_assist", "lab_dsp"),
    ),
    ProcessingMethod(
        id="bus_group_balance_proposal",
        label="Bus / Group Balance Proposal Layer",
        domain="bus_group",
        status="proposal_only",
        summary="Transport-free stem, group, DCA and matrix balance analyzer using Ayaic planes, summed-bus evidence, and fresh current assignment/fader state without live writes.",
        module="backend/bus_group_balance_proposal.py",
        can_analyze=True,
        can_propose=True,
        pipeline_profiles=("main_live_assist", "shadow_quality", "lab_dsp"),
        blocked_reason="bus_group_dca_matrix_live_writes_need_fresh_assignment_fader_readback_true_peak_rollback_and_hardware_proof",
        promotion_next="connect bus/group balance proposals to operator review only after stale assignment/fader cache states are blocked; keep DCA/group/matrix writes out of the live executor",
        notes=("transport_free", "proposal_only", "ayaic_bus_planes", "current_bus_group_freshness_gate", "no_live_writes"),
    ),
    ProcessingMethod(
        id="mute_management_proposal",
        label="Mute Management Proposal Layer",
        domain="mute",
        status="proposal_only",
        summary="Transport-free mute/unmute analyzer using source role, signal activity, operator intent, fresh current mute state and dropout/feedback risk without live writes.",
        module="backend/mute_management_proposal.py",
        can_analyze=True,
        can_propose=True,
        pipeline_profiles=("main_live_assist", "shadow_quality", "lab_dsp"),
        blocked_reason="mute_live_writes_need_fresh_intent_readback_rollback_and_hardware_proof",
        promotion_next="connect mute-state proposals to operator review only after stale mute-state cache is blocked; keep mute writes out of the live executor",
        notes=("transport_free", "proposal_only", "manual_mute_authority", "current_mute_freshness_gate", "no_live_writes"),
    ),
    ProcessingMethod(
        id="routing_management_proposal",
        label="Routing Management Proposal Layer",
        domain="routing",
        status="proposal_only",
        summary="Transport-free routing analyzer that emits route-diff proposals with fresh route readback, rollback, intent and activity evidence without live writes.",
        module="backend/routing_management_proposal.py",
        can_analyze=True,
        can_propose=True,
        pipeline_profiles=("main_live_assist", "shadow_quality", "lab_dsp"),
        blocked_reason="routing_live_writes_need_fresh_full_route_readback_rollback_scope_limits_and_hardware_proof",
        promotion_next="connect routing proposals to operator review only after stale route cache states are blocked; keep routing writes out of the live executor",
        notes=("transport_free", "proposal_only", "route_diff_evidence", "current_route_freshness_gate", "no_live_writes"),
    ),
    ProcessingMethod(
        id="scene_preset_management_proposal",
        label="Scene / Preset Management Proposal Layer",
        domain="scene_preset",
        status="proposal_only",
        summary="Transport-free scene/preset analyzer that emits scoped scene and user-preset proposals with fresh state readback, diff, rollback and intent evidence without live writes.",
        module="backend/scene_preset_management_proposal.py",
        can_analyze=True,
        can_propose=True,
        pipeline_profiles=("main_live_assist", "shadow_quality", "lab_dsp"),
        blocked_reason="scene_preset_live_writes_need_fresh_full_state_readback_scoped_diff_rollback_scope_limits_and_hardware_proof",
        promotion_next="connect scene/preset proposals to operator review only after stale scene/preset cache states are blocked; keep scene, snapshot and preset writes out of the live executor",
        notes=("transport_free", "proposal_only", "scoped_scene_preset_diff", "scoped_diff_evidence_contract", "rollback_evidence_contract", "current_scene_preset_freshness_gate", "product_snapshot_freshness_bridge", "snapshots_scenes_snapshot_bridge", "user_presets_snapshot_bridge", "no_live_writes"),
    ),
    ProcessingMethod(
        id="mix_variant_profile_proposal",
        label="Mix Variant Profile Proposal Layer",
        domain="mix_variant",
        status="proposal_only",
        summary="Transport-free genre/task mix profile analyzer using fresh arrangement/reference context, Ayaic level planes and safety evidence without live writes.",
        module="backend/mix_variant_profile_proposal.py",
        can_analyze=True,
        can_propose=True,
        pipeline_profiles=("main_live_assist", "offline_mix", "shadow_quality", "lab_dsp"),
        blocked_reason="mix_variant_profiles_need_fresh_reference_arrangement_operator_review_and_followup_proposals_before_any_live_or_preset_apply",
        promotion_next="route fresh-context mix-variant profiles to operator review, then fan out through proposal-only bus, FX, master and scene/preset layers",
        notes=("transport_free", "proposal_only", "ayaic_level_planes", "current_mix_variant_context_freshness_gate", "no_style_clone", "no_live_writes"),
    ),
    ProcessingMethod(
        id="performance_adaptation_proposal",
        label="Performance Adaptation Proposal Layer",
        domain="adaptation",
        status="proposal_only",
        summary="Transport-free level-drift analyzer for musician, concert-time and real-time adaptation proposals with fresh trend/readback, cooldown, headroom and feedback evidence without live writes.",
        module="backend/performance_adaptation_proposal.py",
        can_analyze=True,
        can_propose=True,
        pipeline_profiles=("main_live_assist", "shadow_quality", "lab_dsp"),
        blocked_reason="performance_adaptation_needs_fresh_trend_readback_replay_hardware_proof_and_rate_limited_supervised_family_promotion_before_apply",
        promotion_next="compare fresh-state adaptation proposals against engineer actions in replay, then promote only one reversible fader/gain family at a time",
        notes=("transport_free", "proposal_only", "rate_limited", "current_performance_context_freshness_gate", "no_live_writes"),
    ),
    ProcessingMethod(
        id="arrangement_role_proposal",
        label="Arrangement Role Proposal Layer",
        domain="arrangement",
        status="proposal_only",
        summary="Transport-free arrangement-role analyzer using source type, section, fresh inter-channel/arrangement context, reference and accepted-history evidence without live writes.",
        module="backend/arrangement_role_proposal.py",
        can_analyze=True,
        can_propose=True,
        pipeline_profiles=("main_live_assist", "shadow_quality", "lab_dsp"),
        blocked_reason="arrangement_role_learning_needs_fresh_context_operator_acceptance_and_replay_correlation_before_authority",
        promotion_next="attach fresh-context role proposals to operator review and replay accepted/rejected role decisions before feeding downstream mix actions",
        notes=("transport_free", "proposal_only", "current_arrangement_role_context_freshness_gate", "decision_evidence_packet", "no_live_writes"),
    ),
    ProcessingMethod(
        id="decision_evidence_packet_proposal",
        label="Decision Evidence Packet Proposal Layer",
        domain="decision_evidence",
        status="proposal_only",
        summary="Transport-free evidence-packet analyzer tying fresh audio, psychoacoustic, statistical, history, reference, inter-channel, arrangement and genre evidence to operator-review decisions without live writes.",
        module="backend/decision_evidence_packet_proposal.py",
        can_analyze=True,
        can_propose=True,
        pipeline_profiles=("main_live_assist", "shadow_quality", "lab_dsp"),
        blocked_reason="decision_evidence_packets_need_fresh_context_and_are_review_contracts_not_live_apply_authority",
        promotion_next="attach fresh-context evidence packets to all proposal and rejected shadow decisions, then replay against accepted engineer actions before any authority promotion",
        notes=("transport_free", "proposal_only", "current_decision_evidence_context_freshness_gate", "decision_evidence_packet", "no_live_writes"),
    ),
    ProcessingMethod(
        id="long_term_memory_learning_proposal",
        label="Long-Term Memory Learning Proposal Layer",
        domain="memory_learning",
        status="proposal_only",
        summary="Transport-free learning-record analyzer using fresh verified decisions, offline replay, telemetry, operator outcomes, evidence packets and safety regressions without live writes.",
        module="backend/long_term_memory_learning_proposal.py",
        can_analyze=True,
        can_propose=True,
        pipeline_profiles=("main_live_assist", "shadow_quality", "lab_dsp"),
        blocked_reason="memory_learning_records_require_fresh_context_operator_acceptance_and_replay_correlation_before_authority",
        promotion_next="attach fresh-context learning proposals to memory review, then compare future proposals against accepted engineer actions before any autonomous influence",
        notes=("transport_free", "proposal_only", "current_memory_learning_context_freshness_gate", "decision_memory", "no_live_writes"),
    ),
    ProcessingMethod(
        id="agentic_method_foundation",
        label="Agentic Feature / Recommendation Foundation",
        domain="evaluation",
        status="shadow_only",
        summary="Transport-free feature extraction, inter-channel context and dry-run recommendations for future gain/EQ/dynamics/pan/FX agents.",
        module="backend/agentic_method_foundation.py",
        can_analyze=True,
        can_propose=True,
        pipeline_profiles=("main_live_assist", "shadow_quality", "lab_dsp"),
        blocked_reason="dry_run_recommendation_layer_no_live_transport",
        promotion_next="connect selected outputs through proposal adapters and supervised fader/gain gate only",
        notes=("no_osc_imports", "all_recommendations_dry_run_by_default"),
    ),
    ProcessingMethod(
        id="context_aware_gate_agent",
        label="Context-Aware Gate / Bleed Agent",
        domain="gate",
        status="proposal_only",
        summary=(
            "ACAD-inspired dry-run gate/expander recommendations using channel "
            "context, bleed correlation, spectral outliers, false-open/false-close "
            "risk, freshness, and source identity trust."
        ),
        module="backend/context_aware_gate.py",
        can_analyze=True,
        can_propose=True,
        pipeline_profiles=("main_live_assist", "shadow_quality", "lab_dsp"),
        blocked_reason=(
            "gate_expander_live_writes_need_fresh_bleed_telemetry_source_identity_"
            "trust_readback_rollback_and_physical_proof"
        ),
        promotion_next=(
            "collect live/offline bleed replay evidence with trusted source identity; "
            "keep gate/expander changes proposal-only until separate physical proof exists"
        ),
        notes=(
            "no_live_transport",
            "dry_run_true",
            "decision_trace_ready",
            "current_gate_bleed_context_freshness_gate",
            "source_identity_downstream_trust",
        ),
    ),
    ProcessingMethod(
        id="auto_eq",
        label="Auto EQ",
        domain="eq",
        status="proposal_only",
        summary="Single/multi-channel EQ analysis may propose changes; direct apply/reset paths are fail-closed blocked evidence.",
        module="backend/auto_eq.py",
        can_analyze=True,
        can_propose=True,
        pipeline_profiles=("lab_dsp",),
        blocked_reason="eq_live_writes_need_filter_audit_feedback_safety_readback_rollback_and_hardware_proof",
        promotion_next="proposal_adapter_blocked_eq_evidence: route blocked EQ evidence to operator review; require filter audit, feedback safety, readback and rollback proof before live apply",
        notes=("proposal_only", "direct_eq_apply_quarantined", "direct_eq_reset_quarantined", "blocked_apply_evidence", "no_live_writes"),
    ),
    ProcessingMethod(
        id="autoeq_audit_validation",
        label="AutoEQ Audit Validation",
        domain="eq",
        status="shadow_only",
        summary="Transport-free audit checks for EQ units, phase contracts, IIR stability, split leakage, export reproducibility and headroom.",
        module="backend/autoeq_audit.py",
        can_analyze=True,
        pipeline_profiles=("shadow_quality", "lab_dsp"),
        blocked_reason="audit_only_no_mixer_transport",
        promotion_next="require_passed_audit_before_any_future_eq_apply_promotion",
        notes=("no_osc_imports", "no_live_writes", "ci_gate_for_filter_packages"),
    ),
    ProcessingMethod(
        id="diff_deq_playbook_bridge",
        label="Diff-DEQ Dynamic EQ Playbook Bridge",
        domain="eq",
        status="shadow_only",
        summary="Imports external six-band PEQ+DRC dynamic-EQ predictions into dry-run playbook entries with safety validation.",
        module="backend/dynamic_eq_playbook.py",
        can_analyze=True,
        can_propose=True,
        pipeline_profiles=("shadow_quality", "lab_dsp"),
        blocked_reason="research_import_only_no_model_runtime_no_live_transport",
        promotion_next="use_as_offline_fixture_source_before_any_operator_facing_eq_dynamics_proposal",
        notes=("no_model_weights", "no_torch_dependency", "no_live_writes", "peq_drc_predictions_shadow_only"),
    ),
    ProcessingMethod(
        id="cross_adaptive_eq",
        label="Cross-Adaptive EQ",
        domain="eq",
        status="proposal_only",
        summary=(
            "Masking-aware mirror EQ proposal layer that emits linked cut/boost "
            "candidates with freshness-gated masking context, source identity "
            "downstream trust, and no live EQ writes."
        ),
        module="backend/cross_adaptive_eq.py",
        can_analyze=True,
        can_propose=True,
        pipeline_profiles=("lab_dsp", "offline_mix"),
        blocked_reason=(
            "eq_live_writes_need_fresh_masking_context_band_readback_sum_bus_check_"
            "source_identity_trust_and_hardware_proof"
        ),
        promotion_next=(
            "route fresh-context linked unmasking proposals to operator review "
            "with trusted source identity and rollback EQ-band evidence; keep direct "
            "EQ writes quarantined"
        ),
        notes=(
            "transport_free",
            "dry_run_true",
            "current_masking_context_freshness_gate",
            "source_identity_downstream_trust",
            "linked_mirror_eq_pair",
            "no_live_writes",
        ),
    ),
    ProcessingMethod(
        id="frequency_separation_proposal",
        label="Frequency Separation Proposal",
        domain="eq",
        status="proposal_only",
        summary=(
            "Transport-free source/stem frequency-lane analyzer that emits "
            "freshness-gated split proposals with contribution and rollback evidence."
        ),
        module="backend/frequency_separation_proposal.py",
        can_analyze=True,
        can_propose=True,
        pipeline_profiles=("lab_dsp", "offline_mix", "main_live_assist"),
        blocked_reason=(
            "frequency_split_live_writes_need_fresh_source_stem_spectrum_readback_"
            "sum_bus_check_rollback_and_hardware_proof"
        ),
        promotion_next=(
            "route fresh-context frequency-lane proposals to operator review; "
            "keep derived EQ or fader moves quarantined until readback and rollback proof"
        ),
        notes=(
            "transport_free",
            "proposal_only",
            "source_stem_contribution",
            "current_frequency_context_freshness_gate",
            "no_live_writes",
        ),
    ),
    ProcessingMethod(
        id="auto_compressor",
        label="Auto Compressor",
        domain="dynamics",
        status="proposal_only",
        summary="Dynamics analysis may recommend compressor settings; direct compressor enable/apply paths are fail-closed blocked evidence.",
        module="backend/auto_compressor.py",
        can_analyze=True,
        can_propose=True,
        pipeline_profiles=("lab_dsp",),
        blocked_reason="compressor_live_writes_need_fresh_dynamics_readback_makeup_headroom_rollback_and_hardware_proof",
        promotion_next="route blocked compressor evidence to operator review; require dynamics readback, makeup/headroom proof and rollback before live apply",
        notes=("proposal_only", "direct_compressor_apply_quarantined", "blocked_apply_evidence", "bounded_makeup", "no_live_writes"),
    ),
    ProcessingMethod(
        id="compressor_dynamics_proposal",
        label="Compressor Dynamics Proposal Layer",
        domain="dynamics",
        status="proposal_only",
        summary="Transport-free dynamics analyzer that emits compressor proposals without live writes and gates stale cached dynamics state.",
        module="backend/compressor_dynamics_proposal.py",
        can_analyze=True,
        can_propose=True,
        pipeline_profiles=("main_live_assist", "shadow_quality", "lab_dsp"),
        blocked_reason="compressor_live_writes_need_dynamics_readback_rollback_and_hardware_proof",
        promotion_next="connect compressor proposals to fresh dynamics readback and operator review; keep dynamics writes out of the live executor",
        notes=("transport_free", "proposal_only", "bounded_makeup", "freshness_gated", "no_live_writes"),
    ),
    ProcessingMethod(
        id="auto_compressor_cf",
        label="Crest-Factor Compressor",
        domain="dynamics",
        status="proposal_only",
        summary="Experimental crest-factor/LUFS compressor variant; direct apply path is fail-closed blocked evidence.",
        module="backend/auto_compressor_cf.py",
        can_analyze=True,
        can_propose=True,
        pipeline_profiles=("lab_dsp",),
        blocked_reason="cf_compressor_live_writes_need_fresh_dynamics_readback_makeup_headroom_rollback_and_hardware_proof",
        promotion_next="use CF/LUFS parameters as proposal evidence only until hardware proof and supervised dynamics apply exist",
        notes=("proposal_only", "direct_cf_compressor_apply_quarantined", "blocked_apply_evidence", "no_live_writes"),
    ),
    ProcessingMethod(
        id="deesser_vocal_sibilance",
        label="Vocal De-esser Proposal Layer",
        domain="dynamics",
        status="proposal_only",
        summary="Transport-free vocal sibilance detector that emits de-esser proposals without live writes and gates stale dynamics state.",
        module="backend/deesser_vocal_sibilance.py",
        can_analyze=True,
        can_propose=True,
        pipeline_profiles=("main_live_assist", "shadow_quality", "lab_dsp"),
        blocked_reason="deesser_live_writes_need_dynamics_or_dynamic_eq_hardware_proof",
        promotion_next="connect proposals to fresh dynamics or dynamic-EQ readback; keep de-esser writes out of the live executor",
        notes=("transport_free", "proposal_only", "freshness_gated", "no_live_writes"),
    ),
    ProcessingMethod(
        id="auto_gate_caig",
        label="CAIG Auto Gate",
        domain="gate",
        status="proposal_only",
        summary="Gate analysis and recommendation source; direct _send_osc is fail-closed and not a live executor.",
        module="backend/auto_gate_caig.py",
        can_analyze=True,
        can_propose=True,
        pipeline_profiles=("lab_dsp",),
        blocked_reason="gate_live_writes_need_false_decision_replay_physical_readback_and_rollback_proof",
        promotion_next="keep direct CAIG gate sends quarantined; promote only through a proposal-only contract with physical proof",
        notes=("proposal_only", "direct_gate_send_quarantined", "dry_run_default", "no_live_writes"),
    ),
    ProcessingMethod(
        id="auto_panner",
        label="Auto Panner",
        domain="pan",
        status="proposal_only",
        summary="Panning automation candidate for offline/proposal review; direct apply_to_mixer and adaptive _send_osc are fail-closed and do not mutate mixer state.",
        module="backend/auto_panner.py + backend/auto_panner_adaptive.py",
        can_analyze=True,
        can_propose=True,
        pipeline_profiles=("lab_dsp", "offline_mix"),
        blocked_reason="pan_live_writes_need_current_pan_readback_mono_check_and_rollback_proof",
        promotion_next="keep AutoPanner, adaptive panner, and pan-separation apply calls as blocked evidence; promote only through proposal review plus physical pan readback proof",
        notes=(
            "proposal_only",
            "direct_pan_apply_quarantined",
            "adaptive_pan_send_quarantined",
            "blocked_apply_evidence",
            "no_live_writes",
        ),
    ),
    ProcessingMethod(
        id="pan_spatial_proposal",
        label="Spatial Pan Proposal Layer",
        domain="pan",
        status="proposal_only",
        summary="Transport-free spatial analyzer that emits pan proposals from role, stereo-pair, stage-position, mono-compatibility, and fresh current-pan evidence.",
        module="backend/pan_spatial_proposal.py",
        can_analyze=True,
        can_propose=True,
        pipeline_profiles=("main_live_assist", "shadow_quality", "lab_dsp"),
        blocked_reason="pan_live_writes_need_fresh_pan_readback_mono_check_and_hardware_proof",
        promotion_next="connect pan proposals to operator review only after stale current-pan cache states are blocked; keep pan writes out of the live executor",
        notes=("transport_free", "proposal_only", "current_pan_freshness_gate", "mono_compatibility_guard", "no_live_writes"),
    ),
    ProcessingMethod(
        id="phase_alignment",
        label="Phase / Delay Alignment",
        domain="phase",
        status="proposal_only",
        summary="Phase and delay analysis can produce recommendations; direct delay/polarity apply and reset are fail-closed blocked evidence.",
        module="backend/phase_alignment.py + backend/auto_phase_gcc_phat.py",
        can_analyze=True,
        can_propose=True,
        pipeline_profiles=("lab_dsp", "offline_mix"),
        blocked_reason="phase_delay_polarity_live_writes_need_fresh_phase_context_readback_rollback_and_hardware_proof",
        promotion_next="connect blocked phase/delay evidence to operator review only after fresh phase context plus delay/polarity readback and rollback proof exist",
        notes=("proposal_only", "phase_delay_apply_quarantined", "phase_delay_reset_quarantined", "blocked_apply_evidence", "no_live_writes"),
    ),
    ProcessingMethod(
        id="phase_aware_eq_guard",
        label="Phase-Aware EQ Guard",
        domain="phase",
        status="proposal_only",
        summary="Pi-inspired original group phase guard that blocks risky EQ boosts and emits delay/polarity proposals only from fresh phase context without live writes.",
        module="backend/phase_aware_eq_guard.py",
        can_analyze=True,
        can_propose=True,
        can_render_offline=True,
        pipeline_profiles=("shadow_quality", "lab_dsp", "offline_mix"),
        blocked_reason="phase_delay_polarity_live_writes_need_fresh_phase_context_and_separate_supervised_hardware_proof",
        promotion_next="connect only fresh-context proposal records to operator review; keep WING phase/delay writes quarantined",
        notes=("transport_free", "dry_run_true", "current_phase_context_freshness_gate", "no_pace_or_proprietary_pi_logic", "phase_before_eq_guard"),
    ),
    ProcessingMethod(
        id="auto_fx",
        label="FX / Reverb / Sends",
        domain="fx",
        status="proposal_only",
        summary="FX, reverb and send planners emit reviewable plans; direct apply_to_mixer paths are fail-closed and do not mutate mixer state.",
        module="backend/auto_fx.py + backend/auto_reverb.py + backend/auto_effects.py",
        can_analyze=True,
        can_propose=True,
        pipeline_profiles=("lab_dsp", "offline_mix"),
        blocked_reason="fx_rack_send_live_writes_need_send_return_readback_feedback_check_and_rollback_proof",
        promotion_next="keep AutoFX and AutoReverb apply calls as blocked evidence; promote only through proposal review plus physical FX readback proof",
        notes=(
            "proposal_only",
            "direct_fx_apply_quarantined",
            "direct_reverb_apply_quarantined",
            "blocked_apply_evidence",
            "no_live_writes",
        ),
    ),
    ProcessingMethod(
        id="fx_send_proposal",
        label="FX Send Proposal Layer",
        domain="fx",
        status="proposal_only",
        summary="Transport-free FX send analyzer that emits send/return proposals from fresh current-FX state and scoped Product Layer FX snapshot schema hints without live writes.",
        module="backend/fx_send_proposal.py",
        can_analyze=True,
        can_propose=True,
        pipeline_profiles=("main_live_assist", "shadow_quality", "lab_dsp"),
        blocked_reason="fx_send_live_writes_need_fresh_send_return_readback_schema_hints_feedback_check_and_hardware_proof",
        promotion_next="connect FX send proposals to operator review only after stale send/return cache states and unknown FX parameter schemas are blocked; keep FX writes out of the live executor",
        notes=("transport_free", "proposal_only", "current_fx_freshness_gate", "fx_snapshot_freshness_gate", "fx_parameter_schema_gate", "fx_inserts_snapshot_schema_bridge", "batch_fx_snapshot_schema_bridge", "feedback_guard", "no_live_writes"),
    ),
    ProcessingMethod(
        id="master_bus_optimization_proposal",
        label="Master Bus Optimization Proposal Layer",
        domain="master_bus",
        status="proposal_only",
        summary="Transport-free master bus analyzer that emits loudness, true-peak, tonal, and glue-compression proposals from fresh current main-bus state without live writes.",
        module="backend/master_bus_optimization_proposal.py",
        can_analyze=True,
        can_propose=True,
        pipeline_profiles=("main_live_assist", "shadow_quality", "lab_dsp"),
        blocked_reason="master_bus_live_writes_need_fresh_main_bus_readback_limiter_dynamics_rollback_and_hardware_proof",
        promotion_next="connect master-bus proposals to operator review only after stale main-bus processing cache states are blocked; keep main bus writes out of the live executor",
        notes=("transport_free", "proposal_only", "current_main_bus_freshness_gate", "true_peak_guard", "no_live_writes"),
    ),
    ProcessingMethod(
        id="mix_agent_conservative",
        label="Mix Agent Conservative Renderer",
        domain="offline_render",
        status="offline_only",
        summary="Offline conservative render/report path; unsupported actions remain placeholders.",
        module="mix_agent/",
        can_analyze=True,
        can_propose=True,
        can_render_offline=True,
        requires_supervision=False,
        pipeline_profiles=("offline_mix",),
    ),
    ProcessingMethod(
        id="offline_agent_mix",
        label="Offline Agent Mix",
        domain="offline_render",
        status="offline_only",
        summary="Extended offline renderer with references, FX and reporting; no live console imports by policy.",
        module="tools/offline_agent_mix.py",
        can_analyze=True,
        can_propose=True,
        can_render_offline=True,
        requires_supervision=False,
        pipeline_profiles=("offline_mix",),
    ),
    ProcessingMethod(
        id="muq_perceptual_shadow",
        label="MuQ / Perceptual Shadow Quality",
        domain="evaluation",
        status="shadow_only",
        summary="Quality scoring and warnings may prioritize proposals but never execute console writes.",
        module="backend/evaluation/muq_eval_service.py + backend/perceptual/",
        can_analyze=True,
        pipeline_profiles=("shadow_quality",),
    ),
    ProcessingMethod(
        id="ml_processing_graph",
        label="ML Processing Graph",
        domain="evaluation",
        status="lab",
        summary="Research graph/differentiable processors; not part of live executor.",
        module="backend/ml/",
        can_analyze=True,
        can_propose=True,
        pipeline_profiles=("shadow_quality", "lab_dsp"),
    ),
    ProcessingMethod(
        id="autoeq_direct_apply",
        label="Direct AutoEQ Apply",
        domain="eq",
        status="quarantined",
        summary="Lower-level EQ write path must not enter main live executor.",
        module="backend/auto_eq.py",
        blocked_reason="direct_set_eq_surface",
        pipeline_profiles=("quarantine",),
    ),
    ProcessingMethod(
        id="compressor_direct_apply",
        label="Direct Compressor Apply",
        domain="dynamics",
        status="quarantined",
        summary="Compressor writes stay blocked on WING deployment boundary.",
        module="backend/auto_compressor.py + backend/server.py",
        blocked_reason="direct_set_compressor_surface",
        pipeline_profiles=("quarantine",),
    ),
    ProcessingMethod(
        id="routing_snapshot_reset_writes",
        label="Routing / Snapshot / Reset Writes",
        domain="evaluation",
        status="quarantined",
        summary="Routing, snapshot, reset and input-routing writes are not processing pipeline executors.",
        module="backend/wing_client.py + backend/handlers/",
        blocked_reason="destructive_or_broad_mixer_state_surface",
        pipeline_profiles=("quarantine",),
    ),
    ProcessingMethod(
        id="fx_direct_apply",
        label="Direct FX / Insert Apply",
        domain="fx",
        status="quarantined",
        summary="FX rack, insert, send and FX parameter writes are blocked from the main live executor.",
        module="backend/auto_fx.py + backend/handlers/fx_handlers.py",
        blocked_reason="direct_fx_or_insert_write_surface",
        pipeline_profiles=("quarantine",),
    ),
    ProcessingMethod(
        id="mix_agent_unsupported_actions",
        label="Mix Agent Unsupported Actions",
        domain="dynamics",
        status="placeholder",
        summary="Compression, sends, stereo width, dynamic EQ and sidechain are advisory placeholders in mix_agent.",
        module="mix_agent/actions/",
        blocked_reason="placeholder_not_applied",
        pipeline_profiles=("offline_mix", "quarantine"),
    ),
    ProcessingMethod(
        id="automixer_production_mix_v1",
        label="Production Mix v1 Package",
        domain="offline_render",
        status="placeholder",
        summary="Documented offline package is not a current executable entrypoint in this checkout.",
        module="automixer/production_mix_v1/",
        blocked_reason="empty_or_missing_entrypoint",
        pipeline_profiles=("offline_mix", "quarantine"),
    ),
    ProcessingMethod(
        id="ai_mixing_pipeline_legacy",
        label="AI Mixing Pipeline Legacy Skeleton",
        domain="offline_render",
        status="deprecated",
        summary="Older empty/legacy pipeline skeleton; not selected as a current executor.",
        module="ai_mixing_pipeline/",
        blocked_reason="legacy_empty_skeleton",
        pipeline_profiles=("quarantine",),
    ),
)


PIPELINE_PROFILES: Tuple[PipelineProfile, ...] = (
    PipelineProfile(
        id="main_live_assist",
        label="Main Live Assist",
        status="canonical",
        summary="Live-safe operator pipeline: analyze, propose, safety-check and apply only supervised channel fader/gain.",
        method_ids=(
            "product_operator_state",
            "auto_soundcheck_engine",
            "source_identity_confidence_proposal",
            "source_identity_replay_calibration",
            "auto_fader_v1",
            "lufs_gain_staging",
            "live_shared_mix",
            "bus_group_balance_proposal",
            "mute_management_proposal",
            "routing_management_proposal",
            "scene_preset_management_proposal",
            "mix_variant_profile_proposal",
            "performance_adaptation_proposal",
            "arrangement_role_proposal",
            "decision_evidence_packet_proposal",
            "long_term_memory_learning_proposal",
            "agentic_method_foundation",
            "context_aware_gate_agent",
            "compressor_dynamics_proposal",
            "deesser_vocal_sibilance",
            "pan_spatial_proposal",
            "fx_send_proposal",
            "master_bus_optimization_proposal",
            "supervised_manual_fader_gain",
        ),
        proposal_method_ids=(
            "auto_soundcheck_engine",
            "source_identity_confidence_proposal",
            "source_identity_replay_calibration",
            "auto_fader_v1",
            "lufs_gain_staging",
            "live_shared_mix",
            "bus_group_balance_proposal",
            "mute_management_proposal",
            "routing_management_proposal",
            "scene_preset_management_proposal",
            "mix_variant_profile_proposal",
            "performance_adaptation_proposal",
            "arrangement_role_proposal",
            "decision_evidence_packet_proposal",
            "long_term_memory_learning_proposal",
            "agentic_method_foundation",
            "context_aware_gate_agent",
            "compressor_dynamics_proposal",
            "deesser_vocal_sibilance",
            "pan_spatial_proposal",
            "fx_send_proposal",
            "master_bus_optimization_proposal",
        ),
        executor_method_ids=("supervised_manual_fader_gain",),
        can_write_live=True,
        requires_supervision=True,
        live_write_kinds=LIVE_ALLOWED_WRITE_KINDS,
        mode="operator_supervised",
        notes=("not_autonomous_live_mixing", "no_eq_or_compressor_apply_in_v1"),
    ),
    PipelineProfile(
        id="offline_mix",
        label="Offline Mix",
        status="offline_only",
        summary="Offline render and comparison pipeline; produces WAV/report/JSON traces without OSC.",
        method_ids=(
            "mix_agent_conservative",
            "offline_agent_mix",
            "lufs_gain_staging",
            "cross_adaptive_eq",
            "auto_panner",
            "phase_alignment",
            "phase_aware_eq_guard",
            "auto_fx",
            "mix_variant_profile_proposal",
        ),
        blocked_method_ids=("mix_agent_unsupported_actions", "automixer_production_mix_v1"),
        can_write_live=False,
        requires_supervision=False,
        mode="offline_render",
    ),
    PipelineProfile(
        id="shadow_quality",
        label="Shadow Quality",
        status="shadow_only",
        summary="Quality/evaluation pipeline; returns scores, warnings and confidence only.",
        method_ids=(
            "agentic_method_foundation",
            "source_identity_confidence_proposal",
            "source_identity_replay_calibration",
            "bus_group_balance_proposal",
            "mute_management_proposal",
            "routing_management_proposal",
            "scene_preset_management_proposal",
            "mix_variant_profile_proposal",
            "performance_adaptation_proposal",
            "arrangement_role_proposal",
            "decision_evidence_packet_proposal",
            "long_term_memory_learning_proposal",
            "context_aware_gate_agent",
            "supervised_proof_artifact_evaluator",
            "supervised_promotion_evidence_bridge",
            "supervised_wing_proof_plan",
            "supervised_proof_remediation_plan",
            "autonomy_evidence_overlay",
            "autonomy_mode_transition_gate",
            "read_surface_remediation_plan",
            "autoeq_audit_validation",
            "diff_deq_playbook_bridge",
            "compressor_dynamics_proposal",
            "deesser_vocal_sibilance",
            "pan_spatial_proposal",
            "fx_send_proposal",
            "master_bus_optimization_proposal",
            "phase_aware_eq_guard",
            "muq_perceptual_shadow",
            "ml_processing_graph",
        ),
        can_write_live=False,
        requires_supervision=False,
        mode="shadow_evaluation",
    ),
    PipelineProfile(
        id="lab_dsp",
        label="Lab DSP",
        status="lab",
        summary="DSP research/proposal pool kept outside live runtime defaults.",
        method_ids=(
            "auto_eq",
            "source_identity_confidence_proposal",
            "source_identity_replay_calibration",
            "cross_adaptive_eq",
            "autoeq_audit_validation",
            "diff_deq_playbook_bridge",
            "auto_compressor",
            "compressor_dynamics_proposal",
            "auto_compressor_cf",
            "deesser_vocal_sibilance",
            "context_aware_gate_agent",
            "auto_gate_caig",
            "auto_panner",
            "pan_spatial_proposal",
            "phase_alignment",
            "phase_aware_eq_guard",
            "auto_fx",
            "fx_send_proposal",
            "master_bus_optimization_proposal",
            "bus_group_balance_proposal",
            "mute_management_proposal",
            "routing_management_proposal",
            "scene_preset_management_proposal",
            "mix_variant_profile_proposal",
            "performance_adaptation_proposal",
            "arrangement_role_proposal",
            "decision_evidence_packet_proposal",
            "long_term_memory_learning_proposal",
            "agentic_method_foundation",
            "supervised_proof_artifact_evaluator",
            "supervised_promotion_evidence_bridge",
            "supervised_wing_proof_plan",
            "ml_processing_graph",
        ),
        can_write_live=False,
        requires_supervision=False,
        mode="proposal_or_lab_only",
    ),
    PipelineProfile(
        id="quarantine",
        label="Quarantine",
        status="quarantined",
        summary="Known direct-write, placeholder, empty or deprecated paths blocked from canonical execution.",
        method_ids=(
            "autoeq_direct_apply",
            "compressor_direct_apply",
            "routing_snapshot_reset_writes",
            "fx_direct_apply",
            "mix_agent_unsupported_actions",
            "automixer_production_mix_v1",
            "ai_mixing_pipeline_legacy",
        ),
        can_write_live=False,
        requires_supervision=True,
        mode="blocked",
        notes=("requires_adr_and_tests_before_promotion",),
    ),
)


PROMOTION_LADDER = (
    {
        "from": "lab",
        "to": "proposal_only",
        "requirements": [
            "adapter emits dry-run trace",
            "tests prove no direct mixer writes",
            "operator-facing reason and confidence are present",
        ],
    },
    {
        "from": "proposal_only",
        "to": "supervised_live_candidate",
        "requirements": [
            "fake mixer tests cover bounds and blocked cases",
            "readback and rollback schema exists",
            "cooldown and emergency-stop behavior are preserved",
        ],
    },
    {
        "from": "supervised_live_candidate",
        "to": "canonical_live",
        "requirements": [
            "physical WING proof with readback",
            "rollback succeeds",
            "cooldown and emergency stop are verified",
            "operator approval is explicit",
        ],
    },
)

MODULE_STATUS_OVERRIDES: Dict[str, Dict[str, str]] = {
    "product_operator_state": {
        "entrypoint": "get_dashboard_snapshot / get_decision_queue / get_safety_monitor",
        "ui_surface": "Control Center dashboard, Task Queue, Safety Monitor",
        "next_step": "Keep as the canonical read model and route new automation through proposals.",
    },
    "auto_soundcheck_engine": {
        "entrypoint": "run_operator_analysis / import_soundcheck_recommendations",
        "ui_surface": "Dashboard analysis and Task Queue proposals",
        "next_step": "Keep live behavior guarded; emit fader/gain proposals before any apply.",
    },
    "source_identity_confidence_proposal": {
        "entrypoint": "source_identity_confidence_proposal.analyze_source_identity_confidence / build_source_identity_confidence_proposals",
        "ui_surface": "Channel Intelligence, source identity review, downstream proposal priority evidence",
        "next_step": "Keep source identity review proposal-only; replay accepted/rejected labels before allowing downstream proposals to trust low-confidence classifications.",
    },
    "source_identity_replay_calibration": {
        "entrypoint": "source_identity_replay_calibration.analyze_source_identity_replay_calibration / build_source_identity_replay_calibration_proposals",
        "ui_surface": "Channel Intelligence, replay calibration evidence, downstream proposal priority review",
        "next_step": "Use accepted/rejected replay outcomes only as operator-reviewed calibration evidence before changing source-specific proposal priority.",
    },
    "supervised_manual_fader_gain": {
        "entrypoint": "apply_operator_proposal -> supervised manual gate",
        "ui_surface": "Task Queue supervised apply",
        "next_step": "Complete separate physical gain proof before widening readiness claims.",
    },
    "supervised_proof_artifact_evaluator": {
        "entrypoint": "evaluate_supervised_proof_artifact / tools/evaluate_supervised_proof_artifacts.py",
        "ui_surface": "Processing registry, safety reports, proof artifact review",
        "next_step": "Evaluate dry-run and physical proof artifacts before changing safety coverage or promotion readiness rows.",
    },
    "supervised_promotion_evidence_bridge": {
        "entrypoint": "build_supervised_promotion_evidence_snapshot",
        "ui_surface": "Safety reports, proof artifact report, promotion readiness review",
        "next_step": "Use report evidence to justify manual safety-row review; never auto-promote live authority.",
    },
    "supervised_wing_proof_plan": {
        "entrypoint": "build_supervised_wing_proof_plan",
        "ui_surface": "Proof artifact report, next WING test runbook",
        "next_step": "Use only as an operator checklist; do not execute physical proof without explicit approval.",
    },
    "supervised_proof_remediation_plan": {
        "entrypoint": "build_supervised_proof_remediation_plan",
        "ui_surface": "Proof artifact report, blocked proof remediation checklist",
        "next_step": "Use only to identify rerun/fix actions for blocked proof artifacts; do not execute physical reruns without explicit approval.",
    },
    "autonomy_evidence_overlay": {
        "entrypoint": "build_autonomy_evidence_overlay",
        "ui_surface": "Proof artifact report, autonomy promotion evidence review",
        "next_step": "Use only to connect proof artifacts to manual safety review; do not auto-promote any live authority.",
    },
    "autonomy_mode_transition_gate": {
        "entrypoint": "build_autonomy_mode_transition_gate",
        "ui_surface": "Proof artifact report, operating mode readiness gate",
        "next_step": "Use only to explain mode availability and blockers; do not switch runtime mode automatically.",
    },
    "read_surface_remediation_plan": {
        "entrypoint": "build_read_surface_remediation_plan",
        "ui_surface": "Processing registry, console read coverage remediation",
        "next_step": "Use only to plan read DTO and fixture work; do not query or mutate a live console from this plan.",
    },
    "read_surface_physical_readback_capture_plan": {
        "entrypoint": "build_read_surface_physical_readback_capture_plan",
        "ui_surface": "Processing registry, operator readback capture checklist",
        "next_step": "Use only as a non-executing checklist; do not capture or query hardware without explicit operator action.",
    },
    "raw_mixer_state_physical_readback_contract": {
        "entrypoint": (
            "build_raw_mixer_state_physical_readback_contract / "
            "validate_raw_mixer_state_physical_readback_artifact"
        ),
        "ui_surface": "Processing registry, read-surface remediation artifact contract",
        "next_step": "Use only to validate a separately captured physical readback artifact; do not infer live authority.",
    },
    "channel_basic_controls_physical_readback_contract": {
        "entrypoint": (
            "build_channel_basic_controls_physical_readback_contract / "
            "validate_channel_basic_controls_physical_readback_artifact"
        ),
        "ui_surface": "Processing registry, read-surface remediation artifact contract",
        "next_step": "Use only to validate separately captured channel identity/control readback; do not infer live authority.",
    },
    "channel_processing_physical_readback_contract": {
        "entrypoint": (
            "build_channel_processing_physical_readback_contract / "
            "validate_channel_processing_physical_readback_artifact"
        ),
        "ui_surface": "Processing registry, read-surface remediation artifact contract",
        "next_step": "Use only to validate separately captured HPF/EQ/dynamics/gate readback; do not infer live authority.",
    },
    "bus_group_main_dca_physical_readback_contract": {
        "entrypoint": (
            "build_bus_group_main_dca_physical_readback_contract / "
            "validate_bus_group_main_dca_physical_readback_artifact"
        ),
        "ui_surface": "Processing registry, read-surface remediation artifact contract",
        "next_step": "Use only to validate separately captured bus/group/main/DCA readback; do not infer live authority.",
    },
    "matrix_state_physical_readback_contract": {
        "entrypoint": (
            "build_matrix_state_physical_readback_contract / "
            "validate_matrix_state_physical_readback_artifact"
        ),
        "ui_surface": "Processing registry, read-surface remediation artifact contract",
        "next_step": "Use only to validate separately captured matrix-state readback; do not infer live authority.",
    },
    "fx_slots_inserts_physical_readback_contract": {
        "entrypoint": (
            "build_fx_slots_inserts_physical_readback_contract / "
            "validate_fx_slots_inserts_physical_readback_artifact"
        ),
        "ui_surface": "Processing registry, read-surface remediation artifact contract",
        "next_step": "Use only to validate separately captured FX slots/inserts readback; do not infer live authority.",
    },
    "routing_sends_physical_readback_contract": {
        "entrypoint": (
            "build_routing_sends_physical_readback_contract / "
            "validate_routing_sends_physical_readback_artifact"
        ),
        "ui_surface": "Processing registry, read-surface remediation artifact contract",
        "next_step": "Use only to validate separately captured routing/sends readback; do not infer live authority.",
    },
    "snapshots_scenes_physical_readback_contract": {
        "entrypoint": (
            "build_snapshots_scenes_physical_readback_contract / "
            "validate_snapshots_scenes_physical_readback_artifact"
        ),
        "ui_surface": "Processing registry, read-surface remediation artifact contract",
        "next_step": "Use only to validate separately captured scene/snapshot readback; do not infer live authority.",
    },
    "user_presets_physical_readback_contract": {
        "entrypoint": (
            "build_user_presets_physical_readback_contract / "
            "validate_user_presets_physical_readback_artifact"
        ),
        "ui_surface": "Processing registry, read-surface remediation artifact contract",
        "next_step": "Use only to validate separately captured user-preset readback; do not infer live authority.",
    },
    "dlive_state_physical_readback_contract": {
        "entrypoint": (
            "build_dlive_state_physical_readback_contract / "
            "validate_dlive_state_physical_readback_artifact"
        ),
        "ui_surface": "Processing registry, read-surface remediation artifact contract",
        "next_step": "Use only to validate separately captured dLive state/parity readback; do not infer live authority.",
    },
    "auto_fader_v1": {
        "entrypoint": "main_pipeline_preview proposal adapter",
        "ui_surface": "Task Queue preview/proposal rows",
        "next_step": "Expand proposal evidence without adding direct fader writes.",
    },
    "lufs_gain_staging": {
        "entrypoint": "safe_gain_suggestions / input-trim proposal adapter / tools/offline_input_trim_autogain_mix.py",
        "ui_surface": "Gain staging, Task Queue proposals, offline stem render",
        "next_step": "Keep live trim moves proposal-first with rollback/readback metadata.",
    },
    "live_shared_mix": {
        "entrypoint": "main_pipeline_preview proposal adapter",
        "ui_surface": "Task Queue linked fader proposals",
        "next_step": "Keep grouped balance as linked fader/gain proposals only.",
    },
    "bus_group_balance_proposal": {
        "entrypoint": "bus_group_balance_proposal.analyze_bus_group_balance_need / build_bus_group_balance_proposals",
        "ui_surface": "Group/DCA/matrix balance evidence, future Bus Balance panel",
        "next_step": "Keep bus/group balance proposal-only until fresh assignment/fader readback, true-peak proof, and rollback are proven.",
    },
    "mute_management_proposal": {
        "entrypoint": "mute_management_proposal.analyze_mute_management_need / build_mute_management_proposals",
        "ui_surface": "Channel Intelligence, Safety Monitor, future Mute panel",
        "next_step": "Keep mute-state output proposal-only until fresh intent/readback, rollback, and hardware proof are complete.",
    },
    "routing_management_proposal": {
        "entrypoint": "routing_management_proposal.analyze_routing_management_need / build_routing_management_proposals",
        "ui_surface": "Routing evidence, Safety Monitor, future Routing panel",
        "next_step": "Keep routing output proposal-only until fresh full route readback, rollback, scope limits, and hardware proof are complete.",
    },
    "scene_preset_management_proposal": {
        "entrypoint": "scene_preset_management_proposal.analyze_scene_preset_management_need / build_scene_preset_management_proposals",
        "ui_surface": "Scene/preset evidence, Safety Monitor, future Scene panel",
        "next_step": "Keep scene/preset output proposal-only; use Product Layer snapshot freshness, read-only scoped diff evidence, and read-only rollback evidence until hardware proof is complete.",
    },
    "mix_variant_profile_proposal": {
        "entrypoint": "mix_variant_profile_proposal.analyze_mix_variant_profile_need / build_mix_variant_profile_proposals",
        "ui_surface": "Mix variant evidence, reference comparison, future Variant panel",
        "next_step": "Keep mix variant output proposal-only until fresh references, arrangement context, and follow-up proposal fan-out are proven.",
    },
    "performance_adaptation_proposal": {
        "entrypoint": "performance_adaptation_proposal.analyze_performance_adaptation_need / build_performance_adaptation_proposals",
        "ui_surface": "Live adaptation evidence, Safety Monitor, future Ride panel",
        "next_step": "Keep performance adaptation output proposal-only until fresh trend/readback, replay match, cooldown, rollback, and hardware proof are complete.",
    },
    "arrangement_role_proposal": {
        "entrypoint": "arrangement_role_proposal.analyze_arrangement_role_need / build_arrangement_role_proposals",
        "ui_surface": "Arrangement role evidence, Channel Intelligence, future Role panel",
        "next_step": "Keep arrangement role output proposal-only until fresh context, accepted/rejected operator decisions, and replay correlation are proven.",
    },
    "decision_evidence_packet_proposal": {
        "entrypoint": "decision_evidence_packet_proposal.analyze_decision_evidence_packet_need / build_decision_evidence_packet_proposals",
        "ui_surface": "Decision evidence, Safety Monitor, Task Queue review metadata",
        "next_step": "Attach fresh-context evidence packets to proposal and rejected shadow decisions; keep them non-queueable until replay learning proves value.",
    },
    "long_term_memory_learning_proposal": {
        "entrypoint": "long_term_memory_learning_proposal.analyze_long_term_memory_learning_need / build_long_term_memory_learning_proposals",
        "ui_surface": "Memory learning evidence, replay reports, future Learning Review panel",
        "next_step": "Keep memory learning proposal-only until fresh accepted/rejected outcomes and replay correlation prove useful future decisions.",
    },
    "phase_aware_eq_guard": {
        "entrypoint": "phase_aware_eq_guard.analyze_phase_aware_eq_guard / PhaseAwareEQGuardPlugin.process",
        "ui_surface": "Future AutoEQ guard, Channel Intelligence, phase/EQ proposal traces",
        "next_step": "Use as fresh-context proposal-only phase-before-EQ evidence; do not promote phase/delay/polarity writes without separate hardware proof.",
    },
    "agentic_method_foundation": {
        "entrypoint": "channel_intelligence foundation DTO / BaselineAgenticRecommender",
        "ui_surface": "Channel Intelligence, Safety Monitor evidence, future agent panels",
        "next_step": "Promote only bounded fader/gain outputs through existing proposal adapters; keep EQ/dynamics/gate/pan/FX proposal-only.",
    },
    "context_aware_gate_agent": {
        "entrypoint": "context_aware_gate.extract_bleed_telemetry / ContextAwareGateAgent.recommend",
        "ui_surface": "Channel Intelligence, Decision Log evidence, future Gate/Bleed panel",
        "next_step": "Feed real telemetry/offline bleed simulations; keep all gate/expander outputs dry-run/proposal-only.",
    },
    "compressor_dynamics_proposal": {
        "entrypoint": "compressor_dynamics_proposal.analyze_compressor_need / build_compressor_proposals",
        "ui_surface": "Channel Intelligence, vocal/instrument dynamics proposals, future Compressor panel",
        "next_step": "Keep compressor output proposal-only until fresh dynamics readback, bounded makeup, and rollback are proven.",
    },
    "deesser_vocal_sibilance": {
        "entrypoint": "deesser_vocal_sibilance.analyze_deesser_need / build_deesser_proposals",
        "ui_surface": "Channel Intelligence, vocal processing proposals, future De-esser panel",
        "next_step": "Keep de-esser output proposal-only until fresh dynamics/dynamic-EQ readback and rollback are proven.",
    },
    "pan_spatial_proposal": {
        "entrypoint": "pan_spatial_proposal.analyze_pan_need / build_pan_proposals",
        "ui_surface": "Channel Intelligence, stereo-pair/stage pan proposals, future Panning panel",
        "next_step": "Keep pan output proposal-only until fresh current-pan readback, mono checks, and rollback are proven.",
    },
    "fx_send_proposal": {
        "entrypoint": "fx_send_proposal.analyze_fx_send_need / build_fx_send_proposals",
        "ui_surface": "Channel Intelligence, FX send proposals, future FX panel",
        "next_step": "Keep FX sends proposal-only until fresh send/return readback, feedback checks, and rollback are proven.",
    },
    "master_bus_optimization_proposal": {
        "entrypoint": "master_bus_optimization_proposal.analyze_master_bus_need / build_master_bus_proposals",
        "ui_surface": "Master bus evidence, Safety Monitor, future Master panel",
        "next_step": "Keep master-bus output proposal-only until fresh main-bus readback, limiter/dynamics proof, and rollback are proven.",
    },
    "mix_agent_conservative": {
        "entrypoint": "python -m mix_agent analyze|suggest|apply",
        "ui_surface": "Offline reports; no live cockpit apply",
        "next_step": "Import useful findings as proposals/reports, never direct WING writes.",
    },
    "offline_agent_mix": {
        "entrypoint": "tools/offline_agent_mix.py",
        "ui_surface": "Offline reports; no live cockpit apply",
        "next_step": "Keep as offline/lab renderer until a narrow adapter exists.",
    },
    "automixer_production_mix_v1": {
        "entrypoint": "none in this checkout",
        "ui_surface": "Hidden from live cockpit",
        "next_step": "Restore as an explicit offline entrypoint or retire as legacy.",
    },
    "ai_mixing_pipeline_legacy": {
        "entrypoint": "none in this checkout",
        "ui_surface": "Hidden from live cockpit",
        "next_step": "Keep deprecated until a replacement decision is made.",
    },
}


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


def iter_processing_methods() -> Iterable[ProcessingMethod]:
    return PROCESSING_METHODS


def iter_pipeline_profiles() -> Iterable[PipelineProfile]:
    return PIPELINE_PROFILES


def get_processing_method(method_id: str) -> Optional[ProcessingMethod]:
    return next((method for method in PROCESSING_METHODS if method.id == method_id), None)


def get_pipeline_profile(profile_id: str) -> Optional[PipelineProfile]:
    return next((profile for profile in PIPELINE_PROFILES if profile.id == profile_id), None)


def build_module_status_map() -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for method in PROCESSING_METHODS:
        override = MODULE_STATUS_OVERRIDES.get(method.id, {})
        next_step = (
            override.get("next_step")
            or method.promotion_next
            or (
                "Requires separate ADR, tests, physical readback, rollback, cooldown, and emergency-stop proof."
                if method.status in {"proposal_only", "lab"}
                else "Keep isolated from the main live pipeline."
            )
        )
        rows.append(
            {
                "module_id": method.id,
                "label": method.label,
                "domain": method.domain,
                "status": method.status,
                "entrypoint": override.get("entrypoint", method.module),
                "module": method.module,
                "ui_surface": override.get("ui_surface", "Processing registry / status-only surfaces"),
                "can_write_live": method.can_write_live,
                "live_write_kinds": list(method.live_write_kinds),
                "pipeline_profiles": list(method.pipeline_profiles),
                "blocked_reason": method.blocked_reason,
                "next_step": next_step,
            }
        )
    return rows


def build_processing_registry_snapshot(*, include_methods: bool = True) -> Dict[str, Any]:
    methods = [method.to_dict() for method in PROCESSING_METHODS]
    profiles = [profile.to_dict() for profile in PIPELINE_PROFILES]
    autonomy_promotion_readiness = build_autonomy_promotion_readiness_snapshot()
    autonomy_mode_transition_gate = build_autonomy_mode_transition_gate(
        {"autonomy_promotion_readiness": autonomy_promotion_readiness}
    )
    status_counts: Dict[str, int] = {status: 0 for status in sorted(PROCESSING_STATUSES)}
    domain_counts: Dict[str, int] = {domain: 0 for domain in sorted(PROCESSING_DOMAINS)}
    for method in PROCESSING_METHODS:
        status_counts[method.status] += 1
        domain_counts[method.domain] += 1

    live_executor_ids = [
        method.id
        for method in PROCESSING_METHODS
        if method.can_write_live
    ]
    return {
        "type": "processing_pipeline_snapshot",
        "generated_at": _utc_now(),
        "status": "ok",
        "summary": {
            "method_count": len(PROCESSING_METHODS),
            "pipeline_count": len(PIPELINE_PROFILES),
            "status_counts": status_counts,
            "domain_counts": domain_counts,
            "live_executor_method_ids": live_executor_ids,
            "live_allowed_write_kinds": list(LIVE_ALLOWED_WRITE_KINDS),
            "main_pipeline_id": "main_live_assist",
            "live_apply_policy": "supervised_fader_gain_only",
        },
        "pipeline_profiles": profiles,
        "module_status_map": build_module_status_map(),
        "project_goal_capability_matrix": build_project_goal_capability_matrix_snapshot(),
        "autonomy_promotion_readiness": autonomy_promotion_readiness,
        "autonomy_mode_transition_gate": autonomy_mode_transition_gate,
        "read_surface_inventory": build_read_surface_inventory_snapshot(),
        "read_surface_remediation_plan": build_read_surface_remediation_plan(),
        "read_surface_physical_readback_capture_plan": (
            build_read_surface_physical_readback_capture_plan()
        ),
        "read_surface_fixture_evidence": build_read_surface_fixture_evidence_snapshot(),
        "raw_mixer_state_physical_readback_contract": build_raw_mixer_state_physical_readback_contract(),
        "channel_basic_controls_physical_readback_contract": (
            build_channel_basic_controls_physical_readback_contract()
        ),
        "channel_processing_physical_readback_contract": (
            build_channel_processing_physical_readback_contract()
        ),
        "bus_group_main_dca_physical_readback_contract": (
            build_bus_group_main_dca_physical_readback_contract()
        ),
        "matrix_state_physical_readback_contract": (
            build_matrix_state_physical_readback_contract()
        ),
        "fx_slots_inserts_physical_readback_contract": (
            build_fx_slots_inserts_physical_readback_contract()
        ),
        "routing_sends_physical_readback_contract": (
            build_routing_sends_physical_readback_contract()
        ),
        "snapshots_scenes_physical_readback_contract": (
            build_snapshots_scenes_physical_readback_contract()
        ),
        "user_presets_physical_readback_contract": (
            build_user_presets_physical_readback_contract()
        ),
        "dlive_state_physical_readback_contract": (
            build_dlive_state_physical_readback_contract()
        ),
        "safety_coverage_matrix": build_safety_coverage_matrix_snapshot(),
        "write_surface_inventory": build_write_surface_inventory_snapshot(),
        "methods": methods if include_methods else [],
        "promotion_ladder": list(PROMOTION_LADDER),
    }


def build_pipeline_runtime_summary(operator_mode: Dict[str, Any], manual_write_supervision: Dict[str, Any]) -> Dict[str, Any]:
    main = get_pipeline_profile("main_live_assist")
    return {
        "main_pipeline_id": "main_live_assist",
        "main_pipeline_status": main.status if main else "missing",
        "live_apply_policy": "supervised_fader_gain_only",
        "allowed_live_write_kinds": list(LIVE_ALLOWED_WRITE_KINDS),
        "operator_mode": operator_mode.get("mode"),
        "can_apply_to_console": bool(operator_mode.get("capabilities", {}).get("can_apply_to_console")),
        "manual_write_supervision": manual_write_supervision,
        "secondary_pipeline_ids": ["offline_mix", "shadow_quality", "lab_dsp", "quarantine"],
    }
