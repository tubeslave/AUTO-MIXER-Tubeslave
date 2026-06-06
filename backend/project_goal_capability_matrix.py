"""Read-only project-goal capability matrix for Automixer autonomy.

This module maps the long-term AI mixing engineer goal to the current repo
surfaces. It is deliberately declarative: it does not import mixer clients, open
audio devices, or create proposal/apply side effects.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from typing import Any, Dict, Iterable, Tuple


GOAL_CAPABILITY_STATUSES = {
    "canonical",
    "proposal_only",
    "shadow_only",
    "offline_only",
    "lab",
    "missing",
    "blocked_live",
    "quarantined",
}

AUTONOMY_MODE_CEILINGS = {
    "analysis",
    "assisted",
    "supervised_narrow",
    "offline_only",
    "shadow_only",
    "lab_only",
    "blocked",
    "missing",
}

CURRENT_PROVEN_LIVE_WRITE_KINDS = ("fader", "gain")


@dataclass(frozen=True)
class ProjectGoalCapabilityRow:
    id: str
    label: str
    goal_area: str
    project_requirement: str
    status: str
    autonomy_mode_ceiling: str
    method_ids: Tuple[str, ...] = ()
    support_paths: Tuple[str, ...] = ()
    target_console_surfaces: Tuple[str, ...] = ()
    current_live_write_kinds: Tuple[str, ...] = ()
    blocker: str = ""
    next_step: str = ""

    def __post_init__(self) -> None:
        if self.status not in GOAL_CAPABILITY_STATUSES:
            raise ValueError(f"{self.id} has unknown goal capability status: {self.status}")
        if self.autonomy_mode_ceiling not in AUTONOMY_MODE_CEILINGS:
            raise ValueError(f"{self.id} has unknown autonomy mode ceiling")
        for kind in self.current_live_write_kinds:
            if kind not in CURRENT_PROVEN_LIVE_WRITE_KINDS:
                raise ValueError(f"{self.id} widens current live write kind: {kind}")
        if self.current_live_write_kinds and self.autonomy_mode_ceiling != "supervised_narrow":
            raise ValueError(f"{self.id} declares live writes outside supervised_narrow")

    @property
    def currently_live_writable(self) -> bool:
        return bool(self.current_live_write_kinds)

    def to_dict(self) -> Dict[str, Any]:
        payload = asdict(self)
        payload["method_ids"] = list(self.method_ids)
        payload["support_paths"] = list(self.support_paths)
        payload["target_console_surfaces"] = list(self.target_console_surfaces)
        payload["current_live_write_kinds"] = list(self.current_live_write_kinds)
        payload["currently_live_writable"] = self.currently_live_writable
        return payload


PROJECT_GOAL_CAPABILITIES: Tuple[ProjectGoalCapabilityRow, ...] = (
    ProjectGoalCapabilityRow(
        id="mode_analysis",
        label="Analysis Mode",
        goal_area="runtime_mode",
        project_requirement="Only analysis and recommendations.",
        status="canonical",
        autonomy_mode_ceiling="analysis",
        method_ids=("product_operator_state",),
        support_paths=("backend/operator_mode_policy.py",),
        next_step="Keep analysis read-only and use it as the default fallback for uncertain states.",
    ),
    ProjectGoalCapabilityRow(
        id="mode_assisted",
        label="Assisted Mixing Mode",
        goal_area="runtime_mode",
        project_requirement="Suggestions with engineer confirmation.",
        status="canonical",
        autonomy_mode_ceiling="assisted",
        method_ids=("product_operator_state",),
        support_paths=("backend/operator_proposal_queue.py",),
        next_step="Route every new processing family through proposals before any apply path.",
    ),
    ProjectGoalCapabilityRow(
        id="mode_supervised_automix",
        label="Supervised AutoMix Mode",
        goal_area="runtime_mode",
        project_requirement="Automatic execution of allowed actions under safety control.",
        status="proposal_only",
        autonomy_mode_ceiling="supervised_narrow",
        method_ids=("supervised_manual_fader_gain",),
        support_paths=("backend/operator_mode_policy.py", "backend/autofoh_safety.py"),
        current_live_write_kinds=CURRENT_PROVEN_LIVE_WRITE_KINDS,
        blocker="Only narrow fader/gain surfaces are allowed; broad processing families need proof.",
        next_step="Add fake-mixer and physical proof one action family at a time before widening.",
    ),
    ProjectGoalCapabilityRow(
        id="mode_full_autonomous",
        label="Full Autonomous Mixing Mode",
        goal_area="runtime_mode",
        project_requirement="Fully automatic mix and console control without human participation.",
        status="blocked_live",
        autonomy_mode_ceiling="blocked",
        method_ids=("product_operator_state",),
        support_paths=("backend/operator_mode_policy.py",),
        blocker="Full autonomous WING deployment is locked until safety proof covers every write family.",
        next_step="Use shadow/offline decisions and supervised proofs before enabling autonomous apply.",
    ),
    ProjectGoalCapabilityRow(
        id="analyze_multichannel_audio",
        label="Multichannel audio analysis",
        goal_area="audio_intelligence",
        project_requirement="Automatically analyze multichannel audio signals.",
        status="canonical",
        autonomy_mode_ceiling="analysis",
        method_ids=("auto_soundcheck_engine", "agentic_method_foundation"),
        support_paths=("backend/audio_capture.py", "backend/autofoh_analysis.py"),
        next_step="Keep expanding real session telemetry coverage across channels, stems, and buses.",
    ),
    ProjectGoalCapabilityRow(
        id="identify_source_types",
        label="Source type classification",
        goal_area="audio_intelligence",
        project_requirement="Automatically determine instrument and source types.",
        status="canonical",
        autonomy_mode_ceiling="analysis",
        method_ids=(
            "auto_soundcheck_engine",
            "source_identity_confidence_proposal",
            "source_identity_replay_calibration",
            "source_identity_calibration_acceptance_ledger",
            "source_identity_calibration_persistence_plan",
            "source_identity_calibration_memory_draft",
            "source_identity_calibration_memory_writer",
            "source_identity_calibration_memory_index",
            "source_identity_calibration_memory_trust_review",
            "source_identity_calibration_memory_trust_acceptance_ledger",
            "source_identity_calibration_memory_trust_promotion_plan",
            "source_identity_calibration_memory_trust_promotion_contract",
            "source_identity_calibration_memory_trust_promotion_acceptance_ledger",
            "source_identity_calibration_memory_trust_promotion_execution_plan",
            "source_identity_calibration_memory_trust_promotion_executor_dry_run",
            "source_identity_calibration_memory_trust_promotion_executor",
            "source_identity_calibration_memory_trust_promotion_apply_audit",
            "source_identity_calibration_memory_trust_promotion_rollback_verification",
            "source_identity_calibration_memory_trust_promotion_priority_gate",
        ),
        support_paths=(
            "backend/ml/channel_classifier.py",
            "backend/channel_recognizer.py",
            "backend/source_identity_confidence_proposal.py",
            "backend/source_identity_replay_calibration.py",
            "backend/source_identity_calibration_acceptance_ledger.py",
            "backend/source_identity_calibration_persistence_plan.py",
            "backend/source_identity_calibration_memory_draft.py",
            "backend/source_identity_calibration_memory_writer.py",
            "backend/source_identity_calibration_memory_index.py",
            "backend/source_identity_calibration_memory_trust_review.py",
            "backend/source_identity_calibration_memory_trust_acceptance_ledger.py",
            "backend/source_identity_calibration_memory_trust_promotion_plan.py",
            "backend/source_identity_calibration_memory_trust_promotion_contract.py",
            "backend/source_identity_calibration_memory_trust_promotion_acceptance_ledger.py",
            "backend/source_identity_calibration_memory_trust_promotion_execution_plan.py",
            "backend/source_identity_calibration_memory_trust_promotion_executor_dry_run.py",
            "backend/source_identity_calibration_memory_trust_promotion_executor.py",
            "backend/source_identity_calibration_memory_trust_promotion_apply_audit.py",
            "backend/source_identity_calibration_memory_trust_promotion_rollback_verification.py",
            "backend/source_identity_calibration_memory_trust_promotion_priority_gate.py",
            "backend/source_identity_downstream_gate.py",
        ),
        next_step=(
            "Keep replay-calibrated source identity trust states in operator review, "
            "record acceptance, plan manual persistence, and draft append-only "
            "memory before acknowledged writes, read-only indexing, and operator "
            "memory-trust promotion acceptance plus non-executing execution "
            "planning, executor dry-run specs, operator-gated trust-store apply, "
            "rollback audit, rollback verification, and read-only priority-gate "
            "eligibility before downstream-priority policy consumer use."
        ),
    ),
    ProjectGoalCapabilityRow(
        id="identify_arrangement_roles",
        label="Arrangement role detection",
        goal_area="audio_intelligence",
        project_requirement="Automatically determine roles of instruments in the arrangement.",
        status="proposal_only",
        autonomy_mode_ceiling="assisted",
        method_ids=("arrangement_role_proposal", "agentic_method_foundation"),
        support_paths=(
            "backend/arrangement_role_proposal.py",
            "backend/source_identity_downstream_gate.py",
            "backend/autofoh_profiles.py",
        ),
        blocker="Role confidence is not yet a live apply authority.",
        next_step="Route fresh-context arrangement-role proposals to operator review; replay accepted/rejected role decisions before feeding downstream mix actions.",
    ),
    ProjectGoalCapabilityRow(
        id="gain_staging",
        label="Gain staging",
        goal_area="processing",
        project_requirement="Automatically perform gain staging.",
        status="proposal_only",
        autonomy_mode_ceiling="supervised_narrow",
        method_ids=("lufs_gain_staging", "supervised_manual_fader_gain"),
        support_paths=("tools/offline_input_trim_autogain_mix.py",),
        target_console_surfaces=("preamp", "trim", "fader"),
        current_live_write_kinds=CURRENT_PROVEN_LIVE_WRITE_KINDS,
        blocker="Gain-up and broad trim behavior still need headroom, bleed, and physical proof.",
        next_step="Complete supervised gain-down proof and keep gain-up opt-in.",
    ),
    ProjectGoalCapabilityRow(
        id="equalization",
        label="Equalization",
        goal_area="processing",
        project_requirement="Automatically perform equalization.",
        status="proposal_only",
        autonomy_mode_ceiling="assisted",
        method_ids=("auto_eq", "autoeq_audit_validation", "phase_aware_eq_guard"),
        support_paths=(
            "backend/auto_eq.py",
            "backend/auto_fader_v2/controller.py",
            "backend/auto_soundcheck_engine.py",
            "backend/autoeq_audit.py",
            "backend/phase_aware_eq_guard.py",
        ),
        target_console_surfaces=("eq",),
        blocker="EQ live writes need filter audit, feedback safety, readback, and rollback proof; AutoEQ direct apply/reset, AutoFaderV2 masking EQ, and AutoSoundcheck EQ-enable writes now return blocked evidence.",
        next_step="Route AutoEQ, AutoFaderV2 masking EQ, and AutoSoundcheck EQ-enable blocked evidence to operator review; promote only audited EQ proposals with fresh phase context.",
    ),
    ProjectGoalCapabilityRow(
        id="compression",
        label="Compression",
        goal_area="processing",
        project_requirement="Automatically perform compression.",
        status="proposal_only",
        autonomy_mode_ceiling="assisted",
        method_ids=("compressor_dynamics_proposal", "auto_compressor", "auto_compressor_cf"),
        support_paths=(
            "backend/compressor_dynamics_proposal.py",
            "backend/auto_compressor.py",
            "backend/auto_compressor_cf.py",
            "backend/auto_soundcheck_engine.py",
            "backend/server.py",
            "backend/source_identity_downstream_gate.py",
        ),
        target_console_surfaces=("compressor",),
        blocker="Compressor apply paths are not proven through the supervised gate; AutoCompressor/CF, server reset helper, and AutoSoundcheck threshold writes now return blocked evidence instead of mutating the console.",
        next_step="Route AutoCompressor/CF, server reset helper, and AutoSoundcheck threshold blocked evidence to operator review; require fresh dynamics readback, makeup/headroom and rollback proof before assisted apply.",
    ),
    ProjectGoalCapabilityRow(
        id="gating",
        label="Gate and expander control",
        goal_area="processing",
        project_requirement="Automatically perform gating.",
        status="proposal_only",
        autonomy_mode_ceiling="assisted",
        method_ids=("context_aware_gate_agent", "auto_gate_caig"),
        support_paths=(
            "backend/auto_gate_caig.py",
            "backend/auto_fader_v2/controller.py",
            "backend/context_aware_gate.py",
            "backend/source_identity_downstream_gate.py",
        ),
        target_console_surfaces=("gate",),
        blocker="Gate live writes remain blocked; CAIG direct gate sends and AutoFaderV2 adaptive gate return blocked evidence until false-open/false-close evidence is replayed.",
        next_step=(
            "Evaluate gate decisions against fresh bleed telemetry, trusted source "
            "identity, false-open/false-close replay, and rollback readback before any supervised promotion."
        ),
    ),
    ProjectGoalCapabilityRow(
        id="deessing",
        label="De-essing",
        goal_area="processing",
        project_requirement="Automatically perform de-essing.",
        status="proposal_only",
        autonomy_mode_ceiling="assisted",
        method_ids=("deesser_vocal_sibilance",),
        support_paths=(
            "backend/deesser_vocal_sibilance.py",
            "backend/source_identity_downstream_gate.py",
            "backend/ai/rule_engine.py",
            "backend/ai/agent.py",
        ),
        target_console_surfaces=("compressor", "eq"),
        blocker="De-esser live writes need dynamics or dynamic-EQ readback and rollback proof.",
        next_step="Connect vocal de-esser proposals to operator review without creating live writes.",
    ),
    ProjectGoalCapabilityRow(
        id="panning",
        label="Panning",
        goal_area="processing",
        project_requirement="Automatically perform panning.",
        status="proposal_only",
        autonomy_mode_ceiling="assisted",
        method_ids=("pan_spatial_proposal", "auto_panner"),
        support_paths=(
            "backend/auto_panner.py",
            "backend/auto_panner_adaptive.py",
            "backend/pan_spatial_proposal.py",
            "backend/source_identity_downstream_gate.py",
        ),
        target_console_surfaces=("pan",),
        blocker="Pan live writes need stage/stereo evidence, mono check, readback, and rollback proof; AutoPanner and adaptive panner direct apply are fail-closed.",
        next_step="Route AutoPanner/adaptive evidence and spatial pan proposals to operator review before any supervised pan-write proof.",
    ),
    ProjectGoalCapabilityRow(
        id="frequency_separation",
        label="Frequency separation",
        goal_area="processing",
        project_requirement="Automatically perform frequency separation of instruments.",
        status="proposal_only",
        autonomy_mode_ceiling="assisted",
        method_ids=("frequency_separation_proposal", "cross_adaptive_eq", "agentic_method_foundation"),
        support_paths=(
            "backend/frequency_separation_proposal.py",
            "backend/source_identity_downstream_gate.py",
        ),
        blocker="Frequency split decisions are not yet tied to safe console moves.",
        next_step=(
            "Route fresh-context source/stem frequency-lane proposals to operator review "
            "before deriving any EQ or fader move."
        ),
    ),
    ProjectGoalCapabilityRow(
        id="cross_channel_unmasking",
        label="Cross-channel unmasking",
        goal_area="processing",
        project_requirement="Automatically perform cross-channel unmasking.",
        status="proposal_only",
        autonomy_mode_ceiling="assisted",
        method_ids=("cross_adaptive_eq", "live_shared_mix"),
        support_paths=(
            "backend/auto_fader_v2/controller.py",
            "backend/cross_adaptive_eq.py",
            "backend/source_identity_downstream_gate.py",
        ),
        blocker="Unmasking can recommend changes but cannot write EQ/fader relationships live; AutoFaderV2 masking EQ returns blocked evidence.",
        next_step=(
            "Route fresh-context unmasking output with trusted source identity into "
            "linked cut/boost proposals with EQ-band rollback evidence."
        ),
    ),
    ProjectGoalCapabilityRow(
        id="phase_correction",
        label="Phase correction",
        goal_area="processing",
        project_requirement="Automatically perform phase correction.",
        status="proposal_only",
        autonomy_mode_ceiling="assisted",
        method_ids=("phase_alignment", "phase_aware_eq_guard"),
        target_console_surfaces=("pan",),
        blocker="Delay/polarity writes are not physically proven, can damage mono compatibility, and now return blocked evidence instead of mutating the console.",
        next_step="Route phase alignment blocked evidence to operator review; require fresh phase context plus delay/polarity readback and rollback proof before any live apply.",
    ),
    ProjectGoalCapabilityRow(
        id="effects_management",
        label="Effects management",
        goal_area="processing",
        project_requirement="Automatically manage effects.",
        status="proposal_only",
        autonomy_mode_ceiling="assisted",
        method_ids=("fx_send_proposal", "auto_fx"),
        support_paths=(
            "backend/operator_product_state.py",
            "backend/auto_fx.py",
            "backend/fx_send_proposal.py",
            "backend/source_identity_downstream_gate.py",
        ),
        target_console_surfaces=("fx",),
        blocker="FX rack/insert/reverb writes are broad state changes; AutoFX and AutoReverb direct apply are fail-closed and blocked from live runtime.",
        next_step="Route AutoFX/AutoReverb plans and FX send/return proposals through read-only FX schema hints and operator review before any supervised FX-write proof.",
    ),
    ProjectGoalCapabilityRow(
        id="effect_sends",
        label="Effect send tuning",
        goal_area="processing",
        project_requirement="Automatically tune effect sends.",
        status="proposal_only",
        autonomy_mode_ceiling="assisted",
        method_ids=("fx_send_proposal", "auto_fx"),
        support_paths=(
            "backend/auto_fx.py",
            "backend/fx_send_proposal.py",
            "backend/source_identity_downstream_gate.py",
        ),
        target_console_surfaces=("fx",),
        blocker="Send level automation is fail-closed until feedback, intelligibility, and readback proof exist.",
        next_step="Keep sends proposal-only until feedback risk, return headroom, readback, and rollback proof are complete.",
    ),
    ProjectGoalCapabilityRow(
        id="mix_balance",
        label="Mix balance",
        goal_area="mix_decision",
        project_requirement="Automatically build mix balance.",
        status="proposal_only",
        autonomy_mode_ceiling="supervised_narrow",
        method_ids=("bus_group_balance_proposal", "auto_fader_v1", "live_shared_mix", "lufs_gain_staging"),
        support_paths=(
            "backend/bus_group_balance_proposal.py",
            "backend/source_identity_downstream_gate.py",
        ),
        target_console_surfaces=("fader", "groups", "dca"),
        current_live_write_kinds=("fader",),
        blocker="Only channel fader/gain paths are allowed; bus/DCA/group balance is not proven and AutoFOH bus/DCA/master typed actions are blocked evidence.",
        next_step="Route bus/group/DCA balance proposals to operator review and prove trusted source identity plus fresh assignment/fader readback before any write promotion.",
    ),
    ProjectGoalCapabilityRow(
        id="musician_level_adaptation",
        label="Musician level adaptation",
        goal_area="mix_decision",
        project_requirement="Automatically adapt mix to changing musician levels.",
        status="proposal_only",
        autonomy_mode_ceiling="assisted",
        method_ids=("performance_adaptation_proposal", "agentic_method_foundation", "auto_fader_v1"),
        support_paths=(
            "backend/performance_adaptation_proposal.py",
            "backend/source_identity_downstream_gate.py",
        ),
        blocker="Shadow gain rides are not yet trusted live authority.",
        next_step="Route persistent musician-level drift to operator review with fresh trend/readback, cooldown, rollback, headroom, and feedback evidence before any supervised ride promotion.",
    ),
    ProjectGoalCapabilityRow(
        id="live_adaptation",
        label="Concert-time adaptation",
        goal_area="mix_decision",
        project_requirement="Automatically adapt the mix during the concert.",
        status="proposal_only",
        autonomy_mode_ceiling="assisted",
        method_ids=("performance_adaptation_proposal", "agentic_method_foundation", "muq_perceptual_shadow"),
        support_paths=(
            "backend/performance_adaptation_proposal.py",
            "backend/source_identity_downstream_gate.py",
        ),
        blocker="Live adaptation needs latency, quality, and safety proof per action family.",
        next_step="Keep concert-time adaptation proposal-only until fresh trend/readback, replay match, latency budget, operator review, and per-family hardware proof are complete.",
    ),
    ProjectGoalCapabilityRow(
        id="genre_variants",
        label="Genre and task variants",
        goal_area="mix_decision",
        project_requirement="Automatically form mix variants for genres and tasks.",
        status="proposal_only",
        autonomy_mode_ceiling="assisted",
        method_ids=(
            "mix_variant_profile_proposal",
            "offline_agent_mix",
            "mix_agent_conservative",
            "ml_processing_graph",
        ),
        support_paths=(
            "backend/mix_variant_profile_proposal.py",
            "backend/source_identity_downstream_gate.py",
            "backend/ml/reference_profiles.py",
        ),
        blocker="Genre variants can fan out into many mix and console surfaces.",
        next_step="Route mix-variant profiles to operator review; require trusted source identity, fresh reference metrics, arrangement context, and follow-up proposal fan-out before any live or preset apply.",
    ),
    ProjectGoalCapabilityRow(
        id="master_bus_optimization",
        label="Master bus optimization",
        goal_area="mix_decision",
        project_requirement="Automatically optimize the master bus.",
        status="proposal_only",
        autonomy_mode_ceiling="assisted",
        method_ids=("master_bus_optimization_proposal", "muq_perceptual_shadow"),
        support_paths=(
            "backend/master_bus_optimization_proposal.py",
            "backend/auto_mastering.py",
            "backend/system_measurement.py",
        ),
        target_console_surfaces=("main_bus", "matrices", "eq", "compressor", "limiter"),
        blocker="Master-bus writes can affect the entire PA and are not live-approved.",
        next_step="Route master-bus recommendations to operator review; require main-bus readback, limiter/dynamics proof, and rollback before any write promotion.",
    ),
    ProjectGoalCapabilityRow(
        id="realtime_corrections",
        label="Real-time corrections",
        goal_area="mix_decision",
        project_requirement="Automatically perform corrections during performance.",
        status="proposal_only",
        autonomy_mode_ceiling="assisted",
        method_ids=("performance_adaptation_proposal", "agentic_method_foundation", "supervised_manual_fader_gain"),
        support_paths=(
            "backend/performance_adaptation_proposal.py",
            "backend/source_identity_downstream_gate.py",
        ),
        blocker="Real-time corrections are limited by per-family safety proof and operator mode.",
        next_step="Route reversible, rate-limited correction candidates to operator review with fresh trend/readback; promote only one fader/gain family after replay and hardware proof.",
    ),
    ProjectGoalCapabilityRow(
        id="console_read_parameters",
        label="Read console parameters",
        goal_area="console_control",
        project_requirement="Read all available console parameters.",
        status="proposal_only",
        autonomy_mode_ceiling="analysis",
        method_ids=(
            "product_operator_state",
            "read_surface_remediation_plan",
            "read_surface_physical_readback_capture_plan",
            "read_surface_fixture_evidence",
            "raw_mixer_state_physical_readback_contract",
            "raw_mixer_state_physical_readback_artifact_intake",
            "channel_basic_controls_physical_readback_contract",
            "channel_basic_controls_physical_readback_artifact_intake",
            "channel_processing_physical_readback_contract",
            "channel_processing_physical_readback_artifact_intake",
            "bus_group_main_dca_physical_readback_contract",
            "bus_group_main_dca_physical_readback_artifact_intake",
            "matrix_state_physical_readback_contract",
            "matrix_state_physical_readback_artifact_intake",
            "fx_slots_inserts_physical_readback_contract",
            "fx_slots_inserts_physical_readback_artifact_intake",
            "routing_sends_physical_readback_contract",
            "routing_sends_physical_readback_artifact_intake",
            "snapshots_scenes_physical_readback_contract",
            "snapshots_scenes_physical_readback_artifact_intake",
            "user_presets_physical_readback_contract",
            "user_presets_physical_readback_artifact_intake",
            "dlive_state_physical_readback_contract",
            "dlive_state_physical_readback_artifact_intake",
            "read_surface_physical_readback_artifact_review_queue",
            "read_surface_physical_readback_fixture_acceptance_ledger",
            "read_surface_physical_readback_fixture_closure_plan",
        ),
        support_paths=(
            "backend/read_surface_inventory.py",
            "backend/read_surface_remediation_plan.py",
            "backend/read_surface_physical_readback_capture_plan.py",
            "backend/read_surface_fixture_evidence.py",
            "backend/raw_mixer_state_physical_readback_contract.py",
            "backend/raw_mixer_state_physical_readback_artifact_intake.py",
            "backend/channel_basic_controls_physical_readback_contract.py",
            "backend/channel_basic_controls_physical_readback_artifact_intake.py",
            "backend/channel_processing_physical_readback_contract.py",
            "backend/channel_processing_physical_readback_artifact_intake.py",
            "backend/bus_group_main_dca_physical_readback_contract.py",
            "backend/bus_group_main_dca_physical_readback_artifact_intake.py",
            "backend/matrix_state_physical_readback_contract.py",
            "backend/matrix_state_physical_readback_artifact_intake.py",
            "backend/fx_slots_inserts_physical_readback_contract.py",
            "backend/fx_slots_inserts_physical_readback_artifact_intake.py",
            "backend/routing_sends_physical_readback_contract.py",
            "backend/routing_sends_physical_readback_artifact_intake.py",
            "backend/snapshots_scenes_physical_readback_contract.py",
            "backend/snapshots_scenes_physical_readback_artifact_intake.py",
            "backend/user_presets_physical_readback_contract.py",
            "backend/user_presets_physical_readback_artifact_intake.py",
            "backend/dlive_state_physical_readback_contract.py",
            "backend/dlive_state_physical_readback_artifact_intake.py",
            "backend/read_surface_physical_readback_artifact_review_queue.py",
            "backend/read_surface_physical_readback_fixture_acceptance_ledger.py",
            "backend/read_surface_physical_readback_fixture_closure_plan.py",
            "backend/wing_client.py",
            "backend/dlive_client.py",
        ),
        blocker="Read coverage is inventoried but still partial for matrix, routing/send, scene contents, presets, and dLive parity.",
        next_step=(
            "Close partial read surfaces with physical fixtures, stale-cache evidence, "
            "scene/preset diffs, and dLive/WING parity before claiming full console-state awareness."
        ),
    ),
    ProjectGoalCapabilityRow(
        id="console_analyze_state",
        label="Analyze console state",
        goal_area="console_control",
        project_requirement="Analyze channel, group, matrix, bus, and FX state.",
        status="canonical",
        autonomy_mode_ceiling="analysis",
        method_ids=("product_operator_state", "agentic_method_foundation"),
        support_paths=("backend/operator_product_state.py",),
        next_step="Keep expanding bus/matrix/FX read-only snapshots with schema, freshness, and physical fixture evidence before proposing writes.",
    ),
    ProjectGoalCapabilityRow(
        id="console_preamp_trim_control",
        label="Preamp and trim control",
        goal_area="console_control",
        project_requirement="Control preamps and trim.",
        status="proposal_only",
        autonomy_mode_ceiling="supervised_narrow",
        method_ids=("lufs_gain_staging", "supervised_manual_fader_gain"),
        target_console_surfaces=("preamp", "trim"),
        current_live_write_kinds=("gain",),
        blocker="Physical trim/gain proof is incomplete and gain-up remains constrained.",
        next_step="Finish small supervised gain-down proof before widening trim automation.",
    ),
    ProjectGoalCapabilityRow(
        id="console_fader_control",
        label="Fader control",
        goal_area="console_control",
        project_requirement="Control faders.",
        status="canonical",
        autonomy_mode_ceiling="supervised_narrow",
        method_ids=("supervised_manual_fader_gain",),
        support_paths=("backend/auto_soundcheck_engine.py",),
        target_console_surfaces=("fader",),
        current_live_write_kinds=("fader",),
        blocker="Only supervised bounded channel fader moves are current live scope; AutoSoundcheck fader restore is blocked evidence, not a live executor.",
        next_step="Route AutoSoundcheck restore evidence through Product Layer approval and repeat proof across representative channels before any autonomous fader mode.",
    ),
    ProjectGoalCapabilityRow(
        id="console_mute_control",
        label="Mute control",
        goal_area="console_control",
        project_requirement="Control mute.",
        status="proposal_only",
        autonomy_mode_ceiling="assisted",
        method_ids=("mute_management_proposal",),
        support_paths=(
            "backend/auto_soundcheck_engine.py",
            "backend/mute_management_proposal.py",
            "backend/source_identity_downstream_gate.py",
        ),
        target_console_surfaces=("mute",),
        blocker="Mute automation can create dropouts and is not in the supervised proof path; AutoSoundcheck mute restore returns blocked evidence only.",
        next_step="Route mute/unmute recommendations to operator review; require fresh intent/readback, rollback, and hardware proof before any mute write promotion.",
    ),
    ProjectGoalCapabilityRow(
        id="console_pan_control",
        label="Pan control",
        goal_area="console_control",
        project_requirement="Control panorama.",
        status="proposal_only",
        autonomy_mode_ceiling="assisted",
        method_ids=("pan_spatial_proposal", "auto_panner"),
        support_paths=(
            "backend/auto_panner.py",
            "backend/auto_panner_adaptive.py",
            "backend/pan_spatial_proposal.py",
            "backend/source_identity_downstream_gate.py",
        ),
        target_console_surfaces=("pan",),
        blocker="No supervised pan write proof; AutoPanner, adaptive panner, and controller pan separation return blocked evidence only.",
        next_step="Keep pan settings proposal-only until readback, mono compatibility, and rollback proof are complete.",
    ),
    ProjectGoalCapabilityRow(
        id="console_eq_control",
        label="Console EQ control",
        goal_area="console_control",
        project_requirement="Control equalizers.",
        status="proposal_only",
        autonomy_mode_ceiling="assisted",
        method_ids=("auto_eq", "phase_aware_eq_guard"),
        support_paths=(
            "backend/auto_eq.py",
            "backend/auto_fader_v2/controller.py",
            "backend/auto_soundcheck_engine.py",
            "backend/phase_aware_eq_guard.py",
        ),
        target_console_surfaces=("eq",),
        blocker="Direct EQ apply/reset, AutoFaderV2 masking EQ, and AutoSoundcheck EQ-enable writes remain quarantined and return blocked evidence only.",
        next_step="Require AutoEQ audit, feedback safety, readback and rollback proof before supervised EQ candidates.",
    ),
    ProjectGoalCapabilityRow(
        id="console_compressor_control",
        label="Console compressor control",
        goal_area="console_control",
        project_requirement="Control compressors.",
        status="proposal_only",
        autonomy_mode_ceiling="assisted",
        method_ids=("compressor_dynamics_proposal", "auto_compressor", "auto_compressor_cf"),
        support_paths=(
            "backend/auto_soundcheck_engine.py",
            "backend/server.py",
            "backend/compressor_dynamics_proposal.py",
            "backend/auto_compressor.py",
            "backend/auto_compressor_cf.py",
            "backend/source_identity_downstream_gate.py",
        ),
        target_console_surfaces=("compressor",),
        blocker="Compressor writes are not in current live executor; AutoCompressor/CF apply, server reset helper, and AutoSoundcheck compressor-threshold writes return blocked evidence only.",
        next_step="Keep dynamics settings proposal-only until readback, rollback, and bounded makeup/headroom proof are complete.",
    ),
    ProjectGoalCapabilityRow(
        id="console_gate_control",
        label="Console gate control",
        goal_area="console_control",
        project_requirement="Control gates.",
        status="proposal_only",
        autonomy_mode_ceiling="assisted",
        method_ids=("context_aware_gate_agent", "auto_gate_caig"),
        support_paths=(
            "backend/auto_gate_caig.py",
            "backend/auto_fader_v2/controller.py",
            "backend/context_aware_gate.py",
            "backend/source_identity_downstream_gate.py",
        ),
        target_console_surfaces=("gate",),
        blocker="Gate writes are blocked; CAIG direct gate sends and AutoFaderV2 adaptive gate are quarantined until replay evidence reduces false decisions.",
        next_step="Create supervised gate candidate only after fresh telemetry validation, trusted source identity, and rollback readback proof.",
    ),
    ProjectGoalCapabilityRow(
        id="console_routing_control",
        label="Routing control",
        goal_area="console_control",
        project_requirement="Control routing.",
        status="proposal_only",
        autonomy_mode_ceiling="assisted",
        method_ids=("routing_management_proposal",),
        support_paths=(
            "backend/routing_management_proposal.py",
            "backend/source_identity_downstream_gate.py",
        ),
        target_console_surfaces=("routing",),
        blocker="Routing writes can rewrite broad mixer state.",
        next_step="Route routing diffs to operator review; require trusted source identity, fresh full route readback, rollback, scope limits, and hardware proof before any routing write promotion.",
    ),
    ProjectGoalCapabilityRow(
        id="console_dca_group_matrix_control",
        label="DCA, group, and matrix control",
        goal_area="console_control",
        project_requirement="Control DCA, groups, and matrices.",
        status="proposal_only",
        autonomy_mode_ceiling="assisted",
        method_ids=("bus_group_balance_proposal",),
        support_paths=(
            "backend/bus_group_balance_proposal.py",
            "backend/autofoh_safety.py",
            "backend/source_identity_downstream_gate.py",
        ),
        target_console_surfaces=("dca", "groups", "matrices"),
        blocker="Group/DCA/matrix writes have no supervised proof path; AutoFOH bus/DCA/master typed actions are blocked evidence only.",
        next_step="Keep DCA/group/matrix settings proposal-only until trusted source identity, fresh assignment/fader readback, true-peak safety, rollback, and hardware proof are complete.",
    ),
    ProjectGoalCapabilityRow(
        id="console_fx_control",
        label="Console FX control",
        goal_area="console_control",
        project_requirement="Control FX.",
        status="proposal_only",
        autonomy_mode_ceiling="assisted",
        method_ids=("fx_send_proposal", "auto_fx"),
        support_paths=(
            "backend/operator_product_state.py",
            "backend/auto_fx.py",
            "backend/fx_send_proposal.py",
            "backend/source_identity_downstream_gate.py",
        ),
        target_console_surfaces=("fx",),
        blocker="FX direct apply is blocked from the canonical live runtime; AutoFX and AutoReverb apply_to_mixer return blocked evidence only.",
        next_step="Keep FX settings proposal-only until schema hints become verified model readback, feedback checks, and rollback proof are complete.",
    ),
    ProjectGoalCapabilityRow(
        id="console_scene_preset_control",
        label="Scene and preset control",
        goal_area="console_control",
        project_requirement="Control scenes and user presets.",
        status="proposal_only",
        autonomy_mode_ceiling="assisted",
        method_ids=("scene_preset_management_proposal",),
        support_paths=(
            "backend/operator_product_state.py",
            "backend/scene_preset_management_proposal.py",
        ),
        target_console_surfaces=("scenes", "presets"),
        blocker="Scenes and presets can overwrite broad console state.",
        next_step="Route read-only scoped scene/preset diffs to operator review with Product Layer snapshot freshness and read-only rollback evidence; require scope limits and hardware proof before any scene/preset write promotion.",
    ),
    ProjectGoalCapabilityRow(
        id="decision_basis",
        label="Decision evidence basis",
        goal_area="learning_memory",
        project_requirement="Base decisions on audio, psychoacoustics, statistics, history, references, inter-channel analysis, arrangement, and genre.",
        status="proposal_only",
        autonomy_mode_ceiling="assisted",
        method_ids=(
            "decision_evidence_packet_proposal",
            "agentic_method_foundation",
            "muq_perceptual_shadow",
            "ml_processing_graph",
            "source_identity_proposal_priority_policy",
            "source_identity_proposal_ordering_preview",
            "source_identity_proposal_queue_ordering_plan",
            "source_identity_proposal_queue_importer",
        ),
        support_paths=(
            "backend/decision_evidence_packet_proposal.py",
            "backend/source_identity_downstream_gate.py",
            "backend/source_identity_proposal_priority_policy.py",
            "backend/source_identity_proposal_ordering_preview.py",
            "backend/source_identity_proposal_queue_ordering_plan.py",
            "backend/source_identity_proposal_queue_importer.py",
            "backend/source_knowledge.py",
            "backend/decision_memory.py",
        ),
        blocker="Evidence packets are operator-review contracts, not live apply authority.",
        next_step="Attach fresh-context and trusted-source-identity evidence packets to every proposal and rejected shadow decision, project read-only source-identity-aware ranking records, ordering previews, queue-ordering plans, and queue-import dry runs, then replay accepted/rejected outcomes before promotion.",
    ),
    ProjectGoalCapabilityRow(
        id="long_term_memory_learning",
        label="Long-term memory and learning",
        goal_area="learning_memory",
        project_requirement="Improve decisions from accumulated data, logs, tests, and previous concerts.",
        status="proposal_only",
        autonomy_mode_ceiling="assisted",
        method_ids=(
            "long_term_memory_learning_proposal",
            "source_identity_calibration_memory_draft",
            "source_identity_calibration_memory_writer",
            "source_identity_calibration_memory_index",
            "operator_proposal_apply_sequence_policy_memory_index",
            "operator_proposal_apply_sequence_policy_memory_trust_review",
            "operator_proposal_apply_sequence_policy_memory_trust_acceptance_ledger",
            "operator_proposal_apply_sequence_policy_memory_trust_promotion_plan",
            "operator_proposal_apply_sequence_policy_memory_trust_promotion_contract",
            "operator_proposal_apply_sequence_policy_memory_trust_promotion_acceptance_ledger",
            "operator_proposal_apply_sequence_policy_memory_trust_promotion_execution_plan",
            "operator_proposal_apply_sequence_policy_memory_trust_promotion_executor_dry_run",
            "operator_proposal_apply_sequence_policy_memory_trust_promotion_executor",
            "operator_proposal_apply_sequence_policy_memory_trust_promotion_apply_audit",
            "source_identity_calibration_memory_trust_review",
            "source_identity_calibration_memory_trust_acceptance_ledger",
            "source_identity_calibration_memory_trust_promotion_plan",
            "source_identity_calibration_memory_trust_promotion_contract",
            "source_identity_calibration_memory_trust_promotion_acceptance_ledger",
            "source_identity_calibration_memory_trust_promotion_execution_plan",
            "source_identity_calibration_memory_trust_promotion_executor_dry_run",
            "source_identity_calibration_memory_trust_promotion_executor",
            "source_identity_calibration_memory_trust_promotion_apply_audit",
            "source_identity_calibration_memory_trust_promotion_rollback_verification",
            "source_identity_calibration_memory_trust_promotion_priority_gate",
            "source_identity_proposal_priority_policy",
            "source_identity_proposal_ordering_preview",
            "source_identity_proposal_queue_ordering_plan",
            "source_identity_proposal_queue_importer",
            "product_operator_state",
        ),
        support_paths=(
            "backend/long_term_memory_learning_proposal.py",
            "backend/source_identity_calibration_memory_draft.py",
            "backend/source_identity_calibration_memory_writer.py",
            "backend/source_identity_calibration_memory_index.py",
            "backend/operator_proposal_apply_sequence_policy_memory_index.py",
            "backend/operator_proposal_apply_sequence_policy_memory_trust_review.py",
            "backend/operator_proposal_apply_sequence_policy_memory_trust_acceptance_ledger.py",
            "backend/operator_proposal_apply_sequence_policy_memory_trust_promotion_plan.py",
            "backend/operator_proposal_apply_sequence_policy_memory_trust_promotion_contract.py",
            "backend/operator_proposal_apply_sequence_policy_memory_trust_promotion_acceptance_ledger.py",
            "backend/operator_proposal_apply_sequence_policy_memory_trust_promotion_execution_plan.py",
            "backend/operator_proposal_apply_sequence_policy_memory_trust_promotion_executor_dry_run.py",
            "backend/operator_proposal_apply_sequence_policy_memory_trust_promotion_executor.py",
            "backend/operator_proposal_apply_sequence_policy_memory_trust_promotion_apply_audit.py",
            "backend/source_identity_calibration_memory_trust_review.py",
            "backend/source_identity_calibration_memory_trust_acceptance_ledger.py",
            "backend/source_identity_calibration_memory_trust_promotion_plan.py",
            "backend/source_identity_calibration_memory_trust_promotion_contract.py",
            "backend/source_identity_calibration_memory_trust_promotion_acceptance_ledger.py",
            "backend/source_identity_calibration_memory_trust_promotion_execution_plan.py",
            "backend/source_identity_calibration_memory_trust_promotion_executor_dry_run.py",
            "backend/source_identity_calibration_memory_trust_promotion_executor.py",
            "backend/source_identity_calibration_memory_trust_promotion_apply_audit.py",
            "backend/source_identity_calibration_memory_trust_promotion_rollback_verification.py",
            "backend/source_identity_calibration_memory_trust_promotion_priority_gate.py",
            "backend/source_identity_proposal_priority_policy.py",
            "backend/source_identity_proposal_ordering_preview.py",
            "backend/source_identity_proposal_queue_ordering_plan.py",
            "backend/source_identity_proposal_queue_importer.py",
            "backend/source_identity_downstream_gate.py",
            "backend/decision_memory.py",
            "backend/telemetry/",
            ".ai/memory/project.md",
        ),
        blocker="Memory-learning proposals are review contracts, not autonomous console authority.",
        next_step="Route trusted-source, fresh-context learning records through replay correlation and operator review, then append-only memory drafts, acknowledged writes, read-only indexing, memory-trust review, operator acceptance, manual promotion planning, promotion contract review, promotion acceptance, non-executing execution planning, executor dry-run specs, operator-gated trust-store apply, rollback audit, rollback verification, read-only priority-gate eligibility, read-only proposal-priority policy projection, read-only ordering preview, read-only queue-ordering plan, and operator-gated queue import before any live console authority.",
    ),
)


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


def iter_project_goal_capabilities() -> Iterable[ProjectGoalCapabilityRow]:
    return PROJECT_GOAL_CAPABILITIES


def get_project_goal_capability(row_id: str) -> ProjectGoalCapabilityRow | None:
    return next((row for row in PROJECT_GOAL_CAPABILITIES if row.id == row_id), None)


def build_project_goal_capability_matrix_snapshot() -> Dict[str, Any]:
    rows = [row.to_dict() for row in PROJECT_GOAL_CAPABILITIES]
    status_counts = {status: 0 for status in sorted(GOAL_CAPABILITY_STATUSES)}
    mode_counts = {mode: 0 for mode in sorted(AUTONOMY_MODE_CEILINGS)}
    for row in PROJECT_GOAL_CAPABILITIES:
        status_counts[row.status] += 1
        mode_counts[row.autonomy_mode_ceiling] += 1

    missing_ids = [
        row.id for row in PROJECT_GOAL_CAPABILITIES if row.status == "missing"
    ]
    blocked_live_ids = [
        row.id
        for row in PROJECT_GOAL_CAPABILITIES
        if row.status in {"blocked_live", "quarantined"}
    ]
    live_writable_ids = [
        row.id for row in PROJECT_GOAL_CAPABILITIES if row.currently_live_writable
    ]

    return {
        "type": "project_goal_capability_matrix",
        "generated_at": _utc_now(),
        "status": "ok",
        "safety": {
            "read_only": True,
            "live_write": False,
            "live_widening_allowed": False,
            "current_proven_live_write_kinds": list(CURRENT_PROVEN_LIVE_WRITE_KINDS),
        },
        "summary": {
            "row_count": len(PROJECT_GOAL_CAPABILITIES),
            "status_counts": status_counts,
            "autonomy_mode_ceiling_counts": mode_counts,
            "missing_capability_ids": missing_ids,
            "blocked_live_capability_ids": blocked_live_ids,
            "currently_live_writable_capability_ids": live_writable_ids,
            "next_global_step": "Run promotion readiness gates before widening supervised or autonomous live writes.",
        },
        "rows": rows,
    }
