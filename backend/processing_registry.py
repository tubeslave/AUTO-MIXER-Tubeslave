"""Read-only registry for processing methods and pipeline profiles.

The registry is intentionally declarative. It must not import mixer clients or
DSP modules because Product Layer snapshots need to be safe in offline and live
WING deployments.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, Iterable, List, Optional, Tuple


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
    "eq",
    "dynamics",
    "gate",
    "pan",
    "phase",
    "fx",
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
        id="auto_eq",
        label="Auto EQ",
        domain="eq",
        status="proposal_only",
        summary="Single/multi-channel EQ analysis may propose changes; direct apply remains outside main live executor.",
        module="backend/auto_eq.py",
        can_analyze=True,
        can_propose=True,
        pipeline_profiles=("lab_dsp",),
        promotion_next="proposal_adapter_no_direct_set_eq",
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
        summary="Masking-aware EQ candidate for proposals or offline comparison.",
        module="backend/cross_adaptive_eq.py",
        can_analyze=True,
        can_propose=True,
        pipeline_profiles=("lab_dsp", "offline_mix"),
    ),
    ProcessingMethod(
        id="auto_compressor",
        label="Auto Compressor",
        domain="dynamics",
        status="proposal_only",
        summary="Dynamics analysis may recommend compressor settings; live compressor writes stay blocked.",
        module="backend/auto_compressor.py",
        can_analyze=True,
        can_propose=True,
        pipeline_profiles=("lab_dsp",),
    ),
    ProcessingMethod(
        id="auto_compressor_cf",
        label="Crest-Factor Compressor",
        domain="dynamics",
        status="lab",
        summary="Experimental crest-factor/LUFS compressor variant.",
        module="backend/auto_compressor_cf.py",
        can_analyze=True,
        can_propose=True,
        pipeline_profiles=("lab_dsp",),
    ),
    ProcessingMethod(
        id="auto_gate_caig",
        label="CAIG Auto Gate",
        domain="gate",
        status="proposal_only",
        summary="Gate analysis and recommendation source; not live executor.",
        module="backend/auto_gate_caig.py",
        can_analyze=True,
        can_propose=True,
        pipeline_profiles=("lab_dsp",),
    ),
    ProcessingMethod(
        id="auto_panner",
        label="Auto Panner",
        domain="pan",
        status="lab",
        summary="Panning automation candidate for offline or proposal-only review.",
        module="backend/auto_panner.py + backend/auto_panner_adaptive.py",
        can_analyze=True,
        can_propose=True,
        pipeline_profiles=("lab_dsp", "offline_mix"),
    ),
    ProcessingMethod(
        id="phase_alignment",
        label="Phase / Delay Alignment",
        domain="phase",
        status="proposal_only",
        summary="Phase and delay analysis can produce recommendations; live delay/polarity needs separate proof.",
        module="backend/phase_alignment.py + backend/auto_phase_gcc_phat.py",
        can_analyze=True,
        can_propose=True,
        pipeline_profiles=("lab_dsp", "offline_mix"),
    ),
    ProcessingMethod(
        id="auto_fx",
        label="FX / Reverb / Sends",
        domain="fx",
        status="lab",
        summary="FX and send logic is offline/lab until explicit live send safety exists.",
        module="backend/auto_reverb.py + backend/auto_effects.py",
        can_analyze=True,
        can_propose=True,
        pipeline_profiles=("lab_dsp", "offline_mix"),
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
        summary="FX rack, insert and FX parameter writes are blocked from the main live executor.",
        module="backend/handlers/fx_handlers.py",
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
            "auto_fader_v1",
            "lufs_gain_staging",
            "live_shared_mix",
            "agentic_method_foundation",
            "supervised_manual_fader_gain",
        ),
        proposal_method_ids=(
            "auto_soundcheck_engine",
            "auto_fader_v1",
            "lufs_gain_staging",
            "live_shared_mix",
            "agentic_method_foundation",
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
            "auto_fx",
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
            "autoeq_audit_validation",
            "diff_deq_playbook_bridge",
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
            "cross_adaptive_eq",
            "autoeq_audit_validation",
            "diff_deq_playbook_bridge",
            "auto_compressor",
            "auto_compressor_cf",
            "auto_gate_caig",
            "auto_panner",
            "phase_alignment",
            "auto_fx",
            "agentic_method_foundation",
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
    "supervised_manual_fader_gain": {
        "entrypoint": "apply_operator_proposal -> supervised manual gate",
        "ui_surface": "Task Queue supervised apply",
        "next_step": "Complete separate physical gain proof before widening readiness claims.",
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
    "agentic_method_foundation": {
        "entrypoint": "channel_intelligence foundation DTO / BaselineAgenticRecommender",
        "ui_surface": "Channel Intelligence, Safety Monitor evidence, future agent panels",
        "next_step": "Promote only bounded fader/gain outputs through existing proposal adapters; keep EQ/dynamics/gate/pan/FX proposal-only.",
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
