"""Declarative inventory for Automixer write-capable surfaces.

This module is intentionally data-only. It does not import mixer clients,
handlers, websocket services, OSC transports, or DSP runtime modules. Its job
is to make every known write-capable path visible to Product Layer snapshots,
tests, docs, and future operators before a path can be promoted.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from typing import Any, Dict, Iterable, Tuple
try:
    from operator_mode_policy import CURRENT_PROVEN_LIVE_WRITE_KINDS
except ImportError:  # pragma: no cover - package import compatibility
    from backend.operator_mode_policy import CURRENT_PROVEN_LIVE_WRITE_KINDS


WRITE_SURFACE_STATUSES = {
    "canonical_supervised",
    "proposal_only",
    "lab_only",
    "quarantined",
    "blocked",
}

LIVE_ALLOWED_WRITE_KINDS = tuple(CURRENT_PROVEN_LIVE_WRITE_KINDS)


@dataclass(frozen=True)
class WriteSurface:
    id: str
    label: str
    status: str
    paths: Tuple[str, ...]
    entrypoints: Tuple[str, ...]
    write_kinds: Tuple[str, ...]
    allowed_live: bool
    gate: str
    current_policy: str
    next_action: str
    notes: Tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if self.status not in WRITE_SURFACE_STATUSES:
            raise ValueError(f"Unknown write surface status: {self.status}")
        if not self.paths:
            raise ValueError(f"{self.id} must declare at least one path")
        if not self.entrypoints:
            raise ValueError(f"{self.id} must declare at least one entrypoint")
        if self.allowed_live and self.status != "canonical_supervised":
            raise ValueError(f"{self.id} allows live writes outside canonical supervised status")
        if self.status == "canonical_supervised":
            if not self.allowed_live:
                raise ValueError(f"{self.id} is canonical supervised but allowed_live=False")
            if any(kind not in LIVE_ALLOWED_WRITE_KINDS for kind in self.write_kinds):
                raise ValueError(f"{self.id} declares unsupported live write kinds: {self.write_kinds}")
            if "apply_operator_proposal" not in self.gate:
                raise ValueError(f"{self.id} must route live apply through apply_operator_proposal")
        elif self.write_kinds and self.allowed_live:
            raise ValueError(f"{self.id} has write kinds but should not be live allowed")

    def to_dict(self) -> Dict[str, Any]:
        payload = asdict(self)
        payload["paths"] = list(self.paths)
        payload["entrypoints"] = list(self.entrypoints)
        payload["write_kinds"] = list(self.write_kinds)
        payload["notes"] = list(self.notes)
        return payload


WRITE_SURFACES: Tuple[WriteSurface, ...] = (
    WriteSurface(
        id="product_layer_fader_gain_apply",
        label="Product Layer supervised fader/gain apply",
        status="canonical_supervised",
        paths=(
            "backend/operator_proposal_queue.py",
            "backend/handlers/mixer_handlers.py",
            "backend/server.py",
            "tools/wing_operator_proposal_validation_runner.py",
        ),
        entrypoints=("apply_operator_proposal", "rollback_last_manual_write", "manual_write_supervision"),
        write_kinds=LIVE_ALLOWED_WRITE_KINDS,
        allowed_live=True,
        gate="apply_operator_proposal -> supervised manual write gate -> physical readback -> rollback evidence",
        current_policy="Only operator-approved fader/gain proposals may reach the WING executor.",
        next_action="Keep as the only live executor; do not add EQ/dynamics/routing/FX here without separate proof.",
        notes=("hardware_proven_for_narrow_fader_gain_paths", "requires_idempotency_key"),
    ),
    WriteSurface(
        id="mixer_transport_clients",
        label="Low-level mixer transport clients",
        status="quarantined",
        paths=(
            "backend/mixer_client_base.py",
            "backend/wing_client.py",
            "backend/dlive_client.py",
            "backend/mixing_station_client.py",
            "backend/osc_manager.py",
        ),
        entrypoints=(
            "send",
            "set_fader",
            "set_gain",
            "set_mute",
            "set_eq_band",
            "set_compressor",
            "route_output",
            "load_snap",
            "save_snap",
        ),
        write_kinds=(
            "transport",
            "fader",
            "gain",
            "mute",
            "eq",
            "compressor",
            "routing",
            "snapshot",
            "pan",
        ),
        allowed_live=False,
        gate="Transport only; live authority must come from product_layer_fader_gain_apply.",
        current_policy="Low-level mixer clients are not an automation or UI live-write authority.",
        next_action=(
            "Keep direct imports behind supervised gates and include any new transport "
            "client in this inventory."
        ),
    ),
    WriteSurface(
        id="gain_fader_proposal_sources",
        label="Gain/fader proposal generators",
        status="proposal_only",
        paths=(
            "backend/auto_soundcheck_engine.py",
            "backend/lufs_gain_staging.py",
            "backend/auto_fader.py",
            "backend/live_shared_mix.py",
            "backend/soundcheck_recommendation_workflow.py",
            "backend/processing_proposal_adapters.py",
            "backend/operator_recommendation_bridge.py",
        ),
        entrypoints=("main_pipeline_preview", "run_operator_analysis", "import_soundcheck_recommendations"),
        write_kinds=LIVE_ALLOWED_WRITE_KINDS,
        allowed_live=False,
        gate="Product Layer proposal queue; live apply requires product_layer_fader_gain_apply",
        current_policy="May create bounded fader/gain recommendations only; no direct mixer mutation.",
        next_action="Continue feeding proposal evidence and rollback values into the queue.",
    ),
    WriteSurface(
        id="context_gate_bleed_recommendations",
        label="Context-aware gate/bleed recommendations",
        status="proposal_only",
        paths=("backend/context_aware_gate.py", "backend/agentic_method_foundation.py"),
        entrypoints=("ContextAwareGateAgent.recommend", "recommendation_to_decision_trace"),
        write_kinds=("gate", "expander"),
        allowed_live=False,
        gate="Decision trace only; gate/expander live writes are quarantined.",
        current_policy="Dry-run recommendations with bleed telemetry and review-required safety status.",
        next_action="Connect real telemetry and offline bleed fixtures, but keep gate apply blocked.",
        notes=("no_live_transport_imports", "dry_run_true"),
    ),
    WriteSurface(
        id="eq_recommendation_sources",
        label="EQ recommendation and audit sources",
        status="proposal_only",
        paths=(
            "backend/auto_eq.py",
            "backend/auto_fader_v2/controller.py",
            "backend/cross_adaptive_eq.py",
            "backend/autoeq_audit.py",
            "backend/dynamic_eq_playbook.py",
        ),
        entrypoints=("main_pipeline_preview proposal-only contracts", "autoeq audit validation", "AutoFaderControllerV2._block_eq_adjustments"),
        write_kinds=("eq", "hpf"),
        allowed_live=False,
        gate="Proposal-only contracts; AutoEQ direct apply/reset and AutoFaderV2 masking EQ paths are fail-closed.",
        current_policy="May emit EQ evidence, blockers, blocked apply/reset or masking EQ calls, and filter-package audit results only.",
        next_action="Route blocked EQ evidence to operator review before any supervised EQ promotion.",
    ),
    WriteSurface(
        id="dynamics_recommendation_sources",
        label="Compressor/dynamics recommendation sources",
        status="proposal_only",
        paths=("backend/auto_compressor.py", "backend/auto_compressor_cf.py", "backend/autofoh_safety.py"),
        entrypoints=("compressor analysis", "AutoFOH action planning"),
        write_kinds=("compressor",),
        allowed_live=False,
        gate="Proposal-only contracts; AutoCompressor, CF compressor, and AutoFOH compressor actions are fail-closed outside the Product Layer fader/gain gate.",
        current_policy="Dynamics settings may be recommended or kept as blocked evidence, not applied to WING live.",
        next_action="Route blocked compressor evidence to operator review before any supervised dynamics promotion.",
    ),
    WriteSurface(
        id="offline_render_local_mixers",
        label="Offline render local mixer facades",
        status="lab_only",
        paths=("tools/offline_agent_mix.py", "mix_agent/", "backend/offline_validation_harness.py"),
        entrypoints=("offline render", "offline validation harness", "local in-memory mixer facade"),
        write_kinds=(*(LIVE_ALLOWED_WRITE_KINDS), "eq", "compressor", "gate", "pan", "fx"),
        allowed_live=False,
        gate="Offline files and local fake mixer state only.",
        current_policy="Safe for offline renders/tests; must not import live WING/OSC transport.",
        next_action="Keep outputs as reports/traces; import only validated proposals into Product Layer.",
        notes=("offline_only", "no_wing_transport"),
    ),
    WriteSurface(
        id="latency_benchmark_tools",
        label="Latency benchmark tools",
        status="lab_only",
        paths=("tools/latency/audio_e2e.py", "tools/latency/osc_rtt_benchmark.py", "tools/latency/capture_network.sh"),
        entrypoints=("local OSC RTT benchmark", "audio end-to-end smoke", "tcpdump capture"),
        write_kinds=("osc_test_packet",),
        allowed_live=False,
        gate="Local loopback/default smoke target; not an Automixer/WING write path.",
        current_policy="Benchmark only. Must not be treated as console automation.",
        next_action="Keep separate from Product Layer apply and label artifacts as lab evidence.",
    ),
    WriteSurface(
        id="proof_and_release_check_tools",
        label="Proof and release-check tooling",
        status="lab_only",
        paths=(
            "tools/no_wing_release_check.py",
            "tools/gain_proof_dry_run_runner.py",
            "tools/wing_method_proof_runner.py",
            "tools/wing_supervised_smoke_runner.py",
            "tools/wing_readonly_preflight_runner.py",
            "tools/build_wing_readonly_preflight_handoff_packet.py",
            "tools/review_wing_readonly_preflight_artifacts.py",
            "backend/wing_readonly_preflight_evidence.py",
        ),
        entrypoints=(
            "no-WING release check",
            "fake gain proof",
            "method proof runner",
            "supervised WING smoke runner",
            "read-only WING preflight runner",
            "read-only WING preflight handoff packet builder",
            "read-only WING preflight artifact review",
            "read-only WING preflight evidence intake",
        ),
        write_kinds=("proof_fader", "proof_gain", "proof_eq", "forbidden_token_scan"),
        allowed_live=False,
        gate=(
            "Manual proof tooling only; autoloop/cockpit must not use it as live "
            "executor authority."
        ),
        current_policy=(
            "Dry-run by default; real-WING proof requires separate operator approval "
            "and explicit flags. Read-only preflight may query backend-observed "
            "state/readback only and must not send write, proposal, connect, "
            "disconnect, rollback, or emergency-stop messages. Evidence intake may "
            "contain forbidden request names only as audit tokens and never as "
            "transport calls."
        ),
        next_action=(
            "Keep proof artifacts separate from live readiness claims until physical "
            "readback and rollback are preserved."
        ),
    ),
    WriteSurface(
        id="ableton_control_surface",
        label="Ableton OSC control surface",
        status="lab_only",
        paths=("backend/ableton_client.py",),
        entrypoints=("Ableton track volume/mute/pan/device/send methods",),
        write_kinds=("ableton_volume", "ableton_mute", "ableton_pan", "ableton_device", "ableton_send"),
        allowed_live=False,
        gate="External DAW/lab integration; not WING live apply.",
        current_policy="May support offline/DAW experiments only; no Product Layer live authority.",
        next_action="Add an explicit lab-only wrapper if this becomes part of evaluation playback.",
    ),
    WriteSurface(
        id="observation_mode_write_interceptor",
        label="Observation-mode write interception proxy",
        status="lab_only",
        paths=("backend/observation_mixer.py",),
        entrypoints=(
            "ObservationMixerClient",
            "_record",
            "set_fader",
            "set_gain",
            "set_mute",
            "set_eq_band",
        ),
        write_kinds=(
            "shadow_fader",
            "shadow_gain",
            "shadow_mute",
            "shadow_eq",
            "shadow_compressor",
        ),
        allowed_live=False,
        gate=(
            "Shadow-state recording only; mutating calls are intercepted as operations, "
            "not forwarded to WING."
        ),
        current_policy="Useful for audit/offline observation, but not a live apply path.",
        next_action=(
            "Keep observation outputs as evidence rows and never treat them as readback "
            "proof."
        ),
    ),
    WriteSurface(
        id="routing_snapshot_scene_writes",
        label="Routing, snapshot, and scene writes",
        status="quarantined",
        paths=(
            "backend/handlers/routing_handlers.py",
            "backend/handlers/snapshot_handlers.py",
            "backend/load_snap.py",
            "backend/load_snap_v2.py",
            "backend/load_snap_final.py",
            "backend/find_and_load_snap.py",
            "backend/scan_and_load_snap.py",
            "backend/route_custom_outputs.py",
            "backend/auto_test_routing.py",
        ),
        entrypoints=("route_output", "load_snap", "save_snap", "load_snap_by_index", "auto_test_routing"),
        write_kinds=("routing", "snapshot", "scene"),
        allowed_live=False,
        gate="Deployment boundary/quarantine; not part of main_live_assist executor.",
        current_policy="Blocked from canonical pipeline because it can rewrite broad mixer state.",
        next_action="Keep manual scripts lab-only or migrate to explicit operator runbooks with preview diffs.",
    ),
    WriteSurface(
        id="eq_direct_apply_and_reset",
        label="Direct EQ apply and reset",
        status="quarantined",
        paths=(
            "backend/handlers/eq_handlers.py",
            "backend/server.py",
            "backend/auto_eq.py",
            "backend/auto_fader_v2/controller.py",
            "backend/autofoh_safety.py",
            "backend/auto_soundcheck_engine.py",
            "backend/system_measurement.py",
        ),
        entrypoints=(
            "set_eq",
            "reset_eq",
            "auto_eq_controller.reset_eq",
            "AutoEQController.apply_to_mixer",
            "MultiChannelAutoEQController.apply_channel_correction",
            "AutoFaderControllerV2._block_eq_adjustments",
            "AutoFOHSafetyController._blocked_direct_write_payload",
            "AutoSoundcheckEngine._record_blocked_console_write",
            "SystemMeasurement.apply_corrections",
            "SystemMeasurement.reset_corrections",
        ),
        write_kinds=("eq", "hpf"),
        allowed_live=False,
        gate="Quarantined; EQ can only appear as proposal/audit/blocked evidence and AutoEQ, AutoFaderV2 masking EQ, AutoFOH EQ actions, and AutoSoundcheck legacy EQ-enable writes are fail-closed.",
        current_policy="No live EQ apply; AutoEQ apply/reset, AutoFaderV2 spectral/cross-adaptive masking EQ, AutoFOH EQ/emergency-notch actions, and AutoSoundcheck EQ-enable writes return blocked evidence until filter audit, readback, rollback, and feedback proof exist.",
        next_action="Use EQ proposal DTOs and require filter audit, feedback safety, readback and rollback proof before any live apply.",
    ),
    WriteSurface(
        id="compressor_direct_apply",
        label="Direct compressor apply",
        status="quarantined",
        paths=("backend/handlers/mixer_handlers.py", "backend/server.py", "backend/auto_compressor.py", "backend/auto_compressor_cf.py", "backend/autofoh_safety.py", "backend/auto_soundcheck_engine.py"),
        entrypoints=("set_compressor", "mixer_client.set_compressor", "_apply_compressor_smooth", "AutoCFCompressorController.apply_params", "AutoMixerServer._reset_compressor_on_channel", "AutoFOHSafetyController._blocked_direct_write_payload", "AutoSoundcheckEngine._record_blocked_console_write"),
        write_kinds=("compressor",),
        allowed_live=False,
        gate="Quarantined; compressor can only appear as proposal or blocked evidence and direct apply paths are fail-closed.",
        current_policy="No live compressor writes in MVP; AutoCompressor/CF apply, server compressor reset helper, AutoFOH compressor typed actions, and AutoSoundcheck legacy compressor-threshold writes return blocked evidence only.",
        next_action="Use compressor proposal adapter and require dynamics readback, makeup/headroom and rollback proof before any live apply.",
    ),
    WriteSurface(
        id="bus_dca_master_fader_direct_apply",
        label="Bus, DCA, and master fader direct apply",
        status="quarantined",
        paths=(
            "backend/autofoh_safety.py",
            "backend/auto_soundcheck_engine.py",
            "backend/bus_group_balance_proposal.py",
            "backend/master_bus_optimization_proposal.py",
        ),
        entrypoints=("BusFaderMove", "DCAFaderMove", "MasterFaderMove", "AutoFOHSafetyController._blocked_direct_write_payload"),
        write_kinds=("bus_fader", "dca_fader", "master_fader"),
        allowed_live=False,
        gate="Quarantined; AutoFOH bus/DCA/master fader actions return blocked evidence and only channel fader/gain may reach Product Layer supervised apply.",
        current_policy="No bus, DCA, group, matrix, or master fader live writes from automation modules.",
        next_action="Promote only after fresh assignment/fader readback, true-peak safety, rollback, and hardware proof.",
    ),
    WriteSurface(
        id="gate_hpf_send_direct_apply",
        label="Gate, HPF, and send direct apply",
        status="quarantined",
        paths=("backend/auto_gate_caig.py", "backend/auto_soundcheck_engine.py", "backend/auto_fader_v2/controller.py", "backend/autofoh_safety.py"),
        entrypoints=("set_gate", "set_gate_on", "AutoFaderControllerV2._apply_adaptive_noise_gate", "set_hpf", "set_send_level"),
        write_kinds=("gate", "hpf", "send"),
        allowed_live=False,
        gate="Quarantined; context-aware gate remains dry-run only, CAIG _send_osc is fail-closed, and AutoFaderV2/AutoFOH adaptive or typed gate actions return blocked evidence.",
        current_policy="Do not apply gate/HPF/send changes live from automation modules; CAIG direct gate sends, AutoFaderV2 adaptive gate/fader-mute fallback, and AutoFOH gate/HPF/send actions are blocked even when a mixer client is present.",
        next_action="Keep direct gate sends quarantined; route all gate/HPF/send outputs to proposal-only contracts with physical proof requirements.",
    ),
    WriteSurface(
        id="pan_phase_delay_direct_apply",
        label="Pan, phase, polarity, and delay direct apply",
        status="quarantined",
        paths=(
            "backend/auto_panner.py",
            "backend/auto_panner_adaptive.py",
            "backend/auto_fader_v2/controller.py",
            "backend/phase_alignment.py",
            "backend/autofoh_safety.py",
        ),
        entrypoints=("_send_osc", "set_channel_pan", "set_polarity", "set_delay", "set_pan"),
        write_kinds=("pan", "phase", "delay", "polarity"),
        allowed_live=False,
        gate="Quarantined; AutoPanner, adaptive panner, AutoFOH pan actions, and pan separation are fail-closed; phase alignment and AutoFOH phase actions are fail-closed until stage position/mono/readback proof exists.",
        current_policy="No live pan/phase/delay automation; direct pan and phase/delay apply/reset paths return blocked evidence only.",
        next_action="Keep stage-position and pan/phase outputs as evidence until separate hardware readback and rollback proof.",
    ),
    WriteSurface(
        id="fx_insert_send_direct_apply",
        label="FX, insert, and send direct apply",
        status="quarantined",
        paths=(
            "backend/auto_fx.py",
            "backend/auto_effects.py",
            "backend/auto_reverb.py",
            "backend/handlers/fx_handlers.py",
            "backend/autofoh_safety.py",
            "backend/example_fx_correction.py",
            "backend/restore_channels.py",
            "backend/reset_all_channels.py",
            "backend/reset_modules_trim_faders.py",
        ),
        entrypoints=(
            "set_fx_model",
            "set_fx_on",
            "set_fx_mix",
            "set_fx_parameter",
            "set_send_level",
            "set_reverb_send",
            "set_reverb_params",
        ),
        write_kinds=("fx", "insert", "send"),
        allowed_live=False,
        gate="Quarantined; FX/color agent stays recommendation-only and AutoFX/AutoReverb/AutoFOH send apply paths are fail-closed.",
        current_policy="No live FX/send/reverb writes from automation or reset scripts; AutoFX, AutoReverb, and AutoFOH send actions return blocked evidence only.",
        next_action="Keep direct FX sends quarantined; route useful FX behavior into explainable FX plans with physical proof requirements.",
    ),
    WriteSurface(
        id="voice_ai_direct_commands",
        label="Voice and AI direct command surfaces",
        status="quarantined",
        paths=(
            "backend/voice_control.py",
            "backend/voice_control_v2.py",
            "backend/voice_control_vosk.py",
            "backend/voice_control_sherpa.py",
            "backend/ai/agent.py",
            "backend/ai/rule_engine.py",
        ),
        entrypoints=(
            "set_fader",
            "set_gain",
            "load_snap",
            "mute_channel",
            "unmute_channel",
            "_safe_set_fader",
        ),
        write_kinds=("fader", "gain", "snapshot", "mute", "compressor"),
        allowed_live=False,
        gate="Voice/AI may request proposals only; no bypass of Product Layer or manual gate.",
        current_policy="Direct voice/AI commands are not canonical live control.",
        next_action="Convert commands into operator proposal intents and blocked reasons.",
    ),
    WriteSurface(
        id="manual_reset_restore_scripts",
        label="Manual reset, restore, and bulk mutation scripts",
        status="blocked",
        paths=("backend/reset_all_channels.py", "backend/reset_modules_trim_faders.py", "backend/restore_channels.py", "backend/disable_modules_set_faders.py", "backend/auto_soundcheck_engine.py"),
        entrypoints=("reset all channels", "restore channels", "disable modules and set faders", "AutoSoundcheckEngine._record_blocked_console_write"),
        write_kinds=("fader", "mute", "eq", "compressor", "gate", "fx"),
        allowed_live=False,
        gate="Blocked from automated runs; destructive operator-only recovery scripts.",
        current_policy=(
            "Never run from Codex/autoloop/automation. CLI auto-confirm flags are "
            "not supported; destructive use requires an explicit human live procedure. "
            "AutoSoundcheck fader/mute restore emits blocked evidence only and "
            "must be applied through Product Layer approval."
        ),
        next_action=(
            "Move to lab_only/manual recovery namespace and keep backend restore "
            "entrypoints blocked by the quarantined write-surface gate."
        ),
    ),
    WriteSurface(
        id="legacy_frontend_direct_ws_methods",
        label="Legacy frontend direct websocket methods",
        status="blocked",
        paths=("frontend/src/services/websocket.js", "frontend/src/App.js"),
        entrypoints=("set_fader", "set_gain", "route_output", "load_snap", "save_snap", "reset_eq"),
        write_kinds=("fader", "gain", "routing", "snapshot", "eq"),
        allowed_live=False,
        gate="Modern cockpit must use backend Product Layer actions only.",
        current_policy="No new frontend view may call direct websocket write helpers.",
        next_action="Delete or wrap legacy helpers after compatibility audit; keep tests forbidding cockpit use.",
    ),
    WriteSurface(
        id="legacy_ws_to_osc_router",
        label="Legacy websocket to OSC router",
        status="blocked",
        paths=("backend/routing.py",),
        entrypoints=("command_to_osc", "set_fader", "set_mute", "set_pan"),
        write_kinds=("fader", "mute", "pan"),
        allowed_live=False,
        gate="Blocked from canonical runtime; Product Layer/manual gate supersedes it.",
        current_policy="Do not route frontend commands directly to OSC.",
        next_action="Keep only as historical reference or remove after dependency scan.",
    ),
    WriteSurface(
        id="ml_generated_osc_commands",
        label="ML generated OSC command output",
        status="blocked",
        paths=("backend/ml/style_transfer.py", "backend/ml/"),
        entrypoints=("generate OSC commands",),
        write_kinds=("osc_generated_command",),
        allowed_live=False,
        gate="Research output only; no live transport authority.",
        current_policy="Generated commands must become proposal/trace evidence, not OSC messages.",
        next_action="Add adapter tests before any ML output reaches Product Layer.",
    ),
)


def iter_write_surfaces() -> Iterable[WriteSurface]:
    return WRITE_SURFACES


def get_write_surface(surface_id: str) -> WriteSurface | None:
    return next((surface for surface in WRITE_SURFACES if surface.id == surface_id), None)


def build_write_surface_inventory_snapshot() -> Dict[str, Any]:
    status_counts: Dict[str, int] = {status: 0 for status in sorted(WRITE_SURFACE_STATUSES)}
    allowed_live_ids = []
    for surface in WRITE_SURFACES:
        status_counts[surface.status] += 1
        if surface.allowed_live:
            allowed_live_ids.append(surface.id)
    return {
        "type": "write_surface_inventory",
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z"),
        "status": "ok",
        "summary": {
            "surface_count": len(WRITE_SURFACES),
            "status_counts": status_counts,
            "allowed_live_surface_ids": allowed_live_ids,
            "live_apply_policy": "canonical_supervised_fader_gain_only",
            "allowed_live_write_kinds": list(LIVE_ALLOWED_WRITE_KINDS),
        },
        "surfaces": [surface.to_dict() for surface in WRITE_SURFACES],
    }
