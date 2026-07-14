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


WRITE_SURFACE_STATUSES = {
    "canonical_supervised",
    "proposal_only",
    "lab_only",
    "quarantined",
    "blocked",
}

LIVE_ALLOWED_WRITE_KINDS = ("fader", "gain")


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
        id="dca_groups_setup_runner",
        label="DCA groups SETUP runner (operator-invoked)",
        status="lab_only",
        paths=("tools/dca_groups_setup.py",),
        entrypoints=("dca_groups_setup CLI",),
        write_kinds=("dca_name", "dca_fader", "dca_mute", "channel_tags"),
        allowed_live=False,
        gate="explicit --confirm-live-wing-write + rollback file before first "
             "write + per-address readback; DCA assignment via #DN tag",
        current_policy="SETUP-фаза по явному запуску оператора (2026-07-06): "
                       "группы DRUMS/BASS/GTRS/PLAYBK/KEYS/VOX.",
        next_action="Синхронизировать карту групп с config autofoh.dca_balance.",
        notes=("wing_tags_dca_assignment", "rollback_file_first"),
    ),
    WriteSurface(
        id="stem_bus_routing_setup_runner",
        label="Stem bus routing + processing reset SETUP runner "
              "(operator-invoked)",
        status="lab_only",
        paths=("tools/stem_bus_routing_setup.py",),
        entrypoints=("stem_bus_routing_setup CLI",),
        write_kinds=("channel_eq", "channel_gate", "channel_dyn",
                     "channel_filter", "channel_delay", "channel_polarity",
                     "channel_trim", "channel_send", "channel_main_assign",
                     "bus_name", "bus_fader", "bus_mute", "bus_main_assign",
                     "main_eq", "main_dyn", "usr_signal", "usb_output_routing"),
        allowed_live=False,
        gate="rollback file before first write + per-address readback + "
             "audio probe of new taps; buses built muted, channels leave "
             "main before buses unmute (no level doubling)",
        current_policy="SETUP-фаза по явной директиве оператора (2026-07-06): "
                       "сброс обработки ch1-16 + main, стем-шины 1-6 (GRP "
                       "sends), USB 33-44 = стемы post-fader, 45/46 = Main LR "
                       "pre-fader через USR 17/18.",
        next_action="Подключить стем-тапы 33-44 и Main-pre 45/46 к "
                    "измерительным контурам движка.",
        notes=("wing_send_mode_grp", "rollback_file_first",
               "usr_signal_pool_taps"),
    ),
    WriteSurface(
        id="mixing_rule_trial_runner",
        label="Mixing-rule learning trial runner (operator-invoked)",
        status="lab_only",
        paths=("tools/mixing_rule_trial.py",),
        entrypoints=("mixing_rule_trial CLI",),
        write_kinds=("channel_eq", "channel_filter"),
        allowed_live=False,
        gate="rollback file before first write + per-address readback + "
             "master-loop A/B (before/after) + operator verdict updates "
             "rule score; occupied EQ bands are never re-purposed",
        current_policy="Пайплайн обучения (директива 2026-07-06): проба "
                       "правил-кандидатов из source_knowledge на реальном "
                       "пульте, счёт helped/hurt, статус candidate -> "
                       "active/rejected.",
        next_action="После промоута правил — перенос tonal targets в "
                    "EQ-стадию движка.",
        notes=("source_knowledge_rules", "rollback_file_first",
               "operator_verdict_scoring"),
    ),
    WriteSurface(
        id="master_pink_balance_runner",
        label="Master pink-noise balance SETUP runner (operator-invoked)",
        status="lab_only",
        paths=("tools/master_pink_balance.py",),
        entrypoints=("master_pink_balance CLI",),
        write_kinds=("main_eq",),
        allowed_live=False,
        gate="explicit --confirm-live-wing-write + rollback file before first "
             "write + per-address readback; closed loop via master tap",
        current_policy="SETUP-фаза по явному запуску оператора (2026-07-06): "
                       "спектр мастера к розовой кривой, prefer-cut ≤6 дБ.",
        next_action="Повторять по мере смены материала; freeze при NO_SIGNAL.",
        notes=("solver_pa_v3", "rollback_file_first"),
    ),
    WriteSurface(
        id="fx_rack_setup_runner",
        label="FX rack SETUP runner (operator-invoked)",
        status="lab_only",
        paths=("tools/fx_rack_setup.py",),
        entrypoints=("fx_rack_setup CLI",),
        write_kinds=("fx_model", "fx_params", "bus_insert", "bus_fader", "bus_name"),
        allowed_live=False,
        gate="explicit --confirm-live-wing-write + rollback file written before "
             "first write + per-address readback; premium models only slots 1-8",
        current_policy="SETUP-фаза по явному запуску оператора (2026-07-06); "
                       "не песенная автоматика. Слот с несовпавшим readback "
                       "откатывается в NONE.",
        next_action="Дополнить операторским прослушиванием и калибровкой "
                    "посылов -10±3 дБ по консольным метрам (fx_v3).",
        notes=("plan_source_auto_fx_planner", "rollback_file_first"),
    ),
    WriteSurface(
        id="product_layer_fader_gain_apply",
        label="Product Layer supervised fader/gain apply",
        status="canonical_supervised",
        paths=(
            "backend/live_apply.py",
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
            "backend/osc/enhanced_osc_client.py",
            "backend/wing_supervised_console_adapters.py",
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
        id="m110_attended_one_shot_maintenance",
        label="M110 attended one-shot DSP maintenance executor",
        status="lab_only",
        paths=("backend/maintenance_dsp_reset.py",),
        entrypoints=("M110MaintenanceDspResetExecutor.execute",),
        write_kinds=("processor_bypass",),
        allowed_live=False,
        gate=(
            "Fixed server-owned 64-address bundle; CP05 hardware gate, exact "
            "one-shot capability, WAL-before-send, physical readback and reverse "
            "rollback. Never part of the normal live allowlist."
        ),
        current_policy=(
            "Available only inside an attended proof-owned M110 process after "
            "software and hardware provenance gates pass."
        ),
        next_action=(
            "Keep the address set server-constant and require a byte-identical "
            "restored proof before permanent reset commit."
        ),
        notes=("m110_only", "one_shot_capability", "not_normal_live_authority"),
    ),
    WriteSurface(
        id="gain_fader_proposal_sources",
        label="Gain/fader proposal generators",
        status="proposal_only",
        paths=(
            "backend/heuristics/gain_v3.py",
            "backend/heuristics/balance_v3.py",
            "backend/heuristics/bleed_leakage.py",
            "backend/auto_soundcheck_engine.py",
            "backend/agents/fader_agent.py",
            "backend/agents/gain_agent.py",
            "backend/lufs_gain_staging.py",
            "backend/auto_fader.py",
            "backend/live_shared_mix.py",
            "backend/soundcheck_recommendation_workflow.py",
            "backend/processing_proposal_adapters.py",
            "backend/operator_recommendation_bridge.py",
        ),
        entrypoints=(
            "main_pipeline_preview",
            "run_operator_analysis",
            "import_soundcheck_recommendations",
            "FaderAgent.act blocked fader evidence",
            "GainAgent.act blocked trim evidence",
        ),
        write_kinds=LIVE_ALLOWED_WRITE_KINDS,
        allowed_live=False,
        gate="Product Layer proposal queue; live apply requires product_layer_fader_gain_apply",
        current_policy="May create bounded fader/gain recommendations or blocked trim evidence only; no direct mixer mutation.",
        next_action="Continue feeding proposal evidence and rollback values into the queue.",
    ),
    WriteSurface(
        id="realtime_arrangement_gain_fader_gate_clients",
        label="Realtime arrangement fader/trim live-apply clients",
        status="quarantined",
        paths=(
            "backend/arrangement_automation/controller.py",
            "backend/arrangement_automation/live_input_trim_controller.py",
        ),
        entrypoints=(
            "ArrangementAutomationController._apply_safe_offsets",
            "LiveInputTrimController._apply_trim_decision",
        ),
        write_kinds=LIVE_ALLOWED_WRITE_KINDS,
        allowed_live=False,
        gate=(
            "Disabled/dry-run by default and routed through LiveApplyService; not a "
            "standalone WING live authority or release gate."
        ),
        current_policy=(
            "Realtime arrangement fader and input-trim controllers may produce "
            "proposal/replay evidence, but must not bypass Product Layer approval, "
            "operator confirmation, readback, rollback, or emergency-stop proof."
        ),
        next_action=(
            "Keep these clients inventoried as write-adjacent and require explicit "
            "Product Layer handoff before any release promotion."
        ),
        notes=("not_allowed_live_surface", "live_apply_service_client"),
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
            "backend/heuristics/eq_v3.py",
            "backend/heuristics/resonance_eq.py",
            "backend/heuristics/erb_masking.py",
            "backend/heuristics/eq_ab_check.py",
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
        paths=(
            "backend/heuristics/comp_v3.py",
            "backend/heuristics/drc_giannoulis.py",
            "backend/auto_compressor.py",
            "backend/auto_compressor_cf.py",
            "backend/autofoh_safety.py",
        ),
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
            "tools/soundcheck_runtime_observation_runner.py",
            "tools/soundcheck_long_run_drift_audit.py",
            "tools/build_wing_readonly_preflight_handoff_packet.py",
            "tools/review_wing_readonly_preflight_artifacts.py",
            "tools/migrate_m005_current_release_provenance.py",
            "backend/tub345_supervised_write_runner.py",
            "backend/wing_readonly_preflight_contract.py",
            "backend/wing_readonly_preflight_evidence.py",
        ),
        entrypoints=(
            "no-WING release check",
            "fake gain proof",
            "method proof runner",
            "supervised WING smoke runner",
            "read-only WING preflight runner",
            "read-only soundcheck runtime observation runner",
            "offline soundcheck long-run drift audit",
            "read-only WING preflight handoff packet builder",
            "read-only WING preflight artifact review",
            "M005 current-release provenance migration and isolated replay",
            "TUB supervised WING write runner",
            "read-only WING preflight safety contract",
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
            "transport calls. Runtime observation may subscribe to meters and "
            "request dashboard/channel/decision/safety snapshots only. Long-run "
            "drift audit reads existing logs only and cannot create or apply "
            "corrections. M005 provenance migration writes private evidence "
            "artifacts and runs a sealed no-network/no-audio replay; correction "
            "names inside that replay never grant console transport authority."
        ),
        next_action=(
            "Keep proof artifacts separate from live readiness claims until physical "
            "readback and rollback are preserved."
        ),
    ),
    WriteSurface(
        id="broad_live_wing_soundcheck_v2_tool",
        label="Broad live WING soundcheck v2 tool",
        status="blocked",
        paths=("tools/live_wing_soundcheck_v2.py",),
        entrypoints=("live_wing_soundcheck_v2 main",),
        write_kinds=("fader", "gain", "hpf", "eq", "compressor", "gate"),
        allowed_live=False,
        gate=(
            "Blocked from automation and release readiness. Non-dry-run execution "
            "requires the explicit --confirm-live-wing-write acknowledgement plus "
            "operator supervision, physical readback, and rollback review."
        ),
        current_policy=(
            "This script can write broad WING soundcheck v2 settings across active "
            "channels and is not part of the supervised MVP fader/gain apply path."
        ),
        next_action=(
            "Prefer the narrow supervised fader/gain proof runbook. Keep this v2 "
            "setup tool blocked unless a future operator runbook gives it separate "
            "readback, rollback, and emergency-stop proof requirements."
        ),
        notes=("requires_explicit_live_write_ack", "not_mvp_release_authority"),
    ),
    WriteSurface(
        id="broad_live_wing_soundcheck_v2_stage2_tool",
        label="Broad live WING soundcheck v2 stage2 tool",
        status="blocked",
        paths=("tools/live_wing_soundcheck_v2_stage2.py",),
        entrypoints=("live_wing_soundcheck_v2_stage2 main",),
        write_kinds=("gain", "hpf", "eq", "compressor", "fader", "pan", "phase", "delay"),
        allowed_live=False,
        gate=(
            "Blocked from automation and release readiness. Non-dry-run execution "
            "requires the explicit --confirm-live-wing-write acknowledgement plus "
            "operator supervision, physical readback, and rollback review."
        ),
        current_policy=(
            "This script can write broad WING stage2 soundcheck settings including "
            "phase, delay, pan, EQ, dynamics, trim and fader; it is not part of the "
            "supervised MVP fader/gain apply path."
        ),
        next_action=(
            "Prefer the narrow supervised fader/gain proof runbook. Keep this "
            "stage2 setup tool blocked unless a future operator runbook gives it "
            "separate readback, rollback, and emergency-stop proof requirements."
        ),
        notes=("requires_explicit_live_write_ack", "not_mvp_release_authority"),
    ),
    WriteSurface(
        id="broad_live_wing_soundcheck_v2_stage3_tool",
        label="Broad live WING soundcheck v2 stage3 tool",
        status="blocked",
        paths=("tools/live_wing_soundcheck_v2_stage3.py",),
        entrypoints=("live_wing_soundcheck_v2_stage3 main",),
        write_kinds=("phase", "delay", "fx", "bus_fader", "send"),
        allowed_live=False,
        gate=(
            "Blocked from automation and release readiness. Non-dry-run execution "
            "requires the explicit --confirm-live-wing-write acknowledgement plus "
            "operator supervision, physical readback, and rollback review."
        ),
        current_policy=(
            "This script can write broad WING stage3 soundcheck settings including "
            "phase, delay, FX models/parameters, bus faders and channel sends; it is "
            "not part of the supervised MVP fader/gain apply path."
        ),
        next_action=(
            "Prefer the narrow supervised fader/gain proof runbook. Keep this "
            "stage3 setup tool blocked unless a future operator runbook gives it "
            "separate readback, rollback, and emergency-stop proof requirements."
        ),
        notes=("requires_explicit_live_write_ack", "not_mvp_release_authority"),
    ),
    WriteSurface(
        id="broad_live_wing_soundcheck_daemon_tool",
        label="Broad live WING soundcheck daemon tool",
        status="blocked",
        paths=("tools/live_wing_soundcheck_daemon.py",),
        entrypoints=("live_wing_soundcheck_daemon main",),
        write_kinds=("fader", "gain", "eq"),
        allowed_live=False,
        gate=(
            "Blocked from automation and release readiness. Non-dry-run execution "
            "requires the explicit --confirm-live-wing-write acknowledgement plus "
            "operator supervision, physical readback, and rollback review."
        ),
        current_policy=(
            "This daemon can continuously write WING fader, trim and EQ settings "
            "during a broad live soundcheck loop; it is not part of the supervised "
            "MVP fader/gain apply path."
        ),
        next_action=(
            "Prefer the narrow supervised fader/gain proof runbook. Keep this "
            "daemon blocked unless a future operator runbook gives it separate "
            "readback, rollback, stop and emergency-stop proof requirements."
        ),
        notes=("requires_explicit_live_write_ack", "continuous_broad_live_loop", "not_mvp_release_authority"),
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
        id="wing_meter_subscription_transport",
        label="WING meter subscription transport",
        status="quarantined",
        paths=("backend/arrangement_automation/wing_meter_reader.py",),
        entrypoints=("WingMeterReader.start", "mixer_client.send meter subscribe"),
        write_kinds=("transport",),
        allowed_live=False,
        gate=(
            "Read telemetry subscription only; it may contact WING for meter "
            "frames but must not be treated as console-state mutation or live "
            "apply authority."
        ),
        current_policy=(
            "Meter subscriptions can support supervised measurements, but they "
            "do not grant readiness closure, apply authority, or proof of a "
            "write-capable correction family."
        ),
        next_action=(
            "Keep meter transport behind explicit supervised/read-only flows "
            "and preserve untrusted/stale readings as fail-closed evidence."
        ),
    ),
    WriteSurface(
        id="routing_snapshot_scene_writes",
        label="Routing, snapshot, and scene writes",
        status="quarantined",
        paths=(
            "backend/handlers/routing_handlers.py",
            "backend/handlers/snapshot_handlers.py",
            "backend/list_all_snapshots.py",
            "backend/list_snapshots.py",
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
            "backend/heuristics/pa_v3.py",
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
        id="mute_direct_apply",
        label="Direct mute apply",
        status="quarantined",
        paths=(
            "backend/handlers/mixer_handlers.py",
            "backend/autofoh_safety.py",
            "backend/auto_soundcheck_engine.py",
        ),
        entrypoints=(
            "set_mute",
            "mute_channel",
            "unmute_channel",
            "set_channel_mute",
            "AutoFOHSafetyController._blocked_direct_write_payload",
            "AutoSoundcheckEngine._record_blocked_console_write",
        ),
        write_kinds=("mute",),
        allowed_live=False,
        gate="Quarantined for WING deployment; direct mute writes are not part of the supervised Product Layer apply path.",
        current_policy="No WING-deployment mute writes from websocket, AutoFOH, or AutoSoundcheck automation surfaces.",
        next_action="Promote only after explicit mute-specific operator UX, readback, rollback, and emergency-state proof.",
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
            "backend/heuristics/pan_v3.py",
            "backend/auto_panner.py",
            "backend/auto_panner_adaptive.py",
            "backend/auto_fader_v2/controller.py",
            "backend/heuristics/phase_v3.py",
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
            "backend/heuristics/fx_v3.py",
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
        write_kinds=(*(LIVE_ALLOWED_WRITE_KINDS), "snapshot", "mute", "compressor"),
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
        write_kinds=(*(LIVE_ALLOWED_WRITE_KINDS), "routing", "snapshot", "eq"),
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
