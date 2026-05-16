"""Canonical gain/fader runtime classification for deployment and lab surfaces."""

from __future__ import annotations

from typing import Any, Dict, List


def build_gain_fader_runtime_summary(
    *,
    connection_mode: str | None,
    wing_boundary_active: bool,
) -> Dict[str, Any]:
    deployment_surface = connection_mode == "wing" and wing_boundary_active

    if deployment_surface:
        selected_runtime = {
            "id": "wing_supervised_soundcheck",
            "surface": "deployment",
            "analyzer_runtime": "AutoSoundcheckEngine",
            "proposal_transport": "ReplayWriteIntentAdapter",
            "live_write_path": "AutoMixerServer._apply_manual_console_write",
            "policy": "dry_run_proposals_plus_supervised_manual_promotion",
            "summary": (
                "WING deployment runs analyzer output through AutoSoundcheckEngine dry-run "
                "write intents and permits live gain/fader changes only through the "
                "supervised manual write gate."
            ),
        }
    else:
        selected_runtime = {
            "id": "legacy_split_lab_runtime",
            "surface": "lab_or_non_wing",
            "analyzer_runtime": "LUFSGainStagingController + AutoFaderController",
            "proposal_transport": "direct_controller_writes",
            "live_write_path": "controller_managed",
            "policy": "legacy_runtime_not_approved_for_wing_deployment",
            "summary": (
                "Non-WING and lab flows still use the legacy split gain/fader controllers; "
                "that path is not the approved WING deployment runtime."
            ),
        }

    runtimes: List[Dict[str, Any]] = [
        {
            "id": "auto_soundcheck_engine",
            "module": "backend/auto_soundcheck_engine.py",
            "role": "headless analyzer and recommendation runtime",
            "status": "deployment_active_dry_run" if deployment_surface else "available",
            "write_policy": "replay_write_intents_first",
            "notes": (
                "Forced to auto_apply=False on WING by soundcheck handlers; gain/fader proposals "
                "emit replay intents before any live write attempt."
            ),
        },
        {
            "id": "manual_supervised_writes",
            "module": "backend/server.py",
            "role": "only approved live gain/fader mutation path on WING deployment",
            "status": "deployment_active" if deployment_surface else "available",
            "write_policy": "armed_manual_gate_with_readback",
            "notes": (
                "Applies fader/gain changes through the supervised write gate with approval, "
                "cooldown, readback, and rollback tracking."
            ),
        },
        {
            "id": "lufs_gain_staging_controller",
            "module": "backend/lufs_gain_staging.py",
            "role": "legacy realtime gain controller",
            "status": "deployment_quarantined" if deployment_surface else "legacy_lab_runtime",
            "write_policy": "controller_direct",
            "notes": "UI-wired through start_realtime_correction; blocked on WING deployment surface.",
        },
        {
            "id": "safe_gain_calibrator",
            "module": "backend/lufs_gain_staging.py",
            "role": "one-shot legacy gain analysis/apply helper",
            "status": "deployment_quarantined" if deployment_surface else "legacy_lab_runtime",
            "write_policy": "controller_direct",
            "notes": (
                "Wraps legacy gain staging buffers for a learning/apply flow; not the deployment "
                "runtime for WING."
            ),
        },
        {
            "id": "auto_fader_controller",
            "module": "backend/auto_fader.py",
            "role": "legacy realtime fader controller",
            "status": "deployment_quarantined" if deployment_surface else "legacy_lab_runtime",
            "write_policy": "controller_direct_with_wing_intent_guard",
            "notes": "UI-wired through start_realtime_fader and auto-balance; quarantined on WING deployment.",
        },
        {
            "id": "auto_fader_v2_controller",
            "module": "backend/auto_fader_v2/controller.py",
            "role": "experimental analyzer/fader runtime",
            "status": "experimental_unwired",
            "write_policy": "not_on_server_surface",
            "notes": "Covered by focused tests and helper modules, but not instantiated by backend/server.py.",
        },
    ]

    return {
        "selected_runtime": selected_runtime,
        "deployment_surface": deployment_surface,
        "runtimes": runtimes,
    }
