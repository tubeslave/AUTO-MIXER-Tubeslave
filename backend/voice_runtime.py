"""Declared voice-runtime contract for the supervised pilot surface."""

from __future__ import annotations

from typing import Any, Optional


def build_voice_runtime_summary(
    *,
    connection_mode: Optional[str],
    wing_boundary_active: bool,
    sherpa_available: bool,
) -> dict[str, Any]:
    """Summarize the supported voice runtime and quarantined alternatives."""

    deployment_surface = connection_mode == "wing" and wing_boundary_active
    selected_runtime = {
        "id": (
            "wing_voice_supervised_runtime"
            if deployment_surface
            else "lab_voice_runtime_family"
        ),
        "entrypoint": "AutoMixerServer.start_voice_control",
        "primary_backend": "VoiceControlSherpa" if sherpa_available else "VoiceControlV2",
        "fallback_backend": "VoiceControlV2" if sherpa_available else None,
        "command_execution_path": "AutoMixerServer._execute_voice_command",
        "live_write_path": "AutoMixerServer._apply_manual_console_write",
        "allowed_live_write_kinds": ["fader", "gain"],
        "blocked_voice_write_kinds": ["snapshot", "mute"],
        "notes": (
            "Sherpa/GigaAM is the declared pilot runtime; Whisper V2 is fallback-only."
            if sherpa_available
            else "Sherpa is unavailable, so the runtime falls back to Whisper V2."
        ),
    }

    runtimes = [
        {
            "id": "voice_control_sherpa",
            "module": "backend/voice_control_sherpa.py",
            "status": (
                "deployment_primary"
                if deployment_surface and sherpa_available
                else "available_primary_candidate"
                if sherpa_available
                else "dependency_unavailable"
            ),
            "role": "pilot_primary",
            "notes": "Primary Russian pilot runtime when Sherpa dependencies are installed.",
        },
        {
            "id": "voice_control_v2",
            "module": "backend/voice_control_v2.py",
            "status": (
                "deployment_fallback"
                if deployment_surface
                else "lab_supported_fallback"
            ),
            "role": "pilot_fallback",
            "notes": "Fallback backend kept for continuity if Sherpa init/load fails.",
        },
        {
            "id": "voice_control",
            "module": "backend/voice_control.py",
            "status": (
                "deployment_quarantined_legacy"
                if deployment_surface
                else "lab_only_legacy"
            ),
            "role": "legacy_non_pilot",
            "notes": "Legacy Whisper backend retained for offline tests only; not a supported pilot runtime.",
        },
        {
            "id": "voice_control_vosk",
            "module": "backend/voice_control_vosk.py",
            "status": (
                "deployment_quarantined_legacy"
                if deployment_surface
                else "lab_only_legacy"
            ),
            "role": "legacy_non_pilot",
            "notes": "Unused Vosk backend is quarantined from the pilot surface.",
        },
    ]

    return {
        "deployment_surface": deployment_surface,
        "selected_runtime": selected_runtime,
        "runtimes": runtimes,
    }
