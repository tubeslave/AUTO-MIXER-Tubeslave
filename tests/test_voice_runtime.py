from backend.voice_runtime import build_voice_runtime_summary


def test_wing_deployment_voice_runtime_declares_sherpa_primary():
    summary = build_voice_runtime_summary(
        connection_mode="wing",
        wing_boundary_active=True,
        sherpa_available=True,
    )

    assert summary["deployment_surface"] is True
    assert summary["selected_runtime"]["id"] == "wing_voice_supervised_runtime"
    assert summary["selected_runtime"]["primary_backend"] == "VoiceControlSherpa"
    assert summary["selected_runtime"]["fallback_backend"] == "VoiceControlV2"
    assert summary["selected_runtime"]["allowed_live_write_kinds"] == ["fader", "gain"]
    assert summary["selected_runtime"]["blocked_voice_write_kinds"] == ["snapshot", "mute"]

    runtime_status = {item["id"]: item["status"] for item in summary["runtimes"]}
    assert runtime_status["voice_control_sherpa"] == "deployment_primary"
    assert runtime_status["voice_control_v2"] == "deployment_fallback"
    assert runtime_status["voice_control"] == "deployment_quarantined_legacy"
    assert runtime_status["voice_control_vosk"] == "deployment_quarantined_legacy"


def test_voice_runtime_falls_back_to_whisper_when_sherpa_is_unavailable():
    summary = build_voice_runtime_summary(
        connection_mode="wing",
        wing_boundary_active=True,
        sherpa_available=False,
    )

    assert summary["selected_runtime"]["primary_backend"] == "VoiceControlV2"
    assert summary["selected_runtime"]["fallback_backend"] is None

    runtime_status = {item["id"]: item["status"] for item in summary["runtimes"]}
    assert runtime_status["voice_control_sherpa"] == "dependency_unavailable"
    assert runtime_status["voice_control_v2"] == "deployment_fallback"


def test_non_wing_voice_runtime_keeps_legacy_backends_lab_only():
    summary = build_voice_runtime_summary(
        connection_mode="dlive",
        wing_boundary_active=False,
        sherpa_available=True,
    )

    assert summary["deployment_surface"] is False
    assert summary["selected_runtime"]["id"] == "lab_voice_runtime_family"

    runtime_status = {item["id"]: item["status"] for item in summary["runtimes"]}
    assert runtime_status["voice_control_sherpa"] == "available_primary_candidate"
    assert runtime_status["voice_control_v2"] == "lab_supported_fallback"
    assert runtime_status["voice_control"] == "lab_only_legacy"
    assert runtime_status["voice_control_vosk"] == "lab_only_legacy"
