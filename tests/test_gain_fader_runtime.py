from backend.gain_fader_runtime import build_gain_fader_runtime_summary


def test_wing_deployment_runtime_selects_supervised_soundcheck_path():
    summary = build_gain_fader_runtime_summary(
        connection_mode="wing",
        wing_boundary_active=True,
    )

    assert summary["deployment_surface"] is True
    assert summary["selected_runtime"]["id"] == "wing_supervised_soundcheck"
    assert summary["selected_runtime"]["analyzer_runtime"] == "AutoSoundcheckEngine"
    assert summary["selected_runtime"]["live_write_path"] == "AutoMixerServer._apply_manual_console_write"

    runtime_status = {item["id"]: item["status"] for item in summary["runtimes"]}
    assert runtime_status["auto_soundcheck_engine"] == "deployment_active_dry_run"
    assert runtime_status["manual_supervised_writes"] == "deployment_active"
    assert runtime_status["lufs_gain_staging_controller"] == "deployment_quarantined"
    assert runtime_status["auto_fader_controller"] == "deployment_quarantined"
    assert runtime_status["auto_fader_v2_controller"] == "experimental_unwired"


def test_non_wing_runtime_keeps_legacy_lab_split():
    summary = build_gain_fader_runtime_summary(
        connection_mode="dlive",
        wing_boundary_active=False,
    )

    assert summary["deployment_surface"] is False
    assert summary["selected_runtime"]["id"] == "legacy_split_lab_runtime"
    assert summary["selected_runtime"]["analyzer_runtime"] == "LUFSGainStagingController + AutoFaderController"

    runtime_status = {item["id"]: item["status"] for item in summary["runtimes"]}
    assert runtime_status["auto_soundcheck_engine"] == "available"
    assert runtime_status["lufs_gain_staging_controller"] == "legacy_lab_runtime"
    assert runtime_status["safe_gain_calibrator"] == "legacy_lab_runtime"
    assert runtime_status["auto_fader_controller"] == "legacy_lab_runtime"
