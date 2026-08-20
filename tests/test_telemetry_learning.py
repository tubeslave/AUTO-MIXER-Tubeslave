import ast
import json
import subprocess
from pathlib import Path

import pytest

from backend.config_manager import ConfigManager
from backend.telemetry.collectors import (
    MockTelemetryCollector,
    WingTelemetryCollector,
    duplicate_decision_trace_as_engineer_action,
)
from backend.telemetry.learning_export import export_training_timeline, generate_summary_report, load_session
from backend.telemetry.schemas import (
    ChannelTelemetry,
    EngineerAction,
    LabelAnnotation,
    SceneMetadata,
    SPLTelemetry,
    timestamp_ms,
)
from backend.telemetry.session_manager import (
    TelemetrySessionManager,
    default_telemetry_config,
    project_git_commit,
)
from backend.telemetry.spl_provider import FileSPLProvider, MockSPLProvider
from backend.telemetry.streaming_alerts import (
    StreamingAlertEngine,
    run_streaming_alerts_for_session,
)


def _manager(tmp_path, *, enabled=True):
    config = default_telemetry_config()
    config["enabled"] = enabled
    config["storage_dir"] = str(tmp_path / "sessions")
    return TelemetrySessionManager(storage_dir=config["storage_dir"], config=config, repo_root=Path.cwd())


def test_project_git_commit_times_out_fail_closed(monkeypatch, tmp_path):
    def fake_check_output(*args, **kwargs):
        assert kwargs["timeout"] == 2.0
        raise subprocess.TimeoutExpired(cmd=args[0], timeout=kwargs["timeout"])

    monkeypatch.setattr("backend.telemetry.session_manager.subprocess.check_output", fake_check_output)

    assert project_git_commit(tmp_path) is None


def test_start_session_writes_session_scene_and_manifest(tmp_path):
    manager = _manager(tmp_path)
    session = manager.start_session(
        mode="soundcheck",
        venue_name="Test Venue",
        artist_name="Test Artist",
        scene_metadata=SceneMetadata(room_type="club", estimated_capacity=300),
    )

    assert session is not None
    session_dir = tmp_path / "sessions" / session.session_id
    assert (session_dir / "session.json").exists()
    assert (session_dir / "scene_metadata.json").exists()
    manifest = json.loads((session_dir / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["session_id"] == session.session_id
    assert manifest["schema_version"] == "0.1"
    assert manifest["safety_mode"] == "dry-run"
    assert manifest["files"]["channel_telemetry"] == "channel_telemetry.jsonl"


def test_telemetry_disabled_does_not_create_session_files(tmp_path):
    manager = _manager(tmp_path, enabled=False)
    session = manager.start_session(mode="soundcheck")

    assert session is None
    assert not (tmp_path / "sessions").exists()


@pytest.mark.parametrize("session_id", ["../escape", "nested/session", "nested\\session", "..", ""])
def test_telemetry_rejects_path_like_session_ids_before_writing(tmp_path, session_id):
    manager = _manager(tmp_path)

    with pytest.raises(ValueError, match="safe path component"):
        manager.start_session(session_id=session_id)

    assert not (tmp_path / "escape").exists()


def test_append_jsonl_events_and_manifest_counters(tmp_path):
    manager = _manager(tmp_path)
    session = manager.start_session(mode="rehearsal", venue_name="Room")
    now = timestamp_ms()

    manager.append_event(
        "channel_telemetry",
        ChannelTelemetry(
            timestamp_ms=now,
            session_id=session.session_id,
            channel_id=1,
            channel_name="Lead Vocal",
            detected_instrument="vocal",
            fader_db=-8.0,
            trim_db=1.0,
            mute=False,
            solo=False,
            pan=0.0,
            meter_rms_dbfs=-18.0,
            meter_peak_dbfs=-6.0,
        ).to_dict(),
    )
    manager.append_event(
        "engineer_actions",
        EngineerAction(
            timestamp_ms=now,
            session_id=session.session_id,
            actor="human",
            target_type="channel",
            target_id="1",
            action_type="fader_move",
            before_value=-9.0,
            after_value=-8.0,
            reason="vocal lift",
            confidence=1.0,
            safety_status="logged_only",
        ).to_dict(),
    )

    session_dir = tmp_path / "sessions" / session.session_id
    channel_rows = (session_dir / "channel_telemetry.jsonl").read_text(encoding="utf-8").splitlines()
    manifest = json.loads((session_dir / "manifest.json").read_text(encoding="utf-8"))
    assert len(channel_rows) == 1
    assert manifest["counters"]["channel_telemetry"] == 1
    assert manifest["counters"]["engineer_actions"] == 1


def test_stop_session_sets_ended_at_and_preserves_manifest(tmp_path):
    manager = _manager(tmp_path)
    session = manager.start_session(mode="concert")
    stopped = manager.stop_session(session.session_id, reason="show ended")

    assert stopped is not None
    payload = json.loads((tmp_path / "sessions" / session.session_id / "session.json").read_text(encoding="utf-8"))
    assert payload["ended_at"]
    assert "show ended" in payload["notes"]
    assert not (tmp_path / "sessions" / "current_session.json").exists()


def test_mock_collector_writes_channel_bus_and_action_events(tmp_path):
    manager = _manager(tmp_path)
    session = manager.start_session(mode="offline_test")

    counts = MockTelemetryCollector(channel_count=3).collect_once(manager)

    session_dir = tmp_path / "sessions" / session.session_id
    assert counts["channel_telemetry"] == 3
    assert len((session_dir / "channel_telemetry.jsonl").read_text(encoding="utf-8").splitlines()) == 3
    assert len((session_dir / "bus_telemetry.jsonl").read_text(encoding="utf-8").splitlines()) == 1
    assert len((session_dir / "engineer_actions.jsonl").read_text(encoding="utf-8").splitlines()) == 1


def test_mock_and_file_spl_providers_write_spl_telemetry(tmp_path):
    manager = _manager(tmp_path)
    session = manager.start_session(mode="soundcheck")

    mock_row = MockSPLProvider(laeq_1s=86.5).read_once(session_id=session.session_id)
    manager.append_event("spl_telemetry", mock_row.to_dict())
    csv_path = tmp_path / "spl.csv"
    csv_path.write_text("timestamp_ms,laeq_1s,lcpeak,lafmax,source\n123,88.1,105.0,91.2,meter\n", encoding="utf-8")
    file_row = FileSPLProvider(csv_path).read_once(session_id=session.session_id)
    manager.append_event("spl_telemetry", file_row.to_dict())

    rows = [
        json.loads(line)
        for line in (tmp_path / "sessions" / session.session_id / "spl_telemetry.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    assert rows[0]["laeq_1s"] == 86.5
    assert rows[1]["source"] == "meter"
    assert rows[1]["timestamp_ms"] == 123


def test_export_training_timeline_merges_events_by_timestamp(tmp_path):
    manager = _manager(tmp_path)
    session = manager.start_session(mode="offline_test")
    manager.append_event(
        "labels",
        LabelAnnotation(
            timestamp_ms=300,
            session_id=session.session_id,
            label="vocal_clear",
            source="human",
        ).to_dict(),
    )
    manager.append_event(
        "spl_telemetry",
        SPLTelemetry(timestamp_ms=100, session_id=session.session_id, laeq_1s=84.0).to_dict(),
    )
    timeline_path = tmp_path / "timeline.jsonl"

    timeline = export_training_timeline(
        session.session_id,
        storage_dir=tmp_path / "sessions",
        output_path=timeline_path,
    )
    summary = generate_summary_report(session.session_id, storage_dir=tmp_path / "sessions")

    assert [row["event_type"] for row in timeline] == ["spl_telemetry", "labels"]
    assert timeline_path.exists()
    assert summary["timeline_event_count"] == 2
    assert Path(summary["report_path"]).exists()
    loaded = load_session(session.session_id, storage_dir=tmp_path / "sessions")
    assert loaded["events"]["labels"][0]["label"] == "vocal_clear"


class _CachedMixer:
    state = {
        "/ch/1/$name": "Kick",
        "/ch/1/fdr": -10.0,
        "/ch/1/in/set/trim": 1.5,
        "/ch/1/mute": 0,
        "/ch/1/pan": 0.0,
        "/ch/1/gate/on": 1,
        "/ch/1/dyn/on": 1,
    }


class _Capture:
    def get_rms(self, channel):
        return -20.0 + channel

    def get_peak(self, channel):
        return -8.0 + channel

    def get_lufs(self, channel):
        return -21.0 + channel


def test_wing_collector_reads_cached_state_only(tmp_path):
    manager = _manager(tmp_path)
    session = manager.start_session(mode="soundcheck")
    collector = WingTelemetryCollector(_CachedMixer(), _Capture(), channel_count=1)

    counts = collector.collect_once(manager)

    assert counts["channel_telemetry"] == 1
    row = json.loads((tmp_path / "sessions" / session.session_id / "channel_telemetry.jsonl").read_text(encoding="utf-8").splitlines()[0])
    assert row["channel_name"] == "Kick"
    assert row["fader_db"] == -10.0
    assert row["trim_db"] == 1.5
    assert row["meter_peak_dbfs"] == -7.0


def test_decision_trace_can_be_duplicated_as_automixer_action(tmp_path):
    manager = _manager(tmp_path)
    session = manager.start_session(mode="offline_test")

    row = duplicate_decision_trace_as_engineer_action(
        manager,
        {
            "decision_id": "decision-1",
            "source": "unit",
            "target": 2,
            "current_value": -8.0,
            "proposed_value": -8.5,
            "proof_status": "offline",
        },
    )

    assert row["actor"] == "automixer"
    assert row["target_type"] == "channel"
    assert row["target_id"] == "2"
    rows = (tmp_path / "sessions" / session.session_id / "engineer_actions.jsonl").read_text(encoding="utf-8").splitlines()
    assert len(rows) == 1


def test_config_default_telemetry_disabled():
    config = ConfigManager().to_dict()
    assert config["telemetry"]["enabled"] is False
    assert config["telemetry"]["spl"]["enabled"] is False
    assert config["telemetry"]["safety"]["allow_live_writes"] is False
    assert config["telemetry"]["safety"]["telemetry_read_only"] is True


def test_telemetry_modules_do_not_expose_live_write_calls():
    root = Path("backend/telemetry")
    combined = "\n".join(path.read_text(encoding="utf-8") for path in root.glob("*.py"))
    for forbidden in (
        "set_fader(",
        "set_gain(",
        "set_eq(",
        "set_compressor(",
        "route_output(",
        "route_multiple_outputs(",
        "load_snap(",
        "save_snap(",
        "reset(",
        "mute(",
        "unmute(",
        "set_channel_input(",
        "set_channel_alt_input(",
        "set_insert(",
        "set_fx_model(",
        "set_fx_on(",
        "set_fx_mix(",
        "set_fx_parameter(",
        "send(",
    ):
        assert forbidden not in combined


def test_telemetry_modules_do_not_import_live_mixer_transports():
    root = Path("backend/telemetry")
    forbidden_modules = (
        "backend.wing_client",
        "wing_client",
        "backend.dlive_client",
        "dlive_client",
        "backend.osc",
        "osc",
        "pythonosc",
        "backend.mixer_client_base",
        "mixer_client_base",
    )
    violations = []

    for path in root.glob("*.py"):
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                modules = [alias.name for alias in node.names]
            elif isinstance(node, ast.ImportFrom) and node.level == 0:
                modules = [node.module or ""]
            else:
                continue

            for module in modules:
                if any(module == forbidden or module.startswith(f"{forbidden}.") for forbidden in forbidden_modules):
                    violations.append(f"{path}: {module}")

    assert violations == []


def test_streaming_alert_engine_detects_lufs_fader_override_and_headroom(tmp_path):
    manager = _manager(tmp_path)
    session = manager.start_session(mode="offline_test")
    sid = session.session_id

    manager.append_event(
        "bus_telemetry",
        {
            "timestamp_ms": 1000,
            "session_id": sid,
            "bus_name": "Main",
            "lufs_short": -20.0,
            "peak_dbfs": -8.0,
        },
    )
    manager.append_event(
        "bus_telemetry",
        {
            "timestamp_ms": 5900,
            "session_id": sid,
            "bus_name": "Main",
            "lufs_short": -17.5,
            "peak_dbfs": -2.5,
        },
    )
    manager.append_event(
        "channel_telemetry",
        {
            "timestamp_ms": 10000,
            "session_id": sid,
            "channel_id": 12,
            "channel_name": "Lead Vocal",
            "fader_db": -12.0,
            "meter_peak_dbfs": -9.0,
        },
    )
    manager.append_event(
        "channel_telemetry",
        {
            "timestamp_ms": 10150,
            "session_id": sid,
            "channel_id": 12,
            "channel_name": "Lead Vocal",
            "fader_db": -5.5,
            "meter_peak_dbfs": -0.05,
        },
    )
    for offset in (0, 1000, 2000, 3000):
        manager.append_event(
            "engineer_actions",
            EngineerAction(
                timestamp_ms=20000 + offset,
                session_id=sid,
                actor="human",
                target_type="channel",
                target_id="12",
                action_type="fader_move",
                before_value=-8.0,
                after_value=-7.5,
                reason="manual correction",
            ).to_dict(),
        )

    summary = run_streaming_alerts_for_session(
        sid,
        storage_dir=tmp_path / "sessions",
        output_dir=tmp_path / "alerts",
    )
    alert_rows = [
        json.loads(line)
        for line in (tmp_path / "alerts" / "alerts.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    rules = {row["rule"] for row in alert_rows}

    assert summary["alert_count"] == len(alert_rows)
    assert {"lufs_drift", "headroom_low", "fader_jump", "clipping_risk", "manual_override_burst"} <= rules
    assert (tmp_path / "alerts" / "dashboard.csv").exists()
    dashboard = (tmp_path / "alerts" / "dashboard.csv").read_text(encoding="utf-8")
    assert "manual_override_burst" in dashboard
    assert summary["safety"] == {"read_only": True, "live_write": False}


def test_streaming_alert_engine_ignores_automixer_actions_for_override_burst():
    engine = StreamingAlertEngine()
    events = []
    for idx in range(5):
        events.append(
            {
                "event_type": "engineer_actions",
                "timestamp_ms": idx * 1000,
                "payload": {
                    "timestamp_ms": idx * 1000,
                    "session_id": "session-1",
                    "actor": "automixer",
                    "target_type": "channel",
                    "target_id": "1",
                    "action_type": "fader_move",
                },
            }
        )

    alerts = engine.process(events)

    assert alerts == []
    assert len(engine.dashboard_rows) == 5
