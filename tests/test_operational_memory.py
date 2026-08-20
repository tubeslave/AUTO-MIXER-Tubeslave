from __future__ import annotations

import json
from pathlib import Path

from operational_memory import OperationalMemoryStore
from replay_write_intent_adapter import ReplayWriteIntentAdapter
from session_manager import SessionManager
from wing_telemetry_recorder import WingTelemetryRecorder


class WingMixerDouble:
    pass


def test_operational_memory_store_builds_gpt_readable_views(tmp_path: Path):
    store = OperationalMemoryStore(tmp_path / "memory")

    outcome = store.record_task_outcome(
        issue_identifier="TUB-164",
        title="Create centralized operational memory system",
        status="in_progress",
        summary="Initial integration is in place.",
        evidence_paths=[tmp_path / "evidence.json"],
        context={"owner": "qa"},
    )
    sync = store.record_context_sync(
        agent_id="agent-1",
        issue_identifier="TUB-164",
        summary="Hand off current event inventory to the next agent.",
        related_event_ids=[outcome["event_id"]],
        context={"step": "handoff"},
    )
    knowledge = store.write_knowledge_snapshot(
        snapshot_name="wing-write-safety",
        summary="Supervised WING writes remain dry-run first.",
        content={"approved_paths": ["fader", "gain"], "default_mode": "dry_run_only"},
        tags=["wing", "safety"],
    )

    snapshot = store.read_snapshot()
    summary_text = store.summary_path.read_text(encoding="utf-8")
    knowledge_payload = json.loads(Path(knowledge["snapshot_path"]).read_text(encoding="utf-8"))

    assert snapshot["event_count"] == 3
    assert snapshot["counts_by_type"]["task_outcome"] == 1
    assert snapshot["counts_by_type"]["context_sync"] == 1
    assert snapshot["counts_by_type"]["knowledge_snapshot"] == 1
    assert snapshot["task_outcomes"][0]["payload"]["issue_identifier"] == "TUB-164"
    assert snapshot["context_sync_events"][0]["event_id"] == sync["event_id"]
    assert "Operational Memory Summary" in summary_text
    assert "task_outcome: 1" in summary_text
    assert knowledge_payload["content"]["default_mode"] == "dry_run_only"


def test_session_manager_records_centralized_memory_events(tmp_path: Path):
    store = OperationalMemoryStore(tmp_path / "memory")
    manager = SessionManager(sessions_dir=str(tmp_path / "sessions"), memory_store=store)

    session = manager.create_session("festival", venue="Main Hall")
    manager.update_config(target_lufs=-16.0)
    manager.end_session(session.id)

    snapshot = store.read_snapshot()
    actions = [event["summary"] for event in snapshot["session_events"]]

    assert snapshot["counts_by_type"]["session_event"] == 3
    assert any("created" in action for action in actions)
    assert any("config_updated" in action for action in actions)
    assert any("ended" in action for action in actions)


def test_replay_write_intents_are_copied_into_operational_memory(tmp_path: Path):
    store = OperationalMemoryStore(tmp_path / "memory")
    adapter = ReplayWriteIntentAdapter(source_system="unit_test", memory_store=store)

    record = adapter.emit_fader_intent(
        mixer_client=WingMixerDouble(),
        mixer_type="wing",
        channel=5,
        current_fader_db=-12.0,
        target_fader_db=-9.0,
        readback_source="unit_test_snapshot",
        context_metadata={"scene_id": "intro"},
    )

    snapshot = store.read_snapshot()

    assert record is not None
    assert snapshot["counts_by_type"]["write_intent"] == 1
    assert snapshot["write_intents"][0]["payload"]["write_kind"] == "fader"
    assert snapshot["write_intents"][0]["payload"]["metadata"]["scene_id"] == "intro"


def test_operational_memory_recursively_redacts_credentials_before_any_view_is_written(tmp_path: Path):
    store = OperationalMemoryStore(tmp_path / "memory")
    raw_secret = "sk-super-secret-value-1234567890"
    telegram_secret = "123456789:abcdefghijklmnopqrstuvwxyzABCDE"

    event = store.record_event(
        "security-test",
        {
            "metadata": {
                "openai_api_key": raw_secret,
                "nested": [{"password": "do-not-store"}],
                "authorization_header": f"Bearer {raw_secret}",
                "free_text": f"token accidentally embedded: {telegram_secret}",
            }
        },
        summary=f"Bearer {raw_secret}",
    )

    assert event["payload"]["metadata"]["openai_api_key"] == "[REDACTED]"
    assert event["payload"]["metadata"]["nested"][0]["password"] == "[REDACTED]"
    persisted = "\n".join(
        path.read_text(encoding="utf-8")
        for path in (store.events_path, store.snapshot_path, store.summary_path)
    )
    assert raw_secret not in persisted
    assert telegram_secret not in persisted
    assert "do-not-store" not in persisted


def test_wing_telemetry_recorder_captures_start_and_stop_in_operational_memory(tmp_path: Path):
    store = OperationalMemoryStore(tmp_path / "memory")
    recorder = WingTelemetryRecorder(
        base_dir=str(tmp_path / "telemetry"),
        session_label="ops_memory",
        metadata={"observation_only": True},
        memory_store=store,
    )

    recorder.start()
    recorder.record_event("query_sent", address="/ch/1/fdr", source="send")
    recorder.stop(reason="unit_test")

    snapshot = store.read_snapshot()
    telemetry_events = snapshot["telemetry_sessions"]

    assert snapshot["counts_by_type"]["telemetry_session"] == 2
    assert telemetry_events[0]["payload"]["phase"] == "started"
    assert telemetry_events[-1]["payload"]["phase"] == "stopped"
    assert telemetry_events[-1]["payload"]["metadata"]["stop_reason"] == "unit_test"
