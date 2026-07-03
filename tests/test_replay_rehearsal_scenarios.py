from __future__ import annotations

import json
from pathlib import Path

from ai_mixing_pipeline.decision_layer.replay_rehearsal_scenarios import (
    REPLAY_REHEARSAL_SCHEMA_VERSION,
    REPLAY_REHEARSAL_TIMELINE_SCHEMA_VERSION,
    ReplayRehearsalScenario,
    build_replay_rehearsal_timeline,
    normalize_replay_rehearsal_scenario,
)


def _fixture_payload() -> dict:
    path = Path("tests/fixtures/replay_rehearsal_scenario_v1.json")
    return json.loads(path.read_text(encoding="utf-8"))


def _timeline_fixture_payload() -> dict:
    path = Path("tests/fixtures/replay_rehearsal_timeline_v1.json")
    return json.loads(path.read_text(encoding="utf-8"))


def test_replay_rehearsal_scenario_normalizes_and_roundtrips():
    payload = _fixture_payload()
    scenario = normalize_replay_rehearsal_scenario(payload)
    roundtrip = ReplayRehearsalScenario.from_dict(scenario.to_dict())

    assert scenario.schema_version == REPLAY_REHEARSAL_SCHEMA_VERSION
    assert scenario.session.dry_run_only is True
    assert scenario.session.live_mixer_writes is False
    assert [stage.ordinal for stage in scenario.stages] == list(range(1, 10))
    assert scenario.replay_correlation_ids == sorted(scenario.replay_correlation_ids)
    assert roundtrip.to_dict() == scenario.to_dict()


def test_replay_rehearsal_timeline_is_deterministic_for_reordered_input():
    payload = _fixture_payload()
    reversed_payload = dict(payload)
    reversed_payload["stages"] = list(reversed(payload["stages"]))

    baseline = build_replay_rehearsal_timeline(payload)
    reordered = build_replay_rehearsal_timeline(reversed_payload)

    assert baseline["schema_version"] == REPLAY_REHEARSAL_TIMELINE_SCHEMA_VERSION
    assert baseline["dry_run_only"] is True
    assert baseline["live_mixer_writes"] is False
    assert baseline["artifact_signature"] == reordered["artifact_signature"]
    assert baseline["timeline"] == reordered["timeline"]
    assert [event["seq"] for event in baseline["timeline"]] == list(range(1, 10))


def test_replay_rehearsal_timeline_preserves_stage_order_and_signatures():
    timeline = build_replay_rehearsal_timeline(_fixture_payload())

    assert [event["stage"] for event in timeline["timeline"]] == [
        "pre_show",
        "soundcheck",
        "replay_validation",
        "shadow_mode",
        "operator_review",
        "rollback",
        "escalation",
        "fallback",
        "post_show_review",
    ]
    assert all(event["dry_run_only"] is True for event in timeline["timeline"])
    assert all(event["live_mixer_writes"] is False for event in timeline["timeline"])
    assert all(event["event_signature"] for event in timeline["timeline"])


def test_replay_rehearsal_timeline_matches_frozen_fixture():
    assert build_replay_rehearsal_timeline(_fixture_payload()) == _timeline_fixture_payload()
