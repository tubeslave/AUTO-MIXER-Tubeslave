"""Tests for source-grounded rule retrieval and logging."""

import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "backend"))

from config_manager import ConfigManager
from source_knowledge import (
    DecisionTrace,
    FeedbackRecord,
    SourceDecisionLogger,
    SourceGroundedConfig,
    SourceKnowledgeLayer,
    SourceKnowledgeStore,
    iter_jsonl_events,
)


def test_seeded_sources_and_rules_validate():
    store = SourceKnowledgeStore()

    assert "senior_mixing_secrets" in store.sources
    assert "eq.reason_only" in store.rules
    assert store.validate() == []


def test_search_rules_filters_by_problem_instrument_and_domain():
    store = SourceKnowledgeStore({"source_knowledge": {"min_rule_confidence": 0.0}})

    matches = store.search_rules(
        "sharp vocal harshness",
        domains=["eq"],
        instruments=["lead_vocal"],
        problems=["harshness"],
        limit=3,
    )

    assert matches
    assert matches[0].rule.rule_id in {
        "harshness.small_targeted_cut",
        "eq.reason_only",
    }
    assert matches[0].rule.source_ids


def test_config_defaults_keep_source_knowledge_disabled():
    config = ConfigManager()

    source_config = SourceGroundedConfig.from_mapping(config.get("source_knowledge"))

    assert source_config.enabled is False
    assert source_config.log_path == "logs/source_grounded_decisions.jsonl"
    assert source_config.allow_unsourced_rules is False


def test_decision_logger_writes_decision_and_feedback(tmp_path):
    log_path = tmp_path / "source_decisions.jsonl"
    logger = SourceDecisionLogger(log_path, queue_maxsize=8)
    trace = DecisionTrace(
        session_id="session-1",
        decision_id="decision-1",
        channel="Vocal",
        instrument="lead_vocal",
        problem="harshness",
        selected_rule_ids=["harshness.small_targeted_cut"],
        source_ids=["senior_mixing_secrets", "izhaki_mixing_audio"],
        selected_action={"action_type": "eq_candidate", "gain_db": -1.5, "freq_hz": 3500},
        before_metrics={"presence_db": 1.8},
        after_metrics={"presence_db": 0.4},
        outcome="pending_feedback",
        confidence=0.78,
        osc_sent=False,
    )
    feedback = FeedbackRecord(
        session_id="session-1",
        decision_id="decision-1",
        rating="better",
        comment="less sharp",
    )

    logger.start()
    assert logger.log_decision(trace) is True
    assert logger.log_feedback(feedback) is True
    logger.stop()

    rows = list(iter_jsonl_events(log_path))
    assert [row["event_type"] for row in rows] == ["source_decision", "source_feedback"]
    assert rows[0]["selected_rule_ids"] == ["harshness.small_targeted_cut"]
    assert rows[1]["rating"] == "better"


def test_layer_retrieval_logging_is_shadow_only(tmp_path):
    log_path = tmp_path / "source_decisions.jsonl"
    layer = SourceKnowledgeLayer({
        "source_knowledge": {
            "enabled": True,
            "log_path": str(log_path),
            "min_rule_confidence": 0.0,
        }
    })

    layer.start()
    matches = layer.retrieve("mud low mid", domains=["eq"], problems=["mud"], limit=2)
    layer.stop()

    assert matches
    rows = log_path.read_text(encoding="utf-8").strip().splitlines()
    assert rows
    payload = json.loads(rows[0])
    assert payload["event_type"] == "source_rule_retrieval"
    assert payload["matches"]
