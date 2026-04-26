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


def test_drum_time_alignment_rule_is_source_grounded_and_shadow_only():
    store = SourceKnowledgeStore({"source_knowledge": {"min_rule_confidence": 0.0}})

    rule = store.get_rule("phase.drum_time_alignment_audition")
    assert rule is not None
    assert rule.mode == "shadow"
    assert "aes_kruk_sobecki_phase_alignment" in rule.source_ids
    assert store.get_source("sos_multimic_time_alignment") is not None
    assert rule.bounds["requires_mono_check"] is True
    assert rule.bounds["max_live_delay_ms"] <= 5.0

    matches = store.search_rules(
        "drum delay phase alignment mono",
        domains=["phase"],
        instruments=["snare"],
        problems=["comb_filtering"],
        action_types=["phase_alignment_candidate"],
        limit=1,
    )

    assert matches
    assert matches[0].rule.rule_id == "phase.drum_time_alignment_audition"


def test_modulation_and_spatial_fx_rules_are_retrievable():
    store = SourceKnowledgeStore({"source_knowledge": {"min_rule_confidence": 0.0}})

    predelay_rule = store.get_rule("fx.predelay_by_role_preserve_attack")
    modulation_rule = store.get_rule("fx.modulation_support_width_texture")
    assert predelay_rule is not None
    assert modulation_rule is not None
    assert "izotope_reverb_predelay" in predelay_rule.source_ids
    assert "eventide_h90_modulation" in modulation_rule.source_ids
    assert store.get_source("sos_optimise_reverb_treatments") is not None
    assert predelay_rule.bounds["vocal_predelay_ms"] == [20.0, 80.0]
    assert modulation_rule.bounds["chorus_delay_ms"] == [1.5, 35.0]

    fx_matches = store.search_rules(
        "vocal filtered ducked predelay tempo delay reverb depth",
        domains=["fx"],
        instruments=["lead_vocal"],
        problems=["washed_out_front"],
        action_types=["fx_send_candidate"],
        limit=6,
    )
    fx_rule_ids = {match.rule.rule_id for match in fx_matches}

    assert "fx.predelay_by_role_preserve_attack" in fx_rule_ids
    assert "fx.ducked_returns_front_clarity" in fx_rule_ids

    modulation_matches = store.search_rules(
        "chorus phaser flanger support width texture",
        domains=["modulation"],
        instruments=["guitar"],
        problems=["narrow_mix"],
        action_types=["fx_send_candidate"],
        limit=3,
    )

    assert modulation_matches
    assert modulation_matches[0].rule.rule_id == "fx.modulation_support_width_texture"


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
