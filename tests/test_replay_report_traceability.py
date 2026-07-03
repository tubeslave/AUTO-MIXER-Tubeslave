"""Replay traceability tests for report writers."""

from __future__ import annotations

import json

from ai_mixing_pipeline.models import CandidateAction, DecisionResult, MixCandidate, SafetyResult
from ai_mixing_pipeline.reports.decision_report_writer import write_decision_layer_reports
from ai_mixing_pipeline.reports.writers import write_accepted_rejected_actions, write_decision_log


def test_offline_report_writers_emit_replay_correlation_ids(tmp_path):
    candidate = MixCandidate(
        candidate_id="candidate-1",
        label="Candidate 1",
        actions=[
            CandidateAction(
                action_type="gain_change",
                target="Lead Vox",
                parameters={"gain_db": -1.5, "replay_correlation_id": "corr::offline::1"},
                reason="Replay safety test",
            )
        ],
    )
    decision = DecisionResult(
        selected_candidate_id="candidate-1",
        final_scores={"candidate-1": 0.9},
        normalized_weights={"critic": 1.0},
        explanations={"candidate-1": "selected"},
        no_change_selected=False,
    )
    safety = SafetyResult(candidate_id="candidate-1", passed=True, safety_score=1.0)

    log_path = write_decision_log(
        tmp_path / "decision_log.jsonl",
        mode="offline_test",
        candidates=[candidate],
        evaluations={"candidate-1": {}},
        decision=decision,
        safety_results={"candidate-1": safety},
        render_paths={"candidate-1": "renders/candidate-1.wav"},
    )
    accepted_path, _ = write_accepted_rejected_actions(
        tmp_path,
        [candidate],
        decision,
        {"candidate-1": safety},
    )

    row = json.loads(log_path.read_text(encoding="utf-8").strip())
    accepted = json.loads(accepted_path.read_text(encoding="utf-8"))

    assert row["replay_correlation_ids"] == ["corr::offline::1"]
    assert accepted[0]["parameters"]["replay_correlation_id"] == "corr::offline::1"


def test_decision_layer_report_writer_emits_replay_correlation_ids(tmp_path):
    class _Action:
        def to_dict(self):
            return {
                "action_type": "gain_change",
                "target": "Lead Vox",
                "parameters": {"gain_db": -1.0, "replay_correlation_id": "corr::decision-layer::1"},
            }

    class _Candidate:
        candidate_id = "candidate-1"
        description = "Candidate 1"
        actions = [_Action()]
        is_no_change = False

        def to_dict(self):
            return {
                "candidate_id": self.candidate_id,
                "description": self.description,
                "actions": [self.actions[0].to_dict()],
            }

    paths = write_decision_layer_reports(
        tmp_path,
        run_id="run-1",
        candidates=[_Candidate()],
        render_results={"candidate-1": {"path": "renders/candidate-1.wav"}},
        critic_scores={"candidate-1": {}},
        safety_results={"candidate-1": {"passed": True}},
        decision={
            "selected_candidate_id": "candidate-1",
            "final_scores": {"candidate-1": 0.75},
            "reason": "selected",
            "decision": "accept",
        },
        optimizer_status={},
        mixer_status={},
        dependency_status={},
        module_status={},
    )

    row = json.loads((tmp_path / "decision_log.jsonl").read_text(encoding="utf-8").strip())
    accepted = json.loads((tmp_path / "accepted_actions.json").read_text(encoding="utf-8"))
    summary = (tmp_path / "summary_report.md").read_text(encoding="utf-8")

    assert paths["decision_log"].endswith("decision_log.jsonl")
    assert row["replay_correlation_ids"] == ["corr::decision-layer::1"]
    assert accepted[0]["parameters"]["replay_correlation_id"] == "corr::decision-layer::1"
    assert "corr::decision-layer::1" in summary
