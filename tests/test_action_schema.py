import json

from ai_mixing_pipeline.decision_layer.action_schema import (
    CandidateActionSet,
    CompressorAction,
    EQAction,
    FXSendAction,
    GainAction,
    GateExpanderAction,
    NoChangeAction,
    PanAction,
    ensure_no_change_candidate,
)


def test_decision_layer_actions_serialize_to_json():
    candidate = CandidateActionSet(
        candidate_id="candidate_test",
        actions=[
            GainAction("vocal", 0.5),
            EQAction("vocal", "presence", 3500.0, -1.0, 1.2, "peaking"),
            CompressorAction("drums", -18.0, 1.4, 15.0, 140.0, 0.0),
            GateExpanderAction("snare", -50.0, 2.0, 5.0, 120.0),
            PanAction("guitar_l", -0.2),
            FXSendAction("vocal", "reverb", -0.5),
        ],
        description="schema smoke test",
        source="manual_rule",
    )

    encoded = json.dumps(candidate.to_dict())

    assert "candidate_test" in encoded
    assert "gain" in encoded


def test_no_change_action_is_valid_and_forced_first():
    candidates = ensure_no_change_candidate([CandidateActionSet("candidate_001", [GainAction("vocal", 0.5)])])

    assert candidates[0].is_no_change is True
    assert isinstance(candidates[0].actions[0], NoChangeAction)
