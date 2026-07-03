from ai_mixing_pipeline.decision_engine import DecisionEngine
from ai_mixing_pipeline.decision_layer.action_schema import CandidateActionSet, GainAction, NoChangeAction
from ai_mixing_pipeline.decision_layer.decision_engine import CorrectionDecisionEngine
from ai_mixing_pipeline.models import MixCandidate, SafetyResult


def _candidate(candidate_id):
    return MixCandidate(candidate_id=candidate_id, label=candidate_id)


def _critic(delta, confidence=0.5):
    return {
        "critic_name": "muq_eval",
        "role": "chief_music_critic",
        "scores": {"overall": 0.5 + delta},
        "delta": {"overall": delta},
        "confidence": confidence,
        "warnings": [],
        "explanation": "",
        "model_available": False,
    }


def test_decision_engine_selects_best_safe_candidate_and_renormalizes():
    candidates = [_candidate("000_initial_mix"), _candidate("001_candidate_gain_balance")]
    evaluations = {
        "000_initial_mix": {"muq_eval": _critic(0.0)},
        "001_candidate_gain_balance": {"muq_eval": _critic(0.2)},
    }
    safety = {
        "000_initial_mix": SafetyResult("000_initial_mix", True, 1.0),
        "001_candidate_gain_balance": SafetyResult("001_candidate_gain_balance", True, 1.0),
    }
    engine = DecisionEngine(
        {
            "critics": {"muq_eval": {"enabled": True, "weight": 0.30}},
            "safety": {"weight": 0.05, "min_score_improvement": 0.01},
        }
    )

    result = engine.choose_best(candidates, evaluations, safety)

    assert result.selected_candidate_id == "001_candidate_gain_balance"
    assert set(result.normalized_weights) == {"muq_eval", "safety"}
    assert abs(sum(result.normalized_weights.values()) - 1.0) < 1e-9


def test_decision_engine_keeps_no_change_when_improvement_is_too_small():
    candidates = [_candidate("000_initial_mix"), _candidate("001_candidate_gain_balance")]
    evaluations = {
        "000_initial_mix": {"muq_eval": _critic(0.0)},
        "001_candidate_gain_balance": {"muq_eval": _critic(0.001)},
    }
    safety = {
        "000_initial_mix": SafetyResult("000_initial_mix", True, 1.0),
        "001_candidate_gain_balance": SafetyResult("001_candidate_gain_balance", True, 1.0),
    }
    result = DecisionEngine(
        {
            "critics": {"muq_eval": {"enabled": True, "weight": 0.30}},
            "safety": {"weight": 0.05, "min_score_improvement": 0.03},
        }
    ).choose_best(candidates, evaluations, safety)

    assert result.selected_candidate_id == "000_initial_mix"
    assert result.no_change_selected is True


def test_correction_decision_engine_selects_best_safe_candidate():
    candidates = [
        CandidateActionSet("candidate_000_no_change", [NoChangeAction()]),
        CandidateActionSet("candidate_001_vocal_up", [GainAction("vocal", 0.5)]),
    ]
    critics = {
        "candidate_000_no_change": {"muq_eval": {"delta": {"overall": 0.0}, "confidence": 0.5}},
        "candidate_001_vocal_up": {"muq_eval": {"delta": {"overall": 0.2}, "confidence": 0.5}},
    }
    safety = {
        "candidate_000_no_change": {"passed": True, "safety_score": 1.0},
        "candidate_001_vocal_up": {"passed": True, "safety_score": 1.0},
    }

    decision = CorrectionDecisionEngine(
        {"critics": {"muq_eval": {"enabled": True, "weight": 0.30}}, "safety": {"min_score_improvement": 0.01}}
    ).choose_best("run", candidates, critics, safety)

    assert decision["selected_candidate_id"] == "candidate_001_vocal_up"
    assert decision["decision"] == "accept"


def test_correction_decision_engine_keeps_no_change_for_small_improvement():
    candidates = [
        CandidateActionSet("candidate_000_no_change", [NoChangeAction()]),
        CandidateActionSet("candidate_001_vocal_up", [GainAction("vocal", 0.5)]),
    ]
    critics = {
        "candidate_000_no_change": {"muq_eval": {"delta": {"overall": 0.0}, "confidence": 0.5}},
        "candidate_001_vocal_up": {"muq_eval": {"delta": {"overall": 0.001}, "confidence": 0.5}},
    }
    safety = {
        "candidate_000_no_change": {"passed": True, "safety_score": 1.0},
        "candidate_001_vocal_up": {"passed": True, "safety_score": 1.0},
    }

    decision = CorrectionDecisionEngine({"safety": {"min_score_improvement": 0.03}}).choose_best(
        "run", candidates, critics, safety
    )

    assert decision["selected_candidate_id"] == "candidate_000_no_change"
    assert decision["decision"] == "no_change"


def test_correction_decision_engine_downweights_proxy_scores():
    candidates = [
        CandidateActionSet("candidate_000_no_change", [NoChangeAction()]),
        CandidateActionSet("candidate_001_proxy_up", [GainAction("vocal", 0.5)]),
    ]
    critics = {
        "candidate_000_no_change": {
            "muq_eval": {"delta": {"overall": 0.0}, "confidence": 0.25, "score_source": "proxy", "model_available": False}
        },
        "candidate_001_proxy_up": {
            "muq_eval": {"delta": {"overall": 0.2}, "confidence": 0.25, "score_source": "proxy", "model_available": False}
        },
    }
    safety = {
        "candidate_000_no_change": {"passed": True, "safety_score": 1.0},
        "candidate_001_proxy_up": {"passed": True, "safety_score": 1.0},
    }

    decision = CorrectionDecisionEngine(
        {
            "critics": {"muq_eval": {"enabled": True, "weight": 0.30}},
            "safety": {"min_score_improvement": 0.01},
            "proxy_weight_multiplier": 0.25,
        }
    ).choose_best("run", candidates, critics, safety)

    breakdown = decision["critic_breakdown"]["candidate_001_proxy_up"]
    assert breakdown["weight_sources"]["muq_eval"] == "proxy"
    assert breakdown["effective_weights"]["muq_eval"] == 0.075
    assert breakdown["normalized_weights"]["muq_eval"] < 0.30 / (0.30 + 0.05)
