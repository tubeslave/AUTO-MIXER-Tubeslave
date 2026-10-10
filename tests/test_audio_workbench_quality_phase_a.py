import pytest
from audio_workbench.song_model import build_hierarchy, section_priorities, contributors
from audio_workbench.causal import make_plan, may_auto_accept
from audio_workbench.quality_loop import evaluate_candidate, advance_perceptual_iteration


def manifest():
    return {"tracks":[
      {"name":"Kick In","path":"k.wav","role_guess":"kick"},
      {"name":"Lead Vocal","path":"v.wav","role_guess":"vocal"},
      {"name":"GTR L","path":"g.wav","role_guess":"guitar"}],
      "sections":[{"name":"solo","role_priorities":{"GTR L":1.0,"Lead Vocal":.2}}]}


def confident_plan(**extra):
    p=make_plan("vocal masked","guitar overlap","GTR L",
       [{"type":"eq_bell","params":{"freq_hz":3000,"db":-1,"q":1}}],
       "improve vocal clarity",["guitar_body","chorus_energy"],
       {"cause":.8,"intervention":.8})
    p.update(extra)
    return p


def critic(decision="machine_safe", *, human=True, failures=None, regressions=None):
    return {
        "machine_decision":decision,
        "requires_human_listening":human,
        "failures":list(failures or []),
        "protected_regressions":list(regressions or []),
        "evidence":[{"metric":"punch_db","role":"target","passed":True}],
        "uncertainty":{"score":.22,"reasons":["target:punch_db"]},
    }


def test_hierarchy_and_section_override():
    h=build_hierarchy(manifest())
    assert h["groups"]["drums"][0]["name"]=="Kick In"
    p=section_priorities(manifest(),"solo")
    assert p["GTR L"]==1.0 and p["Lead Vocal"]==.2


def test_contributor_attribution():
    h=build_hierarchy(manifest())
    c=contributors({"Kick In":2,"Lead Vocal":1,"GTR L":7},h)
    assert c["groups"]["guitars"]["share"]==pytest.approx(.7)


def test_causal_plan_requires_no_change_and_confidence_gate():
    p=confident_plan()
    assert p["candidates"][0]["label"]=="no_change"
    assert not may_auto_accept(p,True,["chorus_energy"],.9)["allowed"]
    assert may_auto_accept(p,True,[],.9)["allowed"]


def test_subjective_candidate_stays_pending_after_machine_gate():
    result=evaluate_candidate(
        confident_plan(),
        {"id":"model-cleanup","requires_human_review":True},
        True,[],.9,
    )
    assert result["gate"]["allowed"]
    assert not result["accepted"]
    assert result["acceptance_state"]=="pending_human_review"
    assert result["human_review_status"]=="pending"


def test_subjective_candidate_accepts_only_after_human_and_machine_pass():
    result=evaluate_candidate(
        confident_plan(),
        {"id":"model-cleanup","requires_human_review":True,
         "human_review":{"status":"accepted"}},
        True,[],.9,
    )
    assert result["accepted"]
    assert result["acceptance_state"]=="accepted"

    failed_machine=evaluate_candidate(
        confident_plan(),
        {"id":"model-cleanup","requires_human_review":True,
         "human_review":{"status":"accepted"}},
        True,["vocal_timbre"],.9,
    )
    assert not failed_machine["accepted"]
    assert failed_machine["acceptance_state"]=="machine_gate_failed"


def test_subjective_candidate_human_reject_is_terminal_for_candidate():
    result=evaluate_candidate(
        confident_plan(),
        {"id":"model-cleanup","requires_human_review":True,
         "human_review":"rejected"},
        True,[],.9,
    )
    assert result["gate"]["allowed"]
    assert not result["accepted"]
    assert result["acceptance_state"]=="human_rejected"


def test_non_subjective_candidate_keeps_machine_gate_behavior():
    result=evaluate_candidate(confident_plan(),{"id":"eq-candidate"},True,[],.9)
    assert result["accepted"]
    assert result["acceptance_state"]=="accepted"
    assert result["human_review_status"]=="not_required"


def test_iteration_pending_keeps_baseline_and_persists_critic_evidence():
    result=advance_perceptual_iteration(
        confident_plan(), {"id":"candidate-2"},
        critic("pending_human_review"), "baseline-1", .9,
    )
    assert not result["accepted"]
    assert result["acceptance_state"]=="pending_human_review"
    assert result["baseline_before"]==result["baseline_after"]=="baseline-1"
    assert not result["promote_baseline"]
    assert not result["rollback_candidate"]
    assert result["evidence"][0]["metric"]=="punch_db"
    assert result["uncertainty"]["score"]==pytest.approx(.22)


def test_iteration_rejected_vetoes_human_accept_and_rolls_back():
    result=advance_perceptual_iteration(
        confident_plan(),
        {"id":"candidate-2","human_review":"accepted"},
        critic("rejected", failures=["width_regression"], regressions=["width_regression"]),
        "baseline-1", .9,
    )
    assert not result["accepted"]
    assert result["acceptance_state"]=="critic_rejected"
    assert result["baseline_after"]=="baseline-1"
    assert result["rollback_candidate"]
    assert result["protected_regressions"]==["width_regression"]


def test_iteration_machine_safe_subjective_waits_for_listening_then_promotes():
    pending=advance_perceptual_iteration(
        confident_plan(), {"id":"candidate-2"},
        critic("machine_safe", human=True), "baseline-1", .9,
    )
    assert pending["acceptance_state"]=="pending_human_review"
    assert pending["baseline_after"]=="baseline-1"

    accepted=advance_perceptual_iteration(
        confident_plan(),
        {"id":"candidate-2","human_review":{"status":"accepted"}},
        critic("machine_safe", human=True), "baseline-1", .9,
    )
    assert accepted["accepted"]
    assert accepted["acceptance_state"]=="accepted"
    assert accepted["baseline_after"]=="candidate-2"
    assert accepted["promote_baseline"]


def test_iteration_pending_can_be_resolved_by_explicit_listening_acceptance():
    result=advance_perceptual_iteration(
        confident_plan(),
        {"id":"candidate-2","human_review":"accepted"},
        critic("pending_human_review", human=True), "baseline-1", .9,
    )
    assert result["accepted"]
    assert result["baseline_after"]=="candidate-2"


def test_iteration_objective_machine_safe_preserves_auto_accept_behavior():
    result=advance_perceptual_iteration(
        confident_plan(), {"id":"candidate-2"},
        critic("machine_safe", human=False), "baseline-1", .9,
    )
    assert result["accepted"]
    assert result["human_review_status"]=="not_required"
    assert result["baseline_after"]=="candidate-2"


def test_bad_plan_rejected():
    with pytest.raises(ValueError):
        make_plan("","","x",[],"",[],{})
