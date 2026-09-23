from audio_workbench.autonomous_loop import next_iteration, resolve_candidate_iteration, stop_decision


def test_loop_stops_without_problem():
    assert next_iteration([],{"observation":1,"cause":1,"intervention":1,"evaluation":1})["status"]=="stop"


def test_loop_routes_uncertainty():
    r=next_iteration([{"name":"mask","importance":.9,"confidence":.9,"expected_impact":.8}],
                     {"observation":.9,"cause":.4,"intervention":.8,"evaluation":.8})
    assert r["next_action"]=="diagnostic_experiment"


def _plan():
    return {"confidence":{"cause":.9,"intervention":.9}}


def _critic(decision, *, human=True, regressions=None):
    regressions=list(regressions or [])
    return {
        "machine_decision":decision,
        "requires_human_listening":human,
        "failures":regressions,
        "protected_regressions":regressions,
        "evidence":[{"metric":"punch_db","role":"target","passed":True}],
        "uncertainty":{"score":.2,"reasons":[]},
    }


def test_autonomous_iteration_routes_pending_to_human_without_baseline_change():
    r=resolve_candidate_iteration(
        _plan(),{"id":"candidate-2"},_critic("pending_human_review"),
        "baseline-1",.9,
    )
    assert r["status"]=="pending_human_review"
    assert r["next_action"]=="human_listening"
    assert r["baseline_after"]=="baseline-1"
    assert not r["promote_baseline"]


def test_autonomous_iteration_routes_critic_rejection_to_rollback():
    r=resolve_candidate_iteration(
        _plan(),{"id":"candidate-2","human_review":"accepted"},
        _critic("rejected",regressions=["width_regression"]),
        "baseline-1",.9,
    )
    assert r["status"]=="rejected"
    assert r["next_action"]=="rollback_candidate"
    assert r["baseline_after"]=="baseline-1"
    assert r["rollback_candidate"]


def test_autonomous_iteration_promotes_only_after_required_listening_acceptance():
    pending=resolve_candidate_iteration(
        _plan(),{"id":"candidate-2"},_critic("machine_safe",human=True),
        "baseline-1",.9,
    )
    assert pending["next_action"]=="human_listening"
    assert pending["baseline_after"]=="baseline-1"

    accepted=resolve_candidate_iteration(
        _plan(),{"id":"candidate-2","human_review":"accepted"},
        _critic("machine_safe",human=True),"baseline-1",.9,
    )
    assert accepted["status"]=="accepted"
    assert accepted["next_action"]=="promote_baseline"
    assert accepted["baseline_after"]=="candidate-2"


def test_stop_after_repeated_failed_experiments():
    p=[{"importance":.9,"confidence":.9}]
    assert stop_decision(p,1,3)["stop"]
