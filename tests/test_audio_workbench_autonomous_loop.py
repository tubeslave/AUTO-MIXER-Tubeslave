from audio_workbench.autonomous_loop import next_iteration, stop_decision

def test_loop_stops_without_problem():
    assert next_iteration([],{"observation":1,"cause":1,"intervention":1,"evaluation":1})["status"]=="stop"

def test_loop_routes_uncertainty():
    r=next_iteration([{"name":"mask","importance":.9,"confidence":.9,"expected_impact":.8}],
                     {"observation":.9,"cause":.4,"intervention":.8,"evaluation":.8})
    assert r["next_action"]=="diagnostic_experiment"

def test_stop_after_repeated_failed_experiments():
    p=[{"importance":.9,"confidence":.9}]
    assert stop_decision(p,1,3)["stop"]
