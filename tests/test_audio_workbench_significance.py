from audio_workbench.significance import significance_gate, update_problem_after_tie

def test_tiny_change_is_not_meaningful():
    assert not significance_gate(-65,.13)["meaningful"]

def test_human_tie_deprioritizes_even_larger_change():
    assert not significance_gate(-35,.5,"tie")["meaningful"]

def test_tie_updates_problem():
    p=update_problem_after_tie({"expected_impact":.8,"uncertainty":.1})
    assert p["expected_impact"]<=.2 and p["uncertainty"]>=.7
