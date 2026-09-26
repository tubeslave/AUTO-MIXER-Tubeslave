from audio_workbench.perceptual import aggregate, catch_trial_result
from audio_workbench.uncertainty import assess
from audio_workbench.preference_memory import remember, retrieve

def test_identical_catch_trial_blocks_confident_preference():
    r=catch_trial_result("same","same",{"preference":"A","confidence":.9,"axes":{}})
    assert not r["passed"]

def test_judge_can_be_uncertain():
    r=aggregate([{"preference":"tie","confidence":.9,"axes":{}},
                 {"preference":"A","confidence":.4,"axes":{}}])
    assert r["preference"]=="uncertain"

def test_uncertainty_routes_low_cause_to_diagnostic():
    r=assess({"observation":.9,"cause":.5,"intervention":.8,"evaluation":.9})
    assert r["next_action"]=="diagnostic_experiment"

def test_observer_disagreement_reduces_evaluation():
    r=assess({"observation":.9,"cause":.9,"intervention":.9,"evaluation":.9},.8)
    assert r["confidence"]["evaluation_adjusted"] < .75

def test_memory_keeps_rejected_decisions(tmp_path):
    remember(str(tmp_path),"project:premiera",{"section":"chorus"},{"type":"eq"},"rejected","lost body")
    rows=retrieve(str(tmp_path),"project:premiera")
    assert rows[0]["result"]=="rejected"
