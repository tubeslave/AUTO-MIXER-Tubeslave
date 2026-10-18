from audio_workbench.judge_calibration import authority_map, filter_verdict
from audio_workbench.observer_ensemble import evaluate

def test_failed_capability_has_no_authority():
    trials=[{"capability":"clarity","correct":False},{"capability":"clarity","correct":True}]
    a=authority_map(trials)
    assert not a["capabilities"]["clarity"]["enabled"]
    v=filter_verdict({"preference":"A","confidence":.9,"axes":{"clarity":"A"}},a)
    assert "clarity" not in v["axes"]

def test_good_capability_can_pass():
    trials=[{"capability":"dynamics","correct":True} for _ in range(5)]
    a=authority_map(trials)
    assert a["capabilities"]["dynamics"]["enabled"]

def test_disagreement_is_visible():
    r=evaluate([{"preference":"A","confidence":.9,"axes":{}},
                {"preference":"B","confidence":.9,"axes":{}}])
    assert r["disagreement"]==1.0
    assert r["aggregate"]["preference"]=="uncertain"
