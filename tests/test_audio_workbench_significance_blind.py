from audio_workbench.significance import assess_blind_trials, effect_budget

def test_three_baseline_votes_reject_family():
    key=[{"A":"stronger","B":"baseline","C":"moderate"} for _ in range(3)]
    r=assess_blind_trials(["B","B","B"],key,{"moderate","stronger"})
    assert r["verdict"]=="reject_intervention_family"
    assert effect_budget([r])["next_action"]=="change_problem_or_intervention_family"

def test_mixed_does_not_fake_winner():
    key=[{"A":"baseline","B":"moderate"} for _ in range(3)]
    r=assess_blind_trials(["A","B","tie"],key,{"moderate"})
    assert r["verdict"]=="insufficient_or_mixed"
