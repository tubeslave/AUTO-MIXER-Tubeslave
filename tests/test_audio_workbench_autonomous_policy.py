from audio_workbench.autonomous_policy import autonomous_first_policy,dense_guitar_guard,calibration_update

def test_autonomous_first_excludes_external_targets_initially():
    p=autonomous_first_policy()
    assert "reference_targets" in p["initially_forbidden"]
    assert p["stages"].index("self_critique") < p["stages"].index("optional_external_constraints")

def test_dense_guitar_guard_is_bounded():
    r=dense_guitar_guard(-2.0,.8,True)
    assert r["risk"] and r["suggested_delta_db"]==-2.0
    assert r["max_auto_delta_db"]==-2.0

def test_single_song_learning_is_not_universal():
    r=calibration_update({"preferred_delta_db":-4.0})
    assert r["generalization"]=="candidate_prior_only_until_repeated_across_songs"
