import numpy as np

from audio_workbench.mastering import (
    MasteringConfig,
    MasteringDirector,
    MasteringTargetController,
    MasteringTargetSearchConfig,
)


def tone(sr=48000, seconds=1.0, amplitude=0.03):
    t=np.arange(int(sr*seconds),dtype=np.float64)/sr
    mono=(amplitude*np.sin(2*np.pi*997*t)).astype("float32")
    return np.column_stack([mono,mono]).astype("float32")


def minimal_config(**changes):
    config=MasteringConfig(
        stabilizer=False,
        clarity=False,
        impact=False,
        clipper=False,
        maximizer=True,
        ceiling_db=-1.0,
        max_final_limiter_gr_db=3.0,
        max_band_limiter_gr_db=4.0,
    )
    for key,value in changes.items():
        setattr(config,key,value)
    return config


def test_pregain_is_auditable_and_does_not_mutate_input():
    x=tone();original=x.copy()
    config=MasteringConfig(stabilizer=False,clarity=False,impact=False,clipper=False,maximizer=False,pregain_db=6.0)
    y,report=MasteringDirector(config).render(x,48000)
    np.testing.assert_array_equal(x,original)
    assert [event["module"] for event in report["events"]]==["pregain"]
    assert report["events"][0]["gain_db"]==6.0
    assert report["before"]["rms_dbfs"] < report["after"]["rms_dbfs"]-5.9
    assert y.shape==x.shape


def test_target_controller_finds_machine_safe_candidate_but_never_promotes_baseline():
    x=tone(amplitude=.03)
    known_config=minimal_config(pregain_db=4.0,maximizer_drive_db=1.0)
    _,known_report=MasteringDirector(known_config).render(x,48000)
    target=float(known_report["after"]["integrated_lufs"])

    controller=MasteringTargetController(
        base_config=minimal_config(),
        search_config=MasteringTargetSearchConfig(
            target_lufs=target,
            tolerance_lu=.08,
            pregain_grid_db=(0.0,2.0,4.0,6.0),
            maximizer_drive_grid_db=(0.0,1.0,2.0),
            max_candidates=16,
        ),
    )
    y,result=controller.search(x,48000)
    chosen=result["chosen"]
    assert chosen is not None
    assert chosen["machine_safe"] is True
    assert chosen["safety"]["evidence"]["true_peak_dbtp"] <= -0.95
    assert chosen["safety"]["evidence"]["final_limiter_gr_db"] <= 3.0
    assert chosen["safety"]["evidence"]["worst_band_limiter_gr_db"] <= 4.0
    assert abs(chosen["safety"]["evidence"]["loudness_error_lu"]) <= .08
    assert result["status"]=="pending_human_review"
    assert result["requires_human_listening"] is True
    assert result["baseline_eligible"] is False
    assert result["rolled_back_to_source"] is False
    assert y.shape==x.shape


def test_target_controller_rolls_back_when_no_candidate_is_safe():
    x=tone(amplitude=.02);original=x.copy()
    controller=MasteringTargetController(
        base_config=minimal_config(),
        search_config=MasteringTargetSearchConfig(
            target_lufs=-6.0,
            tolerance_lu=.05,
            pregain_grid_db=(0.0,),
            maximizer_drive_grid_db=(0.0,),
            max_candidates=1,
        ),
    )
    y,result=controller.search(x,48000)
    assert result["status"]=="rejected"
    assert result["chosen"] is None
    assert result["rolled_back_to_source"] is True
    assert result["baseline_eligible"] is False
    np.testing.assert_array_equal(y,original)


def test_target_controller_rejects_unbounded_grid():
    config=MasteringTargetSearchConfig(
        pregain_grid_db=(0.0,1.0,2.0),
        maximizer_drive_grid_db=(0.0,1.0),
        max_candidates=5,
    )
    try:
        MasteringTargetController(minimal_config(),config)
    except ValueError as exc:
        assert "max_candidates" in str(exc)
    else:
        raise AssertionError("expected bounded-search guard to fail")
