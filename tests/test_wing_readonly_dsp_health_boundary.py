from smart_dsp_strategy_engine import (
    WING_DYNAMICS_THRESHOLD_PATH_SUFFIX_BY_SURFACE,
    WING_GATE_THRESHOLD_PATH_SUFFIX_BY_SURFACE,
)
from tools import wing_readonly_preflight_runner


def test_readonly_dsp_health_aliases_do_not_widen_apply_paths():
    assert (
        wing_readonly_preflight_runner.address_for(3, "dyn_threshold")
        == "/ch/3/dyn/thr"
    )
    assert (
        wing_readonly_preflight_runner.address_for(3, "gate_threshold")
        == "/ch/3/gate/thr"
    )
    assert WING_DYNAMICS_THRESHOLD_PATH_SUFFIX_BY_SURFACE["channel"] == "thr24"
    assert WING_GATE_THRESHOLD_PATH_SUFFIX_BY_SURFACE["channel"] == "thr21"
