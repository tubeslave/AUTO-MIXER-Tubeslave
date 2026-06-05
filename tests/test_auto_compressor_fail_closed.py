from pathlib import Path

from auto_compressor import AutoCompressorController
from auto_compressor_cf import AutoCFCompressorController, CFClass, CFCompressorParams


class FakeCompressorMixer:
    is_connected = True

    def __init__(self):
        self.calls = []

    def get_compressor_on(self, channel):
        self.calls.append(("get_compressor_on", channel))
        return 0

    def set_compressor_on(self, *args, **kwargs):
        self.calls.append(("set_compressor_on", args, kwargs))

    def set_compressor(self, *args, **kwargs):
        self.calls.append(("set_compressor", args, kwargs))


def test_auto_compressor_smooth_apply_is_fail_closed_evidence():
    mixer = FakeCompressorMixer()
    controller = AutoCompressorController(mixer_client=mixer)

    result = controller._apply_compressor_smooth(
        3,
        {
            "threshold": -18.5,
            "ratio_wing": "3.0",
            "attack_ms": 12.0,
            "release_ms": 180.0,
            "knee": 3.0,
            "gain": 1.5,
        },
    )

    assert result["applied"] is False
    assert result["blocked"] is True
    assert result["calls"] == []
    assert result["reason"] == "auto_compressor_direct_apply_quarantined"
    assert result["write_policy"] == "proposal_only_quarantined"
    assert result["live_write_allowed"] is False
    assert result["can_apply_live"] is False
    assert result["mixer_client_present"] is True
    assert result["mutates_mixer"] is False
    assert mixer.calls == []
    assert controller.current_params[3]["threshold"] == -18.5
    assert {
        "cmd": "set_compressor_on",
        "channel": 3,
        "on": 1,
        "semantic": "enable_before_compressor_params",
    } in result["blocked_calls"]
    assert {
        "cmd": "set_compressor",
        "channel": 3,
        "threshold": -18.5,
        "ratio": "3.0",
        "attack": 12.0,
        "release": 180.0,
        "knee": 3,
        "gain": 1.5,
        "semantic": "apply_compressor_params",
    } in result["blocked_calls"]


def test_auto_compressor_on_toggle_is_fail_closed():
    mixer = FakeCompressorMixer()
    controller = AutoCompressorController(mixer_client=mixer)

    result = controller._block_compressor_on(4, 0, "soundcheck_clean_analysis_disable")

    assert result["applied"] is False
    assert result["blocked"] is True
    assert result["reason"] == "auto_compressor_direct_apply_quarantined"
    assert mixer.calls == []
    assert result["blocked_calls"] == [
        {
            "cmd": "set_compressor_on",
            "channel": 4,
            "on": 0,
            "semantic": "soundcheck_clean_analysis_disable",
        }
    ]


def test_auto_cf_compressor_apply_params_is_fail_closed_evidence():
    mixer = FakeCompressorMixer()
    controller = AutoCFCompressorController(mixer_client=mixer)
    params = CFCompressorParams(
        attack_ms=8.0,
        release_ms=110.0,
        ratio=4.0,
        threshold_db=-21.0,
        knee_db=2.0,
        makeup_gain_db=1.0,
        cf_class=CFClass.DRUMS,
        cf_db=14.0,
    )

    result = controller.apply_params(5, params)

    assert result["applied"] is False
    assert result["blocked"] is True
    assert result["calls"] == []
    assert result["reason"] == "auto_cf_compressor_direct_apply_quarantined"
    assert result["write_policy"] == "proposal_only_quarantined"
    assert result["mutates_mixer"] is False
    assert mixer.calls == []
    assert {
        "cmd": "set_compressor_on",
        "channel": 5,
        "on": 1,
        "semantic": "enable_before_cf_params",
    } in result["blocked_calls"]
    assert {
        "cmd": "set_compressor",
        "channel": 5,
        "threshold": -21.0,
        "ratio": "4.0",
        "attack": 8.0,
        "release": 110.0,
        "knee": 2,
        "gain": 1.0,
        "cf_db": 14.0,
        "cf_class": "drums",
        "semantic": "apply_cf_compressor_params",
    } in result["blocked_calls"]


def test_auto_compressor_modules_have_no_direct_mixer_compressor_writes():
    for source_path in ("backend/auto_compressor.py", "backend/auto_compressor_cf.py"):
        source = Path(source_path).read_text()

        assert "mixer_client.set_compressor(" not in source
        assert "mixer_client.set_compressor_on(" not in source
        assert '"cmd": "set_compressor"' in source
        assert '"cmd": "set_compressor_on"' in source
