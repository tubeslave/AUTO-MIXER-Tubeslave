"""Executable safety/DSP contracts; no console, network or model downloads."""
from copy import deepcopy
from dataclasses import asdict
import json
import math
from types import SimpleNamespace

import numpy as np
import pytest

from autofoh_models import RuntimeState
from autofoh_safety import (
    AutoFOHSafetyConfig, AutoFOHSafetyController, ChannelEQMove, ChannelFaderMove,
    CompressorAdjust, HighPassAdjust, MasterFaderMove,
)
from decision_guard import DecisionGuard, GuardConfig, ReviewDecision
from shadow_renderer import Compressor, EQBand, RenderChannel, ShadowRenderer, measure
from shadow_pipeline import ShadowPipeline, live_states
from target_corridor import FEATURE_VERSION, TargetCorridor

SR = 48000


def audio(phase=0.0, seconds=1):
    t = np.arange(int(SR * seconds)) / SR
    return 0.03 * np.sin(2 * np.pi * 1000 * t + phase)


def state():
    return RenderChannel(1, fader_db=-6, eq_bands={i: EQBand(1000, 0, 1) for i in range(1, 5)})


def corridor(lufs=-40):
    return TargetCorridor("rock:chorus", {"lufs": (lufs - 0.5, lufs, lufs + 0.5)}, 7)


def rows():
    return [{"mix_id": str(i), "approved": True, "context": "rock:chorus",
             "feature_version": FEATURE_VERSION, "features": {"lufs": -20 + i / 100}}
            for i in range(9)]


def test_corridor_robust_outliers_and_explicit_approval(tmp_path):
    data = rows()
    data += [dict(data[0], mix_id="outlier", features={"lufs": 1e6}),
             dict(data[0], mix_id="not-approved", approved=False),
             dict(data[0], mix_id="wrong-context", context="folk:verse"), data[0]]
    fitted = TargetCorridor.fit(data, context="rock:chorus", features=["lufs"])
    assert fitted.sample_count == 9
    assert fitted.rejected_count == 4
    assert abs(fitted.bands["lufs"][1] + 19.96) < 1e-9
    path = tmp_path / "corridor.json"
    fitted.save(path)
    assert asdict(TargetCorridor.load(path)) == asdict(fitted)
    assert fitted.loss({"lufs": -20}, context=fitted.context) == 0
    assert fitted.loss({"lufs": -23}, context=fitted.context) > 0


@pytest.mark.parametrize("bad", [float("nan"), float("inf"), -float("inf")])
def test_corridor_excludes_nonfinite_rows(bad):
    data = rows() + [dict(rows()[0], mix_id="bad", features={"lufs": bad})]
    fitted = TargetCorridor.fit(data, context="rock:chorus", features=["lufs"])
    assert fitted.rejected_count == 1
    with pytest.raises(ValueError):
        fitted.loss({"lufs": bad}, context=fitted.context)


@pytest.mark.parametrize("data", [[], rows()[:4], [rows()[0]] * 8])
def test_corridor_requires_five_distinct_approved_mixes(data):
    with pytest.raises(ValueError):
        TargetCorridor.fit(data, context="rock:chorus", features=["lufs"])


def test_corridor_rejects_wrong_context_schema_and_missing_features():
    with pytest.raises(ValueError):
        corridor().loss({"lufs": -40}, context="different")
    with pytest.raises(KeyError):
        corridor().loss({}, context="rock:chorus")
    with pytest.raises(ValueError):
        TargetCorridor("x", {"lufs": (-3, -2, -1)}, 9, feature_version="old")


def review(guard, *, frame="a", now=10.0, delta=-2, before=1, after=0.2, unc=0.05):
    return guard.review(("fader", 1), changes={"fader": delta}, before=before, after=after,
                        uncertainty=unc, frame_id=frame, now=now)


def test_guard_requires_distinct_frames_and_commit_for_applied_history():
    guard = DecisionGuard()
    assert review(guard).reason == "awaiting_fresh_confirmation"
    assert review(guard, now=11).reason == "awaiting_fresh_confirmation"
    accepted = review(guard, frame="b", now=12)
    assert accepted.allowed and not guard._applied
    accepted.commit()
    snapshot = deepcopy(guard._applied)
    accepted.commit()
    assert guard._applied == snapshot
    assert review(guard, delta=2, frame="c", now=13).reason == "reversal_hold"


@pytest.mark.parametrize("kwargs,reason", [
    ({"delta": .99}, "deadband"),
    ({"after": 1.1}, "no_significant_improvement"),
    ({"after": 1}, "no_significant_improvement"),
    ({"after": .99}, "no_significant_improvement"),
    ({"unc": 0.6}, "uncertainty_too_high"),
    ({"before": float("nan")}, "invalid_evidence"),
    ({"delta": float("inf")}, "invalid_evidence"),
    ({"now": float("nan")}, "invalid_evidence"),
])
def test_guard_fail_closed(kwargs, reason):
    assert review(DecisionGuard(), **kwargs).reason == reason


def test_guard_reversal_hysteresis_after_hold_and_clock_regression():
    guard = DecisionGuard(GuardConfig(confirmations=1))
    accepted = review(guard)
    accepted.commit()
    assert review(guard, now=9).reason == "invalid_evidence"
    assert review(guard, now=30, delta=2, before=.13, after=.05, unc=.05).reason == "reversal_hysteresis"
    assert review(guard, now=31, delta=2).allowed


def test_guard_memory_is_bounded():
    guard = DecisionGuard(GuardConfig(confirmations=1, max_keys=5))
    for i in range(20):
        result = guard.review((i,), changes={"gain": 2}, before=1, after=.2,
                              uncertainty=.01, frame_id=str(i), now=float(i))
        result.commit()
    assert len(guard._applied) <= 5 and len(guard._pending) <= 5


def test_renderer_noop_and_input_ownership():
    buffers, states = {1: audio()}, {1: state()}
    original = buffers[1].copy()
    result = ShadowRenderer(SR).render(buffers, states)
    expected = original * 10 ** (-6 / 20) / math.sqrt(2)
    np.testing.assert_allclose(result.audio[:, 0], expected, atol=1e-14)
    np.testing.assert_array_equal(buffers[1], original)
    assert states[1].fader_db == -6
    assert np.isfinite(list(result.features.values())).all()


@pytest.mark.parametrize("bad", [float("nan"), float("inf"), -float("inf")])
def test_renderer_refuses_nonfinite_audio(bad):
    x = audio(); x[42] = bad
    with pytest.raises(ValueError):
        ShadowRenderer(SR).render({1: x}, {1: state()})


@pytest.mark.parametrize("bad", [np.zeros((2, SR)), np.zeros(10), np.ones(SR) * 1.01])
def test_renderer_refuses_ambiguous_short_or_overrange_audio(bad):
    with pytest.raises(ValueError):
        ShadowRenderer(SR).render({1: bad}, {1: state()})


def test_renderer_no_implicit_alignment():
    with pytest.raises(ValueError):
        ShadowRenderer(SR).render({1: audio(), 2: audio()[:-1]},
                                  {1: state(), 2: RenderChannel(2)})


def test_renderer_eq_is_actual_audio_and_replaces_physical_slot():
    renderer = ShadowRenderer(SR)
    states = {1: state()}
    states[1].eq_enabled = True
    action = ChannelEQMove("test", 1, 1, 1000, -3, 2)
    after, _ = renderer.propose(states, action)
    base = renderer.render({1: audio()}, states)
    result = renderer.render({1: audio()}, after)
    gain = 20 * np.log10(np.std(result.audio[4096:]) / np.std(base.audio[4096:]))
    assert gain == pytest.approx(-3, abs=.02)
    assert states[1].eq_bands[1].gain_db == 0
    final_action = ChannelEQMove("test", 1, 1, 500, -1, 3)
    replaced, _ = renderer.propose(after, final_action)
    direct, _ = renderer.propose(states, final_action)
    np.testing.assert_array_equal(renderer.render({1: audio()}, replaced).audio,
                                  renderer.render({1: audio()}, direct).audio)


def test_renderer_hpf_and_compressor_change_audio_not_only_metadata():
    renderer = ShadowRenderer(SR)
    t = np.arange(SR) / SR
    x = .05 * np.sin(2 * np.pi * 40 * t)
    filtered = state(); filtered.hpf_hz = 200
    base = renderer.render({1: x}, {1: state()})
    hp = renderer.render({1: x}, {1: filtered})
    assert np.std(hp.audio) < np.std(base.audio) / 10
    unity = state(); unity.compressor = Compressor(ratio=1)
    np.testing.assert_allclose(renderer.render({1: x}, {1: unity}).audio, base.audio, atol=1e-14)
    compressed = state(); compressed.compressor = Compressor(-40, 4, 1, 100)
    assert np.std(renderer.render({1: x}, {1: compressed}).audio) < np.std(base.audio) / 2


def test_antiphase_stereo_does_not_look_like_silence():
    x = audio()
    anti = measure(np.column_stack([x, -x]), SR)
    same = measure(np.column_stack([x, x]), SR)
    assert anti["lufs"] == pytest.approx(same["lufs"], abs=1e-9)
    assert anti["1000_2500"] == pytest.approx(same["1000_2500"], abs=1e-9)


def pipeline_fixture(mode="shadow", confirmations=1):
    states, buffers = {1: state()}, {1: audio()}
    base = ShadowRenderer(SR).render(buffers, states)
    target = corridor(base.features["lufs"] - 2)
    pipeline = ShadowPipeline({"mode": mode, "context": "rock:chorus",
                               "guard": {"confirmations": confirmations},
                               "max_render_age_sec": 30, "model_validated": True}, target)
    return pipeline, states, buffers


def test_pipeline_approves_only_remeasured_improvement():
    pipeline, states, buffers = pipeline_fixture()
    good = ChannelFaderMove("test", 1, -6.5)
    verdict = pipeline.evaluate(buffers, states, good, SR)
    assert verdict.allowed, verdict
    assert verdict.report["after"] < verdict.report["before"]
    assert verdict.report["rendered_action"]["target_db"] == -6.5
    assert not pipeline.guard._applied
    verdict.commit()
    assert pipeline.guard._applied
    bad = ChannelFaderMove("test", 1, -5.5)
    assert not pipeline.evaluate(buffers, states, bad, SR).allowed


def test_pipeline_peak_ceiling_blocks_otherwise_useful_eq():
    pipeline, states, buffers = pipeline_fixture()
    buffers[1] *= 25
    states[1].fader_db = 0
    states[1].eq_enabled = True
    verdict = pipeline.evaluate(buffers, states, ChannelEQMove("test", 1, 1, 1000, 6, 1), SR)
    assert not verdict.allowed
    assert verdict.reason == "predicted_peak_ceiling"


def test_pipeline_no_target_or_unvalidated_model_cannot_authorize():
    pipeline, states, buffers = pipeline_fixture("guarded")
    pipeline.config["model_validated"] = False
    action = ChannelFaderMove("test", 1, -6.5)
    assert pipeline.evaluate(buffers, states, action, SR).reason == "model_or_target_not_validated_for_writes"
    pipeline.corridor = None
    assert pipeline.evaluate(buffers, states, action, SR).reason == "no_approved_target_corridor"


def test_pipeline_rejects_expired_render():
    pipeline, states, buffers = pipeline_fixture()
    pipeline.max_age_sec = 1e-9
    assert pipeline.evaluate(buffers, states, ChannelFaderMove("test", 1, -6.5), SR).reason == "render_expired"


def live_fixture():
    raw = {"gate_enabled": False, "active_bus_sends": [], "dca_assignments": [],
           "main_send": {"on": 1, "pre": 0, "level_db": 0},
           "eq_bands": [(1000, 0, 1)] * 4, "hpf_enabled": False,
           "compressor_enabled": False, "fader_db": -6, "eq_on": False,
           "muted": False, "pan": 0}
    config = {"tap": "post_input_pre_processing", "aligned_complete_direct_sum": True,
              "unmodelled_processing_bypassed": True}
    return [SimpleNamespace(channel_id=1, role="guitars", raw_settings=raw, audio=audio())], config


def test_live_adapter_rejects_missing_processing_or_unknown_routing():
    channels, config = live_fixture()
    assert live_states(channels, config)[1].fader_db == -6
    del channels[0].raw_settings["gate_enabled"]
    with pytest.raises(KeyError):
        live_states(channels, config)
    with pytest.raises(ValueError):
        live_states([], {})


def test_live_adapter_protects_feedback_notches():
    pipeline, _, _ = pipeline_fixture()
    channels, config = live_fixture()
    pipeline.config.update(config)
    channels[0].raw_settings["eq_bands"][0] = (1000, -6, 10)
    result = pipeline.evaluate_live(channels, ChannelEQMove("test", 1, 1, 500, -1, 2), SR)
    assert not result.allowed and "Protected" in result.report["detail"]


class Mixer:
    def __init__(self):
        self.calls = []
        self.fader = -6
        self.success = True

    def get_fader(self, channel):
        return self.fader

    def set_fader(self, channel, value):
        self.calls.append((channel, value))
        if self.success:
            self.fader = value
        return self.success


def test_failed_send_does_not_consume_rate_budget_or_commit_evidence():
    mixer = Mixer(); mixer.success = False
    committed = []
    controller = AutoFOHSafetyController(mixer, time_provider=lambda: 1,
        reviewer=lambda a, s: ReviewDecision(True, "ok", {}, lambda: committed.append(a)))
    action = ChannelFaderMove("test", 1, -6.5)
    assert not controller.execute(action, RuntimeState.FULL_BAND_LEARNING).sent
    assert not controller._last_sent_at and not committed
    mixer.success = True
    assert controller.execute(action, RuntimeState.FULL_BAND_LEARNING).sent
    assert len(committed) == 1


@pytest.mark.parametrize("bad", [float("nan"), float("inf"), -float("inf")])
def test_safety_nonfinite_never_reaches_writer(bad):
    mixer = Mixer()
    result = AutoFOHSafetyController(mixer).execute(ChannelFaderMove("bad", 1, bad),
                                                    RuntimeState.FULL_BAND_LEARNING)
    assert not result.allowed and not mixer.calls


def test_safety_reviews_exact_bounded_action_and_shadow_never_sends():
    mixer = Mixer(); observed = []
    controller = AutoFOHSafetyController(mixer, AutoFOHSafetyConfig(shadow_live=True),
        reviewer=lambda a, s: (observed.append(a) or ReviewDecision(True, "ok")))
    result = controller.execute(ChannelFaderMove("test", 1, 0), RuntimeState.FULL_BAND_LEARNING)
    assert result.simulated and not result.sent and not mixer.calls
    assert result.action.target_db == observed[0].target_db == -5


def test_reviewer_failure_and_denial_both_block_console():
    mixer = Mixer()
    for reviewer in (lambda a, s: ReviewDecision(False, "worse"),
                     lambda a, s: 1 / 0):
        result = AutoFOHSafetyController(mixer, reviewer=reviewer).execute(
            ChannelFaderMove("test", 1, -6.5), RuntimeState.FULL_BAND_LEARNING)
        assert not result.sent and not result.allowed and not mixer.calls


def test_full_render_review_at_safety_boundary_has_no_shadow_writes():
    pipeline, states, buffers = pipeline_fixture()
    mixer = Mixer()
    controller = AutoFOHSafetyController(mixer, AutoFOHSafetyConfig(shadow_live=True),
        reviewer=lambda a, s: pipeline.evaluate(buffers, states, a, SR))
    decision = controller.execute(ChannelFaderMove("test", 1, -6.5), RuntimeState.FULL_BAND_LEARNING)
    assert decision.simulated and not decision.sent and not mixer.calls
    assert decision.payload["shadow_review"]["after"] < decision.payload["shadow_review"]["before"]


def test_live_end_to_end_pipeline_accepts_complete_state_without_mutation():
    pipeline, _, _ = pipeline_fixture()
    channels, config = live_fixture()
    pipeline.config.update(config)
    before = deepcopy(channels[0].raw_settings)
    result = pipeline.evaluate_live(channels, ChannelFaderMove("test", 1, -6.5), SR)
    assert result.allowed, result
    assert channels[0].raw_settings == before


def test_eq_write_does_not_implicitly_enable_bypassed_eq_bank():
    renderer = ShadowRenderer(SR)
    old = {1: state()}
    updated, _ = renderer.propose(old, ChannelEQMove("test", 1, 1, 1000, 6, 1))
    assert not updated[1].eq_enabled
    np.testing.assert_array_equal(renderer.render({1: audio()}, old).audio,
                                  renderer.render({1: audio()}, updated).audio)


def test_three_confirmations_do_not_accept_two_alternating_stale_frames():
    guard = DecisionGuard(GuardConfig(confirmations=3))
    for i, frame in enumerate(["a", "b", "a", "b", "a"]):
        assert not review(guard, frame=frame, now=10 + i).allowed
    assert review(guard, frame="c", now=16).allowed


@pytest.mark.parametrize("bad", [float("nan"), float("inf"), -float("inf")])
def test_nonfinite_mixer_readback_fails_closed(bad):
    mixer = Mixer(); mixer.fader = bad
    result = AutoFOHSafetyController(mixer).execute(ChannelFaderMove("test", 1, -6.5),
                                                    RuntimeState.FULL_BAND_LEARNING)
    assert not result.allowed and not mixer.calls


def test_readback_drift_during_review_cancels_send():
    mixer = Mixer()
    def change_console(action, runtime):
        mixer.fader = -30
        return ReviewDecision(True, "ok")
    result = AutoFOHSafetyController(mixer, reviewer=change_console).execute(
        ChannelFaderMove("test", 1, -6.5), RuntimeState.FULL_BAND_LEARNING)
    assert not result.sent and result.message == "readback changed during shadow review"
    assert not mixer.calls


def test_guarded_proxy_only_releases_approved_typed_write_and_relocks():
    from guarded_mixer import GuardedMixerClient
    base = Mixer(); blocked = []
    mixer = GuardedMixerClient(base, blocked.append)
    assert mixer.set_fader(1, 0) is False
    assert base.calls == [] and blocked == ["set_fader"]
    safety = AutoFOHSafetyController(mixer, reviewer=lambda a, s: ReviewDecision(True, "ok"))
    result = safety.execute(ChannelFaderMove("test", 1, -6.5), RuntimeState.FULL_BAND_LEARNING)
    assert result.sent and base.calls == [(1, -6.5)]
    assert mixer.set_fader(1, 0) is False
    with pytest.raises(RuntimeError):
        with mixer.approved_action():
            raise RuntimeError("test")
    assert mixer.set_fader(1, 0) is False


def test_journal_includes_missing_target_rejection(tmp_path):
    log = tmp_path / "decisions.jsonl"
    pipeline = ShadowPipeline({"mode": "shadow", "journal_path": str(log)})
    result = pipeline.evaluate({1: audio()}, {1: state()}, ChannelFaderMove("test", 1, -6.5), SR)
    assert not result.allowed
    assert json.loads(log.read_text())["reason"] == "no_approved_target_corridor"


def test_offline_cli_fits_and_audits_actual_wav_files(tmp_path):
    import soundfile as sf
    from shadow_mix_cli import main
    x = audio()
    sf.write(tmp_path / "channel.wav", x, SR, subtype="FLOAT")
    target_audio = x * 10 ** (-8 / 20) / math.sqrt(2)
    for i in range(5):
        reference = audio(phase=i * .01) * 10 ** (-8 / 20) / math.sqrt(2)
        sf.write(tmp_path / f"approved_{i}.wav", np.column_stack([reference, reference]), SR, subtype="FLOAT")
    manifest = [{"mix_id": str(i), "approved": True, "context": "rock:chorus",
                 "audio_path": f"approved_{i}.wav"} for i in range(5)]
    (tmp_path / "approved.json").write_text(json.dumps(manifest))
    assert main(["fit", "--manifest", str(tmp_path / "approved.json"), "--context", "rock:chorus",
                 "--output", str(tmp_path / "target.json"), "--features", "lufs"]) == 0
    session = {"sample_rate": SR, "channels": [{"audio_path": "channel.wav", "state": asdict(state())}],
               "action": {"type": "ChannelFaderMove", "reason": "offline", "channel_id": 1,
                          "target_db": -6.5}}
    (tmp_path / "session.json").write_text(json.dumps(session))
    assert main(["audit", "--session", str(tmp_path / "session.json"),
                 "--target", str(tmp_path / "target.json"), "--report", str(tmp_path / "report.json"),
                 "--preview", str(tmp_path / "preview.wav")]) == 0
    assert json.loads((tmp_path / "report.json").read_text())["console_writes"] == 0
    assert sf.info(tmp_path / "preview.wav").samplerate == SR
