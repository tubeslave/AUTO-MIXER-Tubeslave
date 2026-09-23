import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'backend'))

from live_runtime.contracts import ChannelFeatures, LiveMode, MixFeatures, SoundcheckState
from live_runtime.iteration import IterationPhase
from live_runtime.service import LiveSoundcheckService, LiveStartRequest


def _request(mode=LiveMode.AUTO_SAFE):
    return LiveStartRequest(
        mixer_type="wing",
        mixer_ip="10.0.0.5",
        mixer_port=2223,
        audio_device_name="USB",
        num_channels=48,
        selected_channels=[1, 2],
        mode=mode,
    )


def _features(main_peak_dbfs):
    return MixFeatures(
        channels=[
            ChannelFeatures(
                channel=1,
                name="Lead Vocal",
                rms_dbfs=-20.0,
                peak_dbfs=-9.0,
                crest_db=11.0,
                activity=0.8,
            )
        ],
        main_rms_dbfs=-14.0,
        main_peak_dbfs=main_peak_dbfs,
        main_crest_db=13.0,
    )


class FakeWingClient:
    def __init__(self, values=None):
        self.values = dict(values or {})
        self.callbacks = {}
        self.sent = []

    def subscribe(self, address, callback):
        self.callbacks.setdefault(address, []).append(callback)

    def send(self, address, *values):
        self.sent.append((address, values))
        if values:
            self.values[address] = values[0] if len(values) == 1 else values
            return True
        actual = self.values.get(address)
        for callback in self.callbacks.get(address, []):
            callback(address, str(actual), 0.5, actual)
        return True


class ConnectedWingEngine:
    transport = None

    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.state = type("State", (), {"value": "idle"})()
        self._real_mixer_client = type(self).transport
        self.mixer_client = self._real_mixer_client

    def start_async(self):
        self.state = type("State", (), {"value": "running"})()

    def stop(self):
        self.state = type("State", (), {"value": "stopped"})()

    def get_status(self):
        return {"state": self.state.value, "mixer_connected": True, "audio_running": True}


def _service(window=0.0):
    client = FakeWingClient({"/main/1/fdr": -6.0})
    ConnectedWingEngine.transport = client
    service = LiveSoundcheckService(
        engine_factory=ConnectedWingEngine,
        iteration_verification_window_s=window,
    )
    service.start(_request())
    return service, client


def test_sequential_feature_snapshots_apply_then_keep_one_hypothesis():
    service, client = _service(window=0.0)

    applied = service.process_feature_snapshot(_features(-0.5), roles={})
    assert applied.state is SoundcheckState.VERIFY
    assert applied.iteration.phase is IterationPhase.VERIFY_PENDING
    assert applied.hypothesis.name == "main_headroom_protection"
    assert client.values["/main/1/fdr"] == -6.5
    assert service.get_status()["iteration_active"] is True

    kept = service.process_feature_snapshot(_features(-1.0), roles={})
    assert kept.state is SoundcheckState.LISTEN
    assert kept.iteration.phase is IterationPhase.KEPT
    assert client.values["/main/1/fdr"] == -6.5
    assert service.get_status()["iteration_active"] is False


def test_active_iteration_blocks_a_second_proposal_until_verify_window_closes():
    service, client = _service(window=60.0)

    first = service.process_feature_snapshot(_features(-0.5), roles={})
    sent_after_apply = list(client.sent)
    waiting = service.process_feature_snapshot(_features(-0.4), roles={})

    assert first.iteration.phase is IterationPhase.VERIFY_PENDING
    assert waiting.state is SoundcheckState.VERIFY
    assert waiting.iteration.phase is IterationPhase.VERIFY_WAIT
    assert waiting.hypothesis.name == first.hypothesis.name
    assert client.sent == sent_after_apply
    assert client.values["/main/1/fdr"] == -6.5


def test_regressing_feature_snapshot_triggers_verified_rollback():
    service, client = _service(window=0.0)

    service.process_feature_snapshot(_features(-0.5), roles={})
    rejected = service.process_feature_snapshot(_features(-0.4), roles={})

    assert rejected.state is SoundcheckState.LISTEN
    assert rejected.iteration.phase is IterationPhase.ROLLED_BACK
    assert rejected.iteration.rollback.restored is True
    assert client.values["/main/1/fdr"] == -6.0
    assert client.sent[-3:] == [
        ("/main/1/fdr", ()),
        ("/main/1/fdr", (-6.0,)),
        ("/main/1/fdr", ()),
    ]


def test_operator_takeover_propagates_hold_without_fighting_manual_control():
    service, client = _service(window=0.0)

    service.process_feature_snapshot(_features(-0.5), roles={})
    sent_after_apply = list(client.sent)
    held = service.process_feature_snapshot(
        _features(-1.0),
        roles={},
        operator_took_control=True,
    )

    assert held.state is SoundcheckState.HOLD
    assert held.iteration.phase is IterationPhase.HOLD
    assert held.reason == "operator_took_control"
    assert client.sent == sent_after_apply
    assert client.values["/main/1/fdr"] == -6.5
    assert service.get_status()["soundcheck_state"] == "hold"
    assert service.get_status()["iteration_hold_reason"] == "operator_took_control"

    ignored = service.process_feature_snapshot(_features(-0.2), roles={})
    assert ignored.state is SoundcheckState.HOLD
    assert ignored.reason == "operator_took_control"
    assert client.sent == sent_after_apply
