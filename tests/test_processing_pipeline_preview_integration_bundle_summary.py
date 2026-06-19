import importlib
import sys
import time
from unittest.mock import MagicMock, patch

import pytest

from backend.processing_pipeline_preview import build_main_pipeline_preview


server_module = importlib.import_module("backend.server")
sys.modules.setdefault("server", server_module)
AutoMixerServer = server_module.AutoMixerServer


class TimestampedState(dict):
    def __init__(self, owner):
        super().__init__()
        self.owner = owner

    def __setitem__(self, key, value):
        super().__setitem__(key, value)
        self.owner.state_timestamps[key] = time.time()

    def update(self, *args, **kwargs):
        for key, value in dict(*args, **kwargs).items():
            self[key] = value


class CachedMixer:
    is_connected = True
    ip = "192.168.1.102"
    port = 2223

    def __init__(self):
        self.state_timestamps = {}
        self.state = TimestampedState(self)
        self.state.update(
            {
                "/ch/1/$name": "Kick In",
                "/ch/2/$name": "Lead Vocal",
                "/ch/2/fdr": -8.0,
                "/ch/2/in/set/trim": 1.0,
            }
        )
        self.send = MagicMock()
        self.get_all_channel_names = MagicMock(return_value={1: "Should Not Scan"})
        self.set_fader = MagicMock()
        self.set_gain = MagicMock()
        self.set_eq = MagicMock()
        self.set_compressor = MagicMock()

    def get_state(self):
        return {"connected": True}

    @staticmethod
    def _parse_dca_assignments(tags):
        if not tags:
            return []
        assignments = []
        for token in str(tags).replace(",", " ").split():
            if token.upper().startswith("#D") and token[2:].isdigit():
                assignments.append(int(token[2:]))
        return sorted(set(assignments))


class FakeAudioCapture:
    running = True
    num_channels = 4
    sample_rate = 48000

    def get_status(self):
        return {
            "running": True,
            "source_type": "device",
            "num_channels": self.num_channels,
            "sample_rate": self.sample_rate,
            "buffer_seconds": 5.0,
            "subscribers": 0,
        }

    def get_rms(self, channel):
        return {1: -18.2, 2: -12.4, 3: -100.0, 4: -42.0}.get(channel, -100.0)

    def get_peak(self, channel):
        return {1: -5.2, 2: -2.3, 3: -100.0, 4: -20.0}.get(channel, -100.0)

    def get_lufs(self, channel):
        return {1: -22.1, 2: -18.8, 3: -100.0, 4: -34.0}.get(channel, -100.0)


@pytest.fixture
def server():
    with patch.object(AutoMixerServer, "_load_config", return_value={}), patch(
        "server.BleedService"
    ):
        yield AutoMixerServer()


def test_main_pipeline_preview_summary_surfaces_integration_bundle_readiness(server):
    server.mixer_client = CachedMixer()
    server.connection_mode = "wing"
    server.audio_capture = FakeAudioCapture()

    payload = build_main_pipeline_preview(server, total_channels=4)
    summary = payload["summary"]
    integration = payload["registry"]["autonomy_promotion_readiness"]["summary"][
        "offline_integration_bundle"
    ]

    assert summary["autonomy_promotion_offline_integration_bundle_ready"] is False
    assert summary["autonomy_promotion_offline_integration_bundle_status"] == "blocked"
    assert summary["autonomy_promotion_offline_integration_bundle_blockers"] == (
        integration["blockers"]
    )
    assert "offline_only_required" in summary[
        "autonomy_promotion_offline_integration_bundle_blockers"
    ]
    assert "operator_report_bound" in summary[
        "autonomy_promotion_offline_integration_bundle_missing_checks"
    ]
    assert summary[
        "autonomy_promotion_offline_integration_bundle_missing_checks"
    ] == integration["missing_integration_checks"]
    assert (
        summary["autonomy_promotion_offline_integration_bundle_source_authority_denied"]
        is True
    )
    assert (
        summary[
            "autonomy_promotion_offline_integration_bundle_live_authority_granted"
        ]
        is False
    )
    assert (
        summary[
            "autonomy_promotion_offline_integration_bundle_readiness_closure_allowed"
        ]
        is False
    )
    assert summary["autonomy_promotion_offline_integration_bundle_live_write"] is False
    assert (
        summary["autonomy_promotion_offline_integration_bundle_hardware_evidence"]
        is False
    )
    assert payload["executor_policy"]["preview_only"] is True
    assert payload["summary"]["queue_mutation"] is False
    server.mixer_client.set_fader.assert_not_called()
    server.mixer_client.set_gain.assert_not_called()
    server.mixer_client.set_eq.assert_not_called()
    server.mixer_client.set_compressor.assert_not_called()
