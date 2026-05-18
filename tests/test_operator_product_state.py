import importlib
import sys
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from backend.operator_analysis import build_operator_analysis_report
from backend.operator_product_state import build_channel_inventory, build_decision_queue
from backend.operator_recommendation_bridge import (
    build_safe_gain_operator_proposals,
    build_soundcheck_operator_proposals,
)
from soundcheck_recommendation_workflow import NoWriteSoundcheckRecommendationWorkflow


server_module = importlib.import_module("backend.server")
sys.modules.setdefault("server", server_module)
AutoMixerServer = server_module.AutoMixerServer


class CachedMixer:
    is_connected = True
    ip = "192.168.1.102"
    port = 2223

    def __init__(self):
        self.state = {
            "/ch/1/$name": "Kick In",
            "/ch/2/$name": "Lead Vocal",
            "/ch/2/fdr": -8.0,
            "/ch/2/in/set/trim": 1.0,
        }
        self.get_all_channel_names = MagicMock(return_value={1: "Should Not Scan"})

    def get_state(self):
        return {"connected": True}


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
        return {1: -19.1, 2: -13.5, 3: -100.0, 4: -43.2}.get(channel, -100.0)


class FakeSafeGainCalibrator:
    channel_mapping = {1: 2}

    def get_suggestions(self):
        return {
            1: {
                "channel": 1,
                "peak_db": -5.5,
                "lufs": -21.0,
                "crest_factor_db": 12.0,
                "signal_presence": 88.0,
                "suggested_gain_db": 2.5,
                "limited_by": "lufs",
            }
        }


@pytest.fixture
def server():
    with patch.object(AutoMixerServer, "_load_config", return_value={}), patch("server.BleedService"):
        yield AutoMixerServer()


def test_channel_inventory_uses_cached_runtime_state_without_scanning(server):
    mixer = CachedMixer()
    server.connection_mode = "wing"
    server.mixer_client = mixer
    server.audio_capture = FakeAudioCapture()

    inventory = build_channel_inventory(server, total_channels=4)

    assert inventory["type"] == "channel_inventory"
    assert inventory["connection"]["connected"] is True
    assert inventory["summary"]["total_channels"] == 4
    assert inventory["summary"]["active_channels"] == 3
    assert inventory["channels"][0]["name"] == "Kick In"
    assert inventory["channels"][0]["group"] == "DRUMS"
    assert inventory["channels"][1]["name"] == "Lead Vocal"
    assert inventory["channels"][1]["group"] == "VOCALS"
    assert inventory["channels"][1]["control_state"]["fader_db"] == -8.0
    mixer.get_all_channel_names.assert_not_called()


def test_decision_queue_uses_backend_native_proposal_queue(server):
    queue = build_decision_queue(server)

    assert queue["type"] == "decision_queue"
    assert queue["status"] == "ok"
    assert queue["available"] is True
    assert queue["source"] == "operator_proposal_queue"
    assert queue["agent_runtime_available"] is False
    assert queue["reason"] == "mixing_agent_runtime_not_initialized"
    assert queue["pending_actions"] == []
    assert queue["summary"]["queue_accepts_proposals"] is True


@pytest.mark.asyncio
async def test_product_state_handlers_round_trip(server):
    server.mixer_client = CachedMixer()
    server.connection_mode = "wing"
    server.audio_capture = FakeAudioCapture()
    server.send_to_client = AsyncMock()

    await server._dispatch["get_channel_inventory"](object(), {"total_channels": 2})
    await server._dispatch["get_decision_queue"](object(), {"limit": 5})
    await server._dispatch["get_connection_topology"](object(), {})

    payloads = [call.args[1] for call in server.send_to_client.await_args_list]
    assert [payload["type"] for payload in payloads] == [
        "channel_inventory",
        "decision_queue",
        "connection_topology",
    ]
    assert payloads[0]["channels"][1]["name"] == "Lead Vocal"
    assert payloads[1]["reason"] == "mixing_agent_runtime_not_initialized"
    assert payloads[1]["proposal_queue_available"] is True
    assert payloads[2]["connection"]["connected"] is True


@pytest.mark.asyncio
async def test_operator_proposal_create_is_blocked_in_analyze_mode(server):
    server.operator_mode = "analyze"
    server.send_to_client = AsyncMock()

    await server._dispatch["create_operator_proposal"](
        object(),
        {"title": "Lead Vocal -1 dB", "channel": 2, "value_type": "fader", "value": -1.0},
    )

    payload = server.send_to_client.await_args.args[1]
    assert payload["type"] == "operator_proposal_created"
    assert payload["status"] == "blocked"
    assert payload["reason"] == "operator_mode_blocks_proposal_creation"
    assert build_decision_queue(server)["summary"]["pending_count"] == 0


@pytest.mark.asyncio
async def test_operator_proposal_queue_accepts_assist_suggestions(server):
    server.send_to_client = AsyncMock()

    await server._dispatch["create_operator_proposal"](
        object(),
        {
            "title": "Lead Vocal: lower fader",
            "channel": 2,
            "value_type": "fader",
            "value": -1.5,
            "reason": "unit",
        },
    )
    queue = build_decision_queue(server)

    assert queue["summary"]["pending_count"] == 1
    assert queue["pending_actions"][0]["title"] == "Lead Vocal: lower fader"
    assert queue["pending_actions"][0]["can_apply"] is True

    proposal_id = queue["pending_actions"][0]["id"]
    await server._dispatch["accept_operator_proposal"](object(), {"proposal_id": proposal_id})
    queue = build_decision_queue(server)

    assert queue["summary"]["pending_count"] == 0
    assert queue["summary"]["accepted_count"] == 1
    assert queue["history"][0]["status"] == "accepted"


@pytest.mark.asyncio
async def test_operator_proposal_without_complete_change_is_analysis_only(server):
    server.send_to_client = AsyncMock()

    await server._dispatch["create_operator_proposal"](
        object(),
        {
            "title": "Lead Vocal: inspect fader",
            "kind": "fader",
            "channel": 2,
            "reason": "missing target value",
        },
    )

    proposal = build_decision_queue(server)["pending_actions"][0]
    assert proposal["can_apply"] is False
    assert proposal["requires_approval"] is False
    assert proposal["requested_change"] is None


@pytest.mark.asyncio
async def test_operator_proposal_apply_is_blocked_outside_supervised_mode(server):
    server.send_to_client = AsyncMock()

    await server._dispatch["create_operator_proposal"](
        object(),
        {"title": "Kick: trim", "channel": 1, "value_type": "gain", "value": 1.0, "reason": "unit"},
    )
    proposal_id = build_decision_queue(server)["pending_actions"][0]["id"]
    await server._dispatch["apply_operator_proposal"](
        object(),
        {"proposal_id": proposal_id, "approved": True},
    )

    payload = server.send_to_client.await_args.args[1]
    assert payload["type"] == "operator_proposal_apply_blocked"
    assert payload["status"] == "apply_blocked"
    assert payload["apply_result"]["reason"] == "operator_mode_blocks_live_write"
    assert build_decision_queue(server)["summary"]["pending_count"] == 1


@pytest.mark.asyncio
async def test_supervised_apply_blocks_incomplete_requested_change(server):
    server.operator_mode = "supervised"
    server.mixer_client = MagicMock()
    server.mixer_client.is_connected = True
    server._apply_manual_console_write = AsyncMock()
    server.send_to_client = AsyncMock()

    await server._dispatch["create_operator_proposal"](
        object(),
        {
            "title": "Lead Vocal: incomplete fader request",
            "kind": "fader",
            "channel": 2,
            "reason": "unit",
        },
    )
    proposal_id = build_decision_queue(server)["pending_actions"][0]["id"]
    await server._dispatch["apply_operator_proposal"](
        object(),
        {"proposal_id": proposal_id, "approved": True},
    )

    server._apply_manual_console_write.assert_not_called()
    payload = server.send_to_client.await_args.args[1]
    assert payload["type"] == "operator_proposal_apply_blocked"
    assert payload["status"] == "apply_blocked"
    assert payload["apply_result"]["reason"] == "proposal_has_no_applyable_requested_change"


@pytest.mark.asyncio
async def test_supervised_operator_proposal_apply_routes_through_manual_gate(server):
    server.operator_mode = "supervised"
    server.mixer_client = MagicMock()
    server.mixer_client.is_connected = True
    server._apply_manual_console_write = AsyncMock(return_value={
        "status": "applied",
        "channel": 2,
        "value_type": "fader",
        "effective_value": -3.0,
    })
    server.send_to_client = AsyncMock()

    await server._dispatch["create_operator_proposal"](
        object(),
        {
            "title": "Lead Vocal: supervised fader",
            "channel": 2,
            "value_type": "fader",
            "value": -3.0,
            "reason": "unit",
        },
    )
    proposal_id = build_decision_queue(server)["pending_actions"][0]["id"]
    await server._dispatch["apply_operator_proposal"](
        object(),
        {"proposal_id": proposal_id, "approved": True, "reason": "operator approved"},
    )

    server._apply_manual_console_write.assert_awaited_once()
    payload = server.send_to_client.await_args.args[1]
    assert payload["type"] == "operator_proposal_apply_result"
    assert payload["status"] == "applied"
    queue = build_decision_queue(server)
    assert queue["summary"]["pending_count"] == 0
    assert queue["summary"]["applied_count"] == 1


def test_operator_analysis_reports_without_creating_in_analyze_mode(server):
    server.operator_mode = "analyze"
    server.mixer_client = CachedMixer()
    server.audio_capture = FakeAudioCapture()

    report = build_operator_analysis_report(server, total_channels=4, create_proposals=True)
    queue = build_decision_queue(server)

    assert report["type"] == "operator_analysis_report"
    assert report["status"] == "ok"
    assert report["observation_count"] == 1
    assert report["created_count"] == 0
    assert report["observations"][0]["rule"] == "peak_risk"
    assert queue["summary"]["pending_count"] == 0


def test_operator_analysis_creates_applyable_proposal_when_fader_cache_exists(server):
    server.mixer_client = CachedMixer()
    server.audio_capture = FakeAudioCapture()

    report = build_operator_analysis_report(server, total_channels=4, create_proposals=True)
    queue = build_decision_queue(server)

    assert report["status"] == "ok"
    assert report["created_count"] == 1
    assert queue["summary"]["pending_count"] == 1
    proposal = queue["pending_actions"][0]
    assert proposal["id"] == "analysis:peak_risk:2"
    assert proposal["can_apply"] is True
    assert proposal["requested_change"]["value_type"] == "fader"
    assert proposal["requested_change"]["channel"] == 2
    assert proposal["requested_change"]["value"] < -8.0


@pytest.mark.asyncio
async def test_run_operator_analysis_handler_refreshes_queue(server):
    server.mixer_client = CachedMixer()
    server.audio_capture = FakeAudioCapture()
    server.send_to_client = AsyncMock()

    await server._dispatch["run_operator_analysis"](
        object(),
        {"total_channels": 4, "create_proposals": True},
    )

    payload = server.send_to_client.await_args.args[1]
    assert payload["type"] == "operator_analysis_report"
    assert payload["created_count"] == 1
    assert build_decision_queue(server)["summary"]["pending_count"] == 1


def test_safe_gain_recommendation_bridge_builds_applyable_trim_proposal():
    mixer = CachedMixer()

    proposals = build_safe_gain_operator_proposals(
        FakeSafeGainCalibrator().get_suggestions(),
        channel_mapping={1: 2},
        mixer_client=mixer,
    )

    assert len(proposals) == 1
    proposal = proposals[0]
    assert proposal["id"] == "safe_gain:1:2"
    assert proposal["requested_change"]["value_type"] == "gain"
    assert proposal["requested_change"]["channel"] == 2
    assert proposal["requested_change"]["current_value"] == 1.0
    assert proposal["requested_change"]["value"] == 3.5


def test_soundcheck_recommendation_bridge_maps_gain_and_fader_to_applyable_proposals():
    workflow = NoWriteSoundcheckRecommendationWorkflow()
    workflow.recommend_input_gain(channel=1, current_trim_db=0.0, target_trim_db=-4.0)
    workflow.recommend_fader(channel=2, current_fader_db=-10.0, target_fader_db=-8.5)

    proposals = build_soundcheck_operator_proposals(workflow.build_bundle())

    assert [proposal["requested_change"]["value_type"] for proposal in proposals] == ["gain", "fader"]
    assert proposals[0]["requested_change"]["value"] == -4.0
    assert proposals[1]["requested_change"]["value"] == -8.5


@pytest.mark.asyncio
async def test_import_safe_gain_suggestions_handler_feeds_operator_queue(server):
    server.mixer_client = CachedMixer()
    server.safe_gain_calibrator = FakeSafeGainCalibrator()
    server.send_to_client = AsyncMock()

    await server._dispatch["import_safe_gain_suggestions"](object(), {})

    payload = server.send_to_client.await_args.args[1]
    assert payload["type"] == "operator_proposals_imported"
    assert payload["source"] == "safe_gain_calibrator"
    assert payload["created_count"] == 1
    queue = build_decision_queue(server)
    assert queue["summary"]["pending_count"] == 1
    assert queue["pending_actions"][0]["requested_change"]["value"] == 3.5


def test_import_soundcheck_recommendations_respects_analyze_mode(server):
    server.operator_mode = "analyze"
    workflow = NoWriteSoundcheckRecommendationWorkflow()
    workflow.recommend_fader(channel=2, current_fader_db=-10.0, target_fader_db=-8.5)

    result = server.import_soundcheck_recommendations_to_operator_queue(workflow.build_bundle())

    assert result["type"] == "operator_proposals_imported"
    assert result["blocked_count"] == 1
    assert result["created_count"] == 0
    assert build_decision_queue(server)["summary"]["pending_count"] == 0
