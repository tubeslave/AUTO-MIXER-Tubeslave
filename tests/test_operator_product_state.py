import importlib
import sys
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from backend.operator_analysis import build_operator_analysis_report
from backend.operator_product_state import (
    build_agent_communications_snapshot,
    build_channel_inventory,
    build_decision_queue,
    build_groups_buses_snapshot,
    build_memory_knowledge_snapshot,
    build_operator_agent_roster,
    build_processing_pipeline_snapshot,
    build_rules_library_snapshot,
    build_training_sessions_snapshot,
)
from backend.processing_pipeline_preview import build_main_pipeline_preview
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
        return {1: -26.0, 2: -13.5, 3: -100.0, 4: -43.2}.get(channel, -100.0)


class VocalLevelCapture:
    running = True
    num_channels = 16
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
        return -17.5 if channel == 16 else -100.0

    def get_peak(self, channel):
        return -12.4 if channel == 16 else -100.0

    def get_lufs(self, channel):
        return -17.7 if channel == 16 else -100.0


class StaticMeterCapture:
    running = True
    sample_rate = 48000

    def __init__(self, *, num_channels=16, rms=None, peak=None, lufs=None):
        self.num_channels = num_channels
        self.rms = dict(rms or {})
        self.peak = dict(peak or {})
        self.lufs = dict(lufs or {})

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
        return self.rms.get(channel, -100.0)

    def get_peak(self, channel):
        return self.peak.get(channel, -100.0)

    def get_lufs(self, channel):
        return self.lufs.get(channel, -100.0)


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


def test_channel_inventory_uses_wing_linked_source_name_for_high_channels(server):
    mixer = CachedMixer()
    mixer.state.update({
        "/ch/16/$name": "VOCAL",
        "/ch/16/name": "",
        "/ch/16/fdr": 0.0,
        "/ch/16/mute": 0,
    })
    server.connection_mode = "wing"
    server.mixer_client = mixer
    server.audio_capture = None

    inventory = build_channel_inventory(server, total_channels=16)
    channel_16 = next(channel for channel in inventory["channels"] if channel["number"] == 16)

    assert channel_16["name"] == "VOCAL"
    assert channel_16["role"] == "Vocal"
    assert channel_16["group"] == "VOCALS"
    assert channel_16["control_state"]["fader_db"] == 0.0
    assert channel_16["control_state"]["mute"] == 0
    assert channel_16["ai_recognition"]["source"] == "cached_mixer_name"
    mixer.get_all_channel_names.assert_not_called()


def test_channel_inventory_does_not_classify_playback_as_vocal(server):
    mixer = CachedMixer()
    mixer.state.update({
        "/ch/11/$name": "PLAYBACK L",
        "/ch/12/$name": "PLAYBACK R",
    })
    server.connection_mode = "wing"
    server.mixer_client = mixer
    server.audio_capture = None

    inventory = build_channel_inventory(server, total_channels=12)
    by_number = {channel["number"]: channel for channel in inventory["channels"]}

    assert by_number[11]["role"] == "Playback"
    assert by_number[11]["group"] == "PLAYBACK"
    assert by_number[12]["role"] == "Playback"
    assert by_number[12]["group"] == "PLAYBACK"


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


def test_operator_agent_roster_exposes_director_agents_and_blocked_executor(server):
    roster = build_operator_agent_roster(server)
    agents = {agent["id"]: agent for agent in roster["agents"]}

    assert roster["type"] == "operator_agent_roster"
    assert roster["summary"]["agent_count"] >= 8
    assert agents["automixer_director"]["status"] == "ACTIVE"
    assert agents["safety_agent"]["status"] == "ACTIVE"
    assert agents["proposal_queue_agent"]["status"] == "ACTIVE"
    assert agents["console_executor_agent"]["status"] == "BLOCKED"
    assert agents["console_executor_agent"]["can_write"] is False
    assert agents["console_executor_agent"]["blocked_reason"] == "suggestions_only"


def test_operator_agent_roster_marks_executor_ready_only_for_supervised_connected_mixer(server):
    server.operator_mode = "supervised"
    server.connection_mode = "wing"
    server.mixer_client = CachedMixer()

    roster = build_operator_agent_roster(server)
    executor = {agent["id"]: agent for agent in roster["agents"]}["console_executor_agent"]

    assert roster["summary"]["can_live_execute"] is True
    assert executor["status"] == "ACTIVE"
    assert executor["can_write"] is True
    assert executor["write_scope"] == ["fader", "gain"]


def _walk_keys(value):
    if isinstance(value, dict):
        for key, child in value.items():
            yield key
            yield from _walk_keys(child)
    elif isinstance(value, list):
        for child in value:
            yield from _walk_keys(child)


def test_frontend_read_model_snapshots_are_display_only(server):
    server.mixer_client = CachedMixer()
    server.connection_mode = "wing"
    server.audio_capture = FakeAudioCapture()

    snapshots = [
        build_groups_buses_snapshot(server),
        build_memory_knowledge_snapshot(server),
        build_training_sessions_snapshot(server),
        build_rules_library_snapshot(server),
        build_agent_communications_snapshot(server),
    ]

    assert [snapshot["type"] for snapshot in snapshots] == [
        "groups_buses_snapshot",
        "memory_knowledge_snapshot",
        "training_sessions_snapshot",
        "rules_library_snapshot",
        "agent_communications_snapshot",
    ]
    assert snapshots[0]["buses"]
    assert snapshots[1]["memories"]
    assert snapshots[2]["sessions"]
    assert snapshots[3]["rules"]
    assert snapshots[4]["communications"]

    forbidden_keys = {
        "action",
        "actions",
        "apply",
        "apply_operator_proposal",
        "load_snap",
        "requested_change",
        "requested_changes",
        "route_output",
        "route_multiple_outputs",
        "save_snap",
        "set_channel_alt_input",
        "set_channel_input",
        "set_compressor",
        "set_eq",
        "set_fx_mix",
        "set_fx_model",
        "set_fx_on",
        "set_fx_parameter",
        "set_fader",
        "set_gain",
        "set_insert",
        "start_auto_compressor",
        "start_auto_compressor_live",
        "start_auto_compressor_soundcheck",
        "write",
        "writes",
    }
    for snapshot in snapshots:
        assert not (set(_walk_keys(snapshot)) & forbidden_keys)


def test_dashboard_snapshot_tolerates_missing_supervised_pilot_status(server):
    if hasattr(server, "get_supervised_pilot_status"):
        delattr(server, "get_supervised_pilot_status")

    snapshot = server._dispatch  # keep handler registration intact
    from backend.operator_product_state import build_dashboard_snapshot

    payload = build_dashboard_snapshot(server)

    assert snapshot
    assert payload["type"] == "dashboard_snapshot"
    assert payload["safety"]["pilot_primary_state"] is None
    assert payload["safety"]["pilot_ready"] is None
    assert payload["runtime"]["processing"]["main_pipeline_id"] == "main_live_assist"
    assert payload["runtime"]["processing"]["live_apply_policy"] == "supervised_fader_gain_only"


def test_processing_pipeline_snapshot_exposes_method_selection_without_writes(server):
    payload = build_processing_pipeline_snapshot(server)

    assert payload["type"] == "processing_pipeline_snapshot"
    assert payload["runtime"]["main_pipeline_id"] == "main_live_assist"
    assert payload["summary"]["live_executor_method_ids"] == ["supervised_manual_fader_gain"]
    assert payload["summary"]["live_allowed_write_kinds"] == ["fader", "gain"]
    assert any(profile["id"] == "quarantine" for profile in payload["pipeline_profiles"])
    assert any(method["status"] == "placeholder" for method in payload["methods"])
    assert all(
        method["can_write_live"] is False
        for method in payload["methods"]
        if method["status"] in {"placeholder", "quarantined", "deprecated"}
    )


def test_main_pipeline_preview_is_read_only_and_does_not_mutate_queue(server):
    server.mixer_client = CachedMixer()
    server.mixer_client.set_fader = MagicMock()
    server.mixer_client.set_gain = MagicMock()
    server.mixer_client.set_eq = MagicMock()
    server.mixer_client.set_compressor = MagicMock()
    server.audio_capture = FakeAudioCapture()

    payload = build_main_pipeline_preview(server, total_channels=4)
    queue = build_decision_queue(server)

    assert payload["type"] == "main_pipeline_preview"
    assert payload["pipeline_id"] == "main_live_assist"
    assert payload["executor_policy"]["preview_only"] is True
    assert payload["executor_policy"]["queue_mutation"] is False
    assert payload["executor_policy"]["live_write"] is False
    assert payload["executor_policy"]["allowed_requested_change_value_types"] == ["fader", "gain"]
    assert payload["summary"]["proposal_count"] == 1
    assert queue["summary"]["pending_count"] == 0
    assert payload["proposals"][0]["adapter_id"] == "operator_analysis_gain_fader"
    assert payload["proposals"][0]["requested_change"]["value_type"] == "fader"
    assert payload["proposals"][0]["requested_change"]["channel"] == 2
    server.mixer_client.set_fader.assert_not_called()
    server.mixer_client.set_gain.assert_not_called()
    server.mixer_client.set_eq.assert_not_called()
    server.mixer_client.set_compressor.assert_not_called()


def test_main_pipeline_preview_keeps_non_fader_gain_methods_blocked(server):
    payload = build_main_pipeline_preview(server, total_channels=4)
    blockers = {blocker["method_id"]: blocker for blocker in payload["blockers"]}

    for method_id in (
        "auto_eq",
        "auto_compressor",
        "auto_fx",
        "routing_snapshot_reset_writes",
        "fx_direct_apply",
    ):
        assert method_id in blockers
        assert blockers[method_id]["adapter_id"] == "registry_blockers"
        assert blockers[method_id]["can_import_to_queue"] is False
        assert "requested_change" not in blockers[method_id]

    executor_policy = payload["executor_policy"]
    assert executor_policy["executor_method_ids"] == ["supervised_manual_fader_gain"]


def test_main_pipeline_preview_includes_safe_gain_as_preview_only(server):
    server.mixer_client = CachedMixer()
    server.safe_gain_calibrator = FakeSafeGainCalibrator()

    payload = build_main_pipeline_preview(server, total_channels=4)
    queue = build_decision_queue(server)
    safe_gain_proposals = [
        proposal
        for proposal in payload["proposals"]
        if proposal["adapter_id"] == "safe_gain_suggestions"
    ]

    assert payload["safe_gain"]["available"] is True
    assert payload["summary"]["safe_gain_proposal_count"] == 1
    assert len(safe_gain_proposals) == 1
    assert safe_gain_proposals[0]["method_id"] == "lufs_gain_staging"
    assert safe_gain_proposals[0]["requested_change"]["value_type"] == "gain"
    assert safe_gain_proposals[0]["queue_action"] == "preview_only_not_queued"
    assert queue["summary"]["pending_count"] == 0


@pytest.mark.asyncio
async def test_product_state_handlers_round_trip(server):
    server.mixer_client = CachedMixer()
    server.connection_mode = "wing"
    server.audio_capture = FakeAudioCapture()
    server.send_to_client = AsyncMock()

    await server._dispatch["get_channel_inventory"](object(), {"total_channels": 2})
    await server._dispatch["get_decision_queue"](object(), {"limit": 5})
    await server._dispatch["get_operator_agent_roster"](object(), {})
    await server._dispatch["get_connection_topology"](object(), {})
    await server._dispatch["get_groups_buses_snapshot"](object(), {})
    await server._dispatch["get_memory_knowledge_snapshot"](object(), {})
    await server._dispatch["get_training_sessions_snapshot"](object(), {})
    await server._dispatch["get_rules_library_snapshot"](object(), {})
    await server._dispatch["get_agent_communications_snapshot"](object(), {})
    await server._dispatch["get_processing_pipeline_snapshot"](object(), {"include_methods": False})
    await server._dispatch["get_main_pipeline_preview"](object(), {"total_channels": 4})

    payloads = [call.args[1] for call in server.send_to_client.await_args_list]
    assert [payload["type"] for payload in payloads] == [
        "channel_inventory",
        "decision_queue",
        "operator_agent_roster",
        "connection_topology",
        "groups_buses_snapshot",
        "memory_knowledge_snapshot",
        "training_sessions_snapshot",
        "rules_library_snapshot",
        "agent_communications_snapshot",
        "processing_pipeline_snapshot",
        "main_pipeline_preview",
    ]
    assert payloads[0]["channels"][1]["name"] == "Lead Vocal"
    assert payloads[1]["reason"] == "mixing_agent_runtime_not_initialized"
    assert payloads[1]["proposal_queue_available"] is True
    assert payloads[2]["summary"]["agent_count"] >= 8
    assert payloads[3]["connection"]["connected"] is True
    assert payloads[4]["buses"]
    assert payloads[5]["memories"]
    assert payloads[6]["sessions"]
    assert payloads[7]["rules"]
    assert payloads[8]["communications"]
    assert payloads[9]["methods"] == []
    assert payloads[9]["summary"]["live_apply_policy"] == "supervised_fader_gain_only"
    assert payloads[10]["executor_policy"]["preview_only"] is True
    assert payloads[10]["summary"]["queue_mutation"] is False


@pytest.mark.asyncio
async def test_operator_mode_handlers_round_trip(server):
    server.send_to_client = AsyncMock()

    await server._dispatch["get_operator_mode"](object(), {})
    await server._dispatch["set_operator_mode"](
        object(),
        {"mode": "supervised", "reason": "unit_mode_change"},
    )

    payloads = [call.args[1] for call in server.send_to_client.await_args_list]
    assert payloads[0]["type"] == "operator_mode_status"
    assert payloads[0]["mode"] == "assist"
    assert payloads[1]["type"] == "operator_mode_status"
    assert payloads[1]["mode"] == "supervised"
    assert payloads[1]["previous_mode"] == "assist"
    assert payloads[1]["reason"] == "unit_mode_change"


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
async def test_supervised_linked_proposal_apply_routes_each_change_through_manual_gate(server):
    server.operator_mode = "supervised"
    server.mixer_client = MagicMock()
    server.mixer_client.is_connected = True
    server.mixer_client.state = {"/ch/9/fdr": 0.0, "/ch/10/fdr": 0.0}
    server._apply_manual_console_write = AsyncMock(side_effect=[
        {
            "status": "applied",
            "channel": 9,
            "value_type": "fader",
            "effective_value": -4.5,
            "old_value": 0.0,
            "readback_value": -4.5,
            "readback_status": "verified_from_cache",
            "rollback_available": True,
            "rollback_value": 0.0,
            "approval_id": "linked-unit:1",
            "source": "operator_proposal_linked:proposal:1",
        },
        {
            "status": "applied",
            "channel": 10,
            "value_type": "fader",
            "effective_value": -4.5,
            "old_value": 0.0,
            "readback_value": -4.5,
            "readback_status": "verified_from_cache",
            "rollback_available": True,
            "rollback_value": 0.0,
            "approval_id": "linked-unit:2",
            "source": "operator_proposal_linked:proposal:2",
        },
    ])
    server.send_to_client = AsyncMock()

    await server._dispatch["create_operator_proposal"](
        object(),
        {
            "title": "GUITAR L/R: linked faders",
            "kind": "linked_fader",
            "requested_changes": [
                {"value_type": "fader", "channel": 9, "value": -4.5, "current_value": 0.0, "delta_db": -4.5},
                {"value_type": "fader", "channel": 10, "value": -4.5, "current_value": 0.0, "delta_db": -4.5},
            ],
            "reason": "unit",
        },
    )
    proposal = build_decision_queue(server)["pending_actions"][0]
    assert proposal["can_apply"] is True
    assert proposal["requested_change"] is None
    assert proposal["requested_changes"][0]["channel"] == 9

    await server._dispatch["apply_operator_proposal"](
        object(),
        {"proposal_id": proposal["id"], "approved": True, "approval_id": "linked-unit"},
    )

    assert server._apply_manual_console_write.await_count == 2
    payload = server.send_to_client.await_args.args[1]
    assert payload["type"] == "operator_proposal_apply_result"
    assert payload["status"] == "applied"
    assert payload["apply_result"]["linked"] is True
    assert len(payload["apply_result"]["apply_results"]) == 2
    assert build_decision_queue(server)["summary"]["pending_count"] == 0


@pytest.mark.asyncio
async def test_supervised_linked_apply_blocks_when_pre_read_is_missing(server):
    server.operator_mode = "supervised"
    server.mixer_client = MagicMock()
    server.mixer_client.is_connected = True
    server.mixer_client.state = {"/ch/9/fdr": 0.0}
    server._apply_manual_console_write = AsyncMock()
    server.send_to_client = AsyncMock()

    await server._dispatch["create_operator_proposal"](
        object(),
        {
            "title": "GUITAR L/R: linked faders",
            "kind": "linked_fader",
            "requested_changes": [
                {"value_type": "fader", "channel": 9, "value": -4.5, "current_value": 0.0, "delta_db": -4.5},
                {"value_type": "fader", "channel": 10, "value": -4.5, "current_value": 0.0, "delta_db": -4.5},
            ],
            "reason": "unit",
        },
    )
    proposal = build_decision_queue(server)["pending_actions"][0]

    await server._dispatch["apply_operator_proposal"](
        object(),
        {"proposal_id": proposal["id"], "approved": True, "approval_id": "linked-unit"},
    )

    server._apply_manual_console_write.assert_not_called()
    payload = server.send_to_client.await_args.args[1]
    assert payload["type"] == "operator_proposal_apply_blocked"
    assert payload["status"] == "apply_blocked"
    assert payload["apply_result"]["reason"] == "linked_pre_read_unavailable"
    assert payload["apply_result"]["channel"] == 10


@pytest.mark.asyncio
async def test_supervised_linked_apply_rolls_back_first_channel_when_second_fails(server):
    server.operator_mode = "supervised"
    server.mixer_client = MagicMock()
    server.mixer_client.is_connected = True
    server.mixer_client.state = {"/ch/9/fdr": 0.0, "/ch/10/fdr": 0.0}
    server._apply_manual_console_write = AsyncMock(side_effect=[
        {
            "status": "applied",
            "channel": 9,
            "value_type": "fader",
            "effective_value": -4.5,
            "old_value": 0.0,
            "readback_value": -4.5,
            "readback_status": "verified_from_cache",
            "rollback_available": True,
            "rollback_value": 0.0,
            "approval_id": "linked-unit:1",
            "source": "operator_proposal_linked:proposal:1",
        },
        {
            "status": "blocked",
            "reason": "mixer_write_failed",
            "channel": 10,
            "value_type": "fader",
            "rollback_available": False,
        },
        {
            "status": "applied",
            "channel": 9,
            "value_type": "fader",
            "effective_value": 0.0,
            "old_value": -4.5,
            "readback_value": 0.0,
            "readback_status": "verified_from_cache",
            "rollback_available": True,
            "rollback_value": -4.5,
            "approval_id": "rollback:linked-unit:1",
            "source": "operator_proposal_linked_rollback:proposal",
        },
    ])
    server.send_to_client = AsyncMock()

    await server._dispatch["create_operator_proposal"](
        object(),
        {
            "title": "GUITAR L/R: linked faders",
            "kind": "linked_fader",
            "requested_changes": [
                {"value_type": "fader", "channel": 9, "value": -4.5, "current_value": 0.0, "delta_db": -4.5},
                {"value_type": "fader", "channel": 10, "value": -4.5, "current_value": 0.0, "delta_db": -4.5},
            ],
            "reason": "unit",
        },
    )
    proposal = build_decision_queue(server)["pending_actions"][0]

    await server._dispatch["apply_operator_proposal"](
        object(),
        {"proposal_id": proposal["id"], "approved": True, "approval_id": "linked-unit"},
    )

    assert server._apply_manual_console_write.await_count == 3
    rollback_call = server._apply_manual_console_write.await_args_list[2]
    assert rollback_call.kwargs["channel"] == 9
    assert rollback_call.kwargs["value"] == 0.0
    assert rollback_call.kwargs["bypass_cooldown"] is True
    payload = server.send_to_client.await_args.args[1]
    assert payload["type"] == "operator_proposal_apply_blocked"
    assert payload["status"] == "apply_blocked"
    assert payload["apply_result"]["reason"] == "linked_apply_failed_rolled_back"
    assert payload["apply_result"]["rollback_results"][0]["rollback"] is True


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


def test_operator_analysis_creates_gain_trim_down_for_near_clipping(server):
    class ClippingCapture(FakeAudioCapture):
        def get_peak(self, channel):
            return -0.4 if channel == 2 else super().get_peak(channel)

    server.mixer_client = CachedMixer()
    server.audio_capture = ClippingCapture()

    report = build_operator_analysis_report(server, total_channels=4, create_proposals=True)
    queue = build_decision_queue(server)

    assert report["status"] == "ok"
    assert report["observations"][0]["rule"] == "gain_trim_down"
    proposal = queue["pending_actions"][0]
    change = proposal["requested_change"]
    assert proposal["id"] == "analysis:gain_trim_down:2"
    assert proposal["kind"] == "gain"
    assert change["value_type"] == "gain"
    assert change["channel"] == 2
    assert change["current_value"] == 1.0
    assert change["value"] < change["current_value"]
    assert change["rollback_value"] == 1.0
    assert change["readback_required"] is True
    assert any(
        item["label"] == "Direction" and item["value"] == "gain_down_only"
        for item in change["safety_evidence"]
    )


def test_operator_analysis_creates_vocal_level_reduction_proposal(server):
    mixer = CachedMixer()
    mixer.state.update({
        "/ch/16/$name": "VOCAL",
        "/ch/16/fdr": 0.0,
        "/ch/16/mute": 0,
    })
    server.mixer_client = mixer
    server.audio_capture = VocalLevelCapture()

    report = build_operator_analysis_report(server, total_channels=16, create_proposals=True)
    queue = build_decision_queue(server)

    assert report["status"] == "ok"
    assert report["created_count"] == 1
    assert report["observations"][0]["rule"] == "vocal_level_high"
    proposal = queue["pending_actions"][0]
    assert proposal["id"] == "analysis:vocal_level_high:16"
    assert proposal["requested_change"]["value_type"] == "fader"
    assert proposal["requested_change"]["channel"] == 16
    assert proposal["requested_change"]["current_value"] == 0.0
    assert proposal["requested_change"]["delta_db"] < 0
    assert proposal["requested_change"]["value"] < 0.0


def test_operator_analysis_creates_role_level_reduction_proposals_for_sources(server):
    mixer = CachedMixer()
    mixer.state.update({
        "/ch/1/$name": "Kick In",
        "/ch/1/fdr": -2.0,
        "/ch/3/$name": "Snare Top",
        "/ch/3/fdr": -1.0,
        "/ch/10/$name": "Bass DI",
        "/ch/10/fdr": 0.0,
        "/ch/11/$name": "Gtr 1",
        "/ch/11/fdr": -4.0,
    })
    server.mixer_client = mixer
    server.audio_capture = StaticMeterCapture(
        num_channels=12,
        rms={1: -18.0, 3: -18.0, 10: -18.0, 11: -18.0},
        peak={1: -12.0, 3: -12.0, 10: -12.0, 11: -12.0},
        lufs={1: -22.0, 3: -22.0, 10: -22.0, 11: -22.0},
    )

    report = build_operator_analysis_report(server, total_channels=12, create_proposals=True)
    queue = build_decision_queue(server)
    proposals = {proposal["id"]: proposal for proposal in queue["pending_actions"]}

    assert report["status"] == "ok"
    assert report["created_count"] == 4
    assert {observation["rule"] for observation in report["observations"]} == {
        "kick_level_high",
        "snare_level_high",
        "bass_level_high",
        "guitar_level_high",
    }
    assert set(proposals) == {
        "analysis:kick_level_high:1",
        "analysis:snare_level_high:3",
        "analysis:bass_level_high:10",
        "analysis:guitar_level_high:11",
    }
    for proposal in proposals.values():
        assert proposal["requested_change"]["value_type"] == "fader"
        assert proposal["requested_change"]["delta_db"] < 0
        assert proposal["requested_change"]["value"] <= proposal["requested_change"]["current_value"]


def test_operator_analysis_does_not_boost_quiet_sources(server):
    mixer = CachedMixer()
    mixer.state.update({
        "/ch/1/$name": "Kick In",
        "/ch/1/fdr": -2.0,
        "/ch/10/$name": "Bass DI",
        "/ch/10/fdr": 0.0,
    })
    server.mixer_client = mixer
    server.audio_capture = StaticMeterCapture(
        num_channels=10,
        rms={1: -30.0, 10: -30.0},
        peak={1: -20.0, 10: -20.0},
        lufs={1: -32.0, 10: -33.0},
    )

    report = build_operator_analysis_report(server, total_channels=10, create_proposals=True)
    queue = build_decision_queue(server)

    assert report["status"] == "clear"
    assert report["reason"] == "no_operator_analysis_observations"
    assert report["created_count"] == 0
    assert queue["summary"]["pending_count"] == 0


def test_operator_analysis_links_stereo_guitar_pair_into_one_applyable_proposal(server):
    mixer = CachedMixer()
    mixer.state.update({
        "/ch/9/$name": "GUITAR L",
        "/ch/9/fdr": 0.0,
        "/ch/10/$name": "GUITAR R",
        "/ch/10/fdr": 0.0,
    })
    server.mixer_client = mixer
    server.audio_capture = StaticMeterCapture(
        num_channels=10,
        rms={9: -18.0, 10: -18.0},
        peak={9: -9.2, 10: -8.1},
        lufs={9: -20.5, 10: -19.2},
    )

    report = build_operator_analysis_report(server, total_channels=10, create_proposals=True)
    queue = build_decision_queue(server)

    assert report["status"] == "ok"
    assert report["created_count"] == 1
    assert report["observations"][0]["rule"] == "stereo_guitar_level_high"
    proposal = queue["pending_actions"][0]
    assert proposal["id"] == "analysis:stereo_guitar_level_high:9_10"
    assert proposal["kind"] == "linked_fader"
    assert proposal["can_apply"] is True
    assert proposal["requires_approval"] is True
    assert proposal["requested_change"] is None
    assert [change["channel"] for change in proposal["requested_changes"]] == [9, 10]
    assert proposal["requested_changes"][0]["delta_db"] == proposal["requested_changes"][1]["delta_db"]
    assert proposal["requested_changes"][0]["delta_db"] < 0


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
