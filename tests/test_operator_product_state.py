import importlib
import json
import sys
import time
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from backend.automation_trace import AutomationTraceStore
from backend.decision_memory import DecisionMemoryStore, decision_memory_entry_from_trace
from backend.operator_analysis import build_operator_analysis_report
from backend.shadow_gain_rides import build_shadow_gain_ride_candidates, shadow_gain_ride_trace
from backend.operator_product_state import (
    build_agent_communications_snapshot,
    build_bus_main_dca_snapshot,
    build_channel_intelligence,
    build_channel_inventory,
    build_channel_processing_snapshot,
    build_decision_queue,
    build_decision_traces,
    build_groups_buses_snapshot,
    build_matrix_snapshot,
    build_memory_knowledge_snapshot,
    build_operator_agent_roster,
    build_processing_pipeline_snapshot,
    build_replay_runs,
    build_routing_sends_snapshot,
    build_rules_library_snapshot,
    build_safety_monitor,
    build_snapshots_scenes_snapshot,
    build_telemetry_alerts_snapshot,
    build_training_sessions_snapshot,
    build_user_presets_snapshot,
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
        self.send = MagicMock()
        self.get_all_channel_names = MagicMock(return_value={1: "Should Not Scan"})

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
    assert inventory["summary"]["read_coverage_status"] == "partial"
    assert inventory["read_coverage"]["read_only"] is True
    assert inventory["read_coverage"]["mutates_console_state"] is False
    assert inventory["read_coverage"]["section_status_counts"]["metering"]["complete"] == 4
    assert inventory["channels"][0]["name"] == "Kick In"
    assert inventory["channels"][0]["group"] == "DRUMS"
    assert inventory["channels"][1]["name"] == "Lead Vocal"
    assert inventory["channels"][1]["group"] == "VOCALS"
    assert inventory["channels"][1]["control_state"]["fader_db"] == -8.0
    mixer.get_all_channel_names.assert_not_called()


def test_channel_inventory_publishes_cached_channel_detail_readback_only(server):
    mixer = CachedMixer()
    mixer.state.update({
        "/ch/2/mute": 0,
        "/ch/2/pan": 0.25,
        "/ch/2/gate/on": 1,
        "/ch/2/gate/thr": -47.5,
        "/ch/2/dyn/on": 1,
        "/ch/2/dyn/thr26": -18.0,
        "/ch/2/dyn/gr": 2.4,
    })
    server.connection_mode = "wing"
    server.mixer_client = mixer
    server.audio_capture = None

    inventory = build_channel_inventory(server, total_channels=2)
    channel_2 = inventory["channels"][1]
    control = channel_2["control_state"]

    assert control["pan"] == 0.25
    assert control["mute"] == 0
    assert control["gate_enabled"] == 1
    assert control["gate_threshold_db"] == -47.5
    assert control["gate_label"] == "G -47.5"
    assert control["compressor_enabled"] == 1
    assert control["compressor_threshold_db"] == -18.0
    assert control["compressor_gain_reduction_db"] == 2.4
    assert control["comp_label"] == "C 2.4 GR"
    coverage = channel_2["read_coverage"]
    assert coverage["read_only"] is True
    assert coverage["live_write"] is False
    assert coverage["mutates_console_state"] is False
    assert coverage["status"] == "partial"
    assert coverage["sections"]["basic_controls"]["status"] == "complete"
    assert coverage["sections"]["basic_controls"]["source"] == "mixer_state_cache"
    assert coverage["sections"]["processing"]["status"] == "complete"
    assert coverage["sections"]["processing"]["source"] == "mixer_state_cache"
    assert coverage["sections"]["metering"]["status"] == "missing"
    assert "metering" in coverage["missing_sections"]
    assert inventory["read_coverage"]["channels_with_complete_basic_controls"] == 1
    assert inventory["read_coverage"]["channels_with_complete_processing"] == 1
    mixer.send.assert_not_called()
    mixer.get_all_channel_names.assert_not_called()


def test_channel_inventory_marks_missing_cached_read_sections_without_writes(server):
    mixer = CachedMixer()
    mixer.state = {}
    server.connection_mode = "wing"
    server.mixer_client = mixer
    server.audio_capture = None

    inventory = build_channel_inventory(server, total_channels=1)
    channel = inventory["channels"][0]
    coverage = channel["read_coverage"]

    assert channel["name"] == "Ch 1"
    assert channel["control_state"]["fader_db"] is None
    assert coverage["read_only"] is True
    assert coverage["mutates_console_state"] is False
    assert coverage["sections"]["basic_controls"]["status"] == "missing"
    assert coverage["sections"]["basic_controls"]["source"] == "unavailable"
    assert coverage["sections"]["processing"]["status"] == "missing"
    assert coverage["sections"]["metering"]["status"] == "missing"
    assert "mixer_state_cache" not in coverage["source_kinds"]
    mixer.send.assert_not_called()
    mixer.get_all_channel_names.assert_not_called()


def test_channel_processing_snapshot_publishes_cached_processing_readback_only(server):
    mixer = CachedMixer()
    mixer.state.update({
        "/ch/2/flt/lc": 1,
        "/ch/2/flt/lcf": 120.0,
        "/ch/2/eq/on": 1,
        "/ch/2/dyn/on": 1,
        "/ch/2/dyn/mdl": "COMP",
        "/ch/2/dyn/thr26": -18.0,
        "/ch/2/dyn/ratio": "3.0",
        "/ch/2/dyn/att": 12.0,
        "/ch/2/dyn/rel": 180.0,
        "/ch/2/dyn/gain": 2.0,
        "/ch/2/dyn/mix": 80.0,
        "/ch/2/dyn/det": "PEAK",
        "/ch/2/dyn/gr": 2.4,
        "/ch/2/gate/on": 1,
        "/ch/2/gate/mdl": "GATE",
        "/ch/2/gate/thr": -47.5,
        "/ch/2/gate/range": 18.0,
        "/ch/2/gate/att": 4.0,
        "/ch/2/gate/hld": 80.0,
        "/ch/2/gate/rel": 240.0,
        "/ch/2/gate/acc": 3.0,
        "/ch/2/gate/ratio": "4.0",
    })
    for band in range(1, 5):
        mixer.state[f"/ch/2/eq/{band}f"] = 100.0 * band
        mixer.state[f"/ch/2/eq/{band}g"] = -1.0 * band
        mixer.state[f"/ch/2/eq/{band}q"] = 0.7 + band / 10
    server.connection_mode = "wing"
    server.mixer_client = mixer
    server.audio_capture = None

    snapshot = build_channel_processing_snapshot(server, total_channels=2)
    channel_2 = snapshot["channels"][1]

    assert snapshot["type"] == "channel_processing_snapshot"
    assert snapshot["safety"] == {
        "read_only": True,
        "live_write": False,
        "mutates_console_state": False,
        "queries_console": False,
        "full_autonomous_ready": False,
    }
    assert snapshot["summary"]["read_coverage_status"] == "partial"
    assert snapshot["summary"]["channels_with_complete_processing_cache"] == 1
    assert channel_2["name"] == "Lead Vocal"
    assert channel_2["hpf"]["enabled"] == 1
    assert channel_2["hpf"]["frequency_hz"] == 120.0
    assert channel_2["eq"]["enabled"] == 1
    assert channel_2["eq"]["bands"][3]["frequency_hz"] == 400.0
    assert channel_2["eq"]["bands"][3]["gain_db"] == -4.0
    assert channel_2["compressor"]["model"] == "COMP"
    assert channel_2["compressor"]["threshold_db"] == -18.0
    assert channel_2["compressor"]["gain_reduction_db"] == 2.4
    assert channel_2["gate"]["threshold_db"] == -47.5
    assert channel_2["read_coverage"]["status"] == "complete"
    assert channel_2["read_coverage"]["sections"]["hpf"]["status"] == "complete"
    assert channel_2["read_coverage"]["sections"]["eq"]["status"] == "complete"
    assert channel_2["read_coverage"]["sections"]["compressor"]["status"] == "complete"
    assert channel_2["read_coverage"]["sections"]["gate"]["status"] == "complete"
    mixer.send.assert_not_called()
    mixer.get_all_channel_names.assert_not_called()


def test_channel_processing_snapshot_marks_missing_cache_without_queries(server):
    mixer = CachedMixer()
    mixer.state = {}
    server.connection_mode = "wing"
    server.mixer_client = mixer
    server.audio_capture = None

    snapshot = build_channel_processing_snapshot(server, total_channels=1)
    channel = snapshot["channels"][0]

    assert snapshot["summary"]["read_coverage_status"] == "missing"
    assert snapshot["summary"]["channels_with_any_processing_cache"] == 0
    assert channel["hpf"]["enabled"] is None
    assert channel["eq"]["bands"][0]["read_coverage"]["status"] == "missing"
    assert channel["read_coverage"]["status"] == "missing"
    assert channel["read_coverage"]["missing_sections"] == ["hpf", "eq", "compressor", "gate"]
    mixer.send.assert_not_called()
    mixer.get_all_channel_names.assert_not_called()


def test_bus_main_dca_snapshot_publishes_cached_console_surfaces_without_queries(server):
    mixer = CachedMixer()
    mixer.state.update({
        "/bus/1/$name": "Vocal Bus",
        "/bus/1/mute": 0,
        "/bus/1/fdr": -4.0,
        "/bus/1/pan": 0.0,
        "/bus/1/wid": 100.0,
        "/bus/1/busmono": 0,
        "/bus/1/eq/on": 1,
        "/bus/1/dyn/on": 1,
        "/bus/1/dyn/mdl": "BUSCOMP",
        "/bus/1/dyn/thr37": -20.0,
        "/bus/1/dyn/ratio": "2.0",
        "/bus/1/dyn/knee": 3.0,
        "/bus/1/dyn/det": "RMS",
        "/bus/1/dyn/att": 20.0,
        "/bus/1/dyn/rel": 250.0,
        "/bus/1/dyn/gain": 1.5,
        "/bus/1/dyn/mix": 75.0,
        "/bus/1/main/1/on": 1,
        "/bus/1/main/1/lvl": -2.0,
        "/bus/1/main/1/pre": 0,
        "/bus/1/tags": "#D1",
        "/main/1/fdr": -2.5,
        "/main/1/mute": 0,
        "/main/1/pan": 0.0,
        "/main/1/eq/on": 1,
        "/dca/1/name": "Vocals",
        "/dca/1/mute": 0,
        "/dca/1/fdr": -3.0,
        "/dca/1/col": "purple",
        "/dca/1/icon": "mic",
        "/dca/1/mon": 0,
    })
    for band in range(1, 9):
        mixer.state[f"/bus/1/eq/{band}f"] = 125.0 * band
        mixer.state[f"/bus/1/eq/{band}g"] = -0.5 * band
        mixer.state[f"/bus/1/eq/{band}q"] = 0.8 + band / 10
        mixer.state[f"/main/1/eq/{band}f"] = 100.0 * band
        mixer.state[f"/main/1/eq/{band}g"] = -0.25 * band
        mixer.state[f"/main/1/eq/{band}q"] = 0.7 + band / 10
    server.connection_mode = "wing"
    server.mixer_client = mixer

    snapshot = build_bus_main_dca_snapshot(server, bus_count=1, main_count=1, dca_count=1)
    bus = snapshot["buses"][0]
    main = snapshot["mains"][0]
    dca = snapshot["dcas"][0]

    assert snapshot["type"] == "bus_main_dca_snapshot"
    assert snapshot["safety"] == {
        "read_only": True,
        "live_write": False,
        "mutates_console_state": False,
        "queries_console": False,
        "full_autonomous_ready": False,
    }
    assert snapshot["summary"]["read_coverage_status"] == "complete"
    assert bus["name"] == "Vocal Bus"
    assert bus["fader_db"] == -4.0
    assert bus["eq"]["bands"][7]["frequency_hz"] == 1000.0
    assert bus["dynamics"]["threshold_db"] == -20.0
    assert bus["main_send"]["level_db"] == -2.0
    assert bus["dca_assignments"] == [1]
    assert bus["read_coverage"]["status"] == "complete"
    assert main["fader_db"] == -2.5
    assert main["read_coverage"]["status"] == "complete"
    assert dca["name"] == "Vocals"
    assert dca["read_coverage"]["status"] == "complete"
    mixer.send.assert_not_called()
    mixer.get_all_channel_names.assert_not_called()


def test_bus_main_dca_snapshot_marks_missing_cache_without_queries(server):
    mixer = CachedMixer()
    mixer.state = {}
    server.connection_mode = "wing"
    server.mixer_client = mixer

    snapshot = build_bus_main_dca_snapshot(server, bus_count=1, main_count=1, dca_count=1)

    assert snapshot["summary"]["read_coverage_status"] == "missing"
    assert snapshot["read_coverage"]["surface_status_counts"]["buses"]["missing"] == 1
    assert snapshot["read_coverage"]["surface_status_counts"]["mains"]["missing"] == 1
    assert snapshot["read_coverage"]["surface_status_counts"]["dcas"]["missing"] == 1
    assert snapshot["buses"][0]["read_coverage"]["missing_sections"] == [
        "identity",
        "basic_controls",
        "eq",
        "dynamics",
        "main_send",
        "assignments",
    ]
    assert snapshot["mains"][0]["read_coverage"]["status"] == "missing"
    assert snapshot["dcas"][0]["read_coverage"]["status"] == "missing"
    mixer.send.assert_not_called()
    mixer.get_all_channel_names.assert_not_called()


def test_matrix_snapshot_publishes_cached_console_state_without_queries(server):
    mixer = CachedMixer()
    mixer.state.update({
        "/mtx/1/fdr": -5.0,
        "/mtx/1/mute": 0,
        "/mtx/1/pan": 0.0,
        "/mtx/1/eq/on": 1,
        "/mtx/1/preins/on": 1,
        "/mtx/1/preins/ins": "FX1",
        "/mtx/1/preins/$stat": "OK",
        "/mtx/1/postins/on": 1,
        "/mtx/1/postins/ins": "FX2",
        "/mtx/1/postins/$stat": "OK",
        "/mtx/1/postins/mode": "insert",
        "/mtx/1/postins/w": 100.0,
    })
    for band in range(1, 9):
        mixer.state[f"/mtx/1/eq/{band}f"] = 160.0 * band
        mixer.state[f"/mtx/1/eq/{band}g"] = -0.25 * band
        mixer.state[f"/mtx/1/eq/{band}q"] = 0.7 + band / 10
    server.connection_mode = "wing"
    server.mixer_client = mixer

    snapshot = build_matrix_snapshot(server, matrix_count=1)
    matrix = snapshot["matrices"][0]

    assert snapshot["type"] == "matrix_snapshot"
    assert snapshot["safety"] == {
        "read_only": True,
        "live_write": False,
        "mutates_console_state": False,
        "queries_console": False,
        "full_autonomous_ready": False,
    }
    assert snapshot["summary"]["read_coverage_status"] == "complete"
    assert matrix["fader_db"] == -5.0
    assert matrix["eq"]["bands"][7]["frequency_hz"] == 1280.0
    assert matrix["inserts"]["pre_insert"]["slot"] == "FX1"
    assert matrix["inserts"]["post_insert"]["mode"] == "insert"
    assert matrix["read_coverage"]["status"] == "complete"
    assert matrix["read_coverage"]["sections"]["basic_controls"]["status"] == "complete"
    assert matrix["read_coverage"]["sections"]["eq"]["status"] == "complete"
    assert matrix["read_coverage"]["sections"]["inserts"]["status"] == "complete"
    mixer.send.assert_not_called()
    mixer.get_all_channel_names.assert_not_called()


def test_matrix_snapshot_marks_missing_cache_without_queries(server):
    mixer = CachedMixer()
    mixer.state = {}
    server.connection_mode = "wing"
    server.mixer_client = mixer

    snapshot = build_matrix_snapshot(server, matrix_count=1)
    matrix = snapshot["matrices"][0]

    assert snapshot["summary"]["read_coverage_status"] == "missing"
    assert snapshot["read_coverage"]["surface_status_counts"]["matrices"]["missing"] == 1
    assert matrix["read_coverage"]["missing_sections"] == ["basic_controls", "eq", "inserts"]
    assert matrix["inserts"]["pre_insert"]["read_coverage"]["status"] == "missing"
    assert matrix["inserts"]["post_insert"]["read_coverage"]["status"] == "missing"
    mixer.send.assert_not_called()
    mixer.get_all_channel_names.assert_not_called()


def test_routing_sends_snapshot_publishes_static_and_cached_state_without_queries(server):
    mixer = CachedMixer()
    mixer.state.update({
        "/ch/2/send/1/on": 1,
        "/ch/2/send/1/lvl": -12.0,
        "/ch/2/send/1/pon": 1,
        "/ch/2/send/1/mode": "PRE",
        "/ch/2/send/1/plink": 0,
        "/ch/2/send/1/pan": 0.0,
        "/ch/2/main/1/on": 1,
        "/ch/2/main/1/lvl": -3.0,
        "/ch/2/main/1/pre": 0,
        "/io/out/MOD/1/grp": "CH",
        "/io/out/MOD/1/in": 2,
    })
    server.connection_mode = "wing"
    server.mixer_client = mixer

    snapshot = build_routing_sends_snapshot(
        server,
        total_channels=2,
        send_count=1,
        main_count=1,
        output_groups=["MOD"],
        output_count=1,
    )

    assert snapshot["type"] == "routing_sends_snapshot"
    assert snapshot["safety"] == {
        "read_only": True,
        "live_write": False,
        "mutates_console_state": False,
        "queries_console": False,
        "full_autonomous_ready": False,
    }
    assert snapshot["summary"]["read_coverage_status"] == "partial"
    assert snapshot["summary"]["active_bus_send_count"] == 1
    assert snapshot["summary"]["active_main_send_count"] == 1
    assert snapshot["static_dante_routes"][0]["read_coverage"]["source"] == "static_config"
    send = next(row for row in snapshot["channel_bus_sends"] if row["channel"] == 2)
    main_send = next(row for row in snapshot["channel_main_sends"] if row["channel"] == 2)
    route = snapshot["output_routes"][0]
    assert send["active"] is True
    assert send["read_coverage"]["status"] == "complete"
    assert main_send["active"] is True
    assert main_send["read_coverage"]["status"] == "complete"
    assert route["source_group"] == "CH"
    assert route["source_channel"] == 2
    assert route["read_coverage"]["status"] == "complete"
    assert route["destructive_change_marker"] == "output_routing_write_would_repatch_console"
    mixer.send.assert_not_called()
    mixer.get_all_channel_names.assert_not_called()


def test_routing_sends_snapshot_marks_missing_cache_without_queries(server):
    mixer = CachedMixer()
    mixer.state = {}
    server.connection_mode = "wing"
    server.mixer_client = mixer

    snapshot = build_routing_sends_snapshot(
        server,
        total_channels=1,
        send_count=1,
        main_count=1,
        output_groups=["MOD"],
        output_count=1,
    )

    assert snapshot["summary"]["read_coverage_status"] == "partial"
    assert snapshot["read_coverage"]["surface_status_counts"]["static_dante_routes"]["complete"] >= 1
    assert snapshot["read_coverage"]["surface_status_counts"]["channel_bus_sends"]["missing"] == 1
    assert snapshot["read_coverage"]["surface_status_counts"]["channel_main_sends"]["missing"] == 1
    assert snapshot["read_coverage"]["surface_status_counts"]["output_routes"]["missing"] == 1
    assert snapshot["channel_bus_sends"][0]["read_coverage"]["status"] == "missing"
    assert snapshot["channel_main_sends"][0]["read_coverage"]["status"] == "missing"
    assert snapshot["output_routes"][0]["read_coverage"]["status"] == "missing"
    mixer.send.assert_not_called()
    mixer.get_all_channel_names.assert_not_called()


def test_snapshots_scenes_snapshot_publishes_cached_catalog_without_queries(server):
    mixer = CachedMixer()
    mixer.state.update({
        "snapshots": {
            1: {"name": "Pre-show baseline"},
            "2": "Band A live start",
        },
        "/$ctl/lib/$actidx": 2,
        "/$ctl/lib/$active": "Band A live start",
        "/$ctl/lib/$actshow": "Sunday Show",
    })
    server.connection_mode = "wing"
    server.mixer_client = mixer

    snapshot = build_snapshots_scenes_snapshot(server, snapshot_count=3)
    active = snapshot["snapshots"][1]

    assert snapshot["type"] == "snapshots_scenes_snapshot"
    assert snapshot["safety"] == {
        "read_only": True,
        "live_write": False,
        "mutates_console_state": False,
        "queries_console": False,
        "full_autonomous_ready": False,
        "scene_load_allowed": False,
        "scene_save_allowed": False,
    }
    assert snapshot["summary"]["snapshot_count"] == 3
    assert snapshot["summary"]["named_snapshot_count"] == 2
    assert snapshot["summary"]["active_snapshot_index"] == 2
    assert snapshot["summary"]["active_snapshot_name"] == "Band A live start"
    assert snapshot["summary"]["current_show"] == "Sunday Show"
    assert snapshot["summary"]["read_coverage_status"] == "partial"
    assert snapshot["summary"]["live_apply_allowed"] is False
    assert snapshot["active_scene"]["read_coverage"]["status"] == "complete"
    assert active["snapshot_index"] == 2
    assert active["snapshot_name"] == "Band A live start"
    assert active["active"] is True
    assert active["scene_load_policy"] == "blocked_manual_only"
    assert active["scene_save_policy"] == "blocked_manual_only"
    assert active["destructive_change_marker"] == "scene_recall_would_overwrite_console_state"
    assert snapshot["snapshots"][2]["read_coverage"]["status"] == "partial"
    mixer.send.assert_not_called()
    mixer.get_all_channel_names.assert_not_called()


def test_snapshots_scenes_snapshot_marks_missing_cache_without_queries(server):
    mixer = CachedMixer()
    mixer.state = {}
    server.connection_mode = "wing"
    server.mixer_client = mixer

    snapshot = build_snapshots_scenes_snapshot(server, snapshot_count=2)

    assert snapshot["summary"]["read_coverage_status"] == "missing"
    assert snapshot["summary"]["named_snapshot_count"] == 0
    assert snapshot["summary"]["active_snapshot_index"] is None
    assert snapshot["active_scene"]["read_coverage"]["status"] == "missing"
    assert snapshot["read_coverage"]["surface_status_counts"]["snapshot_slots"]["missing"] == 2
    assert snapshot["snapshots"][0]["snapshot_name"] is None
    assert snapshot["snapshots"][0]["read_coverage"]["status"] == "missing"
    assert snapshot["snapshots"][0]["scene_load_policy"] == "blocked_manual_only"
    assert snapshot["snapshots"][0]["scene_save_policy"] == "blocked_manual_only"
    mixer.send.assert_not_called()
    mixer.get_all_channel_names.assert_not_called()


def test_user_presets_snapshot_publishes_cached_catalog_without_queries(server):
    mixer = CachedMixer()
    mixer.state.update({
        "user_presets": {
            1: {
                "name": "Lead Vocal Strip",
                "scope": "channel_strip",
                "target_type": "channel",
                "target": "CH 2",
            },
            "2": "Drum Bus Glue",
        },
    })
    server.connection_mode = "wing"
    server.mixer_client = mixer

    snapshot = build_user_presets_snapshot(server, preset_count=3)
    preset = snapshot["presets"][0]

    assert snapshot["type"] == "user_presets_snapshot"
    assert snapshot["safety"] == {
        "read_only": True,
        "live_write": False,
        "mutates_console_state": False,
        "queries_console": False,
        "full_autonomous_ready": False,
        "preset_recall_allowed": False,
        "preset_store_allowed": False,
    }
    assert snapshot["summary"]["preset_count"] == 3
    assert snapshot["summary"]["named_preset_count"] == 2
    assert snapshot["summary"]["read_coverage_status"] == "partial"
    assert snapshot["summary"]["live_apply_allowed"] is False
    assert preset["preset_index"] == 1
    assert preset["preset_name"] == "Lead Vocal Strip"
    assert preset["preset_scope"] == "channel_strip"
    assert preset["target_type"] == "channel"
    assert preset["target_label"] == "CH 2"
    assert preset["preset_recall_policy"] == "blocked_manual_only"
    assert preset["preset_store_policy"] == "blocked_manual_only"
    assert preset["destructive_recall_marker"] == "preset_recall_would_change_console_parameters"
    assert preset["read_coverage"]["status"] == "complete"
    assert snapshot["presets"][1]["read_coverage"]["status"] == "partial"
    assert snapshot["presets"][2]["read_coverage"]["status"] == "missing"
    mixer.send.assert_not_called()
    mixer.get_all_channel_names.assert_not_called()


def test_user_presets_snapshot_marks_missing_cache_without_queries(server):
    mixer = CachedMixer()
    mixer.state = {}
    server.connection_mode = "wing"
    server.mixer_client = mixer

    snapshot = build_user_presets_snapshot(server, preset_count=2)

    assert snapshot["summary"]["read_coverage_status"] == "missing"
    assert snapshot["summary"]["named_preset_count"] == 0
    assert snapshot["read_coverage"]["surface_status_counts"]["user_preset_slots"]["missing"] == 2
    assert snapshot["presets"][0]["preset_name"] is None
    assert snapshot["presets"][0]["read_coverage"]["status"] == "missing"
    assert snapshot["presets"][0]["preset_recall_policy"] == "blocked_manual_only"
    assert snapshot["presets"][0]["preset_store_policy"] == "blocked_manual_only"
    mixer.send.assert_not_called()
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
        build_channel_processing_snapshot(server, total_channels=4),
        build_bus_main_dca_snapshot(server, bus_count=2, main_count=1, dca_count=2),
        build_matrix_snapshot(server, matrix_count=2),
        build_routing_sends_snapshot(
            server,
            total_channels=2,
            send_count=1,
            output_groups=["MOD"],
            output_count=1,
        ),
        build_snapshots_scenes_snapshot(server, snapshot_count=2),
        build_user_presets_snapshot(server, preset_count=2),
        build_groups_buses_snapshot(server),
        build_memory_knowledge_snapshot(server),
        build_training_sessions_snapshot(server),
        build_rules_library_snapshot(server),
        build_agent_communications_snapshot(server),
    ]

    assert [snapshot["type"] for snapshot in snapshots] == [
        "channel_processing_snapshot",
        "bus_main_dca_snapshot",
        "matrix_snapshot",
        "routing_sends_snapshot",
        "snapshots_scenes_snapshot",
        "user_presets_snapshot",
        "groups_buses_snapshot",
        "memory_knowledge_snapshot",
        "training_sessions_snapshot",
        "rules_library_snapshot",
        "agent_communications_snapshot",
    ]
    assert snapshots[0]["safety"]["live_write"] is False
    assert snapshots[1]["safety"]["queries_console"] is False
    assert snapshots[2]["safety"]["queries_console"] is False
    assert snapshots[3]["safety"]["queries_console"] is False
    assert snapshots[4]["safety"]["queries_console"] is False
    assert snapshots[4]["safety"]["scene_load_allowed"] is False
    assert snapshots[5]["safety"]["queries_console"] is False
    assert snapshots[5]["safety"]["preset_recall_allowed"] is False
    assert snapshots[6]["buses"]
    assert snapshots[7]["memories"]
    assert snapshots[8]["sessions"]
    assert snapshots[9]["rules"]
    assert snapshots[10]["communications"]

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


def test_foundation_read_model_snapshots_are_display_only(server, tmp_path):
    server.mixer_client = CachedMixer()
    server.connection_mode = "wing"
    server.audio_capture = FakeAudioCapture()
    server.automation_trace_store = AutomationTraceStore(tmp_path)
    server.record_automation_decision_trace(
        source="unit",
        mode="dry_run",
        target=2,
        action_type="fader",
        status="blocked",
        current_value=-2.5,
        proposed_value=-1.0,
        delta=1.5,
        blocked_reason="unit_blocked",
    )

    snapshots = [
        build_decision_traces(server),
        build_safety_monitor(server),
        build_replay_runs(server),
        build_channel_intelligence(server, total_channels=4),
    ]

    assert [snapshot["type"] for snapshot in snapshots] == [
        "decision_traces",
        "safety_monitor",
        "replay_runs",
        "channel_intelligence",
    ]
    assert snapshots[0]["summary"]["blocked_count"] == 1
    assert snapshots[1]["live_write_policy"]["direct_writes_allowed"] is False
    assert snapshots[2]["summary"]["offline_only"] is True
    assert snapshots[2]["summary"]["live_write"] is False
    assert snapshots[3]["summary"]["shadow_only"] is True
    assert snapshots[3]["summary"]["live_apply_allowed"] is False

    forbidden_direct_command_keys = {
        "apply_operator_proposal",
        "load_snap",
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
    }
    for snapshot in snapshots:
        assert not (set(_walk_keys(snapshot)) & forbidden_direct_command_keys)


def test_decision_memory_accepts_verified_and_offline_evidence_only():
    verified = decision_memory_entry_from_trace({
        "decision_id": "trace-verified",
        "source": "operator",
        "mode": "supervised",
        "target": 2,
        "action_type": "fader",
        "status": "applied",
        "proof_status": "verified",
    })
    offline = decision_memory_entry_from_trace({
        "decision_id": "trace-offline",
        "source": "offline_replay",
        "mode": "offline",
        "target": 2,
        "action_type": "gain",
        "status": "simulated",
        "proof_status": "advisory",
    })
    shadow = decision_memory_entry_from_trace({
        "decision_id": "trace-shadow",
        "source": "shadow_gain_ride_v0",
        "mode": "assist",
        "target": 2,
        "action_type": "gain_ride_shadow",
        "status": "shadow_proposed",
        "proof_status": "shadow_only",
    })

    assert verified is not None
    assert verified["evidence_quality"] == "verified"
    assert offline is not None
    assert offline["evidence_quality"] == "offline"
    assert shadow is None


def test_verified_traces_feed_decision_memory_snapshot(server, tmp_path):
    server.automation_trace_store = AutomationTraceStore(tmp_path)
    server.decision_memory_store = DecisionMemoryStore(tmp_path)

    server.record_automation_decision_trace(
        source="unit",
        mode="supervised",
        target=2,
        action_type="fader",
        status="applied",
        current_value=-8.0,
        proposed_value=-8.5,
        delta=-0.5,
        proof_status="verified",
    )
    server.record_automation_decision_trace(
        source="shadow_gain_ride_v0",
        mode="assist",
        target=2,
        action_type="gain_ride_shadow",
        status="shadow_proposed",
        delta=0.5,
        proof_status="shadow_only",
    )

    entries = server.get_decision_memory_entries()
    snapshot = build_memory_knowledge_snapshot(server)

    assert len(entries) == 1
    assert entries[0]["source_decision_id"]
    assert entries[0]["evidence_quality"] == "verified"
    assert snapshot["summary"]["decision_memory_records"] == 1
    assert snapshot["memories"][0]["type"] == "Verified Decision Evidence"
    assert "gain_ride_shadow" not in snapshot["memories"][0]["name"]


@pytest.mark.asyncio
async def test_start_replay_run_records_offline_evidence_without_live_gate(server, tmp_path):
    artifact_root = tmp_path / "replay"
    trace_root = tmp_path / "trace"
    server._replay_artifacts_dir = artifact_root
    server.mixer_client = CachedMixer()
    server.audio_capture = FakeAudioCapture()
    server.automation_trace_store = AutomationTraceStore(trace_root)
    server.decision_memory_store = DecisionMemoryStore(trace_root)
    server._apply_manual_console_write = AsyncMock()
    server.send_to_client = AsyncMock()

    await server._dispatch["start_replay_run"](
        object(),
        {"run_id": "unit-replay", "total_channels": 8},
    )

    server._apply_manual_console_write.assert_not_called()
    payload = server.send_to_client.await_args.args[1]
    assert payload["type"] == "replay_run_started"
    assert payload["offline_only"] is True
    assert payload["live_write"] is False
    assert payload["run_id"] == "unit-replay"
    assert payload["foundation_teacher_example_count"] >= 1
    assert (artifact_root / "unit-replay" / "summary.json").exists()
    assert (artifact_root / "unit-replay" / "decision_traces.jsonl").exists()
    assert (artifact_root / "unit-replay" / "foundation_teacher_examples.jsonl").exists()
    teacher_rows = (artifact_root / "unit-replay" / "foundation_teacher_examples.jsonl").read_text(encoding="utf-8").strip().splitlines()
    assert teacher_rows

    traces = server.get_automation_decision_traces()
    assert traces[-1]["source"] == "offline_replay_v1"
    assert traces[-1]["action_type"] == "replay_run"
    assert traces[-1]["proof_status"] == "offline"
    assert traces[-1]["safety_limits"]["mixer_transport_allowed"] is False
    assert traces[-1]["metadata"]["live_write"] is False
    assert traces[-1]["metadata"]["foundation_teacher_example_count"] >= 1

    memory_entries = server.get_decision_memory_entries()
    assert memory_entries[-1]["evidence_quality"] == "offline"
    assert memory_entries[-1]["action_type"] == "replay_run"

    replay_runs = build_replay_runs(server)
    assert replay_runs["runs"][0]["run_id"] == "unit-replay"
    assert replay_runs["runs"][0]["trace_count"] >= 1
    assert replay_runs["runs"][0]["foundation_teacher_example_count"] >= 1
    assert replay_runs["runs"][0]["offline_only"] is True
    assert replay_runs["runs"][0]["live_write"] is False


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
    goal_matrix = payload["project_goal_capability_matrix"]
    assert goal_matrix["type"] == "project_goal_capability_matrix"
    assert goal_matrix["safety"]["live_write"] is False
    assert goal_matrix["safety"]["live_widening_allowed"] is False
    assert "mode_full_autonomous" in goal_matrix["summary"]["blocked_live_capability_ids"]
    assert "console_fader_control" in goal_matrix["summary"][
        "currently_live_writable_capability_ids"
    ]
    assert any(profile["id"] == "quarantine" for profile in payload["pipeline_profiles"])
    assert any(method["status"] == "placeholder" for method in payload["methods"])
    assert all(
        method["can_write_live"] is False
        for method in payload["methods"]
        if method["status"] in {"placeholder", "quarantined", "deprecated"}
    )
    module_map = {row["module_id"]: row for row in payload["module_status_map"]}
    assert module_map["product_operator_state"]["ui_surface"].startswith("Control Center")
    assert module_map["supervised_manual_fader_gain"]["entrypoint"] == "apply_operator_proposal -> supervised manual gate"
    assert module_map["supervised_manual_fader_gain"]["can_write_live"] is True
    assert module_map["supervised_manual_fader_gain"]["live_write_kinds"] == ["fader", "gain"]
    assert module_map["auto_eq"]["can_write_live"] is False
    assert module_map["auto_eq"]["next_step"].startswith("proposal_adapter") or "ADR" in module_map["auto_eq"]["next_step"]
    assert module_map["automixer_production_mix_v1"]["entrypoint"] == "none in this checkout"
    assert module_map["ai_mixing_pipeline_legacy"]["status"] == "deprecated"


def test_telemetry_alerts_snapshot_reads_existing_artifacts_without_generating(server, tmp_path):
    storage_dir = tmp_path / "sessions"
    alerts_dir = storage_dir / "session-a" / "alerts"
    alerts_dir.mkdir(parents=True)
    (alerts_dir / "summary.json").write_text(
        json.dumps(
            {
                "session_id": "session-a",
                "generated_at": "2026-06-01T10:00:00Z",
                "event_count": 4,
                "alert_count": 2,
                "dashboard_row_count": 4,
                "dashboard_path": str(alerts_dir / "dashboard.csv"),
                "safety": {"read_only": True, "live_write": False},
            }
        ),
        encoding="utf-8",
    )
    (alerts_dir / "alerts.jsonl").write_text(
        "\n".join(
            [
                json.dumps(
                    {
                        "timestamp_ms": 1000,
                        "session_id": "session-a",
                        "rule": "headroom_low",
                        "severity": "warning",
                        "target_type": "bus",
                        "target_id": "Main",
                    }
                ),
                json.dumps(
                    {
                        "timestamp_ms": 2000,
                        "session_id": "session-a",
                        "rule": "clipping_risk",
                        "severity": "critical",
                        "target_type": "channel",
                        "target_id": "2",
                    }
                ),
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    server.config["telemetry"] = {"storage_dir": str(storage_dir)}

    payload = build_telemetry_alerts_snapshot(server, limit=1)

    assert payload["type"] == "telemetry_alerts_snapshot"
    assert payload["available"] is True
    assert payload["safety"] == {
        "read_only": True,
        "live_write": False,
        "generates_artifacts": False,
    }
    assert payload["summary"]["artifact_session_count"] == 1
    assert payload["summary"]["alert_count"] == 2
    assert payload["summary"]["critical_count"] == 1
    assert payload["summary"]["warning_count"] == 1
    assert payload["sessions"][0]["session_id"] == "session-a"
    assert len(payload["alerts"]) == 1
    assert payload["alerts"][0]["rule"] == "clipping_risk"


def test_telemetry_alerts_snapshot_handles_malformed_limits(server, tmp_path):
    storage_dir = tmp_path / "sessions"
    alerts_dir = storage_dir / "session-a" / "alerts"
    alerts_dir.mkdir(parents=True)
    (alerts_dir / "summary.json").write_text(
        json.dumps({"session_id": "session-a", "alert_count": 1}),
        encoding="utf-8",
    )
    (alerts_dir / "alerts.jsonl").write_text(
        json.dumps(
            {
                "timestamp_ms": 1000,
                "rule": "headroom_low",
                "severity": "warning",
            }
        )
        + "\n",
        encoding="utf-8",
    )
    server.config["telemetry"] = {"storage_dir": str(storage_dir)}

    payload = build_telemetry_alerts_snapshot(
        server,
        limit="not-a-number",
        session_limit="also-not-a-number",
    )

    assert payload["type"] == "telemetry_alerts_snapshot"
    assert payload["available"] is True
    assert payload["safety"]["live_write"] is False
    assert payload["summary"]["artifact_session_count"] == 1
    assert payload["alerts"][0]["rule"] == "headroom_low"


def test_telemetry_alerts_snapshot_normalizes_untrusted_summary_safety(server, tmp_path):
    storage_dir = tmp_path / "sessions"
    alerts_dir = storage_dir / "session-a" / "alerts"
    alerts_dir.mkdir(parents=True)
    (alerts_dir / "summary.json").write_text(
        json.dumps(
            {
                "session_id": "session-a",
                "alert_count": 1,
                "dashboard_path": "/tmp/not-automixer/dashboard.csv",
                "safety": {"read_only": False, "live_write": True},
            }
        ),
        encoding="utf-8",
    )
    (alerts_dir / "alerts.jsonl").write_text(
        json.dumps(
            {
                "timestamp_ms": 1000,
                "rule": "clipping_risk",
                "severity": "critical",
                "set_gain": {"channel": 12, "value": 3.0},
                "context": {
                    "peak_dbfs": -0.05,
                    "route_output": "main",
                    "action_type": "fader_move",
                },
            }
        )
        + "\n",
        encoding="utf-8",
    )
    server.config["telemetry"] = {"storage_dir": str(storage_dir)}

    payload = build_telemetry_alerts_snapshot(server)

    assert payload["safety"] == {
        "read_only": True,
        "live_write": False,
        "generates_artifacts": False,
    }
    assert payload["sessions"][0]["dashboard_path"] == str(alerts_dir / "dashboard.csv")
    assert payload["sessions"][0]["safety"] == {
        "read_only": True,
        "live_write": False,
        "generates_artifacts": False,
        "source_live_write": True,
        "source_safety_normalized": True,
    }
    assert payload["alerts"][0]["context"] == {
        "action_type": "fader_move",
        "peak_dbfs": -0.05,
    }
    assert "set_gain" not in set(_walk_keys(payload))
    assert "route_output" not in set(_walk_keys(payload))


def test_telemetry_alerts_snapshot_marks_storage_without_alert_artifacts_unavailable(server, tmp_path):
    storage_dir = tmp_path / "sessions"
    session_dir = storage_dir / "session-without-alerts"
    session_dir.mkdir(parents=True)
    (session_dir / "channel_telemetry.jsonl").write_text(
        json.dumps(
            {
                "timestamp_ms": 1000,
                "session_id": "session-without-alerts",
                "channel_id": 1,
                "meter_peak_dbfs": -12.0,
            }
        )
        + "\n",
        encoding="utf-8",
    )
    server.config["telemetry"] = {"storage_dir": str(storage_dir)}

    payload = build_telemetry_alerts_snapshot(server)

    assert payload["type"] == "telemetry_alerts_snapshot"
    assert payload["available"] is False
    assert payload["summary"]["session_count"] == 1
    assert payload["summary"]["artifact_session_count"] == 0
    assert payload["summary"]["alert_count"] == 0
    assert payload["sessions"] == []
    assert payload["alerts"] == []
    assert payload["safety"] == {
        "read_only": True,
        "live_write": False,
        "generates_artifacts": False,
    }


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
    assert payload["summary"]["proposal_count"] == 2
    assert payload["summary"]["operator_analysis_proposal_count"] == 1
    assert payload["summary"]["agentic_foundation_proposal_count"] == 1
    assert queue["summary"]["pending_count"] == 0
    assert payload["proposals"][0]["adapter_id"] == "operator_analysis_gain_fader"
    assert payload["proposals"][0]["requested_change"]["value_type"] == "fader"
    assert payload["proposals"][0]["requested_change"]["channel"] == 2
    agentic_proposals = [
        proposal
        for proposal in payload["proposals"]
        if proposal["adapter_id"] == "agentic_foundation_gain_fader"
    ]
    assert len(agentic_proposals) == 1
    assert agentic_proposals[0]["method_id"] == "agentic_method_foundation"
    assert agentic_proposals[0]["queue_action"] == "preview_only_not_queued"
    assert agentic_proposals[0]["can_import_to_queue"] is False
    assert agentic_proposals[0]["requested_change"]["value_type"] == "fader"
    assert agentic_proposals[0]["requested_change"]["channel"] == 2
    assert agentic_proposals[0]["requested_change"]["delta_db"] < 0
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


def test_main_pipeline_preview_exposes_non_fader_gain_proposal_contracts(server):
    payload = build_main_pipeline_preview(server, total_channels=4)
    contracts = {item["method_id"]: item for item in payload["proposal_only_contracts"]}

    for method_id, value_type in {
        "auto_eq": "eq",
        "auto_compressor": "compressor",
        "auto_gate_caig": "gate",
        "auto_panner": "pan",
        "phase_alignment": "phase_delay",
        "auto_fx": "fx_send",
        "routing_snapshot_reset_writes": "routing",
    }.items():
        contract = contracts[method_id]
        assert contract["adapter_id"] == "proposal_only_contracts"
        assert contract["queue_action"] == "not_queueable_until_promoted"
        assert contract["can_import_to_queue"] is False
        assert contract["can_apply_live"] is False
        assert contract["proposal_contract"]["value_type"] == value_type
        assert contract["proposal_contract"]["live_apply_allowed"] is False
        assert contract["proposal_contract"]["readback_required"] is True
        assert contract["proposal_contract"]["rollback_required"] is True
        assert contract["proposal_contract"]["allowed_live_write_kinds"] == ["fader", "gain"]
        assert f"unsupported_value_type:{value_type}" in contract["blockers"]
        assert "requested_change" not in contract

    assert payload["summary"]["proposal_only_contract_count"] >= len(contracts)


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
async def test_product_state_handlers_round_trip(server, tmp_path):
    server.mixer_client = CachedMixer()
    server.connection_mode = "wing"
    server.audio_capture = FakeAudioCapture()
    server.automation_trace_store = AutomationTraceStore(tmp_path)
    server.send_to_client = AsyncMock()

    await server._dispatch["get_channel_inventory"](object(), {"total_channels": 2})
    await server._dispatch["get_channel_processing_snapshot"](object(), {"total_channels": 2})
    await server._dispatch["get_decision_queue"](object(), {"limit": 5})
    await server._dispatch["get_decision_traces"](object(), {"limit": 5})
    await server._dispatch["get_safety_monitor"](object(), {})
    await server._dispatch["get_replay_runs"](object(), {"limit": 2})
    await server._dispatch["get_channel_intelligence"](object(), {"total_channels": 4})
    await server._dispatch["get_operator_agent_roster"](object(), {})
    await server._dispatch["get_connection_topology"](object(), {})
    await server._dispatch["get_groups_buses_snapshot"](object(), {})
    await server._dispatch["get_bus_main_dca_snapshot"](
        object(),
        {"bus_count": 1, "main_count": 1, "dca_count": 1},
    )
    await server._dispatch["get_matrix_snapshot"](object(), {"matrix_count": 1})
    await server._dispatch["get_routing_sends_snapshot"](
        object(),
        {"total_channels": 1, "send_count": 1, "output_groups": ["MOD"], "output_count": 1},
    )
    await server._dispatch["get_snapshots_scenes_snapshot"](object(), {"snapshot_count": 2})
    await server._dispatch["get_user_presets_snapshot"](object(), {"preset_count": 2})
    await server._dispatch["get_memory_knowledge_snapshot"](object(), {})
    await server._dispatch["get_training_sessions_snapshot"](object(), {})
    await server._dispatch["get_telemetry_alerts_snapshot"](object(), {"limit": 2})
    await server._dispatch["get_rules_library_snapshot"](object(), {})
    await server._dispatch["get_agent_communications_snapshot"](object(), {})
    await server._dispatch["get_processing_pipeline_snapshot"](object(), {"include_methods": False})
    await server._dispatch["get_main_pipeline_preview"](object(), {"total_channels": 4})

    payloads = [call.args[1] for call in server.send_to_client.await_args_list]
    assert [payload["type"] for payload in payloads] == [
        "channel_inventory",
        "channel_processing_snapshot",
        "decision_queue",
        "decision_traces",
        "safety_monitor",
        "replay_runs",
        "channel_intelligence",
        "operator_agent_roster",
        "connection_topology",
        "groups_buses_snapshot",
        "bus_main_dca_snapshot",
        "matrix_snapshot",
        "routing_sends_snapshot",
        "snapshots_scenes_snapshot",
        "user_presets_snapshot",
        "memory_knowledge_snapshot",
        "training_sessions_snapshot",
        "telemetry_alerts_snapshot",
        "rules_library_snapshot",
        "agent_communications_snapshot",
        "processing_pipeline_snapshot",
        "main_pipeline_preview",
    ]
    by_type = {payload["type"]: payload for payload in payloads}
    assert by_type["channel_inventory"]["channels"][1]["name"] == "Lead Vocal"
    assert by_type["channel_processing_snapshot"]["safety"]["queries_console"] is False
    assert by_type["bus_main_dca_snapshot"]["safety"]["queries_console"] is False
    assert by_type["bus_main_dca_snapshot"]["safety"]["live_write"] is False
    assert by_type["matrix_snapshot"]["safety"]["queries_console"] is False
    assert by_type["matrix_snapshot"]["safety"]["live_write"] is False
    assert by_type["routing_sends_snapshot"]["safety"]["queries_console"] is False
    assert by_type["routing_sends_snapshot"]["safety"]["live_write"] is False
    assert by_type["snapshots_scenes_snapshot"]["safety"]["queries_console"] is False
    assert by_type["snapshots_scenes_snapshot"]["safety"]["scene_load_allowed"] is False
    assert by_type["user_presets_snapshot"]["safety"]["queries_console"] is False
    assert by_type["user_presets_snapshot"]["safety"]["preset_recall_allowed"] is False
    assert by_type["decision_queue"]["reason"] == "mixing_agent_runtime_not_initialized"
    assert by_type["decision_queue"]["proposal_queue_available"] is True
    assert by_type["decision_traces"]["summary"]["trace_count"] == 0
    assert by_type["safety_monitor"]["live_write_policy"]["direct_writes_allowed"] is False
    assert by_type["replay_runs"]["summary"]["offline_only"] is True
    assert by_type["channel_intelligence"]["summary"]["shadow_only"] is True
    assert by_type["operator_agent_roster"]["summary"]["agent_count"] >= 8
    assert by_type["connection_topology"]["connection"]["connected"] is True
    assert by_type["groups_buses_snapshot"]["buses"]
    assert by_type["memory_knowledge_snapshot"]["memories"]
    assert by_type["training_sessions_snapshot"]["sessions"]
    assert by_type["telemetry_alerts_snapshot"]["safety"]["live_write"] is False
    assert by_type["rules_library_snapshot"]["rules"]
    assert by_type["agent_communications_snapshot"]["communications"]
    assert by_type["processing_pipeline_snapshot"]["methods"] == []
    assert by_type["processing_pipeline_snapshot"]["summary"]["live_apply_policy"] == "supervised_fader_gain_only"
    assert by_type["main_pipeline_preview"]["executor_policy"]["preview_only"] is True
    assert by_type["main_pipeline_preview"]["summary"]["queue_mutation"] is False


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
        {"proposal_id": proposal_id, "approved": True, "idempotency_key": "unit-outside-mode"},
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
        {"proposal_id": proposal_id, "approved": True, "idempotency_key": "unit-incomplete"},
    )

    server._apply_manual_console_write.assert_not_called()
    payload = server.send_to_client.await_args.args[1]
    assert payload["type"] == "operator_proposal_apply_blocked"
    assert payload["status"] == "apply_blocked"
    assert payload["apply_result"]["reason"] == "proposal_has_no_applyable_requested_change"


@pytest.mark.asyncio
async def test_non_fader_gain_operator_proposal_is_review_only(server):
    server.operator_mode = "supervised"
    server.mixer_client = MagicMock()
    server.mixer_client.is_connected = True
    server._apply_manual_console_write = AsyncMock()
    server.send_to_client = AsyncMock()

    await server._dispatch["create_operator_proposal"](
        object(),
        {
            "title": "Lead Vocal: review EQ cut",
            "requested_change": {
                "value_type": "eq",
                "channel": 2,
                "value": -2.0,
                "current_value": 0.0,
                "rollback_value": 0.0,
                "safety_evidence": [{"label": "Frequency", "value": "350 Hz"}],
            },
            "reason": "proposal-only EQ review",
        },
    )

    proposal = build_decision_queue(server)["pending_actions"][0]
    assert proposal["can_apply"] is False
    assert proposal["requires_approval"] is False
    assert "unsupported_value_type:eq" in proposal["apply_blockers"]

    await server._dispatch["apply_operator_proposal"](
        object(),
        {"proposal_id": proposal["id"], "approved": True, "idempotency_key": "unit-eq-review"},
    )

    server._apply_manual_console_write.assert_not_called()
    payload = server.send_to_client.await_args.args[1]
    assert payload["type"] == "operator_proposal_apply_blocked"
    assert payload["apply_result"]["reason"] == "unsupported_value_type"
    assert payload["apply_result"]["value_type"] == "eq"


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
            "effective_value": -0.5,
            "old_value": 0.0,
            "readback_value": -0.5,
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
            "effective_value": -0.5,
            "old_value": 0.0,
            "readback_value": -0.5,
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
                {"value_type": "fader", "channel": 9, "value": -0.5, "current_value": 0.0, "delta_db": -0.5},
                {"value_type": "fader", "channel": 10, "value": -0.5, "current_value": 0.0, "delta_db": -0.5},
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
        {"proposal_id": proposal["id"], "approved": True, "approval_id": "linked-unit", "idempotency_key": "linked-unit"},
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
                {"value_type": "fader", "channel": 9, "value": -0.5, "current_value": 0.0, "delta_db": -0.5},
                {"value_type": "fader", "channel": 10, "value": -0.5, "current_value": 0.0, "delta_db": -0.5},
            ],
            "reason": "unit",
        },
    )
    proposal = build_decision_queue(server)["pending_actions"][0]

    await server._dispatch["apply_operator_proposal"](
        object(),
        {"proposal_id": proposal["id"], "approved": True, "approval_id": "linked-unit", "idempotency_key": "linked-pre-read"},
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
            "effective_value": -0.5,
            "old_value": 0.0,
            "readback_value": -0.5,
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
            "old_value": -0.5,
            "readback_value": 0.0,
            "readback_status": "verified_from_cache",
            "rollback_available": True,
            "rollback_value": -0.5,
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
                {"value_type": "fader", "channel": 9, "value": -0.5, "current_value": 0.0, "delta_db": -0.5},
                {"value_type": "fader", "channel": 10, "value": -0.5, "current_value": 0.0, "delta_db": -0.5},
            ],
            "reason": "unit",
        },
    )
    proposal = build_decision_queue(server)["pending_actions"][0]

    await server._dispatch["apply_operator_proposal"](
        object(),
        {"proposal_id": proposal["id"], "approved": True, "approval_id": "linked-unit", "idempotency_key": "linked-rollback"},
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
    server.mixer_client.state = {"/ch/2/fdr": -2.5}
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
        {"proposal_id": proposal_id, "approved": True, "reason": "operator approved", "idempotency_key": "unit-supervised-apply"},
    )

    server._apply_manual_console_write.assert_awaited_once()
    payload = server.send_to_client.await_args.args[1]
    assert payload["type"] == "operator_proposal_apply_result"
    assert payload["status"] == "applied"
    queue = build_decision_queue(server)
    assert queue["summary"]["pending_count"] == 0
    assert queue["summary"]["applied_count"] == 1


@pytest.mark.asyncio
async def test_supervised_apply_requires_idempotency_key(server):
    server.operator_mode = "supervised"
    server.mixer_client = MagicMock()
    server.mixer_client.is_connected = True
    server.mixer_client.state = {"/ch/2/fdr": -2.5}
    server._apply_manual_console_write = AsyncMock()
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
        {"proposal_id": proposal_id, "approved": True},
    )

    server._apply_manual_console_write.assert_not_called()
    payload = server.send_to_client.await_args.args[1]
    assert payload["type"] == "operator_proposal_apply_blocked"
    assert payload["reason"] == "missing_idempotency_key"


@pytest.mark.asyncio
async def test_supervised_apply_blocks_duplicate_idempotency_key(server):
    server.operator_mode = "supervised"
    server.mixer_client = MagicMock()
    server.mixer_client.is_connected = True
    server.mixer_client.state = {"/ch/2/fdr": -2.5}
    server._apply_manual_console_write = AsyncMock(return_value={
        "status": "applied",
        "channel": 2,
        "value_type": "fader",
        "effective_value": -3.0,
        "old_value": -2.5,
        "readback_value": -3.0,
        "readback_status": "verified_from_cache",
        "rollback_available": True,
        "rollback_value": -2.5,
    })
    server.send_to_client = AsyncMock()

    for proposal_id in ("dup-1", "dup-2"):
        await server._dispatch["create_operator_proposal"](
            object(),
            {
                "id": proposal_id,
                "title": f"{proposal_id}: supervised fader",
                "channel": 2,
                "value_type": "fader",
                "value": -3.0,
                "reason": "unit",
            },
        )

    await server._dispatch["apply_operator_proposal"](
        object(),
        {"proposal_id": "dup-1", "approved": True, "idempotency_key": "same-key"},
    )
    await server._dispatch["apply_operator_proposal"](
        object(),
        {"proposal_id": "dup-2", "approved": True, "idempotency_key": "same-key"},
    )

    assert server._apply_manual_console_write.await_count == 1
    payload = server.send_to_client.await_args.args[1]
    assert payload["type"] == "operator_proposal_apply_blocked"
    assert payload["reason"] == "duplicate_idempotency_key"


@pytest.mark.asyncio
async def test_supervised_apply_blocks_expired_proposal_before_idempotency_reservation(server):
    server.operator_mode = "supervised"
    server.mixer_client = MagicMock()
    server.mixer_client.is_connected = True
    server.mixer_client.state = {"/ch/2/fdr": -2.5}
    server._apply_manual_console_write = AsyncMock()
    server.send_to_client = AsyncMock()

    await server._dispatch["create_operator_proposal"](
        object(),
        {
            "id": "expired-proposal",
            "title": "Lead Vocal: stale fader",
            "channel": 2,
            "value_type": "fader",
            "value": -3.0,
            "reason": "unit",
            "ttl_seconds": 5,
        },
    )
    proposal = server.operator_proposal_queue.find(proposal_id="expired-proposal")
    proposal.expires_at = "2000-01-01T00:00:00Z"

    await server._dispatch["apply_operator_proposal"](
        object(),
        {"proposal_id": "expired-proposal", "approved": True, "idempotency_key": "expired-key"},
    )

    server._apply_manual_console_write.assert_not_called()
    assert "expired-key" not in server._operator_apply_idempotency_keys
    payload = server.send_to_client.await_args.args[1]
    assert payload["type"] == "operator_proposal_apply_blocked"
    assert payload["apply_result"]["reason"] == "proposal_expired"
    assert payload["proposal"]["expired"] is True


@pytest.mark.asyncio
async def test_supervised_apply_blocks_cumulative_delta_before_manual_gate(server):
    server.operator_mode = "supervised"
    server.mixer_client = MagicMock()
    server.mixer_client.is_connected = True
    server.mixer_client.state = {"/ch/2/fdr": -2.5}
    server._operator_apply_delta_history = {
        (2, "fader"): [{"ts": time.monotonic(), "delta_db": -2.4}]
    }
    server._apply_manual_console_write = AsyncMock()
    server.send_to_client = AsyncMock()

    await server._dispatch["create_operator_proposal"](
        object(),
        {
            "title": "Lead Vocal: cumulative fader",
            "channel": 2,
            "value_type": "fader",
            "value": -3.2,
            "reason": "unit",
        },
    )
    proposal_id = build_decision_queue(server)["pending_actions"][0]["id"]
    await server._dispatch["apply_operator_proposal"](
        object(),
        {"proposal_id": proposal_id, "approved": True, "idempotency_key": "cumulative-limit"},
    )

    server._apply_manual_console_write.assert_not_called()
    payload = server.send_to_client.await_args.args[1]
    assert payload["type"] == "operator_proposal_apply_blocked"
    assert payload["apply_result"]["reason"] == "cumulative_delta_limit_exceeded"
    assert payload["apply_result"]["max_cumulative_delta_db"] == 3.0


@pytest.mark.asyncio
async def test_supervised_apply_blocks_large_delta_before_manual_gate(server):
    server.operator_mode = "supervised"
    server.mixer_client = MagicMock()
    server.mixer_client.is_connected = True
    server.mixer_client.state = {"/ch/2/fdr": 0.0}
    server._apply_manual_console_write = AsyncMock()
    server.send_to_client = AsyncMock()

    await server._dispatch["create_operator_proposal"](
        object(),
        {
            "title": "Lead Vocal: too much fader",
            "channel": 2,
            "value_type": "fader",
            "value": -2.5,
            "reason": "unit",
        },
    )
    proposal_id = build_decision_queue(server)["pending_actions"][0]["id"]
    await server._dispatch["apply_operator_proposal"](
        object(),
        {"proposal_id": proposal_id, "approved": True, "idempotency_key": "large-delta"},
    )

    server._apply_manual_console_write.assert_not_called()
    payload = server.send_to_client.await_args.args[1]
    assert payload["type"] == "operator_proposal_apply_blocked"
    assert payload["apply_result"]["reason"] == "delta_limit_exceeded"
    assert payload["apply_result"]["max_delta_db"] == 1.0


@pytest.mark.asyncio
async def test_supervised_apply_uses_normalized_clamped_value_after_validation(server):
    server.operator_mode = "supervised"
    server.mixer_client = MagicMock()
    server.mixer_client.is_connected = True
    server.mixer_client.state = {"/ch/2/in/set/trim": 18.0}
    server._apply_manual_console_write = AsyncMock(return_value={
        "status": "applied",
        "channel": 2,
        "value_type": "gain",
        "effective_value": 18.0,
    })
    server.send_to_client = AsyncMock()

    await server._dispatch["create_operator_proposal"](
        object(),
        {
            "title": "Lead Vocal: clamped gain",
            "channel": 2,
            "value_type": "gain",
            "value": 22.0,
            "reason": "unit",
        },
    )
    proposal_id = build_decision_queue(server)["pending_actions"][0]["id"]
    await server._dispatch["apply_operator_proposal"](
        object(),
        {"proposal_id": proposal_id, "approved": True, "idempotency_key": "clamped-gain"},
    )

    server._apply_manual_console_write.assert_awaited_once()
    call = server._apply_manual_console_write.await_args
    assert call.kwargs["channel"] == 2
    assert call.kwargs["value_type"] == "gain"
    assert call.kwargs["value"] == 18.0


@pytest.mark.asyncio
async def test_decision_trace_and_safety_monitor_expose_apply_evidence(server, tmp_path):
    server.operator_mode = "supervised"
    server.automation_trace_store = AutomationTraceStore(tmp_path)
    server.mixer_client = MagicMock()
    server.mixer_client.is_connected = True
    server.mixer_client.state = {"/ch/2/fdr": -2.5}
    server._apply_manual_console_write = AsyncMock(return_value={
        "status": "applied",
        "channel": 2,
        "value_type": "fader",
        "effective_value": -3.0,
        "old_value": -2.5,
        "pre_read_status": "verified_from_cache",
        "pre_read_source": "mixer_state_cache",
        "readback_value": -3.0,
        "readback_status": "verified_from_cache",
        "readback_source": "mixer_state_cache",
        "proof_status": "verified",
        "rollback_available": True,
        "rollback_value": -2.5,
    })
    server.send_to_client = AsyncMock()

    await server._dispatch["create_operator_proposal"](
        object(),
        {
            "title": "Lead Vocal: traceable fader",
            "channel": 2,
            "value_type": "fader",
            "value": -3.0,
            "reason": "unit",
        },
    )
    proposal_id = build_decision_queue(server)["pending_actions"][0]["id"]
    await server._dispatch["apply_operator_proposal"](
        object(),
        {"proposal_id": proposal_id, "approved": True, "idempotency_key": "trace-key"},
    )

    traces = build_decision_traces(server)
    assert traces["summary"]["trace_count"] >= 2
    assert traces["traces"][-1]["idempotency_key"] == "trace-key"
    assert traces["traces"][-1]["proof_status"] == "verified"
    monitor = build_safety_monitor(server)
    assert monitor["type"] == "safety_monitor"
    assert monitor["live_write_policy"]["direct_writes_allowed"] is False
    assert "eq" in monitor["blocked_surfaces"]


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


def test_shadow_gain_ride_candidates_are_trace_only_and_bounded():
    candidates = build_shadow_gain_ride_candidates([
        {
            "id": 2,
            "name": "Lead Vox",
            "role": "Vocal",
            "group": "VOCALS",
            "status": "active",
            "lufs": -27.2,
            "peak_db": -9.0,
            "control_state": {"fader_db": -8.0},
        }
    ])

    assert len(candidates) == 1
    candidate = candidates[0]
    assert candidate["action_type"] == "gain_ride_shadow"
    assert candidate["delta_db"] == 0.5
    assert candidate["live_apply_allowed"] is False
    assert candidate["safety_limits"]["live_promotion_allowed"] is False
    assert candidate["scoring"]["intelligibility_proxy"] == "improved"

    trace = shadow_gain_ride_trace(candidate, mode="assist")
    assert trace["action_type"] == "gain_ride_shadow"
    assert trace["proof_status"] == "shadow_only"
    assert trace["readback"]["status"] == "not_available_shadow_only"
    assert trace["metadata"]["live_apply_allowed"] is False


def test_shadow_gain_ride_candidates_block_oscillation():
    recent_traces = [
        {"target": 2, "action_type": "gain_ride_shadow", "delta": 0.5},
        {"target": 2, "action_type": "gain_ride_shadow", "delta": -0.5},
        {"target": 2, "action_type": "gain_ride_shadow", "delta": 0.5},
    ]

    candidates = build_shadow_gain_ride_candidates(
        [
            {
                "id": 2,
                "name": "Lead Vox",
                "role": "Vocal",
                "group": "VOCALS",
                "status": "active",
                "lufs": -27.2,
                "peak_db": -9.0,
                "control_state": {"fader_db": -8.0},
            }
        ],
        recent_traces=recent_traces,
    )

    assert candidates[0]["status"] == "blocked"
    assert candidates[0]["blocked_reason"] == "shadow_gain_ride_oscillation_risk"
    assert candidates[0]["scoring"]["oscillation_risk"] == "blocked"


def test_operator_analysis_records_shadow_gain_ride_traces_without_queue_mutation(server):
    server.mixer_client = CachedMixer()
    server.mixer_client.state.update({
        "/ch/16/$name": "Lead Vox",
        "/ch/16/fdr": -8.0,
        "/ch/16/in/set/trim": 0.0,
    })
    server.connection_mode = "wing"
    server.audio_capture = StaticMeterCapture(
        num_channels=16,
        rms={16: -24.0},
        peak={16: -9.0},
        lufs={16: -27.2},
    )

    report = build_operator_analysis_report(server, total_channels=16, create_proposals=True)
    queue = build_decision_queue(server)
    traces = build_decision_traces(server)

    assert report["shadow_gain_ride_count"] == 1
    assert report["shadow_gain_ride_trace_count"] == 1
    assert report["shadow_gain_rides"][0]["live_apply_allowed"] is False
    assert queue["summary"]["pending_count"] == 0
    assert traces["traces"][-1]["action_type"] == "gain_ride_shadow"
    assert traces["traces"][-1]["proof_status"] == "shadow_only"


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
