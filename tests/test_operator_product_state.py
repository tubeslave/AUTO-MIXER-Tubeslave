import importlib
import json
import os
import sys
import time
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from backend.automation_trace import AutomationTraceStore
from backend.decision_memory import DecisionMemoryStore, decision_memory_entry_from_trace
from backend.operator_analysis import build_operator_analysis_report
from backend.shadow_gain_rides import build_shadow_gain_ride_candidates, shadow_gain_ride_trace
from backend.operator_product_state import (
    DEFAULT_SUPERVISED_PROOF_REPORT_DIRS,
    build_agent_communications_snapshot,
    build_bus_main_dca_snapshot,
    build_channel_intelligence,
    build_channel_inventory,
    build_channel_processing_snapshot,
    build_decision_queue,
    build_decision_traces,
    build_dlive_state_snapshot,
    build_fx_inserts_snapshot,
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
    build_supervised_proof_evidence_snapshot,
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
        self.state.update({
            "/ch/1/$name": "Kick In",
            "/ch/2/$name": "Lead Vocal",
            "/ch/2/fdr": -8.0,
            "/ch/2/in/set/trim": 1.0,
        })
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


class CachedDLiveMixer:
    is_connected = True
    ip = "192.168.3.70"
    port = 51328
    midi_base_channel = 0

    def __init__(self):
        self.state_timestamps = {}
        self.state = TimestampedState(self)
        self.state.update({
            "name/1": "Lead Vox",
            "fader/1": 0x2AAA,
            "mute/1": False,
            "note/0/1": 0x7F,
            "cc/0/99": 0x00,
            "pitchbend/0": 0x2000,
        })
        self._channel_names = {1: "Lead Vox"}
        self.send = MagicMock()
        self._send_raw = MagicMock()
        self.get_fader = MagicMock(return_value=-99.0)
        self.get_channel_settings = MagicMock(return_value={})
        self.request_channel_names = MagicMock()

    def get_state(self):
        return {"cached": True, "raw_state_key_count": len(self.state)}


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


def test_channel_inventory_publishes_cached_source_patch_readback_only(server):
    mixer = CachedMixer()
    mixer.state.update({
        "/ch/2/in/conn/grp": "MOD",
        "/ch/2/in/conn/in": 12,
        "/ch/2/in/conn/altgrp": "CRD",
        "/ch/2/in/conn/altin": 5,
        "/ch/2/mute": 0,
        "/ch/2/pan": 0.25,
    })
    server.connection_mode = "wing"
    server.mixer_client = mixer
    server.audio_capture = None

    inventory = build_channel_inventory(server, total_channels=2)
    channel_2 = inventory["channels"][1]
    source_patch = channel_2["source_patch"]

    assert channel_2["source"] == "MOD 12"
    assert channel_2["fallback_source"] == "In 02"
    assert source_patch["main"] == {
        "source_group": "MOD",
        "source_channel": 12,
        "label": "MOD 12",
        "available": True,
    }
    assert source_patch["alt"] == {
        "source_group": "CRD",
        "source_channel": 5,
        "label": "CRD 5",
        "available": True,
    }
    assert source_patch["input_assignment_policy"] == "blocked_manual_only"
    assert source_patch["destructive_change_marker"] == "source_patch_change_would_repatch_channel_input"
    assert source_patch["read_coverage"]["status"] == "complete"
    assert source_patch["read_coverage"]["source"] == "mixer_state_cache"
    assert source_patch["cache_freshness"]["status"] == "fresh"
    assert source_patch["cache_freshness"]["field_count"] == 4
    assert channel_2["read_coverage"]["sections"]["source_patch"]["status"] == "complete"
    assert inventory["cache_freshness"]["present_address_count"] == 10
    mixer.send.assert_not_called()
    mixer.get_all_channel_names.assert_not_called()


def test_channel_inventory_exposes_fresh_cache_timestamps_without_queries(server):
    mixer = CachedMixer()
    now = time.time()
    mixer.state_timestamps = {address: now - 2.0 for address in mixer.state}
    server.connection_mode = "wing"
    server.mixer_client = mixer
    server.audio_capture = None

    inventory = build_channel_inventory(server, total_channels=2)
    freshness = inventory["cache_freshness"]

    assert inventory["summary"]["cache_freshness_status"] == "fresh"
    assert freshness["status"] == "fresh"
    assert freshness["read_only"] is True
    assert freshness["live_write"] is False
    assert freshness["queries_console"] is False
    assert freshness["present_address_count"] == 4
    assert freshness["timestamped_address_count"] == 4
    assert freshness["max_age_sec"] < freshness["stale_after_sec"]
    assert freshness["untimestamped_present_addresses"] == []
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
    assert coverage["sections"]["source_patch"]["status"] == "missing"
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
    assert channel_2["hpf"]["read_coverage"]["cache_freshness"]["status"] == "fresh"
    assert channel_2["eq"]["read_coverage"]["cache_freshness"]["field_count"] == 13
    assert channel_2["eq"]["read_coverage"]["cache_freshness"]["status"] == "fresh"
    assert channel_2["eq"]["bands"][0]["cache_freshness"]["fields"]["gain_db"]["status"] == "fresh"
    assert channel_2["compressor"]["read_coverage"]["cache_freshness"]["status"] == "fresh"
    assert channel_2["gate"]["read_coverage"]["cache_freshness"]["status"] == "fresh"
    mixer.send.assert_not_called()
    mixer.get_all_channel_names.assert_not_called()


def test_channel_processing_snapshot_marks_stale_cache_timestamps_without_queries(server):
    mixer = CachedMixer()
    mixer.state.update({
        "/ch/2/flt/lc": 1,
        "/ch/2/flt/lcf": 120.0,
        "/ch/2/eq/on": 1,
    })
    stale_time = time.time() - 120.0
    mixer.state_timestamps = {
        "/ch/2/flt/lc": stale_time,
        "/ch/2/flt/lcf": stale_time,
        "/ch/2/eq/on": stale_time,
    }
    server.connection_mode = "wing"
    server.mixer_client = mixer
    server.audio_capture = None

    snapshot = build_channel_processing_snapshot(server, total_channels=2)
    freshness = snapshot["cache_freshness"]
    channel_2 = snapshot["channels"][1]
    hpf_freshness = channel_2["hpf"]["read_coverage"]["cache_freshness"]
    eq_freshness = channel_2["eq"]["read_coverage"]["cache_freshness"]

    assert snapshot["summary"]["cache_freshness_status"] == "stale"
    assert freshness["status"] == "stale"
    assert freshness["present_address_count"] == 3
    assert freshness["timestamped_address_count"] == 3
    assert freshness["max_age_sec"] > freshness["stale_after_sec"]
    assert freshness["live_write"] is False
    assert freshness["queries_console"] is False
    assert hpf_freshness["status"] == "stale"
    assert hpf_freshness["fields"]["enabled"]["status"] == "stale"
    assert hpf_freshness["fields"]["frequency_hz"]["status"] == "stale"
    assert hpf_freshness["queries_console"] is False
    assert eq_freshness["status"] == "stale"
    assert eq_freshness["stale_fields"] == ["enabled"]
    assert eq_freshness["missing_field_count"] == 12
    assert eq_freshness["fields"]["band_1_frequency_hz"]["status"] == "missing"
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
    assert snapshot["summary"]["cache_freshness_status"] == "fresh"
    assert snapshot["cache_freshness"]["status"] == "fresh"
    assert snapshot["cache_freshness"]["present_address_count"] == 79
    assert snapshot["cache_freshness"]["timestamped_address_count"] == 79
    assert snapshot["cache_freshness"]["queries_console"] is False
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
    assert snapshot["summary"]["cache_freshness_status"] == "missing"
    assert snapshot["cache_freshness"]["status"] == "missing"
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


def test_bus_main_dca_snapshot_marks_stale_cache_timestamps_without_queries(server):
    mixer = CachedMixer()
    mixer.state.update({
        "/bus/1/fdr": -5.0,
        "/main/1/fdr": -2.0,
        "/dca/1/fdr": -4.0,
    })
    stale_time = time.time() - 120.0
    mixer.state_timestamps = {address: stale_time for address in mixer.state}
    server.connection_mode = "wing"
    server.mixer_client = mixer

    snapshot = build_bus_main_dca_snapshot(server, bus_count=1, main_count=1, dca_count=1)

    assert snapshot["summary"]["cache_freshness_status"] == "stale"
    assert snapshot["cache_freshness"]["status"] == "stale"
    assert snapshot["cache_freshness"]["present_address_count"] == 3
    assert snapshot["cache_freshness"]["timestamped_address_count"] == 3
    assert snapshot["cache_freshness"]["max_age_sec"] > snapshot["cache_freshness"]["stale_after_sec"]
    assert snapshot["buses"][0]["read_coverage"]["sections"]["basic_controls"]["status"] == "partial"
    assert snapshot["mains"][0]["read_coverage"]["sections"]["basic_controls"]["status"] == "partial"
    assert snapshot["dcas"][0]["read_coverage"]["sections"]["basic_controls"]["status"] == "partial"
    mixer.send.assert_not_called()
    mixer.get_all_channel_names.assert_not_called()


def test_matrix_snapshot_publishes_cached_console_state_without_queries(server):
    mixer = CachedMixer()
    mixer.state.update({
        "/mtx/1/fdr": -5.0,
        "/mtx/1/mute": 0,
        "/mtx/1/pan": 0.0,
        "/mtx/1/eq/on": 1,
        "/mtx/1/dyn/on": 1,
        "/mtx/1/dyn/mdl": "COMP",
        "/mtx/1/dyn/thr37": -24.0,
        "/mtx/1/dyn/ratio": "2.5",
        "/mtx/1/dyn/knee": 2.0,
        "/mtx/1/dyn/det": "PEAK",
        "/mtx/1/dyn/att": 18.0,
        "/mtx/1/dyn/rel": 220.0,
        "/mtx/1/dyn/gain": 1.5,
        "/mtx/1/dyn/mix": 90.0,
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
    assert snapshot["summary"]["cache_freshness_status"] == "fresh"
    assert snapshot["cache_freshness"]["status"] == "fresh"
    assert snapshot["cache_freshness"]["present_address_count"] == 46
    assert snapshot["cache_freshness"]["timestamped_address_count"] == 46
    assert snapshot["cache_freshness"]["queries_console"] is False
    assert matrix["fader_db"] == -5.0
    assert matrix["eq"]["bands"][7]["frequency_hz"] == 1280.0
    assert matrix["dynamics"]["enabled"] == 1
    assert matrix["dynamics"]["threshold_db"] == -24.0
    assert matrix["dynamics"]["ratio"] == "2.5"
    assert matrix["dynamics"]["makeup_db"] == 1.5
    assert matrix["dynamics"]["read_coverage"]["cache_freshness"]["status"] == "fresh"
    assert matrix["inserts"]["pre_insert"]["slot"] == "FX1"
    assert matrix["inserts"]["post_insert"]["mode"] == "insert"
    assert matrix["read_coverage"]["status"] == "complete"
    assert matrix["read_coverage"]["sections"]["basic_controls"]["status"] == "complete"
    assert matrix["read_coverage"]["sections"]["eq"]["status"] == "complete"
    assert matrix["read_coverage"]["sections"]["dynamics"]["status"] == "complete"
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
    assert snapshot["summary"]["cache_freshness_status"] == "missing"
    assert snapshot["cache_freshness"]["status"] == "missing"
    assert snapshot["read_coverage"]["surface_status_counts"]["matrices"]["missing"] == 1
    assert matrix["read_coverage"]["missing_sections"] == ["basic_controls", "eq", "dynamics", "inserts"]
    assert matrix["dynamics"]["read_coverage"]["status"] == "missing"
    assert matrix["dynamics"]["read_coverage"]["cache_freshness"]["status"] == "missing"
    assert matrix["inserts"]["pre_insert"]["read_coverage"]["status"] == "missing"
    assert matrix["inserts"]["post_insert"]["read_coverage"]["status"] == "missing"
    mixer.send.assert_not_called()
    mixer.get_all_channel_names.assert_not_called()


def test_matrix_snapshot_marks_stale_cache_timestamps_without_queries(server):
    mixer = CachedMixer()
    mixer.state.update({
        "/mtx/1/fdr": -6.0,
        "/mtx/1/mute": 0,
        "/mtx/1/pan": 0.0,
        "/mtx/1/eq/on": 1,
    })
    stale_time = time.time() - 120.0
    mixer.state_timestamps = {address: stale_time for address in mixer.state}
    server.connection_mode = "wing"
    server.mixer_client = mixer

    snapshot = build_matrix_snapshot(server, matrix_count=1)
    matrix = snapshot["matrices"][0]

    assert snapshot["summary"]["cache_freshness_status"] == "stale"
    assert snapshot["cache_freshness"]["status"] == "stale"
    assert snapshot["cache_freshness"]["present_address_count"] == 4
    assert snapshot["cache_freshness"]["timestamped_address_count"] == 4
    assert snapshot["cache_freshness"]["max_age_sec"] > snapshot["cache_freshness"]["stale_after_sec"]
    assert matrix["read_coverage"]["sections"]["basic_controls"]["status"] == "complete"
    assert matrix["read_coverage"]["sections"]["eq"]["status"] == "partial"
    assert matrix["dynamics"]["read_coverage"]["cache_freshness"]["status"] == "missing"
    mixer.send.assert_not_called()
    mixer.get_all_channel_names.assert_not_called()


def test_fx_inserts_snapshot_publishes_cached_fx_and_insert_state_without_queries(server):
    mixer = CachedMixer()
    mixer.state.update({
        "/fx/1/mdl": "REV",
        "/fx/1/on": 1,
        "/fx/1/fxmix": 35.0,
        "/fx/1/$esrc": "BUS1",
        "/fx/1/$emode": "send",
        "/fx/1/$a_chn": 2,
        "/fx/1/$a_pos": "post",
        "/fx/1/1": 0.25,
        "/fx/1/2": 0.75,
        "/ch/2/preins/on": 1,
        "/ch/2/preins/ins": "FX1",
        "/ch/2/preins/$stat": "OK",
        "/ch/2/postins/on": 0,
        "/ch/2/postins/ins": "NONE",
        "/ch/2/postins/$stat": "OK",
        "/ch/2/postins/mode": "insert",
        "/ch/2/postins/w": 100.0,
    })
    server.connection_mode = "wing"
    server.mixer_client = mixer

    snapshot = build_fx_inserts_snapshot(
        server,
        fx_slot_count=2,
        total_channels=2,
        bus_count=1,
        main_count=1,
        matrix_count=1,
        parameter_count=2,
    )
    fx_1 = snapshot["fx_slots"][0]
    insert = next(
        row
        for row in snapshot["inserts"]
        if row["target_type"] == "channel" and row["target"] == 2 and row["position"] == "pre"
    )

    assert snapshot["type"] == "fx_inserts_snapshot"
    assert snapshot["safety"] == {
        "read_only": True,
        "live_write": False,
        "mutates_console_state": False,
        "queries_console": False,
        "full_autonomous_ready": False,
        "fx_apply_allowed": False,
        "insert_assignment_allowed": False,
    }
    assert snapshot["summary"]["fx_slot_count"] == 2
    assert snapshot["summary"]["active_fx_slot_count"] == 1
    assert snapshot["summary"]["active_insert_count"] == 1
    assert snapshot["summary"]["cached_parameter_count"] == 2
    assert snapshot["summary"]["read_coverage_status"] == "partial"
    assert snapshot["summary"]["parameter_schema_status"] == "mapped"
    assert snapshot["summary"]["cache_freshness_status"] == "fresh"
    assert snapshot["parameter_schema"]["status"] == "mapped"
    assert snapshot["parameter_schema"]["mapped_parameter_count"] == 2
    assert snapshot["parameter_schema"]["generic_parameter_count"] == 0
    assert snapshot["parameter_schema"]["physical_model_schema_verified"] is False
    assert snapshot["parameter_schema"]["live_write_authority"] is False
    assert snapshot["cache_freshness"]["status"] == "fresh"
    assert snapshot["cache_freshness"]["present_address_count"] == 17
    assert snapshot["cache_freshness"]["timestamped_address_count"] == 17
    assert snapshot["cache_freshness"]["queries_console"] is False
    assert fx_1["slot"] == "FX1"
    assert fx_1["model"] == "REV"
    assert fx_1["mix"] == 35.0
    assert fx_1["cached_parameter_count"] == 2
    assert fx_1["parameter_schema"]["status"] == "mapped"
    assert fx_1["parameter_schema"]["schema_family"] == "reverb"
    assert fx_1["parameters"][0]["parameter_index"] == 1
    assert fx_1["parameters"][0]["schema_status"] == "mapped"
    assert fx_1["parameters"][0]["semantic_name"] == "decay_time"
    assert fx_1["parameters"][0]["unit"] == "seconds"
    assert fx_1["parameters"][0]["safe_range"] == {"min": 0.1, "max": 8.0}
    assert fx_1["read_coverage"]["status"] == "complete"
    assert insert["linked_fx_slot"] == "FX1"
    assert insert["active"] is True
    assert insert["insert_assignment_policy"] == "blocked_manual_only"
    assert insert["destructive_change_marker"] == "insert_assignment_would_repatch_processing_chain"
    assert insert["read_coverage"]["status"] == "complete"
    mixer.send.assert_not_called()
    mixer.get_all_channel_names.assert_not_called()


def test_fx_inserts_snapshot_uses_generic_schema_for_unknown_fx_models(server):
    mixer = CachedMixer()
    mixer.state.update({
        "/fx/1/mdl": "CUSTOM-FX",
        "/fx/1/on": 1,
        "/fx/1/fxmix": 50.0,
        "/fx/1/1": 0.33,
    })
    server.connection_mode = "wing"
    server.mixer_client = mixer

    snapshot = build_fx_inserts_snapshot(
        server,
        fx_slot_count=1,
        total_channels=1,
        bus_count=1,
        main_count=1,
        matrix_count=1,
        parameter_count=1,
    )
    fx_1 = snapshot["fx_slots"][0]
    parameter = fx_1["parameters"][0]

    assert snapshot["summary"]["parameter_schema_status"] == "partial"
    assert snapshot["parameter_schema"]["status"] == "partial"
    assert snapshot["parameter_schema"]["mapped_parameter_count"] == 0
    assert snapshot["parameter_schema"]["generic_parameter_count"] == 1
    assert snapshot["parameter_schema"]["physical_model_schema_verified"] is False
    assert fx_1["parameter_schema"]["status"] == "partial"
    assert fx_1["parameter_schema"]["schema_family"] == "unknown"
    assert parameter["schema_status"] == "generic_numeric"
    assert parameter["semantic_name"] == "parameter_1"
    assert parameter["unit"] == "raw"
    assert parameter["safe_range"] == {"min": None, "max": None}
    assert snapshot["safety"]["fx_apply_allowed"] is False
    mixer.send.assert_not_called()
    mixer.get_all_channel_names.assert_not_called()


def test_fx_inserts_snapshot_marks_missing_cache_without_queries(server):
    mixer = CachedMixer()
    mixer.state = {}
    server.connection_mode = "wing"
    server.mixer_client = mixer

    snapshot = build_fx_inserts_snapshot(
        server,
        fx_slot_count=1,
        total_channels=1,
        bus_count=1,
        main_count=1,
        matrix_count=1,
        parameter_count=1,
    )

    assert snapshot["summary"]["read_coverage_status"] == "missing"
    assert snapshot["summary"]["parameter_schema_status"] == "missing"
    assert snapshot["parameter_schema"]["status"] == "missing"
    assert snapshot["parameter_schema"]["physical_model_schema_verified"] is False
    assert snapshot["summary"]["cache_freshness_status"] == "missing"
    assert snapshot["cache_freshness"]["status"] == "missing"
    assert snapshot["summary"]["active_fx_slot_count"] == 0
    assert snapshot["summary"]["active_insert_count"] == 0
    assert snapshot["read_coverage"]["surface_status_counts"]["fx_slots"]["missing"] == 1
    assert snapshot["read_coverage"]["surface_status_counts"]["inserts"]["missing"] == 8
    assert snapshot["fx_slots"][0]["model"] is None
    assert snapshot["fx_slots"][0]["read_coverage"]["status"] == "missing"
    assert snapshot["inserts"][0]["insert_assignment_policy"] == "blocked_manual_only"
    mixer.send.assert_not_called()
    mixer.get_all_channel_names.assert_not_called()


def test_fx_inserts_snapshot_marks_stale_cache_timestamps_without_queries(server):
    mixer = CachedMixer()
    mixer.state.update({
        "/fx/1/mdl": "DLY",
        "/fx/1/on": 1,
        "/fx/1/fxmix": 40.0,
        "/fx/1/1": 0.5,
        "/ch/1/postins/on": 1,
        "/ch/1/postins/ins": "FX1",
        "/ch/1/postins/$stat": "OK",
        "/ch/1/postins/mode": "insert",
        "/ch/1/postins/w": 95.0,
    })
    stale_time = time.time() - 120.0
    mixer.state_timestamps = {address: stale_time for address in mixer.state}
    server.connection_mode = "wing"
    server.mixer_client = mixer

    snapshot = build_fx_inserts_snapshot(
        server,
        fx_slot_count=1,
        total_channels=1,
        bus_count=1,
        main_count=1,
        matrix_count=1,
        parameter_count=1,
    )

    assert snapshot["summary"]["cache_freshness_status"] == "stale"
    assert snapshot["cache_freshness"]["status"] == "stale"
    assert snapshot["cache_freshness"]["present_address_count"] == 9
    assert snapshot["cache_freshness"]["timestamped_address_count"] == 9
    assert snapshot["cache_freshness"]["max_age_sec"] > snapshot["cache_freshness"]["stale_after_sec"]
    assert snapshot["fx_slots"][0]["read_coverage"]["status"] == "partial"
    active_insert = next(
        row
        for row in snapshot["inserts"]
        if row["target_type"] == "channel" and row["target"] == 1 and row["position"] == "post"
    )
    assert active_insert["read_coverage"]["status"] == "complete"
    mixer.send.assert_not_called()
    mixer.get_all_channel_names.assert_not_called()


def test_dlive_state_snapshot_publishes_cached_midi_tcp_state_without_queries(server):
    mixer = CachedDLiveMixer()
    server.connection_mode = "dlive"
    server.mixer_client = mixer

    snapshot = build_dlive_state_snapshot(server, total_channels=2)
    channel_1 = snapshot["channels"][0]
    channel_2 = snapshot["channels"][1]

    assert snapshot["type"] == "dlive_state_snapshot"
    assert snapshot["source"] == "midi_tcp_cached_state"
    assert snapshot["safety"] == {
        "read_only": True,
        "live_write": False,
        "mutates_console_state": False,
        "queries_console": False,
        "full_autonomous_ready": False,
        "dlive_apply_allowed": False,
        "midi_send_allowed": False,
    }
    assert snapshot["summary"]["channel_count"] == 2
    assert snapshot["summary"]["read_coverage_status"] == "partial"
    assert snapshot["summary"]["channels_with_cached_name"] == 1
    assert snapshot["summary"]["channels_with_cached_fader"] == 1
    assert snapshot["summary"]["channels_with_cached_mute"] == 2
    assert snapshot["summary"]["cache_freshness_status"] == "fresh"
    assert snapshot["summary"]["wing_parity_status"] == "partial"
    assert snapshot["summary"]["live_apply_allowed"] is False
    assert snapshot["state_key_counts"]["raw_state_keys"] == 6
    assert snapshot["state_key_counts"]["raw_pitchbend_events"] == 1
    assert snapshot["cache_freshness"]["status"] == "fresh"
    assert snapshot["cache_freshness"]["queries_console"] is False
    assert channel_1["name"] == "Lead Vox"
    assert channel_1["fader_raw_14bit"] == 0x2AAA
    assert channel_1["fader_db"] == 0.0
    assert channel_1["mute"] is False
    assert channel_1["read_coverage"]["status"] == "partial"
    assert channel_1["read_coverage"]["sections"]["identity"]["status"] == "complete"
    assert channel_1["read_coverage"]["sections"]["fader"]["status"] == "complete"
    assert channel_1["read_coverage"]["sections"]["mute"]["status"] == "partial"
    assert channel_2["name"] == "dLive Ch 2"
    assert channel_2["mute"] is True
    assert channel_2["fader_db"] is None
    assert channel_2["read_coverage"]["sections"]["mute"]["status"] == "complete"
    assert "physical readback fixtures" in snapshot["known_gaps"][2]
    mixer.send.assert_not_called()
    mixer._send_raw.assert_not_called()
    mixer.get_fader.assert_not_called()
    mixer.get_channel_settings.assert_not_called()
    mixer.request_channel_names.assert_not_called()


def test_dlive_state_snapshot_marks_missing_cache_without_queries(server):
    mixer = CachedDLiveMixer()
    mixer.state = {}
    mixer._channel_names = {}
    server.connection_mode = "dlive"
    server.mixer_client = mixer

    snapshot = build_dlive_state_snapshot(server, total_channels=2)

    assert snapshot["summary"]["read_coverage_status"] == "missing"
    assert snapshot["summary"]["channels_with_cached_name"] == 0
    assert snapshot["summary"]["channels_with_cached_fader"] == 0
    assert snapshot["summary"]["channels_with_cached_mute"] == 0
    assert snapshot["summary"]["cache_freshness_status"] == "missing"
    assert snapshot["state_key_counts"]["raw_state_keys"] == 0
    assert snapshot["cache_freshness"]["status"] == "missing"
    assert snapshot["read_coverage"]["channel_status_counts"]["missing"] == 2
    assert snapshot["channels"][0]["name"] == "dLive Ch 1"
    assert snapshot["channels"][0]["read_coverage"]["missing_sections"] == ["identity", "fader", "mute"]
    mixer.send.assert_not_called()
    mixer._send_raw.assert_not_called()
    mixer.get_fader.assert_not_called()
    mixer.get_channel_settings.assert_not_called()
    mixer.request_channel_names.assert_not_called()


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
        "/ch/2/send/MX1/on": 1,
        "/ch/2/send/MX1/lvl": -18.0,
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
        matrix_send_count=1,
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
    assert snapshot["summary"]["cache_freshness_status"] == "fresh"
    assert snapshot["summary"]["active_bus_send_count"] == 1
    assert snapshot["summary"]["active_main_send_count"] == 1
    assert snapshot["summary"]["active_matrix_send_count"] == 1
    assert snapshot["cache_freshness"]["status"] == "fresh"
    assert snapshot["cache_freshness"]["queries_console"] is False
    assert snapshot["static_dante_routes"][0]["read_coverage"]["source"] == "static_config"
    send = next(row for row in snapshot["channel_bus_sends"] if row["channel"] == 2)
    main_send = next(row for row in snapshot["channel_main_sends"] if row["channel"] == 2)
    matrix_send = next(row for row in snapshot["channel_matrix_sends"] if row["channel"] == 2)
    route = snapshot["output_routes"][0]
    assert send["active"] is True
    assert send["read_coverage"]["status"] == "complete"
    assert send["read_coverage"]["cache_freshness"]["status"] == "fresh"
    assert main_send["active"] is True
    assert main_send["read_coverage"]["status"] == "complete"
    assert main_send["read_coverage"]["cache_freshness"]["status"] == "fresh"
    assert matrix_send["target_matrix"] == 1
    assert matrix_send["active"] is True
    assert matrix_send["level_db"] == -18.0
    assert matrix_send["read_coverage"]["status"] == "complete"
    assert matrix_send["read_coverage"]["cache_freshness"]["status"] == "fresh"
    assert matrix_send["destructive_change_marker"] == "matrix_send_write_would_change_console_routing"
    assert route["source_group"] == "CH"
    assert route["source_channel"] == 2
    assert route["read_coverage"]["status"] == "complete"
    assert route["read_coverage"]["cache_freshness"]["status"] == "fresh"
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
        matrix_send_count=1,
        output_groups=["MOD"],
        output_count=1,
    )

    assert snapshot["summary"]["read_coverage_status"] == "partial"
    assert snapshot["summary"]["cache_freshness_status"] == "missing"
    assert snapshot["cache_freshness"]["status"] == "missing"
    assert snapshot["read_coverage"]["surface_status_counts"]["static_dante_routes"]["complete"] >= 1
    assert snapshot["read_coverage"]["surface_status_counts"]["channel_bus_sends"]["missing"] == 1
    assert snapshot["read_coverage"]["surface_status_counts"]["channel_main_sends"]["missing"] == 1
    assert snapshot["read_coverage"]["surface_status_counts"]["channel_matrix_sends"]["missing"] == 1
    assert snapshot["read_coverage"]["surface_status_counts"]["output_routes"]["missing"] == 1
    assert snapshot["channel_bus_sends"][0]["read_coverage"]["status"] == "missing"
    assert snapshot["channel_main_sends"][0]["read_coverage"]["status"] == "missing"
    assert snapshot["channel_matrix_sends"][0]["read_coverage"]["status"] == "missing"
    assert snapshot["channel_matrix_sends"][0]["read_coverage"]["cache_freshness"]["status"] == "missing"
    assert snapshot["output_routes"][0]["read_coverage"]["status"] == "missing"
    mixer.send.assert_not_called()
    mixer.get_all_channel_names.assert_not_called()


def test_routing_sends_snapshot_marks_stale_cache_timestamps_without_queries(server):
    mixer = CachedMixer()
    mixer.state.update({
        "/ch/1/send/1/on": 1,
        "/ch/1/send/1/lvl": -14.0,
        "/ch/1/send/1/pon": 1,
        "/ch/1/send/1/mode": "POST",
        "/ch/1/send/1/plink": 1,
        "/ch/1/send/1/pan": -10.0,
        "/ch/1/main/1/on": 1,
        "/ch/1/main/1/lvl": -4.0,
        "/ch/1/main/1/pre": 0,
        "/ch/1/send/MX1/on": 1,
        "/ch/1/send/MX1/lvl": -20.0,
        "/io/out/MOD/1/grp": "CH",
        "/io/out/MOD/1/in": 1,
    })
    stale_time = time.time() - 120.0
    mixer.state_timestamps = {address: stale_time for address in mixer.state}
    server.connection_mode = "wing"
    server.mixer_client = mixer

    snapshot = build_routing_sends_snapshot(
        server,
        total_channels=1,
        send_count=1,
        main_count=1,
        matrix_send_count=1,
        output_groups=["MOD"],
        output_count=1,
    )

    bus_send = snapshot["channel_bus_sends"][0]
    main_send = snapshot["channel_main_sends"][0]
    matrix_send = snapshot["channel_matrix_sends"][0]
    route = snapshot["output_routes"][0]

    assert snapshot["summary"]["cache_freshness_status"] == "stale"
    assert snapshot["cache_freshness"]["status"] == "stale"
    assert snapshot["cache_freshness"]["present_address_count"] == 13
    assert snapshot["cache_freshness"]["timestamped_address_count"] == 13
    assert snapshot["cache_freshness"]["max_age_sec"] > snapshot["cache_freshness"]["stale_after_sec"]
    assert bus_send["read_coverage"]["cache_freshness"]["status"] == "stale"
    assert main_send["read_coverage"]["cache_freshness"]["status"] == "stale"
    assert matrix_send["read_coverage"]["cache_freshness"]["status"] == "stale"
    assert route["read_coverage"]["cache_freshness"]["status"] == "stale"
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
    assert snapshot["summary"]["cache_freshness_status"] == "fresh"
    assert snapshot["cache_freshness"]["status"] == "fresh"
    assert snapshot["cache_freshness"]["present_address_count"] == 4
    assert snapshot["cache_freshness"]["timestamped_address_count"] == 4
    assert snapshot["cache_freshness"]["queries_console"] is False
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


def test_snapshots_scenes_snapshot_marks_stale_cache_timestamps_without_queries(server):
    mixer = CachedMixer()
    mixer.state.update({
        "snapshots": {
            1: {"name": "Pre-show baseline"},
            "2": "Band A live start",
        },
        "/$ctl/lib/$actidx": 2,
        "/$ctl/lib/$active": "Band A live start",
    })
    stale_time = time.time() - 120.0
    mixer.state_timestamps = {address: stale_time for address in mixer.state}
    server.connection_mode = "wing"
    server.mixer_client = mixer

    snapshot = build_snapshots_scenes_snapshot(server, snapshot_count=2)

    assert snapshot["summary"]["cache_freshness_status"] == "stale"
    assert snapshot["cache_freshness"]["status"] == "stale"
    assert snapshot["cache_freshness"]["present_address_count"] == 3
    assert snapshot["cache_freshness"]["timestamped_address_count"] == 3
    assert snapshot["cache_freshness"]["max_age_sec"] > snapshot["cache_freshness"]["stale_after_sec"]
    assert snapshot["cache_freshness"]["live_write"] is False
    assert snapshot["cache_freshness"]["queries_console"] is False
    mixer.send.assert_not_called()
    mixer.get_all_channel_names.assert_not_called()


def test_snapshots_scenes_snapshot_marks_missing_cache_without_queries(server):
    mixer = CachedMixer()
    mixer.state = {}
    server.connection_mode = "wing"
    server.mixer_client = mixer

    snapshot = build_snapshots_scenes_snapshot(server, snapshot_count=2)

    assert snapshot["summary"]["read_coverage_status"] == "missing"
    assert snapshot["summary"]["cache_freshness_status"] == "missing"
    assert snapshot["cache_freshness"]["status"] == "missing"
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
    assert snapshot["summary"]["cache_freshness_status"] == "fresh"
    assert snapshot["cache_freshness"]["status"] == "fresh"
    assert snapshot["cache_freshness"]["present_address_count"] == 1
    assert snapshot["cache_freshness"]["timestamped_address_count"] == 1
    assert snapshot["cache_freshness"]["queries_console"] is False
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


def test_user_presets_snapshot_marks_stale_cache_timestamps_without_queries(server):
    mixer = CachedMixer()
    mixer.state.update({
        "user_presets": {
            1: {
                "name": "Lead Vocal Strip",
                "scope": "channel_strip",
                "target_type": "channel",
                "target": "CH 2",
            },
        },
    })
    stale_time = time.time() - 120.0
    mixer.state_timestamps = {address: stale_time for address in mixer.state}
    server.connection_mode = "wing"
    server.mixer_client = mixer

    snapshot = build_user_presets_snapshot(server, preset_count=1)

    assert snapshot["summary"]["cache_freshness_status"] == "stale"
    assert snapshot["cache_freshness"]["status"] == "stale"
    assert snapshot["cache_freshness"]["present_address_count"] == 1
    assert snapshot["cache_freshness"]["timestamped_address_count"] == 1
    assert snapshot["cache_freshness"]["max_age_sec"] > snapshot["cache_freshness"]["stale_after_sec"]
    assert snapshot["cache_freshness"]["live_write"] is False
    assert snapshot["cache_freshness"]["queries_console"] is False
    mixer.send.assert_not_called()
    mixer.get_all_channel_names.assert_not_called()


def test_user_presets_snapshot_marks_missing_cache_without_queries(server):
    mixer = CachedMixer()
    mixer.state = {}
    server.connection_mode = "wing"
    server.mixer_client = mixer

    snapshot = build_user_presets_snapshot(server, preset_count=2)

    assert snapshot["summary"]["read_coverage_status"] == "missing"
    assert snapshot["summary"]["cache_freshness_status"] == "missing"
    assert snapshot["cache_freshness"]["status"] == "missing"
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
        build_fx_inserts_snapshot(
            server,
            fx_slot_count=2,
            total_channels=2,
            bus_count=1,
            main_count=1,
            matrix_count=1,
            parameter_count=2,
        ),
        build_dlive_state_snapshot(server, total_channels=2),
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
        "fx_inserts_snapshot",
        "dlive_state_snapshot",
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
    assert snapshots[4]["safety"]["fx_apply_allowed"] is False
    assert snapshots[5]["safety"]["queries_console"] is False
    assert snapshots[5]["safety"]["dlive_apply_allowed"] is False
    assert snapshots[6]["safety"]["queries_console"] is False
    assert snapshots[6]["safety"]["scene_load_allowed"] is False
    assert snapshots[7]["safety"]["queries_console"] is False
    assert snapshots[7]["safety"]["preset_recall_allowed"] is False
    assert snapshots[8]["buses"]
    assert snapshots[9]["memories"]
    assert snapshots[10]["sessions"]
    assert snapshots[11]["rules"]
    assert snapshots[12]["communications"]

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


def proof_report_with_autonomy_overlay(report_status="ok"):
    return {
        "type": "supervised_proof_artifact_report",
        "generated_at": "2026-06-05T08:20:00Z",
        "status": report_status,
        "safety": {
            "read_only": False,
            "live_write": True,
            "live_widening_allowed": True,
            "full_autonomous_ready": True,
        },
        "summary": {
            "artifact_count": 1,
            "invalid_summary_count": 0,
        },
        "autonomy_evidence_overlay": {
            "type": "autonomy_evidence_overlay",
            "generated_at": "2026-06-05T08:20:00Z",
            "status": "dry_run_only",
            "safety": {
                "read_only": False,
                "live_write": True,
                "live_widening_allowed": True,
                "automatic_safety_update_allowed": True,
                "full_autonomous_ready": True,
                "operator_required": False,
            },
            "summary": {
                "status_counts": {
                    "dry_run_only": 1,
                    "manual_safety_review_candidate": 0,
                    "no_current_evidence": 1,
                    "remediation_required": 0,
                },
                "manual_safety_review_candidate_target_ids": [],
                "remediation_required_target_ids": [],
                "dry_run_only_target_ids": ["gain_trim_supervised_promotion"],
                "no_current_evidence_target_ids": ["fader_supervised_promotion"],
                "autonomous_ready_target_ids": ["gain_trim_supervised_promotion"],
                "global_blockers": ["source_only_blocker"],
            },
            "rows": [
                {
                    "id": "gain_autonomy_evidence_overlay",
                    "target_id": "gain_trim_supervised_promotion",
                    "value_type": "gain",
                    "status": "dry_run_only",
                    "safety_review_candidate": False,
                    "supervised_live_allowed_static": False,
                    "autonomous_ready": True,
                    "proof_artifact_count": 1,
                    "physical_candidate_count": 0,
                    "dry_run_ready_count": 1,
                    "blocked_artifact_count": 0,
                    "remediation_action_count": 0,
                    "physical_candidate_proof_ids": [],
                    "dry_run_ready_proof_ids": ["gain-proof-dry-run"],
                    "blocked_proof_ids": [],
                    "remediation_action_ids": [],
                    "blockers": ["physical_proof_required_for_promotion"],
                    "next_required_actions": ["Run one small supervised gain proof on WING."],
                    "set_gain": {"channel": 1, "value": 3.0},
                }
            ],
        },
    }


def test_supervised_proof_evidence_default_roots_include_current_cycle():
    assert "artifacts/no_wing_release_check_current_cycle" in DEFAULT_SUPERVISED_PROOF_REPORT_DIRS


def test_supervised_proof_evidence_snapshot_reads_latest_artifact_as_read_only(server, tmp_path):
    old_dir = tmp_path / "old"
    latest_dir = tmp_path / "latest"
    old_dir.mkdir()
    latest_dir.mkdir()
    old_report = old_dir / "supervised_proof_artifact_report.json"
    latest_report = latest_dir / "supervised_proof_artifact_report.json"
    old_report.write_text(json.dumps(proof_report_with_autonomy_overlay("old")), encoding="utf-8")
    latest_report.write_text(json.dumps(proof_report_with_autonomy_overlay("ok")), encoding="utf-8")
    (latest_dir / "summary.json").write_text(
        json.dumps(
            {
                "type": "no_wing_release_check",
                "status": "passed",
                "read_surface_fixture_evidence": {
                    "status": "passed",
                    "snapshot_status": "fixture_evidence_incomplete",
                    "surface_count": 10,
                    "surface_with_fixture_count": 10,
                    "declared_fixture_path_count": 16,
                    "declared_test_name_count": 43,
                    "physical_gap_surface_count": 10,
                    "summary_recomputed_from_rows": True,
                    "fixture_declaration_validation": True,
                    "contract_gap_consistency_enforced": True,
                    "read_only": True,
                    "live_write": False,
                    "full_autonomous_ready": True,
                    "physical_fixture_missing_surface_ids": [
                        "raw_mixer_state_cache",
                        "channel_identity_and_basic_controls",
                    ],
                    "next_physical_readback_surface_id": "raw_mixer_state_cache",
                    "violations": [],
                },
                "read_surface_physical_readback_capture_plan": {
                    "status": "passed",
                    "plan_status": "operator_capture_required",
                    "read_only": True,
                    "live_write": False,
                    "mutates_console_state": False,
                    "queries_console": False,
                    "executes_tests": False,
                    "operator_required": True,
                    "execution_allowed_by_plan": False,
                    "live_widening_allowed": False,
                    "full_autonomous_ready": False,
                    "capture_step_count": 1,
                    "physical_gap_surface_count": 2,
                    "physical_fixture_missing_surface_ids": [
                        "raw_mixer_state_cache",
                        "channel_identity_and_basic_controls",
                    ],
                    "next_physical_readback_surface_id": "raw_mixer_state_cache",
                    "next_physical_readback_action_id": "remediate_raw_mixer_state_cache",
                    "target_action_consistency": True,
                    "required_artifact_contracts": [
                        "physical_readback_fixture",
                        "raw_mixer_state_physical_readback_contract",
                    ],
                    "blocked_by": [
                        "physical_readback_fixtures_incomplete",
                        "operator_must_capture_and_review_physical_readback_artifact",
                    ],
                    "step_surface_id": "raw_mixer_state_cache",
                    "step_remediation_action_id": "remediate_raw_mixer_state_cache",
                    "step_forbidden_actions": [
                        "do not mark cached state as physical proof",
                    ],
                    "violations": [],
                },
                "raw_mixer_state_physical_readback_contract": {
                    "status": "passed",
                    "schema_id": "raw_mixer_state_physical_readback.v1",
                    "contract_status": "contract_ready_physical_fixture_missing",
                    "surface_id": "raw_mixer_state_cache",
                    "remediation_action_id": "remediate_raw_mixer_state_cache",
                    "next_physical_readback_surface_id": "raw_mixer_state_cache",
                    "next_physical_readback_action_id": "remediate_raw_mixer_state_cache",
                    "artifact_required": True,
                    "physical_fixture_gap_remains": True,
                    "accepted_as_physical_fixture_candidate": False,
                    "read_only": True,
                    "live_write": False,
                    "mutates_console_state": False,
                    "queries_console": False,
                    "executes_tests": False,
                    "live_widening_allowed": False,
                    "live_authority_granted": False,
                    "full_autonomous_ready": False,
                    "target_consistency": True,
                    "allowed_sources": ["physical_readback", "wing_physical_readback"],
                    "accepted_row_statuses": ["verified_from_physical_readback"],
                    "forbidden_mutation_evidence_fields": ["manual_write_events"],
                    "violations": [],
                },
                "raw_mixer_state_physical_readback_artifact_intake": {
                    "status": "passed",
                    "schema_id": "raw_mixer_state_physical_readback.v1",
                    "intake_status": "operator_artifact_missing",
                    "surface_id": "raw_mixer_state_cache",
                    "remediation_action_id": "remediate_raw_mixer_state_cache",
                    "required_contract_method_id": "raw_mixer_state_physical_readback_contract",
                    "artifact_search_roots": [
                        "artifacts/read_surface_physical_readback/raw_mixer_state",
                    ],
                    "artifact_count": 0,
                    "valid_candidate_count": 0,
                    "invalid_candidate_count": 0,
                    "accepted_as_physical_fixture_candidate": False,
                    "operator_review_required": True,
                    "physical_fixture_gap_remains": True,
                    "fixture_closure_allowed_by_intake": False,
                    "read_only": True,
                    "live_write": False,
                    "mutates_console_state": False,
                    "queries_console": False,
                    "executes_tests": False,
                    "operator_required": True,
                    "execution_allowed_by_intake": False,
                    "live_widening_allowed": False,
                    "live_authority_granted": False,
                    "full_autonomous_ready": False,
                    "blocked_by": [
                        "operator_must_capture_raw_mixer_state_physical_readback_artifact",
                    ],
                    "artifact_source_ids": [],
                    "intake_violations": [],
                    "violations": [],
                },
                "channel_basic_controls_physical_readback_contract": {
                    "status": "passed",
                    "schema_id": "channel_basic_controls_physical_readback.v1",
                    "contract_status": "contract_ready_physical_fixture_missing",
                    "surface_id": "channel_identity_and_basic_controls",
                    "remediation_action_id": "remediate_channel_identity_and_basic_controls",
                    "physical_readback_surface_id": "channel_identity_and_basic_controls",
                    "physical_readback_action_id": "remediate_channel_identity_and_basic_controls",
                    "fixture_physical_gap_declared": True,
                    "remediation_physical_gap_declared": True,
                    "artifact_required": True,
                    "physical_fixture_gap_remains": True,
                    "accepted_as_physical_fixture_candidate": False,
                    "read_only": True,
                    "live_write": False,
                    "mutates_console_state": False,
                    "queries_console": False,
                    "executes_tests": False,
                    "live_widening_allowed": False,
                    "live_authority_granted": False,
                    "full_autonomous_ready": False,
                    "target_consistency": True,
                    "promotion_families_requiring_surface": [
                        "fader",
                        "gain",
                        "mute",
                        "pan_phase_delay",
                    ],
                    "required_control_fields": ["fader_db", "trim_db", "mute", "pan"],
                    "required_coverage_flags": [
                        "identity",
                        "basic_controls",
                        "cache_matches_physical_readback",
                    ],
                    "allowed_sources": ["physical_readback", "wing_physical_readback"],
                    "accepted_channel_statuses": ["verified_from_physical_readback"],
                    "forbidden_mutation_evidence_fields": [
                        "manual_write_events",
                        "mute_write_events",
                    ],
                    "violations": [],
                },
                "channel_basic_controls_physical_readback_artifact_intake": {
                    "status": "passed",
                    "schema_id": "channel_basic_controls_physical_readback.v1",
                    "intake_status": "operator_artifact_missing",
                    "surface_id": "channel_identity_and_basic_controls",
                    "remediation_action_id": "remediate_channel_identity_and_basic_controls",
                    "required_contract_method_id": "channel_basic_controls_physical_readback_contract",
                    "artifact_search_roots": [
                        "artifacts/read_surface_physical_readback/channel_basic_controls",
                    ],
                    "artifact_count": 0,
                    "valid_candidate_count": 0,
                    "invalid_candidate_count": 0,
                    "valid_channel_count": 0,
                    "accepted_as_physical_fixture_candidate": False,
                    "operator_review_required": True,
                    "physical_fixture_gap_remains": True,
                    "fixture_closure_allowed_by_intake": False,
                    "read_only": True,
                    "live_write": False,
                    "mutates_console_state": False,
                    "queries_console": False,
                    "executes_tests": False,
                    "operator_required": True,
                    "execution_allowed_by_intake": False,
                    "live_widening_allowed": False,
                    "live_authority_granted": False,
                    "full_autonomous_ready": False,
                    "target_consistency": True,
                    "promotion_families_requiring_surface": [
                        "fader",
                        "gain",
                        "mute",
                        "pan_phase_delay",
                    ],
                    "blocked_by": [
                        "operator_must_capture_channel_basic_controls_physical_readback_artifact",
                    ],
                    "artifact_source_ids": [],
                    "intake_violations": [],
                    "violations": [],
                },
                "channel_processing_physical_readback_contract": {
                    "status": "passed",
                    "schema_id": "channel_processing_physical_readback.v1",
                    "contract_status": "contract_ready_physical_fixture_missing",
                    "surface_id": "channel_eq_dynamics_gate",
                    "remediation_action_id": "remediate_channel_eq_dynamics_gate",
                    "physical_readback_surface_id": "channel_eq_dynamics_gate",
                    "physical_readback_action_id": "remediate_channel_eq_dynamics_gate",
                    "fixture_physical_gap_declared": True,
                    "remediation_physical_gap_declared": True,
                    "artifact_required": True,
                    "physical_fixture_gap_remains": True,
                    "accepted_as_physical_fixture_candidate": False,
                    "read_only": True,
                    "live_write": False,
                    "mutates_console_state": False,
                    "queries_console": False,
                    "executes_tests": False,
                    "live_widening_allowed": False,
                    "live_authority_granted": False,
                    "full_autonomous_ready": False,
                    "target_consistency": True,
                    "promotion_families_requiring_surface": [
                        "eq",
                        "compressor",
                        "gate",
                        "deessing",
                    ],
                    "required_hpf_fields": ["enabled", "frequency_hz"],
                    "required_eq_fields": ["enabled", "bands"],
                    "required_eq_band_fields": ["band", "frequency_hz", "gain_db", "q"],
                    "required_compressor_fields": ["enabled", "model", "gain_reduction_db"],
                    "required_gate_fields": ["enabled", "model", "range_db"],
                    "required_coverage_flags": [
                        "hpf",
                        "eq",
                        "compressor",
                        "gate",
                        "cache_matches_physical_readback",
                    ],
                    "minimum_eq_band_count": 4,
                    "allowed_sources": ["physical_readback", "wing_physical_readback"],
                    "accepted_channel_statuses": ["verified_from_physical_readback"],
                    "forbidden_mutation_evidence_fields": [
                        "manual_write_events",
                        "eq_write_events",
                    ],
                    "violations": [],
                },
                "channel_processing_physical_readback_artifact_intake": {
                    "status": "passed",
                    "schema_id": "channel_processing_physical_readback.v1",
                    "intake_status": "operator_artifact_missing",
                    "surface_id": "channel_eq_dynamics_gate",
                    "remediation_action_id": "remediate_channel_eq_dynamics_gate",
                    "required_contract_method_id": "channel_processing_physical_readback_contract",
                    "artifact_search_roots": [
                        "artifacts/read_surface_physical_readback/channel_processing",
                    ],
                    "artifact_count": 0,
                    "valid_candidate_count": 0,
                    "invalid_candidate_count": 0,
                    "valid_channel_count": 0,
                    "accepted_as_physical_fixture_candidate": False,
                    "operator_review_required": True,
                    "physical_fixture_gap_remains": True,
                    "fixture_closure_allowed_by_intake": False,
                    "read_only": True,
                    "live_write": False,
                    "mutates_console_state": False,
                    "queries_console": False,
                    "executes_tests": False,
                    "operator_required": True,
                    "execution_allowed_by_intake": False,
                    "live_widening_allowed": False,
                    "live_authority_granted": False,
                    "full_autonomous_ready": False,
                    "target_consistency": True,
                    "promotion_families_requiring_surface": [
                        "eq",
                        "compressor",
                        "gate",
                        "deessing",
                        "frequency_separation",
                        "cross_channel_unmasking",
                    ],
                    "blocked_by": [
                        "operator_must_capture_channel_processing_physical_readback_artifact",
                    ],
                    "artifact_source_ids": [],
                    "intake_violations": [],
                    "violations": [],
                },
                "bus_group_main_dca_physical_readback_contract": {
                    "status": "passed",
                    "schema_id": "bus_group_main_dca_physical_readback.v1",
                    "contract_status": "contract_ready_physical_fixture_missing",
                    "surface_id": "bus_group_main_dca_state",
                    "remediation_action_id": "remediate_bus_group_main_dca_state",
                    "physical_readback_surface_id": "bus_group_main_dca_state",
                    "physical_readback_action_id": "remediate_bus_group_main_dca_state",
                    "fixture_physical_gap_declared": True,
                    "remediation_physical_gap_declared": True,
                    "artifact_required": True,
                    "physical_fixture_gap_remains": True,
                    "accepted_as_physical_fixture_candidate": False,
                    "read_only": True,
                    "live_write": False,
                    "mutates_console_state": False,
                    "queries_console": False,
                    "executes_tests": False,
                    "live_widening_allowed": False,
                    "live_authority_granted": False,
                    "full_autonomous_ready": False,
                    "target_consistency": True,
                    "promotion_families_requiring_surface": [
                        "bus_group_matrix",
                        "dca",
                        "master_bus",
                    ],
                    "required_bus_control_fields": ["fader_db", "mute", "pan", "width", "mono"],
                    "required_main_control_fields": ["fader_db", "mute", "pan"],
                    "required_dca_control_fields": ["fader_db", "mute", "monitor"],
                    "required_bus_assignment_fields": [
                        "dca_assignments",
                        "main_send_on",
                        "main_send_level_db",
                        "main_send_pre",
                    ],
                    "required_dca_assignment_fields": ["member_channels", "member_buses"],
                    "required_metering_fields": ["lufs", "true_peak_dbtp", "headroom_db"],
                    "required_bus_coverage_flags": [
                        "controls",
                        "assignments",
                        "metering",
                        "cache_matches_physical_readback",
                    ],
                    "required_main_coverage_flags": [
                        "controls",
                        "metering",
                        "cache_matches_physical_readback",
                    ],
                    "required_dca_coverage_flags": [
                        "identity",
                        "controls",
                        "assignments",
                        "cache_matches_physical_readback",
                    ],
                    "minimum_bus_count": 1,
                    "minimum_main_count": 1,
                    "minimum_dca_count": 1,
                    "allowed_sources": ["physical_readback", "wing_physical_readback"],
                    "accepted_surface_statuses": ["verified_from_physical_readback"],
                    "forbidden_mutation_evidence_fields": [
                        "manual_write_events",
                        "dca_write_events",
                    ],
                    "violations": [],
                },
                "bus_group_main_dca_physical_readback_artifact_intake": {
                    "status": "passed",
                    "schema_id": "bus_group_main_dca_physical_readback.v1",
                    "intake_status": "operator_artifact_missing",
                    "surface_id": "bus_group_main_dca_state",
                    "remediation_action_id": "remediate_bus_group_main_dca_state",
                    "required_contract_method_id": "bus_group_main_dca_physical_readback_contract",
                    "artifact_search_roots": [
                        "artifacts/read_surface_physical_readback/bus_group_main_dca",
                    ],
                    "artifact_count": 0,
                    "valid_candidate_count": 0,
                    "invalid_candidate_count": 0,
                    "valid_bus_count": 0,
                    "valid_main_count": 0,
                    "valid_dca_count": 0,
                    "accepted_as_physical_fixture_candidate": False,
                    "operator_review_required": True,
                    "physical_fixture_gap_remains": True,
                    "fixture_closure_allowed_by_intake": False,
                    "read_only": True,
                    "live_write": False,
                    "mutates_console_state": False,
                    "queries_console": False,
                    "executes_tests": False,
                    "operator_required": True,
                    "execution_allowed_by_intake": False,
                    "live_widening_allowed": False,
                    "live_authority_granted": False,
                    "full_autonomous_ready": False,
                    "target_consistency": True,
                    "promotion_families_requiring_surface": [
                        "bus_group_matrix",
                        "dca",
                        "master_bus",
                        "mix_balance",
                        "performance_adaptation",
                    ],
                    "blocked_by": [
                        "operator_must_capture_bus_group_main_dca_physical_readback_artifact",
                    ],
                    "artifact_source_ids": [],
                    "intake_violations": [],
                    "violations": [],
                },
                "matrix_state_physical_readback_contract": {
                    "status": "passed",
                    "schema_id": "matrix_state_physical_readback.v1",
                    "contract_status": "contract_ready_physical_fixture_missing",
                    "surface_id": "matrix_state",
                    "remediation_action_id": "remediate_matrix_state",
                    "physical_readback_surface_id": "matrix_state",
                    "physical_readback_action_id": "remediate_matrix_state",
                    "fixture_physical_gap_declared": True,
                    "remediation_physical_gap_declared": True,
                    "artifact_required": True,
                    "physical_fixture_gap_remains": True,
                    "accepted_as_physical_fixture_candidate": False,
                    "read_only": True,
                    "live_write": False,
                    "mutates_console_state": False,
                    "queries_console": False,
                    "executes_tests": False,
                    "live_widening_allowed": False,
                    "live_authority_granted": False,
                    "full_autonomous_ready": False,
                    "target_consistency": True,
                    "promotion_families_requiring_surface": [
                        "bus_group_matrix",
                        "fx_send_insert",
                        "routing",
                    ],
                    "required_control_fields": ["fader_db", "mute", "pan"],
                    "required_eq_fields": ["enabled", "bands"],
                    "required_eq_band_fields": ["band", "frequency_hz", "gain_db", "q"],
                    "required_dynamics_fields": ["enabled", "model", "gain_reduction_db"],
                    "required_insert_fields": ["pre_insert_on", "post_insert_weight"],
                    "required_metering_fields": ["lufs", "true_peak_dbtp", "headroom_db"],
                    "required_coverage_flags": [
                        "controls",
                        "eq",
                        "dynamics",
                        "inserts",
                        "metering",
                        "cache_matches_physical_readback",
                    ],
                    "minimum_matrix_count": 1,
                    "minimum_eq_band_count": 8,
                    "allowed_sources": ["physical_readback", "wing_physical_readback"],
                    "accepted_matrix_statuses": ["verified_from_physical_readback"],
                    "forbidden_mutation_evidence_fields": [
                        "manual_write_events",
                        "matrix_write_events",
                    ],
                    "violations": [],
                },
                "matrix_state_physical_readback_artifact_intake": {
                    "status": "passed",
                    "schema_id": "matrix_state_physical_readback.v1",
                    "intake_status": "operator_artifact_missing",
                    "surface_id": "matrix_state",
                    "remediation_action_id": "remediate_matrix_state",
                    "required_contract_method_id": "matrix_state_physical_readback_contract",
                    "artifact_search_roots": [
                        "artifacts/read_surface_physical_readback/matrix_state",
                    ],
                    "artifact_count": 0,
                    "valid_candidate_count": 0,
                    "invalid_candidate_count": 0,
                    "valid_matrix_count": 0,
                    "accepted_as_physical_fixture_candidate": False,
                    "operator_review_required": True,
                    "physical_fixture_gap_remains": True,
                    "fixture_closure_allowed_by_intake": False,
                    "read_only": True,
                    "live_write": False,
                    "mutates_console_state": False,
                    "queries_console": False,
                    "executes_tests": False,
                    "operator_required": True,
                    "execution_allowed_by_intake": False,
                    "live_widening_allowed": False,
                    "live_authority_granted": False,
                    "full_autonomous_ready": False,
                    "target_consistency": True,
                    "promotion_families_requiring_surface": [
                        "bus_group_matrix",
                        "fx_send_insert",
                        "master_bus",
                        "mix_balance",
                        "routing",
                        "performance_adaptation",
                    ],
                    "blocked_by": [
                        "operator_must_capture_matrix_state_physical_readback_artifact",
                    ],
                    "artifact_source_ids": [],
                    "intake_violations": [],
                    "violations": [],
                },
                "fx_slots_inserts_physical_readback_contract": {
                    "status": "passed",
                    "schema_id": "fx_slots_inserts_physical_readback.v1",
                    "contract_status": "contract_ready_physical_fixture_missing",
                    "surface_id": "fx_slots_and_inserts",
                    "remediation_action_id": "remediate_fx_slots_and_inserts",
                    "physical_readback_surface_id": "fx_slots_and_inserts",
                    "physical_readback_action_id": "remediate_fx_slots_and_inserts",
                    "fixture_physical_gap_declared": True,
                    "remediation_physical_gap_declared": True,
                    "artifact_required": True,
                    "physical_fixture_gap_remains": True,
                    "accepted_as_physical_fixture_candidate": False,
                    "read_only": True,
                    "live_write": False,
                    "mutates_console_state": False,
                    "queries_console": False,
                    "executes_tests": False,
                    "live_widening_allowed": False,
                    "live_authority_granted": False,
                    "full_autonomous_ready": False,
                    "target_consistency": True,
                    "promotion_families_requiring_surface": [
                        "fx_send_insert",
                        "routing",
                        "mix_balance",
                    ],
                    "required_fx_slot_fields": ["slot", "slot_number", "parameter_schema"],
                    "required_parameter_fields": ["parameter_index", "value", "safe_range"],
                    "required_parameter_schema_fields": [
                        "status",
                        "physical_model_schema_verified",
                    ],
                    "required_fx_slot_coverage_flags": [
                        "identity",
                        "assignment",
                        "cache_matches_physical_readback",
                    ],
                    "required_insert_fields": ["target_type", "position", "linked_fx_slot"],
                    "required_insert_coverage_flags": [
                        "assignment",
                        "status",
                        "scoped_diff_marker",
                    ],
                    "required_false_safety_flags": [
                        "live_write",
                        "mutates_console_state",
                        "live_widening_allowed",
                        "full_autonomous_ready",
                    ],
                    "minimum_fx_slot_count": 1,
                    "minimum_fx_parameter_count": 4,
                    "minimum_insert_count": 1,
                    "allowed_sources": ["physical_readback", "wing_physical_readback"],
                    "accepted_row_statuses": ["verified_from_physical_readback"],
                    "allowed_insert_target_types": ["channel", "bus", "main", "matrix"],
                    "allowed_insert_positions": ["pre", "post"],
                    "forbidden_mutation_evidence_fields": [
                        "manual_write_events",
                        "fx_parameter_write_events",
                    ],
                    "violations": [],
                },
                "fx_slots_inserts_physical_readback_artifact_intake": {
                    "status": "passed",
                    "schema_id": "fx_slots_inserts_physical_readback.v1",
                    "intake_status": "operator_artifact_missing",
                    "surface_id": "fx_slots_and_inserts",
                    "remediation_action_id": "remediate_fx_slots_and_inserts",
                    "required_contract_method_id": "fx_slots_inserts_physical_readback_contract",
                    "artifact_search_roots": [
                        "artifacts/read_surface_physical_readback/fx_slots_and_inserts",
                    ],
                    "artifact_count": 0,
                    "valid_candidate_count": 0,
                    "invalid_candidate_count": 0,
                    "valid_fx_slot_count": 0,
                    "valid_insert_count": 0,
                    "accepted_as_physical_fixture_candidate": False,
                    "operator_review_required": True,
                    "physical_fixture_gap_remains": True,
                    "fixture_closure_allowed_by_intake": False,
                    "read_only": True,
                    "live_write": False,
                    "mutates_console_state": False,
                    "queries_console": False,
                    "executes_tests": False,
                    "operator_required": True,
                    "execution_allowed_by_intake": False,
                    "live_widening_allowed": False,
                    "live_authority_granted": False,
                    "full_autonomous_ready": False,
                    "target_consistency": True,
                    "promotion_families_requiring_surface": [
                        "fx_send_insert",
                        "routing",
                        "mix_balance",
                        "performance_adaptation",
                    ],
                    "blocked_by": [
                        "operator_must_capture_fx_slots_inserts_physical_readback_artifact",
                    ],
                    "artifact_source_ids": [],
                    "intake_violations": [],
                    "violations": [],
                },
                "routing_sends_physical_readback_contract": {
                    "status": "passed",
                    "schema_id": "routing_sends_physical_readback.v1",
                    "contract_status": "contract_ready_physical_fixture_missing",
                    "surface_id": "routing_and_sends",
                    "remediation_action_id": "remediate_routing_and_sends",
                    "physical_readback_surface_id": "routing_and_sends",
                    "physical_readback_action_id": "remediate_routing_and_sends",
                    "fixture_physical_gap_declared": True,
                    "remediation_physical_gap_declared": True,
                    "artifact_required": True,
                    "physical_fixture_gap_remains": True,
                    "accepted_as_physical_fixture_candidate": False,
                    "read_only": True,
                    "live_write": False,
                    "mutates_console_state": False,
                    "queries_console": False,
                    "executes_tests": False,
                    "live_widening_allowed": False,
                    "live_authority_granted": False,
                    "full_autonomous_ready": False,
                    "target_consistency": True,
                    "promotion_families_requiring_surface": [
                        "routing",
                        "fx_send_insert",
                        "bus_group_matrix",
                        "scene_preset",
                        "mix_balance",
                        "performance_adaptation",
                    ],
                    "required_static_route_fields": [
                        "start",
                        "end",
                        "role",
                        "wing_routing_hint",
                    ],
                    "required_bus_send_fields": ["channel", "send", "level_db", "pre_on"],
                    "required_main_send_fields": ["channel", "main", "level_db", "pre"],
                    "required_matrix_send_fields": ["channel", "send", "target_matrix"],
                    "required_output_route_fields": [
                        "output_group",
                        "output_number",
                        "source_channel",
                    ],
                    "required_static_route_coverage_flags": [
                        "identity",
                        "static_config_match",
                        "physical_route_mapping",
                        "scoped_diff_marker",
                    ],
                    "required_send_coverage_flags": [
                        "assignment",
                        "level",
                        "mode",
                        "scoped_diff_marker",
                        "cache_matches_physical_readback",
                    ],
                    "required_output_route_coverage_flags": [
                        "assignment",
                        "source",
                        "scoped_diff_marker",
                        "cache_matches_physical_readback",
                    ],
                    "required_false_safety_flags": [
                        "live_write",
                        "mutates_console_state",
                        "live_widening_allowed",
                        "full_autonomous_ready",
                    ],
                    "minimum_static_dante_route_count": 1,
                    "minimum_channel_bus_send_count": 1,
                    "minimum_channel_main_send_count": 1,
                    "minimum_channel_matrix_send_count": 1,
                    "minimum_output_route_count": 1,
                    "allowed_sources": [
                        "physical_readback",
                        "wing_physical_readback",
                        "dlive_physical_readback",
                    ],
                    "accepted_row_statuses": ["verified_from_physical_readback"],
                    "forbidden_mutation_evidence_fields": [
                        "manual_write_events",
                        "routing_write_events",
                    ],
                    "violations": [],
                },
                "routing_sends_physical_readback_artifact_intake": {
                    "status": "passed",
                    "schema_id": "routing_sends_physical_readback.v1",
                    "intake_status": "operator_artifact_missing",
                    "surface_id": "routing_and_sends",
                    "remediation_action_id": "remediate_routing_and_sends",
                    "required_contract_method_id": "routing_sends_physical_readback_contract",
                    "artifact_search_roots": [
                        "artifacts/read_surface_physical_readback/routing_and_sends",
                    ],
                    "artifact_count": 0,
                    "valid_candidate_count": 0,
                    "invalid_candidate_count": 0,
                    "valid_static_dante_route_count": 0,
                    "valid_channel_bus_send_count": 0,
                    "valid_channel_main_send_count": 0,
                    "valid_channel_matrix_send_count": 0,
                    "valid_output_route_count": 0,
                    "accepted_as_physical_fixture_candidate": False,
                    "operator_review_required": True,
                    "physical_fixture_gap_remains": True,
                    "fixture_closure_allowed_by_intake": False,
                    "read_only": True,
                    "live_write": False,
                    "mutates_console_state": False,
                    "queries_console": False,
                    "executes_tests": False,
                    "operator_required": True,
                    "execution_allowed_by_intake": False,
                    "live_widening_allowed": False,
                    "live_authority_granted": False,
                    "full_autonomous_ready": False,
                    "target_consistency": True,
                    "promotion_families_requiring_surface": [
                        "routing",
                        "fx_send_insert",
                        "bus_group_matrix",
                        "scene_preset",
                        "mix_balance",
                        "performance_adaptation",
                    ],
                    "blocked_by": [
                        "operator_must_capture_routing_sends_physical_readback_artifact",
                    ],
                    "artifact_source_ids": [],
                    "intake_violations": [],
                    "violations": [],
                },
                "snapshots_scenes_physical_readback_contract": {
                    "status": "passed",
                    "schema_id": "snapshots_scenes_physical_readback.v1",
                    "contract_status": "contract_ready_physical_fixture_missing",
                    "surface_id": "snapshots_scenes",
                    "remediation_action_id": "remediate_snapshots_scenes",
                    "physical_readback_surface_id": "snapshots_scenes",
                    "physical_readback_action_id": "remediate_snapshots_scenes",
                    "fixture_physical_gap_declared": True,
                    "remediation_physical_gap_declared": True,
                    "artifact_required": True,
                    "physical_fixture_gap_remains": True,
                    "accepted_as_physical_fixture_candidate": False,
                    "read_only": True,
                    "live_write": False,
                    "mutates_console_state": False,
                    "queries_console": False,
                    "executes_tests": False,
                    "live_widening_allowed": False,
                    "live_authority_granted": False,
                    "full_autonomous_ready": False,
                    "scene_load_allowed": False,
                    "scene_save_allowed": False,
                    "target_consistency": True,
                    "promotion_families_requiring_surface": [
                        "scene_preset",
                        "routing",
                        "mix_balance",
                        "performance_adaptation",
                    ],
                    "required_active_scene_fields": [
                        "active_snapshot_index",
                        "active_snapshot_name",
                        "current_show",
                    ],
                    "required_snapshot_fields": [
                        "snapshot_index",
                        "snapshot_name",
                        "scene_content_fingerprint",
                        "rollback_reference",
                    ],
                    "required_coverage_flags": [
                        "catalog_identity",
                        "active_scene",
                        "rollback_reference",
                    ],
                    "required_false_safety_flags": [
                        "live_write",
                        "mutates_console_state",
                        "scene_load_allowed",
                        "scene_save_allowed",
                    ],
                    "minimum_snapshot_count": 1,
                    "allowed_sources": ["physical_readback", "wing_physical_readback"],
                    "accepted_row_statuses": ["verified_from_physical_readback"],
                    "forbidden_mutation_evidence_fields": [
                        "manual_write_events",
                        "scene_recall_events",
                    ],
                    "violations": [],
                },
                "snapshots_scenes_physical_readback_artifact_intake": {
                    "status": "passed",
                    "schema_id": "snapshots_scenes_physical_readback.v1",
                    "intake_status": "operator_artifact_missing",
                    "surface_id": "snapshots_scenes",
                    "remediation_action_id": "remediate_snapshots_scenes",
                    "required_contract_method_id": "snapshots_scenes_physical_readback_contract",
                    "artifact_search_roots": [
                        "artifacts/read_surface_physical_readback/snapshots_scenes",
                    ],
                    "artifact_count": 0,
                    "valid_candidate_count": 0,
                    "invalid_candidate_count": 0,
                    "valid_snapshot_count": 0,
                    "accepted_as_physical_fixture_candidate": False,
                    "operator_review_required": True,
                    "physical_fixture_gap_remains": True,
                    "fixture_closure_allowed_by_intake": False,
                    "scene_load_allowed": False,
                    "scene_save_allowed": False,
                    "read_only": True,
                    "live_write": False,
                    "mutates_console_state": False,
                    "queries_console": False,
                    "executes_tests": False,
                    "operator_required": True,
                    "execution_allowed_by_intake": False,
                    "live_widening_allowed": False,
                    "live_authority_granted": False,
                    "full_autonomous_ready": False,
                    "target_consistency": True,
                    "promotion_families_requiring_surface": [
                        "scene_preset",
                        "routing",
                        "mix_balance",
                        "performance_adaptation",
                    ],
                    "blocked_by": [
                        "operator_must_capture_snapshots_scenes_physical_readback_artifact",
                    ],
                    "artifact_source_ids": [],
                    "intake_violations": [],
                    "violations": [],
                },
                "user_presets_physical_readback_contract": {
                    "status": "passed",
                    "schema_id": "user_presets_physical_readback.v1",
                    "contract_status": "contract_ready_physical_fixture_missing",
                    "surface_id": "user_presets",
                    "remediation_action_id": "remediate_user_presets",
                    "physical_readback_surface_id": "user_presets",
                    "physical_readback_action_id": "remediate_user_presets",
                    "fixture_physical_gap_declared": True,
                    "remediation_physical_gap_declared": True,
                    "artifact_required": True,
                    "physical_fixture_gap_remains": True,
                    "accepted_as_physical_fixture_candidate": False,
                    "read_only": True,
                    "live_write": False,
                    "mutates_console_state": False,
                    "queries_console": False,
                    "executes_tests": False,
                    "live_widening_allowed": False,
                    "live_authority_granted": False,
                    "full_autonomous_ready": False,
                    "preset_recall_allowed": False,
                    "preset_store_allowed": False,
                    "target_consistency": True,
                    "promotion_families_requiring_surface": [
                        "scene_preset",
                        "routing",
                        "mix_balance",
                        "performance_adaptation",
                    ],
                    "required_preset_fields": [
                        "preset_index",
                        "preset_name",
                        "preset_scope",
                        "target_type",
                        "target_label",
                        "preset_content_fingerprint",
                        "rollback_reference",
                    ],
                    "required_coverage_flags": [
                        "catalog_identity",
                        "scope",
                        "target",
                        "rollback_reference",
                    ],
                    "required_false_safety_flags": [
                        "live_write",
                        "mutates_console_state",
                        "preset_recall_allowed",
                        "preset_store_allowed",
                    ],
                    "minimum_preset_count": 1,
                    "allowed_sources": ["physical_readback", "wing_physical_readback"],
                    "accepted_row_statuses": ["verified_from_physical_readback"],
                    "forbidden_mutation_evidence_fields": [
                        "manual_write_events",
                        "preset_recall_events",
                        "preset_store_events",
                    ],
                    "violations": [],
                },
                "user_presets_physical_readback_artifact_intake": {
                    "status": "passed",
                    "schema_id": "user_presets_physical_readback.v1",
                    "intake_status": "operator_artifact_missing",
                    "surface_id": "user_presets",
                    "remediation_action_id": "remediate_user_presets",
                    "required_contract_method_id": (
                        "user_presets_physical_readback_contract"
                    ),
                    "artifact_count": 0,
                    "valid_candidate_count": 0,
                    "invalid_candidate_count": 0,
                    "valid_preset_count": 0,
                    "accepted_as_physical_fixture_candidate": False,
                    "operator_review_required": True,
                    "physical_fixture_gap_remains": True,
                    "fixture_closure_allowed_by_intake": False,
                    "preset_recall_allowed": False,
                    "preset_store_allowed": False,
                    "live_authority_granted": False,
                    "full_autonomous_ready": False,
                    "target_consistency": True,
                    "promotion_families_requiring_surface": [
                        "scene_preset",
                        "routing",
                        "mix_balance",
                        "performance_adaptation",
                    ],
                    "blocked_by": [
                        "operator_must_capture_user_presets_physical_readback_artifact",
                    ],
                    "artifact_source_ids": [],
                    "intake_violations": [],
                    "violations": [],
                },
                "dlive_state_physical_readback_contract": {
                    "status": "passed",
                    "schema_id": "dlive_state_physical_readback.v1",
                    "contract_status": "contract_ready_physical_fixture_missing",
                    "surface_id": "dlive_state",
                    "remediation_action_id": "remediate_dlive_state",
                    "physical_readback_surface_id": "dlive_state",
                    "physical_readback_action_id": "remediate_dlive_state",
                    "fixture_physical_gap_declared": True,
                    "remediation_physical_gap_declared": True,
                    "artifact_required": True,
                    "physical_fixture_gap_remains": True,
                    "accepted_as_physical_fixture_candidate": False,
                    "read_only": True,
                    "live_write": False,
                    "mutates_console_state": False,
                    "queries_console": False,
                    "executes_tests": False,
                    "live_widening_allowed": False,
                    "live_authority_granted": False,
                    "full_autonomous_ready": False,
                    "midi_send_allowed": False,
                    "tcp_send_allowed": False,
                    "dlive_write_allowed": False,
                    "dlive_apply_allowed": False,
                    "target_consistency": True,
                    "promotion_families_requiring_surface": [
                        "fader",
                        "gain",
                        "mute",
                        "routing",
                        "performance_adaptation",
                    ],
                    "required_channel_fields": [
                        "channel",
                        "cached_channel_name",
                        "fader_raw_14bit",
                    ],
                    "required_midi_state_fields": [
                        "midi_base_channel",
                        "raw_state_key_count",
                    ],
                    "required_parity_fields": [
                        "wing_surface_equivalent",
                        "mapping_status",
                    ],
                    "required_coverage_flags": [
                        "channel_identity",
                        "wing_parity_mapping",
                    ],
                    "required_false_safety_flags": [
                        "live_write",
                        "midi_send_allowed",
                        "tcp_send_allowed",
                        "dlive_write_allowed",
                    ],
                    "minimum_channel_count": 1,
                    "allowed_sources": ["physical_readback", "dlive_physical_readback"],
                    "accepted_row_statuses": ["verified_from_physical_readback"],
                    "forbidden_mutation_evidence_fields": [
                        "midi_send_events",
                        "tcp_send_events",
                        "dlive_write_events",
                    ],
                    "violations": [],
                },
                "dlive_state_physical_readback_artifact_intake": {
                    "status": "passed",
                    "schema_id": "dlive_state_physical_readback.v1",
                    "intake_status": "operator_artifact_missing",
                    "surface_id": "dlive_state",
                    "remediation_action_id": "remediate_dlive_state",
                    "required_contract_method_id": (
                        "dlive_state_physical_readback_contract"
                    ),
                    "artifact_count": 0,
                    "valid_candidate_count": 0,
                    "invalid_candidate_count": 0,
                    "valid_channel_count": 0,
                    "accepted_as_physical_fixture_candidate": False,
                    "operator_review_required": True,
                    "physical_fixture_gap_remains": True,
                    "fixture_closure_allowed_by_intake": False,
                    "midi_send_allowed": False,
                    "tcp_send_allowed": False,
                    "dlive_write_allowed": False,
                    "dlive_apply_allowed": False,
                    "live_authority_granted": False,
                    "full_autonomous_ready": False,
                    "target_consistency": True,
                    "promotion_families_requiring_surface": [
                        "fader",
                        "gain",
                        "mute",
                        "bus_group_matrix",
                        "routing",
                        "scene_preset",
                        "mix_balance",
                        "performance_adaptation",
                    ],
                    "blocked_by": [
                        "operator_must_capture_dlive_state_physical_readback_artifact",
                    ],
                    "artifact_source_ids": [],
                    "intake_violations": [],
                    "violations": [],
                },
                "read_surface_physical_readback_fixture_closure_plan": {
                    "status": "passed",
                    "plan_status": "manual_fixture_closure_candidate_available",
                    "ledger_status": "operator_acceptance_records_available",
                    "review_candidate_count": 2,
                    "acceptance_record_count": 2,
                    "accepted_record_count": 1,
                    "rejected_record_count": 1,
                    "invalid_record_count": 0,
                    "unreviewed_candidate_count": 0,
                    "closure_candidate_count": 1,
                    "closure_candidate_surface_ids": ["raw_mixer_state_cache"],
                    "rejected_surface_ids": ["dlive_state"],
                    "invalid_record_ids": [],
                    "unreviewed_candidate_surface_ids": [],
                    "manual_fixture_closure_candidate_available": True,
                    "manual_fixture_closure_required": True,
                    "operator_review_required": True,
                    "physical_fixture_gap_remains": True,
                    "fixture_update_allowed_by_plan": False,
                    "fixture_closure_allowed_by_plan": False,
                    "live_write": False,
                    "mutates_console_state": False,
                    "queries_console": False,
                    "executes_tests": False,
                    "execution_allowed_by_plan": False,
                    "live_authority_granted": False,
                    "full_autonomous_ready": False,
                    "blocked_by": [
                        "manual_fixture_baseline_update_required_before_gap_closure",
                    ],
                    "violations": [],
                },
            }
        ),
        encoding="utf-8",
    )
    old_time = time.time() - 100
    latest_time = time.time()
    os.utime(old_report, (old_time, old_time))
    os.utime(latest_report, (latest_time, latest_time))
    server.config["supervised_proof_evidence"] = {"artifact_root": str(tmp_path)}

    payload = build_supervised_proof_evidence_snapshot(server)

    assert payload["type"] == "supervised_proof_evidence_snapshot"
    assert payload["available"] is True
    assert payload["report_path"] == str(latest_report)
    assert payload["safety"] == {
        "read_only": True,
        "live_write": False,
        "generates_artifacts": False,
        "live_widening_allowed": False,
        "automatic_safety_update_allowed": False,
        "full_autonomous_ready": False,
    }
    assert payload["summary"]["latest_report_stale"] is False
    assert payload["summary"]["latest_report_age_sec"] is not None
    assert payload["summary"]["stale_after_sec"] == 24 * 60 * 60
    proof_quality = payload["summary"]["read_surface_proof_quality"]
    assert proof_quality["available"] is True
    assert proof_quality["summary_path"] == str(latest_dir / "summary.json")
    assert proof_quality["declared_fixture_path_count"] == 16
    assert proof_quality["declared_test_name_count"] == 43
    assert proof_quality["physical_gap_surface_count"] == 10
    assert proof_quality["summary_recomputed_from_rows"] is True
    assert proof_quality["fixture_declaration_validation"] is True
    assert proof_quality["contract_gap_consistency_enforced"] is True
    assert proof_quality["full_autonomous_ready"] is False
    assert proof_quality["source_full_autonomous_ready_claimed"] is True
    assert proof_quality["physical_fixture_missing_surface_ids"] == [
        "raw_mixer_state_cache",
        "channel_identity_and_basic_controls",
    ]
    assert proof_quality["next_physical_readback_surface_id"] == "raw_mixer_state_cache"
    assert proof_quality["source_next_physical_readback_surface_id"] == "raw_mixer_state_cache"
    assert proof_quality["next_physical_readback_target_consistent"] is True
    capture_plan = proof_quality["read_surface_physical_readback_capture_plan"]
    assert capture_plan["available"] is True
    assert capture_plan["status"] == "passed"
    assert capture_plan["plan_status"] == "operator_capture_required"
    assert capture_plan["read_only"] is True
    assert capture_plan["live_write"] is False
    assert capture_plan["mutates_console_state"] is False
    assert capture_plan["queries_console"] is False
    assert capture_plan["executes_tests"] is False
    assert capture_plan["operator_required"] is True
    assert capture_plan["execution_allowed_by_plan"] is False
    assert capture_plan["live_authority_granted"] is False
    assert capture_plan["full_autonomous_ready"] is False
    assert capture_plan["capture_step_count"] == 1
    assert capture_plan["next_physical_readback_surface_id"] == "raw_mixer_state_cache"
    assert capture_plan["next_physical_readback_action_id"] == "remediate_raw_mixer_state_cache"
    assert capture_plan["target_consistency"] is True
    assert "physical_readback_fixture" in capture_plan["required_artifact_contracts"]
    assert "operator_must_capture_and_review_physical_readback_artifact" in capture_plan["blocked_by"]
    raw_contract = proof_quality["raw_mixer_state_physical_readback_contract"]
    assert raw_contract["available"] is True
    assert raw_contract["status"] == "passed"
    assert raw_contract["schema_id"] == "raw_mixer_state_physical_readback.v1"
    assert raw_contract["contract_status"] == "contract_ready_physical_fixture_missing"
    assert raw_contract["surface_id"] == "raw_mixer_state_cache"
    assert raw_contract["remediation_action_id"] == "remediate_raw_mixer_state_cache"
    assert raw_contract["artifact_required"] is True
    assert raw_contract["physical_fixture_gap_remains"] is True
    assert raw_contract["accepted_as_physical_fixture_candidate"] is False
    assert raw_contract["live_authority_granted"] is False
    assert raw_contract["full_autonomous_ready"] is False
    assert raw_contract["target_consistency"] is True
    assert raw_contract["next_physical_readback_surface_id"] == "raw_mixer_state_cache"
    assert raw_contract["next_physical_readback_action_id"] == "remediate_raw_mixer_state_cache"
    assert "physical_readback" in raw_contract["allowed_sources"]
    assert "manual_write_events" in raw_contract["forbidden_mutation_evidence_fields"]
    raw_intake = proof_quality["raw_mixer_state_physical_readback_artifact_intake"]
    assert raw_intake["available"] is True
    assert raw_intake["status"] == "passed"
    assert raw_intake["schema_id"] == "raw_mixer_state_physical_readback.v1"
    assert raw_intake["intake_status"] == "operator_artifact_missing"
    assert raw_intake["surface_id"] == "raw_mixer_state_cache"
    assert raw_intake["remediation_action_id"] == "remediate_raw_mixer_state_cache"
    assert raw_intake["required_contract_method_id"] == "raw_mixer_state_physical_readback_contract"
    assert raw_intake["artifact_count"] == 0
    assert raw_intake["valid_candidate_count"] == 0
    assert raw_intake["invalid_candidate_count"] == 0
    assert raw_intake["accepted_as_physical_fixture_candidate"] is False
    assert raw_intake["operator_review_required"] is True
    assert raw_intake["physical_fixture_gap_remains"] is True
    assert raw_intake["fixture_closure_allowed_by_intake"] is False
    assert raw_intake["live_authority_granted"] is False
    assert raw_intake["full_autonomous_ready"] is False
    assert raw_intake["target_consistency"] is True
    assert "operator_must_capture_raw_mixer_state_physical_readback_artifact" in raw_intake["blocked_by"]
    channel_contract = proof_quality["channel_basic_controls_physical_readback_contract"]
    assert channel_contract["available"] is True
    assert channel_contract["status"] == "passed"
    assert channel_contract["schema_id"] == "channel_basic_controls_physical_readback.v1"
    assert channel_contract["contract_status"] == "contract_ready_physical_fixture_missing"
    assert channel_contract["surface_id"] == "channel_identity_and_basic_controls"
    assert channel_contract["remediation_action_id"] == "remediate_channel_identity_and_basic_controls"
    assert channel_contract["artifact_required"] is True
    assert channel_contract["physical_fixture_gap_remains"] is True
    assert channel_contract["accepted_as_physical_fixture_candidate"] is False
    assert channel_contract["live_authority_granted"] is False
    assert channel_contract["full_autonomous_ready"] is False
    assert channel_contract["fixture_physical_gap_declared"] is True
    assert channel_contract["remediation_physical_gap_declared"] is True
    assert channel_contract["target_consistency"] is True
    assert channel_contract["promotion_families_requiring_surface"] == [
        "fader",
        "gain",
        "mute",
        "pan_phase_delay",
    ]
    assert channel_contract["required_control_fields"] == ["fader_db", "trim_db", "mute", "pan"]
    assert "cache_matches_physical_readback" in channel_contract["required_coverage_flags"]
    assert "physical_readback" in channel_contract["allowed_sources"]
    assert "mute_write_events" in channel_contract["forbidden_mutation_evidence_fields"]
    channel_intake = proof_quality["channel_basic_controls_physical_readback_artifact_intake"]
    assert channel_intake["available"] is True
    assert channel_intake["status"] == "passed"
    assert channel_intake["schema_id"] == "channel_basic_controls_physical_readback.v1"
    assert channel_intake["intake_status"] == "operator_artifact_missing"
    assert channel_intake["surface_id"] == "channel_identity_and_basic_controls"
    assert channel_intake["remediation_action_id"] == "remediate_channel_identity_and_basic_controls"
    assert (
        channel_intake["required_contract_method_id"]
        == "channel_basic_controls_physical_readback_contract"
    )
    assert channel_intake["artifact_count"] == 0
    assert channel_intake["valid_candidate_count"] == 0
    assert channel_intake["invalid_candidate_count"] == 0
    assert channel_intake["valid_channel_count"] == 0
    assert channel_intake["accepted_as_physical_fixture_candidate"] is False
    assert channel_intake["operator_review_required"] is True
    assert channel_intake["physical_fixture_gap_remains"] is True
    assert channel_intake["fixture_closure_allowed_by_intake"] is False
    assert channel_intake["live_authority_granted"] is False
    assert channel_intake["full_autonomous_ready"] is False
    assert channel_intake["target_consistency"] is True
    assert channel_intake["promotion_families_requiring_surface"] == [
        "fader",
        "gain",
        "mute",
        "pan_phase_delay",
    ]
    assert (
        "operator_must_capture_channel_basic_controls_physical_readback_artifact"
        in channel_intake["blocked_by"]
    )
    processing_contract = proof_quality["channel_processing_physical_readback_contract"]
    assert processing_contract["available"] is True
    assert processing_contract["status"] == "passed"
    assert processing_contract["schema_id"] == "channel_processing_physical_readback.v1"
    assert processing_contract["contract_status"] == "contract_ready_physical_fixture_missing"
    assert processing_contract["surface_id"] == "channel_eq_dynamics_gate"
    assert processing_contract["remediation_action_id"] == "remediate_channel_eq_dynamics_gate"
    assert processing_contract["artifact_required"] is True
    assert processing_contract["physical_fixture_gap_remains"] is True
    assert processing_contract["accepted_as_physical_fixture_candidate"] is False
    assert processing_contract["live_authority_granted"] is False
    assert processing_contract["full_autonomous_ready"] is False
    assert processing_contract["fixture_physical_gap_declared"] is True
    assert processing_contract["remediation_physical_gap_declared"] is True
    assert processing_contract["target_consistency"] is True
    assert processing_contract["promotion_families_requiring_surface"] == [
        "eq",
        "compressor",
        "gate",
        "deessing",
    ]
    assert processing_contract["required_eq_band_fields"] == ["band", "frequency_hz", "gain_db", "q"]
    assert "gain_reduction_db" in processing_contract["required_compressor_fields"]
    assert "range_db" in processing_contract["required_gate_fields"]
    assert processing_contract["minimum_eq_band_count"] == 4
    assert "cache_matches_physical_readback" in processing_contract["required_coverage_flags"]
    assert "physical_readback" in processing_contract["allowed_sources"]
    assert "eq_write_events" in processing_contract["forbidden_mutation_evidence_fields"]
    processing_intake = proof_quality["channel_processing_physical_readback_artifact_intake"]
    assert processing_intake["available"] is True
    assert processing_intake["status"] == "passed"
    assert processing_intake["schema_id"] == "channel_processing_physical_readback.v1"
    assert processing_intake["intake_status"] == "operator_artifact_missing"
    assert processing_intake["surface_id"] == "channel_eq_dynamics_gate"
    assert processing_intake["remediation_action_id"] == "remediate_channel_eq_dynamics_gate"
    assert (
        processing_intake["required_contract_method_id"]
        == "channel_processing_physical_readback_contract"
    )
    assert processing_intake["artifact_count"] == 0
    assert processing_intake["valid_candidate_count"] == 0
    assert processing_intake["invalid_candidate_count"] == 0
    assert processing_intake["valid_channel_count"] == 0
    assert processing_intake["accepted_as_physical_fixture_candidate"] is False
    assert processing_intake["operator_review_required"] is True
    assert processing_intake["physical_fixture_gap_remains"] is True
    assert processing_intake["fixture_closure_allowed_by_intake"] is False
    assert processing_intake["live_authority_granted"] is False
    assert processing_intake["full_autonomous_ready"] is False
    assert processing_intake["target_consistency"] is True
    assert processing_intake["promotion_families_requiring_surface"] == [
        "eq",
        "compressor",
        "gate",
        "deessing",
        "frequency_separation",
        "cross_channel_unmasking",
    ]
    assert (
        "operator_must_capture_channel_processing_physical_readback_artifact"
        in processing_intake["blocked_by"]
    )
    bus_contract = proof_quality["bus_group_main_dca_physical_readback_contract"]
    assert bus_contract["available"] is True
    assert bus_contract["status"] == "passed"
    assert bus_contract["schema_id"] == "bus_group_main_dca_physical_readback.v1"
    assert bus_contract["contract_status"] == "contract_ready_physical_fixture_missing"
    assert bus_contract["surface_id"] == "bus_group_main_dca_state"
    assert bus_contract["remediation_action_id"] == "remediate_bus_group_main_dca_state"
    assert bus_contract["artifact_required"] is True
    assert bus_contract["physical_fixture_gap_remains"] is True
    assert bus_contract["accepted_as_physical_fixture_candidate"] is False
    assert bus_contract["live_authority_granted"] is False
    assert bus_contract["full_autonomous_ready"] is False
    assert bus_contract["fixture_physical_gap_declared"] is True
    assert bus_contract["remediation_physical_gap_declared"] is True
    assert bus_contract["target_consistency"] is True
    assert bus_contract["promotion_families_requiring_surface"] == [
        "bus_group_matrix",
        "dca",
        "master_bus",
    ]
    assert bus_contract["required_bus_control_fields"] == [
        "fader_db",
        "mute",
        "pan",
        "width",
        "mono",
    ]
    assert "true_peak_dbtp" in bus_contract["required_metering_fields"]
    assert "member_buses" in bus_contract["required_dca_assignment_fields"]
    assert bus_contract["minimum_bus_count"] == 1
    assert bus_contract["minimum_main_count"] == 1
    assert bus_contract["minimum_dca_count"] == 1
    assert "physical_readback" in bus_contract["allowed_sources"]
    assert "dca_write_events" in bus_contract["forbidden_mutation_evidence_fields"]
    bus_intake = proof_quality["bus_group_main_dca_physical_readback_artifact_intake"]
    assert bus_intake["available"] is True
    assert bus_intake["status"] == "passed"
    assert bus_intake["schema_id"] == "bus_group_main_dca_physical_readback.v1"
    assert bus_intake["intake_status"] == "operator_artifact_missing"
    assert bus_intake["surface_id"] == "bus_group_main_dca_state"
    assert bus_intake["remediation_action_id"] == "remediate_bus_group_main_dca_state"
    assert (
        bus_intake["required_contract_method_id"]
        == "bus_group_main_dca_physical_readback_contract"
    )
    assert bus_intake["artifact_count"] == 0
    assert bus_intake["valid_candidate_count"] == 0
    assert bus_intake["invalid_candidate_count"] == 0
    assert bus_intake["valid_bus_count"] == 0
    assert bus_intake["valid_main_count"] == 0
    assert bus_intake["valid_dca_count"] == 0
    assert bus_intake["accepted_as_physical_fixture_candidate"] is False
    assert bus_intake["operator_review_required"] is True
    assert bus_intake["physical_fixture_gap_remains"] is True
    assert bus_intake["fixture_closure_allowed_by_intake"] is False
    assert bus_intake["live_authority_granted"] is False
    assert bus_intake["full_autonomous_ready"] is False
    assert bus_intake["target_consistency"] is True
    assert bus_intake["promotion_families_requiring_surface"] == [
        "bus_group_matrix",
        "dca",
        "master_bus",
        "mix_balance",
        "performance_adaptation",
    ]
    assert (
        "operator_must_capture_bus_group_main_dca_physical_readback_artifact"
        in bus_intake["blocked_by"]
    )
    matrix_contract = proof_quality["matrix_state_physical_readback_contract"]
    assert matrix_contract["available"] is True
    assert matrix_contract["status"] == "passed"
    assert matrix_contract["schema_id"] == "matrix_state_physical_readback.v1"
    assert matrix_contract["contract_status"] == "contract_ready_physical_fixture_missing"
    assert matrix_contract["surface_id"] == "matrix_state"
    assert matrix_contract["remediation_action_id"] == "remediate_matrix_state"
    assert matrix_contract["artifact_required"] is True
    assert matrix_contract["physical_fixture_gap_remains"] is True
    assert matrix_contract["accepted_as_physical_fixture_candidate"] is False
    assert matrix_contract["live_authority_granted"] is False
    assert matrix_contract["full_autonomous_ready"] is False
    assert matrix_contract["fixture_physical_gap_declared"] is True
    assert matrix_contract["remediation_physical_gap_declared"] is True
    assert matrix_contract["target_consistency"] is True
    assert matrix_contract["promotion_families_requiring_surface"] == [
        "bus_group_matrix",
        "fx_send_insert",
        "routing",
    ]
    assert matrix_contract["required_control_fields"] == ["fader_db", "mute", "pan"]
    assert matrix_contract["required_eq_band_fields"] == ["band", "frequency_hz", "gain_db", "q"]
    assert "gain_reduction_db" in matrix_contract["required_dynamics_fields"]
    assert "post_insert_weight" in matrix_contract["required_insert_fields"]
    assert "true_peak_dbtp" in matrix_contract["required_metering_fields"]
    assert matrix_contract["minimum_matrix_count"] == 1
    assert matrix_contract["minimum_eq_band_count"] == 8
    assert "physical_readback" in matrix_contract["allowed_sources"]
    assert "matrix_write_events" in matrix_contract["forbidden_mutation_evidence_fields"]
    matrix_intake = proof_quality["matrix_state_physical_readback_artifact_intake"]
    assert matrix_intake["available"] is True
    assert matrix_intake["status"] == "passed"
    assert matrix_intake["schema_id"] == "matrix_state_physical_readback.v1"
    assert matrix_intake["intake_status"] == "operator_artifact_missing"
    assert matrix_intake["surface_id"] == "matrix_state"
    assert matrix_intake["remediation_action_id"] == "remediate_matrix_state"
    assert matrix_intake["required_contract_method_id"] == "matrix_state_physical_readback_contract"
    assert matrix_intake["artifact_count"] == 0
    assert matrix_intake["valid_candidate_count"] == 0
    assert matrix_intake["invalid_candidate_count"] == 0
    assert matrix_intake["valid_matrix_count"] == 0
    assert matrix_intake["accepted_as_physical_fixture_candidate"] is False
    assert matrix_intake["operator_review_required"] is True
    assert matrix_intake["physical_fixture_gap_remains"] is True
    assert matrix_intake["fixture_closure_allowed_by_intake"] is False
    assert matrix_intake["live_authority_granted"] is False
    assert matrix_intake["full_autonomous_ready"] is False
    assert matrix_intake["target_consistency"] is True
    assert matrix_intake["promotion_families_requiring_surface"] == [
        "bus_group_matrix",
        "fx_send_insert",
        "master_bus",
        "mix_balance",
        "routing",
        "performance_adaptation",
    ]
    assert "operator_must_capture_matrix_state_physical_readback_artifact" in matrix_intake[
        "blocked_by"
    ]
    fx_contract = proof_quality["fx_slots_inserts_physical_readback_contract"]
    assert fx_contract["available"] is True
    assert fx_contract["status"] == "passed"
    assert fx_contract["schema_id"] == "fx_slots_inserts_physical_readback.v1"
    assert fx_contract["contract_status"] == "contract_ready_physical_fixture_missing"
    assert fx_contract["surface_id"] == "fx_slots_and_inserts"
    assert fx_contract["remediation_action_id"] == "remediate_fx_slots_and_inserts"
    assert fx_contract["artifact_required"] is True
    assert fx_contract["physical_fixture_gap_remains"] is True
    assert fx_contract["accepted_as_physical_fixture_candidate"] is False
    assert fx_contract["live_authority_granted"] is False
    assert fx_contract["full_autonomous_ready"] is False
    assert fx_contract["fixture_physical_gap_declared"] is True
    assert fx_contract["remediation_physical_gap_declared"] is True
    assert fx_contract["target_consistency"] is True
    assert fx_contract["promotion_families_requiring_surface"] == [
        "fx_send_insert",
        "routing",
        "mix_balance",
    ]
    assert "parameter_schema" in fx_contract["required_fx_slot_fields"]
    assert "safe_range" in fx_contract["required_parameter_fields"]
    assert "physical_model_schema_verified" in fx_contract[
        "required_parameter_schema_fields"
    ]
    assert "scoped_diff_marker" in fx_contract["required_insert_coverage_flags"]
    assert fx_contract["required_false_safety_flags"] == [
        "live_write",
        "mutates_console_state",
        "live_widening_allowed",
        "full_autonomous_ready",
    ]
    assert fx_contract["minimum_fx_slot_count"] == 1
    assert fx_contract["minimum_fx_parameter_count"] == 4
    assert fx_contract["minimum_insert_count"] == 1
    assert "physical_readback" in fx_contract["allowed_sources"]
    assert "fx_parameter_write_events" in fx_contract[
        "forbidden_mutation_evidence_fields"
    ]
    fx_intake = proof_quality["fx_slots_inserts_physical_readback_artifact_intake"]
    assert fx_intake["available"] is True
    assert fx_intake["status"] == "passed"
    assert fx_intake["schema_id"] == "fx_slots_inserts_physical_readback.v1"
    assert fx_intake["intake_status"] == "operator_artifact_missing"
    assert fx_intake["surface_id"] == "fx_slots_and_inserts"
    assert fx_intake["remediation_action_id"] == "remediate_fx_slots_and_inserts"
    assert fx_intake["required_contract_method_id"] == "fx_slots_inserts_physical_readback_contract"
    assert fx_intake["artifact_count"] == 0
    assert fx_intake["valid_candidate_count"] == 0
    assert fx_intake["invalid_candidate_count"] == 0
    assert fx_intake["valid_fx_slot_count"] == 0
    assert fx_intake["valid_insert_count"] == 0
    assert fx_intake["accepted_as_physical_fixture_candidate"] is False
    assert fx_intake["operator_review_required"] is True
    assert fx_intake["physical_fixture_gap_remains"] is True
    assert fx_intake["fixture_closure_allowed_by_intake"] is False
    assert fx_intake["live_authority_granted"] is False
    assert fx_intake["full_autonomous_ready"] is False
    assert fx_intake["target_consistency"] is True
    assert fx_intake["promotion_families_requiring_surface"] == [
        "fx_send_insert",
        "routing",
        "mix_balance",
        "performance_adaptation",
    ]
    assert "operator_must_capture_fx_slots_inserts_physical_readback_artifact" in fx_intake[
        "blocked_by"
    ]
    routing_contract = proof_quality["routing_sends_physical_readback_contract"]
    assert routing_contract["available"] is True
    assert routing_contract["status"] == "passed"
    assert routing_contract["schema_id"] == "routing_sends_physical_readback.v1"
    assert routing_contract["contract_status"] == "contract_ready_physical_fixture_missing"
    assert routing_contract["surface_id"] == "routing_and_sends"
    assert routing_contract["remediation_action_id"] == "remediate_routing_and_sends"
    assert routing_contract["artifact_required"] is True
    assert routing_contract["physical_fixture_gap_remains"] is True
    assert routing_contract["accepted_as_physical_fixture_candidate"] is False
    assert routing_contract["live_authority_granted"] is False
    assert routing_contract["full_autonomous_ready"] is False
    assert routing_contract["fixture_physical_gap_declared"] is True
    assert routing_contract["remediation_physical_gap_declared"] is True
    assert routing_contract["target_consistency"] is True
    assert routing_contract["promotion_families_requiring_surface"] == [
        "routing",
        "fx_send_insert",
        "bus_group_matrix",
        "scene_preset",
        "mix_balance",
        "performance_adaptation",
    ]
    assert "wing_routing_hint" in routing_contract["required_static_route_fields"]
    assert "pre_on" in routing_contract["required_bus_send_fields"]
    assert "main" in routing_contract["required_main_send_fields"]
    assert "target_matrix" in routing_contract["required_matrix_send_fields"]
    assert "source_channel" in routing_contract["required_output_route_fields"]
    assert "cache_matches_physical_readback" in routing_contract[
        "required_send_coverage_flags"
    ]
    assert "cache_matches_physical_readback" in routing_contract[
        "required_output_route_coverage_flags"
    ]
    assert routing_contract["required_false_safety_flags"] == [
        "live_write",
        "mutates_console_state",
        "live_widening_allowed",
        "full_autonomous_ready",
    ]
    assert routing_contract["minimum_static_dante_route_count"] == 1
    assert routing_contract["minimum_channel_bus_send_count"] == 1
    assert routing_contract["minimum_channel_main_send_count"] == 1
    assert routing_contract["minimum_channel_matrix_send_count"] == 1
    assert routing_contract["minimum_output_route_count"] == 1
    assert "physical_readback" in routing_contract["allowed_sources"]
    assert "routing_write_events" in routing_contract[
        "forbidden_mutation_evidence_fields"
    ]
    routing_intake = proof_quality["routing_sends_physical_readback_artifact_intake"]
    assert routing_intake["available"] is True
    assert routing_intake["status"] == "passed"
    assert routing_intake["schema_id"] == "routing_sends_physical_readback.v1"
    assert routing_intake["intake_status"] == "operator_artifact_missing"
    assert routing_intake["surface_id"] == "routing_and_sends"
    assert routing_intake["remediation_action_id"] == "remediate_routing_and_sends"
    assert routing_intake["required_contract_method_id"] == "routing_sends_physical_readback_contract"
    assert routing_intake["artifact_count"] == 0
    assert routing_intake["valid_candidate_count"] == 0
    assert routing_intake["invalid_candidate_count"] == 0
    assert routing_intake["valid_static_dante_route_count"] == 0
    assert routing_intake["valid_channel_bus_send_count"] == 0
    assert routing_intake["valid_channel_main_send_count"] == 0
    assert routing_intake["valid_channel_matrix_send_count"] == 0
    assert routing_intake["valid_output_route_count"] == 0
    assert routing_intake["accepted_as_physical_fixture_candidate"] is False
    assert routing_intake["operator_review_required"] is True
    assert routing_intake["physical_fixture_gap_remains"] is True
    assert routing_intake["fixture_closure_allowed_by_intake"] is False
    assert routing_intake["live_authority_granted"] is False
    assert routing_intake["full_autonomous_ready"] is False
    assert routing_intake["target_consistency"] is True
    assert routing_intake["promotion_families_requiring_surface"] == [
        "routing",
        "fx_send_insert",
        "bus_group_matrix",
        "scene_preset",
        "mix_balance",
        "performance_adaptation",
    ]
    assert "operator_must_capture_routing_sends_physical_readback_artifact" in routing_intake[
        "blocked_by"
    ]
    scenes_contract = proof_quality["snapshots_scenes_physical_readback_contract"]
    assert scenes_contract["available"] is True
    assert scenes_contract["status"] == "passed"
    assert scenes_contract["schema_id"] == "snapshots_scenes_physical_readback.v1"
    assert scenes_contract["contract_status"] == "contract_ready_physical_fixture_missing"
    assert scenes_contract["surface_id"] == "snapshots_scenes"
    assert scenes_contract["remediation_action_id"] == "remediate_snapshots_scenes"
    assert scenes_contract["artifact_required"] is True
    assert scenes_contract["physical_fixture_gap_remains"] is True
    assert scenes_contract["accepted_as_physical_fixture_candidate"] is False
    assert scenes_contract["live_authority_granted"] is False
    assert scenes_contract["full_autonomous_ready"] is False
    assert scenes_contract["scene_load_allowed"] is False
    assert scenes_contract["scene_save_allowed"] is False
    assert scenes_contract["fixture_physical_gap_declared"] is True
    assert scenes_contract["remediation_physical_gap_declared"] is True
    assert scenes_contract["target_consistency"] is True
    assert "scene_preset" in scenes_contract["promotion_families_requiring_surface"]
    assert "active_snapshot_index" in scenes_contract["required_active_scene_fields"]
    assert "scene_content_fingerprint" in scenes_contract["required_snapshot_fields"]
    assert "rollback_reference" in scenes_contract["required_coverage_flags"]
    assert "scene_load_allowed" in scenes_contract["required_false_safety_flags"]
    assert scenes_contract["minimum_snapshot_count"] == 1
    assert "physical_readback" in scenes_contract["allowed_sources"]
    assert "scene_recall_events" in scenes_contract[
        "forbidden_mutation_evidence_fields"
    ]
    scenes_intake = proof_quality["snapshots_scenes_physical_readback_artifact_intake"]
    assert scenes_intake["available"] is True
    assert scenes_intake["status"] == "passed"
    assert scenes_intake["schema_id"] == "snapshots_scenes_physical_readback.v1"
    assert scenes_intake["intake_status"] == "operator_artifact_missing"
    assert scenes_intake["surface_id"] == "snapshots_scenes"
    assert scenes_intake["remediation_action_id"] == "remediate_snapshots_scenes"
    assert scenes_intake["required_contract_method_id"] == (
        "snapshots_scenes_physical_readback_contract"
    )
    assert scenes_intake["artifact_count"] == 0
    assert scenes_intake["valid_candidate_count"] == 0
    assert scenes_intake["invalid_candidate_count"] == 0
    assert scenes_intake["valid_snapshot_count"] == 0
    assert scenes_intake["accepted_as_physical_fixture_candidate"] is False
    assert scenes_intake["operator_review_required"] is True
    assert scenes_intake["physical_fixture_gap_remains"] is True
    assert scenes_intake["fixture_closure_allowed_by_intake"] is False
    assert scenes_intake["scene_load_allowed"] is False
    assert scenes_intake["scene_save_allowed"] is False
    assert scenes_intake["live_authority_granted"] is False
    assert scenes_intake["full_autonomous_ready"] is False
    assert scenes_intake["target_consistency"] is True
    assert scenes_intake["promotion_families_requiring_surface"] == [
        "scene_preset",
        "routing",
        "mix_balance",
        "performance_adaptation",
    ]
    assert "operator_must_capture_snapshots_scenes_physical_readback_artifact" in scenes_intake[
        "blocked_by"
    ]
    presets_contract = proof_quality["user_presets_physical_readback_contract"]
    assert presets_contract["available"] is True
    assert presets_contract["status"] == "passed"
    assert presets_contract["schema_id"] == "user_presets_physical_readback.v1"
    assert presets_contract["contract_status"] == "contract_ready_physical_fixture_missing"
    assert presets_contract["surface_id"] == "user_presets"
    assert presets_contract["remediation_action_id"] == "remediate_user_presets"
    assert presets_contract["artifact_required"] is True
    assert presets_contract["physical_fixture_gap_remains"] is True
    assert presets_contract["accepted_as_physical_fixture_candidate"] is False
    assert presets_contract["live_authority_granted"] is False
    assert presets_contract["full_autonomous_ready"] is False
    assert presets_contract["preset_recall_allowed"] is False
    assert presets_contract["preset_store_allowed"] is False
    assert presets_contract["fixture_physical_gap_declared"] is True
    assert presets_contract["remediation_physical_gap_declared"] is True
    assert presets_contract["target_consistency"] is True
    assert "scene_preset" in presets_contract["promotion_families_requiring_surface"]
    assert "preset_content_fingerprint" in presets_contract["required_preset_fields"]
    assert "rollback_reference" in presets_contract["required_coverage_flags"]
    assert "preset_recall_allowed" in presets_contract["required_false_safety_flags"]
    assert "preset_store_allowed" in presets_contract["required_false_safety_flags"]
    assert presets_contract["minimum_preset_count"] == 1
    assert "physical_readback" in presets_contract["allowed_sources"]
    assert "preset_store_events" in presets_contract[
        "forbidden_mutation_evidence_fields"
    ]
    presets_intake = proof_quality["user_presets_physical_readback_artifact_intake"]
    assert presets_intake["available"] is True
    assert presets_intake["status"] == "passed"
    assert presets_intake["schema_id"] == "user_presets_physical_readback.v1"
    assert presets_intake["intake_status"] == "operator_artifact_missing"
    assert presets_intake["surface_id"] == "user_presets"
    assert presets_intake["remediation_action_id"] == "remediate_user_presets"
    assert presets_intake["required_contract_method_id"] == (
        "user_presets_physical_readback_contract"
    )
    assert presets_intake["artifact_count"] == 0
    assert presets_intake["valid_candidate_count"] == 0
    assert presets_intake["invalid_candidate_count"] == 0
    assert presets_intake["valid_preset_count"] == 0
    assert presets_intake["accepted_as_physical_fixture_candidate"] is False
    assert presets_intake["operator_review_required"] is True
    assert presets_intake["physical_fixture_gap_remains"] is True
    assert presets_intake["fixture_closure_allowed_by_intake"] is False
    assert presets_intake["preset_recall_allowed"] is False
    assert presets_intake["preset_store_allowed"] is False
    assert presets_intake["live_authority_granted"] is False
    assert presets_intake["full_autonomous_ready"] is False
    assert presets_intake["target_consistency"] is True
    assert "scene_preset" in presets_intake["promotion_families_requiring_surface"]
    assert "operator_must_capture_user_presets_physical_readback_artifact" in presets_intake[
        "blocked_by"
    ]
    dlive_contract = proof_quality["dlive_state_physical_readback_contract"]
    assert dlive_contract["available"] is True
    assert dlive_contract["status"] == "passed"
    assert dlive_contract["schema_id"] == "dlive_state_physical_readback.v1"
    assert dlive_contract["contract_status"] == "contract_ready_physical_fixture_missing"
    assert dlive_contract["surface_id"] == "dlive_state"
    assert dlive_contract["remediation_action_id"] == "remediate_dlive_state"
    assert dlive_contract["artifact_required"] is True
    assert dlive_contract["physical_fixture_gap_remains"] is True
    assert dlive_contract["accepted_as_physical_fixture_candidate"] is False
    assert dlive_contract["live_authority_granted"] is False
    assert dlive_contract["full_autonomous_ready"] is False
    assert dlive_contract["midi_send_allowed"] is False
    assert dlive_contract["tcp_send_allowed"] is False
    assert dlive_contract["dlive_write_allowed"] is False
    assert dlive_contract["fixture_physical_gap_declared"] is True
    assert dlive_contract["remediation_physical_gap_declared"] is True
    assert dlive_contract["target_consistency"] is True
    assert "performance_adaptation" in dlive_contract[
        "promotion_families_requiring_surface"
    ]
    assert "fader_raw_14bit" in dlive_contract["required_channel_fields"]
    assert "midi_base_channel" in dlive_contract["required_midi_state_fields"]
    assert "wing_surface_equivalent" in dlive_contract["required_parity_fields"]
    assert "wing_parity_mapping" in dlive_contract["required_coverage_flags"]
    assert "midi_send_allowed" in dlive_contract["required_false_safety_flags"]
    assert "tcp_send_allowed" in dlive_contract["required_false_safety_flags"]
    assert "dlive_write_allowed" in dlive_contract["required_false_safety_flags"]
    assert dlive_contract["minimum_channel_count"] == 1
    assert "physical_readback" in dlive_contract["allowed_sources"]
    assert "midi_send_events" in dlive_contract[
        "forbidden_mutation_evidence_fields"
    ]
    dlive_intake = proof_quality["dlive_state_physical_readback_artifact_intake"]
    assert dlive_intake["available"] is True
    assert dlive_intake["status"] == "passed"
    assert dlive_intake["schema_id"] == "dlive_state_physical_readback.v1"
    assert dlive_intake["intake_status"] == "operator_artifact_missing"
    assert dlive_intake["surface_id"] == "dlive_state"
    assert dlive_intake["remediation_action_id"] == "remediate_dlive_state"
    assert dlive_intake["required_contract_method_id"] == (
        "dlive_state_physical_readback_contract"
    )
    assert dlive_intake["artifact_count"] == 0
    assert dlive_intake["valid_candidate_count"] == 0
    assert dlive_intake["invalid_candidate_count"] == 0
    assert dlive_intake["valid_channel_count"] == 0
    assert dlive_intake["accepted_as_physical_fixture_candidate"] is False
    assert dlive_intake["operator_review_required"] is True
    assert dlive_intake["physical_fixture_gap_remains"] is True
    assert dlive_intake["fixture_closure_allowed_by_intake"] is False
    assert dlive_intake["midi_send_allowed"] is False
    assert dlive_intake["tcp_send_allowed"] is False
    assert dlive_intake["dlive_write_allowed"] is False
    assert dlive_intake["dlive_apply_allowed"] is False
    assert dlive_intake["live_authority_granted"] is False
    assert dlive_intake["full_autonomous_ready"] is False
    assert dlive_intake["target_consistency"] is True
    assert "performance_adaptation" in dlive_intake[
        "promotion_families_requiring_surface"
    ]
    assert "operator_must_capture_dlive_state_physical_readback_artifact" in dlive_intake[
        "blocked_by"
    ]
    overlay = payload["autonomy_evidence_overlay"]
    assert overlay["source_report_available"] is True
    assert overlay["source_report_path"] == str(latest_report)
    assert overlay["safety"] == {
        "read_only": True,
        "live_write": False,
        "live_widening_allowed": False,
        "automatic_safety_update_allowed": False,
        "full_autonomous_ready": False,
        "operator_required": True,
    }
    assert overlay["summary"]["autonomous_ready_target_ids"] == []
    assert "overlay_does_not_modify_safety_rows" in overlay["summary"]["global_blockers"]
    rows = {row["value_type"]: row for row in overlay["rows"]}
    assert set(rows) == {"fader", "gain"}
    assert rows["gain"]["autonomous_ready"] is False
    assert "source_autonomous_ready_normalized_false" in rows["gain"]["blockers"]
    assert rows["gain"]["dry_run_ready_proof_ids"] == ["gain-proof-dry-run"]
    assert rows["fader"]["status"] == "no_current_evidence"
    assert "set_gain" not in set(_walk_keys(payload))


def test_supervised_proof_evidence_normalizes_raw_mixer_contract_claims(
    server, tmp_path
):
    latest_dir = tmp_path / "latest"
    latest_dir.mkdir()
    latest_report = latest_dir / "supervised_proof_artifact_report.json"
    latest_report.write_text(json.dumps(proof_report_with_autonomy_overlay("ok")), encoding="utf-8")
    (latest_dir / "summary.json").write_text(
        json.dumps(
            {
                "type": "no_wing_release_check",
                "status": "passed",
                "read_surface_fixture_evidence": {
                    "status": "passed",
                    "snapshot_status": "fixture_evidence_incomplete",
                    "physical_gap_surface_count": 2,
                    "summary_recomputed_from_rows": True,
                    "fixture_declaration_validation": True,
                    "contract_gap_consistency_enforced": True,
                    "read_only": True,
                    "live_write": False,
                    "full_autonomous_ready": False,
                    "physical_fixture_missing_surface_ids": [
                        "raw_mixer_state_cache",
                        "channel_identity_and_basic_controls",
                    ],
                    "next_physical_readback_surface_id": "raw_mixer_state_cache",
                    "violations": [],
                },
                "read_surface_physical_readback_capture_plan": {
                    "status": "passed",
                    "plan_status": "operator_capture_required",
                    "read_only": False,
                    "live_write": True,
                    "mutates_console_state": True,
                    "queries_console": True,
                    "executes_tests": True,
                    "operator_required": False,
                    "execution_allowed_by_plan": True,
                    "live_widening_allowed": True,
                    "live_authority_granted": True,
                    "full_autonomous_ready": True,
                    "capture_step_count": 1,
                    "physical_gap_surface_count": 2,
                    "physical_fixture_missing_surface_ids": [
                        "raw_mixer_state_cache",
                        "channel_identity_and_basic_controls",
                    ],
                    "next_physical_readback_surface_id": "dlive_state",
                    "next_physical_readback_action_id": "remediate_dlive_state",
                    "target_action_consistency": False,
                    "required_artifact_contracts": ["physical_readback_fixture"],
                    "blocked_by": [],
                    "step_surface_id": "dlive_state",
                    "step_remediation_action_id": "remediate_dlive_state",
                    "step_forbidden_actions": [],
                    "violations": ["capture_plan_claimed_execution_authority"],
                },
                "raw_mixer_state_physical_readback_contract": {
                    "status": "passed",
                    "schema_id": "raw_mixer_state_physical_readback.v1",
                    "contract_status": "contract_ready_physical_fixture_missing",
                    "surface_id": "raw_mixer_state_cache",
                    "remediation_action_id": "remediate_raw_mixer_state_cache",
                    "next_physical_readback_surface_id": "dlive_state",
                    "next_physical_readback_action_id": "remediate_dlive_state",
                    "artifact_required": True,
                    "physical_fixture_gap_remains": True,
                    "accepted_as_physical_fixture_candidate": True,
                    "read_only": False,
                    "live_write": True,
                    "live_authority_granted": True,
                    "full_autonomous_ready": True,
                    "target_consistency": False,
                    "allowed_sources": ["physical_readback", "mixer_state_cache"],
                    "accepted_row_statuses": ["verified_from_cache"],
                    "forbidden_mutation_evidence_fields": [],
                    "violations": ["source_claimed_live_authority"],
                },
                "raw_mixer_state_physical_readback_artifact_intake": {
                    "status": "passed",
                    "schema_id": "raw_mixer_state_physical_readback.v1",
                    "intake_status": "valid_operator_artifact_available",
                    "surface_id": "dlive_state",
                    "remediation_action_id": "remediate_dlive_state",
                    "required_contract_method_id": "raw_mixer_state_physical_readback_contract",
                    "artifact_search_roots": [
                        "artifacts/read_surface_physical_readback/raw_mixer_state",
                    ],
                    "artifact_count": 1,
                    "valid_candidate_count": 1,
                    "invalid_candidate_count": 0,
                    "accepted_as_physical_fixture_candidate": True,
                    "operator_review_required": False,
                    "physical_fixture_gap_remains": False,
                    "fixture_closure_allowed_by_intake": True,
                    "read_only": False,
                    "live_write": True,
                    "mutates_console_state": True,
                    "queries_console": True,
                    "executes_tests": True,
                    "operator_required": False,
                    "execution_allowed_by_intake": True,
                    "live_widening_allowed": True,
                    "live_authority_granted": True,
                    "full_autonomous_ready": True,
                    "blocked_by": [],
                    "artifact_source_ids": ["unsafe.json"],
                    "intake_violations": [],
                    "violations": ["intake_claimed_live_authority"],
                },
                "channel_basic_controls_physical_readback_contract": {
                    "status": "passed",
                    "schema_id": "channel_basic_controls_physical_readback.v1",
                    "contract_status": "contract_ready_physical_fixture_missing",
                    "surface_id": "dlive_state",
                    "remediation_action_id": "remediate_dlive_state",
                    "physical_readback_surface_id": "dlive_state",
                    "physical_readback_action_id": "remediate_dlive_state",
                    "fixture_physical_gap_declared": False,
                    "remediation_physical_gap_declared": False,
                    "artifact_required": True,
                    "physical_fixture_gap_remains": True,
                    "accepted_as_physical_fixture_candidate": True,
                    "read_only": False,
                    "live_write": True,
                    "live_authority_granted": True,
                    "full_autonomous_ready": True,
                    "target_consistency": False,
                    "promotion_families_requiring_surface": ["fader"],
                    "required_control_fields": ["fader_db"],
                    "required_coverage_flags": [],
                    "allowed_sources": ["physical_readback", "mixer_state_cache"],
                    "accepted_channel_statuses": ["verified_from_cache"],
                    "forbidden_mutation_evidence_fields": [],
                    "violations": ["channel_contract_claimed_live_authority"],
                },
                "channel_basic_controls_physical_readback_artifact_intake": {
                    "status": "passed",
                    "schema_id": "channel_basic_controls_physical_readback.v1",
                    "intake_status": "valid_operator_artifact_available",
                    "surface_id": "dlive_state",
                    "remediation_action_id": "remediate_dlive_state",
                    "required_contract_method_id": "channel_basic_controls_physical_readback_contract",
                    "artifact_search_roots": [
                        "artifacts/read_surface_physical_readback/channel_basic_controls",
                    ],
                    "artifact_count": 1,
                    "valid_candidate_count": 1,
                    "invalid_candidate_count": 0,
                    "valid_channel_count": 24,
                    "accepted_as_physical_fixture_candidate": True,
                    "operator_review_required": False,
                    "physical_fixture_gap_remains": False,
                    "fixture_closure_allowed_by_intake": True,
                    "read_only": False,
                    "live_write": True,
                    "mutates_console_state": True,
                    "queries_console": True,
                    "executes_tests": True,
                    "operator_required": False,
                    "execution_allowed_by_intake": True,
                    "live_widening_allowed": True,
                    "live_authority_granted": True,
                    "full_autonomous_ready": True,
                    "target_consistency": False,
                    "promotion_families_requiring_surface": ["fader"],
                    "blocked_by": [],
                    "artifact_source_ids": ["unsafe-channel-basic.json"],
                    "intake_violations": [],
                    "violations": ["channel_intake_claimed_live_authority"],
                },
                "channel_processing_physical_readback_contract": {
                    "status": "passed",
                    "schema_id": "channel_processing_physical_readback.v1",
                    "contract_status": "contract_ready_physical_fixture_missing",
                    "surface_id": "dlive_state",
                    "remediation_action_id": "remediate_dlive_state",
                    "physical_readback_surface_id": "dlive_state",
                    "physical_readback_action_id": "remediate_dlive_state",
                    "fixture_physical_gap_declared": False,
                    "remediation_physical_gap_declared": False,
                    "artifact_required": True,
                    "physical_fixture_gap_remains": True,
                    "accepted_as_physical_fixture_candidate": True,
                    "read_only": False,
                    "live_write": True,
                    "live_authority_granted": True,
                    "full_autonomous_ready": True,
                    "target_consistency": False,
                    "promotion_families_requiring_surface": ["eq"],
                    "required_hpf_fields": ["enabled"],
                    "required_eq_fields": ["enabled"],
                    "required_eq_band_fields": ["band"],
                    "required_compressor_fields": ["enabled"],
                    "required_gate_fields": ["enabled"],
                    "required_coverage_flags": [],
                    "minimum_eq_band_count": 2,
                    "allowed_sources": ["physical_readback", "mixer_state_cache"],
                    "accepted_channel_statuses": ["verified_from_cache"],
                    "forbidden_mutation_evidence_fields": [],
                    "violations": ["processing_contract_claimed_live_authority"],
                },
                "channel_processing_physical_readback_artifact_intake": {
                    "status": "passed",
                    "schema_id": "channel_processing_physical_readback.v1",
                    "intake_status": "valid_operator_artifact_available",
                    "surface_id": "dlive_state",
                    "remediation_action_id": "remediate_dlive_state",
                    "required_contract_method_id": "channel_processing_physical_readback_contract",
                    "artifact_search_roots": [
                        "artifacts/read_surface_physical_readback/channel_processing",
                    ],
                    "artifact_count": 1,
                    "valid_candidate_count": 1,
                    "invalid_candidate_count": 0,
                    "valid_channel_count": 24,
                    "accepted_as_physical_fixture_candidate": True,
                    "operator_review_required": False,
                    "physical_fixture_gap_remains": False,
                    "fixture_closure_allowed_by_intake": True,
                    "read_only": False,
                    "live_write": True,
                    "mutates_console_state": True,
                    "queries_console": True,
                    "executes_tests": True,
                    "operator_required": False,
                    "execution_allowed_by_intake": True,
                    "live_widening_allowed": True,
                    "live_authority_granted": True,
                    "full_autonomous_ready": True,
                    "target_consistency": False,
                    "promotion_families_requiring_surface": ["eq"],
                    "blocked_by": [],
                    "artifact_source_ids": ["unsafe-processing.json"],
                    "intake_violations": [],
                    "violations": ["processing_intake_claimed_live_authority"],
                },
                "bus_group_main_dca_physical_readback_contract": {
                    "status": "passed",
                    "schema_id": "bus_group_main_dca_physical_readback.v1",
                    "contract_status": "contract_ready_physical_fixture_missing",
                    "surface_id": "dlive_state",
                    "remediation_action_id": "remediate_dlive_state",
                    "physical_readback_surface_id": "dlive_state",
                    "physical_readback_action_id": "remediate_dlive_state",
                    "fixture_physical_gap_declared": False,
                    "remediation_physical_gap_declared": False,
                    "artifact_required": True,
                    "physical_fixture_gap_remains": True,
                    "accepted_as_physical_fixture_candidate": True,
                    "read_only": False,
                    "live_write": True,
                    "live_authority_granted": True,
                    "full_autonomous_ready": True,
                    "target_consistency": False,
                    "promotion_families_requiring_surface": ["dca"],
                    "required_bus_control_fields": ["fader_db"],
                    "required_main_control_fields": ["fader_db"],
                    "required_dca_control_fields": ["fader_db"],
                    "required_bus_assignment_fields": ["dca_assignments"],
                    "required_dca_assignment_fields": ["member_channels"],
                    "required_metering_fields": ["lufs"],
                    "required_bus_coverage_flags": [],
                    "required_main_coverage_flags": [],
                    "required_dca_coverage_flags": [],
                    "minimum_bus_count": 0,
                    "minimum_main_count": 0,
                    "minimum_dca_count": 0,
                    "allowed_sources": ["physical_readback", "mixer_state_cache"],
                    "accepted_surface_statuses": ["verified_from_cache"],
                    "forbidden_mutation_evidence_fields": [],
                    "violations": ["bus_contract_claimed_live_authority"],
                },
                "bus_group_main_dca_physical_readback_artifact_intake": {
                    "status": "passed",
                    "schema_id": "bus_group_main_dca_physical_readback.v1",
                    "intake_status": "valid_operator_artifact_available",
                    "surface_id": "dlive_state",
                    "remediation_action_id": "remediate_dlive_state",
                    "required_contract_method_id": "bus_group_main_dca_physical_readback_contract",
                    "artifact_search_roots": [
                        "artifacts/read_surface_physical_readback/bus_group_main_dca",
                    ],
                    "artifact_count": 1,
                    "valid_candidate_count": 1,
                    "invalid_candidate_count": 0,
                    "valid_bus_count": 8,
                    "valid_main_count": 1,
                    "valid_dca_count": 8,
                    "accepted_as_physical_fixture_candidate": True,
                    "operator_review_required": False,
                    "physical_fixture_gap_remains": False,
                    "fixture_closure_allowed_by_intake": True,
                    "read_only": False,
                    "live_write": True,
                    "mutates_console_state": True,
                    "queries_console": True,
                    "executes_tests": True,
                    "operator_required": False,
                    "execution_allowed_by_intake": True,
                    "live_widening_allowed": True,
                    "live_authority_granted": True,
                    "full_autonomous_ready": True,
                    "target_consistency": False,
                    "promotion_families_requiring_surface": ["dca"],
                    "blocked_by": [],
                    "artifact_source_ids": ["unsafe-bus-main-dca.json"],
                    "intake_violations": [],
                    "violations": ["bus_intake_claimed_live_authority"],
                },
                "matrix_state_physical_readback_contract": {
                    "status": "passed",
                    "schema_id": "matrix_state_physical_readback.v1",
                    "contract_status": "contract_ready_physical_fixture_missing",
                    "surface_id": "dlive_state",
                    "remediation_action_id": "remediate_dlive_state",
                    "physical_readback_surface_id": "dlive_state",
                    "physical_readback_action_id": "remediate_dlive_state",
                    "fixture_physical_gap_declared": False,
                    "remediation_physical_gap_declared": False,
                    "artifact_required": True,
                    "physical_fixture_gap_remains": True,
                    "accepted_as_physical_fixture_candidate": True,
                    "read_only": False,
                    "live_write": True,
                    "live_authority_granted": True,
                    "full_autonomous_ready": True,
                    "target_consistency": False,
                    "promotion_families_requiring_surface": ["routing"],
                    "required_control_fields": ["fader_db"],
                    "required_eq_fields": ["enabled"],
                    "required_eq_band_fields": ["band"],
                    "required_dynamics_fields": ["enabled"],
                    "required_insert_fields": ["pre_insert_on"],
                    "required_metering_fields": ["lufs"],
                    "required_coverage_flags": [],
                    "minimum_matrix_count": 0,
                    "minimum_eq_band_count": 2,
                    "allowed_sources": ["physical_readback", "mixer_state_cache"],
                    "accepted_matrix_statuses": ["verified_from_cache"],
                    "forbidden_mutation_evidence_fields": [],
                    "violations": ["matrix_contract_claimed_live_authority"],
                },
                "matrix_state_physical_readback_artifact_intake": {
                    "status": "passed",
                    "schema_id": "matrix_state_physical_readback.v1",
                    "intake_status": "valid_operator_artifact_available",
                    "surface_id": "dlive_state",
                    "remediation_action_id": "remediate_dlive_state",
                    "required_contract_method_id": "matrix_state_physical_readback_contract",
                    "artifact_search_roots": [
                        "artifacts/read_surface_physical_readback/matrix_state",
                    ],
                    "artifact_count": 1,
                    "valid_candidate_count": 1,
                    "invalid_candidate_count": 0,
                    "valid_matrix_count": 8,
                    "accepted_as_physical_fixture_candidate": True,
                    "operator_review_required": False,
                    "physical_fixture_gap_remains": False,
                    "fixture_closure_allowed_by_intake": True,
                    "read_only": False,
                    "live_write": True,
                    "mutates_console_state": True,
                    "queries_console": True,
                    "executes_tests": True,
                    "operator_required": False,
                    "execution_allowed_by_intake": True,
                    "live_widening_allowed": True,
                    "live_authority_granted": True,
                    "full_autonomous_ready": True,
                    "target_consistency": False,
                    "promotion_families_requiring_surface": ["routing"],
                    "blocked_by": [],
                    "artifact_source_ids": ["unsafe-matrix.json"],
                    "intake_violations": [],
                    "violations": ["matrix_intake_claimed_live_authority"],
                },
                "fx_slots_inserts_physical_readback_contract": {
                    "status": "passed",
                    "schema_id": "fx_slots_inserts_physical_readback.v1",
                    "contract_status": "contract_ready_physical_fixture_missing",
                    "surface_id": "dlive_state",
                    "remediation_action_id": "remediate_dlive_state",
                    "physical_readback_surface_id": "dlive_state",
                    "physical_readback_action_id": "remediate_dlive_state",
                    "fixture_physical_gap_declared": False,
                    "remediation_physical_gap_declared": False,
                    "artifact_required": True,
                    "physical_fixture_gap_remains": True,
                    "accepted_as_physical_fixture_candidate": True,
                    "read_only": False,
                    "live_write": True,
                    "live_authority_granted": True,
                    "full_autonomous_ready": True,
                    "target_consistency": False,
                    "promotion_families_requiring_surface": ["routing"],
                    "required_fx_slot_fields": ["slot"],
                    "required_parameter_fields": ["value"],
                    "required_parameter_schema_fields": ["status"],
                    "required_fx_slot_coverage_flags": [],
                    "required_insert_fields": ["target_type"],
                    "required_insert_coverage_flags": [],
                    "required_false_safety_flags": ["live_write"],
                    "minimum_fx_slot_count": 0,
                    "minimum_fx_parameter_count": 0,
                    "minimum_insert_count": 0,
                    "allowed_sources": ["physical_readback", "mixer_state_cache"],
                    "accepted_row_statuses": ["verified_from_cache"],
                    "allowed_insert_target_types": ["channel"],
                    "allowed_insert_positions": ["post"],
                    "forbidden_mutation_evidence_fields": [],
                    "violations": ["fx_contract_claimed_live_authority"],
                },
                "fx_slots_inserts_physical_readback_artifact_intake": {
                    "status": "passed",
                    "schema_id": "fx_slots_inserts_physical_readback.v1",
                    "intake_status": "valid_operator_artifact_available",
                    "surface_id": "dlive_state",
                    "remediation_action_id": "remediate_dlive_state",
                    "required_contract_method_id": "fx_slots_inserts_physical_readback_contract",
                    "artifact_search_roots": [
                        "artifacts/read_surface_physical_readback/fx_slots_and_inserts",
                    ],
                    "artifact_count": 1,
                    "valid_candidate_count": 1,
                    "invalid_candidate_count": 0,
                    "valid_fx_slot_count": 4,
                    "valid_insert_count": 8,
                    "accepted_as_physical_fixture_candidate": True,
                    "operator_review_required": False,
                    "physical_fixture_gap_remains": False,
                    "fixture_closure_allowed_by_intake": True,
                    "read_only": False,
                    "live_write": True,
                    "mutates_console_state": True,
                    "queries_console": True,
                    "executes_tests": True,
                    "operator_required": False,
                    "execution_allowed_by_intake": True,
                    "live_widening_allowed": True,
                    "live_authority_granted": True,
                    "full_autonomous_ready": True,
                    "target_consistency": False,
                    "promotion_families_requiring_surface": ["fx_send_insert"],
                    "blocked_by": [],
                    "artifact_source_ids": ["unsafe-fx.json"],
                    "intake_violations": [],
                    "violations": ["fx_intake_claimed_live_authority"],
                },
                "routing_sends_physical_readback_contract": {
                    "status": "passed",
                    "schema_id": "routing_sends_physical_readback.v1",
                    "contract_status": "contract_ready_physical_fixture_missing",
                    "surface_id": "dlive_state",
                    "remediation_action_id": "remediate_dlive_state",
                    "physical_readback_surface_id": "dlive_state",
                    "physical_readback_action_id": "remediate_dlive_state",
                    "fixture_physical_gap_declared": False,
                    "remediation_physical_gap_declared": False,
                    "artifact_required": True,
                    "physical_fixture_gap_remains": True,
                    "accepted_as_physical_fixture_candidate": True,
                    "read_only": False,
                    "live_write": True,
                    "live_authority_granted": True,
                    "full_autonomous_ready": True,
                    "target_consistency": False,
                    "promotion_families_requiring_surface": ["routing"],
                    "required_static_route_fields": ["start"],
                    "required_bus_send_fields": ["channel"],
                    "required_main_send_fields": ["channel"],
                    "required_matrix_send_fields": ["channel"],
                    "required_output_route_fields": ["output_group"],
                    "required_static_route_coverage_flags": [],
                    "required_send_coverage_flags": [],
                    "required_output_route_coverage_flags": [],
                    "required_false_safety_flags": ["live_write"],
                    "minimum_static_dante_route_count": 0,
                    "minimum_channel_bus_send_count": 0,
                    "minimum_channel_main_send_count": 0,
                    "minimum_channel_matrix_send_count": 0,
                    "minimum_output_route_count": 0,
                    "allowed_sources": ["physical_readback", "mixer_state_cache"],
                    "accepted_row_statuses": ["verified_from_cache"],
                    "forbidden_mutation_evidence_fields": [],
                    "violations": ["routing_contract_claimed_live_authority"],
                },
                "routing_sends_physical_readback_artifact_intake": {
                    "status": "passed",
                    "schema_id": "routing_sends_physical_readback.v1",
                    "intake_status": "valid_operator_artifact_available",
                    "surface_id": "dlive_state",
                    "remediation_action_id": "remediate_dlive_state",
                    "required_contract_method_id": "routing_sends_physical_readback_contract",
                    "artifact_search_roots": [
                        "artifacts/read_surface_physical_readback/routing_and_sends",
                    ],
                    "artifact_count": 1,
                    "valid_candidate_count": 1,
                    "invalid_candidate_count": 0,
                    "valid_static_dante_route_count": 2,
                    "valid_channel_bus_send_count": 3,
                    "valid_channel_main_send_count": 4,
                    "valid_channel_matrix_send_count": 5,
                    "valid_output_route_count": 6,
                    "accepted_as_physical_fixture_candidate": True,
                    "operator_review_required": False,
                    "physical_fixture_gap_remains": False,
                    "fixture_closure_allowed_by_intake": True,
                    "read_only": False,
                    "live_write": True,
                    "mutates_console_state": True,
                    "queries_console": True,
                    "executes_tests": True,
                    "operator_required": False,
                    "execution_allowed_by_intake": True,
                    "live_widening_allowed": True,
                    "live_authority_granted": True,
                    "full_autonomous_ready": True,
                    "target_consistency": False,
                    "promotion_families_requiring_surface": ["routing"],
                    "blocked_by": [],
                    "artifact_source_ids": ["unsafe-routing.json"],
                    "intake_violations": [],
                    "violations": ["routing_intake_claimed_live_authority"],
                },
                "snapshots_scenes_physical_readback_artifact_intake": {
                    "status": "passed",
                    "schema_id": "snapshots_scenes_physical_readback.v1",
                    "intake_status": "valid_operator_artifact_available",
                    "surface_id": "dlive_state",
                    "remediation_action_id": "remediate_dlive_state",
                    "required_contract_method_id": "snapshots_scenes_physical_readback_contract",
                    "artifact_search_roots": [
                        "artifacts/read_surface_physical_readback/snapshots_scenes",
                    ],
                    "artifact_count": 1,
                    "valid_candidate_count": 1,
                    "invalid_candidate_count": 0,
                    "valid_snapshot_count": 2,
                    "accepted_as_physical_fixture_candidate": True,
                    "operator_review_required": False,
                    "physical_fixture_gap_remains": False,
                    "fixture_closure_allowed_by_intake": True,
                    "scene_load_allowed": True,
                    "scene_save_allowed": True,
                    "read_only": False,
                    "live_write": True,
                    "mutates_console_state": True,
                    "queries_console": True,
                    "executes_tests": True,
                    "operator_required": False,
                    "execution_allowed_by_intake": True,
                    "live_widening_allowed": True,
                    "live_authority_granted": True,
                    "full_autonomous_ready": True,
                    "target_consistency": False,
                    "promotion_families_requiring_surface": ["scene_preset"],
                    "blocked_by": [],
                    "artifact_source_ids": ["unsafe-scenes.json"],
                    "intake_violations": [],
                    "violations": ["scenes_intake_claimed_live_authority"],
                },
                "user_presets_physical_readback_contract": {
                    "status": "passed",
                    "schema_id": "user_presets_physical_readback.v1",
                    "contract_status": "contract_ready_physical_fixture_missing",
                    "surface_id": "dlive_state",
                    "remediation_action_id": "remediate_dlive_state",
                    "physical_readback_surface_id": "dlive_state",
                    "physical_readback_action_id": "remediate_dlive_state",
                    "fixture_physical_gap_declared": False,
                    "remediation_physical_gap_declared": False,
                    "artifact_required": True,
                    "physical_fixture_gap_remains": True,
                    "accepted_as_physical_fixture_candidate": True,
                    "read_only": False,
                    "live_write": True,
                    "live_authority_granted": True,
                    "full_autonomous_ready": True,
                    "preset_recall_allowed": True,
                    "preset_store_allowed": True,
                    "target_consistency": False,
                    "promotion_families_requiring_surface": ["scene_preset"],
                    "required_preset_fields": ["preset_index"],
                    "required_coverage_flags": [],
                    "required_false_safety_flags": ["live_write"],
                    "minimum_preset_count": 0,
                    "allowed_sources": ["physical_readback", "mixer_state_cache"],
                    "accepted_row_statuses": ["verified_from_cache"],
                    "forbidden_mutation_evidence_fields": [],
                    "violations": ["preset_contract_claimed_live_authority"],
                },
                "dlive_state_physical_readback_contract": {
                    "status": "passed",
                    "schema_id": "dlive_state_physical_readback.v1",
                    "contract_status": "contract_ready_physical_fixture_missing",
                    "surface_id": "dlive_state",
                    "remediation_action_id": "remediate_dlive_state",
                    "physical_readback_surface_id": "dlive_state",
                    "physical_readback_action_id": "remediate_dlive_state",
                    "fixture_physical_gap_declared": False,
                    "remediation_physical_gap_declared": False,
                    "artifact_required": True,
                    "physical_fixture_gap_remains": True,
                    "accepted_as_physical_fixture_candidate": True,
                    "read_only": False,
                    "live_write": True,
                    "live_authority_granted": True,
                    "full_autonomous_ready": True,
                    "midi_send_allowed": True,
                    "tcp_send_allowed": True,
                    "dlive_write_allowed": True,
                    "dlive_apply_allowed": True,
                    "target_consistency": False,
                    "promotion_families_requiring_surface": ["performance_adaptation"],
                    "required_channel_fields": ["channel"],
                    "required_midi_state_fields": ["midi_base_channel"],
                    "required_parity_fields": ["wing_surface_equivalent"],
                    "required_coverage_flags": [],
                    "required_false_safety_flags": ["live_write"],
                    "minimum_channel_count": 0,
                    "allowed_sources": ["physical_readback", "mixer_state_cache"],
                    "accepted_row_statuses": ["verified_from_cache"],
                    "forbidden_mutation_evidence_fields": [],
                    "violations": ["dlive_contract_claimed_live_authority"],
                },
                "read_surface_physical_readback_fixture_closure_plan": {
                    "status": "passed",
                    "plan_status": "manual_fixture_closure_candidate_available",
                    "ledger_status": "operator_acceptance_records_available",
                    "review_candidate_count": 2,
                    "acceptance_record_count": 2,
                    "accepted_record_count": 1,
                    "rejected_record_count": 1,
                    "invalid_record_count": 0,
                    "unreviewed_candidate_count": 0,
                    "closure_candidate_count": 1,
                    "closure_candidate_surface_ids": ["raw_mixer_state_cache"],
                    "rejected_surface_ids": ["dlive_state"],
                    "invalid_record_ids": [],
                    "unreviewed_candidate_surface_ids": [],
                    "manual_fixture_closure_candidate_available": True,
                    "manual_fixture_closure_required": True,
                    "operator_review_required": True,
                    "physical_fixture_gap_remains": True,
                    "fixture_update_allowed_by_plan": True,
                    "fixture_closure_allowed_by_plan": True,
                    "live_write": True,
                    "mutates_console_state": True,
                    "queries_console": True,
                    "executes_tests": True,
                    "execution_allowed_by_plan": True,
                    "live_authority_granted": True,
                    "full_autonomous_ready": True,
                    "blocked_by": [
                        "manual_fixture_baseline_update_required_before_gap_closure",
                    ],
                    "violations": ["claimed_closure_authority"],
                },
            }
        ),
        encoding="utf-8",
    )
    server.config["supervised_proof_evidence"] = {"artifact_root": str(tmp_path)}

    payload = build_supervised_proof_evidence_snapshot(server)

    capture_plan = payload["summary"]["read_surface_proof_quality"][
        "read_surface_physical_readback_capture_plan"
    ]
    assert capture_plan["available"] is True
    assert capture_plan["next_physical_readback_surface_id"] == "raw_mixer_state_cache"
    assert capture_plan["source_next_physical_readback_surface_id"] == "dlive_state"
    assert capture_plan["next_physical_readback_action_id"] == "remediate_raw_mixer_state_cache"
    assert capture_plan["source_next_physical_readback_action_id"] == "remediate_dlive_state"
    assert capture_plan["target_consistency"] is False
    assert capture_plan["read_only"] is True
    assert capture_plan["live_write"] is False
    assert capture_plan["mutates_console_state"] is False
    assert capture_plan["queries_console"] is False
    assert capture_plan["executes_tests"] is False
    assert capture_plan["operator_required"] is True
    assert capture_plan["execution_allowed_by_plan"] is False
    assert capture_plan["live_authority_granted"] is False
    assert capture_plan["full_autonomous_ready"] is False
    assert capture_plan["source_execution_allowed_by_plan_claimed"] is True
    assert capture_plan["source_live_write_claimed"] is True
    assert capture_plan["source_live_authority_granted_claimed"] is True
    assert capture_plan["source_full_autonomous_ready_claimed"] is True
    assert "capture_plan_claimed_execution_authority" in capture_plan["violations"]
    raw_contract = payload["summary"]["read_surface_proof_quality"][
        "raw_mixer_state_physical_readback_contract"
    ]
    assert raw_contract["available"] is True
    assert raw_contract["next_physical_readback_surface_id"] == "raw_mixer_state_cache"
    assert raw_contract["source_next_physical_readback_surface_id"] == "dlive_state"
    assert raw_contract["next_physical_readback_action_id"] == "remediate_raw_mixer_state_cache"
    assert raw_contract["source_next_physical_readback_action_id"] == "remediate_dlive_state"
    assert raw_contract["target_consistency"] is False
    assert raw_contract["accepted_as_physical_fixture_candidate"] is False
    assert raw_contract["read_only"] is True
    assert raw_contract["live_write"] is False
    assert raw_contract["live_authority_granted"] is False
    assert raw_contract["full_autonomous_ready"] is False
    assert raw_contract["source_accepted_as_physical_fixture_candidate_claimed"] is True
    assert raw_contract["source_live_write_claimed"] is True
    assert raw_contract["source_live_authority_granted_claimed"] is True
    assert raw_contract["source_full_autonomous_ready_claimed"] is True
    assert "source_claimed_live_authority" in raw_contract["violations"]
    raw_intake = payload["summary"]["read_surface_proof_quality"][
        "raw_mixer_state_physical_readback_artifact_intake"
    ]
    assert raw_intake["available"] is True
    assert raw_intake["surface_id"] == "raw_mixer_state_cache"
    assert raw_intake["source_surface_id"] == "dlive_state"
    assert raw_intake["remediation_action_id"] == "remediate_raw_mixer_state_cache"
    assert raw_intake["source_remediation_action_id"] == "remediate_dlive_state"
    assert raw_intake["target_consistency"] is False
    assert raw_intake["artifact_count"] == 1
    assert raw_intake["valid_candidate_count"] == 1
    assert raw_intake["invalid_candidate_count"] == 0
    assert raw_intake["accepted_as_physical_fixture_candidate"] is False
    assert raw_intake["operator_review_required"] is True
    assert raw_intake["physical_fixture_gap_remains"] is True
    assert raw_intake["fixture_closure_allowed_by_intake"] is False
    assert raw_intake["live_authority_granted"] is False
    assert raw_intake["full_autonomous_ready"] is False
    assert raw_intake["source_accepted_as_physical_fixture_candidate_claimed"] is True
    assert raw_intake["source_operator_review_required_claimed"] is False
    assert raw_intake["source_physical_fixture_gap_remains_claimed"] is False
    assert raw_intake["source_fixture_closure_allowed_by_intake_claimed"] is True
    assert raw_intake["source_execution_allowed_by_intake_claimed"] is True
    assert raw_intake["source_live_write_claimed"] is True
    assert raw_intake["source_live_authority_granted_claimed"] is True
    assert raw_intake["source_full_autonomous_ready_claimed"] is True
    assert "intake_claimed_live_authority" in raw_intake["violations"]
    channel_contract = payload["summary"]["read_surface_proof_quality"][
        "channel_basic_controls_physical_readback_contract"
    ]
    assert channel_contract["available"] is True
    assert channel_contract["surface_id"] == "dlive_state"
    assert channel_contract["physical_readback_surface_id"] == "dlive_state"
    assert channel_contract["target_consistency"] is False
    assert channel_contract["accepted_as_physical_fixture_candidate"] is False
    assert channel_contract["read_only"] is True
    assert channel_contract["live_write"] is False
    assert channel_contract["live_authority_granted"] is False
    assert channel_contract["full_autonomous_ready"] is False
    assert channel_contract["source_accepted_as_physical_fixture_candidate_claimed"] is True
    assert channel_contract["source_live_write_claimed"] is True
    assert channel_contract["source_live_authority_granted_claimed"] is True
    assert channel_contract["source_full_autonomous_ready_claimed"] is True
    assert channel_contract["promotion_families_requiring_surface"] == ["fader"]
    assert "channel_contract_claimed_live_authority" in channel_contract["violations"]
    channel_intake = payload["summary"]["read_surface_proof_quality"][
        "channel_basic_controls_physical_readback_artifact_intake"
    ]
    assert channel_intake["available"] is True
    assert channel_intake["surface_id"] == "channel_identity_and_basic_controls"
    assert channel_intake["source_surface_id"] == "dlive_state"
    assert channel_intake["remediation_action_id"] == "remediate_channel_identity_and_basic_controls"
    assert channel_intake["source_remediation_action_id"] == "remediate_dlive_state"
    assert channel_intake["target_consistency"] is False
    assert channel_intake["artifact_count"] == 1
    assert channel_intake["valid_candidate_count"] == 1
    assert channel_intake["invalid_candidate_count"] == 0
    assert channel_intake["valid_channel_count"] == 24
    assert channel_intake["accepted_as_physical_fixture_candidate"] is False
    assert channel_intake["operator_review_required"] is True
    assert channel_intake["physical_fixture_gap_remains"] is True
    assert channel_intake["fixture_closure_allowed_by_intake"] is False
    assert channel_intake["live_authority_granted"] is False
    assert channel_intake["full_autonomous_ready"] is False
    assert channel_intake["source_accepted_as_physical_fixture_candidate_claimed"] is True
    assert channel_intake["source_operator_review_required_claimed"] is False
    assert channel_intake["source_physical_fixture_gap_remains_claimed"] is False
    assert channel_intake["source_fixture_closure_allowed_by_intake_claimed"] is True
    assert channel_intake["source_execution_allowed_by_intake_claimed"] is True
    assert channel_intake["source_live_write_claimed"] is True
    assert channel_intake["source_live_authority_granted_claimed"] is True
    assert channel_intake["source_full_autonomous_ready_claimed"] is True
    assert channel_intake["promotion_families_requiring_surface"] == ["fader"]
    assert "channel_intake_claimed_live_authority" in channel_intake["violations"]
    processing_contract = payload["summary"]["read_surface_proof_quality"][
        "channel_processing_physical_readback_contract"
    ]
    assert processing_contract["available"] is True
    assert processing_contract["surface_id"] == "dlive_state"
    assert processing_contract["physical_readback_surface_id"] == "dlive_state"
    assert processing_contract["target_consistency"] is False
    assert processing_contract["accepted_as_physical_fixture_candidate"] is False
    assert processing_contract["read_only"] is True
    assert processing_contract["live_write"] is False
    assert processing_contract["live_authority_granted"] is False
    assert processing_contract["full_autonomous_ready"] is False
    assert processing_contract["source_accepted_as_physical_fixture_candidate_claimed"] is True
    assert processing_contract["source_live_write_claimed"] is True
    assert processing_contract["source_live_authority_granted_claimed"] is True
    assert processing_contract["source_full_autonomous_ready_claimed"] is True
    assert processing_contract["promotion_families_requiring_surface"] == ["eq"]
    assert processing_contract["minimum_eq_band_count"] == 2
    assert "processing_contract_claimed_live_authority" in processing_contract["violations"]
    processing_intake = payload["summary"]["read_surface_proof_quality"][
        "channel_processing_physical_readback_artifact_intake"
    ]
    assert processing_intake["available"] is True
    assert processing_intake["surface_id"] == "channel_eq_dynamics_gate"
    assert processing_intake["source_surface_id"] == "dlive_state"
    assert processing_intake["remediation_action_id"] == "remediate_channel_eq_dynamics_gate"
    assert processing_intake["source_remediation_action_id"] == "remediate_dlive_state"
    assert processing_intake["target_consistency"] is False
    assert processing_intake["artifact_count"] == 1
    assert processing_intake["valid_candidate_count"] == 1
    assert processing_intake["invalid_candidate_count"] == 0
    assert processing_intake["valid_channel_count"] == 24
    assert processing_intake["accepted_as_physical_fixture_candidate"] is False
    assert processing_intake["operator_review_required"] is True
    assert processing_intake["physical_fixture_gap_remains"] is True
    assert processing_intake["fixture_closure_allowed_by_intake"] is False
    assert processing_intake["live_authority_granted"] is False
    assert processing_intake["full_autonomous_ready"] is False
    assert processing_intake["source_accepted_as_physical_fixture_candidate_claimed"] is True
    assert processing_intake["source_operator_review_required_claimed"] is False
    assert processing_intake["source_physical_fixture_gap_remains_claimed"] is False
    assert processing_intake["source_fixture_closure_allowed_by_intake_claimed"] is True
    assert processing_intake["source_execution_allowed_by_intake_claimed"] is True
    assert processing_intake["source_live_write_claimed"] is True
    assert processing_intake["source_live_authority_granted_claimed"] is True
    assert processing_intake["source_full_autonomous_ready_claimed"] is True
    assert processing_intake["promotion_families_requiring_surface"] == ["eq"]
    assert "processing_intake_claimed_live_authority" in processing_intake["violations"]
    bus_contract = payload["summary"]["read_surface_proof_quality"][
        "bus_group_main_dca_physical_readback_contract"
    ]
    assert bus_contract["available"] is True
    assert bus_contract["surface_id"] == "dlive_state"
    assert bus_contract["physical_readback_surface_id"] == "dlive_state"
    assert bus_contract["target_consistency"] is False
    assert bus_contract["accepted_as_physical_fixture_candidate"] is False
    assert bus_contract["read_only"] is True
    assert bus_contract["live_write"] is False
    assert bus_contract["live_authority_granted"] is False
    assert bus_contract["full_autonomous_ready"] is False
    assert bus_contract["source_accepted_as_physical_fixture_candidate_claimed"] is True
    assert bus_contract["source_live_write_claimed"] is True
    assert bus_contract["source_live_authority_granted_claimed"] is True
    assert bus_contract["source_full_autonomous_ready_claimed"] is True
    assert bus_contract["promotion_families_requiring_surface"] == ["dca"]
    assert bus_contract["minimum_bus_count"] == 0
    assert bus_contract["minimum_main_count"] == 0
    assert bus_contract["minimum_dca_count"] == 0
    assert "bus_contract_claimed_live_authority" in bus_contract["violations"]
    bus_intake = payload["summary"]["read_surface_proof_quality"][
        "bus_group_main_dca_physical_readback_artifact_intake"
    ]
    assert bus_intake["available"] is True
    assert bus_intake["surface_id"] == "bus_group_main_dca_state"
    assert bus_intake["source_surface_id"] == "dlive_state"
    assert bus_intake["remediation_action_id"] == "remediate_bus_group_main_dca_state"
    assert bus_intake["source_remediation_action_id"] == "remediate_dlive_state"
    assert bus_intake["target_consistency"] is False
    assert bus_intake["artifact_count"] == 1
    assert bus_intake["valid_candidate_count"] == 1
    assert bus_intake["invalid_candidate_count"] == 0
    assert bus_intake["valid_bus_count"] == 8
    assert bus_intake["valid_main_count"] == 1
    assert bus_intake["valid_dca_count"] == 8
    assert bus_intake["accepted_as_physical_fixture_candidate"] is False
    assert bus_intake["operator_review_required"] is True
    assert bus_intake["physical_fixture_gap_remains"] is True
    assert bus_intake["fixture_closure_allowed_by_intake"] is False
    assert bus_intake["live_authority_granted"] is False
    assert bus_intake["full_autonomous_ready"] is False
    assert bus_intake["source_accepted_as_physical_fixture_candidate_claimed"] is True
    assert bus_intake["source_operator_review_required_claimed"] is False
    assert bus_intake["source_physical_fixture_gap_remains_claimed"] is False
    assert bus_intake["source_fixture_closure_allowed_by_intake_claimed"] is True
    assert bus_intake["source_execution_allowed_by_intake_claimed"] is True
    assert bus_intake["source_live_write_claimed"] is True
    assert bus_intake["source_live_authority_granted_claimed"] is True
    assert bus_intake["source_full_autonomous_ready_claimed"] is True
    assert bus_intake["promotion_families_requiring_surface"] == ["dca"]
    assert "bus_intake_claimed_live_authority" in bus_intake["violations"]
    matrix_contract = payload["summary"]["read_surface_proof_quality"][
        "matrix_state_physical_readback_contract"
    ]
    assert matrix_contract["available"] is True
    assert matrix_contract["surface_id"] == "dlive_state"
    assert matrix_contract["physical_readback_surface_id"] == "dlive_state"
    assert matrix_contract["target_consistency"] is False
    assert matrix_contract["accepted_as_physical_fixture_candidate"] is False
    assert matrix_contract["read_only"] is True
    assert matrix_contract["live_write"] is False
    assert matrix_contract["live_authority_granted"] is False
    assert matrix_contract["full_autonomous_ready"] is False
    assert matrix_contract["source_accepted_as_physical_fixture_candidate_claimed"] is True
    assert matrix_contract["source_live_write_claimed"] is True
    assert matrix_contract["source_live_authority_granted_claimed"] is True
    assert matrix_contract["source_full_autonomous_ready_claimed"] is True
    assert matrix_contract["promotion_families_requiring_surface"] == ["routing"]
    assert matrix_contract["minimum_matrix_count"] == 0
    assert matrix_contract["minimum_eq_band_count"] == 2
    assert "matrix_contract_claimed_live_authority" in matrix_contract["violations"]
    matrix_intake = payload["summary"]["read_surface_proof_quality"][
        "matrix_state_physical_readback_artifact_intake"
    ]
    assert matrix_intake["available"] is True
    assert matrix_intake["surface_id"] == "matrix_state"
    assert matrix_intake["source_surface_id"] == "dlive_state"
    assert matrix_intake["remediation_action_id"] == "remediate_matrix_state"
    assert matrix_intake["source_remediation_action_id"] == "remediate_dlive_state"
    assert matrix_intake["target_consistency"] is False
    assert matrix_intake["artifact_count"] == 1
    assert matrix_intake["valid_candidate_count"] == 1
    assert matrix_intake["invalid_candidate_count"] == 0
    assert matrix_intake["valid_matrix_count"] == 8
    assert matrix_intake["accepted_as_physical_fixture_candidate"] is False
    assert matrix_intake["operator_review_required"] is True
    assert matrix_intake["physical_fixture_gap_remains"] is True
    assert matrix_intake["fixture_closure_allowed_by_intake"] is False
    assert matrix_intake["read_only"] is True
    assert matrix_intake["live_write"] is False
    assert matrix_intake["live_authority_granted"] is False
    assert matrix_intake["full_autonomous_ready"] is False
    assert matrix_intake["source_accepted_as_physical_fixture_candidate_claimed"] is True
    assert matrix_intake["source_operator_review_required_claimed"] is False
    assert matrix_intake["source_physical_fixture_gap_remains_claimed"] is False
    assert matrix_intake["source_fixture_closure_allowed_by_intake_claimed"] is True
    assert matrix_intake["source_execution_allowed_by_intake_claimed"] is True
    assert matrix_intake["source_live_write_claimed"] is True
    assert matrix_intake["source_live_authority_granted_claimed"] is True
    assert matrix_intake["source_full_autonomous_ready_claimed"] is True
    assert matrix_intake["promotion_families_requiring_surface"] == ["routing"]
    assert "matrix_intake_claimed_live_authority" in matrix_intake["violations"]
    fx_contract = payload["summary"]["read_surface_proof_quality"][
        "fx_slots_inserts_physical_readback_contract"
    ]
    assert fx_contract["available"] is True
    assert fx_contract["surface_id"] == "dlive_state"
    assert fx_contract["physical_readback_surface_id"] == "dlive_state"
    assert fx_contract["target_consistency"] is False
    assert fx_contract["accepted_as_physical_fixture_candidate"] is False
    assert fx_contract["read_only"] is True
    assert fx_contract["live_write"] is False
    assert fx_contract["live_authority_granted"] is False
    assert fx_contract["full_autonomous_ready"] is False
    assert fx_contract["source_accepted_as_physical_fixture_candidate_claimed"] is True
    assert fx_contract["source_live_write_claimed"] is True
    assert fx_contract["source_live_authority_granted_claimed"] is True
    assert fx_contract["source_full_autonomous_ready_claimed"] is True
    assert fx_contract["promotion_families_requiring_surface"] == ["routing"]
    assert fx_contract["required_false_safety_flags"] == ["live_write"]
    assert fx_contract["minimum_fx_slot_count"] == 0
    assert fx_contract["minimum_fx_parameter_count"] == 0
    assert fx_contract["minimum_insert_count"] == 0
    assert "fx_contract_claimed_live_authority" in fx_contract["violations"]
    fx_intake = payload["summary"]["read_surface_proof_quality"][
        "fx_slots_inserts_physical_readback_artifact_intake"
    ]
    assert fx_intake["available"] is True
    assert fx_intake["surface_id"] == "fx_slots_and_inserts"
    assert fx_intake["source_surface_id"] == "dlive_state"
    assert fx_intake["remediation_action_id"] == "remediate_fx_slots_and_inserts"
    assert fx_intake["source_remediation_action_id"] == "remediate_dlive_state"
    assert fx_intake["target_consistency"] is False
    assert fx_intake["artifact_count"] == 1
    assert fx_intake["valid_candidate_count"] == 1
    assert fx_intake["invalid_candidate_count"] == 0
    assert fx_intake["valid_fx_slot_count"] == 4
    assert fx_intake["valid_insert_count"] == 8
    assert fx_intake["accepted_as_physical_fixture_candidate"] is False
    assert fx_intake["operator_review_required"] is True
    assert fx_intake["physical_fixture_gap_remains"] is True
    assert fx_intake["fixture_closure_allowed_by_intake"] is False
    assert fx_intake["read_only"] is True
    assert fx_intake["live_write"] is False
    assert fx_intake["live_authority_granted"] is False
    assert fx_intake["full_autonomous_ready"] is False
    assert fx_intake["source_accepted_as_physical_fixture_candidate_claimed"] is True
    assert fx_intake["source_operator_review_required_claimed"] is False
    assert fx_intake["source_physical_fixture_gap_remains_claimed"] is False
    assert fx_intake["source_fixture_closure_allowed_by_intake_claimed"] is True
    assert fx_intake["source_execution_allowed_by_intake_claimed"] is True
    assert fx_intake["source_live_write_claimed"] is True
    assert fx_intake["source_queries_console_claimed"] is True
    assert fx_intake["source_live_authority_granted_claimed"] is True
    assert fx_intake["source_full_autonomous_ready_claimed"] is True
    assert fx_intake["promotion_families_requiring_surface"] == ["fx_send_insert"]
    assert fx_intake["artifact_source_ids"] == ["unsafe-fx.json"]
    assert "fx_intake_claimed_live_authority" in fx_intake["violations"]
    routing_contract = payload["summary"]["read_surface_proof_quality"][
        "routing_sends_physical_readback_contract"
    ]
    assert routing_contract["available"] is True
    assert routing_contract["surface_id"] == "dlive_state"
    assert routing_contract["physical_readback_surface_id"] == "dlive_state"
    assert routing_contract["target_consistency"] is False
    assert routing_contract["accepted_as_physical_fixture_candidate"] is False
    assert routing_contract["read_only"] is True
    assert routing_contract["live_write"] is False
    assert routing_contract["live_authority_granted"] is False
    assert routing_contract["full_autonomous_ready"] is False
    assert routing_contract["source_accepted_as_physical_fixture_candidate_claimed"] is True
    assert routing_contract["source_live_write_claimed"] is True
    assert routing_contract["source_live_authority_granted_claimed"] is True
    assert routing_contract["source_full_autonomous_ready_claimed"] is True
    assert routing_contract["promotion_families_requiring_surface"] == ["routing"]
    assert routing_contract["required_false_safety_flags"] == ["live_write"]
    assert routing_contract["minimum_static_dante_route_count"] == 0
    assert routing_contract["minimum_channel_bus_send_count"] == 0
    assert routing_contract["minimum_channel_main_send_count"] == 0
    assert routing_contract["minimum_channel_matrix_send_count"] == 0
    assert routing_contract["minimum_output_route_count"] == 0
    assert "routing_contract_claimed_live_authority" in routing_contract["violations"]
    routing_intake = payload["summary"]["read_surface_proof_quality"][
        "routing_sends_physical_readback_artifact_intake"
    ]
    assert routing_intake["available"] is True
    assert routing_intake["surface_id"] == "routing_and_sends"
    assert routing_intake["source_surface_id"] == "dlive_state"
    assert routing_intake["remediation_action_id"] == "remediate_routing_and_sends"
    assert routing_intake["source_remediation_action_id"] == "remediate_dlive_state"
    assert routing_intake["target_consistency"] is False
    assert routing_intake["artifact_count"] == 1
    assert routing_intake["valid_candidate_count"] == 1
    assert routing_intake["invalid_candidate_count"] == 0
    assert routing_intake["valid_static_dante_route_count"] == 2
    assert routing_intake["valid_channel_bus_send_count"] == 3
    assert routing_intake["valid_channel_main_send_count"] == 4
    assert routing_intake["valid_channel_matrix_send_count"] == 5
    assert routing_intake["valid_output_route_count"] == 6
    assert routing_intake["accepted_as_physical_fixture_candidate"] is False
    assert routing_intake["operator_review_required"] is True
    assert routing_intake["physical_fixture_gap_remains"] is True
    assert routing_intake["fixture_closure_allowed_by_intake"] is False
    assert routing_intake["read_only"] is True
    assert routing_intake["live_write"] is False
    assert routing_intake["live_authority_granted"] is False
    assert routing_intake["full_autonomous_ready"] is False
    assert routing_intake["source_accepted_as_physical_fixture_candidate_claimed"] is True
    assert routing_intake["source_operator_review_required_claimed"] is False
    assert routing_intake["source_physical_fixture_gap_remains_claimed"] is False
    assert routing_intake["source_fixture_closure_allowed_by_intake_claimed"] is True
    assert routing_intake["source_execution_allowed_by_intake_claimed"] is True
    assert routing_intake["source_live_write_claimed"] is True
    assert routing_intake["source_queries_console_claimed"] is True
    assert routing_intake["source_live_authority_granted_claimed"] is True
    assert routing_intake["source_full_autonomous_ready_claimed"] is True
    assert routing_intake["promotion_families_requiring_surface"] == ["routing"]
    assert routing_intake["artifact_source_ids"] == ["unsafe-routing.json"]
    assert "routing_intake_claimed_live_authority" in routing_intake["violations"]
    scenes_intake = payload["summary"]["read_surface_proof_quality"][
        "snapshots_scenes_physical_readback_artifact_intake"
    ]
    assert scenes_intake["available"] is True
    assert scenes_intake["surface_id"] == "snapshots_scenes"
    assert scenes_intake["source_surface_id"] == "dlive_state"
    assert scenes_intake["remediation_action_id"] == "remediate_snapshots_scenes"
    assert scenes_intake["source_remediation_action_id"] == "remediate_dlive_state"
    assert scenes_intake["target_consistency"] is False
    assert scenes_intake["artifact_count"] == 1
    assert scenes_intake["valid_candidate_count"] == 1
    assert scenes_intake["invalid_candidate_count"] == 0
    assert scenes_intake["valid_snapshot_count"] == 2
    assert scenes_intake["accepted_as_physical_fixture_candidate"] is False
    assert scenes_intake["operator_review_required"] is True
    assert scenes_intake["physical_fixture_gap_remains"] is True
    assert scenes_intake["fixture_closure_allowed_by_intake"] is False
    assert scenes_intake["scene_load_allowed"] is False
    assert scenes_intake["scene_save_allowed"] is False
    assert scenes_intake["read_only"] is True
    assert scenes_intake["live_write"] is False
    assert scenes_intake["live_authority_granted"] is False
    assert scenes_intake["full_autonomous_ready"] is False
    assert scenes_intake["source_accepted_as_physical_fixture_candidate_claimed"] is True
    assert scenes_intake["source_operator_review_required_claimed"] is False
    assert scenes_intake["source_physical_fixture_gap_remains_claimed"] is False
    assert scenes_intake["source_fixture_closure_allowed_by_intake_claimed"] is True
    assert scenes_intake["source_execution_allowed_by_intake_claimed"] is True
    assert scenes_intake["source_scene_load_allowed_claimed"] is True
    assert scenes_intake["source_scene_save_allowed_claimed"] is True
    assert scenes_intake["source_live_write_claimed"] is True
    assert scenes_intake["source_queries_console_claimed"] is True
    assert scenes_intake["source_live_authority_granted_claimed"] is True
    assert scenes_intake["source_full_autonomous_ready_claimed"] is True
    assert scenes_intake["promotion_families_requiring_surface"] == ["scene_preset"]
    assert scenes_intake["artifact_source_ids"] == ["unsafe-scenes.json"]
    assert "scenes_intake_claimed_live_authority" in scenes_intake["violations"]
    presets_contract = payload["summary"]["read_surface_proof_quality"][
        "user_presets_physical_readback_contract"
    ]
    assert presets_contract["available"] is True
    assert presets_contract["surface_id"] == "dlive_state"
    assert presets_contract["physical_readback_surface_id"] == "dlive_state"
    assert presets_contract["target_consistency"] is False
    assert presets_contract["accepted_as_physical_fixture_candidate"] is False
    assert presets_contract["read_only"] is True
    assert presets_contract["live_write"] is False
    assert presets_contract["live_authority_granted"] is False
    assert presets_contract["full_autonomous_ready"] is False
    assert presets_contract["preset_recall_allowed"] is False
    assert presets_contract["preset_store_allowed"] is False
    assert presets_contract["source_accepted_as_physical_fixture_candidate_claimed"] is True
    assert presets_contract["source_live_write_claimed"] is True
    assert presets_contract["source_live_authority_granted_claimed"] is True
    assert presets_contract["source_full_autonomous_ready_claimed"] is True
    assert presets_contract["source_preset_recall_allowed_claimed"] is True
    assert presets_contract["source_preset_store_allowed_claimed"] is True
    assert presets_contract["promotion_families_requiring_surface"] == ["scene_preset"]
    assert presets_contract["required_false_safety_flags"] == ["live_write"]
    assert presets_contract["minimum_preset_count"] == 0
    assert "preset_contract_claimed_live_authority" in presets_contract["violations"]
    dlive_contract = payload["summary"]["read_surface_proof_quality"][
        "dlive_state_physical_readback_contract"
    ]
    assert dlive_contract["available"] is True
    assert dlive_contract["surface_id"] == "dlive_state"
    assert dlive_contract["physical_readback_surface_id"] == "dlive_state"
    assert dlive_contract["target_consistency"] is False
    assert dlive_contract["accepted_as_physical_fixture_candidate"] is False
    assert dlive_contract["read_only"] is True
    assert dlive_contract["live_write"] is False
    assert dlive_contract["live_authority_granted"] is False
    assert dlive_contract["full_autonomous_ready"] is False
    assert dlive_contract["midi_send_allowed"] is False
    assert dlive_contract["tcp_send_allowed"] is False
    assert dlive_contract["dlive_write_allowed"] is False
    assert dlive_contract["dlive_apply_allowed"] is False
    assert dlive_contract["source_accepted_as_physical_fixture_candidate_claimed"] is True
    assert dlive_contract["source_live_write_claimed"] is True
    assert dlive_contract["source_live_authority_granted_claimed"] is True
    assert dlive_contract["source_full_autonomous_ready_claimed"] is True
    assert dlive_contract["source_midi_send_allowed_claimed"] is True
    assert dlive_contract["source_tcp_send_allowed_claimed"] is True
    assert dlive_contract["source_dlive_write_allowed_claimed"] is True
    assert dlive_contract["source_dlive_apply_allowed_claimed"] is True
    assert dlive_contract["promotion_families_requiring_surface"] == [
        "performance_adaptation"
    ]
    assert dlive_contract["required_false_safety_flags"] == ["live_write"]
    assert dlive_contract["minimum_channel_count"] == 0
    assert "dlive_contract_claimed_live_authority" in dlive_contract["violations"]
    fixture_closure_plan = payload["summary"]["read_surface_proof_quality"][
        "read_surface_physical_readback_fixture_closure_plan"
    ]
    assert fixture_closure_plan["available"] is True
    assert fixture_closure_plan["status"] == "passed"
    assert fixture_closure_plan["plan_status"] == "manual_fixture_closure_candidate_available"
    assert fixture_closure_plan["ledger_status"] == "operator_acceptance_records_available"
    assert fixture_closure_plan["review_candidate_count"] == 2
    assert fixture_closure_plan["acceptance_record_count"] == 2
    assert fixture_closure_plan["accepted_record_count"] == 1
    assert fixture_closure_plan["rejected_record_count"] == 1
    assert fixture_closure_plan["invalid_record_count"] == 0
    assert fixture_closure_plan["unreviewed_candidate_count"] == 0
    assert fixture_closure_plan["closure_candidate_count"] == 1
    assert fixture_closure_plan["closure_candidate_surface_ids"] == [
        "raw_mixer_state_cache"
    ]
    assert fixture_closure_plan["rejected_surface_ids"] == ["dlive_state"]
    assert fixture_closure_plan["manual_fixture_closure_candidate_available"] is True
    assert fixture_closure_plan["manual_fixture_closure_required"] is True
    assert fixture_closure_plan["operator_review_required"] is True
    assert fixture_closure_plan["physical_fixture_gap_remains"] is True
    assert fixture_closure_plan["fixture_update_allowed_by_plan"] is False
    assert fixture_closure_plan["fixture_closure_allowed_by_plan"] is False
    assert fixture_closure_plan["read_only"] is True
    assert fixture_closure_plan["live_write"] is False
    assert fixture_closure_plan["mutates_console_state"] is False
    assert fixture_closure_plan["queries_console"] is False
    assert fixture_closure_plan["executes_tests"] is False
    assert fixture_closure_plan["execution_allowed_by_plan"] is False
    assert fixture_closure_plan["live_authority_granted"] is False
    assert fixture_closure_plan["full_autonomous_ready"] is False
    assert fixture_closure_plan["source_fixture_update_allowed_by_plan_claimed"] is True
    assert fixture_closure_plan["source_fixture_closure_allowed_by_plan_claimed"] is True
    assert fixture_closure_plan["source_live_write_claimed"] is True
    assert fixture_closure_plan["source_mutates_console_state_claimed"] is True
    assert fixture_closure_plan["source_queries_console_claimed"] is True
    assert fixture_closure_plan["source_executes_tests_claimed"] is True
    assert fixture_closure_plan["source_execution_allowed_by_plan_claimed"] is True
    assert fixture_closure_plan["source_live_authority_granted_claimed"] is True
    assert fixture_closure_plan["source_full_autonomous_ready_claimed"] is True
    assert fixture_closure_plan["blocked_by"] == [
        "manual_fixture_baseline_update_required_before_gap_closure"
    ]
    assert "claimed_closure_authority" in fixture_closure_plan["violations"]
    assert "set_gain" not in set(_walk_keys(payload))


def test_supervised_proof_evidence_snapshot_marks_mismatched_next_readback_target(
    server, tmp_path
):
    latest_dir = tmp_path / "latest"
    latest_dir.mkdir()
    latest_report = latest_dir / "supervised_proof_artifact_report.json"
    latest_report.write_text(json.dumps(proof_report_with_autonomy_overlay("ok")), encoding="utf-8")
    (latest_dir / "summary.json").write_text(
        json.dumps(
            {
                "type": "no_wing_release_check",
                "status": "passed",
                "read_surface_fixture_evidence": {
                    "status": "passed",
                    "snapshot_status": "fixture_evidence_incomplete",
                    "physical_gap_surface_count": 2,
                    "summary_recomputed_from_rows": True,
                    "fixture_declaration_validation": True,
                    "contract_gap_consistency_enforced": True,
                    "read_only": True,
                    "live_write": False,
                    "full_autonomous_ready": False,
                    "physical_fixture_missing_surface_ids": [
                        "raw_mixer_state_cache",
                        "channel_identity_and_basic_controls",
                    ],
                    "next_physical_readback_surface_id": "dlive_state",
                    "violations": [],
                },
            }
        ),
        encoding="utf-8",
    )
    server.config["supervised_proof_evidence"] = {"artifact_root": str(tmp_path)}

    payload = build_supervised_proof_evidence_snapshot(server)

    proof_quality = payload["summary"]["read_surface_proof_quality"]
    assert proof_quality["next_physical_readback_surface_id"] == "raw_mixer_state_cache"
    assert proof_quality["source_next_physical_readback_surface_id"] == "dlive_state"
    assert proof_quality["next_physical_readback_target_consistent"] is False
    assert proof_quality["full_autonomous_ready"] is False
    assert proof_quality["read_only"] is True
    assert proof_quality["live_write"] is False
    assert payload["safety"]["live_write"] is False
    assert "set_gain" not in set(_walk_keys(payload))


def test_supervised_proof_evidence_snapshot_marks_stale_artifacts(server, tmp_path):
    report_path = tmp_path / "supervised_proof_artifact_report.json"
    report_path.write_text(
        json.dumps({**proof_report_with_autonomy_overlay(), "generated_at": "2020-01-01T00:00:00Z"}),
        encoding="utf-8",
    )
    server.config["supervised_proof_evidence"] = {
        "report_path": str(report_path),
        "stale_after_sec": 1,
    }

    payload = build_supervised_proof_evidence_snapshot(server)

    assert payload["available"] is True
    assert payload["summary"]["latest_report_stale"] is True
    assert payload["summary"]["latest_report_age_sec"] > 1
    assert payload["summary"]["stale_after_sec"] == 1
    assert payload["safety"]["live_write"] is False
    assert payload["autonomy_evidence_overlay"]["safety"]["full_autonomous_ready"] is False


def test_processing_pipeline_snapshot_includes_sanitized_autonomy_overlay(server, tmp_path):
    report_path = tmp_path / "custom-proof-report.json"
    report_path.write_text(json.dumps(proof_report_with_autonomy_overlay()), encoding="utf-8")
    server.config["supervised_proof_evidence"] = {"report_path": str(report_path)}

    payload = build_processing_pipeline_snapshot(server, include_methods=False)

    assert payload["methods"] == []
    assert payload["supervised_proof_evidence"]["available"] is True
    assert payload["supervised_proof_evidence"]["safety"]["live_write"] is False
    overlay = payload["autonomy_evidence_overlay"]
    assert overlay == payload["supervised_proof_evidence"]["autonomy_evidence_overlay"]
    assert overlay["source_report_path"] == str(report_path)
    assert overlay["safety"]["live_write"] is False
    assert overlay["safety"]["automatic_safety_update_allowed"] is False
    assert overlay["safety"]["full_autonomous_ready"] is False
    assert any(row["value_type"] == "gain" for row in overlay["rows"])
    assert any(row["value_type"] == "fader" for row in overlay["rows"])
    assert "set_gain" not in set(_walk_keys(payload))


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


def test_main_pipeline_preview_exposes_source_identity_replay_calibration_review(
    server,
):
    server.source_identity_replay_context_cache_freshness = {
        "status": "fresh",
        "field_count": 3,
        "fresh_field_count": 3,
        "stale_field_count": 0,
        "read_only": True,
        "live_write": False,
        "queries_console": False,
    }
    server.source_identity_replay_events = [
        {
            "event_id": f"lead:{index}",
            "source_id": f"ch{index}",
            "predicted_source_type": "lead_vocal",
            "predicted_confidence": 0.86 + index * 0.01,
            "operator_decision": "accepted",
            "accepted_source_type": "lead_vocal",
            "replay_matched": True,
            "downstream_family": "arrangement_role",
            "evidence_packet_id": f"packet:{index}",
            "session_id": "show:source-id",
        }
        for index in range(1, 4)
    ]

    payload = build_main_pipeline_preview(server, total_channels=4)
    queue = build_decision_queue(server)
    proposals = [
        proposal
        for proposal in payload["proposals"]
        if proposal["adapter_id"] == "source_identity_replay_calibration"
    ]

    assert payload["summary"]["source_identity_replay_event_count"] == 3
    assert payload["summary"]["source_identity_replay_calibration_proposal_count"] == 1
    assert payload["source_identity_replay_calibration"]["available"] is True
    assert payload["source_identity_replay_calibration"]["metadata"]["queue_mutation"] is False
    assert payload["source_identity_replay_calibration"]["metadata"]["live_write"] is False
    assert (
        payload["source_identity_replay_calibration"]["metadata"][
            "downstream_priority_mutation"
        ]
        is False
    )
    assert len(proposals) == 1
    proposal = proposals[0]
    assert proposal["method_id"] == "source_identity_replay_calibration"
    assert proposal["queue_action"] == "preview_only_not_queued"
    assert proposal["can_import_to_queue"] is False
    assert proposal["can_apply_live"] is False
    assert "requested_change" not in proposal
    assert "unsupported_value_type:source_identity_calibration" in proposal["blockers"]
    assert proposal["proposed_action"]["recommended_trust_state"] == (
        "trusted_for_downstream_priority"
    )
    assert proposal["evidence"]["downstream_priority_mutation"] is False
    ledger = payload["source_identity_calibration_acceptance_ledger"]
    plan = payload["source_identity_calibration_persistence_plan"]
    draft = payload["source_identity_calibration_memory_draft"]
    writer = payload["source_identity_calibration_memory_writer"]
    assert ledger["status"] == "operator_acceptance_required"
    assert ledger["summary"]["calibration_candidate_count"] == 1
    assert ledger["summary"]["unreviewed_candidate_ids"] == [proposal["id"]]
    assert ledger["summary"]["downstream_priority_change_allowed_by_acceptance"] is False
    assert ledger["summary"]["memory_mutation_allowed_by_acceptance"] is False
    assert plan["status"] == "calibration_persistence_not_ready"
    assert plan["summary"]["persistence_candidate_count"] == 0
    assert plan["summary"]["memory_mutation_allowed_by_plan"] is False
    assert plan["summary"]["downstream_priority_change_allowed_by_plan"] is False
    assert draft["status"] == "memory_draft_not_ready"
    assert draft["summary"]["draft_record_count"] == 0
    assert draft["summary"]["memory_mutation_allowed_by_draft"] is False
    assert draft["summary"]["downstream_priority_change_allowed_by_draft"] is False
    assert writer["status"] == "memory_write_not_ready"
    assert writer["summary"]["write_result_count"] == 0
    assert writer["summary"]["runtime_memory_mutation"] is False
    assert writer["summary"]["downstream_priority_mutation"] is False
    assert queue["summary"]["pending_count"] == 0


def test_main_pipeline_preview_exposes_source_identity_acceptance_ledger_records(
    server,
):
    server.source_identity_replay_context_cache_freshness = {
        "status": "fresh",
        "field_count": 3,
        "fresh_field_count": 3,
        "stale_field_count": 0,
        "read_only": True,
        "live_write": False,
        "queries_console": False,
    }
    server.source_identity_replay_events = [
        {
            "event_id": f"lead:{index}",
            "source_id": f"ch{index}",
            "predicted_source_type": "lead_vocal",
            "predicted_confidence": 0.86 + index * 0.01,
            "operator_decision": "accepted",
            "accepted_source_type": "lead_vocal",
            "replay_matched": True,
            "downstream_family": "arrangement_role",
            "evidence_packet_id": f"packet:{index}",
            "session_id": "show:source-id",
        }
        for index in range(1, 4)
    ]
    server.source_identity_calibration_acceptance_records = [
        {
            "review_id": "source-id-acceptance-1",
            "candidate_id": (
                "main_live_assist:source_identity_replay_calibration:"
                "source_identity_replay_calibration:lead_vocal"
            ),
            "calibration_method_id": "source_identity_replay_calibration",
            "source_type": "lead_vocal",
            "recommended_trust_state": "trusted_for_downstream_priority",
            "decision": "accepted",
            "reviewed_by": "operator",
            "reviewed_at": "2026-06-06T07:30:00Z",
        }
    ]

    payload = build_main_pipeline_preview(server, total_channels=4)
    queue = build_decision_queue(server)
    ledger = payload["source_identity_calibration_acceptance_ledger"]
    plan = payload["source_identity_calibration_persistence_plan"]
    draft = payload["source_identity_calibration_memory_draft"]
    writer = payload["source_identity_calibration_memory_writer"]

    assert payload["summary"]["source_identity_calibration_acceptance_record_count"] == 1
    assert payload["summary"]["source_identity_calibration_accepted_record_count"] == 1
    assert payload["summary"]["source_identity_calibration_invalid_record_count"] == 0
    assert payload["summary"]["source_identity_calibration_persistence_candidate_count"] == 1
    assert payload["summary"]["source_identity_calibration_memory_draft_record_count"] == 1
    assert payload["summary"]["source_identity_calibration_memory_writer_write_result_count"] == 0
    assert ledger["status"] == "operator_acceptance_records_available"
    assert ledger["summary"]["accepted_source_types"] == ["lead_vocal"]
    assert ledger["summary"]["calibration_persistence_allowed_by_acceptance"] is False
    assert ledger["summary"]["downstream_priority_change_allowed_by_acceptance"] is False
    assert ledger["summary"]["model_mutation_allowed_by_acceptance"] is False
    assert ledger["summary"]["memory_mutation_allowed_by_acceptance"] is False
    assert ledger["records"][0]["operator_accepted_calibration_candidate"] is True
    assert ledger["records"][0]["live_authority_granted"] is False
    assert plan["status"] == "manual_calibration_persistence_candidate_available"
    assert plan["summary"]["persistence_candidate_count"] == 1
    assert plan["summary"]["accepted_source_types"] == ["lead_vocal"]
    assert plan["summary"]["calibration_persistence_write_allowed_by_plan"] is False
    assert plan["summary"]["memory_mutation_allowed_by_plan"] is False
    assert plan["summary"]["downstream_priority_change_allowed_by_plan"] is False
    assert plan["persistence_candidates"][0]["manual_calibration_persistence_required"] is True
    assert (
        plan["persistence_candidates"][0]["calibration_persistence_write_allowed_by_plan"]
        is False
    )
    assert draft["status"] == "append_only_memory_draft_available"
    assert draft["summary"]["draft_record_count"] == 1
    assert draft["summary"]["draft_source_types"] == ["lead_vocal"]
    assert draft["summary"]["append_only_memory_write_allowed_by_draft"] is False
    assert draft["summary"]["memory_mutation_allowed_by_draft"] is False
    assert draft["summary"]["downstream_priority_change_allowed_by_draft"] is False
    assert draft["draft_records"][0]["append_only"] is True
    assert draft["draft_records"][0]["memory_write_allowed_by_draft"] is False
    assert draft["draft_records"][0]["source_type"] == "lead_vocal"
    assert writer["status"] == "append_only_memory_write_dry_run_ready"
    assert writer["summary"]["draft_record_count"] == 1
    assert writer["summary"]["write_result_count"] == 0
    assert writer["summary"]["append_only_memory_write_applied"] is False
    assert writer["summary"]["runtime_memory_mutation"] is False
    assert writer["summary"]["downstream_priority_mutation"] is False
    assert writer["write_results"] == []
    assert queue["summary"]["pending_count"] == 0


def test_main_pipeline_preview_exposes_source_identity_memory_index_records(server):
    server.source_identity_replay_context_cache_freshness = {
        "status": "fresh",
        "field_count": 3,
        "fresh_field_count": 3,
        "stale_field_count": 0,
        "read_only": True,
        "live_write": False,
        "queries_console": False,
    }
    server.source_identity_calibration_memory_text = "\n".join(
        [
            "# Project Memory",
            "",
            "---",
            "",
            "### Source identity calibration memory: lead_vocal",
            "",
            "- Draft ID: `source_identity_memory:candidate-1`",
            "- Source candidate: `candidate-1`",
            "- Source review: `review-1`",
            "- Recommended trust state: `trusted_for_downstream_priority`",
            "- Reviewed by: `operator`",
            "- Reviewed at: `2026-06-06T11:00:00Z`",
            (
                "- Runtime authority: no model, memory-priority, queue, "
                "or live-console mutation"
            ),
            "",
            "Source identity calibration for `lead_vocal` was operator-reviewed.",
            "",
        ]
    )

    payload = build_main_pipeline_preview(server, total_channels=4)
    queue = build_decision_queue(server)
    index = payload["source_identity_calibration_memory_index"]
    review = payload["source_identity_calibration_memory_trust_review"]
    ledger = payload["source_identity_calibration_memory_trust_acceptance_ledger"]
    adapter = [
        item
        for item in payload["adapters"]
        if item["adapter_id"] == "source_identity_calibration_memory_index"
    ][0]
    review_adapter = [
        item
        for item in payload["adapters"]
        if item["adapter_id"] == "source_identity_calibration_memory_trust_review"
    ][0]

    assert payload["summary"]["source_identity_calibration_memory_index_record_count"] == 1
    assert payload["summary"]["source_identity_calibration_memory_index_invalid_record_count"] == 0
    assert (
        payload["summary"][
            "source_identity_calibration_memory_trust_review_candidate_count"
        ]
        == 1
    )
    assert adapter["method_id"] == "source_identity_calibration_memory_index"
    assert adapter["output"] == "source_identity_append_only_memory_index_review"
    assert review_adapter["method_id"] == "source_identity_calibration_memory_trust_review"
    assert review_adapter["output"] == "source_identity_memory_trust_operator_review"
    assert index["status"] == "append_only_memory_index_available"
    assert index["summary"]["memory_record_count"] == 1
    assert index["summary"]["runtime_memory_mutation"] is False
    assert index["summary"]["downstream_priority_mutation"] is False
    assert index["memory_records"][0]["source_type"] == "lead_vocal"
    assert index["memory_records"][0]["usable_for_downstream_priority"] is False
    assert review["status"] == "memory_trust_review_candidate_available"
    assert review["summary"]["trust_review_candidate_count"] == 1
    assert review["summary"]["runtime_memory_mutation"] is False
    assert review["summary"]["downstream_priority_mutation"] is False
    assert review["trust_review_candidates"][0]["source_type"] == "lead_vocal"
    assert (
        review["trust_review_candidates"][0][
            "downstream_priority_change_allowed_by_review"
        ]
        is False
    )
    assert ledger["status"] == "operator_memory_trust_acceptance_required"
    assert ledger["summary"]["trust_review_candidate_count"] == 1
    assert ledger["summary"]["acceptance_record_count"] == 0
    plan = payload["source_identity_calibration_memory_trust_promotion_plan"]
    assert plan["status"] == "memory_trust_promotion_not_ready"
    assert plan["summary"]["promotion_candidate_count"] == 0
    contract = payload["source_identity_calibration_memory_trust_promotion_contract"]
    assert contract["status"] == "memory_trust_promotion_contract_not_ready"
    assert contract["summary"]["contract_record_count"] == 0
    assert queue["summary"]["pending_count"] == 0


def test_main_pipeline_preview_exposes_source_identity_memory_trust_acceptance(server):
    server.source_identity_replay_context_cache_freshness = {
        "status": "fresh",
        "field_count": 3,
        "fresh_field_count": 3,
        "stale_field_count": 0,
        "read_only": True,
        "live_write": False,
        "queries_console": False,
    }
    server.source_identity_calibration_memory_text = "\n".join(
        [
            "# Project Memory",
            "",
            "---",
            "",
            "### Source identity calibration memory: lead_vocal",
            "",
            "- Draft ID: `source_identity_memory:candidate-1`",
            "- Source candidate: `candidate-1`",
            "- Source review: `review-1`",
            "- Recommended trust state: `trusted_for_downstream_priority`",
            "- Reviewed by: `operator`",
            "- Reviewed at: `2026-06-06T11:00:00Z`",
            (
                "- Runtime authority: no model, memory-priority, queue, "
                "or live-console mutation"
            ),
            "",
            "Source identity calibration for `lead_vocal` was operator-reviewed.",
            "",
        ]
    )
    server.source_identity_memory_trust_acceptance_records = [
        {
            "acceptance_id": "memory-trust-acceptance-1",
            "review_candidate_id": (
                "source_identity_memory_trust_review:"
                "source_identity_memory_index:source_identity_memory:candidate-1"
            ),
            "memory_record_id": (
                "source_identity_memory_index:source_identity_memory:candidate-1"
            ),
            "source_candidate_id": "candidate-1",
            "source_type": "lead_vocal",
            "recommended_trust_state": "trusted_for_downstream_priority",
            "decision": "accepted",
            "reviewed_by": "operator",
            "reviewed_at": "2026-06-06T12:00:00Z",
        }
    ]

    payload = build_main_pipeline_preview(server, total_channels=4)
    queue = build_decision_queue(server)
    ledger = payload["source_identity_calibration_memory_trust_acceptance_ledger"]
    plan = payload["source_identity_calibration_memory_trust_promotion_plan"]
    contract = payload["source_identity_calibration_memory_trust_promotion_contract"]
    promotion_acceptance = payload[
        "source_identity_calibration_memory_trust_promotion_acceptance_ledger"
    ]
    adapter = [
        item
        for item in payload["adapters"]
        if item["adapter_id"]
        == "source_identity_calibration_memory_trust_acceptance_ledger"
    ][0]
    plan_adapter = [
        item
        for item in payload["adapters"]
        if item["adapter_id"]
        == "source_identity_calibration_memory_trust_promotion_plan"
    ][0]
    contract_adapter = [
        item
        for item in payload["adapters"]
        if item["adapter_id"]
        == "source_identity_calibration_memory_trust_promotion_contract"
    ][0]

    assert (
        payload["summary"][
            "source_identity_calibration_memory_trust_acceptance_record_count"
        ]
        == 1
    )
    assert (
        payload["summary"][
            "source_identity_calibration_memory_trust_promotion_candidate_count"
        ]
        == 1
    )
    assert (
        payload["summary"][
            "source_identity_calibration_memory_trust_accepted_record_count"
        ]
        == 1
    )
    assert (
        payload["summary"][
            "source_identity_calibration_memory_trust_promotion_contract_record_count"
        ]
        == 0
    )
    assert (
        adapter["method_id"]
        == "source_identity_calibration_memory_trust_acceptance_ledger"
    )
    assert adapter["output"] == "source_identity_memory_trust_acceptance_ledger"
    assert plan_adapter["method_id"] == "source_identity_calibration_memory_trust_promotion_plan"
    assert plan_adapter["output"] == "source_identity_memory_trust_manual_promotion_plan"
    assert (
        contract_adapter["method_id"]
        == "source_identity_calibration_memory_trust_promotion_contract"
    )
    assert contract_adapter["output"] == "source_identity_memory_trust_promotion_contract_review"
    assert ledger["status"] == "operator_memory_trust_acceptance_records_available"
    assert ledger["summary"]["accepted_record_count"] == 1
    assert ledger["summary"]["memory_trust_promotion_allowed_by_acceptance"] is False
    assert (
        ledger["summary"]["downstream_priority_change_allowed_by_acceptance"]
        is False
    )
    assert ledger["records"][0]["operator_accepted_memory_trust_candidate"] is True
    assert (
        ledger["records"][0]["downstream_priority_change_allowed_by_acceptance"]
        is False
    )
    assert plan["status"] == "manual_memory_trust_promotion_candidate_available"
    assert plan["summary"]["promotion_candidate_count"] == 1
    assert plan["summary"]["memory_trust_promotion_allowed_by_plan"] is False
    assert plan["summary"]["downstream_priority_change_allowed_by_plan"] is False
    assert plan["promotion_candidates"][0]["source_type"] == "lead_vocal"
    assert (
        plan["promotion_candidates"][0]["downstream_priority_change_allowed_by_plan"]
        is False
    )
    assert contract["status"] == "memory_trust_promotion_contract_not_ready"
    assert contract["summary"]["contract_record_count"] == 0
    assert "rollback_evidence_required_before_promotion_contract" in contract["summary"][
        "blocked_by"
    ]
    assert contract["summary"]["downstream_priority_change_allowed_by_contract"] is False
    assert queue["summary"]["pending_count"] == 0


def test_main_pipeline_preview_exposes_source_identity_memory_trust_promotion_contract(server):
    server.source_identity_replay_context_cache_freshness = {
        "status": "fresh",
        "field_count": 3,
        "fresh_field_count": 3,
        "stale_field_count": 0,
        "read_only": True,
        "live_write": False,
        "queries_console": False,
    }
    server.source_identity_calibration_memory_text = "\n".join(
        [
            "# Project Memory",
            "",
            "---",
            "",
            "### Source identity calibration memory: lead_vocal",
            "",
            "- Draft ID: `source_identity_memory:candidate-1`",
            "- Source candidate: `candidate-1`",
            "- Source review: `review-1`",
            "- Recommended trust state: `trusted_for_downstream_priority`",
            "- Reviewed by: `operator`",
            "- Reviewed at: `2026-06-06T11:00:00Z`",
            (
                "- Runtime authority: no model, memory-priority, queue, "
                "or live-console mutation"
            ),
            "",
            "Source identity calibration for `lead_vocal` was operator-reviewed.",
            "",
        ]
    )
    server.source_identity_memory_trust_acceptance_records = [
        {
            "acceptance_id": "memory-trust-acceptance-1",
            "review_candidate_id": (
                "source_identity_memory_trust_review:"
                "source_identity_memory_index:source_identity_memory:candidate-1"
            ),
            "memory_record_id": (
                "source_identity_memory_index:source_identity_memory:candidate-1"
            ),
            "source_candidate_id": "candidate-1",
            "source_type": "lead_vocal",
            "recommended_trust_state": "trusted_for_downstream_priority",
            "decision": "accepted",
            "reviewed_by": "operator",
            "reviewed_at": "2026-06-06T12:00:00Z",
        }
    ]
    server.source_identity_memory_trust_promotion_rollback_evidence = {
        "status": "ready",
        "read_only": True,
        "live_write": False,
        "mutates_console_state": False,
        "queries_console": False,
        "queue_mutation": False,
        "model_mutation": False,
        "runtime_memory_mutation": False,
        "downstream_priority_mutation": False,
        "live_authority_granted": False,
        "full_autonomous_ready": False,
        "rollback_strategy": "manual_revert_to_previous_trust_state",
        "previous_trust_state": "legacy_unverified",
        "target_store": "source_identity_downstream_gate",
        "reviewed_by": "operator",
        "reviewed_at": "2026-06-06T12:30:00Z",
    }

    payload = build_main_pipeline_preview(server, total_channels=4)
    queue = build_decision_queue(server)
    contract = payload["source_identity_calibration_memory_trust_promotion_contract"]
    promotion_acceptance = payload[
        "source_identity_calibration_memory_trust_promotion_acceptance_ledger"
    ]
    adapter = [
        item
        for item in payload["adapters"]
        if item["adapter_id"]
        == "source_identity_calibration_memory_trust_promotion_contract"
    ][0]
    acceptance_adapter = [
        item
        for item in payload["adapters"]
        if item["adapter_id"]
        == "source_identity_calibration_memory_trust_promotion_acceptance_ledger"
    ][0]

    assert (
        payload["summary"][
            "source_identity_calibration_memory_trust_promotion_contract_record_count"
        ]
        == 1
    )
    assert (
        payload["summary"][
            "source_identity_calibration_memory_trust_promotion_acceptance_record_count"
        ]
        == 0
    )
    assert (
        adapter["method_id"]
        == "source_identity_calibration_memory_trust_promotion_contract"
    )
    assert adapter["output"] == "source_identity_memory_trust_promotion_contract_review"
    assert (
        acceptance_adapter["method_id"]
        == "source_identity_calibration_memory_trust_promotion_acceptance_ledger"
    )
    assert (
        acceptance_adapter["output"]
        == "source_identity_memory_trust_promotion_acceptance_ledger"
    )
    assert contract["status"] == "manual_memory_trust_promotion_contract_ready"
    assert contract["summary"]["contract_record_count"] == 1
    assert contract["summary"]["memory_trust_promotion_allowed_by_contract"] is False
    assert (
        contract["summary"]["downstream_priority_change_allowed_by_contract"]
        is False
    )
    assert contract["contract_records"][0]["source_type"] == "lead_vocal"
    assert contract["contract_records"][0]["read_only"] is True
    assert contract["contract_records"][0]["live_write"] is False
    assert contract["contract_records"][0]["mutates_console_state"] is False
    assert contract["contract_records"][0]["queries_console"] is False
    assert (
        contract["contract_records"][0][
            "memory_trust_promotion_allowed_by_contract"
        ]
        is False
    )
    assert (
        contract["contract_records"][0][
            "downstream_priority_change_allowed_by_contract"
        ]
        is False
    )
    assert contract["contract_records"][0]["live_widening_allowed"] is False
    assert (
        promotion_acceptance["status"]
        == "operator_memory_trust_promotion_acceptance_required"
    )
    assert promotion_acceptance["summary"]["acceptance_record_count"] == 0
    assert promotion_acceptance["summary"]["accepted_record_count"] == 0
    assert (
        "operator_must_review_memory_trust_promotion_contract_records"
        in promotion_acceptance["summary"]["blocked_by"]
    )
    assert (
        promotion_acceptance["summary"][
            "downstream_priority_change_allowed_by_acceptance"
        ]
        is False
    )
    assert queue["summary"]["pending_count"] == 0


def test_main_pipeline_preview_exposes_source_identity_memory_trust_promotion_acceptance(server):
    server.source_identity_replay_context_cache_freshness = {
        "status": "fresh",
        "field_count": 3,
        "fresh_field_count": 3,
        "stale_field_count": 0,
        "read_only": True,
        "live_write": False,
        "queries_console": False,
    }
    server.source_identity_calibration_memory_text = "\n".join(
        [
            "# Project Memory",
            "",
            "---",
            "",
            "### Source identity calibration memory: lead_vocal",
            "",
            "- Draft ID: `source_identity_memory:candidate-1`",
            "- Source candidate: `candidate-1`",
            "- Source review: `review-1`",
            "- Recommended trust state: `trusted_for_downstream_priority`",
            "- Reviewed by: `operator`",
            "- Reviewed at: `2026-06-06T11:00:00Z`",
            (
                "- Runtime authority: no model, memory-priority, queue, "
                "or live-console mutation"
            ),
            "",
            "Source identity calibration for `lead_vocal` was operator-reviewed.",
            "",
        ]
    )
    server.source_identity_memory_trust_acceptance_records = [
        {
            "acceptance_id": "memory-trust-acceptance-1",
            "review_candidate_id": (
                "source_identity_memory_trust_review:"
                "source_identity_memory_index:source_identity_memory:candidate-1"
            ),
            "memory_record_id": (
                "source_identity_memory_index:source_identity_memory:candidate-1"
            ),
            "source_candidate_id": "candidate-1",
            "source_type": "lead_vocal",
            "recommended_trust_state": "trusted_for_downstream_priority",
            "decision": "accepted",
            "reviewed_by": "operator",
            "reviewed_at": "2026-06-06T12:00:00Z",
        }
    ]
    server.source_identity_memory_trust_promotion_rollback_evidence = {
        "status": "ready",
        "read_only": True,
        "live_write": False,
        "mutates_console_state": False,
        "queries_console": False,
        "queue_mutation": False,
        "model_mutation": False,
        "runtime_memory_mutation": False,
        "downstream_priority_mutation": False,
        "live_authority_granted": False,
        "full_autonomous_ready": False,
        "rollback_strategy": "manual_revert_to_previous_trust_state",
        "previous_trust_state": "legacy_unverified",
        "target_store": "source_identity_downstream_gate",
        "reviewed_by": "operator",
        "reviewed_at": "2026-06-06T12:30:00Z",
    }
    server.source_identity_memory_trust_promotion_acceptance_records = [
        {
            "promotion_acceptance_id": "memory-trust-promotion-acceptance-1",
            "promotion_contract_record_id": (
                "source_identity_memory_trust_promotion_contract:"
                "source_identity_memory_trust_promotion:memory-trust-acceptance-1"
            ),
            "promotion_candidate_id": (
                "source_identity_memory_trust_promotion:memory-trust-acceptance-1"
            ),
            "source_candidate_id": "candidate-1",
            "source_type": "lead_vocal",
            "recommended_trust_state": "trusted_for_downstream_priority",
            "previous_trust_state": "legacy_unverified",
            "rollback_strategy": "manual_revert_to_previous_trust_state",
            "decision": "accepted",
            "reviewed_by": "operator",
            "reviewed_at": "2026-06-06T13:00:00Z",
        }
    ]

    payload = build_main_pipeline_preview(server, total_channels=4)
    queue = build_decision_queue(server)
    ledger = payload[
        "source_identity_calibration_memory_trust_promotion_acceptance_ledger"
    ]
    execution_plan = payload[
        "source_identity_calibration_memory_trust_promotion_execution_plan"
    ]
    executor_dry_run = payload[
        "source_identity_calibration_memory_trust_promotion_executor_dry_run"
    ]
    executor = payload[
        "source_identity_calibration_memory_trust_promotion_executor"
    ]
    apply_audit = payload[
        "source_identity_calibration_memory_trust_promotion_apply_audit"
    ]
    rollback_verification = payload[
        "source_identity_calibration_memory_trust_promotion_rollback_verification"
    ]
    priority_gate = payload[
        "source_identity_calibration_memory_trust_promotion_priority_gate"
    ]
    priority_policy = payload["source_identity_proposal_priority_policy"]
    ordering_preview = payload["source_identity_proposal_ordering_preview"]
    queue_ordering_plan = payload["source_identity_proposal_queue_ordering_plan"]
    adapter = [
        item
        for item in payload["adapters"]
        if item["adapter_id"]
        == "source_identity_calibration_memory_trust_promotion_acceptance_ledger"
    ][0]
    execution_adapter = [
        item
        for item in payload["adapters"]
        if item["adapter_id"]
        == "source_identity_calibration_memory_trust_promotion_execution_plan"
    ][0]
    dry_run_adapter = [
        item
        for item in payload["adapters"]
        if item["adapter_id"]
        == "source_identity_calibration_memory_trust_promotion_executor_dry_run"
    ][0]
    executor_adapter = [
        item
        for item in payload["adapters"]
        if item["adapter_id"]
        == "source_identity_calibration_memory_trust_promotion_executor"
    ][0]
    apply_audit_adapter = [
        item
        for item in payload["adapters"]
        if item["adapter_id"]
        == "source_identity_calibration_memory_trust_promotion_apply_audit"
    ][0]
    rollback_verification_adapter = [
        item
        for item in payload["adapters"]
        if item["adapter_id"]
        == "source_identity_calibration_memory_trust_promotion_rollback_verification"
    ][0]
    priority_gate_adapter = [
        item
        for item in payload["adapters"]
        if item["adapter_id"]
        == "source_identity_calibration_memory_trust_promotion_priority_gate"
    ][0]
    priority_policy_adapter = [
        item
        for item in payload["adapters"]
        if item["adapter_id"] == "source_identity_proposal_priority_policy"
    ][0]
    ordering_preview_adapter = [
        item
        for item in payload["adapters"]
        if item["adapter_id"] == "source_identity_proposal_ordering_preview"
    ][0]
    queue_ordering_plan_adapter = [
        item
        for item in payload["adapters"]
        if item["adapter_id"] == "source_identity_proposal_queue_ordering_plan"
    ][0]

    assert (
        payload["summary"][
            "source_identity_calibration_memory_trust_promotion_acceptance_record_count"
        ]
        == 1
    )
    assert (
        payload["summary"][
            "source_identity_calibration_memory_trust_promotion_accepted_record_count"
        ]
        == 1
    )
    assert (
        payload["summary"][
            "source_identity_calibration_memory_trust_promotion_execution_candidate_count"
        ]
        == 1
    )
    assert (
        payload["summary"][
            "source_identity_calibration_memory_trust_promotion_executor_dry_run_record_count"
        ]
        == 1
    )
    assert (
        payload["summary"][
            "source_identity_calibration_memory_trust_promotion_executor_apply_result_count"
        ]
        == 0
    )
    assert (
        payload["summary"][
            "source_identity_calibration_memory_trust_promotion_apply_audit_record_count"
        ]
        == 0
    )
    assert (
        payload["summary"][
            "source_identity_calibration_memory_trust_promotion_rollback_verified_record_count"
        ]
        == 0
    )
    assert (
        payload["summary"][
            "source_identity_calibration_memory_trust_promotion_priority_gate_record_count"
        ]
        == 0
    )
    assert (
        payload["summary"][
            "source_identity_calibration_memory_trust_promotion_priority_eligible_record_count"
        ]
        == 0
    )
    assert (
        payload["summary"]["source_identity_proposal_priority_policy_record_count"]
        == 0
    )
    assert (
        payload["summary"][
            "source_identity_proposal_priority_policy_ranked_proposal_count"
        ]
        == 0
    )
    assert payload["summary"]["source_identity_proposal_ordering_preview_count"] == 0
    assert (
        payload["summary"][
            "source_identity_proposal_ordering_preview_ranked_proposal_count"
        ]
        == 0
    )
    assert (
        payload["summary"][
            "source_identity_proposal_queue_ordering_plan_record_count"
        ]
        == 0
    )
    assert (
        payload["summary"][
            "source_identity_proposal_queue_ordering_plan_ranked_record_count"
        ]
        == 0
    )
    assert (
        adapter["method_id"]
        == "source_identity_calibration_memory_trust_promotion_acceptance_ledger"
    )
    assert adapter["output"] == "source_identity_memory_trust_promotion_acceptance_ledger"
    assert (
        execution_adapter["method_id"]
        == "source_identity_calibration_memory_trust_promotion_execution_plan"
    )
    assert (
        execution_adapter["output"]
        == "source_identity_memory_trust_promotion_execution_plan"
    )
    assert (
        dry_run_adapter["method_id"]
        == "source_identity_calibration_memory_trust_promotion_executor_dry_run"
    )
    assert (
        dry_run_adapter["output"]
        == "source_identity_memory_trust_promotion_executor_dry_run"
    )
    assert (
        executor_adapter["method_id"]
        == "source_identity_calibration_memory_trust_promotion_executor"
    )
    assert (
        executor_adapter["output"]
        == "source_identity_memory_trust_promotion_executor"
    )
    assert (
        apply_audit_adapter["method_id"]
        == "source_identity_calibration_memory_trust_promotion_apply_audit"
    )
    assert (
        apply_audit_adapter["output"]
        == "source_identity_memory_trust_promotion_apply_audit"
    )
    assert (
        rollback_verification_adapter["method_id"]
        == "source_identity_calibration_memory_trust_promotion_rollback_verification"
    )
    assert (
        rollback_verification_adapter["output"]
        == "source_identity_memory_trust_promotion_rollback_verification"
    )
    assert (
        priority_gate_adapter["method_id"]
        == "source_identity_calibration_memory_trust_promotion_priority_gate"
    )
    assert (
        priority_gate_adapter["output"]
        == "source_identity_memory_trust_promotion_priority_gate"
    )
    assert priority_policy_adapter["method_id"] == "source_identity_proposal_priority_policy"
    assert (
        priority_policy_adapter["output"]
        == "source_identity_proposal_priority_policy_projection"
    )
    assert (
        ordering_preview_adapter["method_id"]
        == "source_identity_proposal_ordering_preview"
    )
    assert (
        ordering_preview_adapter["output"]
        == "source_identity_proposal_ordering_preview"
    )
    assert (
        queue_ordering_plan_adapter["method_id"]
        == "source_identity_proposal_queue_ordering_plan"
    )
    assert (
        queue_ordering_plan_adapter["output"]
        == "source_identity_proposal_queue_ordering_plan"
    )
    assert (
        ledger["status"]
        == "operator_memory_trust_promotion_acceptance_records_available"
    )
    assert ledger["summary"]["accepted_record_count"] == 1
    assert ledger["summary"]["memory_trust_promotion_allowed_by_acceptance"] is False
    assert (
        ledger["summary"]["downstream_priority_change_allowed_by_acceptance"]
        is False
    )
    assert (
        ledger["records"][0]["operator_accepted_memory_trust_promotion_contract"]
        is True
    )
    assert ledger["records"][0]["read_only"] is True
    assert ledger["records"][0]["live_write"] is False
    assert ledger["records"][0]["queries_console"] is False
    assert (
        ledger["records"][0]["downstream_priority_change_allowed_by_acceptance"]
        is False
    )
    assert ledger["records"][0]["downstream_priority_mutation"] is False
    assert ledger["records"][0]["runtime_memory_mutation_allowed_by_acceptance"] is False
    assert ledger["records"][0]["can_import_to_queue"] is False
    assert ledger["records"][0]["can_apply_live"] is False
    assert (
        execution_plan["status"]
        == "manual_memory_trust_promotion_execution_candidate_available"
    )
    assert execution_plan["summary"]["execution_candidate_count"] == 1
    assert execution_plan["summary"]["execution_allowed_by_plan"] is False
    assert execution_plan["summary"]["memory_trust_promotion_allowed_by_plan"] is False
    assert (
        execution_plan["summary"]["downstream_priority_change_allowed_by_plan"]
        is False
    )
    assert (
        execution_plan["execution_candidates"][0]["separate_executor_required"]
        is True
    )
    assert (
        execution_plan["execution_candidates"][0]["execution_allowed_by_plan"]
        is False
    )
    assert (
        execution_plan["execution_candidates"][0][
            "downstream_priority_change_allowed_by_plan"
        ]
        is False
    )
    assert executor_dry_run["status"] == "memory_trust_promotion_executor_dry_run_ready"
    assert executor_dry_run["summary"]["dry_run_record_count"] == 1
    assert executor_dry_run["summary"]["execution_allowed_by_dry_run"] is False
    assert executor_dry_run["summary"]["memory_trust_promotion_applied"] is False
    assert (
        executor_dry_run["summary"][
            "downstream_priority_change_allowed_by_dry_run"
        ]
        is False
    )
    assert (
        executor_dry_run["dry_run_records"][0]["target_path"]
        == "source_identity_downstream_gate/candidate-1/trust_state"
    )
    assert (
        executor_dry_run["dry_run_records"][0]["idempotency_key"]
        == (
            "source_identity_memory_trust_promotion:"
            "candidate-1:trusted_for_downstream_priority:"
            "memory-trust-promotion-acceptance-1"
        )
    )
    assert executor_dry_run["dry_run_records"][0]["dry_run"] is True
    assert (
        executor_dry_run["dry_run_records"][0]["memory_trust_promotion_applied"]
        is False
    )
    assert (
        executor_dry_run["dry_run_records"][0][
            "runtime_memory_mutation_allowed_by_dry_run"
        ]
        is False
    )
    assert executor["status"] == "memory_trust_promotion_executor_ready"
    assert executor["summary"]["dry_run_record_count"] == 1
    assert executor["summary"]["apply_result_count"] == 0
    assert executor["summary"]["apply_requested"] is False
    assert executor["summary"]["memory_trust_promotion_applied"] is False
    assert executor["summary"]["runtime_memory_mutation"] is False
    assert executor["summary"]["queue_mutation"] is False
    assert executor["summary"]["live_write"] is False
    assert executor["apply_results"] == []
    assert apply_audit["status"] == "memory_trust_promotion_apply_audit_not_ready"
    assert apply_audit["summary"]["executor_apply_result_count"] == 0
    assert apply_audit["summary"]["apply_audit_record_count"] == 0
    assert apply_audit["summary"]["runtime_memory_mutation"] is False
    assert apply_audit["summary"]["direct_downstream_priority_mutation"] is False
    assert apply_audit["summary"]["live_write"] is False
    assert apply_audit["safety"]["read_only"] is True
    assert apply_audit["safety"]["rollback_verified"] is False
    assert apply_audit["audit_records"] == []
    assert (
        rollback_verification["status"]
        == "memory_trust_promotion_rollback_verification_not_ready"
    )
    assert rollback_verification["summary"]["apply_audit_record_count"] == 0
    assert rollback_verification["summary"]["rollback_verified_record_count"] == 0
    assert rollback_verification["summary"]["runtime_memory_mutation"] is False
    assert (
        rollback_verification["summary"]["direct_downstream_priority_mutation"]
        is False
    )
    assert rollback_verification["summary"]["live_write"] is False
    assert rollback_verification["safety"]["read_only"] is True
    assert rollback_verification["safety"]["rollback_verified"] is False
    assert rollback_verification["verified_records"] == []
    assert (
        priority_gate["status"]
        == "memory_trust_promotion_priority_gate_not_ready"
    )
    assert priority_gate["summary"]["rollback_verified_record_count"] == 0
    assert priority_gate["summary"]["priority_gate_record_count"] == 0
    assert priority_gate["summary"]["eligible_record_count"] == 0
    assert priority_gate["summary"]["direct_downstream_priority_mutation"] is False
    assert priority_gate["summary"]["can_import_to_queue"] is False
    assert priority_gate["summary"]["can_apply_live"] is False
    assert priority_gate["safety"]["read_only"] is True
    assert priority_gate["safety"]["live_write"] is False
    assert priority_gate["priority_records"] == []
    assert priority_policy["status"] == "source_identity_proposal_priority_policy_not_ready"
    assert priority_policy["summary"]["policy_record_count"] == 0
    assert priority_policy["summary"]["ranked_proposal_ids"] == []
    assert priority_policy["summary"]["proposal_order_mutation"] is False
    assert priority_policy["summary"]["queue_mutation"] is False
    assert (
        priority_policy["summary"]["downstream_priority_change_allowed_by_policy"]
        is False
    )
    assert priority_policy["summary"]["live_widening_allowed"] is False
    assert priority_policy["summary"]["can_import_to_queue"] is False
    assert priority_policy["summary"]["can_apply_live"] is False
    assert priority_policy["safety"]["read_only"] is True
    assert priority_policy["safety"]["proposal_order_mutation"] is False
    assert (
        priority_policy["safety"]["downstream_priority_change_allowed_by_policy"]
        is False
    )
    assert priority_policy["safety"]["live_widening_allowed"] is False
    assert priority_policy["safety"]["live_write"] is False
    assert priority_policy["policy_records"] == []
    assert (
        ordering_preview["status"]
        == "source_identity_proposal_ordering_preview_not_ready"
    )
    assert ordering_preview["summary"]["ordered_preview_count"] == 0
    assert ordering_preview["summary"]["recommended_ordered_proposal_ids"] == []
    assert ordering_preview["summary"]["proposal_order_mutation"] is False
    assert ordering_preview["summary"]["queue_mutation"] is False
    assert (
        ordering_preview["summary"]["downstream_priority_change_allowed_by_policy"]
        is False
    )
    assert ordering_preview["summary"]["live_widening_allowed"] is False
    assert ordering_preview["summary"]["can_import_to_queue"] is False
    assert ordering_preview["summary"]["can_apply_live"] is False
    assert ordering_preview["safety"]["read_only"] is True
    assert ordering_preview["safety"]["proposal_order_mutation"] is False
    assert (
        ordering_preview["safety"]["downstream_priority_change_allowed_by_policy"]
        is False
    )
    assert ordering_preview["safety"]["live_widening_allowed"] is False
    assert ordering_preview["safety"]["live_write"] is False
    assert ordering_preview["ordered_proposals"] == []
    assert (
        queue_ordering_plan["status"]
        == "source_identity_proposal_queue_ordering_plan_not_ready"
    )
    assert (
        queue_ordering_plan["summary"]["queue_ordering_plan_record_count"]
        == 0
    )
    assert queue_ordering_plan["summary"]["planned_queue_proposal_ids"] == []
    assert queue_ordering_plan["summary"]["proposal_order_mutation"] is False
    assert queue_ordering_plan["summary"]["queue_mutation"] is False
    assert (
        queue_ordering_plan["summary"]["queue_import_allowed_by_plan"]
        is False
    )
    assert (
        queue_ordering_plan["summary"]["queue_reorder_allowed_by_plan"]
        is False
    )
    assert (
        queue_ordering_plan["summary"]["downstream_priority_change_allowed_by_policy"]
        is False
    )
    assert queue_ordering_plan["summary"]["live_widening_allowed"] is False
    assert queue_ordering_plan["summary"]["can_import_to_queue"] is False
    assert queue_ordering_plan["summary"]["can_apply_live"] is False
    assert queue_ordering_plan["safety"]["read_only"] is True
    assert queue_ordering_plan["safety"]["proposal_order_mutation"] is False
    assert queue_ordering_plan["safety"]["queue_mutation"] is False
    assert (
        queue_ordering_plan["safety"]["queue_import_allowed_by_plan"]
        is False
    )
    assert (
        queue_ordering_plan["safety"]["queue_reorder_allowed_by_plan"]
        is False
    )
    assert (
        queue_ordering_plan["safety"]["downstream_priority_change_allowed_by_policy"]
        is False
    )
    assert queue_ordering_plan["safety"]["live_widening_allowed"] is False
    assert queue_ordering_plan["safety"]["live_write"] is False
    assert queue_ordering_plan["queue_ordering_plan_records"] == []
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
    await server._dispatch["get_fx_inserts_snapshot"](
        object(),
        {
            "fx_slot_count": 1,
            "total_channels": 1,
            "bus_count": 1,
            "main_count": 1,
            "matrix_count": 1,
            "parameter_count": 1,
        },
    )
    await server._dispatch["get_dlive_state_snapshot"](object(), {"total_channels": 2})
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
        "fx_inserts_snapshot",
        "dlive_state_snapshot",
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
    assert by_type["fx_inserts_snapshot"]["safety"]["queries_console"] is False
    assert by_type["fx_inserts_snapshot"]["safety"]["fx_apply_allowed"] is False
    assert by_type["dlive_state_snapshot"]["safety"]["queries_console"] is False
    assert by_type["dlive_state_snapshot"]["safety"]["dlive_apply_allowed"] is False
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
    server.mixer_client.state_timestamps = {address: time.time() for address in server.mixer_client.state}
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
    server.mixer_client.state_timestamps = {address: time.time() for address in server.mixer_client.state}
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
    server.mixer_client.state_timestamps = {address: time.time() for address in server.mixer_client.state}
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
    server.mixer_client.state_timestamps = {address: time.time() for address in server.mixer_client.state}
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
async def test_supervised_operator_proposal_apply_blocks_stale_cache_pre_read(server):
    server.operator_mode = "supervised"
    server.mixer_client = CachedMixer()
    server.mixer_client.state_timestamps["/ch/2/fdr"] = time.time() - 120.0
    server._apply_manual_console_write = AsyncMock()
    server.send_to_client = AsyncMock()

    await server._dispatch["create_operator_proposal"](
        object(),
        {
            "title": "Lead Vocal: stale supervised fader",
            "channel": 2,
            "value_type": "fader",
            "value": -8.5,
            "reason": "unit",
        },
    )
    proposal = build_decision_queue(server)["pending_actions"][0]
    assert proposal["can_apply"] is True

    await server._dispatch["apply_operator_proposal"](
        object(),
        {"proposal_id": proposal["id"], "approved": True, "idempotency_key": "stale-pre-read"},
    )

    server._apply_manual_console_write.assert_not_called()
    payload = server.send_to_client.await_args.args[1]
    assert payload["type"] == "operator_proposal_apply_blocked"
    assert payload["status"] == "apply_blocked"
    assert payload["apply_result"]["reason"] == "pre_read_cache_not_fresh"
    assert payload["apply_result"]["cache_freshness"]["status"] == "stale"
    assert payload["apply_result"]["cache_freshness"]["address"] == "/ch/2/fdr"


@pytest.mark.asyncio
async def test_supervised_apply_requires_idempotency_key(server):
    server.operator_mode = "supervised"
    server.mixer_client = MagicMock()
    server.mixer_client.is_connected = True
    server.mixer_client.state = {"/ch/2/fdr": -2.5}
    server.mixer_client.state_timestamps = {address: time.time() for address in server.mixer_client.state}
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
    server.mixer_client.state_timestamps = {address: time.time() for address in server.mixer_client.state}
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
    server.mixer_client.state_timestamps = {address: time.time() for address in server.mixer_client.state}
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
    server.mixer_client.state_timestamps = {address: time.time() for address in server.mixer_client.state}
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
    server.mixer_client.state_timestamps = {address: time.time() for address in server.mixer_client.state}
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
    server.mixer_client.state_timestamps = {address: time.time() for address in server.mixer_client.state}
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
    assert proposal["requested_change"]["cache_freshness"]["status"] == "fresh"
    assert any(
        item["label"] == "Cache freshness" and item["value"] == "fresh"
        for item in proposal["requested_change"]["safety_evidence"]
    )


def test_operator_analysis_blocks_applyable_proposal_when_fader_cache_is_stale(server):
    server.mixer_client = CachedMixer()
    server.mixer_client.state_timestamps["/ch/2/fdr"] = time.time() - 120.0
    server.audio_capture = FakeAudioCapture()

    report = build_operator_analysis_report(server, total_channels=4, create_proposals=True)
    queue = build_decision_queue(server)

    assert report["status"] == "ok"
    assert report["created_count"] == 1
    assert report["cache_freshness_blocked_count"] == 1
    proposal = queue["pending_actions"][0]
    assert proposal["id"] == "analysis:peak_risk:2"
    assert proposal["can_apply"] is False
    assert proposal["requested_change"] is None
    assert proposal["apply_blockers"] == ["missing_requested_change"]
    assert proposal["raw"]["apply_status"] == "blocked_cache_freshness_not_fresh"
    assert proposal["raw"]["cache_freshness"][0]["status"] == "stale"
    assert proposal["raw"]["blocked_requested_changes"][0]["value_type"] == "fader"


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
    assert proposal["requested_change"]["cache_freshness"]["status"] == "fresh"


def test_safe_gain_recommendation_bridge_blocks_stale_trim_cache():
    mixer = CachedMixer()
    mixer.state_timestamps["/ch/2/in/set/trim"] = time.time() - 120.0

    proposals = build_safe_gain_operator_proposals(
        FakeSafeGainCalibrator().get_suggestions(),
        channel_mapping={1: 2},
        mixer_client=mixer,
    )

    assert len(proposals) == 1
    proposal = proposals[0]
    assert proposal["kind"] == "analysis"
    assert proposal["requested_change"] is None
    assert proposal["raw"]["apply_status"] == "blocked_cache_freshness_not_fresh"
    assert proposal["raw"]["cache_freshness"][0]["status"] == "stale"
    assert proposal["raw"]["blocked_requested_change"]["value_type"] == "gain"


def test_soundcheck_recommendation_bridge_maps_gain_and_fader_to_applyable_proposals():
    workflow = NoWriteSoundcheckRecommendationWorkflow()
    workflow.recommend_input_gain(channel=1, current_trim_db=0.0, target_trim_db=-4.0)
    workflow.recommend_fader(channel=2, current_fader_db=-10.0, target_fader_db=-8.5)

    proposals = build_soundcheck_operator_proposals(workflow.build_bundle())

    assert [proposal["requested_change"]["value_type"] for proposal in proposals] == ["gain", "fader"]
    assert proposals[0]["requested_change"]["value"] == -4.0
    assert proposals[1]["requested_change"]["value"] == -8.5


def test_soundcheck_recommendation_bridge_blocks_stale_cache_when_mixer_client_is_supplied():
    mixer = CachedMixer()
    mixer.state.update({
        "/ch/1/in/set/trim": 0.0,
        "/ch/2/fdr": -10.0,
    })
    mixer.state_timestamps["/ch/2/fdr"] = time.time() - 120.0
    workflow = NoWriteSoundcheckRecommendationWorkflow()
    workflow.recommend_input_gain(channel=1, current_trim_db=0.0, target_trim_db=-4.0)
    workflow.recommend_fader(channel=2, current_fader_db=-10.0, target_fader_db=-11.5)

    proposals = build_soundcheck_operator_proposals(
        workflow.build_bundle(),
        mixer_client=mixer,
    )

    assert proposals[0]["requested_change"]["cache_freshness"]["status"] == "fresh"
    assert proposals[1]["kind"] == "analysis"
    assert proposals[1]["requested_change"] is None
    assert proposals[1]["raw"]["apply_status"] == "blocked_cache_freshness_not_fresh"
    assert proposals[1]["raw"]["cache_freshness"][0]["status"] == "stale"


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
