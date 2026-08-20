"""Regression tests for standard scan b6b64252 (offline only)."""

from __future__ import annotations

import asyncio
import json
import os
from contextlib import contextmanager
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import numpy as np
import pytest

from ai_mixing_pipeline.decision_layer.replay_executor import (
    ReplayExecutionState,
    simulate_replay_decisions,
)
from ai_mixing_pipeline.decision_layer.replay_proposal_ranking import (
    ReplayDecisionProposal,
)
from ai_mixing_pipeline.models import CandidateAction, MixCandidate
from ai_mixing_pipeline.sandbox_renderer import SandboxRenderer
from automixer.safety.safety_gate import SafetyGateConfig
from backend.ml import train_classifier, train_gain_predictor, train_mix_console
from backend.ml.agent_training_service import AgentTrainingService
from tools.gpt_control_bridge import server as bridge
from virtual_mixer.virtual_mixer import VirtualWingMixer


@pytest.mark.asyncio
async def test_listening_runtime_rejects_tokenless_websocket_before_registration():
    from server import AutoMixerServer

    with patch("server.BleedService"), patch.object(
        AutoMixerServer, "_load_config", return_value={}
    ), patch.dict(os.environ, {}, clear=False):
        os.environ.pop("AUTOMIXER_WS_AUTH_TOKEN", None)
        server = AutoMixerServer()
    server._network_runtime_started = True
    websocket = MagicMock()
    websocket.close = AsyncMock()

    assert await server._authenticate_websocket(websocket) is False
    websocket.close.assert_awaited_once_with(
        code=1008,
        reason="launch_token_required",
    )
    assert websocket not in server.connected_clients


@pytest.mark.asyncio
async def test_only_strong_transport_owner_can_disconnect():
    from server import AutoMixerServer

    with patch("server.BleedService"), patch.object(
        AutoMixerServer, "_load_config", return_value={}
    ):
        server = AutoMixerServer()
    owner = object()
    sibling = object()
    server._ws_auth_required = True
    server._authenticated_websocket_ids.update({id(owner), id(sibling)})
    server._mixer_transport_owner_connection_id = id(owner)
    server._mixer_transport_owner_ref = owner
    server.mixer_client = MagicMock()
    server.mixer_client.disconnect.return_value = True
    server.send_to_client = AsyncMock()

    assert await server.disconnect_mixer(requester_websocket=sibling) is False
    server.mixer_client.disconnect.assert_not_called()
    assert await server.disconnect_mixer(requester_websocket=owner) is True


@pytest.mark.asyncio
async def test_mixer_connect_lifecycle_is_serialized_across_transports():
    from server import AutoMixerServer

    with patch("server.BleedService"), patch.object(
        AutoMixerServer, "_load_config", return_value={}
    ):
        server = AutoMixerServer()

    wing_entered = asyncio.Event()
    release_wing = asyncio.Event()
    dlive_entered = asyncio.Event()

    async def wing_attempt(*_args, **_kwargs):
        wing_entered.set()
        await release_wing.wait()
        return "wing"

    async def dlive_attempt(*_args, **_kwargs):
        dlive_entered.set()
        return "dlive"

    server._connect_wing_locked = AsyncMock(side_effect=wing_attempt)
    server._connect_dlive_locked = AsyncMock(side_effect=dlive_attempt)

    wing_task = asyncio.create_task(server.connect_wing("127.0.0.1"))
    await wing_entered.wait()
    dlive_task = asyncio.create_task(server.connect_dlive())
    await asyncio.sleep(0)

    assert not dlive_entered.is_set()
    release_wing.set()
    assert await wing_task == "wing"
    assert await dlive_task == "dlive"


@pytest.mark.asyncio
async def test_wing_reconnect_revalidates_owner_after_disconnect_yield(monkeypatch):
    import server as server_module

    with patch("server.BleedService"), patch.object(
        server_module.AutoMixerServer, "_load_config", return_value={}
    ):
        server = server_module.AutoMixerServer()
    owner = object()
    server._ws_auth_required = True
    server._authenticated_websocket_ids.add(id(owner))
    server._authorize_wing_connection_target = MagicMock(
        return_value={"ip": "127.0.0.1", "send_port": 2223, "receive_port": 2223}
    )
    server.broadcast = AsyncMock()

    async def disconnect_then_tombstone(*_args, **_kwargs):
        server._authenticated_websocket_ids.discard(id(owner))
        return True

    server._disconnect_mixer_locked = AsyncMock(
        side_effect=disconnect_then_tombstone
    )
    candidate = MagicMock()
    candidate.connect.return_value = True
    candidate.disconnect.return_value = True
    monkeypatch.setattr(server_module, "EnhancedOSCClient", lambda **_kwargs: candidate)

    assert await server.connect_wing("127.0.0.1", owner_websocket=owner) is False
    candidate.disconnect.assert_called_once()
    assert server.mixer_client is None
    assert server._mixer_transport_owner_ref is None


@pytest.mark.asyncio
async def test_cancelled_wing_connect_disconnects_live_candidate(monkeypatch):
    import server as server_module

    with patch("server.BleedService"), patch.object(
        server_module.AutoMixerServer, "_load_config", return_value={}
    ):
        server = server_module.AutoMixerServer()
    owner = object()
    server._ws_auth_required = True
    server._authenticated_websocket_ids.add(id(owner))
    server._authorize_wing_connection_target = MagicMock(
        return_value={"ip": "127.0.0.1", "send_port": 2223, "receive_port": 2223}
    )
    candidate = MagicMock()
    candidate.connect.return_value = True
    candidate.disconnect.return_value = True
    monkeypatch.setattr(server_module, "EnhancedOSCClient", lambda **_kwargs: candidate)
    server.broadcast = AsyncMock(side_effect=asyncio.CancelledError)

    with pytest.raises(asyncio.CancelledError):
        await server.connect_wing("127.0.0.1", owner_websocket=owner)

    candidate.disconnect.assert_called_once()
    assert server.mixer_client is None
    assert server._mixer_transport_owner_ref is None


@pytest.mark.asyncio
async def test_failed_disconnect_quarantines_future_transport_replacement(monkeypatch):
    import server as server_module

    with patch("server.BleedService"), patch.object(
        server_module.AutoMixerServer, "_load_config", return_value={}
    ):
        server = server_module.AutoMixerServer()
    installed = MagicMock()
    installed.disconnect.side_effect = RuntimeError("teardown failed")
    server.mixer_client = installed
    server.connection_mode = "wing"
    server.broadcast = AsyncMock()

    assert await server.disconnect_mixer(force=True) is False
    assert server.mixer_client is None
    assert server.connection_mode is None
    assert server._mixer_transport_teardown_failed is True

    constructor = MagicMock()
    dlive_constructor = MagicMock()
    monkeypatch.setattr(server_module, "EnhancedOSCClient", constructor)
    monkeypatch.setattr(server_module, "DLiveClient", dlive_constructor)
    assert await server.connect_wing("127.0.0.1") is False
    assert await server.connect_dlive() is False
    constructor.assert_not_called()
    dlive_constructor.assert_not_called()


@pytest.mark.asyncio
async def test_unverified_disconnect_result_sets_persistent_quarantine():
    from server import AutoMixerServer

    with patch("server.BleedService"), patch.object(
        AutoMixerServer, "_load_config", return_value={}
    ):
        server = AutoMixerServer()
    installed = MagicMock()
    installed.disconnect.return_value = False
    server.mixer_client = installed

    assert await server.disconnect_mixer(force=True) is False
    assert server._mixer_transport_teardown_failed is True
    assert server.mixer_client is None


def test_live_transport_disconnects_report_close_failure():
    from dlive_client import DLiveClient
    from wing_client import WingClient

    wing = WingClient(ip="127.0.0.1")
    wing.sock = MagicMock()
    wing.sock.close.side_effect = OSError("close failed")
    assert wing.disconnect() is False

    dlive = DLiveClient(ip="127.0.0.1")
    dlive._sock = MagicMock()
    dlive._sock.close.side_effect = OSError("close failed")
    assert dlive.disconnect() is False


@pytest.mark.parametrize(
    ("loader", "kwargs", "message"),
    [
        (
            train_classifier._load_dataset_from_file,
            {"n_mels": 8, "n_frames": 8},
            "classifier JSONL byte limit",
        ),
        (
            train_gain_predictor._load_dataset_from_file,
            {"n_channels": 1},
            "gain/pan JSONL byte limit",
        ),
        (
            train_mix_console._load_dataset_from_file,
            {"n_channels": 1},
            "mix-console JSONL byte limit",
        ),
    ],
)
def test_training_parse_sink_enforces_configured_dataset_cap(
    tmp_path,
    loader,
    kwargs,
    message,
):
    dataset = tmp_path / "replaced.jsonl"
    dataset.write_bytes(b"{}\n")
    # Model a file that grew or was replaced after service-level resolution.
    dataset.write_bytes(b"x" * 9)

    with pytest.raises(ValueError, match=message):
        loader(str(dataset), max_dataset_bytes=8, **kwargs)


@pytest.mark.parametrize(
    ("module", "loader", "kwargs", "message"),
    [
        (
            train_classifier,
            train_classifier._load_dataset_from_file,
            {"n_mels": 8, "n_frames": 8},
            "classifier JSONL byte limit",
        ),
        (
            train_gain_predictor,
            train_gain_predictor._load_dataset_from_file,
            {"n_channels": 1},
            "gain/pan JSONL byte limit",
        ),
        (
            train_mix_console,
            train_mix_console._load_dataset_from_file,
            {"n_channels": 1},
            "mix-console JSONL byte limit",
        ),
    ],
)
def test_training_parse_sink_counts_bytes_appended_after_open(
    tmp_path,
    monkeypatch,
    module,
    loader,
    kwargs,
    message,
):
    dataset = tmp_path / "growing.jsonl"
    dataset.write_bytes(b"{}\n")
    real_open = open
    real_open_regular = module.open_regular_binary_file

    class GrowingFile:
        def __init__(self, raw):
            self.raw = raw
            self.grown = False

        def fileno(self):
            return self.raw.fileno()

        def readline(self, limit):
            if not self.grown:
                with real_open(dataset, "ab") as sink:
                    sink.write(b"xxxxxx")
                self.grown = True
            return self.raw.readline(limit)

    @contextmanager
    def growing_open(path, *, error_message):
        with real_open_regular(path, error_message=error_message) as raw:
            yield GrowingFile(raw)

    monkeypatch.setattr(module, "open_regular_binary_file", growing_open)

    with pytest.raises(ValueError, match=message):
        loader(str(dataset), max_dataset_bytes=8, **kwargs)


@pytest.mark.parametrize(
    ("module", "loader", "kwargs", "message"),
    [
        (
            train_classifier,
            train_classifier._load_dataset_from_file,
            {"n_mels": 8, "n_frames": 8},
            "regular file",
        ),
        (
            train_gain_predictor,
            train_gain_predictor._load_dataset_from_file,
            {"n_channels": 1},
            "regular file",
        ),
        (
            train_mix_console,
            train_mix_console._load_dataset_from_file,
            {"n_channels": 1},
            "regular file",
        ),
    ],
)
def test_jsonl_loaders_reject_non_regular_opened_file(
    tmp_path,
    monkeypatch,
    module,
    loader,
    kwargs,
    message,
):
    dataset = tmp_path / "dataset.jsonl"
    os.mkfifo(dataset)

    with pytest.raises(ValueError, match=message):
        loader(str(dataset), **kwargs)


def test_classifier_jsonl_rejects_oversized_line_before_json_parse(
    tmp_path,
    monkeypatch,
):
    dataset = tmp_path / "classifier.jsonl"
    dataset.write_bytes(b"x" * 9)
    monkeypatch.setattr(train_classifier, "MAX_CLASSIFIER_JSONL_LINE_BYTES", 8)
    parse = MagicMock()
    monkeypatch.setattr(train_classifier.json, "loads", parse)

    with pytest.raises(ValueError, match="line limit"):
        train_classifier._load_dataset_from_file(
            str(dataset),
            n_mels=8,
            n_frames=8,
        )
    parse.assert_not_called()


def test_csv_conversion_enforces_sample_and_output_budget(tmp_path, monkeypatch):
    service = AgentTrainingService(
        {
            "max_dataset_bytes": 1024 * 1024,
            "artifacts_dir": str(tmp_path / "models"),
            "dataset_dir": str(tmp_path / "datasets"),
            "state_file": str(tmp_path / "training_state.json"),
            "study": {"output_dir": str(tmp_path / "knowledge")},
        },
        repo_root=str(tmp_path),
    )
    dataset = tmp_path / "classifier.csv"
    dataset.write_text("spectrogram,label\n[0.1],1\n[0.2],2\n", encoding="utf-8")
    monkeypatch.setattr(
        "backend.ml.agent_training_service.MAX_CSV_CONVERTED_ROWS",
        1,
    )

    with pytest.raises(RuntimeError, match="sample limit"):
        service._convert_csv_to_jsonl("channel_classifier", dataset)
    assert not (tmp_path / "classifier.channel_classifier.jsonl").exists()


def test_csv_conversion_enforces_per_target_byte_cap(tmp_path):
    service = AgentTrainingService(
        {
            "max_dataset_bytes": 1024 * 1024,
            "artifacts_dir": str(tmp_path / "models"),
            "dataset_dir": str(tmp_path / "datasets"),
            "state_file": str(tmp_path / "training_state.json"),
            "study": {"output_dir": str(tmp_path / "knowledge")},
        },
        repo_root=str(tmp_path),
    )
    dataset = tmp_path / "classifier.csv"
    dataset.write_text("spectrogram,label\n[0.1],1\n", encoding="utf-8")

    with pytest.raises(RuntimeError, match="max_dataset_bytes"):
        service._convert_csv_to_jsonl(
            "channel_classifier",
            dataset,
            max_dataset_bytes=8,
        )
    assert not (tmp_path / "classifier.channel_classifier.jsonl").exists()


@pytest.mark.asyncio
async def test_dataset_resolver_propagates_target_cap_into_csv_conversion(tmp_path):
    service = AgentTrainingService(
        {
            "max_dataset_bytes": 1024 * 1024,
            "artifacts_dir": str(tmp_path / "models"),
            "dataset_dir": str(tmp_path / "datasets"),
            "state_file": str(tmp_path / "training_state.json"),
            "study": {"output_dir": str(tmp_path / "knowledge")},
        },
        repo_root=str(tmp_path),
    )
    dataset = tmp_path / "classifier.csv"
    dataset.write_text("spectrogram,label\né,1\n", encoding="utf-8")
    source_bytes = dataset.stat().st_size

    with pytest.raises(RuntimeError, match="converted JSONL"):
        await service._resolve_dataset(
            "channel_classifier",
            {
                "dataset_file": str(dataset),
                "max_dataset_bytes": source_bytes,
            },
        )
    assert not (tmp_path / "classifier.channel_classifier.jsonl").exists()


def test_npz_loader_uses_immutable_validated_snapshot(tmp_path, monkeypatch):
    from backend.ml import npz_archive_safety

    dataset = tmp_path / "classifier.npz"
    np.savez(
        dataset,
        features=np.zeros((1, 8, 8), dtype=np.float32),
        labels=np.asarray([0], dtype=np.int64),
    )
    original_validate = npz_archive_safety._validate_npz_stream

    def validate_then_mutate(*args, **kwargs):
        original_validate(*args, **kwargs)
        dataset.write_bytes(b"mutated-after-validation")

    monkeypatch.setattr(
        npz_archive_safety,
        "_validate_npz_stream",
        validate_then_mutate,
    )

    records = train_classifier._load_dataset_from_file(
        str(dataset),
        n_mels=8,
        n_frames=8,
    )
    assert len(records) == 1


@pytest.mark.parametrize(
    "value",
    [float("nan"), float("inf"), float("-inf")],
)
def test_safety_gate_rejects_non_finite_policy_values(value):
    with pytest.raises(ValueError, match="finite"):
        SafetyGateConfig(max_gain_change_db=value)


def test_gain_pan_jsonl_rejects_oversized_line_before_json_parse(tmp_path, monkeypatch):
    dataset = tmp_path / "gain.jsonl"
    dataset.write_bytes(b"x" * 9)
    monkeypatch.setattr(train_gain_predictor, "MAX_GAIN_PAN_JSONL_LINE_BYTES", 8)
    parse = MagicMock()
    monkeypatch.setattr(train_gain_predictor.json, "loads", parse)

    with pytest.raises(ValueError, match="line limit"):
        train_gain_predictor._load_dataset_from_file(str(dataset), n_channels=1)
    parse.assert_not_called()


def test_mix_console_jsonl_rejects_oversized_line_before_json_parse(tmp_path, monkeypatch):
    dataset = tmp_path / "mix.jsonl"
    dataset.write_bytes(b"x" * 9)
    monkeypatch.setattr(train_mix_console, "MAX_MIX_CONSOLE_JSONL_LINE_BYTES", 8)
    parse = MagicMock()
    monkeypatch.setattr(train_mix_console.json, "loads", parse)

    with pytest.raises(ValueError, match="line limit"):
        train_mix_console._load_dataset_from_file(str(dataset), n_channels=1)
    parse.assert_not_called()


def test_sandbox_dsp_failure_does_not_publish_applied_render(tmp_path, monkeypatch):
    renderer = SandboxRenderer()
    monkeypatch.setattr(
        renderer,
        "_high_pass",
        MagicMock(side_effect=RuntimeError("DSP unavailable")),
    )
    candidate = MixCandidate(
        candidate_id="failed-dsp",
        label="failed",
        render_filename="failed.wav",
        actions=[
            CandidateAction(
                action_type="high_pass_filter",
                target="stem",
                parameters={"frequency_hz": 80.0},
            )
        ],
    )

    with pytest.raises(RuntimeError, match="DSP unavailable"):
        renderer.render_candidate(
            candidate,
            {"stem": np.zeros((128, 1), dtype=np.float32)},
            {"stem": "unknown"},
            48_000,
            tmp_path,
            loudness_match=False,
        )
    assert not (tmp_path / "failed.wav").exists()


def _replay_proposal(proposal_id: str, *, dry_run_only: bool) -> ReplayDecisionProposal:
    return ReplayDecisionProposal(
        proposal_id=proposal_id,
        family="gain_adjustment",
        source_system="security_regression",
        action_type="set_gain",
        target={"channel": 1},
        current_state={"gain_db": 0.0},
        requested_state={"gain_db": 1.0},
        confidence=0.9,
        safety={"max_step_db": 2.0},
        dry_run_only=dry_run_only,
        auto_apply_blocked=True,
        replay_correlation_id=f"corr::{proposal_id}",
    )


def test_replay_rejects_mixed_batch_before_any_state_transition():
    initial = ReplayExecutionState(gains={"1": 0.0}).to_dict()
    result = simulate_replay_decisions(
        [
            _replay_proposal("safe", dry_run_only=True),
            _replay_proposal("unsafe", dry_run_only=False),
        ],
        initial_state=initial,
    )

    assert result.proposals_executed == []
    assert all(event.applied is False for event in result.events)
    assert result.rollback_state["gains"] == {"1": 0.0}


@pytest.mark.parametrize("level", ["loud", True, float("nan"), float("inf"), 25.0])
def test_virtual_mixer_rejects_malformed_channel_level(level):
    mixer = VirtualWingMixer()
    before = mixer.input_channels[1].input_signal
    with pytest.raises(ValueError):
        mixer.set_input_signal(1, level)
    assert mixer.input_channels[1].input_signal == before


@pytest.mark.asyncio
async def test_virtual_mixer_osc_listener_binds_loopback(monkeypatch):
    captured = {}
    fake_server = MagicMock()
    fake_server.create_serve_endpoint = AsyncMock(return_value=(object(), object()))

    def build_server(address, dispatcher, loop):
        captured["address"] = address
        return fake_server

    monkeypatch.setattr(
        "virtual_mixer.virtual_mixer.AsyncIOOSCUDPServer",
        build_server,
    )
    mixer = VirtualWingMixer(osc_port=2222)

    await mixer.start()

    assert captured["address"] == ("127.0.0.1", 2222)


def test_gpt_bridge_rejects_oversized_upstream_before_json_decode(monkeypatch):
    class Response:
        headers = {}

        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return False

        def read(self, limit):
            assert limit == bridge.MAX_UPSTREAM_RESPONSE_BYTES + 1
            return b"x" * limit

    monkeypatch.setattr(bridge.urllib.request, "urlopen", lambda *_args, **_kwargs: Response())
    client = bridge.PaperclipClient("http://127.0.0.1:3100")

    with pytest.raises(ValueError, match="byte limit"):
        client.get("/api/health")


def test_gpt_bridge_report_read_is_capped(tmp_path):
    reports = tmp_path / ".paperclip" / "reports"
    reports.mkdir(parents=True)
    report = reports / "watchdog.md"
    report.write_bytes(b"a" * (bridge.REPORT_EXCERPT_CHARS + 100))

    result = bridge.latest_report(tmp_path)

    assert len(result["excerpt"]) == bridge.REPORT_EXCERPT_CHARS
    assert result["truncated"] is True
