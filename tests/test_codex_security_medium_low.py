from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest
from unittest.mock import AsyncMock


def test_remote_training_ingress_is_quarantined_without_network():
    from backend.ml.agent_training_service import AgentTrainingService

    service = object.__new__(AgentTrainingService)
    assert service._collect_study_candidates(manifest={"learning_resources": ["https://evil"]}, max_per_type=10) == []
    assert service._select_candidate_dataset_urls("channel_classifier", {"dataset_url": "https://evil"}) == []


@pytest.mark.asyncio
async def test_remote_training_manifest_and_download_are_quarantined():
    from backend.ml.agent_training_service import AgentTrainingService

    service = object.__new__(AgentTrainingService)
    assert await service._fetch_manifest_with_retries("https://evil/manifest.json") is None
    with pytest.raises(RuntimeError, match="quarantined"):
        await service._download_dataset("channel_classifier", "https://evil/data.npz")


def test_audio_metadata_and_nonfinite_samples_fail_before_processing(monkeypatch):
    from ai_mixing_pipeline import audio_utils

    monkeypatch.setattr(
        audio_utils.sf,
        "info",
        lambda _path: SimpleNamespace(
            frames=audio_utils.MAX_AUDIO_SAMPLE_CELLS + 1,
            channels=2,
            samplerate=48_000,
        ),
    )
    monkeypatch.setattr(
        audio_utils.sf,
        "read",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("decoder must not run for oversized metadata")
        ),
    )
    with pytest.raises(ValueError, match="exceeds limit"):
        audio_utils.read_audio("oversized.wav")

    with pytest.raises(ValueError, match="non-finite"):
        audio_utils.measure_audio(np.array([0.0, np.nan], dtype=np.float32), 48_000)


def test_hardware_discovery_requires_local_wing_label_and_source_port():
    from tools import hardware_validation

    local = {"192.168.10.5"}
    assert hardware_validation._valid_wing_discovery_response(
        b"WING Rack\n3.1", ("192.168.10.20", 2222), local
    )
    assert not hardware_validation._valid_wing_discovery_response(
        b"attacker", ("192.168.10.20", 2222), local
    )
    assert not hardware_validation._valid_wing_discovery_response(
        b"WING Rack", ("192.168.11.20", 2222), local
    )


def test_paperclip_bearer_transport_rejects_non_loopback_http():
    from tools.paperclip_watchdog.automixer_paperclip_watchdog import (
        validate_paperclip_transport,
    )
    from tools.project_vector_index.automixer_chromadb_index import (
        validate_paperclip_base_url,
    )

    for validator in (
        lambda: validate_paperclip_transport("http://10.0.0.2:3100", "token"),
        lambda: validate_paperclip_base_url(
            "http://paperclip.example:3100", token_present=True
        ),
    ):
        with pytest.raises(ValueError, match="require HTTPS"):
            validator()

    validate_paperclip_transport("http://127.0.0.1:3100", "token")
    validate_paperclip_base_url("https://paperclip.example", token_present=True)


def test_trainer_rejects_caller_command_and_escaping_experiment_name(tmp_path):
    from src.agent_ops.trainer_agent import TrainerAgent

    agent = TrainerAgent(project_root=tmp_path)
    with pytest.raises(ValueError, match="caller-selected"):
        agent._resolve_command({"command": ["/bin/sh", "-c", "id"]})
    with pytest.raises(ValueError, match="safe single path component"):
        agent._run_experiment({"name": "../../escape", "command": ["python", "x.py"]})


@pytest.mark.asyncio
async def test_training_handler_rejects_truthy_string_force():
    from backend.handlers.training_handlers import register_handlers

    service = SimpleNamespace(start_once=AsyncMock(), get_status=lambda: {})
    server = SimpleNamespace(
        agent_training_service=service,
        send_to_client=AsyncMock(),
    )
    handlers = register_handlers(server)

    await handlers["start_training"]("ws", {"force": "false"})

    service.start_once.assert_not_awaited()
    assert server.send_to_client.await_args.args[1]["type"] == "training_start_failed"


def test_safety_gate_ignores_self_asserted_emergency_metadata():
    from automixer.decision.models import ACTION_GAIN, ActionDecision, ActionPlan
    from automixer.safety.safety_gate import SafetyGate, SafetyGateConfig

    plan = ActionPlan(
        plan_id="untrusted-emergency-plan",
        decisions=[
            ActionDecision(
                id="untrusted-emergency",
                action_type=ACTION_GAIN,
                target="channel:1",
                parameters={"gain_db": 50.0},
                metadata={"emergency": True},
            )
        ]
    )
    result = SafetyGate(SafetyGateConfig(max_gain_change_db=1.0)).evaluate_plan(
        plan, live_mode=True, dry_run=True
    )
    assert result.allowed_count == 0
    assert result.blocked_count == 1
