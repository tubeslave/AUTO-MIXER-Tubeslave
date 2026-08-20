from __future__ import annotations

import json
import asyncio
from pathlib import Path

import pytest

from backend.agents.eq_agent import EQAgent
from src.agent_ops.architect_agent import ArchitectAgent
from src.agent_ops.evaluator_agent import EvaluatorAgent
from src.experiments.muq_eval_director import (
    MuqDirectorConfig,
    MuqEvalDirectorOfflineTest,
)


def test_architect_agent_checks_real_training_entrypoint(tmp_path: Path):
    scripts = tmp_path / "scripts"
    scripts.mkdir()
    (scripts / "train.py").write_text("print('ok')\n", encoding="utf-8")

    result = ArchitectAgent(project_root=tmp_path).run({
        "train_config": {"script_path": "scripts/train.py"},
        "model_dir": "artifacts",
    })

    assert result["status"] == "ok"
    assert result["result"]["checks"]["training_entrypoint_exists"] is True
    assert result["result"]["issues"] == []
    assert result["result"]["live_authority_granted"] is False


def test_architect_agent_reports_missing_entrypoint_instead_of_placeholder_success(
    tmp_path: Path,
):
    result = ArchitectAgent(project_root=tmp_path).run({
        "train_config": {"script_path": "scripts/missing.py"},
        "model_dir": "artifacts",
    })

    assert result["status"] == "incomplete"
    assert "training_entrypoint_exists" in result["result"]["issues"]


def test_evaluator_parses_existing_metrics_artifact(tmp_path: Path):
    artifact = tmp_path / "evaluation.json"
    artifact.write_text(json.dumps({"metrics": {"loss": 0.2}}), encoding="utf-8")

    result = EvaluatorAgent(project_root=tmp_path, runs_dir=".").run({"model_dir": str(tmp_path)})

    assert result["status"] == "ok"
    assert result["result"]["artifact_count"] == 1
    assert result["result"]["artifacts"][0]["metrics"] == {"loss": 0.2}
    assert result["result"]["listening_test_completed"] is False


def test_evaluator_does_not_claim_success_without_evaluation_artifacts(tmp_path: Path):
    result = EvaluatorAgent(project_root=tmp_path, runs_dir=".").run({"model_dir": str(tmp_path)})

    assert result["status"] == "incomplete"
    assert result["result"]["artifact_count"] == 0
    assert result["result"]["summary"] == "No parseable evaluation artifacts found"


def test_evaluator_does_not_claim_success_for_failed_best_run(tmp_path: Path):
    log_path = tmp_path / "train.log"
    log_path.write_text("training failed\n", encoding="utf-8")

    result = EvaluatorAgent(project_root=tmp_path, runs_dir=".").run({
        "best_run": {
            "run_id": "failed-run",
            "run_dir": str(tmp_path),
            "log_path": str(log_path),
            "success": False,
            "metrics": {},
        },
    })

    assert result["status"] == "failed"
    assert result["checks"]["training_success"] is False
    assert Path(result["evaluation_path"]).exists()


def test_eq_agent_requires_real_spectral_measurement_evidence():
    agent = EQAgent(type("Mixer", (), {"state": {}})())
    agent.set_instrument_profiles({3: "lead_vocal"})

    result = asyncio.run(agent.decide({
        "eq_state": {3: {2: {"freq": 1200.0, "gain": 8.0}}},
    }))

    assert result == {
        "status": "unavailable",
        "reason": "spectral_target_measurement_unavailable",
        "corrections": {},
    }


def test_eq_agent_uses_bounded_measured_delta_instead_of_flattening_heuristic():
    agent = EQAgent(type("Mixer", (), {"state": {}})())
    agent.set_instrument_profiles({3: "lead_vocal"})

    result = asyncio.run(agent.decide({
        "eq_state": {3: {2: {"freq": 1200.0, "gain": 8.0}}},
        "spectral_gain_delta_db": {3: {2: -4.5}},
        "spectral_measurement_provenance": {
            "source": "offline_fft_analyzer",
            "measured": True,
        },
    }))

    assert result["status"] == "proposal_only"
    assert result["corrections"] == {3: {2: -3.0}}


def test_muq_genre_mode_fails_closed_without_real_genre_scorer(tmp_path: Path):
    config = MuqDirectorConfig(
        mode="muq_genre",
        output_root=str(tmp_path),
        audio_output_root=str(tmp_path),
    )

    with pytest.raises(ValueError, match="measured genre scorer"):
        MuqEvalDirectorOfflineTest(config, muq_service=object())
