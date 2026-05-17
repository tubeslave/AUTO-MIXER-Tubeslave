from __future__ import annotations

import subprocess
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]


def git_check_ignore(*paths: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["git", "check-ignore", "-v", *paths],
        cwd=REPO_ROOT,
        text=True,
        capture_output=True,
        check=False,
    )


def git_check_ignore_plain(*paths: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["git", "check-ignore", *paths],
        cwd=REPO_ROOT,
        text=True,
        capture_output=True,
        check=False,
    )


def test_root_env_files_are_ignored_but_example_is_trackable() -> None:
    ignored = git_check_ignore(".env", ".env.local")
    assert ignored.returncode == 0, ignored.stderr
    assert ".gitignore" in ignored.stdout
    assert ".env" in ignored.stdout
    assert ".env.local" in ignored.stdout

    example = git_check_ignore_plain(".env.example")
    assert example.returncode == 1, example.stdout or example.stderr


def test_runtime_artifact_outputs_are_ignored() -> None:
    ignored = git_check_ignore(
        ".paperclip/watchdog_state.json",
        "artifacts/tub345_supervised_live/session_summary.json",
        "artifacts/wing_telemetry/session/events.jsonl",
        "artifacts/wing_telemetry_tub339/events.jsonl",
        "production_mix_v1_out/render.wav",
    )
    assert ignored.returncode == 0, ignored.stderr
    assert ".paperclip" in ignored.stdout
    assert "artifacts/tub345_supervised_live" in ignored.stdout
    assert "artifacts/wing_telemetry" in ignored.stdout
    assert "production_mix_v1_out" in ignored.stdout

    tracked_artifact_sample = git_check_ignore_plain("artifacts/runs/training_summary.json")
    assert tracked_artifact_sample.returncode == 1, (
        tracked_artifact_sample.stdout or tracked_artifact_sample.stderr
    )
