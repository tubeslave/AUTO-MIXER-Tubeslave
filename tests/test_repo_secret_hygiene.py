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
