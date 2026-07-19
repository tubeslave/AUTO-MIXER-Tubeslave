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


def test_env_is_not_tracked_in_git_index() -> None:
    """Regression guard: ignore rules do not untrack files.

    After the 2026-05-16 remediation (0c7752271) removed the tracked root
    `.env`, five merges re-added it to the index, and this guardrail test was
    lost in the same merges, so the regression stayed unnoticed. A tracked
    `.env` must fail CI even when `.gitignore` rules exist.
    """
    tracked = subprocess.run(
        ["git", "ls-files", "--error-unmatch", ".env"],
        cwd=REPO_ROOT,
        text=True,
        capture_output=True,
        check=False,
    )
    assert tracked.returncode != 0, (
        ".env is tracked in the git index; run `git rm --cached .env` and "
        "rotate any secrets it contains per Docs/secrets-rotation.md"
    )
