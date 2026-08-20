from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]


def test_optional_installer_has_no_mutable_vcs_dependency():
    requirements = (REPO_ROOT / "requirements-ai-decision.txt").read_text(
        encoding="utf-8"
    )
    installer = (REPO_ROOT / "scripts/install_ai_decision_layer.sh").read_text(
        encoding="utf-8"
    )

    executable_lines = [
        line.strip()
        for line in requirements.splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    ]
    assert all("git+" not in line.lower() for line in executable_lines)
    assert all("github.com" not in line.lower() for line in executable_lines)
    assert "git+https://github.com/csteinmetz1/pymixconsole" not in installer


def test_documented_fallback_preserves_offline_operation_without_download():
    documentation = (REPO_ROOT / "Docs/AI_DECISION_CORRECTION_LAYER.md").read_text(
        encoding="utf-8"
    )

    assert "installer never" in documentation.lower()
    assert "mutable" in documentation.lower()
    assert "fallback_virtual_mixer" in documentation
