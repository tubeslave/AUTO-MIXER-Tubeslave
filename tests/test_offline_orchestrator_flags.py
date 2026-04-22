import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace


def load_offline_agent_mix():
    spec = importlib.util.spec_from_file_location(
        "offline_agent_mix_flags_module",
        Path(__file__).resolve().parents[1] / "tools" / "offline_agent_mix.py",
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_codex_orchestrator_is_not_dry_run_without_explicit_flag():
    mod = load_offline_agent_mix()

    args = SimpleNamespace(
        codex_orchestrator=True,
        codex_orchestrator_dry_run=False,
    )

    assert mod._orchestrator_dry_run_enabled(args) is False


def test_codex_orchestrator_dry_run_respects_explicit_flag():
    mod = load_offline_agent_mix()

    args = SimpleNamespace(
        codex_orchestrator=True,
        codex_orchestrator_dry_run=True,
    )

    assert mod._orchestrator_dry_run_enabled(args) is True
