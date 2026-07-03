import ast
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]

TIMEOUT_REQUIRED_FILES = [
    "scripts/discover_training_datasets.py",
    "automixer/production_mix_v1/ayaic_offline_pipeline.py",
    "src/agent_ops/trainer_agent.py",
    "tools/automixer_operator/operator.py",
    "tools/channel_triggered_soundcheck_mix.py",
    "tools/chat_only_shared_mix.py",
    "tools/latency/audio_e2e.py",
    "tools/offline_agent_mix.py",
    "tools/online_soundcheck_mix.py",
    "tools/repa_song_soundcheck_pipeline.py",
    "tools/repa_song_uncommitted_smart_dsp_audition_pipeline.py",
    "tests/test_memory_pipeline.py",
    "tests/test_mix_agent/test_offline_pipeline.py",
    "tests/test_offline_correction_runner.py",
    "tests/test_offline_test_chain.py",
    "tests/test_spectral_ceiling_eq.py",
    "tests/test_wing_readonly_preflight_handoff_packet.py",
]


def test_selected_tool_subprocess_calls_have_timeouts():
    missing: list[str] = []
    for relative_path in TIMEOUT_REQUIRED_FILES:
        path = ROOT / relative_path
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            func = node.func
            if not (
                isinstance(func, ast.Attribute)
                and isinstance(func.value, ast.Name)
                and func.value.id == "subprocess"
                and func.attr in {"run", "check_output"}
            ):
                continue
            keywords = {kw.arg for kw in node.keywords if kw.arg}
            if "timeout" not in keywords:
                missing.append(f"{relative_path}:{node.lineno}: subprocess.{func.attr}")

    assert missing == []
