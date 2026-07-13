import os
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_server_import_survives_missing_optional_voice_runtimes():
    script = r'''
import importlib.abc
import sys

class BlockOptionalVoiceModules(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname in {"voice_control_v2", "voice_control_sherpa"}:
            raise ModuleNotFoundError(
                f"blocked optional dependency: {fullname}",
                name=fullname,
            )
        return None

sys.meta_path.insert(0, BlockOptionalVoiceModules())
import server

assert server.VoiceControlV2 is None
assert server.VoiceControlSherpa is None
assert "blocked optional dependency" in server._VOICE_CONTROL_V2_IMPORT_ERROR
assert "blocked optional dependency" in server._VOICE_CONTROL_SHERPA_IMPORT_ERROR
'''
    env = os.environ.copy()
    env["PYTHONPATH"] = str(ROOT / "backend")
    completed = subprocess.run(
        [sys.executable, "-c", script],
        cwd=ROOT,
        env=env,
        capture_output=True,
        text=True,
        timeout=60,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr
