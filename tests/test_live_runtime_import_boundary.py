import os
from pathlib import Path
import subprocess
import sys


REPO_ROOT = Path(__file__).resolve().parents[1]
BACKEND_ROOT = REPO_ROOT / "backend"

# Legacy decision authorities that must never be imported merely by loading the
# canonical live runtime. autofoh_safety is intentionally not listed here: the
# renovation plan currently keeps that safety primitive until live_runtime
# safety reaches parity.
FORBIDDEN_LEGACY_DECISION_ROOTS = (
    "auto_soundcheck_engine",
    "auto_eq",
    "auto_fader",
    "auto_fader_hybrid",
    "auto_fader_v2",
    "auto_compressor",
    "auto_compressor_cf",
    "auto_panner",
    "auto_panner_adaptive",
    "auto_mastering",
    "auto_effects",
    "auto_fx",
    "auto_reverb",
    "autofoh_analysis",
    "autofoh_evaluation",
    "cross_adaptive_eq",
)


def _run_fresh_interpreter(body: str) -> subprocess.CompletedProcess[str]:
    env = dict(os.environ)
    # Make the subprocess prove the same import shape used by the backend
    # runtime, independent of whatever pytest already imported in this process.
    pythonpath = os.pathsep.join((str(BACKEND_ROOT), str(REPO_ROOT)))
    if env.get("PYTHONPATH"):
        pythonpath = os.pathsep.join((pythonpath, env["PYTHONPATH"]))
    env["PYTHONPATH"] = pythonpath
    return subprocess.run(
        [sys.executable, "-c", body],
        cwd=REPO_ROOT,
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )


def _assert_clean_subprocess(body: str) -> None:
    roots = repr(FORBIDDEN_LEGACY_DECISION_ROOTS)
    script = f"""
import sys
{body}
roots = {roots}
loaded = sorted(
    name
    for name in sys.modules
    if any(name == root or name.startswith(root + '.') for root in roots)
)
if loaded:
    raise SystemExit('legacy decision modules imported: ' + ', '.join(loaded))
"""
    result = _run_fresh_interpreter(script)
    assert result.returncode == 0, (
        f"fresh interpreter failed\nstdout:\n{result.stdout}\nstderr:\n{result.stderr}"
    )


def test_importing_canonical_live_service_does_not_load_legacy_decision_modules():
    _assert_clean_subprocess("import live_runtime.service")


def test_constructing_canonical_live_service_is_still_legacy_import_free():
    _assert_clean_subprocess(
        "from live_runtime.service import LiveSoundcheckService\n"
        "service = LiveSoundcheckService()\n"
        "assert service.active_engine is None"
    )
