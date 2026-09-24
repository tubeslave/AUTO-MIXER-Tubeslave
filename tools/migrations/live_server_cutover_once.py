"""One-shot, baseline-checked server/service migration; removed after proof.

No hardware/network calls. GitHub Actions runs the established offline suite
before committing the resulting source changes on the dedicated run11 branch.
"""
from __future__ import annotations

import ast
from datetime import datetime, timezone
from pathlib import Path
import subprocess
import sys
import xml.etree.ElementTree as ET

ROOT = Path(__file__).resolve().parents[2]
EXPECTED = {
    "backend/server.py": "4d148649d0a861e01f10bc4707458b063945d436",
    "backend/handlers/soundcheck_handlers.py": "11bd3f2d1a97b6916c8d228e650027447e33e737",
    "backend/live_runtime/service.py": "bd3dfbf840e35563e9cc36322053dd6a95c9b80f",
    "backend/live_runtime/service_core.py": "d24032d23c3cac41d98757f54041f50d568540ce",
    "tests/test_soundcheck_handlers.py": "52c3c86d24124529f4b9b225241c24382b4ad260",
    "tests/test_live_soundcheck_handler_composition.py": "c1777c16a5103224be21870f4a67ef385e5cec85",
    "tests/test_live_runtime_import_boundary.py": "9c37dbe96bf5a3e689b13c25d0830b11251bd109",
    ".github/workflows/stem_offline_test.yml": "ed3f56c463f2b50115b350b8d3f351b1619fe6c3",
}


def replace_once(text, old, new):
    count = text.count(old)
    if count != 1:
        raise RuntimeError(f"Expected one migration anchor, found {count}: {old[:100]!r}")
    return text.replace(old, new, 1)


def replace_method(text, name, replacement):
    tree = ast.parse(text)
    matches = [node for node in ast.walk(tree) if isinstance(node, ast.FunctionDef) and node.name == name]
    if len(matches) != 1:
        raise RuntimeError(f"Expected exactly one method {name}")
    node = matches[0]
    lines = text.splitlines(keepends=True)
    return "".join(lines[:node.lineno - 1]) + replacement + "".join(lines[node.end_lineno:])


def apply():
    subprocess.run(["git", "merge-base", "--is-ancestor", "69da8b91f5a68894d72940047c579a257e43b468", "HEAD"], cwd=ROOT, check=True)
    for path, expected in EXPECTED.items():
        actual = subprocess.check_output(["git", "hash-object", path], cwd=ROOT, text=True).strip()
        if actual != expected:
            raise RuntimeError(f"Baseline changed: {path}: {actual} != {expected}")
    updates = {}
    server = (ROOT / "backend/server.py").read_text(encoding="utf-8")
    server = replace_once(server, "from auto_soundcheck_engine import AutoSoundcheckEngine\n", "from live_runtime.service import LiveSoundcheckService\n")
    server = replace_once(server,
        "        # Auto soundcheck engine (headless auto-mixing)\n        self.auto_soundcheck_engine: Optional[AutoSoundcheckEngine] = None\n",
        "        # Soundcheck lifecycle belongs to live_runtime, not a server-owned engine.\n        self._live_soundcheck_service: Optional[LiveSoundcheckService] = None\n")
    old_stop = '''        # Stop auto soundcheck engine
        if self.auto_soundcheck_engine:
            try:
                self.auto_soundcheck_engine.stop()
                logger.info("Auto soundcheck engine stopped")
            except Exception as e:
                logger.error(f"Error stopping auto soundcheck engine: {e}")
            self.auto_soundcheck_engine = None
            self.auto_soundcheck_running = False
            self.auto_soundcheck_observe_only = False
'''
    server = replace_once(server, old_stop, "")
    server = replace_once(server, "        # Stop audio capture\n", '''        # Stop the live service before unrelated server-owned audio/mixer handles.
        # Keep the service reference for status/audit, even if teardown fails.
        service = getattr(self, "_live_soundcheck_service", None)
        if service is not None:
            self._safe_cleanup_call(
                service.stop,
                "Error stopping live soundcheck service",
                "Live soundcheck service stopped",
            )
            self.auto_soundcheck_running = False
            self.auto_soundcheck_observe_only = False

        # Stop audio capture
''')
    server = replace_method(server, "_sync_runtime_from_auto_soundcheck", '''    def _sync_runtime_from_live_soundcheck(self) -> Dict[str, Any]:
        """Refresh display status only; never export live hardware ownership."""
        service = getattr(self, "_live_soundcheck_service", None)
        if service is None:
            return {}
        status = service.get_status()
        self.auto_soundcheck_running = bool(service.is_active())
        self.auto_soundcheck_observe_only = (
            self.auto_soundcheck_running
            and status.get("mode") in {"observe", "propose", "freeze"}
        )
        return status
''')
    server = server.replace("self._sync_runtime_from_auto_soundcheck()", "self._sync_runtime_from_live_soundcheck()")
    server = replace_once(server, '''        if self.auto_soundcheck_engine:
            selected = self.auto_soundcheck_engine.get_status().get("selected_channels", [])
            if selected:
                return [int(ch) for ch in selected]
''', '''        selected = self._sync_runtime_from_live_soundcheck().get("selected_channels", [])
        if selected:
            return [int(ch) for ch in selected]
''')
    server = replace_once(server, '''        self._sync_runtime_from_live_soundcheck()
        selected_channels = self._selected_agent_channels(channels)
''', '''        live_status = self._sync_runtime_from_live_soundcheck()
        selected_channels = self._selected_agent_channels(channels)
''')
    server = replace_once(server, '''        engine_channels = {}
        if self.auto_soundcheck_engine:
            try:
                engine_channels = self.auto_soundcheck_engine.get_status().get("channels", {})
            except Exception:
                engine_channels = {}
''', '''        # Compatibility observations only; no mixer/audio handles leave the service.
        engine_channels = live_status.get("channels", {}) or {}
''')
    if "auto_soundcheck_engine" in server or "AutoSoundcheckEngine" in server:
        raise RuntimeError("Unsevered server engine reference")
    updates["backend/server.py"] = server

    path = "backend/handlers/soundcheck_handlers.py"
    handler = (ROOT / path).read_text(encoding="utf-8")
    handler = handler.replace("AutoSoundcheckEngine directly.", "soundcheck engine directly.")
    handler = replace_once(handler, "            engine = service.start(\n", "            service.start(\n")
    handler = replace_once(handler, '''        # Temporary compatibility alias for legacy server cleanup/sync code.
        # New live handlers never use this alias for lifecycle decisions.
        server.auto_soundcheck_engine = engine
''', "")
    handler = replace_once(handler, "        server.auto_soundcheck_engine = None\n", "")
    updates[path] = handler

    helper = '''    def _selected_channel_ids(self) -> list[int]:
        """Copy the explicit input selection; Main-return slots are not inferred."""
        request = self._request
        if request is None:
            return []
        if request.selected_channels:
            return list(request.selected_channels)
        if request.capture_bridge is not None:
            return sorted(request.capture_bridge.roles)
        return []

'''
    for path in ("backend/live_runtime/service_core.py", "backend/live_runtime/service.py"):
        text = (ROOT / path).read_text(encoding="utf-8")
        if path.endswith("service_core.py"):
            text = replace_once(text, "    def get_status(self) -> dict[str, Any]:\n", helper + "    def get_status(self) -> dict[str, Any]:\n")
        text = replace_once(text, '                "state": "idle",\n', '                "state": "idle",\n                "selected_channels": [],\n')
        text = replace_once(text, "        status = dict(raw or {})\n", '        status = dict(raw or {})\n        status["selected_channels"] = self._selected_channel_ids()\n')
        updates[path] = text

    path = "tests/test_soundcheck_handlers.py"
    text = (ROOT / path).read_text(encoding="utf-8")
    text = replace_once(text, "        self.auto_soundcheck_engine = None  # temporary legacy compatibility alias\n", "")
    text = replace_once(text, "    assert server.auto_soundcheck_engine is service.engine\n", '    assert not hasattr(server, "auto_soundcheck_engine")\n')
    text = replace_once(text, "    assert server.auto_soundcheck_engine is None\n", '    assert not hasattr(server, "auto_soundcheck_engine")\n')
    updates[path] = text
    path = "tests/test_live_soundcheck_handler_composition.py"
    updates[path] = replace_once((ROOT / path).read_text(encoding="utf-8"), "        self.auto_soundcheck_engine = None\n", "")

    path = "tests/test_live_runtime_import_boundary.py"
    guard = '''

def test_server_and_soundcheck_handler_cannot_reintroduce_engine_ownership():
    forbidden = {"auto_soundcheck_engine", "AutoSoundcheckEngine", "active_engine", "_sync_runtime_from_auto_soundcheck"}
    for relative in ("server.py", "handlers/soundcheck_handlers.py"):
        path = BACKEND_ROOT / relative
        assert "auto_soundcheck_engine" not in _legacy_imports_in_source(path)
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        found = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Name) and node.id in forbidden:
                found.add(node.id)
            elif isinstance(node, ast.Attribute) and node.attr in forbidden:
                found.add(node.attr)
            elif isinstance(node, ast.Constant) and isinstance(node.value, str) and node.value in forbidden:
                found.add(node.value)
        assert not found, f"{relative}: forbidden live engine references {sorted(found)}"
'''
    updates[path] = (ROOT / path).read_text(encoding="utf-8") + guard
    path = ".github/workflows/stem_offline_test.yml"
    text = (ROOT / path).read_text(encoding="utf-8")
    text = replace_once(text, "      - live-soundcheck-renovation\n", "      - live-soundcheck-renovation\n      - live-soundcheck-renovation-*\n")
    anchor = "            tests/test_live_runtime_import_boundary.py " + chr(92) + "\n"
    text = replace_once(text, anchor, anchor + "            tests/test_live_server_service_boundary.py " + chr(92) + "\n")
    updates[path] = text
    for path, text in updates.items():
        if path.endswith(".py"):
            ast.parse(text, filename=path)
    for path, text in updates.items():
        (ROOT / path).write_text(text, encoding="utf-8")
    print("Applied bounded migration to:", *updates, sep="\n")


def evidence():
    xml = ET.parse(ROOT / "artifacts/live-server-cutover/pytest.xml").getroot()
    suites = [xml] if xml.tag == "testsuite" else list(xml.iter("testsuite"))
    totals = {key: sum(int(s.get(key, "0")) for s in suites) for key in ("tests", "failures", "errors", "skipped")}
    if totals["failures"] or totals["errors"] or totals["tests"] < 180:
        raise RuntimeError(f"Incomplete replacement proof: {totals}")
    now = datetime.now(timezone.utc).isoformat()
    path = ROOT / "Docs/adr/live-server-service-cutover-v1.md"
    text = path.read_text(encoding="utf-8").replace(
        "Status: implementation pending; no replacement-test or HIL claim yet",
        "Status: software cutover tested; physical HIL still outstanding")
    text += f"\n## Executed software evidence\n\nUTC: {now}\n\nThe branch-local one-shot migration applied only baseline-checked source edits. The complete focused test file list from stem_offline_test.yml, including the new server boundary tests, ran with OSC_DISABLED=true and real-model loading disabled. JUnit totals: {totals}. The offline stem loop and report OSC-disabled assertions also passed before this evidence step. No console was contacted.\n\nServer and handlers now retain only the service reference. Selected input IDs are copied from the explicit request (or configured input roles), and legacy observations remain status-only data. The removed synchronization cannot reassign the server's mixer, audio capture, connection mode or agent mixer. Cleanup delegates to the service before independent server-owned audio teardown and retains the service for audit. Existing unconfigured sessions remain behind the lazy legacy adapter.\n\nARCHIVE candidates remain candidates: no legacy module was moved or deleted and no HIL success is claimed. Full Python-matrix CI is separate from this focused evidence. The temporary migration runner/workflow are removed by the resulting source commit.\n"
    path.write_text(text, encoding="utf-8")
    plan = ROOT / "Docs/adr/repository-renovation-plan-v1.md"
    text = plan.read_text(encoding="utf-8")
    text = text.replace("`AutoCompressorController` and `AutoSoundcheckEngine`", "`AutoCompressorController`; the direct `AutoSoundcheckEngine` server dependency has now been severed")
    text += "\n\n### R3 server/service ownership severing — 2026-09-24\n\n`server.py` and soundcheck handlers no longer import, store, stop or inspect the legacy soundcheck engine. `_live_soundcheck_service` is the sole soundcheck lifecycle reference; status-only synchronization cannot export mixer/audio ownership to independent legacy controllers. Explicit selected-channel IDs are available through canonical status, with reserved Main returns excluded by configured role selection. Source guards and behavioral tests run permanently in focused CI, including renovation work branches. Transport/audio primitives remain KEEP_CORE; server/status composition is ADAPT; engine orchestration remains an ARCHIVE candidate behind its lazy compatibility adapter pending remaining-reference and HIL gates. No legacy module deletion. See `live-server-service-cutover-v1.md` for executed replacement evidence.\n"
    plan.write_text(text, encoding="utf-8")
    print("Evidence recorded:", totals)


if __name__ == "__main__":
    if len(sys.argv) != 2 or sys.argv[1] not in {"apply", "evidence"}:
        raise SystemExit("usage: live_server_cutover_once.py apply|evidence")
    {"apply": apply, "evidence": evidence}[sys.argv[1]]()
