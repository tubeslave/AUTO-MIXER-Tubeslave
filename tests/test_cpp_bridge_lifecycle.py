import subprocess

from auto_fader_v2.bridge import cpp_bridge
from auto_fader_v2.bridge.cpp_bridge import CoreStatus, CppBridge


class FakeCppProcess:
    stdout = None

    def __init__(self, *, wait_times_out: bool = False):
        self.wait_times_out = wait_times_out
        self.terminated = False
        self.killed = False
        self.wait_calls = 0

    def terminate(self):
        self.terminated = True

    def wait(self, timeout=None):
        self.wait_calls += 1
        if self.wait_times_out:
            raise subprocess.TimeoutExpired(cmd="auto_fader_core", timeout=timeout)
        return 0

    def kill(self):
        self.killed = True


def test_cpp_bridge_cleans_up_process_when_startup_times_out(monkeypatch):
    process = FakeCppProcess()
    bridge = CppBridge()

    monkeypatch.setattr(cpp_bridge.os.path, "exists", lambda path: True)
    monkeypatch.setattr(cpp_bridge.subprocess, "Popen", lambda *args, **kwargs: process)
    monkeypatch.setattr(bridge, "connect_shared_memory", lambda: False)
    times = iter([0.0, 6.0])
    monkeypatch.setattr(cpp_bridge.time, "time", lambda: next(times))

    assert bridge.start_cpp_core() is False
    assert bridge.status is CoreStatus.ERROR
    assert bridge.last_error == "Startup timeout"
    assert bridge.cpp_process is None
    assert process.terminated is True
    assert process.killed is False
    assert process.wait_calls == 1


def test_cpp_bridge_kills_process_when_failed_start_cleanup_times_out(monkeypatch):
    process = FakeCppProcess(wait_times_out=True)
    bridge = CppBridge()

    monkeypatch.setattr(cpp_bridge.os.path, "exists", lambda path: True)
    monkeypatch.setattr(cpp_bridge.subprocess, "Popen", lambda *args, **kwargs: process)
    monkeypatch.setattr(bridge, "connect_shared_memory", lambda: False)
    times = iter([0.0, 6.0])
    monkeypatch.setattr(cpp_bridge.time, "time", lambda: next(times))

    assert bridge.start_cpp_core() is False
    assert bridge.cpp_process is None
    assert process.terminated is True
    assert process.killed is True
