"""Exercise the real guarded send path without creating network sockets."""
import socket
import threading
import time
import json
import os

import pytest
from pythonosc.osc_message import OscMessage

from wing_client import WingClient
from wing_lab_autowrite import WingLabAutowriteSession


@pytest.fixture(autouse=True)
def no_network(monkeypatch):
    def denied(*args, **kwargs):
        pytest.fail("Hardware/network access is forbidden in lab integration tests")
    monkeypatch.setattr(socket, "socket", denied)
    monkeypatch.setattr(socket, "gethostbyname", denied)


class MemorySocket:
    def __init__(self, transport):
        self.transport = transport

    def sendto(self, packet, peer):
        message = OscMessage(packet)
        self.transport.packets.append((message.address, message.params))
        if message.params:
            self.transport.values[message.address] = message.params[0]
        return len(packet)

    def close(self):
        pass


class MemoryWing(WingClient):
    def __init__(self):
        super().__init__(ip="192.168.1.10")
        self.values = {"/ch/1/fdr": -12.0, "/ch/2/eq/1g": 0.0}
        self.packets = []
        self._osc_throttle_enabled = False

    def connect(self, timeout=5.0):
        self.handshake_info = "WING,192.168.1.10,TEST,wing-rack,SERIAL,3.0.5"
        self.sock = MemorySocket(self)
        self.is_connected = True
        return True

    def query_parameter(self, address, timeout=0.5):
        return {"status": "received", "confirmed_round_trips": 2,
                "peer_ip": self.ip, "peer_port": self.port,
                "value": self.values.get(address)}


def decision(identifier="fader-1", value=-14.0):
    return {"id": identifier, "family": "channel_fader", "channel": 1,
            "accepted": True, "reason": "Measured level above target",
            "confidence": 0.9, "value": value}


@pytest.fixture
def armed(tmp_path):
    transport = MemoryWing()
    session = WingLabAutowriteSession(lambda *_: transport, runtime_root=tmp_path,
                                     enabled=True)
    session.arm(host=transport.ip, console_model="WING Rack", duration_sec=60)
    try:
        yield session, transport
    finally:
        session.emergency_stop(reason="test_cleanup")
        transport.disconnect()


def test_real_transport_batch_and_rollback(armed):
    session, transport = armed
    result = session.apply_decisions([decision()])
    assert result["status"] == "applied_and_verified"
    assert transport.values["/ch/1/fdr"] == -14.0
    assert transport.send("/ch/1/fdr", -20.0) is False
    assert transport.send("/ch/1/mute", 1) is False
    stopped = session.stop()
    assert stopped["rollback"]["failures"] == []
    assert transport.values["/ch/1/fdr"] == -12.0
    assert transport._mutation_boundary is None


def test_real_transport_queued_rollback(armed):
    session, transport = armed
    transport._osc_throttle_enabled = True
    session.apply_decisions([decision()])
    stopped = session.stop()
    assert stopped["rollback"]["failures"] == []
    assert transport.values["/ch/1/fdr"] == -12.0
    assert not transport._osc_pending


@pytest.mark.parametrize("address,values", [
    ("/ch/2/fdr", (-14.0,)), ("/ch/1/fdr", (-13.0,)),
    ("/ch/1/fdr", (-14,)), ("/ch/1/fdr", (True,)),
    ("/ch/1/fdr", (-14.0, 0)), ("/ch/1/fdr", (float("nan"),)),
    ("/ch/1/fdr", (float("inf"),)),
])
def test_exact_typed_grant_only(armed, address, values):
    session, transport = armed
    session._grant = ("/ch/1/fdr", -14.0, False)
    try:
        assert transport.send(address, *values) is False
        assert transport.packets == []
    finally:
        session._grant = None


def test_boundary_present_before_connect(tmp_path):
    t = MemoryWing()
    connect = t.connect
    def checked_connect(timeout):
        assert t._mutation_boundary is not None
        assert not t._mutation_boundary("/ch/1/fdr", (-14.0,))
        return connect(timeout)
    t.connect = checked_connect
    s = WingLabAutowriteSession(lambda *_: t, runtime_root=tmp_path, enabled=True)
    s.arm(host=t.ip, console_model="WING Rack")
    s.stop()


def test_conflicting_owner_not_disconnected_or_removed(tmp_path):
    t = MemoryWing()
    token = t.install_mutation_boundary(lambda *_: False)
    t.connect()
    s = WingLabAutowriteSession(lambda *_: t, runtime_root=tmp_path, enabled=True)
    with pytest.raises(RuntimeError, match="already_active"):
        s.arm(host=t.ip, console_model="WING Rack")
    assert t.is_connected
    assert t.mutation_boundary_matches(token)
    assert s._lock_fd is None
    t.disconnect()


@pytest.mark.parametrize("failure", [RuntimeError("send failed after delivery"), KeyboardInterrupt()])
def test_delivered_send_exception_is_restored(armed, monkeypatch, failure):
    s, t = armed
    send = t.send
    def broken(address, *values):
        result = send(address, *values)
        if values == (-14.0,):
            raise failure
        return result
    monkeypatch.setattr(t, "send", broken)
    with pytest.raises(type(failure)):
        s.apply_decisions([decision()])
    assert t.values["/ch/1/fdr"] == -12.0
    assert s._grant is None
    assert t._mutation_boundary is None
    assert s._lock_fd is None


def test_post_send_wal_failure_restores_registered_preimage(armed, monkeypatch):
    s, t = armed
    append = s._append_event
    def broken(phase, payload):
        if phase == "write_attempted":
            raise OSError("disk failure")
        return append(phase, payload)
    monkeypatch.setattr(s, "_append_event", broken)
    with pytest.raises(OSError, match="disk failure"):
        s.apply_decisions([decision()])
    assert t.values["/ch/1/fdr"] == -12.0
    assert s._grant is None


def test_failed_rollback_preserves_order_and_lock_for_retry(armed, monkeypatch):
    s, t = armed
    s.apply_decisions([decision(), decision("fader-2", -16.0)])
    send = t.send
    monkeypatch.setattr(t, "send", lambda *_: False)
    failed = s.stop()
    assert failed["status"] == "recovery_required"
    assert len(s._applied) == 2
    assert s._lock_fd is not None
    assert s._grant is None
    with pytest.raises(RuntimeError, match="emergency_stopped"):
        s.apply_decisions([decision("forbidden", -18.0)])
    with pytest.raises(RuntimeError, match="already_armed"):
        s.arm(host=t.ip, console_model="WING Rack")
    monkeypatch.setattr(t, "send", send)
    assert s.stop()["rollback"]["failures"] == []
    assert t.values["/ch/1/fdr"] == -12.0
    assert s._lock_fd is None


def test_rollback_does_not_overwrite_human_change(armed):
    s, t = armed
    s.apply_decisions([decision()])
    t.values["/ch/1/fdr"] = -20.0
    assert s.stop()["status"] == "recovery_required"
    assert t.values["/ch/1/fdr"] == -20.0
    t.values["/ch/1/fdr"] = -14.0
    assert s.stop()["rollback"]["failures"] == []


def test_expiry_between_operations_restores_batch(armed, monkeypatch):
    s, t = armed
    append = s._append_event
    def expire(phase, payload):
        result = append(phase, payload)
        if phase == "verified":
            s._expires_at = 0
        return result
    monkeypatch.setattr(s, "_append_event", expire)
    with pytest.raises(RuntimeError, match="session_expired"):
        s.apply_decisions([decision(), decision("fader-2", -16.0)])
    assert t.values["/ch/1/fdr"] == -12.0
    assert not any(values == [-16.0] for _, values in t.packets)


def test_expired_session_can_restore(armed):
    s, t = armed
    s.apply_decisions([decision()])
    s._expires_at = 0
    assert s.stop()["rollback"]["failures"] == []
    assert t.values["/ch/1/fdr"] == -12.0


def test_expiry_revokes_queued_forward_command(armed, monkeypatch):
    s, t = armed
    t._osc_throttle_enabled = True
    t._osc_last_send_time["/ch/1/fdr"] = time.monotonic()
    send = t.send
    def expire_after_enqueue(address, *values):
        result = send(address, *values)
        if values == (-14.0,):
            s._expires_at = 0
        return result
    monkeypatch.setattr(t, "send", expire_after_enqueue)
    with pytest.raises(RuntimeError, match="readback_mismatch"):
        s.apply_decisions([decision()])
    assert t.packets == []
    assert t.values["/ch/1/fdr"] == -12.0


def test_disconnect_failure_retains_lock_until_retry(armed, monkeypatch):
    s, t = armed
    disconnect = t.disconnect
    monkeypatch.setattr(t, "disconnect", lambda: False)
    with pytest.raises(RuntimeError, match="disconnect_unverified"):
        s.stop()
    assert s._lock_fd is not None
    assert s._grant is None
    monkeypatch.setattr(t, "disconnect", disconnect)
    s.emergency_stop()
    assert s._lock_fd is None


def test_factory_failure_releases_lock(tmp_path):
    def fail(*_):
        raise RuntimeError("factory failure")
    s = WingLabAutowriteSession(fail, runtime_root=tmp_path, enabled=True)
    with pytest.raises(RuntimeError, match="factory failure"):
        s.arm(host="192.168.1.10", console_model="WING Rack")
    assert s._lock_fd is None


def test_previous_unfinished_wal_blocks_new_connection(tmp_path):
    root = tmp_path / "wing_lab_autowrite"
    root.mkdir()
    (root / "interrupted.jsonl").write_text('{"phase":"prepared"}\n')
    def forbidden(*_):
        pytest.fail("Factory must not be called with an unresolved journal")
    s = WingLabAutowriteSession(forbidden, runtime_root=tmp_path, enabled=True)
    with pytest.raises(RuntimeError, match="requires_recovery"):
        s.arm(host="192.168.1.10", console_model="WING Rack")
    assert s._lock_fd is None


def test_rearm_resets_emergency_and_audit_chain(armed):
    s, t = armed
    s.apply_decisions([decision()])
    s.stop()
    s.arm(host=t.ip, console_model="WING Rack")
    assert not s.status()["emergency_stopped"]
    assert s.trace()["events"][0]["previous_event_sha256"] == "sha256:" + "0" * 64
    s.apply_decisions([decision()])


def test_emergency_revokes_queue_without_lock_deadlock(armed, monkeypatch):
    s, t = armed
    t._osc_throttle_enabled = True
    t._osc_last_send_time["/ch/1/fdr"] = time.monotonic()
    enqueued = threading.Event()
    send = t.send
    def notifying_send(address, *values):
        result = send(address, *values)
        enqueued.set()
        return result
    monkeypatch.setattr(t, "send", notifying_send)
    errors = []
    def apply():
        try:
            s.apply_decisions([decision()])
        except RuntimeError as exc:
            errors.append(str(exc))
    worker = threading.Thread(target=apply, daemon=True)
    worker.start()
    assert enqueued.wait(1)
    stopper = threading.Thread(target=s.emergency_stop, daemon=True)
    stopper.start()
    worker.join(3)
    stopper.join(3)
    assert not worker.is_alive()
    assert not stopper.is_alive()
    assert errors
    assert t.values["/ch/1/fdr"] == -12.0
    assert s._grant is None


def test_persistent_wal_failure_still_restores_audio(armed, monkeypatch):
    s, t = armed
    append = s._append_event
    broken = False
    def disk_failure(phase, payload):
        nonlocal broken
        broken = broken or phase == "write_attempted"
        if broken:
            raise OSError("disk unavailable")
        return append(phase, payload)
    monkeypatch.setattr(s, "_append_event", disk_failure)
    with pytest.raises(OSError, match="disk unavailable"):
        s.apply_decisions([decision()])
    assert t.values["/ch/1/fdr"] == -12.0
    assert s._lock_fd is not None  # Audit/recovery remains unresolved.
    assert s._grant is None
    monkeypatch.setattr(s, "_append_event", append)
    s.emergency_stop()
    assert s._lock_fd is None


@pytest.mark.parametrize("exit_signal", [False, True])
def test_reentrant_stop_never_reports_applied_after_restore(armed, monkeypatch, exit_signal):
    s, t = armed
    append = s._append_event
    def stop_on_verified(phase, payload):
        result = append(phase, payload)
        if phase == "verified":
            assert s.emergency_stop()["status"] == "stop_requested"
            if exit_signal:
                raise SystemExit(130)
        return result
    monkeypatch.setattr(s, "_append_event", stop_on_verified)
    with pytest.raises(SystemExit if exit_signal else RuntimeError):
        s.apply_decisions([decision()])
    assert t.values["/ch/1/fdr"] == -12.0
    assert s.trace()["events"][-1]["phase"] == "teardown_verified"
    s.arm(host=t.ip, console_model="WING Rack")


def test_expiry_before_batch_restores_previous_corrections(armed):
    s, t = armed
    s.apply_decisions([decision()])
    s._expires_at = 0
    with pytest.raises(RuntimeError, match="session_expired"):
        s.apply_decisions([decision("next", -16.0)])
    assert t.values["/ch/1/fdr"] == -12.0
    assert not t.is_connected


@pytest.mark.parametrize("emergency", [False, True])
def test_terminal_wal_failure_does_not_skip_transport_cleanup(armed, monkeypatch, emergency):
    s, t = armed
    s.apply_decisions([decision()])
    append = s._append_event
    def fail_terminal(phase, payload):
        if phase in {"stopped", "emergency_stopped", "teardown_verified"}:
            raise OSError("terminal log failure")
        return append(phase, payload)
    monkeypatch.setattr(s, "_append_event", fail_terminal)
    with pytest.raises(OSError, match="terminal log failure"):
        s.emergency_stop() if emergency else s.stop()
    assert t.values["/ch/1/fdr"] == -12.0
    assert not t.is_connected
    assert t._mutation_boundary is None
    assert s._lock_fd is not None
    monkeypatch.setattr(s, "_append_event", append)
    s.emergency_stop()
    assert s._lock_fd is None


def test_reentrant_normal_stop_cancels_batch(armed, monkeypatch):
    s, t = armed
    append = s._append_event
    def stop(phase, payload):
        result = append(phase, payload)
        if phase == "verified":
            assert s.stop()["status"] == "stop_requested"
        return result
    monkeypatch.setattr(s, "_append_event", stop)
    with pytest.raises(RuntimeError, match="emergency_stopped"):
        s.apply_decisions([decision(), decision("next", -16.0)])
    assert t.values["/ch/1/fdr"] == -12.0
    assert not any(values == [-16.0] for _, values in t.packets)


def test_short_wal_writes_are_completed(armed, monkeypatch):
    s, _ = armed
    write = os.write
    monkeypatch.setattr(os, "write", lambda fd, data: write(fd, data[:17]))
    s.apply_decisions([decision()])
    s.stop()
    rows = [json.loads(line) for line in s._wal_path.read_text().splitlines()]
    assert rows[-1]["phase"] == "teardown_verified"


def test_corrupted_closed_journal_blocks_rearm(armed):
    s, t = armed
    s.stop()
    text = s._wal_path.read_text().replace('"armed"', '"corrupted"', 1)
    s._wal_path.write_text(text)
    with pytest.raises(RuntimeError, match="requires_recovery"):
        s.arm(host=t.ip, console_model="WING Rack")
    assert s._lock_fd is None


def test_stale_identical_queued_write_cannot_borrow_next_grant(armed):
    s, t = armed
    with t._osc_send_lock:
        t._osc_pending["/ch/1/fdr"] = ((-14.0,), 2, time.monotonic() + .02)
        t._ensure_flusher_running()
        t._osc_flush_wakeup.set()
    s.apply_decisions([decision()])
    assert t.packets == [("/ch/1/fdr", [-14.0])]


def test_queue_timeout_revokes_pending_command_before_teardown(armed, monkeypatch):
    s, t = armed
    t._osc_throttle_enabled = True
    t._osc_last_send_time["/ch/1/fdr"] = time.monotonic()
    flush = t.flush_pending_writes
    calls = 0
    def timeout_once(timeout):
        nonlocal calls
        calls += 1
        return False if calls == 2 else flush(timeout)
    monkeypatch.setattr(t, "flush_pending_writes", timeout_once)
    with pytest.raises(RuntimeError, match="pending_writes_not_drained"):
        s.apply_decisions([decision()])
    assert t.values["/ch/1/fdr"] == -12.0
    assert t.packets == []
    assert not t._osc_pending


def test_repeated_address_disk_failure_restores_original_not_intermediate(armed, monkeypatch):
    s, t = armed
    s.apply_decisions([decision(), decision("second", -16.0)])
    append = s._append_event
    def disk_failed(*args):
        raise OSError("disk unavailable")
    monkeypatch.setattr(s, "_append_event", disk_failed)
    with pytest.raises(OSError):
        s.stop()
    assert t.values["/ch/1/fdr"] == -12.0
    assert len(s._applied) == 2
    assert all(row["physically_restored"] for row in s._applied)
    assert s._lock_fd is not None
    packet_count = len(t.packets)
    monkeypatch.setattr(s, "_append_event", append)
    s.emergency_stop()
    assert len(t.packets) == packet_count
    assert t.values["/ch/1/fdr"] == -12.0
    assert s._lock_fd is None
