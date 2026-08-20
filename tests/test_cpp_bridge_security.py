import os
import stat
import struct
from types import SimpleNamespace

from auto_fader_v2.bridge.cpp_bridge import CppBridge


class FakeProducer:
    def poll(self):
        return None


def _header(*, channels=2, capacity=4, write_idx=1, read_idx=0, producer_alive=1):
    payload = bytearray(CppBridge.HEADER_SIZE + capacity * CppBridge.METRICS_SIZE)
    struct.pack_into(
        "IIIIIIBB",
        payload,
        0,
        CppBridge.MAGIC_NUMBER,
        CppBridge.VERSION,
        channels,
        capacity,
        write_idx,
        read_idx,
        producer_alive,
        0,
    )
    return payload


def test_default_shared_memory_name_is_unpredictable_per_bridge():
    first = CppBridge()
    second = CppBridge()

    assert first.shm_name != second.shm_name
    assert first.shm_name.startswith(f"auto_fader_metrics_{os.getpid()}_")


def test_header_rejects_unbounded_or_inconsistent_indices():
    bridge = CppBridge("security-test")
    bridge.cpp_process = FakeProducer()

    bridge.shm_mmap = _header(capacity=4, write_idx=4)
    assert bridge._validate_header() is False

    bridge.shm_mmap = _header(capacity=CppBridge.MAX_BUFFER_CAPACITY + 1)
    assert bridge._validate_header() is False


def test_header_requires_live_producer_flag_and_complete_mapping():
    bridge = CppBridge("security-test")
    bridge.cpp_process = FakeProducer()

    bridge.shm_mmap = _header(producer_alive=0)
    assert bridge._validate_header() is False

    bridge.shm_mmap = _header()[: CppBridge.HEADER_SIZE]
    assert bridge._validate_header() is False


def test_shared_memory_stat_requires_same_owner_bounded_size_and_private_writes():
    bridge = CppBridge("security-test")
    valid = SimpleNamespace(
        st_uid=os.geteuid(),
        st_size=CppBridge.HEADER_SIZE + CppBridge.METRICS_SIZE,
        st_mode=stat.S_IFREG | 0o600,
    )

    assert bridge._validate_shm_stat(valid) is True
    assert bridge._validate_shm_stat(SimpleNamespace(**{**vars(valid), "st_uid": os.geteuid() + 1})) is False
    assert bridge._validate_shm_stat(SimpleNamespace(**{**vars(valid), "st_mode": stat.S_IFREG | 0o622})) is False
    assert bridge._validate_shm_stat(SimpleNamespace(**{**vars(valid), "st_size": CppBridge.MAX_SHM_SIZE + 1})) is False
