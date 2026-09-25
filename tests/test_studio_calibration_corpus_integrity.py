"""Regression guards for the three reproduced corpus integrity defects."""
from pathlib import Path
from threading import Barrier
from concurrent.futures import ThreadPoolExecutor

import pytest



@pytest.fixture
def remote():
    from audio_workbench.mixing import calibration_corpus
    return calibration_corpus


def args(name):
    return {
        "source_evidence": {"fixture": name}, "subject": {"song": "synthetic"},
        "critic_result": {"target": "vocal_intelligibility", "machine_decision": "rejected",
                          "failures": ["target_not_improved"], "protected_regressions": []},
        "human_review": {"human_review": "accepted", "observations": ["foreground_stability"]},
    }


def test_missing_newline_must_not_corrupt_prefix(remote, tmp_path):
    path = tmp_path / "corpus.jsonl"
    remote.append_calibration_record(path, **args("one"))
    path.write_bytes(path.read_bytes().rstrip(b"\n"))
    before = path.read_bytes()
    try:
        remote.append_calibration_record(path, **args("two"))
    except ValueError:
        assert path.read_bytes() == before, "failed append mutated corpus into invalid JSON"
    else:
        assert remote.replay_calibration_corpus(remote.load_calibration_corpus(path))["records"] == 2


def test_built_record_must_not_alias_input_lists(remote):
    incoming = args("one")
    record = remote.build_calibration_record(**incoming)
    original_id = record["record_id"]
    incoming["critic_result"]["failures"].append("changed_after_build")
    assert record["critic"]["failures"] == ["target_not_improved"], "record aliases caller input"
    assert remote.calibration_record_id(record) == original_id


def test_concurrent_append_must_leave_replayable_history(remote, tmp_path, monkeypatch):
    path = tmp_path / "corpus.jsonl"
    barrier = Barrier(2)
    original = remote.build_calibration_record
    def synchronized_build(**kwargs):
        record = original(**kwargs)
        import time
        time.sleep(0.05)
        return record
    monkeypatch.setattr(remote, "build_calibration_record", synchronized_build)
    def writer(name):
        barrier.wait(timeout=5)
        try:
            return remote.append_calibration_record(path, **args(name))
        except ValueError as exc:
            return str(exc)
    with ThreadPoolExecutor(max_workers=2) as pool:
        list(pool.map(writer, ["one", "two"]))
    summary = remote.replay_calibration_corpus(remote.load_calibration_corpus(path))
    assert summary["records"] in (1, 2)


def test_successful_append_preserves_exact_prefix(remote, tmp_path):
    path = tmp_path / "corpus.jsonl"
    remote.append_calibration_record(path, **args("one"))
    before = path.read_bytes()
    remote.append_calibration_record(path, **args("two"))
    assert path.read_bytes().startswith(before)
    assert remote.replay_calibration_corpus(remote.load_calibration_corpus(path))["records"] == 2


@pytest.mark.parametrize("operation", ["replace", "fsync"])
def test_failed_io_keeps_original_and_cleans_own_files(remote, tmp_path, monkeypatch, operation):
    path = tmp_path / "corpus.jsonl"
    remote.append_calibration_record(path, **args("one"))
    before = path.read_bytes()
    def failure(*values):
        raise OSError("injected failure")
    monkeypatch.setattr(remote.os, operation, failure)
    with pytest.raises(OSError, match="injected"):
        remote.append_calibration_record(path, **args("two"))
    assert path.read_bytes() == before
    assert list(tmp_path.iterdir()) == [path]


def test_existing_lock_is_not_removed(remote, tmp_path):
    path = tmp_path / "corpus.jsonl"
    lock = tmp_path / "corpus.jsonl.lock"
    lock.write_bytes(b"other writer")
    with pytest.raises(ValueError, match="writer lock"):
        remote.append_calibration_record(path, **args("one"))
    assert lock.read_bytes() == b"other writer" and not path.exists()


def test_old_native_json_record_digest_is_unchanged(remote):
    # Independently computed by the fetched, checksum-verified pre-fix implementation.
    record = remote.build_calibration_record(**args("one"))
    assert record["record_id"] == "ce6b27d53920aa1f1e60fbf72814e79371f637d16b20b367014d41be76f0fb1e"
