"""Current-file identity regressions. Synthetic audio only."""
import os

import numpy as np
import pytest
import soundfile as sf

from audio_workbench import core


def seed(root):
    path = root / "render.wav"
    t = np.arange(4800) / 48000
    y = np.column_stack((.1*np.sin(2*np.pi*440*t),
                         .1*np.cos(2*np.pi*440*t))).astype("float32")
    sf.write(path, y, 48000, subtype="FLOAT")
    sha = core.register(str(root), str(path))["sha256"]
    for name in core.CHECKS:
        core.record_check(str(root), sha, name, {"fixture": "synthetic approval"})
    return path, sha, y


def test_unchanged_render_keeps_valid_coverage(tmp_path):
    path, sha, _ = seed(tmp_path)
    result = core.coverage(str(tmp_path), sha)
    assert result["finalizable"]
    assert result["identity"]["actual_sha256"] == sha


@pytest.mark.parametrize("change", ["delete", "corrupt", "replace_same_size"])
def test_invalid_file_fails_closed(tmp_path, change):
    path, sha, y = seed(tmp_path)
    before = path.stat()
    if change == "delete":
        path.unlink()
    elif change == "corrupt":
        path.write_bytes(b"not a wave")
    else:
        sf.write(path, y*.5, 48000, subtype="FLOAT")
        assert path.stat().st_size == before.st_size
        os.utime(path, ns=(before.st_atime_ns, before.st_mtime_ns))
    result = core.coverage(str(tmp_path), sha)
    assert not result["finalizable"]
    assert "render_identity" in result["blockers"]
    assert all(x["status"] == "stale" for x in result["checks"])


def test_changed_file_cannot_receive_new_evidence_for_old_sha(tmp_path):
    path, sha, y = seed(tmp_path)
    sf.write(path, y*.5, 48000, subtype="FLOAT")
    with pytest.raises(ValueError, match="changed render"):
        core.record_check(str(tmp_path), sha, "integrity", {"fixture": "invalid reuse"})


def test_restoring_bytes_does_not_restore_stale_approvals(tmp_path):
    path, sha, y = seed(tmp_path)
    original = path.read_bytes()
    sf.write(path, y*.5, 48000, subtype="FLOAT")
    assert not core.coverage(str(tmp_path), sha)["finalizable"]
    path.write_bytes(original)
    result = core.coverage(str(tmp_path), sha)
    assert result["identity"]["verified"]
    assert not result["finalizable"]
    assert all(x["status"] == "stale" for x in result["checks"])


def test_log_cannot_accept_a_changed_render(tmp_path):
    path, sha, y = seed(tmp_path)
    sf.write(path, y*.5, 48000, subtype="FLOAT")
    with pytest.raises(RuntimeError, match="unverified render"):
        core.log_decision(str(tmp_path), sha, "test", {"type": "bypass"}, "accepted")


def test_analysis_does_not_require_numpy_trapezoid(tmp_path, monkeypatch):
    path, _, _ = seed(tmp_path)
    monkeypatch.delattr(np, "trapezoid", raising=False)
    result = core.analyze(str(path))
    assert np.isfinite(result["crest_db"])
