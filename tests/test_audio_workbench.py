import tempfile
from pathlib import Path

import numpy as np
import soundfile as sf

from audio_workbench import core

def _tone(path: Path, sr=48000, seconds=1.0):
    t = np.arange(int(sr*seconds))/sr
    x = 0.1*np.sin(2*np.pi*440*t)
    y = np.column_stack([x,x]).astype("float32")
    sf.write(path,y,sr,subtype="FLOAT")

def test_register_analyze_invalidate():
    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        wav = root/"tone.wav"
        _tone(wav)
        identity = core.register(str(root),str(wav))
        a = core.analyze(str(wav))
        assert a["channels"] == 2
        assert abs(a["stereo"]["correlation"] - 1.0) < 1e-5
        for name in core.CHECKS:
            core.record_check(str(root),identity["sha256"],name,{"test":True})
        assert core.coverage(str(root),identity["sha256"])["finalizable"]
        core.invalidate(str(root),identity["sha256"],"compression")
        c = core.coverage(str(root),identity["sha256"])
        assert not c["finalizable"]
        assert "dynamics" in c["blockers"]

def test_hash_changes_with_audio():
    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        a = root/"a.wav"
        b = root/"b.wav"
        _tone(a)
        _tone(b)
        data,sr = sf.read(b,always_2d=True)
        data[0,0] += 0.01
        sf.write(b,data,sr,subtype="FLOAT")
        assert core.fingerprint(str(a)).sha256 != core.fingerprint(str(b)).sha256
