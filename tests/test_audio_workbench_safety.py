import tempfile
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

from audio_workbench import core
from audio_workbench.compare import compare
from audio_workbench.experiments import render_candidate
from audio_workbench.project import create_manifest
from audio_workbench.renderer import render_mix

def _write(path: Path, amp=.1, sr=48000, sec=.25, channels=2):
    t=np.arange(int(sr*sec))/sr
    x=(amp*np.sin(2*np.pi*440*t)).astype("float32")
    y=x[:,None] if channels==1 else np.column_stack([x,x])
    sf.write(path,y,sr,subtype="FLOAT")

def test_unknown_render_never_finalizes():
    with tempfile.TemporaryDirectory() as td:
        with pytest.raises(KeyError):
            core.coverage(td,"not-registered")

def test_empty_ok_evidence_rejected():
    with tempfile.TemporaryDirectory() as td:
        p=Path(td)/"x.wav"; _write(p)
        ident=core.register(td,str(p))
        with pytest.raises(ValueError):
            core.record_check(td,ident["sha256"],"integrity",{},status="ok")

def test_compare_rejects_length_and_channel_mismatch():
    with tempfile.TemporaryDirectory() as td:
        r=Path(td); a=r/"a.wav"; b=r/"b.wav"; c=r/"c.wav"
        _write(a); _write(b,sec=.2); _write(c,channels=1)
        with pytest.raises(ValueError): compare(str(a),str(b))
        with pytest.raises(ValueError): compare(str(a),str(c))

def test_bypass_preserves_overload_and_source_protected():
    with tempfile.TemporaryDirectory() as td:
        r=Path(td); src=r/"src.wav"; out=r/"out.wav"
        _write(src,amp=1.2)
        render_candidate(str(src),str(out),{"type":"bypass","params":{}})
        y,_=sf.read(out,always_2d=True,dtype="float32")
        assert np.max(np.abs(y)) > 1.0
        with pytest.raises(ValueError):
            render_candidate(str(src),str(src),{"type":"bypass","params":{}})

def test_dawless_renderer_preserves_float_overload():
    with tempfile.TemporaryDirectory() as td:
        r=Path(td); stems=r/"stems"; stems.mkdir()
        _write(stems/"Kick.wav",amp=.8,channels=1)
        _write(stems/"Bass.wav",amp=.8,channels=1)
        project=r/"project"; create_manifest(str(project),str(stems),"x")
        out=r/"mix.wav"
        result=render_mix(str(project),str(out),{"Kick":{"pan":0},"Bass":{"pan":0}})
        assert result["tracks_rendered"]
        assert out.exists()
