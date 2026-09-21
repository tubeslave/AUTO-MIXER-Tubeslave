import tempfile
from pathlib import Path
import numpy as np
import soundfile as sf

from audio_workbench.experiments import run_experiment, choose_candidate
from audio_workbench.transients import analyze_transients

def _audio(path: Path,sr=48000):
    t=np.arange(sr)/sr
    x=.1*np.sin(2*np.pi*440*t)
    x[::4800]+=.4
    sf.write(path,np.column_stack([x,x]).astype("float32"),sr,subtype="FLOAT")

def test_experiment_has_bypass_and_select():
    with tempfile.TemporaryDirectory() as td:
        root=Path(td); src=root/"src.wav"; _audio(src)
        m=run_experiment(str(root),str(src),"test gain",[
          {"type":"gain","params":{"db":-3}},
          {"type":"eq_bell","params":{"freq_hz":440,"q":1,"db":-2}},
        ])
        assert len(m["candidates"])==3
        assert m["candidates"][0]["action"]["type"]=="bypass"
        chosen=choose_candidate(str(root),m["experiment_id"],1,"controlled test")
        assert chosen["selected"]["candidate"]==1
        tr=analyze_transients(str(src))
        assert tr["event_count"]>=1
