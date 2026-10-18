import tempfile
from pathlib import Path
import numpy as np
import soundfile as sf

from audio_workbench.sections import section_features, find_outlier_windows
from audio_workbench.masking import masking_graph
from audio_workbench.compare import compare, blind_labels

def _tone(path: Path, hz: float, amp=.1, sr=48000, sec=2.0):
    t=np.arange(int(sr*sec))/sr
    x=(amp*np.sin(2*np.pi*hz*t)).astype("float32")
    sf.write(path,np.column_stack([x,x]),sr,subtype="FLOAT")

def test_section_and_compare_and_masking():
    with tempfile.TemporaryDirectory() as td:
        root=Path(td)
        a=root/"a.wav"; b=root/"b.wav"; c=root/"c.wav"
        _tone(a,440,.1); _tone(b,440,.2); _tone(c,100,.1)
        rows=section_features(str(a),[{"name":"all","start_s":0,"end_s":2}])
        assert rows[0]["name"]=="all"
        cmp=compare(str(a),str(b),True)
        assert abs(cmp["b_match_gain_db"]+6.0206)<0.1
        g=masking_graph([{"name":"a","path":str(a)},{"name":"b","path":str(b)},{"name":"c","path":str(c)}])
        assert g["edges"][0]["a"] in ("a","b")
        assert len(find_outlier_windows(str(a),.5,.25,2)["windows"])>1
        assert set(blind_labels(str(a),str(b),"seed"))=={"A","B"}
