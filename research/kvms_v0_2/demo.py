"""Deterministic synthetic test. Illustrates only STFT energy overlap proxy."""
from pathlib import Path
import json
import numpy as np
from kvms_shadow import SceneContext, score_scene, propose_masker_gain_reduction

sr=48000
t=np.arange(sr, dtype=np.float64)/sr
stems={
    "vocal": .15*np.sin(2*np.pi*440*t),
    "guitar": .15*np.sin(2*np.pi*440*t+.4),
}
context=SceneContext(section="chorus",target="vocal",role="foreground",attribution="TARGET",attribution_confidence=.99,annotation_source="synthetic_verified")
base=score_scene(stems,context,sr)
proposal=propose_masker_gain_reduction(stems,context,"guitar",sr,attribution_labels={"guitar":"TARGET"})
result={"study":"KVMS-PILOT-00","data":"synthetic_440Hz_coincident_tones_not_a_real_song","sample_rate":sr,"status":"engineering_smoke_test_only","baseline":base.to_dict(),"proposal":proposal.to_dict(),"human_test":"not_run","live_test":"not_run"}
output=Path(__file__).parent/"synthetic_probe_results.json"
output.write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding="utf-8")
print(json.dumps(result,ensure_ascii=False,indent=2))
