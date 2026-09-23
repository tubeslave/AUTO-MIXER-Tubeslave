import numpy as np
from audio_workbench.mixing.section_context import describe_sections,climax_evidence
def test_peak_sections_exist():
 sr=1000;x=np.zeros((12000,2),dtype="float32")
 x[:4000]=.03;x[4000:8000]=.1;x[8000:]=.25
 rows=describe_sections(x,sr,2)
 assert any(r["role"]=="peak" for r in rows)
 assert climax_evidence(rows)["peak_count"]>0
